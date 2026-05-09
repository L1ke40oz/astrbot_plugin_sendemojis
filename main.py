"""
astrbot_plugin_sendemojis

按概率在 AI 回复后发送表情包：
1. 概率触发（可配置）；
2. 重载时自动扫描表情包主目录下的情绪子文件夹；
3. 命中时将当前上下文交给配置的 LLM，由其判断情绪类别，
   再从对应子文件夹随机挑一张表情图片发送给用户。
4. 发送后以 fake tool call 形式注入对话历史，让主 LLM 知道自己发送了表情包。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register
from astrbot.core.message.message_event_result import MessageChain


@register(
    "astrbot_plugin_sendemojis",
    "you",
    "按概率由模型判断情绪后发送表情包。",
    "1.0.0",
)
class SendEmojisPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None):
        super().__init__(context)
        self.config: dict[str, Any] = config or {}

        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))

        # 情绪 -> [图片绝对路径, ...]
        self.emoji_map: dict[str, list[str]] = {}
        # 所有表情包的扁平列表，用于降级随机
        self.all_emojis: list[str] = []
        # 实际使用的表情包根目录
        self.emojis_path: str = ""

        # 读取配置
        self._load_runtime_config()

    # ---------------------------------------------------------------- 生命周期

    async def initialize(self) -> None:
        """插件加载/重载时调用，自动扫描表情包目录。"""
        self._load_runtime_config()
        self._scan_emojis()
        logger.info(
            f"[sendemojis] 插件已初始化: 概率={self.send_probability}, "
            f"目录={self.emojis_path}, 情绪数={len(self.emoji_map)}, "
            f"总图片={len(self.all_emojis)}"
        )

    async def terminate(self) -> None:
        logger.info("[sendemojis] 插件已停止")

    # ------------------------------------------------------------------ 配置

    def _load_runtime_config(self) -> None:
        cfg = self.config

        self.send_probability: float = float(cfg.get("send_probability", 0.3) or 0.0)
        self.send_probability = max(0.0, min(1.0, self.send_probability))

        raw_path = str(cfg.get("emojis_path", "") or "").strip()
        if not raw_path:
            self.emojis_path = os.path.join(self.plugin_dir, "emojis")
        elif os.path.isabs(raw_path):
            self.emojis_path = raw_path
        else:
            self.emojis_path = os.path.normpath(os.path.join(self.plugin_dir, raw_path))

        self.image_extensions: tuple[str, ...] = (
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"
        )

        self.llm_provider_id: str = str(cfg.get("llm_provider_id", "") or "").strip()
        self.context_rounds: int = int(cfg.get("context_rounds", 3) or 3)
        self.llm_timeout: int = int(cfg.get("llm_timeout", 15) or 15)

        self.fallback_random: bool = bool(cfg.get("fallback_random_on_fail", True))
        self.inject_fake_tool_call: bool = bool(cfg.get("inject_fake_tool_call", False))

    # ------------------------------------------------------------ 表情包扫描

    def _scan_emojis(self) -> None:
        """扫描主目录下的情绪子文件夹，加载所有图片路径。"""
        emoji_map: dict[str, list[str]] = {}
        all_emojis: list[str] = []

        if not os.path.isdir(self.emojis_path):
            try:
                os.makedirs(self.emojis_path, exist_ok=True)
                logger.warning(
                    f"[sendemojis] 表情包目录不存在，已创建空目录: {self.emojis_path}。"
                    f"请在其下放置以情绪命名的子文件夹及图片。"
                )
            except Exception as e:
                logger.error(f"[sendemojis] 表情包目录无效且无法创建: {self.emojis_path} -> {e}")
            self.emoji_map = {}
            self.all_emojis = []
            return

        for entry in sorted(os.listdir(self.emojis_path)):
            sub_dir = os.path.join(self.emojis_path, entry)
            if not os.path.isdir(sub_dir):
                continue
            files: list[str] = []
            for fname in os.listdir(sub_dir):
                fpath = os.path.join(sub_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if fname.lower().endswith(self.image_extensions):
                    files.append(fpath)
            if files:
                emoji_map[entry] = files
                all_emojis.extend(files)

        self.emoji_map = emoji_map
        self.all_emojis = all_emojis

        if not emoji_map:
            logger.warning(
                f"[sendemojis] 未在 {self.emojis_path} 下发现任何情绪子文件夹或图片。"
            )

    # ---------------------------------------------------------------- 工具方法

    def _get_llm_provider(self):
        """根据配置获取 LLM Provider，失败时返回 None。"""
        try:
            if self.llm_provider_id:
                provider = self.context.get_provider_by_id(self.llm_provider_id)
                if provider is not None:
                    return provider
                logger.warning(
                    f"[sendemojis] 未找到指定的 LLM Provider: {self.llm_provider_id}, 改用默认。"
                )
            # 默认 LLM
            if hasattr(self.context, "get_using_provider"):
                return self.context.get_using_provider()
        except Exception as e:
            logger.error(f"[sendemojis] 获取 LLM Provider 失败: {e}")
        return None

    @staticmethod
    def _extract_text_from_chain(chain) -> str:
        if not chain:
            return ""
        parts: list[str] = []
        for comp in chain:
            txt = getattr(comp, "text", None)
            if isinstance(txt, str) and txt:
                parts.append(txt)
        return "".join(parts).strip()

    async def _build_context_text(self, event: AstrMessageEvent, ai_reply: str) -> str:
        """拼接最近若干轮对话 + 本轮用户消息 + AI 回复。"""
        lines: list[str] = []

        # 取会话历史（如果能拿到）
        try:
            uid = event.unified_msg_origin
            cm = getattr(self.context, "conversation_manager", None)
            if cm is not None and self.context_rounds > 0:
                conv_id = await cm.get_curr_conversation_id(uid)
                if conv_id:
                    conv = await cm.get_conversation(uid, conv_id)
                    if conv is not None:
                        history = getattr(conv, "history", None) or []
                        if isinstance(history, str):
                            import json as _json

                            try:
                                history = _json.loads(history)
                            except Exception:
                                history = []
                        # 每轮 = 一问一答 -> 取最后 context_rounds*2 条
                        tail = list(history)[-(self.context_rounds * 2):]
                        for msg in tail:
                            if not isinstance(msg, dict):
                                continue
                            role = str(msg.get("role", "")).lower()
                            content = msg.get("content", "")
                            if not isinstance(content, str):
                                content = str(content)
                            if not content.strip():
                                continue
                            label = "用户" if role == "user" else ("AI" if role == "assistant" else role)
                            lines.append(f"{label}: {content}")
        except Exception as e:
            logger.debug(f"[sendemojis] 读取历史对话失败（忽略）: {e}")

        user_msg = ""
        try:
            user_msg = (event.message_str or "").strip()
        except Exception:
            user_msg = ""
        if user_msg:
            lines.append(f"用户: {user_msg}")
        if ai_reply:
            lines.append(f"AI: {ai_reply}")

        text = "\n".join(lines).strip()
        # 限制传给模型的上下文长度
        max_chars = 800
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    async def _pick_emotion_by_llm(self, context_text: str) -> str | None:
        """让 LLM 从已有情绪类别中选一个。返回情绪名（已校验），否则 None。"""
        emotions = list(self.emoji_map.keys())
        if not emotions:
            return None

        provider = self._get_llm_provider()
        if provider is None:
            logger.debug("[sendemojis] 无可用 LLM Provider，跳过情绪判定。")
            return None

        emotion_list_str = ", ".join(emotions)
        system_prompt = (
            "你是一个对话情绪分类器。根据给定的对话上下文，从候选情绪中挑出最贴合 AI "
            "当前语气/状态的那一项。只输出情绪名本身，不要解释、不要标点、不要多余文字。"
            "如果没有合适的，也必须从候选中挑一个最接近的。"
        )
        user_prompt = (
            f"候选情绪（只能从中选一个，且严格使用原文）：{emotion_list_str}\n\n"
            f"对话上下文：\n{context_text}\n\n"
            f"请只输出一个情绪词："
        )

        try:
            resp = await asyncio.wait_for(
                provider.text_chat(prompt=user_prompt, system_prompt=system_prompt),
                timeout=self.llm_timeout,
            )
            raw = (getattr(resp, "completion_text", "") or "").strip()
        except asyncio.TimeoutError:
            logger.warning(f"[sendemojis] 情绪判定超时 ({self.llm_timeout}s)")
            return None
        except Exception as e:
            logger.warning(f"[sendemojis] 情绪判定调用失败: {e}")
            return None

        if not raw:
            return None

        # 规范化，去标点/引号/空白
        cleaned = re.sub(r"[\s\"'`。,.!?！？：:；;()\[\]【】《》<>]+", "", raw).lower()
        logger.debug(f"[sendemojis] LLM 情绪原始返回={raw!r} -> {cleaned!r}")

        # 精确匹配（忽略大小写）
        for e in emotions:
            if e.lower() == cleaned:
                return e
        # 包含匹配（模型可能输出一句话）
        for e in emotions:
            if e.lower() in raw.lower():
                return e
        return None

    def _pick_emoji_file(self, emotion: str | None) -> str | None:
        """从指定情绪文件夹中随机挑一张；情绪为 None 时从全部中挑。"""
        if emotion and emotion in self.emoji_map and self.emoji_map[emotion]:
            return random.choice(self.emoji_map[emotion])
        if self.fallback_random and self.all_emojis:
            return random.choice(self.all_emojis)
        return None

    # ------------------------------------------------------------------ 钩子

    @filter.on_decorating_result()
    async def on_ai_reply(self, event: AstrMessageEvent):
        """在 AI 回复装饰阶段触发，按概率附带发送表情包。"""
        if self.send_probability <= 0.0 or not self.emoji_map:
            return

        result = event.get_result()
        if result is None or not getattr(result, "chain", None):
            return

        # 必须是 AI 生成的回复才发（避免对命令回复附带表情）
        try:
            if hasattr(result, "is_llm_result") and not result.is_llm_result():
                return
        except Exception:
            pass

        ai_reply = self._extract_text_from_chain(result.chain)
        if not ai_reply:
            return

        # 概率判断
        roll = random.random()
        if roll >= self.send_probability:
            logger.debug(
                f"[sendemojis] 未命中概率: roll={roll:.3f} >= p={self.send_probability:.3f}"
            )
            return
        logger.debug(
            f"[sendemojis] 命中概率: roll={roll:.3f} < p={self.send_probability:.3f}"
        )

        # 异步发送，避免阻塞当前回复
        asyncio.create_task(self._decide_and_send(event, ai_reply))

    async def _decide_and_send(self, event: AstrMessageEvent, ai_reply: str) -> None:
        try:
            context_text = await self._build_context_text(event, ai_reply)
            emotion = await self._pick_emotion_by_llm(context_text)
            logger.debug(f"[sendemojis] 判定情绪: {emotion}")

            emoji_path = self._pick_emoji_file(emotion)
            if not emoji_path or not os.path.isfile(emoji_path):
                logger.debug("[sendemojis] 没有可发送的表情包，跳过。")
                return

            chain = MessageChain([Image(file=emoji_path)])
            await event.send(chain)

            emoji_filename = os.path.basename(emoji_path)
            emotion_label = emotion or "random"
            logger.info(
                f"[sendemojis] 已发送表情包: emotion={emotion_label}, file={emoji_filename}"
            )

            # Fake tool call 注入对话历史，让主 LLM 知道自己发送了表情包
            if self.inject_fake_tool_call:
                await self._inject_fake_tool_call(event, emotion_label, emoji_filename)

        except Exception as e:
            logger.error(f"[sendemojis] 发送表情包失败: {e}", exc_info=True)

    async def _inject_fake_tool_call(
        self, event: AstrMessageEvent, emotion: str, filename: str
    ) -> None:
        """将表情包发送行为以 fake tool call 形式写入对话历史。"""
        try:
            cm = getattr(self.context, "conversation_manager", None)
            if cm is None:
                return

            uid = event.unified_msg_origin
            conv_id = await cm.get_curr_conversation_id(uid)
            if not conv_id:
                return

            # 生成唯一 tool_call_id
            call_id = f"emoji_{uuid.uuid4().hex[:12]}"

            # assistant 发起的 tool_call
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "send_emoji",
                            "arguments": json.dumps(
                                {"emotion": emotion, "file": filename},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }

            # tool 返回结果
            tool_msg = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "send_emoji",
                "content": f"已向用户发送了一张「{emotion}」的表情包（{filename}）。",
            }

            # 读取当前对话历史并追加
            conv = await cm.get_conversation(uid, conv_id)
            if conv is None:
                return

            history = getattr(conv, "history", None) or []
            if isinstance(history, str):
                try:
                    history = json.loads(history)
                except Exception:
                    history = []

            history = list(history)
            history.append(assistant_msg)
            history.append(tool_msg)

            await cm.update_conversation(
                unified_msg_origin=uid,
                conversation_id=conv_id,
                history=history,
            )
            logger.debug(f"[sendemojis] 已注入 fake tool call 到对话历史: {call_id}")

        except Exception as e:
            logger.debug(f"[sendemojis] 注入 fake tool call 失败（不影响发送）: {e}")

    # ------------------------------------------------------------------ 指令

    @filter.command("reload_emojis")
    async def cmd_reload_emojis(self, event: AstrMessageEvent):
        """手动重新扫描表情包目录。"""
        self._load_runtime_config()
        self._scan_emojis()
        emotions = ", ".join(f"{k}({len(v)})" for k, v in self.emoji_map.items()) or "无"
        yield event.plain_result(
            f"表情包已重载\n目录: {self.emojis_path}\n情绪及数量: {emotions}\n总计: {len(self.all_emojis)} 张"
        )

    @filter.command("emojis_status")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看当前表情包配置与扫描结果。"""
        emotions = ", ".join(f"{k}({len(v)})" for k, v in self.emoji_map.items()) or "无"
        yield event.plain_result(
            "📦 表情包插件状态\n"
            f"- 概率: {self.send_probability}\n"
            f"- 目录: {self.emojis_path}\n"
            f"- LLM ID: {self.llm_provider_id or '<默认>'}\n"
            f"- 上下文轮数: {self.context_rounds}\n"
            f"- 情绪分类: {emotions}\n"
            f"- 总图片数: {len(self.all_emojis)}"
        )
