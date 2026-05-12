"""
astrbot_plugin_sendemojis

按概率在 AI 回复后发送表情包：
1. 概率触发（可配置）；
2. 重载时自动扫描表情包主目录下的情绪子文件夹；
3. 命中时将当前上下文交给配置的 LLM，由其判断情绪类别，
   再从对应子文件夹随机挑一张表情图片发送给用户。
4. 可选将发送记录写入对话历史，让主 LLM 知道自己发送了表情包。
5. emoji_position 控制表情包位置：before/after/random。
6. emotion_timing 控制情绪判定时机：parallel（与主LLM并行，最快）/ wait（等主LLM返回后，更准）。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
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

        self.emoji_map: dict[str, list[str]] = {}
        self.all_emojis: list[str] = []
        self.emojis_path: str = ""

        self._load_runtime_config()

        # parallel 模式：缓存预判结果 {uid: asyncio.Task}
        self._emotion_tasks: dict[str, asyncio.Task] = {}

    # ---------------------------------------------------------------- 生命周期

    async def initialize(self) -> None:
        self._load_runtime_config()
        self._scan_emojis()
        logger.info(
            f"[sendemojis] 初始化完成: 概率={self.send_probability}, "
            f"位置={self.emoji_position}, 时机={self.emotion_timing}, "
            f"情绪数={len(self.emoji_map)}, 总图片={len(self.all_emojis)}"
        )

    async def terminate(self) -> None:
        for task in self._emotion_tasks.values():
            task.cancel()
        self._emotion_tasks.clear()
        logger.info("[sendemojis] 插件已停止")

    # ------------------------------------------------------------------ 配置

    def _load_runtime_config(self) -> None:
        cfg = self.config

        self.send_probability: float = max(0.0, min(1.0, float(cfg.get("send_probability", 0.3) or 0.0)))

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
        self.notify_history: bool = bool(cfg.get("notify_history", True))

        pos = str(cfg.get("emoji_position", "after") or "after").strip().lower()
        self.emoji_position: str = pos if pos in ("before", "after", "random") else "after"

        timing = str(cfg.get("emotion_timing", "parallel") or "parallel").strip().lower()
        self.emotion_timing: str = timing if timing in ("parallel", "wait") else "parallel"

        self.segment_separator: str = str(cfg.get("segment_separator", "") or "").strip()

    # ------------------------------------------------------------ 表情包扫描

    def _scan_emojis(self) -> None:
        emoji_map: dict[str, list[str]] = {}
        all_emojis: list[str] = []

        if not os.path.isdir(self.emojis_path):
            try:
                os.makedirs(self.emojis_path, exist_ok=True)
                logger.warning(f"[sendemojis] 表情包目录不存在，已创建: {self.emojis_path}")
            except Exception as e:
                logger.error(f"[sendemojis] 无法创建表情包目录: {self.emojis_path} -> {e}")
            self.emoji_map = {}
            self.all_emojis = []
            return

        for entry in sorted(os.listdir(self.emojis_path)):
            sub_dir = os.path.join(self.emojis_path, entry)
            if not os.path.isdir(sub_dir):
                continue
            files = [
                os.path.join(sub_dir, f)
                for f in os.listdir(sub_dir)
                if os.path.isfile(os.path.join(sub_dir, f))
                and f.lower().endswith(self.image_extensions)
            ]
            if files:
                emoji_map[entry] = files
                all_emojis.extend(files)

        self.emoji_map = emoji_map
        self.all_emojis = all_emojis

        if not emoji_map:
            logger.warning(f"[sendemojis] 未在 {self.emojis_path} 下发现表情包。")

    # ---------------------------------------------------------------- 工具方法

    def _get_llm_provider(self):
        try:
            if self.llm_provider_id:
                provider = self.context.get_provider_by_id(self.llm_provider_id)
                if provider is not None:
                    return provider
                logger.warning(f"[sendemojis] 未找到 Provider: {self.llm_provider_id}, 改用默认")
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

    async def _build_context_text(self, event: AstrMessageEvent, ai_reply: str = "") -> str:
        lines: list[str] = []
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
                            try:
                                history = json.loads(history)
                            except Exception:
                                history = []
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
            logger.debug(f"[sendemojis] 读取历史对话失败: {e}")

        user_msg = (getattr(event, "message_str", "") or "").strip()
        if user_msg:
            lines.append(f"用户: {user_msg}")
        if ai_reply:
            lines.append(f"AI: {ai_reply}")

        text = "\n".join(lines).strip()
        if len(text) > 800:
            text = text[-800:]
        return text

    async def _pick_emotion_by_llm(self, context_text: str) -> str | None:
        emotions = list(self.emoji_map.keys())
        if not emotions:
            return None

        provider = self._get_llm_provider()
        if provider is None:
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

        cleaned = re.sub(r"[\s\"'`。,.!?！？：:；;()\[\]【】《》<>]+", "", raw).lower()

        for e in emotions:
            if e.lower() == cleaned:
                return e
        for e in emotions:
            if e.lower() in raw.lower():
                return e
        return None

    def _pick_emoji_file(self, emotion: str | None) -> str | None:
        if emotion and emotion in self.emoji_map and self.emoji_map[emotion]:
            return random.choice(self.emoji_map[emotion])
        if self.fallback_random and self.all_emojis:
            return random.choice(self.all_emojis)
        return None

    # ------------------------------------------------- parallel 模式：预判钩子

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """主 LLM 请求时，如果是 parallel 模式，同时发起情绪判定。"""
        if self.emotion_timing != "parallel":
            return
        if self.send_probability <= 0.0 or not self.emoji_map:
            return

        # 概率判断提前到这里
        roll = random.random()
        if roll >= self.send_probability:
            return

        uid = event.unified_msg_origin
        # 标记本次事件命中了概率
        event.set_extra("_sendemojis_triggered", True)

        # 基于用户消息+历史（不含 bot 当前回复）构建上下文并发起情绪判定
        async def _do_emotion():
            try:
                context_text = await self._build_context_text(event, ai_reply="")
                return await self._pick_emotion_by_llm(context_text)
            except Exception as e:
                logger.debug(f"[sendemojis] parallel 情绪预判失败: {e}")
                return None

        task = asyncio.create_task(_do_emotion())
        self._emotion_tasks[uid] = task

    # ------------------------------------------------------------------ 钩子

    @filter.on_decorating_result(priority=20)
    async def on_ai_reply(self, event: AstrMessageEvent):
        """AI 回复装饰阶段触发，按概率发送表情包。"""
        if self.send_probability <= 0.0 or not self.emoji_map:
            return

        result = event.get_result()
        if result is None or not getattr(result, "chain", None):
            return

        try:
            if hasattr(result, "is_llm_result") and not result.is_llm_result():
                return
        except Exception:
            pass

        ai_reply = self._extract_text_from_chain(result.chain)
        if not ai_reply:
            return

        uid = event.unified_msg_origin

        if self.emotion_timing == "parallel":
            # parallel 模式：检查是否在 on_llm_request 中命中了概率
            if not event.get_extra("_sendemojis_triggered"):
                return
            # 获取预判结果
            emotion = await self._get_parallel_emotion(uid)
        else:
            # wait 模式：在这里判断概率并调用 LLM
            roll = random.random()
            if roll >= self.send_probability:
                return
            emotion = None  # 稍后在 _process_emoji_with_emotion 中判定

        # 检测引用标签
        has_reply_tag = "[reply:" in ai_reply

        if self.emotion_timing == "parallel":
            # parallel 模式：已有 emotion 结果，直接用
            if self.emoji_position in ("random", "before"):
                await self._process_emoji_with_emotion(event, ai_reply, emotion)
            elif has_reply_tag:
                await self._process_emoji_with_emotion_chain_end(event, emotion)
            else:
                asyncio.create_task(self._send_emoji_directly(event, emotion))
        else:
            # wait 模式：需要调辅助 LLM
            if self.emoji_position in ("random", "before"):
                await self._process_emoji(event, ai_reply)
            elif has_reply_tag:
                await self._process_emoji_into_chain_end(event, ai_reply)
            else:
                asyncio.create_task(self._process_emoji(event, ai_reply))

    async def _get_parallel_emotion(self, uid: str) -> str | None:
        """获取 parallel 模式下预判的情绪结果。"""
        task = self._emotion_tasks.pop(uid, None)
        if task is None:
            return None
        try:
            return await task
        except Exception:
            return None

    # ------------------------------------------- parallel 模式的发送方法

    async def _send_emoji_directly(self, event: AstrMessageEvent, emotion: str | None) -> None:
        """parallel + after：直接用预判结果发送表情包。"""
        try:
            emoji_path = self._pick_emoji_file(emotion)
            if not emoji_path or not os.path.isfile(emoji_path):
                return
            await event.send(MessageChain([Image(file=emoji_path)]))
            emoji_filename = os.path.basename(emoji_path)
            emotion_label = emotion or "random"
            logger.info(f"[sendemojis] 表情包发送(parallel+after): {emotion_label}/{emoji_filename}")
            if self.notify_history:
                await self._notify_history(event, emotion_label, emoji_filename)
        except Exception as e:
            logger.error(f"[sendemojis] 发送表情包失败: {e}", exc_info=True)

    async def _process_emoji_with_emotion(
        self, event: AstrMessageEvent, ai_reply: str, emotion: str | None
    ) -> None:
        """parallel + before/random：用预判结果插入 chain。"""
        try:
            emoji_path = self._pick_emoji_file(emotion)
            if not emoji_path or not os.path.isfile(emoji_path):
                return
            emoji_filename = os.path.basename(emoji_path)
            emotion_label = emotion or "random"

            if self.emoji_position == "random":
                self._insert_emoji_into_chain(event, emoji_path, emotion_label, emoji_filename)
            elif self.emoji_position == "before":
                result = event.get_result()
                if result and getattr(result, "chain", None):
                    result.chain.insert(0, Image(file=emoji_path))
                    logger.info(f"[sendemojis] 表情包插入句前(parallel): {emotion_label}/{emoji_filename}")

            if self.notify_history:
                await self._notify_history(event, emotion_label, emoji_filename)
        except Exception as e:
            logger.error(f"[sendemojis] 发送表情包失败: {e}", exc_info=True)

    async def _process_emoji_with_emotion_chain_end(
        self, event: AstrMessageEvent, emotion: str | None
    ) -> None:
        """parallel + 引用场景：用预判结果追加到 chain 末尾。"""
        try:
            emoji_path = self._pick_emoji_file(emotion)
            if not emoji_path or not os.path.isfile(emoji_path):
                return
            emoji_filename = os.path.basename(emoji_path)
            emotion_label = emotion or "random"

            result = event.get_result()
            if result and getattr(result, "chain", None):
                result.chain.append(Image(file=emoji_path))
                logger.info(f"[sendemojis] 表情包追加chain末尾(parallel): {emotion_label}/{emoji_filename}")

            if self.notify_history:
                await self._notify_history(event, emotion_label, emoji_filename)
        except Exception as e:
            logger.error(f"[sendemojis] 追加表情包失败: {e}", exc_info=True)

    # ------------------------------------------- wait 模式的发送方法

    async def _process_emoji(self, event: AstrMessageEvent, ai_reply: str) -> None:
        try:
            context_text = await self._build_context_text(event, ai_reply)
            emotion = await self._pick_emotion_by_llm(context_text)
            emoji_path = self._pick_emoji_file(emotion)
            if not emoji_path or not os.path.isfile(emoji_path):
                return

            emoji_filename = os.path.basename(emoji_path)
            emotion_label = emotion or "random"

            if self.emoji_position == "random":
                self._insert_emoji_into_chain(event, emoji_path, emotion_label, emoji_filename)
            elif self.emoji_position == "before":
                result = event.get_result()
                if result and getattr(result, "chain", None):
                    result.chain.insert(0, Image(file=emoji_path))
                    logger.info(f"[sendemojis] 表情包插入句前: {emotion_label}/{emoji_filename}")
            else:
                await event.send(MessageChain([Image(file=emoji_path)]))
                logger.info(f"[sendemojis] 表情包发送(句后): {emotion_label}/{emoji_filename}")

            if self.notify_history:
                await self._notify_history(event, emotion_label, emoji_filename)
        except Exception as e:
            logger.error(f"[sendemojis] 发送表情包失败: {e}", exc_info=True)

    async def _process_emoji_into_chain_end(self, event: AstrMessageEvent, ai_reply: str) -> None:
        try:
            context_text = await self._build_context_text(event, ai_reply)
            emotion = await self._pick_emotion_by_llm(context_text)
            emoji_path = self._pick_emoji_file(emotion)
            if not emoji_path or not os.path.isfile(emoji_path):
                return

            emoji_filename = os.path.basename(emoji_path)
            emotion_label = emotion or "random"

            result = event.get_result()
            if result and getattr(result, "chain", None):
                result.chain.append(Image(file=emoji_path))
                logger.info(f"[sendemojis] 表情包追加到chain末尾: {emotion_label}/{emoji_filename}")

            if self.notify_history:
                await self._notify_history(event, emotion_label, emoji_filename)
        except Exception as e:
            logger.error(f"[sendemojis] 追加表情包失败: {e}", exc_info=True)

    # --------------------------------------------------------- 穿插发送逻辑

    def _insert_emoji_into_chain(
        self, event: AstrMessageEvent, emoji_path: str, emotion_label: str, emoji_filename: str
    ) -> None:
        result = event.get_result()
        if result is None or not getattr(result, "chain", None):
            return

        # 计算分段数来决定插入位置，但不修改原始文本
        segment_count = self._count_segments(result.chain)
        if segment_count <= 0:
            segment_count = 1

        # 随机选择插入位置：0=最前, segment_count=最后
        pos = random.randint(0, segment_count)

        # 将 Image 插入到 chain 中对应位置
        # 对于 pos=0 插入最前面，pos=segment_count 插入最后面
        # 中间位置需要拆分 Plain 文本
        new_chain = self._build_chain_with_emoji_at(result.chain, pos, emoji_path)
        result.chain = new_chain

        logger.info(f"[sendemojis] 表情包插入chain: {emotion_label}/{emoji_filename}, 位置={pos}/{segment_count}")

    def _count_segments(self, chain: list) -> int:
        """计算 chain 中的分段数（不修改原始内容）。"""
        count = 0
        if self.segment_separator:
            try:
                pattern = re.compile(self.segment_separator)
            except re.error:
                return max(1, len(chain))
            for comp in chain:
                if isinstance(comp, Plain) and comp.text:
                    parts = pattern.findall(comp.text)
                    count += max(1, len([p for p in parts if p.strip()]))
                else:
                    count += 1
            return count

        # 默认标点分段
        pattern = re.compile(r"(?<=[。？！~…\?\!])")
        for comp in chain:
            if isinstance(comp, Plain) and comp.text:
                parts = pattern.split(comp.text)
                count += max(1, len([p for p in parts if p.strip()]))
            else:
                count += 1
        return count

    def _build_chain_with_emoji_at(self, chain: list, target_pos: int, emoji_path: str) -> list:
        """在第 target_pos 个分段位置插入 Image，保持原始文本不变。"""
        if target_pos == 0:
            return [Image(file=emoji_path)] + list(chain)

        # 遍历 chain，计数分段，在到达 target_pos 时插入 Image
        new_chain = []
        current_pos = 0

        if self.segment_separator:
            try:
                pattern = re.compile(self.segment_separator)
            except re.error:
                pattern = None
        else:
            pattern = None

        split_pattern = re.compile(r"(?<=[。？！~…\?\!])") if pattern is None else None

        for comp in chain:
            if isinstance(comp, Plain) and comp.text:
                if pattern:
                    parts = pattern.findall(comp.text)
                    seg_count = max(1, len([p for p in parts if p.strip()]))
                else:
                    parts = split_pattern.split(comp.text)
                    seg_count = max(1, len([p for p in parts if p.strip()]))

                if current_pos + seg_count >= target_pos and current_pos < target_pos:
                    # Image 需要插入到这个 Plain 内部的某个位置
                    # 但我们不拆分文本，直接在这个 comp 后面插入 Image
                    new_chain.append(comp)
                    new_chain.append(Image(file=emoji_path))
                    current_pos += seg_count
                else:
                    new_chain.append(comp)
                    current_pos += seg_count
            else:
                new_chain.append(comp)
                current_pos += 1

            if current_pos >= target_pos and Image(file=emoji_path) not in new_chain:
                # 安全兜底
                pass

        # 如果还没插入（target_pos >= total），追加到末尾
        if not any(isinstance(c, Image) and getattr(c, "file", None) == emoji_path for c in new_chain):
            new_chain.append(Image(file=emoji_path))

        return new_chain

    # ---------------------------------------------------- 对话历史记录

    async def _notify_history(self, event: AstrMessageEvent, emotion: str, filename: str) -> None:
        """延迟写入一条 assistant 消息到对话历史，告知 LLM 它发送了表情包。"""
        asyncio.create_task(self._write_to_db(event, emotion, filename))

    async def _write_to_db(self, event: AstrMessageEvent, emotion: str, filename: str) -> None:
        await asyncio.sleep(8)
        try:
            cm = getattr(self.context, "conversation_manager", None)
            if cm is None:
                return
            uid = event.unified_msg_origin
            conv_id = await cm.get_curr_conversation_id(uid)
            if not conv_id:
                return
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
            history.append({
                "role": "assistant",
                "content": f"[已向用户发送了一张「{emotion}」情绪的表情包（{filename}）]",
            })
            await cm.update_conversation(unified_msg_origin=uid, conversation_id=conv_id, history=history)
        except Exception as e:
            logger.warning(f"[sendemojis] 表情包记录写入失败: {e}")

    # ------------------------------------------------------------------ 指令

    @filter.command("reload_emojis")
    async def cmd_reload_emojis(self, event: AstrMessageEvent):
        self._load_runtime_config()
        self._scan_emojis()
        emotions = ", ".join(f"{k}({len(v)})" for k, v in self.emoji_map.items()) or "无"
        yield event.plain_result(f"表情包已重载\n情绪: {emotions}\n总计: {len(self.all_emojis)} 张")

    @filter.command("emojis_status")
    async def cmd_status(self, event: AstrMessageEvent):
        emotions = ", ".join(f"{k}({len(v)})" for k, v in self.emoji_map.items()) or "无"
        yield event.plain_result(
            "📦 表情包插件状态\n"
            f"- 概率: {self.send_probability}\n"
            f"- 位置: {self.emoji_position}\n"
            f"- 时机: {self.emotion_timing}\n"
            f"- LLM: {self.llm_provider_id or '默认'}\n"
            f"- 情绪: {emotions}\n"
            f"- 总数: {len(self.all_emojis)}"
        )
