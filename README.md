# astrbot_plugin_sendemojis

AstrBot 表情包发送插件 —— 按概率在 AI 回复后自动发送表情包，通过辅助 LLM 判断上下文情绪，从对应情绪文件夹中随机选取图片。

## 功能

1. **概率触发**：每次 AI 回复后以可配置概率决定是否发送表情包。
2. **情绪分类**：命中概率后，将对话上下文传给辅助 LLM，由模型从已有情绪文件夹名中选出最匹配的情绪。
3. **随机选图**：在对应情绪子文件夹中随机挑一张图片发送。
4. **自动扫描**：插件加载/重载时自动扫描表情包目录。
5. **位置控制**：表情包可在文字之前、之后或随机穿插在分段之间发送。
6. **并行判定**：可与主 LLM 同时请求情绪判定，表情包几乎零延迟发出。
7. **对话历史记录**：可选将发送记录写入对话历史，让主 LLM 知道自己发了表情包。
8. **引用适配**：与 `astrbot_plugin_active_function` 的引用回复功能兼容。

## 表情包目录结构

```
emojis/                  ← 主文件夹（可在配置中自定义路径）
├── happy/               ← 情绪子文件夹
│   ├── 001.jpg
│   ├── 002.png
│   └── ...
├── sad/
│   ├── 001.gif
│   └── ...
├── angry/
├── surprised/
└── neutral/
```

子文件夹名即为情绪类别名，LLM 会从这些名称中选择。

## 配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `send_probability` | 发送概率 (0.0-1.0) | 0.3 |
| `emojis_path` | 表情包主文件夹路径（留空用插件目录下 emojis/） | 空 |
| `llm_provider_id` | 情绪判定 LLM Provider ID（留空用默认） | 空 |
| `emotion_timing` | 情绪判定时机：`parallel`（与主 LLM 并行，最快）/ `wait`（等主 LLM 返回后，更准） | parallel |
| `emoji_position` | 表情包位置：`before`（文字前）/ `after`（文字后）/ `random`（随机穿插） | after |
| `segment_separator` | 分段正则（仅 random 模式），用 findall 提取匹配片段。留空按句末标点分段 | 空 |
| `context_rounds` | 传给情绪判定模型的上下文轮数 | 3 |
| `llm_timeout` | 模型调用超时（秒） | 15 |
| `fallback_random_on_fail` | 模型判定失败时是否随机发送 | true |
| `notify_history` | 发送后在对话历史中追加记录，让主 LLM 知道自己发了表情包 | true |

## 推荐配置组合

| 场景 | emotion_timing | emoji_position | 效果 |
|------|---------------|----------------|------|
| 最快响应 | parallel | after | 表情包紧跟文字发出，几乎无延迟 |
| 更准确的情绪 | wait | after | 等 bot 回复后再判断，表情更贴合 |
| 自然穿插 | wait | random | 表情包随机出现在分段之间 |
| 表情先行 | parallel | before | 表情包在文字之前发出 |

## 指令

| 指令 | 说明 |
|------|------|
| `/reload_emojis` | 重新扫描表情包目录 |
| `/emojis_status` | 查看当前配置与扫描结果 |

## 安装

将本插件目录放置于 AstrBot 的 `data/plugins/` 下，重启或在后台重载插件即可。

## 依赖

无额外依赖，使用 AstrBot 内置 API。
