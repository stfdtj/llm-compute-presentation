# 数据差异与采用口径

制作前核对。主依据为 Downloads 中的 `pre_v2_反向传播简化版_Roofline_MFU.docx`，本地配置、项目教学页与对应在线分支。本文不更改原 Word。

| 项目 | 差异或容易混淆处 | 本次采用 |
|---|---|---|
| MFU | Word 正文为 100.6M tokens/step、60 GFLOPs/token、2048 GPUs、6 s、989.5 TFLOP/s；附录第 4 条仍是 1 TFLOP/token、2000 tokens/s、4 张卡、1000 TFLOP/s | 采用正文及本地 `compute-efficiency/app.js` 当前默认值，49.6423%。旧四卡数据不进入幻灯片。 |
| 远端与本地 MFU | 本次浏览链接分支的远端 `compute-efficiency/app.js`，其 `MFU_PRESETS.generic` 和 `DEFAULTS` 仍是四卡旧版；本地 app.js 已为 2048 卡新版 | 按用户本次明确给出的 100.6M / 60G / 2048 / 6s / 989.5 以及 Word 正文和本地新版制作。不声称远端网页已同步为新版，也不修改远端网页。 |
| 网页范围 | Word 附录链接为通用 Attention Lab；正文 Query Projection 与 MFU 示例来自独立 compute-efficiency 页面 | 真实形状采用模型配置；Query / Roofline / MFU 数值采用独立教学页。不同页面的计算范围不混用。 |
| H100 | Roofline 使用 989 TFLOP/s，MFU 默认 989.5 TFLOP/s | 保留各自原算例，标为舍入口径差异；Roofline 拐点 295.22 FLOPs/Byte，MFU 使用 989.5。 |
| 反向两倍 | Word 正文采用 2MKN 统一近似，单 token 外积的精确加法次数不同 | 主画面 193.99 MFLOPs 明确注明近似且仅含两个 Linear；核对报告另列精确值。 |
| 模型主干 | 40 层主干与 3 个 Draft Blocks 不应合并成 43 个相同主层 | 正常主干 Decode 的 Query Linear ×40；DSpark 单独简述。 |
| Encoder / Decoder | 不能理解为传统双向 Encoder + Cross-Attention Decoder | 两部分均为 Causal；Global KV 来源与 Prefill 分工按讲稿阐释。 |
| BF16 | 模型配置含 FP8 / FP4，Roofline 教学例使用 BF16 | 明确是 BF16 简化流量示例，不声称实际原模型全权重格式为 BF16。 |
| 细节顺序 | 用户建议结构与 Word 中 mHC、MoE、Engram 的先后不完全一致 | 以 Word 讲述顺序为准：架构和 DSpark → CED → Attention → mHC → MoE → Engram → 参数 → FLOPs → 反向 → Roofline → MFU → 总结。 |

后续 PPT 将按以上口径制作。Illustrative Example / Not Measured 将持续出现在 MFU 算例中。
