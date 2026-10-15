# Encoder / Decoder Attention 重设计与验收

## 实现依据

本次依据固定 revision `5254448103b6254fc646dacbca1cbbf20e9216c1` 的本地官方快照。
`references/deepseek/inference/model.py`、`inference/config.json`、`config.json` 的原始字节 SHA256 均与 `references/deepseek/manifest.json` 一致。
工作区 `models/DeepSeek-V4.1-Flash/config.json` 使用 inference 命名；图中维度与层分配已逐项比较，与固定 inference 配置一致。
同时阅读了工作区模型源码、技术报告 CED / CSA2 章节、原 SVG 构建器、HTML 模板、信息流和 Indexer JS，以及计算审计资料。

关键源码位置（固定快照）：

| 位置 | 确认内容 |
| --- | --- |
| `model.py:439–487`，Compressor | ratio=2 双投影、逐特征 softmax 加权池化、完整组才产生 latent；ratio=1 只有投影与 Norm |
| `model.py:495–589`，Indexer | Index Key 来自未旋转 latent；Query 来自本层 qᵣ，Head Weights 来自 X；评分、mask 与 Top-K；L20 候选池 |
| `model.py:614–793`，Attention | 所有层相同的 Q / Local / Attention / Inverse RoPE / 分组输出路径，及 Global 条件分支 |
| `model.py:970–994`，Block.forward | 四路 Hidden State 经 HC Pre / Norm 得到 X；Attention 输出 U 经 HC / FFN / HC 回到四路 |
| `model.py:1169–1181`，SharedAttentionRuntime | Global KV、Index Key、Top-K 索引的跨层交接 |
| `model.py:1245–1275`，Transformer.forward | token embedding、四路展开和顺序执行主干；不存在额外独立 Encoder 输出归一化模块 |

## 共同路径与实际差异

两侧都执行：X → W_Q(L) → RMSNorm → W_Q(R) → reshape / RoPE；同时 X → Local W_KV → RMSNorm → RoPE / FP8 → 本层 Local KV 环形缓存。Global 路径启用时，与 Local KV 在位置维拼接，按两组位置索引进入同一次 sparse_attn，共用 softmax 分母（包括 sink）。输出执行 Inverse RoPE → reshape / FP8 量化反量化 → 分组 W_OA → W_OB，得到 U，再由外围 HC / FFN / HC 更新 Hidden State。

相同结构不表示共用 Q、Local 或 Output 权重：各层有自己的权重与 Local Cache。

| 路径 | Encoder L0–19 | Decoder L20–39 |
| --- | --- | --- |
| Global 源层 | Full：2、8、14；本源层 X 投影、gate、两 token 池化 | Full：20；X20 投影，一 token 一条 KV，无 gate / pooling |
| Global 消费者 | L2–19 的其余层读最近源层 KV / Index Keys / 索引 | 全部读取 L20 的 Global KV / Index Keys |
| Indexer 重新评分 | 仅 Full 2、8、14 | Full 20，Reindex 24、28、32、36 |
| Reuse | 3–7、9–13、15–19；复用 KV 与索引 | 除 20 / 24 / 28 / 32 / 36 以外的层；复用 KV 与索引 |
| 仅局部 | L0、L1，无 Global / Indexer | 无 |
| RoPE 参数 | L0–1 使用 base theta=10000、无 YaRN；Global 层使用 compress theta=160000 / YaRN | compress theta=160000 / YaRN |

Encoder Final 是最后一个 Encoder Block 的四路输出 E=H19 `[1,4,5120]`。它进入 L20 的 HC Pre / Attention RMSNorm，得到 X20 `[1,5120]`；L20 的 Q、Local KV、Global KV 都从 X20 分支产生。后续 Decoder 层的 Q / Local 来自各自 X，Global 仍来自 L20 缓存。此处不虚构一个直接投影未经处理 E 的额外模块。

Full 的实际顺序是：生成未旋转 latent → 生成 / 写 Index Key → Indexer 评分 / Top-K → 对原 latent 做 RoPE / FP4 并写 Global KV → Attention 读取。Reindex 不生成新 Global KV 或 Index Key；Reuse 连 Indexer 也跳过，但本层主要 Attention 始终执行。

共享全局表示及其缓存避免后续 Decoder 层反复投影历史；逐 token 追加新条目。仍为当前 token 计算各层 Q / Local / Attention，不能误读为跳过整个 Decoder。此动机由报告与源码共同支持；没有声称本次运行了部署优化或 GPU 性能测试。

## 维度核对

以下为 B=1、单当前 token、逻辑未分片形状。S 是动态 Global 条目数，不是预设常数。

| 运算 | 配置导出的维度 |
| --- | --- |
| Q 左投影 | `[1,5120] × [5120,1280] = [1,1280]` |
| Q 右投影 | `[1,1280] × [1280,32768] = [1,32768] → [1,64,512]` |
| Local / Global KV 投影 | `[1,5120] × [5120,512] = [1,512]`；为不同权重 |
| Encoder gate 投影 | `[1,5120] × [5120,512] = [1,512]`；每特征跨两 token softmax |
| Local Cache | 每层 `[1,128,512]` 环形分配；单 Query 最多读 128 个有效位置 |
| Global Cache | 可见区 `[1,S,512]`，Top-K 最多 512 个位置 |
| RoPE | Q / Local / Global / Attention 输出的尾 64 维；输出使用逆旋转 |
| 第一段输出投影 | reshape `[1,8,4096]`；8 组独立 `[1,4096] × [4096,1024] = [1,1024]` |
| 第二段输出投影 | flatten `[1,8192] × [8192,5120] = U [1,5120]` |

## 陌生读者审查

审查对象为实际 Chrome 中的主图与“Encoder ↔ Decoder 对照”聚焦视图；细节面板补充矩阵展开、具体层切换和量化顺序，不替代主图信息流。按以下问题逐项人工检查：

| 问题 | 图中直接证据 |
| --- | --- |
| token 从哪里进入 Encoder？ | Encoder 顶部 Embedding → L0；总览保留 Text → Tokenizer → ID → Embedding |
| Encoder 依次计算什么？ | ① 输入分三路，② Q 串行链，③ Local KV / Cache，④ Global 条件分支，⑤ 同一次 Attention，Inverse RoPE、⑥ 输出、⑦ Hidden State |
| Encoder 产生什么？ | ⑦ Hℓ；H19 标为 E，粉色交接条注明四路形状 |
| Decoder 接收什么？ | 粉色 E → L20 Pre / Norm → X20；后续层从上层输入；另有 Local 与共享 Global 缓存 |
| Decoder Q / Local / Global 怎么得到？ | 等位三路；Q / Local 每层投影，Global 明示 Full L20、后续层读 L20 |
| 两侧哪些相同？ | ② / ③ / ⑤ / Inverse RoPE / ⑥ / ⑦ 同位置、同名称、同颜色与维度 |
| 核心差异？ | 橙色 Global 大标记 2 tokens → 1 KV 与 1 token → 1 KV；源层与 gate / pooling 显式区别 |
| 为什么不重算历史？ | Decoder 顶部标注避免各层重建历史 Global KV；Global 分支注明 L21–39 读 L20 缓存 |
| 什么重算，什么复用？ | 本层 Q / Local 均有投影链；Full / Reindex / Reuse 的层号与 KV / 索引复用标记；层选择面板显示具体动作 |
| 一次 Attention 能否追踪完整？ | 三路有独立入端口，沿箭头到共同 Attention、逆 RoPE、分组投影、U 与 H，无箭头穿过文字 |

审查后增大了 Global 的 2→1 / 1→1 标识，避免读者必须读完八行小字才能看到差异；跨模块交接独立使用粉色标签；修正旧信息流箭头坐标，避免旧线路穿过新节点。

## 可复现验收与范围

- `.venv/Scripts/python.exe src/deepseek/build.py`：重新生成单文件 HTML 与 SVG。
- `.venv/Scripts/python.exe src/deepseek/attention_preservation.py`：用未修改的 `before_attention.svg` 与固定 SHA256，逐元素比较两个 Attention 区域以外的完整 SVG。
- `.venv/Scripts/python.exe src/deepseek/verify_single_token.py`、`verify_audit.py`：原计算数据、配置、数值和矩阵维度检查。
- `node src/deepseek/verify_attention.cjs`：40 层模式、共同模块、串行 Q 连线、真实字体文字边界、公式面板、Indexer 入口、信息流及截图。
- `npm run test:browser:deepseek`：原有参数、信息流、矩阵演示、Indexer、推导、HC 浏览器回归。

截图保存在 `src/deepseek/qa/attention_overview_1366.png`、`attention_overview_1920.png`、`attention_pair_1366.png`、`attention_pair_1920.png`、`attention_comparison_1920.png`、`attention_comparison_output.png`、`attention_flow_updated.png`。

旧保护测试只为本次授权的 Attention 区域增加例外，并用独立新基线锁定所有其他元素；仅更新 `step4_preservation.json` 的 architecture.svg 哈希，计算数据与审计文档的哈希未改。

验证边界：本次没有加载模型权重或运行 GPU kernel。在线固定版本 URL 经浏览工具读取不可用，因此证据以经过 SHA256 校验的项目官方快照为准，不声称核对了线上最新版本。上述图示结构在该固定实现中均能确认；没有用假设补齐结构差异。
