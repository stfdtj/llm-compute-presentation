# Deck Outline

- File: D:\working\llm-compute-presentation\visual_deck\DeepSeek_V41_Flash_计算成本可视化.pptx
- Slides: 22

## Slide 1: (No title)

Body:
- 一.1 整体架构
- DeepSeek V4.1 Flash：计算成本从架构开始
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 01 / 22
- Architecture  →  Parameters  →  FLOPs  →  Efficiency
- Text
- Token Embedding
- Image
- Vision + Aligner
- Causal Encoder
- 20 Layers
- Causal Decoder
- 20 Layers
- LM Head
- Next Token
- DSpark · 3 Draft Blocks → 5 候选 Tokens

Notes:
- 对应 Word：一.1 整体架构。
- 段落索引（从 0 起）：[0, 17]。
- 讲述：沿架构 → 参数量 → FLOPs → 硬件效率展开。DSpark 的 3 个 Draft Blocks 是辅助分支，不计为 43 个同样的主层。
- 点击顺序：
- 1. 文本 / 图像进入各自输入模块
- 2. 显示 40 层 Transformer 主干与 LM Head
- 3. 显示 DSpark 独立草稿分支
- 本页完成后再次点击切换下一页。

## Slide 2: (No title)

Body:
- 一.2 Causal Encoder 与 Decoder
- 20 层 + 20 层：两部分都遵守 Causal Mask
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 02 / 22
- Encoder · 0–19
- Decoder · 20–39
- Prefill
- 多数输入 Token
- 形成 Encoder 表示
- 构建 Global KV
- 复用 Encoder 表示
- 减少后半段重复计算
- Decode
- 当前 Token：正常主干仍经过 40 层
- 两部分都遵守 Causal Mask；不是传统的双向 Encoder–Decoder。

Notes:
- 对应 Word：一.2 Causal Encoder 与 Decoder。
- 段落索引（从 0 起）：[18, 25]。
- 讲述：Encoder / Decoder 的区别是计算职责和 Global KV 来源，不是传统双向 Encoder 与 Cross-Attention Decoder。
- 点击顺序：
- 1. 显示 Prefill 主要处理输入的路径
- 2. 显示 Encoder 表示构建 Decoder Global KV
- 3. 显示 Decode 当前 token 经过 40 层
- 本页完成后再次点击切换下一页。

## Slide 3: (No title)

Body:
- 一.3 Encoder Attention
- 局部窗口读取近处，全局路径补充远处
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 03 / 22
- 较早的历史
- 当前 Token
- Local：最近最多 128 个位置
- Encoder Global KV · 2 Tokens → 1 位置
- Global：补充窗口之外的信息
- 位置数为示意；第 0、1 层仅 SWA，从第 2 层起引入压缩 Global KV。

Notes:
- 对应 Word：一.3 Encoder Attention。
- 段落索引（从 0 起）：[26, 42]。
- 讲述：图中只画部分示意位置。Encoder 第 0、1 层只有 SWA；从第 2 层开始增加 Global KV。128 是配置窗口上限。
- 点击顺序：
- 1. 高亮最近最多 128 个 Local 位置
- 2. 显示历史 token 的 2:1 全局压缩
- 3. 连接 Global KV 与当前 Query
- 本页完成后再次点击切换下一页。

## Slide 4: (No title)

Body:
- 一.3 Global KV 与 Indexer
- Indexer 先选位置，Main Attention 再取信息
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 04 / 22
- Indexer Query
- Main Query
- Compressed Global KV
- 0.2   0.9   0.4   0.8   …
- 选择最多 Top-512 个位置
- 选中 Global KV + Local KV
- Softmax
- Attention Output
- Indexer 负责“选位置”，Main Attention 负责“读信息”；两者使用不同的 Query。

Notes:
- 对应 Word：一.3 Global KV 与 Indexer。
- 段落索引（从 0 起）：[43, 47]。
- 讲述：Indexer 与 Main Attention 使用不同 Query。Top-512 是上限；实际数量取决于可用历史。评分和筛选本身也有开销。
- 点击顺序：
- 1. Indexer Query 为全局位置评分
- 2. 高亮最多 Top-512 位置
- 3. Main Query 读取选中 Global KV 与 Local KV
- 4. 显示共同 Softmax 与输出
- 本页完成后再次点击切换下一页。

## Slide 5: (No title)

Body:
- 一.4 CSA2 与层间复用
- CSA2：分清 Full、Reindex 与 Reuse
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 05 / 22
- Full
- Reindex
- Reuse
- 新 KV + 新选址
- 共享 KV + 新选址
- 共享 KV + 共享选址
- Encoder
- 2
- 8
- 14
- → 各自后续层复用
- Decoder
- 20
- 24
- 28
- 32
- 36
- Decoder 候选池：8 positions/block，最多 2048 blocks；层号从 0 开始。

Notes:
- 对应 Word：一.4 CSA2 与层间复用。
- 段落索引（从 0 起）：[48, 70]。
- 讲述：Full 建立 Global KV 和 Index Keys 并选择；Reindex 保留缓存重新选择；Reuse 连最近选址也复用。Decoder 候选池每块 8 位置、最多 2048 块，限制后续搜索。层号从 0 开始。
- 点击顺序：
- 1. 显示 Encoder 全局源层 2 / 8 / 14
- 2. 显示 Decoder 第 20 层提供 Global KV
- 3. 显示 24 / 28 / 32 / 36 层重新选址
- 本页完成后再次点击切换下一页。

## Slide 6: (No title)

Body:
- 一.5 mHC
- mHC：四路残差信息，一路主模块计算
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 06 / 22
- x1
- x2
- x3
- x4
- 4 × 5120
- Pre
- 混合为一路
- Attention / MoE
- 5120 维输入
- Post + Comb
- 四路状态在主模块前汇合；主模块输出再分配，与残差信息组合。

Notes:
- 对应 Word：一.5 mHC。
- 段落索引（从 0 起）：[71, 82]。
- 讲述：每路维度 5120。四路状态不等于 Attention 和 MoE 直接执行四遍；混合权重和混合过程另有参数和成本。
- 点击顺序：
- 1. 四路 Hidden States 通过 Pre 混为一路
- 2. Attention / MoE 执行一次主计算
- 3. Post 分配、Comb 混合回四路
- 本页完成后再次点击切换下一页。

## Slide 7: (No title)

Body:
- 一.6 MoE 与 Dense
- Dense 使用同一组 FFN；MoE 选择执行路径
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 07 / 22
- Dense FFN
- Token
- 同一组 FFN
- 所有输入使用同一组权重
- Mixture of Experts
- Token
- Router
- 384 Routed Experts · 评分示意
- Scores
- 0.12 · 0.91 · …
- Top-6
- 1 Shared
- 6 × Expert FFN
- Σ αᵢ Expertᵢ(x) + Shared(x)

Notes:
- 对应 Word：一.6 MoE 与 Dense。
- 段落索引（从 0 起）：[83, 96]。
- 讲述：专家墙真实包含 384 格。分数与选中的具体专家编号只作教学示意，不是运行轨迹；共享专家不参与这次 Top-6 竞争。
- 点击顺序：
- 1. Token 进入 Router
- 2. 显示 384 个 Routed Experts 的评分示意
- 3. 高亮其中 6 个选中专家
- 4. 显示始终执行的 1 个 Shared Expert
- 5. 选中专家执行 FFN
- 6. 路由加权求和，再加共享专家输出
- 本页完成后再次点击切换下一页。

## Slide 8: (No title)

Body:
- 一.6 Expert 内部计算
- 一个 Expert：Gate、Up、Down 三个投影
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 08 / 22
- X
- [1,5120]
- Gate
- 5120 → 2304
- Up
- 5120 → 2304
- SiLU
- ⊙
- [1,2304]
- Down
- 2304 → 5120
- 三个 Linear；SiLU 与逐元素乘法无新增可学习参数，但仍有计算成本。

Notes:
- 对应 Word：一.6 Expert 内部计算。
- 段落索引（从 0 起）：[93, 96]。
- 讲述：Expert(x)=Down(SiLU(Gate(x)) ⊙ Up(x))。三个 Linear 均无 bias，SiLU 和逐元素乘法无新增权重但有计算。
- 点击顺序：
- 1. X 分别进入 Gate 与 Up
- 2. Gate 经过 SiLU 后与 Up 逐元素相乘
- 3. Down 将 2304 维投影回 5120 维
- 本页完成后再次点击切换下一页。

## Slide 9: (No title)

Body:
- 一.7 Engram
- Engram：按局部 token 组合查找条件记忆
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 09 / 22
- Token 组合
- N-gram
- Hash
- 大型 Embedding Table
- 读取少量记忆
- 参与后续处理
- 放置于 Encoder 第 1、14 层；图示为检索直觉，不是完整 Engram 内部实现。

Notes:
- 对应 Word：一.7 Engram。
- 段落索引（从 0 起）：[97, 103]。
- 讲述：Engram 位于 Encoder 第 1 和 14 层。大量存储参数不等于每 token 都执行整表的稠密乘法；查表仍有访存成本。
- 点击顺序：
- 1. N-gram token 组合进入哈希查找
- 2. 从大型表中读取少量条目
- 3. 读取的记忆参与后续处理
- 本页完成后再次点击切换下一页。

## Slide 10: (No title)

Body:
- 二.1 一个 Expert
- 从矩阵形状数参数：一个 Expert 约 3540 万
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 10 / 22
- Gate
- [5120,2304]
- 11,796,480
- Up
- [5120,2304]
- 11,796,480
- Down
- [2304,5120]
- 11,796,480
- P(Expert) = 3 × 5120 × 2304 = 35,389,440
- 三个 Linear 均无加性 Bias

Notes:
- 对应 Word：二.1 一个 Expert。
- 段落索引（从 0 起）：[104, 124]。
- 讲述：以单 token 输入 [1,5120] 说明形状，参数量本身不随本次输入 token 数增加。Down 形状按 XW 记法写作 [2304,5120]。
- 点击顺序：
- 1. 显示 Gate 权重 5120 × 2304
- 2. 显示 Up 和 Down 的矩阵形状
- 3. 三个矩阵参数量求和
- 本页完成后再次点击切换下一页。

## Slide 11: (No title)

Body:
- 二.2–3 Total vs Active
- 保存 385 个专家，每个 token 使用其中 7 个
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 11 / 22
- Total Parameters
- Active Parameters / Token
- + Shared
- 385 × 35,389,440
- 13,624,934,400
- + Shared
- 7 × 35,389,440
- 247,726,080
- + Router 1,966,080 → 约 136.27 亿
- + Router → 约 2.50 亿

Notes:
- 对应 Word：二.2–3 Total vs Active。
- 段落索引（从 0 起）：[125, 144]。
- 讲述：单层专家总权重 13,624,934,400；激活专家 247,726,080。Router 矩阵 1,966,080。主数值不含选择偏置，且不是全模型参数量。
- 点击顺序：
- 1. 显示全部专家的总参数量
- 2. 仅高亮 6 Routed + 1 Shared
- 3. 分别加上 Router 矩阵权重
- 本页完成后再次点击切换下一页。

## Slide 12: (No title)

Body:
- 三.前向 1 从参数到 FLOPs
- Matrix Multiplication：一行 × 一列，生成一个数
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 12 / 22
- A [2,3]
- 1
- 2
- 3
- 4
- 5
- 6
- ×
- B [3,2]
- 1
- 2
- 3
- 4
- 5
- 6
- =
- C [2,2]
- 每次只生成 C 中的一个元素
- 1
- 2
- 3
- Row 1
- 1
- 2
- 3
- 4
- 5
- 6
- Column 1
- 1 × 1 = 1
- 2 × 3 = 6
- 3 × 5 = 15
- 1 + 6 + 15 = 22
- 22

Notes:
- 对应 Word：三.前向 1 从参数到 FLOPs。
- 段落索引（从 0 起）：[145, 155]。
- 讲述：小矩阵为教学例。A=[[1,2,3],[4,5,6]]，B=[[1,2],[3,4],[5,6]]。每次点击只推进一个乘法或求和动作，等待听众看清再继续。
- 点击顺序：
- 1. 高亮 A 第一行 [1,2,3]
- 2. 高亮 B 第一列 [1,3,5]
- 3. 显示 1×1 = 1
- 4. 显示 2×3 = 6
- 5. 显示 3×5 = 15
- 6. 将乘积相加：1+6+15 = 22
- 7. 将 22 写入 C 第一行第一列
- 本页完成后再次点击切换下一页。

## Slide 13: (No title)

Body:
- 三.前向 1 矩阵乘法与计数
- 换一列，得到下一个输出；重复同样的规则
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 13 / 22
- A [2,3]
- 1
- 2
- 3
- 4
- 5
- 6
- ×
- B [3,2]
- 1
- 2
- 3
- 4
- 5
- 6
- =
- C [2,2]
- 22
- 1
- 2
- 3
- 1
- 2
- 3
- 4
- 5
- 6
- 1 × 2 + 2 × 4 + 3 × 6 = 28
- 22
- 28
- 49
- 64
- 一个输出：K 次乘法 + (K−1) 次加法
- 本例精确：4 × 5 = 20 FLOPs     常用近似：2MKN = 24 FLOPs

Notes:
- 对应 Word：三.前向 1 矩阵乘法与计数。
- 段落索引（从 0 起）：[151, 155]。
- 讲述：C=[[22,28],[49,64]]。每个输出精确 5 次乘加，全矩阵 20 FLOPs；2×2×3×2=24 为常用近似。不是精确 24。
- 点击顺序：
- 1. 高亮 A 第一行与 B 第二列
- 2. 显示 1×2 + 2×4 + 3×6 = 28
- 3. 将 28 写入 C，并补齐第二行
- 4. 显示 3 次乘法 + 2 次加法，以及 2MKN 近似
- 本页完成后再次点击切换下一页。

## Slide 14: (No title)

Body:
- 三.前向 2 Query 投影
- Query Low-Rank Projection：先压缩，再展开
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 14 / 22
- X [1,5120]
- W_QA
- [5120,1280]
- [1,1280]
- 13.11 MFLOPs
- RMSNorm
- W_QB
- [1280,32768]
- Q
- [1,32768]
- 83.89 MFLOPs
- 64 Heads × 512
- 两次 Linear 合计：96.99 MFLOPs
- 宽度为示意；FLOPs 不含 RMSNorm / RoPE

Notes:
- 对应 Word：三.前向 2 Query 投影。
- 段落索引（从 0 起）：[156, 188]。
- 讲述：形状真实，向量宽度为示意，不按维度严格等比。主 FLOPs 仅两个 Linear，排除 RMSNorm / RoPE。Reshape 不增加浮点乘加。
- 点击顺序：
- 1. W_QA：5120 → 1280，13.11 MFLOPs
- 2. RMSNorm：维度仍为 1280
- 3. W_QB：1280 → 32768，83.89 MFLOPs
- 4. 重排为 64 Heads × 512，汇总 96.99 MFLOPs
- 本页完成后再次点击切换下一页。

## Slide 15: (No title)

Body:
- 三.前向 3 从单模块到多层
- 40 层 Query 投影：约 3.88 GFLOPs
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 15 / 22
- 1 Layer
- 96.99 MFLOPs
- 40 次同形状 Query 投影
- ≈ 3.8797 GFLOPs
- 正常主干 Decode；只含 Query 的两个 Linear，其余模块仍需另计。

Notes:
- 对应 Word：三.前向 3 从单模块到多层。
- 段落索引（从 0 起）：[189, 194]。
- 讲述：该乘法针对正常主干 Decode 的两次 Query Linear；不是 Prefill、不是完整全模型前向、也不含 DSpark。
- 点击顺序：
- 1. 显示一层 Query 的 96.99 MFLOPs
- 2. 按 40 个主干层累加
- 3. 显示 3.8797 GFLOPs，并指出其余模块仍需另计
- 本页完成后再次点击切换下一页。

## Slide 16: (No title)

Body:
- 三.Backward 1–2 五步与顺序
- Forward 保存输入；Backward 沿图反向传梯度
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 16 / 22
- X
- W_QA
- RMSNorm
- W_QB
- Q
- Forward →
- 保存 X、H、H′ 与各层权重
- 后续模块
- Loss
- ① dQ
- ② 取回前向输入与权重
- dX
- Backward ←

Notes:
- 对应 Word：三.Backward 1–2 五步与顺序。
- 段落索引（从 0 起）：[196, 214]。
- 讲述：原稿用 dZ 指末端梯度；此处聚焦 Query 子图，以 dQ 表示后续 Attention 已传回的梯度。整个模型的 Loss 不直接等于 Query 局部 Loss。
- 点击顺序：
- 1. 显示前向 X → W_QA → RMSNorm → W_QB → Q
- 2. ① 从 Loss 获得 dQ
- 3. ② 取回本层保存的输入和权重
- 4. 反向从 W_QB 经 RMSNorm 回到 W_QA
- 本页完成后再次点击切换下一页。

## Slide 17: (No title)

Body:
- 三.Backward 3–5 Weight / Input Gradient
- 每个 Linear 同时产生两种去向的 Gradient
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 17 / 22
- H′ [1,1280]
- X [1,5120]
- W_QA
- RMSNorm
- W_QB
- dQ
- ③ dW_QB = H′ᵀ dQ
- [1280,1] × [1,32768] → [1280,32768]
- ④ dH′ = dQ W_QBᵀ
- [1,32768] × [32768,1280] → [1,1280]
- dH′ → dH
- dW_QA = Xᵀ dH
- [5120,1] × [1,1280] → [5120,1280]
- dX = dH W_QAᵀ
- [1,1280] × [1280,5120] → [1,5120]
- ⑤ dX [1,5120] → 前一模块

Notes:
- 对应 Word：三.Backward 3–5 Weight / Input Gradient。
- 段落索引（从 0 起）：[215, 234]。
- 讲述：dW 使用洋红色，dX / 继续传播梯度使用橙色。输入形状、输出形状均列出。RMSNorm 反向必须经过，不是恒等；γ 的梯度未在主图展开，成本小计另排除。
- 点击顺序：
- 1. W_QB 收到 dQ
- 2. ③ dW_QB = H′ᵀ dQ，向权重分支流动
- 3. ④ dH′ = dQ W_QBᵀ，继续向前
- 4. RMSNorm 将 dH′ 转为 dH
- 5. W_QA 分支产生 dW_QA 与 dX
- 6. ⑤ dX 继续传给前一模块
- 本页完成后再次点击切换下一页。

## Slide 18: (No title)

Body:
- 三.Backward 6 反向 FLOPs 汇总
- 两次梯度矩阵乘法，让 Linear 反向约为前向两倍
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 18 / 22
- Forward
- F ≈ 2MKN
- Backward
- dW
- dX
- Query Linear Backward ≈ 193.99 MFLOPs
- 单 token 精确外积计数不同；“两倍”是统一 2MKN 近似。
- 前向 + 反向 ≈ 290.98 MFLOPs；不含 RMSNorm、优化器或额外重计算。

Notes:
- 对应 Word：三.Backward 6 反向 FLOPs 汇总。
- 段落索引（从 0 起）：[235, 241]。
- 讲述：采用 2MKN 近似且输入、权重都需要梯度，排除 RMSNorm、优化器与额外重计算。单 token 精确外积没有跨行求和，因此精确反向不严格等于两倍；见核对报告。
- 点击顺序：
- 1. 前向一次矩阵乘法 F
- 2. 反向分别计算 dW 和 dX
- 3. 汇总 Query Linear Backward 193.99 MFLOPs
- 本页完成后再次点击切换下一页。

## Slide 19: (No title)

Body:
- 四.1–3 概念与屋顶线
- Roofline：两条上限，共同限定可达性能
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 19 / 22
- Performance · TFLOP/s
- Arithmetic Intensity · FLOPs/Byte
- 1
- 10
- 100
- 1000
- 1
- 10
- 100
- 1000
- AI = FLOPs / Bytes
- AI × BW
- 989 TFLOP/s
- Ridge ≈ 295.2
- P_peak / BW
- Memory-bound
- Compute-bound
- P ≤ min(P_peak, AI × BW)
- H100 SXM · BF16 Dense：989 TFLOP/s，3.35 TB/s；双对数坐标；理论上界。

Notes:
- 对应 Word：四.1–3 概念与屋顶线。
- 段落索引（从 0 起）：[243, 266]。
- 讲述：双对数坐标。斜线与水平线的较小值构成屋顶；点的纵坐标是模型给出的性能上界，不是实测。H100 SXM BF16 Dense 989 TFLOP/s，3.35 TB/s。
- 点击顺序：
- 1. 只显示 Arithmetic Intensity 与 Performance 坐标轴
- 2. 绘制 Memory Bandwidth Limit：AI × BW
- 3. 加入 Peak Compute 水平线
- 4. 标出 Ridge Point = Peak / BW
- 5. 标注 Memory-bound 与 Compute-bound
- 本页完成后再次点击切换下一页。

## Slide 20: (No title)

Body:
- 四.3–5 Query 流量与 Batch
- 权重复用提高 Arithmetic Intensity，点向右移动
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 20 / 22
- Performance · TFLOP/s
- Arithmetic Intensity · FLOPs/Byte
- 1
- 10
- 100
- 1000
- 1
- 10
- 100
- 1000
- Memory-bound
- Compute-bound
- BF16 · 2 Bytes
- 权重按批次读一次
- H 写出一次、读回一次
- M=1
- 96.99 MFLOPs
- 97.07 MB · AI≈1
- M=1024
- 99.32 GFLOPs
- 179.83 MB
- AI≈552.3
- M ↑
- 只含两个 Query Linear；简化流量模型，非实测；M 为一次共同处理的行数。

Notes:
- 对应 Word：四.3–5 Query 流量与 Batch。
- 段落索引（从 0 起）：[267, 285]。
- 讲述：F(M)=96,993,280M；B(M)=96,993,280+80,896M。每个权重矩阵按批次读取一次，中间 H 写一次读一次；不含 RMSNorm、真实缓存与融合。M 是一次共同处理的行数，不是同一请求未来 token 可任意并行。
- 点击顺序：
- 1. 显示 BF16 权重和激活流量拆分
- 2. 显示 M=1：AI≈1，Memory-bound
- 3. 显示 M=1024：AI≈552.3，Compute-bound
- 4. 显示向右移动轨迹与优化方向
- 本页完成后再次点击切换下一页。

## Slide 21: (No title)

Body:
- 五.MFU 基本公式与示例
- MFU：把有效模型工作量与可用峰值放在同一尺度
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 21 / 22
- Illustrative Example / Not Measured
- 01  Useful Model FLOPs / Step
- 100.6M tokens × 60 GFLOPs/token
- = 6.036 EFLOPs
- 02  Total Hardware Peak Throughput
- 2048 GPUs × 989.5 TFLOP/s
- = 2.026496
- EFLOP/s
- 03  MFU = 6.036 / (6.0 × 2.026496)
- 49.6%
- 60 GFLOPs/token、2048 GPUs、6.0 s：假设；989.5：示例规格

Notes:
- 对应 Word：五.MFU 基本公式与示例。
- 段落索引（从 0 起）：[286, 320]。
- 讲述：Illustrative Example / Not Measured。报告 batch 100.6M tokens；60 GFLOPs/token、2048 GPUs 和 6 s 均为假设；989.5 为示例硬件规格。60 GFLOPs 不是已验证的全模型训练值。MFU 分子不计额外激活重计算；更高 MFU 不自动表示质量更高或总成本更低。
- 点击顺序：
- 1. Useful FLOPs：100.6M × 60 GFLOPs = 6.036 EFLOPs
- 2. Total Peak：2048 × 989.5 TFLOP/s = 2.026496 EFLOP/s
- 3. Available FLOPs：乘 6 s；进度条自动在约 0.9 秒内填到 49.6%
- 本页完成后再次点击切换下一页。

## Slide 22: (No title)

Body:
- 五.6–7 综合与总结
- 沿着计算路径，把成本与效率接起来
- 输入
- 权重
- 输出
- 传递梯度
- 权重梯度
- 22 / 22
- Architecture
- Parameters
- 哪些路径执行
- 哪些权重被访问
- FLOPs
- 前向 + 反向
- 声明范围与计数口径
- Efficiency
- Roofline：潜在瓶颈    MFU：有效训练吞吐 / 总峰值

Notes:
- 对应 Word：五.6–7 综合与总结。
- 段落索引（从 0 起）：[321, 328]。
- 讲述：Roofline 看潜在算力 / 带宽上限，MFU 结合有效模型 FLOPs 与真实吞吐量。仍需 profiling 核实端到端瓶颈；教学估算不是实测。
- 点击顺序：
- 1. 架构决定执行路径与可学习权重
- 2. 前向与反向决定声明范围的 FLOPs
- 3. Roofline 解释潜在瓶颈，MFU 衡量训练系统效率
- 本页完成后再次点击切换下一页。
