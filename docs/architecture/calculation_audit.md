# 计算正确性审计

官方固定版本：`5254448103b6254fc646dacbca1cbbf20e9216c1`。先按配置和实际构造权重建立独立总账，再对比旧页面。未用旧 HTML 数值证明实现正确。

状态统计：{"Verified": 63, "Estimated": 82, "Unverified": 44, "Mismatch": 0, "historicalMismatchesCorrected": 38}。统计单位为 63 个页面模块 × 3 种模式，共 189 个主记录；参数家族 57 个另附 Verified 源码证据。Mismatch 为修正历史，当前已解决项使用修正后状态。

## 已发现并修正的问题

| Module | Item | Previous | Corrected | Status | Evidence |
|---|---|---|---|---|---|
| indexer | 索引 Key 和 Head 汇合 | 只展示 Query×Key 点积 | 补齐 W_IK / γ、新 latent 的条件、ReLU 与加权求和 | Mismatch → Verified | 源码 Indexer.forward 明确执行这些阶段；选址与主 Attention 分别统计。<br>[inference/model.py:527 · Indexer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L527) |
| compressor | 归一化权重的使用阶段 | γ 只列数量，流程未显示 | 补齐完整分组产生后才执行的归一化 | Mismatch → Verified | 半组 Decode 不执行该阶段；不能误认 γ 不学习或每步都执行。<br>[inference/model.py:458 · Compressor.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L458) |
| hc | base / scale 的明确用途 | 只在文字中列出 base / scale | 补齐 pre / post / comb 的参数变换 | Mismatch → Verified | 核对 hc_split_sinkhorn 的缩放、偏置、epsilon 与先行后列次序。<br>[inference/model.py:950 · Block.hc_mixes](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L950) |
| qnorm | RMSNorm 4d+2 的解释 | 平方、求和、两次逐项缩放共约 4×1280 次乘加；求均值的除法和开平方倒数各简化计 1 次。 | 平方 d 次、求和 d−1 次、两次逐项缩放 2d 次；均值除法、加 ε、rsqrt 共 3 次，合计 4d+2。特殊函数简化记 1 次。 | Mismatch → Estimated | 所有五类 RMSNorm 面板保留正确数值；说明独立列出 d−1 求和、均值除法、epsilon 与 rsqrt，避免口头漏数。<br>[inference/model.py:288 · RMSNorm.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L288) |
| sliding | 明确 sink 的归一化计算 | P 未写出归一化关系 | 分母包含 exp(a_sink)，不附加 Value 行 | Mismatch → Verified | 显式绑定可学习 sink，不误标为零参数。<br>[inference/kernel.py:311 · sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/kernel.py#L311) |
| sparse | 明确 sink 的归一化计算 | P 未写出归一化关系 | 分母包含 exp(a_sink)，不附加 Value 行 | Mismatch → Verified | 显式绑定可学习 sink，不误标为零参数。<br>[inference/kernel.py:311 · sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/kernel.py#L311) |
| multi | 明确 sink 的归一化计算 | P 未写出归一化关系 | 分母包含 exp(a_sink)，不附加 Value 行 | Mismatch → Verified | 显式绑定可学习 sink，不误标为零参数。<br>[inference/kernel.py:311 · sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/kernel.py#L311) |
| encoderattn | 明确 sink 的归一化计算 | P 未写出归一化关系 | 分母包含 exp(a_sink)，不附加 Value 行 | Mismatch → Verified | 显式绑定可学习 sink，不误标为零参数。<br>[inference/kernel.py:311 · sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/kernel.py#L311) |
| decoderattn | 明确 sink 的归一化计算 | P 未写出归一化关系 | 分母包含 exp(a_sink)，不附加 Value 行 | Mismatch → Verified | 显式绑定可学习 sink，不误标为零参数。<br>[inference/kernel.py:311 · sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/kernel.py#L311) |
| engram | Engram 门控权重 | 仅文字概括门值 | 补齐 q_weight / k_weight 的加权归一化点积与 signed sqrt | Mismatch → Verified | 实际 Engram.forward 使用这两组学习权重；不是普通 RMSNorm γ。<br>[inference/model.py:350 · Engram.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L350) |
| engram1 | Engram 门控权重 | 仅文字概括门值 | 补齐 q_weight / k_weight 的加权归一化点积与 signed sqrt | Mismatch → Verified | 实际 Engram.forward 使用这两组学习权重；不是普通 RMSNorm γ。<br>[inference/model.py:350 · Engram.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L350) |
| engram14 | Engram 门控权重 | 仅文字概括门值 | 补齐 q_weight / k_weight 的加权归一化点积与 signed sqrt | Mismatch → Verified | 实际 Engram.forward 使用这两组学习权重；不是普通 RMSNorm γ。<br>[inference/model.py:350 · Engram.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L350) |
| enggate | Engram 门控权重 | 仅文字概括门值 | 补齐 q_weight / k_weight 的加权归一化点积与 signed sqrt | Mismatch → Verified | 实际 Engram.forward 使用这两组学习权重；不是普通 RMSNorm γ。<br>[inference/model.py:350 · Engram.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L350) |
| hidden | 状态与投影的区别 | 状态面板直接展示 KV 投影 | 状态合并、归一化；KV 分支独立统计 | Mismatch → Verified | 防止将 512 维 KV 当成 Encoder 隐藏状态。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| query | Query 含 RMSNorm | [['本模块存储', '48,496,640 个参数']] | [('本模块存储', '48,497,920 个参数')] | Mismatch → Verified | 模块包含的 1,280 个 γ 原本只在子面板计数，主模块总量漏列；正确每层 48,497,920。<br>[inference/model.py:766 · Attention.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L766) |
| hc | comb 方向与 pre 使用次序 | 旧版公式/流程（完整内容见 audit_results.json） | 修正后的公式/范围（完整内容见 audit_results.json） | Mismatch → Verified | hc_post 对 comb 的输入路求和，正确 Σ_i R_i C_ij；本次 pre 交给下一子模块。<br>[inference/model.py:950 · Block.hc_mixes](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L950) |
| hc | 路间梯度与归一化路径 | 旧版公式/流程（完整内容见 audit_results.json） | 修正后的公式/范围（完整内容见 audit_results.json） | Mismatch → Estimated | 改为 dR_i=Σ_j C_ij dY_j；投影小计不代表全部 HC 梯度。<br>[inference/model.py:950 · Block.hc_mixes](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L950) |
| moe | 一层 MoE 边界 | [['本模块存储', '13,624,934,400 个参数']] | [('一层 MoE · 全部逻辑权重', '13,626,901,248 个参数'), ('当前 token · 激活权重', '249,692,544 个参数')] | Mismatch → Verified | 旧主结果只计专家；新增明确含 Router 的 Total / Activated，专家小计仍可在表中查看。<br>[inference/model.py:891 · MoE.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L891) |
| encoder | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 0 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| decoder | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 20 层实际拥有的权重', '13,762,606,326 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer0 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 0 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer1 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 1 层实际拥有的权重', '112,217,418,358 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer2 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 2 层实际拥有的权重', '13,765,227,766 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer13 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 13 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer14 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 14 层实际拥有的权重', '112,230,825,718 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer19 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 19 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer20 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 20 层实际拥有的权重', '13,762,606,326 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer21 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 21 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer22 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 22 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer33 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 33 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer34 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 34 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| layer39 | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 39 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| encoderblock | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 0 层实际拥有的权重', '13,754,511,990 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| decoderblock | 层级统计边界 | [['本模块存储', '13,754,511,990 个参数']] | [('第 20 层实际拥有的权重', '13,762,606,326 个参数')] | Mismatch → Verified | 代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。<br>[inference/model.py:970 · Block.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L970) |
| encoderattn | 完整 Attention 与读取子操作 | [['比较与汇总', '不新增投影权重'], ['每层的 64 个 sink 数字', '64 个可学习参数']] | [('代表层完整 Attention 权重', '126,617,408 个参数')] | Mismatch → Verified | 旧标题指整个 Attention，统计却只显示 kernel 的 sink；现在明确完整参数、读取 FLOPs 的不同边界。<br>[inference/kernel.py:311 · sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/kernel.py#L311) |
| decoderattn | 完整 Attention 与读取子操作 | [['比较与汇总', '不新增投影权重'], ['每层的 64 个 sink 数字', '64 个可学习参数']] | [('代表层完整 Attention 权重', '134,711,744 个参数')] | Mismatch → Verified | 旧标题指整个 Attention，统计却只显示 kernel 的 sink；现在明确完整参数、读取 FLOPs 的不同边界。<br>[inference/kernel.py:311 · sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/kernel.py#L311) |
| overview | 全模型范围 | [['完整主模型存储', '748,494,684,784 个参数']] | 修正后的公式/范围（完整内容见 audit_results.json） | Mismatch → Verified | 原 748,494,684,784 只为 Text + Engram；补齐 Draft、Vision、完整逻辑权重范围。<br>[inference/model.py:1245 · Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245) |
| overview | 当前一步的统计范围 | [['此处已有 1024 个历史 token 时，全模型当前一步', '约 35,584,521,265 次操作']] | [('Text + Engram · 当前一步简化估计', '35,584,521,265 次操作')] | Mismatch → Estimated | 独立按 kernel 逐项核对，原操作总数一致；修正“全模型”边界：这里只是 Text + Engram，未执行 Vision / Draft。64 位块归约为 63 次加法，分母更新另有乘、加两次。<br>[inference/model.py:1245 · Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245) |

## 所有页面模式记录

| Module | Parameters | Forward | Backward | Scope / Evidence |
|---|---|---|---|---|
| kv | Verified | Estimated | Estimated | [Attention.wkv](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L644)；参数拥有关系与 FLOPs 子操作边界见页面 |
| qa | Verified | Estimated | Estimated | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| qb | Verified | Estimated | Estimated | [Attention.wq_b](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L643)；参数拥有关系与 FLOPs 子操作边界见页面 |
| head | Verified | Estimated | Estimated | [ParallelHead.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1008)；参数拥有关系与 FLOPs 子操作边界见页面 |
| qnorm | Verified | Estimated | Unverified | [Attention.q_norm](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L642)；参数拥有关系与 FLOPs 子操作边界见页面 |
| kvnorm | Verified | Estimated | Unverified | [Attention.kv_norm](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L645)；参数拥有关系与 FLOPs 子操作边界见页面 |
| anorm | Verified | Estimated | Unverified | [Block.attn_norm](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L935)；参数拥有关系与 FLOPs 子操作边界见页面 |
| fnorm | Verified | Estimated | Unverified | [Block.ffn_norm](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L936)；参数拥有关系与 FLOPs 子操作边界见页面 |
| finalnorm | Verified | Estimated | Unverified | [Transformer.norm](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1208)；参数拥有关系与 FLOPs 子操作边界见页面 |
| query | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| output | Verified | Estimated | Estimated | [Attention.wo_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L646)；参数拥有关系与 FLOPs 子操作边界见页面 |
| rope | Verified | Estimated | Estimated | [Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245)；参数拥有关系与 FLOPs 子操作边界见页面 |
| sliding | Verified | Estimated | Unverified | [Attention.attn_sink](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L640)；参数拥有关系与 FLOPs 子操作边界见页面 |
| sparse | Verified | Estimated | Unverified | [Attention.attn_sink](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L640)；参数拥有关系与 FLOPs 子操作边界见页面 |
| multi | Verified | Estimated | Unverified | [Attention.attn_sink](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L640)；参数拥有关系与 FLOPs 子操作边界见页面 |
| compressor | Verified | Estimated | Unverified | [Compressor.wkv](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L446)；参数拥有关系与 FLOPs 子操作边界见页面 |
| global | Verified | Estimated | Unverified | [Compressor.wkv](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L446)；参数拥有关系与 FLOPs 子操作边界见页面 |
| indexer | Verified | Estimated | Unverified | [Indexer.wq_b](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L514)；参数拥有关系与 FLOPs 子操作边界见页面 |
| hc | Verified | Estimated | Unverified | [Block.hc_attn_fn](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L943)；参数拥有关系与 FLOPs 子操作边界见页面 |
| router | Verified | Estimated | Unverified | [Gate.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L807)；参数拥有关系与 FLOPs 子操作边界见页面 |
| weights | Verified | Estimated | Unverified | [Gate.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L811)；参数拥有关系与 FLOPs 子操作边界见页面 |
| selection | Verified | Estimated | Unverified | [Gate.bias](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L808)；参数拥有关系与 FLOPs 子操作边界见页面 |
| expert | Verified | Estimated | Estimated | [Expert.w1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L838)；参数拥有关系与 FLOPs 子操作边界见页面 |
| shared | Verified | Estimated | Estimated | [Expert.w1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L838)；参数拥有关系与 FLOPs 子操作边界见页面 |
| swiglu | Verified | Estimated | Unverified | [Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245)；参数拥有关系与 FLOPs 子操作边界见页面 |
| combine | Verified | Estimated | Estimated | [MoE.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L891)；参数拥有关系与 FLOPs 子操作边界见页面 |
| moe | Verified | Estimated | Estimated | [Gate.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L807)；参数拥有关系与 FLOPs 子操作边界见页面 |
| input | Verified | Estimated | Estimated | [Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245)；参数拥有关系与 FLOPs 子操作边界见页面 |
| tokenizer | Verified | Estimated | Estimated | [Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245)；参数拥有关系与 FLOPs 子操作边界见页面 |
| ids | Verified | Estimated | Estimated | [Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245)；参数拥有关系与 FLOPs 子操作边界见页面 |
| embedding | Verified | Estimated | Unverified | [ParallelEmbedding.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L166)；参数拥有关系与 FLOPs 子操作边界见页面 |
| ngram | Verified | Estimated | Estimated | [Engram.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L350)；参数拥有关系与 FLOPs 子操作边界见页面 |
| hash | Verified | Estimated | Estimated | [Engram.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L350)；参数拥有关系与 FLOPs 子操作边界见页面 |
| engembed | Verified | Estimated | Unverified | [ParallelEngramEmbedding.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L309)；参数拥有关系与 FLOPs 子操作边界见页面 |
| enggate | Verified | Estimated | Unverified | [Engram.wkv](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L345)；参数拥有关系与 FLOPs 子操作边界见页面 |
| injection | Verified | Estimated | Unverified | [Engram.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L350)；参数拥有关系与 FLOPs 子操作边界见页面 |
| engram1 | Verified | Estimated | Unverified | [ParallelEngramEmbedding.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L309)；参数拥有关系与 FLOPs 子操作边界见页面 |
| engram14 | Verified | Estimated | Unverified | [ParallelEngramEmbedding.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L309)；参数拥有关系与 FLOPs 子操作边界见页面 |
| engram | Verified | Estimated | Unverified | [ParallelEngramEmbedding.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L309)；参数拥有关系与 FLOPs 子操作边界见页面 |
| streams | Verified | Estimated | Estimated | [Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245)；参数拥有关系与 FLOPs 子操作边界见页面 |
| hidden | Verified | Estimated | Unverified | [Block.attn_norm](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L935)；参数拥有关系与 FLOPs 子操作边界见页面 |
| localstates | Verified | Estimated | Estimated | [Attention.wkv](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L644)；参数拥有关系与 FLOPs 子操作边界见页面 |
| logits | Verified | Estimated | Estimated | [ParallelHead.weight](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1008)；参数拥有关系与 FLOPs 子操作边界见页面 |
| prediction | Verified | Estimated | Estimated | [Transformer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L1245)；参数拥有关系与 FLOPs 子操作边界见页面 |
| encoder | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| decoder | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer0 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer1 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer2 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer13 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer14 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer19 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer20 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer21 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer22 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer33 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer34 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| layer39 | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| encoderblock | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| decoderblock | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| encoderattn | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| decoderattn | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |
| overview | Verified | Estimated | Unverified | [Attention.wq_a](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py#L641)；参数拥有关系与 FLOPs 子操作边界见页面 |

## 逻辑权重总账

| 范围 | 权重数 |
|---|---:|
| Text Backbone | 551,566,180,464 |
| Engram | 196,928,504,320 |
| Draft | 14,225,362,530 |
| Vision | 485,268,480 |
| Text + Engram | 748,494,684,784 |
| All logical weights | 763,205,315,794 |

数量与官方约 552B 文本 Backbone、约 196B Engram、模型文件约 763B 的分范围说明一致。此处统计逻辑权重，不是量化字节、scale 元数据或给定训练任务的 requires_grad 数量。

DSpark 的主 Embedding / LM Head 是引用；唯一总账只计一次。三个草稿阶段使用 128 routed / Top-3；首阶段 main_proj、末阶段 norm、Markov 表/头和 confidence 单独枚举。Vision 从官方 vision.py 核对，包含默认 nn.Linear Bias、32 个 Block、PatchEmbed、最终 RMSNorm、Aligner 和三个图像边界向量。

## 未验证与估算边界

所有数值前向 FLOPs 为明确口径下的 Estimated。参考代码未给训练反向，线性 dX / dW 与已复核非线性梯度是数学推导；完整 HC、RMSNorm、Compressor、Indexer、Engram、Attention、MoE 等反向总量保留 Unverified。没有编造全模型训练反向。

单 token 的 dW 是外积：精确 KN，常用 2KN 只为乘加近似。一般线性精确 MN(2K−1)，不是把近似乘 2 当作精确反向。

特殊操作估计把 rsqrt、exp、sigmoid、softplus、sqrt、除法各记 1 次，忽略比较、Top-K、整数运算、搬运和通信；仅明确计入 Engram 查表反量化的逐项乘法。量化 GEMM、图片或草稿完整执行耗时未验证。

本地 model.py 与官方固定版本有两类差异：Indexer 在没有新 latent 的步骤仍应发布自身索引缓存；Attention / DSpark 的分组输出投影前还执行一次 act_quant。逻辑参数维度不变，本审计以固定官方实现为证据，未改动用户模型源码。

## 可重复验证

运行 `python src/deepseek/audit_model.py` 和 `python src/deepseek/verify_audit.py`。独立总账在 `src/deepseek/independent_audit.json`；源码下载校验和及固定 URL 在 `references/deepseek/manifest.json`；完整修正清单在 `src/deepseek/audit_results.json`。

[正式配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/config.json) · [官方推理实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/5254448103b6254fc646dacbca1cbbf20e9216c1/inference) · [模型说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/README.md)

## 技术报告交叉核验

同 revision 的 51 页技术报告已保存 PDF、提取文本和 SHA256。§2.2 的 CED 与 §2.3 的 CSA2 共同说明分段与跨层 KV / 选址复用；§4.2.1 的正式维度与专家配置和已核对的 config / 构造源码一致。CED 的通用层相关投影表达，与最终 CSA2 部署中的 Full / Reindex / Reuse 配置应分别理解；共享的权重创建以实际源码和 source layer 配置枚举。

§2.5 说明预训练早期 Vision 主体冻结，最终归一化与对齐层保持可训练；学习率衰减阶段解冻。其余关于用户当前任务 requires_grad、完整训练反向内核和量化更新方式的未知项仍保留 Unverified。报告的 8B / 16B 为其激活口径，本页 MoE Activated 是单 token 6+1 个专家与文本 Router 的明确子范围，不能互相替代。

[技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/DeepSeek_V41_Tech_Report.pdf)；本地可核验快照 `references/deepseek/technical_report_manifest.json`。
