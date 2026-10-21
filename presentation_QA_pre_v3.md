# 演讲问答准备 v3：100 道高难度技术答辩问题

本文件全面改写 `presentation_QA_pre_v2.md`，从"能否复述定义"升级为"能否顶住持续追问"。参考材料：本仓库 `references/deepseek/config.json`、`references/deepseek/inference/model.py`（含 `engram.py`/`kernel.py` 接口注释）、`references/deepseek/technical_report.txt`（DeepSeek V4.1 Flash 技术报告，arXiv:2609.19969）、`docs/architecture/assumptions.md`、`docs/architecture/calculation_audit.md`、`deepseek_v4_1_interactive.html`（含 Shape Trace）、`compute-efficiency/calculations.js`/`app.js`，以及 `presentation_QA_pre_v2.md` 已核实的原有问答。

**难度标记：** L1 = 理解设计动机；L2 = 解释实现细节、张量形状与计算；L3 = 质疑假设、指出边界与反例。**★** 标注本文件认为最优先练习的 20 题。每题先给"现场回答"（20–45 秒口语），"追问补充"用于被继续追问时展开，"依据"给出可核查的仓库位置；缺乏证据的地方明确标注为假设或推断，不编造数值。

## 目录

| 节 | 范围 | 题量 | 对应原文件 |
|---|---|---:|---|
| 一、架构与设计决策 | Q01–Q25 | 25 | 深化原 Q01–Q12 |
| 二、张量形状、数据流与模块内部 | Q26–Q45 | 20 | 新增，依据 Shape Trace 与 model.py |
| 三、参数量与 MoE | Q46–Q60 | 15 | 深化原 Q13–Q18 |
| 四、前向与反向 FLOPs | Q61–Q75 | 15 | 深化原 Q19–Q33 |
| 五、Roofline 与硬件性能 | Q76–Q90 | 15 | 深化原 Q34–Q40 |
| 六、MFU 与端到端性能 | Q91–Q100 | 10 | 深化原 Q41–Q50 |
| 附录 A | 20 道最重要问题索引 | — | — |
| 附录 B | 5 条真实追问链 | — | — |
| 附录 C | 10 个最薄弱环节 | — | — |

---

## 一、架构与设计决策（Q01–Q25）

### Q01 ★ L1 为什么用 Causal Encoder–Decoder（CED），而不是普通 Decoder-only？

**现场回答：** "CED 把计算拆成两段：前 20 层 Encoder 处理输入，构建一份压缩的 Global KV；后 20 层 Decoder 复用这份 KV 生成输出。两段都遵守因果约束，不是传统双向 Encoder。好处是 Decoder 不必让每一层都重新从头扫一遍输入，Prefill 阶段的激活参数因此明显更少。"

**追问补充：** 技术报告明确给出数字：Prefill 激活 8B 参数，Decode 激活 16B 参数（同一个 552B 骨干模型）。这不是"模型变小了"，是同一组参数在不同阶段被访问的比例不同。

**依据：** `references/deepseek/technical_report.txt:246-247`（"activate 8B parameters per token during prefill and 16B during decode"）；`references/deepseek/config.json: num_hidden_layers=40`。

### Q02 L2 两段都是因果计算，那 Encoder 和 Decoder 的执行到底哪里不一样？

**现场回答：** "区别不在'能不能看到未来'，两段都不能。区别在于 Decoder 的 Global KV 不是自己从头生成的，而是直接取 Encoder 最后一层（第 20 层，也就是 L/2 层）的隐藏状态投影得到。Decoder 自己的层只需要用这份 KV，不需要每层重新跑一遍压缩。"

**追问补充：** 报告原文写的是"decoder global KV is projected from the final encoder hidden states"，即第 20 层（0 索引下 Encoder 的最后一层）既是 Encoder 的终点，又是 Decoder 侧 Global KV 的源头。

**依据：** `references/deepseek/technical_report.txt:245,676`；本仓库配置中 `kv_source_layer_ids` 含 20。

### Q03 L2 Prefill 阶段，哪些层处理全部输入 token？哪些操作被省掉了？

**现场回答：** "Prefill 时输入的每个 token 都要走完 Encoder 的 20 层——这部分不省。省的是 Decoder 侧：Decoder 不需要对每个输入 token 重新生成一份 Global KV，只有被指定为 KV 源的层（这里是第 20 层）生成一次，其余 Decoder 层直接复用或按 Reindex/Reuse 规则处理。"

**追问补充：** Local/SWA 部分不受影响——报告说 DeepSeek-V4.1-Flash 在每一层都用 Sliding-Window Attention，Prefill 仍要为窗口内的位置做局部计算。

**依据：** `references/deepseek/technical_report.txt:276-277`；`model.py: Attention._window_kv`。

### Q04 L2 这个计算优势在自回归 Decode 阶段还在吗？

**现场回答：** "在，但形式不同。Decode 阶段当前 token 仍然要走完整 40 层主干——不能把 Prefill 的省算比例直接套到生成阶段。Decode 省的地方是：大多数层不用重新生成或重新索引 Global KV，而是复用已有缓存，所以 Decode 激活参数是 16B，比骨干全量参数少，但比 Prefill 的 8B 多。"

**追问补充：** 16B 对应"当前 token 需要经过的、真正参与计算的参数"；552B 是骨干总参数量，二者衡量的是不同的事。

**依据：** `references/deepseek/technical_report.txt:246-247`。

### Q05 ★ L3 怎么量化这个优势，而不是只做定性描述？

**现场回答：** "报告的 Figure 2 给出了具体曲线：把 BF16、FP8、FP4 运算分别按 1、0.5、0.25 加权计数单 token Decode FLOPs，结果是上下文从 4K 扩到 1M（256 倍），Decode FLOPs 只增加约 1/4。这是论文自己的量化方式，不是我现场编的比例。"

**追问补充：** 这个量化口径和本仓库教学页面用的 2MKN 未加权 FLOPs 计数不是同一套方法——如果要把两者放在一起比较，必须先说清楚各自的加权规则（见 Q66）。

**依据：** `references/deepseek/technical_report.txt:271-274,297-298`。

### Q06 L1 Local Attention 已经省算，为什么还需要压缩后的 Global KV？

**现场回答：** "Local 窗口只覆盖最近 128 个位置，窗口外的长距离信息完全看不到。Global KV 走另一条路径，把更早的上下文压缩保存，再配合 Indexer 稀疏挑选，和 Local KV 一起参与同一次 Attention 计算。两者互补，不是二选一。"

**依据：** `references/deepseek/config.json: sliding_window=128`；`model.py: Attention.forward` 中 `kv = torch.cat([kv, compress_kv])`。

### Q07 L2 CSA2 和上一代 DeepSeek-V4 的 CSA-HCA 混合架构有什么本质区别？

**现场回答：** "V4 用的是 CSA 和 HCA 两种注意力机制混合；V4.1 Flash 全部换成纯 CSA2，一种机制贯穿所有层，用 Full / Reindex / Reuse 三种静态模式区分每层的职责，不再需要两套不同的注意力实现并存。"

**追问补充：** 报告原话是"different from DeepSeek-V4 that employs the CSA–HCA hybrid architecture, DeepSeek-V4.1-Flash uses pure CSA2"。

**依据：** `references/deepseek/technical_report.txt:264-266`。

### Q08 L2 compress_ratio=2 和 ratio=1，在代码里到底是两条什么样的执行路径？

**现场回答：** "ratio=1 时 Compressor 只是一次线性投影加 RMSNorm，直接返回，没有门控、没有等待。ratio=2 时要先攒够 2 个 token 的 KV 和 gate 分数，用 softmax 门控把两个位置池化成一个，期间用 fp32 计算；如果当前这组还没攒满，这一步直接返回 None，Decode 要等下一步才产出。"

**追问补充：** ratio>1 的权重会被提升到 fp32，因为门控 softmax 在 fp32 下算；ratio=1 保持 checkpoint 自带精度，不需要额外提升。

**依据：** `model.py: Compressor.__init__/forward`（`if ratio == 1: return self.norm(self.wkv(x))`；`compress_ratio>1` 分支的 `kv_state`/`score_state`）。

### Q09 L3 为什么第 0/1 层没有压缩比，Encoder 用 2，Decoder 用 1？

**现场回答：** "配置里 compress_ratios 前两项是 0，表示这两层只有 Local 窗口，没有 Global 分支——这是 shape_trace.md 里明确标注的'Local only'。Encoder 源层（2/8/14）用 ratio=2，是两个 token 池化成一个压缩位置；Decoder 的 20 层用 ratio=1，每个 token 直接产出一行，不做二合一池化。这是训练时确定的设计取舍，我能读出代码行为，但不能从源码反推'为什么选 2 和 1 而不是别的数'的训练动机。"

**追问补充：** 不要把这里的"0"理解成"没有这一层"——层 0/1 仍然正常执行 Local Attention 和其余子模块，只是没有 Global 分支。

**依据：** `references/deepseek/config.json: compress_ratios` 数组前两项为 0；`shape_trace.md:11`。

### Q10 ★ L1 Full、Reindex、Reuse 三种模式，各自精确复用了什么、重新计算了什么？

**现场回答：** "Full 模式自己生成 Global KV、自己的 Indexer Key，并做一次完整打分和 Top-K 选择。Reindex 模式复用前面某一层已经生成的 Global KV 和 Indexer Key，但用自己这一层的 Indexer Query 重新打分、重新选 Top-K。Reuse 模式连 Top-K 结果也直接复用，不跑自己的 Indexer，直接用这份选择做稀疏 Attention。三种模式下，当前层自己的 Query、Local KV 和输出投影都照常执行。"

**追问补充：** 代码里对应 `Attention.is_kv_source`（是否生成 Global KV）和 `Attention.is_index_source`（是否拥有自己的 Indexer）两个布尔标志的组合；Reuse 既不是 kv_source 也不是 index_source。

**依据：** `references/deepseek/technical_report.txt:259-263`；`model.py: Attention.__init__`（`is_kv_source`/`is_index_source`）、`Indexer.owns_k`。

### Q11 L1 在当前配置下，具体哪些层是 Full、哪些是 Reindex、哪些是 Reuse？

**现场回答：** "KV 源层是 2、8、14、20，这四层是 Full。Indexer 源层是 2、8、14、20、24、28、32、36——减去 KV 源层后，24、28、32、36 这四层是 Reindex：共享 Global KV，但自己重新打分选 Top-K。其余所有 Decoder 层，包括 21、22、23、25-27、29-31、33-35、37-39，都是 Reuse。"

**追问补充：** Encoder 侧（0-19 层）只有 2、8、14 在 `kv_source_layer_ids` 里，不含 20——20 是 Decoder 的 Full 源层，这一点容易说错。

**依据：** `references/deepseek/config.json: kv_source_layer_ids=[2,8,14,20], index_source_layer_ids=[2,8,14,20,24,28,32,36]`。

### Q12 L1 候选池（Hierarchical Sparse Indexer）为什么只在 Decoder 用，Encoder 不需要？

**现场回答：** "候选池解决的是'后续多个 Reindex 层反复对全部历史位置重新打分'这个开销，层数越多、上下文越长，这个重复打分成本越明显。Decoder 里从 20 层到 39 层有多个 Reindex 层，候选池能把每个后续 Indexer 的搜索范围从'随上下文长度线性增长'降到'固定大小'。Encoder 没有这么多层级联的 Reindex 场景，所以报告里明确说这个机制'仅用于 CED 的 Decoder'。"

**追问补充：** 候选池本身由第一个 Full 模式层（这里是第 20 层）构建：按块打分，选出得分最高的 2048 个块（每块 8 个位置），后续 Reindex 层只在这 16384 个候选位置里搜索，而不是扫全部可见位置。

**依据：** `references/deepseek/technical_report.txt:684-700`（"used only in the decoder of CED"；"selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions"）；配置 `candidate_topk_blocks=2048, candidate_block_size=8`。

### Q13 L1 SWA Bounded Replay 为什么只重放最近 n_win 个 token，而不是旧方案的 L×n_win 个？

**现场回答：** "旧的混合策略要精确重建每一层的 Local KV，需要重放最近 层数×窗口大小 个 token，层数越多代价越大。Bounded Replay 换了个思路：只重放最近 n_win 个 token 来近似重建所有层的 Local 状态，用少量精度损失换掉和层数成正比的重放成本，这样就能不把 Local KV 持久化到 SSD。"

**追问补充：** 报告说这个近似"incurs only negligible performance degradation"（基于他们的实验），这是作者的实验结论，我没有独立复现。

**依据：** `references/deepseek/technical_report.txt:278-287`。

### Q14 L2 Local 窗口 KV 和稀疏 Global KV 是怎么"合"在一次 Attention 里的？这为什么重要？

**现场回答：** "代码里是先把 Local 窗口 KV 和 Global 压缩 KV 在位置维拼接成一个张量，索引数组也对应拼接，然后整个送进同一次 `sparse_attn` 调用，做一次联合 Softmax——不是先分别算两次 Attention 再加权合并。这意味着 Local 和 Global 的分数在同一个归一化分母里竞争，包括 attention sink 也在这个联合 Softmax 里。"

**追问补充：** 如果误以为是"两次独立 Attention 再相加"，会漏掉联合归一化这一步——这会直接改变每个位置实际拿到的权重，不是等价实现。

**依据：** `model.py: Attention.forward`（`kv = torch.cat([kv, compress_kv], dim=1)`；`o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)`）。

### Q15 L2 Indexer 自己的 Query 和 Key 分别来自哪里，和 Main Attention 共享吗？

**现场回答：** "Indexer 的 Query 不是从原始输入 x 重新投影的，而是复用 Main Attention 已经算出来的低秩表示 qr（W_QA 的输出），再过 Indexer 自己的 `wq_b` 得到。Key 来自 Compressor 产出的、RoPE 之前的压缩 latent，经 Indexer 自己的 `wk` 和归一化得到——这是独立于 Main Attention 的 Key，但源头数据是共享的中间结果。"

**追问补充：** Indexer 和 Main Attention 是两套不同形状的 Query/Key（Indexer 用 index_n_heads=32、index_head_dim=128），不要以为它们在复用同一组 Q/K。

**依据：** `model.py: Indexer.forward` 参数 `qr`；`Indexer.wq_b`、`Indexer.wk`。

### Q16 L2 为什么 Indexer Key 必须在 RoPE 之前从 latent 取值，这个顺序错了会怎样？

**现场回答：** "代码注释直接写明：Indexer 需要没有旋转过的 latent，所以 Compressor 产出 latent 后，Indexer 要先用它生成 Index Key，然后 Attention 才对同一块存储做 RoPE 和量化覆盖。如果顺序反了，Indexer 拿到的就是已经旋转过的向量，打分依据的位置信息会和设计意图不一致。"

**追问补充：** 这不是性能优化，是正确性约束——latent 这块内存会被原地覆写，错过这个时间窗口就再也拿不到预 RoPE 的版本。

**依据：** `model.py: Compressor` 类文档字符串（"Pre-RoPE is deliberate: the indexer needs the unrotated form"）；`Attention._compress_kv` 中先调用 `_compress_topk_idxs` 再对 latent 做 RoPE。

### Q17 L1 mHC 用 4 路并行残差流，直觉上解决了什么问题？

**现场回答：** "普通 Transformer 的残差流只有一条，所有信息挤在一路里传递。mHC 把残差拆成 4 条并行的流，子模块（Attention/FFN）只处理其中混合出来的一路输入，输出再按学到的系数分配回 4 路——相当于给模型多开了几条信息传递的'车道'，子模块之间的信息路由方式是可学习的，不是固定的恒等残差。"

**追问补充：** 4 路不是 4 份独立参数，混合系数（Pre/Post/Comb）才是这一层新增的可学习部分，具体数学见 Q37/Q38。

**依据：** `references/deepseek/technical_report.txt:712-719`（公式 2：`X_{l+1}=B_l X_l + C_l F_l(A_l X_l)`）。

### Q18 ★ L2 Single-Pass mHC 到底改了什么，为什么能降低访存？

**现场回答：** "原始 mHC 的输入混合系数 A_l 要等本层全部归约完才能算出来，所以要对残差 X_l 读两遍：一遍做残差更新和系数预测，另一遍专门做输入混合，一共要读写 (4n+4)d 的激活量，是理论下界 (2n+2)d 的两倍。Single-Pass mHC 把输入混合用的系数从 A_l 改成上一个 block 产出的 A_{l-1}，这样输入混合就不用等本层归约完成，一次读写就能同时完成残差更新、系数预测和输入混合，Mega-mHC 融合核因此能做到和理论下界一样的 (n+1)d 读 + (n+1)d 写，比原来的实现减少一半激活访存。"

**追问补充：** 这个改动"shifts the input-mixing coefficients by one block"，报告说实验上这个偏移造成的性能下降可以忽略，这是作者的实验结论。预训练阶段仍沿用多核实现，只有偏移的系数索引变了；部署阶段才切换到融合的 Mega-mHC 核。

**依据：** `references/deepseek/technical_report.txt:720-767`（公式 3–6，"halving the activation memory traffic of our original implementation"）。

### Q19 L1 MoE 为什么既要路由专家又要常驻的共享专家？

**现场回答：** "路由专家负责学习专门化的能力，但每个 token 只选 6 个，容易丢掉一些所有 token 都用得到的通用模式。共享专家对每个 token 都执行，负责兜底这部分通用计算，让路由专家可以更专注做差异化的事情，不用重复学习通用部分。"

**依据：** `model.py: MoE.forward`（`y += self.shared_experts(x)`，对所有 token 执行）；配置 `n_shared_experts=1`。

### Q20 L2 这个模型的路由用的是 sqrtsoftplus，不是常见的 softmax 或 sigmoid，这是在做什么？

**现场回答：** "Gate 的打分函数按配置是 sqrtsoftplus：先 softplus(x) 保证非负且平滑，再开根号。这类似 softmax 竞争式归一化和 sigmoid 独立评分之间的一种中间选择，具体为什么选它属于训练期的设计决策，源码只告诉我它是怎么算的，不能替代论文里关于这个选择的消融实验结论。"

**追问补充：** 不要把 sqrtsoftplus 和 softmax 混为一谈去解释——它们数值行为不同，sqrtsoftplus 的输出不强制所有专家分数相加为 1。

**依据：** `references/deepseek/config.json: scoring_func="sqrtsoftplus"`；`model.py: Gate.forward`（`scores=F.softplus(scores).sqrt()`）。

### Q21 L2 Router 的选择偏置"只选专家、不缩放权重"，这个区分为什么重要？

**现场回答：** "代码里很明确：`indices=(scores+bias).topk(...)`用加了偏置的分数去选谁入选，但 `weights=scores.gather(...)`取权重时用的是没加偏置的原始分数。也就是说偏置能决定谁被选中，但不能影响选中之后这个专家对输出贡献多少。如果误以为偏置也参与了输出的加权混合，会错误地认为调节偏置能直接改变专家输出的强弱，实际上它只改变'谁入选'这一步。"

**追问补充：** 注释原话是"the bias picks experts but does not scale them: weights come from the raw scores"——这是无辅助损失（noaux_tc）路由的典型写法，偏置只是选择信号。

**依据：** `model.py: Gate.forward`（`indices = (scores + bias).topk(...)`；`weights = scores.gather(1, indices)`）。

### Q22 L1 Engram 的查表机制，架构层面是怎么工作的？

**现场回答：** "Engram 不靠内容相关性检索，而是把最近几个历史 token 组合成 n-gram，算哈希拿到表里的行号，查出对应的 embedding。查出来的向量再经过投影拆成 Key 和共享 Value，用当前隐藏状态和 Key 算一个匹配度当门控，门控决定这次查表结果往残差流里混入多少。"

**追问补充：** 这是局部 token 组合驱动的检索，和内容相关性驱动的 Attention 是两种不同的信息获取方式，互为补充不是替代。

**依据：** `references/deepseek/technical_report.txt:768-784`；`model.py: Engram.forward`。

### Q23 L3 Engram 的门控同时依赖当前状态和检索到的 Key，而不是固定标量，这给了模型什么固定门控给不了的东西？

**现场回答：** "如果门控是一个固定标量或者只依赖位置，模型没法区分'这次查到的记忆和当前上下文到底匹不匹配'。现在的门控是当前流 h 和检索 Key 的加权点积再过 sigmoid，相当于让模型自己判断这次查表结果可信不可信，匹配度低的时候可以把贡献压到接近 0，而不是每次都无条件混入同样比例。"

**追问补充：** 代码里有个细节：归一化是按'每个 token、每一路 hc 流'分别在 dim 维度上做的，不是跨 4 路联合归一化——这意味着 4 路残差流可以对同一次查表给出不同的信任程度。

**依据：** `model.py: Engram.forward`（`dot=(h*weight*key).sum(-1)*rstd*...`；注释"normalized per (token, hc copy) over dim, NOT jointly"）。

### Q24 L1 DSpark 为什么要单独一个训练阶段、把主干冻结，而不是像经典 MTP 那样联合训练？

**现场回答：** "报告原话说得很直接：和 DeepSeek-V3 的 MTP 模块在预训练全程联合训练不同，DSpark 是在预训练结束后单独开一个阶段训练，这个阶段只训练 DSpark、主干参数冻结；到了后训练阶段才继续和主干一起训练 DSpark，但依然不让 DSpark 的目标函数梯度回传进主干。目的是让草稿模块始终贴合当前策略，同时不让它的训练目标干扰主干本身的优化。"

**追问补充：** 这意味着主干训练和 DSpark 训练在时间上是分段的，不能假设"DSpark 的任何一次反向传播都会更新主干权重"。

**依据：** `references/deepseek/technical_report.txt:798-803`。

### Q25 ★ L3 架构这部分如果只能守住一个结论，你选哪个，为什么它最容易被攻破？

**现场回答：** "我会选'只用 1/3 总参数和 1/4 激活参数，就拿到比 DeepSeek-V4-Pro-Base 高 5%-10% 的评测结果'这条，因为它是报告里最亮眼也最容易被追问边界的一句话。它最脆弱的地方是：这是跟另一个不同规模的模型比较，不是同一模型在相同训练预算下的消融实验，所以不能直接说'架构改进单独贡献了这 5%-10%'，训练数据质量等其他变量也在报告里被提到会影响这个结果。"

**追问补充：** 守住这条的方法不是回避，而是主动承认比较对象和比较维度，把'架构更高效'和'全部收益都来自架构'这两件事分开说。

**依据：** `references/deepseek/technical_report.txt:318-321`。

---

## 二、张量形状、数据流与模块内部（Q26–Q45）

### Q26 L2 Query 的完整张量形状路径是什么？

**现场回答：** "单 token 输入 [1,5120] 先过 W_QA 得到 [1,1280]，RMSNorm 形状不变，再过 W_QB 展开到 [1,32768]，reshape 成 [1,64,512]（64 个 head，每个 512 维），最后对每个 head 最后 64 维做 RoPE，其余维度不变。"

**追问补充：** 这条路径和 compute-efficiency 页面、Shape Trace 页面的 Query Projection 算例完全一致，两次线性层对应 FLOPs ≈97 MFLOPs（见 Q63）。

**依据：** `references/deepseek/config.json: q_lora_rank=1280, n_heads=64, head_dim=512`；`model.py: Attention.forward` 前三行；`src/deepseek/shape_trace.js` 中 `qa`/`qn`/`qb`/`qreshape`/`qrope` 节点。

### Q27 L3 W_QA 为什么恰好是 5120→1280，这个 1280 是算出来的还是定下来的？

**现场回答：** "1280 是 config 里的 q_lora_rank，是训练前定好的超参数，不是从其他维度公式推导出来的。我能说的是它的作用：把 5120 维压缩成一个更小的低秩瓶颈，再展开到 32768，两段投影的参数量远小于直接从 5120 投影到 32768，但'为什么偏偏是 1280'这个具体取值没有公式依据，属于架构设计选择。"

**追问补充：** 不要把这类超参数包装成"有推导公式"，诚实说明它是配置值更稳妥。

**依据：** `references/deepseek/config.json: q_lora_rank=1280`。

### Q28 L2 W_QB 输出的 32768 怎么变成 64 个 head × 512 维？

**现场回答：** "32768 就是 64×512 的乘积，W_QB 输出一个扁平的 [1,32768] 向量后，用 `unflatten` 操作按最后一维重新排列成 [1,64,512]，只是改变张量的视图方式，不涉及额外的矩阵乘法或数据搬运（如果是连续内存的话）。"

**依据：** `model.py: Attention.forward`（`q=self.wq_b(qr).unflatten(-1,(self.n_local_heads,self.head_dim))`）。

### Q29 ★ L3 64 个 Query Head 读的是同一份 512 维 KV，不是每个 head 各有自己的 KV——这是不是实现漏洞？

**现场回答：** "不是漏洞，是设计。配置里 `num_key_value_heads=1`，`Attention.wkv` 本身就只投影出一个 512 维的 KV 向量，所有 64 个 Query Head 在 sparse_attn 里共享这同一份 KV 做计算。这属于类似 Multi-Head Latent Attention 的共享 KV 设计：KV 存储和计算成本不随 Query Head 数线性增长，是节省 KV Cache 体积的关键手段之一。"

**追问补充：** 正因为 KV 只有一份，Attention 部分的参数量里 KV 投影（`wkv`: 5120×512）比 Query 投影（`wq_a`+`wq_b` 共约 4850 万）小得多，二者不是对称的。

**依据：** `references/deepseek/config.json: num_key_value_heads=1`；`model.py: Attention.wkv = Linear(self.dim, self.head_dim)`。

### Q30 L2 RoPE 只作用在 head_dim 的尾部 64 维，为什么要分 rope 和 nope 两部分？

**现场回答：** "配置里 qk_rope_head_dim=64，head_dim=512，意味着每个 head 的 512 维里只有最后 64 维参与旋转位置编码，前面 512-64=448 维（nope 部分）不带位置信息，直接传递内容特征。这是把'携带相对位置关系'和'携带纯内容信息'的维度分开处理的常见做法，RoPE 不需要作用在全部维度上才有效。"

**依据：** `references/deepseek/config.json: qk_rope_head_dim=64`；`model.py: Attention.__init__`（`nope_head_dim=head_dim-rope_head_dim`）；`apply_rotary_emb(q[...,-rd:],...)`。

### Q31 L2 输出投影这一路的张量形状是什么，为什么要分 8 组？

**现场回答：** "Attention 输出 [1,64,512] 先做逆 RoPE，按组 reshape 成 [1,8,4096]（8 组，每组 4096=64×512/8），量化成 FP8，再过分组的 W_OA 投影到 [1,8,1024]（每组低秩 1024 维），flatten 成 [1,8192]，最后过 W_OB 投到 [1,5120]。分组是为了让输出投影也走低秩路径，同时把计算拆成 8 份并行处理，而不是一个巨大的稠密矩阵。"

**依据：** `references/deepseek/config.json: o_groups=8, o_lora_rank=1024`；`model.py: Attention.wo_a/wo_b/forward`；`src/deepseek/shape_trace.js` 中 `ogroup`/`oa`/`oflat`/`ob` 节点（与 `verify_shape_trace.cjs` 的硬编码断言一致）。

### Q32 L2 代码注释说 W_OA 是"按组块对角，所以用 einsum 不用 Linear"，这具体什么意思？用普通 nn.Linear 代替会出什么问题？

**现场回答：** "W_OA 存储上是一个 [8192,4096] 的张量，但使用时 reshape 成 [8 组,1024,4096]，用 einsum 'bsgd,grd->bsgr' 计算——每一组只和自己对应的 4096 维输入切片相乘，组与组之间没有交叉。如果直接用一个普通 nn.Linear(4096,8192) 对整个展平后的输入做矩阵乘法，会让每一组的输出同时依赖全部 8 组的输入，这和设计的'分组独立投影'语义不一致，参数量也会不同。"

**追问补充：** 这种写法本质上等价于 8 个独立的 [4096→1024] 矩阵，只是打包存成一个张量，省内存管理开销，不改变数学语义。

**依据：** `model.py: Attention.forward` 注释（"wo_a is block-diagonal over groups"）；`torch.einsum("bsgd,grd->bsgr", o, wo_a)`。

### Q33 L2 Local KV 环形缓存的形状和写入/读取规则是什么？

**现场回答：** "缓存形状是 [window_size, head_dim]=[128,512]。Decode 阶段每步只写一行：`window_kv_cache[start_pos % win]`，按位置取模循环覆盖最老的一行；Prefill 阶段如果输入长度超过窗口，也是用取模的方式只保留最后 128 个位置。读取时 Decode 直接读整个环形缓存（包含可能还没填满的占位），由索引数组标记哪些槽位是有效的。"

**依据：** `model.py: Attention._window_kv`（`self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`）；`get_window_topk_idxs`。

### Q34 L2 Global KV 缓存从 Compressor 输出到写入缓存，完整走了哪些步骤，量化规格是什么？

**现场回答：** "Compressor 输出 RoPE 前的 latent [ΔS,512]（ΔS 是这一步新增的行数，可能是 0），先给 Indexer 用来生成 Index Key，然后才对 latent 做 RoPE，再用 FP4 量化（MXFP4/E2M1 格式，每 16 个通道共享一个 E4M3 的 scale），最后写入 `compress_kv_cache`。"

**追问补充：** 量化分组大小是 16，不是 Indexer Key 用的 32——这是技术报告和源码都明确区分开的两套量化参数。

**依据：** `model.py: Attention._compress_kv`（`fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`）；`references/deepseek/technical_report.txt:814-818`（E2M1 格式，448×6=2688 动态范围）。

### Q35 L2 Indexer Key 缓存和 Global KV 缓存都来自同一个 latent，为什么量化分组大小不一样（16 对 32）？

**现场回答：** "代码注释明确写了：'Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0'。两条缓存承担的任务不同——Main Attention 的 Global KV 要参与最终的精细计算，精度要求更高一些，用更小的量化块（16）换精度；Indexer Key 只用来做一次粗筛打分，用更大的量化块（32）省存储，对精度的敏感度相对低。这是一个我能从代码读出行为、但背后权衡依据未必能在源码里完全验证的例子。"

**依据：** `model.py: Attention._compress_kv` 注释；`Indexer.forward`（`fp4_act_quant(k, fp4_block_size, True)`，`fp4_block_size=32`）。

### Q36 ★ L2 sparse_attn 把 Local 和 Global 合在一起时，具体拼接的是哪两个张量，又带了哪两个索引数组？

**现场回答：** "KV 张量是 Local 窗口 KV [≤128,512] 和 Global 压缩 KV [S,512] 在位置维拼接成 [≤128+S,512]。索引数组同样拼接：Local 侧的窗口索引和 Global 侧 Indexer 选出的 Top-K 索引（可能含 -1 表示还不可达），拼成一个 [实际读取数] 的索引列表，一起交给 sparse_attn 做一次逻辑读取和联合 Softmax。"

**依据：** `model.py: Attention.forward`（`kv=torch.cat([kv, compress_kv], dim=1)`；`topk_idxs=torch.cat([topk_idxs, compress_idxs], dim=-1)`）。

### Q37 L2 mHC 的 24 个混合系数，具体怎么从一次投影里拆出 pre/post/comb？

**现场回答：** "mix_hc=(2+hc_mult)×hc_mult=(2+4)×4=24。一次线性投影把展平的 4 路残差流映射成 24 维，再按 Sinkhorn 归一化拆成三部分：pre 是 [4] 个数（collapse 4 路到 1 路的权重），post 是 [4] 个数（子模块输出分配回 4 路的权重），comb 是 [4,4] 的矩阵（旧的 4 路残差怎样线性组合进新的 4 路）。4+4+16=24，刚好对上。"

**依据：** `model.py: Block.__init__`（`mix_hc=(2+hc_mult)*hc_mult`）；`src/deepseek/shape_trace.js` 中 `mixes()` 函数；`docs/architecture/assumptions.md` 的 hc 条目（"24 个混合分数，base 24、scale 3"）。

### Q38 ★ L2 hc_post 的公式 `y = post⊗F(x) + Σ comb⊗residual` 怎么对应论文里 `X_{l+1}=B_l X_l + C_l F_l(A_l X_l)` 这个等式？

**现场回答：** "post 对应论文里的 C_l，乘的是子模块输出 F(A_l X_l)；comb 对应 B_l，是一个 [4,4] 矩阵，对旧的 4 路残差 X_l 做加权求和，这部分是跨 4 路的矩阵收缩。两项相加正好是 B_l X_l + C_l F_l(A_l X_l)。pre 对应公式里的 A_l（或者 Single-Pass 版本里的 A_{l-1}），负责把 4 路收缩成 1 路送进子模块。"

**追问补充：** `comb.unsqueeze(-1)*residual.unsqueeze(-2)` 这一步在 dim=2（旧的 4 路维度）上求和，就是矩阵乘法 B_l X_l 的逐元素展开写法；`post.unsqueeze(-1)*x.unsqueeze(-2)` 是标量按路广播乘子模块输出，对应 C_l 作用在单路输出上再扩展回 4 路。

**依据：** `references/deepseek/technical_report.txt:713-719`（公式 2）；`model.py: Block.hc_post`。

### Q39 L2 MoE 的 SwiGLU 具体在哪一步乘上路由权重？

**现场回答：** "Expert.forward 里先算 gate=W1(x)、up=W3(x)，按配置做 clamp，再算 SiLU(gate)*up 得到中间激活，这一步之后、过 W2 之前，如果传入了 weights 参数就直接相乘：`x = weights * x`。也就是路由权重是在 SwiGLU 算完、Down 投影之前乘进去的，不是在最终求和时才加权。"

**依据：** `model.py: Expert.forward`（`if weights is not None: x = weights * x`，随后才 `return self.w2(x.to(dtype))`）。

### Q40 L2 单 token 解码时每个被选中专家只处理 1 个 token，形状很干净；批量/Prefill 场景下这幅图会怎么变？

**现场回答：** "单 token 时每个专家最多处理 1 行输入，形状就是 [1,2304]。批量场景下，`MoE.forward` 用 `torch.where(indices==i)` 为每个专家收集被路由到它的 token，这个数量是不确定的、随 batch 内容变化的——有的专家可能一个 token 都没分到（代码里 `counts[i]==0` 时直接 `continue`），有的专家可能分到很多。张量形状从单 token 时的规整 [1,F] 变成了逐专家长度不一的稀疏聚合，这也是 MoE 真实服务时负载不均衡问题的来源。"

**依据：** `model.py: MoE.forward`（`counts=torch.bincount(...)`；`idx, top = torch.where(indices == i)`）。

### Q41 L2 Engram 的张量形状链条是什么？24 和 25600 这两个数字从哪来？

**现场回答：** "哈希得到的 n-gram 索引经过表查找得到 [1,24,256]（24 = (max_ngram_size-1)×n_heads = 3×8），flatten 成 [1,6144]（24×256），过 wkv 投影到 [1,25600]（(hc_mult+1)×dim = 5×5120），再拆成 Key [1,4,5120] 和共享 Value [1,5120]（4×5120+5120=25600，刚好对上）。"

**依据：** `references/deepseek/config.json: engram_max_ngram_size=4, engram_n_heads=8, engram_head_dim=256`；`model.py: Engram.__init__`（`n_hash_cols=(layout.max_ngram_size-1)*layout.n_heads`；`self.wkv=Linear(n_hash_cols*layout.head_dim, args.dim*(args.hc_mult+1))`）。

### Q42 L3 Engram 门控的归一化是"按每个 token、每一路 hc 流分别算"而不是"跨 4 路联合算"，这个轴选择为什么重要？

**现场回答：** "如果跨 4 路联合归一化，4 路会互相影响彼此的门控强度——某一路的激活特别大，可能把其他路的门控数值也压下去。现在的写法是对每一路各自独立算 RMS 再归一化，4 路可以完全独立地判断'这次查表结果我要不要、要多少'，不会因为别的流而被干扰。这让 4 路残差流在这个注入点上保持了相互独立的表达能力。"

**依据：** `model.py: Engram.forward` 注释（"normalized per (token, hc copy) over dim, NOT jointly over the copies"）。

### Q43 L2 最后一层（第 39 层）为什么没有普通的 hc_post（Post/Comb），直接 collapse 了？

**现场回答：** "普通层的 hc_post 是为了把子模块输出重新分配回 4 路，供下一层继续用。但第 39 层是主干的最后一层，后面不再有层需要 4 路残差了，所以直接做一次'Final mHC Pre'把 4 路收缩成 1 路（用最后一层 FFN Mix 产生的 pre 系数），接 RMSNorm 和 LM Head 输出 [1,129280] 的 logits，不再需要展开。"

**依据：** `src/deepseek/shape_trace.js`（`L===39` 分支）；`model.py: Transformer.forward`（`h = layer.hc_pre(h, pre_mix)`，循环结束后只调用一次 `hc_pre`，不再调用 `hc_post`）。

### Q44 L3 DSpark 的草稿输入是 `h.mean(dim=2)`，对 4 路 hc 流取均值——为什么是均值，不是用 collapse 后的单路形式？

**现场回答：** "代码在 target 层（37/38/39）直接对隐藏状态的 4 路维度取平均，作为喂给草稿模块的主干隐藏表示，而不是用该层自己的 hc_pre 收缩结果——因为 DSpark 读的是这些层的 Attention 输入（进入该层之前的状态），这个时间点上该层自己的 pre 系数还没被用于 collapse，用均值是一种不依赖该层具体系数、直接拿到可用信号的简单做法。这是我能确认的代码行为；为什么训练时选均值而不是别的聚合方式，属于训练期设计决策，源码本身给不出动机解释。"

**追问补充：** 注释明确写"the MTP head reads the attention input of its target layers, not their output"，取的是该层处理前的状态。

**依据：** `model.py: Transformer.forward`（`main_hiddens.append(h.mean(dim=2))`，在 `layer(h,...)` 调用之前）。

### Q45 L3 如果有人说"Shape Trace 页面上的每个张量形状都是 PyTorch 调试器里会看到的真实形状"，这个说法站得住吗？

**现场回答：** "大部分节点是，但有几类是故意用符号形状表示的逻辑阶段，不是字面张量：ΔS、S、N、K 这些符号代表'这一步新增的行数''累计缓存行数''实际读取位置数''选中的 Top-K 数'，它们在不同 Decode 步里数值会变。更重要的是，Gather、Attention Scores、Softmax、Weighted Reduce 这几步，源码里是融合在一次 `sparse_attn` 调用里完成的，页面上画成分开的几个节点，是为了讲清楚逻辑顺序，不代表真的物化出这些中间矩阵。"

**追问补充：** `shape_trace.md` 自己就写明了这个边界——"不声称实现物化了这些中间矩阵，也不复制 64 份 KV"。

**依据：** `src/deepseek/shape_trace.md`（"Gather、Attention Scores、Softmax 和 Weighted Reduce 展示 fused `sparse_attn` 中的逻辑阶段"）。

---

## 三、参数量与 MoE（Q46–Q60）

### Q46 L1 为什么参数量不能直接代表计算量或速度？

**现场回答：** "参数量只说明保存了多少权重。计算量还取决于用了哪些权重、复用了几次；速度还受带宽、并行度和通信影响。MoE 和查表模块就是总参数很多、每次只访问一部分的例子。"

**依据：** `presentation_QA_pre_v2.md` Q13（已核实，原样保留）。

### Q47 L2 一个 Expert 的三个矩阵分别是什么，参数怎么算？有没有 bias？

**现场回答：** "这个 Expert 是 SwiGLU 结构，W1（Gate）、W3（Up）都是 5120→2304，W2（Down）是 2304→5120，不含 bias 的权重总量是 3×5120×2304=35,389,440，约 3540 万。三个 Linear 在源码里都没有传 `bias=True`，所以没有额外的 bias 参数。"

**追问补充：** SiLU 和逐元素相乘本身不引入新的可学习参数，但会产生 FLOPs，"无参数"不等于"无计算"。

**依据：** `references/deepseek/config.json: moe_intermediate_size=2304`；`model.py: Expert.__init__`（`Linear(dim,inter_dim,bias 默认 False)`）。

### Q48 ★ L2 总参数和激活参数具体差多少？Router 算在哪一边？

**现场回答：** "单层保存 384 个路由专家加 1 个共享专家，一个 token 只用 6 个路由专家加 1 个共享专家。专家权重总计约 136.25 亿，当前 token 实际用到的专家权重约 2.48 亿，Router 的打分矩阵（5120×384=1,966,080）在两边都要算一次——因为不管最终选谁，Router 都要对全部 384 个专家打分，这是它和专家权重的本质区别：Router 总是全量参与，专家只有被选中的才参与。"

**备查：**

| 单层项目 | 参数个数 |
|---|---:|
| 一个 Expert | 35,389,440 |
| 385 个专家（384 路由+1 共享） | 13,624,934,400 |
| Router 矩阵 5120×384 | 1,966,080 |
| 总参数 = 专家总量+Router | 13,626,900,480 |
| 7 个激活专家（6 路由+1 共享） | 247,726,080 |
| 激活参数 = 激活专家+Router | 249,692,160 |

**依据：** `compute-efficiency/calculations.js: deepseekMoeActiveVsTotal()`；`references/deepseek/config.json: n_routed_experts=384, n_activated_experts=6, n_shared_experts=1`。

### Q49 L3 只激活 6/384 个路由专家，能不能说速度提升了 64 倍？报告里"1/4 激活参数、提升 5%-10%"这个说法是同口径的 FLOPs 比较吗？

**现场回答：** "不能直接推 64 倍，还有共享专家、Router、Attention 这些始终参与的部分，以及专家分发、通信、负载不均衡的实际开销。报告里'1/3 总参数、1/4 激活参数、提升 5%-10%'这句话，比较对象是另一个不同规模的模型（V4-Pro-Base），不是同一个模型在不同激活比例下的消融实验，所以它说明的是'参数效率更高'，不能直接拿来证明'激活参数减少本身带来了多少倍速度提升'——这是两个不同的问题。"

**依据：** `references/deepseek/technical_report.txt:318-321`；`presentation_QA_pre_v2.md` Q16。

### Q50 L1 Router 的选择偏置（bias）和普通 Linear 的 bias 是一回事吗？

**现场回答：** "不是。这里的专家线性层本身没有加性 bias；Router 的 `self.bias` 是单独的、形状为 [n_routed_experts] 的可学习参数，只参与'选谁'这一步的 topk 排序，不参与'选中后权重多大'的计算（权重来自没加 bias 的原始分数）。代码里还区分了文本和图像场景各自的选择偏置（`bias_vl`），图像场景不激活时用的是普通 `bias`。"

**依据：** `model.py: Gate.__init__`（`self.bias`/`self.bias_vl`）；`Gate.forward`。

### Q51 L2 只算 MoE 核心权重（专家+Router），40 层加总是多少？和论文的 552B 骨干参数差多少，差的部分是什么？

**现场回答：** "单层 MoE 核心权重 13,626,900,480，乘 40 层约等于 545,076,019,200，约 545.08B。论文说的是 552B 骨干参数，差额大约 7B 左右，这部分不在 MoE 核心权重里，包含 Attention 的 Query/KV/输出投影权重、mHC 的混合系数矩阵、RMSNorm 的 γ、Embedding 和 LM Head、以及 Indexer/Compressor 自己的权重——这些加起来大概能补上这个差额（见 Q57 的完整核对）。"

**依据：** `compute-efficiency/calculations.js`；`references/deepseek/technical_report.txt:243`（"552B backbone parameters"）。

### Q52 L2 Engram 的 196B 参数和配置里的 engram_num_embeddings 对得上吗？

**现场回答：** "报告说 196B Engram 参数平均分在两个模块。配置里 `engram_num_embeddings` 是 [384006168, 384016682]，`engram_head_dim`=256，用第一个模块算：384,006,168×256 约等于 983 亿，接近 98B，和'196B 平均分两个模块、每个约 98B'的说法大致对得上，是一次有意义的交叉核对，不是巧合。"

**追问补充：** 这只核对了 Embedding 表本身，没有把 `wkv` 投影层和 `q_weight`/`k_weight` 这些额外参数算进去，严格对账会比 98B 略多一点。

**依据：** `references/deepseek/config.json: engram_num_embeddings, engram_head_dim=256`；`references/deepseek/technical_report.txt:776`。

### Q53 L1 196B 的 Engram 参数算在"552B 骨干参数"里面吗？

**现场回答：** "不算在里面，是额外加的。报告原文是'has 552B backbone parameters and 196B Engram parameters'，用的是'和'（and），说明这是两个并列、相加的数字，不是 552B 已经包含了 Engram。"

**依据：** `references/deepseek/technical_report.txt:421`（"552B backbone parameters and 196B Engram parameters"）。

### Q54 L2 Attention 每层核心权重合计多少？Query 侧和 KV 侧差多少？

**现场回答：** "按源码形状逐项加：W_QA(5120×1280)+W_QB(1280×32768)+W_KV(5120×512)+分组 W_OA(8192×4096)+W_OB(8192×5120)，合计约 1.266 亿。KV 投影只有 262 万，Query 侧两段投影加起来接近 4850 万，差了近 20 倍——这就是共享单头 KV 设计的直接体现：KV 存储和计算成本被压得很小，大头在 Query 和输出投影。"

**追问补充：** 这个小计不含 RMSNorm 的 γ、attn_sink，以及只有源层才有的 Compressor/Indexer 权重，完整核对见 Q57。

**依据：** `model.py: Attention.__init__` 各 Linear 的 in/out_features；本节用 2MKN 同款逐项相乘方法，不含 bias（配置中 attention_bias=false）。

### Q55 L3 mHC 的混合系数矩阵参数量有多大？相对一个 Expert（3540 万）算不算"免费"？

**现场回答：** "`hc_attn_fn` 和 `hc_ffn_fn` 各是 [24,20480]（mix_hc×hc_mult×dim），单个 491,520，两个加上很小的 base/scale 项，每层约 98.3 万，40 层合计约 3930 万——比一个 Expert（3540 万）还略多一点，但相对 545B 的 MoE 核心权重几乎可以忽略，占比不到十万分之一。所以 mHC 的'参数开销'确实很小,但这不等于它'没有成本'——它真正的代价在反复读写激活的内存流量上，这正是报告花大篇幅讲 Single-Pass mHC 和 Mega-mHC 的原因，不是参数量问题。"

**依据：** `model.py: Block.__init__`（`mix_hc=(2+hc_mult)*hc_mult=24`；`hc_dim=hc_mult*args.dim=20480`）。

### Q56 L2 Indexer 每个源层的参数量是多少？为什么不是 8 个 Indexer 源层都一样大？

**现场回答：** "`wq_b`(1280×32×128=5,242,880) 和 `weights_proj`(5120×32=163,840) 是所有 8 个 Indexer 源层都有的。但 `wk`(512×128=65,536) 和对应的 k_norm，只有同时是 KV 源层的那 4 层（2/8/14/20）才有，因为只有这 4 层自己生成压缩 latent、需要自己把它转成 Index Key；另外 4 个 Reindex 层（24/28/32/36）复用已有 Key，不需要 `wk`。"

**依据：** `references/deepseek/config.json: index_n_heads=32, index_head_dim=128`；`model.py: Indexer.__init__`（`if self.owns_k: self.wk=...`）。

### Q57 ★ L3 把 MoE、Attention、mHC、Embedding/Head、Indexer、Compressor 全部加总，能对上论文的 552B 骨干参数吗？

**现场回答：** "仓库里已有一份逐项核对的结果：Text Backbone 合计约 551,566,180,464，大约 551.57B，和论文四舍五入后的 552B 非常接近，差距在十万分之一量级。这份数字我自己用 Q48/Q54/Q55/Q56 的逐项计算又独立核对了一遍，结果几乎一致（差异在几十万这个量级，属于四舍五入和个别小项取舍的差异）。"

**追问补充：** 这个 551.57B 不含 Engram 的 196B（报告说是另算的），也不含 DSpark 和 Vision——加上 Engram 之后，仓库记录的合计是 748,494,684,784，再加上 DSpark 草稿（约 142.25 亿）和 Vision+对齐（约 4.85 亿），总计 763,205,315,794。

**依据：** `presentation_QA_pre_v2.md:23`（"Text Backbone 551,566,180,464"）；`src/deepseek/README.md`"参数与计算审计"节；本题"追问补充"中的逐项求和为现场独立复算。

### Q58 L2 Embedding 和 LM Head 各多少参数？它们绑权重吗？

**现场回答：** "vocab_size×dim=129280×5120=662,153,600，Embedding 和 LM Head 各占这么多。配置里 `tie_word_embeddings=false`，代码里 `ParallelEmbedding` 和 `ParallelHead` 也是两个独立的 `nn.Parameter`，没有共享权重，所以这部分合计约 13.24 亿，是两份独立存储。"

**依据：** `references/deepseek/config.json: vocab_size=129280, hidden_size=5120, tie_word_embeddings=false`；`model.py: ParallelEmbedding`/`ParallelHead`。

### Q59 L2 DSpark 的 3 个 Transformer Block、Markov Head、Confidence Head 应该算进"552B 骨干"吗？论文怎么划的界？

**现场回答：** "不应该算进 552B 骨干，论文一直把 552B 称为'backbone parameters'，DSpark 是单独训练、单独统计的投机解码模块。仓库里的审计给出了一个独立的 DSpark 预算：约 142.25 亿（含 3 个完整 Block 的 Attention+MoE，以及复用主 Embedding/LM Head 的 Markov Head 和独立的 Confidence Head），这个数字是和 552B 骨干、196B Engram 并列列出的，不是其中的一部分。"

**依据：** `presentation_QA_pre_v2.md:23`（"Draft 14,225,362,530"）；`references/deepseek/config.json: dspark_block_size=5, dspark_target_layer_ids=[37,38,39]`。

### Q60 L3 "552B 参数乘 2 字节"能当作这个模型的显存占用吗？

**现场回答：** "不能。配置里的 `quantization_config` 写明权重默认是 fp8，专家权重是 fp4（两个值打包进一个字节），而且是按 32×32 的 block 分别配一个 scale，不是全局统一的缩放系数。简单乘 2 字节既不符合实际存储精度，也没算 scale 本身占的空间；再加上推理还有 KV Cache、激活、工作区，训练还有梯度和优化器状态，没有一个'参数量×固定字节数'的公式能覆盖这些。"

**依据：** `references/deepseek/config.json: quantization_config`（`quant_method=fp8, expert_dtype=fp4, weight_block_size=[32,32]`）；`model.py: Linear.__init__` 中 fp4/fp8 两种 scale 形状。

---

## 四、前向与反向 FLOPs（Q61–Q75）

### Q61 L1 FLOPs、FLOP/s、MAC 和 FMA 怎么区别？

**现场回答：** "FLOPs 是工作总量，FLOP/s 是每秒做多少工作。一个乘法算一个 FLOP，一个加法也算一个；一次乘加按常用口径算两个 FLOPs。MAC 常表示一次乘加，比较数字前要确认计数约定。"

**追问补充：** FMA 可以是一条融合指令，但 FLOPs 计数仍通常记 2。

**依据：** `presentation_QA_pre_v2.md` Q19（已核实，原样保留）。

### Q62 L2 矩阵乘法为什么是 2MKN？精确值是多少？

**现场回答：** "输入 [M,K] 乘权重 [K,N]，输出有 MN 个元素，每个元素需要 K 次乘法和 K−1 次加法，精确乘加计数是 MN(2K−1)，常用近似是 2MKN。小矩阵时两者差异明显，矩阵越大相对差异越小。"

**小例子：** `[1,4]×[4,3]`：前向精确 21 FLOPs，近似 24 FLOPs。

**依据：** `presentation_QA_pre_v2.md` Q20（已核实，原样保留）；`docs/architecture/assumptions.md` Approximation 节。

### Q63 ★ L2 为什么 Query Projection 是约 97 MFLOPs？

**现场回答：** "单 token 先做 5120→1280 的压缩，再做 1280→32768 的展开，两次矩阵乘法按 2MKN 相加：13,107,200+83,886,080=96,993,280 FLOPs，约 97 MFLOPs。32768=64 个 head×512 维。"

**追问补充：** 该小计不含 RMSNorm、RoPE、Attention 分数计算。

**依据：** `presentation_QA_pre_v2.md` Q21；`compute-efficiency/calculations.js: queryProjectionFlops()`。

### Q64 L2 低秩投影比直接投影省多少？

**现场回答：** "直接从 5120 投影到 32768 需要约 1.678 亿个权重；两段投影合计约 4850 万，约为直接投影的 28.9%。单 token 的线性乘法 FLOPs 也约为原来的 28.9%，但这只是针对这两个矩阵的比较基线，不代表全 Attention 或全模型省了 71.1%。"

**依据：** `presentation_QA_pre_v2.md` Q22。

### Q65 L3 Reshape、Top-K、查表没计入 FLOPs，是不是它们不耗时？为什么能乘 40？

**现场回答：** "FLOPs 只衡量一种工作量。Reshape 如果只是视图变化可能很便宜，需要重排或拷贝就有访存成本；Top-K、查表、索引操作同样可能耗时，不能用 FLOPs 表代替运行时间分析。至于乘 40：本例正常主干 Decode 中每层都有同样形状的两次 Query 投影，这一项能乘 40 得到约 3.88 GFLOPs；但跨层复用的 Global KV、Indexer，以及 Prefill、草稿分支要按实际执行次数算，不能统一乘 40。"

**依据：** `presentation_QA_pre_v2.md` Q23/Q24（合并）。

### Q66 ★ L3 论文的 Figure 2 按 BF16:FP8:FP4=1:0.5:0.25 给 FLOPs 加权，仓库这套 2MKN 计数有没有做同样的加权？

**现场回答：** "没有。仓库和本页面用的 2MKN 是未加权的逻辑运算次数计数，不区分精度。这个模型默认权重是 fp8、专家是 fp4，按论文的加权方式，同样的逻辑运算量换算成'BF16 当量 FLOPs'会比未加权的数字更小。这意味着我报的 97 MFLOPs、3.88 GFLOPs 这些数字和论文 Figure 2 里的 Decode FLOPs 曲线不是同一套口径，不能直接放在一起比大小。"

**追问补充：** 如果被问"那实际的加权数字是多少"，诚实的答案是：本仓库没有按精度逐算子统计，这是一个明确的未覆盖项，不能当场编一个加权结果。

**依据：** `references/deepseek/technical_report.txt:271-273`；`references/deepseek/config.json: quantization_config`；`compute-efficiency/calculations.js` 中 FLOPs 公式不含精度加权。

### Q67 L2 为什么有 dW 和 dX？它们分别是什么？

**现场回答：** "dW 告诉优化器怎样调整当前层权重，dX 把损失对输入的影响传给前面的模块。对 Y=XW，收到输出梯度 G 后，dW=XᵀG，dX=GWᵀ，两项都来自链式法则。"

**形状核对：** `X[M,K]，W[K,N]，Y 与 G[M,N]；dW:[K,M]×[M,N]→[K,N]；dX:[M,N]×[N,K]→[M,K]`。

**依据：** `presentation_QA_pre_v2.md` Q25。

### Q68 L1 是不是每层都要重新从 Loss 开始求导？dW 就是更新后的权重吗？

**现场回答：** "不用每层都从 Loss 重新开始，中间层接收后一层传来的梯度，再算本层梯度。dW 也不是更新后的权重，优化器还要根据梯度和学习率等规则完成更新。"

**依据：** `presentation_QA_pre_v2.md` Q26。

### Q69 L2 为什么说反向约为前向两倍？

**现场回答：** "一个线性层前向做一次矩阵乘法，反向通常要做输入梯度、权重梯度两次乘法。按统一的 2MKN 近似，这两次的主项都和前向相同，所以反向约两倍，前向加反向约三倍——这个结论的前提是两种梯度都需要，且用统一的矩阵乘法近似，不含优化器、非线性、缓存完整反向或额外重计算。"

**依据：** `presentation_QA_pre_v2.md` Q27。

### Q70 ★ L3 单 token 时，"反向两倍"是精确结论吗？

**现场回答：** "不是。单 token 的权重梯度是外积，没有跨多行求和，2MKN 近似会多计一部分加法。小矩阵例子 `M=1,K=4,N=3`：前向精确 21，输入梯度精确 20，权重梯度精确 12，反向精确合计 32，前向+反向精确合计 53；而用统一近似算出来是 72——两者差了约 36%。这个差距在单 token 场景下不可忽略，我会明确把'约两倍'称为统一近似，不说成精确值。"

**依据：** `presentation_QA_pre_v2.md` Q28（含精确乘加对照表）。

### Q71 L2 Query Projection 反向约 194 MFLOPs，怎么来的？

**现场回答：** "两个 Linear 都需要输入梯度和权重梯度，按相同近似合计是前向的两倍，即 193,986,560 FLOPs，前向加反向约 290.98 MFLOPs。反向顺序是 W_QB→RMSNorm→W_QA——这个小计不含 RMSNorm 的 FLOPs，但真实梯度仍必须经过 RMSNorm，不能当成恒等映射跳过。"

**依据：** `presentation_QA_pre_v2.md` Q29。

### Q72 L1 RMSNorm 有参数吗？为什么这个算例没算它？

**现场回答：** "有，这里 1280 维 RMSNorm 有 1280 个可学习缩放参数 γ，它也有自己的前向和反向。这个算例为了先讲清矩阵乘法单独把它排除了，完整模块统计时要把它加回来，反向不仅要传入梯度，还可能要算 γ 的梯度。"

**依据：** `presentation_QA_pre_v2.md` Q30；`model.py: RMSNorm`（`self.weight=nn.Parameter(torch.ones(dim))`）。

### Q73 L3 参数冻结或 LoRA 时，还能套"反向两倍"吗？DSpark 冻结主干训练是不是一个现成的反例？

**现场回答：** "不能统一套。冻结 W 之后可以省掉 dW，但如果还要训练前面的模块，dX 仍然需要；LoRA 还有适配器自己的前向反向，应该按实际需要梯度的路径重新统计。DSpark 就是这个模型里一个真实的例子：训练 DSpark 时主干参数被冻结，DSpark 的训练目标明确不向主干传梯度，这种场景下如果照搬'反向两倍'去估算主干这部分的训练开销，会直接算错，因为主干这里根本没有 dW，也没有来自 DSpark 损失的 dX。"

**依据：** `presentation_QA_pre_v2.md` Q31；`references/deepseek/technical_report.txt:798-803`（DSpark 训练阶段冻结主干）。

### Q74 L3 激活检查点重计算会增加 MFU 的分子吗？Mega-mHC 这类核融合又是在改变什么？

**现场回答：** "重计算会增加硬件实际执行的运算，但按这里采用的 MFU 定义，不把额外重计算算成有效模型工作量，它是通过改变显存需求和 step 时间间接影响 MFU 的。Mega-mHC 是另一种情况：它不改变 mHC 做的数学运算本身（还是同一套公式），改变的是访存量——把原来 (4n+4)d 的读写压到 (2n+2)d。这类优化会让实际 step 时间变短，从而让 MFU 分母变小、MFU 数值变高，但 FLOPs 分子没有变化，这正说明 MFU 不是一个纯粹的"计算效率"指标，也掺杂了访存优化的影响。"

**依据：** `presentation_QA_pre_v2.md` Q32（PaLM MFU/HFU 定义引用）；`references/deepseek/technical_report.txt:761-767`（Mega-mHC 访存量）。

### Q75 ★ L3 能不能把网页上的局部反向示例直接当成这个模型的完整训练 FLOPs？

**现场回答：** "不能。架构页提供的是局部数学推导，参考代码本身是推理实现，不含训练反向。要得到完整训练量，还需要确认整段序列、所有模块（包括 Engram、Indexer、mHC、MoE 路由）的梯度汇总、哪些路径在训练时被冻结（比如前面说的 DSpark 阶段）、以及稀疏选择部分的梯度处理规则，这些目前都不在已验证范围内。"

**依据：** `presentation_QA_pre_v2.md` Q33；`docs/architecture/assumptions.md` Unverified 节。

---

## 五、Roofline 与硬件性能（Q76–Q90）

### Q76 L1 Roofline 用一句话怎么解释？

**现场回答：** "它把算力上限和数据搬运上限放在一起，判断在当前工作量与访存假设下，任务更可能先被哪一项限制。"

**公式：** `AI=F/B（FLOPs/Byte）；P_roof=min(P_peak, AI×BW)；AI_ridge=P_peak/BW；t_ideal≥max(F/P_peak, B/BW)`。

**依据：** `presentation_QA_pre_v2.md` Q34；`compute-efficiency/calculations.js: roofline()`。

### Q77 L1 Bytes Moved 为什么不能直接用模型大小或 KV Cache 大小？

**现场回答：** "模型大小和 Cache 大小描述容量，Bytes Moved 描述一次任务实际读写多少。相同权重可能被重复读取，也可能被缓存复用；中间结果还可能写出再读回。要先指定任务和内存层级，再逐项统计流量。"

**依据：** `presentation_QA_pre_v2.md` Q35。

### Q78 L2 Query Projection 的 97.07 MB 流量怎么算出来的？

**现场回答：** "教学模型假设 BF16 每个数两字节，两个权重矩阵各读一次，再加输入读取、中间结果写出与读回、最终输出写出。单 token 合计约 97.07 MB，绝大部分来自权重。"

**公式：** `权重流量=48,496,640×2=96,993,280 B；总流量 B(M)=96,993,280+80,896M Bytes；运算量 F(M)=96,993,280M FLOPs`。

**边界：** 这是两次线性投影的逻辑 HBM 流量模型，不含 RMSNorm，不模拟真实缓存、分块或融合，不是 profiler 记录。

**依据：** `presentation_QA_pre_v2.md` Q36；`compute-efficiency/calculations.js: queryProjectionTraffic()`。

### Q79 ★ L2 为什么单 token 是带宽受限，1024 行变成算力受限？

**现场回答：** "两个算例共享相同的权重。一起处理更多行时，权重读取成本由更多运算分摊，AI 就上升。在当前硬件模板和流量假设下，单行 AI 约 1，1024 行约 552，分别落在拐点两侧。"

| M | FLOPs | 流量 | AI |
|---|---:|---:|---:|
| 1 | 96,993,280 | 97,074,176 B | 0.9992 |
| 1024 | 99,321,118,720 | 179,830,784 B | 552.30 |

**追问补充：** 1024 行可以来自多请求批处理或已有输入位置，不能让同一请求中依赖前一个结果的未来 token 任意并行生成。

**依据：** `presentation_QA_pre_v2.md` Q37。

### Q80 L2 "一起处理的行数"从 1 连续变到 1024，AI 公式里哪部分在变、哪部分不变？

**现场回答：** "AI(M)=96,993,280M/(96,993,280+80,896M)。分子随 M 线性增长，分母里权重流量 96,993,280 是常数（权重只读一次），只有输入/中间/输出流量（80,896×M 这部分）随 M 增长。M 越大，分母里的常数项占比越小，AI 就越接近分子分母都线性的渐近值，这也是为什么 M 从 1 到 1024 时 AI 从约 1 涨到约 552，而不是线性的 1024 倍。"

**依据：** `compute-efficiency/calculations.js: queryProjectionTraffic()`；本题公式由 Q78 的流量分解直接代入。

### Q81 ★ L2 H100 为什么用约 989，而不是 1979 TFLOP/s？

**现场回答：** "这里选 H100 SXM 80GB 的 BF16 Dense Tensor Core 口径。官网 BF16 的 1979 带有'with sparsity'（稀疏加速）标注，不能直接当 Dense 上限；989.5 正好是 1979 的一半，这是行业通行的 dense/sparse 换算惯例，NVIDIA 官网本身没有直接列出 dense 数值，这个 989.5 是推算值而不是官网原文数字。"

**依据：** `presentation_QA_pre_v2.md` Q38；NVIDIA H100 产品页（`https://www.nvidia.com/en-us/data-center/h100/`，"1,979 TFLOPS* with sparsity"）。

### Q82 L3 Roofline 页用 989，MFU 页用 989.5，拐点因此也不完全一样——这个不一致怎么现场说清楚？

**现场回答：** "这是同一个页面两个 Tab 之间一个真实存在的小不一致：Roofline Tab 的 H100 预设用的是 989（更早取的一个近似整数），MFU Tab 后来的算例用的是 989.5（1979 精确除以 2）。拐点因此分别约是 295.22 和 295.26，差异不到 0.02%，不影响结论方向，但如果被追问我会直接承认这是两个数值取整方式不同导致的,后续应该统一成同一个值。"

**依据：** `presentation_QA_pre_v2.md` 开头"先修正或统一的 6 处说法"表格第 5 行；`compute-efficiency/calculations.js: HARDWARE_PRESETS.h100.peakTF=989`；`compute-efficiency/app.js: DEFAULTS['in-mfu-peak-tf']='989.5'`。

### Q83 L3 在屋顶线左边，就证明实测一定是带宽瓶颈吗？

**现场回答：** "不是，它只说明在当前模型假设下，带宽给出的上限更低。真实运行还可能受小矩阵效率、启动开销、访存模式或通信限制，需要用实测流量、时间和 profiling 验证。图上的理想点由公式计算，不是 benchmark。"

**依据：** `presentation_QA_pre_v2.md` Q39。

### Q84 L1 FLOPs 减半为什么不一定速度翻倍？

**现场回答：** "如果主要时间花在搬数据，计算减少一半但搬运量不变，速度就未必明显提升。带宽受限时优先看复用、融合和数据量；算力受限时再看矩阵效率和计算量，最终还是看端到端时间。"

**依据：** `presentation_QA_pre_v2.md` Q40。

### Q85 L3 流量模型假设"权重每次批处理只读一次"——如果 1024 个 token 是 1024 次独立的单 token Decode 调用，而不是一次批量矩阵乘法，这个假设还成立吗？

**现场回答：** "不成立。AI≈552 这个结果的前提是 1024 行在同一次矩阵乘法调用里一起处理，权重只从 HBM 读一次，被 1024 次运算分摊。如果实际是 1024 次相互独立的单 token Decode（比如服务 1024 个不同请求、各自串行调用），权重就要被重复读 1024 次，流量模型完全不同，AI 会重新收敛回接近 1 的水平，而不是 552——这正是教学例子里反复强调'M 是一起处理的行数'而不是'历史 token 数'的原因。"

**依据：** `presentation_QA_pre_v2.md` Q37 追问补充；`compute-efficiency/calculations.js: queryProjectionTraffic()` 中 `weightBytes` 不随调用次数累加，只对应一次调用。

### Q86 ★ L3 更高的 Arithmetic Intensity 是不是总是意味着更好的性能？

**现场回答：** "不是。AI 只是'每搬一个字节能做多少运算'，如果增加的是无意义的冗余计算，AI 照样会升高，但实际有用的工作量和速度都没有变好。页面上的 Roofline 曲线本身也提醒过：Gather、Attention Scores 这类融合在 sparse_attn 内部的逻辑阶段并不是 profiler 实测，它们的真实 AI 目前没有被单独测量，不能假设它们一定落在某个理想位置。"

**依据：** `src/deepseek/shape_trace.md`（"不是实际的 benchmark"类表述）；`compute-efficiency/README.md`"Known limitations"。

### Q87 L2 Indexer 的 Q·K 打分如果也套 Roofline，AI 大概率比 Query Projection 高还是低？为什么？

**现场回答：** "大概率更低、更偏向带宽受限。Query Projection 是两个大矩阵的稠密乘法，权重读一次能摊到很多次乘加上；Indexer 的打分是用当前 Query 去和多达 S 个压缩位置的 Key 逐一做点积，每个位置的计算量很小（index_head_dim=128 这个量级），但要把这些 Key 全部从缓存里读出来，读取量相对计算量的比例比稠密矩阵乘法高得多，所以直觉上更容易偏向带宽受限——这是基于 Roofline 框架的推断，不是仓库里已经跑出来的实测或计算结果。"

**追问补充：** 这个问题是在检验"能不能把框架推广到没有现成算例的模块"，诚实标注"这是推断不是已验证数字"比硬编一个 AI 数值更站得住。

**依据：** `model.py: Indexer.forward`（`index_score=torch.einsum("bshd,btd->bsht", q, index_k)`）；`references/deepseek/config.json: index_head_dim=128`。

### Q88 L3 如果带宽单位被误输成 GB/s 而不是 TB/s（比如把 3.35 当成 GB/s 填进去），拐点和结论会怎么变？现场怎么发现这个错误？

**现场回答：** "拐点公式是 P_peak/BW，如果 BW 被错误地缩小了 1000 倍，拐点会跟着放大 1000 倍，从约 295 变成约 295,000。这样一来 AI≈552 这个点会被远远甩到拐点左边，结论会从'算力受限'错误翻转成'带宽受限'。现场发现的办法是做量级检查：H100 这类加速卡的拐点通常在几百 FLOPs/Byte 量级，如果算出来的拐点是几十万，第一反应应该是去核对带宽的单位选的是 TB/s 还是 GB/s，而不是直接相信这个数字。"

**依据：** `compute-efficiency/calculations.js: bandwidthBytesPerSecond()`（`TB/s`对应`1e12`，`GB/s`对应`1e9`）；`compute-efficiency/index.html` 中 `in-bw-unit` 选择器。

### Q89 L3 这个简化的 Roofline 模型，够不够解释真实 NPU/GPU 的实测性能？缺了什么？

**现场回答：** "不够，它给的是一种理想上界。真实性能还受内核启动开销、小矩阵效率、访存模式（连续还是跨步）、缓存层级、调度和占用率这些因素影响，Roofline 本身不建模这些。它适合用来判断'大方向上更可能被哪种资源限制'，不能单独拿来预测具体的毫秒数或当作性能验证的终点。"

**依据：** `presentation_QA_pre_v2.md` Q39 追问补充（深化为独立问题）。

### Q90 ★ L3 如果有人说"你的 Roofline 图很好看，但它到底证明了这个模型的什么"，怎么回答才不算过度宣称？

**现场回答：** "它证明的是：在给定的流量假设和硬件模板下，Query Projection 这一个孤立算子在两种行数下分别落在带宽受限和算力受限两侧，给出了一个理论上界，不是测量值。它没有覆盖模型里其余接近全部的模块（Indexer、Compressor、MoE、Engram 等都没有做过同样的分析），也不能替代真实 profiling。我会把它定位成'展示分析方法'，而不是'证明了这个模型快在哪里'。"

**依据：** `compute-efficiency/README.md`"Known limitations"（"only the two Query-projection linear layers"）。

---

## 六、MFU 与端到端性能（Q91–Q100）

### Q91 L1 MFU 的公式是什么？50% 到底表示什么？

**现场回答：** "MFU 是有效模型计算吞吐量除以所有设备的理论总峰值。它表示按模型所需的前向、反向工作量核算，当前训练速度对应总峰值的百分之多少，不表示每张卡有多少时间在空闲。"

**公式：** `F_step=训练 FLOPs/token×全局 tokens/step；MFU=F_step/(t_step×N_GPU×P_peak)`。

**依据：** `presentation_QA_pre_v2.md` Q41；PaLM §4.1（`arXiv:2204.02311v5#S4.SS1`）。

### Q92 ★ L3 49.6% 是 DeepSeek 实际训练 MFU 吗？输入从哪里来？

**现场回答：** "不是，这是教学演算。训练 batch 100.6M tokens 是技术报告原文直接给出的数字；每 token 60 GFLOPs、2048 张卡、每步 6 秒都是假设；989.5 TFLOP/s 用的是 H100 Dense BF16 参照口径。不能把这个结果当成模型真实训练数据。"

**公式：**
```text
tokens/step = 100,600,000
F_token     = 60×10^9 FLOPs/token            # 假设
F_step      = 6.036×10^18 FLOPs ≈ 6.04 EFLOPs
集群峰值    = 2048×989.5×10^12 = 2.026496×10^18 FLOP/s
step time   = 6 s                             # 假设
MFU         = 6.036/(6×2.026496) ≈ 49.64%
```

**依据：** `presentation_QA_pre_v2.md` Q42；`references/deepseek/technical_report.txt:1265`（"We keep the batch size fixed at 100.6 million"）；`compute-efficiency/app.js: DEFAULTS`。

### Q93 L3 为什么不能用"6×总参数量"估算 MoE 训练 FLOPs？

**现场回答：** "6P 来自稠密线性权重每 token 前向约 2P、反向约 4P。MoE 的多数专家不会参与当前 token，应该先按实际激活参数核算，再加 Router、Attention 和其他模块。"

**追问补充：** 本仓库 40 层 MoE 专家加 Router 核心矩阵约有 9.99B 每 token 激活权重，由此得出的约 59.93 GFLOPs/token 只是这部分的近似训练小计，和 Q92 用的 60 GFLOPs/token 假设数值接近，但不是同一回事——不能拿这个粗估"验证"教学假设的 60 GFLOPs/token，两者是独立口径。

**依据：** `presentation_QA_pre_v2.md` Q43；`compute-efficiency/calculations.js: deepseekMoeActiveVsTotal()`。

### Q94 L3 MFU 越高，模型就越好或越省钱吗？

**现场回答：** "MFU 反映指定计算口径下的训练效率，不衡量模型质量。如果架构优化让每 token 的工作量减半、吞吐量翻倍，两项乘积不变，MFU 可以不变，但训练同样数量 token 的时间减少了，还要结合质量、吞吐量和总成本判断。"

**依据：** `presentation_QA_pre_v2.md` Q44。

### Q95 L2 GPU 利用率 100%，为什么 MFU 仍然可能不高？

**现场回答：** "监控工具显示 GPU 很忙，不代表一直在做高吞吐的有效矩阵计算。数据搬运、低效率小算子和额外重计算都可能让设备持续繁忙，但有效模型工作量相对理论峰值仍然不高。"

**依据：** `presentation_QA_pre_v2.md` Q45。

### Q96 L2 MFU 算出超过 100%，应该怎么解释？

**现场回答：** "先检查口径是否一致：是不是把总参数当成激活参数、把单卡和全局吞吐混用、或者把 TFLOP/s 的单位换错了，还要核对精度对应的峰值、设备数和计时范围。超过百分之百通常提示估算或分母定义有问题，不能当成突破硬件上限。"

**依据：** `presentation_QA_pre_v2.md` Q46。

### Q97 L2 多卡训练的通信、重计算、优化器时间要放进 MFU 的哪里？

**现场回答：** "这里 MFU 的分子按模型必需的前向和反向 FLOPs 统计，不把通信等待、额外重计算或优化器工作算成有效模型 FLOPs；这些开销会进入完整 step 的实际时间，体现在分母里。梯度累积时 tokens/step 要统计全部累积 microbatches；设备并行副本不要重复计算相同 token。"

**依据：** `presentation_QA_pre_v2.md` Q47。

### Q98 L2 Roofline 和 MFU 怎样一起用？低 MFU 应该先查什么？

**现场回答：** "先用 Roofline 理解主要算子可能受到的限制，再用训练时间线排查整体效率。如果主要算子本身带宽受限，MFU 低未必是内核差；如果高 AI 的主要算子效率也低，再检查矩阵形状、实现、通信和流水线等待。单个 Query Projection 的 Roofline 不能直接解释全训练 step，整步可能包含不同瓶颈的算子，要按耗时占比分析。"

**依据：** `presentation_QA_pre_v2.md` Q48。

### Q99 L2 要把这些理论分析变成真实结论，下一步要做什么？

**现场回答：** "固定模型版本、精度、batch、序列长度和并行策略，明确 Prefill、Decode 或训练范围，然后测同一范围的时间和吞吐量，配合算子与显存流量 profiling，对比理论 FLOPs 和访存估算，最后再做优化前后的实验。训练看全局 tokens/s、step 时间和收敛效果；推理还看首 token 延迟、逐 token 延迟、吞吐量和显存。"

**依据：** `presentation_QA_pre_v2.md` Q49。

### Q100 ★ L3 如果听众问"你能给出这个模型完整的训练 FLOPs 和真实 MFU 吗？"

**现场回答：** "目前这份材料还不能给出经过验证的完整训练 FLOPs，也没有真实集群的训练计时。我能核算的是已声明范围内的模块参数和局部运算，MFU 部分演示的是计算方法本身。要得到真实值，还需要完整训练执行路径和测量数据——我们可以先固定场景，再逐项补齐计算量和计时，这样得到的结论才可复核。"

**依据：** `presentation_QA_pre_v2.md` Q50。

---

## 附录 A：20 道最重要问题索引

以下 20 题是正文中标 ★ 的全部题目，覆盖六个部分、串起附录 B 的全部 5 条追问链，建议优先反复练习：

| Q | 一句话钩子 | 节 |
|---|---|---|
| **Q01** | 为什么用 CED 而不是普通 Decoder-only | 一、架构 |
| **Q05** | 用 Figure 2 的数字量化 Decode FLOPs 近似不变 | 一、架构 |
| **Q10** | Full / Reindex / Reuse 三种模式各自复用什么 | 一、架构 |
| **Q18** | Single-Pass mHC 怎么把访存砍掉一半 | 一、架构 |
| **Q25** | 架构部分最容易被攻破的一句话结论 | 一、架构 |
| **Q29** | 64 个 Query Head 共享 1 份 KV 是设计不是漏洞 | 二、张量形状 |
| **Q36** | sparse_attn 怎样联合拼接 Local 与 Global KV | 二、张量形状 |
| **Q38** | hc_post 公式对应论文里的 B_lX_l+C_lF_l(A_lX_l) | 二、张量形状 |
| **Q48** | 总参数和激活参数具体差多少，Router 算哪边 | 三、参数与 MoE |
| **Q57** | 逐项加总能否对上论文的 552B 骨干参数 | 三、参数与 MoE |
| **Q63** | Query Projection 为什么约 97 MFLOPs | 四、FLOPs |
| **Q66** | 本仓库 FLOPs 计数有没有按精度加权 | 四、FLOPs |
| **Q70** | 单 token"反向两倍"精确吗，差多少 | 四、FLOPs |
| **Q75** | 局部反向示例能否当成完整训练 FLOPs | 四、FLOPs |
| **Q79** | 为什么 M=1 带宽受限、M=1024 算力受限 | 五、Roofline |
| **Q81** | H100 为什么用 989/989.5 而不是 1979 | 五、Roofline |
| **Q86** | 更高的 AI 是否总意味着更好的性能 | 五、Roofline |
| **Q90** | Roofline 图到底证明了这个模型的什么 | 五、Roofline |
| **Q92** | 49.6% 是真实训练 MFU 吗，输入从哪来 | 六、MFU |
| **Q100** | 能否给出完整训练 FLOPs 和真实 MFU | 六、MFU |

## 附录 B：5 条真实追问链

**链 1 — CED 省算到底省在哪（架构开篇，示例链）：**
Q01 → Q02 → Q03 → Q04 → Q05
"为什么用 CED" → "两段因果计算的区别" → "Prefill 哪些层被简化" → "Decode 阶段优势还在吗" → "怎么量化这个优势"。

**链 2 — Global KV 从概念到张量落地：**
Q07 → Q10 → Q11 → Q34 → Q35
"CSA2 和 V4 的区别" → "Full/Reindex/Reuse 复用了什么" → "当前配置哪层是哪种模式" → "Global KV 缓存的张量形状与量化" → "为什么两套量化分组大小不同"。

**链 3 — Indexer 的时序陷阱与联合计算：**
Q15 → Q16 → Q36 → Q87
"Indexer 的 Q/K 来自哪里" → "为什么必须 Pre-RoPE 取 Key" → "联合 Attention 怎么拼接 Local/Global" → "Indexer 自己的打分套 Roofline 会怎样"。

**链 4 — mHC：从直觉到公式到参数成本：**
Q17 → Q18 → Q37 → Q38 → Q55
"4 路残差流的直觉" → "Single-Pass mHC 省了什么访存" → "24 个系数怎么拆分" → "hc_post 对应论文哪个公式" → "mHC 的参数开销算不算免费"。

**链 5 — 一个数字贯穿 FLOPs→Roofline→MFU：**
Q63 → Q66 → Q78 → Q81 → Q92
"Query Projection 前向 97 MFLOPs 怎么来的" → "这个数字有没有按论文方式做精度加权" → "对应的 97.07MB 访存流量怎么算" → "硬件峰值为什么用 989/989.5" → "这些口径最终怎样汇入 49.6% 的 MFU 算例"。

## 附录 C：演讲中最薄弱的 10 个环节

1. **"1/3 总参数、1/4 激活参数、提升 5%-10%"是跨模型比较，不是同模型消融。** 比较对象是 DeepSeek-V4-Pro-Base，不能单独归因给架构改进（见 Q25、Q49）。
2. **仓库的 FLOPs 计数（2MKN）和论文 Figure 2 的精度加权计数（BF16:FP8:FP4=1:0.5:0.25）不是同一套口径**，任何跨两者的数字比较都需要先说明这一点（见 Q66）。
3. **页面内部存在真实的小不一致**：Roofline Tab 用 H100 峰值 989，MFU Tab 用 989.5，拐点因此也不完全一致（见 Q82）。
4. **"60 GFLOPs/token"的 MFU 假设和"约 59.93 GFLOPs/token"的 MoE 核心激活参数粗估"看起来很接近"，但这是两个独立口径，后者不能用来"验证"前者**（见 Q93）。
5. **本地 `model.py` 明确记录落后于官方固定版本**，共享索引缓存释放条件、输出投影前量化时机两处存在已知差异，涉及这两点的具体实现细节需要加上这个边界（见 `docs/architecture/assumptions.md` Unverified 节）。
6. **Roofline 流量模型假设"权重一次批处理只读一次"**，如果被误解为"1024 个历史 token"而不是"1024 行一起处理"，结论会完全错误（见 Q85）。
7. **"反向约为前向两倍"是统一近似，单 token 精确计算下误差约 36%**，这个近似的适用边界需要在被追问时立刻说清楚，不能当成精确结论陈述（见 Q70）。
8. **本仓库没有任何完整、经过验证的训练 FLOPs 或真实训练 MFU 数据**，所有 MFU 数字都是标注清楚的教学假设，这是全场最容易在追问下不小心"听起来像是在说真实数据"的地方（见 Q92、Q100）。
9. **审计记录里 189 项模块×模式记录中有 44 项仍是"Unverified"，0 项"Mismatch"**——"零不匹配"容易被误解成"全部已证实"，实际上 Unverified 只是"证据缺失"，不是"已确认正确"（见 `docs/architecture/calculation_audit.md`）。
10. **DSpark 一次前向能并行出 5 个草稿位置，不等于解码一定提速 5 倍**，实际收益取决于接受率、验证开销和调度策略，报告里对 DSpark 的描述也强调的是"吞吐量"而非固定倍率（见 Q24）。

---

**彩排建议：** 先只回答附录 A 的 20 题，每题控制在 20–45 秒；再按附录 B 的 5 条链练习连续追问的节奏；最后请同事随机抽 3–5 道非 ★ 题，检验能否在不看本文件的情况下，仍然准确说出"这是假设还是已验证"的边界。
