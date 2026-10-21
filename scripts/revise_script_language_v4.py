"""Apply paragraph-level editorial revisions and preserve formulas/navigation verbatim."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
import hashlib, json, re

root = Path(__file__).resolve().parents[1]
source_md = root / 'presentation_script_pre_v3_restructured.md'
source_docx = root / 'presentation_script_pre_v3_final.docx'
target_md = root / 'presentation_script_pre_v4_language.md'
target_docx = target_md.with_suffix('.docx')
before = source_md.read_text(encoding='utf-8')
original_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [source_md, source_docx]}

# Each entry is a separately reviewed paragraph, identified by its original opening.
edits = {
'版本：': '版本：2026-10-21 语言优化版。基于《presentation_script_pre_v3_final.docx》。',
'我们先看这张整体架构图。': '我们先看这张整体架构图。文本经过 Token Embedding，变成 Hidden States；图片经过 Vision Encoder 和投影，也会进入语言模型的特征空间。下面主要沿着文本这条路径讲。模型配置的 Hidden Size 是 5120，每个文本 Token 都使用这个特征宽度。',
'这里的 Encoder 和 Decoder 都保持因果约束。': '这里的 Encoder 和 Decoder 都保持因果约束，每个位置使用当前及之前允许访问的信息。前后两段在同一条主干中依次执行，分别承担对应的计算工作。我们接下来主要看它们怎样生成和使用 Global KV。',
'这套 CED 设计希望': '这套 CED 设计通过调整 Prefill 的执行路径，减少长 Prompt 的计算成本。按照报告描述的优化方式，多数输入位置主要计算 Encoder，Decoder 使用 Encoder 的表示构建 Global KV，再通过有限范围的计算补齐局部窗口状态。公开参考代码的 Backbone Forward 目前循环执行全部 40 层。计算参考代码的 Prefill FLOPs 时，我们按这条完整执行路径统计。',
'旁边的 DSpark 是': '旁边的 DSpark 是同一模型中的辅助 Draft 分支，在权重里放在 mtp 命名空间下。它在 40 层主干之外，另外配置了三个 Draft Blocks。主干承担 Target 的预测和验证，DSpark 提供候选；两边共享 Embedding 和输出 Head 等组件，Draft Blocks 使用自己的参数。我们先看一个主干 Layer 的计算，再看两条分支怎样配合。',
'现在看 Encoder 内部。': '现在看 Encoder 内部。一个 Layer 先把残差通路组合成 Attention 的输入，完成 Attention 后把结果接回残差通路，再经过 MoE，最后传给下一层。沿着这条路径，我们可以逐项分析各个模块的计算成本。',
'Attention 要让当前 Token': 'Attention 让当前 Token 使用历史信息。Local Sliding Window 读取最近最多 128 个有效位置，提供局部上下文。从 Layer 2 开始，Encoder 还会构建压缩的 Global KV，把两个位置的信息压成一个全局位置，用来访问更早的历史。',
'有了 Global KV，': '有了 Global KV，Indexer 会用自己的 Query 和 Index Keys 计算相关性分数，选出最多 512 个可见的全局位置。Main Attention 随后根据这些位置索引读取 Global KV，并结合 Local KV，用自己的 Query 计算 Attention 权重，再聚合 Value，得到输出。',
'所以，这里的 Local 和 Global': 'Local 和 Global 共同提供 Main Attention 使用的历史信息。它们在同一次 Attention 中参与评分和聚合。实际读取的位置满足因果约束，有效位置数量随当前历史长度变化，上限分别是 128 个 Local 和 512 个 Global 位置。',
'Decoder 的 Layer 20 使用': 'Decoder 的 Layer 20 使用 Encoder 最终表示来构建全局数据，这里的位置压缩比是 1。后面的 Decoder 层复用这份 Global KV。其中 Layer 24、28、32、36 会用当前层的 Indexer Query 重新选择位置，属于 Reindex；继续使用已有选择结果的层属于 Reuse。',
'我们把这三种模式区分开': '这三种模式对应三种执行方式：Full 创建全局数据并选择位置，Reindex 复用全局数据并重新选择位置，Reuse 同时复用全局数据和选择结果。Decoder 还用 Candidate Blocks 限制后续重选范围，减少重复搜索。各层继续执行自己的 Query、Local Attention 和 MoE，复用则减少了全局数据生成和索引中的重复工作。',
'进入 Attention 或 MoE 前，': '进入 Attention 或 MoE 前，Pre 按混合系数把四路组合成一路 5120 维输入。计算完成后，Post 把模块结果分配回四路，Comb 对原来的四路残差信息做组合，再将两部分合在一起传下去。后面的 Shape Trace 会沿着这个过程展示维度变化。',
'主干的 Router 给': '主干的 Router 先给 Expert 打分，再结合路由偏置选择 Top-6。选定 Expert 后，Router 从原始分数中取出对应项，经过归一化和缩放，得到输出的组合权重。后面的参数计算，我们就选择 Layer 2 的 MoE 作为例子。',
'每个 Expert 内部都有': '每个 Expert 内部都有 Gate、Up 和 Down 三个投影。Gate 和 Up 把输入映射到中间维度，Gate 的激活结果和 Up 的结果逐元素相乘，再通过 Down 回到 Hidden Size。选中的 Routed Expert 输出按路由权重合并，再加 Shared Expert 输出。后面计算时，我们会分别统计保存的全部 Expert 权重，以及一个 Token 使用的 Expert 权重。',
'这边还有 Engram，': '这边还有 Engram，它放在 Encoder 的 Layer 1 和 14。它根据局部 Token 组合进行哈希查表，把读到的记忆信息注入 Hidden States。每次计算访问表中的一部分条目。统计 Total Parameters 时计入整张表，统计实际工作量时则按本次查表访问和后续投影计算。',
'现在看推理时这些模块': '现在看推理时这些模块怎样配合。页面用“我、喜欢、先”和 A、B、C、D、E 作为教学 Token 符号。我们沿着这组符号，依次看输入准备、候选生成、Target 验证和结果提交。',
'这里先把实现边界说明白。': '公开 model.py 提供了 Backbone Forward 和 DSpark Forward。公开 generate.py 使用逐 Token 生成路径，循环调用 Backbone Forward，每次取一个 next token；forward_spec 和完整的 Verify、Accept、Reject 循环由服务端进一步接入。接下来的页面把这些模块与验证逻辑串起来，演示完整的 Speculative Decoding 流程。',
'普通 Prefill 并不会': '在初始化阶段，调用 start_pos 等于 0 的 forward_spec，会先建立 Draft 自己的窗口 KV，随后返回 None。后续 start_pos 大于 0 的调用，才会执行候选生成。这样，DSpark 可以使用已经建立的历史状态继续预测。',
'我们从已经有历史、': '我们从已经有历史、正在继续 Decode 的位置进入这个例子。当前 Backbone 处理“喜欢”，经过 40 层、Final Norm 和 LM Head，选出了“先”。这个由 Target 选出的 Token 就是本轮 Anchor，作为已确认的起点交给 DSpark。“先”随后还会进入 Target Forward，届时建立它对应位置的 Target KV。',
'同一次 Backbone Forward 还提供': '同一次 Backbone Forward 还提供 Hidden Features。代码在进入 Layer 37、38、39 之前，各取一次输入 Hidden States，对四路 mHC 状态求平均，得到各自的 5120 维向量，再拼成 15360 维，交给 DSpark。',
'另一边，五个输入槽位': '另一边，五个输入槽位初始化为“先、Noise、Noise、Noise、Noise”。第一个位置放 Anchor 的 Token ID，其余四个位置放固定的 Noise Token ID。Noise 是模型定义的占位 Token，用来初始化后续预测位置。',
'这五个 Token ID 经过': '这五个 Token ID 经过共享 Embedding，成为五个 5120 维向量，再扩成四路 mHC 状态。如果显式写出 Batch 维度，Shape 是 Batch 乘五乘四乘 5120。Backbone Features 则沿着上下文路径进入 Draft Attention，与这五个槽位一起参与计算。',
'五个位置一起经过 mtp': '五个位置依次经过 mtp 的第零、一、二个 Draft Block。每个 Block 都处理全部五个位置，最后得到五组 Hidden States。这里的三表示网络深度，五表示本轮候选位置数。',
'Draft Attention 同时使用': 'Draft Attention 同时使用主干特征形成的历史窗口和当前五个槽位。五个槽位在 Draft 内可以互相读取，完成三个 Block 后，得到各自的 Hidden States。后面的 Target Verification 则按因果前缀计算每个位置的预测分布。',
'这五个 Hidden States 经过': '这五个 Hidden States 经过 mHC 合并和 Norm，再由共享 LM Head 产生五行 Base Logits。接下来，Markov Head 按位置顺序修正这些 Logits，并依次选择五个 Draft Token。',
'所以，重的三个 Draft Blocks': '三个 Draft Blocks 先一起处理五个位置，随后 rank 为 256 的 Markov Head 按 A 到 E 的顺序完成修正和选择。第一槽中的“先”提供起始条件，用来预测 Anchor 后面的候选 A；之后每个新选出的候选，再作为下一个位置的 Markov 条件。',
'与此同时，Confidence Head': '与此同时，Confidence Head 把每个位置的 Hidden States 和对应的前词 Markov Embedding 拼起来，通过线性投影输出一个标量分数，五个候选对应五个分数。服务端可以用这些分数辅助选择验证长度，最终接受结果由 Target Verification 决定。分数的概率校准和调度策略还需要服务层实现；当前页面示例选择验证全部五个。',
'到这一步，候选 A、': '接下来，候选 A、B、C、D、E 交给 Target Backbone。本例中，“先”也等待进入 Target，因此验证输入依次是“先、A、B、C、D、E”六个位置。已有 Prompt 的 Cache 保留在前面，这六个位置在一次多位置 Forward 中经过正常的 40 层 Backbone。',
'一次 Forward 能验证多个': '候选给定后，Target 可以把它们排成一批位置，在各层并行计算。Causal Mask 限定每个位置可使用的前缀：“先”的预测使用 Prompt 和“先”，A 的预测再加入 A，后面依次扩展。这样，每一行 Logits 都对应一个合法的历史前缀，一次 Forward 就能为多个候选提供验证所需的预测。',
'这不代表 Target 提前': 'Target 先计算这些候选前缀下的预测分布，再根据验证结果保留连续通过的部分。公开 Backbone 的输出 Head 默认返回最后位置的 Logits。完整验证服务需要取得所有待验证位置的 Logits，并完成对应的位置和状态管理。',
'先看 Greedy Verification。': '先看 Greedy Verification。A 和对应 Target Top-1 一致，所以接受 A。B 也通过验证。到了 C，Target 在“Prompt 加先、A、B”这个前缀后的 Top-1 是 X，与候选 C 出现分歧，因此拒绝 C。',
'后面的 D 和 E 也一起': 'D 和 E 的预测依赖含有 C 的候选前缀，所以在 C 被拒绝后一起作废。本轮确认文本是“Prompt 加先、A、B、X”。其中 A 和 B 是接受的两个 Draft Token，X 是 Target 提供的修正 Token。',
'再看 Cache。': '再看 Cache。“先”、A、B 都已经作为验证输入经过 Target，所以这三个位置的 KV 可以提交。C、D、E 对应的临时状态失效。X 刚从 Logits 中选出，作为下一轮待输入的 Token；它的 KV 会在下一次 Target Forward 中建立。',
'下一轮把 X 作为': '下一轮把 X 作为 Target 的当前输入，继续产生新的预测和 Features，再组织下一次 Draft。服务端按照确认后的前缀提交或回退状态，并同步 Draft 窗口、Target Cache、压缩状态和其他历史状态。这里描述的是逻辑上的有效范围，具体的缓存恢复和显存管理由服务实现完成。',
'页面中 A 和 B 都是': '页面中 A 和 B 都是 q 等于 0.8，p 等于 0.6，所以 alpha 等于 0.75。教学随机数固定为 0.5，小于 0.75，因此它们通过。C 的 q 等于 0.8，p 等于 0.2，alpha 等于 0.25。随机数 0.5 大于这个阈值，所以 C 被拒绝，D、E 随后作废。',
'概率模式下，拒绝后': '概率模式在拒绝候选后，使用完整词表上的正残差构造修正分布。对每个词计算 p 减 q，保留正值，再归一化，然后从这个分布采样修正 Token。',
'在当前教学分布里，': '在当前教学分布里，归一化后 r(X) 等于 0.70，r(Y) 等于 0.25，其余词分到剩余的 0.05。页面固定的抽样值是 0.82，最后选出 Y。这次确认文本就是“Prompt 加先、A、B、Y”。有效 Target KV 到 B，Y 留给下一次 Forward。',
'如果 A 到 E 全部通过，': '如果 A 到 E 全部通过，我们就保留五个候选，再使用 E 位置的 Target 预测得到 Bonus Token，本例是 Z。此时有效 KV 到 E，Z 是下一轮待输入的 Token。Z 会和前面示例中的 X、Y 一样，在下一次 Target Forward 中建立对应的 KV。',
'DSpark 可能更快，': 'DSpark 的加速来自一次 Target Forward 确认多个位置，从而减少逐 Token 调用，并让多个位置分摊权重读取成本。每轮的总开销还包括 Draft、多个位置的验证和状态管理。实际速度取决于接受长度、原有 Batch 的硬件利用率，以及各部分执行时间。接受率较低或额外开销较大时，这种方法也可能更慢。',
'现在我们只跟踪 Layer 2，': '现在我们选择 Layer 2，跟踪一个 Token 的完整数据流。这个 Layer 同时有 Local、Global 和 Indexer，适合把前面的模块串起来看。页面把一个 Token 写成一行，省略单独的 Batch 轴；四路残差输入显示为 1 乘 4 乘 5120。',
'W_QB 的权重是 1280': 'W_QB 的权重是 1280 乘 32768，输出就是 1 乘 32768。32768 等于 64 乘 512，所以可以 Reshape 成 1 乘 64 乘 512，也就是 64 个 Query Heads，每个 Head 512 维。这个 Reshape 使用原有元素，调整它们的分组方式。',
'这里写的是便于计算': '这里按数学公式中的输入维乘输出维写权重矩阵；代码的 Linear 权重通常按输出维乘输入维保存。两种写法互为转置，参数数量相同。后面的 Forward 和 Backward 都统一使用 Y 等于 XW。',
'Query 旁边是 KV 路径。': 'Query 旁边是 KV 路径。Local 投影把当前输入变成 512 维的 KV 表示，写入窗口 Cache。一个窗口的逻辑 Shape 是 128 乘 512，有效位置受历史长度限制。64 个 Query Heads 共享读取这些 KV，分别计算各自的 Attention 权重。',
'Layer 2 还把相邻': 'Layer 2 还把相邻两个位置的特征，通过投影、门控和聚合压成一个 512 维的全局表示。在两个位置凑齐后，这个表示加入 Global KV Cache。如果已有 S 个压缩后的全局位置，Cache 的逻辑 Shape 就是 S 乘 512。',
'所以 Main Query 的': '这里 Main Query 的 Shape 是 64 乘 512，Indexer Query 的 Shape 是 32 乘 128。Indexer 先选出 Top-K 位置索引，Main Attention 根据这些索引读取 KV，再用 Main Query 计算权重并聚合输出。',
'接下来不是直接拿': '接下来，模型把 64 个 Head 分成八组，每组 4096 维，经 Grouped W_OA 变成每组 1024 维。八组合起来是 8192 维，再经 W_OB 回到 5120 维。这个结果作为 Attention 模块的输出，接回 Hidden States。',
'六个 Routed Experts 和一个': '六个 Routed Experts 和一个 Shared Expert 的结果合并后，模块输出仍然是 5120 维。最后经过 mHC 回到四路，也就是 1 乘 4 乘 5120，再传给下一层。各个投影在模块内部调整特征宽度，模块输出则恢复到后续计算使用的 Shape。',
'和 Layer 3 对比一下，': '接着看 Layer 3 的复用路径。它读取 Layer 2 的 Global KV 和 Top-K，继续计算本层的 Query 和局部 Attention。这样，我们就能沿图看出每项输入的来源：一部分由当前层计算，一部分由前面的层提供。',
'三个投影没有额外 Bias，': '这三个投影都采用无 Bias 的 Linear，参数集中在三个权重矩阵中。因此，一个 Expert 的核心权重参数量是三倍的 11796480，也就是 35389440，大约 3540 万。激活函数和逐元素相乘直接处理这三个投影产生的结果。',
'Layer 2 的打分 Router 还有': 'Layer 2 的打分 Router 还有一个 5120 乘 384 的权重矩阵，共 1966080 个权重。把它加进去，一层的 Expert 加 Router 矩阵共有 13626900480 个核心权重。这个小计覆盖矩阵权重；完整清单中的路由偏置、量化 scale 等其他项分别统计。',
'但是对一个 Token，': '接下来统计一个 Token 使用的权重。它经过六个 Routed Experts 和一个 Shared Expert，对应的 Expert 权重规模是七乘 35389440，得到 247726080，大约 2.48 亿。再加 Router 矩阵，是 249692160，大约 2.50 亿。',
'当前是文本 Token，': '当前是文本 Token，页面的 Active 口径再加 384 个文本路由偏置，显示 249692544。Total 则同时包含文本和图像两组路由偏置。',
'这里的 Total 和 Active 都限定': '这里的 Total 和 Active 都统计一层 MoE。不同 Token 可能选择不同 Expert，所以一个 Batch 使用的 Expert 并集还可能更大。Dense FFN 通常每个 Token 都使用整套 FFN 权重；MoE 按每个 Token 的选择执行对应 Expert，我们就按这些实际执行的矩阵统计工作量。',
'参数量也不等于字节数。': '从参数量换算存储大小时，还要结合权重精度。BF16、FP8 和 FP4 的每元素字节数各有对应取值，量化存储还会使用 scale 等额外数据。接下来我们先统计运算次数，看看这些矩阵在一次执行中要完成多少次乘加。',
'这一小步把 1280': '这一小步把 1280 维的中间结果归一化，输出仍是 1280 维。接下来的 FLOPs 小计覆盖两个 Linear 的矩阵乘法，RMSNorm 和 RoPE 的计算另行统计。',
'W_QB 接收 1 乘': 'W_QB 接收 1 乘 1280，权重是 1280 乘 32768。二乘一乘 1280 乘 32768，得到 83886080 FLOPs，也就是 83.88608 MFLOPs。之后的 Reshape 调整元素分组，浮点乘加计数为零。',
'这还只是主干 Query': '这个结果覆盖主干 Query 的两次矩阵乘法。完整 Forward 还需要继续统计 KV、Attention 聚合、MoE、mHC 和输出 Head；启用 DSpark 时，再加入对应分支的工作量。逐项累加时，我们按每个模块实际执行的次数和处理的位置数计算。',
'现在从 Inference 切换到': '现在从 Inference 切换到 Training。训练先根据预测和目标计算 Loss，再从 Loss 沿计算图向前传递 Gradient。到了当前 Query 模块，我们假设后续 Attention 的 Backward 已经计算出对 Query 输出的梯度，记作 dQ。Query 的 Backward 接收这个 dQ，继续计算当前模块的权重和输入 Gradient。',
'按同一个 2MKN 口径，': '按同一个 2MKN 口径，这两次分别是 83886080 FLOPs，相加 167772160 FLOPs。这里 dW 的归约维度等于一，逐项统计标量乘法和加法会得到另一组计数。为了与网页和前向计算一致，我们继续使用 2MKN 的近似口径。',
'刚得到的 dH′ 不能': '刚得到的 dH′ 接下来进入 RMSNorm 的 Backward。它使用前向中间值，把 dH′ 转成 dH，同时计算缩放参数的 Gradient。然后把 Shape 为 1 乘 1280 的 dH 交给 W_QA。这个过程属于完整 Backward 路径，Norm 的计算量另行统计，当前小计继续覆盖两个 Linear。',
'这里 Backward 约等于': '这两个 Linear 的 Backward 约等于 Forward 的两倍，因为每个 Linear 都计算了权重和输入两种 Gradient。其他模块可以按各自的 Backward 路径逐项统计，额外重计算、优化器更新和通信也分别核算。这里分析的是训练中的 Gradient 计算，前面 DSpark 的 Draft 和候选验证属于推理执行路径。',
'现在看硬件这一层。': '现在看硬件这一层。刚才我们算出了 FLOPs，接下来把数据搬运也加入分析。这张通用 Attention 页面用 Roofline 展示算力和带宽两种限制，图里的数据点按页面声明的 Attention 范围统计。后面我们再切回单个 Query 投影的算例。',
'这里 F 是运算次数，': '这里 F 是运算次数，B 是选定内存层级上的搬运字节数，BW 是该层级带宽。我们接下来选择 HBM，并累计计算期间发生的数据读取和写入。同一份数据反复访问时，每次 HBM 搬运都计入对应流量。',
'为了继续使用同一个 Query': '现在切到仓库现有的 Compute Efficiency 页面，继续使用 Query 算例。这里的两个点分别对应一行输入和 1024 行输入。Roofline 和 MFU 都用每张设备 989.5 TFLOP/s 的教学参照，带宽取 3.35 TB/s，精度按 BF16 Dense。后面的结果都按这组硬件假设计算。',
'我们先估算这两次 Linear': '我们先估算这两次 Linear 的 HBM 流量。假设权重在这一批计算中各读一次，输入读一次，最终输出写一次，中间 1280 维结果写回再读出，所以中间项要算两次。每个 BF16 元素是两个 Byte。这是两个 Linear 的简化流量模型；RMSNorm、融合和片上 Cache 的影响留在更完整的分析中处理。',
'再看 M 等于 1024。': '再看 M 等于 1024。运算量变成 99321118720 FLOPs，约 99.32 GFLOPs；搬运量变成 179830784 Bytes，约 179.83 MB。相除得到约 552.3 FLOPs 每 Byte。这一批的运算次数增加了 1024 倍，同时让 1024 行输入复用每批读取一次的权重，所以计算密度提高很多。',
'这里 1024 表示': '这里 1024 表示一次共同处理的 1024 行，可以来自已知 Prompt，也可以来自多个请求组成的 Batch。Decode 同样可以按请求批处理。我们用 M 等于一和 M 等于 1024，观察两种矩阵规模下的权重复用和计算密度。',
'这些是理想下界，': '这些结果表示当前假设下的理想时间下界。实际延迟还受两次投影的前后依赖、Kernel 启动、矩阵形状、融合和 Cache 命中影响，需要通过测量得到。这里对两次投影合并使用 max，得到的是较宽松的下界；逐算子分析还可以进一步考虑执行顺序。',
'这就解释了为什么 FLOPs': '按这个结果选择优化方向，过程就比较直接了。主要受带宽限制时，可以减少数据搬运、提高权重复用；主要受算力限制时，可以减少运算量、提高计算效率。DSpark 通过减少逐 Token 调用、提高一次 Forward 的有效产出争取加速，实际收益同样取决于这些硬件条件。',
'现在从一个算子的性能限制，': '现在从一个算子的性能限制，转到整个训练的效率。MFU 用有效模型训练 FLOPs 每秒，除以全部设备的理论总峰值。计算时，我们先确定整步模型工作量，再结合完整训练步的时间，得到有效吞吐量。',
'这里 F_step 是': '这里 F_step 是整步有效模型的 Forward 加 Backward FLOPs，t_step 是完整训练步的时间，N_GPU 是设备数，P_peak 是单卡同一精度口径下的峰值。分子使用全设备的 Token 总量计算整步 FLOPs，分母中的设备数则用于汇总集群峰值。',
'这一页使用每步': '这一页使用每步 100.6M Tokens，页面把这个 Batch 数标为报告中的配置。每 Token 60 GFLOPs、2048 张设备和每步六秒是教学假设，其中 60 GFLOPs 直接作为本例的训练工作量输入。按这些条件算出的 MFU 约为 49.6%；真实训练的 MFU 需要结合全模型工作量和实际 Step Time 测量。',
'第二步，算集群总峰值。': '第二步，算集群总峰值。2048 张设备，每张 989.5 TFLOP/s，相乘得到 2026496 TFLOP/s，换成基本单位就是 2.026496 乘十的十八次方 FLOP/s。上一步给出整步运算次数，这一步给出硬件每秒的峰值吞吐量，接下来再用 Step Time 把两者联系起来。',
'接近 50% 的意思是：': '接近 50% 的意思是：按这里声明的有效模型计算量核算，训练的有效吞吐量相当于设备总峰值的一半。这个比例把模型工作量、训练速度和硬件峰值联系在一起，用来描述整个训练系统的计算效率。',
'我们这里的有效模型 FLOPs': '这里的有效模型 FLOPs 按模型必需的 Forward 和 Backward 统计。额外的激活重计算、优化器更新和通信等开销，通过完整 Step Time 反映在 MFU 中。报告真实 MFU 时，需要核算全模型实际训练路径的 FLOPs，并使用相同范围的实测时间。对 MoE，我们按实际执行的 Expert 工作量计算。',
'MFU 也不是越高': '我们还可以结合吞吐量看一次架构优化。假设每 Token 的有效计算量减半，吞吐量翻倍，两者乘积保持相同，MFU 就可以基本保持原值，同时完成相同 Token 数的训练时间缩短一半。评估优化效果时，可以一起看 MFU、训练吞吐量、总时间和最终模型效果。',
'Roofline 帮助我们分析': 'Roofline 帮助我们分析主要算子的潜在限制，MFU 帮助我们衡量整体有效吞吐量。对带宽受限的主要算子，可以先看数据搬运和复用；对计算密度较高的算子，可以结合 Profiling 检查算子实现、通信和流水线等待。分析整个训练 Step 时，再按各部分的耗时和执行关系汇总这些结果。',
}

lines = before.splitlines()
counts = {key: 0 for key in edits}
for i, line in enumerate(lines):
    for key, replacement in edits.items():
        if line.startswith(key):
            lines[i] = replacement
            counts[key] += 1
            break
assert all(n == 1 for n in counts.values()), counts
after = '\n'.join(lines) + '\n'
def paragraphs(s):
    return [re.sub(r'^#{1,2} ', '', x) for x in s.splitlines() if x.strip()]
def operations(s):
    return [x for x in s.splitlines() if x.startswith('【操作：')]
def formulas(s):
    return [x for x in s.splitlines() if re.match(r'^(?:P\(|F_|F\(|B\(|AI|T_|MFU =|P_total|dW_|dH′ =|dX =|Y =|alpha =|r\(v\)|40 ×)', x)]
assert operations(before) == operations(after)
assert formulas(before) == formulas(after)
assert [x for x in before.splitlines() if x.startswith('## ')] == [x for x in after.splitlines() if x.startswith('## ')]

ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ET.register_namespace('w', ns)
ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
with ZipFile(source_docx) as z:
    files = {name: z.read(name) for name in z.namelist()}
document = ET.fromstring(files['word/document.xml'])
nodes = document.findall(f'.//{{{ns}}}body/{{{ns}}}p')
old_text = [''.join(p.itertext()) for p in nodes]
assert old_text == paragraphs(before), 'Source DOCX has changes not represented by source Markdown'
new_text = paragraphs(after)
assert len(nodes) == len(new_text)
for p, text in zip(nodes, new_text):
    ts = p.findall(f'.//{{{ns}}}t')
    assert len(ts) == 1
    ts[0].text = text
files['word/document.xml'] = ET.tostring(document, encoding='utf-8', xml_declaration=True)
assert not target_docx.exists() and not target_md.exists(), 'New version already exists'
target_md.write_text(after, encoding='utf-8')
with ZipFile(target_docx, 'w', ZIP_DEFLATED) as z:
    for name, data in files.items():
        z.writestr(name, data)
with ZipFile(target_docx) as z:
    for name in z.namelist():
        if name.endswith(('.xml', '.rels')):
            ET.fromstring(z.read(name))
    restored = ET.fromstring(z.read('word/document.xml'))
    assert [''.join(p.itertext()) for p in restored.findall(f'.//{{{ns}}}body/{{{ns}}}p')] == new_text
assert original_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [source_md, source_docx]}
report = {'paragraphs_reviewed': len(nodes), 'paragraphs_rewritten': len(edits), 'navigation_preserved': len(operations(after)), 'formula_paragraphs_preserved': len(formulas(after)), 'source_files_unchanged': True, 'docx_markdown_text_equal': True}
evidence = root / 'docs/script_revision_evidence'
(evidence / 'language_v4_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
