# DSpark 辅助路径：源码核对与验收

## 数据来源与实现边界

已阅读并比较：

- 用户指定的[官方 main/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)，网页显示更新提交 `4dd75a8`；重点复核 Transformer.forward / forward_spec、DSparkBlock、DSparkAttention、Markov / Confidence Head。
- [官方 inference/config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/config.json)；DSpark 相关配置与项目固定快照一致。
- 项目固定版本 `5254448103b6254fc646dacbca1cbbf20e9216c1` 的源码与配置；manifest SHA256 检查通过。构建继续读取此固定配置，不替换原有计算审计资料。
- V4.1 技术报告 §2.4.3（本地 `references/deepseek/technical_report.txt` 第 785 行起），确认三层、五位置、128 窗口及置信度调度。
- [DSpark 论文](https://arxiv.org/html/2607.05147v1) §2.1、§3.1、§3.2、Algorithm 1，用于补足官方模型文件没有实现的调度、验证与接受流程。

线上相关 DSpark 函数与本地固定实现的逻辑一致。未声称完整模型文件的其他部分与 main 逐字一致。

## 逐步对应源码

| 图中阶段 | 实际执行与形状（B=1） | 固定 model.py 位置 |
| --- | --- | --- |
| 主干正常路径 | 40 个 Block：Encoder 0–19，Decoder 20–39；最终 HC 合并 / RMSNorm / LM Head / sample | Transformer.forward，1245–1275 |
| 来源 A | 进入 37、38、39 Block 前，各自取 h.mean(dim=2)：四路输入 `[1,S,4,5120]` → `[1,S,5120]` | 1264–1274 |
| 条件构造 | 三层最后一维 Concat → `[1,S,15360]`；乘 `[15360,5120]` → RMSNorm → main_x `[1,S,5120]`；仅 Draft 1 拥有 main_proj / main_norm | DSparkBlock.__init__ / forward_embed，1112–1138 |
| 来源 B | 主模型输出 token 作为锚点 t0；建立 `[t0,n,n,n,n]`，n=128799；共享 Embedding → `[1,5,5120]`，扩展四路 → `[1,5,4,5120]` | 1131–1138 |
| 三个 Draft Blocks | mtp.0 → mtp.1 → mtp.2；每层同时计算五位置；HC / Norm / DSparkAttention / HC / FFN / HC；MoE 128 中选 3 + 1 shared | DSparkBlock、Block、ModelArgs.get_moe_config |
| Draft Attention | 本层草稿生成 Q / KV；main_x 用本层 W_KV 生成上下文 KV 并写独立环形缓存；上下文窗口最多 128 + 当前草稿 5 行，一次 Attention，草稿块内双向可见 | 1016–1077 |
| Attention 输出 | Inverse RoPE → reshape 8 组 → FP8 量化 / 反量化 → W_OA / W_OB；本层无 Global / Compressor / Indexer | DSparkAttention.forward |
| 共享 LM Head | 最后四路 HC Pre 合并，保留 Norm 前 H3；最终 RMSNorm 后调用共享 LM Head，生成 `[1,5,129280]` base logits | forward_head，1147–1148 |
| Markov Head | 前一 token 查独立 `[129280,256]` 表 → `[1,256]`，乘 `[256,129280]` 得偏置；加到对应位置 base logits，再顺序 sample | 1081–1089、1152–1157 |
| Confidence Head | 使用未最终 RMSNorm 的 H3 与 Markov Embedding；拼成 5376 维，乘 `[5376,1]`，返回 raw scores `[1,5]` | 1092–1100、1158–1160 |
| 返回 | output_ids `[1,6]` 含一个旧锚点 + 五个候选；logits `[1,5,129280]` 已加 Markov bias；confidence `[1,5]` 未 sigmoid | forward_head |

“Attention 输入状态”在源码的具体读取位置是进入整个 Block 之前；不是该层内部 attn_norm 之后的张量，也不是该层最终输出。图与详情明确采用原代码的 mean(四路)，没有错误替换为本层 HC Pre 权重合并。

## 三层与五位置

`n_layers=40` 决定主干深度；`n_mtp_layers=3` 是附加草稿网络的深度。内部索引 40、41、42 是检查点编号，不表示普通 token 必须经过 43 层。

`dspark_block_size=5` 决定五个预测位置：输入一个锚点 + 四个 Noise，三层 Transformer 一次处理整个五位置块。锚点位置也产生一个预测。随后仅轻量 Markov Head 进行五步顺序采样，因此最终产生五个新候选；不是每个 Draft Block 对应一个候选，也不是五次完整草稿前向。

## Prefill 与 Decode

`start_pos=0` 时 forward_embed 虽然仍构造五位置输入，但 DSparkBlock.forward 仅调用 Attention 的 main_x KV 路径以填充上下文窗口，跳过 Q、草稿 KV、HC、MoE 和最终 Head，forward_spec 返回 None。

`start_pos>0` 时，参考实现接收单 token main_x，执行完整三层草稿网络与最终 Head。各 Draft 层上下文 KV 由各自权重产生；当前草稿 KV 只在本次 Attention 拼接中使用，不写入上下文环形缓存。草稿 Attention 没有主干 Global KV / Indexer 路径。

## 论文验证路径与未提供的代码

主图绿色路径表示论文流程。Confidence 原始分数经 Sigmoid / 校准后用于估计条件接受概率；前缀生存率是这些概率的累计乘积。Algorithm 1 结合引擎 SPS / 当前负载选择每个请求的前缀长度 m∈[0,5]，采用有因果约束的提前停止，不能用“按置信度独立接受 token”替代。

主模型验证输入为锚点与选定前缀 `[t0,c1,…,cm]`，以因果掩码一次得到目标分布 p。与草稿实际采样分布 q 对比，按 `min(1,p(ci)/q(ci))` 检查连续前缀；首次拒绝时丢弃余下草稿并补纠正 token，全部接受时可补充下一 token。Confidence 不承担最终接受判定。

以下在官方 `model.py` 中没有完整实现：

- raw confidence 的 Sigmoid / STS 校准与参数。
- 引擎吞吐曲线的实测数据、负载调度与完整验证调用。
- 拒绝采样纠正、接受后输出提交、多 token 缓存推进与回滚循环。

这些内容的原理可从论文确认，但没有把它们伪装成 forward_spec 已实现的行为。页面中的候选颜色与前缀长度控件只是手动教学示意，不运行真实推测解码，也不承诺每轮接受五个。

## 陌生技术读者的七项检查

| 验收问题 | 主图中的直接证据 |
| --- | --- |
| 为什么 40 层另有 3 层？ | 上方独立蓝色主干；下方虚线可选 DSpark 框，注明不属于 L0–39 |
| Draft 输入来自哪里？ | A 标记连接目标层 Block 输入；B 标记连接主模型输出 t0；分别进入条件投影与 Noise / Embedding |
| 三个 Draft Layers 做什么？ | 三个有独立编号的 Block：DSparkAttention + HC / MoE；点击显示内部顺序与维度 |
| 为什么能生成多个候选？ | 明示“3 层 = 深度，5 位置同时通过每层”；Head 输出五组 logits，Markov 顺序采样 c1…c5 |
| 与普通 AR 有何不同？ | 蓝色每轮正常输出一个 token；紫色五位置成块计算，绿色一次主模型验证多个位置 |
| 为什么候选仍需验证？ | Speculative Tokens 明示尚未接受，必须经过同一主模型目标分布 / 接受规则后才能进入 Accepted Tokens |
| 如何减少串行计算？ | 主图底部一行说明：草稿三层并行位置、轻量采样、主模型一次验证多个位置；详情说明接受多个时减少大模型调用轮数 |

人工截图复核后修正了 H3 标记的重叠，进一步将 Confidence 输入标为“HC Pre（Norm 前）”，并在主图补上验证输入中的锚点。没有靠扩写主图段落替代输入、运算和输出连线。

## 验收与访问

默认打开 `deepseek_v4_1_interactive.html` 显示“主干 + DSpark”总览；“完整架构”保留原有 Attention / HC / MoE / Engram 细节。“主干 + DSpark”按钮可以返回总览。`#architecture` 直接进入原有详细图，旧图回归测试显式使用此入口。

- 构建：`.venv/Scripts/python.exe src/deepseek/build.py`。
- DSpark：`node src/deepseek/verify_dspark.cjs`，检查七个详情页、五步 Markov 交互、0 / 1 / 5 候选前缀、文字边界、视图切换与截图。
- 原有回归：`npm run test:browser:deepseek`，已包含新 DSpark 验收；同时执行源码配置 / 公式审计。
- 原有 `architecture.svg` 没有因 DSpark 改变：DSpark SVG 以独立组注入 HTML；原 Attention 保护测试与逐元素比较仍通过。
- 桌面截图：`src/deepseek/qa/dspark_overview_1920.png`（1920×1080）、`dspark_overview_1366.png`（1366×768）；详情截图 `dspark_detail_inputs / blocks / heads / verification.png`。
- 本次仅验证页面、公式形状与交互，没有加载模型权重或运行 GPU 推理。
