# 技术、公式与视觉核对报告

交付：22 页 16:9 可编辑 PPT，83 组原生点击显现。Speaker Notes 包含原稿章节、段落索引和点击顺序。全部页面以真实 PPTX 导出 PDF，再生成 PNG 检查；另渲染 47 个关键点击状态。

## 1. 来源与差异

主依据为用户提供的 `pre_v2_反向传播简化版_Roofline_MFU.docx`、本地模型 config / 源码、独立 Query 教学页。已阅读旧两页 PPT 的内容，但新稿重新设计为图形场景。

必须保留的差异记录见 [source_differences.md](source_differences.md)。尤其注意：Word 附录和链接分支远端页面仍含旧四卡 MFU；本地新版和本次用户明确要求为 2048 卡、6 秒、49.6%。本交付采用后者，不混用旧值。远端依据：[app.js 的 MFU_PRESETS / DEFAULTS](https://raw.githubusercontent.com/stfdtj/llm-compute-presentation/fix/deepseek-original-attention-details/compute-efficiency/app.js)。

## 2. 核心形状、参数和 FLOPs

前向统一采用 `Y=XW` 记法。源码如按 `[out,in]` 保存权重，转置记法不同但元素总量相同。

| 项目 | 形状 | 参数 | 单 token 前向，2MKN |
|---|---|---:|---:|
| W_QA | [5120,1280] | 6,553,600 | 13,107,200 |
| RMSNorm γ | [1280] | 1,280 | 不含在 Linear 小计 |
| W_QB | [1280,32768] | 41,943,040 | 83,886,080 |
| 两个 Linear 小计 | 32768 = 64 × 512 | 48,496,640 | 96,993,280 |

含中间 RMSNorm 的这段参数为 48,497,920；PPT 的 96.99 MFLOPs 仅统计两个 Linear，不含 RMSNorm / RoPE / Attention。正常主干 Decode 的该项乘 40 得 3,879,731,200 FLOPs，约 3.8797 GFLOPs，不代表完整模型前向。

一个 Expert：`3×5120×2304=35,389,440`。每层保存 384 Routed + 1 Shared，单 token 使用 6 Routed + 1 Shared。专家总权重 13,624,934,400，激活专家权重 247,726,080。Router 矩阵 1,966,080；加 Router 后分别为 13,626,900,480 和 249,692,160。选择偏置单独处理，这些数值不是全模型参数量。

矩阵教学例采用 `A=[[1,2,3],[4,5,6]]`、`B=[[1,2],[3,4],[5,6]]`，结果 `C=[[22,28],[49,64]]`。逐项计算正确；精确 20 FLOPs，2MKN 近似 24 FLOPs。

## 3. 反向计数边界

设 `X[M,K]、W[K,N]、G=dY[M,N]`：

```text
dW = XᵀG      → [K,N]；供优化器更新 W
dX = GWᵀ      → [M,K]；继续向前面的模块传递

前向精确乘加   = MN(2K−1)
输入梯度精确   = MK(2N−1)
权重梯度精确   = KN(2M−1)
```

| 两个 Query Linear，M=1 | 精确乘加 | 统一近似 |
|---|---:|---:|
| Forward | 96,959,232 | 96,993,280 |
| Backward dX 小计 | 96,986,880 | 96,993,280 |
| Backward dW 小计 | 48,496,640 | 96,993,280 |
| Backward 合计 | 145,483,520 | 193,986,560 |

单 token 的 dW 是外积，没有跨行求和，因此近似与精确值差异明显。“反向约两倍”指统一矩阵乘法近似，并要求输入与权重都需要梯度。上述均不含梯度缓冲累加、RMSNorm 反向、优化器和重计算。PPT 主画面保留 Word 的约 194 MFLOPs，并明确标为近似。

Query 子图中的 dQ 来自后续 Attention 的反向，不是直接把 Query 当成整个模型的最终输出。反向必须经过 RMSNorm，不能跳过。局部数学示意不构成完整模型训练实现的验证。

## 4. Roofline

仅两个 Query Linear；BF16 每元素 2 Bytes；十进制 MB。每个权重矩阵每批读取一次；中间 H 写一次、读一次。排除 RMSNorm、缓存、融合与真实分块影响。

```text
F(M) = 96,993,280 M FLOPs
B(M) = 96,993,280 + 80,896 M Bytes
AI = F/B
P_roof = min(P_peak, AI × BW)
AI_ridge = P_peak/BW
```

| M | FLOPs | Bytes | AI，FLOPs/Byte |
|---|---:|---:|---:|
| 1 | 96,993,280 | 97,074,176 | 0.99917 |
| 1024 | 99,321,118,720 | 179,830,784 | 552.30321 |

H100 SXM 参照用 BF16 Dense 989 TFLOP/s、3.35 TB/s，拐点约 295.22。MFU 保留原算例的 989.5，这是舍入差异。官网带稀疏性的数值不能直接用作 Dense 峰值。[NVIDIA 规格](https://www.nvidia.com/en-us/data-center/h100/)、[NVIDIA Dense BF16 参照](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/benchmark/E2E_BENCHMARK.md)。

图中点位是理论性能上界，不是 profiling；M 为一次共同处理的行数，不表示同一请求的未来 token 可任意并行。

## 5. MFU

```text
F_step = 100.6×10⁶ × 60×10⁹ = 6.036×10¹⁸ FLOPs
P_total = 2048 × 989.5×10¹² = 2.026496×10¹⁸ FLOP/s
Available FLOPs = 6 × P_total = 12.158976×10¹⁸ FLOPs
MFU = 6.036 / 12.158976 = 49.6423383%
```

100.6M batch 数来自仓库保存的技术报告；60 GFLOPs/token、2048 GPUs、6 s 为假设；989.5 为示例规格。页面持续显示 **Illustrative Example / Not Measured**。60 GFLOPs/token 不是已验证的完整训练计算量。MFU 按有效前向和反向工作量统计，不把额外激活重计算加成有效模型工作量。[PaLM 定义](https://arxiv.org/html/2204.02311v5#S4.SS1)。

## 6. 实际检查结果与范围

- 22/22 页实际渲染并查看截图；矩阵、低秩、MoE、梯度、Roofline、MFU 的 47 个关键状态另行渲染、检查。
- 原生时间线含 83 个点击组；全部目标形状存在，时间节点与形状 ID 唯一，22 页 Notes 存在。
- LibreOffice 独立导入保留全部 83 个点击组；MFU 填充使用同次点击下的延迟显现，约 0.9 秒完成。
- 实际 PDF 文字审计：无缺失正文、无超出页面的字形、无未解释的文字相撞；上下标作为独立字形核对。矩阵高亮层覆盖原单元格是有意设计。
- 原生图形审计：无越界图形，无非包含关系的意外矩形相交。所有教学图均为可编辑图形和文本，没有整页位图。
- 修复了第一轮截图中的轴标题相撞、Ridge / Batch 标签挤压、梯度箭头穿字、底部说明相撞等问题，并重新渲染。

通用静态 QA 会把“文本放在矩形内”误判成卡片间距问题，也会把动画高亮层判为重叠；原始诊断保留在 `qa/`。最终验收采用原生场景几何检查、实际 PDF 字形检查和逐页截图复核，而不把这些已核实的有意覆盖当作排版错误。

本环境未完成 Microsoft PowerPoint 的交互式放映回放；已完成真实 PPTX 渲染、独立动画导入以及关键状态图像核验，不把这三项说成微软放映实测。PDF 是最终可见状态的静态预览，不播放动画。
