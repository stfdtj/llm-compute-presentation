# 计算定义与验证记录

本文保留初版单 Token 定义与手算记录；连续 Decode、Speedup、Weight Absorption 和已实现 Roofline 的最新验证见 `extension_validation.md`，浏览器最新结果见 `../../src/compute_lab/qa/browser-results.json`。

验证日期：2026-10-13。范围：单层、单序列、一次 Decode；此前历史 S 个 token，包含当前 token 后可见 N=S+1。页面整批内存乘 Batch；整批 FLOPs 另列，不改变每序列曲线。

## 符号与统一约定

D：隐藏维度；H：Query / 递归头数；Hkv：KV 头数；dk：Q/K 头维度（MLA 为内容维度）；dv：Value 头维度；W：含当前 token 的窗口；r：KV latent rank；rq：Query rank；dr：MLA 解耦位置维度；di：索引头维度；k：含强制 local block 的选择预算；block：块大小；b：数据精度字节；bstate：递归状态字节。

FMA 计 2 FLOPs，矩阵乘法 `[M,K] × [K,N]` 近似为 2MKN，严格乘加为 MN(2K−1)。索引、比较、max 与 Top-K 不属于浮点加乘操作，排除但不声称它们没有时间成本。开关启用时，加、减、乘、除、exp 各简化计 1；关闭时只保留矩阵运算，GDA 的外积和逐元素更新也明确排除。

读取 L 个位置的 Softmax 核心，scale 计 HL；稳定 Softmax 计 H(4L−1)：减最大值 L、exp L、求和 L−1、除法 L。比较不计。exp 的 1 FLOP 是教学计数约定，非 GPU 指令数。统一不计 RoPE 旋转、全局 Norm、Bias、残差、FFN、量化与通信。

## 各架构公式（启用标量）

| 架构 | Projection P | Core C | Additional E |
|---|---|---|---|
| MHA | 4DH(dk+dv) | 2HN(dk+dv)+H(5N−1) | 0（声明范围内） |
| GQA | 2D(H+Hkv)(dk+dv) | 同 MHA | 0（声明范围内） |
| SWA | 同 GQA | 2HL(dk+dv)+H(5L−1)，L=min(N,W) | 0（声明范围内） |
| MLA | 2D rq + 2rq H(dk+dr) + 2D(r+dr) + 2Hdk r + 2Hr dv + 2HdvD | 2HN(2r+dr)+H(6N−1) | 0（选定路径与范围内） |
| MSA | P_GQA + 2Ddi(Hkv+1) | 2HL(dk+dv)+H(5L−1) | Hkv N(2di+1)，含显式索引 scale |
| GDA | 已知 4DH(dk+dv)+4DH+2DHdv | H(7dkdv+2dv) | 未核验；E=null |

已完成范围内 Total=P+C+E；GDA 完整 Total=null，但保存 `known=P+C`。跨架构 Ratio=MHA / 当前所选口径；没有 GDA 完整模块比值。

MSA：块数 ceil(N/block)，末块可见长度 ℓ=(N−1) mod block+1。采用总预算 k 中包含 local block 的约定：L=ℓ+[min(k,ceil(N/block))−1]block。其余已选历史块均完整，每组可以选不同块但数量相同，按 H 个查询头聚合 FLOPs。不模拟真实索引排序。训练 KL 不参与 Decode；实际内核可省去不会改变排序的正 scale，此页显式计入。

MLA：qC 从 rq 维 Query latent 恢复，按头做 qLat=qC WUKᵀ；scores=qLat Ccacheᵀ+qR KRcacheᵀ；z=P Ccache；o=z WUV；再做 O 投影。仅聚合后的当前输出恢复 dv，不展开历史 K/V；保留因子的路径与完全预融合权重路径有不同投影开销。不计 RMSNorm / RoPE，不能称为官方完整模块。

GDA：将论文状态 S 改名 R，避免与历史长度 S 混淆。每头状态布局 [dv,dk]：A=αRprev；u=Ak；δ=β(v−u)；Rnew=A+δkᵀ；o=Rnew q。核心逐项为 dk dv（衰减）+2dk dv（预测读取）+2dv（误差与门）+2dk dv（外积与状态相加）+2dk dv（输出读取）。外积只有每元素一次乘法；严格两次 matvec 各 dv(2dk−1)，整个核心严格乘加为 7Hdkdv。完整 ShortConv / Norm / 非线性门与缓冲待核验。

## 内存定义

逻辑容量包含当前 token，并乘 Batch：

| 架构 | 元素 / 字节公式 | Context 增长 |
|---|---|---|
| MHA | N H(dk+dv)b | 线性 |
| GQA | N Hkv(dk+dv)b | 线性；头数更少 |
| SWA | min(N,W) Hkv(dk+dv)b | 到 W 后饱和 |
| MLA | N(r+dr)b | 线性；压缩表示 |
| MSA | N[Hkv(dk+dv)+di]b | 线性；仍保留全历史 |
| GDA | Hdkdv bstate，部分状态 | 不依赖 N；未含卷积缓冲 |

默认 BF16 / FP16 为 2 B，FP32 为 4 B，FP8 为 1 B；GDA 状态默认独立 FP32。权重、临时激活、对齐、量化 scales 不计。HBM 访存和权重访存未验证，字段为 null。

## 手算样例

D=4，H=2，Hkv=1，dk=2，dv=3，S=2（N=3），W=2，k=2，block=2，di=2，r=2，rq=2，dr=2，Batch=1，BF16，GDA 状态 FP32。

| 架构 | Projection | Core | Extra | Total | 缓存 / 状态字节 |
|---|---:|---:|---:|---:|---:|
| MHA | 160 | 88 | 0 | 248 | 60 |
| GQA | 120 | 88 | 0 | 208 | 30 |
| MLA | 168 | 106 | 0 | 274 | 24 |
| SWA | 120 | 58 | 0 | 178 | 20 |
| MSA | 152 | 88 | 15 | 255 | 42 |
| GDA | 240（已知） | 96 | 未知 | 未知；已知 336 | 48（部分） |

例如 MHA 的投影 Q/K/V/O 分别为 32/32/48/48；QKᵀ=24，P×V=36，scale=6，Softmax=22。合计 248；严格点积与同一标量约定合计 218。GQA 的 KV 投影减少到 16/24，但核心保持 88。

## 验证结果

`node src/compute_lab/tests.js`：**23 项全部通过**。

- 小矩阵近似 / 严格计数，包括 K=1 外积边界。
- 上表的 MHA、GQA、MLA、MSA 手算；GDA 递归核心与未知字段。
- 80 组 Hkv=H 的 GQA → MHA 退化测试。
- SWA 等价与窗口饱和，S=0 的当前 token 边界。
- MSA 末块、local block、完整索引增长与缓存。
- MLA latent-space 与展开历史 K/V 的代数等价（容差 1e−12）。
- GDA 递归分解与论文 Eq.10 数值等价（容差 1e−12）、严格 / 近似核心。
- 递归核心与状态对 S 不变；精度、Batch、标量开关。
- 参数、训练 JSON 校验；安全整数保护；导出中未知流量与总量为 null。

`node src/compute_lab/browser_test.mjs`：**21 项全部通过**。本地 Chrome 无界面检查独立 HTML `file://` 打开、六行结果、参数同步到场景 / 曲线 / 架构图、无效头数与空输入、所有路径与步骤、GDA 核心、容量曲线 / 对数坐标、显隐、标量开关、来源弹窗、Roofline Tab、MFU 独立导入、390px 自适应与零浏览器异常。浏览器未进行真实硬件性能测量。

已查看桌面 / 窄屏截图；页面无全局横向溢出，窄屏导航与结果表可在自身区域横向滚动。截图与结果在 `qa/`。所有显示 G/T 数值使用十进制 10⁹ / 10¹²；MiB / GiB 使用二进制 2²⁰ / 2³⁰。

## 依据与后续证据需求

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：MHA 的投影与全注意力。
- [GQA](https://arxiv.org/abs/2305.13245)：多个查询头共享 KV。
- [Mistral 7B](https://arxiv.org/abs/2310.06825)：因果滑动窗口。
- [DeepSeek-V2 §2.1](https://arxiv.org/html/2405.04434v5#S2.SS1)：低秩 KV、解耦位置分支。计数由本页所选矩阵执行路径推导；不复制论文性能结果。
- [MiniMax Sparse Attention §3.1 / §4.1](https://arxiv.org/html/2606.13392v1#S3.SS1)：分组索引、共享 Index K、分块选择。特定内核的 scale 省略与选择规则需按固定源码版本继续核验。
- [Gated Delta Networks Eq.10 / §3.4](https://arxiv.org/html/2412.06464v3#S3.SS1)：门控递归和完整块的额外组件。后续需固定源码、ShortConv 长度、门参数化、Norm 与激活计数才能补全模块。

Roofline 后续需 HBM Bytes Moved 的模型或测量证据、相应精度的硬件上限；MFU 后续需完整模型的训练 FLOPs 和实际全设备吞吐量。接口位于 `../../src/compute_lab/model.js` 的 `rooflineInput` / `validateTraining`，浏览器调用入口在 `window.LLMComputeLab`。两者本轮均未生成估计值。
