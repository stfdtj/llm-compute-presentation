# LLM Compute Presentation

两个可离线打开的教学网页，从架构、参数量、FLOPs 到 Roofline / MFU。

- [DeepSeek 架构交互页](deepseek_v4_1_interactive.html)：参数、前向、局部反向与信息流。
- [LLM Compute Lab](index.html)：六种 Attention 对比、Roofline、MFU 与演示模式。

下载后可直接双击 HTML；阅读成品不需要安装依赖。仓库包含两套网页的源码、测试、固定版本审计资料和参考图，不依赖原来的 sharing 工作区，也不包含模型权重。

## 两个网页分别用来做什么？

### 1. DeepSeek 架构交互页：理解一个真实模型内部怎样计算

入口：[deepseek_v4_1_interactive.html](deepseek_v4_1_interactive.html)。

这个网页适合从模型结构入手学习。它把本项目 DeepSeek-V4.1-Flash 的主要模块画成可点击的架构图，让你沿着数据流理解输入经过哪些模块、矩阵维度怎样变化，以及权重如何参与计算。

点击模块后，可以切换三种说明：

| 模式 | 用来回答的问题 |
|---|---|
| 参数量 | 这个模块有哪些可学习权重？矩阵形状是什么？保存了多少参数？ |
| FLOPs（前向） | 当前 token 从输入得到输出，需要执行哪些运算？计算量怎样推导？ |
| FLOPs（反向） | 输入梯度、权重梯度如何计算？它们与前向矩阵有什么关系？ |

例如，点击“Query 第一步：压缩”，可以观察 `[1,5120] × [5120,1280] → [1,1280]`，再对照参数量与前向 FLOPs。点击“专家混合 / MoE”，可以理解全部专家参数、当前 token 激活参数、Router 和共享专家的区别。

“信息流 / Data Flow”提供分步演示，用来观察隐藏状态、共享 KV、Local KV 和 Attention 输出怎样流动。其中的矩阵教学演示进一步解释位置拼接、索引读取、分数计算、共同 Softmax 和输出投影。

**适用场景：**学习模型源码、核对局部模块的参数和计算公式、讲解 Dense/MoE 执行路径、在分享时展示真实矩阵形状。图支持缩放、拖动与区域聚焦；不需要逐个点击所有模块。

**阅读边界：**常规模块主要围绕当前一个 token 展开；Attention 的示例历史长度在面板中注明。反向内容是局部数学推导，不能直接作为已验证的完整训练实现或全模型训练 FLOPs。详细假设、来源和未核验项可以展开查看。

### 2. LLM Compute Lab：比较计算开销，理解硬件性能与训练效率

入口：[index.html](index.html)。

这个网页适合在理解矩阵和模块之后，进一步探索“改变架构会减少哪些工作”“为什么 FLOPs 少了却不一定同比例加速”。它提供可调参数、实时计算、图形和数字代入，包含三个 Tab：

| Tab | 主要用途 | 可以做的操作 |
|---|---|---|
| Attention Comparison | 比较 MHA、GQA、MLA、SWA、MSA、GDA 的声明范围内 FLOPs、缓存/状态容量和计算量比值 | 修改维度、历史长度与生成数；切换 Single Token / 连续 Decode，以及 Core / Module；查看图表和计算步骤 |
| Roofline Analysis | 将计算量、估算访存量和硬件规格结合起来，判断理论上更受算力还是带宽限制 | 使用默认 H100 BF16 Dense 模板；点击 AI=100/500 教学点；调整带宽、算力和历史长度，观察屋顶线与瓶颈变化 |
| MFU Estimator | 理解训练模型计算吞吐量占硬件理论峰值的比例 | 修改假设训练 FLOPs/token、全设备吞吐量、设备数和峰值；演示速度减半，以及计算减半、速度翻倍的情景 |

右上角“演示模式”会将三个 Tab 的主图、关键结果和动态结论集中展示，适合组内分享。切换模式保留输入，完整参数、表格与推导仍能展开。

Roofline 默认自动计算并绘图，访存量来自明确的假设模型。实心 Attention 点表示理想性能上界，菱形表示独立教学假设；只有提供有效、同范围的 Runtime，才会绘制相应空心实测点。MFU 默认 50% 及两个对照情景都是假设训练数据。

**适用场景：**比较不同 Attention 的工作量，解释缓存容量与访存流量的差别，理解 Roofline 的两类限制，练习 MFU 公式，以及现场修改参数说明结果变化。

**阅读边界：**Attention 研究的是单层自回归 Decode；MFU 示例研究的是全模型 Training，二者不能直接混用。各架构有各自的执行路径与统计边界，GDA 完整总量未核验时保持未知。理论 FLOPs 比值、Roofline 理想时间比值与实测速度比值分别解读。

### 两个网页如何配合使用？

两个网页的关系是：架构页帮助理解”真实模块如何执行”，Compute Lab 帮助分析”工作量与硬件限制如何共同影响效率”。它们是两个独立教学工具，不会自动把 DeepSeek 全模型的数据传入 Lab。Lab 中的 MLA 等示例也不能直接视为本地 DeepSeek 完整模块的复现。

讲解时可以先用架构页展示投影与 MoE 等真实模块，再用 Lab 展示 MHA/GQA、Roofline 和 MFU 的开销对比；现场分享可启用 Lab 演示模式，将关键图表和结论集中展示。

## 目录与修改入口

根目录保留网页成品，打开即可使用。源码及模块测试集中在 `src/`，截图和浏览器检查记录在各模块的 `qa/`；模型、来源资料和技术报告分别收纳在 `models/`、`references/` 与 `docs/`。

```text
llm-compute-presentation/
├─ index.html                        Compute Lab 网页成品
├─ deepseek_v4_1_interactive.html     DeepSeek 网页成品
├─ README.md                         使用与开发导航
├─ src/
│  ├─ compute_lab/                   Lab 模板、计算、交互、构建与测试
│  └─ deepseek/                      架构图、教学内容、审计、构建与测试
├─ models/DeepSeek-V4.1-Flash/        静态分析用模型源码与配置
├─ references/deepseek/              固定版本官方源码、报告与许可
├─ docs/
│  ├─ architecture/                 架构页审计与假设
│  ├─ flops/                        FLOPs 技术报告与审计
│  └─ compute_lab/                  Lab 验证说明
├─ scripts/FLOPs_calculation_verify.py  独立计算复核
└─ requirements*.txt / package*.json  开发依赖与锁定
```

| 路径 | 内容 |
|---|---|
| `src/compute_lab/` | HTML 模板、JS 计算与交互、CSS、打包脚本、测试和 QA 记录 |
| `src/deepseek/` | 架构 SVG 生成、教学数据、HTML 模板、审计与浏览器测试 |
| `references/deepseek/` | 固定 revision 的参考源码、配置、技术报告和 SHA256 |
| `models/DeepSeek-V4.1-Flash/` | 原项目的本地模型源码与配置；用于静态分析，不是完整推理部署 |
| `scripts/FLOPs_calculation_verify.py` | 根目录计算复核，也是 DeepSeek 教学构建的依赖 |
| `docs/flops/FLOPs_calculation_forward.txt`、`docs/flops/FLOPs_calculation_audit.md` | 计算报告与边界 |
| `docs/architecture/calculation_audit.md`、`docs/architecture/assumptions.md` | 架构页构建时生成的审计说明 |

修改源码后再构建，不直接编辑成品 HTML。子目录 README 保留详细功能和验证历史。

## 安装开发依赖

推荐 Python 3.12 与 Node.js 22 或更新版本。在仓库根目录执行（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
```

Python 依赖用于公式转换和数值梯度测试；Playwright 仅供浏览器测试。Linux/macOS 使用 `.venv/bin/python`。首次安装需要联网，安装后常规构建和计算测试不需要联网。

## 构建与计算验证

```powershell
.venv\Scripts\python.exe src/compute_lab/build.py
.venv\Scripts\python.exe src/deepseek/build.py
npm run test:lab
.venv\Scripts\python.exe scripts/FLOPs_calculation_verify.py
.venv\Scripts\python.exe src/deepseek/verify_audit.py
.venv\Scripts\python.exe src/deepseek/verify_single_token.py
.venv\Scripts\python.exe src/deepseek/verify_calculations.py
```

Lab 构建仅需 Python 标准库；DeepSeek 构建同时更新模块 JSON、SVG 和审计文档，不下载或执行模型权重。

## 浏览器验证

```powershell
npm run test:browser:lab
npm run test:browser:deepseek
```

默认使用本机 Chrome / Edge。可设置 `COMPUTE_LAB_CHROME` 为浏览器可执行文件绝对路径。没有本机浏览器时，可运行 `npx playwright install chromium`，并把该 Chromium 路径传给 Lab 测试；DeepSeek 测试也可自动使用 Playwright 安装的 Chromium。

测试启动独立无界面浏览器，不需要网页服务器。QA 图片、JSON 结果和固定几何基线随仓库保留；浏览器 profile、虚拟环境、node_modules 与 Python 缓存不进入 Git。

SHA256 按原始字节核对，`.gitattributes` 禁止 checkout 自动换行转换。不同平台重建若改变生成文件换行，应核查后有意更新基线，不应绕过审计。

## 可选：更新官方审计资料

日常构建使用仓库内固定快照，不运行下载脚本。主动更新来源时才安装：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-sources.txt
```

依次运行 `src/deepseek/fetch_audit_sources.py` 和 `fetch_technical_report.py` 后重新审计；`inspect_report.py` 可渲染报告页。更新会改变 revision / SHA256，应单独审查。

参考源码与报告的来源见 `references/deepseek/manifest.json`。官方源码许可见该目录的 LICENSE；原始参考图保留为 `src/deepseek/qa/reference.png`，用于架构几何对照。

## 数据边界

Roofline 流量是声明假设下的估算；MFU 教学数据是假设值。局部反向示例不代表已验证的完整训练实现，GDA 未核验的完整总量仍为未知。历史截图和 JSON 是对应版本的验证记录，不是训练或硬件性能测量。

## 本次迁移验证（2026-10-14）

已在本仓库新建虚拟环境并从依赖清单安装，不借用原工作区的 Python 包或 node_modules。两个网页均独立重建成功，生成文件与原成品逐字节一致。

- Lab：65 项计算测试、98 项浏览器检查通过。
- DeepSeek：10 类源码与算术审计通过；83 个数值等式、160 个矩阵乘法和 323 个 SVG 几何元素核对通过；数值梯度测试通过。
- 浏览器：189 个模块/模式组合、参数绑定、五步信息流、六段矩阵演示和多尺寸布局通过，固定数据/图形/审计基线 SHA256 保持不变。

这些检查验证教学网页及其计算口径，不涉及真实 GPU/NPU 训练或性能测量。
