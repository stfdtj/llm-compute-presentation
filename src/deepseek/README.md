# DeepSeek-V4.1 单 token 交互教学

## Original layout / expanded Attention blocks

Enlarged Encoder/Decoder views now default to a layer-by-layer sequence matching the supplied drawio: Encoder 0–19, Decoder 20–39, with per-layer jump and two-layer stepping. Switch to **Attention operations** for the projection/Indexer computation diagram. Blue dashed outlines identify CSA2 while leaving mHC/FFN outside. Stream labels are h⁽¹⁾–h⁽⁴⁾ once each. mHC means **Manifold-Constrained Hyper-Connections**; this implementation uses the **Single-Pass mHC** variant.

The original page, 1672×941 canvas and four detail-block positions remain. Encoder and Decoder Attention are updated in place; click **Enlarge detail** or choose C · Encoder / C · Decoder in the original focus menu. The larger drawings are part of the same SVG and include Query, Local KV, Compressor, inline Indexer scoring/Top-k, KV gathering and output projection. Indexer nodes highlight inline; the explicit Full Indexer calculations link retains the eb9356c calculation tool. DSpark is a small overview entry and an original-menu focus target. No replacement workspace or tab bar is built.

Source discrepancies, retained geometry checks, tests and screenshots: [validation](../../docs/architecture/attention_inplace_validation.md).

直接用浏览器打开 `../../deepseek_v4_1_interactive.html`。这是一个离线单文件，内含 SVG、样式、交互、数据和 MathML 公式，无需启动服务器。

点击模块查看参数量、前向 FLOPs 或反向梯度。切换模式保留当前模块；点击空白或按 Esc 关闭。滚轮缩放、拖动平移，按 `0` 或点击“适合窗口”复位。Tab / Enter 可操作图中模块。

## 本次教学范围

普通模块只跟踪当前一个 token，输入采用行向量 `[1,5120]`。面板按“作用、计算公式、矩阵维度、符号说明、数值结果、一句话理解”组织，多阶段运算分步展示；跨层合计默认折叠。

Attention 的数值例子假设此前已有 1024 个历史 token，当前是第 1025 个。历史 KV 从缓存读取，不重新执行历史 token 的投影。Compressor 面板提供 3 / 4 个 token、ratio=2 的分组例子，以及 ratio=1 不等待配对的例子。

Shared KV 投影：每层 2,621,440 个参数，当前 token 前向 5,242,880 FLOPs，输入和权重两个局部梯度合计 10,485,760 FLOPs。40 层前向合计 209,715,200 FLOPs。

去重后的逻辑权重：Text Backbone 551,566,180,464；Engram 196,928,504,320；Draft 14,225,362,530；Vision 与对齐 485,268,480；合计 763,205,315,794。原 748,494,684,784 仅为 Text + Engram。固定历史长度下，文本与 Engram 当前一步估算为 35,584,521,265 次操作，与旧数值独立核对一致；修正其“全模型”统计边界，不含 Vision / Draft 执行，不代表实测 GPU 指令数或延迟。反向内容是局部数学推导，完整未确认部分标为 Unverified。

原图的图形、位置、颜色、箭头和分组保持不变。Encoder / Decoder Attention Detail 底部新增 Indexer 入口，可展开独立的流程细节区域。

## 可编辑源文件

- `teaching_data.py`：当前使用的 63 个教学条目，含三种模式与 Compressor 示例。
- `module_data.json`：自动导出的可读数据，公式保留 LaTeX。
- `build_diagram.py`：SVG 几何与标签。
- `indexer_diagram.py`、`indexer_ui.html`：Indexer 流程、跨层复用与计算面板入口；配置来自 `references/deepseek/config.json`，计算复用原 `module-data`。
- `page_template.html`：页面布局、面板样式与交互。
- `flow_ui.html`：独立信息流开关、五步演示、SVG 路径叠加、缓存写入示例及说明；构建时内嵌到原 HTML。
- `audit_model.py`：直接按固定官方配置和源码构造建立参数总账与独立操作计数；不读取旧 HTML 或旧 FLOPs 报告。
- `audited_data.py`：将统一参数注册表、权重归属、Assumptions 和复核记录绑定到教学条目，生成根目录两份审计文档。
- `audit_ui.html`：可学习参数表、当前权重 / 公式 / 矩阵同步、MoE Total / Activated、全局假设和数值结果审计。
- `build.py`：将公式编译为 MathML，生成独立 HTML。
- `build_data.py`：保留的旧版计算映射，不参与当前构建。

构建命令：`python src/deepseek/build.py`。构建依赖已改为根目录 requirements.txt 中固定版本的 latex2mathml；安装与独立构建见根目录 README。

审计证据是 `audit_sources` 中的固定官方版本（revision 5254448103b6254fc646dacbca1cbbf20e9216c1）及 51 页技术报告；SHA256 和本地文件差异另列。旧报告仅为对比对象。常规教学步骤保持简洁，详细来源、假设与精确乘加放在可展开的结果审计中。`teaching_data.py` 的旧几何与教学基线在构建时经 `audited_data.py` 更新，最终统一数据见 `module_data.json`。

## 验证

- `verify_single_token.py`：核对具体数字等式、矩阵乘法维度、分组示例，并与 `before_single_token.svg` 比较所有非文字图形属性。
- `verify_audit.py`：10 类 PASS / FAIL；正式配置、57 类源码 AST 构造、技术报告、维度、参数、FLOPs 算术、梯度 shape、非线性计数、去重、多层、MoE、HTML 数据一致性。标准库即可离线复跑，结果为 `qa/audit_validation.json`。
- `verify_calculations.py`：核对单 token 计算总账，并用小规模数值差分验证 Query、RMSNorm、共享 KV Attention、SwiGLU、Router 和 HC 的梯度。需要 NumPy。
- `verify_browser.cjs`：离线 Chrome 检查 189 个模块/模式组合、真实点击、模式切换、缩放平移、关闭、区域聚焦与响应式边界。需要根目录 package-lock.json 固定的 Playwright 和 Chrome / Edge（或 Playwright Chromium）。
- `verify_parameters.cjs`：参数表 / MathML / 矩阵同步、梯度类别、数值证据、四类 Assumptions、零参数操作与 MoE 边界，覆盖 189 组合和 1920 / 1366 / 768 / 390 窗口。结果为 `qa/parameter_validation.json`。
- `qa/single_token_validation.json`、`qa/calculation_validation.json`、`qa/browser_validation.json`：各项结果。
- `qa/overview_1920.png`、`qa/shared_kv_parameter.png`、`qa/shared_kv_forward.png`、`qa/shared_kv_backward.png`、`qa/compressor_examples.png`：浏览器实际截图。

这些验证没有加载模型权重，也没有运行 GPU 内核或完整训练。

## 参数与计算审计

点击模块即可见 Learnable Parameters，点击阶段按钮或完整公式步骤同步高亮权重、对应 MathML 符号及矩阵示意。梯度是运行时数据，参数表中的权重才是更新对象。详细角色、共享、存储轴与源码位置默认折叠。

工具栏 Assumptions 展示 Configuration / Scenario / Approximation / Unverified。点击任意计算结果可展开该项公式、Shape、近似 / 精确乘加、假设与固定源码证据。线性前向主结果用 2MKN；单 token dW 的精确乘加是 KN，不能称 2KN 为精确数值。推理缓存读取与训练历史梯度边界分别说明。

根目录 `docs/architecture/calculation_audit.md` 列出旧结果、修正内容、状态、原因与代码位置；`docs/architecture/assumptions.md` 汇总全局和模块条件。状态统计单位为 63 模块 × 3 模式的 189 主记录；当前 Verified 63 / Estimated 82 / Unverified 44 / Mismatch 0，38 项旧结果或解释边界已修正。完整变化见 `audit_results.json`，独立参数清单见 `independent_audit.json`，源码构造检查见 `qa/source_shape_validation.json`。

复跑：`python src/deepseek/audit_model.py`、`python src/deepseek/build.py`、`python src/deepseek/verify_audit.py`、`python src/deepseek/verify_single_token.py`。梯度检查需 NumPy，请使用根目录 requirements.txt 安装的 NumPy；浏览器检查运行 `node src/deepseek/verify_browser.cjs`、`node src/deepseek/verify_parameters.cjs` 和 `node src/deepseek/verify_flow.cjs`。运行 HTML 本身不需要这些构建与验证工具。

## 信息流演示

### Indexer 全局位置选择

点击 Encoder / Decoder Attention Detail 底部橙色的 `Indexer → Top-512 → Global KV ↗`，展开流程并打开现有 Indexer 计算面板。也可以从右上角聚焦菜单选择 `E · Indexer Detail`。点击空白关闭计算面板，保留流程；“返回架构”恢复进入前的视角。

Presentation 建议先选 Full（Layer 20），沿 Query / Key → Scoring + ReLU → Head Aggregation → Top-K → Global KV Gather → Main Attention 讲解，再切换 Reindex（Layer 24）与 Reuse（Layer 21）。淡化的节点表示该层跳过的计算；层选择器可检查 2–39 层的真实配置。0–1 层只有局部 SWA，不运行 Indexer。所有层仍执行自己的主要 Attention。

Query 来自当前层归一化后的 1280 维低秩表示，投影为 32 × 128；Index Key 来自 Compressor 的 512 维 RoPE 前 latent，仅在源层完成新条目时投影、追加缓存。Head Weights 来自当前 X 的可学习投影。Top-K 输出整数位置，Gather 读取已有 Global KV；与最多 128 个 Local KV 位置一起供 Main Attention 使用，Main Query 与 Index Query 相互独立。

KV source layers 为 `[2,8,14,20]`，Index source layers 为 `[2,8,14,20,24,28,32,36]`。Reindex 复用 Global KV 和 Index Keys，重新评分；Reuse 连同已有位置索引一起复用。依据 `references/deepseek/inference/model.py` 的 `Indexer.forward`、`Attention.forward` 与 `SharedAttentionRuntime`。

“评分范围、缓存与计算口径”展开项说明实际评分行数 C、先评分后掩码的参考实现与候选域优化的区别。Top-K 512 不是评分长度。各节点直接调用已有计算记录；切换流程模式不会生成一套新 FLOPs。反向面板继续保留已有 Unverified 状态。Indexer 面板的相关链接指向 Compressor、Global KV 和 Main Attention，避免误连 MoE Top-6；原计算数据不变。

运行 `node src/deepseek/verify_indexer.cjs`，检查真实点击、全部 38 个适用层、三种计算模式、缓存复用、键盘与空白关闭、缩放平移、信息流切换、四种窗口及计算数据 / 原图保留。结果为 `qa/indexer_validation.json`，截图为 `qa/indexer_*.png`。

### 原有五步信息流

在原页面打开“信息流 / Data Flow”，使用上一步、下一步和重新播放。蓝色表示 Hidden State 主干，橙色表示共享 KV，绿色表示 Local KV，紫色表示 Attention 汇总。箭头可悬停查看来源与用途；第 3 步可以逐行写入缓存示例，第 4 步可分别追踪 Q / Global / Local。点击 Global KV 可定位第 2 步，说明面板可返回原计算说明。关闭开关恢复进入演示前的视角。

本地源码确认：

- `config.json`：40 层，KV source 为 2、8、14、20；20–39 层 ratio=1；64 个 Query Head，Head 宽度 512，Local 窗口 128。
- `model.py:978` 附近的 `Block.forward`：跨层传递四路残差；HC 合并与 Attention RMSNorm 后得到单行 Attention 输入。
- `Compressor.forward`、`Attention._compress_kv`：层 20 对自身 Attention 输入投影、归一化、尾部 RoPE、量化并写入共享缓存，随后层复用。
- `Attention._window_kv`：各层独立产生 Local KV 并维护环形缓存。
- `Attention.forward` 与 `kernel.py` 的 `sparse_attn_kernel`：局部和共享 KV 在位置维拼接，选中位置共同归一化；64 个 Query Head 读取共享的 512 维向量，该向量同时用于 Key 和 Value。

教学示意：原图的 Encoder / Decoder 分段及 H_enc 单行标签不是独立源码模块；真正的层 20 分支来自 HC 合并、归一化后的 Attention 输入。Cache 辅助框、无压缩三行例子、移动圆点、简化 Softmax 公式用于说明原理，不是新增层或真实运行中的张量。页面说明已明确这一区别。

`verify_flow.cjs` 与 `qa/flow_validation.json` 覆盖五步导航、三次缓存写入、Q/Global/Local 筛选、计算模式保留、原 SVG 几何保留、关闭和离线运行。`qa/flow_step_1.png` 至 `qa/flow_step_5.png` 为分步截图，另有 1366 / 768 / 390 像素窗口截图。


Step 4 的 Layer 21 矩阵教学演示（增量功能）

打开原页面的信息流并进入第 4 步，即出现六段演示：三条输入、位置维 Concat、索引读取、Attention Scores、共同 Softmax 与 Weighted Sum、8 组输出投影。支持前后切换、键盘左右键 / Home / End、重看，以及改变 2 维教学 Query 后观察权重和输出。小屏矩阵区独立横向 / 纵向滚动。

`flow_attention_ui.html` 只添加教学 UI，不修改模块计算数据。Global 1024 / Local 128 / Global Top-K 512 是独立的教学场景；Global 长度是 Layer 20 写入后的快照（ratio=1 时包含当前位置），不代替审计的历史长度条件。Selected KV 仅是逻辑读取视图；共享 Latent KV 同时用于分数与汇总。分组输出形状为 `[1,64,512] → [1,8,4096] → [1,8,1024] → [1,8192] → [1,5120]`。源码依据使用审计固定的官方版本，并核对 main 中对应实现。

`verify_step4.cjs` 检查六段 × 五种窗口、Concat 行方向 / 固定特征宽度、联合归一化与 sink 数值例子、前后导航、离线运行、原 SVG 几何以及模块数据 / 审计数据 / 原始 SVG / 审计文档的 SHA-256 保持不变；结果见 `qa/step4_validation.json`。
