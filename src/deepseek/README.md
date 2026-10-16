# DeepSeek-V4.1 架构与逐步计算教学

## 当前学习入口（最终聚焦重构）

默认进入“总体架构 / 数据流 / 局部计算 / 实现与证据”四个工作区。局部计算采用 Level 1 概览、Level 2 逻辑步骤、Level 3 原审计账本；Forward A 与 Backward B 分别注明口径。旧完整图可通过 `deepseek_v4_1_interactive.html#architecture` 打开，原 DSpark 总览通过 `#dspark` 打开；在旧图点“返回学习主线”即可返回。

L20 边界显式展示四路 E → HC Pre / Attention RMSNorm → X20。Prefill bounded replay 与 DSpark 验证/调度标为论文设计，未声称固定参考 model.py 已提供完整实现。

构建仍为 `python src/deepseek/build.py`；浏览器聚合命令 `npm run test:browser:deepseek` 包含 Attention、DSpark 和新增 `verify_final_review.cjs`。详见 [最终验收与截图](../../docs/architecture/v2_final_validation.md)。


直接用浏览器打开 `../../deepseek_v4_1_interactive.html`。这是一个离线单文件，内含 SVG、样式、交互、数据和 MathML 公式，无需启动服务器。

点击模块查看参数量、前向 FLOPs 或反向梯度。切换模式保留当前模块；点击空白或按 Esc 关闭。滚轮缩放、拖动平移，按 `0` 或点击“适合窗口”复位。Tab / Enter 可操作图中模块。

## 本次教学范围

保留 Parameter / Forward / Backward。Forward默认场景A：B=1，当前1个token，缓存历史1024个token。Backward默认场景B：B=1，L=128，从序列起点重算的理论实数训练图。A的Backward仅为当前调用的局部VJP练习，梯度止于历史缓存叶子，不是Decode执行反向，也不是训练估算。

每步可展开目的、变量和实际形状、公式、标量运算拆解、步骤小计、路径累计、操作作用与梯度衔接。用Step按钮跳转，参数表与公式联动折叠在“展开参数表与当前步骤的权重联动”中。HC、MoE、Attention、Indexer、Compressor、Engram均有可推导的完整子图；整数/采样和训练策略边界明确列Unknown。

Shared KV线性层单行前向：严格5,242,368，2MKN近似5,242,880。单行反向严格dX=5,237,760、dW=2,621,440，合计7,859,200；旧10,485,760是两项都按2MKN的近似，不能称为精确。B场景独立按M=128计算，不把Decode乘3当训练成本。

逻辑参数总账保持不变。普通加/减/乘与exp、log1p、sqrt、rsqrt、除法分账，严格标量图为Derived、GEMM近似为Estimated、源码前向公式为Verified、缺失策略/内核为Unknown。旧35,584,521,265将特殊函数按1混合，并采用分块在线Attention口径，保留在历史审计中；不作为新口径精确总量。见[逐步推导文档](../../docs/architecture/forward_backward_derivations.md)。

HC Detail的小框与独立展开图保留原有交互。Encoder / Decoder Attention Detail现已重设计为等宽、对齐的三路输入和统一输出流程；底部Indexer入口继续可用。两个Attention细节区以外的SVG元素由独立基线逐元素保护。

## DSpark 主干与辅助路径

旧版 `#dspark` 入口显示独立的“主干 + DSpark”总览：40层蓝色主干、3层紫色可选草稿网络、绿色调度与验证路径。点击任何DSpark节点可查看输入、三层内部、Draft Attention、三个Head、Prefill/Decode、矩阵维度及候选接受过程；“完整架构”或URL的`#architecture`入口保留此前全部细节。右上“主干 + DSpark”返回新总览。

源文件为`dspark_diagram.py`、`dspark_ui.html`，构建沿用固定官方配置。`verify_dspark.cjs`已纳入浏览器回归。核对出处、七项陌生读者审查及参考代码未实现的验证循环边界见[DSpark验收记录](../../docs/architecture/dspark_validation.md)。

## Attention 对照入口

右上“Attention 对照”打开可按层切换的计算图；聚焦菜单的“C · Encoder ↔ Decoder 对照”同时放大主图中的两个模块。主图用橙色的“2 tokens → 1 KV / 1 token → 1 KV”突出全局分支差异，粉色E标签交代Encoder四路输出经L20 HC Pre / RMSNorm后进入Decoder。Q、Local KV、Attention、Inverse RoPE、分组输出投影保持同位展示。

`attention_diagram.py`绘制主图，`attention_ui.html`提供逐层细节，维度与层配置来自固定官方inference/config.json。`verify_attention.cjs`验收40层模式、输入来源、文字边界与现有交互，已纳入`npm run test:browser:deepseek`。`attention_preservation.py`锁定Attention区域以外的全部SVG。源码核对、十项陌生读者审查与截图位置见[验收记录](../../docs/architecture/attention_redesign_validation.md)。

## HC 架构图

HC小框已修正为h⁽¹⁾、h⁽²⁾、h⁽³⁾、h⁽⁴⁾，每路[1,5120]。点击底部“展开HC Pre / Post / Comb”，或右上聚焦菜单的“F · HC Detail”，查看完整图。

上方路径从同一R生成24个动态系数：Flatten→W_HC[20480,24]→投影结果乘r→Scale/Base与非线性；Pre_next交给下一子模块，Post=4与Comb=16用于本次更新。主路径使用前阶段的Pre_in把四路合成一路，经RMSNorm与Attention/FFN得到U；Post分配U，Comb按C[i,j]（旧路i→新路j）混合旧R，两者相加得到[1,4,5120]。

点击图中的Pre、Post、Comb、W_HC等节点，直接定位现有HC计算账本；图本身没有另一套FLOPs公式。点击空白关闭面板，“返回架构”恢复视角。node src/deepseek/verify_hc.cjs检查四路标签、动态系数、步骤定位、文字边界、缩放拖动和与Indexer/信息流的切换。

before_hc.svg保留修改前的架构；hc_preservation.py锁定其原始哈希，并逐元素验证HC区域之外的SVG完全一致。原SVG回归检查只对授权修改的HC区域放行，其他模块仍受原有保护。

## 可编辑源文件

- `teaching_data.py`：当前使用的 63 个教学条目，含三种模式与 Compressor 示例。
- `module_data.json`：自动导出的可读数据，公式保留 LaTeX。
- `derivation_ledger.py`：形状驱动的统一计数、HC/MoE等计算图、A/B场景和反向推导；不维护第二套手填数字。
- `derivation_data.json`：自动导出的规范账本，各场景引用共享步骤，并保留旧行结果的比较记录。
- `derivation_ui.html`：可展开步骤、场景选择、严格/近似/特殊操作分账，构建时编译并内嵌离线HTML。
- `build_diagram.py`：SVG 几何与标签。
- `hc_diagram.py`、`hc_ui.html`：HC小框、展开架构与现有计算步骤的链接。
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

工具栏Assumptions提供配置、场景、计数规则与Unknown边界。各步骤直接展开形状、公式、运算拆解和累计；Parameter保留原有权重追踪。主损失路径和给定额外梯度的条件VJP分列，不把整数Top-K当连续可微操作。

`docs/architecture/forward_backward_derivations.md`是当前数值与推导文档。原`calculation_audit.md` / `assumptions.md`保留固定源码审计与历史计数规则，文首指向新账本。`verify_audit.py`继续核验原源码/参数/历史kernel口径，并检查新账本生成后的数据与网页一致；不以旧结果证明新推导正确。

运行 `python src/deepseek/verify_derivations.py` 检查完整HC、MoE、Compressor、Engram和条件Indexer VJP的微型数值差分，独立检查单行外积及所有步骤累计。`verify_calculations.py`另检Query/RMSNorm、含sink的共享KV Attention与截断SwiGLU。`node src/deepseek/verify_derivations.cjs`检查63模块×2场景×2计算模式=252组、步骤展开、参数联动、响应式和离线运行。结果位于`qa/derivation_validation.json`及`qa/derivation_browser_validation.json`。这些数学验证不等于验证官方训练GPU kernel。

复跑：`python src/deepseek/audit_model.py`、`python src/deepseek/build.py`、`python src/deepseek/verify_audit.py`、`python src/deepseek/verify_single_token.py`。梯度检查需 NumPy，请使用根目录 requirements.txt 安装的 NumPy；浏览器检查运行 `node src/deepseek/verify_browser.cjs`、`node src/deepseek/verify_parameters.cjs` 和 `node src/deepseek/verify_flow.cjs`。运行 HTML 本身不需要这些构建与验证工具。

## 信息流演示

### Indexer 全局位置选择

点击 Encoder / Decoder Attention Detail 底部橙色的 `Indexer → Top-512 → Global KV ↗`，展开流程并打开现有 Indexer 计算面板。也可以从右上角聚焦菜单选择 `E · Indexer Detail`。点击空白关闭计算面板，保留流程；“返回架构”恢复进入前的视角。

Presentation 建议先选 Full（Layer 20），沿 Query / Key → Scoring + ReLU → Head Aggregation → Top-K → Global KV Gather → Main Attention 讲解，再切换 Reindex（Layer 24）与 Reuse（Layer 21）。淡化的节点表示该层跳过的计算；层选择器可检查 2–39 层的真实配置。0–1 层只有局部 SWA，不运行 Indexer。所有层仍执行自己的主要 Attention。

Query 来自当前层归一化后的 1280 维低秩表示，投影为 32 × 128；Index Key 来自 Compressor 的 512 维 RoPE 前 latent，仅在源层完成新条目时投影、追加缓存。Head Weights 来自当前 X 的可学习投影。Top-K 输出整数位置，Gather 读取已有 Global KV；与最多 128 个 Local KV 位置一起供 Main Attention 使用，Main Query 与 Index Query 相互独立。

KV source layers 为 `[2,8,14,20]`，Index source layers 为 `[2,8,14,20,24,28,32,36]`。Reindex 复用 Global KV 和 Index Keys，重新评分；Reuse 连同已有位置索引一起复用。依据 `references/deepseek/inference/model.py` 的 `Indexer.forward`、`Attention.forward` 与 `SharedAttentionRuntime`。

“评分范围、缓存与计算口径”说明C与Top-K的区别。架构按钮解释Full/Reindex/Reuse，面板中的A/B场景独立选择，数值主例明确为Full源层。Backward主路径在整数选择处标Unknown；给定额外dScore的评分子图VJP单列，不猜测真实训练目标。相关链接指向Compressor、Global KV和Main Attention。

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
