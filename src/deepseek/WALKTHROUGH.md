# 统一数据流演示（Step-by-step · Data Flow）

合并了原先两个重叠的教学功能：**Step-by-step · CSA2**（物理动画，23 个动作、5 章节，硬编码 Encoder Layer 2）和 **Shape Trace**（静态可点击的全层 Shape 图，可选 0–39 任意层，但不含动画）。现在只有一个入口，用同一套持久 SVG 动画引擎播放，同时在每一步旁展示结构化的 Input / Op / Weight / Output / Next 读数，数据来自同一套共享的 Shape 计算逻辑。

入口有两条等价路径，打开的是同一个面板：

1. 顶部 Tab 栏 **Step-by-step · Data Flow**（原 "Step-by-step · CSA2"）。
2. 头部工具区按钮 **Data Flow**（`#flow-toggle`，原 Shape Trace 入口）——Indexer / Attention Blocks / Layer Trace / DSpark / Parameters 等面板的"追踪→"跳转链接都走这条路径。

## 现场演示

- **8 个主阶段**（原 23 个细粒度动作合并而来），对应 Query/KV/Indexer/Attention/MoE 的完整单层流程：
  1. Input + mHC Pre　2. RMSNorm + Query　3. Local KV · SWA　4. Global KV Compression　5. Indexer · Top-K　6. Main Attention　7. Output Proj + mHC　8. MoE + Final State
  每个阶段内部再分 1–4 个 Beat，Prev / Next 按 Beat 推进；章节按钮（Stage 1–8）直接跳到某阶段的第一个 Beat。
- **Layer 切换**：顶部 4 个按钮覆盖任务要求的代表性层——`Layer 2 · Full`（ratio=2）、`Layer 3 · Reuse`、`Layer 20 · Full (ratio 1)`、`Layer 24 · Reindex`；右侧"其它 Layer"下拉可选完整的 0–39（供其它面板的跨层跳转使用，例如 Parameters 面板的 Layer 39）。切换 Layer 会重新生成该层对应的 8 阶段内容并回到第 1 步，不替换底层 SVG 节点。
- **Full / Reindex / Reuse 的差异直接体现在相关步骤的文字和配色里**，不单独开一个对比页：例如 Layer 24（Reindex）在第 4 步显示"Global KV 复用自 Layer 20"，第 5 步显示 Indexer Query 独立计算、Index Keys 复用、Top-K 重新选择；Layer 3（Reuse）在第 4/5 步直接显示"复用自 Layer 2"，不展示任何 Indexer 计算。复用的缓存用虚线边框区分。
- Play 连续播放全部阶段与 Beat；Pause 冻结动画进度；Reset 回到 Layer 2 / Stage 1。进度条可拖动到本步动画的任意百分比；速度 0.5×/1×/2×。
- "演示视图"收起阶段导航、Layer 下拉和教学说明，适合投影；再次点击退出。
- 非按钮焦点下，左右方向键推进/回退，空格播放或暂停。
- 窄屏下画布可横向滑动，矩阵保留真实字号。Canvas 高度由 JS 实时测量可用空间（`fitCanvas()`），不是写死的像素值，适配各种窗口尺寸与演示视图。

## 压缩阶段：本 Token 与多 Token 历史的区分

第 4 步（Global KV Compression）明确拆成两种叙述，不混在一起：

- **这个 Token 自己的贡献**：当前 token 的 KV/Gate 投影，ratio=2 时还需要和前一个 token 配对才合并；ratio=1 时直接产出一行。
- **Global KV Cache（多 Token 历史）**：跨步累积的缓存本身，代表许多历史 token，和"刚算完的这一行"是两个概念——缓存会在后续 Reindex/Reuse 层被复用。

## 数据口径（原 Shape Trace 文档合并于此）

- 维度来自构建时读取的 `references/deepseek/config.json / text_config`，由 `build.py` 内联；模型逻辑依据 `references/deepseek/inference/model.py` 的 Transformer、Block、Attention、Compressor、Indexer、Gate、Expert 和 MoE。
- 当前场景为单 Token Decode，省略 batch=1，显示 TP 切分之前的完整模型维度。`[1,5120]` 的 1 表示当前 token，跨层状态为 `[1,4,5120]`。
- mHC 的 Attention Pre 使用前一层 FFN 的 pre_mix，FFN Pre 使用本层 Attention Mix 的 pre；本层 FFN pre 传给下一层。Post 同时接收 sublayer output、四路 residual、post 和 comb。
- Layer 0/1 为 Local only。Global KV 源为 2/8/14/20，Indexer 源为 2/8/14/20/24/28/32/36。其它层读取最近源层的当前 token 对应缓存/索引，不是跨 token 重用旧 Top-K。
- Encoder 源层 ratio=2：每 Token 写入 group buffer，成组后按通道 gate pooling；组未完成时返回 None。Layer 20 ratio=1，不做 gate pooling，ΔS=1。
- Index Keys 取自源层 **RoPE 前**的 compressed latent，Global KV 另行 RoPE / FP4。Reindex 使用当前层自己的 Query / head weights 和共享 Index Keys。
- Layer 20 的 Candidate Mask 是与本层 Top-K 分开的辅助输出，只标注 8 positions/block、最多 2048 blocks。
- Gather、Attention Scores、Softmax 和 Weighted Reduce 展示 fused `sparse_attn` 中的逻辑阶段；不声称实现物化了这些中间矩阵，也不复制 64 份 KV。
- MoE 的 6 个 Routed Experts 使用同构分支展开"每个选中的 expert"；Gate/Up/Down 为 5120→2304→5120；路由权重在 Down 投影前应用；六路 routed output 与一个 shared output 求和。
- 位置、数值和分数（Indexer 分数卡片、Attention 权重条）用于教学示意，不是模型运行或 Profiling 结果。

## 实现和验收

`walkthrough_motion.js` 的持久 SVG 场景（节点/行/箭头）只创建一次；`buildStages(L)` 为任意 Layer 生成 8 阶段的结构化数据（复用原 Shape Trace 的维度常量与 `ratio/owns/scores/mode` 判定逻辑，不重复定义），驱动动画姿态与右侧 Input/Op/Weight/Output/Next 读数两者。`walkthrough_ui.html` 提供外观、Layer 切换与阶段导航。两者由 `build.py` 在生成架构 SVG 之后、Indexer/Attention Blocks 面板之前内联注入——顺序很重要：本面板的 `drawPanel` 包装必须先于 `indexer_ui.html` 的包装注册，才能让后者正确找到本面板注入的 `.flow-entry` 跳转按钮。

构建：

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe src/deepseek/build.py
npm run test:browser:walkthrough
npm run test:browser:layout
```

验收入口：`node src/deepseek/verify_walkthrough_motion.cjs`（推进真实 requestAnimationFrame 时钟，捕获开始/约 50%/结束的真实中间帧；检查 8 阶段结构、4 个代表层的 Beat 数量差异、Reindex/Reuse 的行内文案、Play/Pause/Reset/拖动进度/键盘导航、跨面板契约 `#flow-toggle` / `informationFlow` / `#shape-layer` 0–39 范围、原有 Parameter/Forward/Backward 计算面板、1920/1366/768/390 四种窗口无横向溢出，以及演示视图在 1366×768 下完整可见不溢出）与 `node src/deepseek/verify_walkthrough_layout.cjs`（7 种窗口尺寸 × 4 个滚动位置，检查标题/导航/内容区互不遮挡）。截图与报告在 `src/deepseek/qa/walkthrough/`。

跨面板兼容：`#flow-toggle` 按钮、`#flow-bar` 容器 id、`window.informationFlow.state`（`enabled`/`layer`）、`#shape-layer` 的完整 0–39 选项、`drawPanel` 包装注入的 `.flow-entry` 按钮（模块 key `global`/`hidden`/`multi`/`localstates`，与 `[data-node]`/`.chosen` 高亮）均按原 Shape Trace 的约定保留，供 `indexer_ui.html`、`attention_blocks_ui.html`、`layer_trace_ui.html`、`dspark_ui.html`、`audit_ui.html`、`page_template.html` 继续使用；`#csa-tab-architecture` / `#csa-tab-walkthrough` / `#csa-tab-pipeline` 三个 Tab 与 `csa-walkthrough-active` body class 按原 CSA2 walkthrough 的约定保留。`npm run test:browser:deepseek`（含 `verify_indexer.cjs`、`verify_attention_blocks.cjs`）与单独的 `verify_dspark_overview.cjs`、`verify_inference_pipeline.cjs` 均已重新跑过确认未受影响。

旧的 `verify_shape_trace.cjs`（纯 Shape Trace 验收）已随功能合并一并移除，其覆盖范围被 `verify_walkthrough_motion.cjs` 吸收。
