# Shape Trace

入口：主页面顶部 **Shape Trace**（替代旧 Data Flow）。Layer 选择器覆盖 0–39；章节按钮只滚动到同一张连续图的对应位置。点击或键盘 Enter 选择模块，高亮其祖先和后代；清除高亮恢复整图。

## 数据口径

- 维度来自构建时读取的 `references/deepseek/config.json / text_config`，由 `build.py` 内联；模型逻辑依据 `references/deepseek/inference/model.py` 的 Transformer、Block、Attention、Compressor、Indexer、Gate、Expert 和 MoE。
- 当前场景为单 Token Decode，省略 batch=1，显示 TP 切分之前的完整模型维度。`[1,5120]` 的 1 表示当前 token，跨层状态为 `[1,4,5120]`。Layer 0 展开 Embedding，Layer 39 展示 Final mHC Pre、Norm、LM Head 和 `[1,129280]` Logits。
- mHC 的注意力 Pre 使用前一层 FFN 的 pre_mix，FFN Pre 使用本层 Attention Mix 的 pre；本层 FFN pre 传给下一层。Post 同时接收 sublayer output、四路 residual、post 和 comb。系数路径以金色虚线区分。
- Layer 0/1 为 Local only。Global KV 源为 2/8/14/20，Indexer 源为 2/8/14/20/24/28/32/36。其他层读取最近源层的当前 token 对应缓存/索引，不是跨 token 重用旧 Top-K。
- Encoder 源层 ratio=2：每 Token 写入 group buffer，成组后按通道 gate pooling；组未完成时返回 None，图中 ΔS=0 表示跳过后续 latent 操作，不表示实际创建零行 Tensor。Layer 20 ratio=1，不做 gate pooling，ΔS=1。
- Index Keys 取自源层 **RoPE 前**的 compressed latent，Global KV 另行 RoPE / FP4。Reindex 使用当前层自己的 Query / head weights 和共享 Index Keys。参考实现仍计算 `[1,32,S]` scores，再应用 Candidate Mask；图中不将 score 宽度写成 16384。
- Layer 20 Candidate Mask 是与本层 Top-K 分开的辅助输出，只标注 8 positions/block、最多 2048 blocks，不展开独立计算章节。
- Decode Local ring 的物理形状为 `[128,512]`，有效位置 W≤128；Global Cache 为 `[S,512]`。实际读取位置 N≤W+K，K=min(512,S)，无效索引为 -1。KV 拼接形状 `[128+S,512]` 与实际读取的 `[N,512]` 不混用。
- Gather、Attention Scores、Softmax 和 Weighted Reduce 展示 fused `sparse_attn` 中的逻辑阶段；不声称实现物化了这些中间矩阵，也不复制 64 份 KV。sink 纳入归一化，图中权重只显示 N 个真实 KV 位置。
- SWA Bounded Replay 的依据是 `references/deepseek/README.md` Architecture 段落；本地 model.forward 没有独立 Replay 算子。因此仅在 Local KV Read 的操作标签中标注缺失缓存恢复，不编造串行层或额外输入输出。
- MoE 的 6 个 Routed Experts 使用同构分支展开“每个选中的 expert”，不是声称物化 `[6,1,5120]`。Gate / Up / Down 为 5120→2304→5120；路由 bias 只选专家，权重从 raw scores Gather 后归一化和缩放；路由权重在 Down 前应用。六路 routed output 与一个 shared output 求和。
- Engram 仅在 Layer 1/14 的 Block 前注入。Hash/Lookup 是独立 Token history 依赖，不从 Query/KV 伪造连接。DSpark 在 37/38/39 的输入处附加独立 hidden tap，不插入主输出链。

## 实现和验收

`shape_trace.js` 的图模型显式定义模块输入输出端口；每条连接构建时验证形状相等。SVG 沿可见网格避开节点路由，连接标签避开节点、章节标题和其他连接标签。多端口同源同目标连接合并标注。没有新增框架或网络依赖。

构建：`.venv\Scripts\python.exe src/deepseek/build.py`（Windows 设置 `PYTHONUTF8=1`）。

验收：`node src/deepseek/verify_shape_trace.cjs`。检查 40 层生成/切换、13 个代表层的图形边界、端口连续性、箭头避开模块、节点文字边界、高亮、1366/768/390 窄屏滚动，以及退出后原架构 DOM 保持一致。截图与报告在 `src/deepseek/qa/shape_trace/`。旧 Data Flow 和 Step 4 教学测试已从默认浏览器测试链移出，替换为 Shape Trace 验收；其他功能的测试入口保留。

Indexer 的 preservation fixture 中，架构 SVG 的旧哈希仍对应更早版本，现更新为本次任务开始时分支 HEAD（ff25777）的 SVG 哈希；本次没有更改架构 SVG 或计算数据。参数和 Indexer 测试里的旧 Data Flow 步进断言，已换成 Shape Trace 的 Layer 切换/模块定位断言。
