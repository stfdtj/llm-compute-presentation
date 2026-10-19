# CSA2 Step-by-step

打开仓库根目录的 `deepseek_v4_1_interactive.html`，点击顶部 **Step-by-step · CSA2**。页面完全离线运行，使用原生 HTML/CSS/JS 与可缩放 SVG，不增加运行时依赖。

## 现场演示

- Prev / Next 逐步推进，也可以直接点击六个步骤。
- Play 每 3 秒推进一步，在最后一步停止；Pause、手动导航、离开 tab 或切换到后台都会停止自动播放。
- Reset 返回第一步和 Full 模式。
- “演示视图”收起辅助信息，适合笔记本和投影；再次点击退出。
- 非按钮焦点下可用左右方向键推进，空格播放或暂停。顶部 tab 支持方向键、Home 和 End。
- 小屏保留图中文字尺寸，计算图可以横向滑动。

## 六个步骤

1. 当前 Token 与最近最多 128 个位置的 Local KV（包含当前位置）。
2. Encoder Layer 2 的 compression ratio=2：两个历史 Token 形成一个 Global Slot。
3. 独立的 Indexer Query / Index Keys 评分。
4. Top-512 返回位置索引；仅画出四个入选样例。
5. 本层 Main Query 读取 Local KV 与选中的 Global KV，共同 Softmax 后汇总。
6. Full / Reindex / Reuse 模式切换，缓存与跳过的计算淡化显示。

附加的 Matrix Multiplication 支持八步手动演示：高亮行、列、逐项乘法、求和与写入输出矩阵。

## 技术口径

- 参数读取现有构建配置：Local Window 128、Top-K 512；ratio=2 的压缩示例对应 Encoder Layer 2。
- Reindex 使用 Decoder Layer 24：复用 Layer 20 的 KV / Index Keys，重新计算索引。Reuse 使用 Decoder Layer 21：连同位置索引一起复用。两者本层 ratio=1，不把 Encoder 的压缩率套到 Decoder。
- Main Query 与 Indexer Query 分开显示。Index Keys 从压缩表示分支产生；不是从存储后的 Global KV 重新投影。
- 示例历史长度 4096，压缩为 2048 个 Global Slots；图中只抽样八个。分数、选中位置和 token 方块都是教学示意，并非真实运行输出或 Profiling 数据。
- 矩阵例子 A[2,3] × B[3,2] 的结果为 [[22,28],[49,64]]；精确标量计数为 12 次乘法 + 8 次加法 = 20 FLOPs，2MKN 近似为 24 FLOPs。
- 原有架构 SVG 和模块计算数据保持不变。

## 生成和检查

在仓库根目录运行：

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe src/deepseek/build.py
npm run test:browser:walkthrough
```

源文件为 `walkthrough_ui.html`，由 `build.py` 内联到最终 HTML。浏览器检查使用项目已有的 Playwright 与本机 Chrome。

已实际运行并通过：六步导航、三种模式、播放暂停与计时器清理、Reset、键盘、演示视图、八步矩阵示例、离线加载、原有数据和图形保留。检查 1920 / 1366 / 768 / 390 像素窗口，并查看实际截图，修正了评分区箭头及笔记本整图布局。原有 `verify_flow.cjs` 与 `verify_indexer.cjs` 回归检查通过。

结果：`qa/walkthrough/validation.json`。截图：`qa/walkthrough/step-*.png`、`mode-*.png`、`matrix.png`、`responsive-*.png` 与 `presentation-1366.png`。
