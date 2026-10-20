# CSA2 矩阵教学动画

打开仓库根目录的 `deepseek_v4_1_interactive.html`，点击顶部 **Step-by-step · CSA2**。页面完全离线运行，使用原生 HTML/CSS/JS 与可缩放 SVG，不增加运行时依赖。

## 现场演示

- Prev / Next 推进或回退一个明确的计算动作，共 23 个动作，分为五章。顶部章节按钮用于跳转。
- Play 连续播放；每个运动默认 1.8 秒，动作之间停留 1 秒。最后一步自动停止。
- Pause 在当前动画进度冻结矩阵位置；再次 Play 从中途继续。手动导航会停止连续播放。
- “重播本步”从该动作的开始姿态重播；进度条可拖到任意中间帧。支持 0.5× / 1× / 2× 速度。
- Reset 返回第一步和 Full 模式；离开 tab 或切换到后台时停止动画。
- “演示视图”收起辅助信息，适合笔记本和投影；再次点击退出。
- 非按钮焦点下可用左右方向键推进，空格播放或暂停。顶部 tab 支持方向键、Home 和 End。
- 小屏保留图中文字尺寸，计算图可以横向滑动。
- 标题、播放控制和章节导航占用独立的布局空间，下方内容单独滚动；不使用会覆盖说明文字的 sticky 标题。Reset 与演示视图会恢复内容区滚动位置。

## 动作与矩阵维度

| 章节 | Action | 可见的矩阵变化 | 真实维度 |
|---|---|---|---|
| SWA | 1–4 | 取出 Query 行、窗口滑到最近 128 个位置、复制选中行 | Q [4097,64,512] 为全序列形状示意；当前 q [1,64,512]；Local KV [128,512] |
| KV Compression | 5–10 | 一对行进入 Compressor，随后每次 2→1，八行变四行 | X [8,5120] → Global KV [4,512]；真实历史 4096→2048；ratio=2 |
| Indexer | 11–17 | 逐个生成 Score Vector、分数卡片移动排序、提取位置索引 | Query [32,128]；Index Keys [2048,128]；Score Vector [2048]；实际 Top-K≤512 |
| Main Attention | 18–22 | Local / selected Global 行复制到读取区、评分、共同归一化、加权汇总 | KV_read 最多 [640,512]；每 Head scores [1,640]；O_head [1,64,512] |
| Modes | 23 | 切换 Full / Reindex / Reuse，比较缓存与选址 | Full L2 ratio=2；Reindex L24 / Reuse L21 ratio=1 |

附加的 Matrix Multiplication 支持八步手动演示：高亮行、列、逐项乘法、求和与写入输出矩阵。

## 技术口径

- 参数读取现有构建配置：Local Window 128、Top-K 512；ratio=2 的压缩示例对应 Encoder Layer 2。
- Reindex 使用 Decoder Layer 24：复用 Layer 20 的 KV / Index Keys，重新计算索引。Reuse 使用 Decoder Layer 21：连同位置索引一起复用。两者本层 ratio=1，不把 Encoder 的压缩率套到 Decoder。
- Main Query 与 Indexer Query 分开显示。Index Keys 从压缩表示分支产生；不是从存储后的 Global KV 重新投影。
- 当前位置 t4096（从 0 开始），此前 4096 个 Token 压缩为 2048 个 Global Slots；窗口 t3969…t4096 包含当前 Token。一次增量推理只计算当前 Query，全序列 Q 矩阵用于说明行与 Token 的关系。
- Compression 图示 8→4。压缩为可学习投影、按通道 Softmax 门控和加权汇总，不表示对原始 Token 简单求平均。未满一组的尾部不输出新 Slot。
- Indexer 只画四个候选、示意选两个；实际候选域为 C=2048，Top-K 上限 512。分数排序是教学展示，不要求实现中完整排序。Index Keys 是 128 维，Global KV 是 512 维，两者不能混用。
- Main Attention 图示 6 行 Local + 2 行 Global；真实最多 128+512 行。行复制表示逻辑 Gather，不要求额外分配完整拼接矩阵。共享 Latent KV 同时用作 Key 和 Value。
- 分数、位置与单元格数值都是教学示意，不是模型运行或 Profiling 结果。Softmax 数字保持自洽：示例 Sink logit=0，八个 logits 为 ln(weight/0.10)，Value 权重和为 0.90，Sink 贡献分母中剩余 0.10 而不加 Value。
- 矩阵例子 A[2,3] × B[3,2] 的结果为 [[22,28],[49,64]]；精确标量计数为 12 次乘法 + 8 次加法 = 20 FLOPs，2MKN 近似为 24 FLOPs。
- 原有架构 SVG 和模块计算数据保持不变。

## 生成和检查

在仓库根目录运行：

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe src/deepseek/build.py
npm run test:browser:walkthrough
npm run test:browser:layout
```

`walkthrough_ui.html` 提供外观和控制区，`walkthrough_motion.js` 创建持久 SVG 元素，由 `build.py` 内联到最终 HTML。计算画布只创建一次 SVG；后续插值改变矩阵行的坐标、尺寸、透明度和颜色，不重新渲染整张 SVG。附加矩阵乘法有独立的简单演示区。

`verify_walkthrough.cjs` 调用 `verify_walkthrough_motion.cjs`。浏览器测试推进真实 requestAnimationFrame 时钟，捕获开始、约 50% 和结束的不同几何姿态，同时检查 DOM 身份保留、暂停时坐标冻结、反向移动、重播、进度拖动、全部 23 步播放、三种模式、原有三种计算面板以及四种窗口尺寸。中间帧不通过测试专用接口伪造。

本轮实际执行结果为 `passed: true`，浏览器错误为 0。已查看关键场景的真实截图，修正运动中卡片碰撞、压缩行遮挡公式、数值条压缩文字和输出维度重叠问题。保留 8 组开始／中间／结束截图。原计算面板另用真实时间的浏览器页面验证；原 Architecture / 计算数据 / 其他 UI 的 HTML 前缀与修改前保持一致（仅规范化换行比较）。

验收入口为 [关键帧对照页](qa/walkthrough/review.html)，结果为 `qa/walkthrough/validation.json`（含各场景起止坐标、中间帧进度和矩阵尺寸）。本轮截图采用 `swa-*-start/middle/end.png`、`compression-*.png`、`indexer-*.png`、`attention-*.png`。目录内原有 `step-1` 至 `step-6` 是上一版静态流程图的历史截图，不作为新版验收依据。

数据依据：`references/deepseek/config.json`、`inference/model.py` 的 `Compressor.forward` / `Indexer.forward` / `Attention.forward`，以及 `inference/kernel.py` 的 Attention Sink 归一化。视觉参考为用户提供的 Downloads/imgs 中四张截图；没有复制截图素材到演示页面。
