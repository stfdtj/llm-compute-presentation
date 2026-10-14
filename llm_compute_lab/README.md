# LLM Compute Lab — Attention, Roofline & MFU

直接双击 **index.html** 打开，支持 `file://`，不需要服务器、安装依赖或联网。默认从 S=8192 开始连续生成 1024 个 Token；可调整生成数或切回 Single Token。新增 FLOPs Speedup、累计曲线、首步 / 末步 / 平均值、完整交互式 HBM Roofline，以及可调的 50% MFU 教学示例。按“概念 → 示例 → 公式 → 解读”组织内容，保留六种架构的图、步骤、Context / 容量曲线和训练数据导入。

右上角的 **演示模式** 默认关闭。启用后，三个 Tab 优先展示图形、关键结果与动态结论；原输入、图表和监听器直接复用，切换不重置状态。完整内容与表格仍可展开。Roofline 的演示按钮位于图上方，教学点与 Attention 上界分开；MFU 可一键切到“速度减半”或“计算减半、速度翻倍”。本轮验收见 `final_validation.md`。

此项目独立新建，未改动现有 DeepSeek-V4.1 网页或代码。参考现有项目的单 token 场景、2MKN 与严格乘加的区分、浅色科研教学风格；没有复用旧网页复杂布局或把旧模型配置套给新架构。交付为本地文件，未注册或部署到托管服务。

## 文件清单

| 文件 | 用途 |
|---|---|
| `index.html` | 最终独立离线网页，已内嵌所有 CSS / JS |
| `model.js` | 纯计算函数、校验、导出和后续接口；Browser / Node 共用 |
| `ui.js` | 参数联动、SVG 图表 / 架构图、步骤、Tab 与导入 |
| `extension.js` | 增量的连续 Decode、Speedup、累计曲线、硬件模板与 Roofline UI |
| `learning.js` | 初学者数字代入、图形解读、硬件变化对照与 MFU 教学交互 |
| `presentation.js` | 复用原 DOM 的紧凑演示布局，不维护第二套计算状态 |
| `final_tests.js` | 最后一轮场景边界、教学点、实测单位和数值范围测试 |
| `final_browser_checks.mjs` | 由浏览器测试调用的演示模式、状态与交互验收 |
| `final_validation.md` | 本轮问题修复、教学场景、演示操作与验证记录 |
| `teaching_tests.js` | 官方硬件模板、默认 Roofline 手算与 MFU 假设示例测试 |
| `teaching_validation.md` | 硬件核验来源、默认数字与本轮教学验收 |
| `style.css` | 桌面 / 窄屏自适应样式 |
| `page.html` | 构建模板 |
| `build.py` | 标准库打包；生成独立 `index.html` |
| `tests.js` | 算术、退化、代数和接口测试 |
| `extension_tests.js` | 累计求和、吸收矩阵、Speedup、HBM 估算与 Roofline 测试 |
| `browser_test.mjs` | Chrome / Edge 的本地 CDP 交互与截图检查，无 npm 依赖 |
| `validation.md` | 定义、公式、手算与验证记录 |
| `extension_validation.md` | 本轮累计求和、MLA 吸收、HBM / Roofline 依据与验证结果 |
| `training-flops-template.json` | 后续 MFU 的独立输入模板 |
| `qa/browser-results.json` | 浏览器检查结果 |
| `qa/*.png` | 桌面、完整页面、窄屏、GDA 核心与 Roofline 截图 |
| `.gitignore` | 排除 QA 浏览器临时 profile |

## 修改与复现

修改 `model.js` / `ui.js` / `extension.js` / `learning.js` / `presentation.js` / `style.css` / `page.html` 后运行：

```powershell
node llm_compute_lab/tests.js
node llm_compute_lab/extension_tests.js
node llm_compute_lab/teaching_tests.js
node llm_compute_lab/final_tests.js
python llm_compute_lab/build.py
node llm_compute_lab/browser_test.mjs
```

算术测试只需 Node。构建只需 Python。浏览器测试默认使用 Windows 已安装的 Chrome / Edge；其他位置可设置 `COMPUTE_LAB_CHROME`。测试启动自有无界面浏览器，结束后关闭，只写入 `qa/`。部分受限沙箱需要允许浏览器进程间通信。

不要直接编辑生成的 `index.html`；从源文件构建可以保留计算层和 UI 层的分离。

## 六种定义与边界

| 架构 | 本页计算定义 |
|---|---|
| MHA | H 个独立 K/V 头；全历史 + 当前 token 的 Softmax Decode |
| GQA | H/Hkv 个 Query Head 共享一个 KV Head；核心仍按 H 个 Query 计算 |
| MLA | DeepSeek-V2 低秩 KV 与解耦位置分支；latent-space Decode；可选保留因子、吸收 Query、吸收 Query + O 权重 |
| SWA | 使用 GQA 投影的因果局部窗口；读取并保留最近 min(S+1,W) 个位置 |
| MSA | MiniMax 论文 Index Q（每 group 一头）+ 共享 Index K；完整扫描、block max pooling、exp-free Top-K、选定 block 上的主 GQA；k 包含强制 local block |
| GDA | Gated DeltaNet Eq.10 的逐 token 递归核心；已知 Q/K/V/O、α/β、输出门投影；完整模块 Extra / Total 未核验 |

Total 是声明范围内的小计：统一排除 RoPE 旋转、全局 Norm、Bias、残差、FFN、量化 / 反量化及通信。MLA 的 dk 是内容维度，另有 dr 位置维度，不天然等价于 MHA；GDA 的递归语义也不同于 Softmax。

GDA 的 ShortConv、SiLU、Q/K L2 Norm、门参数化、输出 Norm / 门乘法和卷积缓冲尚未核验。总量与未知额外项使用 `null`，仅展示已知小计与递归核心曲线，不把缺失项当作零。

缓存容量是逻辑存储量，显示全部生成后的整批单层容量；不是 HBM 实际流量。原有单 Token 接口中的 traffic 字段仍为 `null`，新 `analyzeRoofline` 接口独立输出声明假设下的 HBM 估算或用户总量，不把它们标为测量值。FP8 容量不计 scales / 对齐，不表示真实内核性能。FLOPs Ratio 不是实测加速。

## 连续 Decode 与 Speedup

主表和 Speedup 区域按**单层、每序列、完整生成区间**对比；矩阵展开保留第一步示例。Ftotal=Σᵢ₌₀ᴳ⁻¹ Fdecode(S+i)，不是首步 FLOPs 乘以 G。Context 曲线仍显示单 Token 随历史长度的变化；累计曲线显示从初始 S 开始生成 g 个 Token 的总和，两者共享曲线显隐。

`aggregate(name,parameters,scenario)` 使用解析前缀和，并与显式逐步循环交叉验证。`scenario` 为 `{mode:'autoregressive',generated:1024}` 或 `{mode:'single',generated:1}`；G=1 与旧公式一致。`Prefill` 仍禁用并在计算 API 拒绝，不输出冒充 Prefill 的 Decode 结果。

Speedup=MHA/当前；Reduction=(1−当前/MHA)×100%。小于 1× 时显示计算量增加；未知完整模块显示 N/A。MLA 原有分解没有重复展开历史 K/V；新增离线吸收路径会替换相应投影，融合权重可能更大，不能保证运算量更少。离线融合和真实舍入 / 内核差异不计入每 Token FLOPs。

## Roofline 已实现

默认自动使用 NVIDIA H100 SXM 80GB、BF16 Dense 989 TFLOP/s、HBM 3.35 TB/s、1 张设备，与 Attention 默认配置同步，连续生成 1024 tokens。打开 Roofline 页即可看到计算密度、注释图、公式数值代入、性能上界、瓶颈与时间下界。带宽 / 算力 ×2 按钮把修改后的硬件标成教学假设，并给出前后数值；也可恢复官方规格或改为自定义。复杂模型设置可展开查看。

带宽与容量来自 [NVIDIA 产品规格](https://www.nvidia.com/en-us/data-center/h100/)；Dense 峰值来自 [NVIDIA 官方 recsys-examples 基准说明 §2](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/benchmark/E2E_BENCHMARK.md)。产品页 BFLOAT16 1,979 TFLOP/s 有 With sparsity 注脚，不作为 Dense 峰值。未添加没有同时核验精度、Dense 算力与 HBM 带宽的 Ascend 型号模板。详见教学验证记录。

HBM 模型默认每步读取权重（每设备一份）、组内共享历史 KV、当前 KV 片上使用并写入 HBM、中间 score / probability 片上融合。可改变权重 / 历史 KV 命中率、头之间复用与中间回写，或输入六种架构当前整批生成区间的总 MiB。100% 权重命中只表示预先驻留的理想假设，未包含冷启动加载。

MHA/GQA/SWA 按各自历史位置与 KV 头估计；MLA 按压缩 latent 的两次内容读取与一次位置读取；MSA 按已选主 KV 加完整历史索引扫描；GDA 核心按状态完整读写。权重来自选定执行路径的实际矩阵尺寸。完整 GDA FLOPs / 字节未知时不绘制 Module 点、不排名，即使用户给出字节也不能补全未知 FLOPs。

AI=F/B；Pupper=min(Ppeak,AI×BW)；Tlower=max(F/Ppeak,B/BW)。这些是声明假设的理想模型上界 / 下界；区间合并下界可比逐步下界之和更宽松。单峰值近似混合矩阵 / 标量操作，不代表每种指令都可达到 Tensor Core 峰值。无通信、启动延迟、量化 scales 或内核实测。多设备只支持 Batch 中独立序列均分，复制权重、理想聚合峰值和带宽；不模拟张量并行。

用户实测 Runtime 与预测分别展示。改变实验参数 / Token 模式 / FLOPs 范围会清空旧自定义字节与 Runtime；只改变 Peak / BW 保留同一工作量的字节并清空 Runtime。改变设备数也清空字节。自定义输入的来源与准确性需提供者核验。

图例：屋顶线是模型上界；实心圆是 Attention 在当前 FLOPs / 估计 Bytes 下的理想上界；菱形是 AI=100 与 500 的独立教学假设；空心圆仅在用户提供相同范围的正数 Runtime 时出现。空心点纵坐标为 FLOPs ÷ (Runtime_ms / 1000)，横坐标仍依赖估计 Bytes。测量超过模型上界时不截断，提示核对范围、精度与访存假设。只有同时提供 MHA 与当前架构的 Runtime 时才显示 Measured Speedup，并注明运行条件需要一致。

演示按钮先验证完整候选场景，含 `S+G−1≤1048576`、累计计数、单位换算和有限数值；失败时保持配置与图形不变。手动输入无效时，Attention 明示保留上次有效值，Roofline 清空当前分析，避免跨 Tab 使用旧结果。Roofline 独立选中的架构随参数更新保留；只有 Attention 的架构选择改变时才跟随选择。

## 数据接口 / MFU 教学

- **Roofline Tab** 在 `page.html`，新增渲染在 `extension.js`；`model.js: analyzeRoofline(parameters,scenario,hardware,traffic,scope)` 是本轮分析入口。`estimateTraffic` 输出 HBM 分项、公式和复用假设。旧 `rooflineInput(parameters)` 空接口继续保留。
- **MFU Tab** 使用明确标为假设的可调输入：Training FLOPs/token=10¹²、全设备 Tokens/s=2000、设备数=4、每设备峰值=1000 TFLOP/s，得到 50%。支持 Tokens/Step ÷ Step Time 的换算；没有接入真实训练测量。完整模型 forward / backward 数据可独立导入，保持与教学示例分离。`validateTraining(data)` 校验格式；`verified` 只是提供者声明，不能自动证明来源正确。
- **浏览器公开接口**：`window.LLMComputeLab.getExperiment()`、`getRooflineInput()`、`getRooflineAnalysis()`、`getParameters()`、`getScenario()`、`getTrainingInput()`、`importTraining(data)`、`getMfuTeachingExample()`、`applyHardwareTemplate(id)`。
- 新实验 / Roofline 导出为 `schemaVersion: 2`。累计 Attention 为 `FLOPs/generated-interval/layer/sequence`，`tokenResults` 单列旧单 Token 结果，Roofline 的 FLOPs / Bytes 为整批单层生成区间。缓存为 `bytes/batch/layer`。旧 `exportAttention` 和训练模板仍为版本 1。
- MFU 教学计算 `calculateMfu(input)` 输出 `dataKind: hypothetical`，不是实测训练性能；导入数据不会覆盖假设示例。当前单层 Decode 与全模型训练的工作量、范围不同，不能混用。不能使用推理 Roofline 吞吐量当作实测训练吞吐量。
- MFU 情景按钮均从默认假设加载整组输入：A 为 1000 Tokens/s，得到 25%；B 为 0.5×10¹² FLOPs/token 与 4000 Tokens/s，保持 50%。设备数均为 4、每设备峰值均为 1000 TFLOP/s。修改前后公式并列展示，手动编辑会清除旧情景结论。

来源与单 Token 公式见页面 Methodology 和 `validation.md`；本轮累计求和、MLA 吸收、HBM 流量与 Roofline 验证见 `extension_validation.md`。
