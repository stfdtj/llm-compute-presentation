# 最终聚焦重构验收（2026-10-16）

基线 HEAD：`deff9fd9dadcb9f3b8efd93d536ca0d0cee40480`。本次基于现有页面修改，未创建第二套计算引擎、未提交或推送远端。初始跟踪文件无修改；三个审查/规格/讲稿文件为用户未跟踪文件，保持原样。未发现现存 V2 实现。

## 审查逐项结论

| 项目 | 核验与处理 |
| --- | --- |
| P0 测试聚合遗漏 Attention / DSpark | **核验后不成立**。基线 package.json 已有两者，README 与实际一致。本轮追加 verify_final_review.cjs；两套原测试保留并单独执行。 |
| P0 E → Global KV 语义 | Attention 对照本身已正确，但历史总览仍有 E 单行与串行 Global→Decoder 歧义。新首页显式展示 E[1,4,5120] → L20 HC Pre / Attention RMSNorm → X20[1,5120]。boundary_ui 对保留的旧 SVG 做展示层修正：标明边界、移除 KV 替代残差的串行箭头，补四路残差主干。原始 SVG 和哈希基线不改；实际浏览器渲染另验收。 |
| P0 场景与反向 | 默认前向 A（1 当前、1024 历史），反向 B（128 位置理论训练图）；参数与 token 数独立。局部 VJP 与训练图明确区分，历史缓存梯度到叶子停止；既有 Unknown 原样保留。 |
| P1 学习主线 | 统一为总体架构 / 数据流 / 局部计算 / 实现与证据。完整图、HC、Indexer、Attention、DSpark 与原账本保留入口；原图可通过 #architecture 打开，旧 DSpark 总览通过 #dspark 打开。 |
| P1 计算降噪 | 默认 Level 1：目的、形状、公式、当前步骤权重和数量级。Level 2：源账本逻辑分组及梯度形状说明。Level 3：原始计数、条件 VJP、Unknown、参数联动均保留；细账默认折叠，可从当前步骤直接跳转。所有成本读取同一 ledger，未修改 derivation_ledger.py。 |
| P1 小屏与可用性 | 四种宽度；DSpark 七页的键盘、滚动、空白与 Esc 关闭；计算三级返回与场景切换；离线及 JS 错误检查。平板与手机采用纵向主线，避免换行断箭头。 |
| Prefill bounded replay | **不能认定为参考源码已实现**。固定 model.py:1264–1274 对所有输入位置遍历主干，无 bounded replay 分支。页面把源码执行和技术报告优化设计分开，不捏造末尾重放调度。 |
| DSpark 论文边界 | 40 主干 + 可选 3 Draft Blocks；5 草稿位置；L37/38/39 Block 输入 mean → Concat/Projection/Norm。Confidence 是 raw scores，不能直接接受候选。绿色虚线标论文验证/调度，forward_spec 未实现完整循环。 |

## 文件与数据复用

- `learning_ui.html`：四个互通工作区、场景数据流、模块搜索、Evidence。
- `progressive_ui.html`：读取 module_data / derivation ledger 的三级呈现；原始审计 DOM 保留，切换层级不修改数据。
- `boundary_ui.html`：历史图 L20 边界的展示修正。旧图的保存基线不改，修正应用于离线 HTML 的实际渲染。
- `flow_ui.html`：旧逐步信息流改用 E 与 X20，显式 HC Pre / Norm。
- `dspark_diagram.py`：论文验证连接统一绿色虚线。
- `build.py`：内联新呈现层，仍交付同一个离线单文件。
- `verify_final_review.cjs`：新增跨工作区、数值金样例、响应式、键盘与箭头位置验收。

配置、模型参考快照、计算引擎、module_data.json、audit_data.json、derivation_data.json、原图 geometry/hash 基线不变。没有更改公式或数值；只是展示其现有来源。Global 投影权重 2,621,440 与包括 Norm 的模块权重 2,621,952 分开标明；单步前向 2MKN=5,242,880 与严格 5,242,368 保留。

## 自动可验证记录

命令均使用本项目 `.venv/Scripts/python.exe -X utf8`（系统 Python 未作为依赖环境）。完整 stdout 在 `src/deepseek/qa/final_review/`（.log 按仓库规则仅保留本地）；可随变更保存的命令/退出码汇总为 execution_results.json。

| 命令 | 结果 / 记录 |
| --- | --- |
| python src/compute_lab/build.py | PASS；index.html 重建一致 |
| python src/deepseek/build.py | PASS；离线产物重建一致 |
| npm run test:lab | PASS；lab.log |
| npm run test:browser:lab | PASS；98 checks，lab_browser.log |
| python scripts/FLOPs_calculation_verify.py | PASS；同名 log |
| python src/deepseek/verify_audit.py | PASS；同名 log |
| python src/deepseek/verify_single_token.py | PASS；同名 log |
| python src/deepseek/verify_calculations.py | PASS；同名 log |
| python src/deepseek/verify_derivations.py | PASS；同名 log |
| npm run test:browser:deepseek | PASS；final_browser.log 与 validation.json，包含所有原测试和新增验收 |
| node src/deepseek/verify_attention.cjs | PASS；attention_standalone.log |
| node src/deepseek/verify_dspark.cjs | PASS；dspark_standalone.log |
| node src/deepseek/verify_final_review.cjs | PASS；最终版本在聚合命令末尾执行，详见 final_browser.log / validation.json；63 模块 × 3 模式 = 189 概览检查 |

修改前完整浏览器回归通过，保存在 baseline_browser.log。第一次修改后聚合运行在 Attention 截图写入时遇到 Windows 文件打开 UNKNOWN 错误，保留在 first_browser_attempt.log；随后完整重跑，不把失败记录当作通过。最终标签补全又触发 Indexer 的 original diagram unchanged 断言：新文字继承旧 textLength / lengthAdjust，重新 fitLabels 后 DOM 不一致。修正标签初始化并清除旧压缩属性，原断言保留；失败记录在 label_fit_regression.log，独立复验在 indexer_standalone.log。一度因自动审批服务额度错误无法启动复验，用户要求继续后恢复。

产物重复构建的 SHA256 记录在 build_reproducibility.json。新测试校验页面内嵌 canonical ledger 未被 UI 改写，金样例数值不由另一份手抄常量生成。旧细图测试仍针对其 #architecture / #dspark 兼容入口；新测试针对默认首页。

## 人工视觉审查（真实 Chrome 截图）

已实际打开截图审查，未用文字边界检测代替视觉审查。以下路径相对仓库根：

- 修改前：`src/deepseek/qa/final_review/before_1366.png`、`before_390.png`、`before_compute.png`。
- 修改后首页：`after_1920.png`、`after_1366.png`、`after_768.png`、`after_390.png`。
- 草稿旁支：`branch_1920.png`、`branch_1366.png`、`branch_768.png`、`branch_390.png`。
- 计算：`compute_1366.png`、`compute_390.png`。
- 场景：`flow_prefill_1366.png`、`flow_decode_1366.png`、`flow_draft_390.png` 等四尺寸组。
- DSpark 小屏细节：`dspark_dimensions_390.png`、`dspark_verification_390.png` 等。

第一轮人工审查发现 768px 横向流程换行产生悬空箭头，已改为纵向并增加几何断言。计算卡场景长说明挤压结果，已折叠。一次脚本替换导致场景摘要标签乱码，也已从实际截图发现、修正并增加检测。新增旧图边界断言还发现 SVG 多行 tspan 更新不完整，已改为逐行更新，重新截图确认 E / HC Pre / RMSNorm / X20 全部显示。最终桌面主干与草稿在同屏内区分；小屏用正常纵向滚动阅读，无需横向拖动才能知道主线。移动计算面板的细节可滚动，关闭与模式按钮可操作。

### 陌生读者信息追踪

首页蓝色 L0–19 / L20–39 共 40 层，LM Head 明示最终输出。黄色边界显示 E→HC Pre/Norm→X20；L20 Full、四个 Reindex 与其余 Reuse 同屏标记。两侧共同 Q/Local/RoPE/Attention/Inverse RoPE/Output 可进入同位对照图。数据流区分别说明位置数、512 维、TopK512、128 窗口与压缩率；Prefill 和 Decode 不共享错误执行叙述。紫色 DSpark 明示深度 3、位置 5，A/B 来源与绿色论文验证边界；为何减少串行轮数及为何仍需验证可从详情获得。

这是按首次阅读视角进行的界面审查，没有声称进行真实受试者可用性研究。

## 仍为 Unknown / 未执行

- 官方训练 backward、离散 TopK 梯度策略、量化梯度、分布式 kernel / 通信 / 优化器成本缺少对应实现与运行证据。
- bounded replay 的实际引擎调度、DSpark 概率校准、prefix scheduler、接受纠正、缓存提交与回滚未在固定 model.py 中实现。
- 未下载模型权重、未执行 GPU 模型推理、未测吞吐或接受率。不声称教学页面运行了这些算法。
- 本次浏览器为本机 Chrome / Chromium；未额外执行独立 Edge 二进制测试。
