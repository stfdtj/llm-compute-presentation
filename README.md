# LLM Compute Presentation

两个可离线打开的教学网页，以及从架构、参数量、FLOPs 到 Roofline / MFU 的中文讲稿。

- [DeepSeek 架构交互页](deepseek_v4_1_interactive.html)：参数、前向、局部反向与信息流。
- [LLM Compute Lab](llm_compute_lab/index.html)：六种 Attention 对比、Roofline、MFU 与演示模式。
- [完整演讲稿](presentation_script.md)。

下载后可直接双击 HTML；阅读成品不需要安装依赖。仓库包含两套网页的源码、测试、固定版本审计资料和参考图，不依赖原来的 sharing 工作区，也不包含模型权重。

## 目录与修改入口

| 路径 | 内容 |
|---|---|
| `llm_compute_lab/` | HTML 模板、JS 计算与交互、CSS、打包脚本、测试和 QA 记录 |
| `interactive_work/` | 架构 SVG 生成、教学数据、HTML 模板、审计与浏览器测试 |
| `interactive_work/audit_sources/` | 固定 revision 的参考源码、配置、技术报告和 SHA256 |
| `DeepSeek-V4.1-Flash/` | 原项目的本地模型源码与配置；用于静态分析，不是完整推理部署 |
| `FLOPs_calculation_verify.py` | 根目录计算复核，也是 DeepSeek 教学构建的依赖 |
| `FLOPs_calculation_forward.txt`、`FLOPs_calculation_audit.md` | 计算报告与边界 |
| `calculation_audit.md`、`assumptions.md` | 架构页构建时生成的审计说明 |

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
.venv\Scripts\python.exe llm_compute_lab/build.py
.venv\Scripts\python.exe interactive_work/build.py
npm run test:lab
.venv\Scripts\python.exe FLOPs_calculation_verify.py
.venv\Scripts\python.exe interactive_work/verify_audit.py
.venv\Scripts\python.exe interactive_work/verify_single_token.py
.venv\Scripts\python.exe interactive_work/verify_calculations.py
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

依次运行 `interactive_work/fetch_audit_sources.py` 和 `fetch_technical_report.py` 后重新审计；`inspect_report.py` 可渲染报告页。更新会改变 revision / SHA256，应单独审查。

参考源码与报告的来源见 `interactive_work/audit_sources/manifest.json`。官方源码许可见该目录的 LICENSE；原始参考图保留为 `interactive_work/reference.png`，用于架构几何对照。

## 数据边界

Roofline 流量是声明假设下的估算；MFU 教学数据是假设值。局部反向示例不代表已验证的完整训练实现，GDA 未核验的完整总量仍为未知。历史截图和 JSON 是对应版本的验证记录，不是训练或硬件性能测量。

## 本次迁移验证（2026-10-14）

已在本仓库新建虚拟环境并从依赖清单安装，不借用原工作区的 Python 包或 node_modules。两个网页均独立重建成功，生成文件与原成品逐字节一致。

- Lab：65 项计算测试、98 项浏览器检查通过。
- DeepSeek：10 类源码与算术审计通过；83 个数值等式、160 个矩阵乘法和 323 个 SVG 几何元素核对通过；数值梯度测试通过。
- 浏览器：189 个模块/模式组合、参数绑定、五步信息流、六段矩阵演示和多尺寸布局通过，固定数据/图形/审计基线 SHA256 保持不变。

这些检查验证教学网页及其计算口径，不涉及真实 GPU/NPU 训练或性能测量。
