# DeepSeek V4.1 Flash 计算成本可视化

主文件：`DeepSeek_V41_Flash_计算成本可视化.pptx`，22 页，16:9。按原 Word 顺序讲解，83 组原生点击显现，全部使用可编辑的 Shapes、箭头和文本。

## 使用

在 PowerPoint 中按 F5 从头放映，按空格或鼠标依次推进。编辑视图显示所有最终元素是正常现象；点击过程在放映视图中播放。矩阵页分别用 7 次、4 次点击完成；MFU 第三次点击后进度条自动逐段填充约 0.9 秒，最后显示 49.6%。

每页 Speaker Notes 标注 Word 章节、段落索引和点击提示。完整映射见 [slide_mapping.md](slide_mapping.md)。字体使用 Microsoft YaHei；替换字体后应重新检查。

## 文件

- `plan.py`：语义页面计划、讲稿映射、Speaker Notes。
- `outline.json`：可重新生成的页面内容源。
- `scenes.js`：数学场景的原生图形渲染器。
- `build.js`：调用 presentation-skill 源码渲染器，并在源代码加载阶段加入原生 OOXML Appear 时间线；不修改生成后的 PPTX。
- `vendor/presentation-skill/`：构建必需的上游 MIT 授权源码子集，已含许可证。
- `check.py`：数学、目标引用、点击组与图形几何核对。
- `render.py`：真实 PPTX → LibreOffice PDF → PNG。
- `audit_rendered.py`：实际 PDF 字形保留、边界和碰撞核对。
- `audit_import.py`：独立动画导入检查。
- `checkpoints.py`：生成关键点击状态的 QA 源文件。
- `technical_report.md`：公式、统计范围和检查结果。
- `source_differences.md`：Word、本地新版和远端旧版的差异。

## 重建

需要 Node.js 22+ 和 Python 3.12+。在本目录运行：

```powershell
npm install
python plan.py
node build.js
python check.py
```

核心依赖固定为 PptxGenJS 4.0.1、JSZip 3.10.1。PPTX 容器元数据会包含生成时刻，重复构建可见内容一致，不要求 ZIP 字节逐位相同。

## 重新渲染

渲染需要 LibreOffice 和 Python 图像依赖。当前工作区的 `tools/` 是本地解包的运行时，不包含在源码 ZIP 内；重建 PPT 本身不需要它。

```powershell
python -m pip install -r requirements-render.txt
$env:SOFFICE = 'C:\Program Files\LibreOffice\program\soffice.com'
python render.py
python audit_rendered.py
```

PNG 在 `renders/`，PDF 是无动画的最终状态预览。检查关键点击状态：

```powershell
python checkpoints.py
$env:TEACHING_MANIFEST = 'qa/checkpoints/scene_manifest.json'
node build.js --outline qa/checkpoints/outline.json --output qa/checkpoints/animation_checkpoints.pptx --style-preset lab-report
Remove-Item Env:\TEACHING_MANIFEST
python render.py --input qa/checkpoints/animation_checkpoints.pptx --outdir qa/checkpoints/renders
```

数据口径优先遵循本次用户明确要求。远端分支的 MFU 默认值仍为旧四卡算例，不能直接替换本 PPT 的 49.6% 教学输入。
