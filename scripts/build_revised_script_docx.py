"""Build a standalone editable DOCX without external dependencies; assert text parity."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET
import json, hashlib, re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'presentation_script_pre_v3_restructured.md'
TARGET = ROOT / 'presentation_script_pre_v3_final.docx'
NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def paragraph(text, style):
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'

paragraphs, plain = [], []
for line in SOURCE.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    style = 'Normal'
    if line.startswith('# '):
        style, line = 'Title', line[2:]
    elif line.startswith('## '):
        style, line = 'Heading1', line[3:]
    elif line.startswith('【操作：'):
        style = 'Operation'
    elif line.startswith(('版本：', '使用说明')):
        style = 'Meta'
    elif re.match(r'^(?:P\(|F_|F\(|B\(|AI|T_|MFU =|P_total|dW_|dH′ =|dX =|Y =|alpha =|r\(v\)|40 ×)', line):
        style = 'Formula'
    paragraphs.append(paragraph(line, style))
    plain.append(line)

styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="微软雅黑"/><w:sz w:val="23"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="140" w:line="330" w:lineRule="auto"/><w:widowControl/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="260"/></w:pPr><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:keepLines/><w:outlineLvl w:val="0"/><w:spacing w:before="300" w:after="160"/></w:pPr><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="29"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Operation"><w:name w:val="网页操作（不口述）"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:keepLines/><w:shd w:val="clear" w:fill="EDF3FA"/><w:spacing w:before="80" w:after="140"/></w:pPr><w:rPr><w:color w:val="345B85"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Formula"><w:name w:val="Formula"/><w:basedOn w:val="Normal"/><w:pPr><w:keepLines/><w:spacing w:after="160"/></w:pPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:color w:val="203D61"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Meta"><w:name w:val="Meta"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="667085"/><w:sz w:val="19"/></w:rPr></w:style>
</w:styles>'''
document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body>{''.join(paragraphs)}<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567"/></w:sectPr></w:body></w:document>'''
files = {
    '[Content_Types].xml': '''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>''',
    '_rels/.rels': '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>''',
    'word/_rels/document.xml.rels': '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>''',
    'word/document.xml': document,
    'word/styles.xml': styles,
    'word/footer1.xml': f'''<?xml version="1.0" encoding="UTF-8"?><w:ftr xmlns:w="{NS}"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:color w:val="7A8496"/><w:sz w:val="18"/></w:rPr><w:t>DeepSeek V4.1 Flash · 演讲稿  |  </w:t></w:r><w:fldSimple w:instr="PAGE"/></w:p></w:ftr>''',
}
with ZipFile(TARGET, 'w', ZIP_DEFLATED) as z:
    for name, data in files.items():
        ET.fromstring(data)
        z.writestr(name, data.encode('utf-8'))
with ZipFile(TARGET) as z:
    root = ET.fromstring(z.read('word/document.xml'))
    actual = [''.join(p.itertext()) for p in root.findall(f'.//{{{NS}}}p')]
assert actual == plain, 'DOCX/Markdown text differs'
assert len([p for p in plain if p.startswith('【操作：')]) == 37
for forbidden in ['此处补充细节', '这里你可以自由发挥', '视情况解释', '根据听众反应']:
    assert not any(forbidden in p for p in plain)
evidence = ROOT / 'docs/script_revision_evidence'
evidence.mkdir(exist_ok=True)
report = {'docx_markdown_text_equal': True, 'paragraphs': len(plain), 'operations': 37, 'md_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'docx_sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest()}
(evidence / 'document_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
