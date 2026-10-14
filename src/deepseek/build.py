"""Build the offline, self-contained deliverable from independent content/layout."""
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
from latex2mathml.converter import convert
from teaching_data import DATA,check
from audited_data import apply,write_docs
from build_diagram import build

check()
AUDIT=apply(DATA)
write_docs(AUDIT,DATA)
(ROOT/'src/deepseek/audit_data.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2),encoding='utf-8')
svg=build()
root=ET.fromstring(svg)
ids=[e.attrib['data-module'] for e in root.iter() if 'data-module' in e.attrib]
assert all(i in DATA for i in ids)
# Save editable calculation content BEFORE compiling equations to MathML.
(ROOT/'src/deepseek/module_data.json').write_text(json.dumps(DATA,ensure_ascii=False,indent=2),encoding='utf-8')
for d in DATA.values():
    for name in ['parameter','forward','backward']:
        m=d[name]
        m['calculations']=[convert(f,display='block') for f in m['calculations']]
        steps=m['steps']+[s for extra in m['extras'] for s in extra['steps']]
        for s in steps:
            s['formulas']=[convert(f,display='block') for f in s['formulas']]
            s['dimensions']=[convert(f,display='block') for f in s['dimensions']]
template=(ROOT/'src/deepseek/page_template.html').read_text(encoding='utf-8')
result=template.replace('__SVG__',svg).replace('__DATA__',json.dumps(DATA,ensure_ascii=False).replace('</','<\\/'))
audit_ui=(ROOT/'src/deepseek/audit_ui.html').read_text(encoding='utf-8').replace('__AUDIT__',json.dumps(AUDIT,ensure_ascii=False).replace('</','<\\/'))
result=result.replace('</body>',(ROOT/'src/deepseek/flow_ui.html').read_text(encoding='utf-8')+'\n'+audit_ui+'\n</body>')
result=result.replace('</body>',(ROOT/'src/deepseek/flow_attention_ui.html').read_text(encoding='utf-8')+'\n</body>')
(ROOT/'deepseek_v4_1_interactive.html').write_text(result,encoding='utf-8')
(ROOT/'src/deepseek/architecture.svg').write_text(svg,encoding='utf-8')
print(f'Built standalone HTML: {len(result.encode("utf-8")):,} bytes; {len(ids)} interactive SVG elements / {len(set(ids))} keys; {len(DATA)} calculation entries.')
