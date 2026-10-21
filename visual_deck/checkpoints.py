"""Build semantic snapshots of the key click sequences for visual QA."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT/'outline.json').read_text(encoding='utf-8'))
slides=[]
for i in [7,12,13,14,16,17,19,21]:
    s=d['slides'][i-1]
    for cap in range(len(s['clicks'])+1):
        v=dict(s,stage_cap=cap,page_no=i)
        v['slide_id']=s['slide_id']+f'-checkpoint-{cap}'
        v['section']=s['section']+f' | Visual QA · Click {cap}/{len(s["clicks"])}'
        slides.append(v)
d['slides']=slides
p=ROOT/'qa'/'checkpoints';p.mkdir(exist_ok=True,parents=True)
(p/'outline.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print('Checkpoint pages:',len(slides))
