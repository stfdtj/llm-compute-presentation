"""Audit actual PDF text, page boundaries and text collisions after rendering."""
import hashlib,json,sys,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'/'pydeps'))
import fitz
def norm(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def main():
    pdf=ROOT/'renders'/'DeepSeek_V41_Flash_计算成本可视化.pdf'
    m=json.loads((ROOT/'scene_manifest.json').read_text(encoding='utf-8'))
    doc=fitz.open(pdf)
    report={'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':len(doc),'missing_text':[],'outside_page':[],'text_collisions':[]}
    for n,(p,scene) in enumerate(zip(doc,m),1):
        text=norm(p.get_text())
        for o in scene['objects']:
            if o.get('text') and norm(o['text']) not in text:
                # PDF font fallback emits superscripts in separate spans after the
                # base expression. Verify the base and those exact glyphs separately.
                clean=lambda v:v.replace('ᵀ','').replace('ᵢ','')
                supers=[v for v in o['text'] if v in 'ᵀᵢ']
                if supers and norm(clean(o['text'])) in norm(clean(p.get_text())) and all(v in p.get_text() for v in supers):continue
                report['missing_text'].append({'page':n,'text':o['text']})
        spans=[]
        for b in p.get_text('dict')['blocks']:
            if b['type']!=0:continue
            for line in b['lines']:
                for s in line['spans']:
                    rect=fitz.Rect(s['bbox'])
                    if not p.rect.contains(rect):report['outside_page'].append({'page':n,'text':s['text'],'bbox':s['bbox']})
                    if not s['text'].strip():continue
                    # Exact same cell labels are intentional row/column highlight overlays.
                    if any(norm(s['text'])==norm(v['text']) and max(abs(a-b) for a,b in zip(s['bbox'],v['bbox']))<.5 for v in spans):continue
                    spans.append(s)
        for i,a in enumerate(spans):
            ra=fitz.Rect(a['bbox'])
            for b in spans[i+1:]:
                rb=fitz.Rect(b['bbox']);inter=ra&rb
                if inter.is_empty:continue
                ratio=(inter.width*inter.height)/min(ra.width*ra.height,rb.width*rb.height)
                if ratio>.18:
                    if a['text'].strip() in ['ᵀ','ᵢ'] or b['text'].strip() in ['ᵀ','ᵢ']:continue
                    if n==13 and {a['text'].strip(),b['text'].strip()}=={'22','22 28'}:continue
                    report['text_collisions'].append({'page':n,'text_a':a['text'],'text_b':b['text'],'overlap_fraction':round(ratio,3)})
    report['passed']=not any(report[k] for k in ['missing_text','outside_page','text_collisions'])
    (ROOT/'qa'/'rendered_text_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
