"""Independently audit arithmetic, shape bounds and native click animation targets."""
import hashlib,json,re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E

ROOT=Path(__file__).resolve().parent
PPTX=ROOT/'DeepSeek_V41_Flash_计算成本可视化.pptx'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def check():
    d=json.loads((ROOT/'outline.json').read_text(encoding='utf-8'))
    results=[]
    with ZipFile(PPTX) as z:
        for i,s in enumerate(d['slides'],1):
            r=E.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
            ids=[x.attrib['id'] for x in r.findall('.//p:cNvPr',NS)]
            assert len(ids)==len(set(ids)),f'Duplicate shape ids slide {i}'
            targets=[x.attrib['spid'] for x in r.findall('.//p:timing//p:spTgt',NS)]
            assert set(targets)<=set(ids),f'Unknown animation target slide {i}'
            click=r.findall('.//p:cTn[@nodeType="clickEffect"]',NS)
            assert len(click)==len(s['clicks']),(i,len(click),len(s['clicks']))
            tids=[x.attrib['id'] for x in r.findall('.//p:timing//p:cTn',NS)]
            assert len(tids)==len(set(tids)),f'Duplicate timing ids slide {i}'
            notes=z.read(f'ppt/notesSlides/notesSlide{i}.xml').decode()
            assert s['section'] in notes
            results.append(dict(page=i,shapes=len(ids)-1,clicks=len(click),targets=len(targets),notes=True))
    shapes=json.loads((ROOT/'scene_manifest.json').read_text(encoding='utf-8'))
    outside=[]
    for slide in shapes:
        for o in slide['objects']:
            b=o['bounds']
            if b['x']<0 or b['y']<0 or b['x']+b['w']>10.001 or b['y']+b['h']>5.626:outside.append((slide['page'],o))
    assert not outside,outside
    unexpected_rect_overlaps=[]
    for slide in shapes:
        rects=[o for o in slide['objects'] if o['kind']=='rect']
        for i,a in enumerate(rects):
            for b in rects[i+1:]:
                x,y=a['bounds'],b['bounds']
                ix=min(x['x']+x['w'],y['x']+y['w'])-max(x['x'],y['x'])
                iy=min(x['y']+x['h'],y['y']+y['h'])-max(x['y'],y['y'])
                if ix<=.001 or iy<=.001:continue
                contains=lambda u,v:u['x']<=v['x']+.001 and u['y']<=v['y']+.001 and u['x']+u['w']>=v['x']+v['w']-.001 and u['y']+u['h']>=v['y']+v['h']-.001
                # Highlight cells and the progress fill intentionally cover their base.
                if contains(x,y) or contains(y,x):continue
                unexpected_rect_overlaps.append(dict(page=slide['page'],a=a['name'],b=b['name']))
    assert not unexpected_rect_overlaps,unexpected_rect_overlaps
    K1,N1,K2,N2=5120,1280,1280,32768
    P1,P2=K1*N1,K2*N2
    F1,F2=2*P1,2*P2
    exact_f=sum(n*(2*k-1) for k,n in [(K1,N1),(K2,N2)])
    exact_dx=sum(k*(2*n-1) for k,n in [(K1,N1),(K2,N2)])
    exact_dw=P1+P2
    numeric=dict(query_parameters=[P1,P2],query_linear_parameters=P1+P2,query_with_norm_parameters=P1+P2+1280,forward_approx=[F1,F2],forward_total=F1+F2,backward_approx=2*(F1+F2),forward_exact=exact_f,backward_exact_dx=exact_dx,backward_exact_dw=exact_dw,backward_exact=exact_dx+exact_dw,expert_params=3*5120*2304,router=5120*384,mfu=100.6e6*60e9/(6*2048*989.5e12))
    numeric['roofline']=[dict(M=m,F=(F1+F2)*m,B=(P1+P2)*2+80896*m,AI=(F1+F2)*m/((P1+P2)*2+80896*m)) for m in [1,1024]]
    assert (P1,P2,F1+F2,exact_dx+exact_dw)==(6553600,41943040,96993280,145483520)
    report=dict(passed=True,slide_count=len(results),click_count=sum(r['clicks'] for r in results),native_targets=sum(r['targets'] for r in results),out_of_bounds=len(outside),unexpected_rectangle_overlaps=len(unexpected_rect_overlaps),pptx_sha256=hashlib.sha256(PPTX.read_bytes()).hexdigest(),numeric=numeric,slides=results)
    (ROOT/'technical_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':check()
