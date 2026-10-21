"""Confirm that an independent presentation importer retains native click effects."""
import json,os,subprocess
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
ROOT=Path(__file__).resolve().parent
def main():
    pptx=ROOT/'DeepSeek_V41_Flash_计算成本可视化.pptx'
    out=ROOT/'qa'/'import_audit';out.mkdir(exist_ok=True,parents=True)
    cmd=[str(ROOT/'tools/lo/program/soffice.com'),'-env:UserInstallation='+(ROOT/'tools/lo_profile').as_uri(),'--headless','--convert-to','odp','--outdir',str(out),str(pptx)]
    r=subprocess.run(cmd,capture_output=True,encoding='utf-8',errors='replace',timeout=180)
    if r.returncode:raise RuntimeError(r.stderr)
    ns={'anim':'urn:oasis:names:tc:opendocument:xmlns:animation:1.0','draw':'urn:oasis:names:tc:opendocument:xmlns:drawing:1.0','pres':'urn:oasis:names:tc:opendocument:xmlns:presentation:1.0'}
    outline=json.loads((ROOT/'outline.json').read_text(encoding='utf-8'))
    with ZipFile(out/(pptx.stem+'.odp')) as z:root=E.fromstring(z.read('content.xml'))
    rows=[]
    for i,(p,s) in enumerate(zip(root.findall('.//draw:page',ns),outline['slides']),1):
        clicks=[x for x in p.iter() if x.attrib.get('{'+ns['pres']+'}node-type')=='on-click']
        assert len(clicks)==len(s['clicks']),(i,len(clicks),len(s['clicks']))
        rows.append(dict(page=i,click_groups=len(clicks),animation_sets=len(p.findall('.//anim:set',ns))))
    assert len(rows)==22
    mfu=root.findall('.//draw:page',ns)[20]
    begin_key='{urn:oasis:names:tc:opendocument:xmlns:smil-compatible:1.0}begin'
    delays=[x.attrib[begin_key] for x in mfu.findall('.//anim:set',ns) if begin_key in x.attrib]
    assert all(v in delays for v in ['0.09s','0.18s','0.27s','0.36s','0.45s','0.54s','0.63s','0.72s','0.81s','0.9s']),delays
    report=dict(passed=True,engine='LibreOffice 26.2.6.3 independent OOXML animation import',click_groups=sum(r['click_groups'] for r in rows),mfu_progress_delays=delays,slides=rows,limitation='This checks native timeline interpretation, not a live Microsoft PowerPoint slideshow session.')
    (ROOT/'qa'/'animation_import_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Independent animation import: 22 pages,',report['click_groups'],'click groups retained.')
if __name__=='__main__':main()
