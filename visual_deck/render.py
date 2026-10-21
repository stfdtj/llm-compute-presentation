"""Render the actual PPTX with LibreOffice, then inspectable PNGs from its PDF."""
import argparse,hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'/'pydeps'))
import fitz
from PIL import Image,ImageDraw

def render(pptx,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    soffice=os.environ.get('SOFFICE') or shutil.which('soffice') or str(ROOT/'tools'/'lo'/'program'/'soffice.com')
    profile=(ROOT/'tools'/'lo_profile').as_uri()
    cmd=[str(soffice),'-env:UserInstallation='+profile,'--headless','--convert-to','pdf','--outdir',str(outdir),str(pptx)]
    pdf=outdir/(pptx.stem+'.pdf')
    # Fail closed: an old successful PDF must never mask a conversion failure.
    pdf.unlink(missing_ok=True)
    start=time.time()
    r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180,env=dict(os.environ,SAL_USE_VCLPLUGIN='svp'))
    if r.returncode or not pdf.exists(): raise RuntimeError(r.stdout+'\n'+r.stderr)
    doc=fitz.open(pdf)
    records=[]
    for i,p in enumerate(doc,1):
        dest=outdir/f'slide-{i:02d}.png'
        pix=p.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False);pix.save(dest)
        records.append(dict(page=i,file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),size=[pix.width,pix.height]))
    # Contact sheets are review aids, not replacement slide assets.
    for start_page in range(0,len(records),6):
        chunk=records[start_page:start_page+6]
        sheet=Image.new('RGB',(1440,3*435),'#e8edf5');draw=ImageDraw.Draw(sheet)
        for j,rec in enumerate(chunk):
            im=Image.open(outdir/rec['file']);im.thumbnail((700,395))
            x=10+(j%2)*720;y=12+(j//2)*435
            sheet.paste(im,(x,y));draw.text((x,y+400),str(rec['page']).zfill(2),fill='#15253d')
        sheet.save(outdir/f'contact-{start_page//6+1:02d}.jpg',quality=94)
    report=dict(engine='LibreOffice 26.2.6.3 PDF export + PyMuPDF 1.26.5 rasterization',pptx=str(pptx),pptx_sha256=hashlib.sha256(pptx.read_bytes()).hexdigest(),pdf=str(pdf),count=len(records),elapsed_seconds=round(time.time()-start,2),slides=records,stdout=r.stdout,stderr=r.stderr)
    (outdir/'render_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(rendered=len(records),outdir=str(outdir)),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'DeepSeek_V41_Flash_计算成本可视化.pptx');p.add_argument('--outdir',type=Path,default=ROOT/'renders')
    a=p.parse_args();render(a.input.resolve(),a.outdir.resolve())
