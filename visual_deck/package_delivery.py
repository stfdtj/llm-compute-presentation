"""Package portable build sources and the four requested deliverables."""
import hashlib,json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parent
def main():
    code=['README.md','plan.py','outline.json','scenes.js','build.js','check.py','render.py','audit_rendered.py','audit_import.py','checkpoints.py','package_delivery.py','package.json','requirements-render.txt','slide_mapping.md','technical_report.md','source_differences.md','script_extracted.txt','source_manifest.json','technical_checks.json','scene_manifest.json']
    files=[ROOT/p for p in code]+list((ROOT/'vendor').rglob('*'))+list((ROOT/'evidence').rglob('*'))
    source_zip=ROOT/'DeepSeek_V41_Flash_sources.zip'
    with ZipFile(source_zip,'w',ZIP_DEFLATED) as z:
        for p in files:
            if p.is_file():z.write(p,p.relative_to(ROOT))
    requested=[ROOT/'DeepSeek_V41_Flash_计算成本可视化.pptx',ROOT/'renders/DeepSeek_V41_Flash_计算成本可视化.pdf',source_zip,ROOT/'slide_mapping.md',ROOT/'technical_report.md',ROOT/'source_differences.md',ROOT/'README.md']
    report=dict(slides=22,click_groups=83,rendered_pages=22,reviewed_key_states=47,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in requested])
    (ROOT/'delivery_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with ZipFile(ROOT/'DeepSeek_V41_Flash_delivery.zip','w',ZIP_DEFLATED) as z:
        for p in requested:z.write(p,p.name)
        z.write(ROOT/'delivery_manifest.json','delivery_manifest.json')
        for p in sorted((ROOT/'renders').glob('contact-*.jpg')):z.write(p,'preview/'+p.name)
        for name in ['visual_review_receipt.json','visual_judgment.json','rendered_text_audit.json','animation_import_audit.json']:
            p=ROOT/'qa'/name
            if p.exists():z.write(p,'checks/'+name)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
