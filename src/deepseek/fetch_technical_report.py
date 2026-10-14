"""Read the official technical report at the audit's fixed revision."""
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent
import pymupdf
manifest=json.loads((HERE.parents[1]/'references/deepseek/manifest.json').read_text(encoding='utf-8'))
url='https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/'+manifest['revision']+'/DeepSeek_V41_Tech_Report.pdf?download=true'
with urlopen(url,timeout=45) as response:content=response.read()
pdf=HERE.parents[1]/'references/deepseek/DeepSeek_V41_Tech_Report.pdf';pdf.write_bytes(content)
with pymupdf.open(pdf) as document:
    pages=[p.get_text() for p in document]
(HERE.parents[1]/'references/deepseek/technical_report.txt').write_text('\n'.join('PAGE '+str(i+1)+'\n'+text for i,text in enumerate(pages)),encoding='utf-8')
(HERE.parents[1]/'references/deepseek/technical_report_manifest.json').write_text(json.dumps(dict(url=url,revision=manifest['revision'],sha256=hashlib.sha256(content).hexdigest(),pages=len(pages)),indent=2),encoding='utf-8')
print('Read',len(pages),'report pages; SHA256',hashlib.sha256(content).hexdigest())
