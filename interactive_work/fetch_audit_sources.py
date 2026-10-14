"""Pin official reference sources for a reproducible, read-only model audit."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def fetch(url):
    with urlopen(Request(url,headers={'User-Agent':'Codex-local-calculation-audit'}),timeout=45) as r:return r.read()
def main():
    info=json.loads(fetch('https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash'))
    revision=info['sha']
    folder=HERE/'audit_sources';folder.mkdir(exist_ok=True)
    files=['config.json','README.md','inference/config.json','inference/model.py','inference/kernel.py','inference/engram.py','inference/vision.py']
    def get(name):
        url=f'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{revision}/{name}'
        content=fetch(url);target=folder/name;target.parent.mkdir(exist_ok=True);target.write_bytes(content)
        local=ROOT/'DeepSeek-V4.1-Flash'/Path(name).name
        same=local.exists() and local.read_text(encoding='utf-8').replace('\r\n','\n')==content.decode('utf-8').replace('\r\n','\n')
        return dict(path=name,url=url,sha256=hashlib.sha256(content).hexdigest(),localExists=local.exists(),matchesLocalNormalized=same)
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(get,files))
    report={'revision':revision,'files':records}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
