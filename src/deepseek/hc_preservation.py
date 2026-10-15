"""Keep the authorized HC edit isolated from every other architecture element."""
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib

HERE=Path(__file__).resolve().parent
ORIGINAL_SHA256='8861c3b502047b9ba507645343bedf1494cc1719d10060c22ba67072542ec84c'

def without_hc(root):
    root=deepcopy(root)
    content=next(n for n in root.iter() if n.attrib.get('id')=='diagram-content')
    children=list(content)
    modern=next((n for n in children if n.attrib.get('id')=='hc-overview'),None)
    if modern is not None:
        content.remove(modern)
    else:
        first=next(i for i,n in enumerate(children) if n.attrib.get('aria-label')=='Hyper-Connection HC')
        end=next(i for i,n in enumerate(children) if i>first and n.attrib.get('aria-label')=='Mixture of Experts MoE')
        for n in children[first:end]:content.remove(n)
    return root

def verify():
    baseline=HERE/'before_hc.svg'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==ORIGINAL_SHA256,'HC baseline was changed'
    from attention_preservation import without_attention,verify as verify_attention
    verify_attention()
    before=without_attention(without_hc(ET.parse(baseline).getroot()))
    after=without_attention(without_hc(ET.parse(HERE/'architecture.svg').getroot()))
    assert ET.tostring(before)==ET.tostring(after),'An architecture element outside HC changed'
    return True
