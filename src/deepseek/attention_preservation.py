"""Allow only the two Attention frames and their labelled handoff to change."""
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib

HERE = Path(__file__).resolve().parent


def without_attention(root):
    root = deepcopy(root)
    content = next(n for n in root.iter() if n.attrib.get('id') == 'diagram-content')
    children = list(content)
    first = next(i for i, n in enumerate(children) if
                 n.attrib.get('id') == 'encoder-attention-overview' or
                 n.attrib.get('aria-label') == '1.  Encoder Attention (Detail)')
    end = next(i for i, n in enumerate(children) if i > first and
               n.attrib.get('aria-label') == '3.  Hyper-Connection + MoE (Detail)')
    for n in children[first:end]:
        content.remove(n)
    return root


def verify():
    baseline = HERE / 'before_attention.svg'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == '9c9b6b9cdd832a0f5ebe85a39819cfa0dd351e4b2dd13abfc9e2fd872f31dc92'
    before = without_attention(ET.parse(baseline).getroot())
    after = without_attention(ET.parse(HERE / 'architecture.svg').getroot())
    assert ET.tostring(before) == ET.tostring(after), 'Element outside the Attention details changed'
    return True


if __name__ == '__main__':
    verify()
    print('PASS: every SVG element outside the two Attention details is unchanged.')
