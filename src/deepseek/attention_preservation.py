"""Protect the original layout outside the two authorized Attention blocks and KV boundary arrows."""
import re
from pathlib import Path
from xml.etree import ElementTree as ET

HERE=Path(__file__).resolve().parent


def validate(svg):
    before=ET.parse(HERE/'before_single_token.svg').getroot()
    after=ET.fromstring(svg) if isinstance(svg,str) else svg
    assert before.attrib['viewBox']==after.attrib['viewBox']=='0 0 1672 941'

    def elements(root):
        out=[]
        def visit(node):
            if node.attrib.get('id','').endswith('-expanded'):return
            name=node.tag.rsplit('}',1)[-1]
            if name in ['rect','path']:
                attrs=dict(node.attrib)
                if name=='rect':
                    x,y=float(attrs.get('x',0)),float(attrs.get('y',0))
                    authorized=x<885 and y>=580
                else:
                    # Exact unchanged path membership protects arrows in the original sections.
                    nums=list(map(float,re.findall(r'-?\d+(?:\.\d+)?',attrs.get('d',''))))
                    # H/V path syntax lacks paired coordinates; use the initial point and all
                    # numeric coordinates conservatively for the two authorized areas.
                    x,y=(nums+[0,0])[:2]
                    authorized=(x<885 and y>=580) or (950<=x<=1075 and y<300)
                if not authorized:out.append((name,attrs))
            for child in node:visit(child)
        visit(root)
        return out
    actual=elements(after)
    protected=elements(before)
    for item in protected:
        # User-requested removal of the standalone R19 state box. Its incoming
        # stub is replaced by direct Encoder → layer-20 connections.
        name,attrs=item
        if name=='rect' and tuple(attrs.get(k) for k in ('x','y','width','height'))==('870','184','86','110'):
            continue
        if name=='path' and attrs.get('d')=='M859 233 H869':
            continue
        assert item in actual, f'Original non-Attention geometry changed: {item}'
    return len(protected)


if __name__=='__main__':
    print('Protected original geometry elements:',validate((HERE/'architecture.svg').read_text(encoding='utf-8')))
