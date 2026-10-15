"""Validate the new teaching mapping and preservation of SVG geometry."""
import ast
import json
import operator
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from teaching_data import DATA, check
from audited_data import apply

HERE=Path(__file__).resolve().parent
check()
apply(DATA)
ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
def value(node):
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,ast.BinOp):return ops[type(node.op)](value(node.left),value(node.right))
    raise AssertionError(ast.dump(node))

arithmetic=dimensions=0
pattern=r'\[(\d+),(\d+)\]\\times\[(\d+),(\d+)\]=\[(\d+),(\d+)\]'
for key,d in DATA.items():
    for mode in ('parameter','forward','backward'):
        m=d[mode]
        for formula in m['calculations']:
            expr=formula.replace(r'\times','*').replace(r'\,','').replace(' ','')
            if re.fullmatch(r'[\d+*/().=\-]+',expr) and expr.count('=')==1:
                a,b=expr.split('=')
                assert value(ast.parse(a,mode='eval').body)==value(ast.parse(b,mode='eval').body),(key,mode,formula)
                arithmetic+=1
        for s in m['steps']+[s for e in m['extras'] for s in e['steps']]:
            assert s['formulas'] and s['dimensions'] and s['symbols'],(key,mode)
            for dim in s['dimensions']:
                for match in re.finditer(pattern,dim):
                    a,b,c,d0,e,f=map(int,match.groups())
                    assert b==c and (a,d0)==(e,f),(key,mode,dim)
                    dimensions+=1
                assert not re.search(r'\[(B|T|M)[,\]]',dim),(key,mode,dim)

raw=json.dumps(DATA,ensure_ascii=False)
for stale in ('Scenario P','Source Verified','Mathematical Estimate','36,438,679,326,720','F_B','F_E'):
    assert stale not in raw,stale
examples=DATA['compressor']['forward']['extras']
assert examples[0]['steps'][0]['dimensions']==[r'[3,5120]\times[5120,512]=[3,512]']
assert '第三个还在等第四个' in examples[0]['steps'][0]['note']
assert examples[1]['steps'][0]['dimensions']==[r'[4,5120]\times[5120,512]=[4,512]']
assert '输出两行 [2,512]' in examples[1]['steps'][0]['note']
assert '不用等待配对' in examples[2]['title']

before=ET.parse(HERE/'before_single_token.svg').getroot()
after=ET.parse(HERE/'architecture.svg').getroot()
from hc_preservation import without_hc,verify as verify_hc_scope
verify_hc_scope()
before=without_hc(before)
after=without_hc(after)
def geometry(root):
    # All diagram boxes, paths, arrows, groups, paint, and positions must remain.
    shapes=[]
    for node in root.iter():
        name=node.tag.rsplit('}',1)[-1]
        if name in ('text','tspan','title','desc'):continue
        attrs={k:v for k,v in node.attrib.items() if k!='aria-label'}
        shapes.append((name,attrs))
    return shapes
assert geometry(before)==geometry(after),'SVG geometry or colors changed'
assert before.attrib['viewBox']==after.attrib['viewBox']
report={'passed':True,'moduleModes':len(DATA)*3,'numericEquationsVerified':arithmetic,'matrixProductsVerified':dimensions,'compressorExamples':[3,4,1],'unchangedSvgElements':len(geometry(after)),'preserved':['shapes','positions','arrows','colors','groups','viewBox']}
(HERE/'qa/single_token_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
