"""Reproducible offline audit: source AST, independent math, ledger, rendered data.

No torch or checkpoint loading. Source constructor checks use the pinned official
files, formal config, and world_size=1 to recover unsharded logical shapes.
"""
import ast
import json
import math
import operator
import re
import traceback
from pathlib import Path
import audit_model as M
from teaching_data import DATA
from audited_data import apply

HERE=Path(__file__).resolve().parent
AUDIT=apply(DATA)
results=[]
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Pow:operator.pow}

def test(name,fn):
    try:
        detail=fn();results.append(dict(test=name,status='PASS',detail=detail));print('PASS '+name+': '+str(detail))
    except Exception as e:
        results.append(dict(test=name,status='FAIL',detail=str(e)));print('FAIL '+name+': '+str(e));traceback.print_exc()

class Unknown(Exception):pass

def expression(n,env):
    if isinstance(n,ast.Constant):return n.value
    if isinstance(n,ast.Name):
        if n.id not in env:raise Unknown(n.id)
        return env[n.id]
    if isinstance(n,ast.Attribute):
        obj=expression(n.value,env)
        if not isinstance(obj,dict) or n.attr not in obj:raise Unknown(n.attr)
        return obj[n.attr]
    if isinstance(n,(ast.Tuple,ast.List)):return [expression(v,env) for v in n.elts]
    if isinstance(n,ast.BinOp):return OPS[type(n.op)](expression(n.left,env),expression(n.right,env))
    if isinstance(n,ast.UnaryOp):return -expression(n.operand,env) if isinstance(n.op,ast.USub) else expression(n.operand,env)
    if isinstance(n,ast.Subscript):return expression(n.value,env)[expression(n.slice,env)]
    if isinstance(n,ast.IfExp):return expression(n.body if expression(n.test,env) else n.orelse,env)
    if isinstance(n,ast.Call):
        name=ast.unparse(n.func)
        if name=='args.get_moe_config':return [M.C['dspark_n_routed_experts'] if env['layer_id']>=40 else M.C['n_routed_experts'],M.C['dspark_n_activated_experts'] if env['layer_id']>=40 else M.C['n_activated_experts']]
        if name=='layout.layer_ids.index':return env['layout']['layer_ids'].index(expression(n.args[0],env))
        if name=='len':return len(expression(n.args[0],env))
        if name=='nn.Parameter':return expression(n.args[0],env)
        if name in ['torch.empty','torch.ones']:return dict(stored=[expression(a,env) for a in n.args],kind='parameter')
        if name in ['Linear','ColumnParallelLinear','RowParallelLinear','nn.Linear']:
            k,out=[expression(a,env) for a in n.args[:2]]
            has_bias=next((expression(k.value,env) for k in n.keywords if k.arg=='bias'),name=='nn.Linear')
            return dict(stored=[out,k],bias=has_bias,kind='linear')
        if name=='RMSNorm':return dict(stored=[expression(n.args[0],env)],kind='norm')
        if name in ['ParallelEmbedding','ParallelEngramEmbedding','ParallelHead']:
            return dict(stored=[expression(n.args[0],env),expression(n.args[1],env)],kind='table')
        raise Unknown(name)
    raise Unknown(ast.dump(n))

def source_member(p):
    file=p['evidence']['file'].split('/')[-1];cls,member=p['evidence']['symbol'].split('.')
    target=next(n for n in M.TREES[file].body if isinstance(n,ast.ClassDef) and n.name==cls)
    init=next(n for n in target.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    pid=p['id'];layer=40 if pid.endswith('_draft') else 14 if pid=='etable14' else 1 if pid=='etable1' else 20
    if cls=='Engram':layer=1
    env=dict(args=dict(M.C),self={},world_size=1,rank=0,layer_id=layer,dim=M.C['vision_dim'] if pid=='vnorm' else M.D if pid.startswith('etable') else M.H,inter_dim=M.F,vocab_size=M.V,num_embeddings=M.C['engram_num_embeddings'][1 if pid=='etable14' else 0],dspark_markov_rank=M.C['dspark_markov_rank'],input_dim=M.H+M.C['dspark_markov_rank'],layout={'layer_ids':M.C['engram_layer_ids'],'num_embeddings':M.C['engram_num_embeddings'],'head_dim':M.C['engram_head_dim'],'n_heads':M.C['engram_n_heads'],'max_ngram_size':M.C['engram_max_ngram_size']})
    if pid.startswith('etable'):env['dim']=M.C['engram_head_dim']
    env['args']['vision_enabled']=M.C['vision_n_layers']>0
    # Constructor assignment expressions are interpreted, not imported/executed.
    found=None
    for node in ast.walk(init):
        if not isinstance(node,(ast.Assign,ast.AnnAssign)):continue
        val=node.value
        if val is None:continue
        try:value=expression(val,env)
        except (Unknown,KeyError,TypeError,IndexError):continue
        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
        for lhs in targets:
            if isinstance(lhs,ast.Name):env[lhs.id]=value
            elif isinstance(lhs,ast.Tuple):
                for l,v in zip(lhs.elts,value):env[l.id]=v
            elif isinstance(lhs,ast.Attribute) and isinstance(lhs.value,ast.Name) and lhs.value.id=='self':
                env['self'][lhs.attr]=value
                if lhs.attr==member:found=value
    assert found is not None,(pid,p['evidence']['symbol'],'source constructor unresolved')
    if pid in ['patchbias','vqkvbias','vobias','align1bias','align2bias']:
        assert found['kind']=='linear' and found['bias'] is True,(pid,'missing source Bias')
        stored=[found['stored'][0]]
    else:stored=found['stored']
    assert stored==p['storedShape'],(pid,stored,p['storedShape'])
    assert math.prod(stored)==p['count'],pid
    return dict(parameter=pid,shapeFromAST=stored,file=p['evidence']['file'],line=p['evidence']['line'])

def sources():
    fields=M.validate_sources()
    derived=[source_member(p) for p in M.REGISTRY.values()]
    (HERE/'source_shape_validation.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2),encoding='utf-8')
    # Independently confirm creation/ownership branches and shared references.
    source=ast.unparse(M.TREES['model.py'])
    for expr in ['if self.is_kv_source:\n    self.compressor = Compressor(args, layer_id)','if self.is_index_source:\n    self.indexer = Indexer(args, layer_id)','if self.owns_k:','if compress_ratio > 1:','self.mtp[-1].embed = self.embed','self.mtp[-1].head = self.head']:
        assert re.sub(r'\n\s*','\n',expr) in re.sub(r'\n\s*','\n',source),expr
    report=(HERE/'audit_sources/technical_report.txt').read_text(encoding='utf-8')
    for statement in ['20-layer causal encoder followed by a 20-layer decoder','552B backbone parameters and 196B Engram parameters','query heads to 64, the head dimension to 512','query compression dimension to 1280','1 shared expert and 384 routed experts','6 experts','32, the hidden dimension to 1024','keep the vision encoder frozen until the learning-rate decay stage']:
        assert statement in report,statement
    import hashlib
    metadata=json.loads((HERE/'audit_sources/technical_report_manifest.json').read_text(encoding='utf-8'))
    assert metadata['revision']==M.REV and hashlib.sha256((HERE/'audit_sources/DeepSeek_V41_Tech_Report.pdf').read_bytes()).hexdigest()==metadata['sha256']
    return dict(formalConfigFields=fields,constructorFamilies=len(derived),reportPages=metadata['pages'],reportSections=['2.2','2.3','2.5','4.2.1'],allSnapshotHashes=True)

def linear_math():
    n=0
    for p in M.REGISTRY.values():
        if len(p['shape'])!=2 or p['evidence']['symbol'].endswith('weight') and 'Gate' not in p['evidence']['symbol']:continue
        k,out=p['shape'];a=M.linear(k,out)
        assert a['input'][1]==a['weight'][0] and a['output']==[1,out]
        assert a['dxShape']==a['input'] and a['dwShape']==a['weight']
        assert a['parameters']==k*out and a['forwardApprox']==2*k*out
        assert a['forwardExact']==out*(2*k-1)
        assert a['dxExact']==k*(2*out-1) and a['dwExact']==k*out
        n+=1
    # Explicit small actual multiply/add loops independently check approximation boundary.
    for m,k,n0 in [(1,2,3),(3,4,2),(1,5120,512)]:
        a=M.linear(k,n0,m)
        assert a['forwardExact']==m*n0*k+m*n0*(k-1)
        assert a['dwExact']==k*n0*m+k*n0*(m-1)
    return dict(linearFamilies=n,singleTokenOuterProduct='KN exact; 2KN approximation')

def page_math():
    count=products=0
    for key,d in DATA.items():
        for mode in ['parameter','forward','backward']:
            for f in d[mode]['calculations']:
                s=f.replace(r'\times','*').replace(r'\,','').replace(' ','')
                if re.fullmatch(r'[\d+*/().=\-]+',s) and s.count('=')==1:
                    a,b=s.split('=');assert expression(ast.parse(a,mode='eval').body,{})==expression(ast.parse(b,mode='eval').body,{}),(key,mode,f);count+=1
            for step in d[mode]['steps']+[s for e in d[mode]['extras'] for s in e['steps']]:
                assert step['symbols'] and step['formulas'] and step['dimensions'],(key,mode)
                for dim in step['dimensions']:
                    for match in re.finditer(r'\[(\d+),(\d+)\]\\times\[(\d+),(\d+)\]=\[(\d+),(\d+)\]',dim):
                        a,b,c,e,f,g=map(int,match.groups());assert b==c and (a,e)==(f,g),(key,mode,dim);products+=1
                ids={p['bindingId'] for p in d['learnable']};assert set(step['parameterRefs'])<=ids
    assert DATA['query']['ownedParameters']==M.H*M.R+M.R+M.R*M.Q
    for key,shape in [('qa',[M.H,M.R]),('qb',[M.R,M.Q]),('kv',[M.H,M.D])]:assert DATA[key]['learnable'][0]['shape']==shape
    return dict(arithmeticEquations=count,matrixProducts=products,moduleModes=len(DATA)*3)

def parameters():
    for p in M.REGISTRY.values():assert p['count']==math.prod(p['shape'])
    for d in DATA.values():
        assert d['ownedParameters']==sum(p['count']*p['multiplicity'] for p in d['learnable'] if p['owned'])
    assert DATA['overview']['ownedParameters']==M.TOTALS['All logical weights']
    assert M.TOTALS['Text Backbone']+M.TOTALS['Engram']==M.TOTALS['Text + Engram']
    return M.TOTALS

def shared():
    paths=[r['path'] for r in M.LEDGER];assert len(paths)==len(set(paths))
    assert sum(r['parameterId']=='embedding' for r in M.LEDGER)==1
    assert sum(r['parameterId']=='head' for r in M.LEDGER)==1
    assert sum(r['parameterId']=='ckv' for r in M.LEDGER)==4
    assert sum(r['parameterId']=='cgate' for r in M.LEDGER)==3
    assert sum(r['parameterId']=='iq' for r in M.LEDGER)==8
    assert sum(r['parameterId']=='ik' for r in M.LEDGER)==4
    assert sum(r['parameterId']=='etable1' for r in M.LEDGER)==1
    assert len([r for r in M.LEDGER if r['path'].startswith('mtp.') and r['parameterId']=='qa'])==3
    return dict(uniquePaths=len(paths),kvProjectionOwners=4,indexQueryOwners=8,indexKeyOwners=4,mainEmbeddingAndHead='counted once each')

def layered():
    for key in ['encoder','decoder']:
        layers=range(20) if key=='encoder' else range(20,40)
        expected=sum(r['count'] for r in M.LEDGER if any(r['path'].startswith(f'layers.{i}.') for i in layers))
        assert int(DATA[key]['parameter']['totals'][0][1].split()[0].replace(',',''))==expected
        assert DATA[key]['_audit']['decode']==sum(M.decode_layer(i) for i in layers)
    assert M.FORWARD==sum(M.DECODE.values())+2*M.engram()+7*M.H+M.norm(M.H)+2*M.H*M.V
    assert M.FORWARD==35584521265
    return dict(layers=40,modeledDecode=M.FORWARD,scope='text + Engram; not full multimodal execution')

def moe():
    s=DATA['moe']['moeSummary'];ep=M.H*M.F*3
    assert s['expertsTotal']==385*ep and s['expertsActivated']==7*ep
    assert s['total']==385*ep+M.H*384+2*384
    assert s['activated']==7*ep+M.H*384+384
    assert s['fortyTotal']==40*s['total'] and s['fortyActivated']==40*s['activated']
    assert s['activated']<s['total']
    return s

def nonlinear():
    def numeral(s):return int(re.search(r'\d[\d,]*',s).group().replace(',',''))
    for key,pid in [('qnorm','qnorm'),('kvnorm','kvnorm'),('anorm','anorm'),('fnorm','fnorm'),('finalnorm','finalnorm')]:
        d=M.REGISTRY[pid]['count']
        operations=d+(d-1)+1+1+1+d+d
        assert M.norm(d)==operations==numeral(DATA[key]['forward']['rows'][0][1])
    # Enumerate the actual online-softmax operations by stage, including the
    # denominator multiplication and addition; no reuse of old report formulas.
    for positions,key in [(128,'sliding'),(640,'multi'),(640,'sparse')]:
        blocks=positions//64;h=M.A;d=M.D
        per_block={'scale':64,'running_max_difference':1,'score_subtract':64,'reduce_sum_adds':63,'running_denominator_mul_add':2,'rescale_output':d,'exponentials':65}
        tail={'sink_subtract':1,'sink_exp':1,'sink_add':1,'output_divide':d}
        expected=4*h*d*64*blocks+h*blocks*sum(per_block.values())+h*sum(tail.values())
        assert M.attn(positions)['modeled']==expected
        assert numeral(DATA[key]['forward']['totals'][0][1])==expected
    # Check the other published main matrix/rotation terms against source shapes.
    expected={'kv':2*M.H*M.D,'qa':2*M.H*M.R,'qb':2*M.R*M.Q,'query':2*M.H*M.R+2*M.R*M.Q,'head':2*M.H*M.V,'global':2*M.H*M.D,'expert':6*M.H*M.F,'shared':6*M.H*M.F,'output':2*(M.REGISTRY['oa']['count']+M.REGISTRY['ob']['count']),'hc':2*M.H*M.HC*M.MIX,'router':2*M.H*384,'enggate':2*M.EW*5*M.H,'rope':M.A*(M.C['rope_head_dim']//2)*6,'swiglu':3*M.F,'weights':5+1+6+6,'compressor':4*M.H*M.D}
    for key,value in expected.items():assert numeral(DATA[key]['forward']['rows'][0][1])==value,(key,value)
    assert M.hc()==1246042 and M.router()==3933714 and M.engram()==314783776
    return dict(normShapes=5,attentionWidths=[128,640],mainOperationResults=len(expected),specialFunctionPolicy='one operation; memory, comparison and bit operations excluded')

def display():
    raw=json.loads((HERE/'module_data.json').read_text(encoding='utf-8'))
    html=(HERE.parent/'deepseek_v4_1_interactive.html').read_text(encoding='utf-8')
    embedded=json.loads(re.search(r'<script type="application/json" id="module-data">(.*?)</script>',html,re.S).group(1).replace(r'<\/','</'))
    for k,d in DATA.items():
        assert raw[k]==json.loads(json.dumps(d)),k
        for name in ['parameter','forward','backward']:
            for field in ['rows','totals','audit','linearArithmetic','resultAudits','totalAudits']:
                assert json.loads(json.dumps(d[name][field]))==embedded[k][name][field],(k,name,field)
        assert d['learnable']==embedded[k]['learnable'],k
    assert 'id="audit-data"' in html and 'id="flow-bar"' in html and 'id="parameter-mini"' in html
    assert not re.search(r'<script[^>]+src=|<link[^>]+stylesheet',html)
    return dict(modules=len(DATA),standalone=True,dataAndDisplayMatch=True)

def status():
    assert sum(AUDIT['summary'][s] for s in ['Verified','Estimated','Unverified','Mismatch'])==189
    assert AUDIT['summary']['Mismatch']==0
    for k,d in DATA.items():
        assert d['parameter']['audit']['status']=='Verified'
        assert d['forward']['audit']['status']=='Estimated'
        assert set(d['moduleAssumptions'])=={'Configuration','Scenario','Approximation','Unverified'}
        for row in d['backward']['resultAudits']:
            if any(w in row['value'] for w in ['未验证','暂不能','另算','未并入']):assert row['status']=='Unverified',(k,row)
    return AUDIT['summary']

for name,fn in [('Pinned config / source constructors',sources),('Linear shapes / forward / backward math',linear_math),('Page formulas / dimensions / bindings',page_math),('Logical parameter counts',parameters),('Layer and group totals',layered),('Shared weights counted once',shared),('MoE Total / Activated',moe),('Nonlinear and operation subtotals',nonlinear),('Page versus calculation source',display),('Statuses and Assumptions',status)]:test(name,fn)
report=dict(passed=all(r['status']=='PASS' for r in results),revision=M.REV,tests=results)
(HERE/'audit_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
raise SystemExit(0 if report['passed'] else 1)
