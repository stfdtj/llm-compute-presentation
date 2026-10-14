"""Independent config/source parameter ledger and mathematical operation audit.

This module never imports teaching_data or the previous FLOPs report. Logical
weights are counted once by model namespace; quantization metadata is separate.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCES=ROOT/'references/deepseek'
MANIFEST=json.loads((SOURCES/'manifest.json').read_text(encoding='utf-8'))
REV=MANIFEST['revision']
CONFIG=json.loads((SOURCES/'inference/config.json').read_text(encoding='utf-8'))
OFFICIAL=json.loads((SOURCES/'config.json').read_text(encoding='utf-8'))
TREES={f:ast.parse((SOURCES/'inference'/f).read_text(encoding='utf-8')) for f in ['model.py','kernel.py','vision.py','engram.py']}

def evidence(cls,member='',file='model.py'):
    tree=TREES[file]
    target=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==cls)
    hits=[n for n in ast.walk(target) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and n.attr==member and isinstance(n.ctx,ast.Store)] if member else []
    methods=[n for n in target.body if isinstance(n,ast.FunctionDef) and n.name==member] if isinstance(target,ast.ClassDef) else []
    line=hits[0].lineno if hits else methods[0].lineno if methods else target.lineno
    return dict(file='inference/'+file,line=line,url=f'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{REV}/inference/{file}#L{line}',symbol=cls+('.'+member if member else ''))

C=CONFIG
H=C['dim'];R=C['q_lora_rank'];A=C['n_heads'];D=C['head_dim'];F=C['moe_inter_dim'];V=C['vocab_size'];HC=C['hc_mult'];MIX=HC*(2+HC)
Q=A*D;GROUP=C['o_groups'];GRP_IN=Q//GROUP;GRP_OUT=C['o_lora_rank'];OUT_RANK=GROUP*GRP_OUT
REGISTRY={}
def p(pid,name,shape,cls,member,role,stored=None,sharing='同一模块在不同 token 上复用；各主干层有自己的权重',file='model.py'):
    REGISTRY[pid]=dict(id=pid,name=name,shape=list(shape),storedShape=list(stored or shape),count=math.prod(shape),role=role,sharing=sharing,evidence=evidence(cls,member,file),status='Verified',requiresGrad='结构参数；本推理实现未提供训练冻结策略，不能据此判断某次训练是否更新')
def lin(pid,name,k,n,cls,member,role,**kw):p(pid,name,(k,n),cls,member,role,stored=(n,k),**kw)
lin('qa','W_QA',H,R,'Attention','wq_a','压缩当前输入')
p('qnorm','RMSNorm γ',(R,),'Attention','q_norm','逐特征缩放 Query 中间向量')
lin('qb','W_QB',R,Q,'Attention','wq_b','展开当前 Query 为 64 个 Head')
lin('kv','W_KV (Local)',H,D,'Attention','wkv','生成该层 Local KV；64 个 Head 读取同一向量')
p('kvnorm','KV RMSNorm γ',(D,),'Attention','kv_norm','逐特征缩放 Local KV')
p('oa','W_OA · 8 组',(GROUP,GRP_IN,GRP_OUT),'Attention','wo_a','每组独立压缩注意力结果',stored=(OUT_RANK,GRP_IN))
lin('ob','W_OB',OUT_RANK,H,'Attention','wo_b','返回模型特征宽度')
p('sink','Attention Sink',(A,),'Attention','attn_sink','每个 Head 一个分母项，不提供 Value')
p('anorm','Attention RMSNorm γ',(H,),'Block','attn_norm','缩放合并后的 Attention 输入')
p('fnorm','FFN RMSNorm γ',(H,),'Block','ffn_norm','缩放合并后的 FFN 输入')
lin('hc_fn','W_HC',HC*H,MIX,'Block','hc_attn_fn','生成混合比例的原始分数')
p('hc_base','HC base',(MIX,),'Block','hc_attn_base','调整混合分数的偏置')
p('hc_scale','HC scale',(3,),'Block','hc_attn_scale','调整 pre / post / comb 三类分数')
for e,suffix in [(C['n_routed_experts'],''),(C['dspark_n_routed_experts'],'_draft')]:
    lin('router'+suffix,'W_R',H,e,'Gate','weight','为候选专家打分')
    p('bias'+suffix,'Router selection bias',(e,),'Gate','bias','只用于选择专家；不是线性层输出 Bias')
    p('bias_vl'+suffix,'Router image bias',(e,),'Gate','bias_vl','图像 token 的选择偏置')
lin('w1','W₁ (Gate)',H,F,'Expert','w1','生成门控分支')
lin('w3','W₃ (Up)',H,F,'Expert','w3','生成内容分支')
lin('w2','W₂ (Down)',F,H,'Expert','w2','把门控结果变回 5120 维')
lin('ckv','W_KV (Global)',H,D,'Compressor','wkv','源层生成可供后续层读取的 KV',sharing='layer 20 生成、20–39 读取；ratio=2 源层分别为 2 / 8 / 14，不计为消费者的额外权重')
lin('cgate','W_gate',H,D,'Compressor','wgate','ratio>1 时生成逐特征分组权重')
p('cnorm','Compressor RMSNorm γ',(D,),'Compressor','norm','缩放压缩向量')
IH=C['index_n_heads'];ID=C['index_head_dim'];IQ=IH*ID
lin('iq','W_IQ',R,IQ,'Indexer','wq_b','生成较小的索引 Query')
lin('ih','W_IH',H,IH,'Indexer','weights_proj','生成各索引 Head 的混合比例')
lin('ik','W_IK',D,ID,'Indexer','wk','仅 KV 源层生成新的索引 Key')
p('inorm','Indexer RMSNorm γ',(ID,),'Indexer','k_norm','只在新索引 Key 所有者创建')
p('embedding','Token Embedding E',(V,H),'ParallelEmbedding','weight','按整数 token 编号查表',sharing='所有 token 复用；三个 DSpark 阶段引用同一张主词表，统计一次')
lin('head','W_head',H,V,'ParallelHead','weight','生成词表分数',sharing='与 Token Embedding 不绑权；DSpark 引用同一 LM Head，统计一次')
p('finalnorm','Final RMSNorm γ',(H,),'Transformer','norm','最终四路合并后缩放')
HASH_COLS=(C['engram_max_ngram_size']-1)*C['engram_n_heads'];ED=C['engram_head_dim'];EW=HASH_COLS*ED
for i,n in zip(C['engram_layer_ids'],C['engram_num_embeddings']):p('etable'+str(i),'Engram E · layer '+str(i),(n,ED),'ParallelEngramEmbedding','weight','根据 N-gram 地址查表',sharing='同一表可重复取行；layer 1 和 layer 14 是两张不同表')
lin('ew','W_E',EW,(HC+1)*H,'Engram','wkv','产生每路 Key 和一个共享 Value')
p('eq','Engram q_weight',(HC,H),'Engram','q_weight','调整残差侧匹配权重')
p('ek','Engram k_weight',(HC,H),'Engram','k_weight','调整记忆侧匹配权重')
lin('dmain','DSpark main_proj',H*len(C['dspark_target_layer_ids']),H,'DSparkBlock','main_proj','仅第一个草稿阶段投影主干特征')
p('dmainnorm','DSpark main_norm γ',(H,),'DSparkBlock','main_norm','首阶段主干投影归一化')
p('dfinalnorm','DSpark final norm γ',(H,),'DSparkBlock','norm','仅最后草稿阶段最终归一化')
MARKOV=C['dspark_markov_rank']
p('markovembed','Markov Embedding',(V,MARKOV),'DSparkMarkovHead','embed','草稿 Markov 输入查表')
lin('markovhead','Markov Head',MARKOV,V,'DSparkMarkovHead','head','草稿 Markov 打分')
lin('confidence','Confidence projection',H+MARKOV,1,'DSparkConfidenceHead','proj','计算草稿置信度')
VD=C['vision_dim'];VI=C['vision_inter_dim']
lin('patch','Vision patch projection',3*C['vision_patch_size']**2,VD,'PatchEmbed','proj','图像 patch 映射',file='vision.py')
p('patchbias','Vision patch bias',(VD,),'PatchEmbed','proj','patch 投影偏置',file='vision.py')
lin('vqkv','Vision W_QKV',VD,3*VD,'Attention','wqkv','图像 Query / Key / Value',file='vision.py')
p('vqkvbias','Vision QKV bias',(3*VD,),'Attention','wqkv','QKV 输出偏置',file='vision.py')
lin('vo','Vision W_O',VD,VD,'Attention','wo','图像 Attention 输出',file='vision.py')
p('vobias','Vision output bias',(VD,),'Attention','wo','图像输出偏置',file='vision.py')
lin('vup','Vision Gate + Up',VD,2*VI,'MLP','w1','图像两条激活分支',file='vision.py')
lin('vdown','Vision Down',VI,VD,'MLP','w2','图像 MLP 输出',file='vision.py')
p('vnorm','Vision RMSNorm γ',(VD,),'RMSNorm','weight','图像层归一化',file='vision.py')
lin('align1','Aligner W₁',VD*C['vision_downsample_ratio']**2,H,'Aligner','w1','下采样图像映射到文本宽度',file='vision.py')
p('align1bias','Aligner bias₁',(H,),'Aligner','w1','第一层对齐偏置',file='vision.py')
lin('align2','Aligner W₂',H,H,'Aligner','w2','第二层对齐',file='vision.py')
p('align2bias','Aligner bias₂',(H,),'Aligner','w2','第二层对齐偏置',file='vision.py')
p('image_delimiter','Image delimiter embedding',(H,),'Transformer','image_start','图像 start / end / newline 三种边界向量')

LEDGER=[]
def add(path,pid,scope,multiplicity=1):
    assert not any(r['path']==path for r in LEDGER),path
    LEDGER.append(dict(path=path,parameterId=pid,scope=scope,multiplicity=multiplicity,count=REGISTRY[pid]['count']*multiplicity))
def block(i,draft=False):
    prefix=('mtp.'+str(i-C['n_layers'])) if draft else 'layers.'+str(i)
    scope='Draft' if draft else 'Text Backbone'
    for name,pid in [('wq_a','qa'),('q_norm','qnorm'),('wq_b','qb'),('wkv','kv'),('kv_norm','kvnorm'),('wo_a','oa'),('wo_b','ob'),('attn_sink','sink')]:add(prefix+'.attn.'+name,pid,scope)
    for pid in ['anorm','fnorm']:add(prefix+'.'+pid,pid,scope)
    for component in ['attn','ffn']:
        for pid in ['hc_fn','hc_base','hc_scale']:add(prefix+'.'+component+'.'+pid,pid,scope)
    suffix='_draft' if draft else ''
    for pid in ['router','bias','bias_vl']:add(prefix+'.ffn.gate.'+pid,pid+suffix,scope)
    e=C['dspark_n_routed_experts'] if draft else C['n_routed_experts']
    for pid in ['w1','w2','w3']:
        add(prefix+'.ffn.experts[*].'+pid,pid,scope,e)
        add(prefix+'.ffn.shared_experts.'+pid,pid,scope)
    if not draft and i in C['kv_source_layers']:
        for pid in ['ckv','cnorm']+(['cgate'] if C['compress_ratios'][i]>1 else []):add(prefix+'.attn.compressor.'+pid,pid,scope)
    if not draft and i in C['index_source_layers']:
        for pid in ['iq','ih']+(['ik','inorm'] if i in C['kv_source_layers'] else []):add(prefix+'.attn.indexer.'+pid,pid,scope)
    if not draft and i in C['engram_layer_ids']:
        for pid in ['etable'+str(i),'ew','eq','ek']:add(prefix+'.engram.'+pid,pid,'Engram')
for i in range(C['n_layers']):block(i)
for path,pid in [('embed','embedding'),('norm','finalnorm'),('head','head')]:add(path,pid,'Text Backbone')
for i in range(C['n_layers'],C['n_layers']+C['n_mtp_layers']):block(i,True)
for path,pid in [('mtp.0.main_proj','dmain'),('mtp.0.main_norm','dmainnorm'),('mtp.2.norm','dfinalnorm'),('mtp.2.markov_head.embed','markovembed'),('mtp.2.markov_head.head','markovhead'),('mtp.2.confidence_head.proj','confidence')]:add(path,pid,'Draft')
for pid in ['patch','patchbias']:add('vision.patch_embed.'+pid,pid,'Vision')
for i in range(C['vision_n_layers']):
    for pid in ['vqkv','vqkvbias','vo','vobias','vup','vdown']:add('vision.blocks.'+str(i)+'.'+pid,pid,'Vision')
    for norm in ['norm1','norm2']:add('vision.blocks.'+str(i)+'.'+norm,'vnorm','Vision')
add('vision.norm','vnorm','Vision')
for pid in ['align1','align1bias','align2','align2bias']:add('aligner.'+pid,pid,'Vision')
for name in ['start','end','newline']:add('image_'+name,'image_delimiter','Vision')
TOTALS={s:sum(r['count'] for r in LEDGER if r['scope']==s) for s in ['Text Backbone','Engram','Draft','Vision']}
TOTALS['Text + Engram']=TOTALS['Text Backbone']+TOTALS['Engram']
TOTALS['All logical weights']=sum(TOTALS[s] for s in ['Text Backbone','Engram','Draft','Vision'])

def linear(k,n,m=1):
    return dict(input=[m,k],weight=[k,n],output=[m,n],parameters=k*n,forwardApprox=2*m*k*n,forwardExact=m*n*(2*k-1),dxShape=[m,k],dwShape=[k,n],dxApprox=2*m*n*k,dwApprox=2*k*m*n,dxExact=m*k*(2*n-1),dwExact=k*n*(2*m-1))
def norm(d):return 4*d+2 # square d + reduction d-1 + /d + eps + rsqrt + two d-wide multiplies
def rope(heads=1):return heads*(C['rope_head_dim']//2)*6
def attn(selected):
    # Independently count the actual tiled online-softmax kernel; comparisons excluded.
    tile=64;t=(selected+tile-1)//tile
    mat=4*A*D*tile*t
    # Per tile: score scaling, max-difference, score subtraction, 63-add reduction,
    # denominator multiply AND add, accumulator rescaling. Tail: sink subtract/add.
    ordinary=A*t*(tile+1+tile+(tile-1)+2+D)+2*A
    special=A*((tile+1)*t+(D+1))
    return dict(matrixApprox=4*A*D*selected,tiledMatrix=mat,modeled=mat+ordinary+special,selected=selected,tiles=t)
def hc():
    c=HC;n=c*H;it=C['hc_sinkhorn_iters']
    projection=2*n*MIX
    flat_norm=2*n+2 # squares, sum, mean, eps, rsqrt (included as 1)
    rescale=MIX
    split=6*c+2*c*c+(3*c*c-c)+(2*it-1)*(2*c*c-c)+2*c+2*c*c+(2*it-1)*c*c
    collapse=(2*c-1)*H
    post=(2*c*c+c)*H
    return projection+flat_norm+rescale+split+collapse+post
def router():
    e=C['n_routed_experts'];t=C['n_activated_experts']
    # output / gate_temp, softplus, sqrt, selection-bias add, normalize selected, scale
    return 2*H*e+e*4+3*t
def expert(routed=False):return 6*H*F+F*3+(F if routed else 0)
def engram():
    matrix=2*EW*(HC+1)*H
    # q*k weights, two RMS statistics per copy, weighted dot, scalar gate, inject.
    # Scalar 3: product of the two rstd statistics, then two score scales.
    # Scalar 2: sqrt and sigmoid; abs/copysign/clamp are excluded bit/comparison ops.
    gates=HC*(H+(4*H+4)+(3*H-1)+3+2)
    return matrix+HASH_COLS*ED+gates+2*HC*H
def indexer(layer,prior=1024):
    count=(prior+1)//C['compress_ratios'][layer]
    new=layer in C['kv_source_layers'] and (prior+1)%C['compress_ratios'][layer]==0
    proj=2*R*IQ+2*H*IH
    qrope=rope(IH);scale=IH
    dots=2*IH*ID*count;combine=(2*IH-1)*count
    key=(2*D*ID+norm(ID)+rope()) if new else 0
    return proj+qrope+scale+dots+combine+key
def compressor(layer,prior=1024):
    ratio=C['compress_ratios'][layer]
    proj=2*H*D*(2 if ratio>1 else 1)
    emit=(prior+1)%ratio==0
    pool=D*(3*ratio+1+2*ratio-1) if ratio>1 and emit else 0
    return proj+((norm(D)+rope()+pool) if emit else 0)
def decode_layer(i,prior=1024):
    positions=C['window_size']+(min(C['index_topk'],(prior+1)//C['compress_ratios'][i]) if C['compress_ratios'][i] else 0)
    base=2*H*R+2*R*Q+norm(R)+2*H*D+norm(D)+rope(A)*2+rope()+2*REGISTRY['oa']['count']+2*REGISTRY['ob']['count']+2*norm(H)+2*hc()+router()+C['n_activated_experts']*expert(True)+expert(False)+(C['n_activated_experts']+1)*H+attn(positions)['modeled']
    return base+(compressor(i,prior) if i in C['kv_source_layers'] else 0)+(indexer(i,prior) if i in C['index_source_layers'] else 0)
DECODE={i:decode_layer(i) for i in range(C['n_layers'])}
FORWARD=sum(DECODE.values())+2*engram()+(2*HC-1)*H+norm(H)+2*H*V

def validate_sources():
    mapping={'dim':'hidden_size','n_layers':'num_hidden_layers','moe_inter_dim':'moe_intermediate_size','n_heads':'num_attention_heads','rope_head_dim':'qk_rope_head_dim','window_size':'sliding_window','n_activated_experts':'num_experts_per_tok','kv_source_layers':'kv_source_layer_ids','index_source_layers':'index_source_layer_ids','n_mtp_layers':'num_nextn_predict_layers','candidate_source_layer':'candidate_source_layer_id','dspark_n_activated_experts':'dspark_num_experts_per_tok'}
    text=OFFICIAL['text_config']
    for key in ['vocab_size','q_lora_rank','head_dim','o_groups','o_lora_rank','n_routed_experts','n_shared_experts','compress_ratios','index_n_heads','index_head_dim','index_topk','hc_mult','hc_sinkhorn_iters','engram_layer_ids','engram_num_embeddings','engram_n_heads','engram_head_dim','dspark_n_routed_experts','dspark_target_layer_ids','dspark_markov_rank']:mapping[key]=key
    for a,b in mapping.items():assert C[a]==text[b],(a,C[a],text[b])
    assert text['tie_word_embeddings'] is False
    for record in MANIFEST['files']:assert hashlib.sha256((SOURCES/record['path']).read_bytes()).hexdigest()==record['sha256'],record['path']
    # Check actual assignment and constructor expressions, not just arithmetic shapes.
    source=ast.unparse(TREES['model.py'])
    for expr in ['self.wq_a = Linear(self.dim, self.q_lora_rank)','self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)','self.wkv = Linear(self.dim, self.head_dim)','self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))','self.bias_vl = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32)) if args.vision_enabled else None','self.mtp[-1].embed = self.embed','self.mtp[-1].head = self.head','self.w1 = Linear(dim, inter_dim, dtype=dtype)','self.w2 = Linear(inter_dim, dim, dtype=dtype)','self.w3 = Linear(dim, inter_dim, dtype=dtype)']:assert expr in source,expr
    vsource=ast.unparse(TREES['vision.py'])
    assert 'self.proj = nn.Linear(3 * args.vision_patch_size ** 2, args.vision_dim)' in vsource
    assert 'self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)' in vsource
    return len(mapping)

def main():
    n=validate_sources()
    report=dict(revision=REV,configFieldsChecked=n,totals=TOTALS,parameterFamilies=len(REGISTRY),uniqueLedgerEntries=len(LEDGER),modeledDecode=FORWARD,parameterRegistry=REGISTRY,ledger=LEDGER,linear={k:linear(*args) for k,args in {'kv':(H,D),'qa':(H,R),'qb':(R,Q),'head':(H,V),'expert_up':(H,F),'expert_down':(F,H)}.items()},nonlinearEstimates=dict(hc=hc(),router=router(),engram=engram()),decodeByLayer=DECODE)
    (HERE/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['revision','configFieldsChecked','totals','parameterFamilies','uniqueLedgerEntries','modeledDecode','nonlinearEstimates']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
