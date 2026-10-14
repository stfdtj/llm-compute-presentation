"""Attach source-linked parameter bindings, assumptions, and audit results.

Independent numbers come exclusively from audit_model. The teaching mapping is
only compared/updated here; it is never used to establish source correctness.
"""
from collections import Counter
from copy import deepcopy
import json
import math
import re
from pathlib import Path
import audit_model as model

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
P=model.REGISTRY
C=model.C
def fmt(n):return f'{n:,}'
def binding(pid,multiple=1,kind='',owned=True):
    return dict(parameterId=pid,multiplicity=multiple,kind=kind,owned=owned)
def expand(spec):
    out=[]
    for n,b in enumerate(spec):
        r=deepcopy(P[b['parameterId']]);r.update(b);r['bindingId']=b['parameterId']+'-'+str(n)
        r['total']=r['count']*r['multiplicity'];out.append(r)
    return out
COMMON_ATTN=['qa','qnorm','qb','kv','kvnorm','oa','ob','sink']
COMMON_BLOCK=COMMON_ATTN+['anorm','fnorm']
def layer_bindings(i):
    spec=[binding(p) for p in COMMON_BLOCK]
    spec+=[binding(p,2,'Attention 与 FFN 各一套') for p in ['hc_fn','hc_base','hc_scale']]
    spec+=[binding(p) for p in ['router','bias','bias_vl']]
    spec+=[binding(p,C['n_routed_experts'],'Routed Expert') for p in ['w1','w2','w3']]
    spec+=[binding(p,1,'Shared Expert') for p in ['w1','w2','w3']]
    if i in C['kv_source_layers']:spec+=[binding(p) for p in ['ckv','cnorm']+(['cgate'] if C['compress_ratios'][i]>1 else [])]
    if i in C['index_source_layers']:spec+=[binding(p) for p in ['iq','ih']+(['ik','inorm'] if i in C['kv_source_layers'] else [])]
    if i in C['engram_layer_ids']:spec+=[binding(p) for p in ['etable'+str(i),'ew','eq','ek']]
    return spec
def owner_sum(layers,include_engram=True):
    return sum(r['count'] for r in model.LEDGER if any(r['path'].startswith(f'layers.{i}.') for i in layers) and (include_engram or r['scope']!='Engram'))

SPECS={k:[binding(k)] for k in ['qa','qb','qnorm','kv','kvnorm','anorm','fnorm','finalnorm','head','embedding']}
SPECS.update(query=[binding(p) for p in ['qa','qnorm','qb']],output=[binding(p) for p in ['oa','ob']],global_=[binding(p) for p in ['ckv','cnorm']],compressor=[binding(p) for p in ['ckv','cgate','cnorm']],indexer=[binding(p) for p in ['iq','ih','ik','inorm']],hc=[binding(p) for p in ['hc_fn','hc_base','hc_scale']],router=[binding(p) for p in ['router','bias','bias_vl']],expert=[binding(p,1,'单个 Routed Expert') for p in ['w1','w3','w2']],shared=[binding(p,1,'单个 Shared Expert') for p in ['w1','w3','w2']],engembed=[binding('etable1')],enggate=[binding(p) for p in ['ew','eq','ek']],moe=[binding(p) for p in ['router','bias','bias_vl']]+[binding(p,384,'Routed Expert') for p in ['w1','w3','w2']]+[binding(p,1,'Shared Expert') for p in ['w1','w3','w2']])
SPECS['global']=SPECS.pop('global_')
for k in ['sliding','sparse','multi']:SPECS[k]=[binding('sink')]
for k in ['engram','engram1','engram14']:SPECS[k]=[binding(p) for p in ['etable14' if k=='engram14' else 'etable1','ew','eq','ek']]
for k,i in [('encoder',0),('decoder',20),('encoderblock',0),('decoderblock',20)]:SPECS[k]=layer_bindings(i)
for i in [0,1,2,13,14,19,20,21,22,33,34,39]:SPECS['layer'+str(i)]=layer_bindings(i)
for k,i in [('encoderattn',0),('decoderattn',20)]:SPECS[k]=[b for b in layer_bindings(i) if b['parameterId'] in COMMON_ATTN+['ckv','cgate','cnorm','iq','ih','ik','inorm']]
SPECS['selection']=[binding('bias',owned=False)]
SPECS['logits']=[binding('head',owned=False)]
SPECS['hidden']=[binding('anorm',owned=False)]
SPECS['localstates']=[binding(p,owned=False) for p in ['kv','kvnorm']]
SPECS['overview']=[binding(pid,sum(r['multiplicity'] for r in model.LEDGER if r['parameterId']==pid),'去重后的全部实例') for pid in P]

ASSUMPTIONS={
 'Configuration':[
 ['Model','DeepSeek-V4.1-Flash；正式 config 与 inference/config 32 项逐项对应'],
 ['Hidden / Layers','5120；文本主干 40 层，CED 教学分段 0–19 / 20–39'],
 ['Query / KV','64 个 Query Head × 512；KV 一份 512 维向量'],
 ['MoE','每层存 384 个路由专家 + 1 个共享专家；当前选择 6 个路由专家'],
 ['KV / Index source','KV 源层 2 / 8 / 14 / 20；Indexer 层 2 / 8 / 14 / 20 / 24 / 28 / 32 / 36'],
 ['窗口 / 选址上限','Local 窗口 128；Indexer 最多选择 512 个 Global 位置'],
 ['绑定与偏置','Token Embedding 与 LM Head 不绑权；文本 Linear 无加性 Bias；Router 选择偏置及 Vision 的真实 Bias 单独计数'],
 ['额外分支','3 个 DSpark 草稿阶段；32 层 Vision + Aligner；主图没有新增这些模块']
 ],
 'Scenario':[
 ['当前 token','Batch Size 1；一次 Decode 更新 1 个 token，不是 Prefill 整段输入'],
 ['历史','演示固定此前 1024 个 token；当前为第 1025 个。128 / 512 是此场景实际读取数，不能与配置上限混为一谈'],
 ['主计算范围','文本 40 层 + 两次 Engram + 最终 LM Head；不执行 Vision 或 DSpark'],
 ['缓存','历史 KV 已存在；本次新的 Local KV 每层生成，Global KV 只在源层生成。ratio=2 当前半组等待，ratio=1 当前产生一行'],
 ['反向','实数数学的局部梯度示例；不代表整批训练、历史缓存的完整反向、优化器更新或官方训练内核'],
 ['共享','重用一个权重或缓存不会增加存储参数；草稿引用的主 Embedding / LM Head 统计一次']
 ],
 'Approximation':[
 ['矩阵 FLOPs','主结果采用 2MKN 近似：乘、加各 1 次；单 token 的 M=1。展开审计另列精确乘加 MN(2K−1)'],
 ['线性反向','dX 与 dW 分别计算；近似分别 2MNK 和 2KMN。精确分别 MK(2N−1) 和 KN(2M−1)，因此单 token 外积不应称作精确 2KN'],
 ['非矩阵操作','加、减、乘各 1 次；rsqrt、exp、sigmoid、softplus、sqrt、除法各简化计 1 次；比较、Top-K、索引、abs/copysign 的符号处理和类型转换不计为浮点 FLOPs'],
 ['Attention','矩阵小计按实际读取位置；在线 Softmax 的展开估计按源码 64 位置分块，含 sink。它不是 GPU 实测指令数'],
 ['量化与内存','仅显式计 Engram 查表反量化的逐项乘法；其他 FP4/FP8 量化、GEMM 内部反量化、缓存搬运、通信和分词耗时不计'],
 ['参数与字节','统计逻辑权重数字；FP4 两值打包、FP8/FP4 scale、缓存和整数哈希 buffer 不无条件当作新增数学学习权重，字节数另算']
 ],
 'Unverified':[
 ['完整反向','参考实现为推理，不含训练反向；非线性总量、量化权重训练规则、选择偏置的更新方式未验证'],
 ['训练状态','技术报告 §2.5 说明预训练早期冻结 Vision 主体，保留最终归一化与对齐层可训练；本页面仍不能确定用户当前任务的 requires_grad 或量化权重更新规则'],
 ['完整 Prefill / Draft / Vision FLOPs','未运行完整图片、Prefill、DSpark 调度，当前未给出这些场景的精确 FLOPs'],
 ['本地差异','本地 model.py 落后于固定官方 revision：共享索引缓存发布条件、输出投影前量化有差异。参数维度不变；证据以官方固定版本为准']
 ]
}

GROUPS={
 'attention':['sliding','sparse','multi','encoderattn','decoderattn'],
 'moe':['moe','router','expert','shared','selection','weights','swiglu','combine'],
 'engram':['engram','engram1','engram14','engembed','enggate','hash','ngram','injection']
}
OPERATIONS={
 'query':('Attention','forward'),'qa':('Attention','forward'),'qb':('Attention','forward'),'kv':('Attention','_window_kv'),'output':('Attention','forward'),'global':('Attention','_compress_kv'),'compressor':('Compressor','forward'),'indexer':('Indexer','forward'),'hc':('Block','hc_mixes'),'router':('Gate','forward'),'selection':('Gate','forward'),'weights':('Gate','forward'),'expert':('Expert','forward'),'shared':('Expert','forward'),'moe':('MoE','forward'),'combine':('MoE','forward'),'hidden':('Block','forward'),'localstates':('Attention','_window_kv'),'head':('ParallelHead','forward'),'logits':('ParallelHead','forward'),'embedding':('ParallelEmbedding','forward')
}

def operation_evidence(key):
    if key in GROUPS['attention']:return [model.evidence('sparse_attn_kernel',file='kernel.py'),model.evidence('Attention','forward')]
    if key in GROUPS['engram']:return [model.evidence('Engram','forward')]
    if key in ['qnorm','kvnorm','anorm','fnorm','finalnorm']:return [model.evidence('RMSNorm','forward')]
    if key=='hc':return [model.evidence('Block','hc_mixes'),model.evidence('Block','hc_post'),model.evidence('hc_split_sinkhorn_kernel',file='kernel.py')]
    if key in OPERATIONS:return [model.evidence(*OPERATIONS[key])]
    if key.startswith('layer') or key in ['encoder','decoder','encoderblock','decoderblock']:return [model.evidence('Block','forward')]
    return [model.evidence('Transformer','forward')]
def module_assumptions(key):
    data={k:[] for k in ASSUMPTIONS}
    data['Scenario']=[['计算行数','普通运算只处理当前 1 个 token；反向为局部理论示例']]
    data['Approximation']=[['FLOPs','线性 2MKN 近似；非矩阵特殊函数按全局简化计数；缓存与访存另计']]
    if key in GROUPS['attention']:
        n=128 if key in ['sliding','encoderattn'] else 640
        data['Configuration']=[['Head / Window / Top-K','64 / 128 / 至多 512；都是配置值'],['Learnable Sink','每层 64 个；只改变归一化分母']]
        data['Scenario']+=[['历史与读取','此前 1024 个 token；本例 '+str(n)+' 个位置，其中 '+('128 Local' if n==128 else '128 Local + 512 Global')],['边界','Attention 读取结果不包含 Query/KV/输出投影、Compressor 或 Indexer；完整 Attention 参数表会列实际拥有的权重']]
        data['Unverified']=[['反向','历史 KV 梯度及跨 Head/Query 的累加、sink 反传未包含在四个矩阵小计里']]
    if key in GROUPS['moe']:
        data['Configuration']=[['专家','384 routed，Top-6，1 shared；每专家三套 5120↔2304 矩阵'],['层与共享','主干 40 层各自有权重；Shared Expert 为所有 token 执行，不是跨层绑权']]
        data['Scenario']+=[['Activated','一个 token 用 6+1 个专家；展示的 I₁…I₆ 为符号，没有运行真实路由 ID'],['Router bias','文本 token 使用文本选择偏置；图像选择偏置存储但本次不激活']]
        data['Unverified']=[['Top-K 训练','离散选择不求普通梯度；如何学习选择偏置未由推理代码给出']]
    if key in ['compressor','global','indexer']:
        data['Configuration']=[['source','KV owners 2/8/14 为 ratio=2，20 为 ratio=1；后续层复用'],['Indexer','8 套 Query / Head mixing，仅 4 套拥有 Key 投影与归一化']]
        data['Scenario']+=[['本次位置','第 1025 个 token：ratio=2 等待，ratio=1 新增 KV；Indexer 主例为 source layer 20 的 1025 个候选'],['量化','已执行位置处理；普通量化和搬运没有算入所示 FLOPs']]
    if key in GROUPS['engram']:
        data['Configuration']=[['记忆','layer 1 / 14 两张表；24 次查表 × 256 维，投影 6144→25600'],['权重','q_weight、k_weight 为 [4,5120]；不是额外 Token 或普通 RMSNorm γ']]
        data['Scenario']+=[['输入','当前 token 和最多前三个历史编号；主例 layer 1，另一张表另列']]
        data['Approximation']+=[['反量化','仅本次 6144 个查表数值乘 scale 计入；整数哈希、查行不算矩阵 FLOPs']]
    if key=='hc':
        data['Configuration']=[['四路','hc_mult=4；24 个混合分数，base 24、scale 3，Sinkhorn 20 轮'],['次序','本次生成 pre 给下一子模块；当前输入用前一子模块的 pre_mix'],['方向','comb[i,j] 表示旧输入路 i 向新输出路 j；Y_j=Σ_i R_i C_ij+b_j U']]
    if not data['Configuration']:data['Configuration']=[['权重','参数表按源码实际定义的 shape 与拥有关系统计，公式采用 XW 行向量']]
    return data

ALIASES={'qa':['WQA'],'qb':['WQB'],'kv':['WKV'],'ckv':['WKV'],'qnorm':['gamma','γ'],'kvnorm':['gamma','γ'],'anorm':['gamma','γ'],'fnorm':['gamma','γ'],'finalnorm':['gamma','γ'],'cnorm':['gamma','γ','RMSNorm'],'inorm':['gamma','γ','RMSNorm'],'oa':['WOAg','WOA'],'ob':['WOB'],'router':['WR'],'bias':['b'],'bias_vl':['biasvl'],'w1':['W1'],'w3':['W3'],'w2':['W2'],'cgate':['Wgate'],'iq':['WI','WIQ'],'ih':['Wh','WIH'],'ik':['WIK'],'hc_fn':['WHC'],'hc_base':['base'],'hc_scale':['scale'],'ew':['WE'],'eq':['wq'],'ek':['wk'],'embedding':['E'],'etable1':['E'],'etable14':['E'],'head':['Whead'],'sink':['sink']}
def normal(s):return re.sub(r'[^a-zA-Z0-9γ]','',s.replace(r'\gamma','γ'))
def refs(step,params,key):
    raw=' '.join(step['formulas']);normalized=normal(raw)
    weights={'W'+normal(a or b) for a,b in re.findall(r'W_(?:\{([^}]+)\}|([a-zA-Z0-9]+))',raw)}
    matches=[]
    for p in params:
        aliases=ALIASES.get(p['parameterId'],[])
        if any((normal(a) in weights if a.startswith('W') else normal(a) in normalized) for a in aliases):matches.append(p['bindingId'])
    if key in ['query','qnorm','kvnorm','anorm','fnorm','finalnorm','hidden'] and 'RMSNorm' in raw:
        matches+=[p['bindingId'] for p in params if 'norm' in p['parameterId']]
    if key in ['global','compressor'] and 'RMSNorm' in raw:matches+=[p['bindingId'] for p in params if p['parameterId']=='cnorm']
    if key=='indexer' and 'RMSNorm' in raw:matches+=[p['bindingId'] for p in params if p['parameterId']=='inorm']
    if key=='moe':matches=[p['bindingId'] for p in params if p['parameterId'] in ['w1','w2','w3']]
    if key=='selection':matches=[p['bindingId'] for p in params]
    return list(dict.fromkeys(matches))

def number(value):
    m=re.search(r'\d[\d,]*',value)
    return int(m[0].replace(',','')) if m else None
def zero_or_missing(value):return bool(re.search('不能|未确定|未验证|尚不能|暂不能|取决于|另算|需再计|需要再计|无法|未并入',value))

def apply(DATA):
    changes=[]
    def change(key,mode,item,before,after,reason,status='Verified'):
        changes.append(dict(module=key,mode=mode,item=item,previous=before,corrected=after,status=status,previousStatus='Mismatch',reason=reason))
    old=json.loads((HERE/'before_parameter_audit.json').read_text(encoding='utf-8'))
    for name in ['parameter','forward']:
        index=DATA['indexer'][name]['steps']
        index[1]['formulas']=[r'w=(XW_h)/\sqrt{128\times32}']
        index[1]['symbols']+=[('√(128×32)','索引 Head 宽度和 Head 数的固定缩放因子')]
        index[2]['formulas']=[r'S_h=Q_{I,h}K_I^{\mathsf T}',r's_j=\sum_{h=1}^{32}w_h\max(S_{h,j},0)',r'\mathcal I=\operatorname{Top512}(s)']
        index[2]['symbols']+=[('max(S,0)','把负分数截为 0，再按 Head 权重汇合'),('ℐ','最多 512 个选中位置；整数编号不是参数')]
        index.insert(2,{'title':'源层有新 latent 时生成索引 Key','formulas':[r'K_I=\operatorname{RMSNorm}(C W_{IK})'],'dimensions':[r'[1,512]\times[512,128]=[1,128]'],'symbols':[('C','Compressor 输出的未做位置旋转的 latent'),('W_IK','源层拥有的 512×128 Key 投影'),('RMSNorm','使用 128 个可学习 γ，随后做尾部 RoPE 与量化'),('K_I','写入索引缓存的一行；后续索引层复用')],'note':'本例 layer 20 的 ratio=1，当前有一行；ratio=2 的半组步骤不执行此项。'})
        comp=DATA['compressor'][name]['steps']
        comp.append({'title':'完整分组产生后，合并并归一化','formulas':[r'C_j=\sum_{t=1}^{r}a_{t,j}Z_{t,j}',r'KV=\operatorname{RoPE}(\operatorname{RMSNorm}(C))'],'dimensions':[r'[r,512]\longrightarrow[1,512]'],'symbols':[('r','分组大小；ratio=2 为两个 token，ratio=1 为当前一行'),('a_t,j','W_gate 分数逐特征 Softmax 的比例；ratio=1 不需要它'),('C / KV','合并后的 latent / 位置旋转后的 KV'),('RMSNorm','使用 [512] 可学习 γ')],'note':'当前第 1025 个 token 的 ratio=2 仍在等待，所以该步骤当前不执行；凑齐分组才发生。'})
        hc=DATA['hc'][name]['steps']
        hc.insert(1,{'title':'base / scale 将分数转为混合比例','formulas':[r'a_i^{next}=\sigma(scale_0 L_i+base_i)+\epsilon',r'b_i=2\sigma(scale_1 L_{4+i}+base_{4+i})',r'C=\operatorname{Sinkhorn}_{20}(\operatorname{Softmax}_{row}(scale_2 L_{8:24}+base_{8:24})+\epsilon)'],'dimensions':[r'L:[1,24]\longrightarrow a^{next},b:[1,4],C:[1,4,4]'],'symbols':[('base / scale','24 个学习偏置 / 3 个学习缩放系数'),('σ / ε','sigmoid / 10⁻⁶ 稳定数'),('a_next','本次生成、交给下一子模块的 pre 比例'),('b / C','本次使用的 post / comb；与参数不同，是运行时比例'),('Softmax_row / Sinkhorn₂₀','先逐行 Softmax，加 ε，再交替列/行归一化；分母也加 ε')],'note':'比例都是当前 token 的中间数据；不是新增的 24 个可学习权重。'})
    change('indexer','forward','索引 Key 和 Head 汇合','只展示 Query×Key 点积','补齐 W_IK / γ、新 latent 的条件、ReLU 与加权求和','源码 Indexer.forward 明确执行这些阶段；选址与主 Attention 分别统计。','Verified')
    change('compressor','forward','归一化权重的使用阶段','γ 只列数量，流程未显示','补齐完整分组产生后才执行的归一化','半组 Decode 不执行该阶段；不能误认 γ 不学习或每步都执行。','Verified')
    change('hc','forward','base / scale 的明确用途','只在文字中列出 base / scale','补齐 pre / post / comb 的参数变换','核对 hc_split_sinkhorn 的缩放、偏置、epsilon 与先行后列次序。','Verified')
    for name in ['parameter','forward']:
        s=DATA['hc'][name]['steps'][1]
        s['formulas']=[r'a_i^{next}=\sigma(scale_0 L_i+base_i)+\epsilon',r'b_i=2\sigma(scale_1 L_{4+i}+base_{4+i})',r'C_0=\operatorname{Softmax}_{row}(scale_2 L_{8:24}+base_{8:24})+\epsilon',r'C=\operatorname{AltNorm}_{19}(\operatorname{ColNorm}_{\epsilon}(C_0))']
        s['symbols']+=[('i','索引从 0 到 3；L 的 0…3 / 4…7 / 8…23 三段分别供 pre / post / comb'),('ColNorm_ε / AltNorm₁₉','先除以列和加 ε；再重复 19 轮行和 / 列和归一化，分母各加 ε')]
        DATA['global'][name]['steps'][1]['formulas'].append(r'\operatorname{RMSNorm}(C)=C\odot r\odot\gamma')
        DATA['global'][name]['steps'][1]['symbols']+=[('γ','Compressor 的 512 个学习系数'),('r','当前 C 的均方开平方倒数，非参数')]
        DATA['indexer'][name]['steps'][2]['formulas'].append(r'\operatorname{RMSNorm}(U)=U\odot r\odot\gamma,\quad U=CW_{IK}')
        DATA['indexer'][name]['steps'][2]['symbols']+=[('γ / r','128 个学习缩放系数 / 当前 U 均方的开平方倒数')]
        DATA['compressor'][name]['steps'][-1]['formulas'].append(r'\operatorname{RMSNorm}(C)=C\odot r_C\odot\gamma')
        DATA['compressor'][name]['steps'][-1]['symbols']+=[('γ / r_C','512 个学习系数 / 从 C 均方计算的缩放因子')]
        DATA['localstates'][name]['steps'].append({'title':'本层 Local KV 归一化、位置旋转并缓存','formulas':[r'KV=\operatorname{RoPE}(KV_{new}\odot r\odot\gamma)'],'dimensions':[r'[1,512]\longrightarrow[1,512]'],'symbols':[('KV_new','当前层投影产生的新一行'),('γ','该层 Local KV 的 512 个学习系数；在 KVNorm 计数'),('r','由该行自身均方算出的缩放比例，非参数'),('KV','写入本层最多 128 位置的环形缓存')],'note':'这里只引用投影与归一化；不是额外创建一套参数，量化与搬运成本另计。'})
    for name in ['parameter','forward']:
        s=DATA['query'][name]['steps'][1]
        s['formulas']=[r'H^{\prime}=\operatorname{RMSNorm}(H)=H\odot r\odot\gamma']
        s['symbols']+=[('γ','1280 个可学习逐特征系数'),('r','从 H 自身均方计算的统一缩放因子；不是参数'),('⊙','对应位置相乘')]
    DATA['qnorm']['forward']['explain']='平方 d 次、求和 d−1 次、两次逐项缩放 2d 次；均值除法、加 ε、rsqrt 共 3 次，合计 4d+2。特殊函数简化记 1 次。'
    for k in ['kvnorm','anorm','fnorm','finalnorm']:
        DATA[k]['forward']['explain']=DATA['qnorm']['forward']['explain']
    change('qnorm','forward','RMSNorm 4d+2 的解释',old['qnorm']['forward']['explain'],DATA['qnorm']['forward']['explain'],'所有五类 RMSNorm 面板保留正确数值；说明独立列出 d−1 求和、均值除法、epsilon 与 rsqrt，避免口头漏数。','Estimated')
    for key in GROUPS['attention']:
        for name in ['parameter','forward']:
            s=DATA[key][name]['steps'][1]
            s['formulas']=[r'P_j=\frac{\exp(S_j)}{\sum_i\exp(S_i)+\exp(a_{sink})}',r'O=P\times V']
            s['symbols']+=[('a_sink','当前 Head 的一个可学习分母项；64 个 Head 各有一个'),('Σᵢ / exp','对实际读取位置求和 / 指数函数；sink 没有 Value'),('Pⱼ、Sⱼ','第 j 个位置的权重 / 前一步计算的分数')]
        change(key,'forward','明确 sink 的归一化计算','P 未写出归一化关系','分母包含 exp(a_sink)，不附加 Value 行','显式绑定可学习 sink，不误标为零参数。','Verified')
    for key in ['engram','engram1','engram14','enggate']:
        for name in ['parameter','forward']:
            gate={'title':'匹配记忆并计算每路门值','formulas':[r's_c=\frac{r_{R,c}r_{K,c}}{\sqrt{5120}}\sum_j R_{c,j}w_{q,c,j}w_{k,c,j}K_{c,j}',r'g_c=\sigma(\operatorname{sign}(s_c)\sqrt{\max(|s_c|,10^{-6})})'],'dimensions':[r'R,K,w_q,w_k:[1,4,5120]\longrightarrow g:[1,4]'],'symbols':[('R、K','当前 token 的四路残差 / 四路记忆 Key'),('w_q、w_k','两组 [4,5120] 可学习逐特征权重'),('r_R、r_K','各路残差和 Key 自身均方的开平方倒数；不带额外 γ'),('Σⱼ、s_c','对 5120 个特征求和 / 第 c 路匹配分数'),('σ、sign、max','sigmoid / 保留正负号 / 将绝对值下限设为 10⁻⁶'),('g_c','第 c 路是否接收记忆的门值')],'note':'归一化统计使用原残差和 Key；q_weight 与 k_weight 的乘积只作用在匹配点积。'}
            gate['dimensions']=[r'R,K:[1,4,5120];\quad w_q,w_k:[4,5120]\longrightarrow g:[1,4]']
            DATA[key][name]['steps'].insert(len(DATA[key][name]['steps'])-1,gate)
        change(key,'forward','Engram 门控权重','仅文字概括门值','补齐 q_weight / k_weight 的加权归一化点积与 signed sqrt','实际 Engram.forward 使用这两组学习权重；不是普通 RMSNorm γ。','Verified')
    for name in ['parameter','forward','backward']:
        DATA['hidden'][name]['steps']=[{'title':'当前状态合并成 Attention 输入','formulas':[r'Z=\operatorname{HC\_pre}(R,a)',r'X=\operatorname{RMSNorm}(Z)=Z\odot r\odot\gamma'],'dimensions':[r'[1,4,5120]\longrightarrow[1,5120]'],'symbols':[('R','进入 layer 20 的当前 token 四路残差'),('a','前一子模块产生的合并比例，是中间数据'),('HC_pre','把四路加权合成一行'),('RMSNorm / γ','使用 layer 20 Attention 输入归一化的 [5120] 系数'),('r / Z','从 Z 均方计算的统一比例 / 合并成的单行；都是中间数据'),('X','随后用于生成 Query / KV 的一行输入')],'note':'此标签是运行时状态视图；投影在 Global / Local KV 模块另算。'}]
    DATA['hidden']['purpose']='原图把 Encoder 末端状态概括为 [1,5120]。源码的跨层状态实际为四路，layer 20 进入 Attention 前先合并并归一化。'
    DATA['hidden']['backward']['rows']=[('状态梯度','通过合并与归一化回到四路；完整操作数未验证')]
    change('hidden','forward','状态与投影的区别','状态面板直接展示 KV 投影','状态合并、归一化；KV 分支独立统计','防止将 512 维 KV 当成 Encoder 隐藏状态。','Verified')
    for key,d in DATA.items():
        params=expand(SPECS.get(key,[]));d['learnable']=params;d['moduleAssumptions']=module_assumptions(key)
        d['ownedParameters']=sum(p['total'] for p in params if p['owned'])
        d['nonLearned']='运行时输入、激活、梯度与缓存不是权重；RoPE 角度、维度与 Top-K 数量为固定设置。'
        if not params:d['nonLearned']='本图示操作没有自己拥有的可学习参数；输入、缓存、整数 ID、重排和中间值不计为权重。'
        if key=='moe':
            routed=3*model.H*model.F
            d['moeSummary']={'total':d['ownedParameters'],'activated':7*routed+model.H*C['n_routed_experts']+C['n_routed_experts'],'expertsTotal':385*routed,'expertsActivated':7*routed,'router':model.H*C['n_routed_experts']+2*C['n_routed_experts'],'fortyTotal':d['ownedParameters']*40,'fortyActivated':(7*routed+model.H*C['n_routed_experts']+C['n_routed_experts'])*40}
            d['parameter']['rows']=[('一层 MoE · 全部逻辑权重',fmt(d['moeSummary']['total'])+' 个参数'),('当前 token · 激活权重',fmt(d['moeSummary']['activated'])+' 个参数')]
            d['parameter']['calculations']=[rf'(384+1)\times{routed}+5120\times384+2\times384={d["moeSummary"]["total"]}']
            d['parameter']['totals']=[('40 层 MoE 总量',fmt(d['moeSummary']['fortyTotal'])+' 个参数'),('40 层当前 token 激活',fmt(d['moeSummary']['fortyActivated'])+' 个参数')]
            d['parameter']['note']='此处包含 Router；Activated 包含当前文本选择偏置，图像选择偏置不参与本次路径。专家与 Router 的子视图是同一权重的引用，不另加一次。'
            change(key,'parameter','一层 MoE 边界',old[key]['parameter']['rows'],d['parameter']['rows'],'旧主结果只计专家；新增明确含 Router 的 Total / Activated，专家小计仍可在表中查看。')
        if key.startswith('layer') or key in ['encoder','decoder','encoderblock','decoderblock']:
            i=int(key[5:]) if key.startswith('layer') else 20 if key.startswith('decoder') else 0
            count=d['ownedParameters']
            d['parameter']['rows']=[(f'第 {i} 层实际拥有的权重',fmt(count)+' 个参数')]
            group=list(range(20,40)) if key=='decoder' else list(range(20)) if key=='encoder' else [i]
            d['parameter']['totals']=[('本组 '+str(len(group))+' 层含其 Engram（若有）',fmt(owner_sum(group))+' 个参数')]
            d['parameter']['note']='先列代表层的实际创建权重；有 Compressor / Indexer / Engram 的层数量不同。全局总账按唯一模型路径去重。'
            d['forward']['totals']=[('本组文本层 · 简化操作估计（Engram 另列）',fmt(sum(model.DECODE[j] for j in group))+' 次估算操作')]
            d['forward']['note']='固定此前 1024 个历史 token；只计该组文本层，Engram 和最终输出另算。估算不含一般量化、搬运或通信。'
            d['_audit']=dict(parameters=owner_sum(group,False),decode=sum(model.DECODE[j] for j in group),layers=group)
            if old[key]['parameter']['rows']!=d['parameter']['rows']:change(key,'parameter','层级统计边界',old[key]['parameter']['rows'],d['parameter']['rows'],'代表层按实际 owner 枚举，不能把普通层固定模板用于 source / Engram 层。')
        if key in ['encoderattn','decoderattn']:
            d['parameter']['rows']=[('代表层完整 Attention 权重',fmt(d['ownedParameters'])+' 个参数')]
            d['parameter']['totals']=[];d['parameter']['explain']='完整 Attention 包含投影、归一化和 sink；源码源层还拥有 Compressor 与 Indexer。前向主例仍只算读取 KV 的矩阵小计。'
            change(key,'parameter','完整 Attention 与读取子操作',old[key]['parameter']['rows'],d['parameter']['rows'],'旧标题指整个 Attention，统计却只显示 kernel 的 sink；现在明确完整参数、读取 FLOPs 的不同边界。')
        if key in ['sparse','multi','sliding','encoderattn','decoderattn']:
            selected=128 if key in ['sliding','encoderattn'] else 640
            count=2 if key in ['sliding','encoderattn'] else 20 if key in ['multi','decoderattn'] else 38
            total=model.attn(selected)['modeled']
            d['forward']['totals']=[('含在线归一化 / sink 的一层简化估计',fmt(total)+' 次操作'),(f'同类 {count} 层',fmt(total*count)+' 次操作')]
        if key=='overview':
            d['parameter']['totals']=[(name,fmt(n)+' 个逻辑参数') for name,n in model.TOTALS.items()]
            d['forward']['totals']=[('Text + Engram · 当前一步简化估计',fmt(model.FORWARD)+' 次操作')]
            d['forward']['note']='估计涵盖文本主干、两次 Engram 与最终词表打分；不包含 Vision、Draft 或完整量化/搬运。不是完整多模态执行总量。'
            change(key,'parameter','全模型范围',old[key]['parameter']['totals'],d['parameter']['totals'],'原 748,494,684,784 只为 Text + Engram；补齐 Draft、Vision、完整逻辑权重范围。')
            change(key,'forward','当前一步的统计范围',old[key]['forward']['totals'],d['forward']['totals'],'独立按 kernel 逐项核对，原操作总数一致；修正“全模型”边界：这里只是 Text + Engram，未执行 Vision / Draft。64 位块归约为 63 次加法，分母更新另有乘、加两次。','Estimated')
        if key=='query':change(key,'parameter','Query 含 RMSNorm',old[key]['parameter']['rows'],d['parameter']['rows'],'模块包含的 1,280 个 γ 原本只在子面板计数，主模块总量漏列；正确每层 48,497,920。')
        if key=='hc':
            change(key,'forward','comb 方向与 pre 使用次序',old[key]['forward']['steps'],d['forward']['steps'],'hc_post 对 comb 的输入路求和，正确 Σ_i R_i C_ij；本次 pre 交给下一子模块。','Verified')
            change(key,'backward','路间梯度与归一化路径',old[key]['backward']['steps'],d['backward']['steps'],'改为 dR_i=Σ_j C_ij dY_j；投影小计不代表全部 HC 梯度。','Estimated')
        for name in ['parameter','forward','backward']:
            m=d[name]
            status='Verified' if name=='parameter' else 'Estimated'
            if name=='backward' and any(zero_or_missing(str(v)) for _,v in m['rows']):status='Unverified'
            source=[p['evidence'] for p in params]+operation_evidence(key)
            if not source:source=[model.evidence('NgramHashState',file='engram.py') if key in ['ngram','hash'] else model.evidence('Transformer')]
            m['audit']={'status':status,'evidence':source,'assumptions':d['moduleAssumptions'],'basis':'配置、源码定义与独立逻辑权重清单' if name=='parameter' else '独立实数数学推导；参考实现仅提供推理，特殊操作采用明确简化计数','mode':name}
            m['resultAudits']=[]
            for label,value in m['rows']:
                row_status='Unverified' if zero_or_missing(value) or (name=='backward' and '完整' in label and not re.search(r'\d[\d,]* FLOPs',value)) else status
                if name=='backward' and re.search(r'\d[\d,]* FLOPs',value):row_status='Estimated'
                m['resultAudits'].append(dict(label=label,value=value,status=row_status))
            m['totalAudits']=[dict(label=l,value=v,status='Verified' if name=='parameter' else 'Estimated') for l,v in m['totals']]
            for s in m['steps']:
                s['parameterRefs']=refs(s,params,key)
                s['tensorKind']='gradient' if name=='backward' else 'activation'
            for extra in m['extras']:
                for s in extra['steps']:s['parameterRefs']=refs(s,params,key)
            # Single linear terms include exact arithmetic, never called measured FLOPs.
            terms={'kv':[(model.H,model.D)],'qa':[(model.H,model.R)],'qb':[(model.R,model.Q)],'head':[(model.H,model.V)],'query':[(model.H,model.R),(model.R,model.Q)],'output':[(model.GRP_IN,model.GRP_OUT,model.GROUP),(model.OUT_RANK,model.H)],'router':[(model.H,C['n_routed_experts'])],'expert':[(model.H,model.F),(model.H,model.F),(model.F,model.H)],'shared':[(model.H,model.F),(model.H,model.F),(model.F,model.H)],'enggate':[(model.EW,(model.HC+1)*model.H)],'global':[(model.H,model.D)],'hc':[(model.HC*model.H,model.MIX)]}.get(key,[])
            m['linearArithmetic']=[dict(**model.linear(t[0],t[1]),multiplicity=t[2] if len(t)>2 else 1) for t in terms]
    # All unknown complete backward totals stay marked Unverified.
    for key in ['qnorm','kvnorm','anorm','fnorm','finalnorm','compressor','global','indexer','weights','swiglu','engembed']:
        DATA[key]['backward']['audit']['status']='Unverified'
    status_counts=Counter(m['audit']['status'] for d in DATA.values() for m in [d['parameter'],d['forward'],d['backward']])
    summary={s:status_counts.get(s,0) for s in ['Verified','Estimated','Unverified','Mismatch']}
    summary['historicalMismatchesCorrected']=len(changes)
    for c in changes:c['evidence']=operation_evidence(c['module'])+DATA[c['module']][c['mode']]['audit']['evidence'][:1]
    report={'summary':summary,'changes':changes,'records':[dict(module=k,mode=n,status=d[n]['audit']['status']) for k,d in DATA.items() for n in ['parameter','forward','backward']],'totals':model.TOTALS,'revision':model.REV,'forwardModeled':model.FORWARD}
    (HERE/'audit_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'registry':P,'assumptions':ASSUMPTIONS,'summary':summary,'totals':model.TOTALS,'revision':model.REV,'changes':changes,'officialSources':model.MANIFEST}

def write_docs(AUDIT,DATA):
    lines=['# 计算正确性审计','',f'官方固定版本：`{model.REV}`。先按配置和实际构造权重建立独立总账，再对比旧页面。未用旧 HTML 数值证明实现正确。','',f'状态统计：{json.dumps(AUDIT["summary"],ensure_ascii=False)}。统计单位为 63 个页面模块 × 3 种模式，共 189 个主记录；参数家族 57 个另附 Verified 源码证据。Mismatch 为修正历史，当前已解决项使用修正后状态。','', '## 已发现并修正的问题','','| Module | Item | Previous | Corrected | Status | Evidence |','|---|---|---|---|---|---|']
    for c in AUDIT['changes']:
        prev=str(c['previous']);new=str(c['corrected'])
        if len(prev)>170:prev='旧版公式/流程（完整内容见 audit_results.json）'
        if len(new)>210:new='修正后的公式/范围（完整内容见 audit_results.json）'
        e=c['evidence'][0]
        lines.append(f'| {c["module"]} | {c["item"]} | {prev.replace(chr(10)," ").replace("|","/")} | {new.replace(chr(10)," ").replace("|","/")} | Mismatch → {c["status"]} | {c["reason"]}<br>[{e["file"]}:{e["line"]} · {e["symbol"]}]({e["url"]}) |')
    lines+=['','## 所有页面模式记录','','| Module | Parameters | Forward | Backward | Scope / Evidence |','|---|---|---|---|---|']
    for k,d in DATA.items():
        e=d['parameter']['audit']['evidence'][0]
        lines.append(f'| {k} | {d["parameter"]["audit"]["status"]} | {d["forward"]["audit"]["status"]} | {d["backward"]["audit"]["status"]} | [{e["symbol"]}]({e["url"]})；参数拥有关系与 FLOPs 子操作边界见页面 |')
    lines+=['','## 逻辑权重总账','','| 范围 | 权重数 |','|---|---:|']+[f'| {k} | {fmt(v)} |' for k,v in model.TOTALS.items()]
    lines+=['','数量与官方约 552B 文本 Backbone、约 196B Engram、模型文件约 763B 的分范围说明一致。此处统计逻辑权重，不是量化字节、scale 元数据或给定训练任务的 requires_grad 数量。','', 'DSpark 的主 Embedding / LM Head 是引用；唯一总账只计一次。三个草稿阶段使用 128 routed / Top-3；首阶段 main_proj、末阶段 norm、Markov 表/头和 confidence 单独枚举。Vision 从官方 vision.py 核对，包含默认 nn.Linear Bias、32 个 Block、PatchEmbed、最终 RMSNorm、Aligner 和三个图像边界向量。','', '## 未验证与估算边界','','所有数值前向 FLOPs 为明确口径下的 Estimated。参考代码未给训练反向，线性 dX / dW 与已复核非线性梯度是数学推导；完整 HC、RMSNorm、Compressor、Indexer、Engram、Attention、MoE 等反向总量保留 Unverified。没有编造全模型训练反向。','', '单 token 的 dW 是外积：精确 KN，常用 2KN 只为乘加近似。一般线性精确 MN(2K−1)，不是把近似乘 2 当作精确反向。','', '特殊操作估计把 rsqrt、exp、sigmoid、softplus、sqrt、除法各记 1 次，忽略比较、Top-K、整数运算、搬运和通信；仅明确计入 Engram 查表反量化的逐项乘法。量化 GEMM、图片或草稿完整执行耗时未验证。','', '本地 model.py 与官方固定版本有两类差异：Indexer 在没有新 latent 的步骤仍应发布自身索引缓存；Attention / DSpark 的分组输出投影前还执行一次 act_quant。逻辑参数维度不变，本审计以固定官方实现为证据，未改动用户模型源码。','', '## 可重复验证','','运行 `python src/deepseek/audit_model.py` 和 `python src/deepseek/verify_audit.py`。独立总账在 `src/deepseek/independent_audit.json`；源码下载校验和及固定 URL 在 `references/deepseek/manifest.json`；完整修正清单在 `src/deepseek/audit_results.json`。','', '[正式配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/'+model.REV+'/config.json) · [官方推理实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/'+model.REV+'/inference) · [模型说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/'+model.REV+'/README.md)']
    lines+=['','## 技术报告交叉核验','','同 revision 的 51 页技术报告已保存 PDF、提取文本和 SHA256。§2.2 的 CED 与 §2.3 的 CSA2 共同说明分段与跨层 KV / 选址复用；§4.2.1 的正式维度与专家配置和已核对的 config / 构造源码一致。CED 的通用层相关投影表达，与最终 CSA2 部署中的 Full / Reindex / Reuse 配置应分别理解；共享的权重创建以实际源码和 source layer 配置枚举。','', '§2.5 说明预训练早期 Vision 主体冻结，最终归一化与对齐层保持可训练；学习率衰减阶段解冻。其余关于用户当前任务 requires_grad、完整训练反向内核和量化更新方式的未知项仍保留 Unverified。报告的 8B / 16B 为其激活口径，本页 MoE Activated 是单 token 6+1 个专家与文本 Router 的明确子范围，不能互相替代。','', '[技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/'+model.REV+'/DeepSeek_V41_Tech_Report.pdf)；本地可核验快照 `references/deepseek/technical_report_manifest.json`。']
    (ROOT/'docs/architecture/calculation_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    alines=['# Assumptions','', '所有普通计算以当前一个 token 为主例。Configuration 是模型固有设置，Scenario 是演示场景，Approximation 是计数约定，Unverified 为当前证据边界。','']
    for cat,rows in ASSUMPTIONS.items():
        alines+=['## '+cat,'','| 项目 | 条件 |','|---|---|']+[f'| {a} | {b} |' for a,b in rows]+['']
    for key in ['query','kv','multi','compressor','indexer','hc','moe','engram','head']:
        alines+=['## '+key,'']
        for cat,rows in DATA[key]['moduleAssumptions'].items():
            if rows:alines+=['### '+cat,'']+[f'- {a}：{b}' for a,b in rows]+['']
    (ROOT/'docs/architecture/assumptions.md').write_text('\n'.join(alines)+'\n',encoding='utf-8')

if __name__=='__main__':
    from teaching_data import DATA
    a=apply(DATA);write_docs(a,DATA);print(json.dumps(a['summary'],ensure_ascii=False,indent=2))
