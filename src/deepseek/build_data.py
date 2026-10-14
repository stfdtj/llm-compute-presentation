"""Auditable calculation mapping. No diagram geometry is defined here."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M, H, R, A, D, F, V, C = 1024, 5120, 1280, 64, 512, 2304, 129280, 4
DATA = {}
P_SOURCE = '参数说明 §§1–15；config.json / model.py'
F_SOURCE = 'FLOPs_calculation_forward.txt §§18–22，Scenario P'
B_SOURCE = '由实数计算图与链式法则推导；未验证训练内核'
CONVENTION = 'B=1，T=1024，M=B×T。F_B 为基本乘加（GEMM=2MKN），F_E 为每项计 1 的特殊操作估算。量化、其他 GEMM padding、比较/Top-K、通信和内存成本另列，未计入而非为零；主 attention 按文档计入 64 槽 tile。'
UNKNOWN = '未验证 / 需要训练实现确认'
CED = '沿用原图的 Causal Encoder / Decoder 与 0–19 / 20–39 分区。源码未定义官方 CED 名称：实际为跨层共享压缩 KV；每层仍独立计算 local KV、Query 与输出投影。'
KV_NOTE = '原图写作 T×64×d（或 2×64×d）；源码的共享 local KV 实为 [B,T,512]，64 个 Query heads 共享同一 KV，且 K/V 使用同一向量。原图标签保留，以下计算采用源码形状。'
HC_NOTE = '原图的 T×5120 简写不显示 residual copy 轴；真实残差为 [B,T,4,5120]。HC 不是单路残差加法；pre 系数由前一子层提供。'
GLOBAL_NOTE = '原图把 Encoder Final Hidden States → Global KV 画成独立阶段。源码对应 layer 20 对其 attention 输入生成 ratio-1 压缩 KV，并供 20–39 复用；不存在另加一套 40 层投影。不能把图示 H_enc 直接等同未经 HC/RMSNorm 的源码输入。'

def fmt(n): return f'{n:,}'
def result(n, unit='parameters'): return f'{fmt(n)} {unit}'
def mode(formulas=(), rows=(), note='', status='', source=''):
    return dict(formulas=list(formulas), rows=list(rows), note=note, status=status, source=source)
def p(formulas=(), rows=(), note=''): return mode(formulas,rows,note,source=P_SOURCE)
def f(formulas=(), rows=(), note=''): return mode(formulas,rows,note,status='Scenario P · 数学估算',source=F_SOURCE)
def b(formulas=(), rows=(), note='', verified=False): return mode(formulas,rows,note,status='解析推导 · 矩阵主项' if verified else UNKNOWN,source=B_SOURCE)
def zero(note=''): return p([r'P=0'], [('额外可学习参数','0')],note)
def add(key,title,purpose,shape,param,forward,backward,insight='',note='',related=(),scope=''):
    DATA[key]=dict(title=title,purpose=purpose,shape=shape,parameter=param,forward=forward,backward=backward,insight=insight,note=note,related=list(related),scope=scope)
def nums(basic,special=0):
    return [('基本 FLOPs · F_B',fmt(basic)),('特殊操作估计 · F_E',fmt(special)),('合计 · F_B + F_E',fmt(basic+special)+' ops')]

add('input','Input Text / 输入文本','把自然语言交给 tokenizer；图中 “It is raining” 仅作输入示例。','字符串 → token 序列',zero(),f([], [('浮点 FLOPs','不适用：文本预处理')]),b([], [('梯度','文本与整数 ID 不可微')]),'文字不是浮点张量。')
add('tokenizer','Tokenizer / 分词','将文字映射为词表索引。图中的逐词切分是示意，真实 tokenizer 文件未提供。','text → [B,T] 整数 IDs',zero(),f([], [('计数','整数/字符串处理，不列入浮点总量')]),b([], [('梯度','离散分词不求导')]),'实际 T 与 token ID 不能从单词数推断。')
add('ids','Token IDs / 词表索引','索引控制 embedding 和 N-gram lookup；不把 ID 数值当作连续特征。','[1,1024] integers',zero(),f([], [('浮点算术','0；索引和读取成本另列')]),b([], [('梯度','整数 IDs 无梯度；被读取的权重可获得梯度')]),'梯度流向表中取出的值，而不是整数地址。')
add('embedding','Token Embedding / 词嵌入','从词表取出 5120 维行向量，再初始化四路残差副本。','[1,1024] → [1,1024,5120] → [1,1024,4,5120]',p([r'E\in\mathbb{R}^{129280\times5120}',r'P=VH=129280\times5120'],[('实例','1 个独立输入词表'),('存储参数',result(V*H)),('单 token 读取','5,120 个权重值')]),f([r'X_{t,:}=E_{u_t,:}'], [('浮点算术','0；lookup/repeat 的内存与整数操作另列')]),b([r'dE_{v,:}=\sum_{t:u_t=v}dX_{t,:}',r'dX_{t,:}=\sum_{c=1}^{4}dR_{t,c,:}'], [('梯度形状','dR [M,4,5120] → dX [M,5120]；dE [129280,5120]'),('计算','散射累加；重复 ID 与实现影响成本')], '词表量化权重的训练规则未提供；此处是可训练实数 embedding 的推导。'), '存储整张词表，每个 token 只查询其中一行。')

def linear(key,title,K,N,n,purpose,shape=None,note=''):
    add(key,title,purpose,shape or f'[M,{K}] → [M,{N}]',p([rf'W\in\mathbb{{R}}^{{{K}\times{N}}}',rf'P={K}\times{N}'],[('单实例',result(K*N)),('实例数',str(n)),('合计',result(K*N*n))]),f([rf'F=2M\times{K}\times{N}\times{n}'],[('实例数',str(n))]+nums(2*M*K*N*n)),b([r'Y=XW,\quad dX=dY W^{\mathsf T},\quad dW=X^{\mathsf T}dY',rf'dX:\ [{M},{N}]\,[{N},{K}]\to[{M},{K}]',rf'dW:\ [{K},{M}]\,[{M},{N}]\to[{K},{N}]',rf'F_{{dX}}=F_{{dW}}=2M\times{K}\times{N}\times{n}'],[('dX FLOPs',fmt(2*M*K*N*n)),('dW FLOPs',fmt(2*M*K*N*n)),('矩阵反向合计',fmt(4*M*K*N*n)+' FLOPs')], 'dX 向前层传梯度，dW 更新权重的梯度；实数、稠密两侧均求导，量化训练规则未包含。',True), '参数只存一次；序列中的每一行都执行矩阵乘法。',note=note,scope=f'×{n}')

def norm(key,title,d,n):
    add(key,title,'按每个 token 的特征均方根归一化，并乘可学习的 γ。',f'[M,{d}] → [M,{d}]',p([rf'\gamma\in\mathbb{{R}}^{{{d}}},\quad P={d}\times{n}'],[('单实例',result(d)),('实例数',str(n)),('合计',result(d*n))]),f([r'y=\gamma\odot x\,r,\quad r=(\operatorname{mean}(x^2)+\epsilon)^{-1/2}',rf'F_B=4M\times{d}\times{n},\quad F_E=2M\times{n}'], nums(4*M*d*n,2*M*n)),b([r'g=dy\odot\gamma',r'dx=r g-\frac{r^3}{d}x\sum_j g_jx_j',r'd\gamma=\sum_{t=1}^{M}dy_t\odot x_t r_t'],[('梯度',f'dy、dx [M,{d}]；dγ [{d}]'),('实例数',str(n)),('反向 FLOPs',UNKNOWN)],'d 是归一化宽度，r 是前向保存的 inverse RMS；可复用中间量，内核归约与融合尚未确定。'), 'RMSNorm 保持张量形状，但具有可学习缩放参数。')
norm('qnorm','Query RMSNorm / Query 潜变量归一化',R,40)
norm('kvnorm','KV RMSNorm / 共享 KV 归一化',D,40)
norm('anorm','Attention RMSNorm / 注意力输入归一化',H,40)
norm('fnorm','FFN RMSNorm / FFN 输入归一化',H,40)
norm('finalnorm','Final RMSNorm / 最终归一化',H,1)
DATA['finalnorm']['note']='原图省略最终 HC collapse 的独立方框。源码先将 [M,4,5120] 合为 [M,5120]：额外 M×7H=36,700,160 基本操作，无新参数；它在最终输出总账内只计一次。'
linear('qa','W_Q⁽ᴸ⁾ / Query 第一投影',H,R,40,'将 5120 维 attention 输入压到 1280 维潜变量。')
linear('qb','W_Q⁽ᴿ⁾ / Query 第二投影',R,A*D,40,'把归一化潜变量展开为 64 个 512 维 Query heads。','[M,1280] → [M,32768] → [M,64,512]')
qP=H*R+R*A*D
add('query','Query Projection (Low-rank) / 两级 Query 投影','先降维，经过 Query RMSNorm，再展开多头 Query。','[M,5120] → [M,1280] → [M,64,512]',p([r'W_{QA}:[5120,1280],\quad W_{QB}:[1280,32768]',r'P=5120\times1280+1280\times32768'],[('每层矩阵',result(qP)),('40 层矩阵合计',result(qP*40)),('中间 RMSNorm','另计 1,280×40=51,200')]),f([r'F_{\rm linear}=2M(5120\times1280+1280\times32768)\times40'],nums(2*M*qP*40)+[('中间 Query RMSNorm','另计 209,797,120 ops；未重复加在矩阵项')]),b([r'Z=XW_{QA},\quad U=\operatorname{RMSNorm}(Z),\quad Q=UW_{QB}',r'dU=dQW_{QB}^{\mathsf T},\quad dW_{QB}=U^{\mathsf T}dQ',r'dZ=\operatorname{RMSNormBack}(dU,Z,\gamma)',r'dX=dZW_{QA}^{\mathsf T},\quad dW_{QA}=X^{\mathsf T}dZ'],[('逆向形状','dQ [M,32768] → dU [M,1280] → dZ [M,1280] → dX [M,5120]'),('权重梯度','dW_QB [1280,32768]；dW_QA [5120,1280]'),('40 层矩阵反向',fmt(4*M*qP*40)+' FLOPs'),('RMSNorm / 总反向',UNKNOWN)],'不能跳过中间 RMSNorm；若 Indexer 复用 U，其梯度还需汇入 dU。',True),'低秩节省投影参数；两级权重都需要计算梯度。',related=['qa','qnorm','qb'],scope='40 层；分项不重复相加')
linear('kv','W_KV / Shared KV Projection',H,D,40,'每层生成共享于全部 Query heads 的 local KV。',note=KV_NOTE)
DATA['kv']['related']=['kvnorm']

add('rope','RoPE / 旋转位置编码','仅旋转最后 64 个坐标；Query、local KV 与输出逆旋转均保持形状。','Q/O [M,64,512]；KV [M,512] → 同形状',zero(),f([r'F_B=40M(6\times64\times64+3\times64)'],nums(1014497280), '含 Query + local KV + 输出逆 RoPE。compressed KV 与 Index RoPE 在各自模块计数。'),b([r'y=R_\theta x,\quad dx=R_\theta^{\mathsf T}dy',r'F_{\rm backward,rot}=F_{\rm forward,rot}'],[('40 层三类旋转', '1,014,497,280 基本 FLOPs')],'角度为固定位置函数；仅计算输入梯度。实数旋转的解析计数，非已验证训练 kernel。',True),'RoPE 没有额外可学习矩阵，也不是加到 token embedding 上。',note='原图 “Apply to Q,K,V” 保留；源码 K/V 共用同一 KV，并对 attention 输出做 inverse RoPE。')
oP=8*4096*1024+8192*H
add('output','Inverse RoPE + Grouped Output / 分组输出投影','逆旋转 attention 输出后，8 个独立 head groups 各自降维，再汇合到 5120 维。','[M,64,512] → [M,8,4096] → [M,8,1024] → [M,8192] → [M,5120]',p([r'P=8(4096\times1024)+8192\times5120'],[('每层',result(oP)),('40 层',result(oP*40)),('inverse RoPE','0 额外参数')]),f([r'F_O=40M\cdot2[8(4096\times1024)+8192\times5120]'],nums(2*M*oP*40), '逆 RoPE 的 503,316,480 基本操作已计入 Main RoPE，不重复添加。'),b([r'dZ=dY W_{OB}^{\mathsf T},\quad dW_{OB}=Z^{\mathsf T}dY',r'dU_g=dZ_g W_{OA,g}^{\mathsf T},\quad dW_{OA,g}=U_g^{\mathsf T}dZ_g',r'dO=R_\theta dU'],[('形状','dY [M,5120] → dZ [M,8192] → dU_g [M,4096] → dO [M,64,512]'),('dW','[8192,5120] 与 8×[4096,1024]'),('40 层矩阵反向',fmt(4*M*oP*40)+' FLOPs')],'8 组为 block-diagonal 映射；不可视为 32768→8192 的稠密全连接。旋转另列。',True),'输出投影的参数比 Query 投影更多。',related=['rope'])

def comp_cost(ids):
    r2=sum(i<20 for i in ids); r1=len(ids)-r2
    jc=r2*M//2+r1*M; groups=r2*M//2
    basic=(2*r2+r1)*2*M*H*D+groups*6*D+int(jc)*(4*D+3*64)
    special=groups*4*D+2*int(jc)
    return int(basic),int(special)
def index_cost(ids):
    owners=[i for i in ids if i in [2,8,14,20]]
    jc=sum(M//2 if i<20 else M for i in owners)
    pairs=M*sum(M//2 if i<20 else M for i in ids)
    basic=len(ids)*M*(2*R*32*128+2*H*32+3*32*64+32)+jc*(2*D*128+4*128+3*64)+(2*32*128+2*32-1)*pairs
    return basic,2*jc
add('compressor','Compressor / 压缩 KV 生产者','四个 source layers 生成可复用的压缩表示；ratio-2 对每两个 token 按特征门控池化。','[1,1024,5120] → [1,Cᵣ,512]；r=2: Cᵣ=512；r=1: Cᵣ=1024',p([r'P_{r=2}=2HD+D,\quad P_{r=1}=HD+D'],[('ratio-2 ×3','5,243,392 × 3'),('ratio-1 ×1','2,621,952 × 1'),('全部四实例',result(18352128))]),f([r'F_B=7(2MHD)+G_2(6D)+J_c(4D+3p)',r'F_E=G_2(4D)+2J_c'],nums(*comp_cost([2,8,14,20])), 'G₂=1536 完整二元组，Jc=2560 新 latent；H=5120，D=512，p=64。每个 token 都投影，不复用上次 forward 的历史。'),b([r'c=\sum_i\alpha_i v_i,\quad dv_i=\alpha_i dc',r'd\alpha_i=dc\odot v_i',r'ds_i=\alpha_i(d\alpha_i-\sum_j\alpha_j d\alpha_j)',r'dX=dV W_{KV}^{\mathsf T}+dS W_{gate}^{\mathsf T}'],[('权重梯度','dW_KV=XᵀdV；dW_gate=XᵀdS；各 [5120,512]'),('形状','dc [B,Cᵣ,512] → dv/ds [B,T,512] → dX [B,T,5120]'),('7 次矩阵投影的反向',fmt(7*4*M*H*D)+' FLOPs'),('池化、Norm、总量',UNKNOWN)], 'ratio-1 没有 gate 投影；多个消费者的 KV 梯度在生产者汇总，不重复计算共享权重。'),'生产 KV 的只有 4 层，消费 KV 的有 38 层。',scope='source layers 2 / 8 / 14 / 20')
add('indexer','Indexer / 稀疏位置索引','对压缩历史打分，选出最多 512 个位置；后续 attention 再计算自身的 QK/PV。','Q [M,32,128]；K [Cᵣ,128] → scores [M,Cᵣ] → IDs [M,min(512,Cᵣ)]',p([r'P=8(1280\times4096+5120\times32)+4(512\times128+128)'],[('Query / head weight','8 套'),('Key projection / norm','4 套'),('全部 Indexer',result(43516416))]),f([r'F_B=8M(2R a_i d_i+2H a_i+3a_i p+a_i)',r'\quad+J_c(2Dd_i+4d_i+3p)+(2a_id_i+2a_i-1)P_{idx}'],nums(*index_cost([2,8,14,20,24,28,32,36])), 'aᵢ=32，dᵢ=128，p=64，Jc=2560；Pidx=6,815,744。先算完整矩形分数，再作因果/候选 mask，不能替换为三角矩阵。'),b([r'\mathcal I=\operatorname{TopK}(s),\quad KV_{sel}=\operatorname{Gather}(KV,\mathcal I)'],[('选中 KV 值','由 attention 得到梯度并 scatter-add 回共享 KV'),('整数 indices','不可微；没有穿过 Top-K IDs 的普通链式梯度'),('Indexer 参数梯度 / FLOPs',UNKNOWN)], '仅下游 Gather 无法把梯度传回打分权重。额外训练目标、代理梯度或专用训练路径均未提供，不能凭前向 GEMM 推出一笔反向总量。'),'Indexer 分数不是主 attention 概率。',scope='8 次打分，4 个 Key owners')

def attn_cost(n,tiles):
    return A*((4*64*D+3*64+2+D)*M*tiles+2*n*M), A*((64+1)*M*tiles+(D+1)*n*M)
ATTN_B,ATTN_E=attn_cost(40,384)
def attention(key,title,n,tiles,note='',scope=''):
    add(key,title,'Query 对 local 与可用 compressed KV 做统一的注意力归一化，并以相同 KV 加权汇聚。','Q [M,64,512] + KV bank [Nkv,512] → O [M,64,512]',p([rf'P_{{sink}}=64\times{n}'],[('QK / Softmax / PV','0 额外投影权重'),('attention sink',result(64*n))], 'Query、KV、output、compressor 与 Indexer 在各自模块计数。'),f([r'A_B(n_t)=64[n_t(4\cdot64\cdot512+3\cdot64+2+512)+2]',r'A_E(n_t)=64[65n_t+513]'],[('范围',scope or f'{n} 层'),('64 槽 tile 合计 / token',str(tiles))]+nums(*attn_cost(n,tiles)), '完整 batch 成本 M×ΣA(nt)。无压缩层 2 tiles，压缩层 10 tiles。mask 槽仍执行 tile 矩阵乘法；sink 仅进分母。'),b([r'S=QK^{\mathsf T}/\sqrt D,\quad O=PV',r'dP=dOV^{\mathsf T},\quad dV=P^{\mathsf T}dO',r'dS=P\odot(dP-\operatorname{rowsum}(P\odot dP))',r'dQ=dSK/\sqrt D,\quad dK=dS^{\mathsf T}Q/\sqrt D'],[('Q 梯度','[M,64,512]'),('共享 KV 梯度','Σheads(dK + dV) → [Nkv,512]；跨消费者再累加'),('按相同槽位执行的 4 个梯度 GEMM',fmt(8*A*D*M*tiles*64)+' FLOPs'),('Softmax / sink / 完整总量',UNKNOWN)],'上面数值只是假设相同 tile 槽位的矩阵主项，不是训练 kernel 总量；sink 参与归一化并有梯度。稀疏 gather/scatter 与实际训练调度未验证。'),'被 mask 的位置不等于该内核跳过了矩阵运算。',note=note,scope=scope,related=['query','kv','output','compressor','indexer'])
attention('sliding','Sliding Window Attention / 早期局部注意力',2,4,'原图写 early layers；源码 0–1 为纯 local，2–39 仍保留 local 分支。此卡仅计 0–1 两层，不能再次加到 Main Attention 总账。','layers 0–1；window=128')
attention('sparse','Compressed KV Sparse Attention / 压缩稀疏注意力',38,380,'原图把 Sliding Window 与 Compressed KV 上下绘出；源码 2–39 将二者拼为一次 sparse_attn，共享一次归一化，不能把两个独立 softmax 相加。','layers 2–39；local 128 + compressed 512 slots')
attention('multi','Multi-Source Attention / Decoder 多源注意力',20,200,GLOBAL_NOTE,'layers 20–39；不含 Q/KV/O 投影')
DATA['sparse']['related']=['compressor','indexer','query','output']
add('global','Global KV Projection / 图示全局 KV','按原图显示 Encoder 到 Decoder 的共享通路；计数映射到 layer 20 的 ratio-1 KV source。','源码：[M,5120] → [M,512]；供 layers 20–39 共享',p([r'P=5120\times512+512'],[('source layer 20 ×1',result(2621952)),('权重共享','后续 19 层不新增这套压缩投影')]),f([r'F_B=2MHD+M(4D+3p),\quad F_E=2M'],nums(*comp_cost([20])), '这里只计 ratio-1 source，含它的 norm 和 RoPE；Indexer 与各层 local KV 另列。'),b([r'dC=\sum_{\ell=20}^{39}dC^{(\ell)}',r'dX=dZ W^{\mathsf T},\quad dW=X^{\mathsf T}dZ'],[('形状','dC [M,512] → inverse RoPE / norm → dZ [M,512] → dX [M,5120]'),('dW','[5120,512]'),('矩阵反向',fmt(4*M*H*D)+' FLOPs'),('总反向',UNKNOWN)], '先累加所有消费者梯度，再穿过 source 的 RoPE/Norm/投影；是否 detach 必须由训练实现确认。'), '共享的是生产者输出；每个 Decoder Query 仍独立执行。',note=GLOBAL_NOTE,related=['compressor','indexer'])

hcP=491547
add('hc','Hyper-Connection / 四路残差混合','动态生成 pre、post 与 comb 系数，在四路残差之间进行压缩、扩展和混合。','[M,4,5120] → pre [M,5120] → post [M,4,5120]',p([r'P=(4\cdot5120)\cdot24+24+3'],[('每套 HC',result(hcP)),('每层两套',result(2*hcP)),('80 套合计',result(80*hcP))]),f([r'F_B=80M[2cHz+2cH+z+(2c-1)H+c(2c+1)H+1192]',r'F_E=80M(2+664)'],nums(102021201920,54558720),'c=4，z=24，H=5120。Sinkhorn 共 20 次列归一化、19 次普通行归一化与初始 row softmax；final collapse 另计。'),b([r'x=\sum_i a_i R_i,\quad Y_j=b_j u+\sum_i C_{ji}R_i',r'du=\sum_j b_j dY_j,\quad dR_i=\sum_j C_{ji}dY_j+a_i dx',r'db_j=\langle dY_j,u\rangle,\quad dC_{ji}=\langle dY_j,R_i\rangle',r'dW=R_{flat}^{\mathsf T}dL,\quad dR_{flat}\mathrel{+}=dLW^{\mathsf T}'],[('形状','dY,dR [M,4,5120]；da,db [M,4]；dC [M,4,4]'),('生成权重','dW [20480,24]；base [24]；scale [3]'),('仅系数投影反向',fmt(80*4*M*20480*24)+' FLOPs'),('Sinkhorn / 全部 HC 反向',UNKNOWN)],'还需传播 da_i=⟨dx,R_i⟩，并反传 sigmoid、统计量与每轮 Sinkhorn。当前 Triton forward 没有给出训练 backward kernel。'),'4+4+16 个动态系数不是额外存储的可学习参数。',note=HC_NOTE)
add('streams','Four Residual Streams / 四路残差状态','四个并行向量保留并混合不同路径的信息。','[M,4,5120]；每一路 [M,5120]',zero(),f([], [('状态本身','没有独立算术；混合成本在 HC 中')]),b([r'dR\in\mathbb R^{M\times4\times5120}'],[('梯度','来自子层、残差 comb 与 coefficient generator 的贡献相加')]),'4 路同时存在，不能当成一条 5120 维残差。',note=HC_NOTE+' 原图左侧 h⁽³⁾ 标签重复一次，视觉照录；数学上四路是索引 1…4。',related=['hc'])

routerP=H*384+768
add('router','MoE Router / 专家路由','为 384 个候选专家打分，用加偏置的分数选 Top-6，再由无偏置分数计算路由权重。','[M,5120] → [M,384] → IDs / weights [M,6]',p([r'P=5120\times384+384+384'],[('每层',result(routerP)),('40 层',result(routerP*40)),('文本实际触及','不使用 384 维 vision-selection bias')]),f([r'F_B=40M(2H\cdot384+384+12)',r'F_E=40M(3\cdot384+6)'],nums(161077493760,47431680),'sqrtsoftplus、温度除法、归一化按文档特殊操作口径；Top-K 比较成本另列。'),b([r's=\sqrt{\operatorname{softplus}(XW/\tau)}',r'w_i=1.5\frac{s_i}{\sum_{j\in\mathcal I}s_j+\epsilon}',r'dX=dL W^{\mathsf T},\quad dW=X^{\mathsf T}dL'],[('梯度形状','dw [M,6] → ds/dL [M,384] → dX [M,5120]；dW [5120,384]'),('矩阵反向主项',fmt(4*M*H*384*40)+' FLOPs'),('离散选路 / bias 更新',UNKNOWN)],'固定选中集合时，归一化权重能向 scores 传梯度；selection-only bias 不能通过整数 Top-K 得到普通梯度。可能的训练偏置更新策略未提供。'),'384 是打分维度，6 是执行的 routed expert 数。',related=['selection','weights'])
add('selection','Top-6 Expert Selection / 离散专家选择','选出每 token 的 6 个 routed experts；只选择索引，不执行整个专家池。','scores [M,384] → indices [M,6]',zero(),f([], [('浮点 FLOPs','比较、Top-K 与 dispatch 成本另列'),('规模','40 次 × 1024 tokens × 384 candidates')]),b([r'\mathcal I=\operatorname{Top6}(s+b)'],[('索引梯度','不可微；不对整数 IDs 求导'),('下游值梯度','选中 scores / expert values 可微')],'不能把“索引不可微”理解为选中专家没有梯度。'),'每 token 只执行 6 个路由专家，再加 1 个共享专家。')
add('weights','Top-k / Normalize Weights / 路由权重','收集被选中的无偏置分数，归一化并乘 route_scale=1.5。','scores [M,384] → selected scores / weights [M,6]',zero(),f([r'w_i=1.5s_i/(\sum_{j\in\mathcal I}s_j+\epsilon)',r'F_B=40M(5+1+6),\quad F_E=40M\cdot6'],nums(40*M*12,40*M*6),'这是 Router 总账中的子集，不可重复相加。'),b([r'S=\sum_j s_j+\epsilon',r'ds_j=\frac{1.5}{S}dw_j-\frac{1.5}{S^2}\sum_i dw_i s_i'],[('形状','dw [M,6] → ds_selected [M,6] → scatter 至 [M,384]'),('数值 FLOPs',UNKNOWN)]),'选路使用带偏置分数；权重使用未加偏置的分数。')
expertP=3*H*F
add('expert','Expert FFN (SwiGLU) / 路由专家网络','Gate 和 Up 两条投影经 SwiGLU 相乘，乘路由权重后由 Down 投影回模型维度。','Xⱼ [mⱼ,5120] → G,U [mⱼ,2304] → Yⱼ [mⱼ,5120]',p([r'W_1,W_3:[5120,2304],\quad W_2:[2304,5120]',r'P_{expert}=3\cdot5120\cdot2304'],[('单专家',result(expertP)),('384×40 routed 存储',result(expertP*384*40)),('6×40 单 token 激活权重',result(expertP*6*40))]),f([r'\sum_j m_j=6M,\quad F_{linear}=40\cdot6M\cdot6HF'],[('路由专家矩阵',fmt(40*6*M*6*H*F)+' FLOPs'),('SwiGLU 基本 / 特殊',f'{fmt(40*6*M*2*F)} / {fmt(40*6*M*F)}'),('路由乘权',fmt(40*6*M*F)+' 基本 FLOPs')], 'shared expert 和输出相加另计；每个 mⱼ 取决于 routing，逻辑 GEMM 不因专家分布变化。'),b([r'Z=w\,[\operatorname{SiLU}(G_c)\odot U_c],\quad Y=ZW_2',r'dZ=dYW_2^{\mathsf T},\quad dW_2=Z^{\mathsf T}dY',r'dX=dG W_1^{\mathsf T}+dU W_3^{\mathsf T}',r'dW_1=X^{\mathsf T}dG,\quad dW_3=X^{\mathsf T}dU'],[('形状','dY [mⱼ,5120] → dZ,dG,dU [mⱼ,2304] → dX [mⱼ,5120]'),('权重梯度','dW₁,dW₃ [5120,2304]；dW₂ [2304,5120]'),('全部 routed 矩阵反向',fmt(40*6*M*12*H*F)+' FLOPs'),('激活、路由权重、总量',UNKNOWN)], '只对实际派发的 Σmⱼ=6M 行求导，不能按 384M 计执行量。FP4 的训练梯度规则另需确认。'),'存储的专家数与每 token 执行的专家数是不同计数。',related=['swiglu','weights','shared'])
add('shared','Shared Expert / 共享专家','每个 token 都执行这一独立专家，且不乘 router 权重。','[M,5120] → [M,2304] → [M,5120]',p([r'P=3HF'],[('每层 1 个',result(expertP)),('40 层',result(expertP*40))]),f([r'F_B=40M(6HF+2F),\quad F_E=40MF'],nums(40*M*(6*H*F+2*F),40*M*F)),b([r'dX=dG W_1^{\mathsf T}+dU W_3^{\mathsf T}',r'dW_2=Z^{\mathsf T}dY,\quad dW_1=X^{\mathsf T}dG,\quad dW_3=X^{\mathsf T}dU'],[('中间形状','dY [M,5120] → dG,dU [M,2304] → dX [M,5120]'),('40 层矩阵反向',fmt(40*M*12*H*F)+' FLOPs'),('激活 / 总量',UNKNOWN)], '和 routed expert 相同三投影结构，但无路由标量。'),'共享专家每层独立存储，不是跨 40 层绑定同一套权重。',related=['expert','swiglu'])
add('swiglu','SwiGLU / 门控激活','SiLU(gate) 与 up 分支逐元素相乘；源码还先对两分支作不同的 clipping。','G,U [mⱼ,2304] → Z [mⱼ,2304]',zero(),f([r'Z=\operatorname{SiLU}(G_c)\odot U_c',r'F_B=40\cdot7M\cdot2F,\quad F_E=40\cdot7M\cdot F'],nums(40*7*M*2*F,40*7*M*F),'范围：6 routed + 1 shared；仅激活，不含三投影。clamp 比较另列。'),b([r's=\sigma(G_c),\quad a=G_c s',r'dU_c=dZ\odot a',r'dG_c=dZ\odot U_c\odot[s+G_cs(1-s)]',r'dG=dG_c\odot\mathbf1_{G<10},\quad dU=dU_c\odot\mathbf1_{-10<U<10}'],[('梯度形状','dZ、dG、dU [mⱼ,2304]'),('反向总量',UNKNOWN)], '公式在阈值以外成立；边界处子梯度策略由训练实现决定。gate 仅上截断，up 双侧截断；路由标量的链式法则另计。'),'两个输入分支都获得梯度，没有额外可学习激活参数。')
add('combine','Combine / Weighted Sum / 专家输出合并','累加 6 个已加权的 routed 输出，再加不经路由加权的 shared 输出。','6×[M,5120] + [M,5120] → [M,5120]',zero(),f([r'Y=\sum_{j\in\mathcal I}Y_j+Y_s',r'F_{add}=40M(6+1)H'],nums(40*M*7*H),'源码从零缓冲累加 6 次再加 shared；图示 Weighted Sum 的权重实际在 routed Down 投影前乘，已列于专家账。'),b([r'dY_j=dY,\quad dY_s=dY',r'dw_j=\langle dZ_j,H_j\rangle'],[('梯度','相加向各分支传相同 dY；随后按各路由权重反传'),('总反向 FLOPs',UNKNOWN)], '共享相加不产生新的 GEMM；路由权重梯度来自专家内部加权点。'),'专家贡献汇总为一个 5120 维更新，再进入 HC。')
add('moe','Mixture of Experts / MoE','384 个 routed experts 中选 6 个执行，再加 1 个 shared expert。','[M,5120] → 6 routed + 1 shared → [M,5120]',p([r'P_{stored}=40[385(3HF)+H\cdot384+768]',r'P_{active,FFN}=40\cdot7\cdot3HF'],[('专家存储',result(544997376000)),('Router 存储',result(78673920)),('二者总计',result(545076049920)),('单 token 激活专家权重',result(9909043200))]),f([r'F_{experts,B}=40M[7(6HF+2F)+6F+7H]',r'F_{experts,E}=40M\cdot7F'],[('Experts + aggregation',fmt(20297736519680)+' ops'),('Router',fmt(161124925440)+' ops'),('MoE 合计',fmt(20458861445120)+' ops')]),b([r'\sum_jm_j=6M,\quad m_{shared}=M',r'F_{back,expert\ matrices}=40\cdot7M\cdot12HF'],[('专家矩阵反向主项',fmt(40*7*M*12*H*F)+' FLOPs'),('Router / 激活 / 完整总量',UNKNOWN)], '专家权重只对本 batch 的派发行计算梯度；单 token 激活 7 个，并不表示整个 batch 仅使用 7 个不同专家。'),'544.997B 存储专家权重不等于实际执行 385 个专家。',related=['router','selection','expert','shared','swiglu','combine'])

engTables=[98305579008,98308270592]
engDense=157327360
engBasic=644635222016
add('ngram','N-gram Construction / N-gram 构造','构造长度 2、3、4 的因果 token 片段，序列开头用 pad 补齐。','IDs [B,T] → 3 个阶数 × 8 个 hash heads / instance',zero(),f([], [('浮点 FLOPs','0；整数构造与 padding 成本另列')]),b([], [('梯度','整数 token 窗口与 padding 不可微')]),'N-gram 是局部离散上下文，不是对整段序列执行 attention。')
add('hash','Hashing & Lookup / 哈希地址','用 N-gram 的整数哈希选择 24 行 memory；两次注入各有对应布局。','[B,T] → hash IDs [B,T,2,24]',zero('查找表权重在 Engram Embedding 计数，哈希本身无学习参数。'),f([], [('浮点 FLOPs','0；hash、索引、数据移动另列')]),b([], [('哈希 / 选址','不可微，整数地址无梯度'),('选中行','lookup values 的梯度可 scatter-add 到相应表行')]),'选址不可微，取出的连续向量可以学习。',related=['engembed'])
add('engembed','Engram Embedding / N-gram 记忆表','每个 token、每实例查 24 行，每行 256 维，再拼为 6144 维。','[B,T,24] → [B,T,24,256] → [M,6144]',p([r'P=384006168\cdot256+384016682\cdot256'],[('layer 1 table',result(engTables[0])),('layer 14 table',result(engTables[1])),('两表总计',result(sum(engTables))),('单 token 实际读取','2×24×256 = 12,288 个权重值')]),f([r'F_B=2M\cdot6144'],nums(2*M*6144),'这项为显式 lookup dequantization 乘法；gather 自身是内存/整数操作。'),b([r'dE_{i,:}=\sum_{t,k:\,hash(t,k)=i}dZ_{t,k,:}'],[('形状','dZ [M,24,256] → table rows [N,256] 的稀疏累加'),('量化表训练规则 / 总量',UNKNOWN)], '不对 hash ID 求导；实数学习值可接收梯度，FP8 权重与 scale 的训练策略未知。'),'巨大的表容量不会导致对整张表执行 GEMM。')
add('enggate','Projection & Gating / 记忆投影与门控','6144 维记忆投影为四个 key 与一个共享 value，按当前四路残差计算匹配门。','[M,6144] → [M,25600] → K [M,4,5120] + V [M,5120]',p([r'P=6144\times25600+2(4\times5120)'],[('单实例',result(engDense)),('两个实例',result(2*engDense))]),f([r'F_{projection}=2\cdot2M\cdot6144\cdot25600',r'F_{gate,B}=2[4H+M\cdot4(7H+2)],\quad F_E=2M\cdot4\cdot6'],nums(2*2*M*6144*25600+2*(4*H+M*4*(7*H+2)),49152), '门控 q_weight×k_weight 每实例只算一次，不按 M 重复。此处不含 lookup 反量化与 residual injection，两者另列。'),b([r'Y_c=R_c+g_cV,\quad dV=\sum_cg_cdY_c',r'dg_c=\langle dY_c,V\rangle,\quad dR_c\mathrel{+}=dY_c',r'dZ=d[K,V]W^{\mathsf T},\quad dW=Z^{\mathsf T}d[K,V]'],[('形状','dK [M,4,5120] + dV [M,5120] → [M,25600] → dZ [M,6144]'),('dW / gate weights','[6144,25600]；dq_weight,dk_weight [4,5120]'),('两实例投影反向',fmt(2*4*M*6144*25600)+' FLOPs'),('门控 / 量化 / 总量',UNKNOWN)], '还须沿归一化点积、signed sqrt、clamp 与 sigmoid 反传至 residual/key 和两组权重。反向 kernel 未提供。'),'两组 [4,5120] gate weights 以乘积参与计算；广播不等于重复计算。')
add('injection','Residual Injection / 四路记忆注入','门控后的同一 value 广播到四路 residual，在 layer 1 与 layer 14 之前注入。','[M,4,5120] + gate [M,4] × value [M,5120] → [M,4,5120]',zero(),f([r'R_c\leftarrow R_c+g_cV,\quad F_B=2\cdot M\cdot4\cdot2H'],nums(2*M*4*2*H)),b([r'dV=\sum_cg_cdR_c^{out},\quad dg_c=\sum_hdR_{c,h}^{out}V_h',r'dR_c^{in}\mathrel{+}=dR_c^{out}'],[('形状','dV [M,5120]；dg [M,4]；dR [M,4,5120]'),('完整梯度','还需经过 gate(R,K) 路径；总量未验证')]),'注入位置是进入 layer 1 / 14 之前，零基索引。')
for key,i in [('engram1',0),('engram14',1),('engram',None)]:
    n=2 if i is None else 1; tables=sum(engTables) if i is None else engTables[i]
    add(key,'Engram Memory / N-gram 记忆'+('' if i is None else f' · layer {[1,14][i]}'),'查局部 N-gram 记忆，投影并门控注入四路残差。','IDs [B,T] → [M,6144] → K/V → residual [M,4,5120]',p([r'P=P_{table}+n(6144\cdot25600+2\cdot4\cdot5120)'],[('实例数',str(n)),('表参数',result(tables)),('Dense + gate',result(n*engDense)),('存储合计',result(tables+n*engDense))]),f([r'F_B=n[M(J+2J(c+1)H+c(9H+2))+cH]',r'F_E=n\cdot6Mc'],nums(engBasic*n//2,49152*n//2),f'n={n}，J=6144，c=4，H=5120。包含显式反量化、投影、gating、residual injection。'),b([r'IDs\xrightarrow{hash}\mathcal I\xrightarrow{lookup}Z\xrightarrow{W}K,V\xrightarrow{gate}R^{out}'],[('可微路径','lookup values、W、q/k gate weights、四路 residual'),('不可微路径','整数 n-gram / hashing / ID selection'),('投影矩阵反向主项',fmt(n*4*M*6144*25600)+' FLOPs'),('完整反向',UNKNOWN)], '需要确认量化表与 gate clamp 的训练规则；不可将整张记忆表的参数量乘 2 当 FLOPs。'),'记忆容量与每个 token 的查表计算量应分开理解。',related=['ngram','hash','engembed','enggate','injection'],scope='layers 1 / 14' if i is None else f'before layer {[1,14][i]}')

linear('head','LM Head / 词表输出投影',H,V,1,'把最终隐藏状态映射为每个词表 token 的 logit。',note='Scenario P 为所有 T 位置输出 logits；源码默认只取最后位置。这里是明确的训练前向代理，不把 last-token decode 混入数值。输入 embedding 与 LM Head 权重不绑定。')
add('logits','Logits / 未归一化词表分数','保存 LM Head 的实数输出；logits 本身不是概率。','[B,T,V]=[1,1024,129280]',zero(),f([], [('额外浮点算术','0；投影已在 LM Head 计数')]),b([r'dL\in\mathbb R^{M\times129280}'],[('梯度来源','由外部目标提供，作为 LM Head 的 dY；loss 不在本图范围')]),'这个矩阵是激活，不是另一套参数。')
add('prediction','Next Token Prediction / 下一 token 示意','根据最后位置 logits 选择下一 token。原图中的词和概率仅是说明性示例。','last logits [1,129280] → token ID [1]',zero(),f([], [('Scenario P','采样排除在训练前向总量之外'),('图中 0.42 / 0.18 / 0.06','不是本地 checkpoint 的实测结果')]),b([], [('离散选取','argmax / sampling 不进行普通反向求导'),('本任务','不引入 loss、optimizer 或采样训练估计')]),'预测展示与模型训练的全位置 logits 计数不是同一执行场景。')

# Aggregate only source-owned work; injection and output modules are separate.
CORE_P=qP+H*D+oP+R+D+64+2*hcP+2*H+routerP+385*expertP
CORE_B=(3972844748800+209715200+214832250880+1014497280+6184752906240+102021201920+1677721600+161077493760+20297075916800)//40
CORE_E=(81920+81920+54558720+163840+47431680+660602880)//40
def layer_group(key,title,ids):
    n=len(ids); owners=[i for i in ids if i in [2,8,14,20]]; ix=[i for i in ids if i in [2,8,14,20,24,28,32,36]]
    cp=sum(5243392 if i<20 else 2621952 for i in owners)
    ip=len(ix)*(5242880+163840)+len(owners)*(65536+128)
    tiles=sum(2 if i<2 else 10 for i in ids)
    cb,ce=comp_cost(owners); ib,ie=index_cost(ix); ab,ae=attn_cost(n,tiles)
    totalp=n*CORE_P+cp+ip; fb=n*CORE_B+cb+ib+ab; fe=n*CORE_E+ce+ie+ae
    add(key,title,'按原图的 Attention → HC → FFN/MoE → HC 分组；实际由四路残差贯穿，并按 source layer 共享压缩 KV。',f'[M,4,5120] → {n} block(s) → [M,4,5120]',p([rf'P={n}P_{{block,base}}+P_{{owned\ comp}}+P_{{owned\ index}}'],[('单层固定部分',result(CORE_P)),('本组拥有 compressor',str(owners)),('本组拥有 Indexer',str(ix)),('本组存储',result(totalp))], '不包含图外置的两次 Engram 注入、embedding、final norm、LM Head；Global KV source20 已包含，不可重复相加。'),f([rf'F_B={n}F_{{block,base,B}}+F_{{comp,B}}+F_{{index,B}}+F_{{attn,B}}'],nums(fb,fe),'按照具体层号分配 4 个 compressor、8 个 Indexer、4 个 index-key owners；单层成本并非全部相同。'),b([r'dR_{in}=dR_{residual}+dR_{sublayer}+dR_{mix}',r'dKV_{source}=\sum_{\ell\in consumers}dKV^{(\ell)}'],[('残差梯度形状','[M,4,5120]'),('矩阵子项','可在 Query、Output、MoE 等面板查看解析主项'),('整个分组反向总量',UNKNOWN)],'源文件只有 inference_mode 路径；不以 2×整个前向作为反向总量。共享路径不得默认 detach。'), '生产者和消费者必须按层号分配，不能把所有工作机械地乘 40。',note=CED+' '+HC_NOTE,related=['query','kv','sparse' if min(ids)<20 else 'multi','hc','moe','compressor','indexer'],scope=f'layers {min(ids)}–{max(ids)}')
layer_group('encoder','Causal Encoder / 20 层 Encoder',list(range(20)))
layer_group('decoder','Causal Decoder / 20 层 Decoder',list(range(20,40)))
for i in [0,1,2,13,14,19,20,21,22,33,34,39]: layer_group(f'layer{i}',f'Transformer Layer {i}',[i])
for key,base,title in [('encoderblock','encoder','Causal Encoder Block ×20'),('decoderblock','decoder','Causal Decoder Block ×20')]:
    DATA[key]=copy.deepcopy(DATA[base]); DATA[key]['title']=title
for key,ids,title in [('encoderattn',list(range(20)),'Encoder Attention / 分组'),('decoderattn',list(range(20,40)),'Decoder Attention / 分组')]:
    n=len(ids);owners=[i for i in ids if i in [2,8,14,20]];ix=[i for i in ids if i in [2,8,14,20,24,28,32,36]]
    cb,ce=comp_cost(owners);ib,ie=index_cost(ix);ab,ae=attn_cost(n,sum(2 if i<2 else 10 for i in ids))
    pb=n*(qP+H*D+oP+R+D+64)+sum(5243392 if i<20 else 2621952 for i in owners)+len(ix)*(5242880+163840)+len(owners)*65664
    fb=n*((3972844748800+209715200+214832250880+1014497280+6184752906240)//40)+cb+ib+ab
    fe=n*4*M+ce+ie+ae
    add(key,title,'汇总本组 Query、local KV、RoPE、输出、主 attention 及其拥有的 compressor / Indexer。','[M,5120] → Q / KV / attention → [M,5120]',p([], [('20 层 attention 存储',result(pb)),('范围','含内部 Query/KV Norm 与 sink；不含外部 attention RMSNorm')]),f([],nums(fb,fe),'source20 的 Global KV 在 decoder 组内只计一次。'),b([], [('图上的分组','由各可微子图反向连接；共享 KV 梯度汇总'),('组反向总量',UNKNOWN)]),'encoder 与 decoder 的 source / indexer 调度不同。',note=CED,related=['query','kv','sliding','sparse','multi','output','compressor','indexer'])
add('hidden','Encoder Final Hidden States / 图示隐藏状态','图中作为 Encoder 输出连接 Global KV；状态本身不新增学习参数。','图示 [T,5120]；源码 residual [B,T,4,5120] → HC pre / norm → source 输入',zero(),f([], [('独立阶段成本','图示张量无额外算术；实际 HC/Norm 在已有模块计数')]),b([r'dH_{enc}=\sum_{paths}dH_{enc}^{(path)}'],[('共享梯度','从消费者路径回传；精确映射依训练实现确认')]),'张量形状与投影参数量是两类对象。',note=GLOBAL_NOTE)
add('localstates','Local Decoder States / 局部 KV','每层仍为所有 token 独立计算本层 local KV，窗口限制可见位置。','local KV bank [B,T,512]；每 query 可见 min(128,t+1) 个位置',zero('激活 bank 不重复计权重；local KV 投影见 W_KV。'),f([], [('独立成本','状态存取另列；投影与 attention 运算分别计账')]),b([r'dKV=dK+dV'],[('梯度','按索引回填 local KV，并累加所有 heads 的贡献')]),'复用 global KV 不会取消每层的 local KV。',note=KV_NOTE,related=['kv','multi'])

add('overview','DeepSeek-V4.1 / 全图口径','沿用提供图片的架构解释，计算以本地参数表与修订 FLOPs 文档为依据。','B=1，T=1024；H=5120；40 layers；4 residual streams',p([r'P_{core}=748494684784'],[('全部存储',result(748494684784)),('warm decode 单 token touched','16,130,590,832（完整压缩组路径）')], 'touched 不是 activations，也不是完整序列的独立专家数；不含 draft、ViT/Aligner 与量化 scale。'),f([r'F_M=F_B+F_E'],nums(36434932621312,3746705408)+[('完整 batch','36.438679326720 TFLOPs-equivalent'),('平均 token','35.584647780 GFLOPs-equivalent')], CONVENTION),b([r'Y=XW:\quad F_{back,linear}\approx4MKN'],[('矩阵模块','在 dX、dW 都计算时，矩阵反向是其前向的 2 倍'),('整模型反向总量',UNKNOWN)],'不是所有算子都满足 2×规则；未纳入 optimizer、loss、activation replay。'),'图示分区保留；可验证实现与解释性架构名称分开陈述。',note='输入三词与预测概率均属原图示意。实际 Tokenizer / checkpoint 未提供。'+CED)

# Preserve visual shorthand while explicitly recording source disagreements.
DATA['moe']['note']='原图把 Expert FFN → Shared Expert → Combine 纵向排列。源码 routed / shared experts 都接收同一个归一化输入，shared 不是 routed 的后续层。布局照录；计算按各专家路径相加。'
DATA['query']['note']='图中的 d / q 是示意符号；实际低秩宽度为 1280，Query 输出为 64×512。两级投影间的 RMSNorm 单独计数。'
DATA['hc']['backward']['note']+=' 这里 L=R_flat W 表示未缩放投影，dL 已包含系数变换与缩放的反传；C 的行 j 对应输出 copy、列 i 对应输入 copy。'
DATA['router']['backward']['note']+=' L=XW 是温度/激活之前的 logits，τ=1。'
DATA['compressor']['backward']['note']+=' dc 表示经逆 RoPE / Norm 反传后，到达池化结果的梯度。'
for key in ['sliding','sparse','multi']:
    DATA[key]['backward']['formulas'].append(r'ds_{sink}=-p_{sink}\sum_jP_jdP_j')
    DATA[key]['backward']['note']+=' p_sink 是分母中 sink 的概率质量；sink 梯度需沿 batch / token 轴累加。'

def check():
    assert comp_cost([2,8,14,20]) == (37591416832,3150848)
    assert index_cost([2,8,14,20,24,28,32,36]) == (145235607552,5120)
    assert (ATTN_B,ATTN_E)==(3316307197952,2980577280)
    assert CORE_P*40+18352128+43516416+196928504320+2*V*H+H==748494684784
    base=CORE_B*40+CORE_E*40+sum(comp_cost([2,8,14,20]))+sum(index_cost([2,8,14,20,24,28,32,36]))+ATTN_B+ATTN_E+engBasic+49152+1355656726528
    assert base==36438679326720, base
    for key,d in DATA.items():
        assert all(x in d for x in ['parameter','forward','backward'])
        assert all(x in DATA for x in d['related']),key

if __name__=='__main__':
    check(); (ROOT/'src/deepseek/module_data.json').write_text(json.dumps(DATA,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(DATA)} calculation entries validated.')

