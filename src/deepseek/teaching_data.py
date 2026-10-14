"""Single-current-token teaching prose, subsequently bound to the independent audit.

Numerical context: revised decode, 1024 PRIOR positions, current position 1025.
This is the editable teaching baseline, not the source of audit truth. audited_data
applies independently reconstructed counts, source bindings, and assumptions.
Backward is real-valued local differentiation, not a whole training-step ledger.
"""
from copy import deepcopy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))

DATA={}
H,R,A,D,F,V=5120,1280,64,512,2304,129280
QP=H*R+R*A*D
OP=8*4096*1024+8192*H
EP=3*H*F
HP=491547
CORE_P=QP+H*D+OP+R+D+64+2*HP+2*H+(H*384+768)+385*EP
def fmt(n):return f'{n:,}'
def eqn(n):return str(n)
def step(title,formula,dims,symbols,note=''):
    return dict(title=title,formulas=[formula] if isinstance(formula,str) else formula,dimensions=[dims] if isinstance(dims,str) else dims,symbols=symbols,note=note)
def mode(steps,calcs=(),rows=(),explain='',totals=(),note='',extras=()):
    return deepcopy(dict(steps=steps,calculations=list(calcs),rows=list(rows),explain=explain,totals=list(totals),note=note,extras=list(extras)))
def card(key,title,purpose,steps,param,forward,backward,insight,related=(),note='',scope='当前 1 个 token'):
    DATA[key]=dict(title=title,purpose=purpose,parameter=param,forward=forward,backward=backward,insight=insight,related=list(related),note=note,scope=scope)
def param(steps,n,formula='',totals=(),note=''):
    return mode(steps,[formula] if formula else [],[('本模块存储',fmt(n)+' 个参数')],'把权重矩阵的行数乘以列数，就是需要学习的数字个数。',totals,note)
def noparam(steps,note='这一步没有新增的可学习权重。'):return mode(steps,[],[('新增参数','0')],note)
def forward(steps,n,formula='',totals=(),note='',explain='2 表示一次乘法和一次累加；1 表示当前只有一个 token。其余两个数分别是输入和输出的特征数。'):
    return mode(steps,formula if isinstance(formula,list) else [formula] if formula else [],[('当前 token · 一次计算',fmt(n)+' FLOPs')],explain,totals,note)
def backward(steps,rows=(),calcs=(),totals=(),note='',explain='d 表示梯度，即结果对学习目标 Loss 的影响；ᵀ 表示转置，把矩阵的行与列交换。'):
    return mode(steps,calcs,rows,explain,totals,note)
def info(steps,rows,note=''):return mode(steps,[],rows,note=note)
def unknown(steps,note):return backward(steps,[('完整反向计算量','暂不能给出完整数值')],note=note)
def mul(k,n,w='W',y='Y',x='X',title='计算这一次投影'):
    return step(title,rf'{y}={x}\times {w}',rf'[1,{k}]\times[{k},{n}]=[1,{n}]',[(x,f'当前 token 的 {k} 个输入数字'),(w,f'学习得到的权重，{k} 行、{n} 列'),(y,f'得到的 {n} 个输出数字')])
def grad_steps(k,n,w='W',x='X',y='Y'):
    return [step('① 把梯度传回输入',rf'd{x}=d{y}\times {w}^{{\mathsf T}}',rf'[1,{n}]\times[{n},{k}]=[1,{k}]',[(f'd{y}',f'后一个模块传来的梯度，有 {n} 个数字'),(f'{w}ᵀ','把原权重矩阵的行与列交换'),(f'd{x}','输入对 Loss 的影响，继续传给前面的模块')]),step('② 计算权重梯度',rf'd{w}={x}^{{\mathsf T}}\times d{y}',rf'[{k},1]\times[1,{n}]=[{k},{n}]',[(f'{x}ᵀ','当前 token 的输入改为一列'),(f'd{y}','同一份输出梯度'),(f'd{w}','每个权重对 Loss 的影响，供后续更新使用')])]
def linear(key,title,k,n,count,purpose,w='W',y='Y'):
    steps=[mul(k,n,w,y)]
    f=2*k*n
    card(key,title,purpose,steps,
      param(steps,k*n,rf'{k}\times{n}={k*n}',[(f'{count} 套权重合计',fmt(k*n*count)+' 个参数')] if count>1 else ()),
      forward(steps,f,rf'2\times1\times{k}\times{n}={f}',[(f'当前 token 经过 {count} 层',fmt(f*count)+' FLOPs')] if count>1 else ()),
      backward(grad_steps(k,n,w,y=y),[('输入梯度 dX',fmt(f)+' FLOPs'),('权重梯度',fmt(f)+' FLOPs'),('本次反向合计',fmt(2*f)+' FLOPs')],[rf'2\times1\times{n}\times{k}={f}',rf'2\times{k}\times1\times{n}={f}'],[(f'{count} 层的这些矩阵反向',fmt(2*f*count)+' FLOPs')] if count>1 else (),note='这里只算当前 token 对这一次投影的梯度；多次使用同一权重时，还需要累加各次贡献。'),
      '参数量由权重矩阵大小决定；这次的计算量还取决于输入几行数字。')

linear('kv','共享 KV 投影 / Shared KV',5120,512,40,'模型用 5120 个数字描述当前 token。把它乘以一个有 512 列的权重矩阵，得到 512 个数字；64 个 Query Head 都使用这一份结果。','W_{KV}','KV')
DATA['kv']['related']=['kvnorm']
DATA['kv']['purpose']='5120 是这个模型选定的特征宽度，用来描述当前 token。乘以一个有 512 列的权重矩阵后，得到 512 个数字；64 个 Query Head 都使用这一份结果。'
DATA['kv']['forward']['note']='这里只计算投影。后面的 KV RMSNorm 另算，不重复加在这一项里。'
linear('qa','Query 第一步：压缩',5120,1280,40,'把当前 token 的 5120 个数字组合成 1280 个数字，为后续生成 Query 做准备。','W_{QA}','H')
linear('qb','Query 第三步：展开',1280,32768,40,'把归一化后的 1280 个数字展开成 32768 个，再分成 64 组，每组 512 个。','W_{QB}','Q')
linear('head','词表打分 / LM Head',5120,129280,1,'当前 token 的最终向量有 5120 个数字。乘以词表权重后，给 129280 个候选 token 各打一个分数。','W_{head}','Y')
DATA['head']['insight']='得到的是 129280 个分数，不是 129280 个新 token。'

def norm_steps(d):
    return [step('调整数字的整体大小',[r'r=\frac{1}{\sqrt{\frac{x_1^2+\cdots+x_{%d}^2}{%d}+\epsilon}}' % (d,d),r'Y=X\odot r\odot\gamma'],rf'[1,{d}]\longrightarrow[1,{d}]', [('X',f'当前这组 {d} 个数字；x₁… 是其中各个数字'),('r','从这组数字算出的统一缩放比例'),('γ',f'{d} 个可学习的逐项缩放系数'),('ε','一个很小的数，避免除以零'),('⊙','对应位置相乘；Y 是调整后的向量')])]
def norm_backward(d):
    return [step('① 把梯度传回原向量',[r'G=dY\odot\gamma',rf'dX=rG-\frac{{r^3}}{{{d}}}X\sum_j(G_jX_j)'],rf'dY:[1,{d}]\longrightarrow dX:[1,{d}]',[('dY','调整后的数字对 Loss 的影响'),('γ','前向使用的逐项缩放权重'),('G','先把输出梯度乘以 γ'),('r','前向算出的统一缩放比例'),('Σⱼ','把这组向量各位置的结果相加'),('dX','原来每个输入数字对 Loss 的影响')]),step('② 计算缩放权重的梯度',r'd\gamma=dY\odot X\odot r',rf'[1,{d}]\odot[1,{d}]\odot[1,1]=[1,{d}]',[('X','前向时的原向量'),('r','前向的缩放比例'),('dY','传入的输出梯度'),('dγ','当前 token 对各缩放权重的梯度'),('⊙','逐项相乘，不是矩阵乘法')])]
for key,title,d,n in [('qnorm','Query 归一化 / RMSNorm',1280,40),('kvnorm','KV 归一化 / RMSNorm',512,40),('anorm','Attention 输入归一化',5120,40),('fnorm','FFN 输入归一化',5120,40),('finalnorm','最终归一化 / Final RMSNorm',5120,1)]:
    s=norm_steps(d)
    card(key,title,'调整这组数字的整体大小，让数值更稳定，但不改变向量的维度。',s,
      param(s,d,rf'{d}\times1={d}',[(f'{n} 次独立权重',fmt(n*d)+' 个参数')]),
      forward(s,4*d+2,rf'4\times{d}+2={4*d+2}',[(f'当前 token 经过 {n} 次',fmt((4*d+2)*n)+' 次估算操作')],explain=f'平方、求和、两次逐项缩放共约 4×{d} 次乘加；求均值的除法和开平方倒数各简化计 1 次。',note='这是便于比较的算术估计，开平方等操作的实际耗时并不等于一次乘法。'),
      unknown(norm_backward(d),'复用哪些前向中间结果会影响反向的具体操作数，这里先展示正确的梯度关系。'),
      'RMSNorm 改变数字的大小，不改变数字的个数。')
DATA['finalnorm']['note']='进入这一块之前，先把当前 token 的四路残差合成一行 [1,5120]；这次合并另需 35,840 次乘加，没有新权重。'

qs=[mul(5120,1280,'W_{QA}','H',title='① 压缩到 1280 维'),step('② 稳定数值',r'H^{\prime}=\operatorname{RMSNorm}(H)',r'[1,1280]\longrightarrow[1,1280]',[('H','刚压缩得到的 1280 个数字'),('RMSNorm','调整数字整体大小的归一化操作'),('H′','归一化后的结果，数字个数不变')]),mul(1280,32768,'W_{QB}','Q',r'H^{\prime}',title='③ 展开成 64 个 Query Head')]
qs[-1]['symbols']=[('H′','归一化后的 1280 个数字'),('W_QB','1280 行、32768 列的学习权重'),('Q','展开后的 32768 个数字')]
qs[-1]['dimensions'].append(r'[1,32768]\longrightarrow[1,64,512]')
qs[-1]['note']='最后只是重新分组：64 个 Head，每个 Head 有 512 个数字，不增加乘法。'
qb=[step('① 先穿过展开矩阵',[r'dH^{\prime}=dQW_{QB}^{\mathsf T}',r'dW_{QB}=(H^{\prime})^{\mathsf T}dQ'],[r'[1,32768]\times[32768,1280]=[1,1280]',r'[1280,1]\times[1,32768]=[1280,32768]'],[('dQ','传给 64 个 Query Head 的梯度，先并成一行'),('W_QB','前向展开矩阵；ᵀ 交换行与列'),('H′','前向归一化后的向量'),('dH′ / dW_QB','归一化结果的梯度 / 展开权重的梯度')]),step('② 穿过中间的 RMSNorm',r'dH=\operatorname{RMSNormBackward}(dH^{\prime})',r'[1,1280]\longrightarrow[1,1280]',[('dH′','上一步得到的梯度'),('RMSNormBackward','使用前向保存的 H、缩放权重和比例来反传'),('dH','回到压缩结果的梯度')], '归一化本身也要反传，不能直接跳过；具体公式可点下方的 Query 归一化。'),step('③ 穿过压缩矩阵',[r'dX=dH W_{QA}^{\mathsf T}',r'dW_{QA}=X^{\mathsf T}dH'],[r'[1,1280]\times[1280,5120]=[1,5120]',r'[5120,1]\times[1,1280]=[5120,1280]'],[('X','当前 token 的原始输入'),('W_QA','压缩矩阵；ᵀ 表示转置'),('dH','压缩结果的梯度'),('dX / dW_QA','原输入的梯度 / 压缩权重的梯度')])]
card('query','Query 投影：先压缩，再展开','当前 token 先从 5120 维压到 1280 维，归一化后，再展开为 64 个 Query Head。',qs,
 param(qs,QP+R,r'5120\times1280+1280+1280\times32768=48497920',[('40 层 Query 含归一化',fmt((QP+R)*40)+' 个参数')],note='这个 Query 总览包含两套投影和中间 RMSNorm；子模块的引用不会再次加入模型总量。'),
 forward(qs,2*QP,[r'2\times1\times5120\times1280=13107200',r'2\times1\times1280\times32768=83886080'],[('40 层投影',fmt(2*QP*40)+' FLOPs')],note='这是两个矩阵的合计；中间 RMSNorm 另约 5,122 次操作。'),
 backward(qb,[('两套矩阵反向',fmt(4*QP)+' FLOPs'),('归一化反向','另算；未并入这个矩阵小计')],[r'2\times(13107200+83886080)=193986560'],[( '40 层矩阵反向',fmt(4*QP*40)+' FLOPs')],note='这里的 2 表示每个矩阵都要计算输入梯度和权重梯度；若同一中间向量还供 Indexer 使用，还要合并那条路径的梯度。'),
 '最终是一个 token 的 64 个 Query Head，不是 64 个 token。',related=['qa','qnorm','qb'])

os=[step('① 每 8 个 Head 放在一组',r'Z_g=O_g\times W_{OA,g}',[r'[1,64,512]\longrightarrow[1,8,4096]',r'[1,4096]\times[4096,1024]=[1,1024]'],[('O_g','第 g 组的 4096 个输入数字；g 从 1 到 8'),('W_OA,g','这一组自己的学习矩阵，8 组各一套'),('Z_g','该组得到的 1024 个数字')]),step('② 拼接，再回到模型宽度',r'Y=Z\times W_{OB}',[r'8\text{ groups of }[1,1024]\longrightarrow[1,8192]',r'[1,8192]\times[8192,5120]=[1,5120]'],[('Z','把 8 组结果按顺序拼成一行'),('W_OB','8192 行、5120 列的权重'),('Y','返回给后面模块的 5120 个数字')])]
ob=[step('① 先回到拼接后的向量',[r'dZ=dY W_{OB}^{\mathsf T}',r'dW_{OB}=Z^{\mathsf T}dY'],[r'[1,5120]\times[5120,8192]=[1,8192]',r'[8192,1]\times[1,5120]=[8192,5120]'],[('dY','输出传来的梯度'),('Z / W_OB','前向拼接向量 / 最后一个投影矩阵'),('dZ / dW_OB','拼接向量的梯度 / 权重梯度'),('ᵀ','交换矩阵的行与列')]),step('② 分回 8 组，各自反传',[r'dO_g=dZ_g W_{OA,g}^{\mathsf T}',r'dW_{OA,g}=O_g^{\mathsf T}dZ_g'],[r'[1,1024]\times[1024,4096]=[1,4096]',r'[4096,1]\times[1,1024]=[4096,1024]'],[('g','第 1 到第 8 组'),('O_g / W_OA,g','前向该组输入 / 权重'),('dZ_g','把 dZ 分组后得到的梯度'),('dO_g / dW_OA,g','该组输入梯度 / 权重梯度'),('ᵀ','转置')])]
card('output','Attention 输出投影','先把注意力输出做逆向位置旋转，再用 8 组小矩阵压缩，最后回到 5120 维。',os,
 param(os,OP,r'8\times4096\times1024+8192\times5120=75497472',[('40 层',fmt(OP*40)+' 个参数')]),
 forward(os,2*OP,[r'8\times(2\times1\times4096\times1024)=67108864',r'2\times1\times8192\times5120=83886080'],[('40 层',fmt(2*OP*40)+' FLOPs')],note='逆 RoPE 另需 12,288 次乘加，放在 RoPE 中计数。8 组矩阵之间不互相连接。'),
 backward(ob,[('两级矩阵反向',fmt(4*OP)+' FLOPs')],[r'2\times150994944=301989888'],note='该小计只含矩阵；还要让梯度经过逆 RoPE 对应的反向旋转。'),
 '8 组各自压缩，不是把所有 Head 连成一个大矩阵。',related=['rope'])

rs=[step('每两个数字做一次旋转',r'[u^{\prime},v^{\prime}]=[u,v]\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix}',r'[1,2]\times[2,2]=[1,2]',[('u、v','一对相邻的特征数字'),('θ','根据当前 token 位置确定的角度'),('u′、v′','旋转后的两个数字，数量不变')], '每个 Head 只旋转最后 64 个数字，也就是 32 对；角度已提前准备好。')]
rb=[step('把梯度沿相反方向旋转',r'[du,dv]=[du^{\prime},dv^{\prime}]R_\theta^{\mathsf T}',r'[1,2]\times[2,2]=[1,2]',[('du′、dv′','旋转结果传来的梯度'),('Rθ','前向使用的 2×2 旋转矩阵'),('ᵀ','转置；旋转矩阵的转置就是逆向旋转'),('du、dv','原来两个数字的梯度')])]
card('rope','位置旋转 / RoPE','让当前 token 的向量带上位置信息。旋转不改变维度，也没有新权重。',rs,noparam(rs),forward(rs,12288,r'64\times32\times6=12288',[('本层 local KV 的旋转','192 FLOPs'),('本层输出的逆旋转','12,288 FLOPs'),('一层这三类旋转','24,768 FLOPs'),('40 层这三类旋转','990,720 FLOPs')],explain='64 个 Query Head，每个旋转 32 对数字；一对需要 4 次乘法和 2 次加减。',note='上方先计算 Query 的旋转；压缩 KV 和 Indexer 的旋转分别计入它们自己的模块。'),backward(rb,[('当前 Query 的反向旋转','12,288 FLOPs')],explain='旋转矩阵固定，只求输入梯度，不需要计算角度权重的梯度。'),'旋转只是变换当前 token 的数字，不会生成更多 token。')

def attention(key,title,count,selected,note):
    s=[step('① 当前 Query 和可见的 Key 比较',r'S=Q\times K^{\mathsf T}/\sqrt{512}',rf'[1,512]\times[512,{selected}]=[1,{selected}]',[('Q','当前 token 的一个 Query Head，512 个数字'),('K',f'可读取的 {selected} 行 Key，每行 512 个数字'),('ᵀ','转置后才能按特征维相乘'),('S','每个可见位置与当前 Query 的相似程度'),('√512','缩放分数，避免数字过大')]),step('② 按相似程度汇总 Value',r'O=P\times V',rf'[1,{selected}]\times[{selected},512]=[1,512]',[('P','把分数归一化得到的各位置权重'),('V','这些位置的 Value，与 Key 共用同一份 512 维向量'),('O','这一个 Query Head 汇总得到的 512 个数字')], '64 个 Head 都做这两步；历史 Key/Value 已缓存，不重新计算历史 token 的投影。')]
    ss=[step('① 从汇总结果回到权重和 Value',[r'dP=dO V^{\mathsf T}',r'dV=P^{\mathsf T}dO'],[rf'[1,512]\times[512,{selected}]=[1,{selected}]',rf'[{selected},1]\times[1,512]=[{selected},512]'],[('dO','当前 Head 的输出梯度'),('V / P','前向读取的 Value / 注意力权重'),('dP / dV','权重梯度 / 被读取的各个 Value 的梯度'),('ᵀ','转置')]),step('② 通过归一化回到分数',r'dS=P\odot\left(dP-\sum_j P_jdP_j\right)',rf'[1,{selected}]\longrightarrow[1,{selected}]',[('P / dP','前向权重 / 上一步的权重梯度'),('Σⱼ','对这一 Query 读取的位置求和'),('⊙','逐项相乘'),('dS','相似度分数的梯度')]),step('③ 回到当前 Query 和缓存的 Key',[r'dQ=dS K/\sqrt{512}',r'dK=dS^{\mathsf T}Q/\sqrt{512}'],[rf'[1,{selected}]\times[{selected},512]=[1,512]',rf'[{selected},1]\times[1,512]=[{selected},512]'],[('Q / K','前向当前 Query / 缓存的 Key'),('dS','分数的梯度'),('dQ / dK','当前 Query 的梯度 / 各 Key 的梯度'),('ᵀ、√512','转置，以及与前向一致的缩放')])]
    gemm=4*512*selected*64
    from FLOPs_calculation_verify import attention_kernel
    total=sum(attention_kernel(1,selected//64))
    card(key,title,f'只跟着当前 token 的 Query 走，让它从缓存中读取相关信息。这里先看 1 个 Head，再合计 64 个 Head。',s,
      mode(s,[],[('比较与汇总','不新增投影权重'),('每层的 64 个 sink 数字','64 个可学习参数')],explain='Query/KV 的权重已在投影模块计算；sink 只影响归一化，不携带一行 Value。',totals=[(f'{count} 层 sink',fmt(count*64)+' 个参数')]),
      forward(s,gemm,rf'2\times(2\times1\times512\times{selected})\times64={gemm}',[(f'含缩放、归一化等的一层估算',fmt(total)+' 次操作'),(f'同类 {count} 层的估算',fmt(total*count)+' 次操作')],explain='内层的 2 是乘加计数；外层的 2 表示先算 Query×Key，再算权重×Value；最后乘 64 个 Head。',note=note+' 先显示两个矩阵乘法的计算量，归一化等工作放在展开项里。'),
      backward(ss,[('四个梯度矩阵乘法 · 64 Heads',fmt(2*gemm)+' FLOPs'),('完整 attention 反向','不能由当前 token 单独确定')],[rf'4\times(2\times1\times512\times{selected})\times64={2*gemm}'],note='这是当前 Query 的局部梯度。历史 KV 还会收到其他 Query 的贡献；归一化、sink 梯度和缓存中的累加需另算。K/V 共用向量，所以还要把 dK 与 dV 相加。'),
      '只有当前 token 生成 Query；它可以读取很多历史位置。',related=['query','kv','output','compressor','indexer'],note=note)
attention('sliding','滑动窗口注意力',2,128,'假设此前已有 1024 个 token；窗口读取当前 token 加最近 127 个，共 128 个位置。这里只算 layers 0–1 的纯局部注意力。')
attention('sparse','压缩 KV 稀疏注意力',38,640,'假设此前已有 1024 个 token；128 个局部位置加上选中的 512 个压缩位置，共 640 行 KV。它们共同归一化。')
attention('multi','多来源注意力',20,640,'假设此前已有 1024 个 token；当前 Query 同时读取 128 个局部位置与选中的 512 个共享 KV 位置。')
DATA['sparse']['related']=['compressor','indexer','query','output']

# Compressor examples are deliberately the only new-token multirow illustrations.
cs=[mul(5120,512,'W_{KV}','Z',title='① 当前 token 先产生 512 个数字'),mul(5120,512,'W_{gate}','G',title='② ratio=2 再计算混合分数')]
cs[1]['note']='前一个 token 的结果保存在缓存里。凑齐两个 token 才做一次合并；若当前是新一对中的第一个，就先等待。'
examples=[]
for n,out,txt in [(3,1,'前两个 token 合成一组，第三个还在等第四个。输出只有一行 [1,512]。'),(4,2,'第 1、2 个 token 合成第一组，第 3、4 个合成第二组。输出两行 [2,512]。')]:
    examples.append(dict(title=f'分组小例子：{n} 个 token，ratio=2',steps=[step('先逐行投影',r'Z=X\times W_{KV}',rf'[{n},5120]\times[5120,512]=[{n},512]',[('X',f'用于说明分组的 {n} 个 token，每个 5120 维'),('W_KV','同一套 5120×512 权重，每行都使用它'),('Z','每个 token 各自投影后的结果')],txt),step('完整的一对合成一行',r'C_j=a_jZ_{1,j}+(1-a_j)Z_{2,j}',r'[2,512]\longrightarrow[1,512]',[('j','第 j 个特征，j 从 1 到 512'),('Z₁,j、Z₂,j','一对 token 在该特征上的数字'),('aⱼ','由 W_gate 的输出归一化得到的混合比例，每个特征都可不同'),('Cⱼ','合并后第 j 个数字')],f'这个 {n} token 示例总共形成 {out} 个完整分组。示例用于解释分组，不替代页面的当前单 token FLOPs。')]))
cb=[step('① 梯度先分给一组中的两个 token',[r'dZ_{1,j}=a_jdC_j',r'dZ_{2,j}=(1-a_j)dC_j',r'da_j=(Z_{1,j}-Z_{2,j})dC_j'],r'dC:[1,512]\longrightarrow dZ:[2,512]',[('dCⱼ','压缩结果第 j 个数字的梯度'),('aⱼ','前向用的第 j 个混合比例'),('Z₁,j、Z₂,j','这一对 token 的投影结果'),('dZ、da','两个输入分支的梯度 / 混合比例的梯度')]),step('② 当前 token 的投影反传',[r'dX=dZ W_{KV}^{\mathsf T}+dG W_{gate}^{\mathsf T}',r'dW_{KV}=X^{\mathsf T}dZ'],[r'[1,512]\times[512,5120]=[1,5120]',r'[5120,1]\times[1,512]=[5120,512]'],[('X','只取这一组中当前 token 的输入'),('dZ','分回当前 token 的投影梯度'),('dG','混合比例经过归一化反传到门控分数的梯度'),('W_KV、W_gate','前向两套矩阵；ᵀ 表示转置'),('dW_KV','当前 token 的 KV 权重梯度；门控权重同样用 XᵀdG')])]
card('compressor','KV 压缩 / Compressor','把相邻 token 的信息合到较少的向量中。ratio=2 每两个合成一个；ratio=1 每个 token 自己产生一个。',cs,
 param(cs,5243392,r'2\times5120\times512+512=5243392',[('3 套 ratio=2 + 1 套 ratio=1','18,352,128 个参数')],note='ratio=2 有两套矩阵，再加 512 个归一化权重；ratio=1 没有 W_gate，共 2,621,952 个参数。'),
 forward(cs,10485760,r'2\times(2\times1\times5120\times512)=10485760',[('凑齐一对时：合并、归一化、位置旋转','另约 7,362 次操作'),('ratio=1 当前 token 的完整计算','约 5,245,122 次操作'),('当前第 1025 个 token：全部四套','约 36,702,402 次操作')],note='已有 1024 个历史 token 时，当前是第 1025 个：ratio=2 只投影并等待，不产生新压缩向量；ratio=1 正常产生一行。',explain='两套投影各算 2×1×5120×512。之前 token 的投影已缓存，不重算。'),
 unknown(cb,'如果当前只完成半组，就要等下一 token 到来后才能得到该组的完整梯度；历史项和归一化的反传还需一起处理。'),
 '分组决定何时产生压缩向量，不决定当前 token 是否需要先投影。',related=['global','indexer'])
examples.append(dict(title='ratio=1：不用等待配对',steps=[mul(5120,512,'W_{KV}','C',title='每个 token 各自产生一行')]))
examples[-1]['steps'][0]['note']='没有 W_gate，也不混合两个 token；当前 token 投影后，经过归一化和位置旋转就产生一行 [1,512]。'
for m in ['parameter','forward','backward']:DATA['compressor'][m]['extras']=deepcopy(examples)
gs=[mul(5120,512,'W_{KV}','C'),step('再调整数值与位置',r'KV=\operatorname{RoPE}(\operatorname{RMSNorm}(C))',r'[1,512]\longrightarrow[1,512]',[('C','当前 token 刚投影出的 512 个数字'),('RMSNorm / RoPE','调整大小 / 添加位置信息'),('KV','这一 token 产生的新共享向量')])]
card('global','全局 KV 投影 / ratio=1','当前 token 产生一个 512 维共享向量，后续 Decoder 层可以反复读取它。这里不把多个 token 两两合并。',gs,param(gs,2621952,r'5120\times512+512=2621952',note='只保存这一套共享投影和它的归一化权重，不因后面 20 层读取它而存 20 份。'),forward(gs,5242880,r'2\times1\times5120\times512=5242880',[('加上归一化与位置旋转','约 5,245,122 次操作')]),unknown(grad_steps(5120,512,'W_{KV}',y='C'),'多个 Decoder 层都读取这个向量，所以要先合并它们的梯度，再经过位置旋转、归一化和这里的投影反传。'), '新 token 只产生一个新向量；缓存中可以已经存着许多历史向量。',related=['compressor','multi'])

ix=[mul(1280,4096,'W_I','Q_I',title='① 生成 32 个小 Query'),mul(5120,32,'W_h','w',title='② 决定各 Head 的打分权重'),step('③ 与已缓存的索引 Key 比较',r'S=Q_I\times K_I^{\mathsf T}',r'[1,128]\times[128,1025]=[1,1025]',[('Q_I','先看其中一个小 Query，128 维'),('K_I','1024 个历史项加当前项，共 1025 个缓存索引 Key'),('ᵀ','转置，按 128 个特征相乘'),('S','这一 Head 对每个候选位置的分数')], '32 个 Head 分别打分后，按上一步的权重 w 合成分数，再选最多 512 个位置。')]
ix[0]['formulas']=[r'Q_I=H^{\prime}\times W_I']
ix[0]['symbols']=[('H′','当前 token 的 1280 维归一化 Query 中间向量'),('W_I','1280×4096 的索引 Query 权重'),('Q_I','4096 个数字，分成 32 个 128 维 Head')]
ix[1]['symbols']=[('X','当前 token 的 5120 维输入'),('W_h','5120×32 的学习矩阵'),('w','32 个 Head 的分数混合权重')]
ixback=[step('选中位置后，取出向量',r'V_{sel}=\operatorname{Gather}(V,\mathcal I)',r'V:[1025,512]\longrightarrow V_{sel}:[512,512]',[('V','缓存里的候选向量'),('ℐ','选出的整数位置编号'),('Gather','按编号取出对应行'),('V_sel','实际交给 attention 的向量')], '反向把选中行的梯度加回 V。整数编号本身不能像连续数字一样求导。')]
card('indexer','挑选历史位置 / Indexer','先用较小的 Query 给历史位置打分，挑出值得读取的位置。主 Attention 随后才计算自己的注意力权重。',ix,
 param(ix,5472384,r'1280\times4096+5120\times32+512\times128+128=5472384',[('全部 8 个 Indexer','43,516,416 个参数')],note='这里以还负责生成新索引 Key 的一套为例；只有 4 套带 Key 投影和归一化，其余复用已有 Key。'),
 mode(ix,[r'2\times1\times128\times1025\times32=8396800'],[('一次：32 Heads 与历史比较','8,396,800 FLOPs')],'1 是当前 Query；128 是每个小 Head 的宽度；1025 是候选历史/当前位置数；32 是 Head 数。',totals=[('当前 Query 投影','10,485,760 FLOPs'),('32 个 Head 权重','327,680 FLOPs'),('本次新 Key 投影','131,072 FLOPs'),('此步全部 8 个 Indexer','约 141,675,261 次操作')],note='这里以 layer 20 的 1025 个候选为例；ratio=2 的候选只有 512 个。历史 Key 已缓存，只有新产生的 Key 需要投影。'),
 unknown(ixback,'只通过整数位置去取值，不会把梯度传回这些打分矩阵；如何训练选址本身，需要另有可微的学习方式才能确定。'),
 'Indexer 选位置，Attention 再从那些位置读取信息。',related=['sparse','selection'])

# Four residual streams remain four, even though only one token is followed.
hcs=[step('① 为这一个 token 生成混合系数',[r'r=\operatorname{rsqrt}(\operatorname{mean}(R_{flat}^2)+\epsilon)',r'L=(R_{flat}\times W_{HC})r'],r'[1,20480]\times[20480,24]=[1,24]',[('R_flat','当前 token 的 4 路向量拼成一行，4×5120=20480'),('W_HC','学习得到的 20480×24 权重'),('r、ε','该拼接向量算出的缩放比例 / 稳定小数，没有额外 γ'),('L','24 个分数，经过 base、scale 及 Sinkhorn 调整后产生 4+4+16 个比例')], '本次生成的 pre 比例交给下一子模块；当前子模块的输入合并使用前一子模块传来的 pre_mix。'),step('② 合成一行，送入 Attention 或 FFN',r'X=a_1R_1+a_2R_2+a_3R_3+a_4R_4',r'4\text{ vectors }[1,5120]\longrightarrow[1,5120]',[('R₁…R₄','同一个 token 的四路向量'),('a₁…a₄','前一子模块产生的输入比例；模型开头使用固定 one-hot'),('X','送入当前子模块的一行向量')]),step('③ 将子模块结果送回四路',r'Y_j=b_jU+\sum_{i=1}^{4}R_i C_{ij}',r'Y:[1,4,5120],\quad U:[1,5120]',[('U','Attention 或 FFN 刚算出的结果'),('bⱼ','这一结果送给第 j 路的比例'),('Cᵢⱼ','第 i 路旧状态送给第 j 路；行是输入路、列是输出路'),('Σᵢ','把四路的贡献相加；Yⱼ 是第 j 路的新状态')])]
hcb=[step('① 梯度分回子模块和四路旧状态',[r'dU=\sum_{j=1}^{4}b_jdY_j',r'dR_i=\sum_{j=1}^{4}C_{ij}dY_j'],[r'dY:[1,4,5120]\longrightarrow dU:[1,5120]',r'dR:[1,4,5120]'],[('dYⱼ','第 j 路新状态传来的梯度'),('bⱼ / Cᵢⱼ','前向的子模块比例 / 输入路 i 到输出路 j 的比例'),('dU / dRᵢ','子模块输出的梯度 / 第 i 路旧状态的直接梯度'),('Σⱼ','合并四路的贡献；生成比例的间接路径仍要另算')]),step('② 混合权重也需要梯度',[r'dW_{HC}=R_{flat}^{\mathsf T}dL',r'dR_{flat}=dLW_{HC}^{\mathsf T}'],[r'[20480,1]\times[1,24]=[20480,24]',r'[1,24]\times[24,20480]=[1,20480]'],[('dL','经过比例变换、反复归一化和 r 缩放后，回到线性投影的梯度'),('R_flat / W_HC','前向的四路拼接输入 / 混合矩阵'),('dW_HC / dR_flat','混合矩阵的梯度 / 投影路径传回四路的梯度，r 的路径需另加'),('ᵀ','转置')])]
card('hc','四路残差混合 / Hyper-Connection','一个 token 同时保留四路 5120 维向量。HC 决定怎样把四路合成一行，以及怎样把新结果混回四路。',hcs,
 param(hcs,HP,r'20480\times24+24+3=491547',[('每层两套 HC','983,094 个参数'),('40 层共 80 套','39,323,760 个参数')],note='矩阵外还有 24 个 base 与 3 个 scale 权重；本次算出的混合比例不是新增参数。'),
 forward(hcs,983040,r'2\times1\times20480\times24=983040',[('系数调整、四路合并等也计入','每套约 1,246,042 次操作'),('当前 token 经过 80 套','约 99,683,360 次操作')],note='先展示最主要的权重投影；四路向量相加与混合比例调整也有计算，不能当作零。'),
 backward(hcb,[('只计系数矩阵的两个梯度','1,966,080 FLOPs'),('完整 HC 反向','尚不能确定')],[r'2\times983040=1966080'],note='还要计算混合比例本身的梯度，穿过每次比例调整，并把各路径的梯度加起来；不能用矩阵小计代替完整 HC。'),
 '四路是同一个 token 的四份状态，不是四个 token。',related=['streams'])

rt=[mul(5120,384,'W_R','L',title='① 给 384 个专家打分'),step('② 选 6 个，再算混合权重',r's=\sqrt{\operatorname{softplus}(L)}',r'[1,384]\longrightarrow\text{6 IDs + 6 weights}',[('L','384 个原始分数'),('softplus、平方根','把原始分数变成正数'),('s','用来算路由权重的分数')], '选专家时会对分数加偏置；算混合权重时使用未加偏置的分数。')]
rback=grad_steps(5120,384,'W_R',y='L')
rback[0]['note']='从 6 个选中分数的权重梯度开始，先经过归一化与分数变换，得到这里的 dL。未选中的整数编号不求导。'
card('router','专家路由 / MoE Router','用当前 token 的 5120 个数字给 384 个专家打分，决定这一次交给哪 6 个路由专家。',rt,
 param(rt,1966848,r'5120\times384+384+384=1966848',[('40 层','78,673,920 个参数')],note='两组 384 维偏置分别用于文本和图像选择；这里只走文本路径，但存储参数要包含两组。'),
 forward(rt,3932160,r'2\times1\times5120\times384=3932160',[('加上分数变换与权重归一化','一层约 3,933,714 次操作'),('40 层完整路由估算','157,348,560 次操作')],note='Top-6 的比较与派发不是这次矩阵乘法，耗时另计。'),
 backward(rback,[('打分矩阵的输入、权重梯度','7,864,320 FLOPs'),('完整路由反向','需要再计分数变换与权重归一化')],[r'2\times3932160=7864320'],note='选择用的偏置只影响离散编号，不能通过编号得到普通梯度；学习时如何调整它另需规则。'),
 '384 是候选数量，6 才是当前 token 要执行的路由专家数。',related=['selection','weights'])
wt=[step('把选中的 6 个分数变成比例',r'w_i=1.5\frac{s_i}{s_1+s_2+\cdots+s_6+\epsilon}',r'[1,6]\longrightarrow[1,6]',[('sᵢ','选中专家的原始正分数，不带选择偏置'),('wᵢ','这个专家实际使用的混合权重'),('1.5','模型设置的整体放大比例'),('ε','避免分母为零的小数')])]
wback=[step('先算共同的分母',r'S=s_1+s_2+\cdots+s_6+\epsilon',r'[1,6]\longrightarrow[1,1]',[('s₁…s₆','选中的六个原始分数'),('ε','很小的稳定数'),('S','六项共同的分母')]),step('再把梯度传给每个分数',r'ds_j=\frac{1.5}{S}dw_j-\frac{1.5}{S^2}\sum_i dw_i s_i',r'[1,6]\longrightarrow[1,6]',[('dwᵢ','各个专家混合权重对 Loss 的影响'),('sᵢ / S','前向选中分数 / 分母'),('Σᵢ','把六项相加'),('dsⱼ','第 j 个原始分数的梯度')])]
card('weights','路由权重归一化','先把 6 个分数除以它们的总和，再统一乘 1.5。',wt,noparam(wt),forward(wt,18,r'5+1+6+6=18',explain='6 个数求和用 5 次加法；加小数 1 次；6 次除法和 6 次缩放。除法在此简化按一次操作计。',note='这 18 次已经包含在 Router 完整计算里，不要再加一遍。'),unknown(wback,'各个专家传来的梯度需要先到齐，再一起经过这个归一化；这里只展示局部梯度公式。'),'选中的编号负责派发，选中的分数负责混合。')
st=[step('按分数挑选六个编号',r'\mathcal I=\operatorname{Top6}(s+b)',r'[1,384]\longrightarrow[1,6]\text{ integer IDs}',[('s','384 个专家分数'),('b','选择偏置，帮助决定选谁'),('Top6','挑出分数最高的六项'),('ℐ','六个整数专家编号')])]
card('selection','Top-6 专家选择','挑选当前 token 要访问的 6 个专家；这一步只决定编号。',st,noparam(st,'分数权重和偏置已经在 Router 里计算，不新增一套参数。'),info(st,[('当前工作','比较 384 个分数并选出 6 个编号')],'比较和派发不计入矩阵 FLOPs，但仍有实际计算与搬运成本。'),info(st,[('整数编号','不能像连续数字一样求导'),('选中的专家输出','仍然正常接收梯度')],'是否选中一个专家和选中后怎样计算，是两件事。'),'挑选六个不代表只存六个；模型仍存着全部 384 个路由专家。')

es=[step('① 同一个输入走两条投影',[r'G=XW_1',r'U=XW_3'],r'[1,5120]\times[5120,2304]=[1,2304]',[('X','当前 token 的 5120 维输入'),('W₁ / W₃','两套独立学习权重，大小相同'),('G / U','门控分支 / 内容分支，各有 2304 个数字')]),step('② 逐项相乘形成门控结果',r'Z=\operatorname{SiLU}(G_c)\odot U_c',r'[1,2304]\odot[1,2304]=[1,2304]',[('G_c、U_c','把过大的分支数字截断后的结果'),('SiLU','让 Gate 数字产生平滑的控制作用'),('⊙','对应位置相乘'),('Z','控制后的 2304 个数字')]),step('③ 乘路由比例，再返回 5120 维',r'Y=(wZ)W_2',r'[1,2304]\times[2304,5120]=[1,5120]',[('w','Router 给这一个专家的混合比例'),('Z','上一步的 2304 维向量'),('W₂','返回模型宽度的权重'),('Y','这一专家对当前 token 的贡献')])]
eb=[step('① 先从输出回到中间结果',[r'dA=dY W_2^{\mathsf T}',r'dW_2=A^{\mathsf T}dY',r'dZ=w\,dA'],[r'[1,5120]\times[5120,2304]=[1,2304]',r'[2304,1]\times[1,5120]=[2304,5120]'],[('A=wZ','前向送入最后矩阵的加权向量'),('dY','当前专家的输出梯度'),('W₂ / w','最后投影权重 / 路由比例'),('dA、dZ、dW₂','加权向量梯度、激活梯度和投影权重梯度'),('ᵀ','转置')]),step('② 穿过 SwiGLU，再从两分支回到输入',[r'dX=dG W_1^{\mathsf T}+dU W_3^{\mathsf T}',r'dW_1=X^{\mathsf T}dG,\quad dW_3=X^{\mathsf T}dU'],[r'[1,2304]\times[2304,5120]=[1,5120]',r'[5120,1]\times[1,2304]=[5120,2304]'],[('dG / dU','穿过 SwiGLU 和截断后，两条分支的梯度'),('X / W₁ / W₃','原输入 / Gate 矩阵 / Up 矩阵'),('dX','把两条分支传回的输入梯度相加'),('dW₁ / dW₃','分别求出的两套权重梯度'),('ᵀ','转置')])]
card('expert','专家网络 / Expert FFN','一个专家先展开两条 2304 维分支，用其中一条控制另一条，再回到 5120 维。当前 token 执行 6 个这样的路由专家。',es,
 param(es,EP,r'3\times5120\times2304=35389440',[('一层存储 384 个路由专家','13,589,544,960 个参数'),('40 层所有路由专家','543,581,798,400 个参数'),('当前 token 每层用到 6 个','212,336,640 个专家权重')]),
 forward(es,2*EP,r'3\times(2\times1\times5120\times2304)=70778880',[('六个路由专家的矩阵','424,673,280 FLOPs'),('单专家加上激活与路由乘权','约 70,788,096 次操作')],note='主结果是一个专家的三次矩阵乘法；当前 token 只执行被选中的六个，不执行全部 384 个。'),
 backward(eb,[('单专家三套矩阵的反向','141,557,760 FLOPs'),('六个路由专家的矩阵反向','849,346,560 FLOPs')],[r'2\times70778880=141557760'],note='还需要 SwiGLU、截断、路由权重及两分支相加的计算；这里的数字只包含矩阵乘法。'),
 '每个专家存三套矩阵；一次只为当前 token 选中的专家做运算。',related=['swiglu','shared','weights'])
sh=deepcopy(es);sh[-1]=step('③ 直接返回 5120 维',r'Y=ZW_2',r'[1,2304]\times[2304,5120]=[1,5120]',[('Z','共享专家的门控结果'),('W₂','该专家自己的 Down 投影权重'),('Y','共享专家给出的结果')], '共享专家不乘 Router 的混合权重。')
shb=deepcopy(eb);shb[0]=step('① 从输出反传到中间结果',[r'dZ=dY W_2^{\mathsf T}',r'dW_2=Z^{\mathsf T}dY'],[r'[1,5120]\times[5120,2304]=[1,2304]',r'[2304,1]\times[1,5120]=[2304,5120]'],[('Z / W₂','前向门控结果 / 最后投影矩阵'),('dY / dZ','输出梯度 / 门控结果梯度'),('dW₂','当前 token 的 Down 权重梯度'),('ᵀ','转置')])
card('shared','共享专家 / Shared Expert','每个 token 都执行这个专家，不需要被 Router 选中。它与六个路由专家分别读取同一输入。',sh,param(sh,EP,r'3\times5120\times2304=35389440',[('40 层各一个','1,415,577,600 个参数')]),forward(sh,2*EP,r'3\times(2\times1\times5120\times2304)=70778880',[('加上 SwiGLU','约 70,785,792 次操作')]),backward(shb,[('三套矩阵的反向','141,557,760 FLOPs')],[r'2\times70778880=141557760'],note='激活和两分支梯度相加另算；这里没有路由比例的梯度。'),'“共享”是所有 token 都会执行，不是 40 层共用同一套专家权重。',related=['swiglu'])
sw=[step('让 Gate 控制每个位置',r'Z=G_c\odot\sigma(G_c)\odot U_c',r'[1,2304]\odot[1,2304]\odot[1,2304]=[1,2304]',[('G_c','Gate 分支：大于 10 的数字截到 10'),('U_c','Up 分支：数字限制在 −10 到 10 之间'),('σ','sigmoid，把每个输入变为 0 到 1 的数'),('⊙','逐项相乘；Z 是门控后的向量')])]
swb=[step('梯度分别返回 Gate 和 Up',[r's=\sigma(G_c)',r'dU_c=dZ\odot(G_c\odot s)',r'dG_c=dZ\odot U_c\odot[s+G_c\odot s\odot(1-s)]'],r'dZ:[1,2304]\longrightarrow dG_c,dU_c:[1,2304]',[('s','前向 sigmoid 的结果'),('G_c / U_c','截断后的 Gate / Up 输入'),('dZ','门控结果的梯度'),('dG_c / dU_c','分给 Gate 和 Up 的两份梯度'),('⊙','对应位置相乘')], '穿过截断时，超出限制的部分不再传梯度；刚好在边界处怎样处理由实际训练规则决定。')]
card('swiglu','逐项门控 / SwiGLU','两条分支逐项相乘：Gate 决定 Up 的每个位置保留多少。',sw,noparam(sw),forward(sw,6912,r'2304\times(1+1+1)=6912',explain='每个位置算一次 sigmoid、两次乘法；这里把 sigmoid 简化记为一次操作。',note='这只是一个专家的激活，不含前后投影；截断比较与实际 sigmoid 耗时另计。'),unknown(swb,'可以明确两条分支如何获得梯度，但复用前向结果和截断处理会影响具体操作数。'),'Gate 分支和 Up 分支都要得到梯度，不能只反传其中一条。')
co=[step('将六个路由结果和一个共享结果相加',r'Y=Y_1+Y_2+\cdots+Y_6+Y_s',r'7\text{ vectors }[1,5120]\longrightarrow[1,5120]',[('Y₁…Y₆','六个已经乘过路由比例的专家输出'),('Y_s','不乘路由比例的共享专家输出'),('Y','同一个 token 的合并结果')])]
cob=[step('把同一份梯度传给每条相加分支',r'dY_1=\cdots=dY_6=dY_s=dY',r'[1,5120]\longrightarrow7\text{ branches of }[1,5120]',[('dY','合并结果传来的梯度'),('dY₁…dY₆、dY_s','分别交给七个专家输出的梯度')], '专家内部再按各自权重继续反传；六个 routed 分支的路由比例也在里面处理。')]
card('combine','合并专家结果','将当前 token 的七份专家贡献合成一行，交给后面的 HC。',co,noparam(co),forward(co,35840,r'7\times5120=35840',explain='实现从全零结果开始，依次加 6 个路由结果，再加 1 个共享结果，共 7 次逐项相加。'),info(cob,[('相加的反向','向每个分支传入相同梯度，不新增权重')]),'合并后的宽度仍然是 5120。')
ms=[step('一个 token 经过七个专家',r'Y=\sum_{i\in\mathcal I}w_i\operatorname{Expert}_i(X)+\operatorname{Shared}(X)',r'[1,5120]\longrightarrow7\text{ expert paths}\longrightarrow[1,5120]',[('X','当前 token 的输入'),('ℐ','Router 选中的 6 个专家编号'),('wᵢ','相应的混合比例'),('Expertᵢ / Shared','第 i 个路由专家 / 共享专家'),('Σ、Y','把六个加权结果相加，再加共享结果，得到 Y')], '示意图纵向排列这些步骤，但共享专家与路由专家读取同一个输入，不是把前一个专家输出再送给共享专家。')]
card('moe','专家混合 / MoE','当前 token 只执行 6 个路由专家和 1 个共享专家，再把它们的结果相加。',ms,
 param(ms,13624934400,r'(384+1)\times35389440=13624934400',[('40 层全部专家','544,997,376,000 个参数'),('一层当前 token 激活的专家权重','247,726,080 个参数')],note='主结果只计专家矩阵；Router 的参数另外计算。存储用 384+1，执行用 6+1。'),
 forward(ms,495452160,r'(6+1)\times70778880=495452160',[('加激活、路由乘权、结果合并','一层专家约 495,550,208 次操作'),('再加 Router','一层 MoE 约 499,483,922 次操作'),('当前 token 经过 40 层 MoE','约 19,979,356,880 次操作')],explain='每个专家三次矩阵乘法，共 70,778,880 FLOPs；当前 token 做 7 份。'),
 info([step('沿实际执行的七条路径反传',r'dX=\sum_{i\in\mathcal I}dX_i+dX_s',r'7\text{ gradients }[1,5120]\longrightarrow[1,5120]',[('ℐ','实际选中的六个专家'),('dXᵢ / dX_s','每个路由专家 / 共享专家返回的输入梯度'),('Σ、dX','合并这些梯度，交给前面的模块')])],[('七个专家的矩阵反向','990,904,320 FLOPs'),('完整 MoE 反向','还需激活、Router 与路由权重的反传')],'没有执行的专家不因“存储在那里”而计算这一个 token 的梯度。'),
 '“存了多少专家”与“这次用了多少专家”是两个不同问题。',related=['router','selection','expert','shared','combine'])

# Discrete stages use actual indexing rather than fictitious matrix products.
tok=[step('取当前 token 的编号',r'u=\operatorname{Tokenizer}(\text{text})_{current}',r'\text{current token}\longrightarrow[1]\text{ integer}',[('Tokenizer','把文字拆分并转换为词表编号的工具'),('u','当前所跟踪的一个 token 编号')], '图中 It / is / raining 只是文字示意，真实分词不一定恰好一词一个 token。')]
for key,title,purpose in [('input','输入文字','文字会先被分词；后面的教学只跟踪其中一个 token。'),('tokenizer','分词 / Tokenizer','把文字转换成 token 编号，再取当前正在处理的那个 token。'),('ids','当前 token 编号','一个整数告诉模型到词表的哪一行取出向量。')]:
    card(key,title,purpose,tok,noparam(tok),info(tok,[('这一步','文字、整数与查索引操作')],'不是浮点矩阵乘法，不能用参数量来充当 FLOPs。'),info(tok,[('整数 token 编号','不求连续梯度')],'后面的向量和权重可以得到梯度，编号本身不求导。'),'token 的编号不是它的 5120 维向量。')
emb=[step('按编号读取词表的一行',r'X=E[u,:]',r'E:[129280,5120],\quad u:[1]\longrightarrow X:[1,5120]',[('E','模型学习得到的词表，每行 5120 个数字'),('u','当前 token 的整数编号'),('X','从该编号对应的行取出的 5120 维向量')], '这是查表，不是用 token 编号去乘矩阵。5120 是这个模型选定的特征宽度。')]
embb=[step('把梯度加回被取出的那一行',r'dE[u,:]\mathrel{+}=dX',r'dX:[1,5120]\longrightarrow\text{one row of }dE:[129280,5120]',[('dX','当前向量传回的梯度'),('u','前向读取的同一个整数编号'),('dE','词表权重的梯度；这里只更新编号 u 对应行的贡献'),('+=','累加，若别处也用同一行，贡献要加在一起')])]
card('embedding','词嵌入 / Token Embedding','从学习好的词表里，取出当前 token 对应的 5120 个数字。',emb,param(emb,661913600,r'129280\times5120=661913600',[('当前 token 读取','1 行，共 5,120 个权重值')]),info(emb,[('浮点乘加','0；这一动作是取一行数据')],'查表仍要访问内存。得到向量后会复制成四路残差，不是又处理了四个 token。'),info(embb,[('梯度去向','被读取的那一行'),('完整操作数','取决于是否与其他梯度合并')],'当前 token 的一次贡献不等于整张词表在一个训练步骤中的所有梯度。'),'表可以很大，但一个 token 只取一行。')

ng=[step('当前编号加上前面的编号',r'(u_{-1},u_0),\ (u_{-2},u_{-1},u_0),\ (u_{-3},u_{-2},u_{-1},u_0)',r'1\text{ current ID}+\text{up to 3 previous IDs}',[('u₀','当前 token 编号'),('u₋₁、u₋₂、u₋₃','此前三个 token 的编号'),('2 / 3 / 4-gram','分别用最近 2、3、4 个编号组成片段')], '序列开头不够长时使用填充编号；当前 token 仍只有一个。')]
card('ngram','构造 N-gram','把当前 token 的编号和前面最多三个编号放在一起，形成三个长度不同的短片段。',ng,noparam(ng),info(ng,[('当前工作','读取少量历史整数编号并组成片段')],'整数处理不算浮点矩阵 FLOPs，但也需要计算与数据访问。'),info(ng,[('编号与分组','不求连续梯度')]),'这里使用历史编号，不重新计算历史 token 的隐藏向量。')
ha=[step('把短片段映射为查表地址',r'\mathcal I=\operatorname{Hash}(\text{n-grams})',r'3\text{ lengths}\times8\text{ hash heads}=24\text{ IDs}',[('Hash','由固定整数规则生成表地址'),('n-grams','当前 token 结尾的三种长度片段'),('ℐ','本次要读取的 24 个编号')])]
card('hash','哈希与查找 / Hashing & Lookup','把三个长度的 N-gram 映射到记忆表里的 24 个地址。',ha,noparam(ha,'哈希规则不学习；记忆表参数在下一步计算。'),info(ha,[('当前工作','算出 24 个整数地址')],'整数哈希与访存不列入浮点 FLOPs，不能理解为没有运行成本。'),info(ha,[('地址','不求梯度'),('地址取出的数值','可以收到后面传回的梯度')]),'地址是离散的，但表中读出的向量可以学习。',related=['engembed'])
ee=[step('取 24 行，再拼成一个向量',r'Z=\operatorname{concat}(E[i_1,:],\ldots,E[i_{24},:])',r'24\times[1,256]\longrightarrow[1,6144]',[('E','当前 Engram 的记忆表，每行 256 个数字'),('i₁…i₂₄','当前 token 的 24 个查表地址'),('concat','按顺序拼接，不是把各行相加'),('Z','拼接得到的 6144 维向量')])]
eeb=[step('梯度拆回 24 行',r'dE[i_k,:]\mathrel{+}=dZ_k',r'dZ:[1,6144]\longrightarrow24\times[1,256]',[('i_k','第 k 次查表的地址，k 从 1 到 24'),('dZ_k','从 6144 维梯度拆出的第 k 段'),('dE','记忆表权重梯度；同一地址的贡献要累加')])]
card('engembed','Engram 记忆向量','一个 token 在本次 Engram 里读取 24 行，每行 256 维，拼成一行 6144 维。',ee,param(ee,98305579008,r'384006168\times256=98305579008',[('layer 14 的另一张表','98,308,270,592 个参数'),('两张表合计','196,613,849,600 个参数'),('当前 token 在一张表中取出','6,144 个权重值')],note='主例子是 layer 1 的记忆表；表大小与当前只读取 24 行不矛盾。'),forward(ee,6144,r'24\times256=6144',[('两次 Engram 查表反量化','12,288 次乘法')],explain='查出的低精度数值需要逐项乘缩放系数，共 6144 次乘法；查表和拼接本身主要是内存操作。'),unknown(eeb,'这里解释可学习数值的局部梯度；低精度存储如何参与训练，以及重复地址如何累加，会影响完整反向计算量。'),'记忆表虽大，每个 token 只从中取出少量内容。')
eg=[step('① 由记忆向量算出四个 Key 和一个 Value',r'[K_1,K_2,K_3,K_4,V]=Z W_E',r'[1,6144]\times[6144,25600]=[1,25600]',[('Z','查表拼出的 6144 维向量'),('W_E','学习得到的记忆投影权重'),('K₁…K₄','各 5120 维，分别与当前 token 的四路状态比较'),('V','一个 5120 维内容向量，四路共用')], '25600=4×5120+5120；这五份不是五个 token。'),step('② 每一路决定接收多少记忆',r'Y_c=R_c+g_c V',r'[1,5120]+[1,1]\times[1,5120]=[1,5120]',[('c','当前四路残差中的某一路'),('R_c','这一路原来的 5120 个数字'),('g_c','比较 R_c 与 K_c 后得到的 0 到 1 之间的门值'),('V、Y_c','共享记忆内容 / 更新后的这一路')], '门值先用两组学习缩放权重调整匹配分数，再经过开平方和 sigmoid；不是对全部历史做 attention。')]
egb=[step('① 先把四路梯度分给 Value 和门值',[r'dV=\sum_{c=1}^{4}g_cdY_c',r'dg_c=\sum_{j=1}^{5120}dY_{c,j}V_j'],[r'4\times[1,5120]\longrightarrow dV:[1,5120]',r'dg:[1,4]'],[('dY_c','更新后第 c 路的梯度'),('g_c / V','前向门值 / 共享内容向量'),('Σ','按所示四路或 5120 个特征相加'),('dV / dg_c','内容梯度 / 门值梯度')]),step('② 经过门控后，再反传投影',[r'dZ=dU W_E^{\mathsf T}',r'dW_E=Z^{\mathsf T}dU'],[r'[1,25600]\times[25600,6144]=[1,6144]',r'[6144,1]\times[1,25600]=[6144,25600]'],[('dU','四份 Key 梯度与一份 Value 梯度拼成的 25600 维向量'),('Z / W_E','前向查表结果 / 记忆投影矩阵'),('dZ / dW_E','回到查表结果的梯度 / 权重梯度'),('ᵀ','转置')])]
card('enggate','Engram 投影与门控','把取出的记忆变成适合当前 token 的内容，再决定每一路残差接收多少。',eg,param(eg,157327360,r'6144\times25600+2\times4\times5120=157327360',[('两次 Engram 各一套','314,654,720 个参数')],note='矩阵外两组 [4,5120] 权重参与匹配分数的缩放。'),forward(eg,314572800,r'2\times1\times6144\times25600=314572800',[('再含本实例门值计算','约 314,736,672 次操作'),('四路残差注入','另需 40,960 次乘加')],note='这里先看大矩阵；门值计算仍有额外操作，不能当作免费。'),backward(egb,[('一套投影的矩阵反向','629,145,600 FLOPs'),('门控与整块反向','尚不能确定完整数值')],[r'2\times314572800=629145600'],note='Key 的梯度要经过门控的匹配、数值调整和开平方等步骤；四路 residual 也会从门控路径额外得到梯度。'),'一份 Value 广播给四路，但每一路的接收比例不同。')
inj=[eg[1]]
card('injection','向四路残差注入记忆','每一路加上自己的门值乘同一个记忆 Value，然后继续进入这一层。',inj,noparam(inj),forward(inj,40960,r'4\times5120\times2=40960',[('layer 1 与 layer 14 两次','81,920 FLOPs')],explain='四路各有 5120 个位置，每个位置做一次乘法、一次加法。'),unknown([egb[0]],'除了直接相加的梯度，原残差还参与了门值计算，完整梯度要包含那一条路径。'),'两次注入发生在进入 layer 1 和 layer 14 之前。')
for key,layer,table in [('engram1',1,98305579008),('engram14',14,98308270592),('engram',1,98305579008)]:
    purpose=f'当前 token 在 layer {layer} 前用自身与前面的编号查记忆，再把结果注入四路残差。'
    eshort=[ee[0],eg[0],eg[1]]
    card(key,f'Engram 记忆'+(f' · layer {layer}' if key!='engram' else ''),purpose,eshort,
      param(eshort,table+157327360,rf'{table}+157327360={table+157327360}',[('两次 Engram 完整存储','196,928,504,320 个参数')],note=f'主例子是 layer {layer} 的一次 Engram；表参数和投影参数分别计算。'),
      forward(eshort,314572800,r'2\times1\times6144\times25600=314572800',[('加查表反量化、门控和注入','一次约 314,783,776 次操作'),('当前 token 两次 Engram 合计','约 629,567,552 次操作')],note='先看本次的大矩阵；读取历史 token 编号不等于重新执行这些 token 的网络。'),
      backward(egb,[('一次投影矩阵反向','629,145,600 FLOPs'),('完整记忆模块反向','不能只由这一个矩阵小计确定')],note='梯度可以回到记忆数值和学习权重；整数哈希地址不求导，门控与跨使用位置的累加还需另算。'),
      'Engram 用少量前文编号查表，不对整段历史做注意力。',related=['ngram','hash','engembed','enggate','injection'])

streams=[step('同一个 token 的四路向量',r'R=[R_1,R_2,R_3,R_4]',r'R:[1,4,5120],\quad R_c:[1,5120]',[('R','当前 token 的四路状态整体'),('R₁…R₄','每一路有 5120 个数字'),('c','四路中的一路，不是 token 序号')])]
card('streams','四路残差状态','这个 token 同时保留四路向量，HC 会在它们之间混合信息。',streams,noparam(streams,'状态是运行时算出的数字，不是额外的学习权重。'),info(streams,[('状态本身','不单独计算 FLOPs；混合操作在 HC 中')]),info([step('把各条路径的梯度相加',r'dR=dR_{direct}+dR_{module}+dR_{mix}',r'[1,4,5120]\longrightarrow[1,4,5120]',[('dR_direct','直接残差路径传回的梯度'),('dR_module','Attention / FFN 等子模块传回的梯度'),('dR_mix','生成混合比例这条路径传回的梯度'),('dR','合并后的四路梯度')])],[('反向状态形状','[1,4,5120]')]),'第一个维度 1 始终表示同一个当前 token。',related=['hc'])
for key,title in [('hidden','Encoder 后的当前 token 状态'),('localstates','Decoder 的局部状态与缓存')]:
    s=[step('把当前向量变成当前的 KV',r'KV_{new}=X W_{KV}',r'[1,5120]\times[5120,512]=[1,512]',[('X','当前 token 经过四路合并和归一化后的向量'),('W_KV','这一模块的投影权重'),('KV_new','当前新产生的一行；之前的行保存在缓存里')])]
    card(key,title,'当前只计算一个 token 的新状态。历史状态已经存在缓存中，供后面的注意力读取。',s,noparam(s,'状态本身不增加参数；图示投影权重在 KV 模块计算。'),info(s,[('这个状态标签','不重复记账；乘法计在 KV 投影中')]),info(grad_steps(5120,512,'W_{KV}',y='KV'),[('局部投影梯度','见 KV 投影面板，不再重复相加')],'历史向量如果被多次读取，会收到多次梯度贡献。'),'缓存里有多行历史，不代表这一步重新处理了多行输入。',related=['global' if key=='hidden' else 'kv','multi'])

log=[step('保存当前 token 的词表分数',r'Y=X W_{head}',r'[1,5120]\times[5120,129280]=[1,129280]',[('X','当前 token 的最终 5120 维向量'),('W_head','词表输出权重'),('Y','129280 个未归一化分数，称为 logits')])]
card('logits','当前 token 的 Logits','每个候选 token 对应一个分数，数值越大通常越倾向于被选中。',log,noparam(log,'这是计算结果，LM Head 的权重已在那里计数。'),info(log,[('额外计算','0；矩阵乘法已经计在 LM Head')]),info([step('接收输出侧传来的梯度',r'dY:[1,129280]',r'[1,129280]\longrightarrow\text{LM Head backward}',[('dY','外部学习目标告诉我们：129280 个分数各怎样影响 Loss')])],[('下一步','用 dY 计算 LM Head 的 dX 与 dW')],'不在这张图里另加 Loss 或优化器的计算。'),'logits 是当前向量的计算结果，不是又一套模型参数。',related=['head'])
pr=[step('按最后的分数选一个编号',r'u_{next}=\operatorname{Select}(Y)',r'[1,129280]\longrightarrow[1]\text{ integer ID}',[('Y','当前 token 的词表分数'),('Select','可以选最高分，也可以按概率采样'),('u_next','选出的下一个 token 编号')])]
card('prediction','下一 token 预测','根据当前 token 的词表分数，选出下一个 token。图中候选词和概率只是说明性的示例。',pr,noparam(pr),info(pr,[('采样计算','单独处理，不混入矩阵投影 FLOPs')]),info(pr,[('离散选择','不直接进行普通反向求导')],'学习时的梯度来自输出侧的学习目标，而不是对抽出的整数 token 编号求导。'),'当前 token 提供预测依据，下一个 token 是随后才产生的。')

# Group cards keep the first result at one layer, with whole-region figures secondary.
def comp_cost(ids):
    basic=sum(10485760 if i<20 else 5242880+2048+192 for i in ids)
    return basic,2*sum(i==20 for i in ids)
def index_cost(ids):
    pairs=sum(512 if i<20 else 1025 for i in ids)
    emitted=int(20 in ids)
    return len(ids)*(10485760+327680+6144+32)+emitted*(131072+512+192)+(8192+63)*pairs,2*emitted
def attn_cost(layers,tiles):
    from FLOPs_calculation_verify import attention_kernel
    return attention_kernel(layers,tiles)
TOKEN_BASE=96993280+5122+5244930+24768+150994944+2*1246042+2*20482+3933714+495550208
def group(key,title,layers):
    src=[i for i in layers if i in [2,8,14,20]];idx=[i for i in layers if i in [2,8,14,20,24,28,32,36]]
    pc=CORE_P*len(layers)+sum(5243392 if i<20 else 2621952 for i in src)+len(idx)*5406720+len(src)*65664
    fc=TOKEN_BASE*len(layers)+sum(comp_cost(src))+sum(index_cost(idx))+sum(attn_cost(len(layers),sum(2 if i<2 else 10 for i in layers)))
    ss=[step('当前 token 经过一个完整层',r'R_{next}=\operatorname{Block}(R)',r'[1,4,5120]\longrightarrow[1,4,5120]',[('R','当前 token 进入这一层时的四路状态'),('Block','Attention、HC、MoE 和归一化组成的一层'),('R_next','同一个 token 经过这一层后的四路状态')], 'Attention 可以读取此前 1024 个 token 留下的缓存；并不重新计算这些历史 token 的所有模块。')]
    card(key,title,'沿着当前 token 看每一层：先用 Attention 读取上下文，再经 HC 混合，接着执行 MoE，并更新四路状态。',ss,
      param(ss,CORE_P,'',[(f'本组 {len(layers)} 层含专属压缩/索引权重',fmt(pc)+' 个参数')],note='先显示普通一层的固定权重。拥有 Compressor 或 Indexer 的层另有权重；两次 Engram、输入词表和输出词表另计。'),
      info(ss,[('一层 Query 两级矩阵','96,993,280 FLOPs'),('一层输出两级矩阵','150,994,944 FLOPs'),('一层七个专家矩阵','495,452,160 FLOPs')],'先理解这些主要乘法；其余注意力、HC、归一化和路由也要计算。'),
      info([step('按相反顺序把梯度传回',r'dR=dR_{attention}+dR_{experts}+dR_{residual}+dR_{mix}',r'[1,4,5120]\longrightarrow[1,4,5120]',[('dR','当前 token 这一层输入的梯度'),('各下标','分别表示 Attention、专家、直接残差和混合比例路径的贡献')])],[('整层反向','无法仅凭一个 token 的独立计算给出完整训练数值')],'历史注意力和共享权重还要合并其他 token 或层的梯度。'),
      '层号增加时，还是同一个当前 token 在更新自己的表示。',related=['query','kv','multi' if min(layers)>=20 else 'sparse','hc','moe'],scope=f'当前 1 个 token · layers {min(layers)}–{max(layers)}')
    DATA[key]['forward']['totals']=[(f'这一 token 经过本组 {len(layers)} 层',fmt(fc)+' 次估算操作')]
    DATA[key]['forward']['note']='展开的总量以已有 1024 个历史 token 为例；不含图外置的 Engram、输入词表和最后输出。source 20 的共享 KV 已在其所在层计入一次。'.replace('source 20','layer 20')
    DATA[key]['_audit']=dict(parameters=pc,decode=fc,layers=layers)
group('encoder','当前 token 经过 Encoder',list(range(20)))
group('decoder','当前 token 经过 Decoder',list(range(20,40)))
for i in [0,1,2,13,14,19,20,21,22,33,34,39]:group(f'layer{i}',f'当前 token · 第 {i} 层',[i])
for key,base,title in [('encoderblock','encoder','Encoder 的一层计算'),('decoderblock','decoder','Decoder 的一层计算')]:DATA[key]=deepcopy(DATA[base]);DATA[key]['title']=title
for key,base,title in [('encoderattn','sliding','Encoder Attention'),('decoderattn','multi','Decoder Attention')]:
    DATA[key]=deepcopy(DATA[base]);DATA[key]['title']=title
    DATA[key]['purpose']='当前 token 先生成 Query 与自己的新 KV，再读取缓存，最后把汇总结果投影回 5120 维。'
    DATA[key]['note']='这里用一层的注意力读取过程举例；Query、KV、输出投影的参数和 FLOPs 可分别点击查看。'+('Encoder 的 0–1 层只读局部窗口，2–19 层还读取压缩位置。' if key=='encoderattn' else 'Decoder 每层都有自己的 Query 与局部 KV，并读取共享 KV。')

ov=[step('从当前向量到下一 token 的分数',r'X\longrightarrow\text{40 layers}\longrightarrow Y',r'[1,5120]\longrightarrow[1,129280]',[('X','当前 token 的词嵌入向量'),('40 layers','沿图中的 40 层依次更新这个 token 的状态'),('Y','这个 token 对整个词表的预测分数')], '普通面板先看一个 token 在一次模块调用中的运算，再按需展开多层合计。')]
card('overview','跟着一个 token 走','我们只跟踪当前 token。普通模块计算一行向量；Attention 从缓存读取历史，Compressor 需要凑齐分组。',ov,
 mode(ov,[r'129280\times5120=661913600'],[('先看输入词表','661,913,600 个参数')],'词表的行数乘每行的宽度，与这次输入几个 token 无关。',totals=[('完整主模型存储','748,494,684,784 个参数')]),
 forward(ov,5242880,r'2\times1\times5120\times512=5242880',[('此处已有 1024 个历史 token 时，全模型当前一步','约 35,584,521,265 次操作')],note='主例子只展示一次 KV 投影；展开项是当前一个 token 的完整一步估计，历史 token 不重新执行。量化、内存搬运和设备通信另计，不能把它们当作没有成本。'),
 info(grad_steps(5120,512),[('这一投影的局部反向','10,485,760 FLOPs'),('完整训练反向','不能从这一 token 的例子直接得到')],'矩阵的两个梯度可以逐一算；历史读写、共享权重和多次梯度累加还会影响整体工作量。'),
 '先看这一行数字怎样变化，再看它用了哪些权重。',related=['embedding','query','kv','head'])

def check():
    assert set(DATA)=={'input','tokenizer','ids','embedding','qnorm','kvnorm','anorm','fnorm','finalnorm','qa','qb','query','kv','rope','output','compressor','indexer','sliding','sparse','multi','global','hc','streams','router','selection','weights','expert','shared','swiglu','combine','moe','ngram','hash','engembed','enggate','injection','engram1','engram14','engram','head','logits','prediction','encoder','decoder','layer0','layer1','layer2','layer13','layer14','layer19','layer20','layer21','layer22','layer33','layer34','layer39','encoderblock','decoderblock','encoderattn','decoderattn','hidden','localstates','overview'}
    for key,d in DATA.items():
        assert all(k in DATA for k in d['related']),key
        for name in ['parameter','forward','backward']:
            m=d[name];assert m['steps'] and m['rows'],(key,name)
            for s in m['steps']:
                assert s['formulas'] and s['dimensions'] and s['symbols'],(key,name,s)
    assert DATA['encoder']['_audit']['parameters']+DATA['decoder']['_audit']['parameters']+196928504320+2*661913600+5120==748494684784
    assert DATA['encoder']['_audit']['decode']+DATA['decoder']['_audit']['decode']+629567552+1323883522==35584521265

if __name__=='__main__':check();print(f'{len(DATA)} single-token teaching entries checked.')
