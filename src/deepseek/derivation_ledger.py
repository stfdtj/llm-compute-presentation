"""Executable real-valued compute graphs, with separate scalar and special-op ledgers.

This supplements the pinned inference-source audit; it does not claim an official
training implementation. Counts are generated from shapes, never copied totals.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re
import audit_model as source

C = source.C
H, F, R, D, A = (C[k] for k in ('dim', 'moe_inter_dim', 'q_lora_rank', 'head_dim', 'n_heads'))
HC, IH, ID = C['hc_mult'], C['index_n_heads'], C['index_head_dim']
ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = {
 'A': dict(title='A · 单 Token Decode', batch=1, length=1, prior=1024,
  description='B=1，当前 L=1，计算行数 M=B×L=1；此前1024个token已在缓存。前向只生成当前条目。A不执行训练反向；Backward仅是当前调用的局部VJP数学练习：历史缓存作为独立可求导叶子，梯度止于缓存，不能据此估算训练。'),
 'B': dict(title='B · 理论训练图', batch=1, length=128, prior=0,
  description='B=1，L=128，计算行数 M=B×L=128；从序列起点重算全部位置。保存反向需要的实数状态，历史KV、Compressor、共享权重均参与梯度。固定每行同一组6个专家、查表行互异以提供可复算数字，实际分布须重算归约。离散索引固定，量化用实数代理（保留Engram显式values×scale）；这不是已验证的训练策略。')}
POLICY = [
 '普通算术：标量加、减、乘各记 1；约简 n 项为 n−1 次加法。复制、广播、reshape、比较、整数索引、Top-K 选择及通信不计浮点 FLOPs，单列其工作与边界。',
 'exp、log1p、sqrt、rsqrt、除法分别按调用次数列账，不与普通乘加相加为“精确 FLOPs”。均值按预计算常量1/N乘法计；Sigmoid 展开成负号、加法、exp、除法；SiLU 复用 Sigmoid。Softplus 使用 log1p(exp(z)) 的数学模型，稳定内核可能有分支。',
 '严格标量计数只针对所写实数计算图，不是硬件指令数。并列的 2MKN 口径仅替换 GEMM 项为常见近似，其余普通操作仍逐项统计；不把它称作精确结果。',
 '矩阵反向分别记 dX、dW；dW 的求和轴是 M，精确 KN(2M−1)，单行时为 KN。批内已经形成的梯度再次累加到既有梯度缓冲、优化器与跨设备归约不在局部 VJP 中。',
 'Verified = 前向公式/维度由固定源码确认；Derived = 明确实数图的理论梯度或标量计数；Estimated = GEMM 2MKN 等近似；Unknown = 训练策略、专用内核或数据相关工作量尚未确定。不同状态可以在同一模块并存。']

def fmt(n): return f'{n:,}'
def shp(*ns): return '[' + ','.join(map(str, ns)) + ']'
def weight_symbol(pid):
    return {'qa':r'W_{QA}','qb':r'W_{QB}','kv':r'W_{KV}','head':r'W_{head}','oa':r'W_{OA}','ob':r'W_{OB}',
      'hc_fn':r'W_{HC}','router':r'W_R','w1':r'W_1','w2':r'W_2','w3':r'W_3','ew':r'W_E',
      'iq':r'W_{IQ}','ih':r'W_{IH}','ik':r'W_{IK}','ckv':r'W_{CKV}','cgate':r'W_{gate}'}.get(pid,'W')

@dataclass
class Cost:
    mul: int = 0
    add: int = 0
    sub: int = 0
    special: dict = field(default_factory=dict)
    # Difference introduced ONLY by substituting 2MKN for GEMM scalar counts.
    gemm_delta: int = 0

    def export(self):
        ordinary = self.mul + self.add + self.sub
        return dict(mul=self.mul, add=self.add, sub=self.sub, ordinary=ordinary,
                    special=self.special, approximate=ordinary + self.gemm_delta)

class Flow:
    def __init__(self, label, scope='', unknown=()):
        self.label, self.scope, self.unknown = label, scope, list(unknown)
        self.steps = []

    def step(self, title, purpose, inputs, outputs, formula, why, cost=None,
             work='', refs=(), status='Derived', path='main', source_symbol=''):
        cost = cost or Cost()
        inputs = [inputs] if isinstance(inputs, str) else inputs
        outputs = [outputs] if isinstance(outputs, str) else outputs
        self.steps.append(dict(title=title, purpose=purpose, inputs=inputs, outputs=outputs,
          formulas=[formula], dimensions=[r'\text{' + ' ; '.join(inputs + outputs).replace('_',' ') + '}'],
          symbols=[('输入', '；'.join(inputs)), ('输出', '；'.join(outputs))],
          note=why, why=why, cost=cost.export(), work=work or '此步为视图/读取/离散操作，普通浮点算术 0；不代表没有访存或整数工作。',
          parameterIds=list(refs), status=status, formulaStatus='Verified' if source_symbol else 'Derived',
          path=path, source=source_symbol))

    def extend(self, other, prefix='', path=None):
        for item in other.steps:
            item = deepcopy(item)
            if prefix: item['title'] = prefix + ' · ' + item['title']
            if path: item['path'] = path
            self.steps.append(item)
        self.unknown.extend(other.unknown)
        return self

    def export(self):
        cumulative = {'main': Counter(), 'conditional': Counter()}
        totals = {}
        for step in self.steps:
            acc = cumulative[step['path']]
            for k in ['mul', 'add', 'sub', 'ordinary', 'approximate']: acc[k] += step['cost'][k]
            for k, v in step['cost']['special'].items(): acc['special:' + k] += v
            step['cumulative'] = dict(acc)
        for path, acc in cumulative.items():
            totals[path] = {k: acc[k] for k in ['mul', 'add', 'sub', 'ordinary', 'approximate']}
            totals[path]['special'] = {k[8:]:v for k,v in acc.items() if k.startswith('special:')}
        return dict(label=self.label, scope=self.scope, unknown=list(dict.fromkeys(self.unknown)), steps=self.steps, totals=totals)

def linear_f(name, m, k, n, groups=1, pid=''):
    f = Flow(name)
    mul, add = groups*m*k*n, groups*m*n*(k-1)
    f.step(name+' · XW', '把输入特征投影到目标维度。',
      [f'X {shp(m,k)}', f'{name} {shp(k,n)}'+(f'，{groups} 个独立分组' if groups>1 else '')],
      f'Y {shp(m,n)}'+(f' × {groups} 组' if groups>1 else ''), 'Y=X'+weight_symbol(pid),
      '每个输出是 K 项点积；先有 K 次乘法，再把它们相加。',
      Cost(mul, add, gemm_delta=groups*2*m*k*n-mul-add),
      f'乘法 G×M×K×N = {groups}×{m}×{k}×{n} = {fmt(mul)}；加法 G×M×N×(K−1) = {fmt(add)}。严格合计 {fmt(mul+add)}；GEMM 近似 2GMKN = {fmt(groups*2*m*k*n)}。',
      [pid] if pid else (), source_symbol='model.py · Linear / F.linear')
    return f

def linear_b(name, m, k, n, groups=1, pid=''):
    f=Flow(name)
    for title, inputs, output, formula, mul, add, why in [
      ('dX', [f'dY {shp(m,n)}', f'Wᵀ {shp(n,k)}'], f'dX {shp(m,k)}', 'dX=dY'+weight_symbol(pid)+r'^{\mathsf T}', groups*m*k*n, groups*m*k*(n-1), '求和轴是输出宽度 N，把梯度传回输入。'),
      ('dW', [f'Xᵀ {shp(k,m)}', f'dY {shp(m,n)}'], f'dW {shp(k,n)}', 'd'+weight_symbol(pid)+r'=X^{\mathsf T}dY', groups*m*k*n, groups*k*n*(m-1), '求和轴是本次调用的输入行数 M，把每行对同一权重的贡献累加。')]:
        f.step(name+' · '+title, why, inputs, output, formula,
          '独立分组各自形成梯度；返回的梯度不是新增学习参数。', Cost(mul, add, gemm_delta=groups*2*m*k*n-mul-add),
          f'{groups} 组，乘法 {fmt(mul)} + 加法 {fmt(add)} = {fmt(mul+add)}；2MKN 近似 {fmt(groups*2*m*k*n)}。'+(f' 单行 M=1 时：dW 是外积，严格 GKN={fmt(groups*k*n)}，近似 2GKN={fmt(2*groups*k*n)}，二者不同。' if title=='dW' else ''), [pid] if pid else ())
    return f

def norm_f(name, m, n, pid=''):
    f=Flow(name)
    f.step(name+' · 均方与倒数平方根', '对每行计算稳定的缩放量。', f'X {shp(m,n)}', f'r {shp(m,1)}',
      r'r=(\frac{1}{N}\sum_i X_i^2+\epsilon)^{-1/2}', '统一每行尺度，epsilon 避免零除。',
      Cost(m*(n+1), m*n, special={'rsqrt':m}),
      f'平方 MN={fmt(m*n)} 次乘法；均值常量乘 M={m}；行和 M(N−1)={fmt(m*(n-1))} 次加法；epsilon 加 M={m}；rsqrt {m} 次。', source_symbol='model.py · RMSNorm.forward')
    f.step(name+' · 缩放与 γ', '保留可学习的逐通道缩放。', [f'X {shp(m,n)}',f'r {shp(m,1)}',f'γ {shp(n)}'], f'Y {shp(m,n)}',r'Y=(Xr)\odot\gamma',
      'r 调整行尺度，γ 决定每个通道保留的尺度。', Cost(2*m*n), f'X×r 与 ×γ 各 MN={fmt(m*n)} 次乘法，共 {fmt(2*m*n)}。', [pid] if pid else (), source_symbol='model.py · RMSNorm.forward')
    return f

def norm_b(name, m, n, pid=''):
    f=Flow(name)
    f.step(name+' · dγ', '跨输入行累计缩放参数梯度。', [f'dY {shp(m,n)}',f'已保存 Xr {shp(m,n)}'], f'dγ {shp(n)}',r'd\gamma=\sum_m dY\odot(Xr)',
      'γ 在同一次调用的所有行之间共享；复用前向的 Xr。', Cost(m*n, (m-1)*n), f'MN={fmt(m*n)} 乘法 + N(M−1)={fmt((m-1)*n)} 加法。', [pid] if pid else ())
    f.step(name+' · dX', '同时反传逐项缩放路径和均方统计路径。', [f'dY,X {shp(m,n)}',f'γ {shp(n)}',f'r {shp(m,1)}'], f'dX {shp(m,n)}',
      r'g=dY\odot\gamma,\quad t=\sum_i g_iX_i,\quad dX=rg-X(r^3t/N)',
      'X 既直接参与输出，也影响整行 r；忽略后者会得到错误梯度。',
      Cost(4*m*n+4*m, m*(n-1), m*n),
      f'g、gX、rg、X×系数各 MN 次乘法；r²、r³、×1/N、×t 各 M 次乘法，共 {fmt(4*m*n+4*m)}。点积求和 {fmt(m*(n-1))} 加法；两路径相减 {fmt(m*n)} 次。')
    return f

def rope(name, rows, heads, backward=False, dim=D):
    f=Flow(name); count=rows*heads*(C['rope_head_dim']//2)
    f.step(name+(' · 反向旋转' if backward else ' · 成对旋转'), '只旋转最后 64 维，其他维度原样传递。',
      f'{"dY" if backward else "X"} {shp(rows,heads,dim)}，预计算 cosθ / sinθ', f'{"dX" if backward else "Y"} 同形状',
      r'\binom{y_0}{y_1}=\begin{pmatrix}c&-s\\s&c\end{pmatrix}\binom{x_0}{x_1}' if not backward else r'dX=R(\theta)^{\mathsf T}dY',
      '逆 RoPE 的前向使用 −θ，反向用该旋转的转置；sin/cos 表为常量，不在此重新计算。',
      Cost(4*count,count,count), f'{rows} 行 × {heads} heads × 32 对 = {fmt(count)} 对；每对 4 乘 + 1 加 + 1 减 = 6，共 {fmt(6*count)}。', source_symbol='' if backward else 'model.py · apply_rotary_emb')
    return f

def softmax(name, lengths, backward=False, prefix=1):
    f=Flow(name); n=sum(lengths)*prefix; rows=len(lengths)*prefix
    f.step(name, '将同一归一化组内的分数变成权重。' if not backward else '将权重梯度变成分数梯度。',
      f'{"P,dP" if backward else "Z"}：{rows} 组，共 {fmt(n)} 个有效元素', f'{"dZ" if backward else "P"} 同形状',
      r'dZ=P\odot(dP-\sum_j P_jdP_j)' if backward else r'P_i=\exp(Z_i-\max Z)/\sum_j\exp(Z_j-\max Z)',
      '组内分母把所有项联系起来。前向最大值仅用于数值稳定；Softmax 的平移不变性给出此 VJP，无需另求 max 的梯度。',
      Cost(2*n,n-rows,n) if backward else Cost(0,n-rows,n,{'exp':n,'div':n}),
      (f'P×dP 与 P×括号各 {fmt(n)} 乘；求和 {fmt(n-rows)} 加；减去内积 {fmt(n)} 减。' if backward else f'减去 max {fmt(n)} 减；行和 {fmt(n-rows)} 加；exp {fmt(n)} 次、除法 {fmt(n)} 次；max 的 {fmt(n-rows)} 次比较另列，不算 FLOPs。'), source_symbol='' if backward else 'kernel.py / model.py · softmax')
    return f

def hc(m, backward=False):
    n=HC*H; f=Flow('Hyper-Connection', '一次 HC 包含系数生成、当前 pre 合并和 post/comb 回混；Attention/FFN 及其外部 RMSNorm 在各自模块计数。输入为 R、前一子模块传来的 p_in；输出为四路 Y 和供下一子模块使用的 p_next。')
    if not backward:
        f.step('1a · 展平并计算 r', '从四路残差形成每 token 一个归一化因子。', f'R {shp(m,HC,H)} → R_flat {shp(m,n)}',f'r {shp(m,1)}',r'r=(\operatorname{mean}(R_{flat}^2)+\epsilon)^{-1/2}',
          '归一化统计横跨 4×5120 个数；没有 γ。reshape 不产生浮点算术。',Cost(m*(n+1),m*n,special={'rsqrt':m}),
          f'{fmt(m*n)} 平方乘法 + {m} 次均值常量乘；{fmt(m*(n-1))} 求和加法 + {m} 次 epsilon 加法；rsqrt {m} 次。',source_symbol='model.py · Block.hc_mixes')
        f.extend(linear_f('W_HC',m,n,24,pid='hc_fn'))
        f.step('1b · 投影结果乘 r', '把归一化应用在 24 个分数上。', [f'T=R_flat W_HC {shp(m,24)}',f'r {shp(m,1)}'],f'M {shp(m,24)}',r'M=Tr',
          '源码是先投影后缩放；不是先逐元素归一化 20480 维输入。',Cost(m*24),f'每行 24 个输出 × r，{m}×24={fmt(m*24)} 次乘法。',source_symbol='model.py · Block.hc_mixes')
        f.step('2a · 拆分、scale 与 base', '把 24 个数分成 pre 4、post 4、comb 16。', [f'M {shp(m,24)}','scale [3]，base [24]'],[f'a_pre,a_post {shp(m,4)}',f'A_comb {shp(m,4,4)}'],r'a_g=M_g\,scale_g+base_g',
          'pre 决定四路如何合成单路；post 决定子模块输出写回各路多少；comb[i,j] 决定旧路 i 向新路 j 的残差混合。',Cost(24*m,24*m),f'24M={fmt(24*m)} 次乘法 + 24M={fmt(24*m)} 次加法。', ['hc_scale','hc_base'],source_symbol='kernel.py · hc_split_sinkhorn_kernel')
        f.step('2b · pre / post 的 Sigmoid', '限制混合系数的范围。',f'a_pre,a_post {shp(m,4)}',[f'p_next=σ(a_pre)+ε {shp(m,4)}',f'b=2σ(a_post) {shp(m,4)}'],r'p_{next}=\sigma(a_{pre})+\epsilon,\quad b=2\sigma(a_{post}),\quad\sigma(z)=1/(1+e^{-z})',
          '此处 p_next 交给下一个子模块；本次 Step 3 用输入 p_in，不能误把二者连在同一时序。',Cost(4*m,12*m,8*m,{'exp':8*m,'div':8*m}),
          f'8M 次负号（按减法计）、8M 次 +1；pre +ε 为 4M 次加，post ×2 为 4M 次乘。exp / div 各 {fmt(8*m)} 次。',source_symbol='kernel.py · hc_split_sinkhorn_kernel')
        f.extend(softmax('2c · comb 行 Softmax',[4]*(4*m)))
        f.step('2d · 初始矩阵加 ε', '避免后续行列归一化的退化。',f'P {shp(m,4,4)}',f'C0 {shp(m,4,4)}',r'C_0=P+\epsilon', '所有 16 项加 epsilon。',Cost(add=16*m),f'16M={fmt(16*m)} 次加法。',source_symbol='kernel.py · hc_split_sinkhorn_kernel')
        f.step('2e · Sinkhorn 的 39 次归一化', '先归一化列，再重复 19 次“行、列”。',f'C0 {shp(m,4,4)}',f'C {shp(m,4,4)}',r'C\leftarrow C/(\sum_{axis}C+\epsilon)',
          '配置 iters=20：首次行 Softmax 后有一次列归一化，再有 19×2 次。每步分母在四个元素间复用，按提取公共分母的实数图计数；内核是否重复计算 epsilon 属于指令实现差异。',Cost(add=39*m*16,special={'div':39*m*16}),
          f'每步 4 组×(3 次求和加+1 次 epsilon 加)=16 加，16 除；39M={fmt(39*m)} 步，共 {fmt(39*m*16)} 加、{fmt(39*m*16)} 除。保存每一步 C 供理论反向使用。',source_symbol='kernel.py · hc_split_sinkhorn_kernel')
        f.step('3 · 用传入的 pre 合并四路', '形成当前 Attention / FFN 的单路输入。',[f'R {shp(m,4,H)}',f'p_in {shp(m,4)}'],f'Z {shp(m,H)}',r'Z_d=\sum_{i=1}^{4}p_{in,i}R_{i,d}',
          '当前 p_in 来自前一个子模块。子模块自身计算 F(Z) 的成本由 Attention / MoE 面板承担。',Cost(4*m*H,3*m*H),f'每个通道 4 乘+3 加；乘 {fmt(4*m*H)}，加 {fmt(3*m*H)}，小计 {fmt(7*m*H)}。',source_symbol='model.py · Block.hc_pre')
        f.step('4 · post / comb 回混四路', '同时保留残差与子模块的新结果。',[f'U {shp(m,H)}',f'R {shp(m,4,H)}',f'b {shp(m,4)}, C {shp(m,4,4)}'],f'Y {shp(m,4,H)}',r'Y_{j,d}=b_jU_d+\sum_i C_{ij}R_{i,d}',
          'comb 的第一个轴是旧输入路，第二个轴是新输出路。',Cost(20*m*H,16*m*H),f'comb 乘法 16MH、求和 12MH；post 乘法 4MH，最后相加 4MH。共 {fmt(20*m*H)} 乘 + {fmt(16*m*H)} 加 = {fmt(36*m*H)}。',source_symbol='model.py · Block.hc_post')
    else:
        for title, ins, out, eq, mul, add, why in [
          ('4⁻¹ · 子模块输出梯度', ['dY [M,4,H]','b [M,4]'], 'dU [M,H]',r'dU_d=\sum_j b_jdY_{j,d}',4*m*H,3*m*H,'接下来由 Attention/FFN 自己的反向把 dU 变成 dZ；其成本不算进 HC。'),
          ('4⁻¹ · post 系数梯度',['dY [M,4,H]','U [M,H]'],'db [M,4]',r'db_j=\sum_d dY_{j,d}U_d',4*m*H,4*m*(H-1),'对特征轴求和。'),
          ('4⁻¹ · comb 系数梯度',['R,dY [M,4,H]'],'dC [M,4,4]',r'dC_{ij}=\sum_d R_{i,d}dY_{j,d}',16*m*H,16*m*(H-1),'每对旧路 i、新路 j 都有一个点积。'),
          ('4⁻¹ · 残差直连梯度',['C [M,4,4]','dY [M,4,H]'],'dR_post [M,4,H]',r'dR^{post}_{i,d}=\sum_j C_{ij}dY_{j,d}',16*m*H,12*m*H,'这是 R 的第一条梯度路径。'),
          ('3⁻¹ · pre 合并的输入梯度',['dZ [M,H]','p_in [M,4]'],'dR_pre [M,4,H]',r'dR^{pre}_{i,d}=p_{in,i}dZ_d',4*m*H,0,'dZ 由外部子模块反向提供；R 还有统计和投影路径。'),
          ('3⁻¹ · 传入 pre 的梯度',['dZ [M,H]','R [M,4,H]'],'dp_in [M,4]',r'dp_{in,i}=\sum_d dZ_dR_{i,d}',4*m*H,4*m*(H-1),'dp_in 回传到前一个系数生成器；当前生成器接收下一子模块提供的 dp_next。')]:
            f.step(title,'按链式法则反传 HC 的回混/合并。',[s.replace('M',str(m)).replace('H',str(H)) for s in ins],out.replace('M',str(m)).replace('H',str(H)),eq,why,Cost(mul,add),f'乘法 {fmt(mul)} + 加法 {fmt(add)} = {fmt(mul+add)}；每个长度 H 或 4 的约简分别使用 H−1 或 3 次加法。')
        f.step('2e⁻¹ · 逆序经过 39 次 Sinkhorn', '从最后一列归一化逐步回传到 C0。',f'G,C_in,C_out {shp(m,4,4)}；每步保存分母 s',f'dC0 {shp(m,4,4)}',r'dX_i=G_i/s-\bigl(\sum_jG_jY_j\bigr)/s,\quad Y=X/s,\ s=\sum X+\epsilon',
          '每个归一化组先算 t=Σ GY，再用 (G−t)/s；39 步均反向执行，不能只对最终矩阵做一次归一化反向。',Cost(39*m*16,39*m*12,39*m*16,{'div':39*m*16}),
          f'每步：16 乘，4×3=12 加，16 减，16 除；39M 步共 {fmt(39*m*44)} 普通算术 + {fmt(39*m*16)} 除法。')
        f.extend(softmax('2c⁻¹ · comb 行 Softmax',[4]*(4*m),True))
        f.step('2b⁻¹ · 两路 Sigmoid 梯度', '传播下一模块的 dp_next 与当前回混的 db。',[f'dp_next,db,已保存 σ(a) {shp(m,4)}'],f'da_pre,da_post {shp(m,4)}',r'da_{pre}=dp_{next}\sigma(1-\sigma),\quad da_{post}=2db\sigma(1-\sigma)',
          '加 epsilon 的导数为 1；post 前向有 ×2，因此反向也有 ×2。',Cost(20*m,sub=8*m),f'pre 每项 2 乘+1 减，post 每项 3 乘+1 减；4M 项合计 {fmt(20*m)} 乘、{fmt(8*m)} 减。')
        f.step('2a⁻¹ · scale / base 梯度', '把三组系数梯度还原到 24 维分数。',[f'da,M {shp(m,24)}','scale [3]'],[f'dM {shp(m,24)}','dscale [3], dbase [24]'],r'dM_g=da_g scale_g,\quad dscale_g=\sum_{m,j\in g}da_{mj}M_{mj},\quad dbase_j=\sum_m da_{mj}',
          'scale 每组共享一个数，分别跨 4M、4M、16M 项求和；base 每通道跨 M 行共享。',Cost(48*m,(24*m-3)+24*(m-1)),f'dM 24M 乘；dscale 24M 乘、(24M−3) 加；dbase 24(M−1) 加。', ['hc_scale','hc_base'])
        f.step('1b⁻¹ · 分数归一化的两条梯度', '投影值和 r 都依赖 R。',[f'dM,T {shp(m,24)}',f'r {shp(m,1)}'],[f'dT {shp(m,24)}',f'dr {shp(m,1)}'],r'dT=dM r,\quad dr=\sum_j dM_jT_j',
          '不能只把 dT 送入矩阵反向而忽略 r 的导数。',Cost(48*m,23*m),f'dT 24M 乘；dr 24M 乘+23M 加。')
        f.step('1a⁻¹ · r 的统计梯度', '将均方统计的影响传回所有 20480 个数。',[f'dr,r {shp(m,1)}',f'R_flat {shp(m,n)}'],f'dR_norm {shp(m,n)}',r'dR^{norm}=-(dr\,r^3/N)R_{flat}',
          '复用前向 r；r²、r³、乘 dr、乘 −1/N 四次标量乘，再逐项乘输入。',Cost(m*(n+4)),f'M(N+4)={fmt(m*(n+4))} 次乘法，无需重新执行 rsqrt。')
        f.extend(linear_b('W_HC',m,n,24,pid='hc_fn'))
        f.step('汇总 · 四条输入梯度相加', '形成 HC 对同一残差输入的总梯度。',f'dR_post,dR_pre,dR_norm,dR_projection 各 {shp(m,4,H)}',f'dR {shp(m,4,H)}',r'dR=dR^{post}+dR^{pre}+dR^{norm}+dR^{projection}',
          '四条路径逐元素累加三次；当前 pre 的梯度另外交给前一生成器。',Cost(add=3*m*n),f'3MN=3×{m}×{n}={fmt(3*m*n)} 次加法。')
        f.unknown.append('完整实数 HC VJP 已推导；专用 hc_split_sinkhorn 训练 kernel、保存/重算策略及量化训练规则没有在推理源码中提供。这里保存 39 次归一化状态，不计重计算。')
    return f

def router(m, backward=False):
    e,k=C['n_routed_experts'],C['n_activated_experts']; f=Flow('MoE Router','文本 token，gate_temp=1.0（ModelArgs 默认），route_scale=1.5；固定 Top-6 的局部可微路径。')
    if not backward:
        f.extend(linear_f('W_R',m,H,e,pid='router'))
        f.step('Router · 温度与 sqrtsoftplus','把原始打分转换为非负选择分数。',f'Z {shp(m,e)}',f'S {shp(m,e)}',r'z=Z/\tau,\quad u=\log(1+e^z),\quad S=\sqrt{u}',
          '源码默认 τ=1，仍按显式除法记账；softplus 的数值稳定内核可能使用比较分支，数学模型单列 exp/log1p/sqrt。',Cost(special={'div':m*e,'exp':m*e,'log1p':m*e,'sqrt':m*e}),f'ME={fmt(m*e)} 个元素；每项 1 div、1 exp、1 log1p、1 sqrt；没有把特殊函数直接记成普通乘加。',source_symbol='model.py · Gate.forward')
        f.step('Router · 选择偏置与 Top-6','选出本行执行的 6 个专家。',[f'S {shp(m,e)}',f'bias {shp(e)}'],f'I {shp(m,k)}，整数索引',r'I=\operatorname{Top6Indices}(S+bias)',
          'bias 只影响离散选择，后续权重取原 S。Top-K 的比较、排序、访存不属于浮点乘加；算法依赖的比较次数 Unknown。',Cost(add=m*e),f'选择偏置 ME={fmt(m*e)} 次加法；Top-K 不新增所定义的浮点 FLOPs。', ['bias'],source_symbol='model.py · Gate.forward')
        f.step('Router · Gather、归一化与放大','给选中的专家生成连续权重。',[f'S {shp(m,e)}, I {shp(m,k)}'],f'w {shp(m,k)}',r's=S[I],\quad d=\sum_{j=1}^6s_j+10^{-20},\quad w_j=1.5s_j/d',
          '归一化让 6 路权重可比较；乘 1.5 是配置的 route_scale。Gather 只读索引，不创建新权重矩阵。',Cost(m*k,m*k,special={'div':m*k}),f'每行求和 5 加、epsilon 1 加，共 {fmt(m*k)} 加；除法 {fmt(m*k)} 次，×1.5 为 {fmt(m*k)} 乘。',source_symbol='model.py · Gate.forward')
    else:
        f.step('Router⁻¹ · 归一化权重反向','从每个专家给出的 dw 得到所选分数梯度。',[f'dw,s {shp(m,k)}',f'd {shp(m,1)}'],f'ds {shp(m,k)}',r'g=1.5dw,\quad q=s/d,\quad ds_j=(g_j-\sum_i g_iq_i)/d',
          '复用前向 q 和分母；分母依赖全部 6 个已选分数。epsilon 保留，公式也适用于非零 epsilon。',Cost(2*m*k,m*(k-1),m*k,{'div':m*k}),f'g 6M 乘，gq 6M 乘，求和 5M 加，6M 减，6M 除。')
        f.step('Router⁻¹ · Scatter 与离散边界','只将连续权重梯度放回所选 6 个分数。',f'ds {shp(m,k)}, I int',f'dS {shp(m,e)}',r'dS[I]=ds,\quad dS[\overline I]=0',
          '同一行选择互异专家，Scatter 赋值无浮点加法。没有 dI 的普通连续梯度；选择 bias 也没有从此路径得到梯度。它如何训练或按负载更新是 Unknown。',Cost(),status='Derived')
        f.step('Router⁻¹ · sqrtsoftplus 梯度','沿固定专家选择回到 Router logits。',[f'dS,z,u,S {shp(m,e)}'],f'dZ {shp(m,e)}',r'dZ=dS\,\sigma(z)/(2S\tau)',
          '对所有 384 个槽做稠密 VJP，未选项梯度为零；不假定未知的稀疏训练优化。σ 的 exp/除法单列。',Cost(2*m*e,m*e,m*e,{'exp':m*e,'div':3*m*e}),f'σ：ME 减、ME 加、ME exp、ME div；2S 与 dS×σ 各 ME 乘，再 ME div、ME div（显式按 /2S /τ）。')
        f.extend(linear_b('W_R',m,H,e,pid='router'))
        f.unknown.append('Top-6 / 选择 bias 的更新策略未由推理代码给出；以上仅固定 I 的普通自动微分，不对整数专家 ID 求导。')
    return f

def expert(m, backward=False, routed=True, count=1):
    # Every expert handles m rows in the explicit teaching routing assignment.
    g=count; f=Flow('Routed Experts' if routed else 'Shared Expert',f'{count} 个独立专家，每个接收 {m} 行。Gate/Up clamp 使用配置 ±10；边界点取零子梯度的教学约定。')
    n=g*m*F
    if not backward:
        f.extend(linear_f('W1 · Gate',m,H,F,g,'w1')).extend(linear_f('W3 · Up',m,H,F,g,'w3'))
        f.step('SwiGLU · Clamp 与 SiLU','限制激活范围并生成平滑门。',[f'G,U {shp(g,m,F)}'],f'Gc,Uc,A {shp(g,m,F)}',r'G_c=\min(G,10),\quad U_c=\operatorname{clip}(U,-10,10),\quad A=G_c\sigma(G_c)',
          'clamp 比较不算浮点 FLOPs；SiLU 不只是一个乘法，还需要 Sigmoid 的 exp 和除法。',Cost(n,n,n,{'exp':n,'div':n}),f'{fmt(n)} 个元素：SiLU 的 Gc×σ 为 {fmt(n)} 乘；σ 含 {fmt(n)} 减、{fmt(n)} 加、{fmt(n)} exp、{fmt(n)} div。',source_symbol='model.py · Expert.forward')
        f.step('SwiGLU · 门与 Up 逐项乘','形成送入 Down 的中间表示。',f'A,Uc {shp(g,m,F)}',f'T {shp(g,m,F)}',r'T=A\odot U_c', '非线性门控制 Up 分支保留哪些特征。',Cost(n),f'GMF={fmt(n)} 次乘法。',source_symbol='model.py · Expert.forward')
        if routed:
            f.step('Routed · Down 前乘路由权重','按每行的选中权重缩放专家中间激活。',[f'T {shp(g,m,F)}',f'w {shp(g,m,1)}'],f'V {shp(g,m,F)}',r'V=wT',
              '源码先乘 2304 维激活，再执行无 bias 的 Down。代数上可移到输出，但 FLOPs 和 dw 的点积长度随实现位置变化，必须按源码计数。',Cost(n),f'GMF={fmt(n)} 次乘法；不是 GMH。',source_symbol='model.py · Expert.forward')
        f.extend(linear_f('W2 · Down',m,F,H,g,'w2'))
    else:
        f.extend(linear_b('W2 · Down',m,F,H,g,'w2'))
        if routed:
            f.step('Routed⁻¹ · 路由权重与中间激活梯度','把 Down 的输入梯度分给 w 和 T。',[f'dV,T {shp(g,m,F)}',f'w {shp(g,m,1)}'],[f'dT {shp(g,m,F)}',f'dw {shp(g,m,1)}'],r'dT=w\,dV,\quad dw=\sum_f dV_fT_f',
              'dw 回到 Router 的归一化权重路径，点积宽度为 F=2304。',Cost(2*n,g*m*(F-1)),f'dT GMF 乘；dw GMF 乘、GM(F−1) 加；共 {fmt(2*n)} 乘 + {fmt(g*m*(F-1))} 加。')
        f.step('SwiGLU⁻¹ · 逐项乘法','把梯度分给 SiLU 与 Up。',[f'dT,A,Uc {shp(g,m,F)}'],f'dA,dUc {shp(g,m,F)}',r'dA=dT\odot U_c,\quad dU_c=dT\odot A',
          '两个前向因子都需要各自的梯度。',Cost(2*n),f'每项 2 乘，共 {fmt(2*n)}。')
        f.step('SwiGLU⁻¹ · SiLU 与 Clamp','对已保存的 σ 反传，再应用 clamp 掩码。',f'dA,Gc,σ,dUc {shp(g,m,F)}',f'dG,dU {shp(g,m,F)}',r'dG_c=dA[\sigma+G_c\sigma(1-\sigma)],\quad dG=dG_c1_{G<10},\quad dU=dU_c1_{|U|<10}',
          '复用 σ，不重算 exp。比较不计浮点算术；采用显式乘 0/1 掩码，各多一次乘法。边界点不可微，教学约定为 0。',Cost(5*n,n,n),f'每项：Gc×σ、×(1−σ)、×dA 三乘；两个 clamp mask 两乘；括号一加、一减。共 {fmt(7*n)} 普通算术。')
        f.extend(linear_b('W1 · Gate',m,H,F,g,'w1')).extend(linear_b('W3 · Up',m,H,F,g,'w3'))
        f.step('Expert⁻¹ · 两分支输入梯度合并','同一专家输入同时影响 Gate 与 Up。',f'dX_gate,dX_up {shp(g,m,H)}',f'dX_expert {shp(g,m,H)}',r'dX_e=dX_{gate}+dX_{up}',
          '每个专家的参数梯度独立保留，不在专家之间相加。',Cost(add=g*m*H),f'GMH={fmt(g*m*H)} 次加法。')
    return f

def moe(m, backward=False):
    f=Flow('MoE 完整图','6 个 routed + 1 个 shared；数学示例固定每行同一组 6 个专家，路由 ID 不可微。实际批次用每个专家收到的 m_e 重新计算各自 dW 的归约。')
    if not backward:
        f.extend(router(m)).extend(expert(m,count=6)).extend(expert(m,routed=False))
        f.step('MoE · 专家输出合并','把 6 个路由专家与 1 个共享专家相加。',f'Y_e {shp(7,m,H)}',f'Y {shp(m,H)}',r'Y=\sum_{e\in I}Y_e+Y_{shared}',
          '按源码 y=0 后逐次 += 计 7 次加法；若把首项直接赋值，可改为 6 次，属于不同执行图。这里不混用二者。',Cost(add=7*m*H),f'7MH=7×{m}×{H}={fmt(7*m*H)} 次加法。',source_symbol='model.py · MoE.forward')
    else:
        f.step('MoE⁻¹ · 从最终输出分发梯度','相加的每个输入接收同一个 dY。',f'dY {shp(m,H)}',f'各专家 dY_e {shp(7,m,H)}',r'dY_e=dY',
          '广播/视图本身没有浮点算术；不把它误算成乘法。')
        f.extend(expert(m,True,count=6)).extend(expert(m,True,routed=False)).extend(router(m,True))
        f.step('MoE⁻¹ · 汇总全部输入梯度','同一 X 同时进入 7 个专家和 Router。',f'8 条 dX {shp(m,H)}',f'dX {shp(m,H)}',r'dX=\sum_{e=1}^{7}dX_e+dX_{router}',
          '8 条路径需要 7 次逐项加；前面每个专家内部 Gate/Up 的相加已单独计数。',Cost(add=7*m*H),f'7MH={fmt(7*m*H)} 次加法。')
    f.unknown+=['FP4/FP8 的权重更新、STE 或主权重精度未由推理代码提供。此处权重视为实数可微参数，未添加任何未知辅助 loss。']
    return f

def query(m, backward=False):
    f=Flow('Query 低秩投影','[M,5120] → [M,1280] → RMSNorm → [M,32768]；Indexer 另外读取归一化的 1280 维表示，本面板不重复计 Indexer。')
    if backward:
        f.extend(linear_b('W_QB',m,R,A*D,pid='qb')).extend(norm_b('Q RMSNorm',m,R,'qnorm')).extend(linear_b('W_QA',m,H,R,pid='qa'))
    else:
        f.extend(linear_f('W_QA',m,H,R,pid='qa')).extend(norm_f('Q RMSNorm',m,R,'qnorm')).extend(linear_f('W_QB',m,R,A*D,pid='qb'))
    return f

def output_projection(m, backward=False):
    f=Flow('Attention 输出低秩投影','64×512 的 Head 输出先分 8 组，每组 4096→1024，拼成 8192 后投影到 5120；不是一张稠密 32768→8192。')
    if backward:
        f.extend(linear_b('W_OB',m,8192,H,pid='ob')).extend(linear_b('W_OA 分组',m,4096,1024,8,'oa'))
    else:
        f.extend(linear_f('W_OA 分组',m,4096,1024,8,'oa')).extend(linear_f('W_OB',m,8192,H,pid='ob'))
    return f

def attention_layout(scenario, ratio=1):
    m=SCENARIOS[scenario]['length']; prior=SCENARIOS[scenario]['prior']
    local=[min(C['window_size'],prior+t) for t in range(1,m+1)]
    glob=[min(C['index_topk'],(prior+t)//ratio) if ratio else 0 for t in range(1,m+1)]
    lengths=[x+y for x,y in zip(local,glob)]
    unique_local=min(prior+m,C['window_size']) if m==1 else m
    unique_global=max(glob)
    return m,local,glob,lengths,unique_local+unique_global

def attention_core(scenario, ratio=1, backward=False):
    m,local,glob,lengths,unique=attention_layout(scenario,ratio)
    pairs=sum(lengths); cells=A*pairs
    f=Flow('Attention · 核心读取',f'每行有效 KV 数 = Local min(128,t) + Global min(512,⌊t/{ratio or 1}⌋)'+('；本面板为纯 SWA，Global=0。' if not ratio else '。')+' K 与 V 由同一个 512 维缓存向量提供。只对有效项计数学图算术；FlashAttention 分块、padding 与在线重标定开销不与此简化图混算。')
    if not backward:
        f.step('Attention · 选址读取与逻辑拼接','把已选 Global 和 Local 位置送入同一次归一化。',f'Global IDs、Local IDs；各位置 KV [512]',f'每行 KV [n_t,512]，Σn_t={pairs}',r'KV_t=\operatorname{Gather}(Cache,I_t)',
          'Global 只取最多 512 个索引；Local 最多 128 个。Gather/Concat 是读取语义，不新增浮点乘加。不同来源的相同原始时间位置仍是不同缓存条目。',source_symbol='model.py · Attention.forward / kernel.py · sparse_attn_kernel')
        mul=cells*D; add=cells*(D-1)
        f.step('Attention · QK 相关性','比较每个 Query Head 与选中的 KV。',f'Q_t [64,512]，KV_t [n_t,512]',f'Z_t [64,n_t]',r'Z_{h,j}=\sum_d Q_{h,d}KV_{j,d}',
          '64 个 Query Head 读取同一 KV，不把共享缓存复制后当成新参数。',Cost(mul,add,gemm_delta=2*cells*D-mul-add),f'有效点积数 AΣn_t={fmt(cells)}；每个 D 乘+(D−1) 加；乘 {fmt(mul)}、加 {fmt(add)}，2MKN 近似 {fmt(2*cells*D)}。',source_symbol='kernel.py · sparse_attn_kernel')
        f.step('Attention · 缩放并加入 sink','把普通位置分数与只参与分母的 sink 一起归一化。',f'Z_t [64,n_t]，sink [64]',f'Z_aug,t [64,n_t+1]',r'Z_{aug}=[Z/\sqrt{512},a_{sink}]',
          '1/√512 是预计算常量；sink 没有 Value 向量，不能把它作为 KV 读取。',Cost(cells),f'{fmt(cells)} 次常量乘；追加 sink 与 mask 无浮点算术。', ['sink'],source_symbol='kernel.py · sparse_attn_kernel')
        f.extend(softmax('Attention · 共同 Softmax',[n+1 for n in lengths],prefix=A))
        mul=cells*D;add=A*D*sum(n-1 for n in lengths)
        f.step('Attention · 权重乘 Value','用共同归一化的权重汇总 KV。',f'P_t [64,n_t]，V_t=KV_t [n_t,512]',f'O {shp(m,64,512)}',r'O_{h,d}=\sum_jP_{h,j}KV_{j,d}',
          'Local 与 Global 使用同一个分母。丢弃 sink 的概率质量，不给 sink 补一个 Value。',Cost(mul,add,gemm_delta=2*cells*D-mul-add),f'乘 A D Σn_t={fmt(mul)}；加 A D Σ(n_t−1)={fmt(add)}；矩阵近似 {fmt(2*cells*D)}。',source_symbol='kernel.py · sparse_attn_kernel')
    else:
        mul=cells*D;add=cells*(D-1)
        f.step('Attention⁻¹ · dP 与 Value 路径','从汇总结果反传到概率与缓存 Value。',f'dO {shp(m,64,512)}，KV_t [n_t,512]，P_t [64,n_t]',f'dP_t [64,n_t]；每次读取的 dV [512]',r'dP=dO\,V^{\mathsf T},\quad dV_{read}=P\,dO',
          'dV 是每次读取的外积贡献，此处未合并同一缓存行的重复贡献。',Cost(2*mul,add,gemm_delta=cells+cells*D),f'dP：{fmt(mul)} 乘+{fmt(add)} 加；dV 每次为标量×512，{fmt(mul)} 乘，没有单行内约简。矩阵近似分别 {fmt(2*mul)}、{fmt(2*mul)}。')
        f.extend(softmax('Attention⁻¹ · 含 sink 的 Softmax',[n+1 for n in lengths],True,A))
        f.step('Attention⁻¹ · sink 累加与缩放梯度','sink 在 Query 行间共享；QK 分数缩放也要反传。',f'dZ_aug,t [64,n_t+1]',f'dsink [64]，dZ_t [64,n_t]',r'da_{sink,h}=\sum_t dZ_{aug,t,h,sink},\quad dZ=dZ_{aug,real}/\sqrt{512}',
          'sink 的 dP 输入为 0，但其分母作用让 dsink 通常非零。',Cost(cells,A*(m-1)),f'普通位置 {fmt(cells)} 次常量乘；sink 跨 {m} 行，{fmt(A*(m-1))} 加。', ['sink'])
        mul=cells*D;add=A*D*sum(n-1 for n in lengths)
        f.step('Attention⁻¹ · dQ 与 Key 路径','通过 QK 点积回到 Query 与缓存 Key。',f'dZ_t [64,n_t]，Q_t [64,512]，K_t [n_t,512]',f'dQ {shp(m,64,512)}，每次读取 dK [512]',r'dQ=dZK,\quad dK_{read}=dZ^{\mathsf T}Q',
          'dK 与上一步 dV 都作用到同一共享 KV；两个分支都必须保留。',Cost(2*mul,add,gemm_delta=4*mul-(2*mul+add)),f'dQ：{fmt(mul)} 乘+{fmt(add)} 加；dK 是每 Query/Head 的外积，{fmt(mul)} 乘。矩阵近似两项各 {fmt(2*mul)}。')
        additions=(2*cells-unique)*D
        f.step('Attention⁻¹ · Scatter-add 到共享 KV','合并同一缓存行的 Key、Value、Head 和 Query 梯度。',f'2×64×Σn_t={2*cells} 个 [512] 贡献；本例 {unique} 个互异缓存条目',f'dKV_cache [{unique},512]',r'dKV_j=\sum_{t,h:j\in I_t}(dK_{t,h,j}+dV_{t,h,j})',
          '采用每个目标首项赋值，其余累加的严格归约图。A 的梯度止于缓存叶子；B 将每个条目继续传回产生它的 KV/Compressor。跨层共享还需在拥有层汇总各消费者，不能重跑每层 Key 投影。',Cost(add=additions),f'(2AΣn_t−U)D=({2*cells}−{unique})×512={fmt(additions)} 次加法。此例选中位置可枚举；更长训练序列的 U 依赖实际索引。')
        f.unknown+=['这是有效位置展开的理论注意力 VJP；官方稀疏训练 kernel 的分块、通信、保存/重算与低精度梯度规则未验证。跨层 KV 归约由拥有层负责，本核心面板不声称包含整个网络的缓存反向。']
    return f

def compressor(scenario, ratio=2, backward=False):
    m=SCENARIOS[scenario]['length']; prior=SCENARIOS[scenario]['prior']
    new=(m//ratio if scenario=='B' else int((prior+1)%ratio==0))
    f=Flow('Global KV Compressor',f'源层 ratio={ratio}；当前输入 {m} 行，新完成 {new} 组。Compressor 输出为 RoPE 之前的归一化 latent，外部 Attention 再执行 Global KV RoPE。')
    if not backward:
        f.extend(linear_f('W_CKV',m,H,D,pid='ckv'))
        if ratio>1:f.extend(linear_f('W_Cgate',m,H,D,pid='cgate'))
        if ratio>1 and new:
            f.extend(softmax('Compression · 每通道对分组内 token 做 Softmax',[ratio]*(new*D)))
            f.step('Compression · 按 gate 池化','把一组相邻 token 合成一个 latent。',f'KV,α {shp(new,ratio,D)}',f'Z {shp(new,D)}',r'Z_{g,d}=\sum_{u=1}^{r}\alpha_{g,u,d}KV_{g,u,d}',
              'Softmax 沿 token 轴，每个通道有自己的权重，不是整个 512 向量共享一个门。',Cost(new*ratio*D,new*(ratio-1)*D),f'GrD={fmt(new*ratio*D)} 乘 + G(r−1)D={fmt(new*(ratio-1)*D)} 加。',source_symbol='model.py · Compressor.forward')
        if new:f.extend(norm_f('Compressed RMSNorm',new,D,'cnorm'))
        else:
            f.step('Compression · 等待配对','保存当前 KV 与 gate，等待第 1026 个 token。',f'KV,score {shp(m,D)}', '未完成组缓存；本次返回 None',r'\operatorname{emit}=((1024+1)\bmod2=0)=\mathrm{False}',
              '第 1025 个 token 是 ratio=2 新组的第一项。本次没有池化、归一化和新 Global KV 写入；但两次投影已经执行。',source_symbol='model.py · Compressor.forward')
    else:
        if new:f.extend(norm_b('Compressed RMSNorm',new,D,'cnorm'))
        if ratio>1 and new:
            f.step('Compression⁻¹ · 池化两条梯度','把 latent 梯度送回每个 token 和 gate。',f'dZ {shp(new,D)}，α,KV {shp(new,ratio,D)}',f'dKV,dα {shp(new,ratio,D)}',r'dKV_u=\alpha_u dZ,\quad d\alpha_u=KV_u dZ',
              '被压缩组内的早期 token 同样接收梯度；不能把前一个 token 当作训练时免费的缓存。',Cost(2*new*ratio*D),f'2GrD={fmt(2*new*ratio*D)} 次乘法。')
            f.extend(softmax('Compression⁻¹ · 分组 Softmax',[ratio]*(new*D),True))
        elif ratio>1:
            f.step('Compression⁻¹ · 延迟梯度边界','本次没有输出 latent，尚无来自下游的当前组梯度。',f'当前暂存 KV,score {shp(m,D)}', '未来配对时由缓存回传 dKV / dscore',r'dKV,dscore\ \text{由未来完整组反向提供}',
              '下面两次投影 VJP 是给定未来暂存梯度后的条件计算，不是本次 Decode 已执行反向。',path='conditional')
        path='conditional' if ratio>1 and not new else None
        if ratio>1:f.extend(linear_b('W_Cgate',m,H,D,pid='cgate'),path=path)
        f.extend(linear_b('W_CKV',m,H,D,pid='ckv'),path=path)
        if ratio>1:
            f.step('Compression⁻¹ · 合并两分支输入梯度','KV 与 gate 两个投影都读取同一 X。',f'dX_KV,dX_gate {shp(m,H)}',f'dX {shp(m,H)}',r'dX=dX_{KV}+dX_{gate}',
              '跨层所有消费者的 latent 梯度应先在源层相加，再做一次本 Compressor VJP。',Cost(add=m*H),f'MH={fmt(m*H)} 次加法。',path=path or 'main')
        f.unknown+=['全网络的共享缓存梯度需要消费者的实际索引和使用次数；本面板只计算一次源层调用。低精度量化梯度与缓存生命周期/重算策略未提供。']
    return f

def indexer(scenario, backward=False, ratio=1, owns_key=True):
    m=SCENARIOS[scenario]['length']; prior=SCENARIOS[scenario]['prior']; s=(prior+m)//ratio
    new=s if scenario=='B' and owns_key else int(owns_key and (prior+1)%ratio==0)
    c=s; cells=m*IH*c
    f=Flow('Indexer',f'{"Full 源层" if owns_key else "Reindex"}，ratio={ratio}；Index Key cache 长度 S={s}。参考代码对全部可用 S 行打分后再做因果/候选 mask，因此评分矩阵 C={c}，不是 Top-K=512，也不是只算掩码后的项。')
    if not backward:
        f.extend(linear_f('W_IQ',m,R,IH*ID,pid='iq')).extend(rope('Index Query RoPE',m,IH,dim=ID))
        if new:f.extend(linear_f('W_IK',new,D,ID,pid='ik')).extend(norm_f('Index Key RMSNorm',new,ID,'inorm')).extend(rope('Index Key RoPE',new,1,dim=ID))
        f.step('Indexer · 历史 Key 复用','读取源层缓存，只为新完成条目生成 Key。',f'Index Key cache [{s},128]',f'K_I [{s},128]',r'K_I=Cache_{index}[:S]',
          f'本次仅投影 {new} 个新 Key，不能每次重投影全部 {s} 个历史条目。',source_symbol='model.py · Indexer.forward')
        f.extend(linear_f('W_IH',m,H,IH,pid='ih'))
        f.step('Indexer · Head Weights 缩放','由当前 X 决定各 Index Head 的分数贡献。',f'w_raw {shp(m,IH)}',f'w {shp(m,IH)}',r'w=w_{raw}/\sqrt{128\times32}',
          '分母是预计算常量，按常量乘法执行；不是全局固定的 32 个标量，也不是 MoE 路由权重。',Cost(m*IH),f'M×32={fmt(m*IH)} 次乘法。',source_symbol='model.py · Indexer.forward')
        mul=cells*ID;add=cells*(ID-1)
        f.step('Indexer · 相似度点积','每个 Query 的 32 个 Head 都扫描 Index Keys。',f'Q_I {shp(m,32,128)}，K_I [{c},128]',f'Z {shp(m,32,c)}',r'Z_{t,h,j}=\sum_d Q_{t,h,d}K_{j,d}',
          '实际评分工作量随 C 增长；最终只读取 512 个位置不能减少已执行的评分点积。',Cost(mul,add,gemm_delta=2*mul-mul-add),f'M×32×C={fmt(cells)} 个点积；每个 128 乘+127 加。共 {fmt(mul)} 乘 + {fmt(add)} 加；近似 {fmt(2*mul)}。',source_symbol='model.py · Indexer.forward')
        f.step('Indexer · ReLU 与 Head Aggregation','给每个历史位置形成一个总相关性分数。',f'Z {shp(m,32,c)}，w {shp(m,32)}',f'Score {shp(m,c)}',r'Score_{t,j}=\sum_h w_{t,h}\max(Z_{t,h,j},0)',
          'ReLU 比较另列，不是浮点乘加；聚合前必须使用可学习投影得到的 Head Weights。',Cost(cells,m*c*(IH-1)),f'ReLU {fmt(cells)} 次比较（排除）；加权 {fmt(cells)} 乘；跨 Head 求和 {fmt(m*c*(IH-1))} 加。',source_symbol='model.py · Indexer.forward')
        f.step('Indexer · Mask / 分层候选 / Top-K','根据分数返回最多 512 个整数位置。',f'Score {shp(m,c)}，causal mask / candidate blocks',f'I [M,K≤512] int',r'I=\operatorname{sort}_{position}(\operatorname{TopK}_{score}(Score\mid mask))',
          'Layer 20 的候选块池最多 2048×8=16384 个位置；后续 Reindex 先算全 cache 分数再应用候选 mask。比较、整数选择和索引输出不计浮点 FLOPs，候选块最大值使用比较而非乘加。',source_symbol='model.py · Indexer.forward / select_candidate_blocks')
        f.step('Indexer · Global KV Gather','用整数索引读取已有 Main KV。',f'I [M,K≤512]，Global KV [{s},512]', '每个 Query 至多 [512,512] 的逻辑读取',r'KV_{sel}=GlobalCache[I]',
          '输出不是新生成的 KV。与 Local ≤128 行共同送入 Main Attention，相关点积在 Attention 面板计算。',source_symbol='model.py · Attention.forward')
    else:
        _,_,gl,_,_=attention_layout(scenario,ratio); refs=sum(gl); unique=max(gl)
        f.step('Gather⁻¹ · Global KV 梯度累加','把选中 KV 的梯度散射回原缓存。',f'dKV_sel：{refs} 行 [512]；{unique} 个互异全局行',f'dGlobalKV [{s},512]',r'dGlobalKV_j=\sum_{t:j\in I_t}dKV_{sel,t,j}',
          '这里输入梯度已经合并 Main Attention 的 Head / K / V；本步只演示跨 Query 的 Gather VJP。与完整 Attention 面板中的缓存 Scatter 是同一操作，组合时不能再加一次。',Cost(add=(refs-unique)*D),f'(引用行数−互异行数)×512=({refs}−{unique})×512={fmt((refs-unique)*D)} 加。')
        f.step('Top-K⁻¹ · 主损失的离散边界','位置索引不能携带普通连续梯度。','dKV_sel 与整数 I','主路径没有由 dI 得到的 dScore',r'\frac{\partial I}{\partial Score}\ \text{不存在普通连续导数}',
          'Main loss 可以训练选中的 Global KV，但不会通过整数 Top-K 自动训练 Index Query / Keys / Head Weights。不能把选择器当成 Softmax。',status='Unknown')
        f.step('条件分支 · 给定额外 dScore','展示评分子图本身可推导的 VJP。',f'外部 G_score {shp(m,c)}','以下条件 VJP；不计入主路径小计',r'G=\partial L_{aux}/\partial Score',
          '这里只把 G 作为给定输入，不假设真实训练存在某种辅助损失或 STE；来源 Unknown。',path='conditional',status='Unknown')
        f.step('Aggregation⁻¹ · Head 权重与 ReLU','从给定分数梯度分解到每个 Head。',f'G [M,C]，ReLU(Z) [M,32,C]，w [M,32]',f'dw [M,32]，dZ [M,32,C]',r'dw_{t,h}=\sum_jG_{t,j}ReLU(Z_{t,h,j}),\quad dZ_{t,h,j}=G_{t,j}w_{t,h}1_{Z>0}',
          '采用 mask 选择梯度而非再乘一个浮点 mask；ReLU 在 0 点取零子梯度。',Cost(2*cells,m*IH*(c-1)),f'dw：{fmt(cells)} 乘+{fmt(m*IH*(c-1))} 加；dZ：{fmt(cells)} 乘，ReLU 的选择/比较另列。',path='conditional')
        f.extend(linear_b('Index QK · dQ/dK',m*IH,ID,c),path='conditional')
        f.steps[-1]['note']=f.steps[-1]['why']='这里线性 W=K_Iᵀ 是激活缓存，不是学习权重；dW 转置后为 dK_I [C,128]，已合并全部 Query / Head。'
        f.extend(rope('Index Query RoPE',m,IH,True,ID),path='conditional')
        f.extend(linear_b('W_IQ',m,R,IH*ID,pid='iq'),path='conditional')
        f.step('Head scale⁻¹','将 dw 传回 Head Weight 投影。',f'dw {shp(m,IH)}',f'dw_raw {shp(m,IH)}',r'dw_{raw}=dw/\sqrt{128\times32}',
          '前向使用常量乘，反向也使用同一常量。',Cost(m*IH),f'M×32={fmt(m*IH)} 乘。',path='conditional')
        f.extend(linear_b('W_IH',m,H,IH,pid='ih'),path='conditional')
        if new:
            f.extend(rope('Index Key RoPE',new,1,True,ID),path='conditional').extend(norm_b('Index Key RMSNorm',new,ID,'inorm'),path='conditional').extend(linear_b('W_IK',new,D,ID,pid='ik'),path='conditional')
        f.step('缓存边界 · 两种 latent 梯度','在源层汇总使用同一 latent 的不同分支。',f'主 Attention dlatent 与条件 Index Key dlatent 各 [{new},512]',f'dlatent [{new},512]',r'dlatent=dlatent_{MainKV}+dlatent_{IndexKey}',
          '只有存在额外评分训练梯度时才有第二项；先汇总再执行 Compressor VJP。Reindex 不拥有 Key 投影；复用层返回缓存梯度给拥有层。',Cost(add=new*D),f'{new}×512={fmt(new*D)} 加。',path='conditional')
        f.unknown+=['Indexer 的训练目标、额外 dScore 的来源、离散选择估计器和量化反向规则均未由推理实现给出。主路径和条件评分 VJP 分列，不能相加称作官方训练反向。']
    return f

def engram_lookup(m, backward=False):
    n=m*source.EW; rows=m*source.HASH_COLS
    f=Flow('Engram N-gram / Lookup','每 token 构造 2/3/4-gram ×8 heads =24 个地址；每地址读取256维。仅处理文本有效 token。')
    if not backward:
        f.step('N-gram · 构造历史编号组','读取当前和最多前三个编号。',f'token IDs [{m}] int 与历史编号缓存',f'压缩词表 IDs {shp(m,4)} int',
          r't_{t,u}=token\_map[token_{t-u}],\quad u=0,1,2,3',
          '越界/图像段用 pad 处理，不能跨 DEAD token 构造 n-gram。所有地址构造都是整数/索引，不是浮点 FLOPs。',source_symbol='engram.py · NgramHashState.forward')
        f.step('Hash · 生成24个查表地址','通过乘法、XOR、模和偏移定位记忆行。',f'IDs {shp(m,4)} int，multipliers / primes / offsets',f'hash IDs {shp(m,24)} int',
          r'h_{n,a}=(\operatorname{XOR}_{u<n}(t_u\,mult_u)\bmod prime_{n,a})+offset_{n,a}',
          '2/3/4-gram 分别8个桶。整数乘法不能与浮点矩阵 FLOPs 混算；哈希比较和访存单独说明。',source_symbol='engram.py · NgramHashState.forward')
        f.step('Lookup · 读表与显式反量化','读取24行记忆并还原其幅度。',f'hash IDs {shp(m,24)}，E rows [256]，固定scale',f'E_lookup {shp(m,24,256)} → E_flat {shp(m,6144)}',
          r'E_{lookup}=\operatorname{Gather}(E,ids)\odot scale',
          'Lookup/flatten 无乘加，显式的逐项反量化乘法要计入；表scale的训练规则另外标为 Unknown。',Cost(n),f'M×24×256={fmt(n)} 次乘法。', ['etable1','etable14'],source_symbol='model.py · ParallelEngramEmbedding.forward')
    else:
        f.step('Lookup⁻¹ · 显式反量化乘法的梯度','保留前向values×scale的可微实数路径。',f'dE_lookup,scale {shp(m,24,256)}，scale按32维广播',f'dE_raw_lookup {shp(m,24,256)}',r'dE_{raw}=scale\odot dE_{lookup}',
          '主路径把scale固定、原始表值当实数代理参数；不是宣称FP8舍入有普通导数。若训练反量化后的表本身，这一步应省略；这里明确采用前者，与前向显式反量化对应。',Cost(n),f'M×24×256={fmt(n)} 次乘法。')
        f.step('Lookup⁻¹ · 将梯度写回记忆行','固定地址，把 dE_flat 还原为逐行贡献。',f'dE_flat {shp(m,6144)}，整数 IDs {shp(m,24)}', '稀疏 dE[row,256]；无 dIDs',r'dE_{row}=\operatorname{ScatterAdd}(ids,dE_{lookup})',
          '教学条件设24M个地址互异，Scatter首项赋值，浮点加法0；实际U个不同地址时需(24M−U)×256加法。整数Hash/ID没有普通连续梯度。',Cost(),f'本例 U=24M={rows}，加法 ({rows}−{rows})×256=0。实际 U 必须根据输入 IDs 计算。', ['etable1','etable14'])
        f.step('条件路径 · 表 scale 的理论梯度','如果scale也作为实数训练变量，梯度是每32维一个点积。',f'dE_lookup,E_raw {shp(m,24,256)}',f'dscale_lookup {shp(m,24,8)}',
          r'dscale_b=\sum_{d\in block_b}dE_{lookup,d}E_{raw,d}',
          '源码fp8_block_size=32，256维每行有8个scale。是否实际训练scale、精度或更新策略未知，因此单列条件VJP，不并入固定scale主路径。',Cost(n,m*24*8*31),
          f'M×24×8×32={fmt(n)} 乘 + M×24×8×31={fmt(m*24*8*31)} 加；地址互异时scale行Scatter不额外相加。',path='conditional')
        f.unknown+=['FP8 表与scale的实际训练变量、主精度副本和量化更新策略未知；主路径假设scale固定、原始值为实数代理，scale可训练时的理论VJP已另列。']
    return f

def engram_gate(m, backward=False):
    n=m*HC*H; r=m*HC; f=Flow('Engram Projection / Gating','归一化按每 token、每路的5120维进行；不是把四路一起归一化。q_weight 与 k_weight 的乘积在本次调用中只生成一次。')
    if not backward:
        f.extend(linear_f('W_E',m,source.EW,(HC+1)*H,pid='ew'))
        f.step('Engram · 拆分 Key / Value 与门权重','从记忆产生四路 Key 和共享 Value。',f'Projected {shp(m,25600)}，q_weight,k_weight [4,5120]',[f'K {shp(m,4,H)}, V {shp(m,H)}','w [4,5120]'],
          r'K,V=\operatorname{split}(E_{flat}W_E),\quad w=q_{weight}\odot k_{weight}',
          'split是视图；w在 token 之间共享，不能按 M 重复生成。',Cost(HC*H),f'4×5120={fmt(HC*H)} 次乘法。', ['eq','ek'],source_symbol='model.py · Engram.forward')
        f.step('Engram · 残差和 Key 的两组 RMS 统计','计算 h 与 Key 的相对尺度。',f'h,K {shp(m,4,H)}',f'r_h,r_k,rstd {shp(m,4)}',
          r'r_h=(mean(h^2)+\epsilon)^{-1/2},\ r_k=(mean(K^2)+\epsilon)^{-1/2},\ rstd=r_hr_k',
          '这里只归一化点积，没有乘普通 RMSNorm γ；q_weight/k_weight 已在 w 中。',Cost(2*r*(H+1)+r,2*r*H,special={'rsqrt':2*r}),
          f'两组统计各：{fmt(r*H)} 平方乘、{r} 均值乘、{fmt(r*(H-1))} 行和加、{r} epsilon加、{r} rsqrt；r_h r_k 再 {r} 乘。',source_symbol='model.py · Engram.forward')
        f.step('Engram · 加权归一化点积','判断记忆 Key 与当前残差是否匹配。',f'h,K {shp(m,4,H)}，w [4,5120]，rstd [M,4]',f'dot {shp(m,4)}',
          r'v=\sum_d h_dw_dK_d,\quad dot=v\,rstd/\sqrt{5120}',
          '乘积 h×w×K 需要两次逐项乘，不能只按一次普通点积计数。',Cost(2*n+2*r,r*(H-1)),f'2MCH={fmt(2*n)} 乘 + MC(H−1)={fmt(r*(H-1))} 加；rstd和预计算缩放各 {r} 乘。',source_symbol='model.py · Engram.forward')
        f.step('Engram · signed sqrt 与 Sigmoid','将相关性转成注入门。',f'dot {shp(m,4)}',f'gate {shp(m,4)}',
          r's=\operatorname{copysign}(\sqrt{\max(|dot|,10^{-6})},dot),\quad gate=\sigma(s)',
          '保留正负号的平方根后再做 Sigmoid。abs/clamp/copysign 属于比较或符号操作；不是额外矩阵。',Cost(0,r,r,{'sqrt':r,'exp':r,'div':r}),
          f'MC={r} 个门；Sigmoid {r} 减、{r} 加、{r} exp、{r} div；signed sqrt {r} sqrt，符号/比较另列。',source_symbol='model.py · Engram.forward')
    else:
        f.step('Gate⁻¹ · Sigmoid 与 signed sqrt','沿连续门函数回到点积。',f'dgate,gate,dot {shp(m,4)}',f'ddot {shp(m,4)}',
          r'ds=dgate\,gate(1-gate),\quad ddot=\begin{cases}ds/(2\sqrt{|dot|})&|dot|>10^{-6}\\0&0<|dot|<10^{-6}\end{cases}',
          '复用前向 sqrt/σ；在 clamp 饱和区只做mask选择。dot=0有符号跳变，±阈值处不可微，边界的子梯度约定未由训练实现给出。此计数对每项执行安全分母计算再选择。',Cost(3*r,sub=r,special={'div':r}),f'σ反向2MC乘+MC减；2sqrt再MC乘，MC除。')
        f.step('Dot⁻¹ · 标量缩放与两个 RMS 因子','让点积数值和两组尺度都接收梯度。',f'ddot,v,rstd,r_h,r_k {shp(m,4)}',f'dv,dr_h,dr_k {shp(m,4)}',
          r'dv=ddot\,rstd/\sqrt H,\quad drstd=ddot\,v/\sqrt H,\quad dr_h=drstd\,r_k,\quad dr_k=drstd\,r_h',
          'r_h 与 r_k 都影响最终门，不能停止它们的梯度。',Cost(6*r),f'dv 2MC乘，drstd 2MC乘，dr_h/dr_k各MC乘，共 {fmt(6*r)}。')
        f.step('RMS统计⁻¹ · h 与 Key 的尺度梯度','通过每路均方统计回到向量。',f'dr_h,dr_k,r_h,r_k [M,4]；h,K {shp(m,4,H)}',f'dh_norm,dK_norm {shp(m,4,H)}',
          r'dx_{norm}=-(dr\,r^3/H)x',
          '分别复用两组 r；每组每行4次标量乘，再H次逐项乘。',Cost(2*r*(H+4)),f'2MC(H+4)={fmt(2*r*(H+4))} 次乘法。')
        f.step('Dot⁻¹ · h / Key / w 的逐项梯度','反传三因子加权点积。',f'dv [M,4]；h,K [M,4,5120]；w [4,5120]',f'dh_dot,dK_dot [M,4,5120]；dw [4,5120]',
          r'dh=dv\,wK,\quad dK=dv\,hw,\quad dw=\sum_t dv\,hK',
          'dw 跨所有 token 归约，因为同一 w 在所有行共享。',Cost(6*n,(m-1)*HC*H),f'三个梯度各2MCH乘，共 {fmt(6*n)}；dw跨M行，(M−1)CH={fmt((m-1)*HC*H)} 加。')
        f.step('Weight⁻¹ · q_weight / k_weight','把乘积权重梯度分给两套参数。','dw,q_weight,k_weight [4,5120]','dq_weight,dk_weight [4,5120]',
          r'dq_{weight}=dw\,k_{weight},\quad dk_{weight}=dw\,q_{weight}',
          '只在已经跨 token 累加的 dw 上计算一次。',Cost(2*HC*H),f'2CH={fmt(2*HC*H)} 乘。', ['eq','ek'])
        f.step('Key⁻¹ · 汇总两条路径','Key 同时影响加权点积与归一化。',f'dK_dot,dK_norm {shp(m,4,H)}',f'dK {shp(m,4,H)}',r'dK=dK_{dot}+dK_{norm}',
          '与已从 Injection 得到的 dV 拼接，组成 Projection 的输出梯度。',Cost(add=n),f'MCH={fmt(n)} 加。')
        f.extend(linear_b('W_E',m,source.EW,(HC+1)*H,pid='ew'))
        f.unknown+=['signed sqrt 在 dot=0 或 clamp 阈值的训练导数约定未知；数值检验避开这些点。']
    return f

def injection(m, backward=False):
    n=m*HC*H; r=m*HC; f=Flow('Engram Residual Injection')
    if not backward:
        f.step('Residual Injection','按每路 gate 加入共享记忆 Value。',f'h {shp(m,4,H)}, gate {shp(m,4)}, V {shp(m,H)}',f'Y {shp(m,4,H)}',
          r'Y_{c,d}=h_{c,d}+gate_cV_d','四路读取同一 Value，但每路有自己的门；不是复制四份 Value 参数。',Cost(n,n),f'MCH={fmt(n)} 乘 + MCH={fmt(n)} 加。',source_symbol='model.py · Engram.forward')
    else:
        f.step('Injection⁻¹ · 残差、gate 与共享 Value','从最终输出拆出直连路径和门控路径。',f'G=dY {shp(m,4,H)}，V {shp(m,H)}，gate {shp(m,4)}',
          [f'dh_direct {shp(m,4,H)}',f'dgate {shp(m,4)}',f'dV {shp(m,H)}'],
          r'dh_{direct}=G,\quad dgate_c=\sum_dG_{c,d}V_d,\quad dV_d=\sum_cG_{c,d}gate_c',
          '直连dh是视图；dV要在四路之间累加，dgate要在5120维特征之间累加。',Cost(2*n,r*(H-1)+m*(HC-1)*H),
          f'dgate MCH乘+MC(H−1)加；dV MCH乘+M(C−1)H加。共 {fmt(2*n)} 乘 + {fmt(r*(H-1)+m*(HC-1)*H)} 加。')
    return f

def engram(m, backward=False):
    f=Flow('Engram 完整图','整数地址固定；实数代理图从记忆读取值到Projection、归一化门控、残差注入。前向FP8读表反量化显式计入，反向实数记忆叶子与量化训练分开。')
    if backward:
        f.extend(injection(m,True)).extend(engram_gate(m,True))
        f.step('Engram⁻¹ · 残差三条路径累加','同一个 h 同时影响直连输出、门的点积与门的归一化。',f'dh_direct,dh_dot,dh_norm 各 {shp(m,4,H)}',f'dh {shp(m,4,H)}',
          r'dh=dh_{direct}+dh_{dot}+dh_{norm}','三项需要两次逐项加法。',Cost(add=2*m*HC*H),f'2MCH={fmt(2*m*HC*H)} 加。')
        f.extend(engram_lookup(m,True))
    else:
        f.extend(engram_lookup(m)).extend(engram_gate(m)).extend(injection(m))
    return f

def attention(scenario, layer=20, backward=False):
    m=SCENARIOS[scenario]['length']; ratio=C['compress_ratios'][layer]
    owner=layer in C['kv_source_layers']; indexed=layer in C['index_source_layers']
    f=Flow(f'Layer {layer} 完整 Attention',f'按配置执行：ratio={ratio}，KV owner={owner}，Index source={indexed}。所有层都有自己的 Q、Local KV 和输出投影。')
    if backward:
        f.extend(output_projection(m,True)).extend(rope('Inverse RoPE 的 VJP',m,A,True))
        f.extend(attention_core(scenario,ratio,True))
        f.step('缓存梯度 · 划定本次可继续反传的条目','先完成缓存内的归约，再回到生产这些条目的计算图。',
          '已合并的 Local / Global cache 梯度',f'本次 Local dKV {shp(m,D)}；Global 梯度发送到源层',
          r'dKV_{new}=\operatorname{Slice}(dKV_{cache},new\_positions)',
          'A只向当前新Local条目和完成的新Global条目继续反传，其余历史梯度返回为叶子梯度；B的128个Local位置全部由本序列产生并参与投影反向。复用Global的层不创建或重算源层投影。')
        f.extend(rope('Local KV RoPE 的 VJP',m,1,True)).extend(norm_b('Local KV RMSNorm',m,D,'kvnorm')).extend(linear_b('W_KV',m,H,D,pid='kv'))
        # The main task has no gradient through integer index selection.
        if owner:
            new=m//ratio if scenario=='B' else int(1025%ratio==0)
            if new:f.extend(rope('Global KV RoPE 的 VJP',new,1,True))
            f.extend(compressor(scenario,ratio,True))
        f.extend(rope('Query RoPE 的 VJP',m,A,True)).extend(query(m,True))
        paths=3 if owner else 2
        if owner and scenario=='A' and ratio>1:paths=2
        f.step('Attention⁻¹ · 同一输入的分支梯度相加','合并本层 Q、Local KV 以及拥有时的 Compressor 输入梯度。',f'{paths} 条 dX {shp(m,H)}',f'dX {shp(m,H)}',
          r'dX=dX_Q+dX_{Local}+dX_{Compressor}\ (\text{仅源层})',
          '主损失没有沿 Indexer 整数选址产生 dX；条件评分 VJP 不并入此小计。若本层复用 Global KV，返回的全局缓存梯度交给源层，不当作本层 X 的梯度。',Cost(add=(paths-1)*m*H),f'({paths}−1)MH={fmt((paths-1)*m*H)} 加。')
    else:
        f.extend(query(m)).extend(rope('Main Query RoPE',m,A))
        f.extend(linear_f('W_KV',m,H,D,pid='kv')).extend(norm_f('Local KV RMSNorm',m,D,'kvnorm')).extend(rope('Local KV RoPE',m,1))
        if owner:f.extend(compressor(scenario,ratio))
        if indexed:f.extend(indexer(scenario,False,ratio,owner))
        elif ratio:
            f.step('Attention · 复用 Global KV 与位置索引','读取之前发布的缓存和 Top-K 结果。','源层 Global KV cache；最近 Index source 的 I','本层选中的 Global KV',
              r'KV=shared\_attn.compress\_kv,\quad I=shared\_attn.topk\_idxs',
              'Reuse 不运行 Indexer，但本层仍执行 Main Attention。缓存读取本身0浮点 FLOPs。',source_symbol='model.py · Attention._compress_kv / _compress_topk_idxs')
        if owner:
            new=m//ratio if scenario=='B' else int(1025%ratio==0)
            if new:f.extend(rope('Global KV RoPE / 写入前',new,1))
        f.extend(attention_core(scenario,ratio)).extend(rope('Inverse RoPE',m,A)).extend(output_projection(m))
    f.unknown+=['矩阵输入/权重的 FP4/FP8 量化、量化训练导数和 kernel 实际指令数不在本实数账本内；特殊函数次数单列。']
    return f

def layer_flow(scenario, layer, backward=False):
    m=SCENARIOS[scenario]['length']; f=Flow(f'Layer {layer}', '这里合计本层各子图的已列算术；跨层梯度调度和共享缓存的额外归约边界另列。')
    if backward:
        f.extend(hc(m,True),'FFN HC').extend(moe(m,True)).extend(norm_b('FFN RMSNorm',m,H,'fnorm'))
        f.extend(hc(m,True),'Attention HC').extend(attention(scenario,layer,True)).extend(norm_b('Attention RMSNorm',m,H,'anorm'))
        if layer in C['engram_layer_ids']:f.extend(engram(m,True))
        f.unknown+=['本层列表按子图逆序展示，HC 的 dU→子模块VJP→dZ 接口是联立边界；pre_next 的梯度来自下一子模块。不能把此子图成本清单当作完整网络的训练调度或全局梯度累加账本。']
    else:
        if layer in C['engram_layer_ids']:f.extend(engram(m))
        f.extend(hc(m),'Attention HC').extend(norm_f('Attention RMSNorm',m,H,'anorm')).extend(attention(scenario,layer))
        f.extend(hc(m),'FFN HC').extend(norm_f('FFN RMSNorm',m,H,'fnorm')).extend(moe(m))
    return f

def boundary(key, m, backward):
    f=Flow(key)
    if key=='prediction':
        f.step('预测 / 采样边界','从logits选择下一token，具体采样图需额外设置。','logits [M,129280]','Token ID [M] int',
          r'id=\operatorname{Sampling}(logits)',
          '参考Transformer返回logits；温度、Top-p、输出Softmax及loss未由此架构模块唯一确定。没有为未知采样器编造FLOPs，也不对抽出的整数ID求普通梯度。',status='Unknown')
        f.unknown+=['采样算法/输出Softmax/loss未指定，完整成本Unknown；已列0仅为边界占位，不是采样成本为0。']
    elif key in ['input','tokenizer','ids','ngram','hash']:
        f.step('整数 / 文本边界','准备后续可微模块使用的整数输入。',f'文本或 IDs，{m} 个当前位置','Token IDs / N-gram IDs，整数',
          r'ids=\operatorname{IntegerPreprocess}(input)','分词、整数Hash与离散编号没有普通浮点梯度；0浮点算术不代表CPU工作或内存开销为0。',status='Unknown' if backward else 'Verified',source_symbol='' if backward else 'engram.py · NgramHashState.forward')
        if backward:f.unknown+=['整数ID没有普通连续梯度；不猜测离散优化策略。']
    elif key=='embedding':
        f.step('Embedding · '+('稀疏梯度写回' if backward else '按ID读行'),'根据整数编号读取或更新对应权重行。',f'IDs [{m}] int；E [129280,5120]'+(f'；dY {shp(m,H)}' if backward else ''),f'{"dE 的选中行" if backward else "Y"} {shp(m,H)}',
          r'dE[ids]\mathrel{+}=dY' if backward else r'Y=E[ids]',
          '前向查表0乘加。反向教学条件设M个ID互异，首次赋值0加；若U个不同ID，归约需(M−U)H加。没有dIDs。',refs=['embedding'],source_symbol='' if backward else 'model.py · ParallelEmbedding.forward')
    else:
        f.step('状态 / 输出视图','显示模块边界处已经计算好的状态。',f'已有 Tensor {shp(m,H)}',f'同一 Tensor {shp(m,H)}',r'Y=X' if not backward else r'dX=dY',
          '这里只是读取或视图；它的生成成本应在生产该状态的模块中计算一次，不能再复制到边界上。',source_symbol='model.py · Block.forward')
    return f

def choose_steps(flow, predicate):
    out=Flow(flow.label,flow.scope,flow.unknown)
    out.steps=[deepcopy(s) for s in flow.steps if predicate(s)]
    return out

def for_module(key, scenario, backward=False):
    m=SCENARIOS[scenario]['length']; linear=linear_b if backward else linear_f; norm=norm_b if backward else norm_f
    if key=='hc':return hc(m,backward)
    if key=='moe':return moe(m,backward)
    if key=='router':return router(m,backward)
    if key in ['expert','shared']:return expert(m,backward,key=='expert')
    if key=='swiglu':return choose_steps(expert(m,backward,False),lambda s:'SwiGLU' in s['title'])
    if key=='selection':return choose_steps(router(m,backward),lambda s:('选择偏置' in s['title']) or ('离散边界' in s['title']))
    if key=='weights':return choose_steps(router(m,backward),lambda s:('归一化' in s['title']) or ('sqrtsoftplus 梯度' in s['title']))
    if key=='combine':
        return choose_steps(moe(m,backward),lambda s:s['title'].startswith('MoE⁻¹ · 从最终') if backward else s['title'].startswith('MoE · 专家'))
    if key in ['query']:return query(m,backward)
    if key=='qa':return linear('W_QA',m,H,R,pid='qa')
    if key=='qb':return linear('W_QB',m,R,A*D,pid='qb')
    if key=='kv':return linear('W_KV',m,H,D,pid='kv')
    if key in ['head','logits']:return linear('W_head',m,H,C['vocab_size'],pid='head')
    if key in ['qnorm','kvnorm','anorm','fnorm','finalnorm']:
        return norm(key,m,R if key=='qnorm' else D if key=='kvnorm' else H,key)
    if key=='output':return output_projection(m,backward)
    if key=='rope':return rope('Query RoPE',m,A,backward)
    if key in ['sliding','sparse','multi']:return attention_core(scenario,0 if key=='sliding' else 2 if key=='sparse' else 1,backward)
    if key=='encoderattn':return attention(scenario,0,backward)
    if key=='decoderattn':return attention(scenario,20,backward)
    if key=='localstates':
        f=Flow('Local KV','当前层的局部KV，包含投影、RMSNorm和RoPE；所有层各自产生。')
        if backward:return f.extend(rope('Local KV RoPE',m,1,True)).extend(norm('KV RMSNorm',m,D,'kvnorm')).extend(linear('W_KV',m,H,D,pid='kv'))
        return f.extend(linear('W_KV',m,H,D,pid='kv')).extend(norm('KV RMSNorm',m,D,'kvnorm')).extend(rope('Local KV RoPE',m,1))
    if key=='compressor':return compressor(scenario,2,backward)
    if key=='global':
        f=Flow('Global KV 源层20','ratio=1：Compressor → Global RoPE → 缓存。后续层只读取。')
        if backward:
            if scenario=='B':
                f.step('Global KV⁻¹ · 跨20个消费者层汇总','源层20的Global KV被20–39层共同读取。',f'20份已在各层内部归约的dCache {shp(20,m,D)}',f'dCache_source {shp(m,D)}',r'dCache_{source}=\sum_{\ell=20}^{39}dCache_\ell',
                  'B的L=128小于Top-K上限，每个全局条目在每个消费者层都有可见Query；先跨层汇总，再执行一次源层RoPE/Compressor反向。',Cost(add=19*m*D),f'(20−1)×128×512={fmt(19*m*D)}次加法；不是每层重做源投影。')
            else:
                f.step('Global KV⁻¹ · 跨层缓存接口','收集各层选中的位置梯度。','20–39层的稀疏dCache和整数索引','源层缓存梯度',r'dCache_j=\sum_{\ell:j\in I_\ell}dCache_{\ell,j}',
                  'A各层Top-K可能不同，额外归约取决于实际索引交集；需要真实I才能给严格数字。以下源投影VJP仍可单独推导。',status='Unknown')
                f.unknown+=['A跨层索引交集未运行，跨层Scatter的实际归约次数Unknown。']
            return f.extend(rope('Global KV RoPE',m,1,True)).extend(compressor(scenario,1,True))
        return f.extend(compressor(scenario,1)).extend(rope('Global KV RoPE',m,1))
    if key=='indexer':return indexer(scenario,backward)
    if key in ['engram','engram1','engram14']:return engram(m,backward)
    if key=='engembed':return engram_lookup(m,backward)
    if key=='enggate':return engram_gate(m,backward)
    if key=='injection':return injection(m,backward)
    if key in ['ngram','hash']:
        if backward:return boundary(key,m,True)
        return choose_steps(engram_lookup(m),lambda s:s['title'].startswith('N-gram' if key=='ngram' else 'Hash'))
    if key.startswith('layer') and key[5:].isdigit():return layer_flow(scenario,int(key[5:]),backward)
    if key in ['encoderblock','decoderblock']:return layer_flow(scenario,0 if key=='encoderblock' else 20,backward)
    if key in ['encoder','decoder','overview']:
        # Aggregate summaries link to shared detailed steps; they don't duplicate or
        # count their children a second time in another module's totals.
        f=Flow(key,'组合面板是所属子图成本清单，不能再与其子模块相加；全网反向尚有共享缓存跨层累加与训练策略边界。')
        ids=range(0,20) if key=='encoder' else range(20,40) if key=='decoder' else range(40)
        for layer in (reversed(list(ids)) if backward else ids):
            child=layer_flow(scenario,layer,backward).export(); t=child['totals']['main']
            f.step(f'Layer {layer} · 子图小计','按该层真实 source 配置汇总独立子图。',f'R {shp(m,4,H)}',f'{"dR" if backward else "R_next"} {shp(m,4,H)}',
              r'F_{layer}=F_{HC,attn}+F_{AttnNorm}+F_{Attention}+F_{HC,ffn}+F_{FFNNorm}+F_{MoE}+F_{Engram\ (if\ present)}',
              '点击对应层或 HC / Attention / MoE / Engram 查看逐步明细。此行仅汇总其子步骤，没有重复包含外部子模块。',Cost(t['mul'],t['add'],t['sub'],t['special'],t['approximate']-t['ordinary']),
              f'普通乘 {fmt(t["mul"])} + 加 {fmt(t["add"])} + 减 {fmt(t["sub"])} = {fmt(t["ordinary"])}；矩阵近似口径 {fmt(t["approximate"])}；特殊函数见小计。')
        if key=='overview':
            if backward:
                f.extend(linear_b('W_head',m,H,C['vocab_size'],pid='head')).extend(norm_b('最终 RMSNorm',m,H,'finalnorm'))
                f.step('最终合并 VJP 接口','末端四路合并与最后 pre 来自不同路径。',f'dZ {shp(m,H)}，R [M,4,H]，p [M,4]',f'dR [M,4,H]，dp [M,4]',r'dR=p\,dZ,\quad dp_i=\sum_d dZ_dR_{i,d}',
                  'dp必须送回最后一层FFN的HC生成器。',Cost(8*m*H,4*m*(H-1)),f'8MH乘+4M(H−1)加。')
            else:
                f.step('最终四路合并','用最后FFN输出的pre形成单路状态。',f'R {shp(m,4,H)}，p {shp(m,4)}',f'Z {shp(m,H)}',r'Z=\sum_i p_iR_i',
                  '这次末端合并在40层之外，只计算一次。',Cost(4*m*H,3*m*H),f'4MH乘+3MH加。')
                f.extend(norm_f('最终 RMSNorm',m,H,'finalnorm')).extend(linear_f('W_head',m,H,C['vocab_size'],pid='head'))
        if backward:
            f.unknown+=['此处小计只汇总局部VJP，不是完整网络训练反向总量。需要跨层共享缓存的梯度归约、所有pre梯度的依赖调度和Indexer训练目标。不能用Decode×3替代；已知局部推导继续在子模块展示。']
        return f
    return boundary(key,m,backward)

def apply(DATA, AUDIT):
    """Attach a single generated ledger; modes reference flows rather than copies."""
    pack={'scenarios':SCENARIOS,'policy':POLICY,'steps':{},'flows':{},'modules':{},'revision':source.REV}
    comparisons=[]
    historical_summary=deepcopy(AUDIT['summary'])
    for key,d in DATA.items():
        mapping={}
        for scenario in SCENARIOS:
            mapping[scenario]={}
            for mode in ['forward','backward']:
                flow=for_module(key,scenario,mode=='backward').export()
                ids=[]
                for step in flow.pop('steps'):
                    # Cumulation is flow-specific; do not store it inside an interned step.
                    cumulative=step.pop('cumulative')
                    for field in ['inputs','outputs']:
                        step[field]=[re.sub(r'\bH\b',str(H),re.sub(r'\bM\b',str(SCENARIOS[scenario]['length']),x)) for x in step[field]]
                    step['dimensions']=[r'\text{'+' ; '.join(step['inputs']+step['outputs']).replace('_',' ')+'}']
                    step['symbols']=[('输入','；'.join(step['inputs'])),('输出','；'.join(step['outputs']))]
                    step['parameterRefs']=[p['bindingId'] for p in d['learnable'] if p['parameterId'] in step['parameterIds']]
                    step['tensorKind']='gradient' if mode=='backward' else 'activation'
                    signature=hashlib.sha256(json.dumps(step,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
                    pack['steps'][signature]=step
                    ids.append(dict(ref=signature,cumulative=cumulative))
                flow['steps']=ids; flow['scenario']=scenario; flow['mode']=mode
                fid=f'{key}/{scenario}/{mode}';pack['flows'][fid]=flow;mapping[scenario][mode]=fid
        pack['modules'][key]=mapping
        d['derivationModes']=mapping
        d['moduleAssumptions']['Scenario']=[['A',SCENARIOS['A']['description']],['B',SCENARIOS['B']['description']]]
        d['moduleAssumptions']['Approximation']=[['普通与特殊操作',POLICY[0]],['矩阵计数',POLICY[2]],['外积',POLICY[3]]]
        d['moduleAssumptions']['Unverified']=[['训练实现','理论实数图可推导不等于官方训练实现已验证；本模块具体Unknown见逐步账本。']]
        for mode,scenario in [('forward','A'),('backward','B')]:
            old=deepcopy(d[mode]); fid=mapping[scenario][mode];flow=pack['flows'][fid];t=flow['totals']['main']
            comparisons.append(dict(module=key,mode=mode,previousRows=old['rows'],previousTotals=old['totals'],scenario=scenario,ordinary=t['ordinary'],approximate=t['approximate'],special=t['special'],unknown=flow['unknown']))
            rows=[('已列实数图 · 普通标量算术',fmt(t['ordinary'])+' FLOPs'),('矩阵2MKN近似 + 其余普通算术',fmt(t['approximate'])+' FLOPs')]
            if flow['unknown']:rows.append(('训练实现 / 完整执行边界','Unknown：见逐步账本'))
            status='Derived'
            if mode=='backward' and key in ['input','tokenizer','ids','ngram','hash','selection']:status='Unknown'
            if key=='prediction':status='Unknown'
            d[mode].update(ledgerRef=fid,steps=[],calculations=[],rows=rows,totals=[],extras=[],
              explain='普通乘加与特殊函数分列，数值仅适用于所选场景和已列图。',
              note='',linearArithmetic=[],
              resultAudits=[dict(label=l,value=v,status='Unknown' if i==2 else 'Derived' if i==0 else 'Estimated') for i,(l,v) in enumerate(rows)],totalAudits=[])
            d[mode]['audit'].update(status=status,basis='固定源码计算图 + 明确的理论实数VJP；特殊函数独立计次；近似与严格乘加分列')
    counts=Counter(d[mode]['audit']['status'] for d in DATA.values() for mode in ['parameter','forward','backward'])
    AUDIT['historicalSummary']=historical_summary
    AUDIT['summary']={s:counts[s] for s in ['Verified','Derived','Estimated','Unknown']}
    AUDIT['summary']['historicalMismatchesCorrected']=historical_summary['historicalMismatchesCorrected']
    AUDIT['assumptions']['Scenario']=[['A',SCENARIOS['A']['description']],['B',SCENARIOS['B']['description']]]
    AUDIT['assumptions']['Approximation']=[['计数',p] for p in POLICY[:4]]
    AUDIT['assumptions']['Unverified']=[['训练实现','完整量化、离散训练、重计算、通信和优化器未由推理实现确定；理论梯度不因此停止推导。']]
    pack['comparisons']=comparisons
    pack['unknown']=['低精度训练/量化梯度与主权重策略','Indexer评分训练目标或STE','MoE选择偏置更新机制','实际路由、哈希、长序列选中位置分布','官方训练Kernel、保存/重算、通信与优化器成本']
    return pack

def write_docs(pack):
    target=ROOT/'docs/architecture/forward_backward_derivations.md'
    lines=['# Forward / Backward 逐步账本','',f'固定源码 revision: {pack["revision"]}。此文与网页使用同一个生成账本 derivation_ledger.py。','']
    for s in SCENARIOS.values():lines+=['## '+s['title'],'',s['description'],'']
    lines+=['## 计数约定','']+['- '+p for p in POLICY]+['']
    lines+=['## 重点模块数值（特殊函数不并入普通FLOPs）','','| 模块 | 场景 | 模式 | 普通标量算术 Derived | 矩阵2MKN口径 Estimated | 特殊函数次数 |','|---|---|---|---:|---:|---|']
    for key in ['hc','moe','query','kv','multi','decoderattn','compressor','global','indexer','engram']:
        for scenario in SCENARIOS:
            for mode in ['forward','backward']:
                flow=pack['flows'][f'{key}/{scenario}/{mode}'];t=flow['totals']['main']
                lines.append(f'| {key} | {scenario} | {mode} | {fmt(t["ordinary"])} | {fmt(t["approximate"])} | {t["special"]} |')
    lines+=['','## 修正与比较','','旧版结果有的是矩阵小计，有的是把特殊函数按1合并的估计；新表拆出特殊函数并计入明确的图。因此不能直接把所有差额叫做算错。原数据在Git历史，比较记录在src/deepseek/derivation_data.json的comparisons。','',
      '- HC：983,040只是W_HC的2MKN近似；现在包含r、后置24维缩放、scale/base、Sigmoid、行Softmax、39次Sinkhorn、pre合并、post/comb回混。原较完整估计1,246,042把特殊函数统一作1；新模型另外拆出普通加减乘，禁止二者混称精确。',
      '- M=1线性dW：KN次乘法、0次归约加法。Shared KV局部VJP严格dX为5120×(2×512−1)=5,237,760；dW为5120×512=2,621,440，矩阵合计7,859,200。旧10,485,760是两项2MKN近似，不能当严格结果。',
      '- MoE：路由权重在2304维中间激活、Down之前相乘；dw也沿2304维点积，不按5120维输出计算。补齐clamp、SiLU、归一化权重、7路结果与8条输入梯度。',
      '- Attention：旧2×矩阵小计不是完整注意力。补齐sink、Softmax、共享K/V两个梯度路径、跨Head/QueryScatter。新表采用有效位置展开数学图；旧在线Softmax分块估计属于另一执行图，不能相减认定内核错误。',
      '- Indexer：评分使用可用缓存长度；Top-K=512是输出上限。主loss可回传Global KV，但整数选择不能给评分器普通连续梯度；给定额外dScore的理论VJP另列conditional，绝不并入主路径。',
      '- Engram：补齐双RMS统计、q/k权重、signed sqrt、门控与共享Value反向；Hash/索引无普通连续导数，表梯度为固定ID下的scatter。','',
      '## 未确定的部分','']+['- '+p for p in pack['unknown']]+['','## 可重复验证','','python src/deepseek/verify_derivations.py','node src/deepseek/verify_derivations.cjs','','细小实数图的数值差分验证不等于运行官方GPU训练内核。']
    for key in ['hc','moe','multi','indexer','compressor','engram']:
        lines+=['','## '+key+' · B理论反向步骤','']
        flow=pack['flows'][f'{key}/B/backward']
        for i,entry in enumerate(flow['steps']):
            s=pack['steps'][entry['ref']]
            lines += [f'### {i+1}. {s["title"]}', '',s['purpose'],'','输入：'+'；'.join(s['inputs']),'','输出：'+'；'.join(s['outputs']),'', '$$'+s['formulas'][0]+'$$','',s['work'],'',s['why'],'']
    target.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
    # Old audit remains source evidence and a historical-policy comparison, not current totals.
    for name in ['calculation_audit.md','assumptions.md']:
        p=target.parent/name
        p.write_text('> 本文为原审计口径与固定源码证据。当前逐步Forward/Backward、A/B场景、Derived/Unknown和特殊操作分账以[新账本](forward_backward_derivations.md)为准；旧“完整反向未推导”说明已由新理论推导补充。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
