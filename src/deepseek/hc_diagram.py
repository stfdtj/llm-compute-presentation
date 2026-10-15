"""HC-only SVG: four streams, delayed pre, and source-faithful post/comb."""
from html import escape
import build_diagram as d

STREAMS=[('h⁽¹⁾','blue'),('h⁽²⁾','green'),('h⁽³⁾','orange'),('h⁽⁴⁾','purple')]

def node(stage,x,y,w,h,lines,color='blue',size=18,expand=False):
    label=' / '.join(lines)
    d.raw(f'<g class="hc-module" data-module="hc" data-hc-stage="{stage}"'+
          (' data-hc-expand="true" aria-controls="hc-detail" aria-expanded="false"' if expand else '')+
          f' role="button" tabindex="0" aria-label="{escape(label)}"><title>{escape(label)} · 点击查看HC计算步骤</title>')
    d.rect(x,y,w,h,color)
    leading=size*1.13
    d.text(x+w/2,y+h/2-(len(lines)-1)*leading/2+size*.34,lines,size,leading=leading,maxwidth=w-12)
    d.raw('</g>')

def draw_overview():
    """Draw into the original HC rectangle; all other SVG siblings stay intact."""
    d.raw('<g id="hc-overview">')
    d.start('hc','Hyper-Connection HC')
    d.rect(893,648,226,264,'orange',opacity=.15)
    d.text(1006,668,'Hyper-Connection (HC)',17,True)
    d.text(1006,686,'1 token · 4 residual streams',12)
    d.text(1006,702,'4 + 4 + 16 = 24 dynamic coefficients',10.8)
    d.end()
    for i,(label,color) in enumerate(STREAMS):
        y=711+36*i
        d.box('streams',902,y,65,30,[label,'[1,5120]'],color,10.5)
        d.line(f'M967 {y+15} H974',width=1)
    d.line('M974 726 V849',width=1)
    node('pre',982,711,124,30,['Pre = 4 · p_in','4 → 1 · [1,5120]'],size=10.6)
    node('submodule',982,752,124,30,['RMSNorm → Attn / FFN','U [1,5120]'],size=10)
    node('post',982,793,124,30,['Post = 4 · current','将 U 分配到四路'],'green',10.6)
    node('comb',982,834,124,30,['Comb = 4×4 = 16','Comb(R) + Post(U)'],'purple',10.1)
    d.arrow('M974 726 H982')
    d.arrow('M974 849 H982')
    for first,last in [(741,752),(782,793),(823,834)]:d.arrow(f'M1044 {first} V{last}')
    d.arrow('M1044 864 V872')
    d.text(1006,886,'Updated R′ [1,4,5120]',11)
    node('all',902,894,204,15,['展开 HC Pre / Post / Comb ↗'],'blue',9.8,True)
    d.raw('</g>')

def build_detail():
    """Standalone hidden detail, rendered using the existing palette/arrow helpers."""
    offset=len(d.S)
    d.raw('<g id="hc-detail" style="display:none" font-family="Times New Roman, Microsoft YaHei, serif" role="group" aria-label="Hyper-Connection：Pre、Post、Comb与系数生成">')
    d.rect(15,12,1642,916,'blue',opacity=.08)
    d.text(40,47,'Hyper-Connection · Pre / Post / Comb',31,True,anchor='start')
    d.text(40,78,'同一个 Token 的四路 Hidden State · 上标是残差流编号，不是 Token / Layer 编号',19,anchor='start')
    d.text(1625,46,'Pre = 4   |   Post = 4   |   Comb = 4×4 = 16',21,True,anchor='end')
    d.text(1625,77,'4 + 4 + 16 = 24 个运行时动态系数',19,anchor='end')

    # Coefficient generation has its own upper lane, branched from the SAME R.
    d.rect(32,103,1591,177,'blue',opacity=.15)
    d.text(46,126,'系数生成 · 当前阶段 hc_mixes(R)',17,True,anchor='start')
    node('flatten',40,150,180,85,['同一份四路 R','[1,4,5120]'],'orange',18)
    node('flatten',260,150,200,85,['Flatten','R_flat [1,20480]'],size=18)
    node('projection',500,150,230,85,['W_HC [20480,24]','T = R_flat W_HC','T [1,24]'],size=17)
    node('transform',770,150,280,100,['投影结果 × r','Scale / Base 调整','Pre = σ(a_pre) + ε','Post = 2σ(a_post)','Comb：Softmax → Sinkhorn(20)'],'green',16)
    for first,last in [(220,260),(460,500),(730,770)]:d.arrow(f'M{first} 193 H{last}',width=1.8)
    d.arrow('M1050 193 H1075 V138 H1205 V150',dash=True,width=1.6)
    d.arrow('M1075 138 H1400 V150',dash=True,width=1.6)
    d.arrow('M1400 138 H1545 V150',dash=True,width=1.6)
    node('prenext',1100,150,210,85,['Pre_next = 4','本次生成','→ 下一个子模块'],'blue',18)
    node('post',1340,150,120,85,['Post = 4','当前使用'],'green',17)
    node('comb',1490,150,110,85,['Comb','4×4 = 16','当前使用'],'purple',16)
    d.text(46,268,'r = rsqrt(mean(R_flat²) + ε)；先投影，再对 24 维结果乘 r。W_HC 使用 XW 行向量约定。',16,anchor='start')

    # Incoming pre is a distinct input, with no edge from this stage's Pre_next.
    node('pre',350,300,210,70,['Pre_in = 4','由前一个阶段传入'],'blue',18)
    d.arrow('M455 370 V410',dash=True,width=1.8)
    d.arrow('M1400 235 V330 H1150 V410',dash=True,width=1.8)
    d.arrow('M1545 235 V295 H1640 V760 H1150 V710',dash=True,width=1.8)

    # One token carries four learned residual paths, not four tokens.
    d.rect(40,365,220,350,'blue',opacity=.13)
    d.text(150,390,'Residual Streams · R',19,True)
    d.text(150,411,'[1,4,5120]',18)
    for i,(label,color) in enumerate(STREAMS):
        node('streams',55,425+i*65,190,50,[label+'  [1,5120]'],color,20)
    d.arrow('M40 555 H21 V193 H40',dash=True,width=1.6)
    d.arrow('M260 465 H350',width=1.8)
    node('pre',350,410,210,110,['Pre · 输入混合','Z = Σᵢ p_in,i Rᵢ','Z [1,5120]'],size=20)
    d.arrow('M560 465 H640',width=1.8)
    node('submodule',640,410,230,110,['RMSNorm →','Attention / FFN','U [1,5120]'],size=20)
    d.arrow('M870 465 H1040',width=1.8)
    node('post',1040,410,220,110,['Post · 输出分配','bⱼ U → 4 路','[1,4,5120]'],'green',20)
    d.arrow('M1260 465 H1318',width=1.8)
    d.raw('<g class="hc-module" data-module="hc" data-hc-stage="update" role="button" tabindex="0" aria-label="相加得到更新后的四路残差"><title>Post(U) + Comb(R)</title><circle class="surface" cx="1340" cy="465" r="22" fill="#fff" stroke="#42a9ff" stroke-width="1.5"/>')
    d.text(1340,474,'+',29);d.raw('</g>')
    d.arrow('M1362 465 H1430',width=1.8)
    d.arrow('M260 675 H920 V650 H1040',width=1.8)
    d.text(637,660,'旧残差 R 直接参与 Comb',18)
    node('comb',1040,600,220,110,['Comb · 残差混合','C [4,4]：旧 i → 新 j','Σᵢ C[i,j] Rᵢ','[1,4,5120]'],'purple',18)
    d.arrow('M1260 650 H1340 V487',width=1.8)

    d.rect(1430,365,185,350,'blue',opacity=.13)
    d.text(1522.5,390,'Updated R′',20,True)
    d.text(1522.5,411,'[1,4,5120]',18)
    for i,(label,color) in enumerate(STREAMS):
        node('update',1444,425+i*65,157,50,[label.replace('h','h′')+'  [1,5120]'],color,17)
    d.text(440,575,'Pre_in ≠ 本次 Pre_next',19,True)
    d.text(440,602,'当前输入用前阶段的 Pre；',17)
    d.text(440,626,'本阶段的新 Pre 只交给下一子模块。',17)
    d.text(40,805,'更新公式：Yⱼ = Σᵢ C[i,j] Rᵢ + bⱼ U      （C 的行 = 旧输入路 i；列 = 新输出路 j）',23,True,anchor='start')
    d.text(40,847,'为什么保留四路？同一 Token 通过多条残差路径传递信息；Pre 临时合成一路，Post / Comb 再更新四路。',18,anchor='start')
    d.text(40,878,'24 是运行时系数的数量；可学习参数是 W_HC、base[24]、scale[3]，不会再额外增加 24 个学习参数。',18,anchor='start')
    d.text(40,909,'虚线：系数 / 生成分支；实线：Hidden State 数据。所有 HC 节点关联已有计算步骤，不新增另一套 FLOPs。',17,anchor='start')
    d.raw('</g>')
    markup=''.join(d.S[offset:])
    del d.S[offset:]
    return markup
