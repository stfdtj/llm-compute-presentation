"""Vector reconstruction of the supplied 1672 x 941 reference. Geometry only."""
from html import escape

S=[]
INK='#080d59'
COLORS={'blue':('#e3f3ff','#42a9ff'),'green':('#e2f7e9','#60c787'),'purple':('#efe6ff','#a66aff'),'orange':('#fff3d8','#efba36'),'pink':('#ffe7ed','#ff476f'),'white':('#ffffff','#88bfff')}

def raw(t): S.append(t)
def text(x,y,lines,size=14,bold=False,anchor='middle',fill=INK,leading=None,italic=False,maxwidth=None):
    if isinstance(lines,str): lines=[lines]
    raw(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"'+(' font-weight="bold"' if bold else '')+(' font-style="italic"' if italic else '')+(f' data-max-width="{maxwidth}"' if maxwidth else '')+'>')
    for i,line in enumerate(lines): raw(f'<tspan x="{x}" dy="{0 if i==0 else leading or size*1.08}">{escape(line)}</tspan>')
    raw('</text>')
def rect(x,y,w,h,color='blue',dash=False,r=4,opacity=1):
    bg,stroke=COLORS[color]
    raw(f'<rect class="surface" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{bg}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width=".9"'+(' stroke-dasharray="4 3"' if dash else '')+'/>')
def start(key,label=None):
    raw(f'<g class="module" data-module="{key}" role="button" tabindex="0" aria-label="{escape(label or key)}"><title>{escape(label or key)} · 点击查看计算</title>')
def end(): raw('</g>')
def box(key,x,y,w,h,lines,color='blue',size=14,bold=False,dash=False,leading=None):
    start(key, ' / '.join(lines) if isinstance(lines,list) else lines)
    rect(x,y,w,h,color,dash)
    ls=[lines] if isinstance(lines,str) else lines
    lh=leading or size*1.08
    text(x+w/2,y+h/2-(len(ls)-1)*lh/2+size*.34,ls,size,bold,leading=lh,maxwidth=w-8)
    end()
def label(key,x,y,w,h,lines,size=14,bold=False):
    start(key, ' / '.join(lines) if isinstance(lines,list) else lines)
    raw(f'<rect class="hit-label" x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="transparent"/>')
    ls=[lines] if isinstance(lines,str) else lines
    text(x+w/2,y+size,ls,size,bold,maxwidth=w-4)
    end()
def arrow(d,color=INK,dash=False,width=1.2):
    marker='redarrow' if color=='#ee152e' else 'arrow'
    raw(f'<path class="connection" d="{d}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"'+(' stroke-dasharray="5 3"' if dash else '')+'/>')
def line(d,color=INK,width=1,dash=False):
    raw(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="4 3"' if dash else '')+'/>')
def frame(key,x,y,w,h,color,title,size=21):
    start(key,title);rect(x,y,w,h,color,opacity=.2);end()
    bg,_=COLORS[color]
    raw(f'<path d="M{x+4} {y} H{x+w-4} Q{x+w} {y} {x+w} {y+4} V{y+34} H{x} V{y+4} Q{x} {y} {x+4} {y}" fill="{bg}" opacity=".7"/>')
    start(key,title)
    raw(f'<rect class="hit-label" x="{x+7}" y="{y+4}" width="{w-14}" height="28" fill="transparent"/>')
    text(x+12,y+24,title,size,True,anchor='start',maxwidth=w-24,fill='#a40a31' if color=='pink' else INK)
    end()
def bars(x,y,w=85,h=18,gap=5):
    for i,c in enumerate(['blue','green','orange','purple']):
        rect(x,y+i*(h+gap),w,h,c,r=3)

def build():
    S.clear()
    raw('<svg id="architecture" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1672 941" aria-label="DeepSeek-V4.1 architecture, faithful interactive reconstruction" role="group"><defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L8 4 L0 8 Z" fill="#080d59"/></marker><marker id="redarrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#ee152e"/></marker></defs><g id="diagram-content" font-family="Times New Roman, Noto Serif, serif">')
    label('overview',270,1,1120,43,'DeepSeek-V4.1 Causal Encoder-Decoder Overview',36,True)
    text(836,70,'(40 Transformer Layers split into 20 Causal Encoder layers and 20 Decoder layers)',27)
    # A: top-level architecture, faithful positions and connections.
    start('input','Input Text');rect(14,170,96,110);text(62,195,'Input Text',15,True);text(62,241,'“It is raining”',15);end()
    start('tokenizer','Tokenizer');rect(123,161,90,138);text(168,182,'Tokenizer',16,True)
    for y,t in [(194,'“It”'),(220,'“is”'),(245,'“raining”')]:rect(129,y,77,24,'white');text(167.5,y+17,t,14)
    text(168,286,'…',17);end()
    start('ids','Token IDs');rect(223,161,80,138);text(263,182,'Token IDs',16,True)
    for y,t in [(194,'id₁'),(220,'id₂'),(246,'id₃')]:rect(232,y,63,24,'white');text(263.5,y+17,t,15)
    text(263,286,'…',17);end()
    start('embedding','Token Embedding');rect(313,153,128,150);text(377,173,'Token Embedding',16,True);text(377,192,'(129280 × 5120)',16,maxwidth=114)
    for i,c in enumerate(['blue','green','orange']):rect(333,201+i*23,86,18,c)
    text(377,295,'1 × 5120',16,italic=True);end()
    box('rope',282,313,213,42,['RoPE is applied inside attention','(not added to embeddings)'],'purple',14,dash=True)
    for x1,x2 in [(110,123),(213,223),(303,313),(441,508)]:arrow(f'M{x1} 232 H{x2-2}')
    # Engram boxes are placed directly over their original injection points.
    for x,key,lay in [(551,'engram1',1),(723,'engram14',14)]:
        start(key,f'Engram before layer {lay}');rect(x,88,151,86,'pink',True);text(x+75.5,106,'Engram',16,True,fill='#a50027');text(x+75.5,122,'(N-gram Memory)',15,fill='#57042a');end()
        box(key,x+6,130,139,40,['Lookup & Injection',f'(before layer {lay})'],'pink',15)
    start('encoder','Causal Encoder layers 0–19');rect(508,185,351,92,'green');text(684,208,'Causal Encoder',19,True);text(684,227,'Layers 0 – 19',15);end()
    enc=[(518,0,41),(566,1,41),(614,2,41),(695,13,43),(745,14,41),(807,19,43)]
    for x,i,w in enc:box(f'layer{i}',x,234,w,34,str(i),'purple',15)
    text(676,254,'…',16);text(796,254,'…',16)
    arrow('M587 170 V217','#ee152e',True,1.6);arrow('M774 170 V217','#ee152e',True,1.6)
    line('M511 282 Q511 289 517 289 H852 Q858 289 858 282','#003c45',1.4)
    text(684,308,'20 Encoder Layers',18,True);text(684,327,['KV sources: 2 / 8 / 14 (2:1)','Reuse: 3–7 / 9–13 / 15–19'],14)
    start('global','Layer 20 Compressor');rect(963,184,100,109,'orange');text(1013,203,['Layer 20','Compressor'],15,True);text(1013,235,'mHC Pre → Norm',12)
    rect(980,245,63,11,'blue');rect(980,260,63,11,'orange');text(1013,287,'1 × 512',15,italic=True);end()
    text(1013,313,['X20 → new KV20','1 token → 1 KV','shared through 39'],14)
    start('decoder','Causal Decoder layers 20–39');rect(1071,185,252,92,'purple');text(1197,207,'Causal Decoder',19,True);text(1197,226,'Layers 20 – 39',15);end()
    for x,i,w in [(1080,20,30),(1116,21,30),(1152,22,30),(1208,33,30),(1242,34,30),(1288,39,30)]:box(f'layer{i}',x,234,w,34,str(i),'purple',15)
    text(1195,253,'…',14);text(1280,253,'…',14)
    line('M1073 282 Q1073 287 1078 287 H1317 Q1322 287 1322 282','#9847fd',1.4)
    text(1197,307,'20 Decoder Layers',18,True);text(1197,326,['Reuse KV20 + own Local KV','Index: 20/24/28/32/36','mHC streams continue from layer 19'],13)
    box('finalnorm',1333,184,64,95,['Final','RMSNorm'],'blue',14,True)
    box('head',1407,184,72,95,['LM Head','5120 × 129280'],'blue',14)
    box('logits',1488,168,68,124,['Logits','','1 × 129280'],'purple',16)
    start('prediction','Next Token Prediction');rect(1566,134,94,165);text(1613,151,['Next Token','Prediction'],14,True);text(1613,183,'(Illustrative only)',12)
    for y,word,prob in [(194,'“在”','0.42'),(221,'“it”','0.18'),(248,'“is”','0.06')]:
        rect(1572,y,41,24,'white');rect(1618,y,35,24,'white');text(1592.5,y+17,word,13);text(1635.5,y+17,prob,13)
    end()
    for x1,x2 in [(1323,1333),(1397,1407),(1479,1488),(1556,1566)]:arrow(f'M{x1} 233 H{x2-1}')

    arrow('M859 247 H963')
    arrow('M1063 253 H1071')

    from dspark_overview import draw as draw_dspark
    draw_dspark()

    # B: the exact two per-layer chains.
    rect(13,365,1647,207,'purple',True,opacity=.11)
    text(27,391,'Computation Split (per layer)',26,True,anchor='start')
    frame('encoderblock',19,405,834,160,'green','Causal Encoder Block (× 20 layers, 0 – 19)',22)
    frame('decoderblock',862,405,792,160,'purple','Causal Decoder Block (× 20 layers, 20 – 39)',22)
    # The captions align right and do not collide with the block title.
    text(836,429,'Hidden size: 5120   |   Attention heads: 64 (q)   |   Experts: 384 (top-6)',12,anchor='end')
    text(1640,429,'Hidden size: 5120   |   Attention heads: 64 (q)   |   Experts: 384 (top-6)',12,anchor='end')
    label('streams',25,469,80,68,['Input hidden','state','1 × 4 × 5120'],14)
    box('anorm',110,454,73,77,['RMSNorm','1 × 5120'],'green',14)
    box('encoderattn',193,454,153,77,['Attention','(Local SWA in early layers +','Compressed KV Attention','in later layers)'],'blue',14)
    box('hc',355,454,79,77,['mHC','(4 residual','streams)'],'orange',14)
    box('fnorm',444,454,64,77,['FFN','RMSNorm'],'green',14)
    box('moe',518,454,129,77,['MoE','(6 routed + 1 shared','from 384 experts)'],'purple',14)
    box('hc',657,454,84,77,['mHC','(4 residual','streams)'],'orange',14)
    label('streams',755,469,84,68,['Output hidden','state','1 × 4 × 5120'],14)
    for a,z in [(96,110),(183,193),(346,355),(434,444),(508,518),(647,657),(741,756)]:arrow(f'M{a} 495 H{z-2}')
    for x in [269,394,582,699]:text(x,551,'1 × 4 × 5120' if x in [394,699] else '1 × 5120',15,italic=True,maxwidth=80)
    label('streams',866,469,80,68,['Input hidden','state','1 × 4 × 5120'],14)
    box('anorm',952,454,70,77,['RMSNorm','1 × 5120'],'green',14)
    box('decoderattn',1032,454,145,77,['Attention','(Own Query + own Local KV','+ shared KV20; reindex','only at configured layers)'],'blue',12.8)
    box('hc',1186,454,73,77,['mHC','(4 residual','streams)'],'orange',14)
    box('fnorm',1270,454,61,77,['FFN','RMSNorm'],'green',14)
    box('moe',1340,454,120,77,['MoE','(6 routed + 1 shared','from 384 experts)'],'purple',14)
    box('hc',1470,454,84,77,['mHC','(4 residual','streams)'],'orange',14)
    label('streams',1566,469,80,68,['Output hidden','state','1 × 4 × 5120'],14)
    for a,z in [(938,952),(1022,1032),(1177,1186),(1259,1270),(1331,1340),(1460,1470),(1554,1568)]:arrow(f'M{a} 495 H{z-2}')
    for x in [1104,1222,1400,1512]:text(x,551,'1 × 4 × 5120' if x in [1222,1512] else '1 × 5120',15,italic=True,maxwidth=80)
    for d in ['M160 585 Q300 568 484 584','M576 585 Q700 568 867 584','M920 585 Q1140 568 1330 584','M1420 585 Q1530 568 1635 584']:line(d,'#b177ff',.8,True)

    # Preserve the original four-block layout; enlarge details in the same SVG on focus.
    from attention_blocks import compact
    compact()

    # C3: the original four-stream mHC and MoE/routing detail.
    frame('moe',885,585,493,333,'purple','3.  mHC + MoE (Detail)',21)
    text(1131,635,'Same FL and MoE structure used in both encoder and decoder blocks.',14)
    start('hc','mHC · Manifold-Constrained Hyper-Connections');rect(893,648,226,264,'orange',opacity=.15);text(1006,668,'Single-Pass mHC',17,True);text(1006,689,'Four parallel residual streams',14);end()
    for y,t,c in [(728,'h⁽¹⁾','blue'),(766,'h⁽²⁾','green'),(805,'h⁽³⁾','orange'),(843,'h⁽⁴⁾','purple')]:
        box('streams',902,y,34,26,t,c,14)
        text(961,y-2,'1 × 5120',12)
    arrow('M939 739 H981 L992 762');arrow('M939 778 H991');arrow('M939 816 H981 L992 794');arrow('M939 854 H981 L992 822')
    label('hc',994,741,70,77,['Learned','mixing','coefficients','(α₁ … α₄)'],12)
    text(1090,729,['Updated','hidden state','1 × 5120'],12)
    start('streams','Updated four residual streams');bars(1066,764,45,16,4);end()
    for y in [773,793,813,833]:arrow(f'M1052 {y} H1064')
    start('moe','Mixture of Experts MoE');rect(1129,648,242,264,'purple',opacity=.12);text(1250,666,'Mixture of Experts (MoE)',16,True);text(1250,682,'(6 routed + 1 shared from 384 experts)',13);end()
    label('moe',1137,691,117,25,'Input (1 × 5120)',14)
    for key,y,h,ls in [('router',720,24,'Router (top experts)'),('selection',752,24,'Top-6 Expert Selection'),('expert',784,24,'Expert FFN (SwiGLU)'),('shared',816,24,'Shared Expert'),('combine',848,23,'Combine Expert Outputs')]:box(key,1136,y,121,h,ls,'purple',12)
    for y,z in [(710,720),(744,752),(776,784),(808,816),(840,848)]:arrow(f'M1197 {y} V{z}')
    text(1197,897,'1 × 5120',15,italic=True)
    rect(1264,689,103,194,'blue',True,opacity=.12);text(1315,705,'Fetch Token:',14)
    for key,y,h,ls,c in [('router',710,24,'Router Logits','pink'),('selection',742,24,'Top-k indices','blue'),('weights',774,24,'Top-k weights','blue'),('weights',806,24,'Normalize Weights','blue'),('combine',838,36,['Weighted Sum','(to routed+shared)'],'blue')]:box(key,1273,y,86,h,ls,c,11.5)
    for y,z in [(734,742),(766,774),(798,806),(830,838)]:arrow(f'M1316 {y} V{z}')

    # D: Engram, no added modules.
    frame('engram',1388,585,271,333,'pink','4.  Engram Memory (Detail)',20)
    text(1523,636,'Applied before layers 1 and 14 (zero-based).',14)
    box('ids',1395,656,66,47,['Token IDs','1 × 1'],'blue',14)
    box('ngram',1480,656,169,47,['N-gram Construction','(2-gram, 3-gram, 4-gram, …)'],'pink',14)
    arrow('M1461 680 H1480');arrow('M1564 703 V720')
    box('hash',1480,720,169,42,['Hashing & Lookup','(N-gram Memory)'],'pink',14)
    arrow('M1564 762 V776')
    box('engembed',1480,776,169,29,'Engram Embedding','pink',14)
    arrow('M1564 805 V815')
    box('enggate',1480,815,169,31,'Projection & Gating','pink',14)
    arrow('M1564 846 V867')
    label('injection',1479,870,174,43,['Inject into 4 residual streams','1 × 4 × 5120'],14)
    from attention_blocks import draw_expanded
    draw_expanded()
    raw('</g></svg>')
    return ''.join(S)

if __name__=='__main__':
    from pathlib import Path
    Path(__file__).with_name('architecture.svg').write_text(build(),encoding='utf-8')
