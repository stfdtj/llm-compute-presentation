"""Separate main + optional DSpark overview; preserves the existing detailed SVG."""
import build_diagram as d


def build_dspark(c):
    d.S.clear()
    d.raw('<g id="dspark-overview" style="display:none" font-family="Times New Roman, Microsoft YaHei, serif" role="group" aria-label="40-layer main model and optional three-layer DSpark">')
    d.raw('<defs>')
    for name,color in [('main','#2863c5'),('draft','#8b51ce'),('verify','#24875d')]:
        d.raw(f'<marker id="ds-arrow-{name}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="{color}"/></marker>')
    d.raw('</defs>')
    def edge(path,kind='draft',dash=False):
        color={'main':'#2863c5','draft':'#8b51ce','verify':'#24875d'}[kind]
        d.raw(f'<path class="dspark-edge" d="{path}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#ds-arrow-{kind})"'+(' stroke-dasharray="6 4"' if dash or kind=='verify' else '')+'/>')
    def box(stage,x,y,w,h,lines,color='purple',size=19):
        d.raw(f'<g data-dspark-stage="{stage}">')
        d.box('overview',x,y,w,h,lines,color,size)
        d.raw('</g>')
    d.text(836,45,'DeepSeek-V4.1 Flash · 40-Layer Main Model + Optional DSpark',31,True)
    d.rect(25,75,1622,205,'blue',opacity=.13)
    d.text(45,108,'主干计算 · 正常自回归路径始终只有 40 层',23,True,anchor='start')
    main=[(45,180,['Input','Token Embedding']), (265,220,['Encoder · 20 Layers','L0–19']),
          (525,250,['Decoder · 20 Layers','L20–39']), (815,180,['HC Collapse','Final RMSNorm']),
          (1035,220,['LM Head','主模型 logits → sample']), (1295,325,['Output Token · t₀','主模型生成 / 本轮草稿的锚点'])]
    for x,w,lines in main:box('inputs',x,140,w,72,lines,'blue',21)
    for left,right in [(225,265),(485,525),(775,815),(995,1035),(1255,1295)]:edge(f'M{left} 176 H{right}','main')
    box('inputs',525,230,250,34,['A · 取 L37 / 38 / 39 的 Block 输入'],'pink',15)
    edge('M650 212 V230','main',True)
    box('inputs',1295,230,325,34,'B · t₀ → 可选草稿路径','pink',17)
    edge('M1457 212 V230','draft',True)
    d.rect(25,310,1622,575,'purple',True,opacity=.08)
    d.text(45,343,f'可选辅助路径 · DSpark = {c["n_mtp_layers"]} Draft Layers；不属于主干 L0–39',25,True,anchor='start')
    d.text(45,372,'Decode：生成草稿；Prefill：只建立上下文 KV。紫色：草稿生成　绿色：调度 / 验证 / 接受　A / B：同名来源连接',17,anchor='start',maxwidth=1580)
    h=c['dim'];n=len(c['dspark_target_layer_ids']);b=c['dspark_block_size']
    box('inputs',45,395,315,95,['A · h37, h38, h39 → Concat',f'各自 mean(4 路) → [1,{h}]',f'[1,{h*n}] → Projection → RMSNorm',f'main_x [1,{h}]'],'pink',17)
    edge('M360 442 H390 V485 H950')
    d.text(715,477,'main_x → 每个 Draft Attention 的上下文 KV（各自缓存）',17,maxwidth=640)
    for x in [510,730,950]:edge(f'M{x} 485 V520')
    box('positions',45,520,315,85,['B · [t₀, noise, noise, noise, noise]',f'共享 Token Embedding → [1,{b},{h}]',f'扩展 {c["hc_mult"]} 路 → [1,{b},{c["hc_mult"]},{h}]'],'purple',17)
    edge('M360 562 H420')
    for i,x in enumerate([420,640,860]):
        box('blocks',x,520,180,85,[f'Draft Block {i+1}','DSparkAttention','HC + MoE / HC'],'purple',19)
    edge('M600 562 H640');edge('M820 562 H860')
    d.text(730,626,'3 层 = 深度；5 个位置同时通过每一层',19,True,maxwidth=620)
    d.rect(1090,395,530,210,'purple',opacity=.18)
    d.text(1355,422,'Draft Head · Block 3 之后',21,True)
    box('heads',1110,440,235,56,['HC Pre / RMSNorm → LM Head','共享主模型权重 · 5 组 base logits'],'purple',16)
    box('heads',1370,440,230,56,['Markov Head · 前一 token tᵢ',f'Embedding rank={c["dspark_markov_rank"]} → bias'],'purple',17)
    edge('M1040 562 H1060 V468 H1110')
    edge('M1227 496 V530')
    edge('M1370 468 H1357 V514 H1227 V530')
    box('heads',1110,530,235,55,['base logits + Markov bias','依次 sample → c₁…c₅'],'purple',17)
    box('heads',1370,530,230,55,['Confidence Head', 'HC Pre（Norm 前）+ Markov emb', '→ 5 raw scores'],'purple',15)
    edge('M1485 496 V530')
    d.text(1510,518,'emb',13,anchor='start')
    edge('M1227 585 V645 H202 V680')
    edge('M1485 585 V662 H555 V680','verify')
    box('verification',45,680,315,100,['Speculative Tokens · 尚未接受', '[t₀,c₁,c₂,c₃,c₄,c₅] · 6 IDs', '只有 c₁…c₅ 是新候选'],'purple',20)
    box('verification',410,680,290,100,['Confidence → Prefix Scheduler','概率校准 / 前缀生存率 + 负载曲线', '选取前缀长度 0 ≤ m ≤ 5'],'green',17)
    box('verification',750,680,420,100,['Main Model Verification','输入 [t₀,c₁…cₘ] · 因果掩码', '同一 40 层主模型 · 并行算候选分布','按接受规则，从左向右检查前缀'],'green',19)
    box('verification',1220,680,400,100,['Accepted Tokens · 最终提交','接受前缀 + 主模型补充 / 纠正 token','首次拒绝后丢弃剩余草稿'],'green',20)
    for left,right in [(360,410),(700,750),(1170,1220)]:edge(f'M{left} 730 H{right}','verify')
    d.text(836,823,'一次 3 层草稿计算得到 5 组 logits → 轻量顺序采样 → 主模型一次验证多个位置，减少完整主模型的串行轮数',19,True,maxwidth=1550)
    d.text(836,855,'验证 / 调度为 DSpark 论文所述推理流程；forward_spec() 仅实现草稿前向，不含完整验证循环。点击模块查看源码边界。',17,maxwidth=1550)
    d.text(836,919,'蓝色主干可独立生成 token。DSpark 关闭时不执行紫色、绿色辅助路径；原 Attention / HC / MoE 细节保留在“完整架构”。',17,maxwidth=1590)
    d.raw('</g>')
    return ''.join(d.S).replace('class="module"','class="dspark-node"')
