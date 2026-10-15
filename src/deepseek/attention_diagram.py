"""Aligned attention overviews. Logical shapes are read from the pinned config."""
import json
from pathlib import Path
import build_diagram as d

C = json.loads((Path(__file__).resolve().parents[2] / 'references/deepseek/inference/config.json').read_text(encoding='utf-8'))


def draw_overviews():
    h, r, a, k = C['dim'], C['q_lora_rank'], C['n_heads'], C['head_dim']
    groups, rank = C['o_groups'], C['o_lora_rank']
    for decoder, x in [(False, 15), (True, 453)]:
        name = 'decoder' if decoder else 'encoder'
        d.raw(f'<g id="{name}-attention-overview" data-attention-side="{name}">')
        d.frame(name+'attn', x, 585, 423, 333, 'blue',
                ('2.  Decoder' if decoder else '1.  Encoder')+' Attention (Detail)', 18)
        d.text(x+211.5, 632,
               '共享 Encoder 全局表示，避免各 Decoder 层重建历史 Global KV' if decoder else
               '当前 token: Embedding → L0；后续层: 上层 → HC Pre / Norm', 10.5, maxwidth=401)
        d.box('anorm', x+12, 640, 399, 21,
              f'① Input Hidden State · Xℓ [1,{h}] · 本层 Attention 输入', 'blue', 11)
        cols=[x+12,x+146,x+280]
        width=123
        for col, title, key in zip(cols, ['② Query Projection','③ Local KV Projection','④ Global KV / Reuse'], ['query','kv','global' if decoder else 'sparse']):
            d.label(key,col,663,width,15,title,10.5,True)
            d.arrow(f'M{col-5} 661 V690 H{col}')
        # All three query operations are serial; no bypass around RMSNorm.
        for key,y,label,color in [
            ('qa',680,f'W_Q(L) · {h} → {r}','blue'),
            ('qnorm',704,f'RMSNorm · {r}','green'),
            ('qb',728,f'W_Q(R) · {r} → {a*k}','blue'),
            ('rope',752,f'Q [1,{a},{k}] · RoPE 尾 {C["rope_head_dim"]} 维','blue')]:
            d.box(key,cols[0],y,width,21,label,color,10.3)
        for y in [701,725,749]:
            d.arrow(f'M{cols[0]+width/2} {y} v3')
        for key,y,label,color in [
            ('kv',680,f'W_KV · {h} → {k}','blue'),
            ('kvnorm',704,f'RMSNorm · {k}','green'),
            ('rope',728,f'RoPE 尾 {C["rope_head_dim"]} 维 → FP8','blue'),
            ('localstates',752,f'Local KV Cache · ≤{C["window_size"]} × {k}','purple')]:
            d.box(key,cols[1],y,width,21,label,color,10.2,dash=key=='localstates')
        for y in [701,725,749]:
            d.arrow(f'M{cols[1]+width/2} {y} v3')
        full = [i for i in C['kv_source_layers'] if (i>=20)==decoder]
        lines = (['Full L20 · X₂₀ (来自 E)',f'W_KV(g): {h} → {k} · 无 gate',
                  'Norm → RoPE → FP4 写缓存',
                  '写 KV / Index Key + Top-512', 'L21–39 读 L20 Global KV',
                  'Reindex: 24,28,32,36','其余 Reuse KV + 索引'] if decoder else
                 ['Full L'+','.join(map(str,full))+' · 本层 Xℓ',f'W_KV(g), W_gate: {h} → {k}',
                  '加权池化 → Norm → RoPE → FP4',
                  '写 KV / Index Key + Top-512','其余 L2–19 Reuse KV + 索引',
                  'L0–1: Global 关闭（仅 SWA）','完整两 token 才追加全局条目'])
        d.start('global' if decoder else 'compressor','Global KV generation and cross-layer reuse')
        d.rect(cols[2],680,width,93,'orange')
        d.text(cols[2]+width/2,695,'1 token → 1 KV' if decoder else '2 tokens → 1 KV',14,True,maxwidth=width-8)
        d.text(cols[2]+width/2,707,lines,9.4,leading=10,maxwidth=width-8)
        d.end()
        # Each lane enters its own port, so Q never appears to feed a KV cache.
        for i,col in enumerate(cols):
            d.arrow(f'M{col+width/2} 773 V789',dash=i==2)
        d.box('multi' if decoder else 'sliding',x+12,789,399,29,
              ['⑤ Attention · Q + Local KV + Global KV (Top-512'+(')' if decoder else '，仅 L≥2)'),
               'Local / Global 共同 softmax (+ sink) → O [1,64,512]'], 'blue', 10.7)
        d.arrow(f'M{x+211.5} 818 V823')
        d.box('rope',x+12,823,399,17,'Inverse RoPE · Attention 输出的尾 64 维','blue',10.5)
        d.arrow(f'M{x+211.5} 840 V845')
        d.box('output',x+12,845,399,25,
              [f'⑥ Output Projection · {groups} 组 W_OA: {a*k//groups} → {rank}',
               f'Flatten [{groups*rank}] → W_OB [{groups*rank},{h}] → U [1,{h}]'], 'blue',10.5)
        d.arrow(f'M{x+211.5} 870 V875')
        d.box('streams',x+12,875,399,18,
              '⑦ U → HC → MoE / HC → Hℓ [1,4,5120]'+(' → 下层 / Head' if decoder else '；H₁₉ = E →'), 'blue',10.5)
        d.raw('</g>')
    # Labelled cross-module handoff replaces a long arrow cutting through the diagram.
    d.rect(15,920,861,19,'pink')
    d.text(445,933,'E = Encoder Final H₁₉ [1,4,5120] → L20 HC Pre / RMSNorm → X₂₀ → Decoder Q、Local KV、Global KV',11.3,maxwidth=851)
