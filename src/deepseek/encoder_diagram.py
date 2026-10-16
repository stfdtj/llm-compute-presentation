"""Standalone Encoder (layers 0-19) SVG: layer-range strip + one Block.forward flow.

Source-verified (models/DeepSeek-V4.1-Flash/model.py, config.json):
- There is no `Encoder` class. `Transformer` loops over one `Block` class 40 times
  (model.py:1203-1204, 1261-1267). "Encoder" here means layers 0-19.
- compress_ratios[0:20] = [0,0,2,2,...,2] (config.json:52): layers 0-1 run pure local
  sliding-window attention (compress_ratio==0, Attention.forward model.py:775 guard false);
  layers 2-19 additionally read/write a 2:1 compressed shared KV.
- kv_source_layers intersect [0,19] = [2,8,14]; index_source_layers intersect [0,19] = [2,8,14]
  (config.json:27-28) -- these three layers run Compressor+Indexer; the other 15 compressed
  layers (3-7,9-13,15-19) reuse the most recent source layer's shared KV/indices.
- engram_layer_ids = [1,14] (config.json:43): Engram runs in Transformer.forward BEFORE
  Block.forward for these two layers only (see engram_diagram.py for the full mechanism).
- Block.forward (model.py:968-994): hc_mixes -> hc_pre(pre_mix from PREVIOUS block) -> attn_norm
  -> Attention -> hc_post -> hc_mixes -> hc_pre(this block's own attn_pre) -> ffn_norm -> MoE
  -> hc_post; returns (x, ffn_pre) and ffn_pre becomes the NEXT block's pre_mix.
"""
import build_diagram as d

KV_SOURCE = [2, 8, 14]
LOCAL_ONLY = [0, 1]
ENGRAM = [1, 14]


def node(key, x, y, w, h, lines, color='blue', size=16, bold=False):
    d.box(key, x, y, w, h, lines, color, size, bold)


def _layer_strip(y):
    d.text(40, y - 14, '① 20 层总览 · Layers 0–19（同一个 Block 类，按配置逐层略有不同）', 18, True, anchor='start')
    x0, w, gap = 40, 72, 4.5
    for i in range(20):
        x = x0 + i * (w + gap)
        if i in LOCAL_ONLY:
            color, note = 'blue', 'Local only'
        elif i in KV_SOURCE:
            color, note = 'orange', 'KV+Index source'
        else:
            color, note = 'green', 'Reuse KV/Index'
        d.raw(f'<g data-encoder-layer="{i}">')
        d.box('encoderattn', x, y, w, 44, [f'L{i}', note], color, 11.5 if i in KV_SOURCE else 12, i in KV_SOURCE)
        d.raw('</g>')
        if i in ENGRAM:
            d.text(x + w / 2, y - 6, 'Engram ↓', 11, True, fill='#a50027')
    d.text(40, y + 64, 'Engram before layer 1 before layer 14（进入该层 Block 之前单独调用一次，见 Engram Detail）', 14, anchor='start', fill='#a50027')
    d.text(40, y + 86, 'compress_ratios[0..19] = [0,0,2,2,…,2]：L0–1 纯局部注意力；L2–19 额外有 2:1 压缩共享 KV，KV/Index source = 2, 8, 14。', 14.5, anchor='start')


def _block_flow(y):
    d.text(40, y - 14, '② 代表层 Block.forward 执行流程（HC 系数生成 → HC Pre → RMSNorm → Attention → HC Post，两次：Attention 一次、MoE 一次）', 17, True, anchor='start')
    x, w, h, gap = 40, 150, 100, 36
    node('hc', x, y, w, h, ['① HC 系数生成', 'hc_mixes(x, hc_attn_fn)', '→ attn_pre, post, comb'], 'orange', 13.5)
    x2 = x + w + gap
    d.arrow(f'M{x+w} {y+h/2} H{x2}', width=1.8)
    d.arrow(f'M{x2+w/2} {y-28} V{y}', color='#9847fd', dash=True, width=1.6)
    d.text(x2 + w / 2, y - 34, '来自上一层的 pre_mix', 11.5, fill='#6b3fb8')
    node('hc', x2, y, w, h, ['② HC Pre', '用上一个 Block 传来的', 'pre_mix 合成一行'], 'orange', 13.5)
    x3 = x2 + w + gap
    d.arrow(f'M{x2+w} {y+h/2} H{x3}', width=1.8)
    node('anorm', x3, y, 110, h, ['③ RMSNorm', '[1,5120]'], 'green', 14)
    x4 = x3 + 110 + gap
    d.arrow(f'M{x3+110} {y+h/2} H{x4}', width=1.8)
    aw = 330
    variants = [
        ('local', 'sliding', ['④ Attention（L0–1）', '仅 Local Sliding Window', '128 个最近位置', '无 Compressor / Indexer']),
        ('full', 'sparse', ['④ Attention（L2/8/14 · Full）', 'Local KV + 新建 2:1 压缩 KV', '本层运行 Compressor + Indexer', '见 Indexer / Compressor Detail']),
        ('reuse', 'sparse', ['④ Attention（其余 L2–19 · Reuse）', 'Local KV（本层独立）+', '复用最近 Full 层的压缩 KV/索引', '本层不运行 Compressor / Indexer']),
    ]
    for mode, akey, alabel in variants:
        d.raw(f'<g data-encoder-mode="{mode}" class="{"encoder-hidden" if mode!="full" else ""}">')
        node(akey, x4, y, aw, h, alabel, 'blue', 13)
        d.raw('</g>')
    x5 = x4 + aw + gap
    d.arrow(f'M{x4+aw} {y+h/2} H{x5}', width=1.8)
    node('hc', x5, y, w, h, ['⑤ HC Post', '按 post/comb 写回四路', '[1,5120]→[1,4,5120]'], 'orange', 13)

    y2 = y + h + 54
    d.text(40, y2 - 14, '同一层继续：MoE 子层用同样的 HC 系数生成 → HC Pre（这次用本层①产生的 attn_pre）→ RMSNorm → MoE → HC Post', 15, anchor='start', fill='#5b6981')
    node('hc', x, y2, w, h, ['⑥ HC 系数生成', 'hc_mixes(x, hc_ffn_fn)', '→ ffn_pre, post, comb'], 'orange', 13.5)
    x2b = x + w + gap
    d.arrow(f'M{x+w} {y2+h/2} H{x2b}', width=1.8)
    node('hc', x2b, y2, w, h, ['⑦ HC Pre', '用本层①产生的', 'attn_pre 合成一行'], 'orange', 13.5)
    x3b = x2b + w + gap
    d.arrow(f'M{x2b+w} {y2+h/2} H{x3b}', width=1.8)
    node('fnorm', x3b, y2, 110, h, ['⑧ RMSNorm', '[1,5120]'], 'green', 14)
    x4b = x3b + 110 + gap
    d.arrow(f'M{x3b+110} {y2+h/2} H{x4b}', width=1.8)
    node('moe', x4b, y2, aw, h, ['⑨ MoE', '6 路由专家 + 1 共享专家', 'Router → SwiGLU → Combine'], 'purple', 14)
    x5b = x4b + aw + gap
    d.arrow(f'M{x4b+aw} {y2+h/2} H{x5b}', width=1.8)
    node('hc', x5b, y2, w, h, ['⑩ HC Post', '返回 (x, ffn_pre)', 'ffn_pre → 下一层 pre_mix'], 'orange', 13)
    d.arrow(f'M{x5b+w} {y2+h/2} H{x5b+w+34}', color='#9847fd', dash=True, width=1.6)
    d.text(x5b + w + 40, y2 + h/2 - 6, 'ffn_pre', 13, anchor='start', fill='#6b3fb8')
    d.text(x5b + w + 40, y2 + h/2 + 12, '→ 下一层②的 pre_mix', 13, anchor='start', fill='#6b3fb8')
    return y2 + h


def build_detail():
    offset = len(d.S)
    d.raw('<g id="encoder-detail" style="display:none" font-family="Times New Roman, Microsoft YaHei, serif" '
          'role="group" aria-label="Encoder Detail：Layer 0-19 结构与代表层流程">')
    d.rect(15, 12, 1642, 820, 'green', opacity=.08)
    d.text(40, 47, 'Encoder Detail · Layers 0–19', 30, True, anchor='start')
    d.text(40, 78, '同一个 Block 类跑 20 次；下面用一个代表层展开它内部真正的执行顺序', 18, anchor='start')
    d.text(1625, 46, 'compress_ratio = 2（L2–19）', 19, True, anchor='end', fill='#1c8a4c')
    d.text(1625, 75, 'L0–1 的 compress_ratio = 0', 16, anchor='end')

    _layer_strip(130)
    bottom = _block_flow(280)

    y = bottom + 70
    d.rect(32, y, 1591, 190, 'blue', opacity=.1)
    d.text(46, y + 26, '③ 三种层在 Attention 子层的差别（HC / MoE 子层对三者完全一样，这里只比较 Attention）', 17, True, anchor='start')
    cols = [
        ('L0–1 · Local only', ['只有 128 个最近位置的滑动窗口', '没有 Compressor，没有 Indexer', '不参与跨层共享 KV']),
        ('L2/8/14 · Full（KV+Index source）', ['新建本组的 2:1 压缩 KV（Compressor）', '新建 Index Key 并运行 Indexer 选 Top-K', '供本组后续 5 层复用']),
        ('其余 L2–19 · Reuse', ['仍计算自己独立的 Local KV', '直接复用最近 Full 层的压缩 KV', '直接复用最近 Full 层选出的位置索引']),
    ]
    cw = 515
    for i, (title, lines) in enumerate(cols):
        cx = 46 + i * (cw + 10)
        d.text(cx, y + 56, title, 15.5, True, anchor='start', fill='#1c8a4c' if i else '#1c8a4c')
        for j, line in enumerate(lines):
            d.text(cx, y + 82 + j * 24, '· ' + line, 14, anchor='start')
    d.text(46, y + 176, '每一组压缩层固定是 6 层（1 个 Full + 5 个 Reuse）：2–7、8–13、14–19。点击任意 Attention 节点可打开 Indexer / Compressor 的完整流程。', 13.5, anchor='start', fill='#5b6981')
    d.raw('</g>')
    markup = ''.join(d.S[offset:]).replace('class="module"', 'class="encoder-module"')
    del d.S[offset:]
    return markup
