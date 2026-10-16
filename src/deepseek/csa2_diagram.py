"""Standalone CSA2 (Compressed Sparse Attention 2) single-layer data-flow SVG.

This is the "zoomed into one layer's Attention" companion to backbone43_diagram.py
(which shows where in the 43-layer flow CSA2 sits) and indexer_diagram.py (which
already covers the hierarchical Indexer in depth -- this diagram links out to it
rather than redrawing it).

Source-verified against models/DeepSeek-V4.1-Flash/model.py (Attention.forward,
_window_kv, _compress_kv, _compress_topk_idxs, Compressor.forward, Indexer.forward)
and kernel.py (sparse_attn_kernel):
- Three independent input branches per layer: (A) this layer's own low-rank Query,
  (B) this layer's own Local KV (always recomputed, window_size=128), (C) the
  Compressed/Shared Global KV -- either freshly produced by THIS layer (if it is a
  kv_source_layer) or read from the most recent kv_source_layer.
- The Indexer only ever produces integer position indices (Top-k); it never produces
  the attention output itself -- Main Attention gathers the KV rows at those indices
  and runs one shared-normalization sparse attention with the Local KV.
- Full / Reindex / Reuse are properties of the ATTENTION branch only; every layer,
  in every mode, still computes its own Query and its own Local KV.
"""
import build_diagram as d


def node(key, x, y, w, h, lines, color='blue', size=15, bold=False):
    d.box(key, x, y, w, h, lines, color, size, bold)


def build_detail():
    offset = len(d.S)
    d.raw('<g id="csa2-detail" style="display:none" font-family="Arial, Microsoft YaHei, sans-serif" '
          'role="group" aria-label="CSA2: Compressed Sparse Attention 2">')
    d.rect(15, 12, 1642, 1030, 'indexkv', opacity=.08)
    d.text(40, 46, 'CSA2 · Compressed Sparse Attention 2', 30, True, anchor='start')
    d.text(40, 76, '一个层的 Attention 如何同时读取"最近的"和"很久以前但被选中的"历史', 17, anchor='start')
    d.text(1625, 46, '三条独立输入分支 → 一次共同归一化的 Attention', 17, True, anchor='end', fill='#7a1f4e')
    d.text(1625, 73, 'Indexer 只挑选位置，不产生最终输出', 15, anchor='end')

    # ---- Section 0: intuitive analogy ------------------------------------
    cards = [
        ('localkv', 'Local KV', '最近 128 个位置，随时都在，每层自己算一份。'),
        ('sharedkv', 'Compressed KV', '更早之前历史的压缩摘要，少数几层生成、很多层共用。'),
        ('indexkv', 'Indexer', '只负责"选地址"：从压缩摘要里挑出当前最值得读的位置编号。'),
        ('localkv', 'Attention', '真正去读 Local KV + 选中的 Compressed KV，算出新的表示。'),
    ]
    cw = (1591 - 3 * 20) / 4
    for i, (color, title, desc) in enumerate(cards):
        cx = 32 + i * (cw + 20)
        d.rect(cx, 96, cw, 92, color, opacity=.16)
        d.text(cx + cw / 2, 118, title, 16.5, True)
        d.text(cx + cw / 2 - cw / 2 + 10, 140, desc, 12.8, anchor='start', leading=16, maxwidth=cw - 20)
    d.text(40, 212, '一句话：CSA2 让每一层既能看"最近"（Local），也能便宜地看"很久以前"（Compressed + Indexer 挑选），不用对全部历史都做一次完整 Attention。', 14.5, anchor='start', fill='#44546b')

    # ---- Section 1: three branches -----------------------------------
    by = 250
    d.text(40, by - 14, '单层数据流 · 三条输入分支汇入一次 Sparse Attention', 19, True, anchor='start')
    bw, bh, gap = 480, 72, 30
    xA, xB, xC = 40, 40, 40
    yA, yB, yC = by + 10, by + 10 + (bh + 20) * 5 + 40, by + 10 + (bh + 20) * 10 + 80
    # Branch A: Query
    d.text(xA, yA - 10, 'Branch A · Query', 15.5, True, anchor='start', fill='#245aca')
    stepsA = [('query', ['Hidden State [1,5120]', '→ Low-Rank Query Projection']),
              ('query', ['RMSNorm → [1,1280]']),
              ('query', ['Expand → 64 × 512 heads']),
              ('query', ['RoPE on each Query head']),
              ('query', ['Query heads ready [1,64,512]'])]
    yy = yA
    prevA = None
    for key, lines in stepsA:
        node(key, xA, yy, bw, bh, lines, 'blue', 14)
        if prevA:
            d.arrow(f'M{xA+bw/2} {prevA} V{yy}', width=1.6)
        prevA = yy + bh
        yy += bh + 20

    # Branch B: Local KV
    xB2 = 40 + bw + 60
    d.text(xB2, yA - 10, 'Branch B · Local KV (every layer, always)', 15.5, True, anchor='start', fill='#168863')
    stepsB = [('localstates', ['Hidden State [1,5120]', '→ Local KV Projection']),
              ('localstates', ['RMSNorm']),
              ('localstates', ['RoPE on Local KV']),
              ('localstates', ['Write into ring buffer, window = 128']),
              ('localstates', ['Local KV ready ≤128 positions'])]
    yy = yA
    prevB = None
    for key, lines in stepsB:
        node(key, xB2, yy, bw, bh, lines, 'green', 14)
        if prevB:
            d.arrow(f'M{xB2+bw/2} {prevB} V{yy}', width=1.6)
        prevB = yy + bh
        yy += bh + 20

    # Branch C: Compressed Global KV + Indexer
    xC2 = 40 + (bw + 60) * 2
    d.text(xC2, yA - 10, 'Branch C · Compressed Global KV + Indexer', 15.5, True, anchor='start', fill='#5C67C9')
    d.raw('<g data-csa2-mode="full" class="">')
    node('compressor', xC2, yA, bw, bh, ['THIS layer is a KV-source:', 'compress Hidden State → new shared latent'], 'purple', 13.5)
    d.raw('</g>')
    d.raw('<g data-csa2-mode="reindex" class="csa2-hidden">')
    node('global', xC2, yA, bw, bh, ['THIS layer is Reindex:', 'reuse existing shared latent, no new KV'], 'purple', 13.5)
    d.raw('</g>')
    d.raw('<g data-csa2-mode="reuse" class="csa2-hidden">')
    node('global', xC2, yA, bw, bh, ['THIS layer is Reuse:', 'just read the existing shared KV'], 'purple', 13.5)
    d.raw('</g>')
    y2 = yA + bh + 20
    node('global', xC2, y2, bw, bh, ['Shared Compressed KV Cache', '[S_compressed, 512] · cross-layer'], 'purple', 13.5)
    d.arrow(f'M{xC2+bw/2} {yA+bh} V{y2}', width=1.6)
    y3 = y2 + bh + 20
    d.raw('<g data-csa2-mode="full" class="">')
    node('indexer', xC2, y3, bw, bh, ['Full: build Index Keys + run Indexer', 'score → Top-k (≤512) → NEW candidate'], 'pink', 13.5)
    d.raw('</g>')
    d.raw('<g data-csa2-mode="reindex" class="csa2-hidden">')
    node('indexer', xC2, y3, bw, bh, ['Reindex: reuse Index Keys', 'own Query rescoring → NEW Top-k'], 'pink', 13.5)
    d.raw('</g>')
    d.raw('<g data-csa2-mode="reuse" class="csa2-hidden">')
    node('indexer', xC2, y3, bw, bh, ['Reuse: no rescoring', 'just reuse the existing Top-k'], 'pink', 13.5)
    d.raw('</g>')
    d.arrow(f'M{xC2+bw/2} {y2+bh} V{y3}', width=1.6)
    y4 = y3 + bh + 20
    node('indexer', xC2, y4, bw, bh, ['Top-k position indices', '(addresses only, not vectors)'], 'pink', 13.5)
    d.arrow(f'M{xC2+bw/2} {y3+bh} V{y4}', width=1.6)
    y5 = y4 + bh + 20
    node('global', xC2, y5, bw, bh, ['Gather selected rows from', 'Shared Compressed KV Cache'], 'purple', 13.5)
    d.arrow(f'M{xC2+bw/2} {y4+bh} V{y5}', width=1.6)

    # Converge -> Sparse Attention
    conv_y = max(prevA, prevB, y5 + bh) + 50
    d.arrow(f'M{xA+bw/2} {prevA} V{conv_y-40} H{xA+bw/2+300} V{conv_y}', width=1.8)
    d.arrow(f'M{xB2+bw/2} {prevB} V{conv_y}', width=1.8)
    d.arrow(f'M{xC2+bw/2} {y5+bh} V{conv_y}', width=1.8)
    d.text(40, conv_y - 10, 'All three branches meet here — ONE sparse attention, not two separate Attentions added together', 15, True, anchor='start', fill='#7a1f4e')
    node('sparse', xB2, conv_y, bw, 70, ['Concat Local + selected Compressed KV', 'shared softmax normalization over eligible rows',
                                          '+ learned Attention Sink bias'], 'blue', 13.5)
    y6 = conv_y + 70 + 20
    node('output', xB2, y6, bw, 60, ['Inverse RoPE → grouped output projection', '[1,5120]'], 'blue', 14)
    d.arrow(f'M{xB2+bw/2} {conv_y+70} V{y6}', width=1.8)
    d.text(xB2 + bw + 40, y6 + 35, '→ HC Post → MoE (unchanged by CSA2)', 14, anchor='start', fill='#44546b')
    d.text(40, y6 + 90, 'Not two unrelated Attentions added: Local and selected Compressed KV rows are concatenated along the position axis BEFORE one shared-normalization softmax — matching sparse_attn_kernel in kernel.py.', 13.5, anchor='start', fill='#5b6981')
    d.text(40, y6 + 112, 'The compressed latent is also not an ordinary separate K/V pair: one projected latent serves as both Key and Value for the compressed positions (see Compressor Detail / Global KV Projection panel).', 13.5, anchor='start', fill='#5b6981')

    bottom = y6 + 150
    d.raw('</g>')
    markup = ''.join(d.S[offset:]).replace('class="module"', 'class="csa2-module"')
    del d.S[offset:]
    return markup
