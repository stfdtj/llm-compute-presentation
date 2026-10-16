"""Standalone Decoder (layers 20-39) SVG: L19->L20 boundary + layer strip + Block.forward flow.

Source-verified (models/DeepSeek-V4.1-Flash/model.py, config.json):
- compress_ratios[20:40] = [1]*20 (config.json:52): every decoder layer has a "compressed" KV
  path, but ratio=1 means Compressor.forward is a plain Linear+RMSNorm with no pooling
  (model.py:461-462) -- no position reduction happens in the decoder, unlike the encoder's 2:1.
- kv_source_layers intersect [20,39] = [20] ONLY (config.json:27). Layer 20 is the single
  decoder layer that runs Compressor and creates the shared compressed-KV cache that layers
  21-39 reuse (Attention._compress_kv, model.py:739-763). Layers 24/28/32/36 do NOT create a
  new KV cache -- they are "Reindex" layers that only re-run the Indexer with a fresh query
  against the SAME KV layer 20 produced.
- index_source_layers intersect [20,39] = [20,24,28,32,36] (config.json:28): these 5 layers run
  Indexer.forward; the other 14 decoder layers reuse the most recently selected position indices.
- candidate_source_layer = 20 (config.json:37): only layer 20 runs select_candidate_blocks to
  build a 2048x8=16384-position candidate pool; layers 24/28/32/36 (is_candidate_source's
  downstream users) restrict their fresh Top-K search to that candidate pool before re-scoring.
- Every decoder layer, including Reuse layers, still computes its OWN Local KV independently
  via Attention._window_kv (model.py:700-720) -- only the compressed/shared KV and the position
  indices are what get reused.
- The L19->L20 boundary: Transformer.forward keeps looping the SAME Block/Attention class; there
  is no separate cross-attention module. Block.hc_pre(E, pre_mix) + attn_norm produce X20
  [1,5120], which layer 20 uses for its OWN Query, OWN Local KV, OWN new Compressor KV and OWN
  new Indexer -- it is not "encoder output read by a separate decoder cross-attention."
"""
import build_diagram as d

KV_SOURCE = [20]
REINDEX = [24, 28, 32, 36]


def node(key, x, y, w, h, lines, color='purple', size=16, bold=False):
    d.box(key, x, y, w, h, lines, color, size, bold)


def _boundary(y):
    d.text(40, y - 14, '① Layer 19 → Layer 20 边界 · 仍是同一个 Block 循环，没有独立的 Cross-Attention 模块', 18, True, anchor='start')
    d.rect(32, y, 1591, 150, 'blue', opacity=.12)
    for i, c in enumerate(['blue', 'green', 'orange', 'purple']):
        node('streams', 48, y + 16 + i * 30, 150, 24, f'E 第{i+1}路  [1,5120]', c, 12.5)
    d.text(123, y + 142, 'E = Encoder 最终四路状态 [1,4,5120]', 13)
    d.arrow(f'M198 {y+76} H260', width=1.8)
    node('hc', 260, y + 36, 170, 76, ['HC Pre', '用上一层(19)的 pre_mix', '四路合成一行'], 'orange', 14)
    d.arrow(f'M430 {y+74} H480', width=1.8)
    node('anorm', 480, y + 36, 140, 76, ['Attention RMSNorm', '[1,5120]'], 'green', 14)
    d.arrow(f'M620 {y+74} H670', width=1.8)
    d.raw('<g><rect class="surface" x="670" y="36" width="140" height="76" rx="4" fill="#fff" stroke="#9847fd" stroke-width="1.4" transform="translate(0,{})"/>'
          '<text x="740" y="{}" font-size="15" text-anchor="middle" font-weight="bold">X20</text>'
          '<text x="740" y="{}" font-size="13" text-anchor="middle">[1,5120] 单行</text></g>'.format(y, y + 66, y + 86))
    for dy, label, key in [(-58, '本层自己的 Query', 'query'), (-19, '本层自己的 Local KV', 'localstates'),
                            (20, '本层新建压缩 KV →', 'global'), (59, '本层新建候选池+Top-K →', 'indexer')]:
        ty = y + 74 + dy
        d.arrow(f'M810 {y+74} H870 V{ty} H910', width=1.6, dash=True)
        node(key, 910, ty - 14, 260, 28, label, 'purple' if dy > 0 else 'blue', 12.5)
    d.text(40, y + 150 + 18, 'X20 不是“把 Encoder 输出交给一个独立的 Decoder Cross-Attention”；它只是 Layer 20 这一个 Block 的普通输入，Layer 20 用它重新算出自己的 Query、Local KV，并且恰好在这一层新建压缩 KV 与索引。', 14.5, anchor='start')


def _layer_strip(y):
    d.text(40, y - 14, '② 20 层总览 · Layers 20–39（同一个 Block 类；compress_ratio = 1，不做位置压缩）', 18, True, anchor='start')
    x0, w, gap = 40, 72, 4.5
    for i in range(20, 40):
        x = x0 + (i - 20) * (w + gap)
        if i in KV_SOURCE:
            color, note, bold = 'orange', 'KV+Index source', True
        elif i in REINDEX:
            color, note, bold = 'purple', 'Reindex', True
        else:
            color, note, bold = 'green', 'Reuse', False
        d.raw(f'<g data-decoder-layer="{i}">')
        d.box('decoderattn', x, y, w, 44, [f'L{i}', note], color, 11, bold)
        d.raw('</g>')
    d.text(40, y + 64, 'KV source = {20}（唯一新建压缩 KV 的层）；Index source = {20,24,28,32,36}；其余层只读共享 KV / 索引，但仍各自算自己的 Local KV。', 14.5, anchor='start')


def _block_flow(y):
    d.text(40, y - 14, '③ 代表层 Block.forward 执行流程（与 Encoder 完全相同的 HC/Attention/MoE 顺序，只是 Attention 内部来源不同）', 17, True, anchor='start')
    x, w, h, gap = 40, 150, 100, 36
    node('hc', x, y, w, h, ['① HC 系数生成', 'hc_mixes(x, hc_attn_fn)'], 'orange', 13.5)
    x2 = x + w + gap
    d.arrow(f'M{x+w} {y+h/2} H{x2}', width=1.8)
    d.arrow(f'M{x2+w/2} {y-28} V{y}', color='#9847fd', dash=True, width=1.6)
    d.text(x2 + w / 2, y - 34, '来自上一层的 pre_mix', 11.5, fill='#6b3fb8')
    node('hc', x2, y, w, h, ['② HC Pre', '合成一行（同 L19→L20）'], 'orange', 13.5)
    x3 = x2 + w + gap
    d.arrow(f'M{x2+w} {y+h/2} H{x3}', width=1.8)
    node('anorm', x3, y, 110, h, ['③ RMSNorm', '[1,5120]'], 'green', 14)
    x4 = x3 + 110 + gap
    d.arrow(f'M{x3+110} {y+h/2} H{x4}', width=1.8)
    aw = 330
    variants = [
        ('full', 'multi', ['④ Attention（L20 · Full）', '新建压缩 KV + 候选池 + Top-K', 'Local KV + 新压缩 KV 共同参与', '见 Indexer / Compressor Detail']),
        ('reindex', 'multi', ['④ Attention（L24/28/32/36 · Reindex）', '复用 L20 的压缩 KV（不重建）', '用本层新 Query 在候选池内重新评分', '选出新的 Top-K 位置索引']),
        ('reuse', 'multi', ['④ Attention（其余层 · Reuse）', '复用最近一次选出的压缩 KV + 索引', '本层仍独立计算自己的 Local KV', '不运行 Compressor，不运行 Indexer']),
    ]
    for mode, akey, alabel in variants:
        d.raw(f'<g data-decoder-mode="{mode}" class="{"decoder-hidden" if mode!="full" else ""}">')
        node(akey, x4, y, aw, h, alabel, 'blue', 12.5)
        d.raw('</g>')
    x5 = x4 + aw + gap
    d.arrow(f'M{x4+aw} {y+h/2} H{x5}', width=1.8)
    node('hc', x5, y, w, h, ['⑤ HC Post', '写回四路'], 'orange', 13)

    y2 = y + h + 54
    node('hc', x, y2, w, h, ['⑥ HC 系数生成', 'hc_mixes(x, hc_ffn_fn)'], 'orange', 13.5)
    x2b = x + w + gap
    d.arrow(f'M{x+w} {y2+h/2} H{x2b}', width=1.8)
    node('hc', x2b, y2, w, h, ['⑦ HC Pre', '用本层①的 attn_pre'], 'orange', 13.5)
    x3b = x2b + w + gap
    d.arrow(f'M{x2b+w} {y2+h/2} H{x3b}', width=1.8)
    node('fnorm', x3b, y2, 110, h, ['⑧ RMSNorm', '[1,5120]'], 'green', 14)
    x4b = x3b + 110 + gap
    d.arrow(f'M{x3b+110} {y2+h/2} H{x4b}', width=1.8)
    node('moe', x4b, y2, aw, h, ['⑨ MoE', '与 Encoder 结构完全相同', 'Router → SwiGLU → Combine'], 'purple', 14)
    x5b = x4b + aw + gap
    d.arrow(f'M{x4b+aw} {y2+h/2} H{x5b}', width=1.8)
    node('hc', x5b, y2, w, h, ['⑩ HC Post', '返回 (x, ffn_pre)'], 'orange', 13)
    d.arrow(f'M{x5b+w} {y2+h/2} H{x5b+w+34}', color='#9847fd', dash=True, width=1.6)
    d.text(x5b + w + 40, y2 + h/2 - 6, 'ffn_pre', 13, anchor='start', fill='#6b3fb8')
    d.text(x5b + w + 40, y2 + h/2 + 12, '→ 下一层②的 pre_mix', 13, anchor='start', fill='#6b3fb8')
    return y2 + h


def build_detail():
    offset = len(d.S)
    d.raw('<g id="decoder-detail" style="display:none" font-family="Times New Roman, Microsoft YaHei, serif" '
          'role="group" aria-label="Decoder Detail：Layer 19→20 边界、Layer 20-39 结构与代表层流程">')
    d.rect(15, 12, 1642, 1040, 'purple', opacity=.08)
    d.text(40, 47, 'Decoder Detail · Layers 20–39', 30, True, anchor='start')
    d.text(40, 78, '同一个 Block 类继续跑 20 次；Layer 20 是唯一新建共享压缩 KV 的层', 18, anchor='start')
    d.text(1625, 46, 'compress_ratio = 1（L20–39）', 19, True, anchor='end', fill='#6b3fb8')
    d.text(1625, 75, 'candidate_source_layer = 20', 16, anchor='end')

    _boundary(130)
    _layer_strip(330)
    bottom = _block_flow(480)

    y = bottom + 70
    d.rect(32, y, 1591, 220, 'orange', opacity=.1)
    d.text(46, y + 26, '④ 三种层在 Attention 子层的差别（常见误解纠正：不是每个 Index source 层都新建 KV）', 17, True, anchor='start')
    cols = [
        ('L20 · Full（唯一 KV source）', ['新建本层的压缩 KV（ratio=1，无位置压缩）', '新建候选池（最多 2048×8=16384 个位置）', '新建 Index Key 并运行 Indexer 选 Top-K']),
        ('L24/28/32/36 · Reindex', ['不新建 KV：直接复用 L20 写入的压缩 KV', '用本层自己的 Query 在候选池内重新打分', '选出本层自己的新 Top-K 索引']),
        ('其余 15 层 · Reuse', ['不新建 KV，不运行 Indexer', '直接复用最近一次（Full 或 Reindex）的', '压缩 KV 与 Top-K 索引；Local KV 仍独立计算']),
    ]
    cw = 515
    for i, (title, lines) in enumerate(cols):
        cx = 46 + i * (cw + 10)
        d.text(cx, y + 56, title, 15.5, True, anchor='start', fill='#6b3fb8')
        for j, line in enumerate(lines):
            d.text(cx, y + 82 + j * 24, '· ' + line, 14, anchor='start')
    d.text(46, y + 176, '任务草稿曾假设“20/24/28/32/36 都新建 KV”——源码核实后只有 Layer 20 新建；24/28/32/36 只新建 Top-K 索引（Reindex），复用同一份 KV。', 14, anchor='start', fill='#a50027')
    d.text(46, y + 200, '点击任意 Attention 节点可打开 Indexer / Compressor 的完整流程；与 Encoder Detail 对照可见两者是同一套机制，只是起止层不同。', 13.5, anchor='start', fill='#5b6981')
    d.raw('</g>')
    markup = ''.join(d.S[offset:]).replace('class="module"', 'class="decoder-module"')
    del d.S[offset:]
    return markup
