"""Standalone Engram SVG: call site, N-gram -> hash -> lookup -> gate -> injection pipeline.

Source-verified against models/DeepSeek-V4.1-Flash/model.py and engram.py:
- Transformer.forward calls `layer.engram(h, hashes, mask)` directly, BEFORE `layer(h, ...)`
  (model.py:1261-1267) -- Engram is not inside Attention and not inside Block.forward.
- engram_layer_ids = [1, 14] (config.json:43).
- NgramHashState.forward builds (up to) a 2/3/4-gram per token, per Engram layer, per head,
  then combines each gram's token ids with fixed odd multipliers via incremental XOR and
  reduces mod a per-(layer, n-gram-size, head) prime (engram.py:64-184). The exact multiplier
  and prime values are derived at load time from a seeded RNG -- not literal source constants
  we can print -- so the worked example below shows the *structure* of the hash, not invented
  numbers.
- Engram.forward (model.py:350-365): kv = wkv(embed(hash_ids).flatten(-2)) produces 4 Keys
  (one per HC stream) and 1 shared Value; gate = sigmoid(signed-sqrt(dot)); the SAME value is
  added into all 4 Hyper-Connections streams, each scaled by its own stream's gate.
"""
import build_diagram as d

PINK = '#a50027'
MULT_NOTE = '乘数与素数由固定随机种子在加载时生成，不是可直接抄写的源码常量；下方只演示结构。'


def node(key, x, y, w, h, lines, color='pink', size=18, bold=False):
    d.box(key, x, y, w, h, lines, color, size, bold)


def build_detail():
    offset = len(d.S)
    d.raw('<g id="engram-detail" style="display:none" font-family="Times New Roman, Microsoft YaHei, serif" '
          'role="group" aria-label="Engram：N-gram 记忆查找与四路注入">')
    d.rect(15, 12, 1642, 1020, 'pink', opacity=.08)
    d.text(40, 47, 'Engram · N-gram 记忆查找与注入', 30, True, anchor='start')
    d.text(40, 78, '当前 token 的编号 + 最近 3 个历史编号 → 查一张固定大小的记忆表 → 注入四路 Hyper-Connections', 18, anchor='start')
    d.text(1625, 46, 'engram_layer_ids = [1, 14]', 20, True, anchor='end', fill=PINK)
    d.text(1625, 75, '只在进入这两层之前各运行一次', 17, anchor='end')

    # ---- Section 1: call site -------------------------------------------------
    d.rect(32, 100, 1591, 150, 'blue', opacity=.14)
    d.text(46, 123, '① 调用位置 · Transformer.forward 的逐层循环（不在 Attention 内部）', 17, True, anchor='start')
    node('ids', 46, 142, 150, 60, ['Embedding 之后', '四路 Hidden State', '[1,4,5120]'], 'blue', 14)
    d.arrow('M196 172 H230', width=1.8)
    d.raw('<g><rect class="surface" x="230" y="142" width="230" height="60" rx="4" fill="#fff" stroke="#88bfff" stroke-width=".9"/>'
          '<text x="345" y="163" font-size="14" text-anchor="middle" font-weight="bold">for i in range(40): layer = self.layers[i]</text>'
          '<text x="345" y="183" font-size="13" text-anchor="middle">第 i 层循环（i = 0…39）</text></g>')
    d.arrow('M460 172 H494', width=1.8)
    d.raw('<g><rect class="surface" x="494" y="142" width="200" height="60" rx="4" fill="#fff6e8" stroke="#efba36" stroke-width=".9"/>'
          '<text x="594" y="163" font-size="14" text-anchor="middle" font-weight="bold">if layer.engram is not None:</text>'
          '<text x="594" y="183" font-size="13" text-anchor="middle">即 i == 1 或 i == 14</text></g>')
    d.arrow('M694 155 H728 V148', width=1.6)
    d.text(728, 138, '是（i=1 或 14）', 13, anchor='start')
    node('engram', 728, 148, 240, 60, ['h = layer.engram(h, hash_ids, mask)', '本页展开的完整流程 ↓'], 'pink', 14, True)
    d.arrow('M694 196 V222 H1010 V196', width=1.6, dash=True)
    d.text(790, 216, '否（其余 38 层跳过，h 不变）', 13, anchor='start')
    d.arrow('M968 172 H1010', width=1.8)
    d.arrow('M1210 172 H1010', width=0, dash=True)
    d.raw('<g><rect class="surface" x="1010" y="142" width="260" height="60" rx="4" fill="#eef4ff" stroke="#42a9ff" stroke-width=".9"/>'
          '<text x="1140" y="163" font-size="14" text-anchor="middle" font-weight="bold">h, pre_mix = layer(h, ...)</text>'
          '<text x="1140" y="183" font-size="13" text-anchor="middle">Block.forward 正常执行（HC/Attn/MoE）</text></g>')
    d.text(1310, 165, '→ 下一层 i+1', 15, anchor='start')
    d.text(46, 232, 'Engram 修改的是即将进入这一层的 h；它是“进场前”的一次性记忆注入，每层最多发生一次，不是 Attention 的一部分，也不是常驻在每一层。', 15, anchor='start')

    # ---- Section 2: main pipeline ----------------------------------------------
    d.text(46, 285, '② 主计算流程 · 当前 token（History 来自持续累积的编号缓存，不是重新跑一遍历史 token 的网络）', 18, True, anchor='start')
    px, py, pw, ph, gap = 46, 310, 258, 140, 46
    node('ids', px, py, pw, ph, ['历史编号缓存', 'u₋₃  u₋₂  u₋₁  u₀(当前)', '持续累积，跨 prefill/decode', '最多向前看 3 个历史 token'], 'blue', 15)
    x2 = px + pw + gap
    d.arrow(f'M{px+pw} {py+ph/2} H{x2}', width=1.8)
    node('ngram', x2, py, pw, ph, ['N-gram 构造', '(u₋₁,u₀) (u₋₂,u₋₁,u₀)', '(u₋₃,u₋₂,u₋₁,u₀)', '2-gram / 3-gram / 4-gram'], 'pink', 15)
    x3 = x2 + pw + gap
    d.arrow(f'M{x2+pw} {py+ph/2} H{x3}', width=1.8)
    node('hash', x3, py, pw, ph, ['哈希 → 查表地址', '3 种长度 × 8 个 head', '= 24 个整数地址 ℐ', 'XOR 乘积 → mod 质数'], 'pink', 15)
    x4 = x3 + pw + gap
    d.arrow(f'M{x3+pw} {py+ph/2} H{x4}', width=1.8)
    node('engembed', x4, py, pw, ph, ['Embedding Lookup', '取 24 行 × 256 维', 'concat → Z [1,6144]', '只读这一个 token 的 24 行'], 'pink', 15)
    x5 = x4 + pw + gap
    d.arrow(f'M{x4+pw} {py+ph/2} H{x5}', width=1.8)
    node('enggate', x5, py, pw, ph, ['投影 Z·W_E → K₁…K₄,V', '[1,6144]×[6144,25600]', 'Gate = σ(signed-√(h·K·w))', '每路各自的匹配门值'], 'pink', 15)
    x6 = x5 + pw + 30
    d.arrow(f'M{x5+pw} {py+ph/2} H{x6}', width=1.8)
    d.text(x6, py + ph/2 - 6, '同一个 V', 13)
    d.text(x6, py + ph/2 + 12, '四路共用', 13)

    # Residual streams feed into gate (matching) and injection (add)
    d.rect(46, py + ph + 30, 1577, 95, 'orange', opacity=.13)
    d.text(60, py + ph + 50, '四路残差 R = [R₁,R₂,R₃,R₄]（HC 的四路 Hidden State，每路 [1,5120]）同时参与：① 和 Key 比较生成门值  ② 按门值接收 Value', 15, anchor='start')
    ry = py + ph + 64
    for i, c in enumerate(['blue', 'green', 'orange', 'purple']):
        d.rect(60 + i * 70, ry, 58, 22, c, r=3)
    d.arrow(f'M{x5+pw/2} {py+ph} V{ry}', dash=True, width=1.6)
    node('injection', x5 - 40, py + ph + 30 + 95 + 18, pw + 80, 80,
         ['注入：Yⱼ = Rⱼ + gⱼ · V', '四路各自乘自己的 gate', '[1,4,5120] → [1,4,5120]', 'layer 1 与 layer 14 各发生一次'], 'green', 14.5)
    d.arrow(f'M{x5+pw/2} {ry+22} V{py+ph+30+95+18}', width=1.8)
    d.arrow(f'M{x6+30} {py+ph/2} V{py+ph+30+95+58} H{x5+pw+40}', dash=True, width=1.6)
    d.text(x6 + 40, py + ph + 30 + 50, 'V 向下参与注入', 13)
    block2_bottom = py + ph + 30 + 95 + 18 + 80

    # ---- Section 3 (left) + Section 4 (right): worked example + why/comparison --
    ey = block2_bottom + 36
    eh = 270
    d.text(40, ey - 10, '③ 单 token 小例子（左） · ④ 为什么是 layer 1/14、与 Attention KV Cache 的区别（右）', 17, True, anchor='start')
    lw = 950
    d.rect(32, ey, lw, eh, 'blue', opacity=.1)
    d.text(46, ey + 24, '历史序列 “… quick brown fox jumps”，当前 token = “jumps”', 15.5, True, anchor='start')
    seq = ['quick', 'brown', 'fox', 'jumps']
    sx = 46
    for i, w in enumerate(seq):
        color = 'orange' if i == 3 else 'white'
        d.box('ids', sx + i * 108, ey + 36, 94, 30, w, color, 14, i == 3)
    d.text(sx + 55, ey + 83, 'u₋₃      u₋₂      u₋₁      u₀(当前)', 12.5)
    grams = [
        ('2-gram', '(jumps, fox)'),
        ('3-gram', '(jumps, fox, brown)'),
        ('4-gram', '(jumps, fox, brown, quick)'),
    ]
    gy = ey + 102
    for i, (label, g) in enumerate(grams):
        d.box('ngram', 46, gy + i * 33, 400, 27, f'{label} = {g}', 'pink', 13.5)
    d.arrow(f'M446 {gy+41} H540', width=1.8)
    d.text(560, gy + 16, '每种长度 × 8 个 head', 13.5, anchor='start')
    d.text(560, gy + 35, '= 24 个查表地址（示意值，非真实哈希输出）', 13.5, anchor='start')
    d.text(560, gy + 54, MULT_NOTE, 12, anchor='start', fill='#6b7790')
    node('engembed', 560, gy + 65, 350, 70,
         ['记忆表 E（每层一张）→ 24 行 × 256 维', 'concat → Z [1,6144] → K/V/Gate', '"jumps" 的具体结果依赖训练权重'],
         'pink', 12.5)
    d.text(46, ey + eh - 16, '演示“2/3/4-gram → 24 个地址 → 查表 → K/V/Gate”的结构；具体命中哪些地址由训练得到的哈希乘数、质数与权重决定。', 12.5, anchor='start', fill='#5b6981')

    rx = 32 + lw + 30
    rw = 1591 - lw - 30
    d.rect(rx, ey, rw, eh / 2 - 8, 'orange', opacity=.12)
    d.text(rx + 14, ey + 22, '为什么只在 layer 1 和 14？为什么先 N-gram 再哈希？', 14.5, True, anchor='start')
    why = [
        'engram_layer_ids=[1,14]：注入只发生两次，成本随层数',
        '常数增加、不随序列长度增加——这正是相对 Attention 的目的。',
        'N-gram 把“最近几个 token”变成离散身份，哈希再映射到',
        '固定大小的表，表大小与词表、历史长度都无关。',
    ]
    for i, line in enumerate(why):
        d.text(rx + 14, ey + 44 + i * 20, line, 12.8, anchor='start')
    ey2 = ey + eh / 2 + 8
    d.rect(rx, ey2, rw, eh / 2 - 8, 'purple', opacity=.12)
    d.text(rx + 14, ey2 + 22, 'Engram 查表 vs Attention 读 KV Cache', 14.5, True, anchor='start')
    cmp_lines = [
        '· Attention 对所有历史位置做 QKᵀ + Softmax，成本随缓存长度增长；',
        '  Engram 只对 ≤3 个历史编号做一次固定哈希查表，读 24 行，与长度无关。',
        '· Attention 的 K/V 是当次前向算出的；Engram 的表训练后就固定，',
        '  查表地址才是运行时算出来的——更像查词典，不是一次新的注意力。',
    ]
    for i, line in enumerate(cmp_lines):
        d.text(rx + 14, ey2 + 44 + i * 20, line, 12.8, anchor='start')

    d.text(40, ey + eh + 26, '虚线：仅用于匹配/地址计算的辅助数据；实线：Hidden State 或记忆内容的主数据路径。所有节点关联已有计算面板（Parameter / Forward / Backward），不新增另一套 FLOPs。', 14, anchor='start', fill='#6b7790')
    d.raw('</g>')
    markup = ''.join(d.S[offset:])
    del d.S[offset:]
    return markup
