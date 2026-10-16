"""Standalone 43-layer vertical backbone SVG, content/ordering/color-legend faithful to
deepseek_v41_43_layer_mainline_en.drawio (parsed as XML, not screenshotted), geometry
rebuilt as a maintainable, zoomable/pannable SVG using build_diagram's shared helpers.

Source-verified against models/DeepSeek-V4.1-Flash/model.py, engram.py, config.json:
- One `Block` class looped 40 times (model.py:1203-1204); "Encoder"/"Decoder" are the
  drawio's own labels for layers 0-19 / 20-39, matching the technical report's naming,
  not separate classes or a Cross-Attention module.
- CSA2 schedule (config.json): kv_source_layers=[2,8,14,20] (Full -- new shared/compressed
  KV); index_source_layers=[2,8,14,20,24,28,32,36] (24/28/32/36 are Reindex -- rescun their
  own Query against the EXISTING L20 KV inside L20's candidate-block mask, they do NOT
  create new KV); every other compressed layer is Reuse (reads the most recently written
  KV/Top-k, still computes its own Local KV every layer). candidate_source_layer=20 only.
- Engram runs in Transformer.forward before Block.forward, only at engram_layer_ids=[1,14].
- Draft (DSpark) layers 40-42: L37/38/39 capture mean(4 HC streams) BEFORE their own Block
  call (model.py:1265-1266, which runs before `h, pre_mix = layer(...)` at 1267) --
  pre-block hidden state, not the completed block's output. forward_spec is a separate
  entry point, never auto-invoked by forward() (model.py comment at 129-130).
"""
import build_diagram as d

# ---------------------------------------------------------------------------
# Layer spec: (layer_num, phase, kind, steps|text)
#   kind 'full'    -> steps = [(color, data_key, [lines]), ...] stacked sub-boxes
#   kind 'compact' -> steps = a single text line (drawio's one-line collapsed boxes)
# Content and wording are taken directly from the drawio's own cell values.
# ---------------------------------------------------------------------------
ENCODER_FULL = {
    0: ('layer0', [
        ('hcmix', 'hc', ['Collapse 4 HC streams with incoming Pre • normalize']),
        ('localkv', 'query', ['Build Q (5120→1280→64×512) and own Local KV (512)']),
        ('localkv', 'sliding', ['Causal 128-token window → attention-weighted output']),
        ('localkv', 'output', ['Inverse RoPE • grouped output projection → 5120']),
        ('moelane', 'moe', ['HC Post + Comb → MoE (6 routed + 1 shared) → HC Post']),
    ]),
    1: ('layer1', [
        ('engramlane', 'engram1', ['Inject Engram gate × retrieved value into all 4 streams']),
        ('localkv', 'sliding', ['HC Pre → normalize → LOCAL attention (window 128)']),
        ('moelane', 'moe', ['HC Post → 6 routed + 1 shared expert → 4 streams']),
    ]),
    2: ('layer2', [
        ('localkv', 'query', ['Own Q and Local KV; read window of 128']),
        ('sharedkv', 'compressor', ['2:1 gated compression → 512 shared positions']),
        ('indexkv', 'indexer', ['Index keys + scores → select up to 512 positions']),
        ('localkv', 'sparse', ['One sparse attention over Local + selected Shared KV']),
        ('moelane', 'moe', ['Output projection → HC Post → MoE → HC Post']),
    ]),
    3: (None, [
        ('localkv', 'query', ['Own Q and Local KV; no new compressed KV']),
        ('sharedkv', 'global', ['Read shared 2:1 KV produced by Layer 2']),
        ('indexkv', 'indexer', ['Reuse selected compressed positions from Layer 2']),
        ('moelane', 'sparse', ['One sparse attention → HC / MoE update']),
    ]),
    8: (None, [
        ('sharedkv', 'compressor', ['New 2:1 shared KV replaces the prior shared source']),
        ('indexkv', 'indexer', ['Recompute Index Keys and Top-k for the new source']),
    ]),
    14: ('layer14', [
        ('engramlane', 'engram14', ['Engram injects n-gram value BEFORE this layer']),
        ('sharedkv', 'compressor', ['Generate new 2:1 shared KV + index positions']),
        ('moelane', 'sparse', ['Local + shared attention → HC / MoE']),
    ]),
}
ENCODER_REUSE_TPL = 'Own Local KV • shared KV from L{src} • selected positions from L{src}'

DECODER_FULL = {
    20: ('layer20', [
        ('hcmix', 'hc', ['Continue HC stream from Layer 19; no new embedding']),
        ('localkv', 'query', ['Own Q + Local KV for causal window 128']),
        ('sharedkv', 'global', ['Refresh shared KV with 1:1 projection (no 2:1 pooling)']),
        ('indexkv', 'indexer', ['Write index keys; preselect 8-position candidate blocks']),
        ('localkv', 'multi', ['Choose compressed positions → ONE sparse attention']),
        ('moelane', 'moe', ['Output projection → HC Post → MoE → HC Post']),
    ]),
    21: ('layer21', [
        ('localkv', 'query', ['Own Q and Local KV (not reused from Layer 20)']),
        ('sharedkv', 'global', ['Read Layer 20 shared KV and its Index Keys']),
        ('indexkv', 'indexer', ['Reuse Layer 20 selected positions and candidate mask']),
        ('moelane', 'multi', ['One sparse attention → HC → MoE → next layer']),
    ]),
    24: (None, [
        ('sharedkv', 'global', ['Keep Layer 20 compressed KV and index keys']),
        ('indexkv', 'indexer', ['Rescore within Layer 20 candidate blocks → new Top-k']),
    ]),
    28: (None, [
        ('sharedkv', 'global', ['Keep Layer 20 compressed KV']),
        ('indexkv', 'indexer', ['Recompute Top-k inside shared candidate blocks']),
    ]),
    32: (None, [
        ('sharedkv', 'global', ['Keep Layer 20 compressed KV']),
        ('indexkv', 'indexer', ['Recompute Top-k inside shared candidate blocks']),
    ]),
    36: (None, [
        ('sharedkv', 'global', ['Keep Layer 20 compressed KV']),
        ('indexkv', 'indexer', ['Last Top-k refresh before Layers 37–39']),
    ]),
}
DECODER_REUSE_SRC = {22: 20, 23: 20, 25: 24, 26: 24, 27: 24, 29: 28, 30: 28, 31: 28,
                      33: 32, 34: 32, 35: 32}

INDEX_SOURCE = {2, 8, 14, 20, 24, 28, 32, 36}
KV_SOURCE = {2, 8, 14, 20}
ENGRAM_LAYERS = {1, 14}

CANVAS_W = 1500
COL_X, COL_W = 60, 900
NOTE_X, NOTE_W = 990, 450


def node(key, x, y, w, h, lines, color, size=15.5, bold=False):
    d.box(key, x, y, w, h, lines, color, size, bold)


def layer_key(n):
    return f'layer{n}' if n in (0, 1, 2, 13, 14, 19, 20, 21, 22, 33, 34, 39) else None


def _title(y, n, phase):
    d.text(COL_X + COL_W / 2, y, f'LAYER {n:02d}   {phase}', 21, True)


def _note(y, h, lines, color='backbone'):
    d.rect(NOTE_X, y, NOTE_W, h, color, opacity=.5)
    d.text(NOTE_X + 16, y + 24, lines, 14, anchor='start', leading=19)


def build_detail():
    offset = len(d.S)
    y = 20
    marks = {}
    d.raw('<g id="backbone43-detail" style="display:none" font-family="Arial, Microsoft YaHei, sans-serif" '
          'role="group" aria-label="43-Layer Source-Aligned Backbone">')

    # ---- Header -------------------------------------------------------
    d.rect(10, y, CANVAS_W - 20, 74, 'backbone', opacity=.9)
    d.text(CANVAS_W / 2, y + 30, 'DeepSeek V4.1 Flash | Complete 43-Stage Architecture', 24, True, fill='#ffffff')
    d.text(CANVAS_W / 2, y + 56, 'One continuous 40-layer backbone • 1024-token prefill example • 3 connected Draft stages', 14, fill='#dbe8ff')
    y += 94
    d.text(CANVAS_W / 2, y, 'Follow the BLUE central column top → bottom. Colored boxes mark which "lane" each step belongs to — not extra layers.', 13.5, fill='#526781')
    y += 26
    legend = [('backbone', 'Hidden state'), ('hcmix', 'HC mixing'), ('localkv', 'Local KV'),
              ('sharedkv', 'Shared compressed KV'), ('indexkv', 'Index / Top-k'),
              ('engramlane', 'Engram'), ('draftlane', 'Draft')]
    lx = COL_X
    for color, label in legend:
        lw = 16 + len(label) * 7.2
        d.box('overview', lx, y, lw, 26, label, color, 11.5, True)
        lx += lw + 10
    y += 46

    # ---- Input stage ----------------------------------------------------
    marks['input'] = y
    node('ids', COL_X, y, COL_W, 44, '1024 input token IDs   [1, 1024]', 'white', 16, True)
    y2 = y + 44 + 14
    node('embedding', COL_X, y2, COL_W, 50, 'Token Embeddings   [1, 1024, 5120]', 'backbone', 16, True)
    y3 = y2 + 50 + 14
    node('streams', COL_X, y3, COL_W, 50, 'Replicate into 4 HC streams   [1, 1024, 4, 5120]', 'hcmix', 16, True)
    _note(y2 - 4, 90, ['N-gram position preparation (text IDs)', 'Normalize → compressed IDs → 2/3/4-grams',
                        '24 hash addresses per token, for Layers 1 and 14'], 'engramlane')
    _note(y3 - 4, 70, ['Initial HC input weights: [1,0,0,0] for every position',
                        'Next-layer mixing weights travel with the 4 streams'], 'hcmix')
    y = y3 + 50 + 30

    # ---- Encoder section --------------------------------------------
    d.text(COL_X, y, 'ENCODER — 20 consecutive backbone layers (compress_ratio = 2:1 at L2/8/14)', 19, True, anchor='start', fill='#17304C')
    y += 24
    enc_top = y
    for n in range(20):
        if n in ENCODER_FULL:
            marks[f'l{n}'] = y
            lkey, steps = ENCODER_FULL[n]
            sh = 54
            box_h = 40 + len(steps) * sh
            d.rect(COL_X - 10, y, COL_W + 20, box_h, 'white', opacity=.4)
            _title(y + 24, n, 'ENCODER')
            sy = y + 40
            prev = None
            for color, key, lines in steps:
                node(key, COL_X + 20, sy, COL_W - 40, sh - 8, lines, color, 14.5, True)
                if prev:
                    d.arrow(f'M{COL_X + COL_W / 2} {prev} V{sy}', width=1.6)
                prev = sy + sh - 8
                sy += sh
            if n == 2:
                _note(y + 36, 150, ['NEW SHARED KV • source L2', 'Ratio 2:1 • 1024 → 512 entries',
                                     'Index Key created from pre-RoPE latent'], 'sharedkv')
            y += box_h + 16
        else:
            node(layer_key(n) or 'sparse', COL_X, y, COL_W, 46,
                 [f'LAYER {n:02d}   ENCODER', ENCODER_REUSE_TPL.format(src=2 if n < 8 else 8 if n < 14 else 14)],
                 'backbone', 14)
            y += 46 + 10
    d.rect(COL_X - 24, enc_top - 4, COL_W + 48, y - enc_top - 2, 'backbone', opacity=.06)
    y += 10

    # ---- L19 -> L20 boundary -------------------------------------------
    marks['boundary'] = y
    d.rect(COL_X - 10, y, COL_W + 20, 86, 'hcmix', opacity=.35)
    d.text(COL_X + COL_W / 2, y + 26, 'L19 → L20 boundary: same Block loop, no separate Cross-Attention module', 15.5, True)
    d.text(COL_X + COL_W / 2, y + 50, 'E = Encoder final 4-stream state [1,1024,4,5120] → HC Pre + Attention RMSNorm → X20 [1,1024,5120]', 13.5)
    d.text(COL_X + COL_W / 2, y + 70, 'X20 is just Layer 20’s ordinary Block input — Layer 20 uses it for its OWN new Query, Local KV, Compressed KV and Indexer.', 12.8, fill='#6b3fb8')
    y += 86 + 20

    # ---- Decoder section --------------------------------------------
    d.text(COL_X, y, 'DECODER — 20 consecutive backbone layers (compress_ratio = 1:1 at L20)', 19, True, anchor='start', fill='#17304C')
    y += 24
    dec_top = y
    for n in range(20, 40):
        if n in DECODER_FULL:
            marks[f'l{n}'] = y
            lkey, steps = DECODER_FULL[n]
            sh = 46
            box_h = 40 + len(steps) * sh
            d.rect(COL_X - 10, y, COL_W + 20, box_h, 'white', opacity=.4)
            _title(y + 24, n, 'DECODER')
            sy = y + 40
            prev = None
            for color, key, lines in steps:
                node(key, COL_X + 20, sy, COL_W - 40, sh - 8, lines, color, 14, True)
                if prev:
                    d.arrow(f'M{COL_X + COL_W / 2} {prev} V{sy}', width=1.6)
                prev = sy + sh - 8
                sy += sh
            if n == 20:
                _note(y + 36, 150, ['NEW SHARED KV • source L20', 'Ratio 1:1 • 1024 → 1024 entries',
                                     'INDEX + CANDIDATE SOURCE: 8-position blocks → candidate mask → Top-k'], 'sharedkv')
            if n == 24:
                _note(y + 4, 70, ['REFRESH Top-k • L24', 'Use L20 candidate blocks and Index Keys',
                                   'Does NOT rerun candidate-block filtering — only rescores within it'], 'indexkv')
            y += box_h + 16
        elif n in DECODER_REUSE_SRC:
            idx_src = DECODER_REUSE_SRC[n]
            node(layer_key(n) or 'multi', COL_X, y, COL_W, 46,
                 [f'LAYER {n:02d}   DECODER', f'Own Local KV • shared KV from L20 • selected positions from L{idx_src}'],
                 'backbone', 14)
            y += 46 + 10
        else:  # 37, 38, 39 -- pre-block capture for Draft
            if n == 37:
                marks['l37'] = y
            node('streams', COL_X, y, COL_W, 70,
                 [f'LAYER {n:02d}   DECODER', 'Before this block: capture mean of 4 HC streams',
                  'Own Q / Local + shared KV → HC and MoE'],
                 'draftlane', 13.5, True)
            _note(y + 6, 56, [f'CAPTURE BEFORE L{n}', 'Mean over 4 streams → [1,1024,5120]',
                               '(captured BEFORE this Block runs, not its output)'], 'draftlane')
            y += 70 + 10
    d.rect(COL_X - 24, dec_top - 4, COL_W + 48, y - dec_top - 2, 'sharedkv', opacity=.05)
    y += 10

    # ---- Final output ----------------------------------------------
    node('finalnorm', COL_X, y, COL_W / 2 - 10, 50, ['Final HC collapse → RMSNorm'], 'backbone', 15, True)
    node('head', COL_X + COL_W / 2 + 10, y, COL_W / 2 - 10, 50, ['LM Head → logits [1,129280]'], 'backbone', 15, True)
    y += 50 + 14
    node('prediction', COL_X, y, COL_W, 40, 'Last position → next-token ID (normal 40-layer output)', 'white', 14)
    y += 40 + 30

    # ---- Draft entry notice -------------------------------------------
    d.rect(COL_X - 10, y, COL_W + 20, 64, 'draftlane', opacity=.3)
    d.text(COL_X + COL_W / 2, y + 24, 'SPECULATIVE GENERATION ENTRY — separate invocation (forward_spec)', 16, True)
    d.text(COL_X + COL_W / 2, y + 46, 'Main inference does NOT automatically execute Draft 40–42. They are shown here as the complete downstream pipeline.', 12.8)
    y += 64 + 24

    marks['draft'] = y
    d.text(COL_X, y, 'DRAFT — 3 additional stages (separate speculative entry; DSpark)', 19, True, anchor='start', fill='#17304C')
    y += 24
    draft_top = y
    draft_steps = [
        (40, ['Separate speculative decode call receives current token',
              'Concatenate captured L37–39 states → project to 5120',
              'Current token + 4 placeholders → length-5 draft input',
              'Draft local attention reads main-context window + drafts',
              '3 routed + 1 shared expert → pass updated draft streams']),
        (41, ['Draft local attention with main-context KV', 'Draft HC + MoE → continue draft streams']),
        (42, ['Draft local attention + HC + MoE', 'Shared output head + Markov correction + confidence',
              'Produce 5 candidate tokens; acceptance external']),
    ]
    for n, lines in draft_steps:
        box_h = 40 + len(lines) * 32
        d.rect(COL_X - 10, y, COL_W + 20, box_h, 'white', opacity=.4)
        _title(y + 24, n, 'DRAFT')
        sy = y + 40
        tab = {40: 'inputs', 41: 'blocks', 42: 'heads'}[n]
        for i, line in enumerate(lines):
            d.raw(f'<g class="draft-jump-module" data-draft-jump="{tab}" role="button" tabindex="0" '
                  f'aria-label="{line}"><title>{line} • 打开 DSpark 详情</title>')
            d.rect(COL_X + 20, sy, COL_W - 40, 26, 'draftlane')
            d.text(COL_X + 20 + (COL_W - 40) / 2, sy + 13 + 5, line, 13, maxwidth=COL_W - 56)
            d.raw('</g>')
            sy += 32
        y += box_h + 14
    d.rect(COL_X - 24, draft_top - 4, COL_W + 48, y - draft_top - 2, 'draftlane', opacity=.06)
    _note(draft_top, 150, ['JOIN 3 CAPTURES (L37+L38+L39)', '[1,1024,15360] → draft context projection',
                            'Decode: last current-token row drives candidate generation'], 'draftlane')
    _note(draft_top + 160, 110, ['DRAFT EXECUTION RULES', 'Cache-init calls: cache only, no candidates.',
                                  'Later decode calls: 3 stages compute candidates.',
                                  'External scheduler handles accept/reject.'], 'draftlane')
    _note(draft_top + 280, 70, ['DRAFT EXPERTS', '3 routed + 1 shared expert per draft stage',
                                 '(backbone MoE uses 6 routed + 1 shared)'], 'moelane')
    y += 20

    # ---- Footer: reading notes / verified scope -------------------------
    marks['notes'] = y
    d.rect(COL_X - 10, y, COL_W + 20, 160, 'backbone', opacity=.1)
    d.text(COL_X, y + 24, 'READING NOTES · VERIFIED SCOPE', 16, True, anchor='start')
    notes = [
        'Backbone = 40 sequential layers. Encoder and Decoder are contiguous phases of ONE model, not two separately invoked networks.',
        'Every backbone layer has its own Query and local causal window. Shared compressed KV is refreshed only at L2 / L8 / L14 / L20.',
        'Top-k is refreshed at L2 / L8 / L14 / L20 / L24 / L28 / L32 / L36. Engram modifies the 4 streams before L1 and L14.',
        'Draft 40–42 are explicitly shown downstream as the complete speculative path, but must be invoked separately by orchestration.',
        'This diagram simplifies implementation-level quantization/communication details; it does not describe training backward kernels.',
    ]
    for i, line in enumerate(notes):
        d.text(COL_X, y + 50 + i * 21, line, 13, anchor='start', fill='#44546b')
    y += 180

    d.raw('</g>')
    markup = ''.join(d.S[offset:]).replace('class="module"', 'class="backbone43-module"')
    del d.S[offset:]
    return markup, y, marks
