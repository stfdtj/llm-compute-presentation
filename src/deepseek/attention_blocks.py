"""Refine the original Attention blocks; enlarged details live in the SAME SVG."""
import build_diagram as d


def compact():
    """Keep the reference's lower four-block composition and original bounds."""
    for decoder, x, w in [(False, 15, 483), (True, 508, 368)]:
        color = 'orange' if decoder else 'blue'
        key = 'decoderattn' if decoder else 'encoderattn'
        title = ('2. Decoder' if decoder else '1. Encoder') + ' Attention (Detail)'
        d.frame(key, x, 585, w, 333, color, title, 21)
        center = x + w/2
        d.text(x+w-90, 640, 'X: [1,5120]', 13)
        d.raw(f'<rect class="csa2-boundary" data-csa2-compact="{key}" x="{x+8}" y="650" width="{w-16}" height="242" rx="8" fill="none" stroke="#247b9f" stroke-width="1.5" stroke-dasharray="6 4"/>')
        d.text(x+w*.53,648,'CSA2 (L20–39)' if decoder else 'CSA2 (L2–19)',11)
        d.box('query', x+12, 654, w*.48-18, 64,
              ['Low-rank Query + RoPE', '5120 → 1280 → 64 × 512'], 'blue', 14)
        d.box('kv', x+w*.48+6, 654, w*.52-18, 64,
              ['Own Local KV + RoPE', '5120 → 512; window ≤128'], 'green', 14)
        d.label('sliding',x+w*.48+6,696,w*.52-18,20,[],13)
        d.box('compressor', x+12, 734, w-24, 47,
              ['KV20 generated once per call; 1 token → 1 row' if decoder else
               'KV sources 2 / 8 / 14; 2 tokens → 1 row',
               'Later layers reuse KV; every layer still builds Local KV'], color, 13)
        d.raw(f'<g class="inline-indexer" data-inline-region="{ "c2" if decoder else "c1"}">')
        inline_start=len(d.S)
        d.box('indexer', x+12, 795, w-24, 43,
              ['Indexer Q / K → scores → Top-512 positions',
               'Run: 20,24,28,32,36; reuse between' if decoder else
               'Run: 2,8,14; reuse between; 0–1 window only'], 'purple', 13)
        for i in range(inline_start,len(d.S)):
            d.S[i]=d.S[i].replace('class="module"','class="inline-indexer-node"')
        d.raw('</g>')
        d.box('multi', x+12, 851, w-24, 37,
              ['Local + selected Global KV → sparse_attn → output'], 'blue', 13)
        d.label('sparse',x+12,851,100,37,[],13)
        for a,b in [(781,795),(838,851)]:
            d.arrow(f'M{center} {a} V{b}')
        d.arrow(f'M{x+40} 718 H{x+6} V870 H{x+12}')
        d.arrow(f'M{x+w-40} 718 H{x+w-6} V870 H{x+w-12}')
        d.raw(f'<g class="attention-focus" data-attention-focus="{"c2" if decoder else "c1"}" role="button" tabindex="0" aria-label="Enlarge {title}">')
        d.rect(x+12, 623, 142, 21, 'white')
        d.text(x+83, 638, 'Enlarge detail ↗', 13, True)
        d.raw('</g>')


def expanded(decoder=False):
    """Readable 1500×970 block, with the same boxes, serif type and pastel colors."""
    name = 'decoder' if decoder else 'encoder'
    key = 'decoderattn' if decoder else 'encoderattn'
    ox, oy = 1720, 1050 if decoder else 0
    d.raw(f'<g class="attention-expanded" id="{name}-attention-expanded" data-attention-expanded="{name}" transform="translate({ox} {oy})" font-family="Times New Roman, Microsoft YaHei, serif">')
    d.frame(key, 10, 10, 1480, 1025, 'orange' if decoder else 'blue',
            ('Decoder · Layers 20–39' if decoder else 'Encoder · Layers 0–19') + ' — Attention (Detail)', 28)
    d.text(35, 80, 'Input X = this Block’s mHC Pre + RMSNorm: [1,5120]. B=1; sequence axis omitted below.', 21, anchor='start')
    d.text(35, 113, 'Generate KV: 20. Reindex: 24 / 28 / 32 / 36. Reuse KV20 throughout 20–39.' if decoder else
           '0–1: window only. Generate KV + Index Keys + Top-k: 2 / 8 / 14. Reuse: 3–7 / 9–13 / 15–19.', 20, anchor='start')

    d.raw(f'<rect class="csa2-boundary" data-csa2-operations="{name}" x="20" y="139" width="1460" height="776" rx="12" fill="none" stroke="#247b9f" stroke-width="2.5" stroke-dasharray="10 6"/>')
    d.raw('<rect x="1020" y="124" width="430" height="25" fill="white"/>')
    d.text(1235,143,'CSA2 · Layers 20–39' if decoder else 'CSA2 · Layers 2–19',20,True)
    def node(key, x, y, w, h, lines, color='blue', inline=False):
        if inline:d.raw('<g class="inline-indexer">')
        inline_start=len(d.S)
        d.box(key,x,y,w,h,lines,color,20,leading=25)
        if inline:
            for i in range(inline_start,len(d.S)):
                d.S[i]=d.S[i].replace('class="module"','class="inline-indexer-node"')
            d.raw('</g>')

    node('query',35,150,440,135,['Main Query · low-rank projection',
         'X × Wqa[5120,1280] → RMSNorm → qr[1280]',
         'qr × Wqb[1280,32768] → Q[64,512]', 'RoPE on last 64 dimensions of each head'])
    node('kv',530,150,440,135,['Local KV · independently computed each layer',
         'X × Wkv[5120,512] → RMSNorm → [512]',
         'RoPE → FP8 → own 128-slot ring buffer', 'Recent visible rows: [N,512], N ≤128'],'green')
    node('compressor',1025,150,440,135,['Compressed KV · generate only at source layers',
         '20: X × [5120,512] → RMSNorm (ratio 1)' if decoder else
         '2 / 8 / 14: project + learned softmax pool (ratio 2)',
         'Pre-RoPE latent: [new rows,512]',
         'Otherwise reuse the current source cache'],'orange')
    # Reuse the original parameter-panel hit targets on their corresponding formula lines.
    d.label('qa',35,196,295,25,[],18)
    d.label('qnorm',330,196,145,25,[],18)
    d.label('qb',35,222,440,25,[],18)
    d.label('kvnorm',820,196,150,25,[],18)

    node('indexer',35,365,440,145,['Indexer Query + learned head weights',
         'qr × Wqi[1280,4096] → [32,128]',
         'Query RoPE → FP4', 'X × Wweights[5120,32] × 128⁻½ × 32⁻½'],'purple',True)
    node('indexer',530,365,440,145,['Index Keys · source-owned cache',
         'Source latent × Wk[512,128] → RMSNorm',
         'Key RoPE → FP4 → cache [S,128]',
         'Reindex reads cached Keys; no new KV'],'purple',True)
    node('global',1025,365,440,145,['Publish shared Compressed KV',
         'After Index Keys use the pre-RoPE latent:',
         'RoPE → FP4 (group 16, E4M3 scales)',
         'Append source cache [S,512]; retain older rows'],'orange')
    d.arrow('M255 285 V365',width=1.8)
    d.arrow('M1245 285 V320 H750 V365',width=1.8)
    d.arrow('M1245 320 V365',width=1.8)
    d.text(830,346,'Pre-RoPE latent',18,anchor='start')

    node('indexer',35,575,935,155,['Indexer scoring → masks → Top-k integer positions',
         'Qindex[32,128] × Kindexᵀ[128,S] → [32,S]; ReLU × head weights → sum heads → [S].',
         'Causal mask. At 20: publish candidate blocks; later reindexers apply that candidate mask.' if decoder else
         'Causal mask: only completed groups are visible. No candidate-block stage in Encoder.',
         'Top min(512,S) by score → sort position IDs → add Local KV offset. Reuse skips scoring.',
         'Reindex changes selected positions; it does not regenerate the KV vectors.'],'purple',True)
    node('multi',1025,575,440,155,['Gather inside sparse_attn()',
         'Concatenate Local [N,512] + Global [S,512].',
         'Read Local IDs + selected compressed IDs.',
         'Each 512-vector is used as both K and V.',
         'Reusing layers still run main Attention.'],'blue')
    d.arrow('M255 510 V575',width=1.8)
    d.arrow('M750 510 V575',width=1.8)
    d.arrow('M970 650 H1025',width=1.8)
    d.arrow('M1245 510 V575',width=1.8)
    d.arrow('M750 285 V310 H995 V550 H1390 V575',width=1.8)
    d.text(1125,546,'Local KV',18,anchor='start',fill='#238461')
    node('output',35,800,1430,105,['Main Attention → inverse RoPE → grouped output projection',
         'Per head: Q[1,512] × Kᵀ[512,N+k] → scaled softmax with learned sink → weighted V[N+k,512] → [1,512].',
         '64 heads → inverse RoPE → 8 groups [4096] × [4096,1024] → [8192] × [8192,5120] → U[5120].'])
    d.arrow('M35 220 H22 V770 H200 V800',width=1.8)
    d.arrow('M1245 730 V800',width=1.8)
    node('hc',35,933,1430,40,['Outside CSA2: mHC Post distributes U and mixes the four residual streams → [1,1,4,5120].'],'orange')
    d.text(35,1007,'Candidate blocks: 8 positions/block, ≤2048 blocks, newest pinned. Source scores first, masks later.' if decoder else
           'Incomplete ratio-2 groups: local/pinned Index Key publication differs; see Full Indexer calculations note.',18,anchor='start')
    from layer_trace import draw
    draw(decoder)
    d.raw('</g>')


def draw_expanded():
    expanded(False)
    expanded(True)
    draft()


def draft():
    d.raw('<g class="attention-expanded" id="draft-expanded" transform="translate(3300 0)" font-family="Times New Roman, Microsoft YaHei, serif">')
    d.frame('overview',10,10,1480,950,'purple','DSpark · three draft stages, separate from the 40-layer backbone',28)
    d.text(35,85,'Transformer.forward_spec(): a separate draft call. No backbone Layers 40–42 are appended to forward().',22,anchor='start')
    d.box('hidden',35,125,690,145,['Main features before Blocks 37 / 38 / 39',
          'Mean over four mHC streams → concatenate [B,L,15360]',
          'main_proj[15360,5120] → RMSNorm → main_x[B,L,5120]',
          'Each draft stage reprojects main_x into its own window KV.'],'blue',22,leading=29)
    d.box('ids',775,125,690,145,['Five draft slots: [current token, noise, noise, noise, noise]',
          'Noise ID 128799; main embedding table is shared.',
          '[B,5,5120] → expand to [B,5,4,5120].',
          'Initialize mHC pre_mix; pass the draft states through all three stages.'],'purple',22,leading=29)
    for i in range(3):
        x=235+i*365
        d.box('moe',x,355,310,170,[f'mtp[{i}] · DSparkBlock',
              'mHC Pre → Norm → Draft Attn', 'mHC Post → mHC Pre → Norm',
              'MoE → mHC Post', '128 routed, Top-3 + 1 shared'],'purple',22,leading=28)
        if i<2:d.arrow(f'M{x+310} 440 H{x+365}',width=2)
    d.arrow('M1120 270 V310 H390 V355',width=2)
    d.arrow('M380 270 V300 H210 V440 H235',width=2)
    d.box('multi',35,600,690,165,['DSparkAttention: own Local KV, no Compressor / Indexer',
          'Q[B,5,64,512]; private main-conditioned window [B,128,512]',
          '+ current draft KV[B,5,512] → sparse_attn → inverse RoPE',
          '8 grouped projections → [B,5,5120] → mHC Post',
          'All five current draft slots are visible to each draft Query.'],'blue',21,leading=27)
    d.box('head',775,600,690,165,['Final mHC collapse → Norm → shared vocabulary Head',
          'Logits[B,5,129280] + Markov bias (rank 256)',
          'Sample 5 IDs sequentially; return seed + 5 candidates.',
          'Confidence: [hidden5120, Markov256] × [5376,1] → [B,5]',
          'Return candidate IDs, logits and confidence to caller.'],'purple',21,leading=27)
    d.arrow('M1120 525 V560 H380 V600',width=2)
    d.arrow('M725 685 H775',width=2)
    d.text(35,835,'Prefill: start_pos=0 only seeds the three private draft caches; forward_spec returns None.',22,anchor='start')
    d.text(35,875,'Decode: the caller must invoke main-model verification / acceptance. That scheduler is not implemented in model.py.',21,anchor='start')
    d.text(35,920,'Draft-specific FLOPs are not substituted with backbone totals. Existing calculation links retain their original scope.',20,anchor='start')
    d.raw('</g>')
