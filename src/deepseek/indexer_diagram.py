"""Additional SVG only: keep the original diagram and calculation mapping intact."""
import build_diagram as diagram


def build_indexer(config):
    d = diagram
    d.S.clear()
    width, rank = config['dim'], config['q_lora_rank']
    heads, dim = config['index_n_heads'], config['index_head_dim']
    kv_dim, topk, window = config['head_dim'], config['index_topk'], config['window_size']

    def node(name, key, x, y, w, h, lines, color='blue', size=20):
        d.raw(f'<g data-indexer-node="{name}">')
        d.box(key, x, y, w, h, lines, color, size)
        d.raw('</g>')

    def edge(name, path, dashed=False):
        d.raw(f'<g data-indexer-edge="{name}">')
        d.arrow(path, dash=dashed, width=1.8)
        d.raw('</g>')

    d.raw('<g id="indexer-entry-links" font-family="Times New Roman, Microsoft YaHei, serif">')
    for source, layer, x, w in [('encoder', 2, 27, 399), ('decoder', 20, 465, 399)]:
        d.raw(f'<g data-indexer-entry="{source}" data-indexer-layer="{layer}" aria-controls="indexer-detail" aria-expanded="false">')
        d.box('indexer', x, 897, w, 19, 'Indexer → Top-512 → Global KV  ↗', 'orange', 13, True)
        d.raw('</g>')
    d.raw('</g>')

    d.raw('<g id="indexer-detail" style="display:none" font-family="Times New Roman, Microsoft YaHei, serif" role="group" aria-label="Indexer Detail：全局 KV 位置选择流程">')
    d.rect(15, 20, 1642, 1075, 'blue', opacity=.08)
    d.text(38, 60, 'Indexer Detail · Global KV Selection', 32, True, anchor='start')
    d.text(38, 95, '选择 Global KV 位置 · 每个模块可点击，复用原计算面板', 20, anchor='start')
    d.text(1630, 60, 'Attention 路径 · 与 MoE Router 分开', 19, anchor='end', fill='#806019')
    d.text(1630, 94, '1 个当前 Query · Batch 省略 · C = 实际评分位置数', 18, anchor='end')

    node('query', 'indexer', 40, 150, 290, 110,
         ['Indexer Query Projection', f'归一化低秩 Query [1,{rank}]',
          f'× [{rank},{heads * dim}] → [1,{heads * dim}]',
          f'Reshape → [1,{heads},{dim}]', '→ RoPE → FP4'], size=18)
    node('compressor', 'compressor', 40, 310, 290, 80,
         ['Global KV Compressor', f'新完成的 latent [1,{kv_dim}]', 'Index Key 分支在 RoPE 之前'], 'orange', 19)
    node('key', 'indexer', 40, 435, 290, 110,
         ['Indexer Key Projection', f'[1,{kv_dim}] × [{kv_dim},{dim}] = [1,{dim}]',
          'RMSNorm → RoPE → FP4', '仅源层追加新条目'], 'blue', 19)
    node('keycache', 'indexer', 40, 590, 290, 110,
         ['Index Key Cache', f'[S,{dim}] · 历史 Key 复用', '不重新投影全部历史'], 'orange', 21)
    node('score', 'indexer', 400, 330, 270, 110,
         ['Index Scoring + ReLU', f'[{heads},{dim}] × [{dim},C]', f'→ [{heads},C]', 'C 不是 Top-K 的 512'], size=21)
    node('weights', 'indexer', 735, 150, 260, 110,
         ['Head Weight Projection', f'当前 X [1,{width}] × [{width},{heads}]',
          f'→ [1,{heads}] / √({dim} × {heads})', '由当前状态生成权重 w'], 'green', 18)
    node('cachedindices', 'indexer', 735, 150, 260, 110,
         ['Cached Position Indices', '复用最近的有效索引 ℐ', '不运行 Indexer'], 'orange', 21)
    node('aggregate', 'indexer', 735, 330, 260, 110,
         ['Head Aggregation', f'Σ over {heads} heads', 'sⱼ = Σₕ wₕ · ReLU(qₕ · kⱼ)', '→ [1,C] · 每个候选一项评分'], 'green', 19)
    node('topk', 'indexer', 1060, 330, 200, 110,
         [f'Top-K ≤ {topk}', '因果 / 候选 mask', '输出整数位置索引 ℐ', '按位置排序'], 'purple', 19)
    node('reusedtopk', 'indexer', 1060, 330, 200, 110,
         ['Reuse Indices ℐ', f'最多 {topk} 个位置', '沿用已有选取结果', '本层不重新 Top-K'], 'purple', 19)
    node('globalcache', 'global', 735, 535, 260, 120,
         ['Global KV Cache', f'[S,{kv_dim}] · 已有缓存', '跨层共享', 'Index Keys 与 Global KV 分开保存'], 'orange', 18)
    node('gather', 'indexer', 1060, 535, 200, 120,
         ['Global KV Gather', '按索引 ℐ 读取原缓存', f'K ≤ {topk} 行', '逻辑读取，无新增 KV'], 'purple', 18)
    node('mainquery', 'query', 1350, 150, 260, 110,
         ['Main Attention Query', f'当前层自己的 {config["n_heads"]} 个 Head',
          f'[1,{config["n_heads"]},{kv_dim}]', '独立于 Indexer Query'], size=20)
    node('local', 'localstates', 1350, 330, 260, 110,
         ['Local KV', f'当前层最近 ≤ {window} 个位置', '每层独立产生', '不经过全局 Top-K'], 'green', 20)
    node('main', 'multi', 1350, 535, 260, 120,
         ['Main Attention', f'Global ≤ {topk} + Local ≤ {window}',
          '共同归一化与加权汇总', '仍在当前层执行'], size=20)

    # Separate lanes for current-query, cached-key, and main-KV data; no FFN route.
    edge('query', 'M330 205 H535 V330')
    edge('key', 'M185 390 V435')
    edge('key', 'M185 545 V590')
    edge('score', 'M330 645 H365 V415 H400')
    edge('score', 'M670 385 H735')
    edge('weights', 'M865 260 V330')
    edge('aggregate', 'M995 385 H1060')
    edge('cachedindices', 'M995 205 H1160 V330', True)
    edge('indices', 'M1160 440 V535')
    edge('globalwrite', 'M40 350 H25 V735 H865 V655', True)
    edge('gather', 'M995 595 H1060')
    edge('main', 'M1260 595 H1350')
    edge('local', 'M1480 440 V535')
    edge('mainquery', 'M1610 205 H1635 V595 H1610')
    d.raw('<g data-indexer-edge="globalwrite">')
    d.text(500, 760, '同一 Compressor latent → Global KV 位置处理 / 量化 → 缓存写入', 18)
    d.raw('</g>')
    d.text(38, 803, f'KV source layers: {config["kv_source_layers"]}    |    Index source layers: {config["index_source_layers"]}', 20, anchor='start')
    d.text(38, 837, f'仅 {len(config["index_source_layers"])} 层运行 Indexer；0–1 层仅 SWA；每层仍有自己的主要 Attention。Gather 表示按索引读取，不承诺分配独立张量。', 18, anchor='start')

    # Teaching-only worked example: 8 history positions, pick 3. Clearly separated from
    # the real index_topk=512 so nobody mistakes this illustrative count for the config value.
    ey = 880
    d.rect(32, ey, 1591, 195, 'green', opacity=.11)
    d.text(46, ey + 28, f'教学小例子 · 8 个历史位置里选 3 个（真实配置 index_topk = {topk}，这里只是便于演示的小数字）', 18, True, anchor='start')
    positions = [('p1', .11), ('p2', .52), ('p3', .74), ('p4', .05), ('p5', .63), ('p6', .22), ('p7', .81), ('p8', .38)]
    chosen = {'p3', 'p5', 'p7'}
    px0, pw, pgap = 46, 130, 16
    for i, (name, score) in enumerate(positions):
        px = px0 + i * (pw + pgap)
        picked = name in chosen
        d.raw(f'<g data-indexer-node="teach-{name}">')
        d.box('indexer', px, ey + 46, pw, 54, [f'位置 {name}', f'score = {score:.2f}'], 'purple' if picked else 'white', 13, picked)
        d.raw('</g>')
        if picked:
            d.text(px + pw / 2, ey + 116, '✓ Top-3', 12.5, True, fill='#5a21c9')
    d.text(46, ey + 150, '按分数从高到低排序：p7(0.81) · p3(0.74) · p5(0.63) · p2(0.52) · … → 选出 p7、p3、p5 这 3 个位置 → 按原始位置顺序重排为 p3、p5、p7。', 15, anchor='start')
    d.text(46, ey + 174, '真实流程完全一样，只是候选位置可以有上千个、Top-K 最多选 512 个；这个 8 选 3 的小例子不对应任何真实层的真实分数。', 14, anchor='start', fill='#5b6981')
    d.raw('</g>')
    # Own class keeps existing diagram event routing / preservation tests independent.
    return ''.join(d.S).replace('class="module"', 'class="indexer-module"')
