"""Layer-by-layer native SVG trace, following the user-provided drawio's central path and side lanes."""
import json
from pathlib import Path
import build_diagram as d

ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/'models/DeepSeek-V4.1-Flash/config.json').read_text(encoding='utf-8'))
ROW_Y=1190
ROW_H=260


def roles(n):
    kv=max((x for x in CONFIG['kv_source_layers'] if x<=n),default=None)
    idx=max((x for x in CONFIG['index_source_layers'] if x<=n),default=None)
    mode='Window' if CONFIG['compress_ratios'][n]==0 else 'Full' if n in CONFIG['kv_source_layers'] else 'Reindex' if n in CONFIG['index_source_layers'] else 'Reuse'
    return mode,kv,idx


def draw(decoder=False):
    first,last=(20,40) if decoder else (0,20)
    stage='Decoder' if decoder else 'Encoder'
    d.text(750,1115,f'{stage} · every layer in execution order',30,True)
    d.text(750,1152,'BLUE: four-stream state + delayed pre_mix. GOLD: shared KV. PURPLE: indices. Every layer builds its own Q + Local KV.',19)
    previous=None
    for i,n in enumerate(range(first,last)):
        y=ROW_Y+i*ROW_H
        mode,kv,idx=roles(n)
        d.raw(f'<g class="trace-layer" data-trace-layer="{n}" data-trace-mode="{mode}" data-kv-owner="{kv}" data-index-owner="{idx}">')
        d.rect(10,y,1480,225,'purple',False,opacity=.04)
        d.text(750,y+29,f'LAYER {n:02d} · {stage.upper()} · '+('Local window only' if mode=='Window' else f'CSA2({CONFIG["compress_ratios"][n]}, {mode})'),25,True)
        if n in CONFIG['engram_layer_ids']:
            d.text(750,y+57,f'Engram injects into all four streams BEFORE Block {n}.',18,fill='#a50027')
        elif n in CONFIG['dspark_target_layer_ids']:
            d.text(750,y+57,f'Capture mean of four streams BEFORE Block {n} for DSpark.',18,fill='#8653b6')
        elif n==20:
            d.text(750,y+57,'R19 + FFN pre_mix19 continue here; no new embedding or independent Decoder.forward().',18)
        else:
            d.text(750,y+57,'Input R[1,1,4,5120] + incoming pre_mix[1,1,4]',18)
        d.box('hc',465,y+70,570,38,['mHC Pre → RMSNorm → X[1,5120]'],'orange',19)
        # The dashed boundary encloses Attention only; mHC/Norm/MoE remain outside.
        if mode!='Window':
            d.raw(f'<rect class="csa2-boundary" data-csa2-layer="{n}" x="25" y="{y+119}" width="1450" height="53" rx="9" fill="none" stroke="#247b9f" stroke-width="2" stroke-dasharray="8 5"/>')
        d.box('encoderattn' if not decoder else 'decoderattn',465,y+125,570,42,
              ['Own Q + own Local KV → '+('causal SWA (no CSA2)' if mode=='Window' else f'CSA2 {mode} → U[1,5120]')], 'blue',19)
        d.box('moe',465,y+185,570,31,['mHC Post/Comb → FFN Pre/Norm → MoE → mHC Post/Comb'],'purple',17)
        if mode=='Window':
            left=['Own 128-token Local KV window', 'No compressed cache at this layer.']
            right=['No Indexer / no Top-k cache', 'Causal window positions only.']
        else:
            left=[f'{"NEW" if mode=="Full" else "READ"} shared KV · owner L{kv}',
                  ('2:1 pool → [S,512]' if n<20 else '1:1 projection → [S,512]') if mode=='Full' else 'Reuse cached vectors; no new projection.']
            right=[f'{"REFRESH" if mode in ("Full","Reindex") else "REUSE"} Top-k · owner L{idx}',
                   'New Index Keys + Query scores → positions' if mode=='Full' else
                   'Keep L20 Keys; fresh Query / weights → positions' if mode=='Reindex' else 'Skip Indexer; keep selected position IDs.']
        d.box('global' if mode!='Window' else 'sliding',35,y+125,395,42,left,'orange' if mode!='Window' else 'green',16,leading=18)
        d.raw('<g class="inline-indexer">')
        start=len(d.S)
        d.box('indexer',1070,y+125,395,42,right,'purple',16,leading=18)
        for j in range(start,len(d.S)):d.S[j]=d.S[j].replace('class="module"','class="inline-indexer-node"')
        d.raw('</g>')
        d.arrow(f'M430 {y+146} H465',width=1.8)
        d.arrow(f'M1070 {y+146} H1035',width=1.8)
        d.arrow(f'M750 {y+108} V{y+125}',width=1.8)
        d.arrow(f'M750 {y+167} V{y+185}',width=1.8)
        if previous is not None:
            d.arrow(f'M750 {previous+216} V{previous+237} H445 V{y+89} H465',color='#3265b0',width=2)
        previous=y
        d.raw('</g>')
    y=ROW_Y+20*ROW_H
    d.arrow(f'M750 {previous+216} V{y-10}',color='#3265b0',width=2)
    d.text(750,y+15,'Continue to Layer 20: same four streams + delayed pre_mix; Layer 20 publishes its own KV / Keys / selection.' if not decoder else
           'After Layer 39: final mHC collapse → RMSNorm → vocabulary Head. DSpark runs only through a separate forward_spec() call.',22,True)
    d.text(750,y+53,'Full replaces the active shared reference; earlier source caches remain allocated. Reindex replaces position IDs, not KV.',20)
    d.text(750,y+86,'Owner labels describe completed compression groups. See the source note for the local incomplete-group Index Key caveat.',18)
