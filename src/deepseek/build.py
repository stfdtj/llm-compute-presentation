"""Build the offline, self-contained deliverable from independent content/layout."""
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
from latex2mathml.converter import convert
from teaching_data import DATA,check
from audited_data import apply,write_docs
from build_diagram import build
from indexer_diagram import build_indexer
from hc_diagram import build_detail
from dspark_diagram import build_dspark
import derivation_ledger

check()
AUDIT=apply(DATA)
write_docs(AUDIT,DATA)
DERIVATIONS=derivation_ledger.apply(DATA,AUDIT)
derivation_ledger.write_docs(DERIVATIONS)
(ROOT/'src/deepseek/derivation_data.json').write_text(json.dumps(DERIVATIONS,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'src/deepseek/audit_data.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2),encoding='utf-8')
svg=build()
root=ET.fromstring(svg)
ids=[e.attrib['data-module'] for e in root.iter() if 'data-module' in e.attrib]
assert all(i in DATA for i in ids)
# Save editable calculation content BEFORE compiling equations to MathML.
(ROOT/'src/deepseek/module_data.json').write_text(json.dumps(DATA,ensure_ascii=False,indent=2),encoding='utf-8')
for d in DATA.values():
    for name in ['parameter','forward','backward']:
        m=d[name]
        m['calculations']=[convert(f,display='block') for f in m['calculations']]
        steps=m['steps']+[s for extra in m['extras'] for s in extra['steps']]
        for s in steps:
            s['formulas']=[convert(f,display='block') for f in s['formulas']]
            s['dimensions']=[convert(f,display='block') for f in s['dimensions']]
for step in DERIVATIONS['steps'].values():
    step['formulas']=[convert(f,display='block') for f in step['formulas']]
    step['dimensions']=[convert(f,display='block') for f in step['dimensions']]
template=(ROOT/'src/deepseek/page_template.html').read_text(encoding='utf-8')
result=template.replace('__SVG__',svg).replace('__DATA__',json.dumps(DATA,ensure_ascii=False).replace('</','<\\/'))
audit_ui=(ROOT/'src/deepseek/audit_ui.html').read_text(encoding='utf-8').replace('__AUDIT__',json.dumps(AUDIT,ensure_ascii=False).replace('</','<\\/'))
result=result.replace('</body>',(ROOT/'src/deepseek/flow_ui.html').read_text(encoding='utf-8')+'\n'+audit_ui+'\n</body>')
result=result.replace('</body>',(ROOT/'src/deepseek/flow_attention_ui.html').read_text(encoding='utf-8')+'\n</body>')
indexer_text_config=json.loads((ROOT/'references/deepseek/config.json').read_text(encoding='utf-8'))['text_config']
# Hugging Face config names -> inference/model.py ModelArgs names.
indexer_aliases={'dim':'hidden_size','n_layers':'num_hidden_layers','n_heads':'num_attention_heads','window_size':'sliding_window','kv_source_layers':'kv_source_layer_ids','index_source_layers':'index_source_layer_ids'}
indexer_fields=['dim','n_layers','n_heads','q_lora_rank','head_dim','index_n_heads','index_head_dim','index_topk','window_size','kv_source_layers','index_source_layers','compress_ratios']
indexer_config={key:indexer_text_config[indexer_aliases.get(key,key)] for key in indexer_fields}
indexer_ui=(ROOT/'src/deepseek/indexer_ui.html').read_text(encoding='utf-8').replace('__INDEXER_SVG__',build_indexer(indexer_config)).replace('__INDEXER_CONFIG__',json.dumps({k:indexer_config[k] for k in indexer_fields}))
result=result.replace('</body>',indexer_ui+'\n</body>')
derivation_ui=(ROOT/'src/deepseek/derivation_ui.html').read_text(encoding='utf-8').replace('__DERIVATIONS__',json.dumps(DERIVATIONS,ensure_ascii=False).replace('</','<\\/'))
result=result.replace('</body>',derivation_ui+'\n</body>')
hc_ui=(ROOT/'src/deepseek/hc_ui.html').read_text(encoding='utf-8').replace('__HC_SVG__',build_detail())
result=result.replace('</body>',hc_ui+'\n</body>')
attention_ui=(ROOT/'src/deepseek/attention_ui.html').read_text(encoding='utf-8').replace('__ATTENTION_CONFIG__',json.dumps(json.loads((ROOT/'references/deepseek/inference/config.json').read_text(encoding='utf-8'))))
result=result.replace('</body>',attention_ui+'\n</body>')
dspark_config=json.loads((ROOT/'references/deepseek/inference/config.json').read_text(encoding='utf-8'))
dspark_ui=(ROOT/'src/deepseek/dspark_ui.html').read_text(encoding='utf-8').replace('__DSPARK_CONFIG__',json.dumps(dspark_config)).replace('__DSPARK_SVG__',build_dspark(dspark_config))
result=result.replace('</body>',dspark_ui+'\n</body>')
(ROOT/'deepseek_v4_1_interactive.html').write_text(result,encoding='utf-8')
(ROOT/'src/deepseek/architecture.svg').write_text(svg,encoding='utf-8')
print(f'Built standalone HTML: {len(result.encode("utf-8")):,} bytes; {len(ids)} interactive SVG elements / {len(set(ids))} keys; {len(DATA)} calculation entries.')
