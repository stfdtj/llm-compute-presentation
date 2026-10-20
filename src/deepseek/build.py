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

check()
AUDIT=apply(DATA)
write_docs(AUDIT,DATA)
(ROOT/'src/deepseek/audit_data.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2),encoding='utf-8')
svg=build()
from attention_preservation import validate
validate(svg)
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
template=(ROOT/'src/deepseek/page_template.html').read_text(encoding='utf-8')
result=template.replace('__SVG__',svg).replace('__DATA__',json.dumps(DATA,ensure_ascii=False).replace('</','<\\/'))
audit_ui=(ROOT/'src/deepseek/audit_ui.html').read_text(encoding='utf-8').replace('__AUDIT__',json.dumps(AUDIT,ensure_ascii=False).replace('</','<\\/'))
shape_config=json.loads((ROOT/'references/deepseek/config.json').read_text(encoding='utf-8'))['text_config']
shape_ui=(ROOT/'src/deepseek/flow_ui.html').read_text(encoding='utf-8').replace('__SHAPE_CONFIG__',json.dumps(shape_config)).replace('__SHAPE_SCRIPT__',(ROOT/'src/deepseek/shape_trace.js').read_text(encoding='utf-8'))
result=result.replace('</body>',shape_ui+'\n'+audit_ui+'\n</body>')
indexer_text_config=json.loads((ROOT/'references/deepseek/config.json').read_text(encoding='utf-8'))['text_config']
# Hugging Face config names -> inference/model.py ModelArgs names.
indexer_aliases={'dim':'hidden_size','n_layers':'num_hidden_layers','n_heads':'num_attention_heads','window_size':'sliding_window','kv_source_layers':'kv_source_layer_ids','index_source_layers':'index_source_layer_ids'}
indexer_fields=['dim','n_layers','n_heads','q_lora_rank','head_dim','index_n_heads','index_head_dim','index_topk','window_size','kv_source_layers','index_source_layers','compress_ratios']
indexer_config={key:indexer_text_config[indexer_aliases.get(key,key)] for key in indexer_fields}
indexer_ui=(ROOT/'src/deepseek/indexer_ui.html').read_text(encoding='utf-8').replace('__INDEXER_SVG__',build_indexer(indexer_config)).replace('__INDEXER_CONFIG__',json.dumps({k:indexer_config[k] for k in indexer_fields}))
result=result.replace('</body>',indexer_ui+'\n</body>')
result=result.replace('</body>',(ROOT/'src/deepseek/attention_blocks_ui.html').read_text(encoding='utf-8')+'\n</body>')
result=result.replace('</body>',(ROOT/'src/deepseek/layer_trace_ui.html').read_text(encoding='utf-8')+'\n</body>')
walkthrough_ui=(ROOT/'src/deepseek/walkthrough_ui.html').read_text(encoding='utf-8').replace('__CSA_WALKTHROUGH_CONFIG__',json.dumps(indexer_config))
walkthrough_ui=walkthrough_ui.replace('__CSA_WALKTHROUGH_MOTION__',(ROOT/'src/deepseek/walkthrough_motion.js').read_text(encoding='utf-8'))
result=result.replace('</body>',walkthrough_ui+'\n</body>')
pipeline_ui=(ROOT/'src/deepseek/inference_pipeline.html').read_text(encoding='utf-8').replace('__PIPELINE_SCRIPT__',(ROOT/'src/deepseek/inference_pipeline.js').read_text(encoding='utf-8'))
result=result.replace('</body>',pipeline_ui+'\n</body>')
(ROOT/'deepseek_v4_1_interactive.html').write_text(result,encoding='utf-8')
(ROOT/'src/deepseek/architecture.svg').write_text(svg,encoding='utf-8')
print(f'Built standalone HTML: {len(result.encode("utf-8")):,} bytes; {len(ids)} interactive SVG elements / {len(set(ids))} keys; {len(DATA)} calculation entries.')
