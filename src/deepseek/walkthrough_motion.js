/* Unified Data Flow + Shape walkthrough. Merges the former Step-by-step CSA2 animation
 * engine with the former Shape Trace per-layer shape graph into one persistent-SVG scene.
 * Navigation changes numeric poses and the structured shape readout, never replaces the
 * stage. One requestAnimationFrame clock owns all motion, including pause/seek.
 *
 * Cross-feature contract preserved from the former shape_trace.js (read by indexer_ui.html,
 * attention_blocks_ui.html, layer_trace_ui.html, dspark_ui.html, audit_ui.html,
 * page_template.html): #flow-toggle button id, window.informationFlow.state
 * {enabled,layer}, #flow-bar container id, #shape-layer <select>, the drawPanel wrap that
 * injects a `.flow-entry` button for modules ['global','hidden','multi','localstates'], and
 * [data-node]/.chosen highlighting. Also preserved from the former walkthrough_motion.js:
 * #csa-tab-architecture / #csa-tab-walkthrough tab ids and the `csa-walkthrough-active`
 * body class (read by page_template.html and dspark_ui.html's CSS). */
(()=>{
 'use strict';
 const $=id=>document.getElementById(id),root=$('flow-bar'),C=JSON.parse($('shape-config').textContent);
 const scrollArea=root.querySelector('.csa-scroll');
 const NS='http://www.w3.org/2000/svg';
 const COLOR={q:'#2863c5',l:'#22936b',g:'#cb7916',s:'#8a4ec2',hc:'#ae7c1d',m:'#8a98ac'};

 // ----------------------------------------------------------------- shape/config layer
 // Ported from the former shape_trace.js: single source of truth for dimensions and the
 // Full/Reindex/Reuse/Local-only mode derivation, reused rather than retyped.
 const D=C.hidden_size,H=C.num_attention_heads,d=C.head_dim,R=C.q_lora_rank,G=C.o_groups,O=C.o_lora_rank,
  F=C.moe_intermediate_size,M=C.hc_mult,E=C.n_routed_experts,J=C.num_experts_per_tok,IH=C.index_n_heads,
  ID=C.index_head_dim,RD=C.qk_rope_head_dim,WIN=C.sliding_window,TOPK=C.index_topk;
 const totalQDim=H*d,mixHc=(2+M)*M;
 const s=(...a)=>'['+a.join(',')+']';
 const nearest=arr=>L=>arr.filter(i=>i<=L).at(-1);
 const kvNearest=nearest(C.kv_source_layer_ids),idxNearest=nearest(C.index_source_layer_ids);
 function layerInfo(L){
  const ratio=C.compress_ratios[L],owns=C.kv_source_layer_ids.includes(L),scores=C.index_source_layer_ids.includes(L);
  const source=kvNearest(L),indexSource=idxNearest(L);
  const mode=!ratio?'Local only':owns?'Full':scores?'Reindex':'Reuse';
  return {L,ratio,owns,scores,source,indexSource,mode};
 }
 const REPRESENTATIVE=[2,3,20,24];

 // ----------------------------------------------------------------- SVG + motion primitives
 // Unchanged from the original CSA2 walkthrough engine: a persistent scene graph whose
 // nodes are repositioned/recolored by `layout()`, never recreated.
 function el(tag,attrs={},parent){const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(parent)parent.append(e);return e;}
 const svg=el('svg',{viewBox:'0 0 1200 680',role:'img','aria-labelledby':'csa-motion-title csa-motion-desc'});
 const svgTitle=el('title',{id:'csa-motion-title'},svg),svgDesc=el('desc',{id:'csa-motion-desc'},svg);
 const defs=el('defs',{},svg);
 for(const [k,c] of Object.entries(COLOR)){const m=el('marker',{id:'csa-m-'+k,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'},defs);el('path',{d:'M0 0L10 5L0 10Z',fill:c},m);}
 $('csa-canvas').append(svg);
 const sceneGroups=[],nodes=new Map();
 function scene(name){const g=el('g',{'data-scene':name},svg);sceneGroups.push(g);return g;}
 function label(parent,x,y,value,size=17,color='#20375c',anchor='start'){const t=el('text',{x,y,'font-size':size,fill:color,'text-anchor':anchor},parent);t.textContent=value;return t;}
 function node(id,parent,content,base={}){const e=el('g',{'data-motion-id':id},parent);content(e);const p={x:0,y:0,sx:1,sy:1,opacity:1,color:COLOR.q,w:200,h:30,...base};nodes.set(id,{el:e,base:p,now:{...p}});return e;}
 function row(id,parent,x,y,color,tag,width=210,cols=6){return node(id,parent,g=>{
  for(let j=0;j<cols;j++)el('rect',{x:j*width/cols,y:0,width:width/cols-2,height:27,rx:2,fill:'currentColor','fill-opacity':.14,stroke:'currentColor','stroke-width':1},g);
  label(g,-10,19,tag,14,'#536580','end');
 },{x,y,color});}
 function box(id,parent,x,y,w,h,color,tag,dashed=false){return node(id,parent,g=>{el('rect',{'data-box':true,width:w,height:h,rx:5,fill:'currentColor','fill-opacity':.09,stroke:'currentColor','stroke-width':2,...(dashed?{'stroke-dasharray':'6 5'}:{})},g);if(tag)label(g,12,22,tag,14);},{x,y,w,h,color});}
 function textNode(id,parent,x,y,value,color=COLOR.q,size=17){return node(id,parent,g=>label(g,0,0,value,size,color),{x,y,color});}
 function wire(id,parent,d,key){return node(id,parent,g=>el('path',{d,fill:'none',stroke:COLOR[key],'stroke-width':2,'marker-end':'url(#csa-m-'+key+')'},g));}
 function dataNode(groupEl,name){groupEl.setAttribute('data-node',name);return groupEl;}
 function paint(p){for(const [id,v]of p){const n=nodes.get(id);n.now={...v};n.el.setAttribute('transform',`translate(${v.x} ${v.y}) scale(${v.sx} ${v.sy})`);n.el.setAttribute('opacity',String(v.opacity));n.el.style.color=v.color;const r=n.el.querySelector('[data-box]');if(r){r.setAttribute('width',v.w);r.setAttribute('height',v.h);}}}
 function mixColor(a,b,t){const c=v=>[1,3,5].map(i=>parseInt(v.slice(i,i+2),16));const aa=c(a),bb=c(b);return '#'+aa.map((n,i)=>Math.round(n+(bb[i]-n)*t).toString(16).padStart(2,'0')).join('');}

 // ----------------------------------------------------------------- stage/beat data model
 // One function builds, for a given layer L, the full ordered list of 8 stages each with
 // its beats -- the structured "input shape / op / weight shape / output shape / next"
 // readout shown beside the animation, and the single source of truth both scenes read
 // their captions from (no per-layer hand-typed Chinese strings).
 const tx=v=>typeof v==='string'?v:JSON.stringify(v);
 function beat(title,desc,shape){return {title,desc,shape};}
 // shape: {input:[label,shape],op,weight:[label,shape]|null,output:[label,shape],next}
 function buildStages(L){
  const info=layerInfo(L);
  const r=s(1,M,D),h=s(1,D),coef=s(1,M),comb=s(1,M,M),qr=s(1,R),q=s(1,H,d);
  const kv=s('ΔS',d),cache=s('S',d),idx=s(1,'K');
  const stages=[];

  // 1. Input Hidden States + mHC Pre
  stages.push({name:'Input + mHC Pre',scene:'mhc',beats:[
   beat('Four-stream Hidden State',L===0?'Token Embedding 展开为 4 路残差流。':`来自 Layer ${L-1} 的输出，4 路残差流。`,
    {input:[L===0?'Token Embedding':'Previous Layer',L===0?s(1):r],op:L===0?'Embedding Lookup · Expand ×4':'Carry-over',weight:null,output:['Hidden State (4 streams)',r],next:'→ mHC Pre'}),
   beat('mHC Pre：4 路 → 1 路',' Pre 系数（上一层 FFN Mix 产生）对 4 路加权求和，收缩成单路送入 Attention。',
    {input:['Hidden State',r],op:'Weighted Reduce (Σ pre_i × stream_i)',weight:['pre mix',coef],output:['Attention Input',h],next:'→ RMSNorm'}),
  ]});

  // 2. RMSNorm + Query Projection -- the user's literal worked example
  stages.push({name:'RMSNorm + Query',scene:'query',beats:[
   beat('RMSNorm','Attention 输入做 RMSNorm，形状不变。',
    {input:['Hidden State',h],op:'RMSNorm',weight:['scale γ',s(D)],output:['Normalized',h],next:'→ W_QA'}),
   beat('W_QA Projection','低秩压缩：5120 → 1280。',
    {input:['Hidden State',h],op:'Projection',weight:['W_QA',s(D,R)],output:['Compressed Query',qr],next:'→ RMSNorm'}),
   beat('RMSNorm + W_QB','归一化后展开：1280 → 32768（64 Head × 512）。',
    {input:['Compressed Query',qr],op:'RMSNorm → Projection',weight:['W_QB',s(R,totalQDim)],output:['Expanded Query',s(1,totalQDim)],next:'→ Reshape'}),
   beat('Reshape + RoPE','展开为 64 个 Head，每个 512 维；尾部 64 维做 RoPE。',
    {input:['Expanded Query',s(1,totalQDim)],op:`Reshape → [${H},${d}] · RoPE tail ${RD}`,weight:null,output:['Query',q],next:'→ Main Attention'}),
  ]});

  // 3. Local KV generation + Sliding Window
  stages.push({name:'Local KV · SWA',scene:'swa',beats:[
   beat('Local KV Projection','当前 token 自己的 KV 投影，每层独立生成。',
    {input:['Hidden State',h],op:'Projection → RMSNorm → RoPE → FP8 Quant',weight:['W_KV',s(D,d)],output:['Local KV (this token)',s(1,d)],next:'→ Ring Cache'}),
   beat('窗口滑到最近 '+WIN+' 个位置','环形缓存写入当前行，读取最近 '+WIN+' 个位置。',
    {input:['Local KV (this token)',s(1,d)],op:'Ring Cache Write + Read',weight:null,output:['Local KV Window',s(WIN,d)],next:'→ Main Attention'}),
  ]});

  // 4. Global KV Compression -- current token vs multi-position history kept visually distinct
  {
   const beats=[];
   if(info.ratio && info.owns){
    if(info.ratio===2){
     beats.push(beat('这个 Token 自己的贡献','当前 token 先各自投影出 KV 候选值和门控分数，还未合并。',
      {input:['Hidden State (this token)',h],op:'Projection (W_KV / W_gate)',weight:['W_KV, W_gate',s(D,d)+' ×2'],output:['Candidate KV + Gate',s(1,d)+' + '+s(1,d)],next:'→ 与上一个 Token 配对'}));
     beats.push(beat('压缩比 2：两个 Token → 一个 Global 位置','本 token 与前一个 token 的候选值按通道 Softmax 门控加权合并；不是简单平均。',
      {input:['2 个 Token 的候选 KV','2×'+s(1,d)],op:'Channel-wise Softmax Gate → Weighted Sum → RMSNorm → RoPE',weight:null,output:['1 个 Global KV 行',s(1,d)],next:'→ Global KV Cache（历史）'}));
    }else{
     beats.push(beat('压缩比 1：每个 Token 直接产出一行','ratio=1 时不等待配对，不做门控池化，直接投影。',
      {input:['Hidden State (this token)',h],op:'Projection → RMSNorm → RoPE',weight:['W_KV',s(D,d)],output:['1 个 Global KV 行',s(1,d)],next:'→ Global KV Cache（历史）'}));
    }
    beats.push(beat('写入 Global KV Cache（多 Token 历史）','这一行追加到跨步维护的历史缓存；缓存本身代表许多历史 Token，和刚才单个 token 的计算是两回事。',
     {input:['新的 1 行',kv],op:'Append (FP4 Quant, 16-channel scale)',weight:null,output:['Global KV Cache（历史）',cache],next:'→ Indexer / Main Attention'}));
   }else if(info.ratio){
    beats.push(beat(`Global KV 复用自 Layer ${info.source}`,'本层不重新压缩；Local KV（上一节）仍是本层自己独立计算的。',
     {input:[`Layer ${info.source} 的 Global KV Cache`,cache],op:'Cross-layer Cache Reuse（本层不执行 Compressor）',weight:null,output:['Global KV Cache（复用）',cache],next:'→ Indexer / Main Attention'}));
   }else{
    beats.push(beat('Local only：本层没有 Global 分支','该层 compress_ratios=0，只有 Local Attention。',
     {input:['—',''],op:'（跳过）',weight:null,output:['—',''],next:'→ Main Attention'}));
   }
   stages.push({name:'Global KV Compression',scene:'compression',beats});
  }

  // 5. Indexer Query, Index Keys, scoring, Top-K
  {
   const beats=[];
   if(!info.ratio){
    beats.push(beat('无 Indexer','Local only 层没有 Global 分支，也就没有 Indexer。',{input:['—',''],op:'（跳过）',weight:null,output:['—',''],next:'→ Main Attention'}));
   }else if(info.scores){
    beats.push(beat('Indexer Query（本层独立计算）','复用 qr（W_QA 输出），不是重新从 Hidden State 投影。',
     {input:['qr（低秩 Query）',qr],op:'Projection → Reshape → RoPE/FP4',weight:['W_Q (Indexer)',s(R,IH*ID)],output:['Indexer Query',s(1,IH,ID)],next:'→ 与 Index Keys 点积'}));
    beats.push(beat(info.owns?'Index Keys（本层新生成）':`Index Keys 复用自 Layer ${info.source}`,
     info.owns?'来自本层 Compressor 的 RoPE 前 latent，经投影、归一化与位置处理。':'与 Global KV 共用同一个压缩分支，本层不重新生成。',
     {input:[info.owns?'压缩 latent (RoPE 前)':`Layer ${info.source} 的 Index Keys`,info.owns?s(1,d):s('S',ID)],op:info.owns?'Projection → RMSNorm → RoPE/FP4':'Cross-layer Cache Reuse',weight:info.owns?['W_K (Indexer)',s(d,ID)]:null,output:['Index Keys Cache',s('S',ID)],next:'→ 打分'}));
    beats.push(beat('打分 + 掩码'+(L>C.candidate_source_layer_id&&!info.owns?'（候选池内）':''),
     L>C.candidate_source_layer_id&&!info.owns?`只在 Layer ${C.candidate_source_layer_id} 构建的候选池（最多 ${C.candidate_topk_blocks}×${C.candidate_block_size} 个位置）内评分。`:'对全部可见的压缩位置评分。',
     {input:['Query · Keys',s(1,IH,ID)+' , '+s('S',ID)],op:'Dot Product → ReLU → Head 加权汇总 → 因果/候选掩码',weight:['Head Weights',s(D,IH)],output:['Score Vector',s(1,'S')],next:'→ Top-K'}));
    beats.push(beat(`Top-K 选择（上限 ${TOPK}）`,'按分数排序，取索引；输出是位置编号，不是 KV 本身。',
     {input:['Score Vector',s(1,'S')],op:'Top-K → 排序 → +offset',weight:null,output:['Top-K 位置索引',idx],next:'→ Main Attention'}));
   }else{
    beats.push(beat(`Top-K 索引复用自 Layer ${info.indexSource}`,'跳过 Indexer Query / 打分；直接复用最近一次的选择结果。',
     {input:[`Layer ${info.indexSource} 的 Top-K`,idx],op:'Cross-layer Reuse（不计算 Indexer Query，不重新打分）',weight:null,output:['Top-K 位置索引（复用）',idx],next:'→ Main Attention'}));
   }
   stages.push({name:'Indexer · Top-K',scene:'indexer',beats});
  }

  // 6. Main Attention using Local and selected Global KV
  stages.push({name:'Main Attention',scene:'attention',beats:[
   beat('聚合 Local + 选中的 Global KV','按位置维拼接成一次读取；索引数组同样拼接。',
    {input:['Local KV, 选中 Global KV',s(WIN,d)+' , ≤'+TOPK+'×'+s(1,d)],op:'Gather（逻辑读取，非额外拷贝）',weight:null,output:['KV Read View',s(`≤${WIN}+${TOPK}`,d)],next:'→ QKᵀ'}),
   beat('QKᵀ + 联合 Softmax','Local 与 Global 共用一次 Softmax；Attention Sink 也参与分母。',
    {input:['Query, KV Read View',q+' , '+s(`≤${WIN}+${TOPK}`,d)],op:'Dot Product / √'+d+' → Softmax + Sink',weight:null,output:['Attention Weights',s(1,H,'N')],next:'→ 加权求和'}),
   beat('加权求和','各 Head 独立计算，得到 Attention Output。',
    {input:['Weights, Values',s(1,H,'N')+' , '+s('N',d)],op:'Weighted Sum',weight:null,output:['Attention Output',q],next:'→ 输出投影'}),
  ]});

  // 7. Output Projection + mHC update
  stages.push({name:'Output Proj + mHC',scene:'outproj',beats:[
   beat('逆 RoPE + 分组','按 '+G+' 组重排，为分组低秩投影做准备。',
    {input:['Attention Output',q],op:'Inverse RoPE → Reshape → FP8 Quant',weight:null,output:['Grouped Output',s(1,G,totalQDim/G)],next:'→ 分组 W_OA'}),
   beat('分组 W_OA + Flatten + W_OB','每组独立投影（block-diagonal），再展平、合并回 5120 维。',
    {input:['Grouped Output',s(1,G,totalQDim/G)],op:'Grouped Projection → Flatten → Projection',weight:['W_OA ('+G+'×), W_OB',s(totalQDim/G,O)+' ×'+G+' , '+s(G*O,D)],output:['Attention 子层输出',h],next:'→ mHC Post'}),
   beat('mHC Post：回到 4 路','Post / Comb 系数把子层输出和旧的 4 路残差混合，写回 4 路。',
    {input:['子层输出, 旧 4 路残差',h+' , '+r],op:'Broadcast Write (post × output + comb × residual)',weight:['post, comb',coef+' , '+comb],output:['Hidden State (4 streams)',r],next:'→ FFN Pre'}),
  ]});

  // 8. MoE Expert calculation + final hidden state
  stages.push({name:'MoE + Final State',scene:'moe',beats:[
   beat('mHC Pre (FFN) + RMSNorm','本层 Attention Mix 产生的 pre 系数，收缩 4 路到 1 路。',
    {input:['Hidden State',r],op:'Weighted Reduce → RMSNorm',weight:['pre mix',coef],output:['FFN 输入',h],next:'→ Router'}),
   beat(`Router：选 ${J} / ${E} 个专家`,'打分后选择 Top-'+J+'，并取未加偏置的原始分数作为权重。',
    {input:['FFN 输入',h],op:'Projection (sqrtsoftplus) → Top-K Select',weight:['Router',s(D,E)],output:[`${J} 个专家 ID + 权重`,s(1,J)+' + '+s(1,J)],next:'→ Expert FFN'}),
   beat(`Expert FFN (SwiGLU) × ${J} + 1 Shared`,'每个选中专家各自跑一次 Gate/Up/Down；权重在 Down 之前相乘；Shared Expert 对所有 token 都执行。',
    {input:['FFN 输入',h],op:'Gate/Up Projection → SiLU × Up → Down Projection',weight:['Gate, Up, Down',s(D,F)+' ×2 , '+s(F,D)],output:[`${J} 路 routed + 1 路 shared`,h+' ×'+(J+1)],next:'→ 加权求和'}),
   beat('求和 → mHC Post (FFN)','routed 与 shared 输出相加，再混回 4 路，作为下一层输入。',
    {input:[`${J+1} 路专家输出`,h],op:'Sum → Broadcast Write (post × sum + comb × residual)',weight:['post, comb',coef+' , '+comb],output:[L<C.num_hidden_layers-1?`Layer ${L+1} 输入`:'Final Hidden State',r],next:L<C.num_hidden_layers-1?`→ Layer ${L+1}`:'→ Final RMSNorm → LM Head'}),
  ]});

  return {info,stages};
 }

 // ----------------------------------------------------------------- scene builders
 // Each scene is built once (persistent SVG), then re-laid-out per beat via layout().
 // SWA, Compression, Indexer and Attention keep the original CSA2 choreography,
 // generalized to read layer-aware captions instead of hardcoded Encoder-L2 text.

 // 1. mHC (four streams collapse to one, and the mirrored expand-back-to-four in stage 7)
 const mhc=scene('mhc');
 label(mhc,90,62,'Hidden State · 4 streams',21,COLOR.hc);label(mhc,90,89,'每路 [1,5120]',13,COLOR.hc);
 for(let i=0;i<4;i++)dataNode(row('mhc-stream-'+i,mhc,90,140+i*50,COLOR.hc,'h⁽'+(i+1)+'⁾',260),'hidden-stream-'+i);
 box('mhc-pre-box',mhc,470,170,220,180,COLOR.hc,'mHC Pre');label(mhc,580,420,'pre 系数（上一层产生）',14,COLOR.m,'middle');
 const preARow=dataNode(row('mhc-pre-out',mhc,800,255,COLOR.q,'h',270),'preA');
 textNode('mhc-pre-dim',mhc,800,330,'Attention Input [1,5120]',COLOR.q,18);
 wire('mhc-pre-arrow',mhc,'M360 255 H460','hc');wire('mhc-pre-arrow2',mhc,'M700 255 H790','q');

 // 2. Query Projection (new scene, matches the user's literal worked example).
 // Single top-to-bottom column, every wire a straight vertical drop with one clear
 // source and one clear destination -- no branch rejoins the chain out of order.
 const qscene=scene('query');
 label(qscene,170,48,'Hidden State [1,5120]',21,COLOR.q);
 const qIn=row('q-in',qscene,170,75,COLOR.q,'h',260);
 box('q-norm1',qscene,170,122,260,55,COLOR.q,'RMSNorm (Attention Input)');
 box('q-wqa',qscene,170,197,260,68,COLOR.q,'W_QA [5120,1280]');
 const qMid=row('q-mid',qscene,170,285,COLOR.q,'qr',260);textNode('q-mid-dim',qscene,170,332,'Intermediate [1,1280]',COLOR.q,15);
 box('q-norm2',qscene,170,360,260,55,COLOR.q,'RMSNorm (Query)');
 box('q-wqb',qscene,170,435,260,68,COLOR.q,'W_QB [1280,32768]');
 const qOut=row('q-out',qscene,170,523,COLOR.s,'Q',260);textNode('q-out-dim',qscene,170,570,'[1,32768]',COLOR.s,15);
 const qReshape=dataNode(node('q-reshape',qscene,g=>{for(let i=0;i<8;i++)el('rect',{x:i*32,y:0,width:28,height:28,rx:3,fill:'currentColor','fill-opacity':.2,stroke:'currentColor'},g);label(g,0,48,'Query [1,64,512]',17,COLOR.s);},{x:170,y:600,color:COLOR.s}),'query-out');
 wire('q-w1',qscene,'M300 102 V122','q');wire('q-w2',qscene,'M300 177 V197','q');wire('q-w3',qscene,'M300 265 V285','q');
 wire('q-w4',qscene,'M300 340 V360','q');wire('q-w5',qscene,'M300 415 V435','q');wire('q-w6',qscene,'M300 503 V523','s');
 wire('q-w7',qscene,'M300 578 V600','s');

 // 3. Local KV + SWA (generalized from the original SWA scene)
 const swa=scene('swa'),positions=[0,3968,3969,4000,4050,4094,4095,4096];
 label(swa,70,60,'Hidden State → Local KV',21,COLOR.l);
 box('swa-wkv',swa,70,105,210,65,COLOR.l,'W_KV');
 const swaKvOut=row('swa-kv-out',swa,340,120,COLOR.l,'kv',210);textNode('swa-kv-dim',swa,340,180,'[1,512] · 当前 token',COLOR.l,15);
 label(swa,700,60,'Ring Cache · 最近 '+WIN+' 个位置',19,COLOR.l);
 for(let i=0;i<8;i++)row('swa-k-'+i,swa,700,110+i*34,COLOR.l,'t'+positions[i],320);
 const swaWindow=dataNode(box('swa-window',swa,690,108,330,276,COLOR.l),'local');
 textNode('swa-window-label',swa,700,418,'t3969 … t4096 · 共 '+WIN+' 个位置',COLOR.l,16);
 wire('swa-w1',swa,'M280 137 H335','l');wire('swa-w2',swa,'M450 150 H660','l');
 textNode('swa-formula',swa,70,480,'本层独立：每层都有自己的 W_KV、自己的环形缓存。',COLOR.m,15);

 // 4. Global KV Compression (generalized: shows fresh-compute OR reused-from-layer state)
 const comp=scene('compression');
 textNode('comp-this-title',comp,80,55,'这个 Token 自己的贡献',COLOR.q,20);
 const compThis=row('comp-this',comp,80,100,COLOR.q,'t',220);
 const compPrev=row('comp-prev',comp,80,145,COLOR.m,'t-1',220);
 box('comp-machine',comp,370,90,300,260,COLOR.g,'Compressor');
 node('comp-fresh-op',comp,g=>{label(g,0,0,'W_KV / W_gate',16,COLOR.g,'middle');label(g,0,35,'Softmax 门控 → 加权合并',14,COLOR.m,'middle');label(g,0,70,'RMSNorm → RoPE',15,COLOR.g,'middle');},{x:520,y:145,color:COLOR.g});
 const compOut=row('comp-out',comp,730,170,COLOR.g,'g',220);
 textNode('comp-cache-title',comp,730,60,'Global KV Cache（多 Token 历史）',COLOR.g,19);
 for(let i=0;i<4;i++)row('comp-hist-'+i,comp,730,245+i*35,COLOR.g,'g'+i,220);
 const compCache=dataNode(box('comp-cache',comp,720,235,240,165,COLOR.g),'global');
 textNode('comp-cache-dim',comp,730,423,'缓存 [S,512]，S 随步数累积增长',COLOR.g,15);
 wire('comp-w1',comp,'M300 110 H365','q');wire('comp-w1b',comp,'M300 158 H365','m');wire('comp-w2',comp,'M670 170 H720','g');
 const compReused=box('comp-reused',comp,370,150,330,120,COLOR.g,'',true);
 node('comp-reused-label',comp,g=>label(g,0,0,'复用自其它 Layer',18,COLOR.g,'middle'),{x:535,y:215,color:COLOR.g});
 textNode('comp-reused-note',comp,80,460,'',COLOR.m,15);

 // 5. Indexer (generalized: fresh Query/Keys/scoring, or inline reused-from-layer state)
 const idx=scene('indexer'),scores=[.28,.92,.43,.81],rank=[3,0,2,1],chosen=[1,3];
 textNode('idx-q-title',idx,65,55,'Indexer Query',COLOR.q,20);
 const idxQ=row('idx-q',idx,65,100,COLOR.q,'q',200);
 textNode('idx-key-title',idx,420,55,'Index Keys',COLOR.g,20);
 for(let i=0;i<4;i++)row('idx-key-'+i,idx,420,100+i*50,COLOR.g,'#'+i,200);
 const idxKeyBox=dataNode(box('idx-key-box',idx,410,90,220,215,COLOR.g,'',true),'indexkeys');
 textNode('idx-score-title',idx,770,55,'Score Vector',COLOR.s,20);
 for(let i=0;i<4;i++)node('idx-score-'+i,idx,g=>{el('rect',{width:205,height:38,rx:5,fill:'#f2eafa',stroke:COLOR.s},g);el('rect',{x:1,y:30,width:scores[i]*203,height:6,rx:2,fill:COLOR.s},g);label(g,12,24,'#'+i+'   '+scores[i].toFixed(2),17,COLOR.s);},{x:770,y:100+i*50,color:COLOR.s});
 box('idx-dot-area',idx,55,330,290,95,COLOR.q,'');textNode('idx-dot-formula',idx,145,365,'q · kᵀ',COLOR.q,20);textNode('idx-dot-footer',idx,80,405,'→ ReLU → Head 加权',COLOR.q,15);
 box('idx-selection',idx,757,330,235,100,COLOR.s,'',true);
 for(let i=0;i<2;i++)node('idx-pos-'+i,idx,g=>{el('rect',{width:95,height:38,rx:6,fill:'#f2eafa',stroke:COLOR.s},g);label(g,47,25,'#'+chosen[i],19,COLOR.s,'middle');},{x:780+i*105,y:362,color:COLOR.s});
 const idxTop=dataNode(textNode('idx-topk',idx,770,460,'Top-K 位置索引',COLOR.s,17),'topk');
 const idxReused=box('idx-reused',idx,55,90,950,215,COLOR.g,'',true);
 node('idx-reused-label',idx,g=>label(g,0,0,'Top-K 复用自其它 Layer',19,COLOR.g,'middle'),{x:530,y:200,color:COLOR.g});
 textNode('idx-candidate',idx,55,500,'',COLOR.m,13);
 wire('idx-w1',idx,'M165 127 V325','q');wire('idx-w2',idx,'M520 305 V335 H345','g');
 wire('idx-w3',idx,'M345 377 H700 V175 H765','s');wire('idx-w4',idx,'M870 288 V325','s');

 // 6. Main Attention (generalized gather/score/combine)
 const attn=scene('attention'),weights=[.08,.09,.07,.12,.06,.10,.20,.18],logits=weights.map(w=>Math.log(w/.10));
 label(attn,65,55,'Local + Global KV',20,COLOR.l);
 for(let i=0;i<4;i++)row('attn-local-'+i,attn,65,100+i*34,COLOR.l,'L'+i,180);
 for(let i=0;i<4;i++)row('attn-global-'+i,attn,65,250+i*34,chosen.includes(i)?COLOR.s:COLOR.g,'G'+i,180);
 label(attn,330,55,'KV Read View',19,COLOR.s);
 const attnRead=dataNode(box('attn-read',attn,320,90,200,300,COLOR.s,''),'multi');
 textNode('attn-read-dim',attn,420,410,'≤128+512 行',COLOR.s,14,'middle');
 label(attn,620,55,'Query',19,COLOR.q);
 const attnQ=row('attn-q',attn,620,100,COLOR.q,'q',200);
 label(attn,620,160,'Scores / Weights',18,COLOR.s);
 for(let i=0;i<8;i++)node('attn-score-'+i,attn,g=>{el('rect',{'data-box':true,width:220,height:24,rx:3,fill:'#eee6fa',stroke:COLOR.s},g);const t=label(g,9,17,logits[i].toFixed(2),13,COLOR.s);t.dataset.scoreText=String(i);},{x:620,y:205+i*30,color:COLOR.s,w:220,h:24});
 textNode('attn-dot',attn,620,460,'QKᵀ/√512 → Softmax + Sink',COLOR.s,16);
 const attnOut=dataNode(row('attn-output',attn,900,250,COLOR.s,'O',220),'attention');
 textNode('attn-output-dim',attn,900,320,'Attention Output [1,64,512]',COLOR.s,16);
 wire('attn-w1',attn,'M250 165 H315','l');wire('attn-w1b',attn,'M250 314 H315','g');
 wire('attn-w2',attn,'M520 240 H610','s');wire('attn-w3',attn,'M720 127 V200','q');wire('attn-w4',attn,'M850 322 H895 V264','s');

 // 7. Output Projection + mHC Post
 const outp=scene('outproj');
 label(outp,65,55,'Attention Output [1,64,512]',20,COLOR.s);
 const outIn=row('out-in',outp,65,100,COLOR.s,'O',240);
 box('out-group',outp,65,170,240,70,COLOR.s,'Inverse RoPE · Reshape');
 const outGrouped=node('out-grouped',outp,g=>{for(let i=0;i<8;i++)el('rect',{x:i*30,y:0,width:27,height:27,rx:2,fill:'currentColor','fill-opacity':.2,stroke:'currentColor'},g);label(g,0,-10,'[1,8,4096]',15,COLOR.s);},{x:65,y:280,color:COLOR.s});
 box('out-woa',outp,65,340,350,70,COLOR.s,'分组 W_OA （8×，block-diagonal）');
 const outFlat=row('out-flat',outp,65,450,COLOR.s,'flat',350);textNode('out-flat-dim',outp,65,500,'[1,8192]',COLOR.s,15);
 box('out-wob',outp,480,420,260,70,COLOR.s,'W_OB [8192,5120]');
 const outFinal=dataNode(row('out-final',outp,820,340,COLOR.hc,'h',230),'postA');
 textNode('out-final-dim',outp,820,300,'[1,5120] → mHC Post',COLOR.hc,15);
 for(let i=0;i<4;i++)row('out-stream-'+i,outp,820,400+i*34,COLOR.hc,'h⁽'+(i+1)+'⁾',230);
 textNode('out-stream-dim',outp,820,560,'回到 4 路 [1,4,5120]',COLOR.hc,15);
 wire('out-w1',outp,'M185 140 V168','s');wire('out-w2',outp,'M185 310 V338','s');wire('out-w3',outp,'M415 463 H475','s');
 wire('out-w4',outp,'M740 455 H780 V353 H815','s');wire('out-w5',outp,'M935 375 H1005 V415','hc');

 // 8. MoE + final hidden state
 const moe=scene('moe');
 label(moe,65,55,'FFN 输入 [1,5120]',20,COLOR.hc);
 const moeIn=row('moe-in',moe,65,100,COLOR.hc,'h',200);
 box('moe-router',moe,65,160,200,65,COLOR.s,'Router');
 const moeScores=row('moe-scores',moe,320,175,COLOR.s,'',260,8);textNode('moe-scores-tag',moe,320,165,'Scores',COLOR.s,13);textNode('moe-scores-dim',moe,320,230,`[1,${E}]`,COLOR.s,14);
 box('moe-topk',moe,320,250,260,55,COLOR.s,`Top-${J} Select`);
 const moeIds=row('moe-ids',moe,640,175,COLOR.s,'',160,6);textNode('moe-ids-tag',moe,640,165,'IDs + Weights',COLOR.s,13);textNode('moe-ids-dim',moe,640,230,`[1,${J}]×2`,COLOR.s,13);
 label(moe,65,340,`Expert FFN (SwiGLU) × ${J} + 1 Shared`,18,COLOR.s);
 for(let i=0;i<3;i++)box('moe-expert-'+i,moe,65+i*210,375,190,110,COLOR.s,'Expert '+(i<2?i+1:'…'));
 label(moe,65,375+130,'Gate/Up [5120,2304] · Down [2304,5120]',13,COLOR.m);
 const moeSum=dataNode(row('moe-sum',moe,65,530,COLOR.hc,'Σ',260),'moe');
 textNode('moe-sum-dim',moe,65,585,`${J}+1 路求和 → [1,5120]`,COLOR.hc,15);
 for(let i=0;i<4;i++)row('moe-final-'+i,moe,500,520+i*34,COLOR.hc,'h⁽'+(i+1)+'⁾',230);
 textNode('moe-final-dim',moe,500,500,'mHC Post (FFN) → [1,4,5120]',COLOR.hc,15);
 wire('moe-w1',moe,'M265 192 H315','hc');wire('moe-w2',moe,'M580 202 H635','s');
 wire('moe-w3',moe,'M700 215 H600 V375','s');wire('moe-w4',moe,'M370 485 V530','s');
 wire('moe-w5',moe,'M325 557 H495 V533','hc');

 // ----------------------------------------------------------------- layout: numeric poses
 const chosenKV=[1,3];
 function layout(L,sIdx,bIdx){
  const {info,stages}=buildStages(L);
  const p=new Map([...nodes].map(([id,n])=>[id,{...n.base}]));
  const set=(id,props)=>{if(p.has(id))Object.assign(p.get(id),props);},show=(id,on)=>set(id,{opacity:on?1:0});
  const name=stages[sIdx]?stages[sIdx].scene:'mhc';
  // Every node defaults hidden; only the active scene's relevant nodes are shown below,
  // matching the original engine's one-chapter-visible-at-a-time model but per stage.
  for(const n of nodes.values())if(n.el.closest('[data-scene]')?.getAttribute('data-scene')!==name)p.get(n.el.closest('[data-motion-id]')?.getAttribute('data-motion-id'))&&set(n.el.dataset.motionId,{opacity:0});

  if(name==='mhc'){
   for(let i=0;i<4;i++)show('mhc-stream-'+i,true);show('mhc-pre-box',true);show('mhc-pre-out',bIdx>=1);show('mhc-pre-dim',bIdx>=1);show('mhc-pre-arrow',bIdx>=1);show('mhc-pre-arrow2',bIdx>=1);
  }
  if(name==='query'){
   // Strictly top-to-bottom reveal: each beat lights up exactly the next box + the one
   // wire that feeds it, so the chain never shows a connection without its source yet.
   const b=bIdx;
   show('q-in',true);show('q-w1',b>=0);show('q-norm1',b>=0);
   show('q-w2',b>=1);show('q-wqa',b>=1);show('q-w3',b>=1);show('q-mid',b>=1);show('q-mid-dim',b>=1);
   show('q-w4',b>=2);show('q-norm2',b>=2);show('q-w5',b>=2);show('q-wqb',b>=2);show('q-w6',b>=2);show('q-out',b>=2);show('q-out-dim',b>=2);
   show('q-w7',b>=3);show('q-reshape',b>=3);
  }
  if(name==='swa'){
   const b=bIdx;
   show('swa-wkv',true);show('swa-kv-out',b>=0);show('swa-kv-dim',b>=0);show('swa-w1',b>=0);
   for(let i=0;i<8;i++)show('swa-k-'+i,b>=1);show('swa-window',b>=1);show('swa-window-label',b>=1);show('swa-w2',b>=0);show('swa-formula',true);
  }
  if(name==='compression'){
   const b=bIdx,fresh=info.ratio&&info.owns,reused=info.ratio&&!info.owns;
   show('comp-this-title',fresh);show('comp-this',fresh);show('comp-prev',fresh&&info.ratio===2);show('comp-machine',fresh);show('comp-fresh-op',fresh);show('comp-w1',fresh);show('comp-w1b',fresh&&info.ratio===2);
   show('comp-out',fresh&&b>=1);show('comp-w2',fresh&&b>=1);
   for(let i=0;i<4;i++)show('comp-hist-'+i,fresh&&b>=2&&i<2);
   show('comp-cache-title',fresh||reused);show('comp-cache',(fresh&&b>=2)||reused);show('comp-cache-dim',(fresh&&b>=2)||reused);
   show('comp-reused',reused);show('comp-reused-label',reused);show('comp-reused-note',reused);
  }
  if(name==='indexer'){
   // Every node's visibility is set unconditionally in every branch (no implicit
   // "leave at default opacity 1") -- otherwise the fresh-compute visuals stay stuck
   // on screen, overlapping the reused-state box, whenever a layer is Reuse/Local-only.
   const b=bIdx,fresh=!!(info.ratio&&info.scores),reused=!!(info.ratio&&!info.scores);
   show('idx-q-title',fresh);show('idx-q',fresh&&b>=0);
   show('idx-key-title',fresh&&b>=1);for(let i=0;i<4;i++)show('idx-key-'+i,fresh&&b>=1);show('idx-key-box',fresh&&b>=1);
   show('idx-w1',fresh&&b>=2);show('idx-w2',fresh&&b>=2);show('idx-dot-area',fresh&&b>=2);show('idx-dot-formula',fresh&&b>=2);show('idx-dot-footer',fresh&&b>=2);
   show('idx-score-title',fresh&&b>=2);
   for(let i=0;i<4;i++)set('idx-score-'+i,{opacity:fresh&&b>=2?1:0,x:fresh&&b>=3&&chosen.includes(i)?780:770,y:100+(fresh&&b>=3?rank[i]:i)*50});
   show('idx-w3',fresh&&b>=2);
   show('idx-w4',fresh&&b>=3);show('idx-selection',fresh&&b>=3);for(let i=0;i<2;i++)show('idx-pos-'+i,fresh&&b>=3);
   show('idx-topk',(fresh&&b>=3)||reused);
   show('idx-reused',reused);show('idx-reused-label',reused);
  }
  if(name==='attention'){
   const b=bIdx;
   for(let i=0;i<4;i++)show('attn-local-'+i,b>=0);for(let i=0;i<4;i++)show('attn-global-'+i,b>=0);
   show('attn-read',b>=0);show('attn-read-dim',b>=0);show('attn-w1',b>=0);show('attn-w1b',b>=0);
   show('attn-q',true);show('attn-w2',b>=1);show('attn-w3',b>=1);
   for(let i=0;i<8;i++)set('attn-score-'+i,{opacity:b>=1?1:0,w:b>=2?220*(.28+weights[i]*3):220});
   show('attn-dot',b>=1);
   show('attn-w4',b>=2);show('attn-output',b>=2);show('attn-output-dim',b>=2);
  }
  if(name==='outproj'){
   const b=bIdx;
   show('out-in',true);show('out-group',b>=0);show('out-grouped',b>=0);show('out-w1',b>=0);
   show('out-woa',b>=1);show('out-flat',b>=1);show('out-flat-dim',b>=1);show('out-wob',b>=1);show('out-w2',b>=0);show('out-w3',b>=1);show('out-w4',b>=1);
   show('out-final',b>=2);show('out-final-dim',b>=2);for(let i=0;i<4;i++)show('out-stream-'+i,b>=2);show('out-stream-dim',b>=2);show('out-w5',b>=2);
  }
  if(name==='moe'){
   const b=bIdx;
   show('moe-in',true);show('moe-router',b>=1);show('moe-w1',b>=1);
   show('moe-scores',b>=1);show('moe-scores-tag',b>=1);show('moe-scores-dim',b>=1);show('moe-topk',b>=1);show('moe-w2',b>=1);
   show('moe-ids',b>=1);show('moe-ids-tag',b>=1);show('moe-ids-dim',b>=1);show('moe-w3',b>=1);
   for(let i=0;i<3;i++)show('moe-expert-'+i,b>=2);
   show('moe-sum',b>=3);show('moe-sum-dim',b>=3);show('moe-w4',b>=2);show('moe-w5',b>=3);
   for(let i=0;i<4;i++)show('moe-final-'+i,b>=3);show('moe-final-dim',b>=3);
  }
  for(const [id,v]of p)if(nodes.get(id).el.closest('[data-scene]')?.getAttribute('data-scene')!==name)v.opacity=0;
  return p;
 }

 // ----------------------------------------------------------------- navigation state
 let layer=2,sIdx=0,bIdx=0,presentation=false;
 let running=false,auto=false,raf=null,hold=null,elapsed=0,last=null,phase=1,speed=1;
 const duration=1200;
 let from=null,to=null,current=null;
 function totalBeats(){return current.stages.reduce((n,st)=>n+st.beats.length,0);}
 function flatIndex(){let n=0;for(let i=0;i<sIdx;i++)n+=current.stages[i].beats.length;return n+bIdx;}
 function fromFlat(n){for(let i=0;i<current.stages.length;i++){if(n<current.stages[i].beats.length)return [i,n];n-=current.stages[i].beats.length;}return [current.stages.length-1,current.stages.at(-1).beats.length-1];}

 function interpolate(t){
  if(t>=1){paint(to);return;}
  const e=t*t*(3-2*t),p=new Map();
  for(const [id,b]of to){const a=from.get(id)||b;const v={};for(const k of Object.keys(b))v[k]=k==='color'?mixColor(a[k],b[k],e):a[k]+(b[k]-a[k])*e;p.set(id,v);}
  paint(p);
 }
 function clearClock(){if(raf!==null)cancelAnimationFrame(raf);if(hold!==null)clearTimeout(hold);raf=hold=null;last=null;}
 function pause(){clearClock();running=false;auto=false;controls();}
 function controls(){
  const busy=running||auto;$('csa-play').textContent=busy?'Ⅱ Pause':'▶ Play';$('csa-play').setAttribute('aria-pressed',String(busy));
  $('csa-progress').value=String(Math.round(phase*100));$('csa-progress').setAttribute('aria-valuetext',Math.round(phase*100)+'%');
  $('csa-motion-status').textContent=running?'矩阵运动中':phase<1?'已暂停 · 可继续或拖动进度':'等待下一步';
 }
 function tick(now){
  if(!running)return;if(last!==null)elapsed+=(now-last)*speed;last=now;phase=Math.min(1,elapsed/duration);interpolate(phase);controls();
  if(phase<1)raf=requestAnimationFrame(tick);
  else{raf=null;last=null;running=false;
   if(auto){const n=flatIndex();if(n<totalBeats()-1)hold=setTimeout(()=>{hold=null;moveFlat(n+1,true);},700/speed);else auto=false;}
   controls();
  }
 }
 function start(){running=true;last=null;raf=requestAnimationFrame(tick);controls();}

 function renderReadout(){
  const stage=current.stages[sIdx],b=stage.beats[bIdx],sh=b.shape;
  const line=(lab,val)=>`<div class="csa-readout-row"><span class="csa-readout-k">${lab}</span><span class="csa-readout-v">${val}</span></div>`;
  let html=`<div class="csa-readout-step">${esc(b.title)}</div><p class="csa-readout-desc">${esc(b.desc)}</p>`;
  html+=line('Input',`${esc(sh.input[0])} <code>${esc(sh.input[1]||'')}</code>`);
  html+=line('Op',esc(sh.op));
  if(sh.weight)html+=line('Weight',`${esc(sh.weight[0])} <code>${esc(sh.weight[1])}</code>`);
  html+=line('Output',`${esc(sh.output[0])} <code>${esc(sh.output[1])}</code>`);
  html+=line('Next',esc(sh.next));
  $('csa-readout').innerHTML=html;
 }
 function esc(v){return String(v).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

 function metadata(){
  const stage=current.stages[sIdx],b=stage.beats[bIdx],info=current.info;
  $('csa-title').textContent=stage.name+' — '+b.title;$('csa-description').textContent=b.desc;
  $('csa-indicator').textContent=`Stage ${sIdx+1}/${current.stages.length} · Beat ${bIdx+1}/${stage.beats.length}`;
  $('csa-context').textContent=`Layer ${layer} · ${info.mode}`+(info.mode!=='Local only'&&info.mode!=='Full'?` (源 L${info.source}${info.scores?', 索引 L'+info.indexSource:''})`:info.ratio?` · ratio=${info.ratio}`:'');
  svgTitle.textContent=b.title;svgDesc.textContent=b.desc;
  $('csa-prev').disabled=sIdx===0&&bIdx===0;$('csa-next').disabled=sIdx===current.stages.length-1&&bIdx===stage.beats.length-1;
  root.querySelectorAll('[data-csa-stage]').forEach((btn,i)=>{if(i===sIdx)btn.setAttribute('aria-current','step');else btn.removeAttribute('aria-current');btn.classList.toggle('csa-completed',i<sIdx);});
  root.querySelectorAll('[data-csa-layer]').forEach(btn=>btn.setAttribute('aria-pressed',String(Number(btn.dataset.csaLayer)===layer)));
  $('shape-layer').value=String(layer);
  renderReadout();
  for(let i=0;i<8;i++){const n=nodes.get('attn-score-'+i);if(n)n.el.querySelector('text').textContent=sIdx===5&&bIdx>=2?weights[i].toFixed(2):logits[i].toFixed(2);}
 }

 function moveFlat(target,keepAuto=false){
  clearClock();running=false;auto=keepAuto;
  const n=Math.max(0,Math.min(totalBeats()-1,target));const [newS,newB]=fromFlat(n);
  const stageChanged=newS!==sIdx;sIdx=newS;bIdx=newB;
  sceneGroups.forEach(g=>{g.style.display=g.getAttribute('data-scene')===current.stages[sIdx].scene?'':'none';});
  if(stageChanged||!current.__laid){const base=layout(layer,sIdx,0);paint(base);current.__laid=true;}
  from=new Map([...nodes].map(([id,nd])=>[id,{...nd.now}]));to=layout(layer,sIdx,bIdx);elapsed=0;phase=0;metadata();
  if(stageChanged){phase=1;elapsed=duration;paint(to);if(auto&&flatIndex()<totalBeats()-1)hold=setTimeout(()=>moveFlat(flatIndex()+1,true),700/speed);else auto=false;controls();}
  else start();
 }
 function moveStage(target){moveFlat((()=>{let n=0;for(let i=0;i<Math.max(0,Math.min(current.stages.length-1,target));i++)n+=current.stages[i].beats.length;return n;})());}
 function play(){if(running||auto){pause();return;}auto=true;if(phase<1)start();else if(flatIndex()===totalBeats()-1)moveFlat(0,true);else moveFlat(flatIndex()+1,true);}

 function setLayer(L,keepStage=false){
  layer=L;current=buildStages(L);current.__laid=false;
  const target=keepStage?flatIndex():0;
  sIdx=0;bIdx=0;
  sceneGroups.forEach(g=>{g.style.display=g.getAttribute('data-scene')===current.stages[0].scene?'':'none';});
  paint(layout(layer,0,0));current.__laid=true;
  from=new Map([...nodes].map(([id,nd])=>[id,{...nd.now}]));to=layout(layer,0,0);phase=1;elapsed=duration;paint(to);
  metadata();controls();
 }

 // ----------------------------------------------------------------- [data-node] highlighting
 function highlight(id){
  svg.querySelectorAll('[data-node]').forEach(n=>n.classList.toggle('chosen',n.dataset.node===id));
 }
 // Where a cross-feature jump (the .flow-entry button) should land: which stage index
 // contains the target node, and which layer to pre-select -- unchanged mapping from the
 // former shape_trace.js (key -> layer), node ids kept identical for minimal risk.
 const JUMP={hidden:{layer:0,stage:0,node:'preA'},global:{layer:20,stage:3,node:'global'},multi:{layer:20,stage:5,node:'multi'},localstates:{layer:20,stage:2,node:'local'}};
 function focusNode(key){
  const j=JUMP[key];if(!j)return;
  setLayer(j.layer);moveStage(j.stage);
  requestAnimationFrame(()=>highlight(j.node));
 }

 // ----------------------------------------------------------------- entry points
 // #flow-toggle / #flow-bar / informationFlow / .flow-entry preserved verbatim for
 // indexer_ui.html, attention_blocks_ui.html, layer_trace_ui.html, dspark_ui.html,
 // audit_ui.html and page_template.html, which all read this exact contract.
 let enabled=false;
 const toggle=document.createElement('button');toggle.id='flow-toggle';toggle.textContent='Data Flow';toggle.setAttribute('aria-pressed','false');document.querySelector('.modes').after(toggle);

 const nav=document.createElement('nav');nav.className='csa-view-tabs';nav.setAttribute('role','tablist');nav.setAttribute('aria-label','页面视图');
 nav.innerHTML='<button id="csa-tab-architecture" role="tab" aria-selected="true" aria-controls="stage">Model Architecture</button><button id="csa-tab-walkthrough" role="tab" aria-selected="false" aria-controls="flow-bar" tabindex="-1">Step-by-step · Data Flow</button><span class="csa-tab-caption">数据流 + Shape · 可暂停 / 回退 / 切换 Layer</span>';
 document.querySelector('header').append(nav);$('stage').setAttribute('role','tabpanel');$('stage').setAttribute('aria-labelledby','csa-tab-architecture');

 function fitCanvas(){
  if(root.hidden)return;
  const top=$('csa-canvas').getBoundingClientRect().top;
  root.style.setProperty('--csa-canvas-h',Math.max(280,window.innerHeight-top-16)+'px');
 }
 const canvasFitObserver=new ResizeObserver(fitCanvas);canvasFitObserver.observe(root);canvasFitObserver.observe(root.querySelector('.csa-story-head'));window.addEventListener('resize',fitCanvas);
 function activate(view){
  // view: true (open this panel), false (architecture), or 'pipeline' (Inference Pipeline
  // tab, a separate unrelated feature sharing the same nav.csa-view-tabs bar).
  const pipeline=view==='pipeline',on=view===true;
  enabled=on;toggle.setAttribute('aria-pressed',String(on));document.body.classList.toggle('csa-walkthrough-active',on);
  document.body.classList.toggle('csa-focus',on&&presentation);
  window.inferencePipeline?.setActive(pipeline);
  root.hidden=!on;$('stage').hidden=on||pipeline;closePanel();
  for(const [id,selected]of [['architecture',!on&&!pipeline],['walkthrough',on],['pipeline',pipeline]]){const b=$('csa-tab-'+id);if(b){b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;}}
  if(on&&!current)setLayer(layer);
  requestAnimationFrame(()=>{window.dispatchEvent(new Event('resize'));fitCanvas();});
 }
 function switchView(view){pause();activate(view);}
 toggle.addEventListener('click',()=>switchView(!enabled));
 nav.addEventListener('click',e=>{const b=e.target.closest('[role=tab]');if(b)switchView(b.id==='csa-tab-pipeline'?'pipeline':b.id==='csa-tab-walkthrough');});
 nav.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...nav.querySelectorAll('[role=tab]')],cur=tabs.findIndex(b=>b.getAttribute('aria-selected')==='true');const i=e.key==='Home'?0:e.key==='End'?tabs.length-1:(cur+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[i].click();tabs[i].focus();});
 document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{if(enabled)switchView(false);}));

 // drawPanel wrap: inject a .flow-entry button into the shared module detail panel for
 // the same module keys the former shape_trace.js handled, so Indexer / Parameters /
 // Attention-Block panels keep their "trace in Data Flow ->" shortcut working unchanged.
 const oldDraw=drawPanel;
 drawPanel=function(...args){
  oldDraw(...args);
  if(['global','hidden','multi','localstates'].includes(selectedModule())){
   const key=selectedModule(),b=document.createElement('button');
   b.className='flow-entry';b.textContent='在 Data Flow 中追踪 →';b.style.marginBottom='12px';
   b.onclick=()=>{if(!enabled)toggle.click();focusNode(key);};
   body.prepend(b);
  }
 };
 function selectedModule(){return architectureApp.state.selected;}

 // #shape-layer: full 0-39 range for compatibility with existing cross-panel jumps and
 // tests (e.g. selecting layer 39 from the Parameters panel's smoke test); the 4
 // quick-pick buttons below drive the same control for the presenter's main path.
 const layerSelect=$('shape-layer');
 for(let i=0;i<C.num_hidden_layers;i++){const o=document.createElement('option');o.value=i;o.textContent=`Layer ${i} · ${layerInfo(i).mode}`;layerSelect.append(o);}
 layerSelect.value=String(layer);
 layerSelect.addEventListener('change',()=>setLayer(Number(layerSelect.value)));

 // ----------------------------------------------------------------- controls wiring
 const stepNav=root.querySelector('.csa-step-nav');
 function renderStepNav(){stepNav.innerHTML=current.stages.map((st,i)=>`<button data-csa-stage="${i}"><b>${i+1}</b><span>${st.name}</span></button>`).join('');}
 stepNav.addEventListener('click',e=>{const b=e.target.closest('[data-csa-stage]');if(b)moveStage(Number(b.dataset.csaStage));});

 const layerBar=root.querySelector('.csa-layer-switch');
 const layerButtons=REPRESENTATIVE.map(L=>{const info=layerInfo(L);const lbl=info.mode==='Full'&&info.ratio!==2?`Layer ${L} · Full (ratio ${info.ratio})`:`Layer ${L} · ${info.mode}`;return `<button data-csa-layer="${L}" aria-pressed="${L===layer}">${lbl}</button>`;}).join('');
 layerBar.insertAdjacentHTML('afterbegin',layerButtons);
 layerBar.addEventListener('click',e=>{const b=e.target.closest('[data-csa-layer]');if(b)setLayer(Number(b.dataset.csaLayer));});

 $('csa-prev').addEventListener('click',()=>moveFlat(flatIndex()-1));
 $('csa-next').addEventListener('click',()=>moveFlat(flatIndex()+1));
 $('csa-play').addEventListener('click',play);
 $('csa-reset').addEventListener('click',()=>{pause();setLayer(2);scrollArea.scrollTop=0;});
 $('csa-progress').addEventListener('input',e=>{const value=Number(e.target.value);pause();phase=value/100;elapsed=phase*duration;interpolate(phase);controls();});
 $('csa-speed').addEventListener('change',e=>{speed=Number(e.target.value);});
 $('csa-focus').addEventListener('click',()=>{presentation=!presentation;document.body.classList.toggle('csa-focus',presentation&&enabled);$('csa-focus').textContent=presentation?'退出演示':'演示视图';$('csa-focus').setAttribute('aria-pressed',String(presentation));scrollArea.scrollTop=0;requestAnimationFrame(()=>{window.dispatchEvent(new Event('resize'));fitCanvas();});});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);
 document.addEventListener('keydown',e=>{if(!enabled||e.target.closest('button,input,select,textarea,summary,[contenteditable=true]'))return;if(['ArrowRight','ArrowLeft',' '].includes(e.key)){e.preventDefault();e.stopImmediatePropagation();if(e.key===' ')play();else moveFlat(flatIndex()+(e.key==='ArrowRight'?1:-1));}},true);

 /* Optional scalar matrix example retains its separate controls. */
 let matrixStep=0;
 const messages=['显示 A、B 与输出 C。','取 A 的第一行。','取 B 的第一列。','三个对应元素相乘。','相加得到 22。','写入 C₀₀。','换第二列，得到 28。','重复行 × 列，补齐 C。'];
 function matrixRender(){const col=matrixStep>=6?1:0;const mat=(name,v,type)=>`<div class="csa-matrix-unit"><div class="csa-matrix-name">${name}</div><div class="csa-matrix-grid" style="grid-template-columns:repeat(${v[0].length},auto)">${v.map((r,i)=>r.map((n,j)=>`<span class="${type==='a'&&i===0&&matrixStep>=1?'csa-row-hot':type==='b'&&j===col&&matrixStep>=2?'csa-col-hot':type==='c'&&n!==''?'csa-result-hot':''}">${n}</span>`).join('')).join('')}</div></div>`;
  $('csa-matrix-message').textContent=messages[matrixStep];$('csa-matrix-indicator').textContent=`${matrixStep+1} / 8`;$('csa-matrix-prev').disabled=!matrixStep;$('csa-matrix-next').disabled=matrixStep===7;
  $('csa-matrix-scene').innerHTML=`<div class="csa-matrices">${mat('A [2,3]',[[1,2,3],[4,5,6]],'a')} × ${mat('B [3,2]',[[1,2],[3,4],[5,6]],'b')} = ${mat('C [2,2]',[[matrixStep>=5?22:'',matrixStep>=6?28:''],[matrixStep>=7?49:'',matrixStep>=7?64:'']],'c')}</div><div class="csa-products">${matrixStep<3?'':matrixStep>=6?'1 × 2 + 2 × 4 + 3 × 6 = 28':matrixStep>=4?'1 + 6 + 15 = 22':'1 × 1 + 2 × 3 + 3 × 5'}</div><div class="csa-matrix-conclusion">${matrixStep===7?'精确 20 FLOPs；2MKN 近似 24 FLOPs。':''}</div>`;
 }
 $('csa-matrix-prev').addEventListener('click',()=>{matrixStep=Math.max(0,matrixStep-1);matrixRender();});$('csa-matrix-next').addEventListener('click',()=>{matrixStep=Math.min(7,matrixStep+1);matrixRender();});$('csa-matrix-reset').addEventListener('click',()=>{matrixStep=0;matrixRender();});

 // ----------------------------------------------------------------- init
 current=buildStages(layer);renderStepNav();
 paint(layout(layer,0,0));current.__laid=true;from=new Map([...nodes].map(([id,n])=>[id,{...n.now}]));to=from;phase=1;
 metadata();matrixRender();

 // Read-only diagnostics; browser tests drive the same controls as the presenter.
 window.informationFlow={get state(){return {enabled,layer,stage:sIdx+1,beat:bIdx+1,mode:current?.info.mode}},steps:[]};
 window.csaWalkthrough={get state(){return{active:enabled,layer,stage:sIdx+1,beat:bIdx+1,flat:current?flatIndex()+1:1,mode:current?.info.mode,playing:running||auto,running,auto,phase,presentation,matrixStep:matrixStep+1}},get config(){return structuredClone(C)},get stages(){return current.stages.map(st=>({name:st.name,beats:st.beats.map(b=>({title:b.title,shape:b.shape}))}))}};
})();
