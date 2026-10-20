/* Hierarchical indexing teaching scene. Inserted into the existing persistent
 * SVG and RAF clock; candidate positions and per-layer Top-K are separate data. */
function createCandidateWalkthrough({scene,node,box,label,textNode,wire,el,C,config}){
 const blockSize=config.candidate_block_size,realBlocks=config.candidate_topk_blocks;
 const realTopK=config.index_topk,source=config.candidate_source_layer;
 if(blockSize!==8||source!==20)throw new Error('Update the 32-position teaching layout for the changed candidate configuration.');
 const layers=[20,24,28,32,36];
 const sourceScores=[.24,.37,.92,.44,.31,.59,.41,.26,.62,.98,.52,.71,.33,.96,.58,.49,.35,.88,.45,.61,.29,.73,.40,.56,.66,.47,.94,.55,.79,.46,.68,.70];
 const blockScores=Array.from({length:4},(_,b)=>Math.max(...sourceScores.slice(b*8,b*8+8)));
 // Match select_candidate_blocks: reserve the newest visible block, then
 // choose the remaining block by its maximum position score.
 const rankedBlocks=[0,1,2,3].sort((a,b)=>blockScores[b]-blockScores[a]);
 const poolBlocks=[3,...rankedBlocks.filter(b=>b!==3).slice(0,1)].sort((a,b)=>a-b);
 const pool=Object.freeze(Array.from({length:32},(_,i)=>i).filter(i=>poolBlocks.includes(Math.floor(i/8))));
 const score=(layer,id)=>layer===20?sourceScores[id]:(((id*(layer+7)+layer*13+(id%3)*17)%89)+10)/100;
 const top=(ids,layer)=>ids.slice().sort((a,b)=>score(layer,b)-score(layer,a)||a-b).slice(0,4);
 const ownTop=top(Array.from({length:32},(_,i)=>i),20);
 const selected=Object.fromEntries(layers.slice(1).map(l=>[l,top(pool,l)]));
 const g=scene('candidate-pool');
 const actions=[
  ['Decoder L20','Decoder Layer 20 · 全部可见 Global Positions','Decoder 从 ratio=1 的 Layer 20 开始；每个方块代表一个 Global KV 位置，并有对应的 Index Key。'],
  ['Decoder L20','当前 Indexer Query 为所有可见位置评分','用 L20 自己的 Query 与 Index Keys 计算位置分数；此处逐项展示 32 个教学分数。'],
  ['Candidate Pool','每 8 个位置组成一个 Block','32 个位置移动为四组；位置编号保持不变，每个 Block 含 8 个位置。'],
  ['Candidate Pool','每个 Block 取内部最高分','八个位置分数汇聚成一个 Block Score：max(s₀,…,s₇)。'],
  ['Candidate Pool','按 Block Score 选择候选 Blocks','示意选择 2 个 Block；真实上限为 2048 个，参考实现优先保留最新可见 Block。'],
  ['Candidate Pool','展开选中的 Blocks → Candidate Pool','B1 与 B3 的全部 16 个位置移入候选池；真实上限为 2048×8 = 16,384 个位置。'],
  ['Decoder L20','分开保留 L20 自己的 Top-K','L20 本层 Top-K 从完整可见位置独立选择；它与 Candidate Pool 是两条输出，示例 #2 只属于本层 Top-4。']
 ];
 for(const layer of layers.slice(1))actions.push(
  ['Reindex','候选池传到 Decoder Layer '+layer,'复用 L20 的同一组候选位置与 KV / Index Keys；当前层产生自己的 Indexer Query。'],
  ['Reindex','Layer '+layer+' · 对候选域重新评分','动画展示有效候选域内的分数变化；参考代码实际先对缓存评分，再施加候选掩码。'],
  ['Reindex','Layer '+layer+' · 池内选择自己的 Top-K','候选池的 16 个位置不变；当前层用自己的新分数选择教学 Top-4，真实上限 Top-512。'],
  ['Reindex','Layer '+layer+' · Main Attention 读取两种 KV','按本层 Top-K 读取 Global KV，与本层 Local KV 一起执行 Main Attention；候选池不重建。']
 );
 label(g,45,28,'教学缩略：32 positions → 4 blocks → 2 blocks / 16 positions → Reindex Top-4',19,C.q);
 label(g,45,53,`真实配置：${blockSize} positions/block · 最多 ${realBlocks} blocks / ${(realBlocks*blockSize).toLocaleString('en-US')} positions · Top-${realTopK}`,14,C.g);
 label(g,45,77,'方块 = 位置；Global KV 宽 512，Index Key 宽 128；所有数字为教学示例。',12,C.m);
 for(let j=0;j<5;j++){
  box('hp-layer-'+layers[j],g,645+j*106,85,94,33,C.q,'L'+layers[j]);
  box('hp-query-'+layers[j],g,645+j*106,85,215,62,C.q);
  textNode('hp-query-label-'+layers[j],g,0,0,'Q_index · L'+layers[j]+' [32,128]',C.q,16);
 }
 textNode('hp-source-label',g,45,120,'L20 · 全部可见位置 / Scores',C.g,18);
 function original(id){return{x:55+(id%8)*92,y:145+Math.floor(id/8)*73};}
 function grouped(id){return{x:55+Math.floor(id/8)%2*405+(id%4)*78,y:155+Math.floor(id/16)*174+Math.floor((id%8)/4)*60};}
 function pooled(id){const j=pool.indexOf(id);return{x:55+(j%8)*92,y:192+Math.floor(j/8)*100};}
 for(let i=0;i<32;i++){
  const p=original(i);
  node('hp-pos-'+i,g,n=>{el('rect',{width:66,height:46,rx:4,fill:'currentColor','fill-opacity':.12,stroke:'currentColor'},n);label(n,7,16,'#'+i,13);},{...p,color:C.g});
  node('hp-score-'+i,g,n=>{el('rect',{'data-box':true,width:64,height:23,rx:2,fill:'currentColor','fill-opacity':.15},n);const t=label(n,7,17,'0.00',14);t.setAttribute('data-motion-number','');},{x:p.x+1,y:p.y+22,w:64,h:23,value:0,color:C.q,opacity:0});
 }
 for(let b=0;b<4;b++){
  const x=43+(b%2)*405,y=121+Math.floor(b/2)*174;
  box('hp-block-'+b,g,x,y,335,150,C.g);
  textNode('hp-block-title-'+b,g,x+12,y+24,'B'+b+' · #'+b*8+'…#'+(b*8+7),C.g,15);
  node('hp-block-score-'+b,g,n=>{el('rect',{width:95,height:25,rx:4,fill:'#fff2dd',stroke:C.g},n);label(n,8,18,'max '+blockScores[b].toFixed(2),15,C.g);},{x:x+226,y:y+7,color:C.g,opacity:0});
  // A copy of the actual maximum position score moves into the block summary.
  const winner=sourceScores.indexOf(blockScores[b],b*8),p=grouped(winner);
  node('hp-max-fly-'+b,g,n=>{el('rect',{width:49,height:23,rx:4,fill:'#fff',stroke:C.s},n);label(n,6,17,blockScores[b].toFixed(2),16,C.s);},{x:p.x+8,y:p.y+23,color:C.s,opacity:0});
 }
 textNode('hp-latest',g,462,469,'B3：最新可见 Block 优先保留',C.g,13);
 box('hp-pool-frame',g,42,132,757,260,C.g);
 textNode('hp-pool-label',g,55,160,'输出 B · L20 Candidate Pool · 固定 16 个位置',C.g,19);
 textNode('hp-pool-ids',g,55,427,'B1: #8…#15    +    B3: #24…#31',C.g,18);
 textNode('hp-pool-upper',g,55,455,'真实上限 16,384；少于上限时只保留可见候选。',C.m,13);
 wire('hp-score-arrow',g,'M900 165 H825','q');
 textNode('hp-score-formula',g,900,224,'ReLU(Q_index K_indexᵀ)',C.q,16);
 textNode('hp-score-heads',g,900,247,'→ 32 Heads 加权汇总',C.q,14);
 box('hp-own-frame',g,890,274,280,149,C.q);
 textNode('hp-own-title',g,905,299,'独立输出 A · L20 本层 Top-4',C.q,17);
 textNode('hp-own-note',g,900,449,'#2 不在候选池，但属于 L20 Top-4',C.q,14);
 textNode('hp-own-summary',g,45,565,'L20 独立 Top-4：',C.q,15);
 for(let i=0;i<4;i++){
  const id=ownTop[i],p=original(id);
  node('hp-own-'+i,g,n=>{el('rect',{width:112,height:33,rx:4,fill:'#edf4ff',stroke:C.q},n);label(n,10,23,'#'+id+'  '+score(20,id).toFixed(2),16,C.q);},{...p,color:C.q,opacity:0});
  node('hp-top-'+i,g,n=>{el('rect',{width:112,height:33,rx:4,fill:'#f2eafa',stroke:C.s},n);label(n,10,23,'',16,C.s).setAttribute('data-reindex-result',String(i));},{x:55,y:192,color:C.s,opacity:0});
 }
 // Candidate mask is passed, not rebuilt. The large pool matrix stays fixed.
 node('hp-carry',g,n=>{for(let i=0;i<16;i++){el('rect',{x:(i%4)*30,y:Math.floor(i/4)*25,width:26,height:20,rx:2,fill:'#fff0d9',stroke:C.g},n);label(n,(i%4)*30+4,Math.floor(i/4)*25+14,String(pool[i]),9,C.g);}},{x:150,y:190,color:C.g,opacity:0});
 textNode('hp-current-top',g,900,285,'当前层 · 池内 Top-4',C.s,17);
 box('hp-main-frame',g,895,416,280,113,C.s);
 textNode('hp-main-title',g,910,440,'本层 Main Attention',C.s,17);
 textNode('hp-main-shapes',g,910,515,'真实：Local ≤128 + Global ≤512',C.m,12);
 for(let i=0;i<4;i++){
  box('hp-local-'+i,g,912+i*28,453,23,25,C.l,'');
  node('hp-global-read-'+i,g,n=>{el('rect',{width:23,height:25,rx:3,fill:'#f2eafa',stroke:C.s},n);label(n,3,17,'',10,C.s).setAttribute('data-read-id',String(i));},{x:1035+i*28,y:453,color:C.s});
 }
 textNode('hp-main-kv',g,912,496,'Local KV         selected Global KV',C.m,12);
 box('hp-output',g,970,550,154,30,C.s,'Output [1,64,512]');
 wire('hp-main-arrow',g,'M1100 533 V544','s');
 textNode('hp-scope',g,45,499,'候选池复用的是位置集合；每个 Reindex Layer 有自己的 Query 与分数。',C.q,15);
 label(g,45,592,'实现口径：动画突出候选域；参考代码先计算缓存分数再加候选掩码。最新可见 Block 优先保留。',12,C.m);
 function stage(s){const n=s-22;if(n<7)return{n,layer:20,part:-1};const j=Math.min(3,Math.floor((n-7)/4));return{n,layer:layers[j+1],part:(n-7)%4};}
 let transition={from:22,to:22};
 function layout(s,set){const {n,layer,part}=stage(s),group=n>=2,inPool=n>=5,isRe=n>=7;
  const show=(id,on)=>set(id,{opacity:on?1:0});
  for(let i=0;i<32;i++){
   const member=pool.includes(i),p=inPool&&member?pooled(i):group?grouped(i):original(i);
   const picked=isRe&&part>=2&&selected[layer].includes(i),dim=inPool&&!member||n===4&&!poolBlocks.includes(Math.floor(i/8));
   set('hp-pos-'+i,{...p,opacity:dim?(inPool?0:.12):1,color:picked?C.s:C.g});
   const scored=isRe?(part>=1?layer:layers[Math.max(0,layers.indexOf(layer)-1)]):20;
   set('hp-score-'+i,{x:p.x+1,y:p.y+22,value:score(scored,i),w:64,opacity:n<1||inPool&&!member?0:dim?.12:1,color:picked?C.s:C.q});
  }
  for(let b=0;b<4;b++){
   show('hp-block-'+b,group&&!inPool);set('hp-block-'+b,{color:n>=4&&poolBlocks.includes(b)?C.s:C.g,opacity:group&&!inPool?(n===4&&!poolBlocks.includes(b)?.12:1):0});
   set('hp-block-title-'+b,{opacity:group&&!inPool?(n===4&&!poolBlocks.includes(b)?.15:1):0});show('hp-block-score-'+b,n>=3&&!inPool);set('hp-block-score-'+b,{opacity:n>=3&&!inPool?(n===4&&!poolBlocks.includes(b)?.15:1):0});
   const x=43+(b%2)*405,y=121+Math.floor(b/2)*174,winner=sourceScores.indexOf(blockScores[b],b*8),p=grouped(winner);
   set('hp-max-fly-'+b,{x:n>=3?x+263:p.x+8,y:n>=3?y+8:p.y+23,opacity:0});
  }
  show('hp-source-label',!group);show('hp-latest',n===4);
  for(const id of ['hp-pool-frame','hp-pool-label','hp-pool-ids','hp-pool-upper'])show(id,inPool);
  for(const id of ['hp-own-frame','hp-own-title','hp-own-note'])show(id,n===6);show('hp-own-summary',isRe);
  for(let i=0;i<4;i++){const origin=inPool&&pool.includes(ownTop[i])?pooled(ownTop[i]):grouped(ownTop[i]);set('hp-own-'+i,{x:n<6?origin.x:isRe?210+i*125:905+(i%2)*132,y:n<6?origin.y:isRe?545:315+Math.floor(i/2)*50,sx:isRe?.8:1,sy:isRe?.8:1,opacity:n>=6?1:0});}
  for(let j=0;j<5;j++){
   const current=layers[j]===layer;
   set('hp-layer-'+layers[j],{color:current?C.s:C.m,opacity:current?1:.4});
   set('hp-query-'+layers[j],{x:current?905:645+j*106,y:current?133:85,opacity:current?1:0});
   set('hp-query-label-'+layers[j],{x:current?917:645+j*106,y:current?159:105,opacity:current?1:0});
  }
  show('hp-score-arrow',n===1||isRe&&part===1);show('hp-score-formula',n===1||isRe&&part===1);show('hp-score-heads',n===1||isRe&&part===1);
  set('hp-carry',{x:isRe?645+layers.indexOf(layer)*106:150,y:isRe?85:190,sx:isRe?.45:1,sy:isRe?.45:1,opacity:0});
  for(let i=0;i<4;i++){
   const id=isRe?selected[layer][i]:selected[24][i],p=pooled(id);
   set('hp-top-'+i,{x:isRe&&part>=2?905+(i%2)*132:p.x,y:isRe&&part>=2?301+Math.floor(i/2)*50:p.y,opacity:isRe&&part>=2?1:0});
   show('hp-local-'+i,isRe&&part>=3);
   set('hp-global-read-'+i,{x:isRe&&part>=3?1035+i*28:905+(i%2)*132,y:isRe&&part>=3?453:301+Math.floor(i/2)*50,opacity:isRe&&part>=3?1:0});
  }
  show('hp-current-top',isRe&&part>=2);
  for(const id of ['hp-main-frame','hp-main-title','hp-main-shapes','hp-main-kv','hp-output','hp-main-arrow'])show(id,isRe&&part>=3);
  show('hp-scope',inPool&&!(isRe&&part===0));
 }
 function metadata(s){const {layer,part}=stage(s),displayLayer=layer!==20&&part<2?layers[layers.indexOf(layer)-1]:layer;for(let i=0;i<4;i++){const id=(selected[displayLayer]||ownTop)[i];g.querySelector('[data-reindex-result="'+i+'"]').textContent='#'+id+'  '+score(displayLayer,id).toFixed(2);g.querySelector('[data-read-id="'+i+'"]').textContent=String(id);}return'Decoder Layer '+layer+' · ratio = 1 · '+(layer===20?'Full / Candidate Source':'Reindex · Candidate Pool from L20');}
 function motion(s,t,id,a,b,v){
  if(s<22||s>44)return;
  const active=transition.to===s;
  if(active&&s===23&&id.startsWith('hp-score-')&&a.opacity!==b.opacity){const j=Number(id.slice(9)),u=Math.max(0,Math.min(1,t*1.8-j/40));v.opacity=u;v.value=b.value*u;v.w=64*u;}
  if(active&&(s===25||transition.from===25&&s===24)&&id.startsWith('hp-max-fly-'))v.opacity=Math.sin(Math.PI*t);
  if(active&&s>=29&&(s-29)%4===0&&id==='hp-carry')v.opacity=Math.sin(Math.PI*t);
  if(active&&s>=29&&(s-29)%4===0&&id.startsWith('hp-query-')){if(a.opacity>b.opacity)v.opacity=a.opacity*Math.max(0,1-t*2.5);else if(a.opacity<b.opacity)v.opacity=Math.max(0,(t-.35)/.65);}
  if(active&&s===29&&id.startsWith('hp-own-')&&/^hp-own-\d$/.test(id))v.y+=Math.sin(Math.PI*t)*(Number(id.slice(7))<2?-25:25);
 }
 function prepare(previous,s,from,to){transition={from:previous,to:s};if(s>=29&&s<=41&&(s-29)%4===0){Object.assign(from.get('hp-carry'),{x:150,y:190,sx:1,sy:1,opacity:0});}}
 return{actions,layout,metadata,motion,prepare,get state(){return{...stage(transition.to),pool:pool.slice(),poolBlocks:poolBlocks.slice(),blockScores:blockScores.slice(),sourceTop:ownTop.slice(),topByLayer:structuredClone(selected),real:{blockSize,blocks:realBlocks,poolLimit:blockSize*realBlocks,topK:realTopK}};}};
}
