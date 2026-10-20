/* Persistent SVG scene graph. Navigation changes numeric poses, never replaces the
 * stage. One requestAnimationFrame clock owns all motion, including pause/seek. */
(()=>{
 'use strict';
 const $=id=>document.getElementById(id),root=$('csa-walkthrough'),config=JSON.parse($('csa-config').textContent);
 const scrollArea=root.querySelector('.csa-scroll');
 const NS='http://www.w3.org/2000/svg',C={q:'#2863c5',l:'#22936b',g:'#cb7916',s:'#8a4ec2',m:'#8a98ac'};
 const chapters=[['SWA',0],['KV Compression',4],['Indexer / Top-K',10],['Main Attention',17],['Full / Reindex / Reuse',22]];
 const actions=[
  ['SWA','Query 矩阵与 Token 序列','矩阵的每一行对应一个 Token；本次计算当前位置 t4096。'],
  ['SWA','取出当前 Token 的 Query','高亮 t4096 对应的 Query 行，复制到本次计算区域。'],
  ['SWA','窗口滑到最近 128 个位置','Local Window 只保留 t3969 至 t4096，包含当前位置。'],
  ['SWA','把窗口内的行组织成 Local KV','选中的 KV 行复制到读取矩阵，原缓存继续保留。'],
  ['KV Compression','用八行表示八个历史 Token','每两行构成一组；真实 Compression Ratio = 2。'],
  ['KV Compression','第一对 Token 进入 Compressor','先做可学习的投影和门控，再按通道加权合并；不是简单平均。'],
  ['KV Compression','两行合并成第一个 Global Slot','两个 Token 的信息汇入一个 512 维 Global KV 行。'],
  ['KV Compression','压缩第二对：2 → 1','t2、t3 合并，输出矩阵新增一行 g1。'],
  ['KV Compression','压缩第三对：2 → 1','t4、t5 合并，输出矩阵新增一行 g2。'],
  ['KV Compression','压缩第四对：八行变成四行','t6、t7 合并为 g3；4096 个完整历史 Token 对应 2048 个 Global Slots。'],
  ['Indexer','独立的 Indexer Query 与 Index Keys','图示四个候选位置；每个位置的 Index Key 宽度为 128。'],
  ['Indexer','计算位置 #0 的分数','Query 与第 0 行 Index Key 点积，经过 ReLU 与 Head 加权汇总，写入分数。'],
  ['Indexer','计算位置 #1 的分数','换下一行 Index Key，Score Vector 继续增长。'],
  ['Indexer','计算位置 #2 的分数','每次只计算一个示意位置；其余分数保留。'],
  ['Indexer','计算位置 #3 的分数','四个示意分数齐备，接下来按分数选择位置。'],
  ['Indexer','按分数整理候选位置','分数卡片沿纵向移动，高分位置排在前面；这只是 Top-K 的教学排序。'],
  ['Indexer','取出高分位置索引','图中示意选择 2 个位置；真实 Top-K 上限为 512，输出是索引而不是 KV。'],
  ['Main Attention','准备本层 Query 和两种 KV','Local KV 保留近处上下文；选中的 Global KV 提供压缩历史信息。'],
  ['Main Attention','按索引复制 Global KV，并组织读取区域','示意的 6 行 Local 与 2 行 Global 拼成 8 行；真实上限为 128 + 512 = 640。'],
  ['Main Attention','Main Query 与每一行 KV 计算相关性','图示一个 Head：q [1,512] × KVᵀ [512,640] → scores [1,640]。'],
  ['Main Attention','所有读取位置共同归一化','Local 与 Global 共用一次 Softmax；模型的 Attention Sink 也参与分母。'],
  ['Main Attention','按权重汇总，得到 Attention Output','各行 Value 乘以权重后相加；64 个 Head 得到 O_head [1,64,512]。'],
  ['Modes','Full / Reindex / Reuse','缓存复用与是否重新选址不同；三种模式都执行本层 Main Attention。']
 ];
 let step=0,active=false,mode='Full',presentation=false,matrixStep=0;
 let running=false,auto=false,raf=null,hold=null,elapsed=0,last=null,phase=1,speed=1,motionPair=null,motionSort=false;
 const duration=1800,nodes=new Map(),sceneGroups=[];
 function el(tag,attrs={},parent){const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(parent)parent.append(e);return e;}
 const svg=el('svg',{viewBox:'0 0 1200 600',role:'img','aria-labelledby':'csa-motion-title csa-motion-desc'});
 const svgTitle=el('title',{id:'csa-motion-title'},svg),svgDesc=el('desc',{id:'csa-motion-desc'},svg);
 const defs=el('defs',{},svg);
 for(const [k,c] of Object.entries(C)){const m=el('marker',{id:'csa-m-'+k,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'},defs);el('path',{d:'M0 0L10 5L0 10Z',fill:c},m);}
 $('csa-canvas').append(svg);
 function scene(name){const g=el('g',{'data-scene':name},svg);sceneGroups.push(g);return g;}
 function label(parent,x,y,value,size=17,color='#20375c',anchor='start'){const t=el('text',{x,y,'font-size':size,fill:color,'text-anchor':anchor},parent);t.textContent=value;return t;}
 function node(id,parent,content,base={}){const e=el('g',{'data-motion-id':id},parent);content(e);const p={x:0,y:0,sx:1,sy:1,opacity:1,color:C.q,w:200,h:30,...base};nodes.set(id,{el:e,base:p,now:{...p}});return e;}
 function row(id,parent,x,y,color,tag,width=210,cols=6){return node(id,parent,g=>{
  for(let j=0;j<cols;j++)el('rect',{x:j*width/cols,y:0,width:width/cols-2,height:27,rx:2,fill:'currentColor','fill-opacity':.14,stroke:'currentColor','stroke-width':1},g);
  label(g,-10,19,tag,14,'#536580','end');
 },{x,y,color});}
 function box(id,parent,x,y,w,h,color,tag){return node(id,parent,g=>{el('rect',{'data-box':true,width:w,height:h,rx:5,fill:'currentColor','fill-opacity':.09,stroke:'currentColor','stroke-width':2},g);if(tag)label(g,12,22,tag,14);},{x,y,w,h,color});}
 function textNode(id,parent,x,y,value,color=C.q,size=17){return node(id,parent,g=>label(g,0,0,value,size,color),{x,y,color});}
 function wire(id,parent,d,key){return node(id,parent,g=>el('path',{d,fill:'none',stroke:C[key],'stroke-width':2,'marker-end':'url(#csa-m-'+key+')'},g));}

 // SWA: source rows and gathered copies are different persistent objects.
 const swa=scene('swa'),positions=[0,3968,3969,4000,4050,4094,4095,4096];
 label(swa,90,62,'Q · [4097,64,512]',21,C.q);label(swa,90,89,'行 = Token；列缩略表示 64 Heads × 512',13);
 label(swa,490,62,'Token 位置 / KV',21,C.l);label(swa,490,89,'每个位置一个共享的 512 维 KV',13);
 label(swa,920,62,'本次读取的 Local KV',21,C.l);label(swa,920,89,'真实形状 [128,512]',16,C.l);
 for(let i=0;i<8;i++){row('swa-q-'+i,swa,90,133+i*36,C.q,'t'+positions[i],270);row('swa-k-'+i,swa,490,133+i*36,C.l,'t'+positions[i],216);}
 box('swa-window',swa,480,133,236,288,C.l);
 row('swa-q-copy',swa,90,385,C.q,'qₜ',270);
 textNode('swa-q-dim',swa,90,502,'当前 Query [1,64,512]',C.q,19);
 for(let i=0;i<6;i++)row('swa-local-'+i,swa,490,205+i*36,C.l,'t'+positions[i+2],216);
 textNode('swa-window-label',swa,490,451,'t3969 … t4096 · 共 128 个位置',C.l,16);
 wire('swa-gather-arrow',swa,'M735 298 H873','l');
 textNode('swa-formula',swa,90,550,'qₜ [64,512]  ×  Local KVᵀ [512,128]  →  Local scores [64,128]',C.q,20);
 label(swa,920,497,'图中 6 行代表 128 行',14,C.m);label(swa,920,520,'原缓存保留；这里显示读取视图',13,C.m);

 // Compression: two row copies physically converge to each output row.
 const comp=scene('compression');
 label(comp,105,62,'历史输入 X',22,C.q);label(comp,105,91,'示意 [8,5120]',17,C.q);
 label(comp,855,62,'Compressed Global KV',22,C.g);label(comp,855,91,'示意 [4,512] · 真实 ratio = 2',17,C.g);
 for(let i=0;i<8;i++){row('comp-source-'+i,comp,105,142+i*43,C.q,'t'+i,216);row('comp-fly-'+i,comp,105,142+i*43,C.g,'t'+i,216);}
 box('comp-machine',comp,420,135,340,355,C.g);
 label(comp,590,180,'Compressor',23,C.g,'middle');label(comp,590,220,'W_KV  /  W_gate',19,C.g,'middle');
 label(comp,590,365,'按通道 Softmax → 加权合并',16,'#536580','middle');label(comp,590,403,'RMSNorm → RoPE',18,C.g,'middle');
 label(comp,590,456,'2 rows  →  1 row',23,C.g,'middle');
 for(let i=0;i<4;i++){row('comp-out-'+i,comp,855,167+i*77,C.g,'g'+i,216);textNode('comp-desc-'+i,comp,855,211+i*77,'t'+i*2+' + t'+(i*2+1),C.m,13);}
 textNode('comp-count',comp,855,518,'8 Tokens → 4 Global Slots',C.g,21);
 label(comp,105,558,'真实示例：4096 个完整历史 Token → [2048,512] Global KV；未满两项的尾组暂不输出。',16);

 // Indexer: scores are built one at a time, physically sorted, then indices gathered.
 const idx=scene('indexer'),scores=[.28,.92,.43,.81],rank=[3,0,2,1],chosen=[1,3];
 label(idx,65,60,'Indexer Query [32,128]',22,C.q);label(idx,65,89,'图示其中 1 个 Head：[1,128]',14);
 row('idx-q',idx,65,145,C.q,'q',216);
 label(idx,465,60,'Index Keys [C,128]',22,C.g);label(idx,465,89,'抽样 4 行；真实 C = 2048',14);
 label(idx,850,60,'Score Vector [C]',22,C.s);label(idx,850,89,'ReLU + 32 Heads 加权汇总',14);
 for(let i=0;i<4;i++){
  row('idx-key-'+i,idx,465,145+i*63,C.g,'#'+i,216);
  row('idx-key-fly-'+i,idx,465,145+i*63,C.g,'k'+i,216);
  node('idx-score-'+i,idx,g=>{el('rect',{width:225,height:42,rx:5,fill:'#f2eafa',stroke:C.s},g);el('rect',{x:1,y:33,width:scores[i]*223,height:7,rx:2,fill:C.s},g);label(g,15,25,'#'+i+'     '+scores[i].toFixed(2),20,C.s);},{x:850,y:137+i*63,color:C.s});
 }
 box('idx-dot-area',idx,55,265,290,130,C.q);textNode('idx-dot-formula',idx,145,298,'q · kᵢᵀ',C.q,22);textNode('idx-dot-footer',idx,85,378,'→ ReLU → Head 汇总',C.q,16);
 wire('idx-to-score',idx,'M700 242 H818','s');
 box('idx-selection',idx,837,126,251,125,C.s);
 for(let i=0;i<2;i++)node('idx-pos-'+i,idx,g=>{el('rect',{width:95,height:42,rx:6,fill:'#f2eafa',stroke:C.s},g);label(g,47,27,'#'+chosen[i],21,C.s,'middle');},{x:850,y:137+i*63,color:C.s});
 textNode('idx-topk',idx,850,438,'示意 Top-2 → 位置索引',C.s,18);
 label(idx,65,506,'Index Keys 来自 Compressor 的 RoPE 前 latent，经 512 → 128 投影、归一化与位置处理。',15);
 label(idx,65,538,'实际 Top-K = 512（上限）；此处仅示意四个候选选两个，并不表示实际只保留两个。',16,C.s);
 label(idx,65,574,'Candidate Pool（Decoder）：每 8 个 KV 位置组成一个 Block，最多保留 2048 个候选 Blocks，供后续 Reindex 使用。',12,C.m).setAttribute('data-candidate-note','');

 // Main attention: copying rows shows logical gather, not an extra mandatory allocation.
 const attn=scene('attention'),weights=[.08,.09,.07,.12,.06,.10,.20,.18],logits=weights.map(w=>Math.log(w/.10));
 label(attn,75,60,'Global KV',21,C.g);label(attn,75,88,'[2048,512] · 抽样 4 行',13);
 label(attn,370,60,'Local KV',21,C.l);label(attn,370,88,'[128,512] · 抽样 6 行',13);
 label(attn,720,60,'KV Read View',21,C.s);label(attn,720,88,'最多 [640,512]',16,C.s);
 label(attn,990,60,'Scores / Weights',19,C.s);
 for(let i=0;i<4;i++)row('attn-global-'+i,attn,75,140+i*65,chosen.includes(i)?C.s:C.g,'#'+i,180);
 for(let i=0;i<6;i++){row('attn-local-'+i,attn,370,140+i*34,C.l,i===0?'3969':i===5?'4096':'…',180);row('attn-local-fly-'+i,attn,370,140+i*34,C.l,'L'+i,180);}
 for(let i=0;i<2;i++)row('attn-global-fly-'+i,attn,75,140+chosen[i]*65,C.s,'G'+chosen[i],180);
 row('attn-q',attn,75,446,C.q,'qₜ',216);label(attn,75,507,'Main Query [64,512]',18,C.q);label(attn,75,533,'图示一个 Head 的 q [1,512]',14);
 for(let i=0;i<8;i++){
  node('attn-score-'+i,attn,g=>{el('rect',{'data-box':true,width:155,height:25,rx:3,fill:'#eee6fa',stroke:C.s},g);const t=label(g,9,18,logits[i].toFixed(2),14,C.s);t.dataset.scoreText=String(i);},{x:990,y:140+i*34,color:C.s,w:155,h:25});
  row('attn-value-fly-'+i,attn,720,140+i*34,i<6?C.l:C.s,'',180);
 }
 wire('attn-q-arrow',attn,'M297 458 H658 V298 H700','q');
 textNode('attn-dot',attn,720,469,'QKᵀ / √512',C.q,20);textNode('attn-softmax',attn,990,453,'Softmax + Sink',C.s,18);
 row('attn-output',attn,820,512,C.s,'O',216);
 textNode('attn-output-dim',attn,790,571,'O_head [1,64,512] · 输出投影之前',C.s,18);
 label(attn,370,591,'行复制表示逻辑 Gather，不要求实际复制缓存。',12,C.m);
 textNode('attn-read-dim',attn,720,429,'128 Local + 512 Global',C.s,15);
 textNode('attn-sink',attn,990,490,'示例权重总和 0.90',C.m,13);
 textNode('attn-sink2',attn,990,510,'Sink 占 0.10，不加 Value',C.m,12);

 const modes=scene('modes');
 const modeInfo={Full:['Encoder L2 · ratio=2','生成 Global KV / Index Keys','本层评分 → Top-K'],Reindex:['Decoder L24 · ratio=1','复用 L20 KV / Index Keys','本层重新评分 → Top-K'],Reuse:['Decoder L21 · ratio=1','复用 L20 KV / Index Keys','连同 L20 位置索引复用']};
 ['Full','Reindex','Reuse'].forEach((name,i)=>{
  const x=65+i*380;box('mode-card-'+name,modes,x,80,340,400,C.q);label(modes,x+28,126,name,26,C.q);
  modeInfo[name].forEach((t,j)=>label(modes,x+28,169+j*44,t,j===0?15:17));
  for(let j=0;j<4;j++)row('mode-row-'+name+'-'+j,modes,x+75,305+j*30,j<2?C.g:C.s,'',180);
  label(modes,x+28,456,'本层 Main Query → Attention',18,C.q);
 });
 label(modes,65,550,'Full 的 ratio=2 示例属于 Encoder；Decoder 的两种复用模式不重复压缩。',17);

 function chapterOf(s){return s<4?0:s<10?1:s<17?2:s<22?3:4;}
 function layout(s){
  const p=new Map([...nodes].map(([id,n])=>[id,{...n.base}]));
  const set=(id,props)=>Object.assign(p.get(id),props),show=(id,on)=>set(id,{opacity:on?1:0});
  for(let i=0;i<8;i++){set('swa-q-'+i,{opacity:s>=1&&i!==7?.24:1,color:i===7&&s>=1?C.s:C.q});set('swa-k-'+i,{opacity:s>=2&&i<2?.15:1});}
  set('swa-q-copy',{y:s>=1?445:385,opacity:s>=1?1:0,color:C.s});show('swa-q-dim',s>=1);
  set('swa-window',{y:s>=2?197:133,h:s>=2?224:288,opacity:s>=2?1:0});show('swa-window-label',s>=2);
  for(let i=0;i<6;i++)set('swa-local-'+i,{x:s>=3?920:490,y:s>=3?160+i*43:205+i*36,opacity:s>=3?1:0});
  show('swa-gather-arrow',s>=3);show('swa-formula',s>=3);
  const completed=Math.max(0,Math.min(4,s-5));
  for(let i=0;i<8;i++){
   const pair=Math.floor(i/2),done=pair<completed;
   set('comp-source-'+i,{opacity:done?.22:1});
   set('comp-fly-'+i,{x:done?855:s>=5&&pair===0?470:105,y:done?167+pair*77:s>=5&&pair===0?270+(i%2)*38:142+i*43,sy:done?.35:1,opacity:done?0:s>=5&&pair===0?1:0});
  }
  for(let i=0;i<4;i++){set('comp-out-'+i,{opacity:i<completed?1:0,sy:i<completed?1:.2});show('comp-desc-'+i,i<completed);}show('comp-count',s>=9);
  const nScores=Math.max(0,Math.min(4,s-10));
  for(let i=0;i<4;i++){
   const current=i===nScores-1,selected=chosen.includes(i);
   set('idx-key-'+i,{opacity:s>=16&&!selected?.18:1,color:s>=16&&selected?C.s:C.g});
   set('idx-key-fly-'+i,{x:current?85:465,y:current?325:145+i*63,opacity:current&&s<15?1:0});
   set('idx-score-'+i,{x:s>=16&&selected?860:850,y:137+(s>=15?rank[i]:i)*63,opacity:i<nScores?(s>=16&&!selected?.16:1):0,sx:i<nScores?1:.08});
  }
  for(const id of ['idx-dot-area','idx-dot-formula','idx-dot-footer','idx-to-score'])show(id,s>=11&&s<15);show('idx-selection',s>=16);show('idx-topk',s>=16);
  for(let i=0;i<2;i++)set('idx-pos-'+i,{x:s>=16?850+i*120:865,y:s>=16?465:137+i*63,opacity:s>=16?1:0});
  for(let i=0;i<4;i++)set('attn-global-'+i,{opacity:chosen.includes(i)?1:.24});
  for(let i=0;i<6;i++)set('attn-local-fly-'+i,{x:s>=18?720:370,opacity:s>=18?1:0});
  for(let i=0;i<2;i++)set('attn-global-fly-'+i,{x:s>=18?720:75,y:s>=18?140+(i+6)*34:140+chosen[i]*65,opacity:s>=18?1:0});
  for(let i=0;i<8;i++){
   set('attn-score-'+i,{opacity:s>=19?1:0,w:s>=20?155*(.28+weights[i]*3):155,x:s>=19?990:900});
   set('attn-value-fly-'+i,{x:s>=21?820:720,y:s>=21?512:140+i*34,sy:s>=21?weights[i]*2:1,opacity:s===20?.7:0});
  }
  show('attn-read-dim',s>=18);show('attn-q-arrow',s>=19);show('attn-dot',s>=19);show('attn-softmax',s>=20);show('attn-sink',s>=20);show('attn-sink2',s>=20);
  set('attn-output',{opacity:s>=21?1:0,sy:s>=21?1:.1});show('attn-output-dim',s>=21);
  for(const name of ['Full','Reindex','Reuse'])set('mode-card-'+name,{opacity:mode===name?1:.3,color:mode===name?C.s:C.m});
  return p;
 }
 function paint(p){for(const [id,v]of p){const n=nodes.get(id);n.now={...v};n.el.setAttribute('transform',`translate(${v.x} ${v.y}) scale(${v.sx} ${v.sy})`);n.el.setAttribute('opacity',String(v.opacity));n.el.style.color=v.color;const r=n.el.querySelector('[data-box]');if(r){r.setAttribute('width',v.w);r.setAttribute('height',v.h);}}
 }
 let from=layout(0),to=layout(0);
 function mixColor(a,b,t){const c=v=>[1,3,5].map(i=>parseInt(v.slice(i,i+2),16));const aa=c(a),bb=c(b);return '#'+aa.map((n,i)=>Math.round(n+(bb[i]-n)*t).toString(16).padStart(2,'0')).join('');}
 function interpolate(t){if(t>=1){paint(to);return;}const e=t*t*(3-2*t),p=new Map();for(const [id,b]of to){const a=from.get(id),v={};for(const k of Object.keys(b))v[k]=k==='color'?mixColor(a[k],b[k],e):a[k]+(b[k]-a[k])*e;
   // A newly processed pair travels through the compressor while the output
   // materialises. At the endpoints only the source/result rows remain visible.
   if(motionPair!==null&&id.startsWith('comp-fly-')&&Math.floor(Number(id.slice(9))/2)===motionPair){v.opacity=Math.max(v.opacity,Math.sin(Math.PI*t));const lane=270+(Number(id.slice(9))%2)*38;v.y+=Math.sin(Math.PI*t)*(lane-(a.y+b.y)/2);}
   if(motionSort&&id.startsWith('idx-score-')){const i=Number(id.slice(10));v.x+=Math.sin(Math.PI*t)*(i%2?110:-110);v.sx*=1-.25*Math.sin(Math.PI*t);v.sy*=1-.35*Math.sin(Math.PI*t);}
   p.set(id,v);}paint(p);}
 function clearClock(){if(raf!==null)cancelAnimationFrame(raf);if(hold!==null)clearTimeout(hold);raf=hold=null;last=null;}
 function pause(){clearClock();running=false;auto=false;controls();}
 function controls(){const busy=running||auto;$('csa-play').textContent=busy?'Ⅱ Pause':'▶ Play';$('csa-play').setAttribute('aria-pressed',String(busy));$('csa-progress').value=String(Math.round(phase*100));$('csa-progress').setAttribute('aria-valuetext',Math.round(phase*100)+'%');$('csa-motion-status').textContent=running?'矩阵运动中':phase<1?'已暂停 · 可继续或拖动进度':'等待下一步';}
 function tick(now){if(!running)return;if(last!==null)elapsed+=(now-last)*speed;last=now;phase=Math.min(1,elapsed/duration);interpolate(phase);controls();if(phase<1)raf=requestAnimationFrame(tick);else{raf=null;last=null;running=false;if(auto&&step<actions.length-1)hold=setTimeout(()=>{hold=null;move(step+1,true);},1000/speed);else auto=false;controls();}}
 function start(){running=true;last=null;raf=requestAnimationFrame(tick);controls();}
 function metadata(){const a=actions[step],ch=chapterOf(step);$('csa-title').textContent=a[1];$('csa-description').textContent=a[2];$('csa-indicator').textContent=`Action ${step+1} / ${actions.length}`;$('csa-context').textContent=step===22?modeInfo[mode][0]:'Encoder Layer 2 · ratio = 2 · 当前位置 t4096';svgTitle.textContent=a[1];svgDesc.textContent=a[2];$('csa-prev').disabled=step===0;$('csa-next').disabled=step===actions.length-1;
  root.querySelectorAll('[data-csa-chapter]').forEach((b,i)=>{if(i===ch)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  root.querySelector('.csa-mode-switch').hidden=step!==22;root.querySelectorAll('[data-csa-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.csaMode===mode)));
  $('csa-mode-readout').hidden=true;
  for(let i=0;i<8;i++)nodes.get('attn-score-'+i).el.querySelector('text').textContent=step>=20?weights[i].toFixed(2):logits[i].toFixed(2);
 }
 function move(target,keepAuto=false,replay=false){
  const previous=step;clearClock();running=false;auto=keepAuto;step=Math.max(0,Math.min(actions.length-1,target));const ch=chapterOf(step),different=chapterOf(previous)!==ch;
  motionPair=ch===1&&!different&&Math.max(previous,step)>=6?Math.min(3,Math.max(previous,step)-6):null;
  motionSort=previous===14&&step===15||previous===15&&step===14||replay&&step===15;
  sceneGroups.forEach((g,i)=>{g.style.display=i===ch?'':'none';});
  if(different||replay){const base=replay&&step>chapters[ch][1]?layout(step-1):layout(chapters[ch][1]);paint(base);}
  from=new Map([...nodes].map(([id,n])=>[id,{...n.now}]));to=layout(step);elapsed=0;phase=0;metadata();
  if(different||step===0){phase=1;elapsed=duration;paint(to);if(auto&&step<actions.length-1)hold=setTimeout(()=>move(step+1,true),1000/speed);else auto=false;controls();}
  else start();
 }
 function play(){if(running||auto){pause();return;}auto=true;if(phase<1)start();else if(step===actions.length-1)move(0,true);else move(step+1,true);}
 function switchView(on){pause();active=on;root.hidden=!on;$('stage').hidden=on;document.body.classList.toggle('csa-walkthrough-active',on);document.body.classList.toggle('csa-focus',on&&presentation);closePanel();for(const [id,selected]of [['architecture',!on],['walkthrough',on]]){const b=$('csa-tab-'+id);b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;}requestAnimationFrame(()=>window.dispatchEvent(new Event('resize')));}
 const nav=document.createElement('nav');nav.className='csa-view-tabs';nav.setAttribute('role','tablist');nav.setAttribute('aria-label','页面视图');nav.innerHTML='<button id="csa-tab-architecture" role="tab" aria-selected="true" aria-controls="stage">Model Architecture</button><button id="csa-tab-walkthrough" role="tab" aria-selected="false" aria-controls="csa-walkthrough" tabindex="-1">Step-by-step · CSA2</button><span class="csa-tab-caption">矩阵逐步计算 · 可暂停 / 回退 / 重播</span>';document.querySelector('header').append(nav);$('stage').setAttribute('role','tabpanel');$('stage').setAttribute('aria-labelledby','csa-tab-architecture');
 nav.addEventListener('click',e=>{const b=e.target.closest('[role=tab]');if(b)switchView(b.id==='csa-tab-walkthrough');});nav.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();switchView(e.key==='Home'?false:e.key==='End'?true:!active);$('csa-tab-'+(active?'walkthrough':'architecture')).focus();});
 const stepNav=root.querySelector('.csa-step-nav');stepNav.innerHTML=chapters.map(([name,s],i)=>`<button data-csa-chapter="${i}"><b>${i+1}</b><span>${name}</span></button>`).join('');stepNav.addEventListener('click',e=>{const b=e.target.closest('[data-csa-chapter]');if(b)move(chapters[Number(b.dataset.csaChapter)][1]);});
 $('csa-prev').addEventListener('click',()=>move(step-1));$('csa-next').addEventListener('click',()=>move(step+1));$('csa-play').addEventListener('click',play);$('csa-reset').addEventListener('click',()=>{pause();mode='Full';move(0);scrollArea.scrollTop=0;});$('csa-replay').addEventListener('click',()=>move(step,false,true));
 $('csa-progress').addEventListener('input',e=>{const value=Number(e.target.value);pause();phase=value/100;elapsed=phase*duration;interpolate(phase);controls();});$('csa-speed').addEventListener('change',e=>{speed=Number(e.target.value);});
 $('csa-focus').addEventListener('click',()=>{presentation=!presentation;document.body.classList.toggle('csa-focus',presentation&&active);$('csa-focus').textContent=presentation?'退出演示':'演示视图';$('csa-focus').setAttribute('aria-pressed',String(presentation));scrollArea.scrollTop=0;requestAnimationFrame(()=>window.dispatchEvent(new Event('resize')));});
 root.querySelector('.csa-mode-switch').addEventListener('click',e=>{const b=e.target.closest('[data-csa-mode]');if(b){mode=b.dataset.csaMode;move(22);}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);
 document.addEventListener('keydown',e=>{if(!active||e.target.closest('button,input,select,textarea,summary,[contenteditable=true]'))return;if(['ArrowRight','ArrowLeft',' '].includes(e.key)){e.preventDefault();e.stopImmediatePropagation();if(e.key===' ')play();else move(step+(e.key==='ArrowRight'?1:-1));}},true);
 /* Optional scalar matrix example retains its separate controls. */
 const messages=['显示 A、B 与输出 C。','取 A 的第一行。','取 B 的第一列。','三个对应元素相乘。','相加得到 22。','写入 C₀₀。','换第二列，得到 28。','重复行 × 列，补齐 C。'];
 function matrixRender(){const col=matrixStep>=6?1:0;const mat=(name,v,type)=>`<div class="csa-matrix-unit"><div class="csa-matrix-name">${name}</div><div class="csa-matrix-grid" style="grid-template-columns:repeat(${v[0].length},auto)">${v.map((r,i)=>r.map((n,j)=>`<span class="${type==='a'&&i===0&&matrixStep>=1?'csa-row-hot':type==='b'&&j===col&&matrixStep>=2?'csa-col-hot':type==='c'&&n!==''?'csa-result-hot':''}">${n}</span>`).join('')).join('')}</div></div>`;
  $('csa-matrix-message').textContent=messages[matrixStep];$('csa-matrix-indicator').textContent=`${matrixStep+1} / 8`;$('csa-matrix-prev').disabled=!matrixStep;$('csa-matrix-next').disabled=matrixStep===7;
  $('csa-matrix-scene').innerHTML=`<div class="csa-matrices">${mat('A [2,3]',[[1,2,3],[4,5,6]],'a')} × ${mat('B [3,2]',[[1,2],[3,4],[5,6]],'b')} = ${mat('C [2,2]',[[matrixStep>=5?22:'',matrixStep>=6?28:''],[matrixStep>=7?49:'',matrixStep>=7?64:'']],'c')}</div><div class="csa-products">${matrixStep<3?'':matrixStep>=6?'1 × 2 + 2 × 4 + 3 × 6 = 28':matrixStep>=4?'1 + 6 + 15 = 22':'1 × 1 + 2 × 3 + 3 × 5'}</div><div class="csa-matrix-conclusion">${matrixStep===7?'精确 20 FLOPs；2MKN 近似 24 FLOPs。':''}</div>`;
 }
 $('csa-matrix-prev').addEventListener('click',()=>{matrixStep=Math.max(0,matrixStep-1);matrixRender();});$('csa-matrix-next').addEventListener('click',()=>{matrixStep=Math.min(7,matrixStep+1);matrixRender();});$('csa-matrix-reset').addEventListener('click',()=>{matrixStep=0;matrixRender();});
 paint(layout(0));move(0);matrixRender();
 // Read-only diagnostics; the browser test drives the same controls as the presenter.
 window.csaWalkthrough={get state(){return{active,step:step+1,mode,playing:running||auto,running,auto,phase,presentation,matrixStep:matrixStep+1}},get config(){return structuredClone(config)},get actions(){return actions.map(a=>a.slice())}};
})();
