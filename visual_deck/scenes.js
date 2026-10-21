'use strict';
// Semantic scene adapter for presentation-skill. All visible objects are native
// PowerPoint shapes, connectors and editable text; no whole-slide bitmap assets.
const fs=require('fs'),path=require('path');
const C={bg:'FAFBFE',ink:'15253D',muted:'586A83',line:'CED8E6',blue:'2563EB',blueL:'EAF1FF',purple:'7C3AED',purpleL:'F1EBFF',teal:'087F75',tealL:'E1F5F0',orange:'C2410C',orangeL:'FFF0E6',pink:'BE185D',pinkL:'FCE7F1',grey:'E5EAF2',white:'FFFFFF'};
const manifest=[];
const FONT='Microsoft YaHei';
function timing(groups){
 let id=3;
 const body=groups.map(g=>{
  const a=id++,b=id++,c=id++;
  const sets=g.ids.map(sp=>{const delay=g.delays?.[sp]||0;const ct=delay?`<p:cTn id="${id++}" dur="1" fill="hold"><p:stCondLst><p:cond delay="${delay}"/></p:stCondLst></p:cTn>`:`<p:cTn id="${id++}" dur="1" fill="hold"/>`;return `<p:set><p:cBhvr>${ct}<p:tgtEl><p:spTgt spid="${sp}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>`;}).join('');
  return `<p:par><p:cTn id="${a}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst><p:childTnLst><p:par><p:cTn id="${b}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst><p:par><p:cTn id="${c}" presetID="1" presetClass="entr" presetSubtype="0" fill="hold" grpId="0" nodeType="clickEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>${sets}</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>`;
 }).join('');
 return `<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>${body}</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>`;
}
function context(pptx,s,d){
 let stage=0,serial=0,delayMs=0;const objects=[],steps=new Map(),delays={};
 const cap=d.stage_cap===undefined?Infinity:d.stage_cap;
 function obj(fn,bounds,kind,text){
  if(stage>cap)return;
  const name=`${d.slide_id}:${stage}:${++serial}:${kind}`;
  const n=s._slideObjects.length;fn(name);const sp=n+2;
  if(stage){if(!steps.has(stage))steps.set(stage,[]);steps.get(stage).push(sp);delays[sp]=delayMs;}
  objects.push({id:sp,stage,kind,text,bounds,name});
 }
 const t=(str,x,y,w,h=0.35,size=18,color=C.ink,bold=false,align='left')=>obj(name=>s.addText(str,{x,y,w,h,fontFace:FONT,fontSize:size,color,bold,align,margin:0,breakLine:false,vertAnchor:'ctr',valign:'mid',paraSpaceAfter:0,objectName:name}),{x,y,w,h},'text',str);
 const rect=(x,y,w,h,fill=C.white,line=C.line,r=0)=>obj(name=>s.addShape(r?pptx.ShapeType.roundRect:pptx.ShapeType.rect,{x,y,w,h,rectRadius:r,fill:{color:fill},line:{color:line,width:1},radius:r,objectName:name}),{x,y,w,h},'rect');
 const line=(x1,y1,x2,y2,color=C.line,width=1.5,arrow=false,dash)=>obj(name=>s.addShape(pptx.ShapeType.line,{x:Math.min(x1,x2),y:Math.min(y1,y2),w:Math.abs(x2-x1)||0.001,h:Math.abs(y2-y1)||0.001,flipH:x2<x1,flipV:y2<y1,line:{color,width,beginArrowType:undefined,endArrowType:arrow?'triangle':undefined,dashType:dash},objectName:name}),{x:Math.min(x1,x2),y:Math.min(y1,y2),w:Math.abs(x2-x1),h:Math.abs(y2-y1)},'line');
 const dot=(x,y,r=.07,fill=C.teal)=>obj(name=>s.addShape(pptx.ShapeType.ellipse,{x:x-r,y:y-r,w:r*2,h:r*2,fill:{color:fill},line:{color:fill},objectName:name}),{x:x-r,y:y-r,w:r*2,h:r*2},'dot');
 const box=(label,x,y,w=1.5,h=.7,fill=C.blueL,color=C.blue,size=18)=>{rect(x,y,w,h,fill,color);t(label,x+.08,y+.05,w-.16,h-.1,size,color,true,'center');};
 const note=(str,y=4.65,color=C.muted)=>t(str,.5,y,9,.4,15,color);
 const arrow=(a,b,y,color=C.muted)=>line(a,y,b,y,color,2,true);
 const grid=(x,y,rows,cols,size=.13,highlight=[],fill=C.grey,color=C.teal)=>{
  for(let i=0;i<rows*cols;i++){const selected=highlight.includes(i);rect(x+(i%cols)*(size+.025),y+Math.floor(i/cols)*(size+.025),size,size,selected?color:fill,selected?color:fill);}
 };
 const matrix=(values,x,y,cell=.48,color=C.blue,label='',highlightRow=-1,highlightCol=-1)=>{
  if(label)t(label,x,y-.48,values[0].length*cell,.35,18,color,true,'center');
  values.forEach((row,i)=>row.forEach((v,j)=>{const hi=i===highlightRow||j===highlightCol;rect(x+j*cell,y+i*cell,cell-.035,cell-.035,hi?(color===C.purple?C.purpleL:C.blueL):C.white,hi?color:C.line);if(v!==null)t(String(v),x+j*cell,y+i*cell,cell-.035,cell-.035,23,hi?color:C.ink,hi,'center');}));
 };
 s.background={color:C.bg};
 t(d.section,.45,.18,8.8,.25,10.5,C.muted);
 t(d.title,.45,.62,9.1,.65,27,C.ink,true);
 line(.45,1.39,9.55,1.39,C.line,.8);
 t('输入',.5,5.22,.4,.2,9.5,C.blue);t('权重',1.05,5.22,.4,.2,9.5,C.purple);t('输出',1.6,5.22,.4,.2,9.5,C.teal);t('传递梯度',2.15,5.22,.8,.2,9.5,C.orange);t('权重梯度',3.05,5.22,.8,.2,9.5,C.pink);
 t(`${String(d.page_no||s._slideNum).padStart(2,'0')} / 22`,8.7,5.18,.85,.25,10.5,C.muted,false,'right');
 return {t,rect,line,dot,box,note,arrow,grid,matrix,stage(n){stage=n;delayMs=0;},delay(ms){delayMs=ms;},finish(){
  const gs=[...steps].sort((a,b)=>a[0]-b[0]).map(([n,ids])=>({stage:n,ids,delays:Object.fromEntries(ids.map(i=>[i,delays[i]])),description:d.clicks[n-1]}));
  if(d.stage_cap===undefined)s._teachingTiming=timing(gs);
  s.addNotes(d.notes);
  manifest.push({slide_id:d.slide_id,scene:d.math_scene,page:s._slideNum,title:d.title,steps:gs,objects});
 }};
}
function architecture(c){const{t,box,arrow,line,stage,note}=c;
 t('Architecture  →  Parameters  →  FLOPs  →  Efficiency',.5,1.63,9,.4,19,C.muted);
 stage(1);box('Text\nToken Embedding',.55,2.55,1.65,1,C.blueL,C.blue,16);box('Image\nVision + Aligner',.55,3.8,1.65,.85,C.blueL,C.blue,15);
 stage(2);arrow(2.3,2.75,3.05);line(2.3,4.15,2.75,3.5,C.blue,1.5,true);box('Causal Encoder\n20 Layers',2.85,2.55,2,1,C.blueL,C.blue);arrow(4.9,5.2,3.05);box('Causal Decoder\n20 Layers',5.3,2.55,2,1,C.purpleL,C.purple);arrow(7.35,7.7,3.05);box('LM Head\nNext Token',7.8,2.55,1.65,1,C.tealL,C.teal,17);
 stage(3);line(6.3,3.6,6.3,4.05,C.muted,1.5,true);box('DSpark · 3 Draft Blocks → 5 候选 Tokens',3.0,4.1,6.45,.65,C.white,C.muted,15);
}
function ced(c){const{t,box,arrow,line,stage,note}=c;
 box('Encoder · 0–19',2.1,2,2.4,.75,C.blueL,C.blue);box('Decoder · 20–39',6.1,2,2.6,.75,C.purpleL,C.purple);
 stage(1);t('Prefill',.5,3.12,1.1,.4,19,C.blue,true);arrow(1.7,2.1,3.35,C.blue);box('多数输入 Token\n形成 Encoder 表示',2.1,3,2.4,.85,C.blueL,C.blue,16);
 stage(2);arrow(4.6,6.05,3.4,C.teal);t('构建 Global KV',4.65,2.98,1.5,.3,13.5,C.teal);box('复用 Encoder 表示\n减少后半段重复计算',6.1,3,2.6,.85,C.tealL,C.teal,16);
 stage(3);t('Decode',.5,4.18,1.2,.4,19,C.orange,true);line(1.75,4.45,8.7,4.45,C.orange,3,true);t('当前 Token：正常主干仍经过 40 层',2.3,4.65,6.5,.3,16,C.orange);note('两部分都遵守 Causal Mask；不是传统的双向 Encoder–Decoder。',1.55);
}
function local(c){const{t,rect,line,dot,stage,note,box}=c;
 t('较早的历史',.6,1.7,3,.35,17,C.muted);t('当前 Token',8.05,1.7,1.6,.35,17,C.blue,true);
 for(let i=0;i<24;i++)rect(.65+i*.34,2.35,.25,.4,C.grey,C.grey);dot(9,2.55,.16,C.blue);
 stage(1);for(let i=17;i<24;i++)rect(.65+i*.34,2.35,.25,.4,C.blueL,C.blue);line(6.55,2.95,8.75,2.95,C.blue,2);t('Local：最近最多 128 个位置',5.6,3.1,3.9,.4,17,C.blue);
 stage(2);for(let i=0;i<8;i++){rect(.8+i*.54,3.55,.36,.37,C.purpleL,C.purple);line(1+i*.54,2.86,1+i*.54,3.48,C.purple,1,true);}t('Encoder Global KV · 2 Tokens → 1 位置',.6,4.05,5,.4,16,C.purple);
 stage(3);line(5.05,3.75,8.85,2.7,C.teal,2,true);box('Global：补充窗口之外的信息',5.55,4,3.8,.65,C.tealL,C.teal,16);note('位置数为示意；第 0、1 层仅 SWA，从第 2 层起引入压缩 Global KV。');
}
function indexer(c){const{t,rect,box,line,arrow,stage,note}=c;
 box('Indexer Query',.5,1.75,2,.65,C.purpleL,C.purple,16);box('Main Query',.5,3.7,2,.65,C.blueL,C.blue,16);
 for(let i=0;i<12;i++)rect(3+i*.36,2.48,.27,.4,C.grey,C.grey);
 t('Compressed Global KV',3,1.8,4.3,.35,19,C.muted);
 stage(1);arrow(2.6,3,2.08,C.purple);t('0.2   0.9   0.4   0.8   …',3.05,2.18,4.6,.3,16,C.purple);
 stage(2);for(const i of[1,3,7,10])rect(3+i*.36,2.48,.27,.4,C.purple,C.purple);t('选择最多 Top-512 个位置',3,3.04,5,.4,18,C.purple,true);
 stage(3);line(4.8,3.5,4.8,3.78,C.teal,2,true);arrow(2.6,3,4.03,C.blue);box('选中 Global KV + Local KV',3.05,3.7,4.25,.65,C.tealL,C.teal,16);
 stage(4);arrow(7.4,7.75,4.03,C.teal);box('Softmax\nAttention Output',7.85,3.55,1.65,1,C.tealL,C.teal,15);note('Indexer 负责“选位置”，Main Attention 负责“读信息”；两者使用不同的 Query。');
}
function reuse(c){const{t,rect,line,box,stage,note}=c;
 const xs=[.6,2.25,3.9,5.55,7.2,8.7];
 t('Full',.6,1.65,1.5,.35,21,C.purple,true);t('Reindex',3.65,1.65,2,.35,21,C.blue,true);t('Reuse',6.75,1.65,2,.35,21,C.teal,true);
 t('新 KV + 新选址',.6,2.08,2.7,.35,17,C.muted);t('共享 KV + 新选址',3.65,2.08,3,.35,17,C.muted);t('共享 KV + 共享选址',6.75,2.08,3,.35,17,C.muted);
 stage(1);t('Encoder',.6,2.72,1.5,.3,16,C.muted);['2','8','14'].forEach((v,i)=>box(v,2.1+i*2.15,2.65,1.4,.55,C.purpleL,C.purple,18));t('→ 各自后续层复用',2.05,3.35,6.5,.4,18,C.purple);
 stage(2);t('Decoder',.6,4.05,1.5,.3,16,C.muted);box('20',2.1,3.95,1.15,.6,C.purpleL,C.purple,18);line(3.35,4.25,9.3,4.25,C.teal,2,true);
 stage(3);[24,28,32,36].forEach((v,i)=>box(String(v),3.65+i*1.45,3.95,1,.6,C.blueL,C.blue,17));note('Decoder 候选池：8 positions/block，最多 2048 blocks；层号从 0 开始。');
}
function mhc(c){const{t,box,line,arrow,stage,note}=c;
 for(let i=0;i<4;i++)box(`x${i+1}`, .6,1.8+i*.63,1,.45,C.blueL,C.blue,16);
 t('4 × 5120',.55,4.2,1.7,.35,18,C.blue,true);
 stage(1);for(let i=0;i<4;i++)line(1.65,2.02+i*.63,2.7,3,C.blue,1.5,true);box('Pre\n混合为一路',2.7,2.6,1.5,.85,C.blueL,C.blue,16);
 stage(2);arrow(4.3,4.65,3,C.blue);box('Attention / MoE\n5120 维输入',4.75,2.5,2.2,1,C.purpleL,C.purple,17);
 stage(3);arrow(7.05,7.4,3,C.teal);box('Post + Comb',7.5,2.6,1.8,.85,C.tealL,C.teal,17);for(let i=0;i<4;i++){line(8.4,3.5,7.66+i*.46,4,C.teal,1.3,true);box('',7.5+i*.46,4.03,.32,.32,C.tealL,C.teal,16);}note('四路状态在主模块前汇合；主模块输出再分配，与残差信息组合。');
}
const picks=[5,39,88,151,233,340];
function expertWall(c,x,y,selected=false){c.grid(x,y,16,24,.07,selected?picks:[],C.grey,C.teal);}
function moe(c){const{t,box,arrow,line,stage,note,rect,dot}=c;
 t('Dense FFN',.5,1.62,1.9,.35,19,C.muted,true);box('Token',.55,2.2,.9,.5,C.blueL,C.blue,15);arrow(1.5,1.8,2.45);box('同一组 FFN',1.9,2.1,2.2,.7,C.purpleL,C.purple,18);t('所有输入使用同一组权重',.6,3.05,3.8,.4,17,C.muted);
 line(4.45,1.6,4.45,4.85,C.line,1);
 t('Mixture of Experts',4.7,1.62,4.8,.35,19,C.ink,true);
 stage(1);box('Token',4.75,2.12,1.1,.5,C.blueL,C.blue,16);arrow(5.95,6.25,2.37,C.blue);box('Router',6.35,2.12,1.4,.5,C.purpleL,C.purple,17);
 stage(2);expertWall(c,5.0,2.9);t('384 Routed Experts · 评分示意',5.0,2.65,3,.25,13.5,C.muted);t('Scores\n0.12 · 0.91 · …',8.0,2.65,1.55,.55,13,C.purple);
 stage(3);picks.forEach(i=>rect(5+(i%24)*.095,2.9+Math.floor(i/24)*.095,.07,.07,C.teal,C.teal));t('Top-6',8.25,3.27,1.2,.35,22,C.teal,true);
 stage(4);box('1 Shared',8.05,3.75,1.4,.6,C.tealL,C.teal,16);
 stage(5);t('6 × Expert FFN',5,4.45,2.8,.3,17,C.teal,true);
 stage(6);line(6.3,4.76,6.3,4.86,C.teal,2,true);line(8.7,4.4,8.7,4.8,C.teal,2,true);t('Σ αᵢ Expertᵢ(x) + Shared(x)',5,4.88,4.6,.27,15,C.teal,true);
}
function swiglu(c){const{t,box,line,arrow,stage,note}=c;
 box('X\n[1,5120]',.5,2.7,1.3,.9,C.blueL,C.blue,17);
 stage(1);line(1.9,3.05,2.65,2.35,C.blue,2,true);line(1.9,3.15,2.65,3.85,C.blue,2,true);box('Gate\n5120 → 2304',2.75,1.95,2,.85,C.purpleL,C.purple,17);box('Up\n5120 → 2304',2.75,3.45,2,.85,C.purpleL,C.purple,17);
 stage(2);box('SiLU',5,2.02,1.2,.65,C.white,C.purple,18);arrow(4.8,4.95,2.37,C.purple);line(6.25,2.37,6.85,3.1,C.purple,2,true);line(4.85,3.85,6.85,3.1,C.purple,2,true);box('⊙',6.9,2.8,.6,.6,C.tealL,C.teal,26);t('[1,2304]',6.55,3.6,1.5,.35,16,C.teal);
 stage(3);arrow(7.55,7.9,3.1,C.teal);box('Down\n2304 → 5120',8.0,2.65,1.5,.9,C.purpleL,C.purple,16);note('三个 Linear；SiLU 与逐元素乘法无新增可学习参数，但仍有计算成本。');
}
function engram(c){const{t,box,line,arrow,grid,stage,note}=c;
 stage(1);box('Token 组合\nN-gram',.55,2.6,1.6,.9,C.blueL,C.blue,18);arrow(2.25,2.65,3.05,C.blue);box('Hash',2.75,2.7,1.15,.7,C.purpleL,C.purple,20);
 stage(2);arrow(4,4.45,3.05,C.purple);grid(4.6,2.15,8,10,.18,[12,44,66],C.grey,C.teal);t('大型 Embedding Table',4.4,4.12,3.4,.35,17,C.muted);
 stage(3);arrow(6.8,7.35,3.05,C.teal);box('读取少量记忆\n参与后续处理',7.45,2.6,2,.9,C.tealL,C.teal,17);note('放置于 Encoder 第 1、14 层；图示为检索直觉，不是完整 Engram 内部实现。');
}
function expertParams(c){const{t,rect,box,stage,note,line}=c;
 const xs=[.7,3.85,7.0];const titles=['Gate','Up','Down'];
 [1,2,2].forEach((st,i)=>{stage(st);box(titles[i],xs[i],1.75,2.2,.5,C.purpleL,C.purple,19);rect(xs[i]+.1,2.5,i===2?1.1:2, i===2?1.2:.75,C.purpleL,C.purple);t(i===2?'[2304,5120]':'[5120,2304]',xs[i],3.95,2.3,.4,18,C.purple,true,'center');t('11,796,480',xs[i]-.1,4.38,2.5,.35,20,C.ink,true,'center');});
 stage(3);t('P(Expert) = 3 × 5120 × 2304 = 35,389,440',.6,4.82,8.8,.35,22,C.teal,true,'center');t('三个 Linear 均无加性 Bias',.6,1.45,8.8,.25,14,C.muted);
}
function active(c){const{t,box,rect,stage,note,line}=c;
 t('Total Parameters',.55,1.7,4,.4,23,C.purple,true);t('Active Parameters / Token',5,1.7,4.5,.4,23,C.teal,true);
 stage(1);expertWall(c,.65,2.4,false);box('+ Shared',3.15,2.7,1.2,.6,C.purpleL,C.purple,15);t('385 × 35,389,440',.6,4.48,4,.35,20,C.purple,true);t('13,624,934,400',.6,4.05,4,.4,24,C.purple,true);
 stage(2);expertWall(c,5.05,2.4,true);box('+ Shared',8.25,2.7,1.2,.6,C.tealL,C.teal,15);t('7 × 35,389,440',5.05,4.48,4.3,.35,20,C.teal,true);t('247,726,080',5.05,4.05,4.3,.4,24,C.teal,true);
 stage(3);t('+ Router 1,966,080 → 约 136.27 亿',.6,4.86,4.2,.24,13.5,C.muted);t('+ Router → 约 2.50 亿',5.05,4.86,4.2,.24,13.5,C.muted);
}
const A=[[1,2,3],[4,5,6]],B=[[1,2],[3,4],[5,6]];
function matrices(c,out,ar=-1,bc=-1){c.matrix(A,.8,2.0,.54,C.blue,'A [2,3]',ar,-1);c.t('×',2.85,2.4,.45,.4,28,C.muted);c.matrix(B,3.65,1.8,.54,C.purple,'B [3,2]',-1,bc);c.t('=',5.4,2.4,.5,.4,28,C.muted);c.matrix(out,6.35,2,.54,C.teal,'C [2,2]');}
function matmul(c){const{t,rect,line,stage,note}=c;
 matrices(c,[[null,null],[null,null]]);t('每次只生成 C 中的一个元素',.6,4.7,8.8,.35,18,C.muted);
 stage(1);c.matrix([A[0]],.8,2,.54,C.blue,'',0,-1);t('Row 1',.8,3.28,1.8,.35,17,C.blue,true);
 stage(2);c.matrix(B,3.65,1.8,.54,C.purple,'',-1,0);t('Column 1',3.35,3.7,2.1,.35,17,C.purple,true);
 stage(3);t('1 × 1 = 1',.7,4.12,2.1,.4,23,C.blue,true);
 stage(4);t('2 × 3 = 6',3.2,4.12,2.1,.4,23,C.purple,true);
 stage(5);t('3 × 5 = 15',5.7,4.12,2.5,.4,23,C.teal,true);
 stage(6);t('1 + 6 + 15 = 22',7.75,2.23,1.7,1.15,24,C.teal,true,'center');
 stage(7);c.matrix([[22,null],[null,null]],6.35,2,.54,C.teal);line(8.05,2.7,7.0,2.3,C.teal,2,true);
}
function matmulNext(c){const{t,stage,note}=c;
 matrices(c,[[22,null],[null,null]]);
 stage(1);c.matrix([A[0]],.8,2,.54,C.blue,'',0,-1);c.matrix(B,3.65,1.8,.54,C.purple,'',-1,1);
 stage(2);t('1 × 2 + 2 × 4 + 3 × 6 = 28',.7,3.78,8.7,.5,26,C.teal,true,'center');
 stage(3);c.matrix([[22,28],[49,64]],6.35,2,.54,C.teal);
 stage(4);t('一个输出：K 次乘法 + (K−1) 次加法',.6,4.45,8.8,.35,19,C.ink,true,'center');t('本例精确：4 × 5 = 20 FLOPs     常用近似：2MKN = 24 FLOPs',.6,4.88,8.8,.3,16,C.muted,false,'center');
}
function lowrank(c){const{t,box,rect,line,arrow,stage,note}=c;
 t('X [1,5120]',.5,1.72,1.8,.35,18,C.blue,true);rect(.65,2.35,1.5,.85,C.blueL,C.blue);
 stage(1);arrow(2.25,2.62,2.78,C.purple);box('W_QA\n[5120,1280]',2.7,2.18,1.5,1.1,C.purpleL,C.purple,16);rect(4.4,2.58,.5,.4,C.blueL,C.blue);t('[1,1280]',4.12,1.72,1.5,.35,15,C.blue,true);t('13.11 MFLOPs',2.2,4.15,2.4,.4,19,C.purple,true);
 stage(2);box('RMSNorm',4.08,3.5,1.65,.55,C.white,C.muted,16);line(4.65,3.05,4.65,3.45,C.muted,1.5,true);
 stage(3);line(5.8,3.77,6.35,3.38,C.purple,1.5,true);box('W_QB\n[1280,32768]',5.52,2.18,1.65,1.1,C.purpleL,C.purple,16);arrow(7.25,7.55,2.78,C.teal);rect(7.65,1.95,1.8,1.65,C.tealL,C.teal);t('Q\n[1,32768]',7.8,2.4,1.5,.8,18,C.teal,true,'center');t('83.89 MFLOPs',5.4,4.15,2.3,.4,19,C.purple,true);
 stage(4);t('64 Heads × 512',7.5,3.95,2,.4,18,C.teal,true,'center');t('两次 Linear 合计：96.99 MFLOPs',.5,4.6,9,.35,24,C.teal,true,'center');t('宽度为示意；FLOPs 不含 RMSNorm / RoPE',.5,5.02,8,.16,9.5,C.muted);
}
function layers(c){const{t,box,stage,rect,note}=c;
 stage(1);box('1 Layer\n96.99 MFLOPs',.6,2.25,2.3,1.25,C.purpleL,C.purple,23);
 stage(2);for(let i=0;i<40;i++)rect(3.6+(i%10)*.4,2+Math.floor(i/10)*.42,.3,.3,C.purpleL,C.purple);t('40 次同形状 Query 投影',3.5,3.9,5.3,.4,20,C.purple,true);
 stage(3);t('≈ 3.8797 GFLOPs',3.5,4.4,5.5,.55,31,C.teal,true);note('正常主干 Decode；只含 Query 的两个 Linear，其余模块仍需另计。',1.6);
}
function forwardGraph(c,y=2.6){const{box,arrow,t}=c;
 const nodes=[['X',.5,.9,C.blueL,C.blue],['W_QA',2,1.25,C.purpleL,C.purple],['RMSNorm',4,1.6,C.white,C.muted],['W_QB',6.4,1.25,C.purpleL,C.purple],['Q',8.6,.9,C.tealL,C.teal]];
 nodes.forEach(([v,x,w,f,co],i)=>{box(v,x,y,w,.7,f,co,18);if(i<4)arrow(x+w+.05,nodes[i+1][1]-.1,y+.35,C.muted);});
}
function backward(c){const{t,box,line,stage,note}=c;
 stage(1);forwardGraph(c);t('Forward →',.5,1.75,2.4,.4,22,C.blue,true);t('保存 X、H、H′ 与各层权重',2.9,1.78,6.6,.35,18,C.muted);
 stage(2);box('后续模块\nLoss',8.15,4.05,1.3,.65,C.orangeL,C.orange,13.5);line(9.05,3.95,9.05,3.42,C.orange,2,true);t('① dQ',7.8,3.35,1.2,.32,20,C.orange,true);
 stage(3);t('② 取回前向输入与权重',.6,4.6,4.6,.4,20,C.ink,true);
 stage(4);line(8.5,3.8,6.95,3.8,C.orange,3,true);line(6.3,3.8,4.8,3.8,C.orange,3,true);line(4,3.8,2.6,3.8,C.orange,3,true);line(1.95,3.8,.7,3.8,C.orange,3,true);t('dX',.55,3.98,1,.4,22,C.orange,true);t('Backward ←',4.9,4.6,2.85,.4,23,C.orange,true,'right');
}
function gradients(c){const{t,box,arrow,line,stage,note}=c;
 t('H′ [1,1280]',5.7,1.57,2.5,.3,15,C.blue,true);t('X [1,5120]',.65,1.57,2,.3,15,C.blue,true);
 box('W_QA',1.25,2.55,1.5,.65,C.purpleL,C.purple,20);box('RMSNorm',3.9,2.55,1.65,.65,C.white,C.muted,18);box('W_QB',6.8,2.55,1.5,.65,C.purpleL,C.purple,20);
 stage(1);line(9.4,2.875,8.4,2.875,C.orange,3,true);t('dQ',8.65,2.18,.9,.35,22,C.orange,true);
 stage(2);line(7.55,2.5,7.55,2.27,C.pink,2.5,true);t('③ dW_QB = H′ᵀ dQ',5.6,1.9,3.9,.35,19,C.pink,true);t('[1280,1] × [1,32768] → [1280,32768]',5.5,3.6,4,.35,13.5,C.pink);
 stage(3);line(6.7,2.875,5.65,2.875,C.orange,3,true);t('④ dH′ = dQ W_QBᵀ',5.65,4.12,3.9,.35,18,C.orange,true);t('[1,32768] × [32768,1280] → [1,1280]',5.5,4.5,4,.25,13,C.orange);
 stage(4);line(3.8,2.875,2.85,2.875,C.orange,3,true);t('dH′ → dH',3.65,3.25,2.3,.3,17,C.orange,true);
 stage(5);line(2,2.5,2,2.27,C.pink,2.5,true);t('dW_QA = Xᵀ dH',.55,1.9,4.5,.35,19,C.pink,true);t('[5120,1] × [1,1280] → [5120,1280]',.55,3.6,4.5,.35,13.5,C.pink);t('dX = dH W_QAᵀ',.55,4.12,4.3,.35,18,C.orange,true);t('[1,1280] × [1280,5120] → [1,5120]',.55,4.5,4.5,.25,13,C.orange);
 stage(6);line(1.15,2.875,.55,2.875,C.orange,3,true);t('⑤ dX [1,5120] → 前一模块',.6,4.88,8.9,.3,20,C.orange,true,'center');
}
function backwardFlops(c){const{t,box,rect,stage,note}=c;
 stage(1);t('Forward',.6,1.8,2.1,.4,22,C.blue,true);rect(3.2,1.85,2.2,.5,C.blue,C.blue);t('F ≈ 2MKN',5.7,1.85,3.4,.4,22,C.blue,true);
 stage(2);t('Backward',.6,2.75,2.2,.4,22,C.orange,true);rect(3.2,2.8,2.2,.5,C.pink,C.pink);rect(5.5,2.8,2.2,.5,C.orange,C.orange);t('dW',3.6,3.45,1.3,.35,20,C.pink,true);t('dX',6.1,3.45,1.3,.35,20,C.orange,true);
 stage(3);t('Query Linear Backward ≈ 193.99 MFLOPs',.6,4.1,8.8,.55,27,C.orange,true,'center');t('单 token 精确外积计数不同；“两倍”是统一 2MKN 近似。',.6,4.75,8.8,.35,15,C.muted,false,'center');note('前向 + 反向 ≈ 290.98 MFLOPs；不含 RMSNorm、优化器或额外重计算。',1.45);
}
function roofAxes(c,stageNum=1){const{t,line,stage}=c;stage(stageNum);
 const x0=1.2,y0=4.52,w=6.2,h=2.6;
 line(x0,y0,x0+w,y0,C.ink,1.7,true);line(x0,y0,x0,1.72,C.ink,1.7,true);
 t('Performance · TFLOP/s',.6,1.46,5,.3,15,C.ink,true);t('Arithmetic Intensity · FLOPs/Byte',1.2,4.8,6.4,.22,14,C.ink,true);
 const px=a=>x0+(Math.log10(a)+.5)/4*w,py=p=>y0-(Math.log10(p)+.5)/4*h;
 [.5,1,10,100,1000].forEach(a=>{const x=px(a);if(a!==.5)t(String(a),x-.25,4.55,.5,.2,10,C.muted,false,'center');});
 [1,10,100,1000].forEach(p=>t(String(p),.65,py(p)-.08,.44,.2,10,C.muted,false,'right'));
 return {px,py,x0,y0};
}
function roofLines(c,a,bwStage,computeStage,ridgeStage,regionsStage){const{line,t,stage,dot,box}=c;const{px,py}=a;
 stage(bwStage);line(px(.5),py(.5*3.35),px(1200),py(1200*3.35),C.blue,2.8);t('AI × BW',3.05,3.35,2.8,.35,20,C.blue,true);
 stage(computeStage);line(px(.5),py(989),px(1700),py(989),C.purple,2.8);t('989 TFLOP/s',7.45,2.1,2.1,.35,17,C.purple,true);
 stage(ridgeStage);dot(px(989/3.35),py(989),.095,C.teal);line(px(989/3.35),py(989),px(989/3.35),4.5,C.teal,1.3,false,'dash');t('Ridge ≈ 295.2',7.45,2.72,2.15,.35,18,C.teal,true);t('P_peak / BW',7.45,3.13,2,.28,14,C.teal);
 stage(regionsStage);t('Memory-bound',1.55,2.72,3,.35,17,C.blue,true);t('Compute-bound',6,2.82,1.4,.35,14.5,C.purple,true);
}
function roofline(c){const{t,stage,note}=c;const a=roofAxes(c,1);t('AI = FLOPs / Bytes',7.45,1.57,2,.4,15,C.ink,true);roofLines(c,a,2,3,4,5);stage(5);t('P ≤ min(P_peak, AI × BW)',7.45,3.55,2,.8,18,C.ink,true);t('H100 SXM · BF16 Dense：989 TFLOP/s，3.35 TB/s；双对数坐标；理论上界。',.6,5.03,8.8,.16,10,C.muted);}
function roofBatch(c){const{t,stage,line,dot,rect,note}=c;const a=roofAxes(c,0);const{px,py}=a;
 line(px(.5),py(.5*3.35),px(1200),py(1200*3.35),C.blue,2.8);line(px(.5),py(989),px(1700),py(989),C.purple,2.8);line(px(295.22),py(989),px(295.22),4.5,C.teal,1.3,false,'dash');t('Memory-bound',1.55,2.72,3,.35,17,C.blue,true);t('Compute-bound',6,2.82,1.4,.35,14.5,C.purple,true);
 stage(1);t('BF16 · 2 Bytes',7.5,1.65,2,.3,15,C.muted,true);t('权重按批次读一次\nH 写出一次、读回一次',7.5,2.05,2,.65,13.5,C.muted);
 stage(2);dot(px(.99917),py(.99917*3.35),.09,C.blue);t('M=1\n96.99 MFLOPs\n97.07 MB · AI≈1',7.5,2.85,2,1,14.5,C.blue,true);
 stage(3);dot(px(552.303),py(989),.09,C.teal);t('M=1024\n99.32 GFLOPs\n179.83 MB\nAI≈552.3',7.5,3.95,2,1.1,14.5,C.teal,true);
 stage(4);line(px(1.8),py(7),px(180),py(650),C.orange,2,true,'dash');t('M ↑',3.8,2.5,1,.35,22,C.orange,true);t('只含两个 Query Linear；简化流量模型，非实测；M 为一次共同处理的行数。',.6,5.03,8.8,.16,10,C.muted);
}
function mfu(c){const{t,rect,line,stage,note}=c;
 t('Illustrative Example / Not Measured',.55,1.48,9,.35,18,C.orange,true);
 stage(1);t('01  Useful Model FLOPs / Step',.55,2.0,5.8,.35,20,C.blue,true);t('100.6M tokens × 60 GFLOPs/token',.55,2.45,5.9,.35,19,C.ink);t('= 6.036 EFLOPs',6.6,2.38,2.9,.5,24,C.blue,true);
 stage(2);t('02  Total Hardware Peak Throughput',.55,3.0,6.1,.35,20,C.purple,true);t('2048 GPUs × 989.5 TFLOP/s',.55,3.45,5.9,.35,19,C.ink);t('= 2.026496\nEFLOP/s',6.6,3.25,2.9,.7,22,C.purple,true);
 stage(3);t('03  MFU = 6.036 / (6.0 × 2.026496)',.55,4.13,6.8,.35,21,C.teal,true);rect(.55,4.7,6.7,.25,C.grey,C.grey);for(let i=0;i<10;i++){c.delay(i*90);rect(.55+i*6.7*.496423383/10,4.7,6.7*.496423383/10,.25,C.teal,C.teal);}c.delay(900);t('49.6%',7.5,4.12,2,.55,34,C.teal,true);c.delay(0);t('60 GFLOPs/token、2048 GPUs、6.0 s：假设；989.5：示例规格',.55,5.02,9,.18,10,C.muted);
}
function summary(c){const{t,box,arrow,stage,note}=c;
 stage(1);box('Architecture',.55,2.1,2,.75,C.blueL,C.blue,20);arrow(2.65,2.9,2.48);box('Parameters',3,2.1,2,.75,C.purpleL,C.purple,20);t('哪些路径执行\n哪些权重被访问',.6,3.12,4.5,.9,21,C.ink,true);
 stage(2);arrow(5.1,5.35,2.48);box('FLOPs',5.45,2.1,1.6,.75,C.orangeL,C.orange,21);t('前向 + 反向\n声明范围与计数口径',5.45,3.12,4,.9,21,C.ink,true);
 stage(3);arrow(7.15,7.4,2.48);box('Efficiency',7.5,2.1,2,.75,C.tealL,C.teal,21);t('Roofline：潜在瓶颈    MFU：有效训练吞吐 / 总峰值',.55,4.38,8.9,.5,22,C.teal,true,'center');
}
const scenes={architecture,ced,local,indexer,reuse,mhc,moe,swiglu,engram,expert_params:expertParams,active,matmul,matmul_next:matmulNext,lowrank,layers,backward,gradients,backward_flops:backwardFlops,roofline,roofline_batch:roofBatch,mfu,summary};
exports.render=(pptx,s,d,preset)=>{if(!scenes[d.math_scene])throw Error('Unknown scene '+d.math_scene);const c=context(pptx,s,d);scenes[d.math_scene](c);c.finish();};
exports.flush=()=>fs.writeFileSync(path.join(__dirname,process.env.TEACHING_MANIFEST||'scene_manifest.json'),JSON.stringify(manifest,null,2));
