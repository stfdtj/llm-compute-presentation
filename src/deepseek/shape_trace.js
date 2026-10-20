/* Forward only; batch omitted; global shapes before Tensor Parallel partitioning.
 * Logical sparse reads are not materialized KV replication. See shape_trace.md. */
(()=>{'use strict';
 const C=JSON.parse(document.getElementById('shape-config').textContent),$=id=>document.getElementById(id);
 const D=C.hidden_size,H=C.num_attention_heads,d=C.head_dim,R=C.q_lora_rank,G=C.o_groups,O=C.o_lora_rank,F=C.moe_intermediate_size,M=C.hc_mult,E=C.n_routed_experts,J=C.num_experts_per_tok,IH=C.index_n_heads,ID=C.index_head_dim;
 const s=(...a)=>'['+a.join(',')+']',h=s(1,D),r=s(1,M,D),coef=s(1,M),post=s(1,M),comb=s(1,M,M),qr=s(1,R),q=s(1,H,d),kv=s('ΔS',d),cache=s('S',d),idx=s(1,'K');
 const colors={hidden:'#2863c5',hc:'#ae7c1d',local:'#22936b',global:'#cb7916',index:'#a16a18',moe:'#8a4ec2',engram:'#c64d76'},xs=[110,510,910,1310],W=300,NH=126,dy=184;
 let enabled=false,layer=0,graph,selected=null;
 const toggle=document.createElement('button');toggle.id='flow-toggle';toggle.textContent='Shape Trace';toggle.setAttribute('aria-pressed','false');document.querySelector('.modes').after(toggle);
 function makeGraph(L){
  const nodes=[],edges=[],sections=[];let y=86;
  const nearest=arr=>arr.filter(i=>i<=L).at(-1),source=nearest(C.kv_source_layer_ids),indexSource=nearest(C.index_source_layer_ids),ratio=C.compress_ratios[L],owns=C.kv_source_layer_ids.includes(L),scores=C.index_source_layer_ids.includes(L),mode=!ratio?'Local only':owns?'Full':scores?'Reindex':'Reuse';
  function section(name){sections.push({name,y:y-20});y+=64;}
  function node(id,name,col,row,inputs,outputs,type='hidden',op=''){nodes.push({id,name,x:xs[col],y:y+row*dy,w:W,h:NH,inputs,outputs,type,op});return id;}
  function edge(a,b,port='out',input='in',type=null){const A=nodes.find(n=>n.id===a),B=nodes.find(n=>n.id===b);if(!A||!B)throw Error('Missing node '+a+' → '+b);const shape=A.outputs[port];if(!shape||shape!==B.inputs[input])throw Error('Shape mismatch '+a+'.'+port+' → '+b+'.'+input);edges.push({a,b,port,input,shape,type:type||A.type});}
  function chain(spec,col,start,initial,shape,type){let prev=initial,sh=shape;spec.forEach(([id,name,out,op],i)=>{node(id,name,col,start+i,{in:sh},{out},type,op);if(prev)edge(prev,id);prev=id;sh=out;});return prev;}
  function advance(rows){y+=rows*dy;}
  function mixes(id,name,col,row,from){
   chain([[id+'flat',name+' · Flatten',s(1,M*D),'Flatten four streams'],[id+'proj',name+' · W_Mix',s(1,(2+M)*M),'Projection · RMS scale'],[id,name+' · Split',coef,'Split · Sinkhorn']],col,row,from,r,'hc');
   const n=nodes.find(n=>n.id===id);n.outputs={pre:coef,post,comb};
  }
  section('Input / mHC');let input;
  if(L===0){node('ids','Input Token ID',1,0,{}, {out:s(1)},'hidden','Input');input=chain([['embed','Embedding',h,'Lookup'],['expand','Expand four streams',r,'Unsqueeze · Repeat']],1,1,'ids',s(1),'hidden');advance(3);}
  else{input=node('previous',`Layer ${L-1} → Layer ${L}`,1,0,{}, {out:r},'hidden','Hidden states');advance(1);}
  if(C.engram_layer_ids.includes(L)){
   const hc=(C.engram_max_ngram_size-1)*C.engram_n_heads;
   node('hash','Token history → N-gram Hash',2,0,{}, {out:s(1,hc)},'engram','Token IDs · auxiliary');
   chain([['lookup','Engram Lookup',s(1,hc,C.engram_head_dim),'Lookup'],['engramflat','Engram Flatten',s(1,hc*C.engram_head_dim),'Flatten'],['engramkv','Engram W_KV',s(1,(M+1)*D),'Projection']],2,1,'hash',s(1,hc),'engram');
   node('engramsplit','Engram Keys / Shared Value',2,4,{in:s(1,(M+1)*D)},{key:r,value:h},'engram','Split · Reshape');edge('engramkv','engramsplit');
   node('engramgate','Engram Stream Gates',1,5,{in:r,key:r},{out:coef},'engram','Normalized match · sigmoid');edge(input,'engramgate');edge('engramsplit','engramgate','key','key');
   node('engram','Engram Injection',1,6,{in:r,gate:coef,value:h},{out:r},'engram','Broadcast · Element-wise add');edge(input,'engram');edge('engramgate','engram','out','gate');edge('engramsplit','engram','value','value');input='engram';advance(7);
  }
  node('premix',L===0?'Identity Pre Mix':`Layer ${L-1} FFN Pre Mix`,0,0,{}, {out:coef},'hc','Coefficient input');
  mixes('mixA','Attention Mix',2,0,input);
  node('preA','mHC Pre · Attention',1,1,{in:r,mix:coef},{out:h},'hc','Weighted reduce · 4 → 1');edge(input,'preA');edge('premix','preA','out','mix');
  chain([['normA','Attention RMSNorm',h,'Norm']],1,2,'preA',h,'hidden');advance(3);
  section('Attention · Query / Local / Global');
  const qEnd=chain([['qa','W_QA',qr,'Projection'],['qn','Query RMSNorm',qr,'Norm'],['qb','W_QB',s(1,H*d),'Projection'],['qreshape','Query Heads',q,'Reshape'],['qrope',`Query RoPE · tail ${C.qk_rope_head_dim}`,q,'Position-wise']],0,0,'normA',h,'hidden');
  chain([['lproj','Local W_KV',s(1,d),'Projection'],['lnorm','Local KV RMSNorm',s(1,d),'Norm'],['lrope',`Local RoPE · tail ${C.qk_rope_head_dim}`,s(1,d),'Position-wise'],['lquant','Local KV FP8',s(1,d),'Quantize']],1,0,'normA',h,'local');
  node('local','Local Window / Ring Cache',1,4,{in:s(1,d)},{out:s(C.sliding_window,d),indices:s(1,C.sliding_window)},'local','Write current · read valid W');edge('lquant','local');
  node('localread','Local KV Read',1,6,{in:s(C.sliding_window,d)},{out:s(C.sliding_window,d)},'local','Cache · Bounded Replay if missing');edge('local','localread');
  if(ratio){
   if(owns){
    if(ratio===2){
     node('cv','Compressor W_KV',2,0,{in:h},{out:s(1,d)},'global','Projection');edge('normA','cv');
     node('cg','Compressor W_Gate',3,0,{in:h},{out:s(1,d)},'global','Projection');edge('normA','cg');
     node('pair','Compression Group Buffers',2,1,{in:s(1,d),gate:s(1,d)},{values:s(2,d),gates:s(2,d)},'global','Write slot · retain earlier token');edge('cv','pair');edge('cg','pair','out','gate');
     node('pool','2 → 1 Channel-wise Pool',2,2,{in:s(2,d),gate:s(2,d)},{out:kv},'global','Softmax gate · weighted reduce');edge('pair','pool','values');edge('pair','pool','gates','gate');
     node('latent','Compressed RMSNorm',2,3,{in:kv},{out:kv},'global','Norm · ΔS = 0 or 1');edge('pool','latent');
    }else chain([['cv','Compressor W_KV',s(1,d),'Projection'],['latent','Compressor RMSNorm',kv,'Norm · ratio 1: ΔS = 1']],2,0,'normA',h,'global');
    chain([['grope','Global KV RoPE',kv,'Position-wise'],['gquant','Global KV FP4',kv,'Quantize']],2,4,'latent',kv,'global');
    node('global',`Global KV Cache · Layer ${source}`,2,6,{in:kv},{out:cache},'global','Append ΔS rows');edge('gquant','global');
    chain([['ik','Indexer W_K',s('ΔS',ID),'Projection'],['iknorm','Index Key Norm / RoPE / FP4',s('ΔS',ID),'Norm · Position · Quantize']],3,4,'latent',kv,'index');
    node('keycache',`Index Keys Cache · Layer ${source}`,3,6,{in:s('ΔS',ID)},{out:s('S',ID)},'index','Append · before Global RoPE');edge('iknorm','keycache');
   }else{
    node('global',`Read Global KV · Layer ${source}`,2,6,{}, {out:cache},'global','Cross-layer cache reuse');
    if(scores)node('keycache',`Read Index Keys · Layer ${source}`,3,6,{}, {out:s('S',ID)},'index','Cross-layer cache reuse');
   }
  }
  advance(7);let top;
  if(ratio){section(`Indexer · ${mode}`);
   if(scores){
    chain([['iqproj','Indexer W_Q',s(1,IH*ID),'Projection'],['iqreshape','Indexer Query Heads',s(1,IH,ID),'Reshape'],['iqrope','Indexer Query RoPE / FP4',s(1,IH,ID),'Position · Quantize']],0,0,'qn',qr,'index');
    node('iweights','Indexer Head Weights',1,2,{in:h},{out:s(1,IH)},'index','Projection · scale');edge('normA','iweights');
    node('dot','Indexer Q · Keys',2,3,{query:s(1,IH,ID),keys:s('S',ID)},{out:s(1,IH,'S')},'index','Dot product');edge('iqrope','dot','out','query');edge('keycache','dot','out','keys');
    node('score','Indexer Score Vector',2,4,{in:s(1,IH,'S'),weights:s(1,IH)},{out:s(1,'S')},'index','ReLU · weight · head reduce');edge('dot','score');edge('iweights','score','out','weights');
    if(L===20)node('candidate','Candidate Mask → Reindex',3,7,{in:s(1,'S')},{out:s(1,'S')},'index',`${C.candidate_block_size} positions/block · ≤${C.candidate_topk_blocks} blocks`);
    if(!owns)node('candidate','Candidate Mask · Layer 20',3,5,{}, {out:s(1,'S')},'index','Cross-layer mask reuse');
    node('mask','Visible Position Mask',2,6,{in:s(1,'S'),...(!owns?{candidate:s(1,'S')}: {})},{out:s(1,'S')},'index',owns?'Causal mask':'Causal + Candidate mask');edge('score','mask');if(!owns)edge('candidate','mask','out','candidate');
    if(L===20)edge('mask','candidate');
    node('top','Top-K Global Positions',2,7,{in:s(1,'S')},{out:idx},'index','Top-K · sort · add Local offset');edge('mask','top');top='top';advance(8);
   }else{node('top',`Reuse Top-K · Layer ${indexSource}`,2,0,{}, {out:idx},'index','Current token · prior index layer');top='top';advance(1);}
  }
  section(ratio?'CSA2 / Main Attention':'SWA / Main Attention');
  const allKV=ratio?s(`${C.sliding_window}+S`,d):s(C.sliding_window,d),allIdx=ratio?s(1,`${C.sliding_window}+K`):s(1,C.sliding_window);
  node('kvcat',ratio?'Local + Global KV':'Local KV',1,0,{in:s(C.sliding_window,d),...(ratio?{global:cache}:{})},{out:allKV},'local',ratio?'Concat · position axis':'Cache view');edge('localread','kvcat');if(ratio)edge('global','kvcat','out','global');
  node('icat','Read Position Indices',2,1,{local:s(1,C.sliding_window),...(ratio?{global:idx}:{})},{out:allIdx},'index',ratio?'Concat · index axis':'Window indices · invalid -1');edge('local','icat','indices','local');if(ratio)edge(top,'icat','out','global');
  node('gather','KV Gather · inside sparse_attn',1,2,{in:allKV,indices:allIdx},{out:s('N',d)},'local','Logical read · no 64× KV copy');edge('kvcat','gather');edge('icat','gather','out','indices');
  node('attention',ratio?'CSA2 · Attention Scores':'SWA · Attention Scores',1,3,{query:q,kv:s('N',d)},{out:s(1,H,'N')},'hidden','Dot product · logical fused stage');edge(qEnd,'attention','out','query');edge('gather','attention','out','kv');
  node('softmax','Attention Weights',1,4,{in:s(1,H,'N')},{out:s(1,H,'N')},'hidden','Joint Softmax · scale / mask / sink');edge('attention','softmax');
  node('attnout','Attention Output',1,5,{in:s(1,H,'N'),values:s('N',d)},{out:q},'hidden','Weighted reduce · fused stage');edge('softmax','attnout');edge('gather','attnout','out','values');
  chain([['inverse','Inverse RoPE',q,'Position-wise'],['ogroup','Output Groups',s(1,G,H*d/G),'Reshape'],['oquant','Output FP8',s(1,G,H*d/G),'Quantize'],['oa','Grouped W_OA',s(1,G,O),'Grouped projection'],['oflat','Flatten Groups',s(1,G*O),'Flatten'],['ob','W_OB',h,'Projection']],1,6,'attnout',q,'hidden');advance(12);
  section('mHC Post / FFN Pre');
  node('postA','mHC Post · Attention',1,0,{in:h,residual:r,post,comb},{out:r},'hc','Residual mix · broadcast write');edge('ob','postA');edge(input,'postA','out','residual');edge('mixA','postA','post','post');edge('mixA','postA','comb','comb');
  mixes('mixF','FFN Mix',2,0,'postA');
  node('preF','mHC Pre · FFN',1,2,{in:r,mix:coef},{out:h},'hc','Weighted reduce · 4 → 1');edge('postA','preF');edge('mixA','preF','pre','mix');chain([['normF','FFN RMSNorm',h,'Norm']],1,3,'preF',h,'hidden');advance(4);
  section('MoE · Routed / Shared');
  chain([['route','Router',s(1,E),'Projection · temperature scale'],['routeScore','Routing Scores',s(1,E),'sqrtsoftplus · raw scores'],['routeBias','Expert Selection Scores',s(1,E),'Add bias · selection only']],0,0,'normF',h,'moe');
  node('routeTop',`Select ${J} / ${E} Experts`,0,3,{in:s(1,E)},{ids:s(1,J)},'moe','Top-K expert IDs');edge('routeBias','routeTop');
  node('routeWeights','Selected Expert Weights',0,4,{in:s(1,E),ids:s(1,J)},{out:s(1,J)},'moe','Gather raw · normalize · scale');edge('routeScore','routeWeights');edge('routeTop','routeWeights','ids','ids');
  node('dispatch',`Dispatch · ${J} Selected Experts`,1,4,{in:h,ids:s(1,J)},{out:h},'moe','Each selected expert · one token');edge('normF','dispatch');edge('routeTop','dispatch','ids','ids');
  for(const [prefix,col,init,start] of [['e',1,'dispatch',5],['s',3,'normF',5]]){
   const upcol=col===1?0:2;
   node(prefix+'gate',prefix==='e'?'Expert W_Gate':'Shared W_Gate',col,start,{in:h},{out:s(1,F)},'moe','Projection');edge(init,prefix+'gate');
   node(prefix+'up',prefix==='e'?'Expert W_Up':'Shared W_Up',upcol,start,{in:h},{out:s(1,F)},'moe','Projection');edge(init,prefix+'up');
   node(prefix+'act','Clamp + SiLU',col,start+1,{in:s(1,F)},{out:s(1,F)},'moe','Element-wise · gate');edge(prefix+'gate',prefix+'act');
   node(prefix+'clamp','Clamp Up',upcol,start+1,{in:s(1,F)},{out:s(1,F)},'moe','Element-wise');edge(prefix+'up',prefix+'clamp');
   node(prefix+'mul','SwiGLU Multiply',col,start+2,{in:s(1,F),up:s(1,F)},{out:s(1,F)},'moe','Element-wise multiply');edge(prefix+'act',prefix+'mul');edge(prefix+'clamp',prefix+'mul','out','up');
   if(prefix==='e'){node('weight','Selected Route Weight',0,start+2,{in:s(1,J)},{out:s(1,1)},'moe','Select this expert weight');edge('routeWeights','weight');node('weighted','Apply Route Weight',col,start+3,{in:s(1,F),weight:s(1,1)},{out:s(1,F)},'moe','Broadcast multiply · before Down');edge('emul','weighted');edge('weight','weighted','out','weight');}
   node(prefix+'down',prefix==='e'?'Expert W_Down':'Shared W_Down',col,start+4,{in:s(1,F)},{out:h},'moe','Projection');edge(prefix==='e'?'weighted':'smul',prefix+'down');
  }
  node('sum','MoE Output',1,10,{routed:h,shared:h},{out:h},'moe',`Sum ${J} routed + ${C.n_shared_experts} shared`);edge('edown','sum','out','routed');edge('sdown','sum','out','shared');advance(11);
  section(L===39?'Final Output':'Next Layer');
  node('postF','mHC Post · FFN',1,0,{in:h,residual:r,post,comb},{out:r},'hc','Residual mix · broadcast write');edge('sum','postF');edge('postA','postF','out','residual');edge('mixF','postF','post','post');edge('mixF','postF','comb','comb');
  if(L<39){node('next',`Layer ${L+1} Input`,1,2,{in:r,mix:coef},{out:r},'hidden','Four streams + FFN Pre Mix');edge('postF','next');edge('mixF','next','pre','mix');advance(3);}
  else{node('collapse','Final mHC Pre',1,2,{in:r,mix:coef},{out:h},'hc','Weighted reduce · 4 → 1');edge('postF','collapse');edge('mixF','collapse','pre','mix');chain([['finalnorm','Final RMSNorm',h,'Norm'],['head','LM Head',s(1,C.vocab_size),'Projection'],['logits','Logits',s(1,C.vocab_size),'Vocabulary scores']],1,3,'collapse',h,'hidden');advance(6);}
  if(C.dspark_target_layer_ids.includes(L)){const n=nodes.find(n=>n.id===input);nodes.push({id:'drafttap',name:'DSpark · auxiliary hidden tap',x:xs[3],y:n.y,w:W,h:NH,inputs:{in:r},outputs:{out:h},type:'hidden',op:'Mean four streams → draft only'});edge(input,'drafttap');}
  return {nodes,edges,sections,height:y+40,layer:L,source,indexSource,ratio,owns,scores,mode};
 }
 const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function render(){graph=makeGraph(layer);selected=null;const {nodes,edges,sections,height}=graph,byId=new Map(nodes.map(n=>[n.id,n]));
  const bundles=new Map();for(const [i,e]of edges.entries()){const key=e.a+'→'+e.b;if(!bundles.has(key))bundles.set(key,[]);bundles.get(key).push(i);}
  const incoming=new Map(),outgoing=new Map();for(const [key,indices]of bundles){const e=edges[indices[0]];if(!incoming.has(e.b))incoming.set(e.b,[]);incoming.get(e.b).push(key);const signature=indices.map(j=>edges[j].port).join('+');e.bus=e.a+':'+signature;if(!outgoing.has(e.a))outgoing.set(e.a,[]);if(!outgoing.get(e.a).includes(e.bus))outgoing.get(e.a).push(e.bus);}
  const portX=(n,list,key)=>n.x+W/2+(list.indexOf(key)-(list.length-1)/2)*50;
  const xgrid=[...new Set([40,80,...xs.flatMap(x=>[x-24,x+W/2,x+W+24]),...edges.flatMap(e=>{const bundle=bundles.get(e.a+'→'+e.b),rep=edges[bundle[0]];return[portX(byId.get(e.a),outgoing.get(e.a),rep.bus),portX(byId.get(e.b),incoming.get(e.b),e.a+'→'+e.b)];}),1660,1720])].sort((a,b)=>a-b),ygrid=[...new Set([35,height-20,...nodes.flatMap(n=>[n.y-18,n.y+NH+18]),...sections.map(t=>t.y+15)])].sort((a,b)=>a-b);
  const blocked=(x,y)=>nodes.some(n=>x>n.x-10&&x<n.x+W+10&&y>n.y-12&&y<n.y+NH+12)||sections.some(t=>x>90&&x<120+t.name.length*9&&y>t.y-25&&y<t.y+11),used=new Map(),labels=[];
  function route(a,b,e){
   const sx=portX(a,outgoing.get(a.id),e.bus),sy=a.y+NH+18,tx=portX(b,incoming.get(b.id),a.id+'→'+b.id),ty=b.y-18,ix=xgrid.indexOf(sx),iy=ygrid.indexOf(sy),jx=xgrid.indexOf(tx),jy=ygrid.indexOf(ty),cols=xgrid.length;
   const start=iy*cols+ix,goal=jy*cols+jx,dist=new Map([[start,0]]),prior=new Map(),open=[start],closed=new Set();
   while(open.length){let bi=0;for(let i=1;i<open.length;i++)if(dist.get(open[i])<dist.get(open[bi]))bi=i;const id=open.splice(bi,1)[0];if(id===goal)break;if(closed.has(id))continue;closed.add(id);const x=id%cols,y=Math.floor(id/cols);
    for(const [nx,ny]of [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]){if(nx<0||nx>=cols||ny<0||ny>=ygrid.length)continue;const X=xgrid[nx],Y=ygrid[ny],midX=(X+xgrid[x])/2,midY=(Y+ygrid[y])/2;if(blocked(X,Y)||blocked(midX,midY))continue;const next=ny*cols+nx,key=[id,next].sort((a,b)=>a-b).join('-'),cost=dist.get(id)+Math.abs(X-xgrid[x])+Math.abs(Y-ygrid[y])+(used.has(key)&&used.get(key)!==e.bus?100000:0)+8;if(cost<(dist.get(next)??Infinity)){dist.set(next,cost);prior.set(next,id);open.push(next);}}
   }
   if(!dist.has(goal))throw Error('No wire route '+a.id+' → '+b.id);
   const path=[goal];while(path.at(-1)!==start)path.push(prior.get(path.at(-1)));path.reverse();for(let i=1;i<path.length;i++){const key=[path[i-1],path[i]].sort((a,b)=>a-b).join('-');used.set(key,e.bus);}
   const points=[[sx,a.y+NH],...path.map(id=>[xgrid[id%cols],ygrid[Math.floor(id/cols)]]),[tx,b.y]];
   return points.filter((p,i)=>i===0||i===points.length-1||(points[i-1][0]!==p[0]||points[i+1][0]!==p[0])&&(points[i-1][1]!==p[1]||points[i+1][1]!==p[1]));
  }
  function intersects(a,b){return a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;}
  let wires='',wireLabels='';for(const [i,e]of edges.entries()){const bundle=bundles.get(e.a+'→'+e.b);if(i!==bundle[0])continue;const A=byId.get(e.a),B=byId.get(e.b),points=route(A,B,e);e.points=points;let label=null;
   const labelText=bundle.length===1?e.shape:bundle.map(j=>edges[j].port+':'+edges[j].shape).join(' · '),width=labelText.length*6.8+12;
   const segments=points.slice(1).map((p,j)=>[points[j],p]).sort((a,b)=>(Math.abs(b[1][1]-b[0][1])+Math.abs(b[1][0]-b[0][0]))-(Math.abs(a[1][1]-a[0][1])+Math.abs(a[1][0]-a[0][0])));
   for(const [p,t]of segments){for(const fraction of [.5,.3,.7,.15,.85]){const x=p[0]+(t[0]-p[0])*fraction,y=p[1]+(t[1]-p[1])*fraction,candidate={x:x-width/2,y:y-9,w:width,h:18};if(!nodes.some(n=>intersects(candidate,n))&&!labels.some(l=>intersects(candidate,l))&&!sections.some(s=>Math.abs(y-s.y)<20)){label=candidate;break;}}if(label)break;}
   if(label)labels.push(label);for(const j of bundle){edges[j].label=label;edges[j].points=points;}
   const path=points.map((p,j)=>(j?'L':'M')+p.join(' ')).join(' ');
   wires+=`<g class="shape-wire" data-edge="${i}"><path d="${path}" fill="none" stroke="white" stroke-width="5"/><path d="${path}" fill="none" stroke="${colors[e.type]}" stroke-width="1.8" ${['global','index','hc'].includes(e.type)?'stroke-dasharray="5 3"':''} marker-end="url(#st-${e.type})"/><title>${esc(A.name+' → '+B.name+': '+labelText)}</title></g>`;
   if(label)wireLabels+=`<g class="shape-wire" data-edge="${i}"><rect x="${label.x}" y="${label.y}" width="${label.w}" height="18" rx="4" fill="white"/><text x="${label.x+label.w/2}" y="${label.y+13}" text-anchor="middle" font-size="12" fill="${colors[e.type]}">${esc(labelText)}</text></g>`;
  }
  function shapeText(n,field){const entries=Object.entries(n[field]);return entries.length?entries.map(([k,v])=>k==='in'||k==='out'?v:k+':'+v).join(' · '):'—';}
  function lines(t,max=34){if(t.length<=max)return[t];const parts=t.split(' · '),out=[''];for(const p of parts){if(out.at(-1).length+p.length>max)out.push(p);else out[out.length-1]+=(out.at(-1)?' · ':'')+p;}return out;}
  const boxes=nodes.map(n=>{const c=colors[n.type],input=lines(shapeText(n,'inputs')),output=lines(shapeText(n,'outputs'));return`<g class="shape-node" data-node="${n.id}" role="button" tabindex="0" aria-label="${esc(n.name+'; Input '+shapeText(n,'inputs')+'; Output '+shapeText(n,'outputs'))}"><rect x="${n.x}" y="${n.y}" width="${W}" height="${NH}" rx="8" fill="${c}0b" stroke="${c}"/><text x="${n.x+13}" y="${n.y+20}" font-size="11" fill="${c}">${esc(n.op)}</text><text x="${n.x+13}" y="${n.y+43}" font-size="15" font-weight="650" fill="#203454">${esc(n.name)}</text>${input.map((t,i)=>`<text x="${n.x+13}" y="${n.y+63+i*16}" font-size="14" fill="#526986">${i?'':'IN  '}${esc(t)}</text>`).join('')}${output.map((t,i)=>`<text x="${n.x+13}" y="${n.y+100+i*16}" font-size="14" fill="${c}">${i?'':'OUT '}${esc(t)}</text>`).join('')}</g>`;}).join('');
  $('shape-scroll').innerHTML=`<svg id="shape-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1760 ${height}" role="group" aria-label="Layer ${layer} Forward Shape Trace"><defs>${Object.entries(colors).map(([k,c])=>`<marker id="st-${k}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="${c}"/></marker>`).join('')}</defs>${sections.map(t=>`<g data-section="${esc(t.name)}"><rect x="85" y="${t.y-23}" width="1540" height="32" rx="5" fill="#f2f5fa"/><text x="100" y="${t.y-1}" font-size="17" font-weight="650" fill="#304a70">${esc(t.name)}</text></g>`).join('')}${wires}${wireLabels}${boxes}</svg>`;
  $('shape-context').textContent=`${layer<20?'Causal Encoder':'Causal Decoder'} · Layer ${layer} · ${graph.mode}${graph.ratio?' · Global KV source L'+graph.source+' · Index source L'+graph.indexSource:''}`;
  $('shape-symbols').textContent=`[1,…]：1 Token，省略 batch。W≤${C.sliding_window}：有效 Local；Ring 固定 ${C.sliding_window} 行。S：Global 行数；ΔS：新增行，0 表示跳过、无 latent。K=min(${C.index_topk},S)，可含 -1；N：有效读取位置。Gather / Scores 是 fused kernel 内的逻辑形状；Bounded Replay 仅恢复缺失的 SWA Cache。`;
  $('shape-layer').value=layer;$('shape-prev').disabled=layer===0;$('shape-next').disabled=layer===39;
  const jumps=[sections[0],sections.find(t=>t.name.startsWith('Attention ·')),sections.find(t=>t.name.startsWith('Indexer')),sections.find(t=>t.name.includes('Main Attention')),sections.find(t=>t.name.startsWith('mHC Post')),sections.find(t=>t.name.startsWith('MoE')),sections.at(-1)].filter(Boolean);
  $('shape-nav').replaceChildren(...jumps.map(t=>{const b=document.createElement('button');b.textContent=t.name.startsWith('Attention ·')?'Q / KV':t.name.includes('Main Attention')?'Attention':t.name.startsWith('Indexer')?'Indexer':t.name.startsWith('mHC Post')?'mHC Post':t.name.startsWith('MoE')?'MoE':t.name;b.onclick=()=>jump(t.y-32);return b;}));$('shape-status').textContent='点击模块，追踪上下游';centerTrace();
 }
 function jump(y){const el=$('shape-scroll'),svg=$('shape-svg');el.scrollTo({top:y*svg.getBoundingClientRect().width/1760,behavior:'instant'});}
 function centerTrace(){const el=$('shape-scroll'),svg=$('shape-svg');if(svg)el.scrollLeft=Math.max(0,(xs[1]+W/2)*svg.getBoundingClientRect().width/1760-el.clientWidth/2);}
 let lastWidth=0;new ResizeObserver(()=>{const width=$('shape-scroll').clientWidth;if(width&&width!==lastWidth){lastWidth=width;centerTrace();}}).observe($('shape-scroll'));
 function highlight(id){selected=id;const related=new Set(id?[id]:graph.nodes.map(n=>n.id)),edgeSet=new Set();if(id){for(const direction of ['a','b']){const visit=[id],seen=new Set(visit);while(visit.length){const n=visit.shift();graph.edges.forEach((e,i)=>{if(e[direction]!==n)return;edgeSet.add(i);const next=e[direction==='a'?'b':'a'];related.add(next);if(!seen.has(next)){seen.add(next);visit.push(next);}});}}}
  document.querySelectorAll('#shape-svg .shape-node').forEach(n=>{n.classList.toggle('dim',!related.has(n.dataset.node));n.classList.toggle('chosen',n.dataset.node===id);});document.querySelectorAll('#shape-svg .shape-wire').forEach(n=>n.classList.toggle('dim',id!==null&&!edgeSet.has(+n.dataset.edge)));$('shape-status').textContent=id?graph.nodes.find(n=>n.id===id).name:'点击模块，追踪上下游';
 }
 function activate(on){enabled=on;toggle.setAttribute('aria-pressed',String(on));document.body.classList.toggle('flow-on',on);closePanel();if(on){render();$('shape-scroll').scrollTop=0;}}
 for(let i=0;i<C.num_hidden_layers;i++){const o=document.createElement('option');o.value=i;o.textContent=`${i} · ${i<20?'Encoder':'Decoder'}`;$('shape-layer').append(o);}
 toggle.addEventListener('click',()=>activate(!enabled));$('shape-layer').addEventListener('change',()=>{layer=+$('shape-layer').value;render();$('shape-scroll').scrollTop=0;});$('shape-prev').onclick=()=>{if(layer){layer--;render();$('shape-scroll').scrollTop=0;}};$('shape-next').onclick=()=>{if(layer<39){layer++;render();$('shape-scroll').scrollTop=0;}};$('shape-clear').onclick=()=>highlight(null);
 $('shape-scroll').addEventListener('click',e=>{const n=e.target.closest('[data-node]');if(n)highlight(selected===n.dataset.node?null:n.dataset.node);});$('shape-scroll').addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.dataset.node){e.preventDefault();highlight(e.target.dataset.node);}});
 document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{if(enabled)activate(false);}));
 // Keep the existing calculation-panel shortcut, now targeting the shape graph.
 const oldDraw=drawPanel;drawPanel=function(...args){oldDraw(...args);if(['global','hidden','multi','localstates'].includes(selectedModule())){const key=selectedModule(),b=document.createElement('button');b.className='flow-entry';b.textContent='在 Shape Trace 中追踪 →';b.style.marginBottom='12px';b.onclick=()=>{layer=key==='hidden'?0:20;if(!enabled)toggle.click();else render();const target=key==='global'?'global':key==='multi'?'attention':key==='localstates'?'local':'preA';jump(graph.nodes.find(n=>n.id===target).y-35);highlight(target);};body.prepend(b);}};
 function selectedModule(){return architectureApp.state.selected;}
 window.informationFlow={get state(){return{enabled,step:1,layer}},steps:[]};window.shapeTrace={get graph(){return graph},makeGraph,highlight,jump};
})();
