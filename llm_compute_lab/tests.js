'use strict';
const assert=require('node:assert/strict');
const C=require('./model.js');
let passed=0;
function test(name,fn){fn();passed++;console.log('PASS '+name);}
const small={...C.defaults,D:4,H:2,Hkv:1,dk:2,dv:3,S:2,W:2,topk:2,block:2,di:2,rank:2,qrank:2,rope:2};
test('small matrix [2,3] × [3,2]: 24 approximate, 20 exact',()=>assert.deepEqual(C.matmul(2,3,2),{approx:24,exact:20}));
test('length-one dot / outer: 2 approximate vs 1 exact per element',()=>assert.deepEqual(C.matmul(2,1,3),{approx:12,exact:6}));
test('manual MHA: projection=160, core=88, total=248, exact=218',()=>{const r=C.calculate('MHA',small);assert.equal(r.projection,160);assert.equal(r.core,88);assert.equal(r.total,248);assert.equal(r.exactKnown,218);});
test('manual GQA: projection=120, unchanged core=88',()=>{const r=C.calculate('GQA',small);assert.equal(r.projection,120);assert.equal(r.core,88);assert.equal(r.total,208);});
test('Hkv=H degeneracy over 80 valid configurations',()=>{for(let i=1;i<=80;i++){const p={...C.defaults,D:128,H:4,Hkv:4,dk:i,dv:i+1,S:i*13,W:100,rank:32,qrank:64};const a=C.calculate('MHA',p),b=C.calculate('GQA',p);for(const k of ['projection','core','total','exactKnown','memoryBytes'])assert.equal(a[k],b[k]);}});
test('GQA reductions apply to KV projections and capacity, not scores',()=>{const p={...small,Hkv:2},a=C.calculate('GQA',p),b=C.calculate('GQA',small);assert.equal(a.core,b.core);assert.equal(a.memoryBytes,2*b.memoryBytes);assert.equal(a.projection-b.projection,40);});
test('SWA equals GQA within window, saturates beyond window',()=>{const p={...small,W:3};assert.equal(C.calculate('SWA',p).total,C.calculate('GQA',p).total);assert.equal(C.calculate('SWA',small).active,2);assert.equal(C.calculate('SWA',{...small,S:999}).total,C.calculate('SWA',small).total);});
test('history S=0 means visible N=1; current token is counted',()=>{const r=C.calculate('MHA',{...small,S:0});assert.equal(r.visible,1);assert.equal(r.core,28);assert.equal(r.memoryBytes,20);});
test('MSA forced-local block accounting, including partial last block',()=>{assert.equal(C.calculate('MSA',small).active,3);assert.equal(C.calculate('MSA',{...small,topk:1}).active,1);assert.equal(C.calculate('MSA',{...small,S:3}).active,4);assert.equal(C.calculate('MSA',{...small,S:4}).active,3);});
test('manual MSA including index scan: projection=152, core=88, extra=15',()=>{const r=C.calculate('MSA',small);assert.equal(r.projection,152);assert.equal(r.core,88);assert.equal(r.extra,15);assert.equal(r.total,255);assert.equal(r.memoryBytes,42);});
test('MSA index scan grows even when main branch is capped',()=>{const a=C.calculate('MSA',{...small,S:20}),b=C.calculate('MSA',{...small,S:40});assert.equal(a.core,b.core);assert.equal(b.extra-a.extra,100);assert.ok(b.memoryBytes>a.memoryBytes);});
test('manual latent MLA: projections=168, core=106, memory=24',()=>{const r=C.calculate('MLA',small);assert.equal(r.projection,168);assert.equal(r.core,106);assert.equal(r.total,274);assert.equal(r.memoryBytes,24);});
test('MLA latent-space algebra equals expanded K/V, each head',()=>{
  const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0),times=(v,w)=>w[0].map((_,j)=>dot(v,w.map(row=>row[j])));
  const cache=[[1,2],[3,-1],[.5,.75]],q=[.5,-2],prob=[.2,.3,.5];
  for(const uk of [[[1,2],[3,4]],[[-2,.5],[1,-1]]]){
    const qlat=times(q,uk[0].map((_,j)=>uk.map(row=>row[j]))),expanded=cache.map(c=>times(c,uk));
    expanded.forEach((k,i)=>assert.ok(Math.abs(dot(q,k)-dot(qlat,cache[i]))<1e-12));
    const uv=[[1,0,2],[3,-1,.5]],latent=cache[0].map((_,j)=>prob.reduce((v,a,i)=>v+a*cache[i][j],0)),values=cache.map(c=>times(c,uv));
    const output=values[0].map((_,j)=>prob.reduce((v,a,i)=>v+a*values[i][j],0));
    times(latent,uv).forEach((x,i)=>assert.ok(Math.abs(x-output[i])<1e-12));
  }
});
test('GDA recurrent core=96, known projections=240; total remains unknown',()=>{const r=C.calculate('GDA',small);assert.equal(r.core,96);assert.equal(r.total,null);assert.equal(r.extra,null);assert.equal(r.projection,240);assert.equal(r.known,336);assert.equal(r.memoryBytes,48);assert.ok(r.unknown.length>0);});
test('GDA constant core and recurrent state over long history',()=>{const a=C.calculate('GDA',small),b=C.calculate('GDA',{...small,S:1048576});assert.equal(a.core,b.core);assert.equal(a.memoryBytes,b.memoryBytes);});
test('GDA factored update equals paper Eq.10; 32 approximate / 28 exact',()=>{
  // One head, dk=dv=2. Explicit recurrence: decay 4, matvec 6 exact,
  // residual/gate 4, outer+add 8, output matvec 6 = 28 exact, 32 approximate.
  const state=[[1,2],[3,4]],k=[.5,-.25],v=[2,-1],q=[1,-.5],alpha=.8,beta=.4;
  const A=state.map(row=>row.map(x=>alpha*x)),u=A.map(row=>row.reduce((s,x,j)=>s+x*k[j],0)),delta=v.map((x,i)=>beta*(x-u[i]));
  const factored=A.map((row,i)=>row.map((x,j)=>x+delta[i]*k[j]));
  const paper=state.map((row,i)=>row.map((_,j)=>row.reduce((s,x,l)=>s+alpha*x*((l===j?1:0)-beta*k[l]*k[j]),0)+beta*v[i]*k[j]));
  factored.forEach((row,i)=>row.forEach((x,j)=>assert.ok(Math.abs(x-paper[i][j])<1e-12)));
  const p={...small,H:1,Hkv:1,dk:2,dv:2},r=C.calculate('GDA',p);assert.equal(r.core,32);assert.equal(r.rows.filter(x=>x.category==='core').reduce((a,x)=>a+x.exact,0),28);
  const output=factored.map(row=>row.reduce((s,x,j)=>s+x*q[j],0));assert.ok(output.every(Number.isFinite));
});
test('scalar toggle consistently removes scale/softmax/state scalar operations',()=>{const a=C.calculate('MHA',{...small,scalar:false});assert.equal(a.core,60);const g=C.calculate('GDA',{...small,scalar:false});assert.equal(g.core,48);assert.ok(g.rows.filter(x=>x.kind==='scalar').every(x=>x.excluded&&x.approx===0));});
test('batch scales capacity and batch total; per-token-per-sequence unchanged',()=>{const a=C.calculate('GQA',small),b=C.calculate('GQA',{...small,batch:3});assert.equal(a.total,b.total);assert.equal(b.batchFlops,3*a.total);assert.equal(b.memoryBytes,3*a.memoryBytes);});
test('precision affects cache, GDA state precision is independent',()=>{const a=C.calculate('MLA',small),b=C.calculate('MLA',{...small,precision:'FP8'});assert.equal(a.total,b.total);assert.equal(a.memoryBytes,2*b.memoryBytes);assert.equal(C.calculate('GDA',{...small,precision:'FP8'}).memoryBytes,48);assert.equal(C.calculate('GDA',{...small,statePrecision:'BF16'}).memoryBytes,24);});
test('validation rejects incompatible, malformed and unsafe configurations',()=>{for(const change of [{H:3,Hkv:2},{S:-1},{D:1.5},{rope:3},{rank:9},{precision:'INT8'},{batch:0},{topk:0},{S:Infinity},{dk:NaN},{D:65536,H:65536,Hkv:65536,dk:65536,dv:65536,rank:65536,qrank:65536,batch:1024}])assert.ok(C.validate({...small,...change}).length,JSON.stringify(change));assert.throws(()=>C.calculate('MSA',{...small,block:0}));});
test('all totals equal row sums; finite outputs over varied inputs',()=>{for(const S of [0,1,127,128,129,8192,1048576])for(const name of C.architectures){const r=C.calculate(name,{...C.defaults,S});assert.equal(r.known,r.rows.reduce((v,s)=>v+s.approx,0));assert.ok(Number.isFinite(r.known));assert.ok(r.memoryBytes>0);}});
test('exports preserve unknown traffic and GDA incomplete total',()=>{const a=C.exportAttention(small),r=C.rooflineInput(small);assert.equal(a.currentTokens,1);assert.equal(a.visibleTokens,3);assert.equal(a.results.find(x=>x.name==='GDA').total,null);assert.equal(r.bytesMoved,null);assert.ok(a.results.every(x=>x.trafficBytes===null&&x.weightTrafficBytes===null));});
test('training schema rejects missing/unverified/invalid throughput data',()=>{const d={architecture:'audit model',source:'local audit',precision:'BF16',forwardFlopsPerToken:12,backwardFlopsPerToken:24,verified:true};assert.equal(C.validateTraining(d).kind,'model-training');assert.throws(()=>C.validateTraining({...d,verified:false}));assert.throws(()=>C.validateTraining({...d,tokensPerSecond:-1}));assert.throws(()=>C.validateTraining({...d,deviceCount:1.5}));assert.throws(()=>C.validateTraining({...d,source:''}));assert.throws(()=>C.validateTraining({...d,schemaVersion:2}));});
console.log(`\n${passed} arithmetic / interface tests passed.`);
