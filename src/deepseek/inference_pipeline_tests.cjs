const assert=require('node:assert/strict');
const D=require('./inference_pipeline_data.js');
const sum=v=>v.reduce((a,b)=>a+b,0),near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const normalized=p=>{assert.equal(p.length,10);assert.ok(p.every(x=>Number.isFinite(x)&&x>=0));near(sum(p),1);};
normalized(D.anchor);normalized(D.bonus);assert.equal(D.vocab[D.argmax(D.anchor)],'先');assert.equal(D.vocab[D.sample(D.anchor,.6)],'先');
for(let i=0;i<5;i++){
 const d=D.draft[i];normalized(d.q);assert.equal(d.selected,'ABCDE'[i]);assert.equal(d.previous,i===0?'先':'ABCDE'[i-1]);
 const recovered=D.softmax(d.baseLogits.map((x,j)=>x+d.markovBias[j]));recovered.forEach((x,j)=>near(x,d.q[j]));near(d.q[D.vocab.indexOf(d.token)],.8);
}
const expected={middle:2,all:5,first:0,last:4};let checked=0;
for(const [scenario,accepted] of Object.entries(expected))for(const alg of ['greedy','probabilistic']){
 const r=D.evaluate(scenario,alg);assert.equal(r.accepted,accepted);assert.deepEqual(r.rows.map(x=>x.status),Array.from({length:5},(_,i)=>i<accepted?'accept':i===accepted?'reject':'discard'));
 assert.equal(r.extra,accepted===5?'Z':alg==='greedy'?'X':'Y');assert.deepEqual(r.verificationInput,['先','A','B','C','D','E']);
 assert.deepEqual(r.committed,['我','喜欢','先',...'ABCDE'.slice(0,accepted)]);assert.deepEqual(r.confirmed,[...r.committed,r.extra]);assert.deepEqual(r.pending,[r.extra]);
 for(const x of r.rows){normalized(x.p);normalized(x.q);assert.equal(x.input,x.index===0?'先':'ABCDE'[x.index-1]);assert.deepEqual(x.context,['我','喜欢','先',...'ABCDE'.slice(0,x.index)]);assert.equal(D.vocab[D.argmax(x.targetLogits)],x.top);D.softmax(x.targetLogits).forEach((v,j)=>near(v,x.p[j]));
  // Exact categorical identity: accepted draft mass + residual correction mass = p.
  const residual=D.residual(x.p,x.q);normalized(residual.probabilities);const rejectedMass=sum(x.q.map((q,j)=>q*(1-Math.min(1,x.p[j]/q))));
  x.p.forEach((p,j)=>near(x.q[j]*Math.min(1,p/x.q[j])+rejectedMass*residual.probabilities[j],p));
 }
 normalized(r.replacement.probabilities);if(accepted<5){assert.equal(r.rows[accepted].top,'X');near(r.rows[accepted].alpha,.25);assert.equal(r.rows[accepted].u,.5);}checked++;
}
const example=D.evaluate('middle','probabilistic');near(example.rows[0].pCandidate,.6);near(example.rows[0].qCandidate,.8);near(example.rows[0].alpha,.75);assert.equal(example.rows[0].status,'accept');near(example.replacement.probabilities[D.vocab.indexOf('X')],.7);near(example.replacement.probabilities[D.vocab.indexOf('Y')],.25);
for(let n=1;n<=5;n++){const r=D.evaluate('all','probabilistic',n);assert.equal(r.accepted,n);assert.deepEqual(r.verificationInput,['先',...'ABCDE'.slice(0,n)]);}
assert.throws(()=>D.sample(D.anchor,1));assert.throws(()=>D.evaluate('missing'));console.log(`Pipeline data: ${checked} algorithm/scenario combinations, all vectors, Markov composition, sampling, residual mass identity and state boundaries passed.`);
