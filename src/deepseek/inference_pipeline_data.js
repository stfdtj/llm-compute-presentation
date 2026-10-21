/* Complete toy distributions, not model inference. All draws are fixed teaching inputs. */
(function(factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else window.PipelineExample=api;})(()=>{
 'use strict';
 const vocab=['A','B','C','D','E','X','Y','Z','先','EOS'];
 const candidates=['A','B','C','D','E'];
 const scenarios={middle:{label:'教学分布 · C 处有分歧',reject:2},all:{label:'教学分布 · 全部匹配',reject:null},first:{label:'教学分布 · A 处有分歧',reject:0},last:{label:'教学分布 · E 处有分歧',reject:4}};
 const sum=a=>a.reduce((s,x)=>s+x,0);
 function distribution(weights){const rest=vocab.filter(v=>!(v in weights)),remaining=1-sum(Object.values(weights));return vocab.map(v=>v in weights?weights[v]:remaining/rest.length);}
 function softmax(logits,temperature=1){if(!(temperature>0))throw new Error('Use argmax for greedy; softmax needs positive temperature');const max=Math.max(...logits),e=logits.map(x=>Math.exp((x-max)/temperature)),z=sum(e);return e.map(x=>x/z);}
 const argmax=a=>a.indexOf(Math.max(...a));
 function sample(prob,u){if(!(u>=0&&u<1))throw new Error('Teaching draw must be in [0,1)');let c=0;for(let i=0;i<prob.length;i++){c+=prob[i];if(u<c)return i;}return prob.length-1;}
 function midpoint(prob,i){return sum(prob.slice(0,i))+prob[i]/2;}
 const log=prob=>prob.map(Math.log);
 const anchor=distribution({'先':.7,A:.1}),bonus=distribution({Z:.55,Y:.3});
 function draftRow(i){
  const token=candidates[i],q=distribution({[token]:.8,X:.03,Y:.1}),base=distribution({[token]:.5,X:.2,Y:.1});
  const baseLogits=log(base),markovBias=q.map((x,j)=>Math.log(x)-baseLogits[j]);
  const logits=baseLogits.map((x,j)=>x+markovBias[j]);
  const probabilities=softmax(logits),draw=midpoint(probabilities,vocab.indexOf(token));
  return {token,previous:i===0?'先':candidates[i-1],context:['我','喜欢','先',...candidates.slice(0,i)],baseLogits,markovBias,logits,q:probabilities,draw,selected:vocab[sample(probabilities,draw)]};
 }
 const draft=candidates.map((_,i)=>draftRow(i));
 function targetRow(i,scenario='middle'){
  const token=candidates[i],reject=scenarios[scenario].reject===i;
  const target=reject?distribution({[token]:.2,X:.45,Y:.25}):distribution({[token]:.6,X:.25,Y:.08});
  const logits=log(target),p=softmax(logits);
  return {input:i===0?'先':candidates[i-1],token,context:draft[i].context,logits,p,top:vocab[argmax(logits)]};
 }
 function residual(p,q){const positive=p.map((x,i)=>Math.max(0,x-q[i])),z=sum(positive);return {positive,mass:z,probabilities:z>0?positive.map(x=>x/z):positive};}
 function evaluate(scenario='middle',algorithm='greedy',length=5){
  if(!scenarios[scenario]||!['greedy','probabilistic'].includes(algorithm)||!Number.isInteger(length)||length<1||length>5)throw new Error('Invalid teaching configuration');
  let rejected=null;const rows=candidates.map((token,i)=>{
   const target=targetRow(i,scenario),proposal=draft[i],j=vocab.indexOf(token),p=target.p[j],q=proposal.q[j],alpha=Math.min(1,p/q),u=.5;
   const matches=target.top===token,pass=algorithm==='greedy'?matches:u<alpha;
   let status='deferred';
   if(i<length){if(rejected!==null)status='discard';else if(pass)status='accept';else{status='reject';rejected=i;}}
   return {index:i,token,input:target.input,context:proposal.context,q:proposal.q,p:target.p,draftLogits:proposal.logits,targetLogits:target.logits,pCandidate:p,qCandidate:q,alpha,u,top:target.top,matches,pass,status,evaluated:status==='accept'||status==='reject'};
  });
  const accepted=rejected===null?length:rejected;
  const replacement=rejected===null?{probabilities:bonus,mass:1,positive:bonus}:residual(rows[rejected].p,rows[rejected].q);
  const draw=rejected===null?.5:.82;
  const extra=algorithm==='greedy'?rejected===null?vocab[argmax(bonus)]:rows[rejected].top:vocab[sample(replacement.probabilities,draw)];
  const committed=['我','喜欢','先',...candidates.slice(0,accepted)],confirmed=[...committed,extra];
  return {scenario,algorithm,length,rows,rejected,accepted,extra,draw,replacement,confirmed,committed,pending:[extra],discarded:candidates.slice(accepted,length),deferred:candidates.slice(length),verificationInput:['先',...candidates.slice(0,length)],bonusContext:['我','喜欢','先',...candidates.slice(0,length)]};
 }
 return {vocab,candidates,scenarios,draft,anchor,bonus,softmax,argmax,sample,residual,evaluate,targetRow};
});
