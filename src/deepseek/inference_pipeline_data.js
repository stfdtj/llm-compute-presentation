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
 // Six lessons share the same deterministic example and computed outcome.
 function lesson(step,r){return [
 ['已有 Prompt：我 喜欢。','Backbone 40 层 → Final RMSNorm → LM Head → Logits → Sample。','选出 Anchor「先」，同时保留层 37/38/39 的输入 Features，交给 DSpark。'],
 ['Anchor「先」和 Backbone Features。','把「先」放在第一个位置，后面放四个 Noise 占位；特征拼接、投影为上下文。','[先, Noise, Noise, Noise, Noise] 和上下文 → 三个 Draft Blocks；Noise 还不是候选。'],
 ['五个位置和来自 Backbone 的上下文。','五个位置一起经过 mtp[0] → mtp[1] → mtp[2]；每块都处理全部五个位置。','五个位置的 Hidden States → LM Head，并供 Confidence Head 使用。'],
 ['五个位置的 Hidden States，以及 Anchor「先」。','LM Head 给出五组 Base Logits；Markov Head 从「先」开始，逐个选出 A → B → C → D → E。','候选 [A, B, C, D, E]；Confidence Head 另用隐藏状态与前词特征给出分数，供调度选验证长度。'],
 ['Anchor「先」、候选 A B C D E 和置信分数。','Confidence / Scheduler 选择验证长度，本例选五个；[先, A, B, C, D, E] 一起经过正常的 40 层 Backbone。','Target Logits → 接受判断：先的位置预测 A，A 预测 B，B 预测 C；依此类推，E 的预测供 Bonus 使用。'],
 ['候选 A B C D E 和 Backbone 的对应预测。',r.rejected===null?'五个候选全部通过，再由 Backbone 产生 Bonus Token。':`保留连续通过的 ${candidates.slice(0,r.accepted).join(' ')||'空前缀'}；${candidates[r.rejected]} 被拒绝，其后候选作废；生成修正 ${r.extra}。`,`确认文本：${r.confirmed.join(' ')}。以 ${r.extra} 为新起点，进入下一轮生成。`]
 ][step];}
 return {vocab,candidates,scenarios,draft,anchor,bonus,softmax,argmax,sample,residual,evaluate,targetRow,lesson};
});
