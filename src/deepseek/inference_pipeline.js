/* Deterministic execution teaching model. No neural inference or performance claims. */
(()=>{
 'use strict';
 const $=id=>document.getElementById(id),root=$('inference-pipeline');
 const tab=document.createElement('button');tab.id='csa-tab-pipeline';tab.role='tab';tab.textContent='Inference Pipeline';tab.setAttribute('aria-selected','false');tab.setAttribute('aria-controls',root.id);tab.tabIndex=-1;
 document.querySelector('.csa-view-tabs').insertBefore(tab,document.querySelector('.csa-tab-caption'));
 const draft=['吃','苹果','呀'],corrections=['去','香蕉','。'];
 const common=[['input','Input Prompt','先把输入变成模型能处理的 token。'],['prefill','Prefill · 建立上下文','Prompt 经过因果计算，构建后续生成需要的 KV Cache。'],['anchor','初始预测 · Target anchor','Prefill 最后一个位置的预测给出起始 token，两个模式共用这个起点。']];
 const spec=[['draft','A · Draft Generation','DSpark 提出一小段候选；这时还不能把它们当作输出。'],['schedule','B · Confidence Scheduling','选择要交给 Target 验证的候选前缀长度。'],['verify','C · Target Backbone Verification','一次 Target Forward 同时计算多个位置的分布；接受判断仍按从左到右的顺序进行。'],['accept','D · Accept / Reject','遇到第一个 Reject 就停止接受；后面的候选即使看起来合理，也不能直接保留。'],['commit','E · 更新有效状态','把确认前缀交给下一轮，让被拒绝的候选离开有效历史。'],['continue','Continue / Finish','从有效状态接着生成，直到 EOS、长度上限或其他终止条件。']];
 const normal=[['normal1','Decode · Forward #2','把尚未写入 KV 的起始 token「先」送进主模型，预测「吃」。'],['normal2','Decode · Forward #3','新确认的「吃」成为本轮输入，预测「苹果」。'],['normal3','Decode · Forward #4','新确认的「苹果」成为本轮输入，预测「呀」。'],['normalcommit','更新状态 · 下一次迭代','每轮复用历史 KV，并维护新位置的有效状态。'],['normalend','Continue / Finish','继续下一轮 Decode，或在满足终止条件时结束。']];
 let active=false,step=0,mode='dspark',outcome='all',length=3,timer=null;
 const steps=()=>[...common,...(mode==='dspark'?spec:normal)];
 const chip=(text,kind='',label='')=>`<span class="ip-chip ${kind}"><strong>${text}</strong>${label?`<small>${label}</small>`:''}</span>`;
 const tokens=(items,kind='',label='')=>`<div class="ip-tokens">${items.map(x=>chip(x,kind,label)).join('')}</div>`;
 const module=(name,text,kind='')=>`<div class="ip-module ${kind}"><b>${name}</b><small>${text}</small></div>`;
 const flow=items=>`<div class="ip-flow">${items.join('<span class="ip-arrow" aria-hidden="true">→</span>')}</div>`;
 const note=(text,warning=false)=>`<div class="ip-callout ${warning?'ip-warning':''}">${text}</div>`;
 function result(){
  const rejected=outcome==='all'?null:outcome==='first'?0:outcome==='second'?1:length-1;
  // Options for positions beyond the scheduled prefix are disabled in the UI.
  const index=rejected!==null?Math.min(rejected,length-1):null;
  return {index,accepted:index===null?length:index,extra:index===null?'！':corrections[index]};
 }
 function snapshot(){
  const id=steps()[step][0],r=result(),after=mode==='dspark'&&step>=6;
  let confirmed=[],cache=['我','喜欢'],pending=[];
  if(step>=2){confirmed=['先'];pending=['先'];}
  if(mode==='normal'&&step>=3){
   const count=Math.min(3,step-2);confirmed.push(...draft.slice(0,count));cache.push('先',...draft.slice(0,count-1));pending=[draft[count-1]];
  }
  if(after){confirmed.push(...draft.slice(0,r.accepted),r.extra);}
  if(mode==='dspark'&&step>=7){cache.push('先',...draft.slice(0,r.accepted));pending=[r.extra];}
  // At Accept/Reject the accepted text is known, but runtime commit is the next stage.
  return {active,step,stage:id,mode,outcome,length,playing:timer!==null,confirmed:['我','喜欢',...confirmed],cache,pending,result:r,committed:mode==='dspark'&&step>=7};
 }
 function states(){
  const s=snapshot(),specMode=mode==='dspark',r=s.result;
  const committed=specMode&&step>=7,after=specMode&&step>=6;
  $('ip-state').innerHTML=`<div class="ip-card"><h3>Confirmed context · 确认文本</h3>${tokens(s.confirmed,'ip-accept')}<p class="ip-muted">确认文本不等于每个位置已写入 KV。</p></div><div class="ip-card"><h3>有效 KV / 模型状态 · 概念边界</h3>${step<1?'<p class="ip-muted">尚未建立</p>':tokens(s.cache,'ip-target')}<p class="ip-muted">${step<2?'Prefill 建立历史状态。':after&&!committed?'已判定结果；下一步提交有效前缀。':`待处理 token：${s.pending.join('、')}。它已确认，但需下次 Forward 才有对应 KV。`}</p></div><div class="ip-card"><h3>${specMode?'本轮候选状态':'执行对象'}</h3>${specMode?`<span class="ip-stat">${after?r.accepted:'—'} / ${length}</span><p>已 Accept / 本轮验证候选</p><p class="ip-muted">Draft 总共画 3 个；实际配置 5 个。${after?`另有 1 个 Target ${r.index===null?'bonus':'修正'} token。`:''}</p>`:'<b>Target Backbone · 40 层</b><p class="ip-muted">此模式不调用 DSpark。</p>'}</div>`;
 }
 function verification(){
  const prefix='我喜欢先',n=length,inputs=['先',...draft.slice(0,n)];
  const rows=inputs.map((x,i)=>`<tr><th>输入 ${x}</th><td>✓</td>${inputs.map((_,j)=>`<td class="${j>i?'ip-blocked':''}">${j<=i?'✓':'×'}</td>`).join('')}</tr>`).join('');
  return flow([module('验证输入',inputs.join(' · ')+'（anchor + 候选）'),module('Target Backbone','Encoder 0–19 → Decoder 20–39'),module('LM Head → Logits','每个输入位置都需要词表 Logits'),module('Softmax','得到各位置的 Target 分布')])+`<div class="ip-verification"><div><h3>预测比输入右移一位</h3><table class="ip-table"><thead><tr><th>输入位置</th><th>对应分布 / 检查项</th></tr></thead><tbody>${draft.slice(0,n).map((x,i)=>`<tr data-ip-check="${i}"><td>${inputs[i]}</td><td>P(${x} | ${prefix+draft.slice(0,i).join('')})</td></tr>`).join('')}<tr><td>${inputs[n]}</td><td>P(next | ${prefix+draft.slice(0,n).join('')})<br><span class="ip-muted">仅当前缀全 Accept 时可取 bonus</span></td></tr></tbody></table></div><div><h3>Causal Mask · 可见关系示意</h3><div class="ip-scroll"><table class="ip-mask"><thead><tr><th>Query ↓ / Key →</th><th>历史</th>${inputs.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div><p class="ip-muted">✓ 允许看当前及过去；× 禁止看未来。这里只画因果边界，CSA2 / SWA 的实际可见集合还受稀疏模式与窗口限制。</p></div></div>`+note('历史 KV 到「我喜欢」；已确认的 anchor「先」还没写入 Target KV，因此和候选一起送入。输入「先」的位置检查「吃」，不是用输入「吃」的同位置 Logits 检查自己。')+note('Target 层与层之间仍依次执行；同一层的多个位置可以一起计算。不需要额外的神经网络 Verifier，Accept / Reject 由 inference runtime 按对应分布完成。参考 forward 默认取末位置 Logits；多位置验证需要 runtime 暴露所有所需位置的 Logits。',true)+timeline();
 }
 function timeline(){return `<div class="ip-timeline"><div class="ip-lane"><span>Normal Decode</span>${draft.slice(0,length).map((x,i)=>`<span class="ip-tick">Target #${i+1} → ${x}</span>${i<length-1?'<span>→</span>':''}`).join('')}</div><div class="ip-lane"><span>DSpark</span><span class="ip-tick ip-draft">Draft + 调度</span><span>→</span><span class="ip-tick ip-wide">1 次 Target Verification · ${length} 个候选位置 + bonus 分布</span><span>→ Accept</span></div></div><p class="ip-muted">时间块只表示依赖顺序，宽度不是耗时。同样从 anchor 之后比较；少跑串行 Target 轮次有机会提速，但还有 Draft、验证和状态维护成本，不代表 ${length}× 加速。</p>`;}
 function acceptance(){
  const r=result();
  return `<h3>固定的 Draft，切换上方 Outcome 比较结果</h3><div class="ip-tokens">${draft.map((x,i)=>chip(x,i>=length?'ip-defer':i<r.accepted?'ip-accept':'ip-reject',i>=length?'未调度':i<r.accepted?'Accept':i===r.index?'首个 Reject':'丢弃后缀')).join('')}</div><p>${r.index===null?'本轮所验证的候选全部 Accept。':`保留前 ${r.accepted} 个候选；第 ${r.index+1} 个 Reject，后续已验证的候选全部丢弃。`} ${length<3?'未调度候选仍未确认，下一轮重新 Draft。':''}</p>${flow([module('保留接受前缀',`我喜欢先${draft.slice(0,r.accepted).join('')}`,'ip-green'),module(r.index===null?'Target bonus token':'Target correction token',r.extra),module('下一轮起点',`我喜欢先${draft.slice(0,r.accepted).join('')}${r.extra}`,'ip-green')])}
   <details open><summary><b>Greedy Verification · 本页默认演示</b></summary><p>从左到右，对比候选与 Target 对应分布的 argmax；第一个不一致处取 Target 所选 token 修正。若全部一致，可从最后一个输入位置的 Target 分布取一个 bonus token。Outcome 是人为设定的教学情境，未运行模型。</p></details>
   <details><summary><b>Probabilistic Verification · 标准理论</b></summary><p>设对应历史下的 Target 分布为 p，实际 Draft 提议分布为 q（包括 Markov 条件项），候选 x 的接受概率：</p><p><code>P(accept) = min(1, p_target(x) / p_draft(x))</code></p><p>拒绝后，在首次拒绝位置从归一化的 <code>max(0, p(v) − q(v))</code> residual 分布采样替代 token；后缀全部丢弃。这里不是总取 Target 概率最高的 token。若全部通过，则从最终 Target 分布采样 bonus。各分布需与实际 sampling 设置一致，且 q(x) &gt; 0。</p><p>这是标准 speculative sampling 理论；本页不随机抽样，也不声称仓库已实现该 runtime 算法。</p></details>`;
 }
 function body(id){
  if(id==='input')return flow([module('Input Prompt','「我喜欢」'),module('Tokenization','教学词块：[我] [喜欢]'),module('Embedding','把 token IDs 映射为向量')])+tokens(['我','喜欢'])+note('词块只是便于讲解的示意，不代表真实 tokenizer 一定这样分词。两种模式的输入与 Prefill 完全一致。');
  if(id==='prefill')return flow([module('Causal Encoder','20 层因果处理 Prompt'),module('全局 KV 投影','从 Encoder 末层状态生成 Decoder 所需全局 KV'),module('Decoder SWA Replay','重放末尾窗口，补齐局部状态'),module('LM Head / Logits','取得初始 next-token 分布')])+note('CED 可以减少长 Prompt 在 Decoder 的重复计算，但不是所有 Decoder 操作都跳过：局部 SWA 仍要重放，初始预测也需要相应的 Target 计算。')+note('以上优化依据技术报告；本仓库参考 model.py 的 forward 仍顺序执行全部 40 个主层，未实现优化 Prefill 路径。',true);
  if(id==='anchor')return flow([module('Prompt 最后一个位置','Target Logits → Softmax'),module('Token selection','本例选出「先」'),module('共享起点','确认文本：我喜欢先','ip-green')])+tokens(['先'],'ip-target','Target anchor')+note('「先」来自 Target，并不是 Draft。此时 Prompt 的 KV 已建立；刚选出的「先」还没有对应 Target KV。下一次 Forward 才会处理它。保留 main_hidden 供 DSpark 使用。');
  if(id==='draft')return `<h3>Confirmed context：我喜欢先</h3>`+flow([module('主模型特征 + anchor','第 37 / 38 / 39 层注意力输入，流均值后拼接 → 投影'),module('Draft Block 1','独立 DSpark 分支 · 私有 128-token 窗口','ip-purple'),module('Draft Block 2','继续处理全部 Draft 槽位','ip-purple'),module('Draft Block 3','输出 Draft 隐藏状态','ip-purple'),module('共享 Head + Markov Head','5 位置基础 Logits 并行；轻量 Head 依次采样','ip-purple')])+tokens(draft,'ip-draft','候选 · 未确认')+note('forward_embed 使用 [anchor, noise, noise, noise, noise]。三块依次处理这 5 个槽位；forward_head 返回 anchor + 5 个候选、Logits 和 Confidence。本图只展示前 3 个候选。Draft 内部槽位可互相可见；下步 Target 验证才遵循 Target 的因果边界。')+note('普通 Target Forward 只走 40 层主模型；Draft Blocks 的结构编号 40–42 不能画成普通 Decode 必经的第 41–43 层。',true);
  if(id==='schedule')return flow([module('Confidence Head','预测每个位置的条件接受概率','ip-purple'),module('Confidence Scheduler','估计前缀存活率 + 引擎吞吐曲线'),module(`本轮验证前 ${length} 个`,`系统负载和预计收益决定验证预算`)])+`<div class="ip-tokens">${draft.map((x,i)=>chip(x,i<length?'ip-draft':'ip-defer',i<length?'交给 Target':'本轮不验证')).join('')}</div>`+note('靠后的候选如果容易被前面的 Reject 连带丢弃，全部验证会浪费 batch 容量。调度会权衡预计能接受多少 token 与引擎在当前负载下的验证成本。')+note('上方长度选择是教学控件；没有虚构 confidence 数值、阈值或调度公式。仓库只返回 Confidence，未包含完整 Scheduler。',true);
  if(id==='verify')return verification();
  if(id==='accept')return acceptance();
  if(id==='commit'){
   const r=result();return acceptance().split('<details open>')[0]+note(`确认：anchor「先」+ 已 Accept 的 ${r.accepted} 个候选 + Target ${r.index===null?'bonus':'修正'}「${r.extra}」。验证时临时算出的拒绝位置和未接受后缀不能留在有效历史里。`)+flow([module('Commit valid prefix',`有效 KV 到「我喜欢先${draft.slice(0,r.accepted).join('')}」`,'ip-green'),module('Discard invalid suffix','包括首次 Reject 的位置与其后缀'),module('Pending Target token',`「${r.extra}」已确认，下次送入 Forward`)])+note('这是逻辑上的 commit / discard：runtime 必须同步维护 KV、位置、共享缓存以及其他模型状态；没有假定具体拷贝、清零或内存回退操作。');
  }
  if(id==='continue'){
   const r=result(),context=`我喜欢先${draft.slice(0,r.accepted).join('')}${r.extra}`;
   return `<h3>下一轮确认上下文：${context}</h3>`+flow([module('Termination check','EOS / 输出长度上限 / 用户停止'),module('继续时：新 anchor',`用已确认的「${r.extra}」与有效特征进入 Draft`),module('Draft → Verification','更新并复用有效历史；重复本轮流程','ip-purple')])+note(`下轮 Target 输入从待处理的「${r.extra}」开始，和新候选一起验证；被丢弃的旧后缀不算历史。此处展示下一轮入口，点击 Reset 可重播完整例子。`)+timeline();
  }
  if(id.startsWith('normal')&&/^normal[123]$/.test(id)){
   const i=Number(id.slice(-1))-1,input=i===0?'先':draft[i-1],context='我喜欢先'+draft.slice(0,i).join('');
   return `<h3>当前上下文：${context}</h3>`+flow([module('当前输入',`「${input}」；先前位置复用 KV`),module('Target Backbone','40 层主模型，DSpark 不参与'),module('LM Head → Logits → Softmax',`P(next | ${context})`),module('Token selection',`确认「${draft[i]}」`,'ip-green')])+tokens([draft[i]],'ip-accept','新确认 token')+note(`本轮为输入「${input}」建立 KV，再选出「${draft[i]}」；新选出的 token 在下一轮作为输入。一次普通 Decode 通常为每条序列预测一个新 token。Prefill 的初始预测算 Forward #1，此处为 #${i+2}。`);
  }
  if(id==='normalcommit')return flow([module('已确认文本','我喜欢先吃苹果呀','ip-green'),module('有效 Target KV','到「我喜欢先吃苹果」'),module('待处理输入','「呀」进入下一轮')])+note('生成 token 的选择与它的 KV 写入相隔一次 Forward；保留有效 KV 和模型状态，无需重新计算全部历史。DSpark 在这个模式中不运行。');
  return flow([module('下一轮输入','呀'),module('Target Forward #5','复用有效历史，预测下一个 token'),module('继续 / 停止','EOS、长度上限或用户停止')])+note('本例停在下一轮入口；尚未展示 EOS，不能把「呀」当作结束标志。');
 }
 function render(){
  const list=steps(),[id,title,description]=list[step];
  $('ip-title').textContent=title;$('ip-description').textContent=description;$('ip-body').innerHTML=body(id);$('ip-step-tag').textContent=`${mode==='dspark'?'DSpark':'Normal'} · Step ${step+1} / ${list.length}`;
  $('ip-progress').innerHTML=list.map(([key,name],i)=>`<button data-ip-step="${i}" ${i===step?'aria-current="step"':''} class="${i<step?'ip-done':''}">${i+1} · ${name.split(' · ').pop()}</button>`).join('');
  $('ip-prev').disabled=step===0;$('ip-next').disabled=step===list.length-1;$('ip-play').textContent=timer===null?'▶ Play':'Ⅱ Pause';$('ip-play').setAttribute('aria-pressed',String(timer!==null));$('ip-status').textContent=timer===null?'已暂停':'自动播放';
  for(const id of ['ip-length','ip-outcome'])$(id).disabled=mode==='normal';
  $('ip-live').textContent=title;states();
 }
 function pause(){if(timer!==null)clearInterval(timer);timer=null;}
 function move(target){pause();step=Math.max(0,Math.min(steps().length-1,target));render();}
 function play(){if(timer!==null){pause();render();return;}if(step===steps().length-1)step=0;timer=setInterval(()=>{if(!active){pause();render();return;}step++;if(step>=steps().length-1){step=steps().length-1;pause();}render();},4200);render();}
 $('ip-prev').onclick=()=>move(step-1);$('ip-next').onclick=()=>move(step+1);$('ip-play').onclick=play;
 $('ip-reset').onclick=()=>{pause();step=0;outcome='all';length=3;$('ip-outcome').value=outcome;$('ip-length').value='3';$('ip-outcome').querySelector('[value=second]').disabled=false;render();root.scrollTop=0;};
 $('ip-mode').onchange=e=>{pause();mode=e.target.value;step=0;render();};
 $('ip-outcome').onchange=e=>{pause();outcome=e.target.value;render();};
 $('ip-length').onchange=e=>{pause();length=Number(e.target.value);$('ip-outcome').querySelector('[value=second]').disabled=length<2;if(length<2&&outcome==='second'){outcome='all';$('ip-outcome').value='all';}render();};
 $('ip-progress').onclick=e=>{const b=e.target.closest('[data-ip-step]');if(b)move(Number(b.dataset.ipStep));};
 document.addEventListener('keydown',e=>{if(!active||e.target.closest('button,input,select,textarea,summary,[contenteditable=true],[role=tablist]'))return;if(['ArrowRight','ArrowLeft',' ','0'].includes(e.key)){e.preventDefault();e.stopImmediatePropagation();if(e.key===' ')play();else if(e.key==='0')$('ip-reset').click();else move(step+(e.key==='ArrowRight'?1:-1));}},true);
 document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();render();}});window.addEventListener('pagehide',pause);
 window.inferencePipeline={get state(){return structuredClone(snapshot());},setActive(on){pause();active=on;root.hidden=!on;document.body.classList.toggle('pipeline-active',on);render();}};
 render();
})();
