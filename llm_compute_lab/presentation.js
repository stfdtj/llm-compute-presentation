/* Presentation changes only layout. Existing controls/charts move, never clone. */
(function(){
  'use strict';
  const A=LLMComputeLab,C=ComputeLab,$=id=>document.getElementById(id);
  const slots=[],sections=[],inlineDetails=[];let enabled=false;
  const n=v=>v==null?'未知':v.toLocaleString('en-US',{maximumFractionDigits:3});
  function prepare(tab,markup){
    const section=$(tab),normal=document.createElement('details');normal.className='presentation-more';normal.open=true;
    const summary=document.createElement('summary');summary.textContent='展开完整学习内容、表格与推导';normal.append(summary);
    const content=document.createElement('div');content.className='study-content';
    while(section.firstChild)content.append(section.firstChild);
    normal.append(content);
    const panel=document.createElement('div');panel.className='presentation-panel';panel.hidden=true;panel.innerHTML=markup;
    section.append(panel,normal);sections.push({normal,panel});
  }
  function slot(selector,id){const node=document.querySelector(selector),marker=document.createComment('presentation position: '+id);node.before(marker);slots.push({node,marker,target:$(id)});}
  function foldInline(nodes,title){
    const d=document.createElement('details'),s=document.createElement('summary');d.className='presentation-inline-more';d.open=true;s.textContent=title;d.append(s);nodes[0].before(d);nodes.forEach(node=>d.append(node));inlineDetails.push(d);
  }
  foldInline([...document.querySelectorAll('.speedup-panel > .table-scroll,.speedup-panel > details')],'展开数值表、比值定义与推导');
  foldInline([document.querySelector('.roof-chart-panel > .chart-guide')],'展开坐标轴与屋顶线读图指南');
  const options=(id,title)=>`<details class="presentation-options"><summary>${title}</summary><div id="${id}"></div></details>`;
  prepare('attention',`<div class="presentation-heading"><div><h1>Attention：少算多少？</h1><p id="presentAttentionContext"></p></div><label class="present-scope">比较口径<span id="presentScope"></span></label></div><p id="presentAttentionTakeaway" class="presentation-takeaway" aria-live="polite"></p>${options('presentAttentionOptions','现场修改 Attention 参数')}<div class="presentation-grid attention-deck"><div id="presentAttentionChart"></div><div id="presentSpeedup"></div></div>`);
  prepare('roofline',`<div class="presentation-heading"><div><h1>Roofline：等数据，还是等计算？</h1><p id="presentRoofContext"></p></div></div><p id="presentRoofTakeaway" class="presentation-takeaway" aria-live="polite"></p><div class="presentation-toolbar">${options('presentRoofOptions','修改硬件 / 访存 / 实测 Runtime')}<div id="presentRoofActions"></div></div><div class="presentation-grid roof-deck"><div id="presentRoofChart"></div><aside><div id="presentRoofMetrics"></div><div id="presentRoofTeaching"></div></aside></div>`);
  prepare('mfu',`<div class="presentation-heading"><div><h1>MFU：训练速度变化，比例怎样变？</h1><p>全部为教学假设 · 全模型 Training 口径，与单层 Decode 分开</p></div></div><div id="presentMfuFormula"></div><p id="presentMfuTakeaway" class="presentation-takeaway" aria-live="polite"></p>${options('presentMfuOptions','现场修改 MFU 教学输入')}<div class="presentation-grid mfu-deck"><div id="presentMfuResult"></div><div id="presentMfuExperiments"></div></div>`);
  slot('#scope','presentScope');slot('.parameters','presentAttentionOptions');slot('.chart-panel','presentAttentionChart');slot('.speedup-panel','presentSpeedup');
  slot('.roof-controls','presentRoofOptions');slot('#customTrafficDetails','presentRoofOptions');slot('.roof-chart-panel','presentRoofChart');slot('#roofMetrics','presentRoofMetrics');slot('#roofTeachingComparison','presentRoofTeaching');
  slot('.roof-chart-panel .demo-controls','presentRoofActions');
  slot('#mfuForm','presentMfuOptions');slot('#mfuMainFormula','presentMfuFormula');slot('#mfuResult','presentMfuResult');slot('#mfuExperiments','presentMfuExperiments');
  function conclusions(){
    const data=A.getExperiment(),scope=A.getScope(),r=data.results.find(x=>x.name===A.getSelection()),f=scope==='core'?r.core:r.total,base=scope==='core'?data.results[0].core:data.results[0].total;
    $('presentAttentionContext').textContent=`历史 S=${n(data.historyTokens)} · 生成 G=${n(data.generatedTokens)} · ${data.parameters.precision} · ${scope==='core'?'核心 Core':'声明范围 Module'} · 单层每序列`;
    $('presentAttentionTakeaway').textContent=!$('error').hidden?'输入尚未通过校验；图表保留上一次有效 Attention 配置，请展开参数修正。':f==null?`${r.name} 的完整 Total 未核验，不能给出完整模块的性能或加速结论。可切到已知 Core 范围。`:`${r.name} 的 FLOPs 为 MHA 的 ${n(f/base*100)}%，计算量比值为 ${n(base/f)}×；运行速度还取决于数据搬运和算子效率，不能按这个比值推断。`;
    const roof=A.getRooflineAnalysis(),q=roof?.results.find(x=>x.name===A.getRooflineSelection());
    $('presentRoofContext').textContent=roof?`${roof.hardware.name} · ${roof.hardware.precision} · ${n(roof.hardware.peakTF)} TFLOP/s / 设备 · ${n(roof.hardware.bandwidth)} ${roof.hardware.bandwidthUnit} / 设备 · ${roof.hardware.devices} 设备 · ${roof.scope==='core'?'Core':'Module'} · G=${n(roof.generatedTokens)}`:'当前参数未通过校验；展开设置查看错误。';
    $('presentRoofTakeaway').textContent=!q?'等待有效场景，暂不判断瓶颈。':q.ai==null?`${q.name} 缺少完整 FLOPs 或访存量，当前范围的瓶颈和时间未知。菱形点仅用于独立教学。`:`${q.name}：AI ${n(q.ai)} ${q.ai<q.ridge?'低于':'达到或超过'} 转折点 ${n(q.ridge)} FLOPs/B，因此在当前假设下受${q.ai<q.ridge?'带宽':'算力'}限制；理想上界 ${n(q.performance/1e12)} TFLOP/s，时间下界 ${n(q.lowerSeconds*1000)} ms。`;
    const m=A.getMfuTeachingExample();
    $('presentMfuTakeaway').textContent=!m?'输入不完整，暂不计算 MFU。':m.exceedsPeak?`当前假设得到 ${n(m.percent)}%，超过理论峰值；请核对 FLOPs、全设备吞吐量和硬件口径。`:`假设吞吐量 ${n(m.tokensPerSecond)} Tokens/s 对应 MFU ${n(m.percent)}%。它衡量模型计算吞吐量与理论峰值的比例，不能说明设备忙碌百分比、训练精度或具体性能瓶颈。`;
  }
  function setMode(next){
    enabled=!!next;document.body.classList.toggle('presentation',enabled);
    for(const s of slots)enabled?s.target.append(s.node):s.marker.after(s.node);
    for(const s of sections){s.panel.hidden=!enabled;s.normal.open=!enabled;}
    inlineDetails.forEach(d=>d.open=!enabled);
    $('presentationToggle').setAttribute('aria-pressed',String(enabled));$('presentationToggle').textContent=enabled?'退出演示模式':'演示模式 · 关闭';
    conclusions();window.scrollTo(0,0);
  }
  $('presentationToggle').addEventListener('click',()=>setMode(!enabled));
  for(const e of ['compute-lab-change','compute-lab-invalid','compute-lab-roofline-change','compute-lab-mfu-change'])document.addEventListener(e,conclusions);
  document.addEventListener('compute-lab-tab-change',()=>{if(enabled)window.scrollTo(0,0);});
  document.addEventListener('compute-lab-demo-change',e=>{if(enabled)$('presentRoofTakeaway').textContent=e.detail.text;});
  A.getPresentationMode=()=>enabled;
  conclusions();
})();
