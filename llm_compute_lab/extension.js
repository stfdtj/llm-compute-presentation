/* Incremental sequence/speedup/Roofline UI. Existing attention explorer stays in ui.js. */
(function(){
  'use strict';
  const C=ComputeLab,A=LLMComputeLab,$=id=>document.getElementById(id),names=C.architectures;
  const fmt=n=>n==null?'N/A':n>=1e12?(n/1e12).toFixed(3)+' T':n>=1e9?(n/1e9).toFixed(3)+' G':n>=1e6?(n/1e6).toFixed(2)+' M':n>=1e3?(n/1e3).toFixed(2)+' K':n.toLocaleString('en-US');
  const num=n=>n==null?'N/A':n.toLocaleString('en-US',{maximumFractionDigits:4});
  const ratio=n=>n==null?'N/A':n.toFixed(2)+'×';
  const percent=n=>n==null?'N/A':n<0?`增加 ${(-n).toFixed(2)}%`:`减少 ${n.toFixed(2)}%`;
  const bytes=n=>n==null?'未知':n>=2**30?(n/2**30).toFixed(3)+' GiB':n>=2**20?(n/2**20).toFixed(3)+' MiB':num(n)+' B';
  const time=n=>n==null?'N/A':n>=1?n.toFixed(4)+' s':n>=.001?(n*1e3).toFixed(3)+' ms':(n*1e6).toFixed(3)+' μs';
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let roofData=null,roofSelected=A.getSelection(),roofVisible=new Set(names),cumulativePlot=null,lastSignature='',lastAttentionSelection=A.getSelection();
  const choice=scope=>r=>scope==='core'?r.core:r.total;
  function snapshot(){const p=A.getParameters(),s=A.getScenario(),scope=A.getScope(),records=C.sequenceResults(p,s);return {p,s,scope,records,selected:A.getSelection(),steps:records[0].steps};}
  function refresh(){
    const {p,s,scope,records,selected,steps:g}=snapshot(),r=records.find(x=>x.name===selected),value=choice(scope),base=value(records[0]),v=value(r);
    $('sequenceHeading').textContent=`${g.toLocaleString()} Tokens · ${s.mode==='single'?'单步 Decode':'自回归逐步累计'}`;
    const first=value(r.first),last=value(r.last);
    $('sequenceMetrics').innerHTML=[['Average / Token',v==null?null:v/g],['First Token',first],['Last Token',last]].map(([t,n])=>`<div><small>${t}</small><b>${fmt(n)} FLOPs</b></div>`).join('');
    $('sequenceProof').innerHTML=`<p><code>F_total = Σᵢ₌₀ᴳ⁻¹ F_decode(S+i)</code>；不是 G × F_decode(S)。每步 X=[1,${p.D}]。</p><p>初始 S=${p.S.toLocaleString()}，G=${g.toLocaleString()}。第一步历史 ${p.S.toLocaleString()}，最后一步历史 ${(p.S+g-1).toLocaleString()}；可见位置分别为 ${p.S+1} 与 ${p.S+g}。</p><p><code>Σ visible = G(2(S+1)+G−1)/2 = ${num(r.sumVisible)}</code>。${selected==='SWA'?`Σ min(S+i+1,W) = ${num(r.sumActive)}`:selected==='MSA'?`Σ selected positions = ${num(r.sumActive)}；按 local 末块周期与 k 预算分段求和`:selected==='GDA'?'递归核心恒定，但完整模块未知':`Σ active positions = ${num(r.sumActive)}`}。</p><p>累计 Projection=${fmt(r.projection)}；Core=${fmt(r.core)}；Additional=${fmt(r.extra)}；Module=${fmt(r.total)} FLOPs。Average=${fmt(v)} / ${g} = ${fmt(v==null?null:v/g)} FLOPs/token。整批所选口径=${fmt(v==null?null:v*p.batch)} FLOPs；均为单层。</p><p>内部使用解析前缀和，已与逐步调用单 Token 公式的显式循环核验。MSA 索引按 Σ visible 计算，不省略历史扫描。</p>`;
    $('inspectStep').max=g;let i=Number($('inspectStep').value);if(!Number.isSafeInteger(i)||i<1||i>g){i=1;$('inspectStep').value=1;}inspect();
    const maxRatio=Math.max(1,...records.map(x=>C.speedup(base,value(x)).ratio||0));
    $('speedupBars').innerHTML=records.map(x=>{const q=C.speedup(base,value(x));return `<div class="speed-row"><button data-arch="${x.name}">${x.name}</button><div class="speed-track"><span class="baseline-marker" style="left:${100/maxRatio}%"></span><i style="width:${q.ratio==null?0:100*q.ratio/maxRatio}%;background:${C.colors[x.name]}"></i></div><b>${ratio(q.ratio)}</b><small>${percent(q.reduction)}</small></div>`;}).join('');
    $('speedupTable').innerHTML=records.map(x=>{const q=C.speedup(base,value(x));return `<tr><td><button data-arch="${x.name}">${x.name}</button></td><td>${fmt(value(x))}</td><td>${ratio(q.ratio)}</td><td>${percent(q.reduction)}</td></tr>`;}).join('');
    $('speedupProof').innerHTML=records.map(x=>{const q=C.speedup(base,value(x));return `<p><b>${x.name}</b>：${q.ratio==null?'完整模块缺失，不能计算加速比。':`Speedup = ${num(base)} / ${num(value(x))} = ${ratio(q.ratio)}；Reduction = [1 − ${num(value(x))} / ${num(base)}] × 100% = ${num(q.reduction)}%。当前 FLOPs 是 MHA 的 ${(100/q.ratio).toFixed(2)}%，${q.ratio<1?'计算量增加':'计算量减少或持平'}。`}</p>`;}).join('');
    const signature=JSON.stringify([p,s,scope]);
    if(lastSignature&&signature!==lastSignature){
      // Manual bytes/runtime belong to an exact experiment and scope. Invalidate on change.
      document.querySelectorAll('#customTrafficInputs input').forEach(e=>e.value='');
      if($('trafficMode').value==='custom')$('roofChartNote').textContent='配置已改变，先前自定义访存量与实测耗时已清空，请为新场景填写。';
    }
    lastSignature=signature;
    if(selected!==lastAttentionSelection)roofSelected=selected;
    lastAttentionSelection=selected;
    $('roofScope').value=scope;
    document.querySelector('.memory-panel .muted').textContent=`KV Cache / State · 最后一步之后 · S+G=${(p.S+g).toLocaleString()} · 单层整批`;
    cumulative();renderRoofline();
  }
  function inspect(){const {p,scope,selected,steps:g}=snapshot(),i=Number($('inspectStep').value);if(!Number.isSafeInteger(i)||i<1||i>g){$('inspectResult').textContent=`请填写 1–${g} 的整数步骤。`;return;}const r=C.calculate(selected,{...p,S:p.S+i-1});$('inspectResult').innerHTML=`<p>第 ${i} 步：History = ${p.S} + ${i} − 1 = ${p.S+i-1}，可见 ${r.visible} 个位置，读取 ${r.active} 个位置；${scope==='core'?'Core':'Module'} = ${fmt(choice(scope)(r))} FLOPs。<br>Projection ${fmt(r.projection)}，Core ${fmt(r.core)}，Additional ${fmt(r.extra)}；缓存 / 状态 ${bytes(r.memoryBytes)}。</p>`;}
  function cumulative(){
    const {p,s,scope,steps:g}=snapshot(),shown=A.getVisible(),get=choice(scope),left=72,right=880,top=25,bottom=236,x=n=>left+n/g*(right-left);
    const counts=new Set([0,g]);for(let i=1;i<=80;i++)counts.add(Math.max(1,Math.round(i*g/80)));
    const series=shown.map(name=>({name,points:[...counts].sort((a,b)=>a-b).map(n=>({n,v:n===0?0:get(C.aggregate(name,p,{...s,mode:'autoregressive',generated:n}))})).filter(q=>q.v!=null)})).filter(q=>q.points.length>1);
    const max=Math.max(1,...series.flatMap(q=>q.points.map(t=>t.v)))*1.08,y=v=>bottom-v/max*(bottom-top);cumulativePlot={g,left,right,top,bottom,x,y,p,s,scope};
    let out=`<text x="${left}" y="13" font-size="10" fill="#718092">累计 FLOPs / layer / sequence · 初始 S=${p.S.toLocaleString()}</text>`;
    for(let i=0;i<=4;i++){const yy=bottom-i*(bottom-top)/4;out+=`<path d="M${left},${yy}H${right}" stroke="#e9eef4"/><text x="${left-10}" y="${yy+4}" text-anchor="end" font-size="10" fill="#718092">${fmt(max*i/4)}</text><text x="${x(g*i/4)}" y="259" text-anchor="middle" font-size="10" fill="#718092">${Math.round(g*i/4).toLocaleString()}</text>`;}
    series.forEach(q=>out+=`<path d="${q.points.map((t,i)=>(i?'L':'M')+x(t.n).toFixed(2)+','+y(t.v).toFixed(2)).join(' ')}" fill="none" stroke="${C.colors[q.name]}" stroke-width="2.3" ${q.name==='GQA'?'stroke-dasharray="7 5"':''}/>`);
    $('cumulativeChart').innerHTML=out+`<text x="475" y="281" text-anchor="middle" font-size="10" fill="#718092">Generated Tokens · 串行生成累计</text><g id="cumulativeHover"></g>`;
  }
  $('cumulativeChart').addEventListener('pointermove',e=>{const q=cumulativePlot;if(!q)return;const box=$('cumulativeChart').getBoundingClientRect(),x=(e.clientX-box.left)*900/box.width,n=Math.max(1,Math.min(q.g,Math.round((x-q.left)/(q.right-q.left)*q.g)));const data=A.getVisible().map(name=>({name,v:choice(q.scope)(C.aggregate(name,q.p,{...q.s,mode:'autoregressive',generated:n}))}));$('cumulativeReadout').textContent=`生成 ${n} tokens；最后历史 ${q.p.S+n-1} · `+data.map(x=>x.name+' '+fmt(x.v)).join(' / ');$('cumulativeHover').innerHTML=`<path d="M${q.x(n)},${q.top}V${q.bottom}" stroke="#9daec3"/>`+data.filter(x=>x.v!=null).map(x=>`<circle cx="${q.x(n)}" cy="${q.y(x.v)}" r="3" fill="${C.colors[x.name]}"/>`).join('');});
  $('inspectStep').addEventListener('input',inspect);
  function readHardware(){return {name:$('hardwareName').value.trim(),peakTF:Number($('peakTF').value),bandwidth:Number($('bandwidth').value),bandwidthUnit:$('bandwidthUnit').value,precision:$('hardwarePrecision').value,devices:Number($('deviceCount').value),provenance:$('hardwareTemplate').value==='h100'?C.hardwarePreset('h100'):{kind:'user-input',computeType:'user-declared'}};}
  function readTraffic(){const customBytes={},runtimesMs={};for(const n of names){const input=$('bytes-'+n),runtime=$('runtime-'+n);customBytes[n]=input.value.trim()===''?null:Number(input.value)*2**20;runtimesMs[n]=runtime.value.trim()===''?null:Number(runtime.value);}return {mode:$('trafficMode').value,weightHitRate:Number($('weightHitRate').value),kvHitRate:Number($('kvHitRate').value),headReuse:$('headReuse').value,intermediate:$('intermediate').value,customBytes,runtimesMs};}
  function renderRoofline(){
    const {p,s,steps:g}=snapshot(),h=readHardware(),o=readTraffic(),scope=$('roofScope').value;
    $('roofContext').textContent=`${s.mode==='single'?'Single Token':'Autoregressive'} Decode · G=${g} · S=${p.S} · ${p.precision} · 单层整批 Batch=${p.batch} · ${scope==='core'?'Core':'Module'}`;
    $('modelTrafficFields').hidden=o.mode!=='model';$('customTrafficDetails').open=o.mode==='custom';
    let errors=C.validateHardware(h,p);
    if(!$('error').hidden)errors.push('Attention 输入尚未通过校验，请先修正；Roofline 暂不使用旧结果');
    if(o.mode==='model')for(const id of ['weightHitRate','kvHitRate'])if($(id).value.trim()==='')errors.push(`${id} 命中率不能为空`);
    for(const n of names){const a=$('bytes-'+n).value,b=$('runtime-'+n).value;if(a.trim()!==''&&(!Number.isFinite(o.customBytes[n])||o.customBytes[n]<=0))errors.push(`${n} 自定义 MiB 必须为正数`);if(b.trim()!==''&&(!Number.isFinite(o.runtimesMs[n])||o.runtimesMs[n]<=0))errors.push(`${n} 实测 Runtime 必须为正数`);}
    if(!errors.length)try{roofData=C.analyzeRoofline(p,s,h,o,scope);}catch(e){errors.push(e.message);}
    $('roofError').hidden=!errors.length;$('roofError').textContent=errors.join('；');$('exportRoofline').disabled=!!errors.length||!$('error').hidden;
    if(errors.length){roofData=null;emptyRoof();document.dispatchEvent(new Event('compute-lab-roofline-change'));return;}
    $('rooflineSchema').textContent=JSON.stringify(roofData,null,2);
    const current=roofData.results.find(x=>x.name===roofSelected),missing=roofData.results.filter(x=>x.ai==null).map(x=>x.name);
    $('roofChartNote').textContent=`${o.mode==='model'?'数据点为声明假设下的 HBM 模型估算':'数据点依赖用户输入，尚未验证测量依据'}；${missing.length?'缺少完整 FLOPs 或 Bytes：'+missing.join(', ')+'，不绘点。':'全部点具有本范围的 FLOPs 与 Bytes。'} 上界不是实测性能。`;
    const peak=h.peakTF*1e12*h.devices,bw=h.bandwidth*(h.bandwidthUnit==='TB/s'?1e12:1e9)*h.devices;
    roofChart(peak,bw);
    $('roofMetrics').innerHTML=`<div class="metric"><small>转折计算密度 · Ridge</small><strong>${num(peak/bw)}</strong><span>FLOPs / Byte · 两种限制的交点</span></div><div class="metric"><small>${roofSelected} / AI</small><strong>${num(current.ai)}</strong><span>${current.bottleneck}</span></div><div class="metric"><small>上界 · ${roofSelected}</small><strong>${current.performance==null?'N/A':(current.performance/1e12).toFixed(3)}</strong><span>TFLOP/s · 理想上界</span></div>`;
    $('roofPointTable').innerHTML=roofData.results.map(r=>`<tr class="${r.name===roofSelected?'selected':''}"><td><button data-roof="${r.name}">${r.name}</button></td><td>${fmt(r.flops)}</td><td>${bytes(r.traffic.totalBytes)}</td><td>${num(r.ai)}</td><td>${r.bottleneck}</td><td>${r.performance==null?'N/A':(r.performance/1e12).toFixed(3)+' TFLOP/s'}</td></tr>`).join('');
    $('roofSpeedupTable').innerHTML=roofData.results.map(r=>`<tr><td><button data-roof="${r.name}">${r.name}</button></td><td>${ratio(r.flopsSpeedup)}</td><td>${time(r.lowerSeconds)}</td><td>${ratio(r.rooflineSpeedup)}</td></tr>`).join('');
    roofDetails();appendTrafficEquations();contrast();
    renderMeasurement(current);
    $('roofAssumptions').innerHTML=current.traffic.assumptions.map(x=>`<p>${esc(x)}</p>`).join('')+'<p>此图用单一峰值近似混合矩阵与标量运算。exp、递归状态等不保证能达到 Tensor Core 峰值；精确分析需按实际算子与计算精度拆分，并测量 HBM 流量。</p>';
    document.dispatchEvent(new Event('compute-lab-roofline-change'));
  }
  function emptyRoof(){
    $('roofChart').innerHTML='<text x="450" y="150" text-anchor="middle" font-size="16" fill="#718092">填写有效的硬件参数后，绘制 Roofline 上界</text><text x="450" y="181" text-anchor="middle" font-size="12" fill="#718092">Peak Compute 与 HBM Bandwidth 不使用未经核验的预设数值</text>';
    $('roofChartNote').textContent='当前没有有效硬件配置，不输出数据点或时间估计。';
    for(const id of ['roofMetrics','roofPointTable','roofSpeedupTable','roofContrast','roofDetails','roofAssumptions','roofMeasurement','rooflineSchema'])$(id).innerHTML='';$('roofExplanation').textContent='';
  }
  function roofChart(peak,bw){
    const ridge=peak/bw,points=roofData.results.filter(r=>r.ai!=null&&roofVisible.has(r.name)),teaching=C.rooflineTeachingPoints(roofData.hardware);
    const measured=points.filter(r=>r.measuredPerformance!=null);
    const allAi=[ridge,...points.map(r=>r.ai),...teaching.map(r=>r.ai)];
    const xmin=10**Math.floor(Math.log10(Math.min(...allAi)/3)),xmax=10**Math.ceil(Math.log10(Math.max(...allAi)*3));
    const values=[peak,xmin*bw,...points.map(r=>r.performance),...measured.map(r=>r.measuredPerformance),...teaching.map(r=>r.performance)];
    const ymin=10**Math.floor(Math.log10(Math.min(...values)*.6)),ymax=10**Math.ceil(Math.log10(Math.max(...values)*1.8));
    const left=80,right=880,top=35,bottom=322,x=v=>left+(Math.log10(v)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*(right-left),y=v=>bottom-(Math.log10(v)-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin))*(bottom-top);
    $('roofChart').dataset.scale=JSON.stringify({xmin,xmax,ymin,ymax,left,right,top,bottom});
    let svg='<text x="80" y="17" font-size="11" fill="#718092">纵轴：性能 · TFLOP/s · 对数刻度</text>';
    // Iterate exponents, not repeated multiplication, to avoid under/overflow loops.
    for(let k=Math.round(Math.log10(xmin));k<=Math.round(Math.log10(xmax));k++){const v=10**k;svg+=`<path d="M${x(v)},${top}V${bottom}" stroke="#edf1f5"/><text x="${x(v)}" y="345" text-anchor="middle" font-size="10" fill="#718092">${k>=3||k< -3?'10^'+k:num(v)}</text>`;}
    for(let k=Math.round(Math.log10(ymin));k<=Math.round(Math.log10(ymax));k++){const v=10**k;svg+=`<path d="M${left},${y(v)}H${right}" stroke="#edf1f5"/><text x="${left-10}" y="${y(v)+4}" text-anchor="end" font-size="10" fill="#718092">${num(v/1e12)}</text>`;}
    svg+=`<path class="roof-ceiling" d="M${x(xmin)},${y(xmin*bw)}L${x(ridge)},${y(peak)}L${x(xmax)},${y(peak)}" fill="none" stroke="#1b2c44" stroke-width="3"/><path d="M${x(ridge)},${top}V${bottom}" stroke="#8fa0b7" stroke-dasharray="5 4"/><text x="${Math.min(x(ridge)+5,right-140)}" y="${top+15}" font-size="11" fill="#62768e">转折 AI: ${num(ridge)}</text><text x="${x(ridge)+(right-x(ridge))/2}" y="${y(peak)-13}" text-anchor="middle" font-size="11" fill="#182b42">算力上限 · 水平线</text><text x="${left+12}" y="${bottom-18}" font-size="11" fill="#182b42">带宽上限 · P = AI × BW</text>`;
    const labels=[];
    points.forEach(r=>{const cx=x(r.ai),cy=y(r.performance),lx=Math.min(cx+12,right-45);let ly=cy-16;for(const offset of [-16,25,-36,45,-56,65,-76,85]){const testY=Math.max(top+14,Math.min(bottom-7,cy+offset));if(!labels.some(q=>Math.abs(q.x-lx)<48&&Math.abs(q.y-testY)<16)){ly=testY;break;}}labels.push({x:lx,y:ly});svg+=`<g data-roof="${r.name}" data-upper="${r.name}" tabindex="0" role="button" aria-label="${r.name} 理想上界，AI ${num(r.ai)}，${r.bottleneck}"><path d="M${cx},${cy}L${lx},${ly-4}" stroke="${C.colors[r.name]}" stroke-width=".7" opacity=".6"/><circle cx="${cx}" cy="${cy}" r="${r.name===roofSelected?7:5}" fill="${C.colors[r.name]}" stroke="white" stroke-width="2"/><text x="${lx}" y="${ly}" font-size="11" fill="${C.colors[r.name]}">${r.name}</text><title>${r.name} · 理想上界 · AI=${num(r.ai)} · ${r.bottleneck}</title></g>`;});
    measured.forEach(r=>{const cx=x(r.ai),cy=y(r.measuredPerformance);svg+=`<path d="M${cx},${y(r.performance)}V${cy}" stroke="${C.colors[r.name]}" stroke-dasharray="3 4"/><g data-roof="${r.name}" data-measured="${r.name}" tabindex="0" role="button" aria-label="${r.name} 用户实测 Runtime 换算的性能 ${num(r.measuredPerformance/1e12)} TFLOP/s"><circle cx="${cx}" cy="${cy}" r="7" fill="white" stroke="${C.colors[r.name]}" stroke-width="2.5"/><text x="${Math.min(cx+11,right-85)}" y="${Math.min(cy+18,bottom-4)}" font-size="10" fill="${C.colors[r.name]}">${r.name} 实测</text><title>${r.name} · FLOPs / 用户实测 Runtime；${num(r.measuredPerformance/1e12)} TFLOP/s；计数和测量范围须一致</title></g>`;});
    teaching.forEach(r=>{const cx=x(r.ai),cy=y(r.performance);svg+=`<g data-roof-teaching="${r.id}" tabindex="0" role="button" aria-label="教学假设 AI=${r.ai}，${r.bottleneck}"><path d="M${cx},${cy-8}L${cx+8},${cy}L${cx},${cy+8}L${cx-8},${cy}Z" fill="#fff2cc" stroke="#916313" stroke-width="2"/><text x="${cx}" y="${Math.min(cy+29,bottom-4)}" text-anchor="middle" font-size="11" fill="#795618">假设 AI=${r.ai}</text><title>独立教学假设点；不是 Attention 数据</title></g>`;});
    svg+=`<text x="${left+20}" y="${top+50}" font-size="12" fill="#677d95">Memory-bound · 等数据</text><text x="${right-12}" y="${bottom-35}" text-anchor="end" font-size="10" fill="#677d95">Compute-bound</text><text x="${right-12}" y="${bottom-20}" text-anchor="end" font-size="10" fill="#677d95">等计算</text><text x="480" y="374" text-anchor="middle" font-size="11" fill="#718092">横轴：计算密度 AI · 每搬运 1 Byte 的 FLOPs · 对数刻度</text>`;
    $('roofChart').innerHTML=svg;
  }
  function renderMeasurement(r){
    if(r.measuredPerformance==null){$('roofMeasurement').innerHTML='';return;}
    const runtime=readTraffic().runtimesMs[r.name],pct=100*r.measuredPerformance/r.performance,base=roofData.results[0],baseRuntime=readTraffic().runtimesMs.MHA;
    const measuredRatio=base.measuredPerformance!=null&&baseRuntime>0?baseRuntime/runtime:null;
    $('roofMeasurement').innerHTML=`<details open><summary>${r.name}：理想上界与用户实测 Runtime 对照</summary><p>P_measured = ${num(r.flops)} FLOPs ÷ (${num(runtime)} ms / 1000) = ${num(r.measuredPerformance/1e12)} TFLOP/s。</p><p>理想上界 ${num(r.performance/1e12)} TFLOP/s；实测换算值约为该上界的 ${num(pct)}%。${pct>100?'超过当前模型上界，请核对 FLOPs、HBM 假设、精度及测量范围。':'差距可能来自算子效率、Kernel Launch、通信、同步或访存模型误差，仅凭此图不能定位原因。'}此比例不是训练 MFU。</p>${measuredRatio==null?'':`<p>Measured Speedup（用户实测时间比值，需相同运行条件）= MHA ${num(baseRuntime)} ms ÷ ${r.name} ${num(runtime)} ms = ${ratio(measuredRatio)}。</p>`}</details>`;
  }
  function roofDetails(){
    if(!roofData)return;const r=roofData.results.find(x=>x.name===roofSelected),h=roofData.hardware,t=r.traffic,peak=h.peakTF*1e12*h.devices,bw=h.bandwidth*(h.bandwidthUnit==='TB/s'?1e12:1e9)*h.devices;
    $('roofDetailHeading').textContent=`${r.name} · 公式、代入与时间下界`;
    if(r.ai==null){$('roofDetails').innerHTML=`<p>FLOPs=${fmt(r.flops)}；Bytes=${bytes(t.totalBytes)}。${r.flops==null?'完整 Attention 模块尚未核验，不使用已知小计进行完整模块排名。':'访存量缺失，请提供当前范围的总 Bytes 或选择模型估算。'}</p>`;return;}
    const parts=[['Weight Traffic',t.weightBytes],['Historical KV Read',t.kvReadBytes],['Cache Write',t.cacheWriteBytes],['Intermediate / Input / Output',t.intermediateBytes],['Other / Recurrent State',t.stateBytes]];
    $('roofDetails').innerHTML=`<div class="time-cards"><div><small>Compute Time Lower Bound</small><b>${time(r.computeSeconds)}</b></div><div><small>Memory Time Lower Bound</small><b>${time(r.memorySeconds)}</b></div><div><small>Roofline Time Lower Bound</small><b>${time(r.lowerSeconds)}</b></div></div><details open><summary>AI、Performance 与时间的具体代入</summary><p>当前 ${roofData.tokenMode}，初始 S=${roofData.initialContext}，生成 ${roofData.generatedTokens} tokens，Batch=${roofData.parameters.batch}，单层整批 ${roofData.scope} 范围。</p><p><code>AI = ${num(r.flops)} FLOPs / ${num(t.totalBytes)} B = ${num(r.ai)} FLOPs/B</code></p><p><code>Ppeak = ${num(h.peakTF)} × 10¹² × ${h.devices} = ${fmt(peak)} FLOP/s</code><br><code>BW = ${num(h.bandwidth)} × ${h.bandwidthUnit==='TB/s'?'10¹²':'10⁹'} × ${h.devices} = ${num(bw)} B/s</code></p><p><code>Pupper = min(${num(peak)}, ${num(r.ai)} × ${num(bw)}) = ${num(r.performance)} FLOP/s</code>；${r.bottleneck}。</p><p><code>Tcompute = ${num(r.flops)} / ${num(peak)} = ${num(r.computeSeconds)} s</code><br><code>Tmemory = ${num(t.totalBytes)} / ${num(bw)} = ${num(r.memorySeconds)} s</code><br><code>Tlower = max(Tcompute, Tmemory) = ${time(r.lowerSeconds)}</code>；F / Tlower = ${num(r.performance)} FLOP/s。</p><p>${r.measuredPerformance==null?'没有同范围实测 Runtime。':`用户实测 Runtime=${num(readTraffic().runtimesMs[r.name])} ms；实测吞吐量 F/t=${(r.measuredPerformance/1e12).toFixed(3)} TFLOP/s。${r.measuredPerformance>r.performance?'超过当前模型上界，需检查计数范围、HBM 假设、精度或测量。':'实测值与模型上界分别展示。'}`}</p></details><details><summary>Bytes Moved 分解与缓存复用依据</summary><p>来源：${t.kind==='user-estimate'?'用户提供的估计总量；没有分项依据':'基于声明假设的逻辑 HBM 搬运模型，非实际测量'}。</p>${parts.map(([label,v])=>`<p><b>${label}</b>：${bytes(v)}${v==null?' · 分项未知':''}</p>`).join('')}<p><code>Total Bytes = Weight + Historical KV Read + Cache Write + Intermediate + State = ${num(t.totalBytes)} B</code></p>${t.assumptions.map(x=>`<p>${esc(x)}</p>`).join('')}<p>逻辑末步缓存 / 状态容量=${bytes(C.aggregate(r.name,roofData.parameters,A.getScenario()).memoryBytes)}；与区间 HBM 流量分开计算。命中率作用于每步历史读取，写入仍单列。权重元素数来自当前实际执行的 Projection 矩阵尺寸，总权重读取=元素数×精度字节×生成步数×设备数×(1−命中率)。</p></details>`;
  }
  function appendTrafficEquations(){const r=roofData?.results.find(x=>x.name===roofSelected);if(!r?.traffic.equations)return;const details=$('roofDetails').querySelector('details:last-child');if(details)details.insertAdjacentHTML('beforeend','<b>逐项字节公式与数值代入</b>'+r.traffic.equations.map(x=>`<p><code>${esc(x)}</code></p>`).join(''));}
  function contrast(){const baseline=roofData.results[0],r=roofData.results.find(x=>x.name===roofSelected);
    if(!baseline.lowerSeconds||!r.lowerSeconds){$('roofContrast').innerHTML='';$('roofExplanation').textContent='缺少完整 FLOPs 或访存量，无法计算完整架构的 Roofline 时间加速比。';return;}
    const ratios=[['FLOPs / MHA',r.flops/baseline.flops],['HBM Bytes / MHA',r.traffic.totalBytes/baseline.traffic.totalBytes],['Time Lower Bound / MHA',r.lowerSeconds/baseline.lowerSeconds]],max=Math.max(1,...ratios.map(x=>x[1]));
    $('roofContrast').innerHTML=ratios.map(([label,v])=>`<div class="contrast-row"><span>${label}</span><div class="bar-track"><div class="bar-fill" style="background:${C.colors[r.name]};width:${100*v/max}%"></div></div><b>${(v*100).toFixed(2)}%</b></div>`).join('');
    $('roofExplanation').textContent=`当前 ${r.name} 的理论 FLOPs 比值为 ${ratio(r.flopsSpeedup)}，HBM 流量为 MHA 的 ${(100*r.traffic.totalBytes/baseline.traffic.totalBytes).toFixed(2)}%，理想时间下界比值为 ${ratio(r.rooflineSpeedup)}。${r.bottleneck==='Memory-Bound'?'该估算由访存限制：减少 FLOPs 不会自动按同一比例减少 HBM 流量。':'该估算由计算上限限制：仍需实际内核利用率和运行测量验证。'}两种比值均不是实测加速。`;
  }
  $('customTrafficInputs').innerHTML=names.map(n=>`<tr><td>${n}</td><td><input id="bytes-${n}" type="number" min="0" step="any" placeholder="未知" aria-label="${n} 区间 HBM 总 MiB"></td><td><input id="runtime-${n}" type="number" min="0" step="any" placeholder="未测量" aria-label="${n} 同范围实测 ms"></td></tr>`).join('');
  $('roofLegend').innerHTML=names.map(n=>`<label><input type="checkbox" data-roof-series="${n}" checked style="accent-color:${C.colors[n]}"><i class="dot" style="background:${C.colors[n]}"></i>${n}</label>`).join('');
  $('roofLegend').addEventListener('change',e=>{const n=e.target.dataset.roofSeries;if(n){e.target.checked?roofVisible.add(n):roofVisible.delete(n);renderRoofline();}});
  $('roofForm').addEventListener('submit',e=>e.preventDefault());
  function hardwareChange(e){if(['hardwareName','peakTF','bandwidth','bandwidthUnit','hardwarePrecision','deviceCount','hardwareTemplate'].includes(e.target.id))for(const n of names)$('runtime-'+n).value='';if(e.target.id==='deviceCount')for(const n of names)$('bytes-'+n).value='';if(['hardwareName','peakTF','bandwidth','bandwidthUnit','hardwarePrecision','deviceCount'].includes(e.target.id))$('hardwareTemplate').value='custom';}
  $('roofForm').addEventListener('input',e=>{hardwareChange(e);renderRoofline();});
  $('roofForm').addEventListener('change',e=>{hardwareChange(e);if(e.target.id==='roofScope'){$('scope').value=$('roofScope').value;$('scope').dispatchEvent(new Event('change'));}else renderRoofline();});
  $('customTrafficInputs').addEventListener('input',renderRoofline);
  function applyHardwareTemplate(id){
    if(!['h100','custom'].includes(id))throw new Error('未知硬件模板');
    const previousDevices=Number($('deviceCount').value);
    $('hardwareTemplate').value=id;
    for(const n of names)$('runtime-'+n).value='';
    if(id==='h100'){
      const h=C.hardwarePreset(id);
      for(const [field,key] of [['hardwareName','name'],['peakTF','peakTF'],['bandwidth','bandwidth'],['bandwidthUnit','bandwidthUnit'],['hardwarePrecision','precision'],['deviceCount','devices']])$(field).value=h[key];
      if($('precision').value!==h.precision){$('precision').value=h.precision;$('precision').dispatchEvent(new Event('change',{bubbles:true}));}
    }else{$('hardwareName').value='';$('peakTF').value='';$('bandwidth').value='';$('bandwidthUnit').value='GB/s';$('deviceCount').value=1;$('hardwarePrecision').value=A.getParameters().precision;}
    if(previousDevices!==Number($('deviceCount').value))for(const n of names)$('bytes-'+n).value='';
    renderRoofline();
  }
  $('hardwareTemplate').addEventListener('change',()=>applyHardwareTemplate($('hardwareTemplate').value));
  document.addEventListener('click',e=>{const b=e.target.closest('[data-roof]');if(b){roofSelected=b.dataset.roof;renderRoofline();}});
  $('roofChart').addEventListener('keydown',e=>{const b=e.target.closest('[data-roof]');if(b&&['Enter',' '].includes(e.key)){e.preventDefault();roofSelected=b.dataset.roof;renderRoofline();}});
  document.addEventListener('compute-lab-change',refresh);document.addEventListener('compute-lab-invalid',renderRoofline);$('legend').addEventListener('change',cumulative);
  $('exportRoofline').addEventListener('click',()=>{if(!roofData)return;const url=URL.createObjectURL(new Blob([JSON.stringify(roofData,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='roofline-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  A.getRooflineInput=()=>roofData||{schemaVersion:2,status:'awaiting-valid-hardware',attention:A.getExperiment(),bytesMoved:null};
  A.getRooflineAnalysis=()=>roofData;
  A.getRooflineSelection=()=>roofSelected;
  A.getRooflineDraft=()=>({hardware:readHardware(),traffic:readTraffic(),scope:$('roofScope').value});
  A.applyHardwareTemplate=applyHardwareTemplate;
  $('methodContent').insertAdjacentHTML('beforeend',`<h3>新增：生成区间与 Weight Absorption</h3><p>G 表示生成 token 数，避免与单步可见位置 N 混淆。默认 G=1024，Ftotal=Σ Fdecode(S+i)，i=0…G−1。MHA/GQA/MLA 的历史项按算术数列求和，SWA 使用 min(N,W) 分段前缀和，MSA 使用 local 末块周期求和，GDA 累计已知递归核心。所有主表与加速比基于单层、每序列的生成区间；展开矩阵默认展示第一步，另有逐步检查。Prefill 禁用。</p><p>MLA 原有路径不重复计算历史 K/V。新增两种离线吸收路径：WUQ[h] WUK[h]ᵀ → [rq,r]，替代内容 Query up 与 Query→latent；WUV[h] WO[h] → [r,D]，替代 Value up 与 O。原因子路径与融合路径经实数矩阵校验；离线预计算不计入每步，融合后的权重大小与 FLOPs 可能增加。此页仍排除 RMSNorm/RoPE，不能声称是完整官方实现。</p><h3>新增：Roofline 的 HBM 流量模型</h3><p>默认每设备每步读取所有当前执行的 Projection 权重，每组 KV 头在 Query Head 内理想复用；MLA latent 按内容 scores 与聚合两遍读取，位置 Key 一遍；MSA 主缓存稀疏读取加完整历史 Index Key 扫描；GDA 核心读写固定状态。当前 token 的 K/V 留在片上参与计算，并计缓存写入。可设置片上命中比例与 score/probability 回写，或输入当前区间总 HBM MiB。所有结果是声明假设的模型估计；完整 GDA 数据缺失时不绘图或排名。</p><p>多设备只按 Batch 中独立序列均分，权重每设备复制，Ppeak 与 BW 理想线性求和；不模拟张量并行。区间时间下界=max(ΣF/Ppeak,ΣBytes/BW)，未计串行阶段和真实利用率。自定义字节与 Runtime 只属于当前实验与范围，配置变化会清空，避免使用旧场景数据。</p>`);
  applyHardwareTemplate('h100');refresh();
})();
