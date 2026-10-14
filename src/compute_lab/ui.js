(function(){
  'use strict';
  const C=ComputeLab, $=id=>document.getElementById(id), names=C.architectures;
  let parameters={...C.defaults}, scenario={...C.scenarioDefaults}, selected='GQA', visible=new Set(names), all=[], training=null, plot=null;
  const metadata={
    D:['Hidden Size · D','隐藏维度'],H:['Query Heads · H','查询头数'],Hkv:['KV Heads · Hkv','共享 KV 头数'],dk:['Q/K Head · dk','MLA 为内容维度'],dv:['Value Head · dv','每头 Value 维度'],S:['Initial Context · S','第一步之前的历史'],batch:['Batch Size','独立序列数量'],
    W:['Window · W','含当前 token 的窗口'],topk:['Top-K Blocks','含强制保留的 local block'],block:['Block Size','每个 KV block 的位置数'],di:['Index Head · di','每 group 一头 / 共享 index key'],rank:['KV Latent Rank · r','缓存压缩维度'],qrank:['Query Rank · rq','查询压缩维度'],rope:['RoPE Head · dr','解耦位置维度，偶数']};
  const fmt=n=>n==null?'未知':n>=1e12?(n/1e12).toFixed(3)+' T':n>=1e9?(n/1e9).toFixed(3)+' G':n>=1e6?(n/1e6).toFixed(2)+' M':n>=1e3?(n/1e3).toFixed(2)+' K':n.toLocaleString('en-US');
  const mem=n=>n>=2**30?(n/2**30).toFixed(2)+' GiB':n>=2**20?(n/2**20).toFixed(2)+' MiB':n>=1024?(n/1024).toFixed(2)+' KiB':n+' B';
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function field(k){const [label,note]=metadata[k];return `<label class="field" for="${k}">${label}<input id="${k}" name="${k}" type="number" min="${k==='S'?0:1}" max="${k==='S'?1048576:k==='batch'?1024:65536}" step="${k==='rope'?2:1}" value="${parameters[k]}"><span>${note}</span></label>`;}
  $('contextField').innerHTML=field('S');
  $('commonFields').innerHTML=['D','H','Hkv','dk','dv','batch'].map(field).join('');
  function special(){
    const keys=selected==='MLA'?['rank','qrank','rope']:selected==='SWA'?['W']:selected==='MSA'?['topk','block','di']:[];
    $('specialFields').innerHTML=keys.length?`<div class="field-grid">${keys.map(field).join('')}</div>`:selected==='GDA'?`<label class="field">递归状态精度<select id="statePrecision">${Object.keys(C.bytes).map(x=>`<option ${parameters.statePrecision===x?'selected':''}>${x}</option>`).join('')}</select></label><p class="caption">每头状态 [dv, dk]。Hkv 不参与本递归定义。</p>`:'<p class="caption">此架构使用基础参数；窗口、索引与 latent 参数不适用。</p>';
    if(selected==='MLA')$('specialFields').insertAdjacentHTML('beforeend',`<label class="field">MLA 执行路径<select id="mlaImplementation"><option value="factorized">保留因子 · 原有路径</option><option value="absorbed-query">吸收 Query 权重</option><option value="absorbed-qo">吸收 Query + O 权重</option></select></label><p class="caption">离线融合不计入每步；融合矩阵可能更大，不保证 FLOPs 更少。</p>`);
    if($('mlaImplementation'))$('mlaImplementation').value=parameters.mlaImplementation;
  }
  function value(r){return $('scope').value==='core'?r.core:r.total;}
  function update(){
    const candidate={...parameters};
    for(const k of Object.keys(C.defaults)) if($(k)) candidate[k]=typeof C.defaults[k]==='number'?Number($(k).value):k==='scalar'?$(k).checked:$(k).value;
    // Empty number fields are invalid, including S (Number('') would be zero).
    const empty=Object.keys(metadata).filter(k=>$(k)&&$(k).value.trim()==='');
    const nextScenario={mode:$('tokenMode').value,generated:Number($('generated').value)};
    const errors=[...empty.map(k=>`${k} 不能为空`),...C.validateScenario(candidate,nextScenario)];
    if(nextScenario.mode!=='single'&&$('generated').value.trim()==='')errors.push('Generated Tokens 不能为空');
    let nextResults;
    if(!errors.length)try{nextResults=C.sequenceResults(candidate,nextScenario);}catch(e){errors.push(e.message);}
    $('error').hidden=!errors.length;
    $('error').textContent=errors.join('；');
    $('parameters').querySelectorAll('input[type=number]').forEach(e=>e.setAttribute('aria-invalid',String(empty.includes(e.id)||!Number.isSafeInteger(candidate[e.id]))));
    document.querySelectorAll('#export,#exportRoofline').forEach(e=>e.disabled=!!errors.length);
    if(errors.length){$('error').textContent+='。Attention 结果保留上一次有效配置；请修正输入后再比较。';document.dispatchEvent(new Event('compute-lab-invalid'));return;}
    parameters=candidate;scenario=nextScenario;all=nextResults;$('generatedField').hidden=scenario.mode==='single';render();
  }
  function render(){
    const p=parameters, current=all.find(r=>r.name===selected), baseline=all[0], v=value(current), base=value(baseline);
    $('scenarioText').textContent=`初始 S = ${p.S.toLocaleString()} · 生成 G = ${current.steps.toLocaleString()} · 首步可见 ${(p.S+1).toLocaleString()}`;
    const ratio=v==null?'未核验':(base/v).toFixed(2)+'×';
    const reduction=v==null?'理论计算量比值 · 非实测加速':v<=base?`较 MHA 减少 ${(100*(1-v/base)).toFixed(1)}% FLOPs`:`较 MHA 增加 ${(100*(v/base-1)).toFixed(1)}% FLOPs`;
    $('metrics').innerHTML=`<div class="metric"><small>${selected} / ${current.steps} Tokens · ${$('scope').value==='core'?'Core':'Module'}</small><strong>${fmt(v)}</strong><span>${selected==='GDA'&&v==null?'已知小计 '+fmt(current.known):'累计 FLOPs / layer / sequence'}</span></div><div class="metric"><small>MHA / ${selected}</small><strong>${ratio}</strong><span>${reduction}</span></div><div class="metric"><small>${selected} / ${current.name==='GDA'?'递归状态':'KV Cache'}</small><strong>${mem(current.memoryBytes)}</strong><span>末步整批容量${current.memoryComplete?'':' · 部分'}</span></div>`;
    $('ratioHeading').textContent='MHA / '+($('scope').value==='core'?'Core':'Total');
    $('comparison').querySelector('tbody').innerHTML=all.map(r=>`<tr class="${r.name===selected?'selected':''}"><td><button data-arch="${r.name}"><i class="dot" style="background:${C.colors[r.name]}"></i>${r.name}</button><small>${r.complete?r.name==='MLA'?'选定路径小计':'声明范围内': '完整模块未核验'}</small></td><td>${fmt(r.projection)}</td><td>${fmt(r.core)}</td><td>${fmt(r.extra)}</td><td>${fmt(r.total)}${!r.complete?`<small>已知 ${fmt(r.known)}</small>`:''}</td><td>${value(r)==null?'—':(base/value(r)).toFixed(2)+'×'}</td></tr>`).join('');
    $('architectureButtons').innerHTML=names.map(n=>`<button data-arch="${n}" class="${n===selected?'active':''}" aria-pressed="${n===selected}">${n}</button>`).join('');
    diagram(current.first);steps(current.first);memory();chart();
    $('architectureSummary').innerHTML=summary(current.first);
    $('rooflineSchema').textContent=JSON.stringify({schemaVersion:1,scenario:'decode',historyTokens:p.S,storageLevel:'HBM',precision:p.precision,flopsPerToken:current.total,bytesMoved:null,bytesEvidence:null},null,2);
    document.dispatchEvent(new CustomEvent('compute-lab-change'));
  }
  function summary(r){const p=parameters,ratio=(C.calculate('MHA',p).memoryBytes/r.memoryBytes).toFixed(2), L=r.active;
    const s={
      MHA:`每个 Query Head 拥有独立 K/V，读取全部 ${L.toLocaleString()} 个位置，作为全注意力基线。没有稀疏选择开销，但计算与缓存都随历史长度线性增长。`,
      GQA:`每 ${p.H/p.Hkv} 个 Query Head 共享一组 KV：KV 投影与缓存减少，当前容量比 MHA 小 ${ratio} 倍。所有 ${p.H} 个 Query Head 仍计算各自分数，所以核心 FLOPs 与 MHA 相同。长上下文、缓存受限时更有价值；共享 KV 改变模型表示能力。`,
      MLA:`历史只存 ${p.rank} 维 latent 和 ${p.rope} 维位置 Key。当前路径：${{'factorized':'保留 Query / Value 因子','absorbed-query':'吸收 Query 权重；保留 Value / O 因子','absorbed-qo':'吸收 Query 与 Value→O 权重'}[p.mlaImplementation]}。计算在 latent 空间进行；大秩可能增加核心 FLOPs。离线融合与原有因子路径在实数代数中等价，不重复执行已吸收的投影；有限精度舍入与实际性能不保证相同。`,
      SWA:`只读取最近 min(S+1, W) = ${L.toLocaleString()} 个位置。超过 W=${p.W.toLocaleString()} 后，核心 FLOPs 与滚动缓存不再随 S 增长。使用 GQA 投影；没有索引分支，但窗口之外的信息无法在本层直接访问。`,
      MSA:`每组独立选择至多 ${p.topk} 个 block（包括 local block），主分支当前读取 ${L.toLocaleString()} 个位置。${p.Hkv} 个 ${p.di} 维索引 Query 仍扫描全部 ${(p.S+1).toLocaleString()} 个 Index Key，因此总量不是完全常数。长上下文可减少主注意力计算；需支付索引、选择与不规则读取开销，缓存仍保留全部历史。`,
      GDA:`每头以 [${p.dv}, ${p.dk}] 状态递归更新，无需读取历史 KV；递归核心不随 S 增长。通过衰减与预测误差控制写入，状态占用 ${mem(r.memoryBytes)}（${p.statePrecision}）。固定状态可能限制历史检索能力。ShortConv、归一化和非线性门尚未核验，完整模块 FLOPs 与完整缓存未知。`
    };return s[r.name];
  }
  function diagram(r){
    const p=parameters, color=C.colors[r.name];
    const rect=(x,y,w,h,title,subtitle,fill='#f1f6ff')=>`<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="#cddbf1"/><text x="${x+w/2}" y="${y+21}" text-anchor="middle" class="diagram-text">${esc(title)}</text><text x="${x+w/2}" y="${y+39}" text-anchor="middle" font-size="10" fill="#718092">${esc(subtitle)}</text></g>`;
    const arrow=(x1,y1,x2,y2)=>`<path d="M${x1},${y1} L${x2},${y2}" fill="none" stroke="${color}" stroke-width="1.5" marker-end="url(#arrow)"/>`;
    let body='';
    if(['MHA','GQA'].includes(r.name)){
      const h=r.name==='MHA'?p.H:p.Hkv, count=Math.min(p.H,8), groups=new Set(Array.from({length:count},(_,i)=>r.name==='MHA'?i:Math.floor(i/(p.H/p.Hkv))));
      body+=rect(5,50,105,55,'当前 X',`[1, ${p.D}]`)+arrow(110,78,148,78);
      for(let i=0;i<count;i++){
        const x=153+i*77,g=r.name==='MHA'?i:Math.floor(i/(p.H/p.Hkv));
        body+=rect(x,10,65,49,`Q${i}`,`[1, ${p.dk}]`);
        const gx=153+([...groups].indexOf(g))*77;
        body+=arrow(x+32,60,gx+32,111);
      }
      [...groups].forEach((g,i)=>{body+=rect(153+i*77,117,66,49,`KV${g}`,`${r.visible} × K/V`,'#e9f7f2');});
      body+=`<text x="153" y="198" class="diagram-text">${p.H} Query Heads → ${h} KV Heads；${p.H>8?'示例显示前 8 头；':''}每组 ${p.H/h} 个 Query Head</text>`;
    }else if(r.name==='MLA'){
      body=rect(5,55,115,55,'当前 X',`[1, ${p.D}]`)+arrow(120,82,154,82)+rect(156,55,145,55,'Compression',`cKV: [1, ${p.rank}]`)+arrow(301,82,334,82)+rect(336,55,195,55,'历史 Latent Cache',`[${r.visible}, ${p.rank}] + KR: ${p.rope}`,'#f2edff')+arrow(531,82,566,82)+rect(568,55,180,55,'Latent Attention',`${p.H} heads → Value up`);
      body+=`<text x="5" y="155" class="diagram-text">Query: D → ${p.qrank} → ${p.mlaImplementation==='factorized'?`H×${p.dk} → `:''}H×${p.rank}；${p.mlaImplementation==='absorbed-qo'?'聚合 latent → 融合 O':'Value up → O'}；不展开历史 K/V。</text>`;
    }else if(r.name==='SWA'){
      const count=24;body+=`<text x="5" y="25" class="diagram-text">历史 + 当前：${r.visible} 个位置 / 窗口读取：${r.active} 个位置</text>`;
      for(let i=0;i<count;i++){const active=i>=count-Math.max(1,Math.round(count*r.active/r.visible));body+=`<rect x="${6+i*30}" y="52" width="24" height="43" rx="4" fill="${active?'#f5ac51':'#e8edf3'}"/>`;}
      body+=arrow(724,76,757,76)+`<text x="7" y="127" class="diagram-text">灰色：窗口外；橙色：最近 min(N, W) = ${r.active} 个位置</text><text x="7" y="155" font-size="11" fill="#718092">格子为比例示意；KV Heads = ${p.Hkv}，每头 dk=${p.dk}, dv=${p.dv}</text>`;
    }else if(r.name==='MSA'){
      body=rect(5,40,185,60,'Index Branch',`${p.Hkv} × ${p.di}；扫描 N=${r.visible}`)+arrow(190,70,225,70)+rect(227,40,240,60,'Block Max → Top-K',`k=${p.topk}, block=${p.block}；含 local`,'#fff0f5')+arrow(467,70,504,70)+rect(506,40,245,60,'Main Attention',`${r.active} positions × ${p.H} Query Heads`);
      const blocks=Math.ceil(r.visible/p.block), shown=Math.min(blocks,16), picks=Math.min(p.topk,blocks);
      for(let i=0;i<shown;i++)body+=`<rect x="${8+i*35}" y="132" width="29" height="18" rx="3" fill="${i>=shown-Math.min(picks,shown)?'#dc537f':'#e8edf3'}"/>`;
      body+=`<text x="8" y="184" font-size="11" fill="#718092">共 ${blocks} blocks；末块可见 ${p.S%p.block+1} 个位置；色块仅示意选择预算，不模拟实际索引分数。</text>`;
    }else{
      body=rect(5,48,150,62,'Recurrent State',`Rprev: [${p.H}, ${p.dv}, ${p.dk}]`,'#eff2f5')+arrow(155,79,190,79)+rect(192,48,180,62,'Decay & Read',`A = αRprev；u = A k`)+arrow(372,79,407,79)+rect(409,48,182,62,'Delta Update',`Rnew = A + β(v−u)kᵀ`)+arrow(591,79,625,79)+rect(627,48,135,62,'Output',`o = Rnew q`);
      body+=`<path d="M496,111 V145 H80 V113" fill="none" stroke="${color}" stroke-dasharray="4 3" marker-end="url(#arrow)"/><text x="235" y="174" class="diagram-text">传给下一个 token；状态尺寸不依赖历史 S</text>`;
    }
    $('architectureDiagram').innerHTML=`<svg viewBox="0 0 780 210" role="img" aria-label="${r.name} 参数驱动架构图"><defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="${color}"/></marker></defs>${body}</svg>`;
  }
  function steps(r){
    $('detailCount').textContent=` · ${r.rows.length} 步`;
    $('detailIntro').innerHTML=`FMA = 2 FLOPs；主计数为 2MKN；严格乘加 MN(2K−1) 只适用于矩阵点积。当前可见 N=S+1=${r.visible}，实际读取 L=${r.active}。<br>D=${parameters.D}，H=${parameters.H}，Hkv=${parameters.Hkv}，dk=${parameters.dk}，dv=${parameters.dv}，r=${parameters.rank}，rq=${parameters.qrank}，dr=${parameters.rope}，di=${parameters.di}；维度均为元素数。<br>当前已知小计 ${fmt(r.known)} FLOPs；整批已知小计 ${fmt(r.known*parameters.batch)} FLOPs。${r.complete?'':`完整模块未知：${r.unknown.join('；')}。`}`;
    $('steps').innerHTML=r.rows.map((s,i)=>`<article class="step"><div class="step-head"><b>${i+1}. ${s.label} <small>· ${{projection:'Projection',core:'Core',extra:'Additional'}[s.category]}</small></b><span>${fmt(s.approx)} FLOPs${s.excluded?' · 本口径排除':''}</span></div><p>${s.purpose}</p><div class="matrix-flow"><span class="matrix-box">${s.input}</span><span>${s.kind==='matmul'?'×':'→'}</span><span class="matrix-box">${s.weight}</span><span>→</span><span class="matrix-box">${s.output}</span>${s.groups>1?`<span>× ${s.groups} heads</span>`:''}</div><p><code>${esc(s.formula)}</code></p>${s.kind==='matmul'?`<p>近似：2 × ${s.input.slice(1,-1).replace(', ',' × ')} × ${s.weight.slice(1,-1).split(', ')[1]} × ${s.groups} = ${s.approx.toLocaleString('en-US')} FLOPs<br>严格标量乘加：MN(2K−1) × ${s.groups} = ${s.exact.toLocaleString('en-US')} FLOPs</p>`:`<p>按上方明示计数约定：${s.approx.toLocaleString('en-US')} FLOPs；不是硬件指令计数。</p>`}</article>`).join('');
  }
  function memory(){const max=Math.max(...all.map(r=>r.memoryBytes)), base=all[0].memoryBytes;
    $('memoryBars').innerHTML=all.map(r=>`<div class="memory-row"><b>${r.name}</b><div class="bar-track"><div class="bar-fill" style="background:${C.colors[r.name]};width:${Math.max(.5,100*r.memoryBytes/max)}%"></div></div><span class="memory-value">${mem(r.memoryBytes)}</span><span class="memory-value">${(100*r.memoryBytes/base).toFixed(1)}%</span><span class="memory-note">${r.memoryNote} · ${r.name==='SWA'?'O(min(N,W))':r.name==='GDA'?'O(H dk dv)，部分状态':'O(N)'}；百分比为 MHA 容量占比</span></div>`).join('');
  }
  function chart(){
    const isMemory=$('chartMode').value==='memory',log=$('yScale').value==='log',getValue=r=>isMemory?r.memoryBytes:value(r),format=isMemory?mem:fmt;
    const maxS=Math.max(Number($('chartMax').value),parameters.S,1), left=68,right=880,top=24,bottom=262;
    const points=new Set([0,maxS,parameters.S,Math.min(maxS,parameters.W-1)]);
    for(let i=0;i<=120;i++)points.add(Math.round(i*maxS/120));
    if(maxS/parameters.block<=512) for(let s=parameters.block;s<=maxS;s+=parameters.block){points.add(s-1);points.add(s);}
    const ss=[...points].sort((a,b)=>a-b), series=names.filter(n=>visible.has(n)).map(n=>({name:n,points:ss.map(s=>({s,v:getValue(C.calculate(n,{...parameters,S:s}))})).filter(a=>a.v!=null)})).filter(a=>a.points.length);
    const ymax=Math.max(1,...series.flatMap(a=>a.points.map(p=>p.v)))*1.12;
    const ymin=log?Math.max(1,Math.min(...series.flatMap(a=>a.points.map(p=>p.v)),ymax/10)*.8):0;
    const x=s=>left+s/maxS*(right-left),y=v=>bottom-(log?(Math.log10(v)-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin)):v/ymax)*(bottom-top);
    plot={maxS,left,right,top,bottom,ymax,x,y,isMemory};
    let svg=`<text x="${left}" y="13" font-size="10" fill="#718092">${isMemory?'Cache / State bytes / batch / layer':'FLOPs / token / layer / sequence'}${log?' · log scale':''}</text>`;
    for(let i=0;i<=4;i++){const yy=bottom-(bottom-top)*i/4,v=log?10**(Math.log10(ymin)+(Math.log10(ymax)-Math.log10(ymin))*i/4):ymax*i/4;svg+=`<path d="M${left},${yy}H${right}" stroke="#e9eef4"/><text x="${left-9}" y="${yy+4}" text-anchor="end" font-size="10" fill="#718092">${format(v)}</text>`;}
    for(let i=0;i<=4;i++){const s=maxS*i/4;svg+=`<text x="${x(s)}" y="285" text-anchor="middle" font-size="10" fill="#718092">${Math.round(s).toLocaleString()}</text>`;}
    svg+=`<text x="${(left+right)/2}" y="307" text-anchor="middle" font-size="10" fill="#718092">History Context Length · S（可见位置 N = S + 1）</text>`;
    series.forEach(a=>{svg+=`<path d="${a.points.map((p,i)=>(i?'L':'M')+x(p.s).toFixed(2)+','+y(p.v).toFixed(2)).join(' ')}" fill="none" stroke="${C.colors[a.name]}" stroke-width="${a.name==='GQA'?2:2.4}" ${a.name==='GQA'?'stroke-dasharray="7 5"':''}/>`;});
    const currentX=x(parameters.S);svg+=`<path d="M${currentX},${top}V${bottom}" stroke="#b7c5d7" stroke-dasharray="3 4"/><text x="${Math.min(currentX+5,right-100)}" y="${top+10}" font-size="10" fill="#7b8ca0">当前 S=${parameters.S.toLocaleString()}</text><g id="hoverMarker"></g>`;
    if(!series.length)svg+=`<text x="450" y="140" text-anchor="middle" font-size="14" fill="#718092">当前没有可绘制的已知曲线</text>`;
    $('chart').innerHTML=svg;
    $('chart').setAttribute('aria-label',isMemory?'历史 Context 长度与缓存 / 状态容量曲线':'历史 Context 长度与每 token FLOPs 曲线');
    $('chartNote').textContent=isMemory?'容量按整批单层计算：MSA 仍保留完整历史；SWA 可淘汰窗口外 KV；GDA 只绘制递归状态，未含 ShortConv 缓冲。容量不是 HBM Bytes Moved。':$('scope').value==='core'?'GQA 虚线与 MHA 重合。MSA 为主分支核心，不含索引；GDA 核心包含衰减、误差、状态更新与输出读取。':'MSA 总量含索引扫描，仍随 S 增长；末块长度导致锯齿变化。GDA 完整模块未核验，不绘制 Total 曲线。';
    if(!parameters.scalar&&!isMemory)$('chartNote').textContent+=' 当前仅统计矩阵运算，GDA 外积与状态逐元素操作未计入。';
    $('hoverReadout').textContent='在图上移动查看数值';
  }
  $('chart').addEventListener('pointermove',e=>{if(!plot)return;const bounds=$('chart').getBoundingClientRect(), px=(e.clientX-bounds.left)*900/bounds.width,s=Math.max(0,Math.min(plot.maxS,Math.round((px-plot.left)/(plot.right-plot.left)*plot.maxS)));
    const data=names.filter(n=>visible.has(n)).map(n=>{const r=C.calculate(n,{...parameters,S:s});return {n,v:plot.isMemory?r.memoryBytes:value(r)};});
    $('hoverReadout').textContent=`S=${s.toLocaleString()} · `+data.map(a=>a.n+' '+(plot.isMemory?mem:fmt)(a.v)).join(' / ');
    $('hoverMarker').innerHTML=`<path d="M${plot.x(s)},${plot.top}V${plot.bottom}" stroke="#8ba0ba"/>`+data.filter(a=>a.v!=null).map(a=>`<circle cx="${plot.x(s)}" cy="${plot.y(a.v)}" r="3.5" fill="${C.colors[a.n]}"/>`).join('');
  });
  function selectArchitecture(n){selected=n;special();render();}
  document.addEventListener('click',e=>{const button=e.target.closest('[data-arch]');if(button)selectArchitecture(button.dataset.arch);const go=e.target.closest('[data-go]');if(go)tab(go.dataset.go);});
  $('parameters').addEventListener('submit',e=>e.preventDefault());$('parameters').addEventListener('input',update);$('parameters').addEventListener('change',update);
  $('scope').addEventListener('change',render);for(const id of ['chartMax','chartMode','yScale'])$(id).addEventListener('change',chart);
  $('legend').innerHTML=names.map(n=>`<label><input type="checkbox" data-series="${n}" checked style="accent-color:${C.colors[n]}"><i class="dot" style="background:${C.colors[n]}"></i>${n}</label>`).join('');
  $('legend').addEventListener('change',e=>{const n=e.target.dataset.series;if(n){e.target.checked?visible.add(n):visible.delete(n);chart();}});
  $('reset').addEventListener('click',()=>{parameters={...C.defaults};scenario={...C.scenarioDefaults};$('tokenMode').value=scenario.mode;$('generated').value=scenario.generated;for(const k of Object.keys(C.defaults))if($(k)){if(k==='scalar')$(k).checked=parameters[k];else $(k).value=parameters[k];}special();update();});
  function tab(id){for(const n of ['attention','roofline','mfu']){$(n).hidden=n!==id;$('tab-'+n).setAttribute('aria-selected',String(n===id));$('tab-'+n).tabIndex=n===id?0:-1;}document.dispatchEvent(new CustomEvent('compute-lab-tab-change',{detail:id}));}
  document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>tab(b.dataset.tab)));
  document.querySelector('.tabs').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;const list=[...document.querySelectorAll('[data-tab]')],i=list.indexOf(document.activeElement),j=e.key==='Home'?0:e.key==='End'?2:(i+(e.key==='ArrowRight'?1:2))%3;e.preventDefault();list[j].focus();tab(list[j].dataset.tab);});
  function download(data,name){const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $('export').addEventListener('click',()=>download(C.exportSequence(parameters,scenario),'attention-experiment.json'));
  const template={schemaVersion:1,architecture:'请填写模型架构',source:'请填写已核验来源或本地审计文件',verified:false,precision:'BF16',forwardFlopsPerToken:null,backwardFlopsPerToken:null,tokensPerSecond:null,tokensPerStep:null,stepSeconds:null,deviceCount:null,peakFlopsPerSecondPerDevice:null,recomputationExecutionFlopsPerToken:null};
  $('trainingSchema').textContent=JSON.stringify(template,null,2);$('trainingTemplate').addEventListener('click',()=>download(template,'training-flops-template.json'));
  $('trainingFile').addEventListener('change',async()=>{const f=$('trainingFile').files[0];if(!f)return;try{if(f.size>1048576)throw new Error('JSON 文件不得超过 1 MiB');training=C.validateTraining(JSON.parse(await f.text()));$('trainingSchema').textContent=JSON.stringify(training,null,2);$('importStatus').textContent='已通过格式校验；verified 是提供者声明，来源仍需人工核验。导入与 MFU 教学假设分开；尚未进行真实训练 MFU 分析。';}catch(e){training=null;$('importStatus').textContent='导入失败：'+e.message;$('trainingSchema').textContent=JSON.stringify(template,null,2);}});
  $('methodContent').innerHTML=`<h3>统一实验与单位</h3><p>默认只做 Decode：输入 X=[1,D]，历史 S 个 token；生成当前第 S+1 个，当前 K/V 参与注意力。N=S+1；W 与 sparse budget 均包含当前位置。按单层 / 每序列 / 每新 token 输出 FLOPs；内存按整批输出。Batch 扩展量单列，曲线不乘 Batch。</p><h3>计数约定</h3><p>FMA = 2 FLOPs。矩阵乘法 X[M,K] × W[K,N] 默认 2MKN，严格点积为 MN(2K−1)。外积每元素只有一次乘法，不能套用近似 2MKN 来称为严格计数。</p><p>启用标量时：加、减、乘、除、exp 各简化计 1 FLOP。稳定 Softmax 为每头 4L−1（L 次减最大值、L 次 exp、L−1 次求和、L 次除法），另计 L 次 scale。max、Top-K、block max pooling 的比较与整数索引不算浮点 FLOPs，但会消耗时间。关闭标量时总量仅包含矩阵运算，图表同步改变。</p><h3>计算范围与可比性</h3><p>MHA / GQA / SWA 是 Q、K、V、O 投影与注意力核心。统一排除位置旋转、Bias、残差、全局归一化、FFN、量化 / 反量化与通信。Total 是这个声明范围内的 Attention 小计，不是模型整体 FLOPs 或硬件指令数。</p><p>MLA 保留 rq 查询压缩、dk 内容查询、dr 解耦位置查询，先将 Query 按头变换至 r 维 latent，直接在 latent Cache 上做注意力，再做 Value up 和 O 投影；不展开历史 K/V，不完全预融合 Query / O 权重。dk 在 MLA 中是内容维度，实际 score 还加 dr 维位置部分；未计 RMSNorm 与 RoPE 旋转，不能称为完整 DeepSeek 实现或等容量对比。</p><p>MSA 使用论文 §3.1 的 group-specific Index Q、单个共享 Index K、block max pooling；Decode 采用 exp-free 选择。k 包含 local block，本工具采用总选择预算 k 的强制保留约定。已选 local 块长度 ℓ=(N−1) mod block+1，其余块均完整，L=ℓ+(min(k,ceil(N/block))−1)×block。没有真实索引输入，不声称选出了某些真实 block。索引 score 仍扫描所有 N 行；正 scale 显式计入，可被具体选择内核省去。训练 KL loss 不在 Decode 范围。</p><p>GDA 按 Eq.10：A=αRprev，u=Ak，δ=β(v−u)，Rnew=A+δkᵀ，o=Rnew q。R 的转置布局不影响 FLOPs。完整标量核心为 H(7dkdv+2dv)，仅矩阵时为 4Hdkdv；这是本递归算法的计数。已知投影包括 Q/K/V/O、α/β 与输出门；ShortConv、激活、归一化和输出门乘法未核验，Extra 与 Total 为 null。仅核心比值可比较，且算法语义与 Softmax 不等价。</p><h3>内存与性能</h3><p>逻辑缓存：MHA N H(dk+dv)b；GQA N Hkv(dk+dv)b；SWA min(N,W)Hkv(dk+dv)b；MLA N(r+dr)b；MSA N[Hkv(dk+dv)+di]b；GDA 仅状态 H dk dv bstate。b 为精度字节。均乘 Batch。GDA 的完整记忆还需 ShortConv 缓冲。量化 scales、对齐、临时激活、权重未计入。</p><p>容量不等于 HBM 流量；trafficBytes、weightTrafficBytes 留 null。FLOPs 是运算量，FLOP/s 是速率，Ratio 是理论计算量比值，均不是实测 wall-clock speedup。FP8 仅按 1 字节存储作教学估算。</p><h3>来源与适用条件</h3><ul>${names.map(n=>`<li>${n}：<a href="${C.sources[n][1]}" target="_blank" rel="noopener noreferrer">${C.sources[n][0]}</a></li>`).join('')}</ul><p>架构来源于上述论文；FLOPs 由其矩阵形状与选定执行路径推导。MLA 与 GDA 不是现有 DeepSeek-V4.1 配置。公式验证请参见同目录 tests.js 与验证报告。</p>`;
  for(const id of ['methodButton','footerMethod'])$(id).addEventListener('click',()=>$('methodology').showModal());$('closeMethod').addEventListener('click',()=>$('methodology').close());
  $('methodology').addEventListener('click',e=>{if(e.target===$('methodology')){const r=$('methodology').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('methodology').close();}});
  window.LLMComputeLab={getExperiment:()=>C.exportSequence(parameters,scenario),getParameters:()=>({...parameters}),getScenario:()=>({...scenario}),getSelection:()=>selected,getScope:()=>$('scope').value,getVisible:()=>[...visible],getRooflineInput:()=>C.rooflineInput(parameters),getTrainingInput:()=>training,importTraining:data=>{training=C.validateTraining(data);$('trainingSchema').textContent=JSON.stringify(training,null,2);$('importStatus').textContent='已通过格式校验；导入与 MFU 教学假设分开；尚未进行真实训练 MFU 分析。';return training;}};
  window.LLMComputeLab.applyAttentionParameters=changes=>{
    const candidate={...parameters,...changes};
    // Validate the entire scenario and accumulated safe-integer counts before any mutation.
    const next=C.sequenceResults(candidate,scenario);
    parameters=candidate;all=next;
    for(const k of Object.keys(changes))if($(k))$(k).value=parameters[k];
    $('error').hidden=true;render();
  };
  special();tab('attention');update();
})();
