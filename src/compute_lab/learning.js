(function(){
  'use strict';
  const C=ComputeLab,A=LLMComputeLab,$=id=>document.getElementById(id);
  const n=v=>v==null?'未知':v.toLocaleString('en-US',{maximumFractionDigits:4});
  const sci=v=>{if(v==null)return '未知';if(v===0)return '0';const e=Math.floor(Math.log10(Math.abs(v)));return `${n(v/10**e)} × 10<sup>${e}</sup>`;};
  const tf=v=>v==null?'未知':n(v/1e12)+' TFLOP/s';
  const ms=v=>v==null?'未知':n(v*1000)+' ms';
  const bound=r=>r.bottleneck==='Memory-Bound'?'Memory-bound（带宽限制，等数据）':'Compute-bound（算力限制，等计算）';
  function step(title,formula,meaning){return `<article><b>${title}</b><div class="substitution">${formula}</div><p>${meaning}</p></article>`;}
  function currentRoof(){return A.getRooflineAnalysis()?.results.find(r=>r.name===A.getRooflineSelection());}
  function roofLesson(){
    $('roofWorkedExample').innerHTML='';
    $('roofDemoResult').textContent='可调整带宽、算力或历史长度，查看同一架构在修改前后的变化。';
    $('roofDemoResult').classList.remove('demo-error');
    roofTeachingLesson();
    const data=A.getRooflineAnalysis(),h=C.hardwarePreset('h100'),official=$('hardwareTemplate').value==='h100';
    $('hardwareEvidence').innerHTML=official?`<b>官方规格 · H100 SXM 80GB</b><br>BF16 Dense：989 TFLOP/s<br>HBM：3.35 TB/s · ${n(Number($('deviceCount').value))} 张设备<details><summary>规格来源与 Dense / Sparse 区别</summary><p><a href="${h.bandwidthSource}" target="_blank" rel="noopener noreferrer">NVIDIA H100 产品规格</a>：SXM 80GB，HBM 3.35 TB/s。官网 BFLOAT16 1,979 TFLOP/s 带 “With sparsity” 注脚，属于稀疏加速口径。</p><p><a href="${h.computeSource}" target="_blank" rel="noopener noreferrer">NVIDIA 官方 recsys-examples 基准说明 §2</a>明确以 989 BF16 Dense Tensor Core TFLOP/s / H100 计算。本模板引用此 Dense 值，公开规格为取整值，不使用 1,979 作为 Dense 峰值。</p><p>核验日期：2026-10-13。带宽按十进制单位换算。Ascend 本轮没有同时核验型号、精度、Dense 峰值和 HBM 带宽的完整预设；可在“自定义”填入有依据的规格。</p></details>`:'<b>自定义硬件 / 教学假设</b><br>修改后的数值不再代表已核验的 H100 规格；请说明型号、精度、Dense / Sparse 与来源。';
    if(!data){$('roofLesson').innerHTML='<h3>等待有效输入</h3><p>查看左侧错误说明，或选择 H100 模板自动填充已核验规格。未计算的数据不会保留为当前示例。</p>';$('roofInterpretation').textContent='参数缺失或不一致，暂时不能给出结论。';$('roofPointHint').textContent='等待有效的硬件参数。';return;}
    const r=currentRoof(),f=r.flops,m=r.traffic.totalBytes;
    const p=data.hardware.peakTF*1e12*data.hardware.devices,b=data.hardware.bandwidth*(data.hardware.bandwidthUnit==='TB/s'?1e12:1e9)*data.hardware.devices;
    const provenance=r.traffic.kind==='modeled-estimate'?'访存为假设模型估算':'访存为用户输入的估计';
    $('roofLesson').innerHTML=`<h3>${r.name} · ${n(data.generatedTokens)} 个 token 的自回归生成示例</h3><p>初始历史 ${n(data.initialContext)} 个 token；${data.hardware.precision}；${n(data.hardware.devices)} 张设备；单层、整批 ${data.parameters.batch} 个独立序列、${data.scope==='core'?'核心运算':'声明范围内模块'}。输入与 Attention 页同步。</p><div class="number-flow"><div><small>理论工作量 F</small><b>${sci(f)}</b><span>FLOPs · 架构推导</span></div><span>÷</span><div><small>估计搬运量 M</small><b>${sci(m)}</b><span>B · ${r.traffic.kind==='modeled-estimate'?'模型假设':'用户估计'}</span></div><span>＝</span><div><small>计算密度 AI</small><b>${n(r.ai)}</b><span>FLOPs / Byte</span></div></div><p class="caption">${provenance}，不是 HBM 实测数据。默认假设：每步读权重、历史 KV 无片上命中、头间理想复用、中间分数留在片上；左侧可以展开修改。</p>`;
    if(r.ai==null){$('roofLesson').insertAdjacentHTML('beforeend','<p class="notice">完整 FLOPs 或搬运量未知，无法代入屋顶线；可选择其他架构，或切换已核验的核心范围。</p>');$('roofInterpretation').textContent='所选架构缺少完整数据，不给出当前范围的瓶颈或加速结论。';$('roofPointHint').textContent='当前选择缺少完整数据，不能判断瓶颈。';return;}
    $('roofWorkedExample').innerHTML=`<h2>把当前示例的数字逐步代入</h2><details class="lesson-formulas" open><summary>用上面的数字，逐步算出性能与时间</summary><div class="formula-steps">${step('① 先算计算密度',`AI = F / M = ${sci(f)} / ${sci(m)} = ${n(r.ai)} FLOPs/B`,`每搬运 1 字节，完成约 ${n(r.ai)} 次浮点运算。`)}${step('② 把硬件单位换成同一口径',`P_peak = ${n(data.hardware.peakTF)} × 10¹² × ${data.hardware.devices} = ${sci(p)} FLOP/s<br>BW = ${n(data.hardware.bandwidth)} × ${data.hardware.bandwidthUnit==='TB/s'?'10¹²':'10⁹'} × ${data.hardware.devices} = ${sci(b)} B/s`,'先换单位，再按设备数求和。TFLOP/s 是速度，TFLOPs 是工作量。')}${step('③ 分别检查两个性能上限',`算力上限 = ${tf(p)}<br>带宽上限 = AI × BW = ${n(r.ai)} × ${sci(b)} = ${tf(r.ai*b)}<br>P = min(P_peak, AI × BW) = ${tf(r.performance)}`,'取两者中较小的数：两个条件都必须满足，才可能完成计算。')}${step('④ 找转折点，判断先碰到哪条线',`AI_ridge = P_peak / BW = ${sci(p)} / ${sci(b)} = ${n(p/b)} FLOPs/B<br>${n(r.ai)} ${r.ai<p/b?'&lt;':'≥'} ${n(p/b)} → ${bound(r)}`,'交点处算力上限和带宽上限相等。交点左侧带宽先限制性能；右侧算力先限制性能。')}${step('⑤ 由工作量和速度，得到理想时间下界',`T_compute = F / P_peak = ${sci(f)} / ${sci(p)} = ${ms(r.computeSeconds)}<br>T_memory = M / BW = ${sci(m)} / ${sci(b)} = ${ms(r.memorySeconds)}<br>T_lower = max(T_compute, T_memory) = ${ms(r.lowerSeconds)}`,'这表示理想模型下至少需要的时间，不是运行预测或实测耗时。')}</div></details>`;
    const rows=data.results.filter(x=>x.ai!=null);
    $('roofInterpretation').innerHTML=`<p class="lesson-result">当前 ${r.name}：${bound(r)}。${r.ai<p/b?'搬数据的理论时间比计算更长；优先观察能否减少字节或增加有效带宽。':'计算的理论时间至少同样长；优先观察能否减少运算或提高有效算力。'}这些判断只在当前访存与硬件假设下成立。</p><div class="architecture-insights">${rows.map(x=>`<p><b style="color:${C.colors[x.name]}">${x.name}</b>：AI ${n(x.ai)}，${bound(x)}；时间下界 ${ms(x.lowerSeconds)}。${x.name==='MHA'?'作为对照。':`相对 MHA，计算量比值 ${n(x.flopsSpeedup)}×，理论时间下界比值 ${n(x.rooflineSpeedup)}×，由计算与搬运共同决定。`}</p>`).join('')}</div>${rows.length<6?'<p class="caption">缺少数据的架构不参加当前范围的瓶颈或时间排名；完整模块中的 GDA 不用已知小计代替总量。</p>':''}`;
    pointHint(r.name);
  }
  function pointHint(name,measured=false){const r=A.getRooflineAnalysis()?.results.find(x=>x.name===name);if(!r||r.ai==null)return;$('roofPointHint').textContent=measured?`${name} 空心点：AI=${n(r.ai)} FLOPs/B，用户实测 Runtime 换算性能 ${tf(r.measuredPerformance)}；理想上界 ${tf(r.performance)}。横坐标仍来自估计访存，测量来源与范围由输入者负责。`:`${name} 实心点：横坐标 ${n(r.ai)} FLOPs/B，纵坐标 ${tf(r.performance)}。${r.ai<r.ridge?'在交点左侧，带宽先限制速度。':'在交点右侧或交点处，算力先限制速度。'}这是假设下的上界；点击可把它作为示例逐步代入。`;}
  for(const event of ['pointerover','focusin'])$('roofChart').addEventListener(event,e=>{const g=e.target.closest('[data-roof]');if(g)pointHint(g.dataset.roof,!!g.dataset.measured);});
  let teachingSelected='100';
  function roofTeachingLesson(){
    const data=A.getRooflineAnalysis();
    if(!data){$('roofTeachingChoices').innerHTML='';$('roofTeachingDetail').textContent='硬件或场景无效，先修正输入再查看教学点。';return;}
    const points=C.rooflineTeachingPoints(data.hardware),r=points.find(x=>x.id===teachingSelected),h=data.hardware,peak=h.peakTF*1e12*h.devices,bw=h.bandwidth*(h.bandwidthUnit==='TB/s'?1e12:1e9)*h.devices;
    $('roofTeachingChoices').innerHTML=points.map(p=>`<button type="button" data-roof-teaching="${p.id}" aria-pressed="${p.id===teachingSelected}">◆ AI ${p.ai}<small>${p.bottleneck==='Memory-Bound'?'带宽限制':'算力限制'} · ${tf(p.performance)}</small></button>`).join('');
    $('roofTeachingDetail').innerHTML=`<p><b>教学假设 AI = ${r.ai} FLOPs/Byte</b>，沿用当前硬件；没有假设 Attention 测量结果。</p><p class="substitution">AI_ridge = ${tf(peak)} ÷ ${n(bw/1e12)} TB/s = ${n(r.ridge)} FLOPs/B<br>P_upper = min(P_peak, AI × BW)<br>= min(${n(peak/1e12)}, ${r.ai} × ${n(bw/1e12)}) = ${n(r.performance/1e12)} TFLOP/s</p><p>${r.ai} ${r.ai<r.ridge?'小于':'大于或等于'} 转折点 ${n(r.ridge)}，所以是 ${bound(r)}。${r.ai<r.ridge?'搬运 1 字节后可做的计算偏少，数据供给先成为限制。':'每字节带来的计算足够多，先碰到芯片的计算上限。'}修改硬件后会重新判断，不固定两点的瓶颈。</p>`;
    document.querySelectorAll('#roofChart [data-roof-teaching]').forEach(g=>g.setAttribute('aria-pressed',String(g.dataset.roofTeaching===teachingSelected)));
  }
  function chooseTeaching(e){const g=e.target.closest('[data-roof-teaching]');if(!g)return;if(e.type==='keydown'){if(!['Enter',' '].includes(e.key))return;e.preventDefault();}teachingSelected=g.dataset.roofTeaching;roofTeachingLesson();}
  document.addEventListener('click',chooseTeaching);$('roofChart').addEventListener('keydown',chooseTeaching);
  document.addEventListener('compute-lab-roofline-change',roofLesson);
  function attentionLesson(){const data=A.getExperiment(),name=A.getSelection(),r=data.results.find(x=>x.name===name),scope=A.getScope(),f=scope==='core'?r.core:r.total,base=scope==='core'?data.results[0].core:data.results[0].total;
    $('attentionLesson').innerHTML=f==null?`当前 ${name} 的完整模块尚未核验；保留已知分项，不用未知总量给出加速比。`:`当前示例：已有 ${n(data.historyTokens)} 个历史 token，再生成 ${n(data.generatedTokens)} 个；${name} 单层每序列累计 ${sci(f)} FLOPs。MHA ÷ ${name} = ${sci(base)} ÷ ${sci(f)} = ${n(base/f)}×。${base/f>=1?'数值越大，说明需要的计算越少。':'低于 1×，说明本配置的计算量更多。'}这是理论工作量对照，不是实测速度。`;
  }
  document.addEventListener('compute-lab-change',attentionLesson);
  function demo(kind){
    const reject=message=>{$('roofDemoResult').textContent=message+' 当前配置与计算结果未修改。';$('roofDemoResult').classList.add('demo-error');document.dispatchEvent(new CustomEvent('compute-lab-demo-change',{detail:{text:$('roofDemoResult').textContent}}));};
    const before=currentRoof();if(!$('error').hidden||!before||before.ai==null){reject('请先修正无效输入，并选择具有完整 FLOPs 与搬运量的架构。');return;}
    const a={...before},p=A.getParameters(),s=A.getScenario(),draft=A.getRooflineDraft(),h={...draft.hardware},o={...draft.traffic,runtimesMs:{}};
    const candidate={...p};
    if(kind==='context'){
      candidate.S=p.S*2;
      const g=s.mode==='single'?1:s.generated;
      if(candidate.S+g-1>1048576){reject(`无法加倍：S=${n(candidate.S)}，G=${n(g)}，最后一步历史 S+G−1=${n(candidate.S+g-1)} 超过 1,048,576。请先减少历史长度或生成数量。`);return;}
      if(candidate.S===p.S){reject('当前历史长度为 0，加倍仍为 0；请先输入正的历史长度。');return;}
      o.customBytes={};
    }else h[kind==='bandwidth'?'bandwidth':'peakTF']*=2;
    try{
      // Evaluate the entire prospective experiment before touching an input or dispatching events.
      C.sequenceResults(candidate,s);
      const preview=C.analyzeRoofline(candidate,s,h,o,draft.scope).results.find(r=>r.name===a.name);
      if(preview.ai==null)throw new Error('新场景缺少访存数据；请先选择模型估算，或为新场景另行提供总字节');
    }catch(e){reject('无法执行演示：'+e.message+'。');return;}
    if(kind==='context')A.applyAttentionParameters({S:candidate.S});else input(kind==='bandwidth'?'bandwidth':'peakTF',kind==='bandwidth'?h.bandwidth:h.peakTF);
    const after=currentRoof(),sameAI=after.ai===a.ai,samePerformance=after.performance===a.performance;
    const shape=kind==='bandwidth'?'带宽加倍，斜线抬高，水平算力线保持不变。':kind==='compute'?'算力加倍，水平线抬高，斜线保持不变。':'历史长度加倍，硬件屋顶线保持不变；FLOPs 与搬运量重新计算。';
    const unchanged=samePerformance?(after.bottleneck==='Compute-Bound'?'算力上限仍是较低的上限，所以当前性能上界没有提高。':'带宽上限仍是较低的上限，所以当前性能上界没有提高。'):'较低的性能上限发生变化，因此性能上界随之变化；实际运行仍需测量。';
    $('roofDemoResult').textContent=`${shape} Ridge ${n(a.ridge)} → ${n(after.ridge)} FLOPs/B。${a.name} 的 AI ${n(a.ai)} → ${n(after.ai)}（${sameAI?'保持不变':'发生变化'}）；${bound(a)} → ${bound(after)}。性能上界 ${tf(a.performance)} → ${tf(after.performance)}；时间下界 ${ms(a.lowerSeconds)} → ${ms(after.lowerSeconds)}。${unchanged}${kind==='context'?'':'修改后的硬件参数是教学假设。'}`;
    document.dispatchEvent(new CustomEvent('compute-lab-demo-change',{detail:{text:`${shape}Ridge ${n(a.ridge)} → ${n(after.ridge)} FLOPs/B；${a.name} AI ${n(a.ai)} → ${n(after.ai)}，${after.bottleneck===a.bottleneck?'仍为':'变为'}${after.bottleneck==='Memory-Bound'?'带宽':'算力'}限制。上界 ${tf(a.performance)} → ${tf(after.performance)}。${samePerformance?'当前限制未改善，上界保持不变。':''}当前变化仅为假设模型对照。`}}));
  }
  const input=(id,value)=>{$(id).value=value;$(id).dispatchEvent(new Event('input',{bubbles:true}));};
  $('demoBandwidth').addEventListener('click',()=>demo('bandwidth'));
  $('demoCompute').addEventListener('click',()=>demo('compute'));
  $('restoreHardware').addEventListener('click',()=>{A.applyHardwareTemplate('h100');$('roofDemoResult').textContent='已恢复 H100 BF16 Dense 989 TFLOP/s、3.35 TB/s、1 张设备，并将 Attention 精度设为 BF16。'+(A.getRooflineAnalysis()?'其余 Attention 维度与访存假设保留。':'Attention 还有无效输入，请先修正。');});
  $('demoContext').addEventListener('click',()=>demo('context'));
  $('restoreAttention').addEventListener('click',()=>{$('reset').click();$('roofDemoResult').textContent='已恢复 Attention 默认维度、BF16、初始历史 8192 和连续生成 1024 个 token；自定义硬件输入保留。'+(A.getRooflineAnalysis()?'':'当前硬件与默认 Attention 不匹配，请检查精度及设备数。');});

  let mfuResult=null,previousPercent=null;
  function readMfu(){return {trainingFlopsPerToken:Number($('mfuTrainingTF').value)*1e12,tokensPerSecond:Number($('mfuTokensRate').value),tokensPerStep:Number($('mfuTokensStep').value),stepSeconds:Number($('mfuStepSeconds').value),deviceCount:Number($('mfuDevices').value),peakTF:Number($('mfuPeakTF').value),rateMode:$('mfuRateMode').value};}
  function mfuLesson(){
    const mode=$('mfuRateMode').value;$('mfuRateField').hidden=mode==='step';$('mfuStepFields').hidden=mode!=='step';
    const data=readMfu();
    try{
      const r=C.calculateMfu(data);mfuResult=r;$('mfuError').hidden=true;
      $('mfuResult').innerHTML=`<div class="mfu-score"><small>MFU · 教学假设结果</small><strong>${n(r.percent)}%</strong><div class="mfu-meter" role="img" aria-label="模型计算吞吐量为理论峰值的 ${n(r.percent)}%"><span style="width:${Math.min(100,r.percent)}%"></span></div><p>${tf(r.modelFlopsPerSecond)} 模型吞吐量 ÷ ${tf(r.hardwarePeak)} 全设备峰值</p></div><p class="lesson-result">每秒假设完成 ${n(r.tokensPerSecond)} 个训练 token，相当于达到理论计算峰值的 ${n(r.percent)}%。${previousPercent!=null&&previousPercent!==r.percent?`上次有效结果 ${n(previousPercent)}% → 本次 ${n(r.percent)}%。`:''}这不是 GPU / NPU 忙碌百分比，也不是训练实测。</p>${r.exceedsPeak?'<p class="notice">比例超过 100%：请检查输入单位、全设备吞吐量、计算精度与 Dense / Sparse 峰值口径。这组假设不一致；原始结果没有被截断。</p>':''}`;
      $('mfuCalculation').innerHTML=(mode==='step'?step('① 先把每步速度换成每秒速度',`Tokens/s = Tokens/Step ÷ Step Time = ${n(data.tokensPerStep)} ÷ ${n(data.stepSeconds)} = ${n(r.tokensPerSecond)}`,'这里是全设备的总 token 数与同一完整训练步的时间。'):step('① 准备模型工作量与吞吐量',`Training FLOPs/token = ${n(Number($('mfuTrainingTF').value))} × 10¹² = ${sci(data.trainingFlopsPerToken)}<br>Measured Tokens/s = ${n(r.tokensPerSecond)}（本页为假设值）`,'在真实分析中，前者来自全模型前向 + 反向推导，后者必须训练测量。'))+step('② 算模型每秒完成的计算量',`Model FLOP/s = Training FLOPs/token × Tokens/s<br>= ${sci(data.trainingFlopsPerToken)} × ${n(r.tokensPerSecond)} = ${sci(r.modelFlopsPerSecond)} FLOP/s`,'把每 token 的工作量乘以每秒 token 数。')+step('③ 算全部设备的理论峰值',`Hardware Peak = Device Count × Peak/device<br>= ${n(data.deviceCount)} × ${n(data.peakTF)} × 10¹² = ${sci(r.hardwarePeak)} FLOP/s`,'这里假设设备相同，峰值对应训练精度与 Dense 口径。')+step('④ 两种速度相除',`MFU = Model Training FLOPs/s / Total Hardware Peak FLOP/s<br>= ${sci(r.modelFlopsPerSecond)} / ${sci(r.hardwarePeak)} = ${n(r.mfu)} = ${n(r.percent)}%`,'工作量定义与峰值口径一致时，这个比例可以帮助比较训练系统效率。');
      previousPercent=r.percent;
    }catch(e){mfuResult=null;$('mfuError').hidden=false;$('mfuError').textContent=e.message;$('mfuResult').textContent='输入不完整或无效，暂不计算 MFU。';$('mfuCalculation').innerHTML='';}
    document.dispatchEvent(new Event('compute-lab-mfu-change'));
  }
  function loadMfuScenario(kind){
    const before={...C.mfuDefaults},after={...before};
    if(kind==='slow')after.tokensPerSecond/=2;
    if(kind==='less'){after.trainingFlopsPerToken/=2;after.tokensPerSecond*=2;}
    const a=C.calculateMfu(before),b=C.calculateMfu(after);
    for(const [id,value] of Object.entries({mfuTrainingTF:after.trainingFlopsPerToken/1e12,mfuTokensRate:after.tokensPerSecond,mfuTokensStep:after.tokensPerStep,mfuStepSeconds:after.stepSeconds,mfuDevices:after.deviceCount,mfuPeakTF:after.peakTF,mfuRateMode:after.rateMode}))$(id).value=value;
    previousPercent=a.percent;mfuLesson();
    const formula=(p,r)=>`(${sci(p.trainingFlopsPerToken)} FLOPs/token × ${n(r.tokensPerSecond)} token/s) ÷ (${p.deviceCount} × ${p.peakTF} × 10¹² FLOP/s) = <b>${n(r.percent)}%</b>`;
    $('mfuScenarioComparison').innerHTML=kind==='reset'?'<p class="lesson-result">已恢复初始教学假设：每 token 1 × 10¹² FLOPs、2000 Tokens/s、4 张设备、每设备 1000 TFLOP/s，MFU = 50%。</p>':`<div class="scenario-pair"><div><small>初始教学假设</small><p>${formula(before,a)}</p></div><div><small>情景 ${kind==='slow'?'A':'B'} · 修改后</small><p>${formula(after,b)}</p></div></div><p class="lesson-result">${kind==='slow'?'单位时间完成的模型计算量减少了，因此 MFU 从 50% 降到 25%。可能涉及访存、通信、数据等待或算子效率，但仅凭 MFU 无法确定具体原因。':'模型训练速度翻倍，但每个 token 所需计算减半，模型每秒计算量不变，MFU 仍为 50%。训练更快不一定让 MFU 提高。'}这些都是假设数据，不代表实测训练性能。</p>`;
  }
  const manualMfu=()=>{$('mfuScenarioComparison').textContent='当前为手动输入的教学假设；可点击 A / B 重新载入对照情景。';mfuLesson();};
  $('mfuForm').addEventListener('submit',e=>e.preventDefault());
  $('mfuForm').addEventListener('input',manualMfu);$('mfuForm').addEventListener('change',manualMfu);
  $('mfuReset').addEventListener('click',()=>loadMfuScenario('reset'));
  $('mfuScenarioReset').addEventListener('click',()=>loadMfuScenario('reset'));
  $('mfuSlow').addEventListener('click',()=>loadMfuScenario('slow'));
  $('mfuLessWork').addEventListener('click',()=>loadMfuScenario('less'));
  A.getMfuTeachingExample=()=>mfuResult?{...mfuResult}:null;
  A.getMfuTeachingInputs=readMfu;
  roofLesson();attentionLesson();mfuLesson();
})();
