/* Pure arithmetic model. Browser + Node; no UI, no external dependencies. */
(function (root) {
  'use strict';
  const defaults = {D:4096,H:32,Hkv:8,dk:128,dv:128,S:8192,W:1024,topk:8,block:128,di:32,rank:512,qrank:1024,rope:64,batch:1,precision:'BF16',statePrecision:'FP32',scalar:true,mlaImplementation:'factorized'};
  const scenarioDefaults={mode:'autoregressive',generated:1024};
  const trafficDefaults={mode:'model',weightHitRate:0,kvHitRate:0,headReuse:'shared',intermediate:'fused',customBytes:{},runtimesMs:{}};
  const architectures = ['MHA','GQA','MLA','SWA','MSA','GDA'];
  const colors = {MHA:'#2563eb',GQA:'#10a090',MLA:'#8659d4',SWA:'#e58b23',MSA:'#dc537f',GDA:'#526578'};
  const sources = {
    MHA:['Attention Is All You Need','https://arxiv.org/abs/1706.03762'],
    GQA:['GQA · §2','https://arxiv.org/abs/2305.13245'],
    SWA:['Mistral 7B · §2','https://arxiv.org/abs/2310.06825'],
    MLA:['DeepSeek-V2 · §2.1','https://arxiv.org/html/2405.04434v5#S2.SS1'],
    MSA:['MiniMax Sparse Attention · §3.1, §4.1','https://arxiv.org/html/2606.13392v1#S3.SS1'],
    GDA:['Gated Delta Networks · Eq.10, §3.4','https://arxiv.org/html/2412.06464v3#S3.SS1']
  };
  const bytes = {FP32:4,BF16:2,FP16:2,FP8:1};
  function validate(p) {
    const errors=[];
    for(const k of Object.keys(defaults).filter(k=>typeof defaults[k]==='number')) {
      const min=k==='S'?0:1, max=k==='S'?1048576:k==='batch'?1024:65536;
      if(!Number.isSafeInteger(p[k])||p[k]<min||p[k]>max) errors.push(`${k} 必须是 ${min}–${max} 范围内的整数`);
    }
    if(p.Hkv>p.H || p.H%p.Hkv!==0) errors.push('GQA / SWA / MSA 要求 H ≥ Hkv 且 H 能被 Hkv 整除');
    if(p.rope%2!==0) errors.push('MLA 的 RoPE 维度必须为偶数');
    if(p.rank>p.D || p.qrank>p.D) errors.push('本实验要求 MLA 的 KV / Query 压缩秩不大于 D');
    if(!bytes[p.precision] || !bytes[p.statePrecision]) errors.push('请选择支持的数据精度');
    if(typeof p.scalar!=='boolean') errors.push('scalar 必须是布尔值');
    if(p.mlaImplementation!=null&&!['factorized','absorbed-query','absorbed-qo'].includes(p.mlaImplementation)) errors.push('MLA 执行路径不受支持');
    // Conservative bound keeps displayed integer operation counts exact in JS Number.
    const bound=8*(p.D*p.H*(p.dk+p.dv+p.rank+p.rope)+p.D*(p.rank+p.qrank+p.Hkv*p.di+p.di)+p.qrank*p.H*(p.dk+p.rope)+p.H*p.dk*p.rank+p.H*p.rank*p.dv+1048577*(p.H*(p.dk+p.dv+2*p.rank+p.rope+8)+p.Hkv*(p.di+1))+p.H*p.dk*p.dv);
    if(Number.isFinite(bound)&&bound*p.batch>Number.MAX_SAFE_INTEGER) errors.push('此配置超过 JavaScript 安全整数计数范围，请减小维度或 Batch');
    return errors;
  }
  function matmul(m,k,n,groups=1) {return {approx:2*m*k*n*groups,exact:m*n*(2*k-1)*groups};}
  function calculate(name,p) {
    const errors=validate(p); if(errors.length) throw new Error(errors.join('；'));
    if(!architectures.includes(name)) throw new Error('未知架构');
    const {D,H,Hkv,dk,dv,S,W,topk,block,di,rank:r,qrank:rq,rope:dr}=p;
    const N=S+1, b=bytes[p.precision], rows=[];
    function mm(label,category,m,k,n,groups,purpose,formula) {
      const f=matmul(m,k,n,groups);
      rows.push({label,category,purpose,formula,input:`[${m}, ${k}]`,weight:`[${k}, ${n}]`,output:`[${m}, ${n}]`,groups,...f,kind:'matmul'});
    }
    function op(label,category,f,formula,shape,purpose) {
      rows.push({label,category,purpose,formula,input:shape,weight:'逐元素 / 标量',output:shape,groups:1,approx:p.scalar?f:0,exact:p.scalar?f:0,excluded:!p.scalar,kind:'scalar'});
    }
    function projections(h) {
      mm('Q Projection','projection',1,D,H*dk,1,'把当前隐藏向量变成每个 Query Head 的查询','Q = X WQ');
      mm('K Projection','projection',1,D,h*dk,1,'只为当前 token 生成 Key，历史 Key 已缓存','Knew = X WK');
      mm('V Projection','projection',1,D,h*dv,1,'只为当前 token 生成 Value','Vnew = X WV');
      mm('O Projection','projection',1,H*dv,D,1,'把拼接的各头输出映射回隐藏空间','Y = concat(Oheads) WO');
    }
    function softmax(n) {
      op('Scale','core',H*n,'H × L；每个 score 乘预计算 scale',`[H, ${n}]`,'把分数缩放到稳定范围');
      op('Softmax','core',H*(4*n-1),'H × (4L−1)：减最大值 L + exp L + 求和 L−1 + 除法 L',`[H, ${n}]`,'把每个头的分数变成权重；max 比较单独排除');
    }
    let active=N, memory, memoryNote, complete=true, unknown=[];
    if(['MHA','GQA','SWA','MSA'].includes(name)) {
      const h=name==='MHA'?H:Hkv;
      if(name==='SWA') active=Math.min(N,W);
      if(name==='MSA') {
        // k INCLUDES the forced current/local block; all preceding blocks are full.
        const blocks=Math.ceil(N/block), local=(N-1)%block+1;
        active=local+(Math.min(topk,blocks)-1)*block;
      }
      projections(h);
      mm('QKᵀ','core',1,dk,active,H,'每个 Query Head 对映射的 KV Head 计算相似度','score[h] = Q[h] K[g(h)]ᵀ；g(h)=floor(h/(H/Hkv))，MHA 为 g(h)=h');
      softmax(active);
      mm('Attention × V','core',1,active,dv,H,'用权重对读取的 Value 加权求和','O[h] = P[h] V[g(h)]');
      memory=active*h*(dk+dv)*b;
      memoryNote=name==='SWA'?'仅保留最近 W 个位置的滚动 KV Cache':'完整历史 KV Cache，含当前 token';
      if(name==='MSA') {
        mm('Index Q Projection','projection',1,D,Hkv*di,1,'每个 GQA group 生成一个小维度索引查询','Qidx = X WQidx');
        mm('Index K Projection','projection',1,D,di,1,'所有 group 共享一个索引 Key；仅新增当前行','Kidx,new = X WKidx');
        mm('Index scores','extra',1,di,N,Hkv,'索引查询扫描全部 N 个 index key，再进行 block max pooling','scoreidx[g] = Qidx[g] Kidxᵀ');
        op('Index scale','extra',Hkv*N,'Hkv × N；显式缩放约定',`[${Hkv}, ${N}]`,'exp-free Top-K 不做索引 Softmax；正 scale 也可由实际实现省略');
        memory=N*(Hkv*(dk+dv)+di)*b;
        memoryNote='全部 GQA KV + 一个共享 Index Key Cache；稀疏读取不等于稀疏存储';
      }
    } else if(name==='MLA') {
      mm('Query down','projection',1,D,rq,1,'先把当前 Query 压缩到低秩空间','cQ = X WDQ');
      const absorbed=p.mlaImplementation&&p.mlaImplementation!=='factorized';
      if(!absorbed) mm('Query content up','projection',1,rq,H*dk,1,'恢复每个头的内容查询','qC = cQ WUQ');
      mm('Query RoPE branch','projection',1,rq,H*dr,1,'单独生成位置查询（旋转操作不在本计数范围）','qR = cQ WQR');
      mm('KV Compression','projection',1,D,r,1,'一份 latent 同时承载历史 Key / Value 内容','cKV = X WDKV');
      mm('Key RoPE branch','projection',1,D,dr,1,'生成所有头共享的位置 Key','kR = X WKR');
      if(absorbed) mm('Absorbed Query → latent','projection',1,rq,H*r,1,'离线合并 WUQ[h] 与 WUK[h]ᵀ；Decode 只读融合权重，不重复做 Query content up','qLat = cQ WQabsorbed；WQabsorbed[h] = WUQ[h] WUK[h]ᵀ');
      else mm('Query → latent','projection',1,dk,r,H,'按头把内容 Query 变到 latent 空间，历史 Key 无需展开','qLat[h] = qC[h] WUK[h]ᵀ');
      mm('Latent scores','core',1,r,N,H,'在缓存的 latent 上计算内容相似度','scoreC[h] = qLat[h] Ccacheᵀ');
      mm('Position scores','core',1,dr,N,H,'加入解耦的位置相似度','scoreR[h] = qR[h] KRcacheᵀ');
      op('Combine scores','core',H*N,'H × N；scoreC + scoreR',`[H, ${N}]`,'内容与位置两部分相加');
      softmax(N);
      mm('Attention × latent','core',1,N,r,H,'直接对 latent 加权，不展开历史 Value','z[h] = P[h] Ccache');
      if(p.mlaImplementation==='absorbed-qo') mm('Absorbed O Projection','projection',1,H*r,D,1,'离线把每头 Value up 合并入 O，Decode 不再单独执行 Value up','Y = concat(z) WOabsorbed；WOabsorbed[h] = WUV[h] WO[h]');
      else {
        mm('Value up','projection',1,r,dv,H,'仅把当前聚合结果恢复成 Value Head 维度','O[h] = z[h] WUV[h]');
        mm('O Projection','projection',1,H*dv,D,1,'把各头当前结果映射回隐藏空间','Y = concat(O) WO');
      }
      memory=N*(r+dr)*b;
      memoryNote='latent cKV + 共享解耦 RoPE Key；不缓存展开后的 K/V';
    } else {
      projections(H);
      mm('α / β Projection','projection',1,D,2*H,1,'为每个递归头生成两种门的原始参数；激活尚未核验','[a,b] = X Wgate');
      mm('Output gate Projection','projection',1,D,H*dv,1,'生成论文输出门的原始向量；激活/归一化尚未计入','g = X Woutgate');
      op('State decay','core',H*dk*dv,'H dk dv；A = α Rprev',`[${H}, ${dv}, ${dk}]`,'先衰减旧记忆；R 为论文状态矩阵 S，避免与历史长度 S 混淆');
      mm('Read prediction','core',1,dk,dv,H,'用当前 Key 读取旧记忆的预测；A 的布局在此取转置','u = A k');
      op('Delta residual / gate','core',2*H*dv,'2 H dv；δ = β (v − u)',`[${H}, ${dv}]`,'只写入预测误差，而非重复写入全部 Value');
      // Outer product is scalar multiplication, not a length-k dot product.
      op('Outer product + update','core',2*H*dk*dv,'2 H dk dv；Rnew = A + δ kᵀ',`[${H}, ${dv}, ${dk}]`,'外积逐元素一次乘法加一次加法');
      mm('Read output','core',1,dk,dv,H,'从更新后的状态取出当前 Query 的输出','o = Rnew q');
      memory=H*dk*dv*bytes[p.statePrecision];
      memoryNote=`仅递归状态 R，使用 ${p.statePrecision}；ShortConv 缓冲未核验`;
      complete=false;
      unknown=['ShortConv 的核长度与缓冲布局','Q/K L2 Norm、SiLU、α/β 参数化','输出 Norm、门激活与门乘法'];
    }
    const sum=(category,field='approx')=>rows.filter(x=>x.category===category).reduce((a,x)=>a+x[field],0);
    const projection=sum('projection'), core=sum('core'), extra=sum('extra'), known=projection+core+extra;
    return {name,projection,core,extra:complete?extra:null,known,knownExtra:extra,total:complete?known:null,exactKnown:rows.reduce((a,x)=>a+x.exact,0),complete,unknown,rows,active,visible:N,memoryBytes:memory*p.batch,memoryComplete:complete,memoryNote,batchFlops:complete?known*p.batch:null,scope:'单层、每序列、当前 1 token；排除位置旋转、全局 Norm、Bias、残差和 FFN',trafficBytes:null,weightTrafficBytes:null};
  }
  function results(p) {return architectures.map(n=>calculate(n,p));}
  function validateScenario(p,s) {
    const errors=validate(p),g=s.mode==='single'?1:s.generated;
    if(!['single','autoregressive','prefill'].includes(s.mode)) errors.push('未知 Token Mode');
    if(!Number.isSafeInteger(g)||g<1||g>65536) errors.push('Generated Tokens 必须是 1–65,536 的整数');
    if(p.S+g-1>1048576) errors.push('最后一步历史 S+G−1 不得超过 1,048,576');
    if(s.mode==='prefill') errors.push('Prefill 尚未验证，不输出 Decode 结果');
    return errors;
  }
  const triangular=n=>n*(n+1)/2;
  function activePrefix(name,t,p) {
    if(name==='SWA') {const a=Math.min(t,p.W);return triangular(a)+(t-a)*p.W;}
    if(name==='MSA') {
      const cutoff=p.topk*p.block;
      if(t<=cutoff)return triangular(t);
      const periodic=n=>Math.floor(n/p.block)*triangular(p.block)+triangular(n%p.block);
      return triangular(cutoff)+(t-cutoff)*(p.topk-1)*p.block+periodic(t)-periodic(cutoff);
    }
    return triangular(t);
  }
  function aggregate(name,p,s=scenarioDefaults) {
    const errors=validateScenario(p,s);if(errors.length)throw new Error(errors.join('；'));
    const g=s.mode==='single'?1:s.generated,first=calculate(name,p),last=calculate(name,{...p,S:p.S+g-1});
    const sumVisible=g*(2*(p.S+1)+g-1)/2,sumActive=activePrefix(name,p.S+g,p)-activePrefix(name,p.S,p),sc=p.scalar?1:0;
    let core,exactCore;
    if(name==='GDA'){core=g*first.core;exactCore=g*first.rows.filter(x=>x.category==='core').reduce((v,x)=>v+x.exact,0);}
    else if(name==='MLA'){
      core=2*p.H*(2*p.rank+p.rope)*sumVisible+sc*p.H*(6*sumVisible-g);
      exactCore=p.H*((4*p.rank+2*p.rope-2)*sumVisible-p.rank*g)+sc*p.H*(6*sumVisible-g);
    }else{
      core=2*p.H*(p.dk+p.dv)*sumActive+sc*p.H*(5*sumActive-g);
      exactCore=p.H*((2*p.dk-1+2*p.dv)*sumActive-p.dv*g)+sc*p.H*(5*sumActive-g);
    }
    const projection=g*first.projection,knownExtra=name==='MSA'?p.Hkv*(2*p.di+sc)*sumVisible:0;
    const known=projection+core+knownExtra,total=first.complete?known:null;
    const exactKnown=g*first.rows.filter(x=>x.category==='projection').reduce((v,x)=>v+x.exact,0)+exactCore+(name==='MSA'?p.Hkv*(2*p.di-1+sc)*sumVisible:0);
    if(!Number.isSafeInteger(known*p.batch)||!Number.isSafeInteger(exactKnown*p.batch))throw new Error('累计 FLOPs 超过安全整数范围，请减小维度、Token 数或 Batch');
    return {...first,projection,core,knownExtra,extra:first.complete?knownExtra:null,total,known,exactKnown,averageFlops:total==null?null:total/g,averageCore:core/g,firstFlops:first.total,lastFlops:last.total,first,last,steps:g,sumVisible,sumActive,memoryBytes:last.memoryBytes,memoryNote:last.memoryNote,batchFlops:total==null?null:total*p.batch,scope:'单层、每序列、当前生成区间累计；不是完整模型或 Prefill'};
  }
  function sequenceResults(p,s=scenarioDefaults){return architectures.map(n=>aggregate(n,p,s));}
  function speedup(base,flops){return flops==null||base==null||flops<=0?{ratio:null,reduction:null}:{ratio:base/flops,reduction:(1-flops/base)*100};}
  function exportSequence(p,s=scenarioDefaults){return {...exportAttention(p),schemaVersion:2,scenario:s.mode,currentTokensPerStep:1,generatedTokens:s.mode==='single'?1:s.generated,lastHistoryTokens:p.S+(s.mode==='single'?1:s.generated)-1,flopsUnit:'FLOPs/generated-interval/layer/sequence',tokenResults:results(p),results:sequenceResults(p,s)};}
  function validateHardware(h,p) {
    const errors=[];
    if(!h.name?.trim())errors.push('请填写 Hardware Name');
    if(!Number.isFinite(h.peakTF)||h.peakTF<=0)errors.push('Peak Compute 必须为正数，单位 TFLOP/s');
    if(!Number.isFinite(h.bandwidth)||h.bandwidth<=0)errors.push('Memory Bandwidth 必须为正数');
    if(!['GB/s','TB/s'].includes(h.bandwidthUnit))errors.push('带宽单位必须为 GB/s 或 TB/s');
    if(h.precision!==p.precision)errors.push('硬件峰值精度必须与当前 Attention 精度相同');
    if(!Number.isSafeInteger(h.devices)||h.devices<1||h.devices>p.batch||p.batch%h.devices!==0)errors.push('本页多设备采用独立序列均分：Device Count 必须整除 Batch，且不得超过 Batch');
    const peak=h.peakTF*1e12*h.devices,bw=h.bandwidth*(h.bandwidthUnit==='TB/s'?1e12:1e9)*h.devices;
    if(![peak,bw,peak/bw].every(v=>Number.isFinite(v)&&v>=1e-280&&v<=1e280))errors.push('硬件参数换算后超出可计算 / 绘图范围，请使用有限的合理数值');
    return errors;
  }
  function roofline(flops,moved,h,runtimeMs=null){
    const peak=h.peakTF*1e12*h.devices,bw=h.bandwidth*(h.bandwidthUnit==='TB/s'?1e12:1e9)*h.devices;
    if(!(Number.isFinite(peak)&&peak>0&&Number.isFinite(bw)&&bw>0))throw new Error('硬件算力或带宽无效');
    const ridge=peak/bw;
    if(!(Number.isFinite(flops)&&flops>0&&Number.isFinite(moved)&&moved>0))return {ai:null,ridge,performance:null,computeSeconds:null,memorySeconds:null,lowerSeconds:null,bottleneck:'N/A',measuredPerformance:null};
    const ai=flops/moved,computeSeconds=flops/peak,memorySeconds=moved/bw,lowerSeconds=Math.max(computeSeconds,memorySeconds),performance=Math.min(peak,ai*bw);
    if(runtimeMs!=null&&(!Number.isFinite(runtimeMs)||runtimeMs<=0))throw new Error('实测 Runtime 必须是有限正数，单位 ms');
    const measuredPerformance=runtimeMs==null?null:flops/(runtimeMs/1000);
    if(![ai,ridge,performance,computeSeconds,memorySeconds,lowerSeconds,...(measuredPerformance==null?[]:[measuredPerformance])].every(v=>Number.isFinite(v)&&v>=1e-280&&v<=1e280))throw new Error('计算结果超出可表示范围，请检查数值和单位');
    return {ai,ridge,performance,computeSeconds,memorySeconds,lowerSeconds,bottleneck:ai<ridge?'Memory-Bound':'Compute-Bound',measuredPerformance};
  }
  function rooflineTeachingPoints(h){
    // A one-byte hypothetical workload reuses exactly the same roofline formula.
    return [100,500].map(ai=>({id:String(ai),kind:'hypothetical-ai',...roofline(ai,1,h)}));
  }
  function estimateTraffic(name,p,s,scope='total',o=trafficDefaults,devices=1){
    if(!['core','total'].includes(scope))throw new Error('Roofline FLOPs 口径不受支持');
    const r=aggregate(name,p,s),g=r.steps,B=p.batch,b=bytes[p.precision],v=r.sumVisible-g,a=r.sumActive-g;
    const opts={...trafficDefaults,...o};
    if(!['model','custom','unknown'].includes(opts.mode))throw new Error('访存模式无效');
    for(const k of ['weightHitRate','kvHitRate'])if(!Number.isFinite(opts[k])||opts[k]<0||opts[k]>100)throw new Error(`${k} 必须为 0–100%`);
    if(!['shared','none'].includes(opts.headReuse)||!['fused','materialized'].includes(opts.intermediate))throw new Error('访存复用或融合选项无效');
    if(opts.mode!=='model'){
      const supplied=opts.customBytes?.[name],valid=opts.mode==='custom'&&Number.isFinite(supplied)&&supplied>0;
      return {totalBytes:valid?supplied:null,weightBytes:null,kvReadBytes:null,cacheWriteBytes:null,intermediateBytes:null,stateBytes:null,kind:valid?'user-estimate':'unknown',assumptions:['用户输入为当前整批、单层、全部生成步的 HBM 总字节；与缓存容量独立'],complete:valid};
    }
    const factor=opts.headReuse==='shared'?(name==='MHA'?p.H:p.Hkv):p.H,hit=1-opts.kvHitRate/100;
    const weightElements=r.first.rows.filter(x=>x.category==='projection').reduce((v,x)=>v+x.approx/2,0);
    let weightBytes=scope==='core'?0:weightElements*b*g*devices*(1-opts.weightHitRate/100),kvReadBytes=0,cacheWriteBytes=0,stateBytes=0;
    if(['MHA','GQA','SWA'].includes(name)){
      kvReadBytes=a*factor*(p.dk+p.dv)*b*B*hit;
      cacheWriteBytes=g*(name==='MHA'?p.H:p.Hkv)*(p.dk+p.dv)*b*B;
    }else if(name==='MSA'){
      kvReadBytes=(a*factor*(p.dk+p.dv)+v*p.di*(opts.headReuse==='shared'?1:p.Hkv))*b*B*hit;
      cacheWriteBytes=g*(p.Hkv*(p.dk+p.dv)+p.di)*b*B;
      // Core-only refers to MAIN attention, so exclude index-cache read/write.
      if(scope==='core'){kvReadBytes=a*factor*(p.dk+p.dv)*b*B*hit;cacheWriteBytes=g*p.Hkv*(p.dk+p.dv)*b*B;}
    }else if(name==='MLA'){
      const repeat=opts.headReuse==='shared'?1:p.H;
      kvReadBytes=v*(2*p.rank+p.rope)*repeat*b*B*hit;
      cacheWriteBytes=g*(p.rank+p.rope)*b*B;
    }else stateBytes=2*p.H*p.dk*p.dv*bytes[p.statePrecision]*B*g;
    const io=scope==='core'?0:2*p.D*b*B*g;
    const spills=opts.intermediate==='materialized'&&name!=='GDA'?(4*p.H*r.sumActive+(name==='MSA'&&scope==='total'?2*p.Hkv*r.sumVisible:0))*b*B:0;
    const intermediateBytes=io+spills,knownBytes=weightBytes+kvReadBytes+cacheWriteBytes+stateBytes+intermediateBytes;
    const complete=scope==='core'||r.complete;
    const readExpression=name==='GDA'?'不读取 KV Cache':name==='MLA'?`${v} 历史行 × (2×${p.rank}+${p.rope}) × ${opts.headReuse==='shared'?1:p.H} 复用系数 × ${b} B × ${B} Batch × ${hit}`:name==='MSA'&&scope==='total'?`[${a} 已选历史位置 × ${factor} 头 × (${p.dk}+${p.dv}) + ${v} 全历史位置 × ${p.di} 索引维度 × ${opts.headReuse==='shared'?1:p.Hkv}] × ${b} B × ${B} Batch × ${hit}`:`${a} 历史读取位置 × ${factor} 头 × (${p.dk}+${p.dv}) × ${b} B × ${B} Batch × ${hit}`;
    const equations=[`Σ历史可见行 = Σvisible − G = ${r.sumVisible} − ${g} = ${v}；Σ已选历史行 = ${r.sumActive} − ${g} = ${a}`,`Weight Traffic = ${scope==='core'?'0（Core 不执行投影）':`${weightElements} 权重元素 × ${b} B × ${g} 步 × ${devices} 设备 × ${1-opts.weightHitRate/100}`} = ${weightBytes} B`,`Historical KV Read = ${readExpression} = ${kvReadBytes} B`,`Cache Write = 每步新增缓存元素 × ${b} B × ${B} Batch × ${g} 步 = ${cacheWriteBytes} B`,`Recurrent State Read+Write = ${name==='GDA'?`2 × ${p.H} × ${p.dk} × ${p.dv} × ${bytes[p.statePrecision]} B × ${B} Batch × ${g} 步`:'0（本架构无递归状态）'} = ${stateBytes} B`,`Intermediate / I/O = ${io} B 隐藏输入输出 + ${spills} B score/probability 回写 = ${intermediateBytes} B`];
    return {totalBytes:complete?knownBytes:null,knownBytes,weightBytes,kvReadBytes,cacheWriteBytes,stateBytes,intermediateBytes,weightElements,equations,kind:'modeled-estimate',complete,assumptions:[`HBM；权重每步按 ${100-opts.weightHitRate}% 从 HBM 读取（每设备一份；不乘每设备 Batch）`,`历史 KV 缓存 HBM 未命中比例 ${100-opts.kvHitRate}%；${opts.headReuse==='shared'?'组内 / latent 各头理想复用':'按查询头重读'}；GDA 状态独立按每步完整读写建模`, `当前 K/V 或 latent 留在片上使用，但每步写入 HBM；${opts.intermediate==='fused'?'中间 score / probability 在片上融合':'score / probability 各写回并读取一次'}`,scope==='core'?'Core 边界：不读投影权重，Q/新 K/V 与输出经片上接口传递；MSA 不包含索引分支':'Module 边界：隐藏输入读取一次、输出写一次',name==='MLA'?'latent 内容按 score 与 value 两遍读取；RoPE key 一遍，跨遍未命中由统一命中率控制':'缓存历史读取随每步 Context / Window / Block 变化','模型忽略分配对齐、量化 scales、通信；MSA 选择索引假定片上完成；GDA 全模块缺少卷积和门开销']};
  }
  function analyzeRoofline(p,s,h,o=trafficDefaults,scope='total'){
    const errors=validateHardware(h,p);if(errors.length)throw new Error(errors.join('；'));
    const data=sequenceResults(p,s).map(r=>{const flops=(scope==='core'?r.core:r.total);const traffic=estimateTraffic(r.name,p,s,scope,o,h.devices);return {name:r.name,flops:flops==null?null:flops*p.batch,traffic,...roofline(flops==null?null:flops*p.batch,traffic.totalBytes,h,o.runtimesMs?.[r.name])};});
    const base=data[0];return {schemaVersion:2,storageLevel:'HBM',hardware:{...h},scope,tokenMode:s.mode,generatedTokens:s.mode==='single'?1:s.generated,initialContext:p.S,parameters:{...p},assumptions:{...o},results:data.map(r=>({...r,flopsSpeedup:speedup(base.flops,r.flops).ratio,rooflineSpeedup:r.lowerSeconds&&base.lowerSeconds?base.lowerSeconds/r.lowerSeconds:null}))};
  }
  function exportAttention(p) {return {schemaVersion:1,scenario:'decode',currentTokens:1,historyTokens:p.S,visibleTokens:p.S+1,parameters:{...p},flopsUnit:'FLOPs/token/layer/sequence',memoryUnit:'bytes/batch/layer',counting:p.scalar?'2MKN + 明示标量约定':'仅矩阵 2MKN（不含外积/逐元素）',results:results(p)};}
  function rooflineInput(p) {return {schemaVersion:1,storageLevel:'HBM',hardware:{vendor:null,peakFlopsPerSecond:null,bandwidthBytesPerSecond:null,precision:p.precision},attention:exportAttention(p),bytesMoved:null,bytesEvidence:null,status:'awaiting-traffic-model'};}
  function validateTraining(data) {
    const required=['architecture','source','precision','forwardFlopsPerToken','backwardFlopsPerToken'];
    if(!data || typeof data!=='object'||Array.isArray(data)) throw new Error('训练数据必须是 JSON 对象');
    if(data.schemaVersion!=null&&data.schemaVersion!==1) throw new Error('仅支持 schemaVersion: 1');
    for(const k of required) if(data[k]==null) throw new Error(`训练数据缺少 ${k}`);
    for(const k of ['architecture','source','precision']) if(typeof data[k]!=='string'||!data[k].trim()) throw new Error(`${k} 必须为非空文本`);
    for(const k of ['forwardFlopsPerToken','backwardFlopsPerToken']) if(!Number.isFinite(data[k])||data[k]<=0) throw new Error(`${k} 必须为正数，单位 FLOPs/token`);
    if(!['FP32','BF16','FP16','FP8'].includes(data.precision)) throw new Error('训练精度不受支持');
    if(data.verified!==true) throw new Error('仅接收 verified: true 的已核验数据；该标签由提供者负责');
    for(const k of ['tokensPerSecond','tokensPerStep','stepSeconds','deviceCount','peakFlopsPerSecondPerDevice']) {
      if(data[k]!=null&&(!Number.isFinite(data[k])||data[k]<=0)) throw new Error(`${k} 必须为正数`);
    }
    if(data.deviceCount!=null&&!Number.isInteger(data.deviceCount)) throw new Error('deviceCount 必须为整数');
    return {...data,schemaVersion:1,kind:'model-training',status:'imported-not-estimated'};
  }
  // Official product bandwidth + NVIDIA's explicit BF16 dense convention.
  // The product page's 1,979 TFLOP/s is WITH sparsity, not our dense ceiling.
  const hardwarePresets=Object.freeze({h100:Object.freeze({
    name:'NVIDIA H100 SXM 80GB',peakTF:989,bandwidth:3.35,bandwidthUnit:'TB/s',precision:'BF16',devices:1,
    computeType:'Dense Tensor Core',memoryGB:80,verifiedOn:'2026-10-13',
    bandwidthSource:'https://www.nvidia.com/en-us/data-center/h100/',
    computeSource:'https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/benchmark/E2E_BENCHMARK.md',
    sparsePeakTF:1979
  })});
  function hardwarePreset(id){if(!hardwarePresets[id])throw new Error('未知硬件模板');return {...hardwarePresets[id]};}
  const mfuDefaults=Object.freeze({trainingFlopsPerToken:1e12,tokensPerSecond:2000,tokensPerStep:8000,stepSeconds:4,deviceCount:4,peakTF:1000,rateMode:'rate'});
  function calculateMfu(input){
    const p={...input};
    if(!['rate','step'].includes(p.rateMode))throw new Error('吞吐量模式应为 rate 或 step');
    for(const k of ['trainingFlopsPerToken','deviceCount','peakTF',...(p.rateMode==='step'?['tokensPerStep','stepSeconds']:['tokensPerSecond'])])
      if(!Number.isFinite(p[k])||p[k]<=0)throw new Error(`${k} 必须为有限正数`);
    if(!Number.isSafeInteger(p.deviceCount))throw new Error('设备数量必须为正整数');
    const tokensPerSecond=p.rateMode==='step'?p.tokensPerStep/p.stepSeconds:p.tokensPerSecond;
    const modelFlopsPerSecond=p.trainingFlopsPerToken*tokensPerSecond,hardwarePeak=p.peakTF*1e12*p.deviceCount,mfu=modelFlopsPerSecond/hardwarePeak;
    if(![tokensPerSecond,modelFlopsPerSecond,hardwarePeak,mfu,100*mfu].every(v=>Number.isFinite(v)&&v>0))throw new Error('数值超出可计算范围');
    return {kind:'training-mfu-teaching',dataKind:'hypothetical',tokensPerSecond,modelFlopsPerSecond,hardwarePeak,mfu,percent:100*mfu,exceedsPeak:mfu>1};
  }
  const api={defaults,scenarioDefaults,trafficDefaults,architectures,colors,sources,bytes,validate,matmul,calculate,results,exportAttention,rooflineInput,validateTraining,validateScenario,activePrefix,aggregate,sequenceResults,speedup,exportSequence,validateHardware,roofline,estimateTraffic,analyzeRoofline,hardwarePresets,hardwarePreset,mfuDefaults,calculateMfu,rooflineTeachingPoints};
  if(typeof module==='object'&&module.exports) module.exports=api;
  root.ComputeLab=api;
})(typeof globalThis==='object'?globalThis:this);
