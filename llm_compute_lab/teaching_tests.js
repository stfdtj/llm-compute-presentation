'use strict';
const assert=require('node:assert/strict'),C=require('./model.js');
let count=0;
function test(name,run){run();count++;console.log('PASS '+name);}
const close=(a,b)=>assert.ok(Math.abs(a-b)<=Math.max(1,Math.abs(b))*1e-12,`${a} vs ${b}`);
test('H100 preset has explicitly dense BF16 compute and official provenance',()=>{
  const h=C.hardwarePreset('h100');assert.equal(h.name,'NVIDIA H100 SXM 80GB');assert.equal(h.precision,'BF16');assert.equal(h.computeType,'Dense Tensor Core');assert.equal(h.peakTF,989);assert.equal(h.sparsePeakTF,1979);assert.equal(h.bandwidth,3.35);assert.equal(h.bandwidthUnit,'TB/s');assert.equal(h.devices,1);
  assert.equal(new URL(h.bandwidthSource).hostname,'www.nvidia.com');assert.ok(h.computeSource.startsWith('https://github.com/NVIDIA/'));assert.deepEqual(C.validateHardware(h,C.defaults),[]);
  h.peakTF=1;assert.equal(C.hardwarePreset('h100').peakTF,989);assert.throws(()=>C.hardwarePreset('ascend-unknown'));
});
test('Default H100 GQA lesson agrees with independent 1024-step hand sum',()=>{
  const r=C.analyzeRoofline(C.defaults,C.scenarioDefaults,C.hardwarePreset('h100')).results.find(r=>r.name==='GQA');
  // Visible rows: 8193..9216. Work: fixed projection + dot/value + scalar ops.
  const rows=8913408;
  const flops=83886080*1024+16544*rows-32*1024;
  const bytes=83886080*1024+4096*(rows-1024)+4194304+16777216;
  assert.equal(r.flops,flops);assert.equal(r.traffic.totalBytes,bytes);assert.equal(r.traffic.kind,'modeled-estimate');assert.equal(r.measuredPerformance,null);
  close(r.ai,flops/bytes);close(r.ridge,989/3.35);close(r.performance,(flops/bytes)*3.35e12);close(r.lowerSeconds,bytes/3.35e12);assert.equal(r.bottleneck,'Memory-Bound');
});
test('Default plot has five usable module points and keeps full GDA unknown',()=>{
  const a=C.analyzeRoofline(C.defaults,C.scenarioDefaults,C.hardwarePreset('h100'));
  assert.equal(a.generatedTokens,1024);assert.equal(a.results.filter(r=>r.ai!=null).length,5);
  for(const r of a.results.filter(r=>r.ai!=null)){close(r.performance,Math.min(989e12,r.ai*3.35e12));close(r.lowerSeconds,Math.max(r.flops/989e12,r.traffic.totalBytes/3.35e12));}
  assert.equal(a.results[5].flops,null);assert.equal(a.results[5].lowerSeconds,null);
});
test('Bandwidth/compute demo changes the appropriate roof without changing AI',()=>{
  const h=C.hardwarePreset('h100'),f=1e12,bytes=1e12,initial=C.roofline(f,bytes,h),wide=C.roofline(f,bytes,{...h,bandwidth:h.bandwidth*2}),fast=C.roofline(f,bytes,{...h,peakTF:h.peakTF*2});
  assert.equal(wide.ai,initial.ai);close(wide.performance,initial.performance*2);close(wide.lowerSeconds,initial.lowerSeconds/2);assert.equal(fast.performance,initial.performance);close(fast.ridge,initial.ridge*2);
  const compute=C.roofline(1e15,1e9,h),wider=C.roofline(1e15,1e9,{...h,bandwidth:6.7});assert.equal(compute.bottleneck,'Compute-Bound');assert.equal(compute.performance,wider.performance);
});
test('TB/s vs GB/s and two independent device sequences preserve unit consistency',()=>{
  const h=C.hardwarePreset('h100'),gb={...h,bandwidth:3350,bandwidthUnit:'GB/s'};assert.deepEqual(C.roofline(1e12,1e11,h),C.roofline(1e12,1e11,gb));
  const a=C.analyzeRoofline({...C.defaults,batch:2},C.scenarioDefaults,{...h,devices:2}),b=C.analyzeRoofline(C.defaults,C.scenarioDefaults,h);close(a.results[1].lowerSeconds,b.results[1].lowerSeconds);
  assert.ok(C.validateHardware({...h,precision:'FP8'},C.defaults).length);assert.ok(C.validateHardware({...h,devices:2},C.defaults).length);
});
test('MFU hypothetical 1e12 × 2000 / (4 × 1000e12) is exactly 50 percent',()=>{
  const r=C.calculateMfu(C.mfuDefaults);assert.equal(r.modelFlopsPerSecond,2e15);assert.equal(r.hardwarePeak,4e15);assert.equal(r.mfu,.5);assert.equal(r.percent,50);assert.equal(r.dataKind,'hypothetical');assert.equal(r.exceedsPeak,false);
});
test('All-device tokens/step conversion avoids multiplying throughput by devices twice',()=>{
  const r=C.calculateMfu({...C.mfuDefaults,rateMode:'step'});assert.equal(r.tokensPerSecond,2000);assert.equal(r.percent,50);
  const slower=C.calculateMfu({...C.mfuDefaults,rateMode:'step',stepSeconds:8});assert.equal(slower.percent,25);
});
test('Less work can double token speed without increasing MFU',()=>{
  assert.equal(C.calculateMfu({...C.mfuDefaults,trainingFlopsPerToken:5e11,tokensPerSecond:4000}).percent,50);
  assert.equal(C.calculateMfu({...C.mfuDefaults,trainingFlopsPerToken:5e11,tokensPerSecond:3000}).percent,37.5);
});
test('MFU invalid values are rejected; impossible hypothetical rates are flagged, not clamped',()=>{
  for(const [k,v] of [['trainingFlopsPerToken',0],['tokensPerSecond',NaN],['deviceCount',1.5],['peakTF',-1],['rateMode','decode'],['trainingFlopsPerToken',1e308]])assert.throws(()=>C.calculateMfu({...C.mfuDefaults,[k]:v}));
  assert.throws(()=>C.calculateMfu({...C.mfuDefaults,rateMode:'step',stepSeconds:0}));
  const r=C.calculateMfu({...C.mfuDefaults,tokensPerSecond:5000});assert.equal(r.percent,125);assert.equal(r.exceedsPeak,true);
});
console.log(`\n${count} teaching calculation tests passed.`);
