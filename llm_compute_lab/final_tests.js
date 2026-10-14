'use strict';
const assert=require('node:assert/strict'),C=require('./model.js');let checks=0;
const test=(name,run)=>{run();checks++;console.log('PASS '+name);};
const close=(a,b)=>assert.ok(Math.abs(a-b)<=Math.max(1,Math.abs(b))*1e-12,`${a} != ${b}`);
const h=C.hardwarePreset('h100');
test('Independent hypothetical AI 100 and 500 straddle H100 ridge',()=>{
  const [a,b]=C.rooflineTeachingPoints(h);assert.equal(a.kind,'hypothetical-ai');assert.equal(a.ai,100);assert.equal(b.ai,500);assert.equal(a.performance,335e12);assert.equal(b.performance,989e12);assert.equal(a.bottleneck,'Memory-Bound');assert.equal(b.bottleneck,'Compute-Bound');close(a.ridge,989/3.35);
});
test('Bandwidth and compute demonstrations can change the teaching bottleneck',()=>{
  const wide=C.rooflineTeachingPoints({...h,bandwidth:13.4}),fast=C.rooflineTeachingPoints({...h,peakTF:1978});
  assert.equal(wide[0].bottleneck,'Compute-Bound');assert.equal(wide[0].performance,989e12);close(wide[0].ridge,989/13.4);
  assert.equal(fast[1].bottleneck,'Memory-Bound');assert.equal(fast[1].performance,1675e12);close(fast[1].ridge,1978/3.35);
});
test('Measured milliseconds are converted to seconds and never invented',()=>{
  assert.equal(C.roofline(1e12,1e11,h).measuredPerformance,null);
  const r=C.roofline(1e12,1e11,h,100);assert.equal(r.measuredPerformance,1e13);close(r.ai,10);assert.ok(r.measuredPerformance<r.performance);
  assert.ok(C.roofline(1e12,1e11,h,1).measuredPerformance>C.roofline(1e12,1e11,h).performance,'above-roof measurement is not clamped');
  for(const t of [0,-1,Infinity,NaN,1e-320])assert.throws(()=>C.roofline(1e12,1e11,h,t));
});
test('Full generated interval, including exact boundary, governs doubled history',()=>{
  const s={...C.scenarioDefaults},p={...C.defaults,S:524000};assert.deepEqual(C.validateScenario(p,s),[]);assert.ok(C.validateScenario({...p,S:p.S*2},s).some(x=>x.includes('S+G')));
  assert.deepEqual(C.validateScenario({...p,S:1047553},s),[]);assert.ok(C.validateScenario({...p,S:1047554},s).length);
  assert.deepEqual(C.validateScenario({...p,S:1048576},{mode:'single',generated:1024}),[]);
});
test('Derived hardware overflow and underflow are rejected before plotting',()=>{
  for(const value of [Infinity,NaN,1e308,1e-320])assert.ok(C.validateHardware({...h,peakTF:value},C.defaults).length);
  assert.throws(()=>C.roofline(1e12,1e-320,h));
});
test('MFU named scenarios retain identical all-device hardware denominator',()=>{
  const a=C.calculateMfu(C.mfuDefaults),b=C.calculateMfu({...C.mfuDefaults,tokensPerSecond:1000}),c=C.calculateMfu({...C.mfuDefaults,trainingFlopsPerToken:5e11,tokensPerSecond:4000});
  assert.equal(a.percent,50);assert.equal(b.percent,25);assert.equal(c.percent,50);assert.equal(b.hardwarePeak,a.hardwarePeak);assert.equal(c.hardwarePeak,a.hardwarePeak);
});
test('MFU derived percent overflow is rejected, rather than rendered as Infinity',()=>{
  assert.throws(()=>C.calculateMfu({...C.mfuDefaults,trainingFlopsPerToken:1e290,tokensPerSecond:1,deviceCount:1,peakTF:1e-29}));
});
test('GDA incomplete total stays null even when runtime and custom bytes are supplied',()=>{
  const a=C.analyzeRoofline(C.defaults,C.scenarioDefaults,h,{...C.trafficDefaults,mode:'custom',customBytes:{GDA:1e9},runtimesMs:{GDA:1}}).results.find(x=>x.name==='GDA');
  assert.equal(a.flops,null);assert.equal(a.performance,null);assert.equal(a.measuredPerformance,null);assert.equal(a.bottleneck,'N/A');
});
console.log(`\n${checks} final-round calculation checks passed.`);
