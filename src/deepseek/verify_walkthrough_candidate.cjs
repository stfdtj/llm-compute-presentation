/* Candidate pool acceptance: real intermediate frames, immutable membership,
 * separate Full-layer output, per-layer reranking, and source preservation. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
let browser;
(async()=>{
 browser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({viewport:{width:1440,height:1050},offline:true});
 const page=await context.newPage(),errors=[],requests=[],evidence={},layers={};
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.clock.install({time:new Date('2026-10-20T00:00:00Z')});await page.clock.pauseAt(new Date('2026-10-20T00:00:00Z'));
 const out=path.join(__dirname,'qa','candidate_pool');fs.mkdirSync(out,{recursive:true});
 const url='file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/');await page.goto(url);await page.evaluate(()=>document.fonts.ready);
 const state=()=>page.evaluate(()=>csaWalkthrough.state),candidate=()=>page.evaluate(()=>csaWalkthrough.candidate);
 const pose=id=>page.locator(`[data-motion-id="${id}"]`).evaluate(n=>{const m=n.transform.baseVal.consolidate().matrix;return{x:m.e,y:m.f,sx:m.a,sy:m.d,opacity:Number(n.getAttribute('opacity')),value:n.querySelector('[data-motion-number]')?.textContent||null};});
 const next=async()=>{await page.locator('#csa-next').click();await page.clock.runFor(1900);};
 const shot=name=>page.screenshot({path:path.join(out,name+'.png')});
 async function motion(name,id){
  const begin=await pose(id);await shot(name+'-start');await page.locator('#csa-next').click();await page.clock.runFor(900);
  const middle=await pose(id),phase=(await state()).phase;assert.ok(phase>.4&&phase<.6);await shot(name+'-middle');
  await page.clock.runFor(1000);const end=await pose(id);await shot(name+'-end');assert.notDeepEqual(begin,middle,name+' changes during animation');assert.notDeepEqual(middle,end,name+' ends at a distinct pose');evidence[name]={begin,middle,end,phase};return evidence[name];
 }
 await page.locator('#csa-tab-walkthrough').click();await page.locator('[data-csa-chapter="4"]').click();await page.clock.runFor(1900);await page.locator('#csa-focus').click();await page.clock.runFor(100);
 assert.equal((await state()).step,23);
 await page.evaluate(()=>{window.__poolRefs=[...document.querySelectorAll('[data-motion-id^="hp-"]')];window.__poolChanges=0;new MutationObserver(ms=>{for(const m of ms)for(const n of [...m.addedNodes,...m.removedNodes])if(n.nodeType===1)__poolChanges++;}).observe(document.querySelector('[data-scene="candidate-pool"]'),{childList:true,subtree:true});});
 assert.deepEqual((await candidate()).real,{blockSize:8,blocks:2048,poolLimit:16384,topK:512});
 await motion('l20-scores','hp-score-31');
 const raw=[];for(let i=0;i<32;i++)raw.push(Number((await pose('hp-score-'+i)).value));
 await motion('eight-per-block','hp-pos-8');
 await motion('block-max','hp-max-fly-0');
 const c=await candidate();for(let b=0;b<4;b++)assert.equal(c.blockScores[b],Math.max(...raw.slice(b*8,b*8+8)));
 await next();assert.deepEqual(c.poolBlocks,[1,3]);assert.equal((await pose('hp-block-0')).opacity,.12);assert.equal((await pose('hp-block-1')).opacity,1);await shot('selected-blocks');
 const gathered=await motion('candidate-pool-gather','hp-pos-8');assert.ok(Math.abs(gathered.end.x-gathered.begin.x)>300);
 const pool=c.pool;assert.equal(pool.length,16);assert.equal(new Set(pool).size,16);assert.deepEqual(pool,[8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31]);
 await motion('l20-independent-topk','hp-own-3');assert.equal(c.sourceTop.length,4);assert.ok(c.sourceTop.some(id=>!pool.includes(id)),'L20 Top-K is not restricted to its Candidate Pool');
 const anchor=await Promise.all(pool.map(id=>pose('hp-pos-'+id)));
 for(const layer of [24,28,32,36]){
  await motion('transfer-l'+layer,'hp-carry');
  assert.equal((await candidate()).layer,layer);
  const scoring=await motion('rescore-l'+layer,'hp-score-8');assert.notEqual(scoring.begin.value,scoring.end.value,'Each layer produces new scores');
  await motion('topk-l'+layer,'hp-top-0');
  const selected=(await candidate()).topByLayer[layer];assert.ok(selected.every(id=>pool.includes(id)));
  const scores={};for(const id of pool)scores[id]=Number((await pose('hp-score-'+id)).value);
  const expected=pool.slice().sort((a,b)=>scores[b]-scores[a]||a-b).slice(0,4);assert.deepEqual(selected,expected);
  const current=await Promise.all(pool.map(id=>pose('hp-pos-'+id)));current.forEach((p,i)=>{assert.equal(p.x,anchor[i].x);assert.equal(p.y,anchor[i].y);});assert.deepEqual((await candidate()).pool,pool,'Pool membership does not change across layers');
  await motion('attention-l'+layer,'hp-global-read-0');assert.equal((await pose('hp-output')).opacity,1);
  const read=await page.locator('[data-read-id]').allTextContents();assert.deepEqual(read.map(Number),selected);
  layers[layer]={pool:pool.slice(),topK:selected,scores};
 }
 assert.equal(new Set(Object.values(layers).map(v=>v.topK.join(','))).size,4,'Different Queries produce four distinct Top-K outputs');
 assert.equal(await page.evaluate(()=>__poolRefs.every(n=>n.isConnected)&&__poolChanges===0),true,'Candidate scene preserves every SVG element');
 // Replay / pause / seek on the new geometry, not a recreated SVG.
 await page.locator('#csa-replay').click();await page.clock.runFor(600);await page.locator('#csa-play').click();const frozen=await pose('hp-global-read-0');await page.clock.runFor(3000);assert.deepEqual(await pose('hp-global-read-0'),frozen);
 await page.locator('#csa-progress').fill('50');assert.equal((await state()).phase,.5);await page.locator('#csa-play').click();await page.clock.runFor(1000);await page.locator('#csa-play').click();
 // New scene fits the presentation viewport and remains usable on a phone.
 await page.setViewportSize({width:1366,height:768});await page.clock.runFor(100);await shot('presentation-1366');const box=await page.locator('#csa-canvas').boundingBox();assert.ok(box.y+box.height<768);
 await page.setViewportSize({width:390,height:844});await page.clock.runFor(100);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await shot('responsive-390');
 // Source check: SWA / Compression definitions and poses remain byte-for-byte
 // identical; the non-walkthrough page prefix is preserved as well.
 const old=cp.execFileSync('git',['show','HEAD:src/deepseek/walkthrough_motion.js'],{encoding:'utf8',maxBuffer:8e6}).replaceAll('\r\n','\n');
 const current=fs.readFileSync(path.join(__dirname,'walkthrough_motion.js'),'utf8').replaceAll('\r\n','\n');
 for(const [start,end]of [[' // SWA:',' // Indexer:'],['  for(let i=0;i<8;i++){set(\'swa-q-', '  const nScores=']])assert.equal(current.slice(current.indexOf(start),current.indexOf(end)),old.slice(old.indexOf(start),old.indexOf(end)));
 const oldHtml=cp.execFileSync('git',['show','HEAD:deepseek_v4_1_interactive.html'],{encoding:'utf8',maxBuffer:8e6}).replaceAll('\r\n','\n'),newHtml=fs.readFileSync(path.resolve(__dirname,'../../deepseek_v4_1_interactive.html'),'utf8').replaceAll('\r\n','\n');assert.equal(newHtml.split('<!-- Persistent matrix teaching scenes.')[0],oldHtml.split('<!-- Persistent matrix teaching scenes.')[0]);
 assert.deepEqual(errors,[]);assert.ok(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')));
 const report={passed:true,actions:(await page.evaluate(()=>csaWalkthrough.actions.length)),realParameters:c.real,sourceTopK:c.sourceTop,poolBlocks:c.poolBlocks,pool,blockScores:c.blockScores,layers,animationEvidence:evidence,poolIdentityPreserved:true,sourceTopKIndependent:true,originalSwaCompressionUnchanged:true,originalPagePrefixUnchanged:true,offline:true,errors};fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(report,null,2));
 const rows=Object.keys(evidence).map(name=>'<section><h2>'+name+'</h2><div>'+['start','middle','end'].map(p=>'<figure><a href="'+name+'-'+p+'.png"><img loading="lazy" src="'+name+'-'+p+'.png" alt="'+p+'"></a><figcaption>'+p+'</figcaption></figure>').join('')+'</div></section>').join('');
 fs.writeFileSync(path.join(out,'review.html'),'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Candidate Pool 动画验收</title><style>body{font:15px system-ui;color:#20375c;background:#f6f8fc;margin:30px auto;padding:0 20px;max-width:1650px}section>div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;padding:8px;background:white;border:1px solid #dce5f0;border-radius:7px}img{width:100%}h2{font-size:20px;margin-top:30px}a{color:#2863c5}@media(max-width:700px){section>div{grid-template-columns:1fr}}</style><h1>Hierarchical Indexing / Candidate Pool</h1><p>实际浏览器动画的开始、中间约 50% 和结束帧；可点击查看原图。矩阵元素持续存在，候选池跨层保持同一组位置。</p><p><a href="../../../../deepseek_v4_1_interactive.html">打开演示</a> · <a href="validation.json">核对与几何数据</a></p>'+rows+'</html>');
 console.log({passed:true,actions:report.actions,sourceTopK:c.sourceTop,pool:pool.length,topByLayer:c.topByLayer,evidenceScenes:Object.keys(evidence).length,errors});await browser.close();
})().catch(async e=>{console.error(e);if(browser)await browser.close();process.exitCode=1;});
