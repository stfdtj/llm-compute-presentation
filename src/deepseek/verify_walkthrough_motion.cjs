/* Drive the unified Data Flow + Shape walkthrough's actual controls and RAF clock;
 * capture real in-between poses. Covers what used to be two separate suites
 * (verify_walkthrough_motion.cjs for CSA2, verify_shape_trace.cjs for Shape Trace),
 * now merged into one feature. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
let activeBrowser;
(async()=>{
 const browser=activeBrowser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({viewport:{width:1440,height:1050},offline:true});
 const page=await context.newPage(),errors=[],requests=[],evidence={};
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 const out=path.join(__dirname,'qa','walkthrough');fs.mkdirSync(out,{recursive:true});
 await page.clock.install({time:new Date('2026-10-21T10:00:00Z')});await page.clock.pauseAt(new Date('2026-10-21T10:00:00Z'));
 await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/'));await page.evaluate(()=>document.fonts.ready);
 const data=await page.locator('#module-data').textContent(),geometry=await page.locator('#diagram-content').innerHTML();
 const state=()=>page.evaluate(()=>csaWalkthrough.state);
 const pose=id=>page.locator(`[data-motion-id="${id}"]`).evaluate(n=>{const m=n.transform.baseVal.consolidate().matrix;return{x:m.e,y:m.f,sx:m.a,sy:m.d,opacity:Number(n.getAttribute('opacity'))};});
 const shot=name=>page.screenshot({path:path.join(out,name+'.png')});
 const next=async()=>{await page.locator('#csa-next').click();await page.clock.runFor(1300);};
 const stageJump=async i=>{await page.locator(`[data-csa-stage="${i}"]`).click();await page.clock.runFor(1300);};
 async function motion(name,id,button='#csa-next'){
  const begin=await pose(id);await shot(name+'-start');await page.locator(button).click();await page.clock.runFor(600);
  const middle=await pose(id),s=await state();assert.ok(s.phase>.3&&s.phase<.7,'Real in-flight frame, got phase='+s.phase);await shot(name+'-middle');
  await page.clock.runFor(700);const end=await pose(id);await shot(name+'-end');
  assert.notDeepEqual(middle,begin,name+' changes geometry');assert.notDeepEqual(middle,end,name+' has a distinct middle frame');
  evidence[name]={begin,middle,end,phaseAtMiddle:s.phase};return {begin,middle,end};
 }

 await page.locator('#csa-tab-walkthrough').click();await page.clock.runFor(100);
 await page.evaluate(()=>{window.__csaNodes=[...document.querySelectorAll('#csa-canvas svg, #csa-canvas [data-motion-id]')];window.__csaElementChanges=0;new MutationObserver(ms=>{for(const m of ms)for(const n of [...m.addedNodes,...m.removedNodes])if(n.nodeType===1)__csaElementChanges++;}).observe(document.getElementById('csa-canvas'),{childList:true,subtree:true});});

 // -------------------------------------------------------------- structure
 assert.equal((await state()).layer,2);assert.equal((await state()).stage,1);assert.equal((await state()).beat,1);
 const stages=await page.evaluate(()=>csaWalkthrough.stages);
 assert.equal(stages.length,8,'8 main stages, grouped from the former 23 fine-grained actions');
 assert.equal(await page.locator('[data-csa-stage]').count(),8);
 assert.equal(await page.locator('[data-csa-layer]').count(),4,'4 representative layer buttons: 2 / 3 / 20 / 24');
 assert.deepEqual(await page.locator('[data-csa-layer]').evaluateAll(els=>els.map(e=>e.dataset.csaLayer)),['2','3','20','24']);

 // -------------------------------------------------------------- cross-feature contract: #flow-toggle / informationFlow / .flow-entry
 // Run this early, right after the tab opens cleanly, rather than after the heavy
 // play/pause/clock-jump choreography below -- avoids flakiness from residual RAF
 // state when switching tabs mid-animation (a Playwright/virtual-clock interaction,
 // not a product bug: manually verified safe in a real browser either way).
 assert.equal(await page.locator('#flow-toggle').count(),1);assert.equal(await page.locator('#flow-bar').count(),1);
 assert.equal(await page.locator('#shape-layer option').count(),40,'Full 0-39 layer range preserved for cross-panel compatibility');
 await page.locator('#csa-tab-architecture').click();await page.clock.runFor(100);
 await page.locator('#flow-toggle').click();assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,true);
 await page.locator('#shape-layer').selectOption('39');assert.equal((await page.evaluate(()=>informationFlow.state)).layer,39);
 await page.locator('#flow-toggle').click();assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,false);
 await page.locator('#csa-tab-walkthrough').click();await page.clock.runFor(100);await page.locator('#csa-reset').click();

 // -------------------------------------------------------------- stage 1-2: mHC Pre + Query Projection
 const mhcReduce=await motion('mhc-pre-reduce','mhc-pre-out');assert.notEqual(mhcReduce.end.opacity,0);
 await next(); // into stage 2
 const qProj=await motion('query-wqa','q-mid');assert.ok(qProj.end.opacity>qProj.begin.opacity||qProj.end.x!==qProj.begin.x,'Query flows into W_QA');
 const readoutQ1=await page.locator('#csa-readout').innerText();
 assert.match(readoutQ1,/W_QA/);assert.match(readoutQ1,/\[5120,1280\]/,'W_QA weight shape shown exactly as [5120,1280]');
 await next();const readoutQ2=await page.locator('#csa-readout').innerText();assert.match(readoutQ2,/\[1280,32768\]/,'W_QB weight shape shown exactly as [1280,32768]');
 await next();const readoutQ3=await page.locator('#csa-readout').innerText();assert.match(readoutQ3,/\[1,64,512\]/,'Final Query shape is [1,64,512]');

 // -------------------------------------------------------------- stage 4: Compression, current token vs history
 await stageJump(3);
 const compReadout=await page.locator('#csa-readout').innerText();
 assert.match(compReadout,/这个 Token 自己的贡献|这个 Token/,'Layer 2 (Full) stage 4 first beat distinguishes the current token\'s own contribution');
 await next();const compReadout2=await page.locator('#csa-readout').innerText();assert.match(compReadout2,/压缩比 2|2 个 Token/,'Second beat is the 2:1 pairing, distinct from the first beat');
 await next();const compReadout3=await page.locator('#csa-readout').innerText();assert.match(compReadout3,/历史/,'Final beat is explicitly the multi-position history cache, not the single token');

 // -------------------------------------------------------------- full walkthrough, every layer
 const totals={};
 for(const L of [2,3,20,24]){
  await page.locator(`[data-csa-layer="${L}"]`).click();await page.clock.runFor(200);
  let guard=0;
  while(!(await page.locator('#csa-next').isDisabled())&&guard<40){await page.locator('#csa-next').click();await page.clock.runFor(400);guard++;}
  const s=await state();assert.equal(s.stage,8);assert.equal(s.beat,4,`Layer ${L} ends on stage 8 beat 4`);
  totals[L]=s.flat;
  assert.ok(await page.locator('#csa-prev').isDisabled()===false);
 }
 assert.ok(totals[3]<totals[2],'Layer 3 (Reuse) has fewer beats than Layer 2 (Full): compression/indexer collapse to one beat each');

 // -------------------------------------------------------------- Reindex/Reuse inline captions (no separate comparison screen)
 await page.locator('[data-csa-layer="24"]').click();await page.clock.runFor(200);
 await stageJump(3);
 assert.match(await page.locator('#csa-readout').innerText(),/复用自 Layer 20/,'Layer 24 stage 4: Global KV reused from Layer 20');
 await stageJump(4);
 const idxBeat1=await page.locator('#csa-readout').innerText();assert.match(idxBeat1,/Indexer Query.*独立计算|本层独立计算/,'Layer 24: Indexer Query freshly computed');
 await next();assert.match(await page.locator('#csa-readout').innerText(),/Index Keys 复用自 Layer 20/,'Layer 24: Index Keys reused from Layer 20');
 await next();await next();assert.match(await page.locator('#csa-readout').innerText(),/Top-K/,'Layer 24: fresh Top-K selection');

 await page.locator('[data-csa-layer="3"]').click();await page.clock.runFor(200);
 await stageJump(4);
 assert.match(await page.locator('#csa-readout').innerText(),/Top-K 索引复用自 Layer 2/,'Layer 3 (Reuse): Top-K indices reused, no Indexer computation shown');
 await page.locator('[data-csa-layer="2"]').click();await page.clock.runFor(200);

 // -------------------------------------------------------------- play/pause/reset/scrub/keyboard
 await page.locator('#csa-reset').click();assert.equal((await state()).layer,2);assert.equal((await state()).flat,1);
 await page.locator('#csa-next').click();await page.clock.runFor(700);await page.locator('#csa-play').click();
 const pausedPose=await pose('mhc-stream-0'),pausedPhase=(await state()).phase;await page.clock.runFor(3000);
 assert.deepEqual(await pose('mhc-stream-0'),pausedPose);assert.equal((await state()).phase,pausedPhase);
 await page.locator('#csa-play').click();await page.clock.runFor(1200);await page.locator('#csa-play').click();
 assert.ok((await state()).flat>=2);assert.equal((await state()).phase,1);
 await page.locator('#csa-progress').fill('40');await page.locator('#csa-progress').dispatchEvent('input');assert.equal((await state()).phase,.4);assert.equal((await state()).running,false);
 await page.locator('#csa-reset').click();assert.equal((await state()).flat,1);assert.equal((await state()).layer,2);
 await page.locator('#csa-play').click();await page.clock.runFor(60000);assert.equal((await state()).playing,false);assert.equal((await state()).stage,8);assert.equal((await state()).beat,4);
 await page.locator('#csa-reset').click();await page.locator('#flow-bar').focus();await page.keyboard.press('ArrowRight');await page.clock.runFor(1300);assert.equal((await state()).flat,2);
 await page.locator('#csa-play').click();await page.clock.runFor(500);await page.locator('#csa-tab-architecture').click();const stopped=await state();await page.clock.runFor(5000);assert.equal((await state()).flat,stopped.flat);assert.equal((await state()).playing,false);

 assert.equal(await page.locator('#module-data').textContent(),data);assert.equal(await page.locator('#diagram-content').innerHTML(),geometry);

 await page.locator('#csa-tab-walkthrough').click();await page.clock.runFor(100);
 assert.equal(await page.evaluate(()=>__csaNodes.every(n=>n.isConnected)&&document.querySelectorAll('#csa-canvas svg').length===1&&__csaElementChanges===0),true,'No SVG/row replacement across the whole run');

 // -------------------------------------------------------------- matrix sub-demo (unrelated, unchanged)
 await page.locator('.csa-matrix-extra summary').click();for(let i=0;i<7;i++)await page.locator('#csa-matrix-next').click();assert.ok((await page.locator('.csa-matrix-conclusion').innerText()).includes('20 FLOPs'));await page.locator('.csa-matrix-extra').screenshot({path:path.join(out,'matrix.png')});await page.locator('.csa-matrix-extra summary').click();

 // -------------------------------------------------------------- original calculation panels still reachable (real-time page)
 const original=await context.newPage();original.on('pageerror',e=>errors.push(e.message));await original.goto(page.url());await original.locator('#csa-tab-walkthrough').click();await original.locator('#csa-tab-architecture').click();
 for(const mode of ['parameter','forward','backward']){await original.locator(`header [data-mode="${mode}"]`).click();await original.locator('.module[data-module="qa"]').first().dispatchEvent('keydown',{key:'Enter'});await original.locator('#panel-body').waitFor({state:'visible'});assert.equal(await original.locator('#inspector').getAttribute('aria-hidden'),'false');assert.ok((await original.locator('#panel-body').textContent()).length>100,mode+' formulas present');await original.keyboard.press('Escape');}
 await original.close();

 // -------------------------------------------------------------- responsive
 await page.locator('#csa-tab-walkthrough').click();await page.locator('#csa-reset').click();
 for(const size of [{width:1920,height:1080},{width:1366,height:768},{width:768,height:1024},{width:390,height:844}]){await page.setViewportSize(size);await page.clock.runFor(100);await page.evaluate(()=>document.getElementById('csa-scroll').scrollTop=0);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'No horizontal overflow at '+size.width);await shot('responsive-'+size.width);}
 await page.setViewportSize({width:1366,height:768});await stageJump(5);await page.locator('#csa-focus').click();await next();await next();await shot('presentation-1366');const canvas=await page.locator('#csa-canvas').boundingBox();assert.ok(canvas.y+canvas.height<768,'Scene fits presentation viewport');
 await page.locator('#csa-focus').click();

 assert.deepEqual(errors,[]);assert.ok(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')));
 const report={passed:true,stages:8,representativeLayers:[2,3,20,24],beatsPerLayer:totals,persistentSvg:true,elementReplacements:0,realAnimationEvidence:evidence,pauseFreezesGeometry:true,resume:true,scrub:true,playToEnd:true,reset:true,crossFeatureContract:{flowToggle:true,informationFlow:true,shapeLayerFullRange:true},originalCalculationModes:['parameter','forward','backward'],architectureDataAndGeometryPreserved:true,responsiveWidths:[1920,1366,768,390],offline:true,errors};
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(report,null,2));
 console.log({passed:true,stages:8,layers:[2,3,20,24],beatsPerLayer:totals,animationScenes:Object.keys(evidence),errors});
 await browser.close();
})().catch(async e=>{console.error(e);if(activeBrowser)await activeBrowser.close();process.exitCode=1;});
