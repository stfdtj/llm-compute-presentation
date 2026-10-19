/* Meaningful browser integration checks: progression, mode semantics, timers,
 * architecture preservation, keyboard access, offline and responsive behavior. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({viewport:{width:1440,height:1000},offline:true});
 const page=await context.newPage(),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 const out=path.join(__dirname,'qa','walkthrough');fs.mkdirSync(out,{recursive:true});
 await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/'));
 await page.evaluate(()=>document.fonts.ready);
 const data=await page.locator('#module-data').textContent();
 const geometry=await page.locator('#diagram-content').innerHTML();
 assert.equal(await page.locator('#csa-walkthrough').isVisible(),false);
 await page.locator('#csa-tab-walkthrough').click();
 await page.waitForFunction(()=>document.getElementById('csa-walkthrough').getBoundingClientRect().top>document.querySelector('header').getBoundingClientRect().bottom);
 assert.equal(await page.locator('#stage').isVisible(),false);
 assert.equal(await page.locator('#csa-prev').isDisabled(),true);
 assert.equal(await page.locator('#csa-indicator').innerText(),'Step 1 / 6');
 for(let step=1;step<=6;step++){
  assert.equal((await page.evaluate(()=>csaWalkthrough.state)).step,step);
  assert.ok((await page.locator('#csa-description').innerText()).length>10);
  assert.equal(await page.locator('[data-csa-step][aria-current="step"]').count(),1);
  await page.screenshot({path:path.join(out,`step-${step}.png`)});
  if(step<6)await page.locator('#csa-next').click();
 }
 assert.equal(await page.locator('#csa-next').isDisabled(),true);
 for(const mode of ['Full','Reindex','Reuse']){
  await page.locator(`[data-csa-mode="${mode}"]`).click();
  assert.equal((await page.evaluate(()=>csaWalkthrough.state)).mode,mode);
  const reuse=mode==='Reuse';
  assert.equal(await page.locator('[data-csa-node="indexer-scoring"]').evaluate(n=>n.classList.contains('csa-cached')),reuse);
  assert.equal(await page.locator('[data-csa-node="main-attention"]').evaluate(n=>n.classList.contains('csa-dim')||n.classList.contains('csa-cached')),false);
  assert.ok((await page.locator('#csa-context').innerText()).includes(mode==='Full'?'ratio = 2':'ratio = 1'));
  await page.screenshot({path:path.join(out,`mode-${mode}.png`)});
 }
 await page.locator('#csa-reset').click();
 assert.deepEqual(await page.evaluate(()=>csaWalkthrough.state),{active:true,step:1,mode:'Full',playing:false,matrixStep:1,presentation:false});
 // A virtual browser clock makes Play/Pause/end-of-sequence checks deterministic.
 await page.clock.install();
 await page.locator('#csa-play').click();await page.clock.runFor(3050);
 assert.equal((await page.evaluate(()=>csaWalkthrough.state)).step,2);
 await page.locator('#csa-play').click();await page.clock.runFor(6000);
 assert.equal((await page.evaluate(()=>csaWalkthrough.state)).step,2,'Pause stops auto-advance');
 await page.locator('#csa-play').click();await page.locator('#csa-next').click();await page.clock.runFor(6000);
 assert.equal((await page.evaluate(()=>csaWalkthrough.state)).step,3,'Manual navigation cancels playback');
 await page.locator('#csa-play').click();await page.clock.runFor(10000);
 assert.equal((await page.evaluate(()=>csaWalkthrough.state)).step,6);assert.equal((await page.evaluate(()=>csaWalkthrough.state)).playing,false,'Playback stops on last step');
 await page.locator('#csa-reset').click();await page.locator('#csa-walkthrough').focus();await page.keyboard.press('ArrowRight');
 assert.equal((await page.evaluate(()=>csaWalkthrough.state)).step,2);
 await page.locator('#csa-play').click();await page.locator('#csa-tab-architecture').click();await page.clock.runFor(6000);
 assert.equal((await page.evaluate(()=>csaWalkthrough.state)).playing,false);
 assert.equal(await page.locator('#stage').isVisible(),true);
 assert.equal(await page.locator('#module-data').textContent(),data);
 assert.equal(await page.locator('#diagram-content').innerHTML(),geometry);
 await page.locator('.module[data-module="qa"]').first().dispatchEvent('keydown',{key:'Enter'});
 assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'false');await page.keyboard.press('Escape');
 await page.locator('#csa-tab-walkthrough').click();await page.locator('.csa-matrix-extra summary').click();
 for(let i=1;i<8;i++)await page.locator('#csa-matrix-next').click();
 assert.equal(await page.locator('#csa-matrix-indicator').innerText(),'8 / 8');
 assert.ok((await page.locator('#csa-matrix-scene').innerText()).includes('64'));
 assert.ok((await page.locator('.csa-matrix-conclusion').innerText()).includes('20 FLOPs'));
 await page.locator('.csa-matrix-extra').screenshot({path:path.join(out,'matrix.png')});
 await page.locator('#csa-matrix-reset').click();assert.equal(await page.locator('#csa-matrix-prev').isDisabled(),true);
 await page.locator('.csa-matrix-extra summary').click();
 for(const size of [{width:1920,height:1080},{width:1366,height:768},{width:768,height:1024},{width:390,height:844}]){
  await page.setViewportSize(size);await page.evaluate(()=>document.getElementById('csa-walkthrough').scrollTop=0);await page.clock.runFor(100);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'No horizontal page overflow');
  const header=await page.locator('header').boundingBox(),root=await page.locator('#csa-walkthrough').boundingBox();
  assert.ok(root.y>=header.y+header.height,'Walkthrough starts below header');
  assert.ok(root.x>=0&&root.x+root.width<=size.width+.1);
  assert.ok(root.height>250,'Usable vertical space');
  await page.screenshot({path:path.join(out,`responsive-${size.width}.png`)});
  await page.locator('#csa-tab-architecture').click();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'Architecture still fits viewport');
  await page.locator('#csa-tab-walkthrough').click();
 }
 await page.setViewportSize({width:1366,height:768});await page.locator('#csa-focus').click();await page.clock.runFor(100);
 assert.equal((await page.evaluate(()=>csaWalkthrough.state)).presentation,true);
 await page.locator('#csa-next').click();await page.locator('#csa-next').click();await page.locator('#csa-next').click();
 await page.screenshot({path:path.join(out,'presentation-1366.png')});
 const output=await page.locator('[data-csa-node="main-attention"]').boundingBox();
 assert.ok(output.y+output.height<=768,'Main Attention output fits presentation viewport');
 await page.locator('#csa-tab-architecture').click();assert.equal(await page.locator('body').evaluate(n=>n.classList.contains('csa-focus')),false);
 assert.deepEqual(errors,[]);assert.ok(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')));
 const report={passed:true,steps:6,modes:['Full','Reindex','Reuse'],matrixSteps:8,playPauseReset:true,timerCleanup:true,keyboard:true,presentationView:true,responsiveWidths:[1920,1366,768,390],architectureDataAndGeometryPreserved:true,offline:true,errors};
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(report,null,2));console.log(report);await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
