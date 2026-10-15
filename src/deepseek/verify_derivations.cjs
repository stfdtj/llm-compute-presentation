const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch(require('./browser_options.cjs')());
 try{
 const context=await browser.newContext({viewport:{width:1920,height:1080},offline:true});
 const page=await context.newPage(),errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/')+'#architecture');await page.evaluate(()=>document.fonts.ready);
 const geometry=()=>page.locator('#diagram-content rect,#diagram-content path').evaluateAll(ns=>ns.map(n=>[n.tagName,...['x','y','width','height','d'].map(k=>n.getAttribute(k))]));
 const before=await geometry(),packed=await page.locator('#derivation-data').textContent();
 // Priority modules: actual visible steps, A/B switching, and no stale partial panels.
 for(const key of ['hc','moe','decoderattn','multi','indexer','compressor','engram']){
  await page.evaluate(k=>openPanel(k),key);
  for(const mode of ['forward','backward']){
   await page.locator('[data-mode="'+mode+'"]').click();
   for(const scenario of ['A','B']){
    await page.locator('#derivation-scenario').selectOption(scenario);
    const ref=await page.evaluate(()=>architectureApp.data[architectureApp.state.selected][architectureApp.state.mode].ledgerRef);
    assert.equal(ref,key+'/'+scenario+'/'+mode);
    const entries=await page.evaluate(ref=>derivationTeaching.data.flows[ref].steps,ref);
    assert.equal(await page.locator('.derivation-step').count(),entries.length);
    assert.equal(await page.locator('.calc-step').count(),0,'old partial prose replaced');
    for(let i=0;i<entries.length;i++){
     const step=page.locator('.derivation-step').nth(i);
     await step.evaluate(n=>n.open=true);
     const text=await step.innerText();
     for(const title of ['这一步在做什么','输入 / 输出','FLOPs 怎样推导','本步骤小计','该路径累计','为什么需要'])assert.ok(text.includes(title),key+' '+title);
    }
    assert.ok(await page.locator('#derivation-walkthrough math').count()>0);
    assert.ok((await page.locator('.teaching-result').innerText()).includes('特殊函数'));
    if(key==='indexer'&&mode==='backward')assert.ok(await page.locator('.derivation-conditional').count());
   }
  }
  await page.locator('#derivation-scenario').selectOption('B');
  await page.locator('.derivation-step').evaluateAll(ns=>ns.forEach((n,i)=>n.open=i===0));
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='1');
  await page.screenshot({path:path.join(__dirname,'qa','derivation_'+key+'_backward.png')});
 }
 let coverage=0;
 const keys=await page.evaluate(()=>Object.keys(architectureApp.data));
 for(const key of keys)for(const mode of ['forward','backward']){
  await page.evaluate(k=>openPanel(k),key);await page.locator('[data-mode="'+mode+'"]').click();
  for(const scenario of ['A','B']){
   await page.locator('#derivation-scenario').selectOption(scenario);
   assert.ok(await page.locator('.derivation-step').count()>0,key+' has a graph');
   const values=await page.evaluate(()=>architectureApp.data[architectureApp.state.selected][architectureApp.state.mode].rows);
   const visible=await page.locator('.teaching-result').innerText();for(const [,v] of values)assert.ok(visible.includes(v));
   coverage++;
  }
 }
 await page.evaluate(()=>openPanel('hc'));await page.locator('[data-mode="forward"]').click();await page.locator('#derivation-scenario').selectOption('A');
 await page.locator('[data-derivation-jump="1"]').click();assert.equal(await page.locator('.derivation-step').nth(1).getAttribute('open'),'');
 await page.locator('.derivation-parameter-ref > summary').click();
 assert.ok((await page.locator('#parameter-mini').innerText()).includes('W_HC'));
 await page.locator('[data-mode="parameter"]').click();assert.equal(await page.locator('#derivation-walkthrough').count(),0);
 assert.ok(await page.locator('.parameter-table').count()>0);
 await page.keyboard.press('Escape');assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true');
 await page.locator('#region-select').selectOption('c2');await page.locator('[data-indexer-entry="decoder"] .indexer-module').click();
 await page.locator('[data-mode="backward"]').click();assert.equal(await page.locator('#derivation-walkthrough').count(),1);
 await page.locator('#close-panel').click();await page.locator('#indexer-close').click();
 await page.locator('#flow-toggle').click();for(let i=0;i<3;i++)await page.locator('#flow-next').click();
 assert.equal((await page.evaluate(()=>attentionTeaching.state)).active,true);await page.locator('#flow-toggle').click();
 for(const [width,height] of [[1366,768],[768,1024],[390,844]]){
  await page.setViewportSize({width,height});await page.evaluate(()=>openPanel('hc'));await page.locator('[data-mode="backward"]').click();
  const panel=await page.locator('#inspector').boundingBox();assert.ok(panel.x>=0&&panel.x+panel.width<=width+.1&&panel.y+panel.height<=height+.1);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='1');
  await page.screenshot({path:path.join(__dirname,'qa','derivation_'+width+'.png')});await page.keyboard.press('Escape');
 }
 assert.deepEqual(await geometry(),before);assert.equal(await page.locator('#derivation-data').textContent(),packed,'canonical ledger not mutated');
 assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith('file:')||r.startsWith('data:')));
 const report={passed:true,moduleScenarioModes:coverage,priorityModules:7,fieldsPerStep:6,parameterBinding:true,originalGeometry:true,offline:true,responsive:[1920,1366,768,390],errors};
 fs.writeFileSync(path.join(__dirname,'qa','derivation_browser_validation.json'),JSON.stringify(report,null,2));console.log(report);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
