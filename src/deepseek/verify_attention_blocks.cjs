const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'../..'),out=path.join(__dirname,'qa/attention_inplace');
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({offline:true}),page=await context.newPage(),errors=[],network=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(!/^(file:|data:)/.test(r.url()))network.push(r.url());});
 await page.goto(require('node:url').pathToFileURL(path.join(root,'deepseek_v4_1_interactive.html')).href);
 assert.equal(await page.locator('#source-nav').count(),0,'no replacement page navigation');
 assert.equal(await page.locator('#source-workspace').count(),0,'no replacement workspace');
 const shots=[];let modes=0;
 for(const [width,height] of [[1920,1080],[1366,768]]){
  await page.setViewportSize({width,height});await page.locator('#reset').click();
  await page.screenshot({path:path.join(out,`original_layout_${width}.png`)});shots.push(`original_layout_${width}.png`);
  for(const [region,id] of [['c1','encoder'],['c2','decoder'],['draft','draft']]){
   await page.locator('#region-select').selectOption(region);
   const graph=page.locator(id==='draft'?'#draft-expanded':`#${id}-attention-expanded`);
   assert.equal(await graph.isVisible(),true);
   const text=await graph.textContent();
   for(const term of id==='draft'?['mtp[0]','mtp[1]','mtp[2]','forward_spec','5376']:['Indexer','[32,128]','[S,128]','sparse_attn','[5120,1280]','[8192,5120]'])assert.ok(text.includes(term),term);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   if(id!=='draft'){
    await graph.locator('.inline-indexer').first().click();assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true');
    assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,false,'inline Indexer never opens the separate calculation view');
    const module=graph.locator('.module[data-module="query"]').first();
    await module.click({position:{x:30,y:20}});
    for(const mode of ['parameter','forward','backward']){
     await page.locator(`[data-mode="${mode}"]`).click();assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'false');assert.ok(await page.locator('#panel-body math').count());modes++;
    }
    await page.screenshot({path:path.join(out,`${id}_ledger_${width}.png`)});
    await page.locator('#close-panel').click();await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='0');assert.equal(await page.locator('#region-select').inputValue(),region);
   }
   await page.screenshot({path:path.join(out,`${id}_${width}.png`)});shots.push(`${id}_${width}.png`);
  }
 }
 // Real in-place launch from the original compact blocks.
 for(const region of ['c1','c2']){
  await page.locator('#reset').click();await page.locator(`[data-attention-focus="${region}"]`).click();assert.equal(await page.locator('#region-select').inputValue(),region);
 }
 await page.locator('#flow-toggle').click();assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,true);assert.equal(await page.locator('#diagram-content').evaluate(n=>n.classList.contains('attention-block-focused')),false);
 await page.locator('#flow-toggle').click();assert.equal(await page.locator('#region-select').inputValue(),'all');assert.equal((await page.evaluate(()=>architectureApp.state)).view.x,0);
 await page.locator('#flow-toggle').click();await page.locator('#region-select').selectOption('c1');assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,false);assert.equal(await page.locator('#encoder-attention-expanded').isVisible(),true);
 assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
 const result={passed:true,originalPage:true,originalCanvas:'1672x941',expandedDetails:'same SVG',viewports:['1920x1080','1366x768'],calculationModes:modes,screenshots:shots,errors,network};
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
