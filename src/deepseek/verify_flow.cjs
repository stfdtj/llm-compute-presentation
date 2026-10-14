const {chromium}=require('playwright');
const assert=require('assert/strict');
const fs=require('fs'),path=require('path');
(async()=>{
const browser=await chromium.launch(require('./browser_options.cjs')());
const context=await browser.newContext({viewport:{width:1920,height:1080},offline:true});
const page=await context.newPage(),errors=[],requests=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/'));
await page.evaluate(()=>document.fonts.ready);
const data=await page.locator('#module-data').textContent();
const geometry=()=>page.locator('#diagram-content').evaluate(n=>[...n.querySelectorAll('rect,path')].map(e=>[e.tagName,...['x','y','width','height','d','fill','stroke'].map(k=>e.getAttribute(k))]));
const before=await geometry();
await page.locator('#flow-toggle').click();assert.equal((await page.evaluate(()=>informationFlow.state)).step,1);
for(let step=1;step<=5;step++){
 assert.equal(await page.locator('#flow-counter').innerText(),step+'/5');
 assert.ok(await page.locator('.flow-hot').count()>0);
 assert.ok(await page.locator('.flow-path[marker-end]').count()>0);
 if(step===3){await page.locator('#flow-write').click();await page.locator('#flow-write').click();assert.equal((await page.evaluate(()=>informationFlow.state)).written,3);assert.ok(await page.locator('#flow-write').isDisabled());}
 if(step===4){for(const filter of ['hidden','global','local','all']){await page.locator(`[data-flow-filter="${filter}"]`).click();assert.equal((await page.evaluate(()=>informationFlow.state)).filter,filter);}}
 await page.waitForFunction(()=>getComputedStyle(document.querySelector('#inspector')).opacity==='0');
 await page.screenshot({path:path.join(__dirname,'qa',`flow_step_${step}.png`)});
 await page.locator('#flow-explain').click();assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'false');
 await page.locator('#flow-to-calc').click();assert.ok(await page.locator('#panel-body math').count()>0);
 for(const mode of ['parameter','forward','backward']){await page.locator(`[data-mode="${mode}"]`).click();assert.equal((await page.evaluate(()=>architectureApp.state)).mode,mode);}
 await page.keyboard.press('Escape');
 if(step<5)await page.locator('#flow-next').click();
}
await page.locator('#flow-prev').click();assert.equal((await page.evaluate(()=>informationFlow.state)).step,4);
await page.locator('#flow-replay').click();assert.equal((await page.evaluate(()=>informationFlow.state)).step,1);
await page.locator('.module[data-module="global"]').first().click();assert.equal((await page.evaluate(()=>informationFlow.state)).step,2);
assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'false');
await page.mouse.click(20,1000);assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true');
await page.locator('#flow-toggle').click();assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,false);
assert.deepEqual(await geometry(),before);assert.equal(await page.locator('#module-data').textContent(),data);
assert.equal(await page.locator('.flow-hot').count(),0);
await page.locator('.module[data-module="global"]').first().dispatchEvent('keydown',{key:'Enter'});
await page.locator('.flow-entry').click();assert.equal((await page.evaluate(()=>informationFlow.state)).step,2);
for(const size of [{width:1366,height:768},{width:768,height:1024},{width:390,height:844}]){
 await page.setViewportSize(size);await page.locator('#flow-next').click();await page.locator('#flow-focus').click();
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'no horizontal page overflow');
 const bar=await page.locator('#flow-bar').boundingBox(),stage=await page.locator('#stage').boundingBox();assert.ok(stage.y>=bar.y+bar.height,'bar outside diagram');
 assert.ok(stage.height>200,'enough diagram space');
 await page.screenshot({path:path.join(__dirname,'qa',`flow_${size.width}.png`)});
 await page.locator('#flow-explain').click();await page.waitForFunction(()=>getComputedStyle(document.querySelector('#inspector')).opacity==='1');
 const panel=await page.locator('#inspector').boundingBox();assert.ok(panel.x>=0&&panel.x+panel.width<=size.width+.1&&panel.y>=0&&panel.y+panel.height<=size.height+.1);
 await page.screenshot({path:path.join(__dirname,'qa',`flow_panel_${size.width}.png`)});await page.keyboard.press('Escape');
}
assert.deepEqual(errors,[]);assert.ok(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')));
const report={passed:true,steps:5,cacheWrites:3,queryGlobalLocalFilters:true,allCalculationModesPreserved:true,geometryPreserved:true,offline:true,errors};
fs.writeFileSync(path.join(__dirname,'qa','flow_validation.json'),JSON.stringify(report,null,2));console.log(report);await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
