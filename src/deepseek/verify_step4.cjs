const {chromium}=require('playwright');
const assert=require('assert/strict'),fs=require('fs'),path=require('path'),crypto=require('crypto');
(async()=>{
 const preservation=JSON.parse(fs.readFileSync(path.join(__dirname,'step4_preservation.json')));
 for(const [file,hash] of Object.entries(preservation))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(__dirname,'../..',file))).digest('hex'),hash,'Audit/base diagram changed: '+file);
 const browser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({offline:true}),page=await context.newPage(),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/')+'#architecture');
 const data=await page.locator('#module-data').textContent();
 const geometry=()=>page.locator('#diagram-content').evaluate(n=>n.innerHTML);
 // Compare physical geometry, since normal highlighting temporarily adds classes/styles.
 const rects=()=>page.locator('#diagram-content').evaluate(n=>[...n.querySelectorAll('rect,path')].map(e=>[e.tagName,...['x','y','width','height','d'].map(k=>e.getAttribute(k))]));
 const before=await rects();
 await page.locator('#flow-toggle').click();for(let i=0;i<3;i++)await page.locator('#flow-next').click();
 await page.waitForFunction(()=>attentionTeaching.state.active);
 const sizes=[[1920,1080],[1366,768],[768,1024],[390,844],[390,667]];
 for(const [width,height] of sizes){
  await page.setViewportSize({width,height});
  for(let i=0;i<6;i++){
   await page.locator('#attention-tab-'+i).click();assert.equal((await page.evaluate(()=>attentionTeaching.state)).phase,i+1);
   await page.locator('#attention-scene').evaluate(n=>Promise.all(n.getAnimations({subtree:true}).map(a=>a.finished)));
   const text=await page.locator('#attention-scene').innerText();
   const required=[['[5120,1280]','[1280,32768]','Layer 20'],['[1,1152,512]','dim=1'],['[1,640,512]','[1,1,640]'],['[512,640]','√512'],['[640,512]','同一分母','[1,512]'],['[1,8,4096]','[1,8,1024]','[8192,5120]']][i];
   for(const t of required)assert.ok(text.includes(t),'Missing '+t);
   const bounds=await page.evaluate(()=>({bar:document.getElementById('flow-bar').getBoundingClientRect().toJSON(),stage:document.querySelector('.stage').getBoundingClientRect().toJSON(),next:document.getElementById('attention-next').getBoundingClientRect().toJSON(),overflow:document.documentElement.scrollWidth>innerWidth}));
   assert.ok(bounds.stage.height>=150,JSON.stringify({width,height,i,bounds}));assert.ok(bounds.next.bottom<height);assert.equal(bounds.overflow,false);
   if(width===1366||width===390&&height===844)await page.screenshot({path:path.join(__dirname,'qa',`attention_${width}_step_${i+1}.png`)});
  }
 }
 await page.locator('#attention-tab-1').click();
 await page.locator('#attention-scene').evaluate(n=>Promise.all(n.getAnimations({subtree:true}).map(a=>a.finished)));
 const concat=await page.locator('#attention-scene .attention-card').last().locator('.matrix-band').evaluateAll(nodes=>nodes.map(n=>({top:n.getBoundingClientRect().top,width:n.getBoundingClientRect().width})));
 assert.equal(concat[0].width,concat[1].width);assert.ok(concat[1].top>concat[0].top);
 await page.locator('#attention-tab-4').click();
 const toyBefore=await page.evaluate(()=>attentionTeaching.toyResult());
 await page.locator('#attention-toy-query').evaluate(n=>{n.value='2';n.dispatchEvent(new Event('input',{bubbles:true}));});
 const toyAfter=await page.evaluate(()=>attentionTeaching.toyResult());assert.notDeepEqual(toyBefore.output,toyAfter.output);assert.ok(Math.abs(toyAfter.weights.reduce((s,x)=>s+x,0)+toyAfter.sink-1)<1e-12);
 await page.locator('#attention-prev').click();assert.equal((await page.evaluate(()=>attentionTeaching.state)).phase,4);
 await page.locator('#attention-next').click();assert.equal((await page.evaluate(()=>attentionTeaching.state)).phase,5);
 await page.locator('#attention-restart').click();assert.equal((await page.evaluate(()=>attentionTeaching.state)).phase,1);
 await page.locator('#attention-tab-0').focus();await page.keyboard.press('End');assert.equal((await page.evaluate(()=>attentionTeaching.state)).phase,6);
 await page.locator('#flow-next').click();assert.equal(await page.locator('#attention-demo').isVisible(),false);assert.equal(await page.locator('#flow-counter').innerText(),'5/5');
 await page.locator('#flow-prev').click();assert.equal((await page.evaluate(()=>attentionTeaching.state)).phase,1);
 await page.locator('#flow-toggle').click();assert.equal(await page.locator('#attention-demo').isVisible(),false);
 assert.equal(await page.locator('#module-data').textContent(),data);assert.deepEqual(await rects(),before);assert.deepEqual(errors,[]);assert.ok(requests.every(x=>x.startsWith('file:')));
 const report={passed:true,substeps:6,viewports:sizes,concatPositionAxis:true,jointSoftmaxWithSink:true,sharedLatentWeightedSum:true,groupedOutputProjection:true,offline:true,originalDiagramPreserved:true,auditHashesPreserved:Object.keys(preservation),errors};
 fs.writeFileSync(path.join(__dirname,'qa','step4_validation.json'),JSON.stringify(report,null,2));
 await browser.close();console.log('PASS: 6 substeps × 5 sizes; concat axes, sink/weighted sum, navigation, offline, diagram/audit preserved.');
})();
