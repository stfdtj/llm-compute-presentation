const {chromium}=require('playwright'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const out=path.join(__dirname,'qa/pipeline');fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch(require('./browser_options.cjs')());
 try{
 const page=await browser.newPage({offline:true}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(require('node:url').pathToFileURL(path.resolve(__dirname,'../../deepseek_v4_1_interactive.html')).href);
 await page.locator('#csa-tab-pipeline').click();
 const scenes=[];
 for(const [width,height] of [[1920,1080],[1366,768],[768,900],[390,844]]){
  await page.setViewportSize({width,height});await page.locator('#ip-reset').click();
  assert.equal(await page.evaluate(()=>inferencePipeline.actions.length),6);
  await page.evaluate(()=>window.savedPipelineNode=document.querySelector('[data-flow-node="draft"]'));
  let lastEdges='';
  for(let step=0;step<6;step++){
   assert.equal(await page.evaluate(()=>inferencePipeline.state.step),step);
   assert.equal(await page.locator('#ip-simple section').count(),3);
   assert.equal(await page.locator('#ip-advanced').getAttribute('open'),null);
   assert.equal(await page.locator('#ip-probability').isVisible(),false);
   assert.equal(await page.locator('#ip-cache').isVisible(),false);
   const edges=await page.locator('.ip-wire.ip-hot').evaluateAll(ns=>ns.map(n=>n.dataset.flowEdge).join(','));assert.ok(edges&&edges!==lastEdges);lastEdges=edges;
   assert.equal(await page.evaluate(()=>window.savedPipelineNode===document.querySelector('[data-flow-node="draft"]')),true);
   const bounds=await page.locator('#ip-simple').boundingBox();assert.ok(bounds.x>=0&&bounds.x+bounds.width<=width+1);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   await page.screenshot({path:path.join(out,`lesson_${width}_${step+1}.png`)});scenes.push({width,height,step:step+1});
   if(step<5)await page.locator('#ip-next').click();
  }
  assert.deepEqual(await page.evaluate(()=>inferencePipeline.state.confirmed),['我','喜欢','先','A','B','X']);
 }
 await page.setViewportSize({width:1366,height:768});
 await page.locator('#ip-advanced>summary').click();
 for(const algorithm of ['greedy','probabilistic'])for(const scenario of ['middle','all','first','last']){
  await page.locator('#ip-algorithm').selectOption(algorithm);await page.locator('#ip-scenario').selectOption(scenario);
  const s=await page.evaluate(()=>inferencePipeline.state);assert.deepEqual(s.confirmed,s.result.confirmed);assert.ok(await page.locator('#ip-body').innerText());
 }
 await page.locator('#ip-reset').click();await page.locator('#ip-progress button').nth(4).click();
 assert.equal(await page.locator('[data-ip-alignment="2"] td').first().innerText(),'B');
 for(let i=0;i<5;i++){
  await page.locator(`[data-ip-alignment="${i}"]`).click();
  assert.equal(await page.locator('[data-inspected-input]').innerText(),['先','A','B','C','D'][i]);
 }
 assert.equal(await page.locator('.ip-mask tbody tr').count(),6);
 assert.equal(await page.locator('.ip-mask tbody tr').first().locator('.ip-blocked').count(),5);
 await page.locator('#ip-advanced>summary').click();await page.locator('#ip-reset').click();
 await page.locator('#ip-play').click();await page.waitForTimeout(3800);assert.equal(await page.evaluate(()=>inferencePipeline.state.step),1);
 await page.locator('#csa-tab-architecture').click();assert.equal(await page.evaluate(()=>inferencePipeline.state.playing),false);
 assert.deepEqual(errors,[]);const report={passed:true,steps:6,scenes,algorithms:8,persistentGraph:true,errors};fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(report,null,2));console.log(report);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
