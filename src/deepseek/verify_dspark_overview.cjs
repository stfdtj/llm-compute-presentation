const {chromium}=require('playwright'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const out=path.join(__dirname,'qa/dspark_overview');
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch(require('./browser_options.cjs')());
 try{
 const context=await browser.newContext({offline:true}),page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(require('node:url').pathToFileURL(path.resolve(__dirname,'../../deepseek_v4_1_interactive.html')).href);
 assert.equal(await page.locator('#dspark-overview').count(),1);
 for(const key of ['anchor','features','input','blocks','head','markov','confidence','verification','implementation'])assert.ok(await page.locator(`[data-dspark-topic="${key}"]`).count());
 const connections=await page.locator('#dspark-overview path.connection').count();assert.equal(connections,10);
 for(const key of ['anchor','features','hidden-confidence','markov-confidence','candidates-verification','scores-scheduler','verification-next-anchor'])assert.equal(await page.locator(`[data-dspark-link="${key}"] path`).count(),1);
 const cases=[];
 for(const [width,height] of [[1366,768],[1920,1080],[768,900],[390,844]]){
  await page.setViewportSize({width,height});await page.locator('#reset').click();
  await page.screenshot({path:path.join(out,`main_${width}.png`)});
  const layout=await page.evaluate(()=>{
   const root=document.querySelector('#dspark-overview'),boxes=[...root.querySelectorAll('.dspark-topic > rect.surface')].map(n=>n.getBBox());
   const overlaps=boxes.flatMap((a,i)=>boxes.slice(i+1).filter(b=>a.x<b.x+b.width&&b.x<a.x+a.width&&a.y<b.y+b.height&&b.y<a.y+a.height).map(()=>i));
   const clipped=[...root.querySelectorAll('.dspark-topic')].flatMap(g=>{const rect=g.querySelector('rect');if(!rect||!rect.classList.contains('surface'))return [];const b=rect.getBBox();return [...g.querySelectorAll('tspan')].filter(n=>{const t=n.getBBox();return t.x<b.x-1||t.x+t.width>b.x+b.width+1||t.y<b.y||t.y+t.height>b.y+b.height+1;}).map(n=>n.textContent);});
   return {overlaps,clipped,scroll:document.documentElement.scrollWidth>innerWidth};
  });assert.deepEqual(layout,{overlaps:[],clipped:[],scroll:false});cases.push({width,height});
  for(const key of ['anchor','features','input','markov','confidence','implementation']){
   await page.locator(`[data-dspark-topic="${key}"]`).first().dispatchEvent('keydown',{key:'Enter'});
   assert.equal(await page.locator('#dspark-card').isVisible(),true);
   assert.equal(await page.locator('#dspark-card details').getAttribute('open'),null);assert.equal(await page.locator('#dspark-card-body>p').count(),3);await page.locator('#dspark-card summary').click();const text=await page.locator('#dspark-card').innerText();assert.ok(text.includes('generate.py'));assert.ok(text.includes('forward_spec'));
   if(key==='anchor')assert.ok(text.includes('Final RMSNorm'));
   if(key==='input')assert.ok(text.includes('[Anchor, Noise, Noise, Noise, Noise]'));
   if(key==='markov')assert.ok(text.includes('前一 token'));
   if(key==='confidence')assert.ok(text.includes('score'));
   const b=await page.locator('#dspark-card').boundingBox();assert.ok(b.x>=0&&b.x+b.width<=width+1&&b.y+b.height<=height+1);
   if(width>=1366)await page.screenshot({path:path.join(out,`${key}_${width}.png`)});
   await page.keyboard.press('Escape');assert.equal(await page.locator('#dspark-card').isVisible(),false);
  }
 }
 await page.setViewportSize({width:1366,height:768});
 await page.locator('[data-dspark-topic="blocks"]').dispatchEvent('click');await page.locator('#dspark-card summary').click();await page.locator('[data-dspark-action="detail"]').click();assert.equal(await page.locator('#region-select').inputValue(),'draft');assert.equal(await page.locator('#draft-expanded').isVisible(),true);assert.equal(await page.locator('#dspark-card').isVisible(),false);
 await page.locator('#reset').click();await page.locator('[data-dspark-topic="verification"]').dispatchEvent('click');await page.locator('[data-dspark-action="pipeline"]').click();assert.equal(await page.locator('#inference-pipeline').isVisible(),true);assert.equal(await page.locator('#dspark-card').isVisible(),false);
 assert.equal(await page.evaluate(()=>inferencePipeline.state.stage),'anchor');assert.equal(await page.evaluate(()=>inferencePipeline.actions.length),6);
 await page.locator('#csa-tab-architecture').click();await page.locator('[data-dspark-topic="implementation"]').dispatchEvent('click');await page.locator('#csa-tab-walkthrough').click();assert.equal(await page.locator('#dspark-card').isVisible(),false);
 assert.deepEqual(errors,[]);const report={passed:true,connectedBranch:true,connections,popups:9,viewports:cases,backboneDecodeBeforeAnchor:true,publicGenerateCaveat:true,errors};fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(report,null,2));console.log(report);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
