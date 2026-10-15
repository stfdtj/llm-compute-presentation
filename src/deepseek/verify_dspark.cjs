const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{
 const browser=await chromium.launch(require('./browser_options.cjs')());
 try{
 const context=await browser.newContext({viewport:{width:1920,height:1080},offline:true}),page=await context.newPage(),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.resolve(__dirname,'../../deepseek_v4_1_interactive.html')).href);await page.evaluate(()=>document.fonts.ready);
 assert.equal((await page.evaluate(()=>dsparkArchitecture.state)).active,true);
 const c=JSON.parse(fs.readFileSync(path.join(__dirname,'../../references/deepseek/inference/config.json'),'utf8'));
 assert.equal(c.n_layers,40);assert.equal(c.n_mtp_layers,3);assert.equal(c.dspark_block_size,5);assert.deepEqual(c.dspark_target_layer_ids,[37,38,39]);
 const original=await page.locator('#diagram-content').evaluate(n=>n.innerHTML);
 const text=await page.locator('#dspark-overview').textContent();
 for(const t of ['Draft Block 1','Draft Block 2','Draft Block 3','Main Model Verification','Speculative Tokens','Accepted Tokens','Markov Head','Confidence Head','main_x','6 IDs'])assert.ok(text.includes(t),t);
 for(const width of [1920,1366]){
  await page.setViewportSize({width,height:width===1920?1080:768});
  await page.screenshot({path:path.join(__dirname,`qa/dspark_overview_${width}.png`)});
 }
 await page.locator('[data-dspark-stage="inputs"] .dspark-node').first().click();assert.equal(await page.locator('#dspark-dialog').isVisible(),true);
 for(const name of ['inputs','blocks','attention','heads','prefill','dimensions','verification']){
  await page.locator(`[data-ds-tab="${name}"]`).click();assert.equal((await page.evaluate(()=>dsparkArchitecture.state)).tab,name);
  const body=await page.locator('#ds-content').innerText();
  if(name==='inputs'){assert.ok(body.includes('mean(dim=2)'));assert.ok(body.includes('15360'));assert.ok(body.includes('尚未执行'));}
  if(name==='blocks'){assert.equal(await page.locator('.ds-stage-grid > span').count(),20);assert.ok(body.includes('128 routed 中选 3'));}
  if(name==='attention'){assert.ok(body.includes('块内双向'));assert.ok(body.includes('compress_ratio=0'));}
  if(name==='heads'){
   assert.ok(body.includes('5376'));assert.ok(body.includes('原始分数'));assert.ok(body.includes('未执行 sigmoid'));
   for(let n=1;n<=5;n++){assert.ok((await page.locator('#ds-markov-view').innerText()).includes('sample c'+n));if(n<5)await page.locator('#ds-markov-next').click();}
  }
  if(name==='prefill'){assert.ok(body.includes('None'));assert.ok(body.includes('跳过'));}
  if(name==='dimensions'){for(const shape of ['[1,15360]','[15360,5120]','[5376,1]','[256,129280]','[8192,5120]'])assert.ok(body.includes(shape),shape);}
  if(name==='verification'){
   assert.ok(body.includes('min(1,pᵢ(cᵢ)/qᵢ(cᵢ))'));assert.ok(body.includes('未在 forward_spec() 中实现'));
   for(const m of [0,1,5]){await page.locator('#ds-verify-length').fill(String(m));const chips=page.locator('#ds-verification-demo .ds-position');assert.equal(await chips.count(),5);assert.equal(await chips.filter({hasText:'未送验'}).count(),5-m);}
  }
  if(['inputs','blocks','heads','verification'].includes(name))await page.screenshot({path:path.join(__dirname,`qa/dspark_detail_${name}.png`)});
 }
 await page.keyboard.press('Escape');assert.equal(await page.locator('#dspark-dialog').isVisible(),false);
 const overflows=await page.locator('#dspark-overview .dspark-node').evaluateAll(nodes=>nodes.flatMap(n=>{const r=n.querySelector(':scope>rect');if(!r)return[];const b=r.getBBox();return [...n.querySelectorAll('text')].flatMap(t=>{const a=t.getBBox();return a.x<b.x-1||a.x+a.width>b.x+b.width+1||a.y<b.y-2||a.y+a.height>b.y+b.height+2?[t.textContent]:[];});}));
 assert.deepEqual(overflows,[]);
 await page.locator('#region-select').selectOption('attention');assert.equal((await page.evaluate(()=>dsparkArchitecture.state)).active,false);
 assert.equal(await page.locator('#diagram-content').isVisible(),true);
 await page.locator('#dspark-launch').click();await page.locator('#flow-toggle').click();assert.equal((await page.evaluate(()=>dsparkArchitecture.state)).active,false);assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,true);await page.locator('#flow-toggle').click();
 await page.locator('#region-select').selectOption('hc');assert.equal((await page.evaluate(()=>hcTeaching.state)).active,true);await page.locator('#dspark-launch').click();assert.equal((await page.evaluate(()=>hcTeaching.state)).active,false);
 await page.locator('#region-select').selectOption('indexer');assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,true);await page.locator('#dspark-launch').click();
 await page.locator('#reset').click();assert.equal((await page.evaluate(()=>dsparkArchitecture.state)).active,false);
 const final=await page.locator('#diagram-content').evaluate(n=>n.innerHTML.replaceAll(' style=""',''));assert.equal(final,original.replaceAll(' style=""',''));
 assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(__dirname,'qa/dspark_validation.json'),JSON.stringify({passed:true,viewports:[1920,1366],draftLayers:3,draftPositions:5,targets:[37,38,39],detailTabs:7,markovSteps:5,verificationPrefixLengths:[0,1,5],textOverflows:overflows,originalDiagramPreserved:true,errors,scope:'Actual Chrome page / interaction tests; mathematical source audit, not GPU execution'},null,2));
 console.log('PASS: DSpark overview, seven detail tabs, source dimensions, five Markov steps, candidate/accepted distinction, text containment and existing view transitions.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
