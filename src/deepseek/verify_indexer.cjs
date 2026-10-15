const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),out=path.join(__dirname,'qa');
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 // The extension must preserve the complete calculation source and original SVG.
 for(const [file,hash] of Object.entries(JSON.parse(fs.readFileSync(path.join(__dirname,'step4_preservation.json')))))
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,file))).digest('hex'),hash,file);
 const browser=await chromium.launch(require('./browser_options.cjs')());
 try{
 const context=await browser.newContext({viewport:{width:1920,height:1080},offline:true});
 const page=await context.newPage(),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file:///'+path.join(root,'deepseek_v4_1_interactive.html').replaceAll('\\','/')+'#architecture');
 await page.evaluate(()=>document.fonts.ready);
 const data=await page.evaluate(()=>JSON.stringify(architectureApp.data));
 // Existing information-flow highlighting leaves empty style attributes behind.
 const originalMarkup=()=>page.locator('#diagram-content').evaluate(n=>{const copy=n.cloneNode(true);copy.querySelectorAll('[style=""]').forEach(e=>e.removeAttribute('style'));return copy.innerHTML;});
 const original=await originalMarkup();
 const config=JSON.parse(fs.readFileSync(path.join(root,'references/deepseek/config.json'))).text_config;
 assert.deepEqual(await page.evaluate(()=>indexerTeaching.config.kv_source_layers),config.kv_source_layer_ids);
 assert.deepEqual(await page.evaluate(()=>indexerTeaching.config.index_source_layers),config.index_source_layer_ids);
 assert.equal(await page.evaluate(()=>indexerTeaching.config.window_size),128);
 assert.equal(await page.evaluate(()=>indexerTeaching.config.index_topk),512);
 await page.screenshot({path:path.join(out,'indexer_overview.png')});
 // Real pointer clicks from both attention detail areas.
 for(const [area,entry,layer] of [['c1','encoder',2],['c2','decoder',20]]){
  await page.locator('#region-select').selectOption(area);
  await page.locator(`[data-indexer-entry="${entry}"] .indexer-module`).click();
  assert.equal(await page.locator('#indexer-detail').isVisible(),true);
  assert.equal((await page.evaluate(()=>indexerTeaching.state)).layer,layer);
  assert.equal((await page.evaluate(()=>architectureApp.state)).selected,'indexer');
  assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'false');
  await page.locator('#close-panel').click();await page.locator('#indexer-close').click();
  assert.equal(await page.locator('#region-select').inputValue(),area,'restore original focus');
 }
 await page.locator('#region-select').selectOption('indexer');
 for(const kind of ['Full','Reindex','Reuse']){
  await page.locator(`[data-indexer-mode="${kind}"]`).click();
  const state=await page.evaluate(()=>indexerTeaching.state);assert.equal(state.mode,kind);
  assert.equal(state.kvSource,20);assert.equal(state.indexSource,kind==='Reindex'?24:20);
  for(const [node,skipped] of [['compressor',kind!=='Full'],['key',kind!=='Full'],['query',kind==='Reuse'],['score',kind==='Reuse'],['main',false],['local',false],['gather',false]])
   assert.equal(await page.locator(`[data-indexer-node="${node}"]`).evaluate(n=>n.classList.contains('indexer-muted')),skipped,kind+' '+node);
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='0');
  await page.screenshot({path:path.join(out,`indexer_${kind.toLowerCase()}.png`)});
  for(const mode of ['parameter','forward','backward']){
   await page.locator('#indexer-calculate').click();await page.locator(`[data-mode="${mode}"]`).click();
   assert.equal((await page.evaluate(()=>architectureApp.state)).selected,'indexer');
   const visible=await page.locator('#panel-body').innerText();
   const rows=await page.evaluate(mode=>architectureApp.data.indexer[mode].rows,mode);
   for(const row of rows)assert.ok(visible.includes(row[1]),kind+' '+mode+' preserves canonical values');
   assert.ok(await page.locator('#panel-body math').count()>0);
   assert.equal(await page.locator('#panel-body [data-related="selection"]').count(),0,'no MoE Top-K link');
   assert.equal(await page.locator('#panel-body [data-related="global"]').count(),1);
   if(mode==='backward')assert.ok(visible.includes('Unknown'));
   await page.locator('#close-panel').click();
  }
 }
 // Every real layer, including full sources, reindex sources and all reuse layers.
 for(let layer=2;layer<40;layer++){
  await page.locator('#indexer-layer').selectOption(String(layer));
  const expected=config.kv_source_layer_ids.includes(layer)?'Full':config.index_source_layer_ids.includes(layer)?'Reindex':'Reuse';
  assert.equal((await page.evaluate(()=>indexerTeaching.state)).mode,expected);
 }
 await page.locator('[data-indexer-mode="Full"]').click();
 // All visible stages are interactive and resolve to the existing calculation record.
 for(const node of ['query','compressor','key','keycache','score','weights','aggregate','topk','globalcache','gather','mainquery','local','main']){
  const locator=page.locator(`[data-indexer-node="${node}"] .indexer-module`);
  const key=await locator.getAttribute('data-module');await locator.click();
  assert.equal((await page.evaluate(()=>architectureApp.state)).selected,key,node);
  await page.locator('#close-panel').click();
 }
 await page.locator('[data-indexer-node="query"] .indexer-module').focus();await page.keyboard.press('Enter');
 assert.equal((await page.evaluate(()=>architectureApp.state)).selected,'indexer');
 // Blank space closes the panel while leaving the expanded detail available.
 const blank=await page.locator('#architecture').evaluate(n=>{const p=n.createSVGPoint();p.x=680;p.y=125;const q=p.matrixTransform(n.getScreenCTM());return{x:q.x,y:q.y};});
 await page.mouse.click(blank.x,blank.y);assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true');
 assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,true);
 let before=await page.evaluate(()=>architectureApp.state.view);
 await page.mouse.move(700,700);await page.mouse.wheel(0,-250);
 assert.ok((await page.evaluate(()=>architectureApp.state.view.w))<before.w,'detail zoom');
 before=await page.evaluate(()=>architectureApp.state.view);
 await page.mouse.move(700,700);await page.mouse.down();await page.mouse.move(790,740,{steps:8});await page.mouse.up();
 assert.notDeepEqual(await page.evaluate(()=>architectureApp.state.view),before,'detail pan');
 await page.locator('#reset').click();assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,false);
 assert.equal((await page.evaluate(()=>architectureApp.state.view.w)),1672);
 await page.locator('#region-select').selectOption('indexer');await page.locator('#flow-toggle').click();
 assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,false);
 assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,true);
 for(let i=0;i<3;i++)await page.locator('#flow-next').click();
 assert.equal((await page.evaluate(()=>attentionTeaching.state)).active,true);
 await page.locator('#region-select').selectOption('indexer');
 assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,false);
 assert.equal((await page.evaluate(()=>attentionTeaching.state)).active,false);
 assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,true);
 await page.locator('[data-indexer-node="globalcache"] .indexer-module').click();
 await page.locator('.flow-entry').click();
 assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,false,'panel shortcut exits Indexer');
 assert.equal((await page.evaluate(()=>informationFlow.state)).step,2,'Global KV shortcut preserves its destination');
 await page.keyboard.press('Escape');await page.locator('#region-select').selectOption('indexer');
 for(const viewport of [{width:1366,height:768},{width:768,height:1024},{width:390,height:844}]){
  await page.setViewportSize(viewport);
  await page.waitForFunction(()=>{const b=document.getElementById('indexer-bar').getBoundingClientRect(),s=document.getElementById('stage').getBoundingClientRect();return s.y>=b.bottom;});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  assert.ok((await page.locator('#stage').boundingBox()).height>200);
  await page.locator('#indexer-boundary summary').click();
  await page.waitForFunction(()=>{const b=document.getElementById('indexer-bar').getBoundingClientRect(),s=document.getElementById('stage').getBoundingClientRect();return s.y>=b.bottom&&s.height>200;});
  await page.locator('#indexer-boundary summary').click();
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='0');
  await page.screenshot({path:path.join(out,`indexer_${viewport.width}.png`)});
  await page.locator('#indexer-calculate').click();
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='1');
  const r=await page.locator('#inspector').boundingBox();assert.ok(r.x>=0&&r.x+r.width<=viewport.width+.1&&r.y>=0&&r.y+r.height<=viewport.height+.1);
  await page.screenshot({path:path.join(out,`indexer_panel_${viewport.width}.png`)});
  await page.keyboard.press('Escape');assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,true);
 }
 await page.locator('#indexer-close').click();
 assert.equal(await page.evaluate(()=>JSON.stringify(architectureApp.data)),data,'canonical calculations untouched');
 // Ignore existing fitLabels adjustments caused solely by responsive resizing.
 await page.setViewportSize({width:1920,height:1080});await page.evaluate(()=>fitLabels());
 assert.equal(await originalMarkup(),original,'original diagram unchanged');
 assert.deepEqual(errors,[]);assert.ok(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')));
 const report={passed:true,entryPoints:2,clickableStages:13,layerModesVerified:38,calculationModes:3,calculationDataUnchanged:true,originalDiagramUnchanged:true,cacheReuse:true,blankDismiss:true,zoomPan:true,flowAndAttentionDemo:true,panelFlowShortcut:true,expandableNotes:true,viewports:[1920,1366,768,390],offline:true,errors};
 fs.writeFileSync(path.join(out,'indexer_validation.json'),JSON.stringify(report,null,2));console.log(report);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
