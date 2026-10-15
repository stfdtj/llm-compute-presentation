const {chromium}=require('playwright');
const assert=require('assert/strict');
const fs=require('fs');
const path=require('path');
const out=path.join(__dirname,'qa');fs.mkdirSync(out,{recursive:true});
const file='file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/');
(async()=>{
 const browser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({viewport:{width:1920,height:1080},deviceScaleFactor:1,offline:true});
 const page=await context.newPage(),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('request',r=>requests.push(r.url()));
 await page.goto(file);await page.evaluate(()=>document.fonts.ready);
 const initialSVG=await page.locator('#architecture').evaluate(n=>n.innerHTML);
 assert.equal(await page.locator('svg image,canvas').count(),0,'diagram is vector, not raster overlay');
 const keys=await page.locator('.module').evaluateAll(ns=>[...new Set(ns.map(n=>n.dataset.module))]);
 let panels=0;
 for(const key of keys){
  await page.locator(`.module[data-module="${key}"]`).first().dispatchEvent('keydown',{key:'Enter'});
  for(const mode of ['parameter','forward','backward']){
   await page.locator(`[data-mode="${mode}"]`).click();
   assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'false');
   const state=await page.evaluate(()=>architectureApp.state);assert.equal(state.selected,key);assert.equal(state.mode,mode);
   const d=await page.evaluate(({key,mode})=>architectureApp.data[key][mode],{key,mode});
   assert.ok(d.rows.length>0);
   const visible=await page.locator('#panel-body').innerText();
   for(const row of d.rows)assert.ok(visible.includes(row[1]));
   assert.ok(d.steps.every(s=>s.formulas.length&&s.dimensions.length&&s.symbols.length));
   assert.ok(!/Scenario P|Source Verified|Mathematical Estimate|F_B|F_E/.test(visible),key+' '+mode+' obsolete teaching text');
   assert.ok(await page.locator('#panel-body math').count()>0);
   panels++;
  }
  await page.keyboard.press('Escape');
 }
 // The only extra entries are panel-only details of existing architecture boxes.
 for(const [parent,child] of [['sparse','compressor'],['sparse','indexer'],['expert','swiglu']]){
  for(const mode of ['parameter','forward','backward']){
   await page.locator(`.module[data-module="${parent}"]`).first().dispatchEvent('keydown',{key:'Enter'});
   await page.locator(`[data-related="${child}"]`).click();
   await page.locator(`[data-mode="${mode}"]`).click();
   assert.equal((await page.evaluate(()=>architectureApp.state)).selected,child);panels++;
   await page.keyboard.press('Escape');
  }
 }
 // Physical pointer clicks from every area in all modes.
 const samples=[['embedding',0],['query',0],['kv',0],['sliding',0],['multi',0],['router',0],['expert',0],['hc',0],['enggate',0],['head',0]];
 for(const [key,index] of samples){for(const mode of ['parameter','forward','backward']){
  await page.locator(`[data-mode="${mode}"]`).click();
  await page.locator(`.module[data-module="${key}"]`).nth(index).click(key==='query'||key==='kv'?{position:{x:35,y:10}}:{});
  assert.equal((await page.evaluate(()=>architectureApp.state)).selected,key);
  const r=await page.locator('#inspector').boundingBox();assert.ok(r.x>=0&&r.y>=0&&r.x+r.width<=1920&&r.y+r.height<=1080);
  await page.keyboard.press('Escape');
 }}
 await page.locator('.module[data-module="embedding"]').click();
 await page.locator('#panel-title').click();assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'false','inside stays open');
 await page.mouse.click(25,990);assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true','blank closes');
 await page.locator('.module[data-module="embedding"]').click();await page.locator('.brand').click();assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true','outside closes');
 await page.locator('.module[data-module="embedding"]').click();await page.keyboard.press('Escape');assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true');
 await page.mouse.move(900,760);await page.mouse.wheel(0,-350);
 assert.ok((await page.evaluate(()=>architectureApp.state.view.w))<1672,'wheel zoom');
 let before=await page.evaluate(()=>architectureApp.state.view);
 await page.mouse.move(600,690);await page.mouse.down();await page.mouse.move(700,750,{steps:10});await page.mouse.up();
 let after=await page.evaluate(()=>architectureApp.state.view);assert.ok(before.x!==after.x||before.y!==after.y,'drag pan');
 await page.locator('#reset').click();assert.equal((await page.evaluate(()=>architectureApp.state.view.w)),1672);
 for(const region of ['a','b','c1','c2','c3','d']){await page.locator('#region-select').selectOption(region);assert.ok((await page.evaluate(()=>architectureApp.state.view.w))>0);}
 await page.locator('#reset').click();await page.locator('[data-mode="parameter"]').click();
 const finalSVG=await page.locator('#architecture').evaluate(n=>n.innerHTML);assert.equal(finalSVG,initialSVG,'all modes preserve diagram geometry and labels');
 await page.screenshot({path:path.join(out,'overview_1920.png')});
 // Side-by-side comparison at the exact reference aspect ratio, in a browser.
 const compare=await context.newPage();await compare.goto(file);
 const svgMarkup=await compare.locator('#architecture').evaluate(n=>n.outerHTML);
 await compare.setContent(`<html><head><style>body{margin:0;background:#e9eef5;font-family:Arial}main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}h2{font-size:14px;margin:4px 0 8px;color:#344467}.item{min-width:0}img,svg{display:block;width:100%;height:auto;background:white}svg .hit-label{fill:transparent}</style></head><body><main><div class="item"><h2>REFERENCE · supplied image</h2><img src="${require('node:url').pathToFileURL(path.join(__dirname,'qa/reference.png')).href}"></div><div class="item"><h2>RECONSTRUCTION · actual SVG in HTML</h2>${svgMarkup}</div></main></body></html>`);
 await compare.setViewportSize({width:2400,height:725});
 await compare.locator('img').evaluate(img=>img.decode());
 await compare.screenshot({path:path.join(out,'reference_comparison.png'),fullPage:true});await compare.close();
 await page.locator('.module[data-module="query"]').first().dispatchEvent('keydown',{key:'Enter'});await page.locator('[data-mode="backward"]').click();
 await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='1');
 await page.screenshot({path:path.join(out,'panel_backward.png')});await page.keyboard.press('Escape');
 for(const mode of ['parameter','forward','backward']){
  await page.locator('.module[data-module="kv"]').first().dispatchEvent('keydown',{key:'Enter'});
  await page.locator(`[data-mode="${mode}"]`).click();
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='1');
  await page.screenshot({path:path.join(out,`shared_kv_${mode}.png`)});
  await page.keyboard.press('Escape');
 }
 await page.locator('.module[data-module="sparse"]').first().dispatchEvent('keydown',{key:'Enter'});
 await page.locator('[data-related="compressor"]').click();
 await page.locator('[data-mode="forward"]').click();
 assert.equal(await page.locator('#derivation-scenario').inputValue(),'A');
 assert.ok((await page.locator('#derivation-walkthrough').innerText()).includes('等待配对'));
 await page.locator('#derivation-scenario').selectOption('B');
 assert.ok((await page.locator('#derivation-walkthrough').innerText()).includes('池化'));
 await page.locator('.derivation-step').first().scrollIntoViewIfNeeded();
 await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='1');
 await page.screenshot({path:path.join(out,'compressor_examples.png')});
 await page.keyboard.press('Escape');
 const sizes=[{width:1366,height:768},{width:768,height:1024},{width:390,height:844}];
 for(const viewport of sizes){await page.setViewportSize(viewport);await page.locator('#reset').click();await page.locator('.module[data-module="kv"]').first().dispatchEvent('keydown',{key:'Enter'});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'no document overflow');
  const r=await page.locator('#inspector').boundingBox();assert.ok(r.x>=0&&r.y>=0&&r.x+r.width<=viewport.width+.1&&r.y+r.height<=viewport.height+.1);
  await page.waitForFunction(()=>getComputedStyle(document.getElementById('inspector')).opacity==='1');
  await page.screenshot({path:path.join(out,`responsive_${viewport.width}.png`)});await page.keyboard.press('Escape');
 }
 assert.deepEqual(errors,[],'no JS or console errors');assert.ok(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')),'offline, no network dependencies');
 const report={passed:true,viewport:'1920x1080',directModuleKeys:keys.length,calculationModesChecked:panels,physicalClicksAcrossAreas:samples.length*3,checks:['all direct module mappings in 3 modes','panel-only compressor/indexer/SwiGLU in 3 modes','blank and outside close','Escape close','inside click remains open','mode switch updates existing panel','wheel zoom','drag pan','reset and region focus','diagram identical across modes','offline local file','MathML equations','responsive panel containment','no JS or console errors','no raster diagram'],responsiveSizes:sizes,errors,networkRequests:requests.filter(u=>!u.startsWith('file:')),screenshots:['overview_1920.png','reference_comparison.png','panel_backward.png']};
 fs.writeFileSync(path.join(out,'browser_validation.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
