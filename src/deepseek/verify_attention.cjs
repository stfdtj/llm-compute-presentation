const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{
 const browser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({viewport:{width:1920,height:1080},offline:true});
 const page=await context.newPage(),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.resolve(__dirname,'../../deepseek_v4_1_interactive.html')).href+'#architecture');
 await page.evaluate(()=>document.fonts.ready);
 const config=JSON.parse(fs.readFileSync(path.join(__dirname,'../../references/deepseek/inference/config.json'),'utf8'));
 // All shared stages exist on both sides; the serial query has no input bypass.
 for(const side of ['encoder','decoder']){
  const root=page.locator(`#${side}-attention-overview`);
  for(const key of ['qa','qnorm','qb','kv','kvnorm','localstates','output','streams'])assert.ok(await root.locator(`[data-module="${key}"]`).count(),side+': '+key);
  const paths=await root.locator('path.connection').evaluateAll(ns=>ns.map(n=>n.getAttribute('d')));
  const qx=side==='encoder'?88.5:526.5;
  for(const y of [701,725,749])assert.ok(paths.includes(`M${qx} ${y} v3`));
  const text=await root.textContent();assert.ok(text.includes('Inverse RoPE'));assert.ok(text.includes('W_OA'));assert.ok(text.includes('W_OB'));
 }
 await page.locator('#region-select').selectOption('attention');
 await page.screenshot({path:path.join(__dirname,'qa/attention_pair_1920.png')});
 for(const w of [1366,1920]){
  await page.setViewportSize({width:w,height:w===1366?900:1080});
  await page.locator('#reset').click();
  await page.screenshot({path:path.join(__dirname,`qa/attention_overview_${w}.png`)});
  await page.locator('#region-select').selectOption('attention');
  await page.screenshot({path:path.join(__dirname,`qa/attention_pair_${w}.png`)});
 }
 await page.locator('#attention-compare-launch').click();
 let checked=0;
 for(let n=0;n<config.n_layers;n++){
  await page.locator(n<20?'#ac-encoder-layer':'#ac-decoder-layer').selectOption(String(n));
  const col=n<20?0:1;
  const globalText=await page.locator('#ac-grid > .ac-global').nth(col).innerText();
  const m=n<2?'SWA':config.kv_source_layers.includes(n)?'Full':config.index_source_layers.includes(n)?'Reindex':'Reuse';
  assert.ok(globalText.includes(m),`L${n} ${m}`);
  if(m==='Full'){
   assert.ok(globalText.includes(`W_KV(global) [${config.dim},${config.head_dim}]`));
   assert.ok(globalText.includes(n<20?'2 个 token':'不运行 W_gate'));
  }else if(m==='Reuse'||m==='Reindex'){
   const src=config.kv_source_layers.filter(i=>i<=n).at(-1);
   assert.ok(globalText.includes(`读取 L${src} Global KV`));
   assert.ok(globalText.includes(m==='Reuse'?'跳过 Indexer':'本层 qᵣ → Indexer Query'));
  }
  checked++;
 }
 await page.locator('#ac-encoder-layer').selectOption('2');await page.locator('#ac-decoder-layer').selectOption('20');
 await page.screenshot({path:path.join(__dirname,'qa/attention_comparison_1920.png')});
 await page.locator('#attention-comparison').evaluate(n=>n.scrollTop=n.scrollHeight);
 await page.screenshot({path:path.join(__dirname,'qa/attention_comparison_output.png')});
 await page.locator('#ac-close').click();
 // Existing calculation panels and detail views still work after a modal transition.
 await page.locator('#attention-compare-launch').click();
 await page.locator('[data-ac-calc="output"]').first().click();
 assert.equal((await page.evaluate(()=>architectureApp.state)).selected,'output');
 await page.keyboard.press('Escape');
 for(const side of ['encoder','decoder']){
  await page.locator(`[data-indexer-entry="${side}"]`).click();
  assert.equal(await page.locator('#indexer-bar').isVisible(),true);
  await page.keyboard.press('Escape');
  await page.locator('#indexer-close').click();
 }
 await page.locator('#flow-toggle').click();
 for(let i=0;i<3;i++)await page.locator('#flow-next').click();
 await page.screenshot({path:path.join(__dirname,'qa/attention_flow_updated.png')});
 await page.locator('#flow-toggle').click();
 // Audit text stays inside the authorized frames at real browser font metrics.
 const overflows=await page.locator('[data-attention-side] .module').evaluateAll(nodes=>nodes.flatMap(n=>{
  const rect=n.querySelector(':scope > rect');if(!rect)return[];
  const b=rect.getBBox();return [...n.querySelectorAll('text')].flatMap(t=>{const r=t.getBBox();return r.x<b.x-1||r.x+r.width>b.x+b.width+1||r.y<b.y-2||r.y+r.height>b.y+b.height+2?[{key:n.dataset.module,text:t.textContent,box:{x:b.x,y:b.y,w:b.width,h:b.height},textBox:{x:r.x,y:r.y,w:r.width,h:r.height}}]:[];});
 }));
 assert.deepEqual(overflows,[],'text must fit in attention nodes');
 assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(__dirname,'qa/attention_validation.json'),JSON.stringify({passed:true,checkedLayers:checked,viewports:[1366,1920],errors,overflows,scope:'Browser / source-configuration verification; no model weights or GPU kernel execution'},null,2));
 await browser.close();console.log('PASS: aligned Attention diagrams, 40 layer modes, cache sources, existing entry points, text bounds, and browser screenshots.');
})().catch(e=>{console.error(e);process.exit(1)});
