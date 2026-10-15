const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch(require('./browser_options.cjs')());
 try{
 const context=await browser.newContext({viewport:{width:1920,height:1080},offline:true}),page=await context.newPage(),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/')+'#architecture');await page.evaluate(()=>document.fonts.ready);
 const canonical=await page.locator('#derivation-data').textContent();
 const overview=page.locator('#hc-overview');
 const inputs=await overview.locator('.module[data-module="streams"]').allTextContents();
 assert.equal(inputs.length,4);
 for(const [i,label] of ['h⁽¹⁾','h⁽²⁾','h⁽³⁾','h⁽⁴⁾'].entries()){assert.ok(inputs[i].includes(label));assert.ok(inputs[i].includes('[1,5120]'));}
 assert.ok(!(await overview.textContent()).includes('α₁'));
 await page.locator('#region-select').selectOption('c3');
 await page.screenshot({path:path.join(__dirname,'qa','hc_overview.png')});
 await page.locator('[data-hc-expand]').click();
 assert.equal((await page.evaluate(()=>hcTeaching.state)).active,true);
 assert.equal(await page.locator('#region-select').inputValue(),'hc');
 assert.equal(await page.locator('#hc-detail').isVisible(),true);
 await page.screenshot({path:path.join(__dirname,'qa','hc_detail.png')});
 const text=await page.locator('#hc-detail').textContent();
 for(const t of ['Pre_in = 4','Pre_next = 4','Post = 4','4×4 = 16','4 + 4 + 16 = 24','[1,20480]','[20480,24]','C [4,4]','C[i,j]','Pre = σ(a_pre) + ε','Post = 2σ(a_post)','base[24]','scale[3]'])assert.ok(text.includes(t),t);
 assert.equal(await page.locator('#hc-detail [data-hc-stage="streams"]').count(),4);
 // Check ALL text against its actual colored node; fitLabels must prevent overlaps.
 const layout=await page.locator('#hc-detail .hc-module').evaluateAll(nodes=>nodes.flatMap(node=>{
  const rect=node.querySelector('rect');if(!rect)return[];
  const r={x:+rect.getAttribute('x'),y:+rect.getAttribute('y'),w:+rect.getAttribute('width'),h:+rect.getAttribute('height')};
  return [...node.querySelectorAll('tspan')].flatMap(t=>{const b=t.getBBox();return b.x<r.x-1||b.x+b.width>r.x+r.w+1||b.y<r.y-1||b.y+b.height>r.y+r.h+1?[{text:t.textContent,box:r,textBox:{x:b.x,y:b.y,w:b.width,h:b.height}}]:[]});
 }));
 assert.deepEqual(layout,[],'all labels inside their nodes');
 const stageTitles={pre:['3 ·','pre 合并的输入'],prenext:['2b ·','2b⁻¹'],post:['4 ·','post 系数'],comb:['4 ·','comb 系数'],projection:['W_HC','W_HC · dW'],flatten:['1a ·','1a⁻¹'],transform:['2a ·','2a⁻¹'],submodule:['3 ·','子模块输出'],update:['4 ·','汇总']};
 for(const [which,titles] of Object.entries(stageTitles)){
  for(const [i,mode] of ['forward','backward'].entries()){
   await page.locator('[data-mode="'+mode+'"]').click();
   await page.locator('#hc-detail [data-hc-stage="'+which+'"]').first().click();
   assert.equal((await page.evaluate(()=>architectureApp.state)).selected,'hc');
   const current=await page.locator('.derivation-step[open] .derivation-title').innerText();assert.ok(current.includes(titles[i]),which+' '+mode+' '+current);
   await page.locator('#close-panel').click();
  }
 }
 await page.locator('#hc-detail [data-hc-stage="post"]').first().focus();await page.keyboard.press('Enter');
 assert.equal((await page.evaluate(()=>architectureApp.state)).selected,'hc');
 const blank=await page.locator('#architecture').evaluate(svg=>{const p=svg.createSVGPoint();p.x=900;p.y=560;const q=p.matrixTransform(svg.getScreenCTM());return{x:q.x,y:q.y};});
 await page.mouse.click(blank.x,blank.y);assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true');
 assert.equal((await page.evaluate(()=>hcTeaching.state)).active,true);
 let previous=await page.evaluate(()=>architectureApp.state.view);
 await page.mouse.move(700,600);await page.mouse.wheel(0,-200);assert.ok((await page.evaluate(()=>architectureApp.state.view.w))<previous.w);
 previous=await page.evaluate(()=>architectureApp.state.view);
 await page.mouse.move(700,600);await page.mouse.down();await page.mouse.move(770,655,{steps:6});await page.mouse.up();
 assert.notDeepEqual(await page.evaluate(()=>architectureApp.state.view),previous);
 await page.locator('#hc-close').click();assert.equal(await page.locator('#region-select').inputValue(),'c3');
 await page.locator('#region-select').selectOption('hc');await page.locator('#hc-close').click();assert.equal(await page.locator('#region-select').inputValue(),'c3','region-menu entry restores the same focus');
 await page.locator('#region-select').selectOption('hc');await page.locator('#reset').click();assert.equal((await page.evaluate(()=>hcTeaching.state)).active,false);
 await page.locator('#region-select').selectOption('hc');await page.locator('#flow-toggle').click();
 assert.equal((await page.evaluate(()=>hcTeaching.state)).active,false);assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,true);
 await page.locator('#region-select').selectOption('hc');assert.equal((await page.evaluate(()=>informationFlow.state)).enabled,false);
 await page.locator('#region-select').selectOption('indexer');assert.equal((await page.evaluate(()=>hcTeaching.state)).active,false);assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,true);
 await page.locator('#region-select').selectOption('hc');assert.equal((await page.evaluate(()=>indexerTeaching.state)).active,false);
 for(const viewport of [{width:1366,height:768},{width:768,height:1024},{width:390,height:844}]){
  await page.setViewportSize(viewport);
  await page.waitForFunction(()=>{const b=document.getElementById('hc-bar').getBoundingClientRect(),s=document.getElementById('stage').getBoundingClientRect();return s.y>=b.bottom&&s.height>200;});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:path.join(__dirname,'qa','hc_'+viewport.width+'.png')});
 }
 assert.equal(await page.locator('#derivation-data').textContent(),canonical,'existing equations/counts unchanged');
 assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith('file:')||r.startsWith('data:')));
 const report={passed:true,streams:4,coefficientCounts:[4,4,16],stagesInBothModes:9,canonicalLedgerUnchanged:true,labelContainment:true,zoomPan:true,keyboardAndBlankDismiss:true,flowAndIndexerTransitions:true,responsive:[1920,1366,768,390],offline:true,errors};
 fs.writeFileSync(path.join(__dirname,'qa','hc_validation.json'),JSON.stringify(report,null,2));console.log(report);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
