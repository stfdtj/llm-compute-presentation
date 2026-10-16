const {chromium}=require('playwright'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{const browser=await chromium.launch(require('./browser_options.cjs')());try{
 const context=await browser.newContext({offline:true}),page=await context.newPage(),errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 const out=path.join(__dirname,'qa/final_review');fs.mkdirSync(out,{recursive:true});
 const url=pathToFileURL(path.resolve(__dirname,'../../deepseek_v4_1_interactive.html')).href;
 await page.goto(url);const packed=await page.locator('#derivation-data').textContent();
 for(const [width,height] of [[1920,1080],[1366,768],[768,1024],[390,844]]){
  await page.setViewportSize({width,height});await page.evaluate(()=>learningWorkspace.show());
  assert.equal(await page.locator('#learning-workspace').evaluate(n=>n.scrollWidth<=n.clientWidth),true);
  const misplaced=await page.locator('.learn-arrow').evaluateAll(arrows=>arrows.filter(a=>{const p=a.previousElementSibling.getBoundingClientRect(),q=a.nextElementSibling.getBoundingClientRect(),r=a.getBoundingClientRect();return innerWidth>800?!(r.left>=p.right-1&&r.right<=q.left+1):!(r.top>=p.bottom-1&&r.bottom<=q.top+1);}).length);assert.equal(misplaced,0,'Arrows must connect adjacent nodes without wrap ambiguity');
  await page.screenshot({path:path.join(out,`after_${width}.png`)});
  const text=await page.locator('#learn-body').innerText();for(const t of ['L0–19','L20–39','HC Pre','[1,4,5120]','Reindex','3 是深度','论文','main_x'])assert.ok(text.includes(t),t);
  await page.locator('[data-draft="blocks"]').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,`branch_${width}.png`)});
  // Real keyboard activation, roving tabs and scrollable dialog at every width.
  await page.locator('[data-draft="blocks"]').focus();await page.keyboard.press('Enter');assert.ok(await page.locator('#dspark-dialog').isVisible());
  for(const tab of ['inputs','blocks','attention','heads','prefill','dimensions','verification']){
   await page.locator(`[data-ds-tab="${tab}"]`).click();
   const size=await page.locator('#dspark-dialog').evaluate(n=>({scroll:n.scrollWidth,width:n.clientWidth}));assert.ok(size.scroll<=size.width+2,`${width} ${tab} horizontal overflow ${JSON.stringify(size)}`);
   await page.locator('#dspark-dialog').evaluate(n=>n.scrollTop=n.scrollHeight);assert.ok(await page.locator('.ds-source').isVisible());
   if(tab==='dimensions'||tab==='verification')await page.screenshot({path:path.join(out,`dspark_${tab}_${width}.png`)});
  }
  await page.locator('[data-ds-tab="verification"]').focus();await page.keyboard.press('Home');assert.equal(await page.locator('[data-ds-tab="inputs"]').getAttribute('aria-selected'),'true');await page.keyboard.press('ArrowRight');assert.equal(await page.locator('[data-ds-tab="blocks"]').getAttribute('aria-selected'),'true');
  await page.keyboard.press('Escape');assert.equal(await page.locator('#dspark-dialog').isVisible(),false);
  await page.locator('[data-workspace="flow"]').click();for(const s of ['prefill','decode','draft']){await page.locator(`[data-scenario="${s}"]`).click();assert.equal((await page.evaluate(()=>learningWorkspace.state)).scenario,s);for(let i=0;i<3;i++)await page.locator('#learn-next').click();assert.equal(await page.locator('.learn-card.current').count(),1);await page.screenshot({path:path.join(out,`flow_${s}_${width}.png`)});await page.locator('#learn-reset').click();}
  await page.locator('[data-workspace="compute"]').first().click();await page.locator('#learn-search').fill('全局 KV 投影');await page.locator('[data-module-open="global"]').click();
  await page.locator('.learning-summary').waitFor({state:'visible'});assert.ok(await page.locator('.learning-summary').isVisible());assert.equal(await page.locator('.learning-audit').isVisible(),false);
  const summary=await page.locator('.learning-summary').innerText();for(const t of ['[1,5120]','[5120,512]','[1,512]','2,621,440','5,242,880'])assert.ok(summary.includes(t),t);
  assert.ok(!(await page.locator('#panel-body').innerText()).includes('???'),'No encoding damage in visible labels');
  await page.waitForTimeout(170);await page.screenshot({path:path.join(out,`compute_${width}.png`)});
  await page.locator('[data-learning-level="2"]').click();assert.ok(await page.locator('.learning-steps').isVisible());assert.ok(await page.locator('.learning-steps > section').count()<=6);
  await page.locator('[data-learning-step="0"]').click();assert.ok((await page.locator('.derivation-step').first().innerText()).includes('5,242,368'));
  await page.locator('[data-learning-level="1"]').click();await page.locator('[data-learning-mode="backward"]').click();assert.equal(await page.locator('#derivation-scenario').inputValue(),'B');assert.ok((await page.locator('.learning-summary').innerText()).includes('128'));
  await page.locator('#derivation-scenario').selectOption('A');assert.ok((await page.locator('.learning-summary').innerText()).includes('局部 VJP'));
  await page.locator('[data-learning-mode="parameter"]').click();assert.ok((await page.locator('.learning-summary').innerText()).includes('存储 / 共享权重'));
  await page.locator('[data-learning-level="2"]').click();await page.locator('[data-learning-step="0"]').click();assert.equal((await page.evaluate(()=>parameterAudit.state)).activeStep,0);assert.ok((await page.locator('#parameter-mini').innerText()).includes('W_KV'));
  await page.keyboard.press('Escape');assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true');
  // Reset independent defaults for the next viewport without mutating ledger data.
  await page.reload();
 }
 await page.setViewportSize({width:1366,height:768});
 for(const key of ['hc','moe','indexer','compressor','engram','decoderattn']){
  await page.evaluate(()=>learningWorkspace.show('compute'));await page.locator('#learn-search').fill('');await page.locator(`[data-module-open="${key}"]`).click();
  assert.equal(await page.locator('.learning-summary').count(),1);await page.locator('[data-learning-level="2"]').click();await page.locator('[data-learning-level="3"]').click();assert.ok(await page.locator('.learning-audit').isVisible());await page.keyboard.press('Escape');
 }
 let summaryCoverage=0;
 for(const key of await page.evaluate(()=>Object.keys(architectureApp.data))){
  await page.evaluate(k=>{learningWorkspace.show('compute');progressiveCompute.reset();openPanel(k);},key);
  for(const m of ['parameter','forward','backward']){
   await page.locator(`[data-learning-mode="${m}"]`).click();
   const summary=await page.locator('.learning-summary').innerText();assert.ok(!/NaN|undefined|\?\?\?/.test(summary),`${key}/${m}: invalid summary`);
   assert.equal(await page.locator('.learning-summary').count(),1);assert.equal(await page.locator('.learning-audit').count(),1);summaryCoverage++;
  }
 }
 await page.evaluate(()=>learningWorkspace.show());await page.locator('[data-detail="attention"]').first().click();assert.ok(await page.locator('#attention-comparison').isVisible());await page.keyboard.press('Escape');await page.locator('#learn-return').click();assert.equal((await page.evaluate(()=>learningWorkspace.state)).active,true);
 await page.locator('[data-draft="inputs"]').click();await page.mouse.click(2,2);assert.equal(await page.locator('#dspark-dialog').isVisible(),false);
 await page.locator('[data-workspace="evidence"]').click();assert.ok((await page.locator('#learn-body').innerText()).includes('Unknown'));await page.locator('#learn-body details > summary').click();await page.locator('#learn-audit').click();await page.locator('#inspector.open').waitFor();await page.keyboard.press('Escape');
 await page.evaluate(()=>learningWorkspace.show());await page.locator('[data-detail="all"]').click();
 const boundary=await page.locator('#diagram-content [data-module="hidden"]').textContent();for(const t of ['E = H19','L20 HC Pre','RMSNorm','X20'])assert.ok(boundary.includes(t));
 await page.locator('#zoom-in').click();await page.locator('#reset').click();assert.equal((await page.evaluate(()=>architectureApp.state)).view.w,1672);
 await page.screenshot({path:path.join(out,'legacy_boundary_1366.png')});await page.locator('#learn-return').click();
 assert.equal(await page.locator('#derivation-data').textContent(),packed);assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith('file:')||r.startsWith('data:')));
 const report={passed:true,viewports:[1920,1366,768,390],workspaces:4,dsparkTabs:7,summaryCoverage,keyboard:true,scroll:true,blankDismiss:true,scenarioModes:true,goldenProjection:true,ledgerUnchanged:true,offline:true,errors};fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(report,null,2));console.log(report);
 }finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1)});
