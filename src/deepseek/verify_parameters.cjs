const {chromium}=require('playwright');
const assert=require('assert/strict'),fs=require('fs'),path=require('path');
(async()=>{
 const browser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({viewport:{width:1366,height:900},offline:true});
 const page=await context.newPage(),errors=[],requests=[],checks=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/')+'#architecture');
 const open=async key=>{await page.locator(`.module[data-module="${key}"]`).first().dispatchEvent('keydown',{key:'Enter'});};
 await open('query');
 for(const [i,id,symbol] of [[0,'qa-0','WQA'],[1,'qnorm-1','γ'],[2,'qb-2','WQB']]){
  await page.locator(`[data-select-step="${i}"]`).click();
  assert.equal(await page.locator('.parameter-active').count(),1);
  assert.equal(await page.locator('.parameter-active').getAttribute('data-parameter'),id);
  const text=await page.locator('.current-step .active-weight').allTextContents();assert.ok(text.some(t=>t.replace(/\s/g,'')===symbol),JSON.stringify(text));
  const current=await page.locator('.current-formula .active-weight').allTextContents();assert.ok(current.some(t=>t.replace(/\s/g,'')===symbol),'current formula is visible alongside its weight');
  assert.ok(await page.locator('#parameter-mini .weight').count());
  assert.ok((await page.locator('#parameter-mini').innerText()).includes('不是权重'));
 }
 checks.push('Query W_QA / gamma / W_QB table, MathML and matrix synchronize');
 await page.locator('.calc-step[data-step-index="0"]').click();
 assert.equal((await page.evaluate(()=>parameterAudit.state)).activeStep,0);
 await page.locator('[data-parameter="qb-2"]').click();assert.equal((await page.evaluate(()=>parameterAudit.state)).activeStep,2);
 await page.locator('[data-select-step="1"]').click();await page.screenshot({path:path.join(__dirname,'qa','parameters_query_1366.png')});
 await page.locator('[data-mode="backward"]').click();
 await page.locator('.derivation-parameter-ref > summary').click();
 assert.ok((await page.locator('.derivation-step').first().innerText()).includes('W_QB'));
 assert.equal(await page.locator('#parameter-mini .gradient').count(),2);
 assert.ok((await page.locator('#parameter-mini').innerText()).includes('梯度不是新增参数'));
 const result=page.locator('.teaching-result .audit-result').first();assert.ok((await result.innerText()).includes('Derived'));
 await page.locator('.derivation-step').nth(1).evaluate(n=>n.open=true);
 const proofText=await page.locator('.derivation-step').nth(1).innerText();assert.ok(proofText.includes('单行 M=1')&&proofText.includes('dW'));
 assert.equal(await page.locator('.audit-assumptions [data-assumption]').count(),4);
 assert.ok(await page.locator('#panel-body a[href*="huggingface"]').count()>0);
 checks.push('Backward gradients separate; results expose exact/approx, four assumptions, pinned source locations');
 await page.locator('[data-mode="parameter"]').click();await open('rope');
 assert.ok((await page.locator('.parameter-box h2').innerText()).endsWith(': 0'));
 await open('qnorm');assert.equal(await page.locator('.parameter-table tbody tr').count(),1);
 assert.ok((await page.locator('.parameter-box').innerText()).includes('1,280'));
 await open('multi');assert.ok((await page.locator('.parameter-box').innerText()).includes('Attention Sink'));
 assert.ok((await page.locator('.parameter-box').innerText()).includes('64'));
 checks.push('RoPE zero; RMSNorm gamma and 64 Attention Sinks counted');
 await open('global');await page.locator('[data-select-step="1"]').click();assert.ok((await page.locator('.current-formula .active-weight').allTextContents()).includes('γ'));
 await page.evaluate(()=>openPanel('indexer'));await page.locator('[data-select-step="2"]').click();
 assert.ok((await page.locator('.current-formula .active-weight').allTextContents()).some(t=>t.replace(/\s/g,'')==='WIK'));
 assert.ok((await page.locator('.current-formula .active-weight').allTextContents()).includes('γ'));
 await open('hc');await page.locator('[data-select-step="1"]').click();assert.equal(await page.locator('.parameter-active').count(),2);
 checks.push('Global / Indexer normalization gamma; HC base and scale stages');
 await open('moe');const s=await page.evaluate(()=>architectureApp.data.moe.moeSummary);
 assert.ok(s.total>s.activated);assert.equal(s.expertsTotal/s.expertsActivated,385/7);
 assert.equal(await page.locator('.expert-demo span').count(),7);
 assert.ok((await page.locator('.parameter-box').innerText()).includes(s.total.toLocaleString('en-US')));
 await page.locator('#more-parameters summary').click();assert.equal(await page.locator('.parameter-table tbody tr').count(),9);
 checks.push('MoE Total / Activated / Shared / Top-6 / Router boundary and complete parameter list');
 await page.screenshot({path:path.join(__dirname,'qa','parameters_moe_1366.png')});
 await page.locator('#assumptions-toggle').click();assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'false');
 assert.equal(await page.locator('#panel-body > [data-assumption]').count(),4);
 assert.ok((await page.locator('#panel-body').innerText()).includes('1024'));
 await page.screenshot({path:path.join(__dirname,'qa','assumptions_1366.png')});
 await page.keyboard.press('Escape');assert.equal(await page.locator('#inspector').getAttribute('aria-hidden'),'true');
 checks.push('Global Assumptions opens, four categories and statuses; Escape closes');
 const keys=await page.evaluate(()=>Object.keys(architectureApp.data));let coverage=0;
 for(const key of keys){
  await page.evaluate(k=>openPanel(k),key);
  for(const mode of ['parameter','forward','backward']){
   await page.locator(`[data-mode="${mode}"]`).click();
   assert.equal(await page.locator('.parameter-box').count(),1,key+' '+mode);
   const metadata=await page.evaluate(({key,mode})=>architectureApp.data[key][mode],{key,mode});
   assert.equal(await page.locator('.teaching-result .audit-result').count(),metadata.rows.length);
   assert.equal(await page.locator('[data-select-step]').count(),metadata.steps.length);
   assert.equal(await page.locator('.audit-assumptions > summary').filter({hasText:'本模块'}).count(),1);
   coverage++;
  }
 }
 checks.push(coverage+' module/mode metadata coverage');
 for(const [width,height] of [[1920,1080],[1366,768],[768,1024],[390,844]]){
  await page.setViewportSize({width,height});await page.locator('[data-mode="parameter"]').click();await open('query');
  await page.locator('[data-select-step="0"]').click();
  await page.waitForTimeout(200);
  const metrics=await page.evaluate(()=>({header:document.querySelector('header').getBoundingClientRect().toJSON(),panel:document.querySelector('#inspector').getBoundingClientRect().toJSON(),stage:document.querySelector('#stage').getBoundingClientRect().toJSON(),scroll:document.body.scrollWidth,width:innerWidth,toolbar:[...document.querySelectorAll('header button')].map(n=>n.getBoundingClientRect().toJSON())}));
  assert.ok(metrics.panel.y>=metrics.header.bottom&&metrics.panel.bottom<=height);
  assert.ok(metrics.scroll<=width&&metrics.stage.y>=metrics.header.bottom);
  for(const r of metrics.toolbar)assert.ok(r.x>=0&&r.right<=width&&r.bottom<=metrics.header.bottom,JSON.stringify({width,r,header:metrics.header}));
  await page.screenshot({path:path.join(__dirname,'qa',`parameters_${width}.png`)});
  await page.keyboard.press('Escape');await page.locator('#flow-toggle').click();
  for(let i=1;i<5;i++)await page.locator('#flow-next').click();
  assert.equal((await page.evaluate(()=>informationFlow.state)).step,5);
  await page.locator('#flow-toggle').click();
 }
 checks.push('1920 / 1366 / 768 / 390 toolbar and inspector bounds; Data Flow five-step preservation');
 assert.deepEqual(errors,[]);assert.ok(requests.every(u=>u.startsWith('file:')));
 const report={passed:true,checks,coverage,errors,offline:true,statuses:await page.evaluate(()=>parameterAudit.data.summary)};
 fs.writeFileSync(path.join(__dirname,'qa','parameter_validation.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
