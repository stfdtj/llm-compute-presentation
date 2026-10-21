const {chromium}=require('playwright'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const out=path.join(__dirname,'qa/pipeline');
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await chromium.launch(require('./browser_options.cjs')());
 try{
 const context=await browser.newContext({viewport:{width:1366,height:768},offline:true}),page=await context.newPage(),errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto(require('node:url').pathToFileURL(path.resolve(__dirname,'../../deepseek_v4_1_interactive.html')).href);
 const architecture=await page.locator('#diagram-content').innerHTML();await page.locator('#csa-tab-pipeline').click();
 const state=()=>page.evaluate(()=>inferencePipeline.state);
 const go=async index=>{let s=await state();while(s.step!==index){await page.locator(index>s.step?'#ip-next':'#ip-prev').click();s=await state();}};
 const index=async(kind,i)=>page.evaluate(({kind,i})=>inferencePipeline.actions.findIndex(a=>a.kind===kind&&(i===undefined||a.i===i)),{kind,i});
 const stage=async(kind,i)=>go(await index(kind,i));
 await page.evaluate(()=>{window.pipelineNodes=[...document.querySelectorAll('[data-flow-node]')];});
 assert.deepEqual((await state()).cache,['我']);assert.equal(await page.locator('[data-flow-edge]').count(),17);
 await stage('slots');assert.deepEqual((await state()).confirmed,['我','喜欢','先']);assert.deepEqual((await state()).cache,['我','喜欢']);assert.deepEqual((await state()).pending,['先']);
 for(let i=0;i<3;i++){await stage('block',i);assert.equal(await page.locator('[data-slot]').count(),15);assert.equal(await page.locator('.ip-layer-active').count(),5);assert.equal((await state()).candidateCount,0);}
 await stage('head');assert.equal((await state()).candidateCount,0);
 for(let i=0;i<5;i++){await stage('sample',i);assert.equal((await state()).candidateCount,i+1);assert.deepEqual((await state()).confirmed,['我','喜欢','先']);assert.equal((await state()).verified,false);}
 await stage('input');assert.deepEqual((await state()).result.verificationInput,['先','A','B','C','D','E']);assert.equal((await state()).provisional.length,0);
 await stage('forward');assert.equal((await state()).provisional.length,6);assert.equal((await state()).verified,true);
 for(let i=0;i<5;i++){
  await page.locator(`[data-ip-alignment="${i}"] button`).click();assert.equal((await state()).selected,i);assert.equal(await page.locator('[data-inspected-input]').innerText(),i===0?'先':'ABCDE'[i-1]);assert.equal(await page.locator(`[data-ip-alignment="${i}"]`).getAttribute('class'),'ip-selected');
 }
 await page.locator('#ip-body summary').filter({hasText:'Causal Mask'}).click();
 const mask=await page.locator('.ip-mask tbody tr').evaluateAll(rows=>rows.map(r=>[...r.querySelectorAll('td')].map(c=>c.textContent)));
 assert.equal(mask.length,6);mask.forEach((r,i)=>assert.deepEqual(r,['✓',...Array.from({length:6},(_,j)=>j<=i?'✓':'×')]));
 const cases={middle:2,all:5,first:0,last:4};
 for(const algorithm of ['greedy','probabilistic'])for(const [scenario,accepted] of Object.entries(cases)){
  await page.locator('#ip-algorithm').selectOption(algorithm);await page.locator('#ip-scenario').selectOption(scenario);
  for(let i=0;i<5;i++){
   await stage('compare',i);let s=await state();assert.equal(s.result.accepted,accepted);assert.deepEqual(s.cache,['我','喜欢']);
   assert.equal(s.statuses[i],i<accepted?'accept':i===accepted?'reject':'discard');
   if(i>=accepted&&accepted<5){assert.ok(s.decided);assert.deepEqual(s.statuses.slice(accepted+1),Array(4-accepted).fill('discard'));}
  }
  let s=await state();assert.equal(s.result.extra,accepted===5?'Z':algorithm==='greedy'?'X':'Y');
  if(algorithm==='probabilistic'&&accepted<5){assert.ok((await page.locator('#ip-body').innerText()).includes('u_residual=0.82'));assert.ok((await page.locator('#ip-body').innerText()).includes('Top-1 是 X'));}
  await stage('commit');s=await state();assert.deepEqual(s.cache,['我','喜欢','先',...'ABCDE'.slice(0,accepted)]);assert.deepEqual(s.pending,[s.result.extra]);assert.deepEqual(s.confirmed,[...s.cache,s.result.extra]);assert.equal(s.provisional.length,0);
  await stage('next');assert.deepEqual((await state()).confirmed,s.confirmed);
 }
 for(const [width,height] of [[1366,768],[1920,1080]]){
  await page.setViewportSize({width,height});await page.locator('#ip-scenario').selectOption('middle');await page.locator('#ip-algorithm').selectOption('greedy');
  await stage('forward');await page.screenshot({path:path.join(out,`${width}x${height}_default_alignment.png`)});
  await stage('compare',2);await page.screenshot({path:path.join(out,`${width}x${height}_greedy_comparison.png`)});
  await page.locator('#ip-algorithm').selectOption('probabilistic');await page.locator('#ip-residual').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,`${width}x${height}_residual_correction.png`)});
 }
 await page.setViewportSize({width:1366,height:768});
 await page.locator('#ip-algorithm').selectOption('probabilistic');await page.locator('#ip-scenario').selectOption('middle');await stage('compare',2);
 await page.locator('#ip-numeric summary').click();await page.locator('#ip-residual summary').click();await page.locator('#ip-body [data-candidate="0"]').first().click();assert.equal(await page.locator('#ip-numeric').getAttribute('open'),'');
 assert.equal(await page.evaluate(()=>pipelineNodes.every((n,i)=>n===document.querySelectorAll('[data-flow-node]')[i])),true,'persistent graph nodes replaced');
 const screenshots=[],geometry=[];
 for(const [width,height] of [[1366,768],[1920,1080],[768,900],[390,844]]){
  await page.setViewportSize({width,height});
  for(const kind of ['prefill','decode','slots','block','sample','input','forward','compare','commit','next']){
   const i=kind==='block'?1:kind==='sample'?4:kind==='compare'?2:undefined;await stage(kind,i);
   await page.locator('#inference-pipeline').evaluate(n=>n.scrollTop=0);await page.locator('#ip-detail').evaluate(n=>n.scrollTop=0);
   const g=await page.evaluate(()=>{
    const root=document.querySelector('#inference-pipeline'),detail=document.querySelector('#ip-detail'),header=document.querySelector('header');
    const labels=[...root.querySelectorAll('button,select,.ip-slot')].filter(n=>n.getClientRects().length&&!n.closest('.ip-scroll')).filter(n=>n.scrollWidth>n.clientWidth+2).map(n=>n.textContent);
    const text=[...document.querySelectorAll('[data-flow-node]')].flatMap(g=>{const r=g.querySelector('rect').getBBox();return [...g.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<r.x||b.x+b.width>r.width||b.y+b.height>r.height+1;}).map(t=>t.textContent);});
    return {rootWidth:root.clientWidth,scrollWidth:root.scrollWidth,detailWidth:detail.clientWidth,detailScrollWidth:detail.scrollWidth,top:root.getBoundingClientRect().top,headerBottom:header.getBoundingClientRect().bottom,labels,text};
   });
   assert.ok(g.scrollWidth<=g.rootWidth+1,`root overflow ${width}/${kind}`);assert.ok(g.detailScrollWidth<=g.detailWidth+1,`detail overflow ${width}/${kind}`);assert.ok(g.top>=g.headerBottom,`header overlap ${width}`);assert.deepEqual(g.labels,[],`labels clip ${width}/${kind}`);assert.deepEqual(g.text,[],`map labels clip ${width}/${kind}`);geometry.push({width,height,kind});
   if(width>=1366||['forward','compare'].includes(kind)){const file=`${width}x${height}_${kind}.png`;await page.screenshot({path:path.join(out,file)});screenshots.push(file);}
  }
 }
 await page.setViewportSize({width:1366,height:768});
 for(const scenario of Object.keys(cases)){
  await page.locator('#ip-mode').selectOption('normal');await page.locator('#ip-scenario').selectOption(scenario);
  const n=(await state()).result.accepted+1;for(let i=0;i<n;i++){await stage('normal',i);const s=await state();assert.deepEqual(s.confirmed,[...s.cache,...s.pending]);assert.equal(s.candidateCount,0);}
  await stage('normalend');assert.deepEqual((await state()).confirmed,(await state()).result.confirmed);
  assert.ok(await page.locator('#ip-algorithm').isDisabled());await page.locator('#ip-mode').selectOption('dspark');
 }
 await stage('forward');await page.locator('#ip-play').click();assert.equal((await state()).playing,true);await page.locator('#csa-tab-walkthrough').click();assert.equal((await state()).playing,false);
 await page.locator('#csa-play').click();await page.locator('#csa-tab-pipeline').click();assert.equal(await page.evaluate(()=>csaWalkthrough.state.playing),false);assert.equal((await state()).stage,'forward');
 const csaStep=await page.evaluate(()=>csaWalkthrough.state.step);await page.locator('body').click({position:{x:4,y:4}});await page.keyboard.press('ArrowRight');assert.equal((await state()).stage,'compare');assert.equal(await page.evaluate(()=>csaWalkthrough.state.step),csaStep);await page.keyboard.press('0');assert.equal((await state()).step,0);
 await page.locator('#csa-tab-architecture').click();assert.equal(await page.locator('#diagram-content').innerHTML(),architecture);await page.locator('#flow-toggle').click();await page.locator('#csa-tab-pipeline').click();assert.equal(await page.locator('#shape-svg').isVisible(),false);await page.locator('#csa-tab-architecture').click();assert.equal(await page.locator('#shape-svg').isVisible(),true);await page.locator('#flow-toggle').click();
 await page.locator('#csa-tab-architecture').focus();await page.keyboard.press('End');assert.equal(await page.locator('#csa-tab-pipeline').getAttribute('aria-selected'),'true');
 await page.clock.install();await page.locator('#ip-play').click();await page.clock.fastForward(3700);assert.equal((await state()).step,1);await page.locator('#ip-play').click();assert.equal((await state()).playing,false);
 assert.deepEqual(errors,[]);assert.ok(requests.every(url=>url.startsWith('file:')));fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify({passed:true,algorithmScenarioCombinations:8,positions:5,persistentGraph:true,geometry,screenshots,errors,networkRequests:requests.filter(url=>!url.startsWith('file:'))},null,2));
 console.log('Pipeline browser: 8 algorithm/data combinations, 5-position provenance, 3x5 slots, shifted causal mask, residual correction, state transitions, persistent graph, 40 responsive scenes, offline and existing tab isolation passed.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
