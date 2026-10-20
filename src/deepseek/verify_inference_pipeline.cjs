const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const out=path.join(__dirname,'qa/pipeline');
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch(require('./browser_options.cjs')());
 try{
 const context=await browser.newContext({viewport:{width:1366,height:768},offline:true});
 const page=await context.newPage(),errors=[],requests=[];page.on('request',r=>requests.push(r.url()));
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(require('node:url').pathToFileURL(path.resolve(__dirname,'../../deepseek_v4_1_interactive.html')).href);
 const architecture=await page.locator('#diagram-content').innerHTML();
 const state=()=>page.evaluate(()=>inferencePipeline.state);
 const go=async i=>{await page.locator(`[data-ip-step="${i}"]`).click();};
 await page.locator('#csa-tab-pipeline').click();
 assert.equal(await page.locator('#stage').isVisible(),false);
 await go(3);assert.deepEqual((await state()).confirmed,['我','喜欢','先']);
 assert.equal(await page.locator('#ip-body .ip-draft').count(),3);
 await go(5);
 for(const [i,text] of ['P(吃 | 我喜欢先)','P(苹果 | 我喜欢先吃)','P(呀 | 我喜欢先吃苹果)'].entries())assert.ok((await page.locator(`[data-ip-check="${i}"]`).innerText()).includes(text));
 const mask=await page.locator('.ip-mask tbody tr').evaluateAll(rows=>rows.map(r=>[...r.querySelectorAll('td')].map(c=>c.textContent)));
 assert.deepEqual(mask,[['✓','✓','×','×','×'],['✓','✓','✓','×','×'],['✓','✓','✓','✓','×'],['✓','✓','✓','✓','✓']]);
 const cases={all:{n:3,text:['我','喜欢','先','吃','苹果','呀','！'],extra:'！'},first:{n:0,text:['我','喜欢','先','去'],extra:'去'},second:{n:1,text:['我','喜欢','先','吃','香蕉'],extra:'香蕉'},last:{n:2,text:['我','喜欢','先','吃','苹果','。'],extra:'。'}};
 for(const [name,c] of Object.entries(cases)){
  await page.locator('#ip-outcome').selectOption(name);await go(6);
  let s=await state();assert.equal(s.result.accepted,c.n);assert.deepEqual(s.confirmed,c.text);assert.equal(await page.locator('#ip-body .ip-reject').count(),3-c.n);
  assert.ok((await page.locator('#ip-body').innerText()).includes(name==='all'?'Target bonus token':'Target correction token'));
  await go(7);s=await state();assert.deepEqual(s.cache,['我','喜欢','先',...['吃','苹果','呀'].slice(0,c.n)]);assert.deepEqual(s.pending,[c.extra]);assert.ok(s.committed);
  await go(8);assert.deepEqual((await state()).confirmed,c.text);
 }
 for(const n of [1,2,3]){
  await page.locator('#ip-length').selectOption(String(n));await page.locator('#ip-outcome').selectOption('all');await go(5);assert.equal(await page.locator('[data-ip-check]').count(),n);
  await go(6);assert.equal((await state()).result.accepted,n);assert.equal(await page.locator('#ip-body .ip-defer').count(),3-n);
  await page.locator('#ip-outcome').selectOption('last');await go(7);assert.equal((await state()).result.accepted,n-1);
 }
 await page.locator('#ip-length').selectOption('3');await page.locator('#ip-outcome').selectOption('second');
 // Both requested presentation sizes, plus compact desktop and phone.
 const screenshots=[];
 for(const [width,height] of [[1366,768],[1920,1080],[768,900],[390,844]]){
  await page.setViewportSize({width,height});
  for(const i of [0,1,2,3,4,5,6,7,8]){
   await go(i);await page.locator('#inference-pipeline').evaluate(n=>n.scrollTop=0);
   const geometry=await page.evaluate(()=>{
    const root=document.querySelector('#inference-pipeline'),header=document.querySelector('header');
    const clipped=[...root.querySelectorAll('button,select,.ip-module,.ip-chip')].filter(n=>n.getClientRects().length&&!n.closest('.ip-scroll')).filter(n=>n.scrollWidth>n.clientWidth+2).map(n=>n.textContent);
    return {width:root.clientWidth,scrollWidth:root.scrollWidth,top:root.getBoundingClientRect().top,headerBottom:header.getBoundingClientRect().bottom,clipped};
   });
   assert.ok(geometry.scrollWidth<=geometry.width+1,`horizontal overflow ${width} step ${i}`);
   assert.ok(geometry.top>=geometry.headerBottom,`header overlaps ${width}`);
   assert.deepEqual(geometry.clipped,[],`labels clip ${width} step ${i}`);
   if([3,5,6,7].includes(i)){
    const file=`${width}x${height}_step_${i}.png`;await page.screenshot({path:path.join(out,file)});screenshots.push(file);
   }
  }
 }
 await page.setViewportSize({width:1366,height:768});await page.locator('#ip-mode').selectOption('normal');
 assert.equal((await state()).step,0);assert.ok(await page.locator('#ip-outcome').isDisabled());
 for(let i=3;i<=5;i++){
  await go(i);const s=await state();assert.deepEqual(s.confirmed,['我','喜欢','先',...['吃','苹果','呀'].slice(0,i-2)]);
  assert.deepEqual(s.pending,[['吃','苹果','呀'][i-3]]);assert.ok((await page.locator('#ip-body').innerText()).includes('DSpark 不参与'));
 }
 await page.screenshot({path:path.join(out,'normal_decode.png')});
 await page.locator('#ip-mode').selectOption('dspark');await go(5);
 await page.locator('#ip-play').click();assert.equal((await state()).playing,true);
 await page.locator('#csa-tab-walkthrough').click();assert.equal((await state()).playing,false);assert.equal(await page.locator('#inference-pipeline').isVisible(),false);
 assert.equal(await page.evaluate(()=>csaWalkthrough.state.active),true);
 await page.locator('#csa-play').click();await page.locator('#csa-tab-pipeline').click();assert.equal(await page.evaluate(()=>csaWalkthrough.state.playing),false);assert.equal((await state()).stage,'verify');
 const csaStep=await page.evaluate(()=>csaWalkthrough.state.step);
 await page.locator('body').click({position:{x:5,y:5}});await page.keyboard.press('ArrowRight');assert.equal((await state()).stage,'accept');assert.equal(await page.evaluate(()=>csaWalkthrough.state.step),csaStep);
 await page.keyboard.press('0');assert.equal((await state()).step,0);assert.equal((await state()).outcome,'all');assert.equal((await state()).length,3);
 await page.locator('#ip-next').click();assert.equal((await state()).step,1);await page.locator('#ip-prev').click();assert.equal((await state()).step,0);
 await page.locator('#csa-tab-pipeline').focus();await page.keyboard.press('Home');assert.equal(await page.locator('#csa-tab-architecture').getAttribute('aria-selected'),'true');
 assert.equal(await page.locator('#stage').isVisible(),true);assert.equal(await page.locator('#diagram-content').innerHTML(),architecture);
 await page.locator('#flow-toggle').click();assert.equal(await page.locator('#shape-svg').isVisible(),true);
 await page.locator('#csa-tab-pipeline').click();assert.equal(await page.locator('#shape-svg').isVisible(),false);
 await page.locator('#csa-tab-architecture').click();assert.equal(await page.locator('#shape-svg').isVisible(),true);
 assert.equal((await state()).playing,false);await page.locator('#flow-toggle').click();
 await page.locator('#csa-tab-architecture').focus();await page.keyboard.press('End');assert.equal(await page.locator('#csa-tab-pipeline').getAttribute('aria-selected'),'true');
 await page.locator('#ip-play').click();await page.waitForTimeout(4400);assert.equal((await state()).step,1);await page.locator('#ip-play').click();assert.equal((await state()).playing,false);
 assert.deepEqual(errors,[]);assert.ok(requests.every(url=>url.startsWith('file:')),'offline page must not request network resources');fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify({passed:true,outcomes:Object.keys(cases),lengths:[1,2,3],viewports:[[1366,768],[1920,1080],[768,900],[390,844]],screenshots,errors},null,2));
 console.log('Inference Pipeline: outcome/state/mask/alignment, scheduler lengths, normal decode, playback, tab isolation, Shape Trace restoration, keyboard navigation and 36 responsive scenes passed.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
