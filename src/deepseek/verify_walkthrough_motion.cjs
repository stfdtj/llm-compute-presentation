/* Drive the actual controls and RAF clock; capture real in-between poses. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
let activeBrowser;
(async()=>{
 const browser=activeBrowser=await chromium.launch(require('./browser_options.cjs')());
 const context=await browser.newContext({viewport:{width:1440,height:1050},offline:true});
 const page=await context.newPage(),errors=[],requests=[],evidence={};
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 const out=path.join(__dirname,'qa','walkthrough');fs.mkdirSync(out,{recursive:true});
 await page.clock.install({time:new Date('2026-10-19T10:00:00Z')});await page.clock.pauseAt(new Date('2026-10-19T10:00:00Z'));
 await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/'));await page.evaluate(()=>document.fonts.ready);
 const data=await page.locator('#module-data').textContent(),geometry=await page.locator('#diagram-content').innerHTML();
 const state=()=>page.evaluate(()=>csaWalkthrough.state);
 const pose=id=>page.locator(`[data-motion-id="${id}"]`).evaluate(n=>{const m=n.transform.baseVal.consolidate().matrix;return{x:m.e,y:m.f,sx:m.a,sy:m.d,opacity:Number(n.getAttribute('opacity')),height:Number(n.querySelector('[data-box]')?.getAttribute('height')||0)};});
 const shot=name=>page.screenshot({path:path.join(out,name+'.png')});
 const next=async()=>{await page.locator('#csa-next').click();await page.clock.runFor(1900);};
 const chapter=async i=>{await page.locator(`[data-csa-chapter="${i}"]`).click();await page.clock.runFor(1900);};
 async function motion(name,id,button='#csa-next'){
  const begin=await pose(id);await shot(name+'-start');await page.locator(button).click();await page.clock.runFor(900);
  const middle=await pose(id),s=await state();assert.ok(s.phase>.4&&s.phase<.6,'Real in-flight frame');await shot(name+'-middle');
  await page.clock.runFor(1000);const end=await pose(id);await shot(name+'-end');
  assert.notDeepEqual(middle,begin,name+' changes geometry');assert.notDeepEqual(middle,end,name+' has a distinct middle frame');
  evidence[name]={begin,middle,end,phaseAtMiddle:s.phase};return {begin,middle,end};
 }
 await page.locator('#csa-tab-walkthrough').click();await page.clock.runFor(100);
 await page.evaluate(()=>{window.__csaNodes=[...document.querySelectorAll('#csa-canvas svg, #csa-canvas [data-motion-id]')];window.__csaElementChanges=0;new MutationObserver(ms=>{for(const m of ms)for(const n of [...m.addedNodes,...m.removedNodes])if(n.nodeType===1)__csaElementChanges++;}).observe(document.getElementById('csa-canvas'),{childList:true,subtree:true});});
 assert.equal((await state()).step,1);await next();
 const window=await motion('swa-window','swa-window');assert.ok(window.end.height<window.begin.height);
 const gather=await motion('swa-gather','swa-local-0');assert.ok(gather.end.x-gather.begin.x>400);
 const back=await motion('swa-backward','swa-local-0','#csa-prev');assert.equal(back.end.x,gather.begin.x);await next();
 await chapter(1);await next();
 const compression=await motion('compression','comp-fly-0');assert.ok(compression.end.x-compression.begin.x>300);assert.ok(compression.end.sy<compression.begin.sy);assert.equal((await pose('comp-out-0')).opacity,1);
 for(let pair=1;pair<4;pair++){await page.locator('#csa-next').click();await page.clock.runFor(900);assert.ok((await pose('comp-fly-'+pair*2)).opacity>.9,'Every pair visibly travels');await page.clock.runFor(1000);}
 for(let i=0;i<4;i++)assert.equal((await pose('comp-out-'+i)).opacity,1);await shot('compression-eight-to-four');
 await chapter(2);
 for(let i=0;i<4;i++){await next();assert.equal((await pose('idx-score-'+i)).opacity,1);if(i<3)assert.equal((await pose('idx-score-'+(i+1))).opacity,0);}
 const sort=await motion('indexer-sort','idx-score-1');assert.ok(sort.end.y<sort.begin.y-50);
 const selection=await motion('indexer-topk','idx-pos-0');assert.ok(selection.end.y-selection.begin.y>250);assert.equal((await pose('idx-key-0')).opacity,.18);
 await chapter(3);const read=await motion('attention-gather','attn-global-fly-0');assert.ok(read.end.x-read.begin.x>600);await next();
 const logits=await page.locator('[data-score-text]').allTextContents();await next();const weights=await page.locator('[data-score-text]').allTextContents();
 const den=1+logits.reduce((sum,v)=>sum+Math.exp(Number(v)),0);weights.forEach((w,i)=>assert.ok(Math.abs(Number(w)-Math.exp(Number(logits[i]))/den)<.002,'Displayed Softmax numbers agree, including Sink logit=0'));
 const output=await motion('attention-output','attn-value-fly-0');assert.ok(output.end.y-output.begin.y>350);
 await chapter(6);for(const mode of ['Full','Reindex','Reuse']){await page.locator(`[data-csa-mode="${mode}"]`).click();await page.clock.runFor(1900);assert.equal((await state()).mode,mode);await shot('mode-'+mode);}
 await chapter(0);await page.locator('#csa-next').click();await page.clock.runFor(700);await page.locator('#csa-play').click();
 const paused=await pose('swa-q-copy'),phase=(await state()).phase;await page.clock.runFor(3000);assert.deepEqual(await pose('swa-q-copy'),paused);assert.equal((await state()).phase,phase);
 await page.locator('#csa-play').click();await page.clock.runFor(1200);await page.locator('#csa-play').click();assert.equal((await state()).step,2);assert.equal((await state()).phase,1);
 await page.locator('#csa-replay').click();await page.clock.runFor(600);assert.ok((await state()).phase<.5);
 await page.locator('#csa-progress').fill('40');await page.locator('#csa-progress').dispatchEvent('input');assert.equal((await state()).phase,.4);assert.equal((await state()).running,false);
 await page.locator('#csa-play').click();await page.clock.runFor(1200);await page.locator('#csa-play').click();
 await page.locator('#csa-reset').click();assert.equal((await state()).step,1);assert.equal((await state()).mode,'Full');
 await page.locator('#csa-play').click();await page.clock.runFor(140000);assert.equal((await state()).step,46);assert.equal((await state()).playing,false);
 await page.locator('#csa-reset').click();await page.locator('#csa-walkthrough').focus();await page.keyboard.press('ArrowRight');await page.clock.runFor(1900);assert.equal((await state()).step,2);
 await page.locator('#csa-play').click();await page.clock.runFor(500);await page.locator('#csa-tab-architecture').click();const stopped=await state();await page.clock.runFor(5000);assert.equal((await state()).step,stopped.step);assert.equal((await state()).playing,false);
 assert.equal(await page.locator('#module-data').textContent(),data);assert.equal(await page.locator('#diagram-content').innerHTML(),geometry);
 // Verify legacy CSS transitions with a real-time page. The motion page uses a
 // virtual JS clock; CSS transition timing is a separate browser timeline.
 const original=await context.newPage();original.on('pageerror',e=>errors.push(e.message));await original.goto(page.url());await original.locator('#csa-tab-walkthrough').click();await original.locator('#csa-tab-architecture').click();
 for(const mode of ['parameter','forward','backward']){await original.locator(`header [data-mode="${mode}"]`).click();await original.locator('.module[data-module="qa"]').first().dispatchEvent('keydown',{key:'Enter'});await original.locator('#panel-body').waitFor({state:'visible'});assert.equal(await original.locator('#inspector').getAttribute('aria-hidden'),'false');assert.ok((await original.locator('#panel-body').textContent()).length>100,mode+' formulas present');await original.keyboard.press('Escape');}
 await original.close();
 await page.locator('#csa-tab-walkthrough').click();assert.equal(await page.evaluate(()=>__csaNodes.every(n=>n.isConnected)&&document.querySelectorAll('#csa-canvas svg').length===1&&__csaElementChanges===0),true,'No SVG/row replacement');
 await page.locator('.csa-matrix-extra summary').click();for(let i=0;i<7;i++)await page.locator('#csa-matrix-next').click();assert.ok((await page.locator('.csa-matrix-conclusion').innerText()).includes('20 FLOPs'));await page.locator('.csa-matrix-extra').screenshot({path:path.join(out,'matrix.png')});await page.locator('.csa-matrix-extra summary').click();
 for(const size of [{width:1920,height:1080},{width:1366,height:768},{width:768,height:1024},{width:390,height:844}]){await page.setViewportSize(size);await page.clock.runFor(100);await page.evaluate(()=>document.getElementById('csa-scroll').scrollTop=0);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await shot('responsive-'+size.width);}
 await page.setViewportSize({width:1366,height:768});await chapter(3);await page.locator('#csa-focus').click();await next();await next();await next();await next();await shot('presentation-1366');const canvas=await page.locator('#csa-canvas').boundingBox();assert.ok(canvas.y+canvas.height<768,'Scene fits presentation viewport');
 assert.deepEqual(errors,[]);assert.ok(requests.every(u=>u.startsWith('file:')||u.startsWith('data:')));
 const report={passed:true,actions:46,persistentSvg:true,elementReplacements:0,realAnimationEvidence:evidence,pauseFreezesGeometry:true,reverse:true,replay:true,scrub:true,playToEnd:true,originalCalculationModes:['parameter','forward','backward'],architectureDataAndGeometryPreserved:true,modes:['Full','Reindex','Reuse'],responsiveWidths:[1920,1366,768,390],offline:true,errors};
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(report,null,2));
 const names={'swa-window':'SWA · 128 位置窗口','swa-gather':'SWA · 行复制到 Local KV','swa-backward':'SWA · 回退','compression':'Compression · 2→1','indexer-sort':'Indexer · 分数移动排序','indexer-topk':'Indexer · 取出位置索引','attention-gather':'Main Attention · Gather','attention-output':'Main Attention · 加权汇总'};
 const gallery='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CSA2 动画验收截图</title><style>body{font:15px system-ui,"Microsoft YaHei";background:#f6f8fc;color:#20375c;max-width:1600px;margin:32px auto;padding:0 18px}h1{font-size:27px}h2{margin-top:32px;font-size:20px}.frames{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;background:white;border:1px solid #dce5f0;border-radius:8px;padding:10px}img{width:100%;height:auto}figcaption{font-size:13px;color:#536580}a{color:#2863c5}@media(max-width:800px){.frames{grid-template-columns:1fr}}</style><h1>CSA2 · 真实动画关键帧</h1><p>各组截图来自同一次浏览器动画，依次为开始、约 50% 进度与结束。点击图片查看原图；中间帧由真实 requestAnimationFrame 推进后截取，并未手工设置矩阵位置。</p><p><a href="../../../../deepseek_v4_1_interactive.html">打开演示网页</a> · <a href="validation.json">数值与回归检查报告</a></p>'+Object.keys(evidence).map(name=>'<section><h2>'+names[name]+'</h2><div class="frames">'+['start','middle','end'].map((phase,i)=>'<figure><a href="'+name+'-'+phase+'.png"><img loading="lazy" src="'+name+'-'+phase+'.png" alt="'+names[name]+' '+phase+'"></a><figcaption>'+['开始','中间 · 约 50%','结束'][i]+'</figcaption></figure>').join('')+'</div></section>').join('')+'</html>';
 fs.writeFileSync(path.join(out,'review.html'),gallery);console.log({passed:true,actions:46,animationScenes:Object.keys(evidence),errors});await browser.close();
})().catch(async e=>{console.error(e);if(activeBrowser)await activeBrowser.close();process.exitCode=1;});
