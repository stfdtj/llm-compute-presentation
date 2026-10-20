/* Reproduce header overlap after scrolling; assert separate, non-overlapping
 * control/navigation/content regions across short and narrow viewports. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
let browser;
(async()=>{
 browser=await chromium.launch(require('./browser_options.cjs')());
 const page=await browser.newPage({viewport:{width:2048,height:720},offline:true});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const out=path.join(__dirname,'qa','layout-overlap');fs.mkdirSync(out,{recursive:true});
 await page.goto('file:///'+path.resolve(__dirname,'../../deepseek_v4_1_interactive.html').replaceAll('\\','/'));
 await page.locator('#csa-tab-walkthrough').click();await page.locator('#csa-next').click();
 await page.waitForFunction(()=>!csaWalkthrough.state.running);
 if(process.argv.includes('--before')){
  await page.evaluate(()=>document.getElementById('csa-walkthrough').scrollTop=130);
  await page.screenshot({path:path.join(out,'before-scroll.png')});
  const h=await page.locator('.csa-heading').boundingBox(),n=await page.locator('.csa-step-nav').boundingBox();
  console.log({reproduced:n.y<h.y+h.height,heading:h,nav:n});await browser.close();return;
 }
 const cases=[];
 for(const size of [{width:2506,height:560},{width:2048,height:720},{width:1920,height:1080},{width:1366,height:768},{width:1100,height:600},{width:900,height:480},{width:390,height:844}]){
  await page.setViewportSize(size);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  await page.waitForFunction(()=>document.getElementById('csa-walkthrough').getBoundingClientRect().top>=document.querySelector('header').getBoundingClientRect().bottom);
  const base=await page.locator('.csa-heading').boundingBox(),nav=await page.locator('.csa-step-nav').boundingBox();
  await page.locator('.csa-scroll').evaluate(e=>e.scrollTop=0);
  await page.screenshot({path:path.join(out,`initial-${size.width}x${size.height}.png`)});
  for(const scroll of [0,80,240,700]){
   await page.locator('.csa-scroll').evaluate((e,y)=>e.scrollTop=y,scroll);
   const heading=await page.locator('.csa-heading').boundingBox(),navigation=await page.locator('.csa-step-nav').boundingBox(),content=await page.locator('.csa-scroll').boundingBox();
   assert.ok(heading.y+heading.height<=navigation.y+1,'Heading never covers chapter navigation');
   assert.ok(navigation.y+navigation.height<=content.y+1,'Navigation never covers content viewport');
   assert.equal(heading.y,base.y,'Scrolling preserves control position');assert.equal(navigation.y,nav.y,'Scrolling preserves chapter position');
   assert.ok(content.height>80,'Content remains scrollable in short windows');
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'No horizontal page overflow');
   assert.ok(await page.locator('#csa-next').isVisible());
  }
  await page.screenshot({path:path.join(out,`scrolled-${size.width}x${size.height}.png`)});cases.push(size);
 }
 await page.setViewportSize({width:1366,height:768});await page.locator('#csa-reset').click();
 assert.equal(await page.locator('.csa-scroll').evaluate(e=>e.scrollTop),0,'Reset restores the new scrolling region');
 await page.locator('[data-csa-chapter="4"]').click();await page.locator('#csa-focus').click();
 assert.equal(await page.locator('.csa-step-nav').isVisible(),false);
 await page.screenshot({path:path.join(out,'presentation-1366.png')});
 assert.equal(await page.locator('.csa-scroll').evaluate(e=>e.scrollTop),0,'Presentation view starts at top');
 await page.locator('#csa-focus').click();await page.locator('.csa-scroll').evaluate(e=>e.scrollTop=200);
 await page.locator('#csa-tab-architecture').click();assert.ok(await page.locator('#stage').isVisible());
 await page.locator('#csa-tab-walkthrough').click();
 const heading=await page.locator('.csa-heading').boundingBox(),nav=await page.locator('.csa-step-nav').boundingBox();assert.ok(heading.y+heading.height<=nav.y);
 assert.deepEqual(errors,[]);
 const report={passed:true,cases,scrollPositions:[0,80,240,700],controlsRemainVisible:true,navigationRemainsVisible:true,reset:true,presentation:true,errors};
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(report,null,2));console.log(report);await browser.close();
})().catch(async e=>{console.error(e);if(browser)await browser.close();process.exitCode=1;});
