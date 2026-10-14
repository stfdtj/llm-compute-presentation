const {chromium}=require('playwright');
const path=require('path');
(async()=>{
 const browser=await chromium.launch(require('./browser_options.cjs')());
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto('file:///'+path.resolve(__dirname,'../deepseek_v4_1_interactive.html').replaceAll('\\','/'));
 await page.screenshot({path:path.join(__dirname,'overview_1920.png')});
 await page.locator('.module[data-module="embedding"]').click();
 await page.screenshot({path:path.join(__dirname,'panel_parameters.png')});
 await page.locator('[data-mode="backward"]').click();
 await page.locator('#close-panel').click();
 await page.locator('.module[data-module="query"]').first().click({position:{x:80,y:10}});
 await page.screenshot({path:path.join(__dirname,'panel_backward.png')});
 console.log(JSON.stringify({errors,title:await page.title(),state:await page.evaluate(()=>architectureApp.state)}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
