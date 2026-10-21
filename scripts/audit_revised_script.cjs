const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
(async()=>{
const out=path.resolve('docs/script_revision_evidence'); fs.mkdirSync(out,{recursive:true});
const b=await chromium.launch(require('../src/deepseek/browser_options.cjs')());
const p=await b.newPage({viewport:{width:1600,height:1000}}); const report={};
const controls=()=>p.evaluate(()=>[...document.querySelectorAll('button,select,summary')].filter(x=>x.getBoundingClientRect().height).map(x=>({id:x.id,text:x.innerText,options:x.tagName==='SELECT'?[...x.options].map(o=>[o.value,o.text]):undefined})));
await p.goto(pathToFileURL(path.resolve('deepseek_v4_1_interactive.html')).href); report.main=await controls();
await p.locator('#csa-tab-pipeline').click(); await p.locator('#ip-reset').click(); report.pipeline=[];
for(let i=0;i<6;i++){report.pipeline.push(await p.locator('#ip-simple').innerText());if(i<5)await p.locator('#ip-next').click();}
await p.locator('#ip-advanced summary').first().click(); report.pipelineControls=await controls();
await p.locator('#ip-algorithm').selectOption('probabilistic'); report.probabilistic=await p.locator('#ip-detail').innerText();
await p.screenshot({path:path.join(out,'pipeline.png')});
await p.goto(pathToFileURL(path.resolve('deepseek_v4_1_interactive.html')).href); await p.locator('#flow-toggle').click(); await p.locator('#shape-layer').selectOption('2'); report.shapeControls=await controls();
report.shape=await p.evaluate(()=>shapeTrace.graph.nodes.map(n=>({id:n.id,name:n.name,inputs:n.inputs,outputs:n.outputs})));
await p.locator('#shape-nav button').filter({hasText:'Q / KV'}).click(); await p.screenshot({path:path.join(out,'shape.png')});
await p.goto(pathToFileURL(path.resolve('deepseek_v4_1_interactive.html')).href);
report.modules=await p.locator('#module-data').textContent(); report.modules=Object.entries(JSON.parse(report.modules)).filter(([k,v])=>/query|expert|moe/i.test(k+' '+v.title)).map(([k,v])=>({key:k,title:v.title}));
await p.goto(pathToFileURL(path.resolve('index.html')).href); report.indexControls=await controls();await p.locator('#tab-roofline').click();report.roofControls=await controls();await p.locator('#tab-mfu').click();report.mfuText=await p.locator('#mfu').innerText();report.mfuControls=await controls();
await p.goto(pathToFileURL(path.resolve('compute-efficiency/index.html')).href); report.effRoof=await p.locator('main').innerText();report.effControls=await controls(); await p.locator('#tab-btn-mfu').click();report.effMFU=await p.locator('#tab-mfu').innerText();
fs.writeFileSync(path.join(out,'browser_audit.json'),JSON.stringify(report,null,2)); await b.close();console.log('Browser audit saved.');
})().catch(e=>{console.error(e);process.exit(1)});
