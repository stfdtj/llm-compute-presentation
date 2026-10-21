/* Reproducible presentation-skill renderer extension.
 * Native editable mathematical scenes + native PowerPoint entrance timings.
 * The source PPTX is generated once; no generated slide package is patched.
 */
'use strict';
const fs=require('fs'),path=require('path'),Module=require('module');
const skill=process.env.PRESENTATION_SKILL || path.join(__dirname,'vendor/presentation-skill');
const modules=path.join(__dirname,'node_modules');
const installed='C:/Users/15344/.codex/skills/presentation-skill/node_modules';
process.env.NODE_PATH=[modules,path.join(skill,'node_modules'),installed,process.env.NODE_PATH||''].join(path.delimiter);Module._initPaths();
// Extend the upstream XML emitter at source-load time. Fail closed on version drift.
const library=require.resolve('pptxgenjs');
const oldLoader=Module._extensions['.js'];
Module._extensions['.js']=function(mod,filename){
  if(filename!==library)return oldLoader(mod,filename);
  let source=fs.readFileSync(filename,'utf8');
  const needle="'<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>');";
  if(!source.includes(needle))throw Error('Unsupported PptxGenJS XML emitter; update animation adapter.');
  source=source.replace(needle,"'<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>' + (slide._teachingTiming || '') + '</p:sld>');");
  mod._compile(source,filename);
};
const templates=require(path.join(skill,'templates/pptxgenjs/slides.js'));
const sceneRenderer=require('./scenes.js');
const base=templates.renderStandard;
templates.renderStandard=function(pptx,slide,data,preset){
  if(!data.math_scene)return base(pptx,slide,data,preset);
  sceneRenderer.render(pptx,slide,data,preset);
  data.__roleContractConsumesSummary=true;
};
const builder=require(path.join(skill,'scripts/build_deck_pptxgenjs.js'));
if(!process.argv.includes('--outline'))process.argv.push('--outline',path.join(__dirname,'outline.json'),'--output',path.join(__dirname,'DeepSeek_V41_Flash_计算成本可视化.pptx'),'--style-preset','lab-report');
builder.main().then(()=>sceneRenderer.flush()).catch(e=>{console.error(e.stack);process.exitCode=1;});
