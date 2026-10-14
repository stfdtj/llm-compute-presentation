const {existsSync}=require('node:fs');
module.exports=()=>{
 const explicit=process.env.COMPUTE_LAB_CHROME;
 if(explicit&&!existsSync(explicit))throw new Error('COMPUTE_LAB_CHROME does not exist: '+explicit);
 const executablePath=explicit||[
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'
 ].find(existsSync);
 return {headless:true,...(executablePath?{executablePath}:{})};
};
