/* gAi Model Health v1 */
import {stats as modelStats} from './micro_transformer.js';
import {stats as datasetStats} from './dataset.js';
import {history as evalHistory} from './production_eval.js';
const KEY='gAiHealthV1';
function snapshot(){const m=modelStats(),d=datasetStats(),e=evalHistory();return{time:Date.now(),steps:m.trainSteps||0,loss:Number(m.loss)||0,samples:d.samples||0,epochs:d.epochs||0,eval:e.length?e.at(-1).score:null}}
export function health(){const h=snapshot();try{localStorage.setItem(KEY,JSON.stringify(h))}catch{};return h}
export function readiness(){const h=health();const checks={dataset:h.samples>=10,training:h.steps>=10,evaluation:h.eval!==null,lossFinite:Number.isFinite(h.loss)};return{...h,checks,ready:Object.values(checks).every(Boolean)}}
export function install(){if(typeof document==='undefined')return;const box=document.getElementById('llmStats');if(!box||document.getElementById('modelHealth'))return;const w=document.createElement('div');w.id='modelHealth';w.style.cssText='margin-top:14px;padding:12px;border-top:1px solid rgba(255,255,255,.1)';w.innerHTML='<div style="font-weight:700">🩺 Model Health</div><div id="healthText" class="muted" style="margin-top:6px"></div><button class="primary mini" id="healthBtn" style="margin-top:8px">🔄 Check Readiness</button>';box.parentNode.appendChild(w);const render=()=>{const r=readiness();document.getElementById('healthText').textContent=`${r.ready?'✅':'🟡'} ${r.ready?'Ready for local inference':'Keep training'} • dataset ${r.samples} • steps ${r.steps} • loss ${r.loss.toFixed(4)} • eval ${r.eval===null?'—':Math.round(r.eval*100)+'%'}`};document.getElementById('healthBtn').onclick=render;render()}
if(typeof window!=='undefined'){import('./production_eval.js').then(()=>{if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',install,{once:true});else setTimeout(install,1300)}).catch(()=>{})}
export default{health,readiness};
