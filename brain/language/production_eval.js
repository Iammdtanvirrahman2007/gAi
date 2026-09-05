/* gAi Production Evaluation v1
 * Turns the training pipeline into a reproducible local evaluation suite.
 * No API, no server, no external model.
 */
import {samples as getSamples} from './dataset.js';
import {generate,stats as modelStats} from './micro_transformer.js';

const KEY='gAiEvalHistoryV1';
const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
const words=s=>clean(s).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu,' ').split(/\s+/).filter(Boolean);
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY))||[]}catch{return[]}};
const save=x=>{try{localStorage.setItem(KEY,JSON.stringify(x.slice(-30)))}catch{}};

function score(reference,prediction){
 const r=words(reference),p=words(prediction),set=new Set(r);
 if(!r.length)return 0;
 return Math.min(1,p.filter(x=>set.has(x)).length/r.length);
}

export function evaluate(limit=30){
 const data=getSamples().slice(0,Math.max(1,Number(limit)||30));
 if(!data.length)return{ok:false,reason:'EMPTY_DATASET'};
 let total=0,good=0;const examples=[];
 for(const s of data){
  const w=words(s.text);if(w.length<2)continue;
  const cut=Math.max(1,Math.floor(w.length*.55));
  const prompt=w.slice(0,cut).join(' '),reference=w.slice(cut,cut+12).join(' ');
  const prediction=clean(generate(prompt,12));const sscore=score(reference,prediction);
  total++;good+=sscore;
  if(examples.length<5)examples.push({topic:s.topic,score:Math.round(sscore*100),prediction,reference});
 }
 const result={time:Date.now(),samples:total,score:total?good/total:0,scorePercent:Math.round((total?good/total:0)*100),model:modelStats(),examples};
 const h=load();h.push(result);save(h);return{ok:true,...result};
}
export function history(){return load()}
export function clearHistory(){try{localStorage.removeItem(KEY)}catch{}}

function install(){
 if(typeof document==='undefined')return;
 const box=document.getElementById('llmStats');if(!box||document.getElementById('productionEval'))return;
 const wrap=document.createElement('div');wrap.id='productionEval';wrap.style.cssText='margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,.1)';
 wrap.innerHTML='<div style="font-weight:700">🏁 Production Evaluation</div><div class="muted" id="prodEvalText" style="margin-top:6px">Reproducible local benchmark</div><button class="primary mini" id="prodEvalBtn" style="margin-top:8px">▶ Run Benchmark</button><div class="muted" id="prodEvalExamples" style="margin-top:7px"></div>';
 box.parentNode.appendChild(wrap);
 document.getElementById('prodEvalBtn').onclick=()=>{const r=evaluate();const t=document.getElementById('prodEvalText');if(!r.ok){t.textContent='⚠ '+r.reason;return}t.textContent=`✓ Benchmark ${r.scorePercent}% • ${r.samples} samples • ${r.model.trainSteps||0} steps`;document.getElementById('prodEvalExamples').innerHTML=r.examples.map(x=>`<div style="margin-top:4px">${String(x.topic)}: <b>${x.score}%</b> • ${String(x.prediction||'∅')}</div>`).join('')};
}
if(typeof window!=='undefined'){if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',install,{once:true});else setTimeout(install,1100)}
export default{evaluate,history,clearHistory};
