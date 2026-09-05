/* gAi Batch Trainer v2
 * Trains the browser Transformer over shuffled dataset batches.
 * No API, no server, browser-local only.
 */
import {batches,markEpoch,stats as datasetStats} from './dataset.js';
import {trainFromLessons,stats as modelStats} from './micro_transformer.js';

export function trainBatch(batchSize=4,epochs=1,onProgress=null){
  const size=Math.max(1,Number(batchSize)||4), totalEpochs=Math.max(1,Number(epochs)||1);
  const before=datasetStats();
  if(!before.samples)return{ok:false,reason:'EMPTY_DATASET',dataset:before,model:modelStats()};
  let batchesDone=0,last={loss:0,steps:0};
  for(let e=0;e<totalEpochs;e++){
    const groups=batches(size);
    for(const group of groups){
      last=trainFromLessons(group);
      batchesDone++;
      if(typeof onProgress==='function')onProgress({epoch:e+1,epochs:totalEpochs,batch:batchesDone,batches:groups.length,loss:last.loss,steps:last.steps});
    }
    markEpoch();
  }
  return{ok:true,epochs:totalEpochs,batches:batchesDone,dataset:datasetStats(),model:modelStats(),last};
}

export function batchPreview(batchSize=4){
  const groups=batches(Math.max(1,Number(batchSize)||4));
  return groups.slice(0,3).map((g,i)=>({batch:i+1,size:g.length,topics:g.map(x=>x.topic)}));
}

function installBatchUI(){
  const install=()=>{
    if(document.getElementById('batchTrainingControls'))return;
    const statsBox=document.getElementById('llmStats');
    if(!statsBox)return;
    const wrap=document.createElement('div');
    wrap.id='batchTrainingControls';
    wrap.style.cssText='margin-top:14px;padding:12px;border:1px solid rgba(255,255,255,.1);border-radius:12px';
    wrap.innerHTML='<div style="font-weight:700;margin-bottom:8px">📚 Dataset Training</div><div style="display:flex;gap:8px;flex-wrap:wrap"><button class="primary mini" id="train1EpochBtn">🧠 Train 1 Epoch</button><button class="primary mini" id="train5EpochBtn">🚀 Train 5 Epochs</button></div><div id="batchProgressText" class="muted" style="margin-top:9px">Ready</div><div style="height:7px;background:rgba(255,255,255,.08);border-radius:9px;overflow:hidden;margin-top:7px"><div id="batchProgressBar" style="height:100%;width:0%;transition:width .2s"></div></div>';
    statsBox.parentNode.appendChild(wrap);
    const run=(epochs)=>{
      const text=document.getElementById('batchProgressText'),bar=document.getElementById('batchProgressBar');
      document.getElementById('train1EpochBtn').disabled=true;document.getElementById('train5EpochBtn').disabled=true;
      const result=trainBatch(4,epochs,p=>{const pct=Math.round((p.batch/Math.max(1,p.batches))*100);text.textContent=`Epoch ${p.epoch}/${p.epochs} • Batch ${p.batch}/${p.batches} • loss ${Number(p.loss).toFixed(4)}`;bar.style.width=pct+'%'});
      document.getElementById('train1EpochBtn').disabled=false;document.getElementById('train5EpochBtn').disabled=false;
      if(result.ok){bar.style.width='100%';text.textContent=`✓ Training complete • ${result.epochs} epoch(s) • ${result.batches} batches • loss ${Number(result.last.loss).toFixed(4)}`}
      else text.textContent='⚠ Dataset is empty. Teach gAi first.';
      renderBatchStats();
    };
    document.getElementById('train1EpochBtn').onclick=()=>run(1);
    document.getElementById('train5EpochBtn').onclick=()=>run(5);
    renderBatchStats();
  };
  const render=()=>{install();renderBatchStats()};
  const renderBatchStats=()=>{};
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',install);else setTimeout(install,0);
}

function renderBatchStats(){
  const box=document.getElementById('batchProgressText');if(!box)return;
  const d=datasetStats(),m=modelStats();
  if(!box.dataset.training)box.textContent=`${d.samples} dataset samples • ${d.epochs} completed epoch(s) • ${m.trainSteps||0} model steps`;
}

installBatchUI();
export default{trainBatch,batchPreview};
