/* gAi Batch Trainer v1
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

export default{trainBatch,batchPreview};
