/* gAi Dataset Pipeline v3
 * Verified local corpus + deterministic train/validation split.
 */
const KEY='gAiDatasetV1',SPLIT='gAiDatasetSplitV1';
const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
function load(){try{return JSON.parse(localStorage.getItem(KEY))||{samples:[],seen:{},epochs:0}}catch{return{samples:[],seen:{},epochs:0}}}
function save(x){try{localStorage.setItem(KEY,JSON.stringify(x))}catch{}}
function hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
export function add(topic,text,source='human-verified'){const d=load(),sample={topic:clean(topic),text:clean(text),source,created:Date.now()};const key=(sample.topic+'|'+sample.text).toLowerCase();if(sample.topic&&sample.text&&!d.seen[key]){d.seen[key]=1;d.samples.push(sample);save(d);return true}return false}
export function addLessons(lessons=[]){let added=0;for(const l of lessons)if(add(l.topic,l.content,'human-verified'))added++;return{added,...stats()}}
export function samples(){return load().samples}
export function splitSets(){const all=load().samples;let map={};try{map=JSON.parse(localStorage.getItem(SPLIT))||{}}catch{};let changed=false;const train=[],validation=[];for(const s of all){const key=(s.topic+'|'+s.text).toLowerCase();if(!map[key]){map[key]=(hash(key)%5===0)?'validation':'train';changed=true};(map[key]==='validation'?validation:train).push(s)}if(changed)try{localStorage.setItem(SPLIT,JSON.stringify(map))}catch{}return{train,validation}}
export function trainingSamples(){return splitSets().train}
export function validationSamples(){return splitSets().validation}
export function splitStats(){const s=splitSets();return{train:s.train.length,validation:s.validation.length,total:s.train.length+s.validation.length,ratio:s.validation.length/(s.train.length+s.validation.length||1)}}
export function resetSplit(){try{localStorage.removeItem(SPLIT)}catch{}return splitSets()}
export function batches(size=4,source=null){const a=(source||load().samples).slice();for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]]}const out=[];for(let i=0;i<a.length;i+=Math.max(1,size))out.push(a.slice(i,i+size));return out}
export function markEpoch(){const d=load();d.epochs++;save(d);return d.epochs}
export function stats(){const d=load();const chars=d.samples.reduce((n,s)=>n+s.text.length,0);const topics=new Set(d.samples.map(s=>s.topic)).size;const sp=splitStats();return{samples:d.samples.length,characters:chars,epochs:d.epochs,topics,train:sp.train,validation:sp.validation}}
export function clear(){try{localStorage.removeItem(KEY);localStorage.removeItem(SPLIT)}catch{}}
export default{add,addLessons,samples,splitSets,trainingSamples,validationSamples,splitStats,resetSplit,batches,markEpoch,stats,clear};
if(typeof window!=='undefined')setTimeout(()=>{import('./batch_trainer.js').catch(()=>{});import('./validation.js').catch(()=>{});import('./metrics_dashboard.js').catch(()=>{});import('./production_eval.js').catch(()=>{});import('./model_health.js').catch(()=>{});},0);
