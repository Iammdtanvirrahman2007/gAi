/* gAi Dataset Pipeline v1
 * Collects verified lessons into a compact local corpus, deduplicates samples,
 * creates shuffled training batches, and persists dataset metadata in the browser.
 */
const KEY='gAiDatasetV1';
const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
function load(){try{return JSON.parse(localStorage.getItem(KEY))||{samples:[],seen:{},epochs:0}}catch{return{samples:[],seen:{},epochs:0}}}
function save(x){try{localStorage.setItem(KEY,JSON.stringify(x))}catch{}}
export function add(topic,text,source='human-verified'){const d=load(),sample={topic:clean(topic),text:clean(text),source,created:Date.now()};const key=(sample.topic+'|'+sample.text).toLowerCase();if(sample.topic&&sample.text&&!d.seen[key]){d.seen[key]=1;d.samples.push(sample);save(d);return true}return false}
export function addLessons(lessons=[]){let added=0;for(const l of lessons)if(add(l.topic,l.content,'human-verified'))added++;return{added,...stats()}}
export function samples(){return load().samples}
export function batches(size=4){const a=load().samples.slice();for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]]}const out=[];for(let i=0;i<a.length;i+=Math.max(1,size))out.push(a.slice(i,i+size));return out}
export function markEpoch(){const d=load();d.epochs++;save(d);return d.epochs}
export function stats(){const d=load();const chars=d.samples.reduce((n,s)=>n+s.text.length,0);return{samples:d.samples.length,characters:chars,epochs:d.epochs}}
export function clear(){try{localStorage.removeItem(KEY)}catch{}}
export default{add,addLessons,samples,batches,markEpoch,stats,clear};