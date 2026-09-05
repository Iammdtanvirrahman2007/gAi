/* gAi Subword Tokenizer v1
 * Lightweight byte-free BPE-style tokenizer for the browser.
 * Learns frequent adjacent symbol pairs from supplied text and persists merges.
 */
const KEY='gAiTokenizerV1';
const clean=s=>String(s||'').normalize('NFKC').toLowerCase().replace(/\s+/g,' ').trim();
const baseUnits=s=>clean(s).split(/\s+/).flatMap(w=>w.length?['▁',...Array.from(w)]:[]);
function load(){try{return JSON.parse(localStorage.getItem(KEY))||{merges:{},steps:0}}catch{return{merges:{},steps:0}}}
function save(x){try{localStorage.setItem(KEY,JSON.stringify(x))}catch{}}
function pairCounts(seq){const c={};for(let i=0;i<seq.length-1;i++){const k=seq[i]+' '+seq[i+1];c[k]=(c[k]||0)+1}return c}
export function train(text,merges=24){const m=load();let seq=baseUnits(text);for(let step=0;step<merges;step++){const c=pairCounts(seq),best=Object.entries(c).sort((a,b)=>b[1]-a[1])[0];if(!best||best[1]<2)break;const [pair,count]=best,name=pair.replace(/ /g,'');m.merges[pair]=name;const next=[];for(let i=0;i<seq.length;i++){if(i<seq.length-1&&seq[i]+' '+seq[i+1]===pair){next.push(name);i++}else next.push(seq[i])}seq=next;m.steps++}save(m);return stats()}
export function encode(text){const m=load();let seq=baseUnits(text);let changed=true;while(changed){changed=false;for(const [pair,name] of Object.entries(m.merges)){for(let i=0;i<seq.length-1;i++){if(seq[i]+' '+seq[i+1]===pair){seq.splice(i,2,name);changed=true;i--}}}}return seq}
export function decode(tokens){return tokens.join('').replace(/▁/g,' ').trim()}
export function stats(){const m=load();return{merges:Object.keys(m.merges).length,trainingSteps:m.steps,vocabEstimate:Object.keys(m.merges).length+128}}
export function reset(){try{localStorage.removeItem(KEY)}catch{}}
export default{train,encode,decode,stats,reset};