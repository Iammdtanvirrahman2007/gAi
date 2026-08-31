/* gAi Tiny Language Model
 * A deliberately small, browser-native word n-gram model.
 * It learns from compact human-provided text instead of shipping a huge dataset.
 * This is a language-model component, not a replacement for a large transformer.
 */

const MODEL_KEY = 'gAiTinyLMV1';
const START = '<s>';
const END = '</s>';
const clean = s => String(s || '').replace(/\s+/g, ' ').trim();
const tokenize = s => clean(s).toLowerCase().replace(/[^\p{L}\p{N}'!?.,-]+/gu, ' ').split(/\s+/).filter(Boolean);

function emptyModel(){ return { order: 3, counts: {}, vocabulary: {} }; }
function key(parts){ return parts.join('\u0001'); }
function load(){ try { return JSON.parse(localStorage.getItem(MODEL_KEY)) || emptyModel(); } catch { return emptyModel(); } }
function save(m){ try { localStorage.setItem(MODEL_KEY, JSON.stringify(m)); } catch {} }

function trainSentence(m, sentence){
  const words = [START, START, ...tokenize(sentence), END];
  words.forEach(w => m.vocabulary[w] = (m.vocabulary[w] || 0) + 1);
  for(let i=2;i<words.length;i++){
    const ctx = key(words.slice(i-2,i));
    const next = words[i];
    if(!m.counts[ctx]) m.counts[ctx] = {};
    m.counts[ctx][next] = (m.counts[ctx][next] || 0) + 1;
  }
}

export function train(text){
  const m = load();
  String(text || '').split(/[.!?\n]+/).map(clean).filter(Boolean).forEach(s => trainSentence(m,s));
  save(m); return m;
}

function candidates(m, context){
  const c = m.counts[key(context)] || {};
  return Object.entries(c).sort((a,b)=>b[1]-a[1]).map(([word,count])=>({word,count}));
}

export function generate(prompt='', maxWords=32){
  const m = load();
  const p = tokenize(prompt).slice(-2);
  let context = p.length===2 ? p : [START, START].slice(0,2-p.length).concat(p);
  while(context.length<2) context.unshift(START);
  const out=[];
  for(let i=0;i<maxWords;i++){
    const options = candidates(m, context).filter(x=>x.word!==START);
    if(!options.length) break;
    const total = options.reduce((n,x)=>n+x.count,0);
    let r=Math.random()*total, chosen=options[0].word;
    for(const x of options){ r-=x.count; if(r<=0){chosen=x.word;break;} }
    if(chosen===END) break;
    out.push(chosen); context=[context[1],chosen];
  }
  return out.join(' ');
}

export function trainFromLessons(lessons=[]){
  lessons.forEach(l=>{ train(`${l.topic}. ${l.content}`); });
}

export function modelStats(){
  const m=load();
  return { vocabulary:Object.keys(m.vocabulary).length, contexts:Object.keys(m.counts).length, order:m.order };
}

export function resetModel(){ try{localStorage.removeItem(MODEL_KEY)}catch{} }

export default {train,generate,trainFromLessons,modelStats,resetModel};
