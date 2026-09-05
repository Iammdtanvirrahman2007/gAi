// gAi Chat Engine v1: deterministic routing before generative inference.
import * as Transformer from './language/micro_transformer.js';
import * as TinyLM from './language/tiny_lm.js';
import * as Dataset from './language/dataset.js';

const MEMORY_KEY='gAiChatMemoryV1';
const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
const norm=s=>clean(s).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu,' ');
const words=s=>norm(s).split(/\s+/).filter(Boolean);
const memory=()=>{try{return JSON.parse(localStorage.getItem(MEMORY_KEY))||{facts:[]}}catch{return{facts:[]}}};
const saveMemory=x=>{try{localStorage.setItem(MEMORY_KEY,JSON.stringify(x))}catch{}};

function remember(text){
  const m=memory(), t=clean(text);
  const patterns=[
    [/^(?:my name is|i am|i'm)\s+(.+)$/i,'User said: their name/identity is $1.'],
    [/^remember that\s+(.+)$/i,'User asked me to remember: $1.'],
    [/^remember\s+(.+)$/i,'User asked me to remember: $1.']
  ];
  for(const [re,fmt] of patterns){const hit=t.match(re);if(hit){const fact=fmt.replace('$1',hit[1].trim());if(!m.facts.includes(fact)){m.facts.push(fact);m.facts=m.facts.slice(-50);saveMemory(m)}return fact}}
  return null;
}

function safeMath(q){
  const c=clean(q).replace(/×/g,'*').replace(/÷/g,'/').replace(/\^/g,'**').replace(/^(?:what is|calculate|solve)\s+/i,'');
  if(!/^[0-9\s()+\-*/%.]+$/.test(c)||!/\d/.test(c)||!/[+\-*/%]/.test(c))return null;
  try{const value=Function('"use strict";return ('+c+')')();return Number.isFinite(value)?value:null}catch{return null}
}

function greeting(q){return /^(hi|hello|hey|hiya|helo|good morning|good afternoon|good evening)[!,.\s]*$/i.test(clean(q))}
function thanks(q){return /^(thanks|thank you|thx|thank u)[!,.\s]*$/i.test(clean(q))}
function identity(q){return /^(who are you|what are you|tell me about yourself)\??$/i.test(clean(q))}
function status(q){return /^(how are you|how r you|how are things)\??$/i.test(clean(q))}
function capabilities(q){return /^(what can you do|what are your abilities|your capabilities)\??$/i.test(clean(q))}
function farewell(q){return /^(bye|goodbye|see you|see ya)[!,.\s]*$/i.test(clean(q))}
function night(q){return /^(good night|gn)[!,.\s]*$/i.test(clean(q))}

function memoryAnswer(q){
  const n=norm(q),m=memory();
  if(/what is my name|do you know my name|who am i/.test(n)){
    const f=m.facts.find(x=>/name\/identity/.test(x));
    return f?f.replace('User said: their name/identity is ','Your name is ').replace(/\.$/,'')+'.':null;
  }
  if(/what do you remember|remember anything about me/.test(n))return m.facts.length?'I remember:\n'+m.facts.map(x=>'• '+x.replace(/^User (?:said|asked me to remember):\s*/,'')).join('\n'):'I do not have any saved personal memories yet.';
  return null;
}

function learnedAnswer(q){
  let samples=[];try{samples=Dataset.samples()}catch{}
  const qWords=new Set(words(q));
  const scored=samples.map(s=>{const sw=words(s.topic+' '+s.text);const hit=sw.filter(w=>qWords.has(w)).length;return{...s,score:hit/Math.max(1,qWords.size)}}).filter(x=>x.score>=.2).sort((a,b)=>b.score-a.score);
  if(!scored.length)return null;
  const best=scored[0];
  return `From what I have learned:\n${best.text}`;
}

function usable(text,prompt){
  const x=clean(text);if(x.length<2||/^\[/.test(x))return false;
  const bad=['i am still learning','teach me more examples'];if(bad.some(s=>x.toLowerCase()===s))return false;
  const toks=words(x),pt=words(prompt),over=pt.length?toks.filter(t=>pt.includes(t)).length/Math.min(pt.length,Math.max(1,toks.length)):0;
  return x.length>=3 && (/[.!?]/.test(x)||toks.length>=3) && !(!/[\p{L}]/u.test(x));
}

export async function answer(prompt,history=[]){
  const q=clean(prompt);if(!q)return 'Please type a message and I will try to help.';
  const remembered=remember(q);if(remembered)return `Got it. I will remember that. 🧠`;
  const mem=memoryAnswer(q);if(mem)return mem;
  const math=safeMath(q);if(math!==null)return `The result is ${math}.`;
  if(greeting(q))return 'Hello! 👋 I am gAi. How can I help you?';
  if(thanks(q))return 'You are welcome! 😊';
  if(identity(q))return 'I am gAi, a local growing AI. I learn from the knowledge you teach me and improve through local training. 🧠';
  if(status(q))return 'I am ready. My local brain is active and I am learning from my saved knowledge. 🧠';
  if(capabilities(q))return 'I can chat, remember selected facts, calculate basic expressions, use learned lessons, analyze supported data, and generate responses locally without an external API.';
  if(farewell(q))return 'Goodbye! 👋';
  if(night(q))return 'Good night! 🌙';
  const learned=learnedAnswer(q);if(learned)return learned;
  const context=history.slice(-10).map(x=>x.role+': '+x.content).join('\n');
  const full=context?context+'\nuser: '+q:q;
  let out='';
  try{out=String(Transformer.generate(full,48,{temperature:.68,topK:8,topP:.9,repeatPenalty:.85})||'').trim()}catch{}
  if(usable(out,q))return out;
  try{out=String(TinyLM.generate(q,48)||'').trim()}catch{}
  if(usable(out,q))return out;
  return 'I do not know that yet. Teach me a verified example and I will add it to my local knowledge. 🧠';
}

export function stats(){const m=memory();let ds={samples:0};try{ds=Dataset.stats()}catch{}return{memoryFacts:m.facts.length,datasetSamples:ds.samples||0};}
export function clearMemory(){try{localStorage.removeItem(MEMORY_KEY)}catch{}}
