// gAi Chat Engine v2: local routing, teachable knowledge, memory and grounded retrieval.
import * as Transformer from './language/micro_transformer.js';
import * as TinyLM from './language/tiny_lm.js';
import * as Dataset from './language/dataset.js';

const MEMORY_KEY='gAiChatMemoryV1';
const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
const norm=s=>clean(s).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu,' ');
const words=s=>norm(s).split(/\s+/).filter(Boolean);
const stop=new Set('the a an and or but is are was were be to of in on for with from this that it as at by about what who how why when where do does did can could would should i you me my your our their they we'.split(' '));
const memory=()=>{try{return JSON.parse(localStorage.getItem(MEMORY_KEY))||{facts:[]}}catch{return{facts:[]}}};
const saveMemory=x=>{try{localStorage.setItem(MEMORY_KEY,JSON.stringify(x))}catch{}};

function remember(text){
  const m=memory(),t=clean(text);
  const patterns=[[/^(?:my name is|i am|i'm)\s+(.+)$/i,'User said: their name/identity is $1.'],[/^remember that\s+(.+)$/i,'User asked me to remember: $1.'],[/^remember\s+(.+)$/i,'User asked me to remember: $1.']];
  for(const [re,fmt] of patterns){const hit=t.match(re);if(hit){const fact=fmt.replace('$1',hit[1].trim());if(!m.facts.includes(fact)){m.facts.push(fact);m.facts=m.facts.slice(-50);saveMemory(m)}return fact}}
  return null;
}

function teachCommand(q){
  let m=clean(q).match(/^teach(?:\s+me)?\s*[:|-]\s*(.+?)\s*(?:\||=>|=)\s*(.+)$/is);
  if(!m)m=clean(q).match(/^learn(?:\s+this)?\s*[:|-]\s*(.+?)\s*(?:\||=>|=)\s*(.+)$/is);
  if(!m)return null;
  const topic=clean(m[1]),text=clean(m[2]);
  if(topic.length<2||text.length<3)return null;
  try{Dataset.add(topic,text,'chat-teaching');TinyLM.train(`${topic}. ${text}`);Transformer.trainFromLessons([{topic,content:text}]);return `Learned and saved. 🧠\n\n**${topic}**\n${text}`;}catch{return 'I understood the lesson, but could not save it yet.'}
}

function safeMath(q){
  const c=clean(q).replace(/×/g,'*').replace(/÷/g,'/').replace(/\^/g,'**').replace(/^(?:what is|calculate|solve)\s+/i,'');
  if(!/^[0-9\s()+\-*/%.]+$/.test(c)||!/\d/.test(c)||!/[+\-*/%]/.test(c))return null;
  try{const value=Function('"use strict";return ('+c+')')();return Number.isFinite(value)?value:null}catch{return null}
}

const exact={
  greeting:/^(hi|hello|hey|hiya|helo|good morning|good afternoon|good evening)[!,.\s]*$/i,
  thanks:/^(thanks|thank you|thx|thank u)[!,.\s]*$/i,
  identity:/^(who are you|what are you|tell me about yourself)\??$/i,
  status:/^(how are you|how r you|how are things)\??$/i,
  capabilities:/^(what can you do|what are your abilities|your capabilities)\??$/i,
  farewell:/^(bye|goodbye|see you|see ya)[!,.\s]*$/i,
  night:/^(good night|gn)[!,.\s]*$/i
};

function memoryAnswer(q){
  const n=norm(q),m=memory();
  if(/what is my name|do you know my name|who am i/.test(n)){
    const f=m.facts.find(x=>/name\/identity/.test(x));
    return f?f.replace('User said: their name/identity is ','Your name is ').replace(/\.$/,'')+'.':null;
  }
  if(/what do you remember|remember anything about me/.test(n))return m.facts.length?'I remember:\n'+m.facts.map(x=>'• '+x.replace(/^User (?:said|asked me to remember):\s*/,'')).join('\n'):'I do not have any saved personal memories yet.';
  return null;
}

function similarity(a,b){
  const A=new Set(words(a).filter(x=>!stop.has(x))),B=new Set(words(b).filter(x=>!stop.has(x)));
  if(!A.size||!B.size)return 0;
  let hit=0;for(const w of A)if(B.has(w))hit++;
  return Math.min(1,hit/Math.sqrt(A.size*B.size)+(norm(a)===norm(b)?0.5:0));
}

function learnedAnswer(q){
  let samples=[];try{samples=Dataset.samples()}catch{return null}
  const scored=samples.map(s=>({...s,score:Math.max(similarity(q,s.topic),similarity(q,s.text),similarity(q,s.topic+' '+s.text))})).filter(x=>x.score>=0.34).sort((a,b)=>b.score-a.score);
  if(!scored.length)return null;
  const best=scored[0];
  return `From my local knowledge:\n\n${best.text}\n\n_Confidence: ${Math.round(best.score*100)}%_`;
}

function recentContext(history=[]){return history.slice(-12).map(x=>`${x.role}: ${clean(x.content)}`).join('\n')}
function usable(text){const x=clean(text);if(x.length<3||/^\[/.test(x))return false;if(/^(i am still learning|teach me more examples)[.!]?$/i.test(x))return false;return /[\p{L}]/u.test(x)}

export async function answer(prompt,history=[]){
  const q=clean(prompt);if(!q)return 'Please type a message and I will try to help.';
  const taught=teachCommand(q);if(taught)return taught;
  const remembered=remember(q);if(remembered)return 'Got it. I will remember that. 🧠';
  const mem=memoryAnswer(q);if(mem)return mem;
  const math=safeMath(q);if(math!==null)return `The result is ${math}.`;
  if(exact.greeting.test(q))return 'Hello! 👋 I am gAi. How can I help you?';
  if(exact.thanks.test(q))return 'You are welcome! 😊';
  if(exact.identity.test(q))return 'I am gAi, a local growing AI. I learn from knowledge you teach me and improve through local training. 🧠';
  if(exact.status.test(q))return 'I am ready. My local brain is active and I am learning from saved knowledge. 🧠';
  if(exact.capabilities.test(q))return 'I can chat, remember selected facts, learn new lessons, calculate basic expressions, retrieve learned knowledge, and generate responses locally without an external AI API.';
  if(exact.farewell.test(q))return 'Goodbye! 👋';
  if(exact.night.test(q))return 'Good night! 🌙';
  const learned=learnedAnswer(q);if(learned)return learned;
  const context=recentContext(history),full=context?context+'\nuser: '+q:q;
  let out='';
  try{out=String(Transformer.generate(full,48,{temperature:.68,topK:8,topP:.9,repeatPenalty:.85})||'').trim()}catch{}
  if(usable(out))return out;
  try{out=String(TinyLM.generate(q,48)||'').trim()}catch{}
  if(usable(out))return out;
  return 'I do not know that yet. Teach me with `teach: topic | answer` and I will save it to my local knowledge. 🧠';
}

export function stats(){const m=memory();let ds={samples:0,topics:0};try{ds=Dataset.stats()}catch{}return{memoryFacts:m.facts.length,datasetSamples:ds.samples||0,datasetTopics:ds.topics||0};}
export function clearMemory(){try{localStorage.removeItem(MEMORY_KEY)}catch{}}
export default{answer,stats,clearMemory};
