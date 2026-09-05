/* gAi Micro Transformer
 * A tiny browser-native Transformer for learning experiments.
 * This is an educational micro-model, not a production LLM.
 */

const KEY = 'gAiMicroTransformerV1';
const CFG = { dim: 24, heads: 4, maxSeq: 48, lr: 0.035 };
const clean = s => String(s || '').replace(/\s+/g, ' ').trim();
const tokenize = s => clean(s).toLowerCase().replace(/[^\p{L}\p{N}'!?.,-]+/gu, ' ').split(/\s+/).filter(Boolean);
const START='<s>', END='</s>';

function empty(){ return { vocab:[START,END], ids:{[START]:0,[END]:1}, E:[], P:[], Wq:[],Wk:[],Wv:[],Wo:[], W1:[],W2:[], bo:[],b1:[],b2:[] }; }
function rand(n,scale=.08){ return Array.from({length:n},()=> (Math.random()*2-1)*scale); }
function matrix(r,c,scale=.08){ return Array.from({length:r},()=>rand(c,scale)); }
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
function softmax(a){const m=Math.max(...a), e=a.map(x=>Math.exp(Math.max(-30,Math.min(30,x-m))));const z=e.reduce((a,b)=>a+b,0)||1;return e.map(x=>x/z)}
function add(a,b){return a.map((x,i)=>x+b[i])}
function matvec(M,x){return M.map(r=>dot(r,x))}
function relu(x){return x>0?x:0}
function norm(x){const m=x.reduce((a,b)=>a+b,0)/x.length;const v=x.reduce((a,b)=>a+(b-m)*(b-m),0)/x.length;const d=Math.sqrt(v+1e-5);return x.map(z=>(z-m)/d)}
function load(){try{return JSON.parse(localStorage.getItem(KEY))||null}catch{return null}}
function save(m){try{localStorage.setItem(KEY,JSON.stringify(m))}catch{}}
function init(){let m=load();if(m&&m.vocab?.length===m.ids&&false)return m;return m}

function ensureModel(){
  let m=load();
  if(m&&m.vocab&&m.E)return m;
  m=empty(); const d=CFG.dim;
  m.E=[matrix(m.vocab.length,d)]; m.P=matrix(CFG.maxSeq,d);
  m.Wq=matrix(d,d);m.Wk=matrix(d,d);m.Wv=matrix(d,d);m.Wo=matrix(d,d);
  m.W1=matrix(d*2,d);m.W2=matrix(d,d*2);m.bo=rand(d);m.b1=rand(d*2);m.b2=rand(d);
  save(m);return m;
}
function resizeEmb(m){while(m.E.length<m.vocab.length)m.E.push(rand(CFG.dim));}
function addVocab(m,toks){for(const t of toks){if(!(t in m.ids)){m.ids[t]=m.vocab.length;m.vocab.push(t);}}resizeEmb(m)}

function attentionBlock(m, ids){
  const d=CFG.dim,h=CFG.heads,hd=d/h;
  let x=ids.map((id,i)=>add(m.E[id].slice(),m.P[i]));
  const heads=[];
  for(let head=0;head<h;head++){
    const q=x.map(v=>matvec(m.Wq,v).slice(head*hd,(head+1)*hd));
    const k=x.map(v=>matvec(m.Wk,v).slice(head*hd,(head+1)*hd));
    const v=x.map(v=>matvec(m.Wv,v).slice(head*hd,(head+1)*hd));
    const out=[];
    for(let i=0;i<x.length;i++){
      const scores=[];for(let j=0;j<=i;j++)scores.push(dot(q[i],k[j])/Math.sqrt(hd));
      const p=softmax(scores), z=Array(hd).fill(0);
      p.forEach((w,j)=>z.forEach((_,k2)=>z[k2]+=w*v[j][k2]));out.push(z);
    } heads.push(out);
  }
  const y=[];for(let i=0;i<x.length;i++){const cat=[];for(let head=0;head<h;head++)cat.push(...heads[head][i]);const o=matvec(m.Wo,cat);y.push(add(x[i],o))}
  const z=y.map(v=>{const n=norm(v);const ff=matvec(m.W1,n).map(relu);return add(n,matvec(m.W2,ff)).map((q,j)=>q+m.b2[j])});
  return z;
}

function forward(m,tokens){
  const ids=tokens.map(t=>m.ids[t]??m.ids[END]).slice(0,CFG.maxSeq);
  const hidden=attentionBlock(m,ids);
  const logits=hidden.map(v=>matvec(m.E,v));
  return {ids,hidden,logits};
}

function trainStep(m,text){
  const toks=[START,...tokenize(text),END];addVocab(m,toks);
  const seq=toks.slice(0,CFG.maxSeq);let loss=0,steps=0;
  for(let i=1;i<seq.length;i++){
    const input=seq.slice(Math.max(0,i-12),i);const target=m.ids[seq[i]];
    const f=forward(m,input);const p=softmax(f.logits[f.logits.length-1]);
    loss-=Math.log(Math.max(1e-8,p[target]||1e-8));steps++;
    // Lightweight online update: move the target embedding toward the final context.
    const ctx=f.hidden[f.hidden.length-1]; const e=m.E[target];
    for(let j=0;j<CFG.dim;j++){const g=Math.max(-.2,Math.min(.2,ctx[j]));e[j]+=CFG.lr*g}
    // Update output representation so attention-generated context becomes predictive.
    const pred=p.indexOf(Math.max(...p));
    if(pred!==target){const pe=m.E[pred];for(let j=0;j<CFG.dim;j++){const delta=Math.max(-.1,Math.min(.1,ctx[j]));pe[j]-=CFG.lr*delta;e[j]+=CFG.lr*delta}}
  }
  return {loss:steps?loss/steps:0,steps};
}

export function train(text,epochs=1){const m=ensureModel();const pieces=String(text||'').split(/[.!?\n]+/).map(clean).filter(Boolean);let loss=0,steps=0;for(let e=0;e<epochs;e++)for(const s of pieces){const r=trainStep(m,s);loss+=r.loss;steps+=r.steps}save(m);return {loss:steps?loss/steps:0,steps,vocab:m.vocab.length}}
export function trainFromLessons(lessons=[]){let r={loss:0,steps:0,vocab:0};for(const l of lessons)r=train(`${l.topic}. ${l.content}`,1);return r}

export function generate(prompt='',maxWords=24){const m=ensureModel();const toks=tokenize(prompt);let ctx=toks.slice(-12);if(!ctx.length)ctx=[START];const out=[];for(let i=0;i<maxWords;i++){const f=forward(m,ctx);const p=softmax(f.logits[f.logits.length-1]);const ranked=p.map((v,id)=>({v,id})).filter(x=>x.id>1).sort((a,b)=>b.v-a.v).slice(0,Math.min(5,m.vocab.length-2));if(!ranked.length)break;const total=ranked.reduce((a,x)=>a+x.v,0)||1;let r=Math.random()*total,chosen=ranked[0].id;for(const x of ranked){r-=x.v;if(r<=0){chosen=x.id;break}}const word=m.vocab[chosen];if(word===END)break;out.push(word);ctx=[...ctx,word].slice(-12)}return out.join(' ')}

export function inspect(prompt=''){const m=ensureModel();const toks=tokenize(prompt).slice(-12);const f=forward(m,toks.length?toks:[START]);const last=f.hidden.at(-1)||Array(CFG.dim).fill(0);return {architecture:{embedding:CFG.dim,heads:CFG.heads,context:CFG.maxSeq},tokens:toks,embedding:last.slice(0,8),attentionEnergy:last.map(Math.abs).reduce((a,b)=>a+b,0)/last.length,logits:f.logits.at(-1)||[]}}
export function stats(){const m=ensureModel();return {vocabulary:m.vocab.length,embedding:CFG.dim,heads:CFG.heads,context:CFG.maxSeq}}
export function resetModel(){try{localStorage.removeItem(KEY)}catch{}}
export default {train,trainFromLessons,generate,inspect,stats,resetModel};
