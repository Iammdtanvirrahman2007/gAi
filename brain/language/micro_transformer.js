/* gAi Micro Transformer v2
 * Tiny browser-native Transformer learning experiment.
 * v2: trainable LM head + cross-entropy backprop + Adam optimizer.
 * Intentionally tiny, educational, and local. Not a production LLM.
 */
const KEY='gAiMicroTransformerV2';
const CFG={dim:24,heads:4,maxSeq:48,lr:.008,beta1:.9,beta2:.999,eps:1e-8};
const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
const tokenize=s=>clean(s).toLowerCase().replace(/[^\p{L}\p{N}'!?.,-]+/gu,' ').split(/\s+/).filter(Boolean);
const START='<s>',END='</s>';
function rand(n,scale=.08){return Array.from({length:n},()=> (Math.random()*2-1)*scale)}
function matrix(r,c,scale=.08){return Array.from({length:r},()=>rand(c,scale))}
function zeros(r,c){return Array.from({length:r},()=>Array(c).fill(0))}
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
function softmax(a){const m=Math.max(...a),e=a.map(x=>Math.exp(Math.max(-30,Math.min(30,x-m)))),z=e.reduce((a,b)=>a+b,0)||1;return e.map(x=>x/z)}
function add(a,b){return a.map((x,i)=>x+b[i])}
function matvec(M,x){return M.map(r=>dot(r,x))}
function relu(x){return x>0?x:0}
function norm(x){const m=x.reduce((a,b)=>a+b,0)/x.length,v=x.reduce((a,b)=>a+(b-m)*(b-m),0)/x.length,d=Math.sqrt(v+1e-5);return x.map(z=>(z-m)/d)}
function load(){try{return JSON.parse(localStorage.getItem(KEY))||null}catch{return null}}
function save(m){try{localStorage.setItem(KEY,JSON.stringify(m))}catch{}}
function adamState(m){if(!m.opt)m.opt={t:0,mW:[],vW:[],mb:[],vb:[]};while(m.opt.mW.length<m.vocab.length){m.opt.mW.push(Array(CFG.dim).fill(0));m.opt.vW.push(Array(CFG.dim).fill(0))}while(m.opt.mb.length< m.vocab.length)m.opt.mb.push(0);while(m.opt.vb.length< m.vocab.length)m.opt.vb.push(0)}
function empty(){return{vocab:[START,END],ids:{[START]:0,[END]:1},E:[],P:[],Wq:[],Wk:[],Wv:[],Wo:[],W1:[],W2:[],b2:[],Wout:[],bout:[],opt:null,trainSteps:0,loss:0}}
function ensureModel(){let m=load();if(!m||!m.vocab||!m.ids||!Array.isArray(m.E)||!m.Wq){m=empty();const d=CFG.dim;m.E=matrix(m.vocab.length,d);m.P=matrix(CFG.maxSeq,d);m.Wq=matrix(d,d);m.Wk=matrix(d,d);m.Wv=matrix(d,d);m.Wo=matrix(d,d);m.W1=matrix(d*2,d);m.W2=matrix(d,d*2);m.b2=rand(d);m.Wout=matrix(m.vocab.length,d);m.bout=Array(m.vocab.length).fill(0);save(m)}if(!Array.isArray(m.Wout)||m.Wout.length!==m.vocab.length)m.Wout=matrix(m.vocab.length,CFG.dim);if(!Array.isArray(m.bout)||m.bout.length!==m.vocab.length)m.bout=Array(m.vocab.length).fill(0);adamState(m);return m}
function resize(m){while(m.E.length<m.vocab.length)m.E.push(rand(CFG.dim));while(m.Wout.length<m.vocab.length)m.Wout.push(rand(CFG.dim));while(m.bout.length<m.vocab.length)m.bout.push(0);adamState(m)}
function addVocab(m,toks){for(const t of toks)if(!(t in m.ids)){m.ids[t]=m.vocab.length;m.vocab.push(t)}resize(m)}
function attentionBlock(m,ids){const d=CFG.dim,h=CFG.heads,hd=d/h,x=ids.map((id,i)=>add(m.E[id].slice(),m.P[i])),heads=[];for(let head=0;head<h;head++){const q=x.map(v=>matvec(m.Wq,v).slice(head*hd,(head+1)*hd)),k=x.map(v=>matvec(m.Wk,v).slice(head*hd,(head+1)*hd)),vv=x.map(v=>matvec(m.Wv,v).slice(head*hd,(head+1)*hd)),out=[];for(let i=0;i<x.length;i++){const scores=[];for(let j=0;j<=i;j++)scores.push(dot(q[i],k[j])/Math.sqrt(hd));const p=softmax(scores),z=Array(hd).fill(0);p.forEach((w,j)=>z.forEach((_,k2)=>z[k2]+=w*vv[j][k2]));out.push(z)}heads.push(out)}const y=[];for(let i=0;i<x.length;i++){const cat=[];for(let head=0;head<h;head++)cat.push(...heads[head][i]);y.push(add(x[i],matvec(m.Wo,cat)))}return y.map(v=>{const n=norm(v),ff=matvec(m.W1,n).map(relu);return add(n,matvec(m.W2,ff)).map((q,j)=>q+m.b2[j])})}
function forward(m,tokens){const ids=tokens.map(t=>m.ids[t]??m.ids[END]).slice(0,CFG.maxSeq),hidden=attentionBlock(m,ids),h=hidden.at(-1)||Array(CFG.dim).fill(0);return{ids,hidden,last:h,logits:add(matvec(m.Wout,h),m.bout)}}
function adamW(m,row,g,j,t){const o=m.opt,b1=CFG.beta1,b2=CFG.beta2;o.mW[row][j]=b1*o.mW[row][j]+(1-b1)*g;o.vW[row][j]=b2*o.vW[row][j]+(1-b2)*g*g;const mh=o.mW[row][j]/(1-Math.pow(b1,t)),vh=o.vW[row][j]/(1-Math.pow(b2,t));m.Wout[row][j]-=CFG.lr*mh/(Math.sqrt(vh)+CFG.eps)}
function adamB(m,row,g,t){const o=m.opt,b1=CFG.beta1,b2=CFG.beta2;o.mb[row]=b1*o.mb[row]+(1-b1)*g;o.vb[row]=b2*o.vb[row]+(1-b2)*g*g;const mh=o.mb[row]/(1-Math.pow(b1,t)),vh=o.vb[row]/(1-Math.pow(b2,t));m.bout[row]-=CFG.lr*mh/(Math.sqrt(vh)+CFG.eps)}
function trainStep(m,text){const toks=[START,...tokenize(text),END];addVocab(m,toks);let loss=0,steps=0;for(let i=1;i<toks.length&&i<CFG.maxSeq;i++){const input=toks.slice(Math.max(0,i-12),i),target=m.ids[toks[i]],f=forward(m,input),p=softmax(f.logits);loss-=Math.log(Math.max(1e-8,p[target]||1e-8));const grad=p.slice();grad[target]-=1;adamState(m);const t=++m.opt.t;for(let row=0;row<m.vocab.length;row++){const g=grad[row];if(Math.abs(g)>1e-8)for(let j=0;j<CFG.dim;j++)adamW(m,row,g*f.last[j],j,t);adamB(m,row,g,t)}const gh=Array(CFG.dim).fill(0);for(let row=0;row<m.vocab.length;row++)for(let j=0;j<CFG.dim;j++)gh[j]+=grad[row]*m.Wout[row][j];const lastId=f.ids.at(-1);if(lastId!=null)for(let j=0;j<CFG.dim;j++)m.E[lastId][j]-=Math.max(-.02,Math.min(.02,CFG.lr*gh[j]));steps++}m.trainSteps=(m.trainSteps||0)+steps;m.loss=steps?loss/steps:m.loss;return{loss:steps?loss/steps:0,steps}}
export function train(text,epochs=1){const m=ensureModel(),pieces=String(text||'').split(/[.!?\n]+/).map(clean).filter(Boolean);let loss=0,steps=0;for(let e=0;e<epochs;e++)for(const s of pieces){const r=trainStep(m,s);loss+=r.loss;steps+=r.steps}save(m);return{loss:steps?loss/steps:0,steps,vocab:m.vocab.length,trainSteps:m.trainSteps}}
export function trainFromLessons(lessons=[]){let r={loss:0,steps:0,vocab:0,trainSteps:0};for(const l of lessons)r=train(`${l.topic}. ${l.content}`,1);return r}
export function generate(prompt='',maxWords=24){const m=ensureModel();let ctx=tokenize(prompt).slice(-12);if(!ctx.length)ctx=[START];const out=[];for(let i=0;i<maxWords;i++){const f=forward(m,ctx),p=softmax(f.logits),ranked=p.map((v,id)=>({v,id})).filter(x=>x.id>1).sort((a,b)=>b.v-a.v).slice(0,Math.min(6,m.vocab.length-2));if(!ranked.length)break;const total=ranked.reduce((a,x)=>a+x.v,0)||1;let r=Math.random()*total,chosen=ranked[0].id;for(const x of ranked){r-=x.v;if(r<=0){chosen=x.id;break}}const word=m.vocab[chosen];if(word===END)break;out.push(word);ctx=[...ctx,word].slice(-12)}return out.join(' ')}
export function inspect(prompt=''){const m=ensureModel(),toks=tokenize(prompt).slice(-12),f=forward(m,toks.length?toks:[START]),last=f.last;return{architecture:{embedding:CFG.dim,heads:CFG.heads,context:CFG.maxSeq},tokens:toks,embedding:last.slice(0,8),attentionEnergy:last.map(Math.abs).reduce((a,b)=>a+b,0)/last.length,logits:f.logits}}
export function stats(){const m=ensureModel();return{vocabulary:m.vocab.length,embedding:CFG.dim,heads:CFG.heads,context:CFG.maxSeq,trainSteps:m.trainSteps||0,loss:m.loss||0,optimizer:'Adam',backprop:true}}
export function resetModel(){try{localStorage.removeItem(KEY)}catch{}}
export default{train,trainFromLessons,generate,inspect,stats,resetModel};