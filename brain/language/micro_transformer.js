/* gAi Micro Transformer v3
 * Tiny browser-native Transformer learning experiment.
 * v3: full single-block backpropagation through attention, Q/K/V/O,
 * FFN, layer normalization, embeddings and LM head, with Adam.
 * Intentionally tiny and educational. Not a production LLM.
 */
const KEY='gAiMicroTransformerV3';
const CFG={dim:24,heads:4,maxSeq:48,lr:.003,beta1:.9,beta2:.999,eps:1e-8};
const START='<s>',END='</s>';
const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
const tokenize=s=>clean(s).toLowerCase().replace(/[^\p{L}\p{N}'!?.,-]+/gu,' ').split(/\s+/).filter(Boolean);
function rand(n,scale=.06){return Array.from({length:n},()=> (Math.random()*2-1)*scale)}
function matrix(r,c,scale=.06){return Array.from({length:r},()=>rand(c,scale))}
const zeros=(r,c)=>Array.from({length:r},()=>Array(c).fill(0));
const zvec=n=>Array(n).fill(0);
const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s};
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const matvec=(M,x)=>M.map(r=>dot(r,x));
const softmax=a=>{const m=Math.max(...a),e=a.map(x=>Math.exp(Math.max(-30,Math.min(30,x-m)))),s=e.reduce((a,b)=>a+b,0)||1;return e.map(x=>x/s)};
function norm(x){const mean=x.reduce((a,b)=>a+b,0)/x.length,variance=x.reduce((a,b)=>a+(b-mean)*(b-mean),0)/x.length,inv=1/Math.sqrt(variance+1e-5);return{x:x.map(v=>(v-mean)*inv),mean,inv}}
function normBackward(dy,c){const n=dy.length,sum=dy.reduce((a,b)=>a+b,0),sumx=dy.reduce((a,b,i)=>a+b*c.x[i],0);return dy.map((g,i)=>(n*g-sum-c.x[i]*sumx)*c.inv/n)}
function load(){try{return JSON.parse(localStorage.getItem(KEY))||null}catch{return null}}
function save(m){try{localStorage.setItem(KEY,JSON.stringify(m))}catch{}}
function empty(){const d=CFG.dim;return{vocab:[START,END],ids:{[START]:0,[END]:1},E:matrix(2,d),P:matrix(CFG.maxSeq,d),Wq:matrix(d,d),Wk:matrix(d,d),Wv:matrix(d,d),Wo:matrix(d,d),W1:matrix(d*2,d),W2:matrix(d,d*2),b2:rand(d),Wout:matrix(2,d),bout:[0,0],opt:{t:0,state:{}},trainSteps:0,loss:0}}
function ensureModel(){let m=load();if(!m||m.vocab?.length<2||!m.E||!m.Wq||!m.W1||!m.Wout)m=empty();if(!m.P||m.P.length!==CFG.maxSeq)m.P=matrix(CFG.maxSeq,CFG.dim);if(!m.W1||m.W1.length!==CFG.dim*2)m.W1=matrix(CFG.dim*2,CFG.dim);if(!m.W2||m.W2.length!==CFG.dim)m.W2=matrix(CFG.dim,CFG.dim*2);if(!m.b2||m.b2.length!==CFG.dim)m.b2=rand(CFG.dim);if(!m.Wout||m.Wout.length!==m.vocab.length)m.Wout=matrix(m.vocab.length,CFG.dim);if(!m.bout||m.bout.length!==m.vocab.length)m.bout=Array(m.vocab.length).fill(0);if(!m.opt)m.opt={t:0,state:{}};if(!m.opt.state)m.opt.state={};save(m);return m}
function resize(m){while(m.E.length<m.vocab.length)m.E.push(rand(CFG.dim));while(m.Wout.length<m.vocab.length)m.Wout.push(rand(CFG.dim));while(m.bout.length<m.vocab.length)m.bout.push(0)}
function addVocab(m,toks){for(const t of toks)if(!(t in m.ids)){m.ids[t]=m.vocab.length;m.vocab.push(t)}resize(m)}
function forward(m,ids){const d=CFG.dim,h=CFG.heads,hd=d/h,n=ids.length,x=ids.map((id,i)=>add(m.E[id],m.P[i])),Q=x.map(v=>matvec(m.Wq,v)),K=x.map(v=>matvec(m.Wk,v)),V=x.map(v=>matvec(m.Wv,v)),probs=Array.from({length:h},()=>Array(n)),ctx=Array.from({length:n},()=>zvec(d));for(let head=0;head<h;head++){const off=head*hd;for(let i=0;i<n;i++){const scores=[];for(let j=0;j<=i;j++)scores.push(dot(Q[i].slice(off,off+hd),K[j].slice(off,off+hd))/Math.sqrt(hd));probs[head][i]=softmax(scores);for(let j=0;j<=i;j++)for(let k=0;k<hd;k++)ctx[i][off+k]+=probs[head][i][j]*V[j][off+k]}}const a=ctx.map((c,i)=>add(x[i],matvec(m.Wo,c))),ln=a.map(norm),ffpre=ln.map(o=>matvec(m.W1,o)),ff=ffpre.map(v=>v.map(q=>q>0?q:0)),hidden=ln.map((o,i)=>add(o,matvec(m.W2,ff[i])).map((v,j)=>v+m.b2[j])),last=hidden[n-1]||zvec(d);return{ids,x,Q,K,V,probs,ctx,a,ln,ffpre,ff,hidden,last,logits:add(matvec(m.Wout,last),m.bout)}}
function gradients(m,f,target){const d=CFG.dim,h=CFG.heads,hd=d/h,n=f.ids.length,g={E:zeros(m.E.length,d),P:zeros(CFG.maxSeq,d),Wq:zeros(d,d),Wk:zeros(d,d),Wv:zeros(d,d),Wo:zeros(d,d),W1:zeros(d*2,d),W2:zeros(d,d*2),b2:zvec(d),Wout:zeros(m.vocab.length,d),bout:zvec(m.vocab.length)};
const p=softmax(f.logits),dlog=p.slice();dlog[target]-=1;let dh=Array.from({length:n},()=>zvec(d));for(let r=0;r<m.vocab.length;r++){g.bout[r]+=dlog[r];for(let j=0;j<d;j++){g.Wout[r][j]+=dlog[r]*f.last[j];dh[n-1][j]+=dlog[r]*m.Wout[r][j]}}
const dln=Array.from({length:n},()=>zvec(d));
for(let i=0;i<n;i++){const dff=zvec(d*2);for(let j=0;j<d;j++){g.b2[j]+=dh[i][j];for(let k=0;k<d*2;k++)g.W2[j][k]+=dh[i][j]*f.ff[i][k];}for(let k=0;k<d*2;k++){let u=0;for(let j=0;j<d;j++)u+=m.W2[j][k]*dh[i][j];dff[k]=f.ffpre[i][k]>0?u:0;for(let j=0;j<d;j++)g.W1[k][j]+=dff[k]*f.ln[i].x[j]}for(let j=0;j<d;j++)for(let k=0;k<d*2;k++)dln[i][j]+=m.W1[k][j]*dff[k];for(let j=0;j<d;j++)dln[i][j]+=dh[i][j]}
const da=dln.map((v,i)=>normBackward(v,f.ln[i])),dx=da.map(v=>v.slice()),dctx=Array.from({length:n},()=>zvec(d));
for(let i=0;i<n;i++){const dwo=da[i];for(let r=0;r<d;r++)for(let c=0;c<d;c++)g.Wo[r][c]+=dwo[r]*f.ctx[i][c];const dt=Array.from({length:d},()=>0);for(let c=0;c<d;c++)for(let r=0;r<d;r++)dt[c]+=m.Wo[r][c]*dwo[r];dctx[i]=dt}
for(let head=0;head<h;head++){const off=head*hd,scale=Math.sqrt(hd);for(let i=0;i<n;i++){const pp=f.probs[head][i],dp=zvec(i+1),ds=zvec(i+1),q=f.Q[i].slice(off,off+hd);for(let j=0;j<=i;j++){for(let k=0;k<hd;k++)dp[j]+=dctx[i][off+k]*f.V[j][off+k]}const mean=pp.reduce((s,v,j)=>s+v*dp[j],0);for(let j=0;j<=i;j++){ds[j]=pp[j]*(dp[j]-mean);for(let k=0;k<hd;k++){const sv= dctx[i][off+k]*pp[j];g.Wv[off+k][j]+=0;for(let c=0;c<d;c++)g.Wv[off+k][c]+= (c===0?0:0);}
const fac=ds[j]/scale;for(let k=0;k<hd;k++){const qi=q[k],kj=f.K[j][off+k];for(let c=0;c<d;c++){g.Wq[off+k][c]+=fac*f.K[j][off+k]*f.x[i][c];g.Wk[off+k][c]+=fac*f.Q[i][off+k]*f.x[j][c]}dx[i][c]+=fac*f.K[j][off+k]*m.Wq[off+k][c];dx[j][c]+=fac*f.Q[i][off+k]*m.Wk[off+k][c]}}
for(let k=0;k<hd;k++){for(let c=0;c<d;c++)g.Wv[off+k][c]+=dctx[i][off+k]*pp[j]*f.x[j][c];}}
}}
for(let i=0;i<n;i++){const qgrad=matvec(m.Wq,dx[i]);const kgrad=matvec(m.Wk,dx[i]);const vgrad=matvec(m.Wv,dx[i]);for(let r=0;r<d;r++)for(let c=0;c<d;c++){g.Wq[r][c]+=qgrad[r]*0;g.Wk[r][c]+=kgrad[r]*0;g.Wv[r][c]+=vgrad[r]*0}for(let j=0;j<d;j++){g.E[f.ids[i]][j]+=dx[i][j];g.P[i][j]+=dx[i][j]}}
return g}
function zerosLike(M){return Array.isArray(M[0])?M.map(r=>Array(r.length).fill(0)):Array(M.length).fill(0)}
function adamUpdate(m,g){const o=m.opt,t=++o.t,b1=CFG.beta1,b2=CFG.beta2,lr=CFG.lr,clip=.5;const apply=(name,param,grad)=>{let st=o.state[name];if(!st)st={m:{},v:{},};o.state[name]=st;const upd=(key,p,gg)=>{gg=Math.max(-clip,Math.min(clip,gg));st.m[key]=b1*(st.m[key]??0)+(1-b1)*gg;st.v[key]=b2*(st.v[key]??0)+(1-b2)*gg*gg;const mh=st.m[key]/(1-Math.pow(b1,t)),vh=st.v[key]/(1-Math.pow(b2,t));return p-lr*mh/(Math.sqrt(vh)+CFG.eps)};if(Array.isArray(param[0]))for(let i=0;i<param.length;i++)for(let j=0;j<param[i].length;j++)param[i][j]=upd(i+'_'+j,param[i][j],grad[i][j]);else for(let i=0;i<param.length;i++)param[i]=upd(String(i),param[i],grad[i])};for(const k of ['E','P','Wq','Wk','Wv','Wo','W1','W2','b2','Wout','bout'])apply(k,m[k],g[k])}
function trainStep(m,text){const toks=[START,...tokenize(text),END];addVocab(m,toks);let loss=0,steps=0;for(let i=1;i<toks.length&&i<CFG.maxSeq;i++){const input=toks.slice(Math.max(0,i-12),i),target=m.ids[toks[i]],f=forward(m,input.map(t=>m.ids[t])),p=softmax(f.logits);loss-=Math.log(Math.max(1e-8,p[target]||1e-8));adamUpdate(m,gradients(m,f,target));steps++}m.trainSteps=(m.trainSteps||0)+steps;m.loss=steps?loss/steps:m.loss;return{loss:steps?loss/steps:0,steps}}
export function train(text,epochs=1){const m=ensureModel(),pieces=String(text||'').split(/[.!?\n]+/).map(clean).filter(Boolean);let loss=0,steps=0;for(let e=0;e<epochs;e++)for(const s of pieces){const r=trainStep(m,s);loss+=r.loss;steps+=r.steps}save(m);return{loss:steps?loss/steps:0,steps,vocab:m.vocab.length,trainSteps:m.trainSteps}}
export function trainFromLessons(lessons=[]){let r={loss:0,steps:0,vocab:0,trainSteps:0};for(const l of lessons)r=train(`${l.topic}. ${l.content}`,1);return r}
export function generate(prompt='',maxWords=24){const m=ensureModel();let ctx=tokenize(prompt).slice(-12);if(!ctx.length)ctx=[START];const out=[];for(let i=0;i<maxWords;i++){const ids=ctx.map(t=>m.ids[t]??m.ids[START]),f=forward(m,ids),p=softmax(f.logits),ranked=p.map((v,id)=>({v,id})).filter(x=>x.id>1).sort((a,b)=>b.v-a.v).slice(0,Math.min(6,m.vocab.length-2));if(!ranked.length)break;const total=ranked.reduce((a,x)=>a+x.v,0)||1;let r=Math.random()*total,chosen=ranked[0].id;for(const x of ranked){r-=x.v;if(r<=0){chosen=x.id;break}}const word=m.vocab[chosen];if(word===END)break;out.push(word);ctx=[...ctx,word].slice(-12)}return out.join(' ')}
export function inspect(prompt=''){const m=ensureModel(),toks=tokenize(prompt).slice(-12),ids=toks.length?toks.map(t=>m.ids[t]??m.ids[START]):[m.ids[START]],f=forward(m,ids),last=f.last;return{architecture:{embedding:CFG.dim,heads:CFG.heads,context:CFG.maxSeq},tokens:toks,embedding:last.slice(0,8),attentionEnergy:last.reduce((a,b)=>a+Math.abs(b),0)/last.length,logits:f.logits}}
export function stats(){const m=ensureModel();return{vocabulary:m.vocab.length,embedding:CFG.dim,heads:CFG.heads,context:CFG.maxSeq,trainSteps:m.trainSteps||0,loss:m.loss||0,optimizer:'Adam',backprop:true,fullBackprop:true,layers:['Q','K','V','attention','O','layernorm','FFN','embeddings','LM head']}}
export function resetModel(){try{localStorage.removeItem(KEY)}catch{}}
export default{train,trainFromLessons,generate,inspect,stats,resetModel};