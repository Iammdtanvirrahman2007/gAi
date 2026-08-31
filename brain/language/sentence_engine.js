/* gAi English Sentence Engine
 * Builds English sentences from meaning, intent, context and register.
 * It uses reusable grammar patterns rather than a phrase -> reply dictionary.
 */

const AUX = new Set(['am','is','are','was','were','have','has','had','do','does','did','can','could','will','would','should','may','might','must']);
function clean(v=''){return String(v).replace(/\s+/g,' ').trim()}
function punct(v,m='.'){return clean(v).replace(/[.!?]+$/,'')+m}
function isVerbPhrase(s=''){
 const first=clean(s).toLowerCase().split(' ')[0];
 return AUX.has(first)||/^(learn|know|understand|think|need|want|like|use|help|answer|explain|build|make|study|work|go|come|see|feel)\b/i.test(s)
}

export const SentenceEngine={
 detectLanguage(text=''){
  const bn=(text.match(/[\u0980-\u09FF]/g)||[]).length,en=(text.match(/[A-Za-z]/g)||[]).length;
  if(bn&&en)return 'mixed'; if(bn)return 'bn'; return 'en';
 },
 plan({intent='statement',subject,predicate,object,tense='present',register='polite',language='en',context=''}={}){return{intent,subject,predicate,object,tense,register,language,context}},
 build(plan={}){return this.buildEnglish({...plan,language:'en'})},
 buildEnglish(p={}){
  const intent=p.intent||'statement',subject=clean(p.subject||'I'),predicate=clean(p.predicate||'am learning'),object=clean(p.object||'');
  if(intent==='question'){
   if(/^(what|why|how|when|where|who|which)\b/i.test(predicate))return punct(`${predicate}${object?' '+object:''}`,'?');
   if(isVerbPhrase(predicate)){const parts=predicate.split(' ');return punct(`${parts[0]} ${subject.toLowerCase()} ${parts.slice(1).join(' ')}${object?' '+object:''}`,'?')}
   return punct(`Does ${subject.toLowerCase()} ${predicate}${object?' '+object:''}`,'?');
  }
  if(intent==='request')return punct(`Please ${predicate}${object?' '+object:''}`);
  if(intent==='command')return punct(`${predicate}${object?' '+object:''}`);
  if(intent==='negative'){
   if(/^(am|is|are|was|were|have|has|can|will|should|could|would)\b/i.test(predicate))return punct(`${subject} ${predicate.replace(/^(am|is|are|was|were|have|has|can|will|should|could|would)\b/i,'$1 not')}${object?' '+object:''}`);
   return punct(`${subject} does not ${predicate}${object?' '+object:''}`)
  }
  return punct(`${subject} ${predicate}${object?' '+object:''}`)
 },
 fromMeaning(meaning={}){return this.build(this.plan(meaning))},
 join(sentences=[]){const a=sentences.filter(Boolean).map(s=>clean(s).replace(/[.!?]+$/,''));return a.length?a.join('. ')+'.':''},
 compose({acknowledgement='',main='',closing=''}={}){return this.join([acknowledgement,main,closing])}
};
export default SentenceEngine;
