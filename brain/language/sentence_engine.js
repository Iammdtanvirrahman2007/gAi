/* gAi Sentence Construction Engine
 * Principle-based sentence generation. It does not keep a response for every phrase.
 * The engine plans a sentence from intent, language, register and semantic slots.
 */

export const SentenceEngine = {
  detectLanguage(text = '') {
    const bn = (text.match(/[\u0980-\u09FF]/g) || []).length;
    const en = (text.match(/[A-Za-z]/g) || []).length;
    if (bn && en) return 'mixed';
    if (bn) return 'bn';
    return 'en';
  },

  plan({ intent = 'statement', subject, predicate, object, tense = 'present', register = 'polite', language = 'auto' } = {}) {
    return { intent, subject, predicate, object, tense, register, language };
  },

  build(plan) {
    const lang = plan.language === 'auto' ? this.detectLanguage([plan.subject, plan.predicate, plan.object].filter(Boolean).join(' ')) : plan.language;
    if (lang === 'bn') return this.buildBangla(plan);
    return this.buildEnglish(plan);
  },

  buildBangla(p) {
    const subject = p.subject || 'আমি';
    const predicate = p.predicate || 'শিখছি';
    const object = p.object ? ` ${p.object}` : '';
    const ending = p.intent === 'question' ? '?' : '।';
    return `${subject} ${predicate}${object}${ending}`.replace(/\s+/g, ' ').trim();
  },

  buildEnglish(p) {
    const subject = p.subject || 'I';
    const predicate = p.predicate || 'am learning';
    const object = p.object ? ` ${p.object}` : '';
    const ending = p.intent === 'question' ? '?' : '.';
    return `${subject} ${predicate}${object}${ending}`.replace(/\s+/g, ' ').trim();
  },

  // Creates a sentence plan from structured meaning rather than memorized replies.
  fromMeaning(meaning = {}) {
    return this.build(this.plan(meaning));
  }
};

export default SentenceEngine;
