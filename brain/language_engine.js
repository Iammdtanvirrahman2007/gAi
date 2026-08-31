/*
 * gAi Language Engine: lightweight, browser-safe primitives.
 *
 * This is an original implementation of general NLP concepts. It does not
 * contain third-party model code or weights. Heavy ML capabilities should be
 * added later as human-provided modules under brain/modules/.
 */
(function (global) {
  const BN_WORDS = new Set(['আমি','আমার','তুমি','আপনি','সে','তিনি','আমরা','তারা','কি','কী','কেন','কিভাবে','কীভাবে','হ্যাঁ','না','ধন্যবাদ','দুঃখিত','ভালো','খারাপ']);
  const QUESTION = /(?:\?|\b(?:কি|কী|কেন|কখন|কোথায়|কোথায়|কে|কীভাবে|কিভাবে|what|why|when|where|who|how|which|can|could|is|are|do|does)\b)/iu;
  const BANGALISH = /[A-Za-z]/.test.bind(/[A-Za-z]/);

  function normalize(text) {
    return String(text || '').normalize('NFC').replace(/[\u200B-\u200D\uFEFF]/g, '').replace(/\s+/g, ' ').trim();
  }

  function detectLanguage(text) {
    const s = normalize(text);
    const bn = (s.match(/[\u0980-\u09FF]/g) || []).length;
    const en = (s.match(/[A-Za-z]/g) || []).length;
    if (!bn && !en) return 'unknown';
    if (bn && en) return 'mixed';
    return bn ? 'bn' : 'en';
  }

  function tokenize(text) {
    return normalize(text).toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
  }

  function isQuestion(text) { return QUESTION.test(normalize(text)); }

  function features(text) {
    const s = normalize(text), tokens = tokenize(s);
    return {
      text: s,
      language: detectLanguage(s),
      tokens,
      tokenCount: tokens.length,
      isQuestion: isQuestion(s),
      hasBangla: /[\u0980-\u09FF]/u.test(s),
      hasEnglish: /[A-Za-z]/.test(s),
      likelyBanglish: /[A-Za-z]/.test(s) && /(?:ami|amr|tumi|kmne|kno|ki|kemon|ache|ase|dau|dao|korbo|hobe|nai)\b/i.test(s)
    };
  }

  function similarity(a, b) {
    const A = new Set(tokenize(a)), B = new Set(tokenize(b));
    if (!A.size || !B.size) return 0;
    let common = 0; A.forEach(x => { if (B.has(x)) common++; });
    return common / Math.max(A.size, B.size);
  }

  global.gAiLanguageEngine = { normalize, detectLanguage, tokenize, isQuestion, features, similarity };
})(window);
