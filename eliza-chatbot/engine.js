const normalize = value => String(value ?? "").toLowerCase().replace(/[^a-z0-9' ]/g, " ").replace(/\s+/g, " ").trim();

function parseScript(source) {
  const data = { initial: "", final: "", quit: [], pre: [], post: [], synonyms: {}, keywords: [] };
  let keyword = null, decomposition = null;
  for (const raw of source.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    let match;
    if ((match = line.match(/^initial:\s*(.*)$/i))) data.initial = match[1];
    else if ((match = line.match(/^final:\s*(.*)$/i))) data.final = match[1];
    else if ((match = line.match(/^quit:\s*(.*)$/i))) data.quit.push(normalize(match[1]));
    else if ((match = line.match(/^pre:\s*(\S+)\s+(.+)$/i))) data.pre.push([match[1].toLowerCase(), match[2].toLowerCase()]);
    else if ((match = line.match(/^post:\s*(\S+)\s+(.+)$/i))) data.post.push([match[1].toLowerCase(), match[2]]);
    else if ((match = line.match(/^synon:\s*(.+)$/i))) {
      const group = match[1].toLowerCase().split(/\s+/);
      for (const word of group) data.synonyms[word] = group;
    } else if ((match = line.match(/^key:\s*(\S+)(?:\s+(\d+))?$/i))) {
      keyword = { word: match[1].toLowerCase(), priority: Number(match[2] || 0), order: data.keywords.length, decompositions: [] };
      data.keywords.push(keyword);
      decomposition = null;
    } else if ((match = line.match(/^decomp:\s*(.*)$/i)) && keyword) {
      decomposition = { pattern: match[1].toLowerCase(), responses: [], cursor: 0 };
      keyword.decompositions.push(decomposition);
    } else if ((match = line.match(/^reasmb:\s*(.*)$/i)) && decomposition) decomposition.responses.push(match[1]);
  }
  return data;
}

function replaceWords(text, pairs) {
  const map = new Map(pairs);
  return text.split(" ").map(word => map.get(word) ?? word).join(" ");
}

function matchPattern(pattern, input, synonyms) {
  const patternWords = pattern.split(/\s+/).filter(Boolean);
  const inputWords = input.split(/\s+/).filter(Boolean);
  function walk(pi, wi, captures) {
    if (pi === patternWords.length) return wi === inputWords.length ? captures : null;
    const token = patternWords[pi];
    if (token === "$") return wi === 0 ? walk(pi + 1, wi, captures) : null;
    if (token === "*") {
      for (let end = inputWords.length; end >= wi; end--) {
        const result = walk(pi + 1, end, [...captures, inputWords.slice(wi, end).join(" ")]);
        if (result) return result;
      }
      return null;
    }
    if (token.startsWith("@")) {
      const group = synonyms[token.slice(1)] || [token.slice(1)];
      return wi < inputWords.length && group.includes(inputWords[wi]) ? walk(pi + 1, wi + 1, captures) : null;
    }
    return wi < inputWords.length && token === inputWords[wi] ? walk(pi + 1, wi + 1, captures) : null;
  }
  return walk(0, 0, []);
}

function renderTemplate(template, captures, post) {
  return template.replace(/\((\d+)\)/g, (_, number) => replaceWords(captures[Number(number) - 1] || "", post))
    .replace(/\s+([?.!,])/g, "$1").replace(/\s+/g, " ").trim();
}

export class ElizaEngine {
  constructor(script) {
    this.data = parseScript(script);
    this.closed = false;
  }

  respond(message) {
    const original = String(message ?? "").trim();
    if (!original) return { text: "Please say something.", done: false };
    let input = replaceWords(normalize(original), this.data.pre);
    if (this.data.quit.includes(input)) {
      this.closed = true;
      return { text: this.data.final || "Goodbye.", done: true };
    }
    const words = input.split(" ");
    const candidates = this.data.keywords.filter(item => item.word !== "xnone" && words.includes(item.word))
      .sort((a, b) => b.priority - a.priority || a.order - b.order);
    const fallback = this.data.keywords.find(item => item.word === "xnone");
    if (fallback) candidates.push(fallback);

    for (const item of candidates) {
      const answer = this.answerFrom(item, input, 0);
      if (answer) return { text: answer, done: false };
    }
    return { text: "Please go on.", done: false };
  }

  answerFrom(item, input, depth) {
    if (!item || depth > 8) return "";
    for (const decomp of item.decompositions) {
      const captures = matchPattern(decomp.pattern, input, this.data.synonyms);
      if (!captures) continue;
      for (let tries = 0; tries < decomp.responses.length; tries++) {
        if (!decomp.responses.length) break;
        const template = decomp.responses[decomp.cursor++ % decomp.responses.length];
        if (template === "newkey") break;
        if (template.toLowerCase().startsWith("goto ")) {
          const target = this.data.keywords.find(entry => entry.word === template.slice(5).trim().toLowerCase());
          const redirected = this.answerFrom(target, input, depth + 1);
          if (redirected) return redirected;
          continue;
        }
        return renderTemplate(template, captures, this.data.post) || "Please go on.";
      }
    }
    return "";
  }
}