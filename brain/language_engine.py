"""Language understanding plus a small persistent learned language model."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .language_model import NGramLanguageModel


@dataclass(frozen=True)
class Intent:
    name: str
    confidence: float


class LanguageEngine:
    """Understand common question patterns and generate grounded responses."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.stopwords = {
            "what", "why", "how", "when", "where", "who", "which", "is", "are",
            "the", "a", "an", "of", "to", "in", "for", "and", "or", "does",
            "do", "can", "could", "would", "should", "i", "you", "it", "this",
            "that", "explain", "tell", "me", "about", "please", "from",
        }
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "knowledge" / "language_model.json"
        self.model = NGramLanguageModel(model_path)

    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def keywords(self, text: str) -> set[str]:
        return {
            word.lower()
            for word in re.findall(r"[a-zA-Z0-9_]+", text)
            if len(word) > 2 and word.lower() not in self.stopwords
        }

    def learn_text(self, text: str) -> int:
        return self.model.train(text)

    def detect_intent(self, question: str) -> Intent:
        q = question.lower()
        patterns = {
            "greeting": (r"\b(hi|hello|hey|assalamu alaikum)\b", 0.95),
            "definition": (r"\b(what is|what are|define|meaning of)\b", 0.9),
            "explanation": (r"\b(why|how does|how do|explain)\b", 0.86),
            "comparison": (r"\b(compare|difference|different|vs|versus|better than)\b", 0.88),
            "calculation": (r"[0-9].*[+*/%^-].*[0-9]|\b(calculate|sum|average|mean|median|percent|percentage)\b", 0.96),
            "data_analysis": (r"\b(data|dataset|csv|column|row|trend|correlation|distribution|statistics)\b", 0.94),
        }
        for name, (pattern, confidence) in patterns.items():
            if re.search(pattern, q):
                return Intent(name, confidence)
        return Intent("general", 0.55)

    def rank_context(self, question: str, contexts: Iterable[tuple[str, str]]) -> list[tuple[float, str, str]]:
        q_words = self.keywords(question)
        ranked: list[tuple[float, str, str]] = []
        for topic, content in contexts:
            words = self.keywords(f"{topic} {content}")
            overlap = len(q_words & words)
            phrase_bonus = 0.0
            q_norm = self.normalize(question).lower()
            if q_norm and q_norm in content.lower():
                phrase_bonus = 3.0
            score = overlap + phrase_bonus
            if score:
                ranked.append((score, topic, content))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def compose_grounded_answer(self, question: str, ranked: list[tuple[float, str, str]]) -> str:
        intent = self.detect_intent(question)
        if intent.name == "greeting":
            return "Hello! I'm gAi. Ask me something I've learned, or give me data to analyze."
        if not ranked:
            return "I don't have enough learned knowledge to answer that reliably yet."

        best = ranked[:3]
        if intent.name == "definition":
            prefix = "From my learned knowledge, the closest explanation is:"
        elif intent.name == "explanation":
            prefix = "Based on what I've learned, here's the explanation:"
        elif intent.name == "comparison":
            prefix = "From the relevant knowledge I have:"
        else:
            prefix = "Based on my current learned knowledge:"

        lines = [prefix]
        for _, topic, content in best:
            lines.append(f"- {topic}: {content}")

        generated = self.model.generate(question, max_tokens=28)
        if generated:
            lines.append(f"Learned-language continuation: {generated}")
        return "\n".join(lines)
