"""A tiny persistent language model for gAi.

This is a deliberately small statistical model, not a transformer. It learns
word transitions from lessons and can generate short continuations. The model
is persisted as JSON so learning survives restarts.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


class NGramLanguageModel:
    """Persistent trigram language model with deterministic top-k generation."""

    def __init__(self, path: Path, order: int = 3) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        self.path = path
        self.order = order
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: Counter[str] = Counter()
        self._load()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9_']+|[.!?]", text.lower())
        return words

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.order = int(raw.get("order", self.order))
            self.counts = defaultdict(Counter, {
                key: Counter(value) for key, value in raw.get("counts", {}).items()
            })
            self.vocab = Counter(raw.get("vocab", {}))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self.counts = defaultdict(Counter)
            self.vocab = Counter()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "order": self.order,
            "counts": {key: dict(value) for key, value in self.counts.items()},
            "vocab": dict(self.vocab),
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def train(self, text: str) -> int:
        tokens = self.tokenize(text)
        if not tokens:
            return 0
        padded = ["<BOS>"] * (self.order - 1) + tokens + ["<EOS>"]
        for i in range(self.order - 1, len(padded)):
            context = " ".join(padded[i - self.order + 1:i])
            token = padded[i]
            self.counts[context][token] += 1
            if token not in {"<BOS>", "<EOS>"}:
                self.vocab[token] += 1
        self.save()
        return len(tokens)

    def _next_token(self, context_tokens: list[str]) -> str | None:
        context = " ".join(context_tokens[-(self.order - 1):])
        choices = self.counts.get(context)
        if not choices:
            return None
        # Most frequent transition first, with lexical tie-break for stability.
        return sorted(choices.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def generate(self, prompt: str, max_tokens: int = 40) -> str:
        if not self.counts:
            return ""
        prompt_tokens = self.tokenize(prompt)
        generated = prompt_tokens[:]
        context = ["<BOS>"] * (self.order - 1) + prompt_tokens

        for _ in range(max_tokens):
            token = self._next_token(context)
            if token is None:
                # Back off to a context that only uses the final token.
                if context:
                    candidates = []
                    final = context[-1]
                    suffix = f" {final}"
                    for key, counter in self.counts.items():
                        if key.endswith(suffix):
                            candidates.extend(counter.items())
                    if candidates:
                        token = sorted(candidates, key=lambda item: (-item[1], item[0]))[0][0]
                if token is None:
                    break
            if token == "<EOS>":
                break
            if token != "<BOS>":
                generated.append(token)
            context.append(token)

        return self._detokenize(generated[len(prompt_tokens):])

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        text = ""
        for token in tokens:
            if token in {".", "!", "?"}:
                text = text.rstrip() + token
            else:
                text += (" " if text else "") + token
        return text.strip()

    @property
    def trained_tokens(self) -> int:
        return sum(self.vocab.values())
