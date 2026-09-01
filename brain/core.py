"""Core learning, reasoning, memory, and data-analysis engine for gAi."""
from __future__ import annotations

import ast
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .capability import CapabilityRegistry
from .data_analyzer import DataAnalyzer
from .language_engine import LanguageEngine

ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = ROOT / "memory"
KNOWLEDGE_DIR = ROOT / "knowledge"
REQUEST_DIR = ROOT / "code_requests"


@dataclass
class Lesson:
    topic: str
    content: str
    learned_at: str


@dataclass
class AnswerResult:
    answer: str
    confidence: float
    known: list[str]
    missing: list[str]
    blockers: list[str]
    request_files: list[str]
    intent: str = "general"
    data_analysis: dict[str, Any] | None = None


class GrowingBrain:
    """Learn, reason over memory, calculate, analyze data, and request human code upgrades."""

    def __init__(self) -> None:
        MEMORY_DIR.mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        REQUEST_DIR.mkdir(exist_ok=True)
        self.memory_file = MEMORY_DIR / "lessons.json"
        self.lessons = self._load_lessons()
        self.capabilities = CapabilityRegistry(KNOWLEDGE_DIR / "capabilities.json")
        self.request_counter = self._next_request_number()
        self.language = LanguageEngine(KNOWLEDGE_DIR / "language_model.json")
        self.data = DataAnalyzer()

        # Rebuild the language model from durable lessons when its state is absent.
        if not self.language.model.counts and self.lessons:
            for lesson in self.lessons:
                self.language.learn_text(f"{lesson.topic}. {lesson.content}")

    def _load_lessons(self) -> list[Lesson]:
        if not self.memory_file.exists():
            return []
        try:
            return [Lesson(**x) for x in json.loads(self.memory_file.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    def _save_lessons(self) -> None:
        self.memory_file.write_text(
            json.dumps([asdict(x) for x in self.lessons], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _next_request_number(self) -> int:
        nums: list[int] = []
        for path in REQUEST_DIR.glob("request_*.md"):
            match = re.match(r"request_(\d+)\.md$", path.name)
            if match:
                nums.append(int(match.group(1)))
        return max(nums, default=0)

    def learn(self, topic: str, content: str) -> Lesson:
        topic, content = topic.strip(), content.strip()
        if not topic or not content:
            raise ValueError("Topic and lesson content cannot be empty.")
        lesson = Lesson(topic, content, datetime.now(timezone.utc).isoformat())
        self.lessons.append(lesson)
        self._save_lessons()
        # Learning updates both declarative memory and language-transition statistics.
        self.language.learn_text(f"{topic}. {content}")
        self._auto_plan_growth(lesson)
        return lesson

    def add_capability(self, name: str, description: str, required_files=None):
        return self.capabilities.add(name, description, required_files)

    def request_code(self, capability: str, reason: str, requirements: list[str], target_file: str = "TBD") -> Path:
        capability, reason = capability.strip(), reason.strip()
        if not capability or not reason:
            raise ValueError("Capability and reason cannot be empty.")
        self.request_counter += 1
        n = self.request_counter
        path = REQUEST_DIR / f"request_{n:03d}.md"
        text = f"""# CODE UPGRADE REQUEST #{n:03d}\n\nStatus: WAITING_FOR_HUMAN_CODE\nCreated: {datetime.now(timezone.utc).isoformat()}\n\n## Capability\n{capability}\n\n## Why the brain needs it\n{reason}\n\n## Required file\n`{target_file.strip() or 'TBD'}`\n\n## Requirements\n{chr(10).join('- ' + x for x in requirements) or '- Define the required interface and tests.'}\n\n## Human implementation rule\nThe AI must not write implementation code. The human supplies the code.\n\n## Verification\nAfter code is supplied, run tests and mark the capability available only after verification.\n"""
        path.write_text(text, encoding="utf-8")
        return path

    def _auto_plan_growth(self, lesson: Lesson) -> None:
        text = (lesson.topic + " " + lesson.content).lower()
        rules = [
            ("neural network", ["neural network", "network architecture"], "brain/modules/neural_network.py", [
                "Represent layers and connections.",
                "Support configurable input and output sizes.",
                "Expose a forward operation.",
            ]),
            ("attention", ["attention", "transformer"], "brain/modules/attention.py", [
                "Implement the learned attention mechanism.",
                "Provide a small testable interface.",
            ]),
            ("image processing", ["image", "computer vision", "vision"], "brain/modules/image_processing.py", [
                "Provide the image-processing operation needed by experiments.",
                "Keep the module independently testable.",
            ]),
        ]
        for cap, keywords, target, reqs in rules:
            if any(k in text for k in keywords) and self.capabilities.get(cap) is None:
                self.capabilities.add(cap, f"Capability inferred from learned topic: {lesson.topic}", [target])
                self.request_code(cap, f"The brain learned '{lesson.topic}' but lacks this implementation capability.", reqs, target)

    @staticmethod
    def _safe_math(question: str) -> float | int | None:
        candidate = question.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
        if not re.fullmatch(r"[0-9\s()+\-*/%.]+", candidate):
            return None
        if not re.search(r"[+\-*/%]", candidate):
            return None
        try:
            tree = ast.parse(candidate, mode="eval")
            allowed = (
                ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                ast.USub, ast.UAdd,
            )
            if not all(isinstance(node, allowed) for node in ast.walk(tree)):
                return None
            value = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {})
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return value
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
            return None
        return None

    def _infer_missing_capability(self, question: str) -> tuple[str, list[str]]:
        words = sorted(self.language.keywords(question))
        topic = "_".join(words[:5]) or "general_reasoning"
        return f"knowledge_reasoning_{topic}", words

    def analyze_data(self, data: Any) -> tuple[str, dict[str, Any]]:
        analysis = self.data.analyze_rows(data) if isinstance(data, list) else self.data.analyze_rows(self._columns_to_rows(data))
        answer = self.data.answer("", analysis)
        if answer is None:
            numeric = analysis.get("numeric", {})
            if numeric:
                chunks = [
                    f"{col}: mean={stats['mean']:.4g}, median={stats['median']:.4g}, "
                    f"min={stats['min']:.4g}, max={stats['max']:.4g}, sum={stats['sum']:.4g}"
                    for col, stats in numeric.items()
                ]
                answer = f"I analyzed {analysis['rows']} rows. " + "; ".join(chunks) + "."
            else:
                answer = f"I analyzed {analysis['rows']} rows across {len(analysis['columns'])} columns, but found no numeric columns."
        return answer, analysis

    @staticmethod
    def _columns_to_rows(columns: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(columns, dict) or not columns:
            return []
        values = list(columns.values())
        if not all(isinstance(v, list) for v in values):
            raise ValueError("Column data must contain lists.")
        lengths = [len(v) for v in values]
        if len(set(lengths)) > 1:
            raise ValueError("Column data must be lists of equal length.")
        return [{key: columns[key][i] for key in columns} for i in range(lengths[0] if lengths else 0)]

    def answer_question(self, question: str, data: Any = None) -> AnswerResult:
        q = question.strip()
        if not q:
            raise ValueError("Question cannot be empty.")

        intent = self.language.detect_intent(q)
        math_value = self._safe_math(q)
        if math_value is not None:
            return AnswerResult(f"The result is {math_value}.", 0.99, [], [], [], [], intent.name)

        if data is not None:
            try:
                analysis = self.data.analyze_rows(data) if isinstance(data, list) else self.data.analyze_rows(self._columns_to_rows(data))
                answer = self.data.answer(q, analysis) or (
                    f"I analyzed {analysis['rows']} rows across {len(analysis['columns'])} columns. "
                    f"Numeric columns: {', '.join(analysis.get('numeric', {})) or 'none'}."
                )
                return AnswerResult(answer, 0.97, list(analysis.get("numeric", {}).keys()), [], [], [], "data_analysis", analysis)
            except ValueError as exc:
                return AnswerResult(f"I could not analyze the supplied data: {exc}", 0.2, [], ["structured_data_analysis"], [str(exc)], [], "data_analysis")

        contexts = [(lesson.topic, lesson.content) for lesson in self.lessons]
        ranked = self.language.rank_context(q, contexts)
        known = [topic for _, topic, _ in ranked[:5]]
        if ranked:
            answer = self.language.compose_grounded_answer(q, ranked)
            confidence = min(0.95, 0.45 + 0.10 * len(ranked))
            return AnswerResult(answer, confidence, known, [], [], [], intent.name)

        capability, _ = self._infer_missing_capability(q)
        request_files: list[str] = []
        if self.capabilities.get(capability) is None:
            target = f"brain/modules/{capability}.py"
            reqs = [
                "Understand the concepts required by the question.",
                "Provide a small, testable interface for the brain.",
                "Add tests so the human implementation can be verified.",
            ]
            path = self.request_code(capability, "Current learned knowledge is insufficient to answer this question reliably.", reqs, target)
            self.capabilities.add(capability, "Capability requested because the brain could not reliably answer a question.", [target])
            request_files.append(str(path.relative_to(ROOT)))
        else:
            cap = self.capabilities.get(capability)
            request_files.extend(cap.required_files or [])

        blocker = f"Missing capability: {capability}. Need more knowledge or a human-provided implementation."
        return AnswerResult(
            "I cannot answer reliably from my current learned knowledge yet. I identified what I am missing and created a code-upgrade request for human implementation.",
            0.05,
            known,
            [capability],
            [blocker],
            request_files,
            intent.name,
        )

    def answer_with_csv(self, question: str, csv_text: str) -> AnswerResult:
        rows = list(csv.DictReader(csv_text.splitlines()))
        return self.answer_question(question, rows)

    def search_memory(self, query: str) -> list[Lesson]:
        q = query.strip().lower()
        return [x for x in self.lessons if not q or q in x.topic.lower() or q in x.content.lower()]
