"""Core of gAi: a human-guided, data-aware growing brain."""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from .capability import CapabilityRegistry

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


class GrowingBrain:
    """Learn, reason over memory, analyze small structured datasets, and request human code upgrades."""

    def __init__(self) -> None:
        MEMORY_DIR.mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        REQUEST_DIR.mkdir(exist_ok=True)
        self.memory_file = MEMORY_DIR / "lessons.json"
        self.lessons = self._load_lessons()
        self.capabilities = CapabilityRegistry(KNOWLEDGE_DIR / "capabilities.json")
        self.request_counter = self._next_request_number()

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
        nums = []
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
        self._auto_plan_growth(lesson)
        return lesson

    def add_capability(self, name, description, required_files=None):
        return self.capabilities.add(name, description, required_files)

    def request_code(self, capability, reason, requirements, target_file="TBD") -> Path:
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
                self.request_code(
                    cap,
                    f"The brain learned '{lesson.topic}' but lacks this implementation capability.",
                    reqs,
                    target,
                )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"[a-zA-Z0-9_]+", text) if len(w) > 2]

    def _retrieve_lessons(self, question: str, limit: int = 5) -> list[Lesson]:
        query = set(self._tokenize(question))
        ranked: list[tuple[float, Lesson]] = []
        for lesson in self.lessons:
            words = set(self._tokenize(lesson.topic + " " + lesson.content))
            overlap = len(query & words)
            if not overlap:
                continue
            phrase_bonus = 0.0
            if lesson.topic.lower() in question.lower():
                phrase_bonus = 2.0
            score = overlap + phrase_bonus
            ranked.append((score, lesson))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [lesson for _, lesson in ranked[:limit]]

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        return [float(x) for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)]

    @staticmethod
    def _answer_simple_math(question: str) -> str | None:
        q = question.lower().strip()
        # Deliberately small and safe arithmetic evaluator. Only numeric expressions/operators are accepted.
        match = re.fullmatch(r"\s*([\d.+\-*/()%\s]+)\s*\??\s*", q)
        if not match or not any(ch.isdigit() for ch in q):
            return None
        expr = match.group(1)
        if not re.fullmatch(r"[\d.+\-*/()%\s]+", expr):
            return None
        try:
            result = eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return None
        if isinstance(result, (int, float)) and math.isfinite(float(result)):
            return f"Result: {result}"
        return None

    @staticmethod
    def analyze_text_data(data: Any) -> dict[str, Any]:
        """Analyze a list of dict rows or a dict of equal-length columns."""
        if isinstance(data, dict):
            rows = GrowingBrain._columns_to_rows(data)
        elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
            rows = data
        else:
            raise ValueError("Data must be a list of objects or a dictionary of columns.")
        if not rows:
            return {"rows": 0, "columns": [], "numeric": {}}

        columns = sorted({key for row in rows for key in row})
        numeric: dict[str, list[float]] = {}
        for col in columns:
            values = []
            for row in rows:
                value = row.get(col)
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, str):
                    try:
                        values.append(float(value.strip()))
                    except ValueError:
                        pass
            if values:
                numeric[col] = values

        summary: dict[str, Any] = {}
        for col, values in numeric.items():
            item: dict[str, Any] = {
                "count": len(values),
                "sum": sum(values),
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
            if len(values) >= 2:
                item["stdev"] = stdev(values)
            summary[col] = item

        return {"rows": len(rows), "columns": columns, "numeric": summary}

    @staticmethod
    def _columns_to_rows(columns: dict[str, Any]) -> list[dict[str, Any]]:
        if not columns:
            return []
        lengths = [len(v) for v in columns.values() if isinstance(v, list)]
        if len(lengths) != len(columns) or len(set(lengths)) > 1:
            raise ValueError("Column data must be lists of equal length.")
        n = lengths[0] if lengths else 0
        return [{key: columns[key][i] for key in columns} for i in range(n)]

    @staticmethod
    def _format_data_analysis(analysis: dict[str, Any], question: str) -> str:
        if not analysis.get("rows"):
            return "The dataset is empty."
        lines = [f"I analyzed {analysis['rows']} rows across {len(analysis['columns'])} columns."]
        for col, stats in analysis["numeric"].items():
            lines.append(
                f"{col}: mean={stats['mean']:.4g}, median={stats['median']:.4g}, "
                f"min={stats['min']:.4g}, max={stats['max']:.4g}, sum={stats['sum']:.4g}."
            )
        if not analysis["numeric"]:
            lines.append("No numeric columns were detected, so I cannot compute numeric statistics.")
        return "\n".join(lines)

    def answer_question(self, question: str, data: Any = None) -> AnswerResult:
        q = question.strip()
        if not q:
            raise ValueError("Question cannot be empty.")

        math_answer = self._answer_simple_math(q)
        if math_answer is not None:
            return AnswerResult(math_answer, 0.99, [], [], [], [])

        if data is not None:
            try:
                analysis = self.analyze_text_data(data)
                answer = self._format_data_analysis(analysis, q)
                return AnswerResult(answer, 0.97, [], [], [], [])
            except ValueError as exc:
                return AnswerResult(f"I could not analyze the supplied data: {exc}", 0.2, [], ["structured_data_analysis"], [str(exc)], [])

        lessons = self._retrieve_lessons(q)
        if lessons:
            snippets = [f"{lesson.topic}: {lesson.content}" for lesson in lessons[:3]]
            answer = "Based on my current learned knowledge:\n" + "\n".join("- " + s for s in snippets)
            confidence = min(0.93, 0.45 + 0.10 * len(lessons))
            return AnswerResult(answer, confidence, [x.topic for x in lessons], [], [], [])

        capability, _ = self._infer_missing_capability(q)
        request_files: list[str] = []
        if self.capabilities.get(capability) is None:
            target = f"brain/modules/{capability}.py"
            reqs = [
                "Understand the concepts required by the question.",
                "Provide a small, testable interface for the brain.",
                "Add tests so the human implementation can be verified.",
            ]
            path = self.request_code(
                capability,
                "Current learned knowledge is insufficient to answer this question reliably.",
                reqs,
                target,
            )
            self.capabilities.add(
                capability,
                "Capability requested because the brain could not reliably answer a question.",
                [target],
            )
            request_files.append(str(path.relative_to(ROOT)))
        else:
            cap = self.capabilities.get(capability)
            request_files.extend(cap.required_files or [])

        blocker = f"Missing capability: {capability}. Need more knowledge or a human-provided implementation."
        return AnswerResult(
            "I cannot answer reliably from my current learned knowledge yet. I identified what I am missing and created a code-upgrade request for human implementation.",
            0.05,
            [],
            [capability],
            [blocker],
            request_files,
        )

    def answer_with_csv(self, question: str, csv_text: str) -> AnswerResult:
        reader = csv.DictReader(csv_text.splitlines())
        return self.answer_question(question, list(reader))

    def _infer_missing_capability(self, question: str):
        stop = {
            "what", "why", "how", "when", "where", "who", "which", "is", "are", "the", "a", "an",
            "of", "to", "in", "for", "and", "or", "does", "do", "can", "could", "would", "should",
            "i", "you", "it", "this", "that", "explain", "tell", "me", "about",
        }
        words = [w for w in self._tokenize(question) if w not in stop]
        topic = "_".join(words[:5]) or "general_reasoning"
        return f"knowledge_reasoning_{topic}", words

    def search_memory(self, query):
        q = query.strip().lower()
        return [x for x in self.lessons if not q or q in x.topic.lower() or q in x.content.lower()]
