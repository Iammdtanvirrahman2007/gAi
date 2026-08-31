"""Core of gAi: a human-guided, non-code-writing growing brain."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = ROOT / "memory"
KNOWLEDGE_DIR = ROOT / "knowledge"
REQUEST_DIR = ROOT / "code_requests"


@dataclass
class Lesson:
    topic: str
    content: str
    learned_at: str


class GrowingBrain:
    """Learn concepts, persist memory, and request human-written code.

    This core intentionally does not generate implementation code.
    """

    def __init__(self) -> None:
        MEMORY_DIR.mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        REQUEST_DIR.mkdir(exist_ok=True)
        self.memory_file = MEMORY_DIR / "lessons.json"
        self.concepts_file = KNOWLEDGE_DIR / "concepts.json"
        self.lessons: list[Lesson] = self._load_lessons()
        self.request_counter = self._next_request_number()

    def _load_lessons(self) -> list[Lesson]:
        if not self.memory_file.exists():
            return []
        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
            return [Lesson(**item) for item in data]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    def _save_lessons(self) -> None:
        data = [asdict(lesson) for lesson in self.lessons]
        self.memory_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _next_request_number(self) -> int:
        numbers = []
        for path in REQUEST_DIR.glob("request_*.md"):
            try:
                numbers.append(int(path.stem.split("_")[-1]))
            except ValueError:
                pass
        return max(numbers, default=0)

    def learn(self, topic: str, content: str) -> Lesson:
        """Store a human-taught lesson permanently."""
        topic = topic.strip()
        content = content.strip()
        if not topic or not content:
            raise ValueError("Topic and lesson content cannot be empty.")

        lesson = Lesson(
            topic=topic,
            content=content,
            learned_at=datetime.now(timezone.utc).isoformat(),
        )
        self.lessons.append(lesson)
        self._save_lessons()
        return lesson

    def request_code(
        self,
        capability: str,
        reason: str,
        requirements: list[str],
        target_file: str = "TBD",
    ) -> Path:
        """Create a Markdown request for a human implementation."""
        capability = capability.strip()
        reason = reason.strip()
        target_file = target_file.strip() or "TBD"
        requirements = [item.strip() for item in requirements if item.strip()]

        if not capability or not reason:
            raise ValueError("Capability and reason cannot be empty.")

        self.request_counter += 1
        number = self.request_counter
        path = REQUEST_DIR / f"request_{number:03d}.md"
        requirements_text = "\n".join(f"- {item}" for item in requirements)
        if not requirements_text:
            requirements_text = "- No detailed requirements supplied yet."

        content = f"""# CODE REQUEST #{number:03d}

Status: WAITING_FOR_HUMAN_CODE
Created: {datetime.now(timezone.utc).isoformat()}

## Capability
{capability}

## Why this is needed
{reason}

## Required file
`{target_file}`

## Requirements
{requirements_text}

## Human implementation rule
The AI must NOT write the implementation code for this request.
A human will provide the implementation.

## After implementation
The system should later verify the file, run appropriate tests, and mark
this request as completed only after the human-provided implementation works.
"""
        path.write_text(content, encoding="utf-8")
        return path

    def search_memory(self, query: str) -> list[Lesson]:
        """Simple local memory search used by the first prototype."""
        query = query.strip().lower()
        if not query:
            return self.lessons
        return [
            lesson
            for lesson in self.lessons
            if query in lesson.topic.lower() or query in lesson.content.lower()
        ]
