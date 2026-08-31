"""Core of gAi: a human-guided, non-code-writing growing brain."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

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

class GrowingBrain:
    """Learn concepts, track capabilities, and request human-written code."""

    def __init__(self) -> None:
        MEMORY_DIR.mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(exist_ok=True)
        REQUEST_DIR.mkdir(exist_ok=True)
        self.memory_file = MEMORY_DIR / "lessons.json"
        self.lessons: list[Lesson] = self._load_lessons()
        self.capabilities = CapabilityRegistry(KNOWLEDGE_DIR / "capabilities.json")
        self.request_counter = self._next_request_number()

    def _load_lessons(self) -> list[Lesson]:
        if not self.memory_file.exists():
            return []
        try:
            return [Lesson(**item) for item in json.loads(self.memory_file.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    def _save_lessons(self) -> None:
        self.memory_file.write_text(json.dumps([asdict(x) for x in self.lessons], indent=2, ensure_ascii=False), encoding="utf-8")

    def _next_request_number(self) -> int:
        numbers = []
        for path in REQUEST_DIR.glob("request_*.md"):
            try: numbers.append(int(path.stem.split("_")[-1]))
            except ValueError: pass
        return max(numbers, default=0)

    def learn(self, topic: str, content: str) -> Lesson:
        topic, content = topic.strip(), content.strip()
        if not topic or not content:
            raise ValueError("Topic and lesson content cannot be empty.")
        lesson = Lesson(topic, content, datetime.now(timezone.utc).isoformat())
        self.lessons.append(lesson)
        self._save_lessons()
        return lesson

    def add_capability(self, name: str, description: str, required_files: list[str] | None = None):
        return self.capabilities.add(name, description, required_files)

    def request_code(self, capability: str, reason: str, requirements: list[str], target_file: str = "TBD") -> Path:
        """Create a human implementation request. This method never writes implementation code."""
        capability, reason = capability.strip(), reason.strip()
        if not capability or not reason: raise ValueError("Capability and reason cannot be empty.")
        requirements = [x.strip() for x in requirements if x.strip()]
        self.request_counter += 1
        number = self.request_counter
        path = REQUEST_DIR / f"request_{number:03d}.md"
        text = f"""# CODE REQUEST #{number:03d}\n\nStatus: WAITING_FOR_HUMAN_CODE\nCreated: {datetime.now(timezone.utc).isoformat()}\n\n## Capability\n{capability}\n\n## Why this is needed\n{reason}\n\n## Required file\n`{target_file.strip() or 'TBD'}`\n\n## Requirements\n{chr(10).join('- ' + x for x in requirements) or '- No detailed requirements supplied yet.'}\n\n## Human implementation rule\nThe AI must NOT write implementation code. A human will provide it.\n\n## Verification\nAfter implementation, tests must verify the human-provided code before completion.\n"""
        path.write_text(text, encoding="utf-8")
        return path

    def search_memory(self, query: str) -> list[Lesson]:
        query = query.strip().lower()
        return [x for x in self.lessons if not query or query in x.topic.lower() or query in x.content.lower()]
