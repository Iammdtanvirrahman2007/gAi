"""Minimal core for the human-guided growing AI."""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Lesson:
    topic: str
    content: str
    learned_at: str


class GrowingBrain:
    """Stores lessons and creates human code requests when needed."""

    def __init__(self):
        self.lessons: list[Lesson] = []
        self.request_counter = 0

    def learn(self, topic: str, content: str) -> Lesson:
        lesson = Lesson(
            topic=topic.strip(),
            content=content.strip(),
            learned_at=datetime.now(timezone.utc).isoformat(),
        )
        self.lessons.append(lesson)
        return lesson

    def request_code(self, capability: str, reason: str, requirements: list[str]) -> str:
        """Create a Markdown request for a human to implement a capability."""
        self.request_counter += 1
        number = self.request_counter
        filename = f"code_requests/request_{number:03d}.md"
        requirements_text = "\n".join(f"- {item}" for item in requirements)

        return f"""# CODE REQUEST #{number:03d}\n\nStatus: WAITING_FOR_HUMAN_CODE\n\n## Capability\n{capability}\n\n## Why this is needed\n{reason}\n\n## Requirements\n{requirements_text}\n\n## Rule\nThe AI must not implement this request itself. A human will provide the code.\n"""
