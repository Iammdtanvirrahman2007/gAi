"""Capability tracking for the human-guided growing brain."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Capability:
    name: str
    description: str
    status: str = "missing"
    required_files: list[str] | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["required_files"] = self.required_files or []
        return data


class CapabilityRegistry:
    """Tracks what gAi can do without allowing it to implement code itself."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capabilities: dict[str, Capability] = self._load()

    def _load(self) -> dict[str, Capability]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return {key: Capability(**value) for key, value in raw.items()}
        except (json.JSONDecodeError, TypeError, KeyError):
            return {}

    def save(self) -> None:
        payload = {key: value.to_dict() for key, value in self.capabilities.items()}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def add(self, name: str, description: str, required_files: list[str] | None = None) -> Capability:
        key = name.strip().lower()
        if not key:
            raise ValueError("Capability name cannot be empty.")
        capability = Capability(
            name=name.strip(),
            description=description.strip(),
            status="missing",
            required_files=required_files or [],
        )
        self.capabilities[key] = capability
        self.save()
        return capability

    def mark_available(self, name: str) -> Capability:
        key = name.strip().lower()
        if key not in self.capabilities:
            raise ValueError(f"Unknown capability: {name}")
        self.capabilities[key].status = "available"
        self.save()
        return self.capabilities[key]

    def get(self, name: str) -> Capability | None:
        return self.capabilities.get(name.strip().lower())

    def all(self) -> list[Capability]:
        return list(self.capabilities.values())
