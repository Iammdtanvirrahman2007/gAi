"""Core of gAi: a human-guided, non-code-writing growing brain."""
from __future__ import annotations
import json
import re
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
@dataclass
class AnswerResult:
    answer: str
    confidence: float
    known: list[str]
    missing: list[str]
    blockers: list[str]
    request_files: list[str]
class GrowingBrain:
    """Learn, answer from accumulated knowledge, and request human code upgrades."""
    def __init__(self) -> None:
        MEMORY_DIR.mkdir(exist_ok=True); KNOWLEDGE_DIR.mkdir(exist_ok=True); REQUEST_DIR.mkdir(exist_ok=True)
        self.memory_file = MEMORY_DIR / "lessons.json"
        self.lessons = self._load_lessons()
        self.capabilities = CapabilityRegistry(KNOWLEDGE_DIR / "capabilities.json")
        self.request_counter = self._next_request_number()
    def _load_lessons(self):
        if not self.memory_file.exists(): return []
        try: return [Lesson(**x) for x in json.loads(self.memory_file.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, TypeError, KeyError): return []
    def _save_lessons(self):
        self.memory_file.write_text(json.dumps([asdict(x) for x in self.lessons], indent=2, ensure_ascii=False), encoding="utf-8")
    def _next_request_number(self):
        nums=[]
        for p in REQUEST_DIR.glob("request_*.md"):
            m=re.match(r"request_(\d+)\.md$",p.name)
            if m: nums.append(int(m.group(1)))
        return max(nums, default=0)
    def learn(self, topic: str, content: str) -> Lesson:
        topic, content = topic.strip(), content.strip()
        if not topic or not content: raise ValueError("Topic and lesson content cannot be empty.")
        lesson=Lesson(topic,content,datetime.now(timezone.utc).isoformat()); self.lessons.append(lesson); self._save_lessons(); self._auto_plan_growth(lesson); return lesson
    def add_capability(self,name,description,required_files=None): return self.capabilities.add(name,description,required_files)
    def request_code(self, capability, reason, requirements, target_file="TBD") -> Path:
        capability,reason=capability.strip(),reason.strip()
        if not capability or not reason: raise ValueError("Capability and reason cannot be empty.")
        self.request_counter+=1; n=self.request_counter; path=REQUEST_DIR/f"request_{n:03d}.md"
        text=f"""# CODE UPGRADE REQUEST #{n:03d}\n\nStatus: WAITING_FOR_HUMAN_CODE\nCreated: {datetime.now(timezone.utc).isoformat()}\n\n## Capability\n{capability}\n\n## Why the brain needs it\n{reason}\n\n## Required file\n`{target_file.strip() or 'TBD'}`\n\n## Requirements\n{chr(10).join('- '+x for x in requirements) or '- Define the required interface and tests.'}\n\n## Human implementation rule\nThe AI must not write implementation code. The human supplies the code.\n\n## Verification\nAfter code is supplied, run tests and mark the capability available only after verification.\n"""
        path.write_text(text,encoding="utf-8"); return path
    def _auto_plan_growth(self, lesson):
        text=(lesson.topic+' '+lesson.content).lower()
        rules=[
        ("neural network",["neural network","network architecture"],"brain/modules/neural_network.py",["Represent layers and connections.","Support configurable input and output sizes.","Expose a forward operation."]),
        ("attention",["attention","transformer"],"brain/modules/attention.py",["Implement the learned attention mechanism.","Provide a small testable interface."]),
        ("image processing",["image","computer vision","vision"],"brain/modules/image_processing.py",["Provide the image-processing operation needed by experiments.","Keep the module independently testable."])]
        for cap,keywords,target,reqs in rules:
            if any(k in text for k in keywords) and self.capabilities.get(cap) is None:
                self.capabilities.add(cap,f"Capability inferred from learned topic: {lesson.topic}",[target])
                self.request_code(cap,f"The brain learned '{lesson.topic}' but lacks this implementation capability.",reqs,target)
    def _infer_missing_capability(self, question: str):
        stop={"what","why","how","when","where","who","which","is","are","the","a","an","of","to","in","for","and","or","does","do","can","could","would","should","i","you","it","this","that","explain","tell","me","about"}
        words=[w for w in re.findall(r"[a-zA-Z0-9_]+",question.lower()) if w not in stop and len(w)>2]
        topic="_".join(words[:5]) or "general_reasoning"
        return f"knowledge_reasoning_{topic}", words
    def answer_question(self, question: str) -> AnswerResult:
        q=question.strip()
        if not q: raise ValueError("Question cannot be empty.")
        tokens={w.lower() for w in re.findall(r"[a-zA-Z0-9_]+",q) if len(w)>2}
        ranked=[]
        for lesson in self.lessons:
            words={w.lower() for w in re.findall(r"[a-zA-Z0-9_]+",lesson.topic+' '+lesson.content) if len(w)>2}
            score=len(tokens & words)
            if score: ranked.append((score,lesson))
        ranked.sort(key=lambda x:x[0],reverse=True)
        known=[x[1].topic for x in ranked[:5]]; missing=[]; blockers=[]; request_files=[]
        for cap in self.capabilities.all():
            if cap.status.lower() != 'available' and any(word in (q+' '+q).lower() for word in cap.name.lower().split()):
                missing.append(cap.name); blockers.append(f"Missing capability: {cap.name}"); request_files.extend(cap.required_files or [])
        if ranked:
            snippets=[f"{x[1].topic}: {x[1].content}" for x in ranked[:3]]
            answer="Based on my current learned knowledge:\n"+'\n'.join('- '+s for s in snippets); confidence=min(0.95,0.35+0.12*ranked[0][0])
            return AnswerResult(answer,confidence,known,missing,blockers,request_files)
        capability,keywords=self._infer_missing_capability(q)
        if self.capabilities.get(capability) is None:
            target=f"brain/modules/{capability}.py"
            reqs=["Understand the concepts required by the question.","Provide a small, testable interface for the brain.","Add tests so the human implementation can be verified."]
            path=self.request_code(capability,"Current learned knowledge is insufficient to answer this question reliably.",reqs,target)
            self.capabilities.add(capability,"Capability requested because the brain could not reliably answer a question.",[target])
            request_files.append(str(path.relative_to(ROOT)))
        else:
            cap=self.capabilities.get(capability)
            request_files.extend(cap.required_files or [])
        blockers.append(f"Missing capability: {capability}. Need more knowledge or a human-provided implementation.")
        return AnswerResult("I cannot answer reliably from my current learned knowledge yet. I identified what I am missing and created a code-upgrade request for human implementation.",0.05,known,missing or [capability],blockers,request_files)
    def search_memory(self, query):
        q=query.strip().lower(); return [x for x in self.lessons if not q or q in x.topic.lower() or q in x.content.lower()]
