"""Contextual reasoning helpers for gAi."""
from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Evidence:
    topic: str
    content: str
    score: float

class ReasoningEngine:
    def __init__(self) -> None:
        self.stopwords={"what","why","how","when","where","who","which","is","are","the","a","an","of","to","in","for","and","or","does","do","can","could","would","should","i","you","it","this","that","explain","tell","me","about"}

    def tokens(self,text:str)->set[str]:
        return {w.lower() for w in re.findall(r"[a-zA-Z0-9_]+",text) if len(w)>2 and w.lower() not in self.stopwords}

    def retrieve(self,question:str,lessons)->list[Evidence]:
        q=self.tokens(question); out=[]
        for lesson in lessons:
            body=f"{lesson.topic} {lesson.content}"; words=self.tokens(body)
            overlap=len(q & words)
            topic_bonus=0.75 if any(t.lower() in q for t in re.findall(r"[a-zA-Z0-9_]+",lesson.topic)) else 0
            exact_bonus=2.0 if question.lower().strip() in lesson.content.lower() else 0
            score=overlap+topic_bonus+exact_bonus
            if score>0: out.append(Evidence(lesson.topic,lesson.content,score))
        return sorted(out,key=lambda x:x.score,reverse=True)

    def synthesize(self,question:str,evidence:list[Evidence])->str:
        if not evidence:
            return "I don't have enough grounded knowledge to answer that reliably yet."
        q=question.lower()
        if re.search(r"\bwhat is\b|\bdefine\b|\bmeaning\b",q): prefix="The relevant learned concept is:"
        elif re.search(r"\bwhy\b|\bhow\b|\bexplain\b",q): prefix="Using the relevant knowledge I've learned:"
        elif re.search(r"\bcompare\b|\bdifference\b|\bversus\b|\bvs\b",q): prefix="From the relevant learned knowledge:"
        else: prefix="Based on the knowledge I've learned:"
        lines=[prefix]
        for item in evidence[:3]: lines.append(f"- {item.topic}: {item.content}")
        return "\n".join(lines)
