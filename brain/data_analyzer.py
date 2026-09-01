"""Dependency-free structured data analysis utilities for gAi."""
from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter
from statistics import mean, median, pstdev
from typing import Any


class DataAnalyzer:
    def _number(self, value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    def analyze_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"rows": 0, "columns": [], "numeric": {}}
        columns = list(dict.fromkeys(k for row in rows for k in row))
        numeric: dict[str, dict[str, float | int]] = {}
        categorical: dict[str, dict[str, Any]] = {}
        for col in columns:
            nums = [n for row in rows if (n := self._number(row.get(col))) is not None]
            if nums:
                numeric[col] = {
                    "count": len(nums), "mean": mean(nums), "median": median(nums),
                    "min": min(nums), "max": max(nums), "sum": sum(nums),
                    "stdev": pstdev(nums) if len(nums) > 1 else 0.0,
                }
            else:
                values = [str(row.get(col, "")).strip() for row in rows]
                categorical[col] = {"unique": len(set(values)), "top": Counter(values).most_common(5)}
        return {"rows": len(rows), "columns": columns, "numeric": numeric, "categorical": categorical}

    def analyze_csv(self, text: str) -> dict[str, Any]:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        return self.analyze_rows(rows)

    def answer(self, question: str, analysis: dict[str, Any]) -> str | None:
        q = question.lower()
        numeric = analysis.get("numeric", {})
        if not numeric:
            return None
        requested = next((c for c in numeric if c.lower() in q), None)
        if requested is None:
            requested = next(iter(numeric)) if len(numeric) == 1 else None
        if requested is None:
            return f"I analyzed {analysis['rows']} rows and found numeric columns: {', '.join(numeric)}. Tell me which column you want to inspect."
        stats = numeric[requested]
        if re.search(r"\b(average|mean)\b", q):
            return f"The mean of {requested} is {stats['mean']:.4g}."
        if "median" in q:
            return f"The median of {requested} is {stats['median']:.4g}."
        if re.search(r"\b(sum|total)\b", q):
            return f"The total of {requested} is {stats['sum']:.4g}."
        if re.search(r"\b(min|minimum|lowest)\b", q):
            return f"The minimum of {requested} is {stats['min']:.4g}."
        if re.search(r"\b(max|maximum|highest)\b", q):
            return f"The maximum of {requested} is {stats['max']:.4g}."
        if "standard deviation" in q or re.search(r"\bstdev\b|\bstd\b", q):
            return f"The population standard deviation of {requested} is {stats['stdev']:.4g}."
        return (
            f"For {requested}: mean={stats['mean']:.4g}, median={stats['median']:.4g}, "
            f"min={stats['min']:.4g}, max={stats['max']:.4g}, sum={stats['sum']:.4g}."
        )
