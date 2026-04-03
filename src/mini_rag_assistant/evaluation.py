from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mini_rag_assistant.pipeline import MiniRAGAssistant


@dataclass(slots=True)
class EvaluationCase:
    question: str
    should_refuse: bool = False
    expected_titles: list[str] | None = None
    expected_answer_contains: list[str] | None = None


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        cases.append(
            EvaluationCase(
                question=payload["question"],
                should_refuse=payload.get("should_refuse", False),
                expected_titles=payload.get("expected_titles"),
                expected_answer_contains=payload.get("expected_answer_contains"),
            )
        )
    if not cases:
        raise ValueError("No evaluation cases were found.")
    return cases


def run_evaluation(
    assistant: MiniRAGAssistant,
    cases: list[EvaluationCase],
    *,
    top_k: int = 4,
    min_score: float = 0.15,
    relative_score_floor: float = 0.55,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    passed = 0

    for case in cases:
        answer, _ = assistant.answer(
            case.question,
            top_k=top_k,
            min_score=min_score,
            relative_score_floor=relative_score_floor,
        )
        citation_titles = [citation.title for citation in answer.citations]
        normalized_answer = answer.answer.lower()

        checks: list[bool] = []
        checks.append(answer.refused == case.should_refuse)

        if case.expected_titles:
            checks.append(any(title in citation_titles for title in case.expected_titles))

        if case.expected_answer_contains:
            checks.extend(fragment.lower() in normalized_answer for fragment in case.expected_answer_contains)

        row_passed = all(checks)
        passed += int(row_passed)
        rows.append(
            {
                "question": case.question,
                "passed": row_passed,
                "refused": answer.refused,
                "answer": answer.answer,
                "citations": citation_titles,
                "confidence": answer.confidence,
            }
        )

    return {
        "total": len(cases),
        "passed": passed,
        "pass_rate": round(passed / len(cases), 3),
        "rows": rows,
    }
