from __future__ import annotations

from typing import Callable

from .types import Evidence


def build_reader_prompt(query: str, evidence: list[Evidence]) -> str:
    blocks = []
    for i, e in enumerate(evidence, 1):
        tr = e.trace
        stamp = tr.created_at.isoformat() if tr.created_at else "unknown-time"
        blocks.append(
            f"[E{i}] trace_id={tr.id} session={tr.session_id} turn={tr.turn_index} time={stamp}\n{tr.text}"
        )
    joined = "\n\n".join(blocks)
    return (
        "Answer the query using only the evidence below. If the evidence is insufficient, say so. "
        "Prefer exact names, dates and values appearing in evidence.\n\n"
        f"QUERY:\n{query}\n\nEVIDENCE:\n{joined}"
    )


def answer_with_reader(query: str, evidence: list[Evidence], llm_call: Callable[[str], str], calibrator=None, profile=None) -> str:
    answer = llm_call(build_reader_prompt(query, evidence))
    if calibrator is not None and profile is not None:
        return calibrator.calibrate_answer(answer, evidence, profile)
    return answer
