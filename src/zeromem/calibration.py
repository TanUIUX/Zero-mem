from __future__ import annotations

import re

from .types import Evidence, QueryProfile, Trace


_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b")
_NUM_RE = re.compile(r"(?<!\w)-?\d+(?:[.,]\d+)?(?!\w)")


class EvidenceCalibrator:
    def filter_and_rank(self, traces: list[Trace], scores: dict[str, float], profile: QueryProfile) -> list[Evidence]:
        out: list[Evidence] = []
        qwords = set(profile.keywords)
        subjects = {s.lower() for s in profile.subjects}
        for tr in traces:
            if profile.session_id and tr.session_id != profile.session_id:
                continue
            text_l = tr.text.lower()
            bonus = 0.0
            reasons = []
            if subjects and any(s in text_l for s in subjects):
                bonus += 0.10
                reasons.append("subject")
            overlap = sum(1 for w in qwords if w in text_l)
            if overlap:
                bonus += min(0.15, overlap * 0.03)
                reasons.append("lexical")
            if profile.answer_type == "date" and _DATE_RE.search(tr.text):
                bonus += 0.08
                reasons.append("answer-type:date")
            elif profile.answer_type == "number" and _NUM_RE.search(tr.text):
                bonus += 0.05
                reasons.append("answer-type:number")
            out.append(Evidence(trace=tr, score=scores.get(tr.id, 0.0) + bonus, reasons=reasons))
        out.sort(key=lambda e: (e.score, e.trace.turn_index), reverse=True)
        return out

    def calibrate_answer(self, answer: str, evidence: list[Evidence], profile: QueryProfile) -> str:
        if profile.answer_type == "date":
            vals = []
            for e in evidence:
                vals.extend(_DATE_RE.findall(e.trace.text))
            uniq = list(dict.fromkeys(vals))
            if len(uniq) == 1 and uniq[0] not in answer:
                return uniq[0]
        if profile.answer_type == "number":
            vals = []
            for e in evidence:
                vals.extend(_NUM_RE.findall(e.trace.text))
            uniq = list(dict.fromkeys(vals))
            if len(uniq) == 1 and uniq[0] not in answer:
                return uniq[0]
        return answer
