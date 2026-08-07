from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(slots=True)
class Trace:
    id: str
    session_id: str
    turn_index: int
    text: str
    created_at: datetime | None = None
    speaker: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryProfile:
    query: str
    subjects: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    answer_type: Literal["date", "time", "number", "person", "location", "list", "text"] = "text"
    temporal_cues: list[str] = field(default_factory=list)
    session_id: str | None = None
    route: Literal["relational", "local"] = "local"


@dataclass(slots=True)
class Candidate:
    trace_id: str
    score: float
    source: str
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Evidence:
    trace: Trace
    score: float
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
