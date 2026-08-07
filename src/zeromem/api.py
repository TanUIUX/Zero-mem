from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .factory import create_engine
from .types import Trace

app = FastAPI(title="Zero-Mem Reference API", version="0.1.0")
engine = create_engine()


class TraceIn(BaseModel):
    id: str
    session_id: str
    turn_index: int
    text: str
    created_at: datetime | None = None
    speaker: str | None = None
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    traces: list[TraceIn]


class RetrieveRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.post("/ingest")
def ingest(req: IngestRequest):
    traces = [Trace(**x.model_dump()) for x in req.traces]
    engine.ingest(traces)
    return {"ingested": len(traces)}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    evidence, debug = engine.retrieve(req.query, req.session_id)
    return {
        "profile": {
            "route": debug["profile"].route,
            "answer_type": debug["profile"].answer_type,
            "subjects": debug["profile"].subjects,
            "temporal_cues": debug["profile"].temporal_cues,
        },
        "evidence": [
            {
                "trace_id": e.trace.id,
                "session_id": e.trace.session_id,
                "turn_index": e.trace.turn_index,
                "text": e.trace.text,
                "score": e.score,
                "reasons": e.reasons,
            }
            for e in evidence
        ],
        "debug": {
            "main_trace_ids": debug["main_trace_ids"],
            "closed_trace_ids": debug["closed_trace_ids"],
        },
    }
