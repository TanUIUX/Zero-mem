from datetime import datetime, timedelta

from zeromem.embedders import HashEmbedder
from zeromem.engine import ZeroMemEngine
from zeromem.ner import RegexEntityExtractor
from zeromem.types import Trace
from zeromem.vector_store import InMemoryVectorStore


def make_engine():
    e = ZeroMemEngine(HashEmbedder(), InMemoryVectorStore(), RegexEntityExtractor(), top_k=3)
    base = datetime(2026, 1, 1, 9, 0)
    e.ingest([
        Trace("t1", "s1", 1, "Alice joined Acme as a product manager.", base),
        Trace("t2", "s1", 2, "Alice later worked closely with Bob on Atlas.", base + timedelta(minutes=1)),
        Trace("t3", "s1", 3, "Bob moved to London after the Atlas launch.", base + timedelta(minutes=2)),
        Trace("t4", "s1", 4, "The Atlas launch date was 2026-07-21.", base + timedelta(minutes=3)),
        Trace("x1", "s2", 1, "Alice ordered coffee in Paris.", base),
    ])
    return e


def test_relational_bridge_retrieves_bob_move():
    engine = make_engine()
    evidence, _ = engine.retrieve("Where did Alice's colleague move?", session_id="s1")
    ids = [e.trace.id for e in evidence]
    assert "t3" in ids


def test_session_boundary():
    engine = make_engine()
    evidence, _ = engine.retrieve("Where did Alice go?", session_id="s1")
    assert all(e.trace.session_id == "s1" for e in evidence)


def test_date_calibration_unique_candidate():
    engine = make_engine()
    evidence, debug = engine.retrieve("When was the Atlas launch?", session_id="s1")
    answer = engine.calibrator.calibrate_answer("It was in July.", evidence, debug["profile"])
    assert answer == "2026-07-21"
