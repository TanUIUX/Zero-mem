from datetime import datetime, timedelta

from zeromem.embedders import HashEmbedder
from zeromem.engine import ZeroMemEngine
from zeromem.ner import RegexEntityExtractor
from zeromem.types import Trace
from zeromem.vector_store import InMemoryVectorStore

base = datetime(2026, 8, 1, 9, 0)
traces = [
    Trace("t1", "s1", 1, "Alice joined Acme as a product manager.", base),
    Trace("t2", "s1", 2, "Alice later worked closely with Bob on the Atlas project.", base + timedelta(minutes=2)),
    Trace("t3", "s1", 3, "Bob moved to London after the Atlas launch.", base + timedelta(minutes=4)),
    Trace("t4", "s1", 4, "The launch date was 2026-07-21.", base + timedelta(minutes=6)),
]

engine = ZeroMemEngine(HashEmbedder(), InMemoryVectorStore(), RegexEntityExtractor())
engine.ingest(traces)
for query in ["Where did Alice's colleague move?", "When was the Atlas launch?"]:
    evidence, debug = engine.retrieve(query)
    print("\nQUERY:", query)
    print("ROUTE:", debug["profile"].route)
    for e in evidence:
        print(f"  {e.score:.3f} {e.trace.id}: {e.trace.text}")
