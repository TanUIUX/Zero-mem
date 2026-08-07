# Zero-Mem Reference Implementation

A paper-inspired engineering implementation of **Zero-Mem: Zero-Token Memory Operations for LLM Agents**.

This project keeps raw interaction traces as the source of truth and builds two non-generative views over them:

1. **Entity-context graph** — NetworkX graph with entity↔context co-occurrence edges and context↔context adjacency; query-conditioned Personalized PageRank.
2. **Temporal hierarchy** — session → episode → window → turn/local-span retrieval.

Both views always run. A deterministic router only changes their fusion weights (`rho=0.6` by default). The pipeline then performs evidence closure (graph bridges + neighboring turns) and deterministic evidence calibration before the final LLM reader.

## Important fidelity note

The paper specifies the architecture, equations, `gamma=0.6`, `rho=0.6`, top-k evaluation setup, BM25 + BGE-M3 access signals, entity-context graph, temporal hierarchy, PPR, fusion, evidence closure and deterministic calibration. It does **not** fully disclose every implementation threshold or the exact episode segmentation/routing rules in v1. This repository therefore distinguishes:

- **Paper-specified behavior:** dual views, provenance, PPR, coarse-to-fine hierarchy, fusion, closure, deterministic filtering.
- **Engineering choices in this repo:** regex/spaCy NER fallback, 30-minute episode gap, 4-turn windows with stride 2, routing heuristics, conservative answer calibration.

These choices are deliberately isolated so they can be swapped when official code is released.

## Architecture

```text
Raw interaction traces
        |
        +----------------------+----------------------+
        |                                             |
        v                                             v
Entity-context graph                            Temporal hierarchy
(entity↔context + adjacency)                    session→episode→window→turn
        |                                             |
        v                                             v
Query-conditioned PPR                          coarse-to-fine search
        |                                             |
        +------------------ fusion -------------------+
                              |
                              v
                    evidence closure
                 (bridges + local neighbors)
                              |
                              v
                 deterministic calibration
                              |
                              v
                     final LLM reader
                   (only generative call)
```

## Quick start — offline demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python demo.py
```

The default demo uses `HashEmbedder` + in-memory vectors so it runs without downloading a model or starting Qdrant.

## Production-like stack: BGE-M3 + Qdrant + NetworkX

Start Qdrant:

```bash
docker compose up -d
```

Install BGE-M3 support:

```bash
pip install -e '.[bge]'
```

Configure:

```bash
export ZEROMEM_VECTOR_BACKEND=qdrant
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=zeromem_traces
export ZEROMEM_EMBEDDER=bge-m3
export BGE_MODEL=BAAI/bge-m3
```

Optional spaCy NER:

```bash
pip install -e '.[spacy]'
python -m spacy download xx_ent_wiki_sm
export ZEROMEM_NER=spacy
export SPACY_MODEL=xx_ent_wiki_sm
```

## Run the retrieval API

```bash
uvicorn zeromem.api:app --reload --port 8000
```

Ingest:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"traces":[
    {"id":"t1","session_id":"s1","turn_index":1,"text":"Alice joined Acme."},
    {"id":"t2","session_id":"s1","turn_index":2,"text":"Alice worked with Bob on Atlas."},
    {"id":"t3","session_id":"s1","turn_index":3,"text":"Bob moved to London."}
  ]}'
```

Retrieve:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Where did Alice collaborator move?","session_id":"s1"}'
```

## Integrate with an LLM agent

The memory layer should return evidence, not an answer. Keep the single generative call at the reader boundary:

```python
from zeromem.factory import create_engine
from zeromem.reader import build_reader_prompt

memory = create_engine()
evidence, debug = memory.retrieve(user_query, session_id=current_session)
prompt = build_reader_prompt(user_query, evidence)
answer = your_llm(prompt)  # only generative call in the memory/QA path
answer = memory.calibrator.calibrate_answer(answer, evidence, debug["profile"])
```

### Recommended agent lifecycle

```text
on user/tool event:
  1. append raw Trace
  2. embed + upsert vector/payload
  3. update BM25 index
  4. update graph + temporal hierarchy

on query:
  1. deterministic QueryProfile
  2. dense + lexical priors
  3. graph retrieval (entity match + PPR)
  4. hierarchy retrieval (episode → window → turn)
  5. min-max normalize + rho-weighted fusion
  6. top-k main evidence
  7. evidence closure
  8. deterministic filter/rerank
  9. one final LLM call
 10. conservative answer calibration
```

## Storage schema

Raw trace (authoritative record):

```json
{
  "id": "trace UUID",
  "session_id": "session UUID",
  "turn_index": 12,
  "text": "original interaction text",
  "created_at": "2026-08-07T12:34:56+07:00",
  "speaker": "user|assistant|tool",
  "metadata": {"tool": "calendar", "boundary": "workspace-A"}
}
```

Qdrant payload mirrors provenance fields (`session_id`, `turn_index`, `created_at`, `speaker`); the dense vector stores BGE-M3 output. NetworkX stores derived entity/context links, but every context node resolves back to a raw trace ID.

## Mapping from paper to code

| Paper component | Reference implementation |
|---|---|
| Provenance-preserving raw traces | `types.Trace`, `engine.traces` |
| BM25 + dense access | `lexical.BM25Index`, `vector_store.*`, `embedders.BGEM3Embedder` |
| Entity-context graph | `graph.EntityContextGraph` |
| Temporal hierarchy | `hierarchy.TemporalHierarchy` |
| Query profile / routing | `router.QueryRouter` |
| PPR graph ranking | `networkx.pagerank()` in `graph.search()` |
| Dual-view fusion | `engine.ZeroMemEngine.retrieve()` |
| Evidence closure | `bridge_contexts()` + `neighbors()` |
| Evidence calibration | `calibration.EvidenceCalibrator` |
| Final reader | `reader.build_reader_prompt()` |

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

## Next engineering upgrades

For a serious production deployment, add: persistent graph storage (Neo4j adapter), incremental BM25 persistence, payload indexes in Qdrant, tenant/workspace boundaries, trace deletion propagation, conflict/version semantics, explicit temporal parser, metrics/tracing, and LoCoMo/HotpotQA evaluation harnesses.
