from __future__ import annotations

from collections import defaultdict

import numpy as np

from .calibration import EvidenceCalibrator
from .embedders import Embedder
from .graph import EntityContextGraph
from .hierarchy import TemporalHierarchy
from .lexical import BM25Index
from .ner import EntityExtractor
from .router import QueryRouter
from .types import Evidence, Trace
from .vector_store import VectorStore


def _minmax(items: list[tuple[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    vals = [s for _, s in items]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {i: 1.0 for i, _ in items}
    return {i: (s - lo) / (hi - lo) for i, s in items}


class ZeroMemEngine:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, entity_extractor: EntityExtractor, rho: float = 0.6, gamma: float = 0.6, top_k: int = 5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.entities = entity_extractor
        self.rho = rho
        self.top_k = top_k
        self.traces: dict[str, Trace] = {}
        self.lexical = BM25Index()
        self.graph = EntityContextGraph(embedder, entity_extractor, gamma=gamma)
        self.hierarchy = TemporalHierarchy()
        self.router = QueryRouter(entity_extractor)
        self.calibrator = EvidenceCalibrator()

    def ingest(self, traces: list[Trace]) -> None:
        if not traces:
            return
        vectors = self.embedder.encode([t.text for t in traces])
        payloads = []
        for tr, vec in zip(traces, vectors):
            self.traces[tr.id] = tr
            self.lexical.add(tr.id, tr.text)
            self.graph.add_trace(tr, vec)
            self.hierarchy.add_trace(tr)
            payloads.append({
                "session_id": tr.session_id,
                "turn_index": tr.turn_index,
                "created_at": tr.created_at.isoformat() if tr.created_at else None,
                "speaker": tr.speaker,
            })
        self.vector_store.upsert([t.id for t in traces], vectors, payloads)

    def _dense_and_lexical(self, query: str, qvec: np.ndarray, session_id: str | None) -> tuple[list[tuple[str, float]], dict[str, float]]:
        dense = self.vector_store.search(qvec, limit=max(30, self.top_k * 6), session_id=session_id)
        dense_scores = {tid: score for tid, score, _ in dense}
        lex = self.lexical.search(query, limit=max(30, self.top_k * 6), candidate_ids=set(dense_scores) if dense_scores else None)
        lex_n = _minmax(lex)
        priors = {tid: score + 0.15 * lex_n.get(tid, 0.0) for tid, score in dense_scores.items()}
        return [(tid, s) for tid, s in priors.items()], priors

    def _hierarchical_search(self, query: str, qvec: np.ndarray, dense_priors: dict[str, float], session_id: str | None) -> list[tuple[str, float]]:
        episode_scores = []
        for eid, unit in self.hierarchy.episodes.items():
            if session_id and unit.session_id != session_id:
                continue
            member_scores = [dense_priors.get(tid, 0.0) + 0.10 * self.lexical.score(query, tid) for tid in unit.trace_ids]
            episode_scores.append((eid, max(member_scores, default=0.0)))
        episode_scores.sort(key=lambda x: x[1], reverse=True)
        chosen_eps = {eid for eid, _ in episode_scores[: max(3, self.top_k)]}
        window_scores = []
        for wid, unit in self.hierarchy.windows.items():
            if not any(self.hierarchy.trace_to_episode.get(tid) in chosen_eps for tid in unit.trace_ids):
                continue
            member_scores = [dense_priors.get(tid, 0.0) + 0.10 * self.lexical.score(query, tid) for tid in unit.trace_ids]
            window_scores.append((wid, max(member_scores, default=0.0)))
        window_scores.sort(key=lambda x: x[1], reverse=True)
        chosen_windows = {wid for wid, _ in window_scores[: max(5, self.top_k * 2)]}
        scores = defaultdict(float)
        for wid in chosen_windows:
            unit = self.hierarchy.windows[wid]
            for tid in unit.trace_ids:
                scores[tid] = max(scores[tid], dense_priors.get(tid, 0.0) + 0.10 * self.lexical.score(query, tid))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(30, self.top_k * 6)]

    def retrieve(self, query: str, session_id: str | None = None) -> tuple[list[Evidence], dict]:
        profile = self.router.profile(query, session_id=session_id)
        qvec = self.embedder.encode([query])[0]
        _, dense_priors = self._dense_and_lexical(query, qvec, session_id)
        graph_raw = self.graph.search(qvec, dense_priors, limit=max(30, self.top_k * 6))
        hierarchy_raw = self._hierarchical_search(query, qvec, dense_priors, session_id)
        g = _minmax(graph_raw)
        h = _minmax(hierarchy_raw)
        all_ids = set(g) | set(h)
        fused = {}
        for tid in all_ids:
            if profile.route == "relational":
                fused[tid] = self.rho * g.get(tid, 0.0) + (1 - self.rho) * h.get(tid, 0.0)
            else:
                fused[tid] = self.rho * h.get(tid, 0.0) + (1 - self.rho) * g.get(tid, 0.0)
        main = [tid for tid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self.top_k]]
        closure = list(main)
        closure.extend(self.graph.bridge_contexts(main, limit=3))
        for tid in main:
            if tid in self.hierarchy.traces:
                closure.extend(self.hierarchy.neighbors(tid, radius=1))
        closure = list(dict.fromkeys(closure))
        scores = dict(fused)
        for tid in closure:
            if tid not in scores:
                scores[tid] = 0.25 * max((fused.get(m, 0.0) for m in main), default=0.0)
        traces = [self.traces[tid] for tid in closure if tid in self.traces]
        evidence = self.calibrator.filter_and_rank(traces, scores, profile)
        evidence = evidence[: max(self.top_k + 3, self.top_k)]
        debug = {
            "profile": profile,
            "graph_ranking": graph_raw[:10],
            "hierarchy_ranking": hierarchy_raw[:10],
            "main_trace_ids": main,
            "closed_trace_ids": closure,
        }
        return evidence, debug
