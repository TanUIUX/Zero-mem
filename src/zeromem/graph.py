from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np

from .embedders import Embedder
from .ner import EntityExtractor
from .types import Trace


class EntityContextGraph:
    def __init__(self, embedder: Embedder, entity_extractor: EntityExtractor, gamma: float = 0.6):
        self.embedder = embedder
        self.entity_extractor = entity_extractor
        self.gamma = gamma
        self.g = nx.Graph()
        self.entity_vectors: dict[str, np.ndarray] = {}
        self.trace_entities: dict[str, list[str]] = defaultdict(list)
        self.session_turn_to_trace: dict[tuple[str, int], str] = {}

    @staticmethod
    def cnode(trace_id: str) -> str:
        return f"ctx::{trace_id}"

    @staticmethod
    def enode(entity: str) -> str:
        return f"ent::{entity.lower()}"

    def add_trace(self, trace: Trace, vector: np.ndarray) -> None:
        c = self.cnode(trace.id)
        self.g.add_node(c, kind="context", trace_id=trace.id, vector=vector)
        self.session_turn_to_trace[(trace.session_id, trace.turn_index)] = trace.id
        entities = self.entity_extractor.extract(trace.text)
        self.trace_entities[trace.id] = entities
        if entities:
            evecs = self.embedder.encode(entities)
            counts = defaultdict(int)
            for e in entities:
                counts[e.lower()] += 1
            total = sum(counts.values())
            for e, evec in zip(entities, evecs):
                en = self.enode(e)
                self.g.add_node(en, kind="entity", label=e)
                self.entity_vectors[en] = evec
                weight = counts[e.lower()] / max(total, 1)
                self.g.add_edge(c, en, weight=max(weight, 1e-3), edge_type="entity_context")
        prev_id = self.session_turn_to_trace.get((trace.session_id, trace.turn_index - 1))
        if prev_id:
            self.g.add_edge(c, self.cnode(prev_id), weight=0.2, edge_type="adjacent_context")

    def matched_entities(self, query_vector: np.ndarray, top_n: int = 3, min_score: float = 0.2) -> list[tuple[str, float]]:
        scored = []
        qn = np.linalg.norm(query_vector) or 1.0
        for node, v in self.entity_vectors.items():
            s = float(np.dot(query_vector, v) / (qn * (np.linalg.norm(v) or 1.0)))
            if s >= min_score:
                scored.append((node, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def search(self, query_vector: np.ndarray, dense_priors: dict[str, float], limit: int = 30) -> list[tuple[str, float]]:
        if self.g.number_of_nodes() == 0:
            return []
        personalization: dict[str, float] = defaultdict(float)
        for tid, score in dense_priors.items():
            c = self.cnode(tid)
            if c in self.g:
                personalization[c] += max(score, 0.0)
        for en, score in self.matched_entities(query_vector):
            personalization[en] += max(score, 0.0)
            for ctx in self.g.neighbors(en):
                if self.g.nodes[ctx].get("kind") != "context":
                    continue
                prior = dense_priors.get(self.g.nodes[ctx]["trace_id"], 0.0)
                for en2 in self.g.neighbors(ctx):
                    if self.g.nodes[en2].get("kind") == "entity":
                        personalization[en2] += max(prior, 0.0) * 0.5
        total = sum(personalization.values())
        if total <= 0:
            return []
        personalization = {k: v / total for k, v in personalization.items()}
        pr = nx.pagerank(self.g, alpha=self.gamma, personalization=personalization, weight="weight", max_iter=200)
        out = []
        for node, score in pr.items():
            if self.g.nodes[node].get("kind") == "context":
                out.append((self.g.nodes[node]["trace_id"], float(score)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:limit]

    def bridge_contexts(self, trace_ids: list[str], limit: int = 3) -> list[str]:
        entity_counts = defaultdict(int)
        for tid in trace_ids:
            for e in self.trace_entities.get(tid, []):
                entity_counts[self.enode(e)] += 1
        scored = defaultdict(float)
        main = set(trace_ids)
        for en, cnt in entity_counts.items():
            if en not in self.g:
                continue
            for ctx in self.g.neighbors(en):
                if self.g.nodes[ctx].get("kind") != "context":
                    continue
                tid = self.g.nodes[ctx]["trace_id"]
                if tid in main:
                    continue
                scored[tid] += cnt
        return [tid for tid, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)[:limit]]
