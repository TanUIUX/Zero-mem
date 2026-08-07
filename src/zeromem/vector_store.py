from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...
    def search(self, vector: np.ndarray, limit: int = 20, session_id: str | None = None) -> list[tuple[str, float, dict]]: ...


class InMemoryVectorStore:
    def __init__(self):
        self.rows: dict[str, tuple[np.ndarray, dict]] = {}

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        for i, v, p in zip(ids, vectors, payloads):
            self.rows[i] = (np.asarray(v, dtype=np.float32), dict(p))

    def search(self, vector: np.ndarray, limit: int = 20, session_id: str | None = None) -> list[tuple[str, float, dict]]:
        q = np.asarray(vector, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        out = []
        for doc_id, (v, p) in self.rows.items():
            if session_id and p.get("session_id") != session_id:
                continue
            score = float(np.dot(q, v) / (qn * (np.linalg.norm(v) or 1.0)))
            out.append((doc_id, score, p))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:limit]


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, vector_size: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url)
        self.collection = collection
        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct
        points = [PointStruct(id=i, vector=v.tolist(), payload=p) for i, v, p in zip(ids, vectors, payloads)]
        self.client.upsert(collection_name=self.collection, wait=True, points=points)

    def search(self, vector: np.ndarray, limit: int = 20, session_id: str | None = None) -> list[tuple[str, float, dict]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        query_filter = None
        if session_id:
            query_filter = Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))])
        points = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
        ).points
        return [(str(p.id), float(p.score), dict(p.payload or {})) for p in points]
