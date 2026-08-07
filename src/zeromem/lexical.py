from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: dict[str, list[str]] = {}
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self.docs:
            self.remove(doc_id)
        tokens = tokenize(text)
        self.docs[doc_id] = tokens
        self.df.update(set(tokens))
        self._recompute_avgdl()

    def remove(self, doc_id: str) -> None:
        tokens = self.docs.pop(doc_id, None)
        if tokens is None:
            return
        for t in set(tokens):
            self.df[t] -= 1
            if self.df[t] <= 0:
                del self.df[t]
        self._recompute_avgdl()

    def _recompute_avgdl(self) -> None:
        self.avgdl = (sum(map(len, self.docs.values())) / len(self.docs)) if self.docs else 0.0

    def score(self, query: str, doc_id: str) -> float:
        tokens = self.docs.get(doc_id)
        if not tokens:
            return 0.0
        q = tokenize(query)
        tf = Counter(tokens)
        n = len(self.docs)
        dl = len(tokens)
        total = 0.0
        for term in q:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            freq = tf.get(term, 0)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            total += idf * (freq * (self.k1 + 1) / max(denom, 1e-9))
        return total

    def search(self, query: str, limit: int = 20, candidate_ids: set[str] | None = None) -> list[tuple[str, float]]:
        ids = candidate_ids if candidate_ids is not None else set(self.docs)
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in ids]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
