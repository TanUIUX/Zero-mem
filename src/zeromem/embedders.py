from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    """Offline deterministic embedder for tests/demo. Not intended for production."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self._dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += sign
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class BGEM3Embedder:
    """Dense BGE-M3 wrapper using FlagEmbedding."""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, max_length: int = 2048):
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.max_length = max_length
        probe = self.encode(["dimension probe"])
        self._dim = int(probe.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "_dim", 1024)), dtype=np.float32)
        result = self.model.encode(list(texts), batch_size=12, max_length=self.max_length)
        vecs = np.asarray(result["dense_vecs"], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
