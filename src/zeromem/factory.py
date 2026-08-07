from __future__ import annotations

import os

from .embedders import BGEM3Embedder, HashEmbedder
from .engine import ZeroMemEngine
from .ner import RegexEntityExtractor, SpacyEntityExtractor
from .vector_store import InMemoryVectorStore, QdrantVectorStore


def create_engine() -> ZeroMemEngine:
    embedder_name = os.getenv("ZEROMEM_EMBEDDER", "hash").lower()
    if embedder_name == "bge-m3":
        embedder = BGEM3Embedder(os.getenv("BGE_MODEL", "BAAI/bge-m3"))
    else:
        embedder = HashEmbedder()

    if os.getenv("ZEROMEM_NER", "regex").lower() == "spacy":
        entities = SpacyEntityExtractor(os.getenv("SPACY_MODEL", "xx_ent_wiki_sm"))
    else:
        entities = RegexEntityExtractor()

    backend = os.getenv("ZEROMEM_VECTOR_BACKEND", "memory").lower()
    if backend == "qdrant":
        store = QdrantVectorStore(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection=os.getenv("QDRANT_COLLECTION", "zeromem_traces"),
            vector_size=embedder.dim,
        )
    else:
        store = InMemoryVectorStore()
    return ZeroMemEngine(embedder, store, entities)
