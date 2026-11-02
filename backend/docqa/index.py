"""Hybrid index: BM25 + dense vectors, merged with Reciprocal Rank Fusion.

RRF(d) = sum over retrievers r of 1 / (k + rank_r(d)), with k = 60.
RRF only uses ranks, so it needs no score calibration between BM25 (unbounded)
and cosine similarity (-1..1).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunking import Chunk
from .dense import make_embedder

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class HybridIndex:
    def __init__(self, embedder: str = "lsa"):
        self.embedder_spec = embedder
        self.chunks: list[Chunk] = []
        self.bm25 = BM25()
        self.embedder = None
        self.vectors: np.ndarray | None = None

    @staticmethod
    def _doc_text(c: Chunk) -> str:
        # The heading is part of what a chunk is "about", so it is indexed with the body.
        return f"{c.heading}\n{c.text}"

    def build(self, chunks: list[Chunk]) -> HybridIndex:
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        self.chunks = list(chunks)
        texts = [self._doc_text(c) for c in self.chunks]
        self.bm25.fit(texts)
        self.embedder = make_embedder(self.embedder_spec).fit(texts)
        self.vectors = self.embedder.encode(texts)
        return self

    def search_bm25(self, query: str, k: int) -> list[int]:
        return [i for i, _ in self.bm25.search(query, k)]

    def search_dense(self, query: str, k: int) -> list[int]:
        q = self.embedder.encode([query])[0]
        sims = self.vectors @ q
        return list(np.argsort(-sims)[:k])

    def search(self, query: str, k: int = 5, mode: str = "hybrid", pool: int = 50) -> list[Hit]:
        if mode == "bm25":
            ranked = self.search_bm25(query, k)
            return [Hit(self.chunks[i], 1.0 / (r + 1), r) for r, i in enumerate(ranked)]
        if mode == "dense":
            ranked = self.search_dense(query, k)
            return [Hit(self.chunks[i], 1.0 / (r + 1), r) for r, i in enumerate(ranked)]
        if mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}")
        fused: dict[int, float] = {}
        for ranked in (self.search_bm25(query, pool), self.search_dense(query, pool)):
            for r, i in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(self.chunks[i], s, r) for r, (i, s) in enumerate(top)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> HybridIndex:
        with open(path, "rb") as f:
            return pickle.load(f)
