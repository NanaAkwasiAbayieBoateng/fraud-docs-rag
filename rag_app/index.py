"""Hybrid vector + BM25 index, persisted as a single JSON file.

The vector side is a plain numpy cosine-similarity store — at this corpus
size (hundreds of documents) a dedicated vector database would be
ceremony, and the swap-in point is this class. BM25 comes from
``rank_bm25``. Hybrid search fuses the two rankings by normalising each
score to [0, 1] within the result set and taking a weighted sum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from rag_app.ollama_client import tokenize


@dataclass
class Chunk:
    text: str
    source: str  # originating document name, used for citations + eval
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            text=data["text"],
            source=data["source"],
            chunk_index=data.get("chunk_index", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    vector_score: float
    bm25_score: float


class HybridIndex:
    def __init__(
        self,
        chunks: Sequence[Chunk] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.chunks: list[Chunk] = list(chunks or [])
        if embeddings is not None and len(embeddings) != len(self.chunks):
            raise ValueError("embeddings and chunks must have the same length")
        self._embeddings: np.ndarray | None = (
            np.asarray(embeddings, dtype=np.float64) if embeddings is not None else None
        )
        self._bm25: BM25Okapi | None = None
        if self.chunks:
            self._build_bm25()

    def __len__(self) -> int:
        return len(self.chunks)

    def _build_bm25(self) -> None:
        corpus = [tokenize(c.text) or [""] for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks.extend(chunks)
        new = np.asarray(embeddings, dtype=np.float64)
        self._embeddings = new if self._embeddings is None else np.vstack([self._embeddings, new])
        if self.chunks:
            self._build_bm25()

    # -- scoring ---------------------------------------------------------
    def _vector_scores(self, query_embedding: Sequence[float]) -> np.ndarray:
        if self._embeddings is None or not len(self.chunks):
            return np.zeros(len(self.chunks))
        q = np.asarray(query_embedding, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        doc_norms = np.linalg.norm(self._embeddings, axis=1)
        denom = doc_norms * q_norm
        denom[denom == 0] = 1.0
        return (self._embeddings @ q) / denom

    def _bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None or not len(self.chunks):
            return np.zeros(len(self.chunks))
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.chunks))
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    @staticmethod
    def _normalise(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        top = float(scores.max())
        if top <= 0:
            return np.zeros_like(scores)
        return scores / top

    # -- search ----------------------------------------------------------
    def search(
        self,
        query: str,
        query_embedding: Sequence[float],
        k: int = 5,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Hybrid search. ``alpha`` weights the vector side (1-alpha = BM25)."""
        if not len(self.chunks):
            return []
        vec = self._vector_scores(query_embedding)
        bm = self._bm25_scores(query)
        fused = alpha * self._normalise(np.clip(vec, 0.0, None)) + (1 - alpha) * self._normalise(bm)
        order = np.argsort(-fused)[:k]
        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(fused[i]),
                vector_score=float(vec[i]),
                bm25_score=float(bm[i]),
            )
            for i in order
        ]

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "embeddings": (
                self._embeddings.tolist() if self._embeddings is not None else []
            ),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(c) for c in payload["chunks"]]
        embeddings = payload.get("embeddings") or None
        return cls(chunks=chunks, embeddings=embeddings)
