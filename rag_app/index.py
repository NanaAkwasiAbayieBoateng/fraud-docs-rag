"""Hybrid vector + BM25 index, with pluggable persistence.

The vector side is a plain numpy cosine-similarity store — at this corpus
size (hundreds of documents) a dedicated vector database would be
ceremony, and the swap-in point is this class. BM25 comes from
``rank_bm25``.

Two fusion strategies are supported: ``linear`` (normalise each score to
[0, 1] and take a weighted sum) and ``rrf`` (Reciprocal Rank Fusion,
which fuses the two *rankings* and so is immune to score-scale
differences). Persistence also has two backends behind ``save_to`` /
``load_from``: a single JSON file, or SQLite for larger corpora that
outgrow loading one JSON blob into memory. A production vector database
(Qdrant, pgvector) would slot in behind the same two methods.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

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
    @staticmethod
    def _rrf_fuse(vec: np.ndarray, bm: np.ndarray, candidates: np.ndarray,
                  rrf_k: int = 60) -> np.ndarray:
        """Reciprocal Rank Fusion over the candidate set.

        Each ranking contributes 1 / (rrf_k + rank) per chunk, so only the
        *order* matters, never the score scales — the property that makes
        RRF robust where normalised-score fusion is not.
        """
        fused = np.zeros(len(candidates))
        position = {int(idx): pos for pos, idx in enumerate(candidates)}
        for scores in (vec, bm):
            order = candidates[np.argsort(-scores[candidates])]
            for rank, idx in enumerate(order):
                fused[position[int(idx)]] += 1.0 / (rrf_k + rank + 1)
        return fused

    def search(
        self,
        query: str,
        query_embedding: Sequence[float],
        k: int = 5,
        alpha: float = 0.5,
        fusion: str = "linear",
        sources: Iterable[str] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search.

        ``alpha`` weights the vector side under ``fusion="linear"``
        (1-alpha = BM25). ``fusion="rrf"`` uses Reciprocal Rank Fusion
        instead. ``sources`` optionally restricts the search to chunks
        from the named documents.
        """
        if fusion not in ("linear", "rrf"):
            raise ValueError(f"unknown fusion strategy: {fusion!r}")
        if not len(self.chunks):
            return []
        candidates = np.arange(len(self.chunks))
        if sources is not None:
            allowed = set(sources)
            candidates = np.array(
                [i for i in candidates if self.chunks[i].source in allowed]
            )
            if candidates.size == 0:
                return []
        vec = self._vector_scores(query_embedding)
        bm = self._bm25_scores(query)
        if fusion == "rrf":
            fused_candidates = self._rrf_fuse(vec, bm, candidates)
            fused = np.zeros(len(self.chunks))
            fused[candidates] = fused_candidates
        else:
            fused = alpha * self._normalise(np.clip(vec, 0.0, None)) + (1 - alpha) * self._normalise(bm)
        order = candidates[np.argsort(-fused[candidates])][:k]
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
    def save_to(self, path: str | Path) -> None:
        """Save via the backend implied by the file suffix (.db/.sqlite → SQLite)."""
        if Path(path).suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            self.save_sqlite(path)
        else:
            self.save(path)

    @classmethod
    def load_from(cls, path: str | Path) -> "HybridIndex":
        """Load via the backend implied by the file suffix (see save_to)."""
        if Path(path).suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            return cls.load_sqlite(path)
        return cls.load(path)

    def save_sqlite(self, path: str | Path) -> None:
        """Persist chunks + embeddings in a SQLite database.

        One row per chunk; embeddings are stored as float64 blobs. Unlike
        the JSON backend, the file can be inspected and updated with
        ordinary SQL tooling, and rows can be read back selectively —
        the shape a real vector store would take behind this interface.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE chunks (id INTEGER PRIMARY KEY, source TEXT, "
                "chunk_index INTEGER, text TEXT, metadata TEXT, embedding BLOB)"
            )
            rows = []
            for i, chunk in enumerate(self.chunks):
                emb = (
                    np.asarray(self._embeddings[i], dtype=np.float64).tobytes()
                    if self._embeddings is not None else None
                )
                rows.append((i, chunk.source, chunk.chunk_index, chunk.text,
                             json.dumps(chunk.metadata), emb))
            conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", rows)

    @classmethod
    def load_sqlite(cls, path: str | Path) -> "HybridIndex":
        with sqlite3.connect(Path(path)) as conn:
            rows = conn.execute(
                "SELECT source, chunk_index, text, metadata, embedding "
                "FROM chunks ORDER BY id"
            ).fetchall()
        chunks, embeddings = [], []
        for source, chunk_index, text, metadata, emb in rows:
            chunks.append(Chunk(text=text, source=source, chunk_index=chunk_index,
                                metadata=json.loads(metadata or "{}")))
            embeddings.append(
                np.frombuffer(emb, dtype=np.float64) if emb is not None else None
            )
        if embeddings and all(e is not None for e in embeddings):
            return cls(chunks=chunks, embeddings=embeddings)
        return cls(chunks=chunks)

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
