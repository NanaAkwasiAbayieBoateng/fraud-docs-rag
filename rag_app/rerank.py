"""Second-stage reranking of retrieved chunks.

Hybrid search is good at recall — getting the right chunk *somewhere* in
the top 20 — but its fused score is a blunt instrument for ordering the
top 5 that actually reach the prompt. A reranker re-scores a small
candidate set with a more expensive, more precise method.

The production version of this stage is a cross-encoder (the query and
each chunk scored jointly by one model). This module ships a dependency-
free stand-in behind the same interface: ``LexicalReranker`` scores each
candidate by how much of the query it covers — distinct query terms
present, plus a bonus for the full phrase — blended with the first-stage
score. It is honest about being a heuristic; swapping in a cross-encoder
means writing one more class with the same ``rerank`` signature.
"""

from __future__ import annotations

from rag_app.index import SearchResult
from rag_app.ollama_client import tokenize


class LexicalReranker:
    """Re-score candidates by query-term coverage and phrase match."""

    def __init__(self, coverage_weight: float = 0.7, phrase_bonus: float = 0.15) -> None:
        self.coverage_weight = coverage_weight
        self.phrase_bonus = phrase_bonus

    def rerank(self, query: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        terms = set(tokenize(query))
        phrase = " ".join(query.lower().split())
        rescored = []
        for r in results:
            chunk_tokens = set(tokenize(r.chunk.text))
            coverage = len(terms & chunk_tokens) / len(terms) if terms else 0.0
            bonus = self.phrase_bonus if phrase and phrase in r.chunk.text.lower() else 0.0
            score = self.coverage_weight * coverage + (1 - self.coverage_weight) * r.score + bonus
            rescored.append(
                SearchResult(chunk=r.chunk, score=score,
                             vector_score=r.vector_score, bm25_score=r.bm25_score)
            )
        rescored.sort(key=lambda r: -r.score)
        return rescored[:k]
