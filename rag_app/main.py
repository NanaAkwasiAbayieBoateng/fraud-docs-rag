"""FastAPI app: GET /health, POST /ask, POST /search.

/ask answers strictly from retrieved passages, returns citations, and
refuses when the top hit's cosine similarity falls below a threshold —
an answer with no evidence behind it is worse than no answer. (The gate
uses the raw cosine, not the fused ranking score: fusion normalises the
best hit to the top of the scale by construction, so it cannot gate.)

/search is the retrieval-only twin of /ask: same ranking, no LLM call.
It exists for debugging — when an answer looks wrong, /search shows
whether retrieval or generation is at fault.

Every /ask is appended to a JSONL query log (RAG_QUERY_LOG) with its
best score and refusal outcome: the data you need to tune the refusal
threshold and to see what users ask that the corpus cannot answer.

Run:
    python -m rag_app.ingest            # build data/index.json first
    uvicorn rag_app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_app.index import HybridIndex
from rag_app.ollama_client import OllamaClient, make_client
from rag_app.rerank import LexicalReranker

DEFAULT_INDEX_PATH = "data/index.json"
DEFAULT_REFUSAL_THRESHOLD = 0.25
DEFAULT_TOP_K = 5
DEFAULT_FUSION = "linear"
DEFAULT_QUERY_LOG = "logs/queries.jsonl"
RERANK_FETCH_CAP = 20

REFUSAL_MESSAGE = (
    "I don't have guidance on that in the documents I can search. "
    "Try rephrasing, or ask about a topic covered by the fraud/AML guidance."
)

SYSTEM_PROMPT = (
    "You are a fraud-prevention assistant. Answer the question using ONLY "
    "the numbered passages provided. If the passages do not answer the "
    "question, say so plainly. Cite passages inline like [1] or [2]. "
    "Keep the answer concise and practical."
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = None
    source: str | None = None  # restrict retrieval to one document


class Citation(BaseModel):
    source: str
    chunk_index: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    best_score: float


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = None
    source: str | None = None


class SearchHit(BaseModel):
    source: str
    chunk_index: int
    score: float
    vector_score: float
    bm25_score: float
    excerpt: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    best_score: float


def _load_index(index_path: str | Path) -> HybridIndex:
    path = Path(index_path)
    if path.is_file():
        return HybridIndex.load_from(path)
    return HybridIndex()


def _append_query_log(log_path: str | Path | None, record: dict) -> None:
    """Append one JSONL record; logging must never break a request."""
    if not log_path:
        return
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


def create_app(
    client=None,
    index: HybridIndex | None = None,
    index_path: str | Path | None = None,
    refusal_threshold: float | None = None,
    fusion: str | None = None,
    rerank: bool | None = None,
    log_path: str | Path | None = None,
) -> FastAPI:
    client = client if client is not None else make_client()
    if index is None:
        index = _load_index(index_path or os.getenv("RAG_INDEX_PATH", DEFAULT_INDEX_PATH))
    threshold = (
        refusal_threshold
        if refusal_threshold is not None
        else float(os.getenv("RAG_REFUSAL_THRESHOLD", DEFAULT_REFUSAL_THRESHOLD))
    )
    top_k_default = int(os.getenv("RAG_TOP_K", DEFAULT_TOP_K))
    fusion = fusion or os.getenv("RAG_FUSION", DEFAULT_FUSION)
    if fusion not in ("linear", "rrf"):
        raise ValueError(f"unknown fusion strategy: {fusion!r}")
    rerank_enabled = (
        rerank if rerank is not None else os.getenv("RAG_RERANK", "0") == "1"
    )
    reranker = LexicalReranker() if rerank_enabled else None
    query_log = log_path if log_path is not None else os.getenv(
        "RAG_QUERY_LOG", DEFAULT_QUERY_LOG
    )

    app = FastAPI(title="Fraud Docs RAG", version="0.2.0")
    app.state.client = client
    app.state.index = index
    app.state.refusal_threshold = threshold
    app.state.fusion = fusion
    app.state.rerank = rerank_enabled

    def _retrieve(question: str, k: int, source: str | None):
        embedding = client.embed_one(question)
        sources = {source} if source else None
        return index.search(question, embedding, k=k, fusion=fusion, sources=sources)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "chat_model": getattr(client, "chat_model", "unknown"),
            "embed_model": getattr(client, "embed_model", "unknown"),
            "chunks": len(index),
            "sources": sorted({c.source for c in index.chunks}),
            "fusion": fusion,
            "rerank": rerank_enabled,
        }

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        k = request.k or top_k_default
        results = _retrieve(request.question, k, request.source)
        return SearchResponse(
            hits=[
                SearchHit(
                    source=r.chunk.source,
                    chunk_index=r.chunk.chunk_index,
                    score=round(r.score, 4),
                    vector_score=round(r.vector_score, 4),
                    bm25_score=round(r.bm25_score, 4),
                    excerpt=r.chunk.text[:200],
                )
                for r in results
            ],
            best_score=round(results[0].vector_score, 4) if results else 0.0,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        k = request.k or top_k_default
        fetch_k = min(max(k * 4, k), RERANK_FETCH_CAP) if reranker else k
        results = _retrieve(request.question, fetch_k, request.source)
        best_score = results[0].vector_score if results else 0.0

        if not results or best_score < threshold:
            _append_query_log(query_log, {
                "ts": datetime.now(timezone.utc).isoformat(),
                "question": request.question,
                "refused": True,
                "best_score": round(best_score, 4),
                "sources": [],
            })
            return AskResponse(
                answer=REFUSAL_MESSAGE,
                refused=True,
                citations=[],
                best_score=best_score,
            )

        if reranker:
            results = reranker.rerank(request.question, results, k)
        else:
            results = results[:k]

        passages = "\n\n".join(
            f"[{i + 1}] ({r.chunk.source}) {r.chunk.text}"
            for i, r in enumerate(results)
        )
        prompt = f"Passages:\n{passages}\n\nQuestion: {request.question}"
        answer = client.chat(prompt, system=SYSTEM_PROMPT)
        citations = [
            Citation(
                source=r.chunk.source,
                chunk_index=r.chunk.chunk_index,
                score=round(r.score, 4),
                excerpt=r.chunk.text[:200],
            )
            for r in results
        ]
        _append_query_log(query_log, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": request.question,
            "refused": False,
            "best_score": round(best_score, 4),
            "sources": sorted({c.source for c in citations}),
        })
        return AskResponse(
            answer=answer, refused=False, citations=citations, best_score=best_score
        )

    return app


app = create_app()
