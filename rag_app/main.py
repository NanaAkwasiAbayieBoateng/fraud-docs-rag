"""FastAPI app: GET /health and POST /ask.

/ask answers strictly from retrieved passages, returns citations, and
refuses when the top hit's cosine similarity falls below a threshold —
an answer with no evidence behind it is worse than no answer. (The gate
uses the raw cosine, not the fused ranking score: fusion normalises the
best hit to the top of the scale by construction, so it cannot gate.)

Run:
    python -m rag_app.ingest            # build data/index.json first
    uvicorn rag_app.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_app.index import HybridIndex
from rag_app.ollama_client import OllamaClient, make_client

DEFAULT_INDEX_PATH = "data/index.json"
DEFAULT_REFUSAL_THRESHOLD = 0.25
DEFAULT_TOP_K = 5

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


def _load_index(index_path: str | Path) -> HybridIndex:
    path = Path(index_path)
    if path.is_file():
        return HybridIndex.load(path)
    return HybridIndex()


def create_app(
    client=None,
    index: HybridIndex | None = None,
    index_path: str | Path | None = None,
    refusal_threshold: float | None = None,
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

    app = FastAPI(title="Fraud Docs RAG", version="0.1.0")
    app.state.client = client
    app.state.index = index
    app.state.refusal_threshold = threshold

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "chat_model": getattr(client, "chat_model", "unknown"),
            "embed_model": getattr(client, "embed_model", "unknown"),
            "chunks": len(index),
            "sources": sorted({c.source for c in index.chunks}),
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        k = request.k or top_k_default
        embedding = client.embed_one(request.question)
        results = index.search(request.question, embedding, k=k)
        best_score = results[0].vector_score if results else 0.0

        if not results or best_score < threshold:
            return AskResponse(
                answer=REFUSAL_MESSAGE,
                refused=True,
                citations=[],
                best_score=best_score,
            )

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
        return AskResponse(
            answer=answer, refused=False, citations=citations, best_score=best_score
        )

    return app


app = create_app()
