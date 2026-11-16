# Fraud Docs RAG — Retrieval over Fraud & AML Guidance

**Status:** Scaffold built — runnable · **Roadmap stage:** Intermediate (Project P3)
Part of the [ML / AI Engineer Roadmap](https://github.com/NanaAkwasiAbayieBoateng/ai-ml-engineer-roadmap) — full plan in the roadmap's [P3 deep-dive](https://github.com/NanaAkwasiAbayieBoateng/ai-ml-engineer-roadmap/blob/main/projects/p3-rag-application.md).

A retrieval-augmented generation app over fraud/AML documents, with measured retrieval quality. The evaluation report is a first-class deliverable. All LLM work runs locally on [Ollama](https://ollama.com) — free and open source, no API keys: chat model `llama3.1:8b`, embedding model `nomic-embed-text` (both configurable via `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL`).

Ships with three short sample documents (`data/sample_docs/`: APP scams, money mules, card-not-present fraud) and an 11-question eval set (`data/eval_set.json`), so it runs out of the box.

📖 New here? Read [BLOG_POST.md](BLOG_POST.md) — a file-by-file walkthrough of the whole project.

## Run commands

Run these in order from the repo root:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the Ollama models (Ollama must be installed and running)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 4. Build the index from the sample documents
python -m rag_app.ingest --docs data/sample_docs --out data/index.json

# 5. Serve the API
uvicorn rag_app.main:app --port 8000

# 6. Compare chunking strategies on the eval set (writes reports/eval_report.md)
python scripts/run_eval.py

# 7. Run the tests (no Ollama or network needed — they use a fake client)
pytest -q
```

Then ask a question:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How are money mules recruited?"}'
```

**No Ollama?** The whole flow also runs offline with deterministic fake
embeddings — useful for CI and a first look:

```bash
python -m rag_app.ingest --fake
RAG_FAKE_LLM=1 uvicorn rag_app.main:app --port 8000
```

## How it works

- `rag_app/ollama_client.py` — Ollama HTTP client (chat + embed) and a deterministic `FakeOllamaClient` for tests/offline runs.
- `rag_app/chunking.py` — three chunking strategies: `fixed_size` (sliding window with overlap), `recursive` (paragraph/sentence aware), `structure_aware` (one chunk per heading section).
- `rag_app/index.py` — hybrid index: numpy cosine vector store + BM25 (`rank_bm25`), fused ranking, persisted to a local JSON file.
- `rag_app/ingest.py` — loads `.md`/`.txt` documents, chunks, embeds, saves the index.
- `rag_app/evaluate.py` — recall@k and MRR against the eval set.
- `rag_app/main.py` — FastAPI app: `GET /health`, `POST /ask`. Answers only from retrieved passages, returns citations, and refuses when the top hit's cosine similarity is below `RAG_REFUSAL_THRESHOLD` (default 0.25).

Configuration is via environment variables — see `.env.example`
(`OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `RAG_INDEX_PATH`,
`RAG_TOP_K`, `RAG_REFUSAL_THRESHOLD`, `RAG_FAKE_LLM`).

## Plan

1. **Corpus:** a few hundred public fraud/AML guidance documents; define the 5–10 question types the app must answer.
2. **Ingestion:** parse PDFs/HTML to clean text, dedupe, keep metadata (title, source, date) for citations and filtering.
3. **Chunking experiment:** fixed-size with overlap vs recursive vs structure-aware — compared, not guessed. *(Scaffold: all three implemented and compared in `reports/eval_report.md`.)*
4. **Index:** embeddings with metadata alongside each vector. *(Scaffold: local numpy store + BM25 instead of pgvector/Qdrant — the `HybridIndex` class is the swap-in point when the corpus outgrows memory.)*
5. **Evaluation set first:** 30–50 questions with known source passages; recall@k and MRR before any tuning. *(Scaffold: 11 questions; grow toward 30–50 with the real corpus.)*
6. **Retrieval:** vector-only baseline → hybrid search (BM25 + vector) → reranker, re-measuring after each change. *(Scaffold: hybrid done; reranker is next.)*
7. **Generation:** answers only from retrieved passages, with citations and explicit refusal when evidence is missing. *(Done in the scaffold.)*
8. **Serving:** FastAPI + a minimal UI, deployed; index/model versions visible. *(Scaffold: FastAPI + `/health` reporting models and index size; UI and deployment next.)*
9. **Monitoring:** log low-confidence and refused queries; weekly sampling with an LLM judge.

## Done when

A live URL plus a written eval report comparing chunking and retrieval variants with numbers.

## Stack

Python · FastAPI · Ollama (llama3.1:8b + nomic-embed-text) · numpy · rank_bm25 · Docker (planned)
