# Fraud Docs RAG — Retrieval over Fraud & AML Guidance

**Status:** Scaffold built — runnable · **Roadmap stage:** Intermediate (Project P3)
Part of the [ML / AI Engineer Roadmap](https://github.com/NanaAkwasiAbayieBoateng/ai-ml-engineer-roadmap) — full plan in the roadmap's [P3 deep-dive](https://github.com/NanaAkwasiAbayieBoateng/ai-ml-engineer-roadmap/blob/main/projects/p3-rag-application.md).

A retrieval-augmented generation app over fraud/AML documents, with measured retrieval quality. The evaluation report is a first-class deliverable. All LLM work runs locally on [Ollama](https://ollama.com) — free and open source, no API keys: chat model `llama3.1:8b`, embedding model `nomic-embed-text` (both configurable via `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL`).

Ships with six short sample documents (`data/sample_docs/`: APP scams, money mules, card-not-present fraud, phishing/smishing, account takeover, crypto & investment scams) and a 31-question eval set (`data/eval_set.json`) — 27 questions mapped to their source document plus 4 off-topic refusal cases — so it runs out of the box.

📖 New here? Read [BLOG_POST.md](BLOG_POST.md) — a file-by-file walkthrough of the whole project.

## Run commands

Run these in order from the repo root:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (runtime only; the Docker image uses this file too)
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
pip install -r requirements-dev.txt
pytest -q
```

Then ask a question:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How are money mules recruited?"}'
```

Retrieval only, no LLM call — for debugging ranking (add
`"source": "money_mules.md"` to either endpoint to restrict it to one
document):

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question": "How are money mules recruited?", "k": 3}'
```

Every `/ask` is appended to a JSONL query log (`logs/queries.jsonl` by
default; set `RAG_QUERY_LOG` to move it, or to an empty value to
disable) with its best score and whether it was refused — the data for
tuning the refusal threshold.

**No Ollama?** The whole flow also runs offline with deterministic fake
embeddings — useful for CI and a first look:

```bash
python -m rag_app.ingest --fake
RAG_FAKE_LLM=1 uvicorn rag_app.main:app --port 8000
```

## Run with Docker

No local Python or Ollama install needed — the compose stack runs both:

```bash
docker compose up --build
```

First run pulls the two Ollama models (~5 GB) via a one-shot
`ollama-pull` service; the app waits for it, then serves at
http://localhost:8000 (`GET /health`, `POST /ask`). On that first start
the app also builds its search index from the sample docs (a fresh
clone has none — `data/index.json` is generated, not committed).
Models persist in a named volume, so later runs start fast.
`docker compose down` stops the stack (add `-v` to also delete the
downloaded models).

App image only, against an Ollama already running on your machine:

```bash
docker build -t fraud-docs-rag .
docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 fraud-docs-rag
```

## How it works

- `rag_app/ollama_client.py` — Ollama HTTP client (chat + embed) and a deterministic `FakeOllamaClient` for tests/offline runs.
- `rag_app/chunking.py` — three chunking strategies: `fixed_size` (sliding window with overlap), `recursive` (paragraph/sentence aware), `structure_aware` (one chunk per heading section).
- `rag_app/index.py` — hybrid index: numpy cosine vector store + BM25 (`rank_bm25`), two fusion strategies (linear score fusion, or Reciprocal Rank Fusion with `fusion="rrf"`), optional per-document filtering, and two persistence backends behind `save_to`/`load_from` — one JSON file, or SQLite when the path ends in `.db`/`.sqlite`.
- `rag_app/ingest.py` — loads `.md`/`.txt` documents, chunks, embeds, saves the index (JSON or SQLite, by output suffix).
- `rag_app/rerank.py` — optional second-stage reranker (`RAG_RERANK=1`): re-scores the top candidates by query-term coverage before they reach the prompt. A cross-encoder slots in behind the same interface.
- `rag_app/evaluate.py` — recall@k, MRR, and refusal accuracy against the eval set.
- `rag_app/main.py` — FastAPI app: `GET /health`, `POST /ask`, `POST /search`. Answers only from retrieved passages, returns citations, refuses when the top hit's cosine similarity is below `RAG_REFUSAL_THRESHOLD` (default 0.25), and logs every question to the query log.

Configuration is via environment variables — see `.env.example`
(`OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `RAG_INDEX_PATH`,
`RAG_TOP_K`, `RAG_REFUSAL_THRESHOLD`, `RAG_FUSION`, `RAG_RERANK`,
`RAG_QUERY_LOG`, `RAG_FAKE_LLM`).

CI (`.github/workflows/ci.yml`) runs the tests, the offline eval, and a
Docker build + container smoke test on every push.

## Plan

1. **Corpus:** a few hundred public fraud/AML guidance documents; define the 5–10 question types the app must answer.
2. **Ingestion:** parse PDFs/HTML to clean text, dedupe, keep metadata (title, source, date) for citations and filtering.
3. **Chunking experiment:** fixed-size with overlap vs recursive vs structure-aware — compared, not guessed. *(Scaffold: all three implemented and compared in `reports/eval_report.md`.)*
4. **Index:** embeddings with metadata alongside each vector. *(Scaffold: local numpy store + BM25 instead of pgvector/Qdrant — the `HybridIndex` class is the swap-in point when the corpus outgrows memory.)*
5. **Evaluation set first:** 30–50 questions with known source passages; recall@k and MRR before any tuning. *(Scaffold: 31 questions including refusal cases; grow with the real corpus.)*
6. **Retrieval:** vector-only baseline → hybrid search (BM25 + vector) → reranker, re-measuring after each change. *(Scaffold: hybrid done with linear and RRF fusion compared in the eval report; a lexical reranker is implemented behind a swap-in interface — a cross-encoder is the production upgrade.)*
7. **Generation:** answers only from retrieved passages, with citations and explicit refusal when evidence is missing. *(Done in the scaffold.)*
8. **Serving:** FastAPI + a minimal UI, deployed; index/model versions visible. *(Scaffold: FastAPI + `/health` reporting models and index size; UI and deployment next.)*
9. **Monitoring:** log low-confidence and refused queries; weekly sampling with an LLM judge. *(Scaffold: the JSONL query log is in; the LLM-judge sampling is next.)*

## Done when

A live URL plus a written eval report comparing chunking and retrieval variants with numbers.

## Stack

Python · FastAPI · Ollama (llama3.1:8b + nomic-embed-text) · numpy · rank_bm25 · Docker · GitHub Actions CI
