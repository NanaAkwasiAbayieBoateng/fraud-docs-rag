# How Fraud Docs RAG Works: A File-by-File Walkthrough

This project is a small **retrieval-augmented generation (RAG)** app. You
give it a folder of fraud and AML guidance documents, and it answers
questions about them — in plain language, with citations pointing back to
the exact document each answer came from.

Two design rules run through everything:

1. **Answers come only from the documents.** The language model is never
   allowed to answer from its own memory. If the documents don't cover a
   question, the app says so instead of guessing.
2. **Off-topic questions get refused.** Ask it how to bake sourdough and
   it will politely decline, rather than inventing an answer with no
   evidence behind it.

Everything runs locally and free: the language model and the embeddings
come from [Ollama](https://ollama.com) running on your own machine. No API
keys, no cloud bill, no data leaving the laptop.

The shipped corpus is deliberately tiny — three short sample documents
(authorised push payment scams, money mules, card-not-present fraud) and
an 11-question test set — so the whole pipeline can be read, run, and
measured in an afternoon. The point of the scaffold is the *shape* of a
real RAG system: ingest → chunk → embed → retrieve → answer → evaluate.

---

## The tools, and why each one

| Tool | What it does here | Why this one |
|---|---|---|
| **Ollama** | Runs the LLM and the embedding model locally | Free, offline, no API keys. The fraud-guidance use case is exactly the kind of thing you don't want to send to a third-party API while prototyping. |
| **`llama3.1:8b`** | The chat model that writes answers | A capable small model that runs on a laptop. Big enough to summarise passages faithfully; the app constrains it with a strict system prompt anyway. |
| **`nomic-embed-text`** | Turns text into vectors for search | A well-regarded open embedding model served by Ollama. Embeddings are what make "find passages about the same *idea*" possible, beyond keyword matching. |
| **FastAPI** | The HTTP API (`/health`, `/ask`) | Small, typed (Pydantic request/response models), and self-documenting. The API surface here is two endpoints — anything heavier would be ceremony. |
| **numpy** | The vector store: one matrix of embeddings, cosine similarity by matrix multiplication | At hundreds of documents, a dedicated vector database (Qdrant, pgvector…) is infrastructure without payoff. A numpy matrix is exact, fast enough, and fits in a JSON file. The swap-in point is clearly marked (see `HybridIndex` below). |
| **rank_bm25** | Keyword search (BM25) | Fraud guidance is full of exact terms — "Confirmation of Payee", "3-D Secure", "chargeback" — where keyword match beats fuzzy semantic match. BM25 is the classic algorithm for that. |
| **Hybrid search** | Fuses the vector ranking and the BM25 ranking | Each method has blind spots: vectors miss exact terminology, keywords miss paraphrases. Averaging the two normalised rankings is simple and measurably better than either alone. |
| **httpx** | HTTP calls to Ollama | Modern, small, and also used by FastAPI's test client — one less dependency to learn. |
| **pytest + a fake Ollama client** | The test suite | Tests must run anywhere, including CI with no Ollama installed. So the codebase has a `FakeOllamaClient` with the same interface as the real one — tests inject it and never touch the network. |

The dependency list really is that short: `fastapi`, `uvicorn`, `httpx`,
`pydantic`, `numpy`, `rank_bm25`, `pytest`.

---

## How it was built, step by step

The build order follows the data. Each stage only depends on the ones
before it, which is also the order to read the code in.

1. **Sample documents first.** Three short guidance documents were
   written (APP scams, money mules, card-not-present fraud), each with
   clear headings. You can't evaluate retrieval without a corpus, and
   you can't judge a corpus without knowing what it says.
2. **An eval set before any tuning.** Eleven questions, each labelled
   with the document that answers it (`data/eval_set.json`). Writing the
   test questions *before* building search keeps you honest: the numbers
   in `reports/eval_report.md` measure something decided in advance.
3. **Chunking.** Documents are too long to embed whole, so three
   chunking strategies were implemented side by side — fixed-size
   windows, recursive splitting, and heading-aware splitting — behind
   one identical interface so they can be swapped and compared.
4. **Embeddings and the index.** Each chunk is embedded once at ingest
   time and stored with its source document name. The index combines
   the vector store and a BM25 index over the same chunks.
5. **Retrieval.** Hybrid search: cosine similarity for meaning, BM25
   for exact terms, fused into one ranking. Every result carries its
   source, which is what makes citations possible.
6. **The API and the refusal gate.** A FastAPI app that embeds the
   question, retrieves passages, and either answers from them with
   citations — or refuses when the best match is too weak. The gate
   turned out to be the subtlest part of the build; it's explained in
   detail under `rag_app/main.py` below.
7. **Evaluation last, and repeatable.** A script rebuilds the index
   under each chunking strategy, runs all 11 questions, and writes the
   comparison table. Change anything, re-run, compare numbers — not
   vibes.

---

## Every file, explained

Read the repo in this order and each file will make sense in terms of
the ones before it.

### `rag_app/__init__.py`

The package marker. It contains no code — just a docstring stating the
project's ground rule (all LLM work runs on local Ollama) and an
`__all__` list naming the five modules. Two minutes in this file tells
you the architecture: a client, chunking, an index, ingest, evaluate.

### `rag_app/ollama_client.py` — talking to the models

Everything that touches Ollama goes through this file, so the rest of
the code never sees an HTTP request.

- **`OllamaClient`** wraps Ollama's HTTP API with two methods:
  `embed(texts)` (batch embedding via `/api/embed`) and
  `chat(prompt, system=...)` (via `/api/chat`). Host and model names
  come from environment variables — `OLLAMA_HOST`,
  `OLLAMA_MODEL` (default `llama3.1:8b`), `OLLAMA_EMBED_MODEL`
  (default `nomic-embed-text`) — so switching models never means
  editing code.
- **`tokenize(text)`** lowercases, extracts word tokens, and drops
  stopwords ("the", "how", "what"…). It's shared: BM25 uses it to index
  chunks, and the fake client uses it to build embeddings.
- **`FakeOllamaClient`** is the most important testing idea in the repo.
  It has the *same interface* as the real client but needs no server:
  its embeddings are built by hashing each content word into a
  fixed 128-dimension vector (the "hashing trick"), so texts that share
  words get high cosine similarity — enough for retrieval tests to be
  meaningful. Its `chat` echoes back the first passage it was given,
  which lets API tests verify that retrieved context really reached
  the answer. Deterministic, offline, instant.
- **`make_client(fake=False)`** is the factory the rest of the app
  calls; it returns the fake when asked, or when `RAG_FAKE_LLM=1` is
  set in the environment.

### `rag_app/chunking.py` — cutting documents into searchable pieces

Embeddings work best on short, focused passages, so documents are cut
into chunks of roughly 800 characters. This file implements three
strategies, all with the same signature — text in, `list[str]` out —
registered in a `STRATEGIES` dict and dispatched by `chunk_text()`:

- **`fixed_size`** — a sliding character window with 100 characters of
  overlap between consecutive chunks, preferring to end a window at a
  whitespace boundary rather than mid-word. Simple and predictable;
  the overlap means a sentence split across a boundary still appears
  whole in one of the chunks.
- **`recursive`** — splits on natural boundaries first (paragraphs,
  then lines, then sentences, then spaces), then packs the pieces up
  to the chunk size, carrying a tail of the previous chunk forward as
  overlap. Chunks respect the document's own structure where possible.
- **`structure_aware`** — one chunk per Markdown heading section, with
  the heading path prepended (e.g. `Money Mules > How mules are
  recruited`). The heading gives every chunk free context. Sections
  longer than the chunk size fall back to `fixed_size` windows, still
  prefixed with their heading; a document with no headings degrades
  gracefully to a plain fixed-size pass.

Why three? Because "which chunking is best?" is an empirical question,
and the eval script (below) answers it with numbers instead of opinion.

### `rag_app/index.py` — the hybrid search index

The heart of retrieval. Two dataclasses and one class:

- **`Chunk`** — a piece of text plus its `source` (the document name),
  its `chunk_index` (position within that document), and a metadata
  dict. Source + index are what citations are made of.
- **`SearchResult`** — a chunk plus three scores: the fused `score`,
  and the raw `vector_score` and `bm25_score` it came from. Keeping
  all three turns out to matter (see the refusal gate, below).
- **`HybridIndex`** — holds the chunks, a numpy matrix of their
  embeddings, and a `BM25Okapi` index built over tokenised chunk text.
  Its `search(query, query_embedding, k, alpha)` method:
  1. computes cosine similarity between the query vector and every
     chunk vector (one matrix multiplication),
  2. computes BM25 scores for the query's tokens,
  3. **normalises each score list to [0, 1] by dividing by its own
     maximum**, then fuses them: `alpha × vector + (1 − alpha) × BM25`
     (default `alpha = 0.5`, an even split),
  4. returns the top-k results in fused order.

  The whole index serialises to a single JSON file (`save` / `load`),
  which is all the persistence this scale needs.

  **Gotcha worth knowing:** step 3's normalisation means the best hit
  *always* has a fused score near the top of the scale, even for a
  nonsense query — there is always a "least bad" chunk. That's fine
  for ranking, but it makes the fused score useless as a confidence
  measure. Remember this when you reach the refusal gate.

### `rag_app/ingest.py` — building the index

The offline half of the system; you run it whenever the documents
change.

- `load_documents(docs_dir)` reads every `.md` / `.txt` file in a
  folder into `(source_name, text)` pairs, in sorted order.
- `build_index(docs_dir, client, strategy, ...)` chunks each document,
  wraps the pieces in `Chunk` objects (recording which strategy made
  them in the metadata), embeds all chunk texts in one batch call, and
  returns a ready `HybridIndex`.
- Run as a module — `python -m rag_app.ingest --docs data/sample_docs
  --out data/index.json` — with flags for strategy, chunk size,
  overlap, and `--fake` for offline fake embeddings.

Its output, `data/index.json`, is a generated artifact (gitignored):
delete it and rebuild any time.

### `rag_app/evaluate.py` — measuring retrieval quality

Small file, two standard metrics:

- **`recall_at_k`** — did a chunk from the right document appear in
  the top k? With one correct document per question, this is a hit
  rate: 1.0 or 0.0 per question, averaged over the set.
- **`reciprocal_rank`** — 1 ÷ the rank of the first correct result.
  Rank 1 → 1.0, rank 2 → 0.5, absent → 0. Averaged over questions
  this is the **MRR**.
- `evaluate_rankings()` aggregates both over precomputed rankings;
  `evaluate_index()` is the convenience wrapper that runs every eval
  question through a live index first.

### `rag_app/main.py` — the API, citations, and the refusal gate

The online half. A FastAPI app built by a `create_app()` factory
(taking the client, index, and threshold as parameters — which is how
tests inject fakes), with two endpoints:

- **`GET /health`** — reports the chat and embedding model names, the
  number of chunks loaded, and the list of source documents. Your
  first check that the app loaded the index you think it did.
- **`POST /ask`** — accepts `{"question": ..., "k": ...}`, and returns
  an answer, a `refused` flag, a list of citations (source, chunk
  index, score, and a 200-character excerpt each), and the best
  match's score.

The flow inside `/ask`: embed the question → hybrid search → **refusal
gate** → build a prompt from the numbered passages → chat → respond.
The system prompt instructs the model to answer *only* from the
numbered passages and to cite them inline like `[1]`, `[2]`.

**The refusal gate — the one real bug this project taught.**
The first version gated on the *fused* hybrid score: refuse if the
best result's fused score is below a threshold. It didn't work —
off-topic questions scored **0.87** and were cheerfully answered.
Why: fusion normalises each score list by its own maximum, so the
best hit is pushed to the top of the scale *by construction*, whatever
the question. The fused score ranks results against each other; it
says nothing about whether any of them is actually relevant.

The fix: gate on the top hit's **raw cosine similarity**
(`results[0].vector_score`, which `SearchResult` had been carrying all
along). Cosine is an absolute measure — it compares the question to
the passage, not to other passages. Measured on the sample corpus,
the separation is wide and clean:

- Topical questions: cosine **0.40 – 0.46**
- Off-topic questions: cosine **0.06 – 0.16**

So the default threshold, `RAG_REFUSAL_THRESHOLD = 0.25`, sits
comfortably between the two bands. Below it, `/ask` returns the
refusal message with `refused: true` and no citations — and never
calls the chat model at all.

### `data/sample_docs/` — the corpus

Three short, original guidance documents, each organised under clear
Markdown headings (which is what `structure_aware` chunking exploits):

- **`app_scams.md`** — authorised push payment scams: what they are,
  how criminals impersonate bank staff, Confirmation of Payee,
  reimbursement.
- **`money_mules.md`** — money mules: recruitment, red flags of a
  mule job offer, consequences when caught.
- **`card_not_present.md`** — card-not-present fraud: 3-D Secure,
  credential stuffing, chargebacks and who bears their cost.

In a production version this folder is where a few hundred real
guidance documents would live; the code doesn't care how many there
are.

### `data/eval_set.json` — the test questions

Eleven questions in a simple JSON list, each shaped like:

```json
{"question": "How are money mules recruited?", "source": "money_mules.md"}
```

The `source` label is the ground truth every metric is computed
against. Small, but written *before* tuning — that's what makes the
eval report meaningful rather than decorative.

### `scripts/run_eval.py` — the chunking comparison

A standalone script (it adds the repo root to `sys.path` itself, so it
runs from anywhere). For each of the three chunking strategies it
builds a fresh index over the sample docs, runs all 11 eval questions
through `evaluate_index`, and writes the comparison table to
`reports/eval_report.md`. It uses the fake client by default, so it
runs fully offline; pass `--real` to measure with real Ollama
embeddings.

### `reports/eval_report.md` — the results

The generated deliverable of the eval script: corpus and eval-set
description, the embeddings used, and the table (current numbers in
the closing section below). It's committed to the repo on purpose —
the evaluation report is a first-class output of the project, not a
by-product. Re-running the script overwrites it.

### `tests/` — 21 tests, no Ollama required

Every test injects `FakeOllamaClient`, so the suite runs offline in
seconds (`pytest -q` → 21 passed):

- **`test_chunking.py`** (8 tests) — all three strategies return
  non-empty chunks; fixed-size respects its size limit and handles
  short/empty text; recursive keeps sentence boundaries intact;
  structure-aware splits on headings with the `Title > Section`
  prefix, and falls back cleanly without headings; unknown strategy
  names raise `ValueError`.
- **`test_index.py`** (5 tests) — on a three-document fixture corpus,
  the right document ranks first for each topic; results come back
  sorted by score; an empty index returns nothing; and a
  save → load round-trip preserves chunks *and* search quality.
- **`test_evaluate.py`** (4 tests) — the metric maths on known
  examples: recall hits and misses, reciprocal rank at ranks 1/2/3,
  aggregation across questions, and a `ValueError` when rankings and
  eval set lengths disagree.
- **`test_api.py`** (4 tests) — end-to-end through FastAPI's
  `TestClient` with a fake client and a two-document index: `/health`
  reports the loaded index; an on-topic question is answered with the
  correct citation and a passing score; the sourdough question is
  refused with the exact refusal message and no citations; an empty
  index refuses everything.

### `requirements.txt`

Seven lines: `fastapi`, `uvicorn` (the server), `httpx`, `pydantic`
(FastAPI's validation layer), `numpy`, `rank_bm25`, `pytest`.
Unpinned — fine for a scaffold; pin before anything production-shaped.

### `.env.example`

The configuration template: copy to `.env` and adjust. Ollama host and
model names, plus the app's three knobs — `RAG_INDEX_PATH`,
`RAG_TOP_K` (default 5), `RAG_REFUSAL_THRESHOLD` (default 0.25) — and
`RAG_FAKE_LLM` to switch the whole app to fake embeddings. Note the app
reads these from the environment (via `os.getenv` in `main.py` and
`ollama_client.py`); `.env` is gitignored so local settings never get
committed.

### `conftest.py`

Five lines at the repo root that exist because of a real papercut:
the bare `pytest` command doesn't put the working directory on
`sys.path` the way `python -m pytest` does, so `import rag_app` can
fail depending on how tests are invoked. A root-level `conftest.py`
makes pytest add the repo root to the path — imports just work either
way.

### `README.md`

The front door: what the project is, the **Run commands** section
(the exact copy-paste sequence to get from clone to a running API,
including the no-Ollama offline variant), a short how-it-works map,
and the roadmap from scaffold to production — bigger corpus, metadata
filtering, a reranker, monitoring of refused queries.

### `.gitignore`

Standard Python ignores (`__pycache__/`, `.venv/`, `.pytest_cache/`,
`.env`) plus one project-specific entry: **`data/index.json`** — the
generated index is a build artifact, rebuilt by the ingest command,
so it isn't committed. The sample documents and eval set *are*
committed; they're source, not output.

---

## A question's journey

Follow one request — `POST /ask` with *"How are money mules
recruited?"* — through the files:

1. **`rag_app/main.py`** receives it. The `AskRequest` model validates
   the body; `k` defaults to `RAG_TOP_K` (5).
2. The app asks the client — **`rag_app/ollama_client.py`** — to embed
   the question: one HTTP call to Ollama's `/api/embed` with
   `nomic-embed-text`, returning a single vector. (With the fake
   client, the same step is the hashing trick, in-process.)
3. **`rag_app/index.py`** takes that vector and the raw question text.
   Cosine similarity against every chunk vector, BM25 against every
   chunk's tokens, normalise, fuse, sort. The top hit is a chunk from
   `money_mules.md` — the recruitment section.
4. Back in **`main.py`**, the gate checks that top hit's *raw cosine*
   — around 0.4 for this question, comfortably above the 0.25
   threshold. (Ask about sourdough instead and the journey ends here:
   cosine ~0.1, refusal returned, chat model never called.)
5. `main.py` formats the top-5 passages as a numbered list —
   `[1] (money_mules.md) …` — and sends them with the question and the
   strict system prompt to the client's `chat`, i.e. `llama3.1:8b`.
6. The model answers from the passages, citing `[1]`, `[2]`… inline.
   `main.py` packages the answer with the citation list — source,
   chunk index, fused score, excerpt for each passage — and returns
   it as an `AskResponse`.

Offline, the identical journey runs in the tests: `test_api.py` drives
steps 1–6 through the `TestClient` with the fake client, whose `chat`
echoes passage `[1]` back — proof the retrieved context made it into
the answer.

---

## Running it, results, and what's next

**To run it**, follow the [README's Run commands](README.md#run-commands):
create a venv, install `requirements.txt`, pull the two Ollama models,
run the ingest command, start `uvicorn`. No Ollama? The README's
offline variant (`ingest --fake` + `RAG_FAKE_LLM=1`) runs the whole
thing on fake embeddings.

**Current eval results** (from `reports/eval_report.md` — sample
corpus, fake embeddings, hybrid search at top-5):

| Chunking strategy | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| fixed_size | 11 | 1.000 | 1.000 | 1.000 | 1.000 |
| recursive | 11 | 1.000 | 1.000 | 1.000 | 1.000 |
| structure_aware | 13 | 1.000 | 1.000 | 1.000 | 1.000 |

Perfect scores — and you should read them with suspicion. Three short
documents and 11 questions is a smoke test, not a benchmark: the
topics barely overlap, so retrieval has nothing hard to do. The honest
conclusion is "the pipeline works end to end and the harness is ready
to measure something harder", not "chunking doesn't matter". Re-run
with `python scripts/run_eval.py --real` for numbers with production
embeddings.

**Limitations and next steps**, in the order they'd bite:

- **Corpus size.** Three documents prove the shape, nothing more. The
  real corpus is a few hundred guidance documents, with the eval set
  grown to 30–50 questions — that's when chunking and fusion choices
  start to separate in the table.
- **The numpy store.** Exact search over a few hundred chunks is
  instant; over a few hundred thousand it's not. `HybridIndex` is the
  deliberate swap-in point for Qdrant or pgvector — same interface,
  real index underneath.
- **No reranking.** Hybrid search returns a good top-5, but a
  cross-encoder reranker over the top 20 is the standard next quality
  step — and the eval harness is already there to prove whether it
  helps.
- **The refusal threshold is corpus-tuned.** 0.25 separates this
  corpus cleanly (0.40+ vs 0.16−); a bigger, more varied corpus will
  narrow that gap, and the threshold — plus logging of refused and
  low-confidence queries — becomes an operational setting to monitor,
  not a constant to forget.

The scaffold's job was to make every one of those steps a small,
measurable change instead of a rewrite. That part, at least, the
numbers and the tests can vouch for.

---

## Added later: Docker (2026-11-22)

Three files package the project so it runs anywhere with one command:

- **`Dockerfile`** — a `python:3.12-slim` image for the API only.
  Dependencies install in their own layer (cached unless
  `requirements.txt` changes), then the `rag_app/` code and the
  prebuilt `data/` index are copied in. The app runs as a non-root
  user, and a `HEALTHCHECK` polls `GET /health`. Ollama is
  deliberately *not* in this image — the app already talks to it over
  HTTP via `OLLAMA_HOST`, so the model server stays a separate,
  replaceable piece.
- **`docker-compose.yml`** — the full stack: an `ollama` service
  (models persisted in a named volume), a one-shot `ollama-pull`
  service that downloads `llama3.1:8b` and `nomic-embed-text`, and the
  `app` service, which waits for the pull to finish before starting.
  `docker compose up --build` is the whole setup.
- **`.dockerignore`** — keeps `.git`, caches, tests and docs out of
  the build context, so the image stays small and rebuilds stay fast.
