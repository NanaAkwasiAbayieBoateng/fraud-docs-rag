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

The shipped corpus is deliberately tiny — seven short sample documents
(authorised push payment scams, money mules, card-not-present fraud,
phishing and smishing, account takeover, crypto and investment scams,
and a chargeback reason-code table) and a 33-question test set — so the
whole pipeline can be read, run, and measured in an afternoon. The point of the scaffold is the *shape* of a
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
| **Reciprocal Rank Fusion** | The alternative fusion strategy | Fuses the vector and BM25 *rankings* instead of their scores, so it is immune to score-scale differences. Implemented alongside the original linear fusion so the eval harness can measure which one actually retrieves better — on this corpus, linear wins (see the results). |
| **SQLite** (stdlib) | The second index-persistence backend | No new dependency: Python's built-in `sqlite3`. One row per chunk, embeddings as blobs — inspectable with ordinary SQL tooling, and the shape a real vector store would take behind the same interface. |
| **pypdf / python-docx / openpyxl** | Reading PDF, Word, and Excel documents | Guidance documents arrive as PDFs, Word files, and spreadsheets at least as often as Markdown. One small, single-purpose library per format, all behind one `extract_text` function — and CSV needs nothing at all, just the standard library. |
| **GitHub Actions** | CI | Every push runs the tests, the offline eval, and a Docker build with a container smoke test — so a broken image or a retrieval regression is caught before it lands. |
| **Docker + Compose** | Packaging and one-command startup | The app already talks to Ollama over HTTP, so the two containerise as separate services: a small image for the API, the stock Ollama image for the models, and Compose to wire them together. Anyone can run the whole system with `docker compose up --build` — no Python or Ollama install to get right first. |

The dependency list is still short: `fastapi`, `uvicorn`, `httpx`,
`pydantic`, `numpy`, `rank_bm25`, plus one parser per document format
— `pypdf`, `python-docx`, `openpyxl` — all pinned in
`requirements.txt`, which is also exactly what the Docker image
installs. `pytest` lives in `requirements-dev.txt`, so the test
runner never ships in the image.

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
   under each chunking strategy, runs the eval questions, and writes the
   comparison table. Change anything, re-run, compare numbers — not
   vibes.
8. **Packaging.** Finally, the whole thing was containerised: a
   Dockerfile for the API, a Compose file that runs it next to
   Ollama, and an entrypoint that builds the index on first start —
   because the index is a generated artifact, a fresh clone doesn't
   have one. The packaging files are walked through after
   `requirements.txt` below.
9. **Hardening, measured.** A second pass hardened the scaffold the
   way the first pass built it — nothing without a measurement or a
   test. CI went in first. The corpus and eval set grew until the
   metrics could actually separate the strategies (at 3 documents
   every score was a perfect 1.000, which measures nothing). Then:
   refusal cases in the eval set, a query log so real refusals can
   be studied, RRF as a second fusion strategy for the harness to
   judge, a retrieval-only `/search` endpoint for debugging, an
   optional reranker, a SQLite persistence backend, and ingestion
   for the formats guidance really arrives in — PDF, Word, Excel,
   CSV — instead of Markdown alone. Each piece is walked through in
   its file's section below.

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

  `search()` takes two more knobs. `sources` restricts the search to
  named documents — the basis of the per-document filter on the API.
  And `fusion` selects the strategy: the `linear` fusion above, or
  `rrf` (Reciprocal Rank Fusion), which ignores score scales entirely
  and fuses the two *rankings* — each chunk earns
  `1 / (60 + rank)` from each list. RRF is the standard answer to the
  normalisation gotcha below; whether it is actually *better* is an
  empirical question, and the eval harness now answers it (on this
  corpus, it isn't — linear fusion's recall@1 is higher; see the
  results section).

  Persistence has two backends behind `save_to` / `load_from`, chosen
  by file suffix. The original is a single JSON file (`save` /
  `load`). The second, `save_sqlite` / `load_sqlite`, writes one row
  per chunk to a SQLite database with the embedding as a float64
  blob — inspectable with SQL, and updatable row by row instead of
  by rewriting one big blob. A served vector database (Qdrant,
  pgvector) remains the production step, behind this same pair of
  methods.

  **Gotcha worth knowing:** step 3's normalisation means the best hit
  *always* has a fused score near the top of the scale, even for a
  nonsense query — there is always a "least bad" chunk. That's fine
  for ranking, but it makes the fused score useless as a confidence
  measure. Remember this when you reach the refusal gate.

### `rag_app/documents.py` — reading real-world formats

Guidance does not arrive as tidy Markdown. It arrives as PDFs from a
regulator, Word documents from a training team, spreadsheets of
reason codes, and CSV exports — so ingestion starts here, with one
`extract_text(path)` function dispatching on file suffix:

- **`.md` / `.txt`** — read as-is.
- **`.pdf`** (pypdf) — text extracted page by page, each page under
  a `## Page N` heading, so the structure-aware chunker keeps page
  context in every chunk. Scanned-image PDFs extract nothing (there
  is no text layer); OCR is deliberately out of scope.
- **`.docx`** (python-docx) — paragraphs in order, then table rows
  rendered as `cell | cell | cell`.
- **`.xlsx`** (openpyxl) — one section per sheet under a
  `## Sheet: name` heading; each data row becomes one line.
- **`.csv`** (standard library) — the same row rendering as Excel:
  the header row names each field, and every row becomes
  `Header: value; Header: value`. A chargeback table row turns into
  "Category: Fraud; Reason code: 10.4; Meaning: Card-not-present
  fraud…" — text an embedding can actually match a question against,
  which a raw comma-joined line does far less well.

Legacy `.doc` / `.xls` files raise a clear "unsupported format"
error — the modern libraries do not read the old binary formats, and
pretending otherwise would fail silently. The sample corpus proves
the CSV path end to end (`chargeback_reason_codes.csv` is a real
corpus document with its own eval questions); the other formats are
covered by tests that generate genuine files — including a committed
two-page PDF fixture — and extract them.

### `rag_app/ingest.py` — building the index

The offline half of the system; you run it whenever the documents
change.

- `load_documents(docs_dir)` reads every supported file in a folder
  (via `documents.extract_text`) into `(source_name, text)` pairs, in
  sorted order. A file that fails to parse is skipped with a warning
  rather than aborting the build — one corrupt PDF in a folder of
  fifty should not cost you the other forty-nine.
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
- `evaluate_refusals()` scores the eval set's refusal cases: for each
  question marked `expect_refusal`, it checks the top hit's raw
  cosine against the same threshold the API's gate uses. Retrieval
  metrics alone can't see a gate that answers everything; this can.

### `rag_app/main.py` — the API, citations, and the refusal gate

The online half. A FastAPI app built by a `create_app()` factory
(taking the client, index, and threshold as parameters — which is how
tests inject fakes), with three endpoints:

- **`GET /health`** — reports the chat and embedding model names, the
  number of chunks loaded, and the list of source documents. Your
  first check that the app loaded the index you think it did.
- **`POST /ask`** — accepts `{"question": ..., "k": ...}`, and returns
  an answer, a `refused` flag, a list of citations (source, chunk
  index, score, and a 200-character excerpt each), and the best
  match's score. An optional `"source"` field restricts retrieval to
  one document.
- **`POST /search`** — the retrieval-only twin: same body, but it
  returns the ranked hits (with all three scores per hit) and never
  calls the chat model. When an answer looks wrong, this is how you
  tell whether retrieval or generation is at fault.

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

**The query log.** Every `/ask` — answered or refused — appends one
JSON line to `logs/queries.jsonl` (`RAG_QUERY_LOG`): timestamp,
question, best score, refusal flag, and the sources cited. It is the
feedback loop the threshold needs: sort the log by score and the
questions sitting just above and below 0.25 are exactly the ones to
read when deciding whether the gate is set right, and the refused
questions are a ready-made list of what the corpus should cover
next. Logging is wrapped so a failed write can never fail a request.

**Optional reranking.** With `RAG_RERANK=1`, `/ask` retrieves a
wider candidate set (up to 20), passes it through the reranker in
`rag_app/rerank.py`, and only then takes the top-k into the prompt.
The refusal gate still reads the first stage's top cosine, before
reranking — the gate's semantics don't change with the ordering
stage behind it.

### `rag_app/rerank.py` — the second stage

Hybrid search is tuned for recall: get the right chunk *into* the
candidate set. Ordering the few chunks that reach the prompt deserves
a second, pickier pass. In production that pass is a cross-encoder —
one model scoring query and chunk jointly. This file ships the
interface plus a dependency-free stand-in, `LexicalReranker`: it
re-scores each candidate by how many distinct query terms the chunk
covers, adds a bonus when the chunk contains the query as an exact
phrase, and blends that with the first-stage score. It is honestly a
heuristic — its job is to make the stage real (wired, tested,
switchable) so a cross-encoder can replace it with one new class.

### `data/sample_docs/` — the corpus

Seven short, original guidance documents — six Markdown files organised
under clear headings (which is what `structure_aware` chunking
exploits), plus one CSV table:

- **`app_scams.md`** — authorised push payment scams: what they are,
  how criminals impersonate bank staff, Confirmation of Payee,
  reimbursement.
- **`money_mules.md`** — money mules: recruitment, red flags of a
  mule job offer, consequences when caught.
- **`card_not_present.md`** — card-not-present fraud: 3-D Secure,
  credential stuffing, chargebacks and who bears their cost.
- **`phishing_smishing.md`** — phishing and smishing: lookalike
  messages and sites, spotting them, what to do after responding.
- **`account_takeover.md`** — account takeover: how attackers get in,
  the warning signs, recovery and prevention.
- **`crypto_investment_scams.md`** — crypto and investment scams:
  fake trading platforms, withdrawal-fee traps, recovery-agent scams.
- **`chargeback_reason_codes.csv`** — a small table of chargeback
  reason codes (category, code, meaning, who bears the cost). The
  corpus's one non-Markdown document, there to prove tabular data
  retrieves properly once rows are rendered as `header: value` text.

In a production version this folder is where a few hundred real
guidance documents would live; the code doesn't care how many there
are.

### `data/eval_set.json` — the test questions

Thirty-three questions in a simple JSON list, each shaped like:

```json
{"question": "How are money mules recruited?", "source": "money_mules.md"}
```

The `source` label is the ground truth every metric is computed
against. Four entries break the pattern on purpose: `"source": null`
with `"expect_refusal": true` — off-topic questions (sourdough,
weather, stock picks, poetry) that a correct system must refuse.
They exist because the first eval set's blind spot was total: at 11
easy questions every configuration scored a perfect 1.000, which
looked like success and measured nothing. The set also includes
near-miss questions whose wording leans toward the wrong document
(a mule question phrased in crypto terms, a credential-stuffing
question that belongs to account takeover, not cards) — those are
the ones that separate the strategies in the results table.

### `scripts/run_eval.py` — the chunking comparison

A standalone script (it adds the repo root to `sys.path` itself, so it
runs from anywhere). For each combination of chunking strategy and
fusion strategy (linear and RRF) it builds a fresh index over the
sample docs, runs the eval questions through `evaluate_index`, scores
the refusal cases with `evaluate_refusals`, and writes the comparison
table to `reports/eval_report.md`. It uses the fake client by
default, so it runs fully offline; pass `--real` to measure with
real Ollama embeddings.

### `reports/eval_report.md` — the results

The generated deliverable of the eval script: corpus and eval-set
description, the embeddings used, and the table (current numbers in
the closing section below). It's committed to the repo on purpose —
the evaluation report is a first-class output of the project, not a
by-product. Re-running the script overwrites it.

### `tests/` — 41 tests, no Ollama required

Every test injects `FakeOllamaClient`, so the suite runs offline in
seconds (`pytest -q` → 41 passed):

- **`test_chunking.py`** (8 tests) — all three strategies return
  non-empty chunks; fixed-size respects its size limit and handles
  short/empty text; recursive keeps sentence boundaries intact;
  structure-aware splits on headings with the `Title > Section`
  prefix, and falls back cleanly without headings; unknown strategy
  names raise `ValueError`.
- **`test_index.py`** (9 tests) — on a three-document fixture corpus,
  the right document ranks first for each topic; results come back
  sorted by score; an empty index returns nothing; a save → load
  round-trip preserves chunks *and* search quality; RRF ranks the
  right document first and rejects unknown fusion names; the source
  filter restricts results (and returns nothing for an unknown
  source); and a SQLite save → load round-trip preserves search
  quality through the suffix-dispatched backend.
- **`test_evaluate.py`** (5 tests) — the metric maths on known
  examples: recall hits and misses, reciprocal rank at ranks 1/2/3,
  aggregation across questions, and a `ValueError` when rankings and
  eval set lengths disagree — plus refusal scoring: refusal items are
  excluded from retrieval metrics and scored by `evaluate_refusals`.
- **`test_api.py`** (10 tests) — end-to-end through FastAPI's
  `TestClient` with a fake client and a two-document index: `/health`
  reports the loaded index; an on-topic question is answered with the
  correct citation and a passing score; the sourdough question is
  refused with the exact refusal message and no citations; an empty
  index refuses everything; `/search` returns ranked hits with no
  answer attached; the source filter restricts both endpoints; the
  query log records one answered and one refused question with their
  scores; a rerank-enabled app still answers correctly; and `/health`
  reports the RRF fusion when the app is configured for it.
- **`test_rerank.py`** (2 tests) — the lexical reranker promotes a
  chunk covering the whole query over a higher-scored chunk covering
  part of it, and the exact-phrase bonus wins at k=1.
- **`test_documents.py`** (7 tests) — one per format, against real
  files: text/Markdown pass through; CSV rows render with their
  headers; a generated .docx yields its paragraphs and table; a
  generated .xlsx renders its data sheet and skips the empty one; a
  committed two-page PDF fixture extracts with page headings; a
  legacy .doc raises the unsupported-format error; and a folder
  containing one corrupt PDF still loads its good document.

### `requirements.txt` and `requirements-dev.txt`

`requirements.txt` is the runtime list, and since the Docker work it
is pinned and runtime-only — nine lines: `fastapi`, `uvicorn` (the
server), `httpx`, `pydantic` (FastAPI's validation layer), `numpy`,
`rank_bm25`, and the three document parsers (`pypdf`, `python-docx`,
`openpyxl`). Pinning matters more once a Dockerfile installs the file:
the image you build next month should be the image you tested.

`pytest` moved to `requirements-dev.txt`, which starts with
`-r requirements.txt` and adds the test runner on top. Local
development installs the dev file; the Docker image installs only the
runtime file, so the shipped image carries no test tooling.

### `Dockerfile`

A `python:3.12-slim` image for the API — and only the API. Ollama is
deliberately *not* in it: the app already reaches the model server
over HTTP via `OLLAMA_HOST`, so the two stay separate, replaceable
pieces (and the multi-gigabyte models never bloat the app image).

The build is layered for caching: `requirements.txt` is copied and
installed first, so that layer is reused unless the dependencies
change; the `rag_app/` code, the `data/` folder, and the entrypoint
script are copied after. The container runs as a non-root user
(`appuser`), exposes port 8000, and carries a `HEALTHCHECK` that
polls `GET /health` — the endpoint already reports the loaded models
and chunk count, so it doubles as proof the index loaded.

### `docker-entrypoint.sh`

The small shell script the image starts with. It exists because of
one awkward fact: `data/index.json` is gitignored (see `.gitignore`
below), so a fresh clone — which is what a Docker build usually
starts from — has no index, and the app would boot empty and refuse
every question. The entrypoint checks: if `data/index.json` is
missing, it builds it from `data/sample_docs` with the ingest module;
either way it then `exec`s uvicorn, so the server runs as the
container's main process and receives its signals properly.

### `docker-compose.yml`

The full stack, three services:

- **`ollama`** — the stock `ollama/ollama` image, with the downloaded
  models persisted in a named volume so they survive restarts.
- **`ollama-pull`** — a one-shot service that pulls `llama3.1:8b` and
  `nomic-embed-text` (~5 GB on first run) and exits.
- **`app`** — built from the Dockerfile, pointed at
  `OLLAMA_HOST=http://ollama:11434`, and held back with
  `depends_on … service_completed_successfully` until the model pull
  finishes — so the entrypoint's first-start ingest has an embedding
  model to talk to.

`docker compose up --build` is the entire setup; the API answers on
http://localhost:8000.

### `.dockerignore`

The build-context filter: `.git`, `__pycache__`, virtualenvs, the
test suite, reports, and the docs stay out of the context, so builds
are fast and the image contains only what it runs. (The sample
documents under `data/` are *not* ignored — the entrypoint needs
them to build the index on first start.)

### `.github/workflows/ci.yml`

The repo's CI, two jobs on every push and pull request. The `test`
job installs the dev requirements, runs the suite, and runs the
offline eval — a retrieval regression that keeps every unit test
green still fails here, in the report step's numbers if not its exit
code. The `docker` job builds the image and smoke-tests the
container: started with `RAG_FAKE_LLM=1`, the entrypoint builds the
index from the sample docs with fake embeddings, and the job asserts
`/health` reports a non-empty index. It is the check this project's
Docker work always needed and a laptop-only workflow can't give:
proof, on a clean machine, that the image builds and boots.

### `.env.example`

The configuration template: copy to `.env` and adjust. Ollama host and
model names, plus the app's knobs — `RAG_INDEX_PATH`, `RAG_TOP_K`
(default 5), `RAG_REFUSAL_THRESHOLD` (default 0.25), `RAG_FUSION`
(`linear` or `rrf`), `RAG_RERANK`, `RAG_QUERY_LOG` — and
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

**To run it**, the shortest path is Docker: `docker compose up
--build` starts Ollama, pulls the models, builds the index, and
serves the API. Without Docker, follow the [README's Run
commands](README.md#run-commands): create a venv, install
`requirements.txt`, pull the two Ollama models, run the ingest
command, start `uvicorn`. No Ollama? The README's offline variant
(`ingest --fake` + `RAG_FAKE_LLM=1`) runs the whole thing on fake
embeddings.

**Current eval results** (from `reports/eval_report.md` — sample
corpus, fake embeddings, hybrid search at top-5, 29 sourced questions
plus 4 refusal cases):

| Chunking | Fusion | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal acc |
|---|---|---|---|---|---|---|---|
| fixed_size | linear | 22 | 0.897 | 1.000 | 1.000 | 0.948 | 1.000 |
| fixed_size | rrf | 22 | 0.828 | 0.966 | 0.966 | 0.891 | 1.000 |
| recursive | linear | 22 | 0.862 | 0.966 | 1.000 | 0.915 | 0.750 |
| recursive | rrf | 22 | 0.793 | 0.966 | 1.000 | 0.876 | 1.000 |
| structure_aware | linear | 24 | 0.931 | 1.000 | 1.000 | 0.966 | 1.000 |
| structure_aware | rrf | 24 | 0.862 | 0.966 | 1.000 | 0.922 | 1.000 |

The first version of this table was three rows of perfect 1.000s —
three documents and 11 easy questions measured nothing. The expanded
set earns its keep: the strategies finally separate
(structure-aware + linear fusion leads), and RRF — the fashionable
default — *loses* to plain linear fusion on this corpus. That is the
harness doing its job: the answer to "which fusion?" is a number, not
a preference, and the number can change when the corpus or the
embeddings do. Refusal accuracy is 1.000 in five of six rows — the
exception is itself a finding: adding the CSV table shifted the
chunk boundaries under `recursive` chunking just enough that one
off-topic question now slips past the gate (0.750). Refusals are a
property of the whole pipeline, not just the threshold. And the
standing caveat applies throughout: fake embeddings blur the gate
(a weather question scores 0.286 against this corpus, uncomfortably
close to the 0.25 line), while real embeddings separate the bands
far more cleanly.
Re-run with `python scripts/run_eval.py --real` for numbers with
production embeddings.

**Limitations and next steps**, in the order they'd bite:

- **Corpus size.** Seven documents prove the shape, nothing more. The
  real corpus is a few hundred guidance documents, with the eval set
  grown alongside — that's when chunking and fusion choices start to
  separate *decisively* in the table, not by a few points.
- **The numpy store.** Exact search over a few hundred chunks is
  instant; over a few hundred thousand it's not. Persistence now has
  a SQLite backend, but search still runs in memory — `HybridIndex`
  remains the deliberate swap-in point for Qdrant or pgvector, same
  interface, real index underneath.
- **No OCR.** A scanned PDF with no text layer extracts nothing and
  is skipped at ingest with a warning. Document-AI/OCR ingestion is
  a real production need for regulator PDFs; it is a deliberate
  non-goal here.
- **Reranking is a heuristic, for now.** The lexical reranker makes
  the second stage real and testable, but a cross-encoder over the
  top 20 is the standard quality step — and the eval harness is
  already there to prove whether it helps.
- **The refusal threshold is corpus-tuned.** 0.25 separates this
  corpus cleanly with real embeddings (0.40+ vs 0.16−); a bigger,
  more varied corpus will narrow that gap. The query log and the
  eval set's refusal cases now exist precisely so the threshold is
  an operational setting that gets *measured*, not a constant that
  gets forgotten.

The scaffold's job was to make every one of those steps a small,
measurable change instead of a rewrite. That part, at least, the
numbers and the tests can vouch for.

