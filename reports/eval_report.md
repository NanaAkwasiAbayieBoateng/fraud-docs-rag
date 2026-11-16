# Retrieval evaluation — chunking strategy comparison

- Corpus: `sample_docs` (sample fraud/AML guidance documents)
- Eval set: 11 questions, each mapped to its source document
- Retrieval: hybrid (vector cosine + BM25, alpha=0.5), top-5
- Embeddings: FakeOllamaClient (deterministic, offline)

| Chunking strategy | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| fixed_size | 11 | 1.000 | 1.000 | 1.000 | 1.000 |
| recursive | 11 | 1.000 | 1.000 | 1.000 | 1.000 |
| structure_aware | 13 | 1.000 | 1.000 | 1.000 | 1.000 |

A question counts as a hit when a chunk from its source document is
retrieved. Re-run with `python scripts/run_eval.py --real` against a
local Ollama server for numbers with production embeddings.
