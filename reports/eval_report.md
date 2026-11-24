# Retrieval evaluation — chunking x fusion comparison

- Corpus: `sample_docs` (sample fraud/AML guidance documents)
- Eval set: 31 questions (27 mapped to a source document, 4 refusal cases)
- Retrieval: hybrid (vector cosine + BM25, alpha=0.5), top-5
- Embeddings: FakeOllamaClient (deterministic, offline)

| Chunking strategy | Fusion | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal acc |
|---|---|---|---|---|---|---|---|
| fixed_size | linear | 20 | 0.889 | 1.000 | 1.000 | 0.944 | 1.000 |
| fixed_size | rrf | 20 | 0.815 | 0.963 | 1.000 | 0.898 | 1.000 |
| recursive | linear | 20 | 0.889 | 0.963 | 1.000 | 0.933 | 1.000 |
| recursive | rrf | 20 | 0.815 | 0.926 | 1.000 | 0.873 | 1.000 |
| structure_aware | linear | 22 | 0.926 | 1.000 | 1.000 | 0.963 | 1.000 |
| structure_aware | rrf | 22 | 0.926 | 0.963 | 1.000 | 0.952 | 1.000 |

A question counts as a hit when a chunk from its source document is
retrieved. Refusal accuracy is the fraction of off-topic questions
whose top hit falls below the refusal gate's cosine threshold (0.25).
Re-run with `python scripts/run_eval.py --real` against a local
Ollama server for numbers with production embeddings.
