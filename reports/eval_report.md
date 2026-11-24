# Retrieval evaluation — chunking x fusion comparison

- Corpus: `sample_docs` (sample fraud/AML guidance documents)
- Eval set: 33 questions (29 mapped to a source document, 4 refusal cases)
- Retrieval: hybrid (vector cosine + BM25, alpha=0.5), top-5
- Embeddings: FakeOllamaClient (deterministic, offline)

| Chunking strategy | Fusion | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Refusal acc |
|---|---|---|---|---|---|---|---|
| fixed_size | linear | 22 | 0.897 | 1.000 | 1.000 | 0.948 | 1.000 |
| fixed_size | rrf | 22 | 0.828 | 0.966 | 0.966 | 0.891 | 1.000 |
| recursive | linear | 22 | 0.862 | 0.966 | 1.000 | 0.915 | 0.750 |
| recursive | rrf | 22 | 0.793 | 0.966 | 1.000 | 0.876 | 1.000 |
| structure_aware | linear | 24 | 0.931 | 1.000 | 1.000 | 0.966 | 1.000 |
| structure_aware | rrf | 24 | 0.862 | 0.966 | 1.000 | 0.922 | 1.000 |

A question counts as a hit when a chunk from its source document is
retrieved. Refusal accuracy is the fraction of off-topic questions
whose top hit falls below the refusal gate's cosine threshold (0.25).
Re-run with `python scripts/run_eval.py --real` against a local
Ollama server for numbers with production embeddings.
