# Fraud Docs RAG — Retrieval over Fraud & AML Guidance

**Status:** Planned · **Roadmap stage:** Intermediate (Project P3)
Part of the [ML / AI Engineer Roadmap](https://github.com/NanaAkwasiAbayieBoateng/ai-ml-engineer-roadmap) — full plan in the roadmap's [P3 deep-dive](https://github.com/NanaAkwasiAbayieBoateng/ai-ml-engineer-roadmap/blob/main/projects/p3-rag-application.md).

A deployed retrieval-augmented generation app over real fraud/AML documents, with measured retrieval quality. The evaluation report is a first-class deliverable.

## Plan

1. **Corpus:** a few hundred public fraud/AML guidance documents; define the 5–10 question types the app must answer.
2. **Ingestion:** parse PDFs/HTML to clean text, dedupe, keep metadata (title, source, date) for citations and filtering.
3. **Chunking experiment:** fixed-size with overlap vs recursive vs structure-aware — compared, not guessed.
4. **Index:** embeddings in pgvector or Qdrant with metadata alongside each vector.
5. **Evaluation set first:** 30–50 questions with known source passages; recall@k and MRR before any tuning.
6. **Retrieval:** vector-only baseline → hybrid search (BM25 + vector) → reranker, re-measuring after each change.
7. **Generation:** answers only from retrieved passages, with citations and explicit refusal when evidence is missing.
8. **Serving:** FastAPI + minimal UI, deployed; index/model versions visible.
9. **Monitoring:** log low-confidence and refused queries; weekly sampling with an LLM judge.

## Done when

A live URL plus a written eval report comparing chunking and retrieval variants with numbers.

## Stack

Python · FastAPI · pgvector/Qdrant · Docker
