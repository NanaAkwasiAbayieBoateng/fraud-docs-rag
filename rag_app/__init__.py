"""Fraud Docs RAG — retrieval over fraud/AML guidance documents.

LLM work runs entirely on a local Ollama instance (free, open source):
chat model ``llama3.1:8b`` and embedding model ``nomic-embed-text`` by
default, both configurable through environment variables.
"""

__all__ = ["ollama_client", "chunking", "index", "ingest", "evaluate"]
