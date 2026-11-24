"""Ingestion: load .md/.txt documents, chunk them, embed, persist the index.

Usage:
    python -m rag_app.ingest --docs data/sample_docs --out data/index.json
    python -m rag_app.ingest --fake          # offline, deterministic embeddings
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_app.chunking import chunk_text
from rag_app.index import Chunk, HybridIndex
from rag_app.ollama_client import make_client

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(docs_dir: str | Path) -> list[tuple[str, str]]:
    """Return (source_name, text) pairs for every supported doc in a folder."""
    docs_dir = Path(docs_dir)
    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.append((path.name, path.read_text(encoding="utf-8")))
    return documents


def build_index(
    docs_dir: str | Path,
    client,
    strategy: str = "fixed_size",
    chunk_size: int = 800,
    overlap: int = 100,
) -> HybridIndex:
    chunks: list[Chunk] = []
    for source, text in load_documents(docs_dir):
        pieces = chunk_text(text, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(text=piece, source=source, chunk_index=i,
                      metadata={"strategy": strategy})
            )
    embeddings = client.embed([c.text for c in chunks]) if chunks else []
    return HybridIndex(chunks=chunks, embeddings=embeddings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the RAG index from documents.")
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--out", default="data/index.json",
                        help="Output path; a .db/.sqlite suffix selects the SQLite backend.")
    parser.add_argument("--strategy", default="fixed_size",
                        choices=["fixed_size", "recursive", "structure_aware"])
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--fake", action="store_true",
                        help="Use deterministic fake embeddings (no Ollama needed).")
    args = parser.parse_args(argv)

    client = make_client(fake=args.fake)
    index = build_index(
        args.docs, client, strategy=args.strategy,
        chunk_size=args.chunk_size, overlap=args.overlap,
    )
    index.save_to(args.out)
    sources = sorted({c.source for c in index.chunks})
    print(f"Indexed {len(index)} chunks from {len(sources)} documents -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
