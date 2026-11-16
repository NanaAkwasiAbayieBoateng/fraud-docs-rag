"""Compare the three chunking strategies on the eval set.

Builds one index per chunking strategy over data/sample_docs, runs every
question in data/eval_set.json through hybrid search, and writes a
markdown comparison table (recall@1/3/5 and MRR) to reports/eval_report.md.

By default it uses the deterministic FakeOllamaClient, so it runs fully
offline with no Ollama server. Pass --real (and have Ollama running with
the embedding model pulled) to measure with real embeddings.

Usage:
    python scripts/run_eval.py [--real] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_app.chunking import STRATEGIES  # noqa: E402
from rag_app.evaluate import evaluate_index  # noqa: E402
from rag_app.ingest import build_index  # noqa: E402
from rag_app.ollama_client import FakeOllamaClient, OllamaClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunking-strategy eval report.")
    parser.add_argument("--docs", default=str(ROOT / "data" / "sample_docs"))
    parser.add_argument("--eval-set", default=str(ROOT / "data" / "eval_set.json"))
    parser.add_argument("--report", default=str(ROOT / "reports" / "eval_report.md"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--real", action="store_true",
                        help="Use a live Ollama server instead of fake embeddings.")
    args = parser.parse_args(argv)

    client = OllamaClient() if args.real else FakeOllamaClient()
    embedder = f"Ollama ({client.embed_model})" if args.real else "FakeOllamaClient (deterministic, offline)"
    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))

    rows = []
    for strategy in STRATEGIES:
        index = build_index(args.docs, client, strategy=strategy)
        metrics = evaluate_index(index, client, eval_set, k=args.k)
        rows.append((strategy, len(index), metrics))
        print(f"{strategy:16s} chunks={len(index):3d}  " +
              "  ".join(f"{name}={value:.3f}" for name, value in metrics.items()))

    lines = [
        "# Retrieval evaluation — chunking strategy comparison",
        "",
        f"- Corpus: `{Path(args.docs).name}` (sample fraud/AML guidance documents)",
        f"- Eval set: {len(eval_set)} questions, each mapped to its source document",
        f"- Retrieval: hybrid (vector cosine + BM25, alpha=0.5), top-{args.k}",
        f"- Embeddings: {embedder}",
        "",
        "| Chunking strategy | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for strategy, n_chunks, m in rows:
        lines.append(
            f"| {strategy} | {n_chunks} | {m['recall@1']:.3f} | {m['recall@3']:.3f} "
            f"| {m['recall@5']:.3f} | {m['mrr']:.3f} |"
        )
    lines += [
        "",
        "A question counts as a hit when a chunk from its source document is",
        "retrieved. Re-run with `python scripts/run_eval.py --real` against a",
        "local Ollama server for numbers with production embeddings.",
        "",
    ]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
