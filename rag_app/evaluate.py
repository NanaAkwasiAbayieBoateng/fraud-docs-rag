"""Retrieval evaluation: recall@k and mean reciprocal rank (MRR).

An eval item is ``{"question": ..., "source": ...}`` where ``source`` is
the name of the document that answers the question. A retrieved chunk is
relevant when its ``source`` matches.
"""

from __future__ import annotations


def recall_at_k(retrieved_sources: list[str], relevant_source: str, k: int) -> float:
    """1.0 when the relevant source appears in the top-k, else 0.0.

    With exactly one relevant document per question, recall@k reduces to
    a hit indicator; averaged over the eval set it is the hit rate.
    """
    return 1.0 if relevant_source in retrieved_sources[:k] else 0.0


def reciprocal_rank(retrieved_sources: list[str], relevant_source: str) -> float:
    """1/rank of the first relevant result, or 0 when absent."""
    for i, source in enumerate(retrieved_sources):
        if source == relevant_source:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_rankings(
    rankings: list[list[str]], eval_set: list[dict], ks: tuple[int, ...] = (1, 3, 5)
) -> dict[str, float]:
    """Aggregate recall@k and MRR over precomputed source rankings."""
    if not eval_set:
        raise ValueError("eval_set is empty")
    if len(rankings) != len(eval_set):
        raise ValueError("rankings and eval_set must have the same length")
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = sum(
            recall_at_k(r, item["source"], k) for r, item in zip(rankings, eval_set)
        ) / len(eval_set)
    metrics["mrr"] = sum(
        reciprocal_rank(r, item["source"]) for r, item in zip(rankings, eval_set)
    ) / len(eval_set)
    return metrics


def evaluate_index(index, client, eval_set: list[dict], k: int = 5) -> dict[str, float]:
    """Run every eval question through an index and aggregate the metrics."""
    rankings = []
    for item in eval_set:
        embedding = client.embed_one(item["question"])
        results = index.search(item["question"], embedding, k=k)
        rankings.append([r.chunk.source for r in results])
    return evaluate_rankings(rankings, eval_set)
