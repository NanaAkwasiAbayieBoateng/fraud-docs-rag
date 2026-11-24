"""Retrieval evaluation: recall@k, mean reciprocal rank (MRR), refusals.

An eval item is ``{"question": ..., "source": ...}`` where ``source`` is
the name of the document that answers the question. A retrieved chunk is
relevant when its ``source`` matches.

Items with ``"expect_refusal": true`` (and no source) measure the other
half of quality: questions the corpus cannot answer should be refused.
``evaluate_refusals`` scores them against the same raw-cosine gate the
API uses.
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


def evaluate_index(index, client, eval_set: list[dict], k: int = 5,
                   fusion: str = "linear") -> dict[str, float]:
    """Run every sourced eval question through an index, aggregate metrics.

    Refusal items (no source) are excluded here; score them with
    ``evaluate_refusals``.
    """
    sourced = [item for item in eval_set if item.get("source")]
    rankings = []
    for item in sourced:
        embedding = client.embed_one(item["question"])
        results = index.search(item["question"], embedding, k=k, fusion=fusion)
        rankings.append([r.chunk.source for r in results])
    return evaluate_rankings(rankings, sourced)


def evaluate_refusals(index, client, eval_set: list[dict], threshold: float = 0.25,
                      fusion: str = "linear") -> float | None:
    """Fraction of refusal-expected questions the gate would refuse.

    Mirrors the API: a question is refused when there are no results or
    the top hit's raw cosine similarity is below ``threshold``. Returns
    None when the eval set has no refusal items.
    """
    items = [item for item in eval_set if item.get("expect_refusal")]
    if not items:
        return None
    correct = 0
    for item in items:
        embedding = client.embed_one(item["question"])
        results = index.search(item["question"], embedding, k=1, fusion=fusion)
        best = results[0].vector_score if results else 0.0
        if best < threshold:
            correct += 1
    return correct / len(items)
