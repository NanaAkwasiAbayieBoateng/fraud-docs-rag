import pytest

from rag_app.evaluate import evaluate_rankings, recall_at_k, reciprocal_rank


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a.md", "b.md"], "a.md", 1) == 1.0
    assert recall_at_k(["a.md", "b.md"], "b.md", 1) == 0.0
    assert recall_at_k(["a.md", "b.md"], "b.md", 2) == 1.0
    assert recall_at_k(["a.md"], "zzz.md", 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a.md", "b.md"], "a.md") == 1.0
    assert reciprocal_rank(["a.md", "b.md"], "b.md") == pytest.approx(0.5)
    assert reciprocal_rank(["a.md", "b.md", "c.md"], "c.md") == pytest.approx(1 / 3)
    assert reciprocal_rank(["a.md"], "zzz.md") == 0.0


def test_evaluate_rankings_aggregates():
    eval_set = [
        {"question": "q1", "source": "a.md"},
        {"question": "q2", "source": "b.md"},
    ]
    rankings = [["a.md", "b.md"], ["a.md", "b.md"]]  # q1 rank 1, q2 rank 2
    metrics = evaluate_rankings(rankings, eval_set, ks=(1, 2))
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["recall@2"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_evaluate_rankings_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_rankings([], [{"question": "q", "source": "a.md"}])
