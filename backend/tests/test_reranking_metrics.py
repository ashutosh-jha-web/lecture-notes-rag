import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reranking_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k():
    ranked = ["a", "b", "c"]
    relevance = {"a": 3, "c": 1, "d": 2}

    assert recall_at_k(ranked, relevance, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, relevance, 3) == pytest.approx(2 / 3)


def test_recall_at_k_with_no_relevant_chunks():
    assert recall_at_k(["a", "b"], {}, 2) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(
        ["a", "b", "c"], {"b": 2}
    ) == pytest.approx(0.5)


def test_mean_reciprocal_rank_with_no_relevant_result():
    assert mean_reciprocal_rank(["a", "b"], {"c": 3}) == 0.0


def test_ndcg_rewards_relevant_chunks_at_the_top():
    relevance = {"a": 3, "b": 1}
    best = ndcg_at_k(["a", "b"], relevance, 2)
    worst = ndcg_at_k(["b", "a"], relevance, 2)

    assert best == pytest.approx(1.0)
    assert worst < best


def test_ndcg_with_no_relevance_labels():
    assert ndcg_at_k(["a", "b"], {}, 2) == 0.0
