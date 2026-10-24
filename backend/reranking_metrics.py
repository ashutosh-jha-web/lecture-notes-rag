"""Ranking metrics for manually judged retrieval results."""

import math


def recall_at_k(ranked_ids, relevance, k):
    """Fraction of all judged-relevant chunks found in the top k."""
    relevant_ids = {key for key, score in relevance.items() if score > 0}
    if not relevant_ids:
        return 0.0

    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(ranked_ids, relevance):
    """Reciprocal rank of the first relevant result."""
    for rank, identity in enumerate(ranked_ids, start=1):
        if relevance.get(identity, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevance, k):
    """Normalized discounted cumulative gain using graded relevance."""
    def gain(score):
        return (2**score) - 1

    actual = sum(
        gain(relevance.get(identity, 0)) / math.log2(rank + 1)
        for rank, identity in enumerate(ranked_ids[:k], start=1)
    )

    ideal_scores = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(
        gain(score) / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )

    return actual / ideal if ideal else 0.0
