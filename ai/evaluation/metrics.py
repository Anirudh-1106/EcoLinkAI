"""
Recommendation Evaluation Metrics.

Computes academic ranking and retrieval metrics:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _mean_over_queries(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    metric_fn,
    k: int,
) -> tuple[float, int]:
    """
    Average a top-K metric over queries instead of across the whole split.

    Precision@K and NDCG@K describe one ranked list shown to one user. Applied
    to an entire evaluation split at once they instead describe the K
    highest-scoring edges in the dataset -- a single sample of K items, which
    on this data pinned precision@5 to 1.0 while recall@5 read 0.018 (five
    hits measured against every positive in the split). Grouping by buyer
    restores the intended meaning: rank each buyer's candidate suppliers,
    score that list, then average across buyers.

    Queries with no positive are skipped rather than counted as zero -- there
    is no correct ranking to find, so they would only dilute the average.

    Returns the mean and the number of queries it was taken over.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for position, group in enumerate(group_ids):
        groups[int(group)].append(position)

    scores = []
    for positions in groups.values():
        idx = np.asarray(positions)
        query_true = y_true[idx]
        if query_true.sum() == 0:
            continue
        scores.append(metric_fn(query_true, y_score[idx], k))

    if not scores:
        return 0.0, 0
    return float(np.mean(scores)), len(scores)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    """Compute Precision@K."""
    if len(y_score) == 0:
        return 0.0
    top_k_indices = np.argsort(y_score)[::-1][:k]
    hits = np.sum(y_true[top_k_indices] == 1)
    return float(hits / min(k, len(y_score)))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    """Compute Recall@K."""
    total_positives = np.sum(y_true == 1)
    if total_positives == 0:
        return 0.0
    top_k_indices = np.argsort(y_score)[::-1][:k]
    hits = np.sum(y_true[top_k_indices] == 1)
    return float(hits / total_positives)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    """Compute NDCG@K."""
    if len(y_score) == 0 or np.sum(y_true) == 0:
        return 0.0

    top_k_indices = np.argsort(y_score)[::-1][:k]
    gains = y_true[top_k_indices]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)

    ideal_gains = np.sort(y_true)[::-1][:k]
    ideal_dcg = np.sum(ideal_gains / discounts)

    if ideal_dcg == 0:
        return 0.0
    return float(dcg / ideal_dcg)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area under the ROC curve, computed from rank statistics.

    Unlike Precision@K and NDCG@K, which only inspect the top K of the
    ranking, this uses every pair in the set. On a test set of a few hundred
    edges that makes it far less sensitive to which handful of items happen
    to land on top, so it is the stabler signal for comparing two models.

    Returns 0.5 (no better than chance) when only one class is present.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # Tied scores must share the average of the ranks they span, otherwise
    # the result depends on the arbitrary order of equally scored items.
    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        stop = start
        while stop + 1 < len(sorted_scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        if stop > start:
            tied = order[start : stop + 1]
            ranks[tied] = ranks[tied].mean()
        start = stop + 1

    rank_sum = float(ranks[y_true == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_model(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 5,
    group_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute all evaluation metrics at K, plus rank-wide AUC.

    group_ids assigns each edge to the query it belongs to -- the buyer whose
    candidate suppliers are being ranked. Supply it: without it the top-K
    metrics are taken over the whole split at once and no longer mean what
    their names say (see _mean_over_queries). AUC is unaffected either way,
    since it reads every pair rather than a top-K slice.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if group_ids is None:
        return {
            f"precision@{k}": precision_at_k(y_true, y_score, k),
            f"recall@{k}": recall_at_k(y_true, y_score, k),
            f"ndcg@{k}": ndcg_at_k(y_true, y_score, k),
            "auc": roc_auc(y_true, y_score),
            "num_queries": 0,
        }

    group_ids = np.asarray(group_ids)
    precision, n_queries = _mean_over_queries(y_true, y_score, group_ids, precision_at_k, k)
    recall, _ = _mean_over_queries(y_true, y_score, group_ids, recall_at_k, k)
    ndcg, _ = _mean_over_queries(y_true, y_score, group_ids, ndcg_at_k, k)

    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg,
        "auc": roc_auc(y_true, y_score),
        "num_queries": n_queries,
    }
