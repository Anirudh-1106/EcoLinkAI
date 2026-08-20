"""
Recommendation Evaluation Metrics.

Computes academic ranking and retrieval metrics:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
"""

from __future__ import annotations

import numpy as np


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


def evaluate_model(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> dict[str, float]:
    """Compute all evaluation metrics at K."""
    return {
        f"precision@{k}": precision_at_k(y_true, y_score, k),
        f"recall@{k}": recall_at_k(y_true, y_score, k),
        f"ndcg@{k}": ndcg_at_k(y_true, y_score, k),
    }
