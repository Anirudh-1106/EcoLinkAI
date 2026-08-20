"""
Baseline weighted rule-based model for comparative evaluation.
"""

from __future__ import annotations

import numpy as np


class BaselineRuleModel:
    """
    Weighted rule-based baseline model.
    Serves as academic benchmark against MC-GNN.
    """

    def __init__(self):
        # Feature weights for scoring
        self.weights = np.array([0.30, 0.40, 0.15, 0.15], dtype=np.float32)

    def predict(self, edge_attr: np.ndarray) -> np.ndarray:
        """
        edge_attr features: [norm_dist, compat_score, transport_cost, carbon_saving]
        """
        # Distance score (1 - norm_dist)
        dist_score = 1.0 - edge_attr[:, 0]
        compat = edge_attr[:, 1]
        t_cost = 1.0 - edge_attr[:, 2]
        c_saving = edge_attr[:, 3]

        feats = np.column_stack([dist_score, compat, t_cost, c_saving])
        scores = np.dot(feats, self.weights)
        return np.clip(scores, 0.0, 1.0)
