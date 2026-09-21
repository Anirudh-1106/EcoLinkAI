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
        # Feature weights for scoring. The relationship features are weighted
        # too: the MC-GNN is only worth preferring if it beats this baseline
        # while both are shown the same inputs, so withholding them here would
        # manufacture the comparison rather than test it.
        self.weights = np.array(
            [0.25, 0.16, 0.16, 0.10, 0.08, 0.06, 0.14, 0.05], dtype=np.float32
        )

    def predict(self, edge_attr: np.ndarray) -> np.ndarray:
        """
        edge_attr columns follow EDGE_FEATURE_NAMES: [norm_dist, material,
        quantity, quality, transport_cost, carbon_saving, prior_successes,
        has_prior_interaction].
        """
        # Distance score (1 - norm_dist)
        dist_score = 1.0 - edge_attr[:, 0]
        material = edge_attr[:, 1]
        quantity = edge_attr[:, 2]
        quality = edge_attr[:, 3]
        t_cost = 1.0 - edge_attr[:, 4]
        c_saving = edge_attr[:, 5]
        prior = edge_attr[:, 6]
        has_prior = edge_attr[:, 7]

        feats = np.column_stack(
            [dist_score, material, quantity, quality, t_cost, c_saving, prior, has_prior]
        )
        scores = np.dot(feats, self.weights)
        return np.clip(scores, 0.0, 1.0)
