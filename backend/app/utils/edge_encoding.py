"""
The edge feature vector the MC-GNN consumes.

Training (ai/features/edge_features.py) and serving
(app/services/recommendation_service.py) have to hand the model an identical
encoding. A feature scaled one way while training and another way at inference
is not a small discrepancy -- the model reads it as a different signal
entirely. Both sides call encode_edge_features() so the two cannot drift.

On scaling: distance and compatibility are naturally bounded and divide down
to 0-1 directly. Transport cost and carbon saving are not. Across the real
dataset they span about seven orders of magnitude, because a listing's
quantity may be recorded in kilograms or in tons -- a 1.7 t collection and a
2,500 t bulk movement sit in the same column. Dividing those by a flat
constant left 27% of cost values and 86% of carbon values above 1.0, topping
out near 8,800, while distance and compatibility stayed inside 0-1. Sheer
magnitude then made shipment size the dominant term, and a listing whose
quantity did not match the buyer's at all outranked one that did.

Rescaling them linearly would fail in the opposite direction: dividing by the
observed maximum drops 70% of the data below 0.002, leaving almost no spread
to separate candidates on. A logarithm compresses the long tail while keeping
the order intact -- larger is still larger -- and against the reference points
below the observed values spread across roughly 0.08-1.0.
"""

from __future__ import annotations

import math

# Reference points, not observed maxima: the value at which a feature is
# treated as "as large as this realistically gets" and saturates at 1.0.
# Chosen above the p99 of real data so the top of the range stays meaningful
# while genuine outliers clamp instead of dwarfing every other feature.
DISTANCE_REFERENCE_KM = 500.0
TRANSPORT_COST_REFERENCE_INR = 5_000_000.0
CARBON_SAVING_REFERENCE_KG = 10_000_000.0

# Order is part of the contract -- the model's weights are bound to these
# positions, so appending or reordering requires retraining.
EDGE_FEATURE_NAMES = (
    "norm_distance",
    "compatibility",
    "transport_cost",
    "carbon_saving",
)


def _log_scale(value: float, reference: float) -> float:
    """Compress a wide-ranging non-negative value into 0-1, order preserved."""
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(reference), 1.0)


def encode_edge_features(
    *,
    distance_km: float,
    compatibility: float,
    transport_cost: float,
    carbon_saving: float,
) -> list[float]:
    """
    Build the 4-dim edge feature vector for one supplier -> buyer pair.

    Args:
        distance_km: Haversine distance between the two plants.
        compatibility: composite material/quantity/quality score, 0-100.
        transport_cost: estimated freight cost in INR.
        carbon_saving: estimated CO2e avoided in kg.

    Returns:
        Four floats, every one of them inside 0-1 so that no single feature
        can dominate the others by raw magnitude alone.
    """
    return [
        min(max(distance_km, 0.0) / DISTANCE_REFERENCE_KM, 1.0),
        min(max(compatibility, 0.0) / 100.0, 1.0),
        _log_scale(transport_cost, TRANSPORT_COST_REFERENCE_INR),
        _log_scale(carbon_saving, CARBON_SAVING_REFERENCE_KG),
    ]
