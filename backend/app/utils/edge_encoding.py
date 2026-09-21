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

# Prior deals stop adding confidence after roughly this many, matching the
# point where the data generator's own rapport term saturates.
PRIOR_SUCCESS_SATURATION = 3.0

# Order is part of the contract -- the model's weights are bound to these
# positions, so appending or reordering requires retraining.
EDGE_FEATURE_NAMES = (
    "norm_distance",
    "material_compatibility",
    "quantity_compatibility",
    "quality_compatibility",
    "transport_cost",
    "carbon_saving",
    "prior_successes",
    "has_prior_interaction",
)


def _log_scale(value: float, reference: float) -> float:
    """Compress a wide-ranging non-negative value into 0-1, order preserved."""
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(reference), 1.0)


def encode_edge_features(
    *,
    distance_km: float,
    material_compatibility: float,
    quantity_compatibility: float,
    quality_compatibility: float,
    transport_cost: float,
    carbon_saving: float,
    prior_successes: int = 0,
    has_prior_interaction: bool = False,
) -> list[float]:
    """
    Build the edge feature vector for one supplier -> buyer pair.

    prior_successes and has_prior_interaction describe the relationship rather
    than the deal, and are the only inputs a graph model can propagate through
    the network -- without them the MC-GNN is asked to beat a per-row formula
    while seeing nothing the formula does not already see.

    They are reported as a pair on purpose. A count alone writes every partner
    a company has never dealt with as 0, which is indistinguishable from
    having tried and failed; the flag separates "no track record" from "a poor
    one". That distinction is what keeps genuinely new entrants -- the real
    KINFRA companies, which carry no invented trading history -- competing on
    material fit, distance and price instead of being ranked last for having
    no past.

    Both must be computed strictly from deals that closed *before* the one
    being scored. Counting the deal itself, or any deal after it, leaks the
    answer into the input and inflates every metric that follows.

    The three compatibility terms arrive separately rather than pre-blended.
    Blended, the score is half material match -- a constant across the exact
    matches that make up the history -- so it only ever moved across a quarter
    of its range, with the quantity term, the strongest of the three, confined
    to a third of that. Splitting them adds no information the blend did not
    already carry, but lets each vary over its own full range instead of being
    recovered from a compressed sum.

    Args:
        distance_km: Haversine distance between the two plants.
        material_compatibility: material match score, 0-100.
        quantity_compatibility: how closely the quantities line up, 0-100.
        quality_compatibility: purity against the requirement, 0-100.
        transport_cost: estimated freight cost in INR.
        carbon_saving: estimated CO2e avoided in kg.
        prior_successes: earlier accepted requests between this ordered pair.
        has_prior_interaction: whether the pair has any earlier request at all,
            whatever its outcome.

    Returns:
        Floats in the order of EDGE_FEATURE_NAMES, every one inside 0-1 so
        that no single feature can dominate the rest by raw magnitude alone.
    """
    return [
        min(max(distance_km, 0.0) / DISTANCE_REFERENCE_KM, 1.0),
        min(max(material_compatibility, 0.0) / 100.0, 1.0),
        min(max(quantity_compatibility, 0.0) / 100.0, 1.0),
        min(max(quality_compatibility, 0.0) / 100.0, 1.0),
        _log_scale(transport_cost, TRANSPORT_COST_REFERENCE_INR),
        _log_scale(carbon_saving, CARBON_SAVING_REFERENCE_KG),
        min(max(prior_successes, 0) / PRIOR_SUCCESS_SATURATION, 1.0),
        1.0 if has_prior_interaction else 0.0,
    ]
