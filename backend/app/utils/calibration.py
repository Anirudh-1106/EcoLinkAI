"""
Turning the model's raw output into a believable probability.

The link predictor is trained to rank, and ranks well, but the number it
emits is not the chance of the deal happening. Measured against held-out
outcomes it ran consistently low: pairs it scored around 55% were accepted
about 66% of the time, and pairs it scored 12% were accepted 23% of the time.
Every band under-stated the truth, so the "AI Match %" a buyer reads was
systematically pessimistic.

Platt scaling corrects that by fitting a one-dimensional logistic map,
sigmoid(a * logit(p) + b), from the model's score to the observed acceptance
rate. Two parameters is deliberate: a small validation split cannot support
anything richer without fitting its noise.

The mapping is strictly increasing, so it moves no candidate above or below
another -- rankings, AUC and NDCG are all untouched. It changes only the
number shown to a person, which is the part that was wrong.

Fitted on validation rather than training data, whose scores the model has
already been optimised against and which would therefore look better
calibrated than reality, and never on the test split, which has to stay an
honest measure.
"""

from __future__ import annotations

import numpy as np

# Keeps logit() finite when the model emits a hard 0 or 1.
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit_platt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 2000,
    learning_rate: float = 0.05,
) -> tuple[float, float] | None:
    """
    Fit sigmoid(a * logit(p) + b) to observed outcomes.

    Returns (a, b), or None when the data cannot support a fit -- too few
    points, or every outcome identical, in which case any mapping would be
    fitting noise and the raw score is the safer answer.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()

    if len(p) < 30 or len(np.unique(y)) < 2:
        return None

    z = _logit(p)
    a, b = 1.0, 0.0

    # Plain gradient descent on the log loss. The surface is convex in two
    # parameters, so this needs no more machinery than it has.
    for _ in range(iterations):
        pred = 1.0 / (1.0 + np.exp(-(a * z + b)))
        err = pred - y
        grad_a = float(np.mean(err * z))
        grad_b = float(np.mean(err))
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b

    if not (np.isfinite(a) and np.isfinite(b)):
        return None

    return float(a), float(b)


def apply_platt(probability: float, a: float, b: float) -> float:
    """Map one raw score through a fitted calibration."""
    z = float(_logit(np.array([probability]))[0])
    return float(1.0 / (1.0 + np.exp(-(a * z + b))))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """
    Mean gap between what was predicted and what happened, weighted by volume.

    0 would mean every band of predictions came true at exactly the rate it
    claimed.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if len(p) == 0:
        return 0.0

    edges = np.linspace(p.min(), p.max(), bins + 1)
    total = 0.0
    for i in range(bins):
        upper_inclusive = i == bins - 1
        mask = (p >= edges[i]) & ((p <= edges[i + 1]) if upper_inclusive else (p < edges[i + 1]))
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)
