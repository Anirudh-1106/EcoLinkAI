"""
How real recovery trade decides between partners, measured for the simulator.

The platform's KINFRA market (scripts/database/generate_market_partners.py)
is synthetic, and its acceptance weights were set by judgement. Measured
against EPA TRI, the result did not behave like real trade: whether a pair
had traded before separated real partners from alternatives at AUC 0.924 on
TRI and 0.508 on the platform. This measures, on TRI, the two things the
simulator needs to reproduce that:

  acceptance weights  a logistic regression of "was this the partner used"
                      on the same 0-1 encodings the simulator computes --
                      proximity (1 - distance/500 km), quantity fit, and prior
                      successes (min(count/3, 1)). Fitted on the training
                      years' candidate sets: each real partner against nine
                      sampled alternatives. Under that case-control sampling
                      the slopes carry over and only the intercept moves, so
                      the intercept is not reported.

  partner retention   of the pairs that traded in one year, the share that
                      trade again the next; and of the pairs trading in a
                      year, the share new to each other.

Caveat recorded with the result: alternatives are drawn preferentially from
within the 90th percentile of real transfer distance, so they are nearer than
an unrestricted draw would be, and the proximity weight is conservative.

Usage:
    python -m ai.experiments.tri_calibration

Writes ai/experiments/results/tri_calibration.json.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from ai.evaluation.metrics import roc_auc
from ai.experiments.tri_loader import DATA_DIR, TRIDataset
from ai.experiments.tri_study import HISTORY_START, TEST_YEAR
from app.utils.edge_encoding import EDGE_FEATURE_NAMES

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_calibration.json"
FIT_YEARS = (2020, 2021, 2022)

# The simulator's inputs, as positions in the platform's edge encoding.
TERMS = {
    "proximity": ("norm_distance", lambda v: 1.0 - v),
    "quantity_fit": ("quantity_compatibility", lambda v: v),
    "prior_successes": ("prior_successes", lambda v: v),
}


def acceptance_weights(graphs: list[dict]) -> dict:
    from sklearn.linear_model import LogisticRegression

    cols = {name: EDGE_FEATURE_NAMES.index(src) for name, (src, _) in TERMS.items()}
    X = np.vstack([
        np.column_stack([fn(g["edge_attr"][:, cols[name]]) for name, (_, fn) in TERMS.items()])
        for g in graphs
    ])
    y = np.concatenate([g["y"] for g in graphs])
    model = LogisticRegression(max_iter=2000).fit(X, y)
    return {
        "weights": {name: round(float(w), 4) for name, w in zip(TERMS, model.coef_[0])},
        "single_term_auc": {name: round(roc_auc(y, X[:, i]), 4) for i, name in enumerate(TERMS)},
        "rows": int(len(y)),
        "positive_share": round(float(y.mean()), 4),
    }


def retention(ds: TRIDataset, years: list[int]) -> dict:
    pairs_by_year: dict[int, set] = defaultdict(set)
    for t in ds.transfers:
        if t.year in years and ds._located(t.sender) and ds._located(t.receiver):
            pairs_by_year[t.year].add((t.sender, t.receiver))
    out, seen = {}, set()
    for prev, cur in zip(years, years[1:]):
        seen |= pairs_by_year[prev]
        kept = pairs_by_year[prev] & pairs_by_year[cur]
        out[f"{prev}->{cur}"] = {
            "retained_share": round(len(kept) / max(len(pairs_by_year[prev]), 1), 4),
            "new_pair_share": round(len(pairs_by_year[cur] - seen) / max(len(pairs_by_year[cur]), 1), 4),
        }
    return {
        "per_year": out,
        "retained_share_mean": round(float(np.mean([v["retained_share"] for v in out.values()])), 4),
        "new_pair_share_mean": round(float(np.mean([v["new_pair_share"] for v in out.values()])), 4),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    years = list(range(HISTORY_START, TEST_YEAR + 1))
    print("Reading EPA TRI ...", flush=True)
    ds, _ = TRIDataset.from_files(years, DATA_DIR)
    print("Building training-year candidate sets ...", flush=True)
    graphs = [ds.build_graph(y, history_start=HISTORY_START) for y in FIT_YEARS]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_years": list(FIT_YEARS),
        "acceptance": acceptance_weights(graphs),
        # Retention is read on years before the test year only.
        "retention": retention(ds, [y for y in years if y < TEST_YEAR]),
        "caveat": "alternatives sampled preferentially within the 90th percentile of real "
                  "transfer distance, so the proximity weight is conservative",
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("acceptance", "retention")}, indent=2))
    print(f"Saved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
