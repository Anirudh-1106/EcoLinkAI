"""
Should a candidate checkpoint replace the model the platform is serving?

Scores the candidate, the serving model and the rule-based baseline on the
platform's own held-out test -- the same split, taken the same way, that
every retrain in ai/training/train.py is judged on -- and applies the same
rule: the candidate must beat the serving model's AUC by more than
PROMOTION_MARGIN. A candidate trained elsewhere gets no easier a test.

Without --promote nothing changes. With it, a winning candidate is promoted
through train.py's own atomic swap and then calibrated on the platform's
validation split, so the percentages it shows describe this platform rather
than the market it learned on. The model it replaces stays on disk under its
timestamped name.

Usage:
    python -m ai.experiments.platform_check --candidate mc_gnn_tri_<stamp>.pt
    python -m ai.experiments.platform_check --candidate mc_gnn_tri_<stamp>.pt --promote
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ai.evaluation.metrics import evaluate_model
from ai.graph.builder import build_industrial_graph
from ai.models.baseline import BaselineRuleModel
from ai.training.train import (
    PROMOTION_MARGIN,
    PROMOTION_METRIC,
    SPLIT_RATIO,
    SPLIT_SEED,
    VAL_RATIO,
    _calibrate_production,
    _evaluate_checkpoint_on_split,
    _promote_checkpoint,
    _split_edges,
)
from app.core.database import SessionLocal

CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"
PRODUCTION = CHECKPOINT_DIR / "mc_gnn_best.pt"
KEYS = ("auc", "ndcg@5", "recall@5", "precision@5")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="checkpoint file name in ai/checkpoints")
    ap.add_argument("--promote", action="store_true", help="promote the candidate if it wins")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    candidate = CHECKPOINT_DIR / args.candidate
    if not candidate.exists():
        sys.exit(f"No checkpoint at {candidate}")

    db = SessionLocal()
    try:
        data, _, _ = build_industrial_graph(db)
    finally:
        db.close()

    edge_times = getattr(data, "edge_time", None)
    split = _split_edges(data.edge_index, data.edge_attr, data.y, edge_times=edge_times)
    test_labels = split["test_labels"].numpy()
    groups = split["test_edge_index"][1].numpy()

    serving = _evaluate_checkpoint_on_split(PRODUCTION, data.x, split)
    challenger = _evaluate_checkpoint_on_split(candidate, data.x, split)
    baseline = evaluate_model(
        test_labels, BaselineRuleModel().predict(split["test_edge_attr"].numpy()), k=5, group_ids=groups
    )
    if serving is None or challenger is None:
        sys.exit("Could not score one of the checkpoints; see the warning above.")

    print(f"\nPlatform held-out test: {split['num_test']:,} requests, "
          f"{data.num_nodes:,} plants ({'temporal' if edge_times is not None else 'random'} split)")
    print(f"  {'':<30}" + "".join(f"{k:>13}" for k in KEYS))
    for label, m in ((f"candidate ({candidate.name})", challenger),
                     ("serving (mc_gnn_best.pt)", serving),
                     ("rule-based baseline", baseline)):
        print(f"  {label[:30]:<30}" + "".join(f"{m[k]:>13.4f}" for k in KEYS))

    gain = challenger[PROMOTION_METRIC] - serving[PROMOTION_METRIC]
    wins = gain > PROMOTION_MARGIN
    print(f"\n  {PROMOTION_METRIC.upper()} change {gain:+.4f}; the bar is +{PROMOTION_MARGIN}. "
          + ("Candidate wins." if wins else "Candidate does not clear it; the serving model stays."))

    if not (args.promote and wins):
        if args.promote:
            print("  Nothing promoted.")
        return

    meta_path = candidate.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    # The site shows these figures as the serving model's held-out scores, so
    # they must be the platform's, not the market it was trained on.
    meta.update({
        "promoted_at": datetime.now().isoformat(),
        "promoted_by": "ai/experiments/platform_check.py",
        "metrics": {k: round(challenger[k], 6) for k in KEYS},
        "baseline_metrics": {k: round(baseline[k], 6) for k in KEYS},
        "replaced_metrics": {k: round(serving[k], 6) for k in KEYS},
        "promotion_metric": PROMOTION_METRIC,
        "metric_type": "held_out",
        "split_kind": "temporal" if edge_times is not None else f"random seed={SPLIT_SEED}",
        "split_ratio": SPLIT_RATIO,
        "val_ratio": VAL_RATIO,
        "platform_test_edges": split["num_test"],
    })
    _promote_checkpoint(candidate, meta, CHECKPOINT_DIR)
    _calibrate_production(CHECKPOINT_DIR, data.x, split, split["val_labels"].numpy())
    print(f"  Promoted {candidate.name} to mc_gnn_best.pt and calibrated it on platform validation.")


if __name__ == "__main__":
    main()
