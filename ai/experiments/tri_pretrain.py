"""
Train the MC-GNN on real US recovery trade and save it as a platform candidate.

tri_study measures whether the architecture can learn real trade; this keeps
the model that does. Same data, same years, same early stopping:

    history 2019-2021 -> learn 2022 -> select on 2023 -> report on 2024

Two things differ from the study's control runs, both so the saved model can
serve a market other than the one it learned:

  * Location is read relative to the market (see LOCATION_FRAMES in
    ai/models/mc_gnn.py). A model that learned US coordinates would see every
    Kerala plant as lying outside anything it knows.
  * Several seeds are trained and the one with the best 2023 validation AUC
    is kept. Selection never looks at 2024.

The result is written next to the platform's checkpoints as
mc_gnn_tri_<timestamp>.pt with its metadata. It does not replace the serving
model: ai/experiments/platform_check.py decides that, on the platform's own
held-out test, by the same margin every retrain has to clear.

Usage:
    python -m ai.experiments.tri_pretrain
    python -m ai.experiments.tri_pretrain --seeds 42 7 13 --epochs 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from ai.evaluation.metrics import evaluate_model
from ai.experiments.tri_loader import DATA_DIR, EDGE_FEATURE_DIM, NODE_FEATURE_DIM, TRIDataset
from ai.experiments.tri_study import (
    HISTORY_START,
    LR,
    MIN_LOCATED_SHARE,
    TEST_YEAR,
    TRAIN_YEAR,
    VAL_YEAR,
    WEIGHT_DECAY,
    measure,
    pick_device,
    score,
    train_control,
)
from ai.models.baseline import BaselineRuleModel
from ai.models.mc_gnn import HSIC_GAMMA
from ai.training.train import PATIENCE
from app.utils.calibration import apply_platt, expected_calibration_error, fit_platt

CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"
LOCATION_FRAME = "market"


def _ranking_metrics(g: dict, scores) -> dict:
    """The fields the platform's checkpoint metadata carries, plus hit@1."""
    at5 = evaluate_model(g["y"], scores, k=5, group_ids=g["group_ids"])
    return {
        "precision@5": round(at5["precision@5"], 6),
        "recall@5": round(at5["recall@5"], 6),
        "ndcg@5": round(at5["ndcg@5"], 6),
        "auc": round(at5["auc"], 6),
        "hit@1": round(measure(g, scores)["hit@1"], 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    # The study's runs were still improving at 300; early stopping ends a run
    # sooner if it has converged.
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--cpu", action="store_true", help="train on the CPU even when a GPU is available")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = pick_device(args.cpu)

    years = list(range(HISTORY_START, TEST_YEAR + 1))
    print(f"Reading EPA TRI {years[0]}-{years[-1]} ...", flush=True)
    ds, coverage = TRIDataset.from_files(years, args.data_dir)
    share = coverage["per_year"][TEST_YEAR]["located_share"]
    if share < MIN_LOCATED_SHARE:
        sys.exit(f"STOP: only {share:.1%} of {TEST_YEAR} transfers located (need {MIN_LOCATED_SHARE:.0%}).")

    print("Building graphs ...", flush=True)
    g_train, g_val, g_test = (
        ds.build_graph(y, history_start=HISTORY_START) for y in (TRAIN_YEAR, VAL_YEAR, TEST_YEAR)
    )

    print(f"Training on {device}, location read relative to the market ...", flush=True)
    candidates = []
    for seed in args.seeds:
        t0 = time.time()
        model, info = train_control(
            g_train, g_val, seed, with_graph=True, epochs=args.epochs,
            location_frame=LOCATION_FRAME, device=device,
        )
        info["seed"], info["seconds"] = seed, round(time.time() - t0, 1)
        print(f"  seed {seed:<4} val AUC {info['val_auc']:.4f}  best epoch {info['best_epoch']} "
              f"of {info['epochs_run']}  ({info['seconds']}s)", flush=True)
        candidates.append((info["val_auc"], seed, model, info))

    _, seed, model, info = max(candidates, key=lambda c: (c[0], -c[1]))
    print(f"Keeping seed {seed} (best validation AUC).")

    # Calibrated on 2023, which chose the weights but never trained them.
    val_scores = score(model, g_val)
    model.calibration = fit_platt(val_scores, g_val["y"])
    ece = None
    if model.calibration is not None:
        adjusted = [apply_platt(float(p), *model.calibration) for p in val_scores]
        ece = {
            "before": round(expected_calibration_error(val_scores, g_val["y"]), 6),
            "after": round(expected_calibration_error(adjusted, g_val["y"]), 6),
        }

    test_metrics = _ranking_metrics(g_test, score(model, g_test))
    baseline_metrics = _ranking_metrics(g_test, BaselineRuleModel().predict(g_test["edge_attr"]))

    # The site loads checkpoints onto the CPU; save it there.
    model = model.cpu().eval()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"mc_gnn_tri_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt, meta_path = CHECKPOINT_DIR / f"{name}.pt", CHECKPOINT_DIR / f"{name}.json"
    torch.save(model, ckpt)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_file": ckpt.name,
        "trained_on": f"EPA TRI off-site recovery transfers, learn {TRAIN_YEAR} "
                      f"(history {HISTORY_START}-{TRAIN_YEAR - 1}), select {VAL_YEAR}, test {TEST_YEAR}",
        "location_frame": LOCATION_FRAME,
        "node_features": NODE_FEATURE_DIM,
        "edge_features": EDGE_FEATURE_DIM,
        "device": str(device),
        "seed": seed,
        "seeds_tried": [
            {"seed": s, "val_auc": round(a, 6), "best_epoch": i["best_epoch"], "epochs_run": i["epochs_run"]}
            for a, s, _, i in candidates
        ],
        "epochs": args.epochs,
        "epochs_run": info["epochs_run"],
        "best_epoch": info["best_epoch"],
        "patience": PATIENCE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "hsic_weight": HSIC_GAMMA,
        "validation_metrics": {"auc": round(info["val_auc"], 6)},
        "calibration": list(model.calibration) if model.calibration else None,
        "calibration_ece": ece,
        "selection_metric": "auc",
        "metric_type": "held_out",
        "us_test_metrics": test_metrics,
        "us_baseline_metrics": baseline_metrics,
        "num_train_edges": int(len(g_train["y"])),
        "num_val_edges": int(len(g_val["y"])),
        "num_test_edges": int(len(g_test["y"])),
        "num_nodes": int(g_train["stats"]["nodes"]),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")

    print(f"\n{TEST_YEAR} test (never seen):")
    for label, m in (("MC-GNN", test_metrics), ("baseline", baseline_metrics)):
        print(f"  {label:<9} AUC {m['auc']:.4f}  NDCG@5 {m['ndcg@5']:.4f}  "
              f"Recall@5 {m['recall@5']:.4f}  Hit@1 {m['hit@1']:.4f}")
    print(f"\nSaved {ckpt.relative_to(PROJECT_ROOT)}")
    print("Next: python -m ai.experiments.platform_check --candidate " + ckpt.name)


if __name__ == "__main__":
    main()
