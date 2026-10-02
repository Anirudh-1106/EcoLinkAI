"""
Tune the pair-aware MC-GNN for top-1 ranking, then compare seed ensembles
with plain XGBoost.

After tri_pairaware and tri_ensemble, the ensembled MC-GNN led plain XGBoost
on AUC (significantly), NDCG@5 and Recall@5, and trailed it on Hit@1. Two
settings bear most directly on Hit@1 and were never searched:

  ranking weight  how strongly the listwise term -- softmax over each
                  question's candidates -- counts against cross-entropy
  capacity        embedding sizes (hidden 64 / out 32 so far)

Stage 1 tries each on one seed, choosing by 2023 validation Hit@1 with AUC as
the tie-break; 2024 is not looked at. Stage 2 trains the chosen setting on
five seeds and averages them, and does the same for plain XGBoost (the 17
facility and 8 pair inputs, no graph), so both sides are ensembles of five.
A paired bootstrap over 2024 questions tests the difference.

The MC-GNN keeps the inputs of tri_pairaware: the graph's three networks,
pair sets, and the two network columns. XGBoost cannot read a graph; the
comparison is between the MC-GNN with what it can read and XGBoost with what
it can.

Usage:
    python -m ai.experiments.tri_mcgnn_tune

Writes ai/experiments/results/tri_mcgnn_tune.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from ai.experiments.tri_improve import PAIR_INPUTS
from ai.experiments.tri_loader import DATA_DIR, NODE_FEATURE_DIM, TRIDataset
from ai.experiments.tri_multiview import TRAIN_POOL_PER_POSITIVE, TRAIN_YEARS, FreshNegatives, fit_xgboost, measure_subset
from ai.experiments.tri_pairaware import PairGraph, predict, train
from ai.experiments.tri_significance import BOOTSTRAP_SEED, Questions, paired_bootstrap
from ai.experiments.tri_study import HISTORY_START, MIN_LOCATED_SHARE, TEST_YEAR, VAL_YEAR, measure, pick_device
from ai.models.mc_gnn import PairAwareMCGNN

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_mcgnn_tune.json"
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"

# From tri_pairaware's search on 2023.
HSIC_WEIGHT = 0.2
LR = 0.01
TUNING_SEED = 42
CONFIGS = [
    {"ranking_weight": 1.0, "hidden_dim": 64, "out_dim": 32},
    {"ranking_weight": 3.0, "hidden_dim": 64, "out_dim": 32},
    {"ranking_weight": 1.0, "hidden_dim": 128, "out_dim": 64},
    {"ranking_weight": 3.0, "hidden_dim": 128, "out_dim": 64},
]
METRICS = ("auc", "ndcg@5", "recall@5", "hit@1")


def factory(cfg: dict):
    return lambda: PairAwareMCGNN(NODE_FEATURE_DIM, edge_dim=PAIR_INPUTS, hsic_weight=HSIC_WEIGHT,
                                  hidden_dim=cfg["hidden_dim"], out_dim=cfg["out_dim"])


def summary(g: dict, scores: list[np.ndarray], new: np.ndarray) -> dict:
    per = [measure(g, s) for s in scores]
    out = {k: float(np.mean([p[k] for p in per])) for k in METRICS}
    out["new_hit@1"] = float(np.mean([measure_subset(g, s, new)["hit@1"] for s in scores]))
    return out


def save(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 21, 99])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    device = pick_device()

    print("Reading EPA TRI and building graphs and pair sets ...", flush=True)
    ds, coverage = TRIDataset.from_files(list(range(HISTORY_START, TEST_YEAR + 1)), DATA_DIR)
    if coverage["per_year"][TEST_YEAR]["located_share"] < MIN_LOCATED_SHARE:
        sys.exit("STOP: too few test-year transfers located.")
    train_graphs = [
        ds.build_graph(y, history_start=HISTORY_START, views=True, negatives_per_positive=TRAIN_POOL_PER_POSITIVE)
        for y in TRAIN_YEARS
    ]
    g_val = ds.build_graph(VAL_YEAR, history_start=HISTORY_START, views=True)
    g_test = ds.build_graph(TEST_YEAR, history_start=HISTORY_START, views=True)
    train_set = [(PairGraph(g, device, keep_sets_on_device=False), FreshNegatives(g, device)) for g in train_graphs]
    val_pg = PairGraph(g_val, device, keep_sets_on_device=True)
    test_pg = PairGraph(g_test, device, keep_sets_on_device=True)
    new_test, new_val = g_test["sender_is_new"], g_val["sender_is_new"]
    print("  ready", flush=True)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"hsic_weight": HSIC_WEIGHT, "lr": LR, "configs": CONFIGS, "seeds": args.seeds,
                   "selection": "2023 validation Hit@1, then AUC", "device": str(device)},
        "tuning": [],
    }

    print("\nStage 1: tuning on 2023 ...", flush=True)
    for cfg in CONFIGS:
        t0 = time.time()
        model, info = train(factory(cfg), "views", train_set, val_pg, TUNING_SEED, args.epochs, LR, device,
                            ranking_weight=cfg["ranking_weight"])
        val = summary(g_val, [predict(model, val_pg, "views")], new_val)
        row = {**cfg, **info, "val": val, "seconds": round(time.time() - t0, 1)}
        results["tuning"].append(row)
        print(f"  ranking {cfg['ranking_weight']:<4} size {cfg['hidden_dim']}/{cfg['out_dim']:<3}  val AUC {val['auc']:.4f}  "
              f"val Hit@1 {val['hit@1']:.4f}  (best epoch {info['best_epoch']}, {row['seconds']}s)", flush=True)
        save(results)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    best = max(results["tuning"], key=lambda r: (round(r["val"]["hit@1"], 4), r["val"]["auc"]))
    chosen = {k: best[k] for k in ("ranking_weight", "hidden_dim", "out_dim")}
    results["chosen"] = chosen
    print(f"  chosen: {chosen}", flush=True)

    print(f"\nStage 2: {len(args.seeds)} seeds each ...", flush=True)
    scores = {"MC-GNN": [], "XGBoost": []}
    best_model = None
    for seed in args.seeds:
        t0 = time.time()
        model, info = train(factory(chosen), "views", train_set, val_pg, seed, args.epochs, LR, device,
                            ranking_weight=chosen["ranking_weight"])
        s = predict(model, test_pg, "views")
        scores["MC-GNN"].append(s)
        m = measure(g_test, s)
        print(f"  MC-GNN   seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  NDCG@5 {m['ndcg@5']:.4f}  "
              f"({round(time.time() - t0)}s)", flush=True)
        if best_model is None or info["val_auc"] > best_model[0]:
            best_model = (info["val_auc"], seed, copy.deepcopy(model).cpu().eval(), info)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    del train_set
    for seed in args.seeds:
        s, _ = fit_xgboost(train_graphs, g_val, g_test, seed, device, False)
        scores["XGBoost"].append(s)
        m = measure(g_test, s)
        print(f"  XGBoost  seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  NDCG@5 {m['ndcg@5']:.4f}", flush=True)

    g = {"y": g_test["y"], "group_ids": g_test["group_ids"]}
    results["models"] = {}
    for name, ss in scores.items():
        results["models"][f"{name} (mean of {len(ss)} seeds)"] = summary(g, ss, new_test)
        results["models"][f"{name} (ensemble of {len(ss)})"] = summary(g, [np.mean(ss, axis=0)], new_test)
    save(results)

    all_q = Questions(g_test["y"], g_test["group_ids"])
    new_q = Questions(g_test["y"][new_test], g_test["group_ids"][new_test])
    ens = {n: [np.mean(ss, axis=0)] for n, ss in scores.items()}
    results["bootstrap"] = {}
    for label, a, b in (("ensemble", ens["MC-GNN"], ens["XGBoost"]), ("seed mean", scores["MC-GNN"], scores["XGBoost"])):
        for subset, q, mask in (("all", all_q, None), ("new facilities", new_q, new_test)):
            aa = a if mask is None else [x[mask] for x in a]
            bb = b if mask is None else [x[mask] for x in b]
            results["bootstrap"][f"{label} | {subset}"] = paired_bootstrap(q, aa, bb, args.bootstrap, BOOTSTRAP_SEED)

    _, seed, model, info = best_model
    ckpt = CHECKPOINT_DIR / f"mc_gnn_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(model, ckpt)
    ckpt.with_suffix(".json").write_text(json.dumps({"checkpoint_file": ckpt.name, "model": "PairAwareMCGNN",
                                                     "seed": seed, **info, **chosen, "hsic_weight": HSIC_WEIGHT},
                                                    indent=2, default=float), encoding="utf-8")
    results["saved_checkpoint"] = ckpt.name
    np.savez_compressed(DATA_DIR / "results" / "tri_mcgnn_tune_scores.npz", y=g_test["y"],
                        group_ids=g_test["group_ids"], sender_is_new=new_test,
                        **{f"{n}|{i}": s for n, ss in scores.items() for i, s in enumerate(ss)})
    save(results)

    print("\n" + "=" * 100)
    for name, r in results["models"].items():
        print(f"  {name:<28} AUC {r['auc']:.4f}  NDCG@5 {r['ndcg@5']:.4f}  Recall@5 {r['recall@5']:.4f}  "
              f"Hit@1 {r['hit@1']:.4f}  new {r['new_hit@1']:.4f}")
    for key, res in results["bootstrap"].items():
        print(f"  MC-GNN minus XGBoost, {key}: " + "  ".join(
            f"{k} {r['difference']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]{'*' if r['significant'] else ''}"
            for k, r in res.items()))
    print(f"Saved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
