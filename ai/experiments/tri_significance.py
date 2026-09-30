"""
Two questions the headline comparison with XGBoost leaves open.

1. Without hand-built features, which model learns the market better?
   XGBoost's best score came with capability_score and peer_votes -- columns
   written by hand from the relationship, capability and peer networks. The
   multi-view MC-GNN can read those networks itself. So the like-for-like
   question is also asked with neither model given the columns: the
   multi-view MC-GNN (channel dropout) reading the three networks, against
   XGBoost, both on the platform's 8 pair inputs and the 17 facility inputs.

2. Are the gaps real? Every figure so far is a mean over three seeds with no
   error bars across the test set. A paired bootstrap over the 2024
   questions -- resampling questions, not rows, since each question's
   candidates move together -- gives a 95% interval and a two-sided p-value
   for each difference, for AUC, Hit@1 and NDCG@5. Seeds are averaged inside
   every resample, so the interval describes the reported numbers.

Both MC-GNN variants use tri_improve's regime: three training years, fresh
negatives each epoch, the ranking term, up to 1000 epochs, selection on 2023.

Usage:
    python -m ai.experiments.tri_significance
    python -m ai.experiments.tri_significance --bootstrap 2000

Writes ai/experiments/results/tri_significance.json; per-row test scores go
to <data dir>/results/ (large, and derived from the raw data, so kept with it).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from ai.evaluation.metrics import ndcg_at_k, precision_at_k
from ai.experiments.tri_identity import CHANNEL_DROPOUT
from ai.experiments.tri_improve import PAIR_INPUTS, RANKING_WEIGHT, train, with_network_columns
from ai.experiments.tri_loader import DATA_DIR, EDGE_FEATURE_DIM, NODE_FEATURE_DIM, TRIDataset
from ai.experiments.tri_multiview import (
    TRAIN_POOL_PER_POSITIVE,
    TRAIN_YEARS,
    FreshNegatives,
    OnDevice,
    _summarise,
    fit_xgboost,
    measure_subset,
    predict,
)
from ai.experiments.tri_study import HISTORY_START, MIN_LOCATED_SHARE, TEST_YEAR, VAL_YEAR, measure, pick_device
from ai.models.mc_gnn import MultiViewMCGNN
from ai.training.train import PATIENCE

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_significance.json"
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"
BOOTSTRAP_SEED = 2024

# (name, uses the two hand-built network columns)
NEURAL = {
    "Multi-view MC-GNN + channel dropout": True,
    "Multi-view MC-GNN + channel dropout, no network columns": False,
}
TABULAR = {
    "XGBoost + network columns": True,
    "XGBoost": False,
}
# The comparisons tested: each MC-GNN against XGBoost given the same information.
PAIRS = [
    ("Multi-view MC-GNN + channel dropout", "XGBoost + network columns"),
    ("Multi-view MC-GNN + channel dropout, no network columns", "XGBoost"),
    ("Multi-view MC-GNN + channel dropout", "XGBoost"),
]
# Facilities with no history are where the designs differ most, and where a
# gap in the overall figure can hide the opposite gap; tested on their own.
NEW_SENDER_PAIRS = [
    ("Multi-view MC-GNN + channel dropout", "XGBoost"),
    ("Multi-view MC-GNN + channel dropout", "XGBoost + network columns"),
    ("Multi-view MC-GNN + channel dropout, no network columns", "XGBoost"),
]


def multiview(edge_dim: int) -> MultiViewMCGNN:
    model = MultiViewMCGNN(NODE_FEATURE_DIM, edge_dim=edge_dim)
    model.fusion.channel_dropout = CHANNEL_DROPOUT
    return model


def prepare(g: dict, device, network_columns: bool) -> OnDevice:
    d = OnDevice(g, device)
    if network_columns:
        return with_network_columns(d)
    d.groups = torch.as_tensor(g["group_ids"], device=device)
    return d


# ── paired bootstrap over questions ─────────────────────────────────────


class Questions:
    """The test set's questions, and each score vector's per-question results."""

    def __init__(self, y: np.ndarray, groups: np.ndarray):
        by_group: dict[int, list[int]] = defaultdict(list)
        for i, gid in enumerate(groups):
            by_group[int(gid)].append(i)
        # Questions with no real partner carry no ranking to find; the
        # project's metrics skip them, and so does this.
        self.rows = [np.asarray(r) for r in by_group.values() if y[np.asarray(r)].sum() > 0]
        self.y = y
        self.row_question = np.full(len(y), -1, dtype=np.int64)
        for q, r in enumerate(self.rows):
            self.row_question[r] = q
        self.kept = self.row_question >= 0

    def per_question(self, scores: np.ndarray) -> dict[str, np.ndarray]:
        """Hit@1 and NDCG@5 of every question, exactly as ai.evaluation.metrics scores them."""
        hit = np.array([precision_at_k(self.y[r], scores[r], 1) for r in self.rows])
        ndcg = np.array([ndcg_at_k(self.y[r], scores[r], 5) for r in self.rows])
        return {"hit@1": hit, "ndcg@5": ndcg}


class WeightedAUC:
    """AUC under per-row weights, ties sharing half credit; one sort, then O(n) per resample."""

    def __init__(self, y: np.ndarray, scores: np.ndarray):
        order = np.argsort(scores, kind="mergesort")
        _, self.block = np.unique(scores[order], return_inverse=True)
        self.order = order
        self.pos = (y[order] > 0.5).astype(np.float64)
        self.n_blocks = int(self.block.max()) + 1

    def __call__(self, row_weight: np.ndarray) -> float:
        w = row_weight[self.order]
        pos_w = np.bincount(self.block, w * self.pos, self.n_blocks)
        neg_w = np.bincount(self.block, w * (1 - self.pos), self.n_blocks)
        below = np.cumsum(neg_w) - neg_w
        total = pos_w.sum() * neg_w.sum()
        return float(np.sum(pos_w * (below + 0.5 * neg_w)) / total) if total else 0.5


def paired_bootstrap(questions: Questions, a: list[np.ndarray], b: list[np.ndarray], n: int, seed: int) -> dict:
    """
    Model a minus model b, each averaged over its seeds, on resampled question sets.

    a and b are per-seed score vectors over the test rows. Returns the observed
    difference, a 95% percentile interval and a two-sided p-value per metric.
    """
    rng = np.random.default_rng(seed)
    q = len(questions.rows)
    pq_a = [questions.per_question(s) for s in a]
    pq_b = [questions.per_question(s) for s in b]
    auc_a = [WeightedAUC(questions.y, s) for s in a]
    auc_b = [WeightedAUC(questions.y, s) for s in b]

    def means(counts: np.ndarray) -> dict[str, float]:
        row_w = np.where(questions.kept, counts[np.maximum(questions.row_question, 0)], 0).astype(np.float64)
        out = {"auc": np.mean([f(row_w) for f in auc_a]) - np.mean([f(row_w) for f in auc_b])}
        for k in ("hit@1", "ndcg@5"):
            wa = np.mean([np.average(p[k], weights=counts) for p in pq_a])
            wb = np.mean([np.average(p[k], weights=counts) for p in pq_b])
            out[k] = wa - wb
        return out

    observed = means(np.ones(q))
    draws = defaultdict(list)
    for _ in range(n):
        counts = np.bincount(rng.integers(0, q, q), minlength=q).astype(np.float64)
        for k, v in means(counts).items():
            draws[k].append(v)

    out = {}
    for k, obs in observed.items():
        d = np.asarray(draws[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[k] = {"difference": float(obs), "ci95": [float(lo), float(hi)], "p_value": float(min(p, 1.0)),
                  "significant": bool(lo > 0 or hi < 0)}
    return out


# ── run ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    device = pick_device(args.cpu)
    scores_dir = args.data_dir / "results"
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores_path = scores_dir / "tri_significance_scores.npz"

    years = list(range(HISTORY_START, TEST_YEAR + 1))
    print(f"Reading EPA TRI {years[0]}-{years[-1]} ...", flush=True)
    ds, coverage = TRIDataset.from_files(years, args.data_dir)
    if coverage["per_year"][TEST_YEAR]["located_share"] < MIN_LOCATED_SHARE:
        sys.exit("STOP: too few test-year transfers located.")
    print("Building graphs ...", flush=True)
    train_graphs = [
        ds.build_graph(y, history_start=HISTORY_START, views=True, negatives_per_positive=TRAIN_POOL_PER_POSITIVE)
        for y in TRAIN_YEARS
    ]
    g_val = ds.build_graph(VAL_YEAR, history_start=HISTORY_START, views=True)
    g_test = ds.build_graph(TEST_YEAR, history_start=HISTORY_START, views=True)
    new_mask = g_test["sender_is_new"]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "train_years": list(TRAIN_YEARS), "val_year": VAL_YEAR, "test_year": TEST_YEAR,
            "seeds": args.seeds, "max_epochs": args.epochs, "patience": PATIENCE,
            "ranking_weight": RANKING_WEIGHT, "channel_dropout": CHANNEL_DROPOUT,
            "bootstrap_resamples": args.bootstrap, "bootstrap_unit": "test question (sender, chemical)",
            "device": str(device),
        },
        "models": {},
    }
    scores: dict[str, list[np.ndarray]] = {}

    print(f"Training on {device} ...", flush=True)
    for name, network_columns in NEURAL.items():
        train_set = [(prepare(g, device, network_columns), FreshNegatives(g, device)) for g in train_graphs]
        val_dev, test_dev = prepare(g_val, device, network_columns), prepare(g_test, device, network_columns)
        edge_dim = PAIR_INPUTS if network_columns else EDGE_FEATURE_DIM
        runs, best = [], None
        for seed in args.seeds:
            t0 = time.time()
            model, info = train(RANKING_WEIGHT, train_set, val_dev, seed, args.epochs, device,
                                model_factory=lambda: multiview(edge_dim), mode="views")
            s = predict(model, test_dev, "views")
            scores.setdefault(name, []).append(s)
            m = measure(g_test, s)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, s, new_mask)})
            print(f"  {name:<56} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"new {runs[-1]['new_senders']['hit@1']:.4f}  ({runs[-1]['seconds']}s)", flush=True)
            if best is None or info["val_auc"] > best[0]:
                best = (info["val_auc"], seed, model.cpu().eval(), info)
            else:
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["models"][name] = _summarise(runs)

        _, seed, model, info = best
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "mv" if network_columns else "mv_plain"
        ckpt = CHECKPOINT_DIR / f"mc_gnn_{tag}_cd_{stamp}.pt"
        torch.save(model, ckpt)
        ckpt.with_suffix(".json").write_text(json.dumps({
            "checkpoint_file": ckpt.name, "model": "MultiViewMCGNN", "variant": name,
            "channel_dropout": CHANNEL_DROPOUT, "pair_inputs": edge_dim, "seed": seed, **info,
            "needs": "three message graphs from ai/experiments/tri_views.py"
                     + (" and pair inputs with capability_score, peer_votes" if network_columns else ""),
        }, indent=2, default=float), encoding="utf-8")
        results["models"][name]["checkpoint"] = ckpt.name
        del train_set, val_dev, test_dev
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    for name, network_columns in TABULAR.items():
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            s, info = fit_xgboost(train_graphs, g_val, g_test, seed, device, network_columns)
            scores.setdefault(name, []).append(s)
            m = measure(g_test, s)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, s, new_mask)})
            print(f"  {name:<56} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"new {runs[-1]['new_senders']['hit@1']:.4f}  ({runs[-1]['seconds']}s)", flush=True)
        results["models"][name] = _summarise(runs)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    np.savez_compressed(scores_path, y=g_test["y"], group_ids=g_test["group_ids"], sender_is_new=new_mask,
                        **{f"{n}|{i}": s for n, ss in scores.items() for i, s in enumerate(ss)})

    print(f"\nPaired bootstrap over questions ({args.bootstrap} resamples) ...", flush=True)
    questions = Questions(g_test["y"], g_test["group_ids"])
    results["bootstrap"] = {}
    for a, b in PAIRS:
        t0 = time.time()
        res = paired_bootstrap(questions, scores[a], scores[b], args.bootstrap, BOOTSTRAP_SEED)
        results["bootstrap"][f"{a} vs {b}"] = res
        print(f"\n  {a}\n    vs {b}   ({time.time() - t0:.0f}s)")
        for k, r in res.items():
            verdict = "significant" if r["significant"] else "not significant"
            print(f"    {k:<7} difference {r['difference']:+.4f}   95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
                  f"   p = {r['p_value']:.3f}   {verdict}", flush=True)

    new_questions = Questions(g_test["y"][new_mask], g_test["group_ids"][new_mask])
    results["bootstrap_new_senders"] = {"questions": len(new_questions.rows)}
    print(f"\n  New senders only ({len(new_questions.rows):,} questions):")
    for a, b in NEW_SENDER_PAIRS:
        res = paired_bootstrap(new_questions, [s[new_mask] for s in scores[a]],
                               [s[new_mask] for s in scores[b]], args.bootstrap, BOOTSTRAP_SEED)
        results["bootstrap_new_senders"][f"{a} vs {b}"] = res
        print(f"\n  {a}\n    vs {b}")
        for k, r in res.items():
            verdict = "significant" if r["significant"] else "not significant"
            print(f"    {k:<7} difference {r['difference']:+.4f}   95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
                  f"   p = {r['p_value']:.3f}   {verdict}", flush=True)

    print("\n" + "=" * 110)
    print(f"  {TEST_YEAR} test (mean over seeds {args.seeds})")
    print("=" * 110)
    for n, r in sorted(results["models"].items(), key=lambda kv: -kv[1]["hit@1_mean"]):
        print(f"  {n:<58} AUC {r['auc_mean']:.4f}  Hit@1 {r['hit@1_mean']:.4f}  NDCG@5 {r['ndcg@5_mean']:.4f}"
              f"  new {r['new_senders']['hit@1']:.4f}")
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
