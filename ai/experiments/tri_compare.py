"""
Is the MC-GNN better than other learned models, or only better than rules?

tri_study showed the MC-GNN beats every ranker that learns nothing. The
question a reviewer asks next is whether a standard model would do as well.
This trains the usual candidates on exactly the MC-GNN's problem:

  * the same EPA TRI graphs -- learn 2022 (history 2019-21), select on 2023,
    report on 2024, which no model sees before the end;
  * the same inputs -- the 17 facts about each facility and the 8 about each
    pair; nothing extra for any model;
  * the same scoring -- AUC, NDCG@5, Recall@5 and Hit@1 over the same
    queries.

Neural models also share one training regime (tri_study.train_control):
optimiser, learning rate, early stopping and best-on-2023 selection. So a gap
between two of them is the architecture, not the tuning.

  MC-GNN            three channels (GCN, GAT, GraphSAGE), attention fusion, HSIC
  GCN / GAT / SAGE  one channel each, same depth, skip and link predictor
  MLP               the same encoder depth with no message passing at all
  XGBoost           gradient-boosted trees on [facility, facility, pair] facts,
                    stopped on 2023
  logistic reg.     the linear model on the same features

Usage:
    python -m ai.experiments.tri_compare
    python -m ai.experiments.tri_compare --seeds 42 7 13 --epochs 500

Writes ai/experiments/results/tri_compare.json.
"""

from __future__ import annotations

import argparse
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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from ai.experiments.tri_loader import DATA_DIR, EDGE_FEATURE_DIM, NODE_FEATURE_DIM, TRIDataset
from ai.experiments.tri_study import (
    HISTORY_START,
    MIN_LOCATED_SHARE,
    TEST_YEAR,
    TRAIN_YEAR,
    VAL_YEAR,
    measure,
    pick_device,
    reference_rankers,
    score,
    train_control,
)
from ai.training.train import PATIENCE

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_compare.json"
METRICS = ("auc", "ndcg@5", "recall@5", "hit@1")

# The MC-GNN's own sizes, so no model wins or loses on capacity.
HIDDEN_DIM, OUT_DIM = 64, 32


def _link_predictor(out_dim: int, edge_dim: int) -> nn.Sequential:
    """MCGNN.link_predictor, layer for layer."""
    return nn.Sequential(
        nn.Linear(out_dim * 2 + edge_dim, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
    )


class _PairScorer(nn.Module):
    """Shared scoring and loss: plain binary cross-entropy, no disparity term."""

    def forward(self, x, edge_index, edge_attr, message_edge_index=None):
        if message_edge_index is None:
            message_edge_index = edge_index
        z = self.encode(x, message_edge_index)
        src, dst = edge_index
        out = self.link_predictor(torch.cat([z[src], z[dst], edge_attr], dim=-1)).squeeze(-1)
        return out, [z]

    def compute_loss(self, pred_scores, targets, channel_embs, hsic_weight=0.0):
        # One channel has nothing to be made independent of.
        return F.binary_cross_entropy(pred_scores, targets)


class SingleChannelGNN(_PairScorer):
    """One MC-GNN channel on its own: two graph layers, a skip, the same predictor."""

    def __init__(self, kind: str, in_features: int, edge_dim: int):
        super().__init__()
        if kind == "gcn":
            self.conv1, self.conv2 = GCNConv(in_features, HIDDEN_DIM), GCNConv(HIDDEN_DIM, OUT_DIM)
        elif kind == "gat":
            self.conv1 = GATConv(in_features, HIDDEN_DIM // 2, heads=2)
            self.conv2 = GATConv(HIDDEN_DIM, OUT_DIM, heads=1)
        elif kind == "sage":
            self.conv1, self.conv2 = SAGEConv(in_features, HIDDEN_DIM), SAGEConv(HIDDEN_DIM, OUT_DIM)
        else:
            raise ValueError(kind)
        self.skip = nn.Linear(in_features, OUT_DIM)
        self.link_predictor = _link_predictor(OUT_DIM, edge_dim)

    def encode(self, x, edge_index):
        z = F.relu(self.conv1(x, edge_index))
        return self.conv2(z, edge_index) + self.skip(x)


class NoGraphMLP(_PairScorer):
    """The same two-layer encoder and skip with the graph layers made ordinary layers."""

    def __init__(self, in_features: int, edge_dim: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(in_features, HIDDEN_DIM), nn.Linear(HIDDEN_DIM, OUT_DIM)
        self.skip = nn.Linear(in_features, OUT_DIM)
        self.link_predictor = _link_predictor(OUT_DIM, edge_dim)

    def encode(self, x, edge_index):
        return self.fc2(F.relu(self.fc1(x))) + self.skip(x)


NEURAL = {
    "MC-GNN (GCN + GAT + SAGE)": (None, True),
    "GCN only": (lambda: SingleChannelGNN("gcn", NODE_FEATURE_DIM, EDGE_FEATURE_DIM), True),
    "GAT only": (lambda: SingleChannelGNN("gat", NODE_FEATURE_DIM, EDGE_FEATURE_DIM), True),
    "GraphSAGE only": (lambda: SingleChannelGNN("sage", NODE_FEATURE_DIM, EDGE_FEATURE_DIM), True),
    "MLP (no graph)": (lambda: NoGraphMLP(NODE_FEATURE_DIM, EDGE_FEATURE_DIM), False),
}


def _tabular(g: dict) -> np.ndarray:
    """What the link predictor sees, before any graph: [sender facts, receiver facts, pair facts]."""
    src, dst = g["edge_index"]
    return np.hstack([g["x"][src], g["x"][dst], g["edge_attr"]]).astype(np.float32)


def run_xgboost(g_train, g_val, g_test, seed: int, device: torch.device) -> tuple[np.ndarray, dict]:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", early_stopping_rounds=PATIENCE, random_state=seed,
        tree_method="hist", device="cuda" if device.type == "cuda" else "cpu",
    )
    model.fit(_tabular(g_train), g_train["y"], eval_set=[(_tabular(g_val), g_val["y"])], verbose=False)
    return model.predict_proba(_tabular(g_test))[:, 1], {"best_iteration": int(model.best_iteration)}


def run_logistic(g_train, g_val, g_test, seed: int, device: torch.device) -> tuple[np.ndarray, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(_tabular(g_train), g_train["y"])
    return model.predict_proba(_tabular(g_test))[:, 1], {}


TABULAR = {"XGBoost": run_xgboost, "Logistic regression": run_logistic}


def _summarise(runs: list[dict]) -> dict:
    return {
        "runs": runs,
        **{f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in METRICS},
        **{f"{k}_sd": float(np.std([r[k] for r in runs])) for k in METRICS},
    }


def report(results: dict) -> None:
    rows = [(n, r) for n, r in results["models"].items()]
    rows.sort(key=lambda nr: -nr[1]["auc_mean"])
    width = 116
    print("\n" + "=" * width)
    print(f"  {TEST_YEAR} EPA TRI recovery transfers, never seen in training or selection "
          f"(mean +/- sd over seeds {results['config']['seeds']})")
    print("=" * width)
    print(f"  {'':<34}" + "".join(f"{k.upper() if k == 'auc' else k:>20}" for k in METRICS))
    print("  " + "-" * (width - 2))
    for name, r in rows:
        print(f"  {name:<34}" + "".join(f"{r[k + '_mean']:>13.4f} +/-{r[k + '_sd']:.3f}" for k in METRICS))
    print("  " + "-" * (width - 2))
    for name, m in results["references"].items():
        print(f"  {name:<34}" + "".join(f"{m[k]:>13.4f}       " for k in METRICS))
    print("=" * width)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--cpu", action="store_true", help="train on the CPU even when a GPU is available")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
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

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "train_year": TRAIN_YEAR, "val_year": VAL_YEAR, "test_year": TEST_YEAR,
            "history_start": HISTORY_START, "seeds": args.seeds, "max_epochs": args.epochs,
            "patience": PATIENCE, "device": str(device),
            "inputs": f"{NODE_FEATURE_DIM} per facility, {EDGE_FEATURE_DIM} per pair, identical for every model",
        },
        "test_queries": int(g_test["stats"].get("queries_used", 0)),
        "models": {},
    }

    print(f"Training on {device} ...", flush=True)
    for name, (factory, with_graph) in NEURAL.items():
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            model, info = train_control(g_train, g_val, seed, with_graph=with_graph, epochs=args.epochs,
                                        device=device, model_factory=factory)
            m = measure(g_test, score(model, g_test, with_graph=with_graph))
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1)})
            print(f"  {name:<28} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"(best epoch {info['best_epoch']}, {runs[-1]['seconds']}s)", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["models"][name] = _summarise(runs)

    for name, fit in TABULAR.items():
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            scores, info = fit(g_train, g_val, g_test, seed, device)
            m = measure(g_test, scores)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1)})
            print(f"  {name:<28} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"({runs[-1]['seconds']}s)", flush=True)
        results["models"][name] = _summarise(runs)

    results["references"] = {
        name: measure(g_test, s) for name, s in reference_rankers(g_test, args.seeds[0]).items()
    }

    report(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
