"""
Does giving each MC-GNN channel its own network make it the better model?

tri_compare found the MC-GNN level with a single GCN and a shade behind
XGBoost, because its three channels read one network three ways. This tests
the design in which each channel reads a different one -- relationship
(GAT), capability (GraphSAGE), peer (GCN); see MultiViewMCGNN and
ai/experiments/tri_views.py -- against the same field as before.

Two training changes apply to every model, so neither can explain a gap:

  * three years to learn from instead of one: 2020, 2021 and 2022, each with
    its own history from 2019; 2023 still selects, 2024 still reports;
  * a fresh draw of "did not happen" examples every epoch, from a pool five
    times the size the test uses, instead of the same nine per real transfer
    seen on every pass. The test keeps its fixed nine, so it is the same
    test as tri_study and tri_compare.

Models that cannot read a graph are offered what the new networks say about
each pair as extra columns (capability_score, peer_votes), so a win for the
multi-view MC-GNN has to come from reading the networks, not from knowing
something the others were not told.

Beyond the usual scores, two things the design is meant to change are
measured directly:

  new facilities   scores on 2024 questions from senders with no recovery
                   transfers in history -- the newly registered company
  attention        how much the multi-view model leans on each network, for
                   new and for established senders

Usage:
    python -m ai.experiments.tri_multiview
    python -m ai.experiments.tri_multiview --seeds 42 7 13 --epochs 500

Writes ai/experiments/results/tri_multiview.json and saves the best
multi-view model (by 2023 validation) under ai/checkpoints/.
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

from ai.evaluation.metrics import roc_auc
from ai.experiments.tri_compare import NoGraphMLP, SingleChannelGNN, _tabular
from ai.experiments.tri_loader import (
    DATA_DIR,
    EDGE_FEATURE_DIM,
    NEGATIVES_PER_POSITIVE,
    NODE_FEATURE_DIM,
    TRIDataset,
)
from ai.experiments.tri_study import (
    HISTORY_START,
    LR,
    MIN_LOCATED_SHARE,
    TEST_YEAR,
    VAL_YEAR,
    WEIGHT_DECAY,
    measure,
    pick_device,
    reference_rankers,
)
from ai.experiments.tri_views import PAIR_GRAPH_FEATURE_NAMES
from ai.models.mc_gnn import HSIC_GAMMA, MCGNN, VIEW_NAMES, MultiViewMCGNN
from ai.training.train import PATIENCE

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_multiview.json"
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"
METRICS = ("auc", "ndcg@5", "recall@5", "hit@1")

TRAIN_YEARS = (2020, 2021, 2022)
# The pool each training query's negatives are redrawn from every epoch.
TRAIN_POOL_PER_POSITIVE = 5 * NEGATIVES_PER_POSITIVE

# graph_mode: which networks a model reads. None reads none.
NEURAL = {
    "Multi-view MC-GNN": (lambda: MultiViewMCGNN(NODE_FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM), "views"),
    "MC-GNN (one network)": (lambda: MCGNN(NODE_FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM), "relationship"),
    "GCN only": (lambda: SingleChannelGNN("gcn", NODE_FEATURE_DIM, EDGE_FEATURE_DIM), "relationship"),
    "GAT only": (lambda: SingleChannelGNN("gat", NODE_FEATURE_DIM, EDGE_FEATURE_DIM), "relationship"),
    "GraphSAGE only": (lambda: SingleChannelGNN("sage", NODE_FEATURE_DIM, EDGE_FEATURE_DIM), "relationship"),
    "MLP (no graph)": (lambda: NoGraphMLP(NODE_FEATURE_DIM, EDGE_FEATURE_DIM), None),
}


# ── graphs on the device ────────────────────────────────────────────────


class OnDevice:
    """One year's graph as tensors, with its message graphs ready for every mode."""

    def __init__(self, g: dict, device: torch.device):
        t = lambda a, **kw: torch.as_tensor(a, **kw).to(device)  # noqa: E731
        self.g = g
        self.x = t(g["x"])
        self.edge_index = t(g["edge_index"])
        self.edge_attr = t(g["edge_attr"])
        self.y = t(g["y"])
        self.messages = {
            None: torch.zeros((2, 0), dtype=torch.long, device=device),
            "relationship": t(g["views"]["relationship"]),
            "views": [t(g["views"][name]) for name in VIEW_NAMES],
        }


class FreshNegatives:
    """
    Every real transfer, plus a new random draw of negatives each epoch.

    Each query keeps NEGATIVES_PER_POSITIVE negatives per real partner, drawn
    from its larger pool. Drawing happens on the device: negatives are held
    sorted by query, a random key reorders them within each query, and the
    first quota of each query is kept.
    """

    def __init__(self, g: dict, device: torch.device, per_positive: int = NEGATIVES_PER_POSITIVE):
        y, groups = g["y"], g["group_ids"]
        self.positives = torch.as_tensor(np.flatnonzero(y > 0.5), device=device)
        neg = np.flatnonzero(y <= 0.5)
        order = np.argsort(groups[neg], kind="stable")
        neg, neg_groups = neg[order], groups[neg][order]
        n_groups = int(groups.max()) + 1 if len(groups) else 0
        first = np.full(n_groups, 0, dtype=np.int64)
        uniq, starts = np.unique(neg_groups, return_index=True)
        first[uniq] = starts
        quota = np.bincount(groups[y > 0.5], minlength=n_groups) * per_positive
        self.neg = torch.as_tensor(neg, device=device)
        self.neg_groups = torch.as_tensor(neg_groups, device=device)
        self.rank_base = torch.as_tensor(first[neg_groups], device=device)
        self.quota = torch.as_tensor(quota[neg_groups], device=device)
        self.position = torch.arange(len(neg), device=device)

    def draw(self, generator: torch.Generator) -> torch.Tensor:
        key = self.neg_groups.double() + torch.rand(
            len(self.neg), generator=generator, device=self.neg.device, dtype=torch.float64)
        shuffled = self.neg[torch.argsort(key)]
        keep = (self.position - self.rank_base) < self.quota
        return torch.cat([self.positives, shuffled[keep]])


# ── training ────────────────────────────────────────────────────────────


@torch.no_grad()
def predict(model, d: OnDevice, mode) -> np.ndarray:
    model.eval()
    out, _ = model(d.x, d.edge_index, d.edge_attr, message_edge_index=d.messages[mode])
    return out.cpu().numpy()


def train_neural(factory, mode, train: list[tuple[OnDevice, FreshNegatives]], val: OnDevice,
                 seed: int, epochs: int, patience: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = {"auc": -1.0, "epoch": 0}
    best_state, stale, epoch = None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        # One step per epoch over all training years, each weighted equally.
        for d, sampler in train:
            idx = sampler.draw(generator)
            scores, channels = model(d.x, d.edge_index[:, idx], d.edge_attr[idx],
                                     message_edge_index=d.messages[mode])
            loss = model.compute_loss(scores, d.y[idx], channels, hsic_weight=HSIC_GAMMA) / len(train)
            loss.backward()
        opt.step()

        val_auc = roc_auc(val.g["y"], predict(model, val, mode))
        if val_auc > best["auc"] + 1e-9:
            best = {"auc": val_auc, "epoch": epoch}
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best["epoch"], "val_auc": best["auc"], "epochs_run": epoch}


# ── models that cannot read a graph ─────────────────────────────────────


def _columns(g: dict, network_features: bool) -> np.ndarray:
    base = _tabular(g)
    return np.hstack([base, g["pair_graph_features"]]) if network_features else base


def fit_xgboost(train_graphs, g_val, g_test, seed, device, network_features):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", early_stopping_rounds=PATIENCE, random_state=seed,
        tree_method="hist", device="cuda" if device.type == "cuda" else "cpu",
    )
    X = np.vstack([_columns(g, network_features) for g in train_graphs])
    y = np.concatenate([g["y"] for g in train_graphs])
    model.fit(X, y, eval_set=[(_columns(g_val, network_features), g_val["y"])], verbose=False)
    return model.predict_proba(_columns(g_test, network_features))[:, 1], {"best_iteration": int(model.best_iteration)}


def fit_logistic(train_graphs, g_val, g_test, seed, device, network_features):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
    X = np.vstack([_columns(g, network_features) for g in train_graphs])
    y = np.concatenate([g["y"] for g in train_graphs])
    model.fit(X, y)
    return model.predict_proba(_columns(g_test, network_features))[:, 1], {}


TABULAR = {
    "XGBoost": (fit_xgboost, False),
    "XGBoost + network columns": (fit_xgboost, True),
    "Logistic regression + network columns": (fit_logistic, True),
}


# ── measuring ───────────────────────────────────────────────────────────


def measure_subset(g: dict, scores: np.ndarray, mask: np.ndarray) -> dict | None:
    if not mask.any():
        return None
    sub = {"y": g["y"][mask], "group_ids": g["group_ids"][mask]}
    return measure(sub, scores[mask])


def attention_by_sender(model: MultiViewMCGNN, d: OnDevice) -> dict:
    """Mean weight on each network, over the senders of new and of established facilities' questions."""
    w = model.channel_weights(d.x, d.messages["views"]).cpu().numpy()
    src = d.g["edge_index"][0]
    new = d.g["sender_is_new"]
    out = {}
    for label, mask in (("new senders", new), ("established senders", ~new)):
        senders = np.unique(src[mask])
        if len(senders):
            out[label] = {name: round(float(w[senders, i].mean()), 4) for i, name in enumerate(VIEW_NAMES)}
            out[label]["facilities"] = int(len(senders))
    return out


def _summarise(runs: list[dict]) -> dict:
    def stats(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    out = {"runs": runs}
    for k in METRICS:
        out[f"{k}_mean"], out[f"{k}_sd"] = stats(k)
    new_runs = [r["new_senders"] for r in runs if r.get("new_senders")]
    if new_runs:
        out["new_senders"] = {k: float(np.mean([r[k] for r in new_runs])) for k in METRICS}
        out["new_senders"]["queries"] = new_runs[0]["queries"]
    return out


def report(results: dict) -> None:
    rows = sorted(results["models"].items(), key=lambda nr: -nr[1]["auc_mean"])
    width = 128
    print("\n" + "=" * width)
    print(f"  {TEST_YEAR} EPA TRI recovery transfers, never seen (mean +/- sd over seeds "
          f"{results['config']['seeds']}); trained on {list(TRAIN_YEARS)}")
    print("=" * width)
    print(f"  {'':<40}" + "".join(f"{k:>18}" for k in METRICS) + f"{'new senders hit@1':>20}")
    print("  " + "-" * (width - 2))
    for name, r in rows:
        new = r.get("new_senders", {}).get("hit@1")
        print(f"  {name:<40}" + "".join(f"{r[k + '_mean']:>11.4f} +/-{r[k + '_sd']:.3f}" for k in METRICS)
              + (f"{new:>20.4f}" if new is not None else ""))
    print("  " + "-" * (width - 2))
    for name, m in results["references"].items():
        new = results["references_new_senders"].get(name, {}).get("hit@1")
        print(f"  {name:<40}" + "".join(f"{m[k]:>11.4f}       " for k in METRICS)
              + (f"{new:>20.4f}" if new is not None else ""))
    print("=" * width)
    if results.get("attention"):
        print("\n  Where the multi-view MC-GNN looks (mean attention, best model):")
        for label, w in results["attention"].items():
            print(f"    {label:<22}" + "  ".join(f"{v}: {w[v]:.3f}" for v in VIEW_NAMES)
                  + f"   ({w['facilities']:,} facilities)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--models", nargs="+", help="run only these (names as in the table)")
    ap.add_argument("--cpu", action="store_true", help="train on the CPU even when a GPU is available")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    device = pick_device(args.cpu)
    wanted = lambda name: not args.models or name in args.models  # noqa: E731

    years = list(range(HISTORY_START, TEST_YEAR + 1))
    print(f"Reading EPA TRI {years[0]}-{years[-1]} ...", flush=True)
    ds, coverage = TRIDataset.from_files(years, args.data_dir)
    share = coverage["per_year"][TEST_YEAR]["located_share"]
    if share < MIN_LOCATED_SHARE:
        sys.exit(f"STOP: only {share:.1%} of {TEST_YEAR} transfers located (need {MIN_LOCATED_SHARE:.0%}).")

    print("Building graphs and their three networks ...", flush=True)
    train_graphs = []
    for y in TRAIN_YEARS:
        t0 = time.time()
        train_graphs.append(ds.build_graph(y, history_start=HISTORY_START, views=True,
                                           negatives_per_positive=TRAIN_POOL_PER_POSITIVE))
        print(f"  {y}: {len(train_graphs[-1]['y']):,} rows ({time.time() - t0:.0f}s)", flush=True)
    g_val = ds.build_graph(VAL_YEAR, history_start=HISTORY_START, views=True)
    g_test = ds.build_graph(TEST_YEAR, history_start=HISTORY_START, views=True)
    for g in (*train_graphs, g_val, g_test):
        print(f"  {g['target_year']} networks: " + ", ".join(
            f"{n} {s['edges']:,} edges (isolated {s['isolated_share']:.0%})"
            for n, s in g["stats"]["views"].items()), flush=True)

    new_mask = g_test["sender_is_new"]
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "train_years": list(TRAIN_YEARS), "val_year": VAL_YEAR, "test_year": TEST_YEAR,
            "history_start": HISTORY_START, "seeds": args.seeds, "max_epochs": args.epochs,
            "patience": PATIENCE, "device": str(device),
            "negatives_per_positive_per_epoch": NEGATIVES_PER_POSITIVE,
            "training_pool_per_positive": TRAIN_POOL_PER_POSITIVE,
            "views": list(VIEW_NAMES), "pair_network_columns": list(PAIR_GRAPH_FEATURE_NAMES),
        },
        "graphs": {str(g["target_year"]): g["stats"] for g in (*train_graphs, g_val, g_test)},
        "test_queries": int(g_test["stats"].get("queries_used", 0)),
        "test_new_sender_queries": int(len(np.unique(g_test["group_ids"][new_mask]))),
        "models": {},
    }

    print(f"\nTraining on {device} ...", flush=True)
    train_dev = [(OnDevice(g, device), FreshNegatives(g, device)) for g in train_graphs]
    val_dev, test_dev = OnDevice(g_val, device), OnDevice(g_test, device)
    best_multiview = None

    for name, (factory, mode) in NEURAL.items():
        if not wanted(name):
            continue
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            model, info = train_neural(factory, mode, train_dev, val_dev, seed, args.epochs, PATIENCE, device)
            scores = predict(model, test_dev, mode)
            m = measure(g_test, scores)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, scores, new_mask)})
            print(f"  {name:<38} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"(best epoch {info['best_epoch']}, {runs[-1]['seconds']}s)", flush=True)
            if mode == "views" and (best_multiview is None or info["val_auc"] > best_multiview[0]):
                best_multiview = (info["val_auc"], seed, model, info)
            else:
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["models"][name] = _summarise(runs)

    for name, (fit, network_features) in TABULAR.items():
        if not wanted(name):
            continue
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            scores, info = fit(train_graphs, g_val, g_test, seed, device, network_features)
            m = measure(g_test, scores)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, scores, new_mask)})
            print(f"  {name:<38} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"({runs[-1]['seconds']}s)", flush=True)
        results["models"][name] = _summarise(runs)

    refs = reference_rankers(g_test, args.seeds[0])
    results["references"] = {name: measure(g_test, s) for name, s in refs.items()}
    results["references_new_senders"] = {
        name: measure_subset(g_test, s, new_mask) for name, s in refs.items()
    }

    if best_multiview is not None:
        val_auc, seed, model, info = best_multiview
        results["attention"] = attention_by_sender(model, test_dev)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = CHECKPOINT_DIR / f"mc_gnn_mv_{stamp}.pt"
        torch.save(model.cpu().eval(), ckpt)
        ckpt.with_suffix(".json").write_text(json.dumps({
            "checkpoint_file": ckpt.name,
            "model": "MultiViewMCGNN",
            "views": list(VIEW_NAMES),
            "needs": "message_edge_index as three edge indices (relationship, capability, peer) "
                     "built by ai/experiments/tri_views.py; not servable by the platform as it stands",
            "trained_on": f"EPA TRI, learn {list(TRAIN_YEARS)}, select {VAL_YEAR}",
            "seed": seed, **info,
            "us_test_metrics": results["models"]["Multi-view MC-GNN"]["runs"][
                args.seeds.index(seed)],
        }, indent=2, default=float), encoding="utf-8")
        results["saved_checkpoint"] = ckpt.name

    report(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
