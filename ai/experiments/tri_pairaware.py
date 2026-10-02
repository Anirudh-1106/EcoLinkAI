"""
The pair-aware MC-GNN with normalised HSIC, against its components and XGBoost.

Two weaknesses were measured in the multi-view MC-GNN:

  * Its scorer concatenates two facility embeddings, so it cannot represent
    what the two share. On TRI, facilities that share a partner with the
    supplier and already trade with the recycler separate real partners at
    AUC 0.890 on their count alone -- information XGBoost receives as a
    hand-built column and the MC-GNN could not reach.
  * Its HSIC term never acted: at gamma = 1e-10 the raw term was about 1e-11
    against a cross-entropy near 0.07 (ai/experiments/results/tri_channels.json).

PairAwareMCGNN (ai/models/mc_gnn.py) keeps the encoder -- three channels on
three networks, attention fusion, channel dropout -- and replaces the scorer
with Neural Common Neighbor pooling over the pair sets of tri_pairs.py, and
the disparity term with normalised HSIC.

Protocol, as in every study here: learn 2020-2022, select on 2023, test 2024
once. Stage 1 tunes the HSIC weight and learning rate on 2023 validation AUC,
one seed. Stage 2 trains the chosen configuration on three seeds beside:

  * GCN, GAT and GraphSAGE alone, each with the same pair-aware scorer, so a
    difference is the multi-channel encoder and not the scorer;
  * XGBoost with the network columns, and with the network columns plus the
    two pair-set sizes, so XGBoost is offered the same structural information
    the scorer reads.

A paired bootstrap over 2024 questions then tests each difference.

Usage:
    python -m ai.experiments.tri_pairaware

Writes ai/experiments/results/tri_pairaware.json; per-row test scores go to
<data dir>/results/.
"""

from __future__ import annotations

import argparse
import copy
import gc
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
from ai.experiments.tri_compare import SingleChannelGNN
from ai.experiments.tri_improve import PAIR_INPUTS, RANKING_WEIGHT, listwise_loss, with_network_columns
from ai.experiments.tri_loader import DATA_DIR, NODE_FEATURE_DIM, TRIDataset
from ai.experiments.tri_multiview import (
    TRAIN_POOL_PER_POSITIVE,
    TRAIN_YEARS,
    FreshNegatives,
    OnDevice,
    _columns,
    _summarise,
    attention_by_sender,
    measure_subset,
)
from ai.experiments.tri_pairs import PAIR_SET_NAMES, pair_neighbourhoods
from ai.experiments.tri_significance import BOOTSTRAP_SEED, Questions, paired_bootstrap
from ai.experiments.tri_study import HISTORY_START, MIN_LOCATED_SHARE, TEST_YEAR, VAL_YEAR, WEIGHT_DECAY, measure, pick_device
from ai.models.mc_gnn import PairAwareMCGNN, PairPooling
from ai.training.train import PATIENCE

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_pairaware.json"
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"

# Searched on 2023 validation AUC only. 0 is the ablation without HSIC.
HSIC_WEIGHTS = (0.0, 0.05, 0.2, 1.0)
LEARNING_RATES = (0.01, 0.003)
TUNING_SEED = 42
NUM_COUNTS = 2
MCGNN = "Pair-aware MC-GNN"


class SingleChannelPairGNN(SingleChannelGNN):
    """One MC-GNN channel with the same pair-aware scorer, for a like-for-like comparison."""

    def __init__(self, kind: str, edge_dim: int):
        super().__init__(kind, NODE_FEATURE_DIM, edge_dim)
        out_dim = self.skip.out_features
        self.link_predictor = PairPooling(out_dim, edge_dim, len(PAIR_SET_NAMES), NUM_COUNTS)

    def forward(self, x, edge_index, edge_attr, message_edge_index=None, pair_sets=(), pair_counts=None):
        z = self.encode(x, message_edge_index)
        return self.link_predictor(z, edge_index, edge_attr, pair_sets, pair_counts), [z]


# ── data ────────────────────────────────────────────────────────────────


class PairGraph:
    """A year's graph on the device, with its pair sets."""

    def __init__(self, g: dict, device, keep_sets_on_device: bool):
        self.d = with_network_columns(OnDevice(g, device))
        pairs = pair_neighbourhoods(g)
        self.g = g
        self.device = device
        target = device if keep_sets_on_device else "cpu"
        self.sets = [torch.as_tensor(pairs[n]).to(target) for n in PAIR_SET_NAMES]
        self.counts = torch.as_tensor(pairs["counts"]).to(target)

    def batch(self, idx: torch.Tensor | None):
        """Pair sets and counts for rows idx (all rows when None), on the device."""
        if idx is None:
            return [s.to(self.device) for s in self.sets], self.counts.to(self.device)
        cpu_idx = idx.cpu() if self.sets[0].device.type == "cpu" else idx
        return ([s[cpu_idx].to(self.device) for s in self.sets], self.counts[cpu_idx].to(self.device))


@torch.no_grad()
def predict(model, pg: PairGraph, mode: str) -> np.ndarray:
    model.eval()
    sets, counts = pg.batch(None)
    d = pg.d
    out, _ = model(d.x, d.edge_index, d.edge_attr, message_edge_index=d.messages[mode],
                   pair_sets=sets, pair_counts=counts)
    return out.cpu().numpy()


def train(factory, mode, train_set, val: PairGraph, seed: int, epochs: int, lr: float, device,
          ranking_weight: float = RANKING_WEIGHT):
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best = {"auc": -1.0, "epoch": 0}
    best_state, stale, epoch = None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        for pg, sampler in train_set:
            idx = sampler.draw(generator)
            sets, counts = pg.batch(idx)
            d = pg.d
            probs, channels = model(d.x, d.edge_index[:, idx], d.edge_attr[idx],
                                    message_edge_index=d.messages[mode], pair_sets=sets, pair_counts=counts)
            loss = model.compute_loss(probs, d.y[idx], channels)
            loss = loss + ranking_weight * listwise_loss(probs, d.y[idx], d.groups[idx])
            (loss / len(train_set)).backward()
        opt.step()

        val_auc = roc_auc(val.g["y"], predict(model, val, mode))
        if val_auc > best["auc"] + 1e-9:
            best = {"auc": val_auc, "epoch": epoch}
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best["epoch"], "val_auc": best["auc"], "epochs_run": epoch}


def fit_xgboost(train_graphs, g_val, g_test, pair_counts, seed, device, with_pair_counts: bool):
    import xgboost as xgb

    def cols(g, counts):
        base = _columns(g, True)
        return np.hstack([base, counts]) if with_pair_counts else base

    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", early_stopping_rounds=PATIENCE, random_state=seed,
        tree_method="hist", device="cuda" if device.type == "cuda" else "cpu",
    )
    X = np.vstack([cols(g, c) for g, c in zip(train_graphs, pair_counts["train"])])
    y = np.concatenate([g["y"] for g in train_graphs])
    model.fit(X, y, eval_set=[(cols(g_val, pair_counts["val"]), g_val["y"])], verbose=False)
    return model.predict_proba(cols(g_test, pair_counts["test"]))[:, 1], {"best_iteration": int(model.best_iteration)}


# ── run ─────────────────────────────────────────────────────────────────


def save(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--resume", action="store_true",
                    help="reuse tuning runs already saved in the results file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    device = pick_device()

    years = list(range(HISTORY_START, TEST_YEAR + 1))
    print(f"Reading EPA TRI {years[0]}-{years[-1]} ...", flush=True)
    ds, coverage = TRIDataset.from_files(years, args.data_dir)
    if coverage["per_year"][TEST_YEAR]["located_share"] < MIN_LOCATED_SHARE:
        sys.exit("STOP: too few test-year transfers located.")

    print("Building graphs and pair sets ...", flush=True)
    train_graphs = [
        ds.build_graph(y, history_start=HISTORY_START, views=True, negatives_per_positive=TRAIN_POOL_PER_POSITIVE)
        for y in TRAIN_YEARS
    ]
    g_val = ds.build_graph(VAL_YEAR, history_start=HISTORY_START, views=True)
    g_test = ds.build_graph(TEST_YEAR, history_start=HISTORY_START, views=True)
    train_set = [(PairGraph(g, device, keep_sets_on_device=False), FreshNegatives(g, device)) for g in train_graphs]
    val_pg = PairGraph(g_val, device, keep_sets_on_device=True)
    test_pg = PairGraph(g_test, device, keep_sets_on_device=True)
    new_mask = g_test["sender_is_new"]
    print("  ready", flush=True)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "train_years": list(TRAIN_YEARS), "val_year": VAL_YEAR, "test_year": TEST_YEAR,
            "seeds": args.seeds, "max_epochs": args.epochs, "patience": PATIENCE,
            "ranking_weight": RANKING_WEIGHT, "pair_sets": list(PAIR_SET_NAMES),
            "hsic": "normalised (linear CKA)", "hsic_grid": list(HSIC_WEIGHTS), "lr_grid": list(LEARNING_RATES),
            "device": str(device),
        },
        "tuning": [],
        "models": {},
    }
    if args.resume and RESULTS_PATH.exists():
        results["tuning"] = json.loads(RESULTS_PATH.read_text(encoding="utf-8")).get("tuning", [])
        print(f"  resuming with {len(results['tuning'])} tuning runs already done", flush=True)

    # ── Stage 1: HSIC weight, then learning rate, on 2023 only ───────────
    print("\nStage 1: tuning on 2023 validation AUC ...", flush=True)

    def tune(hsic_weight, lr):
        for row in results["tuning"]:
            if row["hsic_weight"] == hsic_weight and row["lr"] == lr:
                print(f"  HSIC weight {hsic_weight:<5} lr {lr:<6} val AUC {row['val_auc']:.4f} (saved)", flush=True)
                return row["val_auc"]
        t0 = time.time()
        model, info = train(lambda: PairAwareMCGNN(NODE_FEATURE_DIM, edge_dim=PAIR_INPUTS, hsic_weight=hsic_weight),
                            "views", train_set, val_pg, TUNING_SEED, args.epochs, lr, device)
        row = {"hsic_weight": hsic_weight, "lr": lr, **info, "seconds": round(time.time() - t0, 1)}
        results["tuning"].append(row)
        print(f"  HSIC weight {hsic_weight:<5} lr {lr:<6} val AUC {info['val_auc']:.4f} "
              f"(best epoch {info['best_epoch']} of {info['epochs_run']}, {row['seconds']}s)", flush=True)
        save(results)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        return info["val_auc"]

    hsic_scores = {w: tune(w, LEARNING_RATES[0]) for w in HSIC_WEIGHTS}
    best_hsic = max(hsic_scores, key=hsic_scores.get)
    lr_scores = {LEARNING_RATES[0]: hsic_scores[best_hsic]}
    for lr in LEARNING_RATES[1:]:
        lr_scores[lr] = tune(best_hsic, lr)
    best_lr = max(lr_scores, key=lr_scores.get)
    results["chosen"] = {"hsic_weight": best_hsic, "lr": best_lr}
    print(f"  chosen: HSIC weight {best_hsic}, lr {best_lr}", flush=True)

    # ── Stage 2: three seeds, against single channels and XGBoost ────────
    print("\nStage 2: final comparison ...", flush=True)
    neural = {
        MCGNN: (lambda: PairAwareMCGNN(NODE_FEATURE_DIM, edge_dim=PAIR_INPUTS, hsic_weight=best_hsic), "views"),
        "Pair-aware GCN": (lambda: SingleChannelPairGNN("gcn", PAIR_INPUTS), "relationship"),
        "Pair-aware GAT": (lambda: SingleChannelPairGNN("gat", PAIR_INPUTS), "relationship"),
        "Pair-aware GraphSAGE": (lambda: SingleChannelPairGNN("sage", PAIR_INPUTS), "relationship"),
    }
    scores: dict[str, list[np.ndarray]] = {}
    best_mcgnn = None
    for name, (factory, mode) in neural.items():
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            model, info = train(factory, mode, train_set, val_pg, seed, args.epochs, best_lr, device)
            s = predict(model, test_pg, mode)
            scores.setdefault(name, []).append(s)
            m = measure(g_test, s)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, s, new_mask)})
            print(f"  {name:<24} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"NDCG@5 {m['ndcg@5']:.4f}  new {runs[-1]['new_senders']['hit@1']:.4f}  "
                  f"(best epoch {info['best_epoch']}, {runs[-1]['seconds']}s)", flush=True)
            if name == MCGNN:
                if seed == args.seeds[0]:
                    results["attention"] = attention_by_sender(model, test_pg.d)
                if best_mcgnn is None or info["val_auc"] > best_mcgnn[0]:
                    best_mcgnn = (info["val_auc"], seed, copy.deepcopy(model).cpu().eval(), info)
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        results["models"][name] = _summarise(runs)
        save(results)

    pair_counts = {
        "train": [pg.counts.cpu().numpy() for pg, _ in train_set],
        "val": val_pg.counts.cpu().numpy(),
        "test": test_pg.counts.cpu().numpy(),
    }
    del train_set
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    for name, with_counts in (("XGBoost + network columns", False),
                              ("XGBoost + network columns + pair counts", True)):
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            s, info = fit_xgboost(train_graphs, g_val, g_test, pair_counts, seed, device, with_counts)
            scores.setdefault(name, []).append(s)
            m = measure(g_test, s)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, s, new_mask)})
            print(f"  {name:<40} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"new {runs[-1]['new_senders']['hit@1']:.4f}  ({runs[-1]['seconds']}s)", flush=True)
        results["models"][name] = _summarise(runs)
        save(results)

    scores_dir = args.data_dir / "results"
    scores_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_dir / "tri_pairaware_scores.npz", y=g_test["y"], group_ids=g_test["group_ids"],
                        sender_is_new=new_mask, **{f"{n}|{i}": s for n, ss in scores.items() for i, s in enumerate(ss)})

    _, seed, model, info = best_mcgnn
    ckpt = CHECKPOINT_DIR / f"mc_gnn_pairaware_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(model, ckpt)
    ckpt.with_suffix(".json").write_text(json.dumps({
        "checkpoint_file": ckpt.name, "model": "PairAwareMCGNN", "seed": seed, **info, **results["chosen"],
        "needs": "three message graphs (tri_views), pair sets and counts (tri_pairs), and pair inputs with "
                 "capability_score, peer_votes; not servable by the platform as it stands",
    }, indent=2, default=float), encoding="utf-8")
    results["saved_checkpoint"] = ckpt.name

    print(f"\nPaired bootstrap ({args.bootstrap} resamples) ...", flush=True)
    others = [n for n in scores if n != MCGNN]
    results["bootstrap"], results["bootstrap_new_senders"] = {}, {}
    all_q = Questions(g_test["y"], g_test["group_ids"])
    new_q = Questions(g_test["y"][new_mask], g_test["group_ids"][new_mask])
    for other in others:
        key = f"{MCGNN} vs {other}"
        results["bootstrap"][key] = paired_bootstrap(all_q, scores[MCGNN], scores[other], args.bootstrap, BOOTSTRAP_SEED)
        results["bootstrap_new_senders"][key] = paired_bootstrap(
            new_q, [s[new_mask] for s in scores[MCGNN]], [s[new_mask] for s in scores[other]],
            args.bootstrap, BOOTSTRAP_SEED)
        for label, res in (("all", results["bootstrap"][key]), ("new", results["bootstrap_new_senders"][key])):
            print(f"  vs {other:<40} [{label}] " + "  ".join(
                f"{k} {r['difference']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]{'*' if r['significant'] else ''}"
                for k, r in res.items()), flush=True)

    print("\n" + "=" * 112)
    for n, r in sorted(results["models"].items(), key=lambda kv: -kv[1]["hit@1_mean"]):
        print(f"  {n:<42} AUC {r['auc_mean']:.4f}  NDCG@5 {r['ndcg@5_mean']:.4f}  Recall@5 {r['recall@5_mean']:.4f}"
              f"  Hit@1 {r['hit@1_mean']:.4f}  new {r['new_senders']['hit@1']:.4f}")
    print("=" * 112)
    save(results)
    print(f"Saved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
