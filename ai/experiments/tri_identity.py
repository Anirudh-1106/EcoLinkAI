"""
Two changes aimed at what a graph network can do and a tree ensemble cannot.

After tri_improve the MC-GNN matched XGBoost on AUC and led it on facilities
with no history, but trailed XGBoost given the same network columns by about
half a point of Hit@1. Two things may close that, both of them advantages of
the architecture rather than new information:

  identity         every facility seen in training gets a learned vector of
                   its own, carried across years -- a recycler's particular
                   preferences, beyond what 17 generic inputs describe.
                   Facilities never seen share one "unknown" vector, and in
                   training a facility's identity is hidden at random
                   (IDENTITY_DROPOUT) so that vector learns what a newcomer
                   looks like instead of being a leftover.

  channel dropout  the multi-view MC-GNN settled 94% of its attention on the
                   capability network for nearly every facility. Dropping one
                   channel per facility at random in training
                   (AttentionFusion.channel_dropout) makes the other two
                   carry weight, so all three views are learned.

Everything else is tri_improve's regime -- three training years, fresh
negatives, network columns among the pair inputs, the ranking term, up to
1000 epochs, selection on 2023 AUC, one look at 2024.

XGBoost is offered the facility identities too, as categorical columns, so
any gain is the architecture's use of identity and not identity itself.

Usage:
    python -m ai.experiments.tri_identity
    python -m ai.experiments.tri_identity --models "Improved MC-GNN + identity"

Writes ai/experiments/results/tri_identity.json.
"""

from __future__ import annotations

import argparse
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
import torch.nn as nn

from ai.experiments.tri_improve import PAIR_INPUTS, RANKING_WEIGHT, train, with_network_columns
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
    predict,
)
from ai.experiments.tri_study import (
    HISTORY_START,
    MIN_LOCATED_SHARE,
    TEST_YEAR,
    VAL_YEAR,
    measure,
    pick_device,
)
from ai.models.mc_gnn import MCGNN, MultiViewMCGNN
from ai.training.train import PATIENCE

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_identity.json"
EARLIER = {
    "tri_improve": PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_improve.json",
    "tri_multiview": PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_multiview.json",
}
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"
METRICS = ("auc", "ndcg@5", "recall@5", "hit@1")

IDENTITY_DIM = 16
# Share of facilities whose identity is hidden in each training pass: what a
# newcomer looks like has to be learned from somewhere.
IDENTITY_DROPOUT = 0.15
CHANNEL_DROPOUT = 0.3
UNKNOWN = 0


# ── identity ────────────────────────────────────────────────────────────


class _IdentityMixin:
    """
    Reads a facility identity from the last column of x and replaces it with
    a learned vector. Identity 0 is every facility not seen in training.
    """

    def __init__(self, num_identities: int, **kwargs):
        super().__init__(in_features=NODE_FEATURE_DIM + IDENTITY_DIM, **kwargs)
        self.identity = nn.Embedding(num_identities + 1, IDENTITY_DIM)
        self.identity_dropout = IDENTITY_DROPOUT

    def encode(self, x, message_edge_index):
        ids = x[:, -1].long()
        if self.training and self.identity_dropout > 0:
            hide = torch.rand(ids.shape, device=ids.device) < self.identity_dropout
            ids = torch.where(hide, torch.full_like(ids, UNKNOWN), ids)
        return super().encode(torch.cat([x[:, :-1], self.identity(ids)], dim=1), message_edge_index)


class IdentityMCGNN(_IdentityMixin, MCGNN):
    """The improved MC-GNN with a learned vector per facility."""


class IdentityMultiViewMCGNN(_IdentityMixin, MultiViewMCGNN):
    """The multi-view MC-GNN with a learned vector per facility."""


def identity_vocabulary(train_graphs: list[dict]) -> dict[str, int]:
    """Every facility in the training years, numbered from 1; 0 is reserved for unknown."""
    keys = sorted({k for g in train_graphs for k in g["node_keys"]})
    return {k: i + 1 for i, k in enumerate(keys)}


def attach_identity(d: OnDevice, vocab: dict[str, int]) -> OnDevice:
    ids = torch.tensor([vocab.get(k, UNKNOWN) for k in d.g["node_keys"]],
                       dtype=torch.float32, device=d.x.device)
    d.x_plain = d.x
    d.x_with_identity = torch.cat([d.x, ids.unsqueeze(1)], dim=1)
    return d


def use_identity(d: OnDevice, on: bool) -> None:
    """Hand a model the inputs it was built for: with or without the identity column."""
    d.x = d.x_with_identity if on else d.x_plain


# ── models ──────────────────────────────────────────────────────────────


def _multiview(channel_dropout: float, identity: int | None = None):
    if identity is None:
        model = MultiViewMCGNN(NODE_FEATURE_DIM, edge_dim=PAIR_INPUTS)
    else:
        model = IdentityMultiViewMCGNN(identity, edge_dim=PAIR_INPUTS)
    model.fusion.channel_dropout = channel_dropout
    return model


def model_table(num_identities: int) -> dict:
    """name -> (factory, message graph, uses identity)."""
    return {
        "Improved MC-GNN + identity": (
            lambda: IdentityMCGNN(num_identities, edge_dim=PAIR_INPUTS), "relationship", True),
        "Multi-view MC-GNN + channel dropout": (
            lambda: _multiview(CHANNEL_DROPOUT), "views", False),
        "Multi-view MC-GNN + channel dropout + identity": (
            lambda: _multiview(CHANNEL_DROPOUT, num_identities), "views", True),
    }


# ── XGBoost with the same identities ────────────────────────────────────


def _identity_frame(g: dict, vocab: dict[str, int], network_columns: np.ndarray):
    import pandas as pd

    src, dst = g["edge_index"]
    # Identity per node once, then indexed per row: integer codes throughout,
    # never a Python object per row (1.8M rows per training year).
    node_ids = np.fromiter((vocab.get(k, UNKNOWN) for k in g["node_keys"]), dtype=np.int32)
    frame = pd.DataFrame(network_columns, columns=[f"f{i}" for i in range(network_columns.shape[1])])
    categories = np.arange(len(vocab) + 1)
    for name, idx in (("sender_id", src), ("receiver_id", dst)):
        frame[name] = pd.Categorical.from_codes(node_ids[idx], categories=categories)
    return frame


def fit_xgboost_identity(train_graphs, g_val, g_test, vocab, seed, device):
    import pandas as pd
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", early_stopping_rounds=PATIENCE, random_state=seed,
        tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
        # CPU even with a GPU present: partitioning splits over ~15,000
        # facility categories on 5.4M rows exceeds a 4 GB card.
        device="cpu",
    )
    X = pd.concat([_identity_frame(g, vocab, _columns(g, True)) for g in train_graphs], ignore_index=True)
    y = np.concatenate([g["y"] for g in train_graphs])
    X_val = _identity_frame(g_val, vocab, _columns(g_val, True))
    model.fit(X, y, eval_set=[(X_val, g_val["y"])], verbose=False)
    X_test = _identity_frame(g_test, vocab, _columns(g_test, True))
    return model.predict_proba(X_test)[:, 1], {"best_iteration": int(model.best_iteration)}


# ── run ─────────────────────────────────────────────────────────────────


def save(results: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--models", nargs="+", help="run only these")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    device = pick_device(args.cpu)
    wanted = lambda n: not args.models or n in args.models  # noqa: E731

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

    vocab = identity_vocabulary(train_graphs)
    test_keys = g_test["node_keys"]
    known_share = float(np.mean([k in vocab for k in test_keys]))
    print(f"  {len(vocab):,} facilities with a learned identity; "
          f"{known_share:.1%} of 2024 facilities were seen in training", flush=True)

    def prepare(g):
        return attach_identity(with_network_columns(OnDevice(g, device)), vocab)

    neural = model_table(len(vocab))
    train_set, val_dev, test_dev = [], None, None
    if any(wanted(n) for n in neural):
        train_set = [(prepare(g), FreshNegatives(g, device)) for g in train_graphs]
        val_dev, test_dev = prepare(g_val), prepare(g_test)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "train_years": list(TRAIN_YEARS), "val_year": VAL_YEAR, "test_year": TEST_YEAR,
            "seeds": args.seeds, "max_epochs": args.epochs, "patience": PATIENCE,
            "pair_inputs": PAIR_INPUTS, "ranking_weight": RANKING_WEIGHT,
            "identity_dim": IDENTITY_DIM, "identity_dropout": IDENTITY_DROPOUT,
            "channel_dropout": CHANNEL_DROPOUT, "identities": len(vocab),
            "test_facilities_seen_in_training": round(known_share, 4), "device": str(device),
        },
        "models": {},
        "attention": {},
    }
    # A partial rerun (--models) adds to the saved results instead of
    # replacing them, so models already measured need not be trained again.
    if args.models and RESULTS_PATH.exists():
        previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        results["models"] = {k: v for k, v in previous.get("models", {}).items() if not wanted(k)}
        results["attention"] = {k: v for k, v in previous.get("attention", {}).items() if not wanted(k)}

    print(f"Training on {device} ...", flush=True)
    best = None
    for name, (factory, mode, uses_identity) in neural.items():
        if not wanted(name):
            continue
        runs = []
        for seed in args.seeds:
            for d in [val_dev, test_dev] + [d for d, _ in train_set]:
                use_identity(d, uses_identity)
            t0 = time.time()
            model, info = train(RANKING_WEIGHT, train_set, val_dev, seed, args.epochs, device,
                                model_factory=factory, mode=mode)
            scores = predict(model, test_dev, mode)
            m = measure(g_test, scores)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, scores, new_mask)})
            print(f"  {name:<48} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"NDCG@5 {m['ndcg@5']:.4f}  new Hit@1 {runs[-1]['new_senders']['hit@1']:.4f}  "
                  f"(best epoch {info['best_epoch']} of {info['epochs_run']}, {runs[-1]['seconds']}s)",
                  flush=True)
            if mode == "views" and seed == args.seeds[0]:
                results["attention"][name] = attention_by_sender(model, test_dev)
            if best is None or info["val_auc"] > best[0]:
                best = (info["val_auc"], name, seed, model, info)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["models"][name] = _summarise(runs)
        save(results)

    # The neural models' tensors are no longer needed; free them for XGBoost.
    del train_set, val_dev, test_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    name = "XGBoost + network columns + identity"
    if wanted(name):
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            scores, info = fit_xgboost_identity(train_graphs, g_val, g_test, vocab, seed, device)
            m = measure(g_test, scores)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, scores, new_mask)})
            print(f"  {name:<48} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"({runs[-1]['seconds']}s)", flush=True)
        results["models"][name] = _summarise(runs)
        save(results)

    earlier = {}
    improve = json.loads(EARLIER["tri_improve"].read_text(encoding="utf-8"))["models"]
    multiview = json.loads(EARLIER["tri_multiview"].read_text(encoding="utf-8"))["models"]
    earlier["Improved MC-GNN"] = improve["Improved MC-GNN"]
    for k in ("XGBoost + network columns", "XGBoost", "Multi-view MC-GNN"):
        earlier[k] = multiview[k]
    results["earlier_same_setup"] = earlier

    if best is not None:
        val_auc, bname, seed, model, info = best
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = CHECKPOINT_DIR / f"mc_gnn_identity_{stamp}.pt"
        torch.save(model.cpu().eval(), ckpt)
        ckpt.with_suffix(".json").write_text(json.dumps({
            "checkpoint_file": ckpt.name, "variant": bname, "seed": seed, **info,
            "identities": len(vocab),
            "needs": "x with a facility-identity column appended (tri_identity.attach_identity) and "
                     "pair inputs with capability_score, peer_votes; not servable by the platform as it stands",
        }, indent=2, default=float), encoding="utf-8")
        results["saved_checkpoint"] = ckpt.name

    table = {**results["models"], **earlier}
    print("\n" + "=" * 120)
    print(f"  {TEST_YEAR} test, all trained on {list(TRAIN_YEARS)} (mean over seeds {args.seeds})")
    print("=" * 120)
    print(f"  {'':<50}" + "".join(f"{k:>12}" for k in METRICS) + f"{'new Hit@1':>12}")
    for n, r in sorted(table.items(), key=lambda kv: -kv[1]["hit@1_mean"]):
        new = r.get("new_senders", {}).get("hit@1")
        print(f"  {n:<50}" + "".join(f"{r[k + '_mean']:>12.4f}" for k in METRICS)
              + (f"{new:>12.4f}" if new is not None else ""))
    print("=" * 120)
    for n, a in results["attention"].items():
        print(f"\n  Attention, {n}:")
        for label, w in a.items():
            print(f"    {label:<22}" + "  ".join(f"{v}: {w[v]:.3f}" for v in ("relationship", "capability", "peer")))

    save(results)
    print(f"\nSaved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
