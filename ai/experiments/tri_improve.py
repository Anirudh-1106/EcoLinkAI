"""
The MC-GNN with every improvement the earlier studies pointed to.

tri_multiview left the MC-GNN (one relationship network) within about one
point of XGBoost, and showed where the gap came from. This applies what that
evidence supports, and nothing tuned on the test year:

  * the two network columns (capability_score, peer_votes) as pair inputs,
    beside the eight the platform already encodes. They lifted XGBoost most,
    and XGBoost's number here is the one with them, so the comparison stays
    like for like;
  * three training years with fresh negatives every epoch, as tri_multiview;
  * room to converge: runs there were still improving at epoch 500;
  * a ranking term in the loss. Results are read per question -- is the real
    recycler first, in the top five -- while binary cross-entropy scores each
    pair on its own. A listwise term (softmax over each question's
    candidates) trains for the ordering that is measured. Cross-entropy stays
    in the loss so scores remain usable as probabilities.

Run beside it is the same model without the ranking term, so each change's
share of any gain can be told apart.

Selection is on 2023 AUC and 2024 is scored once, exactly as every other
model in these studies.

Usage:
    python -m ai.experiments.tri_improve
    python -m ai.experiments.tri_improve --seeds 42 7 13 --epochs 1000

Writes ai/experiments/results/tri_improve.json and saves the best improved
model under ai/checkpoints/.
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
from ai.experiments.tri_loader import DATA_DIR, EDGE_FEATURE_DIM, NODE_FEATURE_DIM, TRIDataset
from ai.experiments.tri_multiview import (
    TRAIN_POOL_PER_POSITIVE,
    TRAIN_YEARS,
    FreshNegatives,
    OnDevice,
    _summarise,
    measure_subset,
    predict,
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
)
from ai.experiments.tri_views import PAIR_GRAPH_FEATURE_NAMES
from ai.models.mc_gnn import HSIC_GAMMA, MCGNN
from ai.training.train import PATIENCE

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_improve.json"
EARLIER_RESULTS = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_multiview.json"
CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"
METRICS = ("auc", "ndcg@5", "recall@5", "hit@1")

PAIR_INPUTS = EDGE_FEATURE_DIM + len(PAIR_GRAPH_FEATURE_NAMES)

# Weight of the listwise term against cross-entropy. Equal footing: neither
# is a correction to the other.
RANKING_WEIGHT = 1.0

VARIANTS = {
    "Improved MC-GNN": RANKING_WEIGHT,
    "Improved MC-GNN, no ranking loss": 0.0,
}


def with_network_columns(d: OnDevice) -> OnDevice:
    """Append the two network columns to every pair's inputs."""
    extra = torch.as_tensor(d.g["pair_graph_features"], device=d.edge_attr.device)
    d.edge_attr = torch.cat([d.edge_attr, extra], dim=1)
    d.groups = torch.as_tensor(d.g["group_ids"], device=d.edge_attr.device)
    return d


def listwise_loss(probs: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """
    Softmax cross-entropy over each question's candidates.

    For every real partner, minus the log of its share of the softmax across
    its question's candidates: pushed up only by out-scoring the others in
    the same question, which is what Hit@1 and NDCG@5 reward.
    """
    logits = torch.logit(probs, eps=1e-6)
    _, g = torch.unique(groups, return_inverse=True)
    n = int(g.max()) + 1
    peak = torch.full((n,), float("-inf"), device=logits.device).scatter_reduce(
        0, g, logits, reduce="amax", include_self=True)
    peak = peak.detach()
    denom = torch.zeros(n, device=logits.device).scatter_add(0, g, torch.exp(logits - peak[g]))
    log_share = logits - peak[g] - torch.log(denom[g])
    positive = labels > 0.5
    return -log_share[positive].mean()


def train(ranking_weight: float, train_set, val: OnDevice, seed: int, epochs: int, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = MCGNN(NODE_FEATURE_DIM, edge_dim=PAIR_INPUTS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = {"auc": -1.0, "epoch": 0}
    best_state, stale, epoch = None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        for d, sampler in train_set:
            idx = sampler.draw(generator)
            probs, channels = model(d.x, d.edge_index[:, idx], d.edge_attr[idx],
                                    message_edge_index=d.messages["relationship"])
            loss = model.compute_loss(probs, d.y[idx], channels, hsic_weight=HSIC_GAMMA)
            if ranking_weight:
                loss = loss + ranking_weight * listwise_loss(probs, d.y[idx], d.groups[idx])
            (loss / len(train_set)).backward()
        opt.step()

        val_auc = roc_auc(val.g["y"], predict(model, val, "relationship"))
        if val_auc > best["auc"] + 1e-9:
            best = {"auc": val_auc, "epoch": epoch}
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best["epoch"], "val_auc": best["auc"], "epochs_run": epoch}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    device = pick_device(args.cpu)

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

    train_set = [(with_network_columns(OnDevice(g, device)), FreshNegatives(g, device)) for g in train_graphs]
    val_dev = with_network_columns(OnDevice(g_val, device))
    test_dev = with_network_columns(OnDevice(g_test, device))

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "train_years": list(TRAIN_YEARS), "val_year": VAL_YEAR, "test_year": TEST_YEAR,
            "seeds": args.seeds, "max_epochs": args.epochs, "patience": PATIENCE,
            "pair_inputs": PAIR_INPUTS, "ranking_weight": RANKING_WEIGHT, "device": str(device),
            "message_graph": "relationship (history trades, both directions)",
        },
        "models": {},
    }

    print(f"Training on {device} ...", flush=True)
    best_model = None
    for name, weight in VARIANTS.items():
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            model, info = train(weight, train_set, val_dev, seed, args.epochs, device)
            scores = predict(model, test_dev, "relationship")
            m = measure(g_test, scores)
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1),
                         "new_senders": measure_subset(g_test, scores, new_mask)})
            print(f"  {name:<34} seed {seed:<4} AUC {m['auc']:.4f}  Hit@1 {m['hit@1']:.4f}  "
                  f"NDCG@5 {m['ndcg@5']:.4f}  new Hit@1 {runs[-1]['new_senders']['hit@1']:.4f}  "
                  f"(best epoch {info['best_epoch']} of {info['epochs_run']}, {runs[-1]['seconds']}s)",
                  flush=True)
            if name == "Improved MC-GNN" and (best_model is None or info["val_auc"] > best_model[0]):
                best_model = (info["val_auc"], seed, model, info)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["models"][name] = _summarise(runs)

    # Same data, same test, same training years: the earlier models, for the table.
    earlier = json.loads(EARLIER_RESULTS.read_text(encoding="utf-8"))
    results["earlier_same_setup"] = {
        k: v for k, v in earlier["models"].items()
        if k in ("XGBoost + network columns", "XGBoost", "MC-GNN (one network)", "GCN only")
    }

    if best_model is not None:
        val_auc, seed, model, info = best_model
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = CHECKPOINT_DIR / f"mc_gnn_improved_{stamp}.pt"
        torch.save(model.cpu().eval(), ckpt)
        ckpt.with_suffix(".json").write_text(json.dumps({
            "checkpoint_file": ckpt.name, "model": "MCGNN", "pair_inputs": PAIR_INPUTS,
            "extra_pair_inputs": list(PAIR_GRAPH_FEATURE_NAMES),
            "needs": "pair inputs = the 8 platform edge features + capability_score, peer_votes "
                     "(ai/experiments/tri_views.py); not servable by the platform as it stands",
            "trained_on": f"EPA TRI, learn {list(TRAIN_YEARS)}, select {VAL_YEAR}",
            "seed": seed, **info,
        }, indent=2, default=float), encoding="utf-8")
        results["saved_checkpoint"] = ckpt.name

    table = {**results["models"], **results["earlier_same_setup"]}
    print("\n" + "=" * 112)
    print(f"  {TEST_YEAR} test, all trained on {list(TRAIN_YEARS)} (mean over seeds {args.seeds})")
    print("=" * 112)
    print(f"  {'':<36}" + "".join(f"{k:>14}" for k in METRICS) + f"{'new Hit@1':>14}")
    for name, r in sorted(table.items(), key=lambda kv: -kv[1]["hit@1_mean"]):
        new = r.get("new_senders", {}).get("hit@1")
        print(f"  {name:<36}" + "".join(f"{r[k + '_mean']:>14.4f}" for k in METRICS)
              + (f"{new:>14.4f}" if new is not None else ""))
    print("=" * 112)

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
