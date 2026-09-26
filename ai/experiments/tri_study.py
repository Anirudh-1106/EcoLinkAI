"""
Does the MC-GNN work on real data?

Every result until now was measured on a market this project generated. This
scores the model on real industrial recovery trade: EPA TRI off-site
transfers of waste for recycling and energy recovery, between named US
facilities, with real distances and reported quantities. The target year is
2024; the model never sees it, and every input is built from 2019-2023.

Two tests, and what each answers:

  transfer   The production checkpoint -- trained on the platform's own data
             -- run unchanged. Does what it learned carry over to a real
             market it was never shown?

  control    The same architecture trained on US history (2022, with 2019-21
             as its past), early-stopped on 2023, scored on 2024. Can the
             design learn real recovery trade at all? Without this, a weak
             transfer score could mean a bad design or merely a different
             market, and there would be no telling which.

Both are run with and without the graph, since whether message passing earns
its place is an open question on the platform's own data too.

Alongside them, simple rankers that need no learning at all -- the platform's
rule-based baseline, nearest first, busiest receiver first, past partner
first -- so the numbers have something honest to be measured against.

Usage:
    python -m ai.experiments.tri_study              # everything
    python -m ai.experiments.tri_study --skip-control
    python -m ai.experiments.tri_study --seeds 42 7 13 --epochs 300

Writes ai/experiments/results/tri_study.json.
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

from ai.evaluation.metrics import evaluate_model
from ai.experiments.tri_loader import (
    CANDIDATE_DISTANCE_QUANTILE,
    DATA_DIR,
    DOWNLOADS,
    EDGE_FEATURE_DIM,
    NEGATIVES_PER_POSITIVE,
    NODE_FEATURE_DIM,
    TRIDataset,
)
from ai.models.baseline import BaselineRuleModel
from ai.models.mc_gnn import HSIC_GAMMA, MCGNN
from ai.training.train import PATIENCE
from app.utils.edge_encoding import EDGE_FEATURE_NAMES

logger = logging.getLogger("tri_study")

CHECKPOINT = PROJECT_ROOT / "ai" / "checkpoints" / "mc_gnn_best.pt"
RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_study.json"

HISTORY_START = 2019
TRAIN_YEAR, VAL_YEAR, TEST_YEAR = 2022, 2023, 2024

# The plan's go/no-go line: below this share of 2024 transfers with both ends
# located, distances would be missing for too much of the market to trust the
# result, and the study stops rather than reporting it.
MIN_LOCATED_SHARE = 0.90

LR = 0.01
WEIGHT_DECAY = 1e-4


# ── tensors and scoring ──────────────────────────────────────────────────


def _tensors(g: dict, *, with_graph: bool = True, coords: np.ndarray | None = None):
    x = torch.tensor(g["x"] if coords is None else coords)
    edge_index = torch.tensor(g["edge_index"])
    edge_attr = torch.tensor(g["edge_attr"])
    message = (
        torch.tensor(g["message_edge_index"]) if with_graph
        else torch.empty((2, 0), dtype=torch.long)
    )
    return x, edge_index, edge_attr, message


@torch.no_grad()
def score(model: MCGNN, g: dict, *, with_graph: bool = True, coords: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    x, ei, ea, msg = _tensors(g, with_graph=with_graph, coords=coords)
    out, _ = model(x, ei, ea, message_edge_index=msg)
    return out.numpy()


def measure(g: dict, scores: np.ndarray) -> dict:
    y, groups = g["y"], g["group_ids"]
    at5 = evaluate_model(y, scores, k=5, group_ids=groups)
    at1 = evaluate_model(y, scores, k=1, group_ids=groups)
    return {
        "auc": at5["auc"],
        "ndcg@5": at5["ndcg@5"],
        "recall@5": at5["recall@5"],
        "hit@1": at1["precision@1"],
        "queries": at5["num_queries"],
    }


def reference_rankers(g: dict, seed: int) -> dict[str, np.ndarray]:
    """Rankers that learn nothing. Each isolates one thing a model could lean on."""
    ea = g["edge_attr"]
    dist = EDGE_FEATURE_NAMES.index("norm_distance")
    prior = EDGE_FEATURE_NAMES.index("prior_successes")
    rng = np.random.default_rng(seed)
    return {
        "rule-based baseline (platform)": BaselineRuleModel().predict(ea),
        "nearest receiver first": 1.0 - ea[:, dist],
        "busiest receiver first": g["receiver_popularity"],
        "past partner first": ea[:, prior] + 1e-6 * (1.0 - ea[:, dist]),
        "random": rng.random(len(ea)),
    }


# ── the transfer test ────────────────────────────────────────────────────


def _training_coordinate_range() -> tuple[np.ndarray, np.ndarray] | None:
    """Min and max of the platform graph's two location inputs, read from its own data."""
    try:
        logging.disable(logging.CRITICAL)
        from app.core.database import SessionLocal
        from ai.graph.builder import build_industrial_graph

        db = SessionLocal()
        try:
            data, _, _ = build_industrial_graph(db)
        finally:
            db.close()
        loc = data.x[:, :2].numpy()
        return loc.min(axis=0), loc.max(axis=0)
    except Exception as exc:  # database not running
        logger.warning("Platform database unavailable (%s); skipping the rescaled-location variant.", exc)
        return None
    finally:
        logging.disable(logging.NOTSET)


def _rescaled_coords(g: dict, train_range) -> np.ndarray:
    """
    The same graph with location mapped into the range the model was trained on.

    The platform learned from plants between roughly 8 and 13 degrees north;
    US facilities sit between 25 and 49, at negative longitudes it never saw.
    Mapping the US span linearly onto the training span keeps every facility's
    relative position and asks only whether the model reads location as
    geography or as an absolute number. Reported beside the raw variant, not
    instead of it.
    """
    x = g["x"].copy()
    lo, hi = train_range
    us_lo, us_hi = x[:, :2].min(axis=0), x[:, :2].max(axis=0)
    span = np.where(us_hi > us_lo, us_hi - us_lo, 1.0)
    x[:, :2] = (x[:, :2] - us_lo) / span * (hi - lo) + lo
    return x


def transfer_test(g_test: dict) -> dict:
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    in_dim = model.gcn_conv1.in_channels
    edge_dim = model.link_predictor[0].in_features - 2 * model.fusion.channel_transforms[0].in_features
    if in_dim != NODE_FEATURE_DIM or edge_dim != EDGE_FEATURE_DIM:
        raise RuntimeError(
            f"Checkpoint expects {in_dim} plant and {edge_dim} pair inputs; the "
            f"loader produces {NODE_FEATURE_DIM} and {EDGE_FEATURE_DIM}."
        )

    out = {
        "checkpoint": CHECKPOINT.name,
        "raw location, with graph": measure(g_test, score(model, g_test)),
        "raw location, no graph": measure(g_test, score(model, g_test, with_graph=False)),
    }
    train_range = _training_coordinate_range()
    if train_range is not None:
        coords = _rescaled_coords(g_test, train_range)
        out["rescaled location, with graph"] = measure(g_test, score(model, g_test, coords=coords))
        out["rescaled location, no graph"] = measure(
            g_test, score(model, g_test, with_graph=False, coords=coords))
    return out


# ── the control ──────────────────────────────────────────────────────────


def train_control(g_train: dict, g_val: dict, seed: int, *, with_graph: bool, epochs: int) -> tuple[MCGNN, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MCGNN(in_features=NODE_FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    xt, eit, eat, mt = _tensors(g_train, with_graph=with_graph)
    yt = torch.tensor(g_train["y"])
    best = {"auc": -1.0, "loss": float("inf"), "epoch": 0}
    best_state, stale = None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        s, channels = model(xt, eit, eat, message_edge_index=mt)
        loss = model.compute_loss(s, yt, channels, hsic_weight=HSIC_GAMMA)
        loss.backward()
        opt.step()

        val_scores = score(model, g_val, with_graph=with_graph)
        val_auc = measure(g_val, val_scores)["auc"]
        improved = val_auc > best["auc"] + 1e-9 or (
            abs(val_auc - best["auc"]) <= 1e-9 and loss.item() < best["loss"])
        if improved:
            best = {"auc": val_auc, "loss": loss.item(), "epoch": epoch}
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best["epoch"], "val_auc": best["auc"], "epochs_run": epoch}


def control(g_train: dict, g_val: dict, g_test: dict, seeds: list[int], epochs: int) -> dict:
    out = {}
    for with_graph in (True, False):
        label = "with graph" if with_graph else "no graph"
        runs = []
        for seed in seeds:
            t0 = time.time()
            model, info = train_control(g_train, g_val, seed, with_graph=with_graph, epochs=epochs)
            m = measure(g_test, score(model, g_test, with_graph=with_graph))
            runs.append({**m, **info, "seed": seed, "seconds": round(time.time() - t0, 1)})
            print(f"    control {label:<10} seed {seed:<5} AUC {m['auc']:.4f}  NDCG@5 {m['ndcg@5']:.4f}"
                  f"  (best epoch {info['best_epoch']}, {runs[-1]['seconds']}s)")
        out[label] = {
            "runs": runs,
            **{f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in ("auc", "ndcg@5", "recall@5", "hit@1")},
            **{f"{k}_sd": float(np.std([r[k] for r in runs])) for k in ("auc", "ndcg@5", "recall@5", "hit@1")},
        }
    return out


# ── report ───────────────────────────────────────────────────────────────


def _row(name: str, m: dict, sd: dict | None = None) -> str:
    def cell(k):
        v = f"{m[k]:.4f}"
        return f"{v} +/-{sd[k]:.3f}" if sd else v
    return f"  {name:<50}{cell('auc'):>16}{cell('ndcg@5'):>16}{cell('recall@5'):>16}{cell('hit@1'):>16}"


def report(results: dict) -> None:
    print()
    print("=" * 116)
    print(f"  Scored on {TEST_YEAR} EPA TRI recovery transfers "
          f"({results['graphs'][str(TEST_YEAR)]['queries_used']:,} queries, "
          f"~{NEGATIVES_PER_POSITIVE + 1} candidates each); inputs from {HISTORY_START}-{TEST_YEAR - 1} only")
    print("=" * 116)
    print(f"  {'':<50}{'AUC':>16}{'NDCG@5':>16}{'Recall@5':>16}{'Hit@1':>16}")
    print("  " + "-" * 112)
    for name, m in results["transfer"].items():
        if isinstance(m, dict):
            print(_row(f"MC-GNN as shipped: {name}", m))
    if "control" in results:
        print("  " + "-" * 112)
        for label, c in results["control"].items():
            mean = {k: c[f"{k}_mean"] for k in ("auc", "ndcg@5", "recall@5", "hit@1")}
            sd = {k: c[f"{k}_sd"] for k in ("auc", "ndcg@5", "recall@5", "hit@1")}
            print(_row(f"MC-GNN retrained on US: {label}", mean, sd))
    print("  " + "-" * 112)
    for name, m in results["references"].items():
        print(_row(name, m))
    print("=" * 116)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--min-located", type=float, default=MIN_LOCATED_SHARE)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    years = list(range(HISTORY_START, TEST_YEAR + 1))
    print(f"Reading EPA TRI {years[0]}-{years[-1]} from {args.data_dir} ...")
    t0 = time.time()
    ds, coverage = TRIDataset.from_files(years, args.data_dir)
    print(f"  done in {time.time() - t0:.0f}s")
    print(f"  receivers located by: {coverage['receiver_location_sources']}")
    for y, c in coverage["per_year"].items():
        print(f"  {y}: {c['transfers']:,} recovery transfers, {c['located_share']:.1%} with both ends located")

    share = coverage["per_year"][TEST_YEAR]["located_share"]
    if share < args.min_located:
        sys.exit(
            f"\nSTOP: only {share:.1%} of {TEST_YEAR} transfers could be placed on the map "
            f"(threshold {args.min_located:.0%}). Is the facility registry "
            f"({DOWNLOADS['frs']}) in {args.data_dir}?"
        )

    print("\nBuilding graphs (history only, per target year) ...")
    graphs = {TEST_YEAR: ds.build_graph(TEST_YEAR, history_start=HISTORY_START)}
    if not args.skip_control:
        graphs[TRAIN_YEAR] = ds.build_graph(TRAIN_YEAR, history_start=HISTORY_START)
        graphs[VAL_YEAR] = ds.build_graph(VAL_YEAR, history_start=HISTORY_START)
    for y, g in sorted(graphs.items()):
        s = g["stats"]
        print(f"  {y}: {s['nodes']:,} facilities, {s['message_edges']:,} history links, "
              f"{s.get('queries_used', 0):,} queries, {s.get('positives_used', 0):,} real + "
              f"{s.get('negatives_used', 0):,} sampled; {s.get('positives_cold', 0):,} of "
              f"{s.get('positives_total', 0):,} real partners had no history (excluded)")

    g_test = graphs[TEST_YEAR]
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "history_start": HISTORY_START, "train_year": TRAIN_YEAR, "val_year": VAL_YEAR,
            "test_year": TEST_YEAR, "negatives_per_positive": NEGATIVES_PER_POSITIVE,
            "candidate_distance_quantile": CANDIDATE_DISTANCE_QUANTILE,
            "seeds": args.seeds, "max_epochs": args.epochs, "patience": PATIENCE,
            "lr": LR, "hsic_gamma": HSIC_GAMMA,
        },
        "coverage": coverage,
        "graphs": {str(y): g["stats"] for y, g in graphs.items()},
        "market_profile": ds.market_profile(list(range(HISTORY_START, TEST_YEAR))),
    }

    print("\nTransfer test: production checkpoint, unchanged ...")
    results["transfer"] = transfer_test(g_test)

    if not args.skip_control:
        print("\nControl: same architecture, trained on US history ...")
        results["control"] = control(graphs[TRAIN_YEAR], graphs[VAL_YEAR], g_test, args.seeds, args.epochs)

    results["references"] = {
        name: measure(g_test, s) for name, s in reference_rankers(g_test, args.seeds[0]).items()
    }

    report(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
