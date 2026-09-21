"""
Does the MC-GNN's graph component earn its place, and under what conditions?

On the platform's own data the model ties a weighted rule-based baseline, and
deleting the graph entirely changes its output by under 2%. Two explanations
fit that equally well:

  1. the network is too small and too densely connected for message passing
     to distinguish anyone, or
  2. there is no network effect in the data to find, because whether a deal
     closes depends only on the two companies involved and never on who else
     they trade with.

Neither can be told from the other by looking at the production dataset,
because it has both properties at once. So this varies them separately, on
synthetic networks built here rather than in the platform's pipeline -- the
real data, including the KINFRA companies, is left untouched.

Four conditions, one factor changed at a time:

    small (124 plants)  x  no word-of-mouth
    small               x  word-of-mouth
    large (800 plants)  x  no word-of-mouth
    large               x  word-of-mouth

"Word-of-mouth" means a buyer is likelier to accept a supplier that its own
existing partners already trade with successfully -- a referral effect, and a
documented property of real industrial and supply-chain networks. It is the
one signal a per-row formula structurally cannot read, because it lives in
the neighbourhood rather than the row.

Every condition is scored with the same model, the same baseline, the same
temporal split and the same per-query metrics the production pipeline uses,
so the only thing differing between runs is the data.

Read this alongside the production result. A win here demonstrates the
architecture recovers network effects where they exist; it does not
demonstrate that Kerala's actual exchange network has them.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import torch

from ai.evaluation.metrics import evaluate_model
from ai.models.baseline import BaselineRuleModel
from ai.models.mc_gnn import HSIC_GAMMA, MCGNN
from ai.training.train import PATIENCE, _message_graph, _split_edges
from app.utils.edge_encoding import encode_edge_features

logger = logging.getLogger("network_effect_study")

# Matching the production training setup so results are comparable to it.
EPOCHS, LR = 500, 0.01
HSIC_WEIGHT = HSIC_GAMMA  # the paper's gamma, not the 0.05 used before
INITIALISATIONS = (42, 7, 13)

# Weights on the acceptance decision, mirroring the platform's own generator
# (scripts/database/seed.py) so the synthetic data behaves like the real
# pipeline's, plus the one term under test.
W_PROXIMITY = 2.60
W_QUANTITY = 2.40
W_TRUST = 1.20
W_DYADIC = 1.60      # how often *these two* have traded: readable from the row
W_PRICE = 0.90       # deliberately not shown to either model
NOISE_SD = 0.35
INTERCEPT = -4.30

# Strength of the referral effect in the "word-of-mouth" conditions. Set at
# the same weight as the dyadic term: a partner's endorsement counts as much
# as one's own past dealings, which is strong but not implausible for an
# industry where a bad consignment is expensive.
W_SECOND_ORDER = 1.60

SECTORS = 8


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def generate(
    *,
    n_plants: int,
    avg_degree: float,
    requests_per_pair: float,
    word_of_mouth: bool,
    seed: int,
) -> dict:
    """
    Build a synthetic exchange network and replay its requests in date order.

    Candidate pairs are drawn first, giving a graph of the requested average
    degree, then each pair enquires a few times over a simulated year. Because
    the replay is chronological, a request's outcome can depend on what had
    already happened -- which is what makes a referral effect expressible at
    all.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # ── Plants ────────────────────────────────────────
    # Positioned on a plane and given the same kinds of attribute the real
    # node features carry, so the baseline has something real to work with.
    lat = np_rng.uniform(8.2, 12.8, n_plants)     # roughly Kerala's span
    lon = np_rng.uniform(74.8, 77.4, n_plants)
    trust = np_rng.uniform(40.0, 100.0, n_plants)
    sector = np_rng.integers(0, SECTORS, n_plants)
    capacity = np_rng.lognormal(mean=6.5, sigma=1.1, size=n_plants)

    # ── Candidate pairs ───────────────────────────────
    # Drawn within industrial clusters rather than uniformly at random.
    #
    # This matters for more than realism. A referral only exists when a
    # buyer's partners happen to share a supplier with it, and in a sparse
    # uniform graph that almost never happens: at 800 plants the referral
    # term fired on 4.8% of requests with a mean of 0.017, far too rare to
    # detect anything. Clustering makes neighbourhoods overlap, which is both
    # how real industrial parks are organised and the only way a word-of-mouth
    # effect can be present often enough to be measurable.
    n_clusters = max(2, n_plants // 40)
    cluster_of = np_rng.integers(0, n_clusters, n_plants)
    members: dict[int, list[int]] = defaultdict(list)
    for plant, cluster in enumerate(cluster_of):
        members[int(cluster)].append(plant)

    target_pairs = int(n_plants * avg_degree / 2)
    pairs: set[tuple[int, int]] = set()
    guard = 0
    while len(pairs) < target_pairs and guard < target_pairs * 80:
        guard += 1
        # Most trade happens inside a cluster; the rest reaches across, so
        # the graph stays connected rather than splintering into islands.
        if rng.random() < 0.85:
            group = members[rng.randrange(n_clusters)]
            if len(group) < 2:
                continue
            a, b = rng.sample(group, 2)
        else:
            a = rng.randrange(n_plants)
            b = rng.randrange(n_plants)
        if a == b:
            continue
        pairs.add((a, b))

    ordered_pairs = sorted(pairs)

    # ── Requests, in date order ───────────────────────
    requests = []
    for (supplier, buyer) in ordered_pairs:
        for _ in range(max(1, int(round(rng.gauss(requests_per_pair, 0.8))))):
            requests.append((rng.uniform(0.0, 365.0), supplier, buyer))
    requests.sort(key=lambda r: r[0])

    successes: dict[tuple[int, int], int] = defaultdict(int)
    partners: dict[int, set[int]] = defaultdict(set)

    sources, targets, attrs, labels, times = [], [], [], [], []
    # Recorded so the strength of the treatment can be reported rather than
    # assumed. A referral effect that almost never fires tests nothing, and a
    # null result under those conditions would say nothing about the
    # architecture.
    referral_values: list[float] = []

    for day, supplier, buyer in requests:
        distance = math.hypot(lat[supplier] - lat[buyer], lon[supplier] - lon[buyer]) * 111.0
        proximity = 1.0 - min(distance / 500.0, 1.0)

        # Quantity fit between what the seller has and the buyer wants.
        offered = capacity[supplier] * rng.uniform(0.3, 1.4)
        wanted = capacity[buyer] * rng.uniform(0.3, 1.4)
        quantity_fit = min(offered, wanted) / max(offered, wanted)

        quality_fit = rng.uniform(0.75, 1.0)
        price_appeal = rng.random()
        buyer_trust = trust[buyer] / 100.0

        dyadic = min(successes[(supplier, buyer)] / 3.0, 1.0)

        # The term under test: of the buyer's existing partners, how many
        # already trade with this supplier successfully? Readable only by
        # looking one step beyond the pair itself.
        second_order = 0.0
        if word_of_mouth and partners[buyer]:
            endorsing = sum(
                1 for other in partners[buyer] if successes[(supplier, other)] > 0
            )
            second_order = min(endorsing / 3.0, 1.0)
        referral_values.append(second_order)

        z = (
            INTERCEPT
            + W_PROXIMITY * proximity
            + W_QUANTITY * quantity_fit
            + W_TRUST * buyer_trust
            + W_DYADIC * dyadic
            + W_PRICE * price_appeal
            + (W_SECOND_ORDER * second_order if word_of_mouth else 0.0)
            + rng.gauss(0.0, NOISE_SD)
        )
        accepted = 1.0 if rng.random() < _sigmoid(z) else 0.0

        # Encoded exactly as production encodes an edge, so the models see
        # the same shape of input they were built for.
        transport_cost = distance * max(min(offered, wanted) / 1000.0, 0.1) * 8.0
        carbon_saving = min(offered, wanted) * 1.2
        attrs.append(
            encode_edge_features(
                distance_km=distance,
                material_compatibility=100.0,
                quantity_compatibility=quantity_fit * 100.0,
                quality_compatibility=quality_fit * 100.0,
                transport_cost=transport_cost,
                carbon_saving=carbon_saving,
                prior_successes=successes[(supplier, buyer)],
                has_prior_interaction=(supplier, buyer) in partners.get(buyer, set())
                or successes[(supplier, buyer)] > 0
                or supplier in partners[buyer],
            )
        )
        sources.append(supplier)
        targets.append(buyer)
        labels.append(accepted)
        times.append(day)

        # Recorded only after encoding, so a request never informs itself.
        if accepted:
            successes[(supplier, buyer)] += 1
            partners[buyer].add(supplier)
            partners[supplier].add(buyer)

    # ── Node features ─────────────────────────────────
    degree = np.zeros(n_plants)
    for s, t in zip(sources, targets):
        degree[s] += 1
        degree[t] += 1
    one_hot = np.eye(SECTORS)[sector]
    x = np.column_stack([
        lat / 90.0,
        lon / 180.0,
        trust / 100.0,
        np.log1p(capacity) / 12.0,
        degree / max(degree.max(), 1.0),
        one_hot,
    ]).astype(np.float32)

    referrals = np.array(referral_values) if referral_values else np.zeros(1)
    return {
        "referral_fire_rate": float((referrals > 0).mean()),
        "referral_mean": float(referrals.mean()),
        "x": torch.tensor(x),
        "edge_index": torch.tensor([sources, targets], dtype=torch.long),
        "edge_attr": torch.tensor(np.array(attrs, dtype=np.float32)),
        "y": torch.tensor(np.array(labels, dtype=np.float32)),
        "edge_time": torch.tensor(np.array(times, dtype=np.float64)),
        "n_plants": n_plants,
        "n_pairs": len(ordered_pairs),
    }


def _train_and_score(data: dict, split: dict, message: torch.Tensor, seed: int) -> dict:
    """One training run, early-stopped on validation AUC, scored on test."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MCGNN(in_features=data["x"].size(1), edge_dim=data["edge_attr"].size(1))
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    val_labels = split["val_labels"].numpy()
    val_groups = split["val_edge_index"][1].numpy()
    best_score, best_loss, best_state, stale = -1.0, float("inf"), None, 0

    for _ in range(EPOCHS):
        model.train()
        optimiser.zero_grad()
        scores, channels = model(
            data["x"], split["train_edge_index"], split["train_edge_attr"],
            message_edge_index=message,
        )
        model.compute_loss(
            scores, split["train_labels"], channels, hsic_weight=HSIC_WEIGHT
        ).backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            v_scores, v_channels = model(
                data["x"], split["val_edge_index"], split["val_edge_attr"],
                message_edge_index=message,
            )
            v_loss = model.compute_loss(
                v_scores, split["val_labels"], v_channels, hsic_weight=HSIC_WEIGHT
            ).item()
            v_auc = evaluate_model(
                val_labels, v_scores.numpy(), k=5, group_ids=val_groups
            )["auc"]

        improved = v_auc > best_score + 1e-9 or (
            abs(v_auc - best_score) <= 1e-9 and v_loss < best_loss
        )
        if improved:
            best_score, best_loss = v_auc, v_loss
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        t_scores, _ = model(
            data["x"], split["test_edge_index"], split["test_edge_attr"],
            message_edge_index=message,
        )
    return evaluate_model(
        split["test_labels"].numpy(), t_scores.numpy(), k=5,
        group_ids=split["test_edge_index"][1].numpy(),
    )


def run_condition(*, n_plants: int, avg_degree: float, word_of_mouth: bool, seed: int = 1) -> dict:
    """Generate one dataset, then score the MC-GNN and the baseline on it."""
    data = generate(
        n_plants=n_plants, avg_degree=avg_degree, requests_per_pair=2.4,
        word_of_mouth=word_of_mouth, seed=seed,
    )
    split = _split_edges(
        data["edge_index"], data["edge_attr"], data["y"], edge_times=data["edge_time"]
    )
    # Accepted training edges only, exactly as production propagates.
    message = _message_graph(split["train_edge_index"], split["train_labels"])

    baseline = evaluate_model(
        split["test_labels"].numpy(),
        BaselineRuleModel().predict(split["test_edge_attr"].numpy()),
        k=5, group_ids=split["test_edge_index"][1].numpy(),
    )

    runs = [_train_and_score(data, split, message, s) for s in INITIALISATIONS]
    # The same architecture with the graph removed, to separate what message
    # passing contributes from what the features alone do.
    empty = torch.empty((2, 0), dtype=torch.long)
    no_graph = [_train_and_score(data, split, empty, s) for s in INITIALISATIONS]

    return {
        "edges": data["edge_index"].size(1),
        "pairs": data["n_pairs"],
        "acceptance": float(data["y"].mean()),
        "gnn_auc": float(np.mean([r["auc"] for r in runs])),
        "gnn_sd": float(np.std([r["auc"] for r in runs])),
        "gnn_ndcg": float(np.mean([r["ndcg@5"] for r in runs])),
        "nograph_auc": float(np.mean([r["auc"] for r in no_graph])),
        "baseline_auc": baseline["auc"],
        "baseline_ndcg": baseline["ndcg@5"],
    }


def main() -> None:
    logging.disable(logging.CRITICAL)

    conditions = [
        ("small (124)", 124, 12.0, False),
        ("small (124)", 124, 12.0, True),
        ("large (800)", 800, 14.0, False),
        ("large (800)", 800, 14.0, True),
    ]

    print("MC-GNN vs rule-based baseline, by network size and referral effect")
    print("temporal split, per-query metrics, 3 initialisations, identical features\n")
    print(f"{'network':<13}{'word of mouth':<15}{'edges':>7}{'MC-GNN':>10}{'no graph':>10}"
          f"{'baseline':>10}{'GNN-base':>11}")
    print("-" * 78)

    results = []
    for label, n, degree, wom in conditions:
        r = run_condition(n_plants=n, avg_degree=degree, word_of_mouth=wom)
        gap = r["gnn_auc"] - r["baseline_auc"]
        results.append((label, wom, r, gap))
        print(f"{label:<13}{('yes' if wom else 'no'):<15}{r['edges']:>7}"
              f"{r['gnn_auc']:>10.4f}{r['nograph_auc']:>10.4f}"
              f"{r['baseline_auc']:>10.4f}{gap:>+11.4f}")

    print("-" * 78)
    print("\nWhat the graph itself contributes (MC-GNN minus the same model with no graph):")
    for label, wom, r, _ in results:
        print(f"   {label}, word of mouth {'yes' if wom else 'no ':<3}: "
              f"{r['gnn_auc'] - r['nograph_auc']:+.4f}")

    print("\nPre-registered threshold: the MC-GNN must beat the baseline by more")
    print("than 0.02 AUC, consistently, for the architecture to have earned its place.")
    for label, wom, r, gap in results:
        if wom:
            verdict = "MET" if gap > 0.02 else "not met"
            print(f"   {label} with word of mouth: {gap:+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
