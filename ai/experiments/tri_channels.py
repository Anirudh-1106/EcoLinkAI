"""
Are the MC-GNN's channels actually different from one another?

The architecture's premise is that its channels learn complementary
representations, with HSIC pushing them apart. This measures, on saved
models and the 2024 EPA TRI test graph:

  similarity   normalised HSIC (linear CKA) between every pair of channel
               embeddings: 0 shares nothing, 1 is the same representation
               up to rotation and scale
  HSIC share   the weighted HSIC term (gamma = 1e-10, raw inner-product
               HSIC on 2,048 sampled nodes, as in training) against the
               cross-entropy it is added to -- whether it can influence
               training at all

Usage:
    python -m ai.experiments.tri_channels

Writes ai/experiments/results/tri_channels.json.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

from ai.experiments.tri_improve import with_network_columns
from ai.experiments.tri_loader import DATA_DIR, TRIDataset
from ai.experiments.tri_multiview import OnDevice
from ai.experiments.tri_study import HISTORY_START, TEST_YEAR, pick_device
from ai.models.hsic import hsic_loss, normalized_hsic
from ai.models.mc_gnn import HSIC_GAMMA, HSIC_MAX_NODES

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_channels.json"
CHECKPOINTS = PROJECT_ROOT / "ai" / "checkpoints"

# checkpoint -> (label, message graph, uses the two network columns)
MODELS = {
    "mc_gnn_improved_20260930_195221.pt": ("Improved MC-GNN (one network)", "relationship", True),
    "mc_gnn_mv_20260929_081906.pt": ("Multi-view MC-GNN, no channel dropout", "views", False),
    "mc_gnn_mv_cd_20260930_221806.pt": ("Multi-view MC-GNN + channel dropout", "views", True),
}
CHANNELS = ("GCN", "GAT", "GraphSAGE")


@torch.no_grad()
def channel_report(model, d: OnDevice, mode: str, seed: int = 0) -> dict:
    model.eval()
    _, channels = model.encode(d.x, d.messages[mode])
    pairs = {}
    for i in range(3):
        for j in range(i + 1, 3):
            pairs[f"{CHANNELS[i]}-{CHANNELS[j]}"] = round(normalized_hsic(channels[i], channels[j]).item(), 4)

    # The training-time HSIC term, on the same 2,048-node sample size.
    g = torch.Generator(device=d.x.device).manual_seed(seed)
    idx = torch.randperm(d.x.size(0), generator=g, device=d.x.device)[:HSIC_MAX_NODES]
    sampled = [c[idx] for c in channels]
    raw = (hsic_loss(sampled[0], sampled[1]) + hsic_loss(sampled[0], sampled[2])
           + hsic_loss(sampled[1], sampled[2])) / 3.0
    probs, _ = model(d.x, d.edge_index, d.edge_attr, message_edge_index=d.messages[mode])
    bce = F.binary_cross_entropy(probs, d.y)
    return {
        "normalised_hsic": pairs,
        "normalised_hsic_mean": round(sum(pairs.values()) / 3, 4),
        "raw_hsic": float(raw),
        "weighted_hsic_term": float(HSIC_GAMMA * raw),
        "bce": float(bce),
        "hsic_term_share_of_loss": float(HSIC_GAMMA * raw / (bce + HSIC_GAMMA * raw)),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    device = pick_device()
    years = list(range(HISTORY_START, TEST_YEAR + 1))
    print("Reading EPA TRI and building the 2024 test graph ...", flush=True)
    ds, _ = TRIDataset.from_files(years, DATA_DIR)
    g = ds.build_graph(TEST_YEAR, history_start=HISTORY_START, views=True)
    plain = OnDevice(g, device)
    extended = with_network_columns(OnDevice(g, device))

    results = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "gamma": HSIC_GAMMA, "models": {}}
    for ckpt, (label, mode, network_columns) in MODELS.items():
        model = torch.load(CHECKPOINTS / ckpt, map_location=device, weights_only=False)
        report = channel_report(model, extended if network_columns else plain, mode)
        results["models"][label] = {"checkpoint": ckpt, **report}
        print(f"\n{label}")
        print("  channel similarity (0 = different, 1 = same): "
              + ", ".join(f"{k} {v:.3f}" for k, v in report["normalised_hsic"].items()))
        print(f"  HSIC term {report['weighted_hsic_term']:.2e} vs cross-entropy {report['bce']:.3f} "
              f"-> {100 * report['hsic_term_share_of_loss']:.4f}% of the loss")
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
