"""
Seed ensembles: the pair-aware MC-GNN against plain XGBoost, from saved scores.

Each study trains three seeds and reports their mean. Averaging the three
seeds' scores into one ranking -- an ensemble -- is the standard way to use
them and usually ranks better than any single seed. Both sides are treated
alike: the MC-GNN's three seeds are averaged, and so are plain XGBoost's.

No training happens here. The per-row 2024 test scores were written by
tri_pairaware.py and tri_significance.py, on the same test rows (checked).

Usage:
    python -m ai.experiments.tri_ensemble

Writes ai/experiments/results/tri_ensemble.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for _p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from ai.experiments.tri_loader import DATA_DIR
from ai.experiments.tri_multiview import measure_subset
from ai.experiments.tri_significance import BOOTSTRAP_SEED, Questions, paired_bootstrap
from ai.experiments.tri_study import measure

RESULTS_PATH = PROJECT_ROOT / "ai" / "experiments" / "results" / "tri_ensemble.json"
SOURCES = {
    "Pair-aware MC-GNN": "tri_pairaware_scores.npz",
    "XGBoost": "tri_significance_scores.npz",
}
RESAMPLES = 2000


def load(data_dir: Path):
    seeds, ref = {}, None
    for name, file in SOURCES.items():
        z = np.load(data_dir / "results" / file)
        if ref is None:
            ref = {k: z[k] for k in ("y", "group_ids", "sender_is_new")}
        elif not (np.array_equal(ref["y"], z["y"]) and np.array_equal(ref["group_ids"], z["group_ids"])):
            sys.exit(f"{file} was scored on different rows; cannot compare.")
        seeds[name] = [z[k] for k in sorted(z.files) if k.startswith(f"{name}|")]
    return ref, seeds


def main() -> None:
    ref, seeds = load(DATA_DIR)
    g = {"y": ref["y"], "group_ids": ref["group_ids"]}
    new = ref["sender_is_new"]
    results = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": {}}

    ensembles = {name: [np.mean(s, axis=0)] for name, s in seeds.items()}
    for name, s in seeds.items():
        for label, scores in ((f"{name} (mean of {len(s)} seeds)", s), (f"{name} (ensemble of {len(s)})", ensembles[name])):
            per = [measure(g, x) for x in scores]
            out = {k: float(np.mean([p[k] for p in per])) for k in ("auc", "ndcg@5", "recall@5", "hit@1")}
            out["new_hit@1"] = float(np.mean([measure_subset(g, x, new)["hit@1"] for x in scores]))
            results["models"][label] = out
            print(f"  {label:<40} AUC {out['auc']:.4f}  NDCG@5 {out['ndcg@5']:.4f}  "
                  f"Recall@5 {out['recall@5']:.4f}  Hit@1 {out['hit@1']:.4f}  new {out['new_hit@1']:.4f}")

    all_q, new_q = Questions(ref["y"], ref["group_ids"]), Questions(ref["y"][new], ref["group_ids"][new])
    mc, xg = "Pair-aware MC-GNN", "XGBoost"
    results["bootstrap"] = {}
    for label, a, b in (("ensemble vs ensemble", ensembles[mc], ensembles[xg]),
                        ("seed mean vs seed mean", seeds[mc], seeds[xg])):
        for subset, q, mask in (("all", all_q, None), ("new facilities", new_q, new)):
            aa = a if mask is None else [x[mask] for x in a]
            bb = b if mask is None else [x[mask] for x in b]
            res = paired_bootstrap(q, aa, bb, RESAMPLES, BOOTSTRAP_SEED)
            results["bootstrap"][f"{label} | {subset}"] = res
            print(f"  MC-GNN minus XGBoost, {label}, {subset}: " + "  ".join(
                f"{k} {r['difference']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]{'*' if r['significant'] else ''}"
                for k, r in res.items()))
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
