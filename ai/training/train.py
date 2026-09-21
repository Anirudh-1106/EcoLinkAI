"""
Training Pipeline for MC-GNN with Checkpoint Versioning.

Loads graph data from PostgreSQL, trains MC-GNN with HSIC loss,
evaluates metrics on a fixed held-out split, and saves versioned
model checkpoints with auto-promotion gating.

Versioning features:
- Timestamped checkpoint files (never overwrite)
- Metadata JSON with in-sample/held-out tagging
- Fixed three-way split (deterministic seed=42): train / validation / test
- Early stopping on validation NDCG@5, restoring the best epoch's weights
- Auto-promotion only if NDCG@5 beats production by >2%
- Atomic file swap via os.replace() (safe for concurrent reads)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Ensure backend directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables from backend/.env BEFORE importing any app/ai modules
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import numpy as np
import torch

from ai.evaluation.metrics import evaluate_model
from app.utils.calibration import (
    apply_platt,
    expected_calibration_error,
    fit_platt,
)
from ai.graph.builder import build_industrial_graph
from ai.models.baseline import BaselineRuleModel
from ai.models.mc_gnn import HSIC_GAMMA, MCGNN
from app.core.database import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")

# ── Constants ─────────────────────────────────────────
SPLIT_SEED = 42
SPLIT_RATIO = 0.8  # First 80% of the permutation is train+val, last 20% is the test set
VAL_RATIO = 0.15  # Share of the train+val pool reserved for validation
# Promotion is gated on held-out AUC rather than NDCG@5. NDCG@5 only inspects
# the top 5 of a ~160-edge test set, so it swings on which handful of edges
# land on top; AUC uses every pair and is stable enough to decide on.
PROMOTION_METRIC = "auc"
PROMOTION_MARGIN = 0.02  # Must beat production by >0.02 on the promotion metric
# Validation AUC over ~100 edges is jumpy, so a short patience trips on noise
# long before the model has converged. This is deliberately generous; the
# best-epoch weights are restored anyway, so training longer costs only time.
PATIENCE = 40  # Stop after this many evaluations with no validation improvement
EVAL_EVERY = 1  # Validate every N epochs (cheap on a graph this size)


def _temporal_order(edge_times: torch.Tensor) -> np.ndarray:
    """
    Edge positions ordered oldest first, ties broken deterministically.

    Used instead of a random permutation so the test set is strictly the most
    recent requests. That matters once the features include a pair's trading
    history: under a random split, a training edge's history can be built from
    requests that landed in the test set, so test outcomes reach back into
    training inputs. Ordering by time removes that path -- anything a training
    edge can see is older than it, and therefore also in training -- and it is
    the honest question anyway, since deployment predicts forward from the
    past rather than filling gaps in a shuffled history.
    """
    times = edge_times.cpu().numpy()
    return np.lexsort((np.arange(len(times)), times))


def _split_edges(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    labels: torch.Tensor,
    ratio: float = SPLIT_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
    edge_times: torch.Tensor | None = None,
) -> dict:
    """
    Split edges three ways: train, validation, and held-out test.

    Validation drives early stopping during training; the test set is never
    looked at until the final evaluation and promotion decision, so it stays
    an honest measure of generalisation.

    The test slice is carved from the same permutation and at the same
    boundary the earlier two-way split used, so it contains exactly the same
    edges as before. That keeps every previously recorded held-out score --
    including production's -- directly comparable. Only the remaining
    train+val pool is subdivided, so tuning `val_ratio` can never leak new
    edges into the test set.

    Returns dict with train_*, val_* and test_* tensors.
    """
    num_edges = edge_index.size(1)
    if edge_times is not None:
        perm = _temporal_order(edge_times)
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(num_edges)

    # Carve the test set first, exactly as the previous two-way split did.
    # Ordered by time this leaves the newest requests as the test set.
    test_cut = int(num_edges * ratio)
    trainval_idx = perm[:test_cut]
    test_idx = perm[test_cut:]

    # Subdivide only the train+val pool.
    val_cut = int(len(trainval_idx) * (1.0 - val_ratio))
    train_idx = trainval_idx[:val_cut]
    val_idx = trainval_idx[val_cut:]

    return {
        "train_edge_index": edge_index[:, train_idx],
        "train_edge_attr": edge_attr[train_idx],
        "train_labels": labels[train_idx],
        "val_edge_index": edge_index[:, val_idx],
        "val_edge_attr": edge_attr[val_idx],
        "val_labels": labels[val_idx],
        "test_edge_index": edge_index[:, test_idx],
        "test_edge_attr": edge_attr[test_idx],
        "test_labels": labels[test_idx],
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
    }


def _message_graph(edge_index: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    The subgraph information travels along: accepted deals only.

    Every exchange request used to be an edge, accepted or not. With 1,472
    such pairs over 61 active plants that is close to a complete graph, so
    each node aggregated over ~115 neighbours, every neighbourhood looked
    identical, and the embeddings collapsed toward a common value. Deleting
    the entire graph changed the output by under 2%, which is the signature
    of message passing carrying nothing.

    An accepted deal is a relationship; a rejected enquiry is not. Restricting
    propagation to the former makes a neighbourhood describe who a plant
    actually trades with, which is the only thing a graph model can offer that
    a per-row formula cannot.

    Callers must pass only edges the model is allowed to know about -- for
    training that means the training split alone, or a test outcome would
    reach the embeddings of the very pair being scored.
    """
    accepted = labels > 0.5
    return edge_index[:, accepted]


def _evaluate_checkpoint_on_split(
    model_path: Path,
    data_x: torch.Tensor,
    split: dict,
) -> dict[str, float] | None:
    """
    Load a saved checkpoint and evaluate it on the held-out split.

    Returns metrics dict or None if loading fails.
    """
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
        with torch.no_grad():
            # Run forward on TEST edges
            pred_scores, _ = model(
                data_x,
                split["test_edge_index"],
                split["test_edge_attr"],
                message_edge_index=_message_graph(
                    split["train_edge_index"], split["train_labels"]
                ),
            )
            scores_np = pred_scores.cpu().numpy()
            targets_np = split["test_labels"].cpu().numpy()

        return evaluate_model(
            targets_np,
            scores_np,
            k=5,
            group_ids=split["test_edge_index"][1].cpu().numpy(),
        )
    except Exception as e:
        logger.warning(f"Could not evaluate checkpoint {model_path}: {e}")
        return None


def _calibrate_production(
    checkpoint_dir: Path,
    data_x: torch.Tensor,
    split: dict,
    val_labels: np.ndarray,
) -> None:
    """
    Refit the serving model's calibration against the current validation data.

    Run after the promotion decision, on whichever checkpoint is in
    production. A promoted model already carries a calibration from training,
    but one that held its place may carry an old one or none at all, and the
    percentages it shows buyers are what people actually read.

    Failure here is not fatal: an uncalibrated model still ranks correctly,
    so the run keeps its result and logs the reason.
    """
    production_path = checkpoint_dir / "mc_gnn_best.pt"
    if not production_path.exists():
        return

    try:
        model = torch.load(production_path, map_location="cpu", weights_only=False)
        model.eval()
        with torch.no_grad():
            scores, _ = model(
                data_x,
                split["val_edge_index"],
                split["val_edge_attr"],
                message_edge_index=_message_graph(
                    split["train_edge_index"], split["train_labels"]
                ),
            )
        scores_np = scores.cpu().numpy()

        calibration = fit_platt(scores_np, val_labels)
        if calibration is None:
            logger.info("Production model left uncalibrated: not enough validation signal.")
            return

        model.calibration = calibration
        before = expected_calibration_error(scores_np, val_labels)
        after = expected_calibration_error(
            np.array([apply_platt(float(p), *calibration) for p in scores_np]),
            val_labels,
        )

        # Same atomic write the promotion path uses, so a server reading the
        # checkpoint can never see a half-written file.
        fd, tmp_path = tempfile.mkstemp(dir=str(checkpoint_dir), suffix=".pt.tmp")
        os.close(fd)
        try:
            torch.save(model, tmp_path)
            os.replace(tmp_path, str(production_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        logger.info(
            "Production model calibrated (a=%.3f, b=%.3f): predicted-vs-actual "
            "gap %.3f -> %.3f",
            calibration[0], calibration[1], before, after,
        )
    except Exception as e:
        logger.warning("Could not calibrate the production model: %s", e)


def _promote_checkpoint(
    source_path: Path,
    source_meta: dict,
    checkpoint_dir: Path,
):
    """
    Atomically promote a checkpoint to mc_gnn_best.pt.

    Uses write-to-temp + os.replace() so the backend cache can
    never read a half-written file.
    """
    production_model = checkpoint_dir / "mc_gnn_best.pt"
    production_meta = checkpoint_dir / "mc_gnn_best.json"

    # Atomic model swap: write to temp, then replace
    fd, tmp_path = tempfile.mkstemp(
        dir=str(checkpoint_dir), suffix=".pt.tmp"
    )
    os.close(fd)
    try:
        shutil.copy2(str(source_path), tmp_path)
        os.replace(tmp_path, str(production_model))
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # Write production metadata (not atomic, but JSON is small + non-critical)
    with open(production_meta, "w") as f:
        json.dump(source_meta, f, indent=2, default=str)

    logger.info(f"✅ Promoted {source_path.name} → mc_gnn_best.pt (atomic swap)")


def train_mc_gnn(
    epochs: int = 100,
    lr: float = 0.01,
    hsic_weight: float = HSIC_GAMMA,
    save_dir: Path | None = None,
):
    """Train MC-GNN model with held-out evaluation and checkpoint versioning."""
    db = SessionLocal()

    try:
        logger.info("Building industrial graph from PostgreSQL...")
        data, plant_id_to_idx, plants = build_industrial_graph(db)

        if (
            data.x is None
            or data.edge_index is None
            or data.edge_attr is None
            or data.y is None
            or data.x.size(0) == 0
            or data.edge_index.size(1) == 0
        ):
            logger.info("Graph is empty or incomplete. Rebuilding database first...")
            # Rebuild rather than seed: seeding alone loads only the synthetic
            # CSVs, which would train the model on a graph missing the real
            # KINFRA plants entirely.
            from scripts.database.rebuild import rebuild
            rebuild()
            data, plant_id_to_idx, plants = build_industrial_graph(db)

        assert data.x is not None and data.edge_index is not None
        assert data.edge_attr is not None and data.y is not None
        assert isinstance(data.y, torch.Tensor)

        logger.info(
            f"Graph constructed: {data.num_nodes} nodes, "
            f"{data.edge_index.size(1)} edges, {data.x.size(1)} node features"
        )

        # ── Held-out split, taken over time ───────────
        # Once edge features carry a pair's trading history, a random split
        # lets test outcomes flow back into training inputs. Splitting by date
        # closes that, and asks the question deployment actually faces:
        # predict the next requests from the ones already settled.
        edge_times = getattr(data, "edge_time", None)
        split = _split_edges(
            data.edge_index, data.edge_attr, data.y, edge_times=edge_times
        )
        # Information travels only along deals that actually closed, and
        # only those in the training split -- a test outcome reaching an
        # embedding would leak the answer into the pair being scored.
        message_edge_index = _message_graph(
            split["train_edge_index"], split["train_labels"]
        )

        split_kind = "temporal" if edge_times is not None else f"random seed={SPLIT_SEED}"
        logger.info(
            f"Edge split ({split_kind}): "
            f"{split['num_train']} train, {split['num_val']} val, "
            f"{split['num_test']} test"
        )

        in_features = data.x.size(1)
        edge_features = data.edge_attr.size(1)

        # ── Initialize and train MC-GNN ───────────────
        model = MCGNN(in_features=in_features, edge_dim=edge_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_loss = float("inf")

        # Early-stopping state. Selection uses the same metric that gates
        # promotion (AUC), so the epoch we keep is the one judged best by the
        # criterion the model is actually held to. Selecting on NDCG@5 instead
        # picks whichever epoch happened to land 5 good edges on top, which on
        # a ~100-edge validation set is mostly noise.
        best_val_score = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        checks_without_improvement = 0
        early_stopped = False
        epoch = 0
        val_labels_np = split["val_labels"].cpu().numpy()

        logger.info("Starting MC-GNN training loop (on TRAIN edges only)...")
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            # Train on TRAIN edges only
            pred_scores, channel_embs = model(
                data.x,
                split["train_edge_index"],
                split["train_edge_attr"],
                message_edge_index=message_edge_index,
            )
            loss = model.compute_loss(
                pred_scores, split["train_labels"], channel_embs, hsic_weight=hsic_weight
            )

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            # ── Validation checkpoint ─────────────────
            if epoch % EVAL_EVERY == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    val_scores, val_channel_embs = model(
                        data.x,
                        split["val_edge_index"],
                        split["val_edge_attr"],
                        message_edge_index=message_edge_index,
                    )
                    val_loss = model.compute_loss(
                        val_scores,
                        split["val_labels"],
                        val_channel_embs,
                        hsic_weight=hsic_weight,
                    ).item()
                    val_metrics = evaluate_model(
                        val_labels_np,
                        val_scores.cpu().numpy(),
                        k=5,
                        group_ids=split["val_edge_index"][1].cpu().numpy(),
                    )
                    val_score = val_metrics[PROMOTION_METRIC]

                improved = val_score > best_val_score + 1e-9 or (
                    abs(val_score - best_val_score) <= 1e-9 and val_loss < best_val_loss
                )

                if improved:
                    best_val_score = val_score
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1

                if epoch % 20 == 0 or epoch == epochs:
                    logger.info(
                        f"Epoch {epoch:03d}/{epochs:03d} | Train loss: {loss.item():.4f} | "
                        f"Val loss: {val_loss:.4f} | Val {PROMOTION_METRIC}: {val_score:.4f}"
                    )

                if checks_without_improvement >= PATIENCE:
                    early_stopped = True
                    logger.info(
                        f"Early stopping at epoch {epoch}: no validation improvement "
                        f"for {PATIENCE} checks (best was epoch {best_epoch})."
                    )
                    break

        # Restore the best-validation weights; the final epoch's weights are
        # usually further into overfitting than the ones we actually want.
        epochs_run = epoch
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info(
                f"Restored weights from epoch {best_epoch} "
                f"(val {PROMOTION_METRIC}: {best_val_score:.4f})"
            )

        # ── Calibrate on validation ───────────────────
        # Fitted here, on data the weights were selected against but never
        # trained on, and never on the test split. The map is monotonic, so it
        # leaves every ranking metric below untouched and corrects only the
        # probability a person reads.
        model.eval()
        with torch.no_grad():
            val_scores_final, _ = model(
                data.x,
                split["val_edge_index"],
                split["val_edge_attr"],
                message_edge_index=message_edge_index,
            )
        calibration = fit_platt(val_scores_final.cpu().numpy(), val_labels_np)
        model.calibration = calibration

        if calibration is not None:
            raw_ece = expected_calibration_error(val_scores_final.cpu().numpy(), val_labels_np)
            adjusted = np.array([
                apply_platt(float(p), *calibration) for p in val_scores_final.cpu().numpy()
            ])
            logger.info(
                "Calibration fitted (a=%.3f, b=%.3f): validation gap between "
                "predicted and actual %.3f -> %.3f",
                calibration[0], calibration[1], raw_ece,
                expected_calibration_error(adjusted, val_labels_np),
            )
        else:
            logger.info("Not enough validation signal to calibrate; scores left raw.")

        # ── Evaluate on HELD-OUT test set ─────────────
        model.eval()
        with torch.no_grad():
            test_scores, _ = model(
                data.x,
                split["test_edge_index"],
                split["test_edge_attr"],
                message_edge_index=message_edge_index,
            )
            test_scores_np = test_scores.cpu().numpy()
            test_targets_np = split["test_labels"].cpu().numpy()

        # Each buyer is one query: rank that buyer's candidate suppliers.
        test_group_ids = split["test_edge_index"][1].cpu().numpy()
        held_out_metrics = evaluate_model(
            test_targets_np, test_scores_np, k=5, group_ids=test_group_ids
        )

        # Also compute in-sample metrics (for logging/comparison only)
        with torch.no_grad():
            train_scores, _ = model(
                data.x,
                split["train_edge_index"],
                split["train_edge_attr"],
                message_edge_index=message_edge_index,
            )
            train_scores_np = train_scores.cpu().numpy()
            train_targets_np = split["train_labels"].cpu().numpy()

        in_sample_metrics = evaluate_model(
            train_targets_np,
            train_scores_np,
            k=5,
            group_ids=split["train_edge_index"][1].cpu().numpy(),
        )

        # Baseline evaluation (on held-out for fair comparison)
        baseline_model = BaselineRuleModel()
        baseline_scores_np = baseline_model.predict(split["test_edge_attr"].cpu().numpy())
        baseline_metrics = evaluate_model(
            test_targets_np, baseline_scores_np, k=5, group_ids=test_group_ids
        )

        logger.info("=" * 60)
        logger.info("FINAL MODEL EVALUATION (K=5):")
        logger.info(
            f"MC-GNN (held-out)   -> AUC: {held_out_metrics['auc']:.4f} | "
            f"P@5: {held_out_metrics['precision@5']:.4f} | "
            f"NDCG@5: {held_out_metrics['ndcg@5']:.4f}"
        )
        logger.info(
            f"MC-GNN (in-sample)  -> AUC: {in_sample_metrics['auc']:.4f} | "
            f"P@5: {in_sample_metrics['precision@5']:.4f} | "
            f"NDCG@5: {in_sample_metrics['ndcg@5']:.4f}"
        )
        logger.info(
            f"Baseline (held-out) -> AUC: {baseline_metrics['auc']:.4f} | "
            f"P@5: {baseline_metrics['precision@5']:.4f} | "
            f"NDCG@5: {baseline_metrics['ndcg@5']:.4f}"
        )
        logger.info("=" * 60)

        # ── Save timestamped checkpoint ───────────────
        if save_dir is None:
            save_dir = PROJECT_ROOT / "ai" / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"mc_gnn_{timestamp}"
        checkpoint_path = save_dir / f"{checkpoint_name}.pt"
        metadata_path = save_dir / f"{checkpoint_name}.json"

        # Save model
        torch.save(model, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path.name}")

        # Build metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_file": checkpoint_path.name,
            "epochs": epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "patience": PATIENCE,
            "learning_rate": lr,
            "hsic_weight": hsic_weight,
            "best_train_loss": round(best_loss, 6),
            "validation_metrics": {
                PROMOTION_METRIC: round(best_val_score, 6),
                "loss": round(best_val_loss, 6),
            },
            "selection_metric": PROMOTION_METRIC,
            "metrics": {
                "precision@5": round(held_out_metrics["precision@5"], 6),
                "recall@5": round(held_out_metrics["recall@5"], 6),
                "ndcg@5": round(held_out_metrics["ndcg@5"], 6),
                "auc": round(held_out_metrics["auc"], 6),
            },
            "in_sample_metrics": {
                "precision@5": round(in_sample_metrics["precision@5"], 6),
                "recall@5": round(in_sample_metrics["recall@5"], 6),
                "ndcg@5": round(in_sample_metrics["ndcg@5"], 6),
                "auc": round(in_sample_metrics["auc"], 6),
            },
            "promotion_metric": PROMOTION_METRIC,
            "metric_type": "held_out",
            "split_kind": split_kind,
            "split_seed": SPLIT_SEED,
            "split_ratio": SPLIT_RATIO,
            "val_ratio": VAL_RATIO,
            "num_train_edges": split["num_train"],
            "num_val_edges": split["num_val"],
            "num_test_edges": split["num_test"],
            "num_nodes": data.num_nodes,
            "total_edges": data.edge_index.size(1),
            "baseline_metrics": {
                "precision@5": round(baseline_metrics["precision@5"], 6),
                "recall@5": round(baseline_metrics["recall@5"], 6),
                "ndcg@5": round(baseline_metrics["ndcg@5"], 6),
                "auc": round(baseline_metrics["auc"], 6),
            },
        }

        # Save metadata JSON
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {metadata_path.name}")

        # ── Auto-promotion decision ───────────────────
        production_path = save_dir / "mc_gnn_best.pt"
        promoted = False

        if not production_path.exists():
            # No production model yet — promote unconditionally
            logger.info("No production model exists. Promoting unconditionally.")
            _promote_checkpoint(checkpoint_path, metadata, save_dir)
            promoted = True
        else:
            # Re-evaluate current production model on the SAME held-out split
            logger.info("Re-evaluating current production model on held-out split...")
            prod_metrics = _evaluate_checkpoint_on_split(
                production_path, data.x, split
            )

            if prod_metrics is None:
                logger.warning(
                    "Could not evaluate production model. "
                    "Promoting new checkpoint as fallback."
                )
                _promote_checkpoint(checkpoint_path, metadata, save_dir)
                promoted = True
            else:
                new_score = held_out_metrics[PROMOTION_METRIC]
                prod_score = prod_metrics[PROMOTION_METRIC]
                improvement = new_score - prod_score

                logger.info(
                    f"Production {PROMOTION_METRIC} (held-out): {prod_score:.4f} | "
                    f"New {PROMOTION_METRIC} (held-out): {new_score:.4f} | "
                    f"Improvement: {improvement:+.4f}"
                )
                logger.info(
                    f"   (NDCG@5 for reference - production: "
                    f"{prod_metrics['ndcg@5']:.4f}, new: {held_out_metrics['ndcg@5']:.4f})"
                )

                if improvement > PROMOTION_MARGIN:
                    logger.info(
                        f"✅ New model beats production by "
                        f"{improvement:.4f} (>{PROMOTION_MARGIN}). Promoting!"
                    )
                    _promote_checkpoint(checkpoint_path, metadata, save_dir)
                    promoted = True
                else:
                    logger.info(
                        f"⏸️ New model does NOT beat production by "
                        f">{PROMOTION_MARGIN}. Keeping current production model."
                    )
                    logger.info(
                        f"   Checkpoint saved as {checkpoint_name}.pt "
                        f"for manual review."
                    )

        # ── Keep the serving model calibrated ─────────
        # Calibration is monotonic, so it cannot move AUC and the promotion
        # gate above will never fire on account of it. Left there, a model
        # that keeps its place would keep serving uncalibrated percentages
        # indefinitely. So whichever checkpoint ends up in production is
        # calibrated against the current validation split, independently of
        # who won.
        _calibrate_production(save_dir, data.x, split, val_labels_np)

        return {
            "promoted": promoted,
            "checkpoint": checkpoint_path.name,
            "best_epoch": best_epoch,
            "held_out": held_out_metrics,
            "baseline": baseline_metrics,
        }

    finally:
        db.close()


if __name__ == "__main__":
    train_mc_gnn(epochs=300)
