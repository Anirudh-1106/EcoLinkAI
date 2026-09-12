"""
In-memory cache of the industrial graph and its MC-GNN node embeddings.

Building the graph from PostgreSQL costs ~2s and encoding it through the
GNN another ~0.1s, which is far too slow to repeat on every recommendation
request. Both are built once here and refreshed on a TTL, after which
scoring a single candidate pair is only a ~3ms link-predictor call.

If torch / torch-geometric / the checkpoint are unavailable, every accessor
returns None and callers fall back to baseline scoring.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import BASE_DIR, settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_state: dict = {
    "model": None,
    "embeddings": None,       # torch.Tensor of shape (num_nodes, out_dim)
    "plant_id_to_idx": {},
    "built_at": 0.0,
    "node_count": 0,
    "edge_count": 0,
    "model_loaded": False,
    "load_failed": False,
    "model_mtime": None,      # checkpoint mtime when loaded, to detect promotions
    "last_error": None,
    "metrics": None,          # evaluation of MC-GNN vs baseline, refreshed with the graph
}


def _ensure_ai_importable() -> None:
    """Put the repo root on sys.path so the sibling `ai` package can be imported."""
    repo_root = str(BASE_DIR.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _checkpoint_mtime() -> float | None:
    """Modification time of the production checkpoint, or None if absent."""
    try:
        path = Path(settings.MODEL_PATH) / "mc_gnn_best.pt"
        return path.stat().st_mtime if path.exists() else None
    except OSError:
        return None


def _load_model():
    """
    Load the trained MC-GNN checkpoint. Returns the model or None.

    Reloads when the checkpoint file on disk has changed, so a model promoted
    by a retraining run is picked up by the next cache refresh instead of
    needing the server restarted. Without this, automated promotion would
    never actually reach production.
    """
    current_mtime = _checkpoint_mtime()

    if _state["model"] is not None:
        if current_mtime == _state["model_mtime"]:
            return _state["model"]
        logger.info("MC-GNN checkpoint changed on disk; reloading.")
        _state["model"] = None

    # Don't retry (and re-log) a known-bad load on every scored candidate,
    # unless the checkpoint itself has since changed. invalidate() also clears
    # this so a newly added checkpoint can be picked up.
    if _state["load_failed"] and current_mtime == _state["model_mtime"]:
        return None

    try:
        _ensure_ai_importable()
        import torch

        model_path = Path(settings.MODEL_PATH) / "mc_gnn_best.pt"
        if not model_path.exists():
            _state["last_error"] = f"checkpoint not found at {model_path}"
            _state["load_failed"] = True
            _state["model_mtime"] = None
            logger.info("MC-GNN checkpoint not found at %s, using baseline", model_path)
            return None

        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
        _state["model"] = model
        _state["model_loaded"] = True
        _state["load_failed"] = False
        _state["model_mtime"] = current_mtime
        _state["last_error"] = None
        logger.info("MC-GNN model loaded successfully from %s", model_path)
        return model
    except Exception as e:  # torch missing, unpickling failure, etc.
        _state["last_error"] = str(e)
        _state["load_failed"] = True
        logger.warning("Could not load MC-GNN model: %s. Using baseline.", e)
        return None


def _evaluate(model, data) -> dict | None:
    """
    Measure ranking quality of the MC-GNN against the rule-based baseline
    on the historical exchange-request edges.

    These are in-sample figures (the model was trained on these same edges),
    so treat them as a sanity check rather than held-out generalisation.
    """
    try:
        import torch

        from ai.evaluation.metrics import evaluate_model
        from ai.models.baseline import BaselineRuleModel

        with torch.no_grad():
            gnn_scores, _ = model(data.x, data.edge_index, data.edge_attr)

        y_true = data.y.numpy()
        gnn = gnn_scores.numpy()
        baseline = BaselineRuleModel().predict(data.edge_attr.numpy())

        gnn_metrics = evaluate_model(y_true, gnn, k=5)
        baseline_metrics = evaluate_model(y_true, baseline, k=5)

        return {
            "precision_at_5": round(gnn_metrics["precision@5"], 4),
            "recall_at_5": round(gnn_metrics["recall@5"], 4),
            "ndcg_at_5": round(gnn_metrics["ndcg@5"], 4),
            "baseline_ndcg_at_5": round(baseline_metrics["ndcg@5"], 4),
            "training_samples": int(data.edge_index.size(1)),
        }
    except Exception as e:
        logger.warning("Could not evaluate MC-GNN metrics: %s", e)
        return None


def metrics() -> dict | None:
    """
    Evaluation metrics for the model in production.

    Prefers the held-out figures recorded when the checkpoint was promoted,
    since those measure generalisation to edges the model never trained on.
    Falls back to the in-sample evaluation only when a checkpoint predates
    the versioned training pipeline and carries no metadata of its own.
    """
    promoted = _promoted_metadata()
    if promoted:
        recorded = promoted.get("metrics") or {}
        baseline = promoted.get("baseline_metrics") or {}
        if recorded:
            return {
                "precision_at_5": recorded.get("precision@5", 0.0),
                "recall_at_5": recorded.get("recall@5", 0.0),
                "ndcg_at_5": recorded.get("ndcg@5", 0.0),
                "baseline_ndcg_at_5": baseline.get("ndcg@5", 0.0),
                "training_samples": promoted.get("num_train_edges", 0),
                "metric_type": promoted.get("metric_type", "held_out"),
            }

    in_sample = _state["metrics"]
    if in_sample:
        return {**in_sample, "metric_type": "in_sample"}
    return None


def _promoted_metadata() -> dict | None:
    """Metadata written alongside the currently promoted checkpoint."""
    try:
        path = Path(settings.MODEL_PATH) / "mc_gnn_best.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read promoted checkpoint metadata: %s", e)
        return None


def checkpoint_mtime() -> str | None:
    """Last-modified date of the loaded checkpoint, as an ISO date string."""
    try:
        path = Path(settings.MODEL_PATH) / "mc_gnn_best.pt"
        if not path.exists():
            return None
        return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
    except Exception:
        return None


def _rebuild(db: Session) -> bool:
    """Rebuild the graph and recompute node embeddings. Returns True on success."""
    model = _load_model()
    if model is None:
        return False

    try:
        _ensure_ai_importable()
        import torch

        from ai.graph.builder import build_industrial_graph

        started = time.time()
        data, plant_id_to_idx, _plants = build_industrial_graph(db)

        with torch.no_grad():
            embeddings, _channels = model.encode(data.x, data.edge_index)

        _state["embeddings"] = embeddings
        _state["plant_id_to_idx"] = plant_id_to_idx
        _state["built_at"] = time.time()
        _state["node_count"] = int(data.x.size(0))
        _state["edge_count"] = int(data.edge_index.size(1))
        _state["metrics"] = _evaluate(model, data)
        _state["last_error"] = None

        logger.info(
            "MC-GNN graph cache rebuilt: %d nodes, %d edges in %.2fs",
            _state["node_count"],
            _state["edge_count"],
            time.time() - started,
        )
        return True
    except Exception as e:
        _state["last_error"] = str(e)
        logger.warning("Failed to rebuild MC-GNN graph cache: %s. Using baseline.", e)
        return False


def _is_stale() -> bool:
    if _state["embeddings"] is None:
        return True
    return (time.time() - _state["built_at"]) > settings.GRAPH_CACHE_TTL_SECONDS


def get_context(db: Session) -> dict | None:
    """
    Return the inference context, rebuilding it if stale.

    Returns a dict with `model`, `embeddings` and `plant_id_to_idx`,
    or None when GNN inference is unavailable.
    """
    with _lock:
        if _is_stale() and not _rebuild(db):
            return None

        if _state["embeddings"] is None or _state["model"] is None:
            return None

        return {
            "model": _state["model"],
            "embeddings": _state["embeddings"],
            "plant_id_to_idx": _state["plant_id_to_idx"],
        }


def warm(db: Session) -> bool:
    """Eagerly build the cache (called at application startup)."""
    with _lock:
        return _rebuild(db)


def invalidate() -> None:
    """Force the next request to rebuild the graph and retry a failed model load."""
    with _lock:
        _state["built_at"] = 0.0
        _state["load_failed"] = False


def status() -> dict:
    """Observability snapshot of the cache, safe to expose via the API."""
    built_at = _state["built_at"]
    return {
        "model_loaded": _state["model_loaded"],
        "graph_cached": _state["embeddings"] is not None,
        "node_count": _state["node_count"],
        "edge_count": _state["edge_count"],
        "age_seconds": round(time.time() - built_at, 1) if built_at else None,
        "ttl_seconds": settings.GRAPH_CACHE_TTL_SECONDS,
        "last_error": _state["last_error"],
    }
