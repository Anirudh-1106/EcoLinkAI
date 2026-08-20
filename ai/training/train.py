"""
Training Pipeline for MC-GNN.

Loads graph data from PostgreSQL, trains MC-GNN with HSIC loss,
evaluates metrics against baseline, and saves model checkpoints.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables from backend/.env BEFORE importing any app/ai modules
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import torch

from ai.evaluation.metrics import evaluate_model
from ai.graph.builder import build_industrial_graph
from ai.models.baseline import BaselineRuleModel
from ai.models.mc_gnn import MCGNN
from app.core.database import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")


def train_mc_gnn(
    epochs: int = 100,
    lr: float = 0.01,
    hsic_weight: float = 0.05,
    save_dir: Path | None = None,
):
    """Train MC-GNN model and evaluate against baseline."""
    db = SessionLocal()

    try:
        logger.info("Building industrial graph from PostgreSQL...")
        data, plant_id_to_idx, plants = build_industrial_graph(db)

        if data.x is None or data.edge_index is None or data.edge_attr is None or data.y is None or data.x.size(0) == 0 or data.edge_index.size(1) == 0:
            logger.info("Graph is empty or incomplete. Seeding database first...")
            from scripts.database.seed import seed_database
            seed_database()
            data, plant_id_to_idx, plants = build_industrial_graph(db)

        assert data.x is not None and data.edge_index is not None and data.edge_attr is not None and data.y is not None
        assert isinstance(data.y, torch.Tensor)

        logger.info(
            f"Graph constructed: {data.num_nodes} nodes, "
            f"{data.edge_index.size(1)} edges, {data.x.size(1)} node features"
        )

        in_features = data.x.size(1)
        edge_features = data.edge_attr.size(1)

        # Initialize MC-GNN model
        model = MCGNN(in_features=in_features, edge_dim=edge_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        best_loss = float("inf")

        logger.info("Starting MC-GNN training loop...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            pred_scores, channel_embs = model(data.x, data.edge_index, data.edge_attr)
            loss = model.compute_loss(pred_scores, data.y, channel_embs, hsic_weight=hsic_weight)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if epoch % 20 == 0 or epoch == epochs:
                logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {loss.item():.4f}")

        # ── Evaluation ────────────────────────────────
        model.eval()
        with torch.no_grad():
            final_scores, _ = model(data.x, data.edge_index, data.edge_attr)
            gnn_scores_np = final_scores.cpu().numpy()
            targets_np = data.y.cpu().numpy()

        gnn_metrics = evaluate_model(targets_np, gnn_scores_np, k=5)

        # Baseline evaluation
        baseline_model = BaselineRuleModel()
        baseline_scores_np = baseline_model.predict(data.edge_attr.cpu().numpy())
        baseline_metrics = evaluate_model(targets_np, baseline_scores_np, k=5)

        logger.info("=" * 50)
        logger.info("FINAL MODEL EVALUATION (K=5):")
        logger.info(f"MC-GNN   -> Precision@5: {gnn_metrics['precision@5']:.4f} | Recall@5: {gnn_metrics['recall@5']:.4f} | NDCG@5: {gnn_metrics['ndcg@5']:.4f}")
        logger.info(f"Baseline -> Precision@5: {baseline_metrics['precision@5']:.4f} | Recall@5: {baseline_metrics['recall@5']:.4f} | NDCG@5: {baseline_metrics['ndcg@5']:.4f}")
        logger.info("=" * 50)

        # ── Save Checkpoint ───────────────────────────
        if save_dir is None:
            save_dir = PROJECT_ROOT / "ai" / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / "mc_gnn_best.pt"
        torch.save(model, checkpoint_path)
        logger.info(f"Saved trained model checkpoint to: {checkpoint_path}")

    finally:
        db.close()


if __name__ == "__main__":
    train_mc_gnn(epochs=80)
