"""
Multi-Channel Graph Neural Network (MC-GNN) Architecture.

Implements the multi-channel GNN framework with:
- Multiple GNN backbones (GCN, GAT, GraphSAGE) as distinct channels
- Hilbert-Schmidt Independence Criterion (HSIC) to enlarge disparity between channels
- Attention-based channel fusion mechanism
- Link prediction head for partner matching
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from ai.models.hsic import hsic_loss


class AttentionFusion(nn.Module):
    """Attention-based channel fusion module."""

    def __init__(self, in_features: int, num_channels: int = 3):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, channel_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        channel_embeddings: List of tensors of shape (N, D)
        Returns: Fused embedding tensor of shape (N, D)
        """
        # Stack: (num_channels, N, D)
        stacked = torch.stack(channel_embeddings, dim=0)

        # Compute weights for each node and channel: (num_channels, N, 1)
        weights = []
        for emb in channel_embeddings:
            weights.append(self.attn_mlp(emb))
        weights = torch.stack(weights, dim=0)  # (num_channels, N, 1)

        # Softmax across channels
        attn_weights = F.softmax(weights, dim=0)  # (num_channels, N, 1)

        # Weighted sum: (N, D)
        fused = torch.sum(attn_weights * stacked, dim=0)
        return fused


class MCGNN(nn.Module):
    """
    Multi-Channel Graph Neural Network (MC-GNN).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        edge_dim: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Platt scaling parameters (a, b) fitted on validation after training,
        # mapping the link predictor's raw score to an honest probability. The
        # model ranks well but reads low -- see ai/evaluation/calibration.py.
        # None until fitted, and left None when the data is too thin to
        # support a fit, in which case callers use the raw score.
        self.calibration: tuple[float, float] | None = None

        # Channel 1: Graph Convolution Network (GCN)
        self.gcn_conv1 = GCNConv(in_features, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, out_dim)

        # Channel 2: Graph Attention Network (GAT)
        self.gat_conv1 = GATConv(in_features, hidden_dim // 2, heads=2)
        self.gat_conv2 = GATConv(hidden_dim, out_dim, heads=1)

        # Channel 3: GraphSAGE
        self.sage_conv1 = SAGEConv(in_features, hidden_dim)
        self.sage_conv2 = SAGEConv(hidden_dim, out_dim)

        # Attention Fusion
        self.fusion = AttentionFusion(out_dim, num_channels=3)

        # Edge Link Prediction Head (takes concatenated embeddings of src & dst + edge features)
        self.link_predictor = nn.Sequential(
            nn.Linear(out_dim * 2 + edge_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode nodes through 3 channels and fuse them."""
        # Channel 1: GCN
        z1 = F.relu(self.gcn_conv1(x, edge_index))
        z1 = self.gcn_conv2(z1, edge_index)

        # Channel 2: GAT
        z2 = F.relu(self.gat_conv1(x, edge_index))
        z2 = self.gat_conv2(z2, edge_index)

        # Channel 3: GraphSAGE
        z3 = F.relu(self.sage_conv1(x, edge_index))
        z3 = self.sage_conv2(z3, edge_index)

        channel_embeddings = [z1, z2, z3]

        # Fusion
        z_fused = self.fusion(channel_embeddings)

        return z_fused, channel_embeddings

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass for link prediction."""
        z_fused, channel_embs = self.encode(x, edge_index)

        src, dst = edge_index[0], edge_index[1]
        src_emb = z_fused[src]
        dst_emb = z_fused[dst]

        # Combine src, dst, and edge features
        edge_features = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        pred_scores = self.link_predictor(edge_features).squeeze(-1)

        return pred_scores, channel_embs

    def compute_loss(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        channel_embs: list[torch.Tensor],
        hsic_weight: float = 0.05,
    ) -> torch.Tensor:
        """Compute combined loss = BCE_loss + hsic_weight * HSIC_regularization."""
        bce = F.binary_cross_entropy(pred_scores, targets)

        # HSIC regularization across pairs of channels
        z1, z2, z3 = channel_embs
        h_loss = (
            hsic_loss(z1, z2) + hsic_loss(z1, z3) + hsic_loss(z2, z3)
        ) / 3.0

        return bce + hsic_weight * h_loss
