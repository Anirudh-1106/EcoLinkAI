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

# gamma in equation (15): how much the channel-disparity term counts against
# the task loss.
#
# The paper's own setting (Section IV-C), and its sensitivity study in Fig. 3
# is emphatic about the scale: at gamma = 0.1 accuracy collapses to around 30%
# and the model is described as "unusable", while gamma <= 1e-5 improves it.
# The reason is in Section IV-H -- an inner-product HSIC is "normally very
# large", so gamma is what stops it swamping the thing the model is actually
# being trained to do.
#
# This was previously 0.05: half the value the paper reports as unusable, and
# eight orders of magnitude above its setting.
HSIC_GAMMA = 1e-10

# HSIC compares every node with every other: its Gram matrices are N x N. At
# the platform's few hundred plants that is trivial; at the ~20,000 facilities
# of the EPA transfer network it would need several GB and hours per epoch.
# Past this many nodes the disparity is estimated on a fresh random subset
# each step. Graphs at or below it -- the platform's own included -- are
# untouched, and draw no random numbers, so their training is unchanged.
HSIC_MAX_NODES = 2048

# How the first two node inputs, latitude/90 and longitude/180, are read.
#
#   "absolute"  as given. Every model trained on the platform's own graph.
#   "market"    re-centred on the graph being encoded and scaled by its own
#               spread, so they say where a plant sits within its market
#               rather than where on Earth it is.
#
# A model trained on one country's facilities learns that country's
# coordinate range; handed plants from another, every location input falls
# outside anything it saw. Read relative to the market, a plant at the edge of
# the US network and one at the edge of Kerala's look alike, and distance
# itself still reaches the model through the pair features in kilometres.
LOCATION_FRAMES = ("absolute", "market")


def market_relative_location(x: torch.Tensor) -> torch.Tensor:
    """x with its two location columns standardised over the nodes present."""
    loc = x[:, :2]
    spread = loc.std(dim=0, unbiased=False)
    spread = torch.where(spread > 0, spread, torch.ones_like(spread))
    return torch.cat([(loc - loc.mean(dim=0)) / spread, x[:, 2:]], dim=1)


class AttentionFusion(nn.Module):
    """
    Attention-based channel fusion, per equations (10)-(13) of the MC-GNN paper.

    Each channel gets its own transformation W_i, b_i, and a single attention
    vector q is shared across all of them:

        w_i^v = q . sigma(W_i . z_i^v + b_i)

    The per-channel parameters are the point. An earlier version passed every
    channel through one shared MLP, which cannot express a preference between
    them: identical weights applied to similarly-distributed embeddings give
    identical scores, and the softmax returns a flat 1/T. Measured on the
    trained model, the three weights came out 0.334, 0.325 and 0.341 with a
    standard deviation of 0.002 -- an average wearing the name of attention.
    """

    def __init__(self, in_features: int, num_channels: int = 3, channel_dropout: float = 0.0):
        super().__init__()
        hidden = max(in_features // 2, 1)
        self.channel_dropout = channel_dropout

        # W_i and b_i: one transformation per channel.
        self.channel_transforms = nn.ModuleList(
            nn.Linear(in_features, hidden) for _ in range(num_channels)
        )
        # q: shared across channels, so the comparison between them is made
        # on one common axis.
        self.attention_vector = nn.Parameter(torch.empty(hidden))

        # Xavier uniform, as the paper's experimental settings specify.
        for transform in self.channel_transforms:
            nn.init.xavier_uniform_(transform.weight)
            nn.init.zeros_(transform.bias)
        nn.init.uniform_(self.attention_vector, -1.0, 1.0)

    def forward(self, channel_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        channel_embeddings: List of tensors of shape (N, D)
        Returns: Fused embedding tensor of shape (N, D)
        """
        # Stack: (num_channels, N, D)
        stacked = torch.stack(channel_embeddings, dim=0)
        # Weighted sum: (N, D)
        return torch.sum(self.weights(channel_embeddings) * stacked, dim=0)

    def weights(self, channel_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """How much each node leans on each channel: (num_channels, N, 1), summing to 1."""
        # w_i^v = q . sigma(W_i z_i^v + b_i), one score per node per channel.
        # tanh rather than ReLU: a rectifier floors every negative score at
        # exactly zero, so channels scoring below zero become indistinguishable
        # and the softmax flattens between them -- reintroducing the very
        # behaviour this is meant to fix.
        weights = []
        for embedding, transform in zip(channel_embeddings, self.channel_transforms):
            projected = torch.tanh(transform(embedding))       # (N, hidden)
            weights.append(projected @ self.attention_vector)  # (N,)
        weights = torch.stack(weights, dim=0).unsqueeze(-1)    # (num_channels, N, 1)

        # Channel dropout: in training, each node loses one randomly chosen
        # channel with this probability and must be served by the others.
        # Without it, attention on the multi-view model settled 94% on a
        # single channel for nearly every node, leaving the other two
        # untrained -- one channel doing the work of three. Off unless set;
        # checkpoints saved before it existed carry no such attribute.
        p = getattr(self, "channel_dropout", 0.0)
        if self.training and p > 0:
            num_channels, n = weights.shape[0], weights.shape[1]
            nodes = torch.nonzero(torch.rand(n, device=weights.device) < p).squeeze(1)
            dropped = torch.randint(num_channels, (len(nodes),), device=weights.device)
            mask = torch.zeros(num_channels, n, 1, dtype=torch.bool, device=weights.device)
            mask[dropped, nodes] = True
            weights = weights.masked_fill(mask, float("-inf"))

        # Softmax across channels
        return F.softmax(weights, dim=0)  # (num_channels, N, 1)


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
        location_frame: str = "absolute",
    ):
        super().__init__()
        if location_frame not in LOCATION_FRAMES:
            raise ValueError(f"location_frame must be one of {LOCATION_FRAMES}")
        self.hidden_dim = hidden_dim
        self.location_frame = location_frame

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

        # A direct path from a plant's own features to its embedding, one per
        # channel so the channels stay distinct.
        #
        # Without it two rounds of averaging left every plant looking like
        # every other: mean pairwise embedding similarity reached 0.96 against
        # 0.70 for the same model with no graph, and message passing cost
        # 0.048 AUC rather than adding anything. Aggregation should inform a
        # plant's representation, not replace it.
        self.gcn_skip = nn.Linear(in_features, out_dim)
        self.gat_skip = nn.Linear(in_features, out_dim)
        self.sage_skip = nn.Linear(in_features, out_dim)

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
        return self._encode_channels(x, edge_index, edge_index, edge_index)

    def _encode_channels(
        self,
        x: torch.Tensor,
        gcn_edges: torch.Tensor,
        gat_edges: torch.Tensor,
        sage_edges: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run each channel over its own graph and fuse the results."""
        # Checkpoints saved before location_frame existed carry no such
        # attribute and were all trained on absolute coordinates.
        if getattr(self, "location_frame", "absolute") == "market":
            x = market_relative_location(x)

        # Each channel aggregates over the graph, then adds the plant's own
        # features back so two rounds of averaging cannot wash them away.
        # Channel 1: GCN
        z1 = F.relu(self.gcn_conv1(x, gcn_edges))
        z1 = self.gcn_conv2(z1, gcn_edges) + self.gcn_skip(x)

        # Channel 2: GAT
        z2 = F.relu(self.gat_conv1(x, gat_edges))
        z2 = self.gat_conv2(z2, gat_edges) + self.gat_skip(x)

        # Channel 3: GraphSAGE
        z3 = F.relu(self.sage_conv1(x, sage_edges))
        z3 = self.sage_conv2(z3, sage_edges) + self.sage_skip(x)

        channel_embeddings = [z1, z2, z3]

        # Fusion
        z_fused = self.fusion(channel_embeddings)

        return z_fused, channel_embeddings

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        message_edge_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for link prediction.

        edge_index is the set of pairs being scored. message_edge_index is the
        graph information travels along, and is deliberately allowed to differ:
        a pair is worth scoring whether or not the two have ever dealt, but
        only a real trade should carry a plant's reputation to its neighbours.

        Passing every enquiry as the graph -- the two being the same tensor --
        connected each active plant to nearly all the others, so every
        neighbourhood looked alike and message passing had nothing to
        distinguish anyone by. Defaults to edge_index when omitted.
        """
        if message_edge_index is None:
            message_edge_index = edge_index

        z_fused, channel_embs = self.encode(x, message_edge_index)

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
        hsic_weight: float = HSIC_GAMMA,
    ) -> torch.Tensor:
        """
        Combined loss: task loss + gamma * channel-disparity loss.

        Equation (15) of the paper, L = L_task + gamma * L_disparity.
        """
        bce = F.binary_cross_entropy(pred_scores, targets)

        # HSIC regularization across pairs of channels
        z1, z2, z3 = channel_embs
        n = z1.size(0)
        if n > HSIC_MAX_NODES:
            idx = torch.randperm(n, device=z1.device)[:HSIC_MAX_NODES]
            z1, z2, z3 = z1[idx], z2[idx], z3[idx]
        h_loss = (
            hsic_loss(z1, z2) + hsic_loss(z1, z3) + hsic_loss(z2, z3)
        ) / 3.0

        return bce + hsic_weight * h_loss


# The three networks a multi-view MC-GNN reads, in the order it expects them.
# Each is one reason facilities end up trading, as the industrial symbiosis
# literature frames it:
#
#   relationship  who has traded with whom -- the social tie
#   capability    whose waste another facility can process -- technical fit
#   peer          which facilities resemble each other -- "firms like you"
VIEW_NAMES = ("relationship", "capability", "peer")


class MultiViewMCGNN(MCGNN):
    """
    An MC-GNN whose channels differ in what they read, not only in how.

    MCGNN runs GCN, GAT and GraphSAGE over one graph, so its channels are
    three readings of the same evidence -- and a single GCN, measured on real
    recovery trade, does as well as all three. Here each channel has a network
    of its own and the backbone suited to it:

      GAT        relationship  partners differ in how much they matter, and
                               attention learns which
      GraphSAGE  capability    very large neighbourhoods (every facility
                               handling a common chemical), which mean
                               aggregation and a separate self-weight handle,
                               and which a new facility joins from its first
                               listing
      GCN        peer          similar facilities count roughly equally, so
                               a normalised average suits

    HSIC now keeps three genuinely different viewpoints independent, and the
    attention fusion chooses per facility which one to lean on -- a facility
    with no trading history has nothing on its relationship channel.

    message_edge_index is a sequence of three edge indices in VIEW_NAMES order.
    """

    views = VIEW_NAMES

    def encode(self, x: torch.Tensor, message_edge_index) -> tuple[torch.Tensor, list[torch.Tensor]]:
        relationship, capability, peer = message_edge_index
        return self._encode_channels(x, gcn_edges=peer, gat_edges=relationship, sage_edges=capability)

    @torch.no_grad()
    def channel_weights(self, x: torch.Tensor, message_edge_index) -> torch.Tensor:
        """Attention on each view per node, (N, 3) in VIEW_NAMES order."""
        _, (z_peer, z_rel, z_cap) = self.encode(x, message_edge_index)
        w = self.fusion.weights([z_peer, z_rel, z_cap]).squeeze(-1)  # GCN, GAT, SAGE order
        return torch.stack([w[1], w[2], w[0]], dim=1)


# ── Pair-aware MC-GNN ───────────────────────────────────────────────────


class PairPooling(nn.Module):
    """
    Scores a pair from what the two facilities are and what they share.

    Neural Common Neighbor (Wang, Yang and Zhang, ICLR 2024): one
    message-passing run, then per pair the two embeddings, their Hadamard
    product, and the mean embedding of each pair-dependent node set, with the
    pair's edge features and the sets' sizes. A node-wise scorer -- the
    concatenation MCGNN uses -- cannot represent what two nodes share; this
    can.

    Sets arrive as (rows, k) index tensors padded with -1. Pooling is a sparse
    product, so memory grows with the members present, not rows x k x dim.
    """

    def __init__(self, out_dim: int, edge_dim: int, num_sets: int, num_counts: int):
        super().__init__()
        in_dim = out_dim * (3 + num_sets) + edge_dim + num_counts
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
        )

    @staticmethod
    def _mean_of(z: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
        present = members >= 0
        rows = torch.arange(members.size(0), device=z.device).unsqueeze(1).expand_as(members)[present]
        size = present.sum(dim=1).clamp_min(1).to(z.dtype)
        weights = torch.sparse_coo_tensor(
            torch.stack([rows, members[present].long()]), 1.0 / size[rows],
            (members.size(0), z.size(0)),
        )
        return torch.sparse.mm(weights, z)

    def forward(self, z, edge_index, edge_attr, pair_sets, pair_counts):
        src, dst = edge_index
        parts = [z[src], z[dst], z[src] * z[dst]]
        parts += [self._mean_of(z, members) for members in pair_sets]
        return self.mlp(torch.cat(parts + [edge_attr, pair_counts], dim=-1)).squeeze(-1)


class PairAwareMCGNN(MultiViewMCGNN):
    """
    The multi-view MC-GNN with a pair-aware scorer and normalised HSIC.

    The encoder is unchanged: GAT on relationships, GraphSAGE on capability,
    GCN on peers, skip connections, attention fusion with channel dropout.
    Two things differ:

      scorer  PairPooling in place of the concatenation head, reading the
              pair sets of ai/experiments/tri_pairs.py
      HSIC    normalised (linear CKA, ai.models.hsic.normalized_hsic), so the
              disparity term is scale-free and its weight acts. Measured on
              the earlier models, the raw term at gamma = 1e-10 was about
              1e-11 against a cross-entropy near 0.07: it never shaped
              training.

    forward takes pair_sets (a sequence of (rows, k) index tensors) and
    pair_counts (rows, num_counts) beside the usual arguments.
    """

    def __init__(self, in_features: int, edge_dim: int, num_sets: int = 2, num_counts: int = 2,
                 hsic_weight: float = 0.1, channel_dropout: float = 0.3, **kwargs):
        super().__init__(in_features, edge_dim=edge_dim, **kwargs)
        self.fusion.channel_dropout = channel_dropout
        self.hsic_weight = hsic_weight
        out_dim = self.gcn_conv2.out_channels
        self.link_predictor = PairPooling(out_dim, edge_dim, num_sets, num_counts)

    def forward(self, x, edge_index, edge_attr, message_edge_index=None, pair_sets=(), pair_counts=None):
        z, channel_embs = self.encode(x, message_edge_index)
        if pair_counts is None:
            pair_counts = edge_attr.new_zeros((edge_attr.size(0), 0))
        return self.link_predictor(z, edge_index, edge_attr, pair_sets, pair_counts), channel_embs

    def compute_loss(self, pred_scores, targets, channel_embs, hsic_weight=None):
        from ai.models.hsic import normalized_hsic

        weight = self.hsic_weight if hsic_weight is None else hsic_weight
        bce = F.binary_cross_entropy(pred_scores, targets)
        if not weight:
            return bce
        z1, z2, z3 = channel_embs
        if z1.size(0) > HSIC_MAX_NODES:
            idx = torch.randperm(z1.size(0), device=z1.device)[:HSIC_MAX_NODES]
            z1, z2, z3 = z1[idx], z2[idx], z3[idx]
        disparity = (normalized_hsic(z1, z2) + normalized_hsic(z1, z3) + normalized_hsic(z2, z3)) / 3.0
        return bce + weight * disparity
