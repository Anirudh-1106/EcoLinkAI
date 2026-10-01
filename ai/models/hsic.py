"""
Hilbert-Schmidt Independence Criterion (HSIC) Regularization.

Measures how dependent two channels' embeddings are. Minimising it pushes the
channels apart, so each backbone is forced to encode something the others do
not -- which is the whole reason for having several of them.

Follows Cui, Li and Zhang, "MC-GNN: Multi-Channel Graph Neural Networks With
Hilbert-Schmidt Independence Criterion", IEEE TBD 11(4), 2025, equations
(6)-(9).

The Gram matrices are inner products, as equations (7) and (8) define them.
An earlier version used a Gaussian RBF kernel with a fixed sigma of 1.0
instead. On embeddings whose pairwise distances run around 0.2 that kernel
saturates near 0.95, every entry looks alike, and HSIC collapses to about
0.0003 -- small enough that the regulariser contributed nothing and the model
was effectively the paper's MC-GNN-minus-HSIC ablation, which Table III of
that paper reports as performing worse than a single GNN backbone.
"""

from __future__ import annotations

import torch


def inner_product_kernel(Z: torch.Tensor) -> torch.Tensor:
    """
    Gram matrix of inner products, K[a][b] = <z_a, z_b>.

    Equations (7) and (8) of the paper.
    """
    return Z @ Z.t()


def hsic_loss(Z1: torch.Tensor, Z2: torch.Tensor) -> torch.Tensor:
    """
    Dependence between two channel embeddings.

    HSIC(Z1, Z2) = 1/(N-1)^2 * tr(K1 H K2 H), with H = I - (1/N) 11^T the
    centering matrix -- equations (6) and (9).

    Returns a large positive number by construction: inner-product kernels
    are unnormalised, so the value scales with the embeddings themselves.
    That is expected, and is why the paper weights this term by a gamma
    around 1e-10 rather than something near 1. Section IV-H puts it plainly:
    "the HSIC loss is normally very large", so gamma exists to stop it
    drowning out the task loss.
    """
    N = Z1.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=Z1.device)

    K1 = inner_product_kernel(Z1)
    K2 = inner_product_kernel(Z2)

    H = torch.eye(N, device=Z1.device) - (1.0 / N) * torch.ones((N, N), device=Z1.device)

    K1_centered = H @ K1 @ H
    K2_centered = H @ K2 @ H

    hsic = torch.trace(K1_centered @ K2_centered) / ((N - 1) ** 2)
    return torch.clamp(hsic, min=0.0)


def normalized_hsic(Z1: torch.Tensor, Z2: torch.Tensor) -> torch.Tensor:
    """
    HSIC scaled to 0-1: HSIC(Z1, Z2) / sqrt(HSIC(Z1, Z1) * HSIC(Z2, Z2)).

    With inner-product kernels this is linear centered kernel alignment (CKA;
    Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019): 0 when the channels share no linear structure, 1
    when one is a rotation and rescaling of the other.

    The unnormalised term above grows with the square of the embeddings'
    magnitude in each channel, so a model can lower it by shrinking its
    embeddings without making the channels any more different -- and its
    weight has to sit near 1e-10 to stay below the task loss. Normalised, the
    value cannot be reduced by rescaling, a weight between 0.01 and 1 is
    meaningful, and it states directly how alike two channels are.

    Computed through the d x d cross-covariance rather than the N x N Gram
    matrices -- the same quantity, since tr(K1 H K2 H) = ||Z1c^T Z2c||_F^2 --
    at O(N d^2) instead of O(N^2 d).
    """
    if Z1.size(0) <= 1:
        return torch.tensor(0.0, device=Z1.device)
    a = Z1 - Z1.mean(dim=0, keepdim=True)
    b = Z2 - Z2.mean(dim=0, keepdim=True)
    cross = (a.t() @ b).pow(2).sum()
    self_a = (a.t() @ a).pow(2).sum()
    self_b = (b.t() @ b).pow(2).sum()
    return cross / torch.sqrt(self_a * self_b).clamp_min(1e-12)
