"""
Hilbert-Schmidt Independence Criterion (HSIC) Regularization.

Enforces independence/disparity between representations learned
by different channels in MC-GNN, preventing feature redundancy
and over-squashing as detailed in the MC-GNN reference paper.
"""

from __future__ import annotations

import torch


def rbf_kernel(K: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute Gaussian RBF Gram matrix."""
    n = K.size(0)
    dist_matrix = torch.cdist(K, K, p=2) ** 2
    return torch.exp(-dist_matrix / (2 * sigma ** 2))


def hsic_loss(Z1: torch.Tensor, Z2: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Hilbert-Schmidt Independence Criterion between two channel embeddings Z1 and Z2.

    HSIC(Z1, Z2) = (1 / (N-1)^2) * Tr(K1 * H * K2 * H)
    where H = I - (1/N) * 11^T is the centering matrix.
    """
    N = Z1.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=Z1.device)

    K1 = rbf_kernel(Z1, sigma)
    K2 = rbf_kernel(Z2, sigma)

    H = torch.eye(N, device=Z1.device) - (1.0 / N) * torch.ones((N, N), device=Z1.device)

    K1_centered = torch.matmul(H, torch.matmul(K1, H))
    K2_centered = torch.matmul(H, torch.matmul(K2, H))

    hsic = torch.trace(torch.matmul(K1_centered, K2_centered)) / ((N - 1) ** 2)
    return torch.clamp(hsic, min=0.0)
