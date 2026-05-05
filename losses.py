"""Loss functions for latent-reasoning ASR training."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fmt(x: torch.Tensor) -> str:
    """Format a one-dimensional tensor for logging."""
    return " | ".join([f"d{k}:{x[k].item():.3f}" for k in range(x.numel())])


def contrastive_loss(view_a: torch.Tensor, view_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE contrastive loss over batch (symmetric)."""
    assert view_a.dim() == 2 and view_b.dim() == 2, "contrastive_loss expects (B, D) inputs"
    B = view_a.size(0)
    if B <= 1:
        return view_a.new_tensor(0.0)
    z1 = F.normalize(view_a, dim=-1)
    z2 = F.normalize(view_b, dim=-1)
    logits = torch.matmul(z1, z2.t()) / float(temperature)
    labels = torch.arange(B, device=view_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


def trajectory_regularization_loss(thoughts: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Trajectory regularization loss.

    Encodes the intuition that the latent reasoning trajectory should:
    1. Stay anchored to the initial acoustic state h_0 (semantic grounding)
    2. Evolve smoothly between consecutive steps (temporal coherence)

    Args:
        thoughts: shape (B, N, D), where thoughts[:, 0, :] is the initial state h_0
        alpha: weight balancing anchor vs smoothness (default: 0.5)

    Returns:
        Scalar loss combining anchor and smoothness terms
    """
    N = thoughts.size(1)
    # Guard: empty trajectory (e.g. n_latent=0) – return 0.
    if N == 0:
        return thoughts.new_tensor(0.0)

    # Extract initial acoustic anchor h_0
    h_0 = thoughts[:, 0:1, :]  # shape: (B, 1, D)

    # 1. Acoustic Anchor Loss (distance of all states from h_0)
    anchor_loss = (thoughts - h_0).pow(2).sum(dim=-1).mean()

    # 2. Step-wise Smoothness Loss (distance between adjacent states).
    # When N==1 (e.g. DEQ single fixed-point), there are no consecutive pairs,
    # so smooth_loss is zero – avoids mean() on an empty tensor returning NaN.
    if N > 1:
        diffs = thoughts[:, 1:, :] - thoughts[:, :-1, :]  # shape: (B, N-1, D)
        smooth_loss = diffs.pow(2).sum(dim=-1).mean()
    else:
        smooth_loss = thoughts.new_tensor(0.0)

    # Combine final regularization loss
    total_trajectory_loss = alpha * anchor_loss + (1.0 - alpha) * smooth_loss
    return total_trajectory_loss
