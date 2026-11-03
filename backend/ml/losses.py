"""Pix2Pix adversarial and paired reconstruction losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import PIX2PIX_L1_WEIGHT


def _validate_logits(logits: torch.Tensor, name: str) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError(f"{name} must be a rank-4 NCHW logits tensor")
    if not logits.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if logits.shape[0] < 1 or logits.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W], got {tuple(logits.shape)}")


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Average real/fake BCE-with-logits losses for a conditional discriminator."""
    _validate_logits(real_logits, "real_logits")
    _validate_logits(fake_logits, "fake_logits")
    if real_logits.shape != fake_logits.shape:
        raise ValueError("real_logits and fake_logits must have identical shapes")
    if real_logits.device != fake_logits.device or real_logits.dtype != fake_logits.dtype:
        raise ValueError("real_logits and fake_logits must have the same device and dtype")
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)


def generator_loss(
    fake_logits: torch.Tensor,
    generated: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = PIX2PIX_L1_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, adversarial, weighted_l1)`` for the generator.

    ``fake_logits`` must be produced by the discriminator with the generated
    image attached to the generator graph. The caller computes discriminator
    training logits with a detached fake image for its separate update.
    """
    _validate_logits(fake_logits, "fake_logits")
    if not isinstance(generated, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("generated and target must be torch.Tensor values")
    if generated.shape != target.shape or generated.ndim != 4:
        raise ValueError("generated and target must have identical rank-4 NCHW shapes")
    if generated.shape[0] < 1 or generated.shape[1] != 3:
        raise ValueError(f"generated and target must have shape [B, 3, H, W], got {tuple(generated.shape)}")
    if not generated.is_floating_point() or not target.is_floating_point():
        raise TypeError("generated and target must be floating point tensors in [-1, 1]")
    if generated.device != target.device or generated.dtype != target.dtype:
        raise ValueError("generated and target must have the same device and dtype")
    if not math.isfinite(float(l1_weight)) or l1_weight < 0:
        raise ValueError("l1_weight must be finite and non-negative")
    if fake_logits.shape[0] != generated.shape[0]:
        raise ValueError("fake_logits and generated must have the same batch size")
    if fake_logits.device != generated.device or fake_logits.dtype != generated.dtype:
        raise ValueError("fake_logits and generated must have the same device and dtype")

    adversarial = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    weighted_l1 = F.l1_loss(generated, target) * float(l1_weight)
    return adversarial + weighted_l1, adversarial, weighted_l1
