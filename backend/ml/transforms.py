"""Tensor normalization helpers shared by training and future inference code."""

from __future__ import annotations

import torch


def normalize_uint8_tensor(images: torch.Tensor) -> torch.Tensor:
    """Convert uint8 CHW or NCHW image tensors from [0,255] to float [-1,1].

    Channel count must be one (edge map) or three (RGB target). The caller is
    responsible for decoding RGB channel order correctly before this function.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.dtype != torch.uint8:
        raise TypeError("normalize_uint8_tensor expects torch.uint8 input")
    if images.ndim not in (3, 4):
        raise ValueError("images must have CHW or NCHW shape")
    channels = images.shape[-3]
    if channels not in (1, 3):
        raise ValueError(f"expected one or three image channels, got {channels}")
    return images.to(dtype=torch.float32).div(127.5).sub(1.0)


def denormalize_to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Clamp normalized CHW/NCHW tensors and convert them to uint8 [0,255]."""
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim not in (3, 4) or images.shape[-3] not in (1, 3):
        raise ValueError("images must have CHW or NCHW shape with one or three channels")
    if not images.is_floating_point():
        raise TypeError("images must be floating point in [-1, 1]")
    return images.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
