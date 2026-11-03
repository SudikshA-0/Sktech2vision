

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import (
    PIX2PIX_BASE_CHANNELS,
    PIX2PIX_INPUT_CHANNELS,
    PIX2PIX_NUM_DOWNSAMPLES,
    PIX2PIX_OUTPUT_CHANNELS,
)


def _validate_float_nchw(tensor: torch.Tensor, channels: int, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have NCHW rank 4, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] < 1 or tensor.shape[1] != channels:
        raise ValueError(f"{name} must have shape [B, {channels}, H, W], got {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point and normalized to [-1, 1]")
    if tensor.shape[2] < 1 or tensor.shape[3] < 1:
        raise ValueError(f"{name} height and width must be positive")


class UNetGenerator(nn.Module):
    """U-Net mapping normalized one-channel edges to normalized RGB images.

    Inputs and outputs are NCHW floating tensors in [-1, 1]. ``num_downs``
    controls the spatial divisor; dimensions must be divisible by ``2**num_downs``.
    The default seven stages support the 128x128 project baseline.
    """

    def __init__(
        self,
        base_channels: int = PIX2PIX_BASE_CHANNELS,
        num_downs: int = PIX2PIX_NUM_DOWNSAMPLES,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if num_downs < 2:
            raise ValueError("num_downs must be at least 2")
        self.base_channels = int(base_channels)
        self.num_downs = int(num_downs)
        channels = [min(self.base_channels * (2**index), 512) for index in range(self.num_downs)]

        self.encoders = nn.ModuleList()
        in_channels = PIX2PIX_INPUT_CHANNELS
        for index, out_channels in enumerate(channels):
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=index in (0, self.num_downs - 1))
            ]
            if index not in (0, self.num_downs - 1):
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.encoders.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.decoders = nn.ModuleList()
        current_channels = channels[-1]
        decoder_index = 0
        for skip_index in reversed(range(self.num_downs - 1)):
            out_channels = channels[skip_index]
            layers = [
                nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if decoder_index < 3:
                layers.append(nn.Dropout(p=0.5))
            self.decoders.append(nn.Sequential(*layers))
            current_channels = out_channels * 2  # concatenated with the matching encoder skip
            decoder_index += 1

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                current_channels,
                PIX2PIX_OUTPUT_CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def _validate_spatial_size(self, height: int, width: int) -> None:
        divisor = 2**self.num_downs
        if height < divisor or width < divisor or height % divisor or width % divisor:
            raise ValueError(
                f"height and width must be at least and divisible by 2**num_downs ({divisor}); "
                f"got {height}x{width}"
            )

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        _validate_float_nchw(edges, PIX2PIX_INPUT_CHANNELS, "edges")
        self._validate_spatial_size(edges.shape[2], edges.shape[3])
        encoded: list[torch.Tensor] = []
        features = edges
        for index, encoder in enumerate(self.encoders):
            features = encoder(features)
            if index < self.num_downs - 1:
                encoded.append(features)

        for decoder, skip in zip(self.decoders, reversed(encoded)):
            features = decoder(features)
            if features.shape[2:] != skip.shape[2:]:
                raise RuntimeError("U-Net skip tensor dimensions do not align")
            features = torch.cat((features, skip), dim=1)
        return self.output_layer(features)


class PatchGANDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN; returns raw patch logits (no sigmoid)."""

    def __init__(self, base_channels: int = PIX2PIX_BASE_CHANNELS) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        self.base_channels = int(base_channels)
        widths = [min(self.base_channels * (2**index), 512) for index in range(4)]
        layers: list[nn.Module] = [
            nn.Conv2d(PIX2PIX_INPUT_CHANNELS + PIX2PIX_OUTPUT_CHANNELS, widths[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        previous = widths[0]
        for width in widths[1:3]:
            layers.extend((
                nn.Conv2d(previous, width, 4, 2, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            previous = width
        layers.extend((
            nn.Conv2d(previous, widths[3], 4, 1, 1, bias=False),
            nn.BatchNorm2d(widths[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(widths[3], 1, 4, 1, 1),
        ))
        self.network = nn.Sequential(*layers)

    def forward(self, edges: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        """Accept ``(edges, RGB image)`` or a pre-concatenated four-channel pair."""
        if image is None:
            if not isinstance(edges, torch.Tensor) or edges.ndim != 4 or edges.shape[1] != 4:
                raise ValueError("combined discriminator input must have shape [B, 4, H, W]")
            pair = edges
            if not pair.is_floating_point():
                raise TypeError("discriminator input must be floating point and normalized to [-1, 1]")
        else:
            _validate_float_nchw(edges, PIX2PIX_INPUT_CHANNELS, "edges")
            _validate_float_nchw(image, PIX2PIX_OUTPUT_CHANNELS, "image")
            if edges.shape[0] != image.shape[0] or edges.shape[2:] != image.shape[2:]:
                raise ValueError("edges and image must have the same batch size, height, and width")
            if edges.device != image.device or edges.dtype != image.dtype:
                raise ValueError("edges and image must have the same device and dtype")
            pair = torch.cat((edges, image), dim=1)
        if min(pair.shape[2:]) < 70:
            raise ValueError("PatchGAN inputs must be at least 70 pixels in height and width")
        return self.network(pair)


def initialize_weights(module: nn.Module) -> None:
    """Initialize convolution and BatchNorm layers with the Pix2Pix baseline."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_pix2pix(
    base_channels: int = PIX2PIX_BASE_CHANNELS,
    num_downs: int = PIX2PIX_NUM_DOWNSAMPLES,
    *,
    initialize: bool = True,
) -> tuple[UNetGenerator, PatchGANDiscriminator]:
    """Construct a generator/discriminator pair with matching base widths."""
    generator = UNetGenerator(base_channels=base_channels, num_downs=num_downs)
    discriminator = PatchGANDiscriminator(base_channels=base_channels)
    if initialize:
        generator.apply(initialize_weights)
        discriminator.apply(initialize_weights)
    return generator, discriminator
