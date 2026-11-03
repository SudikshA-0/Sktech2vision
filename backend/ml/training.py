"""Pix2Pix optimization steps; no dataset discovery or application dependencies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .losses import discriminator_loss, generator_loss


@dataclass(frozen=True)
class EpochMetrics:
    generator_total: float
    generator_adversarial: float
    generator_l1: float
    discriminator: float
    examples: int
    seconds: float


def train_one_epoch(generator: torch.nn.Module, discriminator: torch.nn.Module,
                    loader: Iterable[dict[str, Any]], optimizer_g: torch.optim.Optimizer,
                    optimizer_d: torch.optim.Optimizer, device: torch.device,
                    l1_weight: float = 100.0) -> EpochMetrics:
    generator.train()
    discriminator.train()
    totals = {"g": 0.0, "adv": 0.0, "l1": 0.0, "d": 0.0}
    examples = 0
    started = time.perf_counter()
    for batch in loader:
        edges = batch["edges"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        count = edges.shape[0]

        # D update: fake samples are detached so this backward pass cannot update G.
        generated_for_d = generator(edges)
        optimizer_d.zero_grad(set_to_none=True)
        real_logits = discriminator(edges, targets)
        fake_logits = discriminator(edges, generated_for_d.detach())
        loss_d = discriminator_loss(real_logits, fake_logits)
        loss_d.backward()
        optimizer_d.step()

        # G update uses a fresh forward graph. Freeze D parameters while keeping
        # its operations differentiable with respect to the generated image.
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        try:
            generated = generator(edges)
            logits_for_g = discriminator(edges, generated)
            loss_g, loss_adv, loss_l1 = generator_loss(logits_for_g, generated, targets, l1_weight)
            loss_g.backward()
            optimizer_g.step()
        finally:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

        totals["g"] += float(loss_g.detach()) * count
        totals["adv"] += float(loss_adv.detach()) * count
        totals["l1"] += float(loss_l1.detach()) * count
        totals["d"] += float(loss_d.detach()) * count
        examples += count
    if examples == 0:
        raise ValueError("training split is empty")
    return EpochMetrics(
        generator_total=totals["g"] / examples,
        generator_adversarial=totals["adv"] / examples,
        generator_l1=totals["l1"] / examples,
        discriminator=totals["d"] / examples,
        examples=examples,
        seconds=time.perf_counter() - started,
    )

