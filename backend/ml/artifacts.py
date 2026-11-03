"""Training logs and validation sample grids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .transforms import denormalize_to_uint8


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def save_validation_grid(generator: torch.nn.Module, loader: Iterable[dict[str, Any]], path: Path,
                         device: torch.device, max_samples: int = 4) -> None:
    """Write rows of edge | generated | target images, with no autograd graph."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV and NumPy are required to save validation grids") from exc

    previous_mode = generator.training
    generator.eval()
    rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_samples:
                break
            edges = batch["edges"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            generated = generator(edges)
            edges_u8 = denormalize_to_uint8(edges).cpu().numpy()
            generated_u8 = denormalize_to_uint8(generated).cpu().numpy()
            targets_u8 = denormalize_to_uint8(targets).cpu().numpy()
            for i in range(edges.shape[0]):
                edge_rgb = np.repeat(edges_u8[i], 3, axis=0).transpose(1, 2, 0)
                generated_bgr = cv2.cvtColor(generated_u8[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                target_bgr = cv2.cvtColor(targets_u8[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                rows.append(np.concatenate((edge_rgb, generated_bgr, target_bgr), axis=1))
                if len(rows) >= max_samples:
                    break
            if len(rows) >= max_samples:
                break
    generator.train(previous_mode)
    if not rows:
        raise ValueError("validation loader contains no examples for a preview grid")
    grid = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), grid):
        raise OSError(f"could not write validation grid: {path}")

