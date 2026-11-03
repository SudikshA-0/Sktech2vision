"""Validation metrics and atomic checkpoint management."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def evaluate_generator(generator: torch.nn.Module, loader: Iterable[dict[str, Any]], device: torch.device) -> float:
    """Return pixel-mean L1 on validation pairs only, with inference disabled."""
    was_training = generator.training
    generator.eval()
    total_absolute_error = 0.0
    total_values = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                edges = batch["edges"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                generated = generator(edges)
                total_absolute_error += F.l1_loss(generated, targets, reduction="sum").item()
                total_values += targets.numel()
    finally:
        generator.train(was_training)
    if total_values == 0:
        raise ValueError("validation split is empty; cannot select a checkpoint")
    return total_absolute_error / total_values


def dataset_fingerprint(dataset_dir: Path | str, source_ids: Iterable[str]) -> str:
    """Hash split metadata and train/validation pair bytes; deliberately omit test files."""
    root = Path(dataset_dir).resolve()
    manifest = root / "splits.json"
    digest = hashlib.sha256()
    digest.update(manifest.read_bytes())
    for split in ("train", "validation"):
        for source_id in sorted(set(source_ids)):
            for kind in ("edges", "targets"):
                candidates = sorted((root / split / kind).glob(f"{source_id}.*"))
                candidates = [p for p in candidates if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
                if not candidates:
                    continue
                path = candidates[0]
                digest.update(f"{split}/{kind}/{path.name}\0".encode("utf-8"))
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    loader_generator.set_state(state["loader"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(path: Path | str, checkpoint: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path | str, *, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    # Explicit weights_only=False is required because resume bundles include RNG tuples and config metadata.
    try:
        data = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # older PyTorch releases do not expose weights_only
        data = torch.load(Path(path), map_location=map_location)
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError("unsupported or malformed training checkpoint")
    return data


def write_run_metadata(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)

