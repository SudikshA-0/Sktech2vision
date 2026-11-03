"""Isolated helpers for a short Pix2Pix Facades presentation demo.

This module deliberately does not touch the production dataset validator or
training provenance gate. It adapts Berkeley's paired facades composites to
the existing edge-to-RGB model for demonstration only.
"""

from __future__ import annotations

import io
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .evaluation import evaluate_generator, save_checkpoint
from .training import train_one_epoch


FACADE_ARCHIVE_URL = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
DEMO_DATASET_NAME = "CMP Facades / Pix2Pix Facades demonstration subset"


def download_facades(cache_dir: str | Path, url: str = FACADE_ARCHIVE_URL) -> Path:
    """Download the official Pix2Pix Facades archive once; never overwrite it."""
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "facades.tar.gz"
    if archive.exists():
        with tarfile.open(archive, "r:gz") as tar:
            if not any(_is_safe_file_member(member) for member in tar.getmembers()):
                raise ValueError(f"cached Facades archive contains no safe files: {archive}")
        return archive

    temporary = archive.with_suffix(".tar.gz.part")
    request = urllib.request.Request(url, headers={"User-Agent": "Sketch2Vision-Colab-Demo/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        with tarfile.open(temporary, "r:gz") as tar:
            if not any(_is_safe_file_member(member) for member in tar.getmembers()):
                raise ValueError("downloaded archive contains no safe files")
        temporary.replace(archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return archive


def _is_safe_file_member(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    return member.isfile() and not path.is_absolute() and ".." not in path.parts


def extract_facades(archive_path: str | Path, destination: str | Path) -> Path:
    """Extract regular files only, rejecting traversal, links, and devices."""
    archive = Path(archive_path).expanduser().resolve()
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        if (root / "train").is_dir() and (root / "test").is_dir() and any((root / name).is_dir() for name in ("val", "validation")):
            return root
        for candidate in root.rglob("*"):
            if candidate.is_dir() and (candidate / "train").is_dir() and (candidate / "test").is_dir() and any(
                (candidate / name).is_dir() for name in ("val", "validation")
            ):
                return candidate
        raise FileExistsError(f"extraction destination is non-empty and is not a recognized Facades dataset: {root}")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        unsafe = [m.name for m in members if not _is_safe_file_member(m) and not m.isdir()]
        unsafe += [m.name for m in members if m.isdir() and (PurePosixPath(m.name).is_absolute() or ".." in PurePosixPath(m.name).parts)]
        if unsafe:
            raise ValueError(f"unsafe archive member(s): {unsafe[:3]}")
        for member in members:
            relative = PurePosixPath(member.name)
            target = (root / Path(*relative.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"archive path escapes extraction directory: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif _is_safe_file_member(member):
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read archive member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    candidates = [p for p in root.rglob("*") if p.is_dir() and (p / "train").is_dir()]
    if not candidates and (root / "train").is_dir():
        return root
    for candidate in candidates:
        if any((candidate / name).is_dir() for name in ("val", "validation")) and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError("Facades archive must contain train, val/validation, and test folders")


class FacadesPairDataset(Dataset[dict[str, Any]]):
    """Read paired label|photo composites and derive a one-channel label edge map."""

    def __init__(self, dataset_root: str | Path, split: str, image_size: int = 128) -> None:
        if split not in {"train", "val", "validation", "test"}:
            raise ValueError("split must be train, val/validation, or test")
        if image_size < 128 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two of at least 128")
        self.root = Path(dataset_root).expanduser().resolve()
        folder = "val" if split == "validation" and (self.root / "val").is_dir() else split
        folder = "validation" if folder == "val" and not (self.root / "val").is_dir() else folder
        self.files = sorted(p for p in (self.root / folder).rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.files:
            raise FileNotFoundError(f"no paired images found in {self.root / folder}")
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable paired image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        if width != height * 2:
            raise ValueError(f"expected a side-by-side 2:1 pair, got {width}x{height}: {path.name}")
        split = width // 2
        label_map = rgb[:, :split]
        photo = rgb[:, split:]

        gray = cv2.cvtColor(label_map, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        size = (self.image_size, self.image_size)
        edges = cv2.resize(edges, size, interpolation=cv2.INTER_NEAREST)
        photo = cv2.resize(photo, size, interpolation=cv2.INTER_AREA if height > self.image_size else cv2.INTER_LINEAR)
        edge_tensor = torch.from_numpy(edges.copy()).unsqueeze(0).float().div_(127.5).sub_(1.0)
        target_tensor = torch.from_numpy(photo.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        return {"edges": edge_tensor, "targets": target_tensor, "source_id": path.stem}


def deterministic_subset(dataset: Dataset, maximum: int | None, seed: int) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("subset maximum must be positive")
    indexes = list(range(len(dataset)))
    random.Random(seed).shuffle(indexes)
    return Subset(dataset, sorted(indexes[:maximum]))


def train_facades_demo(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    output_dir: str | Path,
    *,
    epochs: int = 3,
    image_size: int = 128,
    learning_rate: float = 2e-4,
    l1_weight: float = 100.0,
    callback: Callable[[int, dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    """Train existing Pix2Pix modules on the isolated demo loaders and checkpoint each epoch."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    generator.to(device)
    discriminator.to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, device, l1_weight)
        validation_l1 = evaluate_generator(generator, validation_loader, device)
        row = {"epoch": float(epoch), "generator_total": metrics.generator_total,
               "generator_adversarial": metrics.generator_adversarial, "generator_l1": metrics.generator_l1,
               "discriminator": metrics.discriminator, "validation_l1": validation_l1}
        history.append(row)
        payload = {"format_version": 1, "demo_only": True, "dataset": DEMO_DATASET_NAME,
                   "epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                   "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                   "history": history, "model_config": {"image_size": image_size, "l1_weight": l1_weight,
                                                            "base_channels": getattr(generator, "base_channels", None)}}
        save_checkpoint(destination / "latest.pt", payload)
        if validation_l1 < best_val:
            best_val = validation_l1
            payload["best_validation_l1"] = best_val
            save_checkpoint(destination / "best.pt", payload)
        print(f"epoch {epoch}/{epochs}: G={metrics.generator_total:.5f} D={metrics.discriminator:.5f} val_L1={validation_l1:.6f}")
        if callback is not None:
            callback(epoch, row)
    return history


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert one normalized CHW tensor to uint8 grayscale or RGB for display."""
    value = tensor.detach().cpu().float().clamp(-1, 1).add(1).mul(127.5).round().byte()
    if value.ndim == 3 and value.shape[0] == 1:
        return value[0].numpy()
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("expected CHW tensor with one or three channels")
    return value.permute(1, 2, 0).numpy()


def predict_one(generator: torch.nn.Module, edge: torch.Tensor, device: torch.device) -> torch.Tensor:
    generator.to(device)
    was_training = generator.training
    generator.eval()
    try:
        with torch.inference_mode():
            return generator(edge.unsqueeze(0).to(device))[0].cpu()
    finally:
        generator.train(was_training)

