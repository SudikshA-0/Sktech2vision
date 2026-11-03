"""Paired edge/target dataset loader for prepared Pix2Pix folders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .transforms import normalize_uint8_tensor


class PairedEdgeImageDataset(Dataset[dict[str, Any]]):
    """Load prepared PNG pairs without resizing or changing their geometry."""

    def __init__(self, dataset_dir: Path | str, split: str, image_size: tuple[int, int] | None = None) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        self.root = Path(dataset_dir).expanduser().resolve()
        self.split = split
        self.edge_dir = self.root / split / "edges"
        self.target_dir = self.root / split / "targets"
        if not self.edge_dir.is_dir() or not self.target_dir.is_dir():
            raise FileNotFoundError(f"missing {split}/edges or {split}/targets under {self.root}")
        supported = {".png", ".jpg", ".jpeg"}
        edge_files = {p.stem: p for p in self.edge_dir.iterdir() if p.is_file() and p.suffix.lower() in supported}
        target_files = {p.stem: p for p in self.target_dir.iterdir() if p.is_file() and p.suffix.lower() in supported}
        if edge_files.keys() != target_files.keys():
            missing_targets = sorted(edge_files.keys() - target_files.keys())
            missing_edges = sorted(target_files.keys() - edge_files.keys())
            raise ValueError(f"unpaired PNGs in {split}: missing_targets={missing_targets}, missing_edges={missing_edges}")
        self.source_ids = sorted(edge_files)
        self.pairs = [(edge_files[source_id], target_files[source_id], source_id) for source_id in self.source_ids]
        self.expected_size = image_size
        if image_size is not None and (len(image_size) != 2 or min(image_size) < 1):
            raise ValueError("image_size must be positive (width, height)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError("OpenCV is required to load prepared PNG pairs") from exc

        edge_path, target_path, source_id = self.pairs[index]
        edge = cv2.imread(str(edge_path), cv2.IMREAD_UNCHANGED)
        target_bgr = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        if edge is None or edge.size == 0:
            raise ValueError(f"unreadable edge map: {edge_path}")
        if target_bgr is None or target_bgr.size == 0:
            raise ValueError(f"unreadable target image: {target_path}")
        if edge.ndim != 2:
            raise ValueError(f"edge map must be one-channel grayscale: {edge_path} has shape {edge.shape}")
        if target_bgr.ndim != 3 or target_bgr.shape[2] != 3:
            raise ValueError(f"target must have three BGR/RGB channels: {target_path} has shape {target_bgr.shape}")
        if edge.shape[:2] != target_bgr.shape[:2]:
            raise ValueError(f"pair dimensions differ for {source_id}")
        height, width = edge.shape
        if self.expected_size is not None and (width, height) != tuple(self.expected_size):
            raise ValueError(
                f"pair {source_id} is {width}x{height}; expected {self.expected_size[0]}x{self.expected_size[1]}"
            )
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        edge_tensor = torch.from_numpy(edge.copy()).unsqueeze(0)
        target_tensor = torch.from_numpy(target_rgb.copy()).permute(2, 0, 1).contiguous()
        return {
            "edges": normalize_uint8_tensor(edge_tensor),
            "targets": normalize_uint8_tensor(target_tensor),
            "source_id": source_id,
        }

