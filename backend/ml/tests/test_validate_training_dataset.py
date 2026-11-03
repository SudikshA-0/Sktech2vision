"""Synthetic tests for the read-only Pix2Pix dataset validator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = PROJECT_ROOT / "backend" / "ml" / "scripts" / "validate_training_dataset.py"
SIZE = (16, 16)


class ValidateTrainingDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix=".validate-training-test-", dir=PROJECT_ROOT)
        self.root = Path(self.temp.name) / "dataset"
        self.root.mkdir()
        for split in ("train", "validation", "test"):
            (self.root / split / "edges").mkdir(parents=True)
            (self.root / split / "targets").mkdir()
        self.manifest = {
            "image_size": {"width": SIZE[0], "height": SIZE[1]},
            "preprocessing": {
                "resize_mode": "stretch",
                "target_padding_rgb": None,
                "edge_padding_grayscale": None,
                "padding_alignment": None,
            },
            "source_ids": {"train": ["sample"], "validation": [], "test": []},
            "skipped": [],
        }
        self.write_manifest()
        self.write_pair("train", "sample")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_manifest(self) -> None:
        (self.root / "splits.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def write_pair(self, split: str, source_id: str, *, dimensions: tuple[int, int] = SIZE,
                   edge_channels: int = 1, target_channels: int = 3) -> None:
        width, height = dimensions
        edge = np.zeros((height, width), dtype=np.uint8)
        if edge_channels == 3:
            edge = np.zeros((height, width, 3), dtype=np.uint8)
        target_shape = (height, width) if target_channels == 1 else (height, width, target_channels)
        target = np.full(target_shape, 180, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.root / split / "edges" / f"{source_id}.png"), edge))
        self.assertTrue(cv2.imwrite(str(self.root / split / "targets" / f"{source_id}.png"), target))

    def run_validator(self, expected: tuple[int, int] | None = None) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(VALIDATOR), "--dataset-dir", str(self.root)]
        if expected is not None:
            command += ["--expected-size", str(expected[0]), str(expected[1])]
        return subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)

    def test_valid_pairs_pass(self) -> None:
        result = self.run_validator(expected=SIZE)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("fatal_errors=0", result.stdout)

    def test_missing_target_fails(self) -> None:
        (self.root / "train" / "targets" / "sample.png").unlink()
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("edge image has no matching target", result.stdout)

    def test_extra_target_fails(self) -> None:
        cv2.imwrite(str(self.root / "train" / "targets" / "orphan.png"), np.zeros((16, 16, 3), np.uint8))
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("target has no matching edge image", result.stdout)

    def test_unreadable_file_fails(self) -> None:
        edge = self.root / "train" / "edges" / "sample.png"
        edge.write_bytes(b"not a PNG")
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unreadable image", result.stdout)

    def test_wrong_dimensions_and_channels_fail(self) -> None:
        self.write_pair("train", "sample", dimensions=(12, 16), edge_channels=3, target_channels=1)
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wrong dimensions", result.stdout)
        self.assertIn("edge map must be one-channel", result.stdout)
        self.assertIn("target must be three-channel", result.stdout)

    def test_split_leakage_fails(self) -> None:
        self.manifest["source_ids"]["validation"] = ["sample"]
        self.write_manifest()
        self.write_pair("validation", "sample")
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("leaks across train and validation", result.stdout)

    def test_malformed_preprocessing_and_missing_dimensions_fail(self) -> None:
        del self.manifest["image_size"]
        self.manifest["preprocessing"] = {"resize_mode": "mystery"}
        self.write_manifest()
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must declare positive integer image_size", result.stdout)
        self.assertIn("resize_mode must be explicitly", result.stdout)

    def test_pending_license_is_reported_as_warning_not_approval(self) -> None:
        self.manifest["source_metadata"] = {
            "sample": {
                "category": "chair",
                "license_name": "CC BY",
                "attribution_required": "true",
                "attribution_text": "",
                "license_review_status": "pending_manual_review",
            }
        }
        self.write_manifest()
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Category counts: chair=1", result.stdout)
        self.assertIn("pending_manual_review=1", result.stdout)
        self.assertIn("not license approval", result.stdout)
        self.assertIn("attribution", result.stdout)


if __name__ == "__main__":
    unittest.main()
