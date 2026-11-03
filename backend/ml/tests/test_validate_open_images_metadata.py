"""Synthetic local tests for Open Images metadata join validation."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "backend" / "ml" / "scripts" / "validate_open_images_metadata.py"
LABELS = "oidv7-train-annotations-human-imagelabels.csv"
CLASSES = "oidv7-class-descriptions.csv"
IMAGE_INFO = "oidv6-train-images-with-labels-with-rotation.csv"


class OpenImagesMetadataValidationTests(unittest.TestCase):
    def write_fixture(self, root: Path, *, info_ids: list[str], label_ids: list[str] | None = None) -> None:
        label_ids = label_ids or info_ids
        with (root / LABELS).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("ImageID", "Source", "LabelName", "Confidence"))
            writer.writerows((image_id, "verification", "/m/test", "1") for image_id in label_ids)
        with (root / CLASSES).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(("/m/test", "Test"))
        fields = ("ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License", "AuthorProfileURL",
                  "Author", "Title", "OriginalSize", "OriginalMD5", "Thumbnail300KURL", "Rotation")
        with (root / IMAGE_INFO).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for image_id in info_ids:
                writer.writerow({"ImageID": image_id, "Subset": "train", "OriginalURL": "url",
                                 "OriginalLandingURL": "page", "License": "license", "AuthorProfileURL": "profile",
                                 "Author": "creator", "Title": "title", "OriginalSize": "10",
                                 "OriginalMD5": "md5", "Thumbnail300KURL": "thumb", "Rotation": "0"})

    def run_validator(self, root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), "--metadata-dir", str(root), *extra],
                              cwd=PROJECT_ROOT, capture_output=True, text=True)

    def test_missing_label_to_image_info_join_is_fatal_and_reported(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".oi-validator-join-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            self.write_fixture(root, info_ids=["0000000000000001"], label_ids=["0000000000000001", "0000000000000002"])
            result = self.run_validator(root)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("FATAL missing image-info joins for 1", result.stdout)

    def test_duplicate_and_malformed_image_info_ids_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".oi-validator-duplicate-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            duplicate = "0000000000000001"
            self.write_fixture(root, info_ids=[duplicate, duplicate, "malformed"], label_ids=[duplicate])
            result = self.run_validator(root)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("FATAL duplicate image-info ImageID records", result.stdout)
            self.assertIn("FATAL invalid/blank ImageID rows", result.stdout)

    def test_selected_manifest_missing_join_and_duplicate_ids_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".oi-validator-selected-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            existing = "0000000000000001"
            missing = "0000000000000002"
            self.write_fixture(root, info_ids=[existing], label_ids=[existing])
            selected = root / "selected.csv"
            with selected.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("image_id", "license_review_status"))
                writer.writerow((existing, "pending_manual_review"))
                writer.writerow((existing, "pending_manual_review"))
                writer.writerow((missing, "pending_manual_review"))
            result = self.run_validator(root, "--selected-ids-csv", str(selected))
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("duplicate_ids", result.stdout)
            self.assertIn("missing_image_info_ids", result.stdout)


if __name__ == "__main__":
    unittest.main()
