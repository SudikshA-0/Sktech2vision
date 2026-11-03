"""Synthetic regression tests for paired dataset resizing and split behavior."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREPARE_SCRIPT = PROJECT_ROOT / "backend" / "ml" / "scripts" / "prepare_dataset.py"
IMAGE_SIZE = (64, 64)


def _pattern(height: int, width: int, marker: tuple[int, int] = (200, 200, 200)) -> np.ndarray:
    image = np.full((height, width, 3), (35, 70, 110), dtype=np.uint8)
    cv2.rectangle(image, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), marker, 2)
    cv2.line(image, (width // 2, 0), (width // 2, height - 1), (0, 0, 0), 1)
    return image


def _jpeg_with_exif_orientation(image: np.ndarray, orientation: int = 6) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode synthetic EXIF test image")
    # TIFF little-endian header + one IFD0 Orientation (0x0112), SHORT entry.
    tiff = (
        b"II*\x00\x08\x00\x00\x00"
        + b"\x01\x00"
        + b"\x12\x01\x03\x00\x01\x00\x00\x00"
        + int(orientation).to_bytes(2, "little")
        + b"\x00\x00\x00\x00\x00\x00"
    )
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    jpeg = encoded.tobytes()
    return jpeg[:2] + app1 + jpeg[2:]


class PrepareDatasetResizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=".prepare-dataset-test-", dir=PROJECT_ROOT
        )
        self.root = Path(self._temporary_directory.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.output_dir = self.root / "pad-output"
        self.repeat_output_dir = self.root / "pad-repeat"
        self.stretch_output_dir = self.root / "stretch-output"

        sources = {
            "landscape.png": _pattern(40, 80),
            "portrait.png": _pattern(80, 40),
            "square.png": _pattern(40, 40),
            "extra.png": _pattern(32, 48),
        }
        for name, image in sources.items():
            self.assertTrue(cv2.imwrite(str(self.input_dir / name), image))
        self.exif_source = self.input_dir / "exif-rotated.jpg"
        self.exif_source.write_bytes(_jpeg_with_exif_orientation(_pattern(40, 80)))

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _run_prepare(self, output_dir: Path, resize_mode: str | None) -> None:
        command = [
            sys.executable, str(PREPARE_SCRIPT), "prepare",
            "--input-dir", str(self.input_dir),
            "--output-dir", str(output_dir),
            "--image-size", str(IMAGE_SIZE[0]), str(IMAGE_SIZE[1]),
            "--seed", "17",
        ]
        if resize_mode is not None:
            command += ["--resize-mode", resize_mode]
        result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @staticmethod
    def _source_id_for_name(source_ids: dict[str, list[str]], filename: str) -> tuple[str, str]:
        for split_name in ("train", "validation", "test"):
            for source_id in source_ids[split_name]:
                if source_id.startswith(Path(filename).stem + "-"):
                    return split_name, source_id
        raise AssertionError(f"No manifest ID found for {filename}")

    def test_pad_geometry_pair_alignment_exif_and_deterministic_splits(self) -> None:
        self._run_prepare(self.output_dir, "pad")
        self._run_prepare(self.repeat_output_dir, "pad")
        manifest = json.loads((self.output_dir / "splits.json").read_text(encoding="utf-8"))
        repeat_manifest = json.loads((self.repeat_output_dir / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_ids"], repeat_manifest["source_ids"])
        self.assertEqual(manifest["transforms"], repeat_manifest["transforms"])
        self.assertEqual(manifest["preprocessing"]["resize_mode"], "pad")
        self.assertEqual(manifest["preprocessing"]["target_padding_rgb"], [255, 255, 255])
        self.assertEqual(manifest["preprocessing"]["edge_padding_grayscale"], 0)
        self.assertTrue(all(manifest["source_ids"][split] for split in ("train", "validation", "test")))

        expected_dimensions = {
            "landscape.png": (80, 40),
            "portrait.png": (40, 80),
            "square.png": (40, 40),
            "exif-rotated.jpg": (40, 80),
        }
        for filename, expected_source_dims in expected_dimensions.items():
            with self.subTest(filename=filename):
                split, source_id = self._source_id_for_name(manifest["source_ids"], filename)
                transform = manifest["transforms"][source_id]
                self.assertEqual(
                    (transform["source_dimensions_after_exif_orientation"]["width"],
                     transform["source_dimensions_after_exif_orientation"]["height"]),
                    expected_source_dims,
                )
                self.assertEqual(transform["exif_orientation_applied"], True)
                resized = transform["resized_dimensions"]
                padding = transform["padding"]
                self.assertLessEqual(abs(padding["left"] - padding["right"]), 1)
                self.assertLessEqual(abs(padding["top"] - padding["bottom"]), 1)
                self.assertAlmostEqual(
                    resized["width"] / resized["height"],
                    expected_source_dims[0] / expected_source_dims[1],
                    delta=0.04,
                )

                target = cv2.imread(str(self.output_dir / split / "targets" / f"{source_id}.png"), cv2.IMREAD_COLOR)
                edges = cv2.imread(str(self.output_dir / split / "edges" / f"{source_id}.png"), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(target.shape, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
                self.assertEqual(edges.shape, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
                left, top = padding["left"], padding["top"]
                content_width, content_height = resized["width"], resized["height"]
                content_target = target[top:top + content_height, left:left + content_width]
                content_edges = edges[top:top + content_height, left:left + content_width]
                gray = cv2.cvtColor(content_target, cv2.COLOR_BGR2GRAY)
                expected_edges = cv2.Canny(gray, 100, 200)
                np.testing.assert_array_equal(content_edges, expected_edges)
                if padding["left"]:
                    self.assertTrue(np.all(target[:, :left] == 255))
                    self.assertTrue(np.all(edges[:, :left] == 0))
                if padding["right"]:
                    self.assertTrue(np.all(target[:, IMAGE_SIZE[0] - padding["right"]:] == 255))
                    self.assertTrue(np.all(edges[:, IMAGE_SIZE[0] - padding["right"]:] == 0))
                if padding["top"]:
                    self.assertTrue(np.all(target[:top, :] == 255))
                    self.assertTrue(np.all(edges[:top, :] == 0))
                if padding["bottom"]:
                    self.assertTrue(np.all(target[IMAGE_SIZE[1] - padding["bottom"]:, :] == 255))
                    self.assertTrue(np.all(edges[IMAGE_SIZE[1] - padding["bottom"]:, :] == 0))

    def test_default_stretch_preserves_legacy_pixels_and_overwrite_protection(self) -> None:
        self._run_prepare(self.stretch_output_dir, None)
        manifest_path = self.stretch_output_dir / "splits.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["preprocessing"]["resize_mode"], "stretch")
        split, source_id = self._source_id_for_name(manifest["source_ids"], "landscape.png")
        original = cv2.imread(str(self.input_dir / "landscape.png"), cv2.IMREAD_UNCHANGED)
        expected = cv2.resize(original, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        actual = cv2.imread(str(self.stretch_output_dir / split / "targets" / f"{source_id}.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(actual, expected)
        transform = manifest["transforms"][source_id]
        self.assertEqual(transform["padding"], {"left": 0, "top": 0, "right": 0, "bottom": 0})
        self.assertFalse(transform["exif_orientation_applied"])

        sentinel = self.stretch_output_dir / "keep.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(PREPARE_SCRIPT), "prepare", "--input-dir", str(self.input_dir),
             "--output-dir", str(self.stretch_output_dir)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 2)
        self.assertTrue(sentinel.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_source_metadata_is_preserved_and_mapped_to_pair_id(self) -> None:
        metadata_path = self.root / "source-metadata.csv"
        fields = [
            "image_id", "image_url", "landing_page_url", "license_name", "license_url",
            "attribution_required", "attribution_text", "author", "license_review_status",
            "reviewer_notes", "source_metadata_url", "selected_category",
        ]
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "image_id": "landscape",
                "image_url": "https://images.example.test/landscape.jpg",
                "landing_page_url": "https://source.example.test/item/landscape",
                "license_name": "CC BY",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution_required": "true",
                "attribution_text": "",
                "author": "Artist Example",
                "license_review_status": "pending_manual_review",
                "reviewer_notes": "",
                "source_metadata_url": "local-review.csv",
                "selected_category": "chair",
            })
        output = self.root / "metadata-output"
        command = [
            sys.executable, str(PREPARE_SCRIPT), "prepare",
            "--input-dir", str(self.input_dir), "--output-dir", str(output),
            "--image-size", "64", "64", "--resize-mode", "pad",
            "--source-metadata-csv", str(metadata_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((output / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_kind"], "real")
        split, pair_id = self._source_id_for_name(manifest["source_ids"], "landscape.png")
        provenance = manifest["source_metadata"][pair_id]
        self.assertEqual(provenance["source_id"], pair_id)
        self.assertEqual(provenance["source_identifier"], "landscape")
        self.assertEqual(provenance["source_path"], "landscape.png")
        self.assertEqual(provenance["source_url"], "https://images.example.test/landscape.jpg")
        self.assertEqual(provenance["source_page_url"], "https://source.example.test/item/landscape")
        self.assertEqual(provenance["license_name"], "CC BY")
        self.assertEqual(provenance["license_identifier"], "https://creativecommons.org/licenses/by/4.0/")
        self.assertEqual(provenance["creator"], "Artist Example")
        self.assertEqual(provenance["attribution_required"], "true")
        self.assertEqual(provenance["license_review_status"], "pending_manual_review")
        self.assertEqual(provenance["selected_category"], "chair")
        self.assertTrue((output / split / "edges" / f"{pair_id}.png").is_file())
        for other_id, other_record in manifest["source_metadata"].items():
            self.assertEqual(other_record["source_id"], other_id)
            if other_id != pair_id:
                self.assertEqual(other_record["license_review_status"], "pending_manual_review")
                self.assertIn("source_metadata_not_provided", other_record["provenance_issues"])

    def test_synthetic_preparation_is_never_license_approved(self) -> None:
        output = self.root / "synthetic-output"
        command = [
            sys.executable, str(PREPARE_SCRIPT), "prepare",
            "--input-dir", str(self.input_dir), "--output-dir", str(output),
            "--image-size", "64", "64", "--dataset-kind", "synthetic",
        ]
        result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((output / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_kind"], "synthetic")
        for record in manifest["source_metadata"].values():
            self.assertEqual(record["record_type"], "synthetic")
            self.assertEqual(record["license_review_status"], "not_applicable_synthetic")
            self.assertNotEqual(record["license_review_status"], "approved")


if __name__ == "__main__":
    unittest.main()
