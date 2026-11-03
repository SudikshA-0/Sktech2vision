"""Synthetic metadata-only tests for the Open Images candidate manifest schema."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "backend" / "ml" / "scripts" / "build_open_images_manifest.py"


class OpenImagesManifestProvenanceTests(unittest.TestCase):
    categories = [f"category-{index}" for index in range(10)]

    def make_inputs(self, root: Path, assignments: list[tuple[str, int]], *, omit_info: set[str] | None = None,
                    duplicate_info: set[str] | None = None,
                    label_overrides: dict[str, tuple[str, str]] | None = None) -> tuple[list[Path], Path, Path]:
        ids = sorted({image_id for image_id, _ in assignments})
        class_csv, labels_csv, info_csv = root / "classes.csv", root / "labels.csv", root / "image-info.csv"
        with class_csv.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows((f"/m/{index}", category) for index, category in enumerate(self.categories))
        with labels_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("ImageID", "Source", "LabelName", "Confidence"))
            writer.writerows((image_id,
                              (label_overrides or {}).get(image_id, ("verification", "1"))[0],
                              f"/m/{category_index}",
                              (label_overrides or {}).get(image_id, ("verification", "1"))[1])
                             for image_id, category_index in assignments)
        info_fields = ("ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "OriginalSize",
                       "Rotation", "Author", "AuthorProfileURL", "Title", "License")
        with info_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=info_fields)
            writer.writeheader()
            for image_id in ids:
                if omit_info and image_id in omit_info:
                    continue
                row = {"ImageID": image_id, "Subset": "train",
                       "OriginalURL": f"https://images.example.test/{image_id}.jpg",
                       "OriginalLandingURL": f"https://source.example.test/{image_id}",
                       "OriginalSize": "1234", "Rotation": "0", "Author": "Creator",
                       "AuthorProfileURL": "https://source.example.test/creator", "Title": "Test image",
                       "License": "https://creativecommons.org/licenses/by/4.0/"}
                writer.writerow(row)
                if duplicate_info and image_id in duplicate_info:
                    writer.writerow(row)
        return [class_csv, labels_csv, info_csv], class_csv, labels_csv

    def run_builder(self, root: Path, paths: list[Path], *, per_category: int = 1,
                    categories: list[str] | None = None) -> subprocess.CompletedProcess[str]:
        class_csv, labels_csv, info_csv = paths
        command = [sys.executable, str(SCRIPT), "--labels-csv", str(labels_csv),
                   "--image-info-csv", str(info_csv), "--class-descriptions-csv", str(class_csv),
                   "--output-csv", str(root / "candidate.csv"), "--summary-json", str(root / "summary.json"),
                   "--per-category", str(per_category)]
        for category in categories or self.categories:
            command.extend(("--category", category))
        return subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)

    def test_candidate_rows_use_canonical_provenance_fields_and_remain_pending(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".openimages-manifest-test-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            categories = self.categories
            ids = [f"{index:016x}" for index in range(1, 11)]
            class_csv = root / "classes.csv"
            with class_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                # Open Images class descriptions are LabelName, DisplayName.
                writer.writerows((f"/m/{index}", category) for index, category in enumerate(categories, start=1))
            labels_csv = root / "labels.csv"
            with labels_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("ImageID", "Source", "LabelName", "Confidence"))
                writer.writerows((image_id, "verification", f"/m/{index}", "1")
                                 for index, image_id in enumerate(ids, start=1))
            info_csv = root / "image-info.csv"
            info_fields = ("ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "OriginalSize",
                           "Rotation", "Author", "AuthorProfileURL", "Title", "License")
            with info_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=info_fields)
                writer.writeheader()
                for image_id in ids:
                    writer.writerow({
                        "ImageID": image_id, "Subset": "train",
                        "OriginalURL": f"https://images.example.test/{image_id}.jpg",
                        "OriginalLandingURL": f"https://source.example.test/{image_id}",
                        "OriginalSize": "1234", "Rotation": "0", "Author": "Creator",
                        "AuthorProfileURL": "https://source.example.test/creator", "Title": "Test image",
                        "License": "https://creativecommons.org/licenses/by/4.0/",
                    })
            output_csv = root / "candidate.csv"
            summary_json = root / "summary.json"
            command = [
                sys.executable, str(SCRIPT), "--labels-csv", str(labels_csv),
                "--image-info-csv", str(info_csv), "--class-descriptions-csv", str(class_csv),
                "--output-csv", str(output_csv), "--summary-json", str(summary_json),
                "--per-category", "1",
            ]
            for category in categories:
                command.extend(("--category", category))
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with output_csv.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            row = rows[0]
            self.assertEqual(row["source_identifier"], row["image_id"])
            self.assertEqual(row["source_url"], row["image_url"])
            self.assertEqual(row["source_page_url"], row["landing_page_url"])
            self.assertEqual(row["license_identifier"], row["license_url"])
            self.assertEqual(row["attribution_required"], "unknown")
            self.assertEqual(row["creator"], "Creator")
            self.assertEqual(row["reviewer_notes"], "")
            self.assertEqual(row["license_review_status"], "pending_manual_review")
            self.assertEqual(row["record_type"], "real")
            self.assertIn("attribution_requirement_requires_manual_review", row["metadata_status"])

    def test_category_assignment_avoids_order_dependent_shortfall_and_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".openimages-matching-test-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            ids = [f"{index:016x}" for index in range(1, 11)]
            # Category 0 is first in the request, but its first shared ID is the
            # only ID for category 1. A greedy pass would unnecessarily short it.
            assignments = [(ids[0], 0), (ids[1], 0), (ids[0], 1)]
            assignments.extend((ids[index], index) for index in range(2, 10))
            paths, _, _ = self.make_inputs(root, assignments)
            result = self.run_builder(root, paths)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import json
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_rows"], 10)
            self.assertTrue(summary["target_achieved"])
            self.assertEqual(summary["eligible_per_category"][self.categories[0]], 2)
            self.assertEqual(summary["selected_per_category"][self.categories[0]], 1)
            self.assertEqual(summary["selected_per_category"][self.categories[1]], 1)
            self.assertTrue(all(value == 0 for value in summary["shortfall_per_category"].values()))
            with (root / "candidate.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len({row["image_id"] for row in rows}), 10)
            self.assertTrue(all(row["license_review_status"] == "pending_manual_review" for row in rows))

    def test_missing_image_info_join_is_reported_and_not_selected(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".openimages-join-test-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            ids = [f"{index:016x}" for index in range(1, 11)]
            assignments = list(zip(ids, range(10)))
            paths, _, _ = self.make_inputs(root, assignments, omit_info={ids[4]})
            result = self.run_builder(root, paths)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import json
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_rows"], 9)
            self.assertFalse(summary["target_achieved"])
            self.assertEqual(summary["missing_image_info_join_candidate_ids"], 1)
            self.assertEqual(summary["eligible_per_category"][self.categories[4]], 0)
            self.assertEqual(summary["shortfall_per_category"][self.categories[4]], 1)
            with (root / "candidate.csv").open(newline="", encoding="utf-8") as stream:
                self.assertNotIn(ids[4], {row["image_id"] for row in csv.DictReader(stream)})
            self.assertIn("selected 9", result.stdout)

    def test_duplicate_selected_image_info_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".openimages-duplicate-test-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            ids = [f"{index:016x}" for index in range(1, 11)]
            paths, _, _ = self.make_inputs(root, list(zip(ids, range(10))), duplicate_info={ids[0]})
            result = self.run_builder(root, paths)
            self.assertEqual(result.returncode, 2)
            self.assertIn("duplicate image-info record", result.stderr)
            self.assertFalse((root / "candidate.csv").exists())

    def test_malformed_candidate_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".openimages-malformed-id-test-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            ids = [f"{index:016x}" for index in range(1, 10)]
            assignments = [("bad-id", 0), *zip(ids, range(1, 10))]
            paths, _, _ = self.make_inputs(root, assignments)
            result = self.run_builder(root, paths)
            self.assertEqual(result.returncode, 2)
            self.assertIn("malformed ImageID", result.stderr)

    def test_confidence_numeric_parsing_and_verification_source_filter(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".openimages-confidence-test-", dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            ids = [f"{index:016x}" for index in range(1, 11)]
            overrides = {
                ids[0]: ("verification", "1.0"),
                ids[1]: ("verification", "1"),
                ids[2]: ("verification", "0.0"),
                ids[3]: ("verification", "not-a-number"),
                ids[4]: ("crowdsource-verification", "1.0"),
                ids[5]: ("verification", "nan"),
                ids[6]: ("verification", "inf"),
            }
            paths, _, _ = self.make_inputs(root, list(zip(ids, range(10))), label_overrides=overrides)
            result = self.run_builder(root, paths)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (root / "candidate.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            selected = {row["image_id"] for row in rows}
            self.assertIn(ids[0], selected, 'Confidence="1.0" should be accepted')
            self.assertIn(ids[1], selected, 'Confidence="1" should be accepted')
            self.assertNotIn(ids[2], selected, 'Confidence="0.0" must be rejected')
            self.assertNotIn(ids[3], selected, "Malformed confidence must be excluded safely")
            self.assertNotIn(ids[4], selected, "crowdsource-verification must not be accepted")
            self.assertNotIn(ids[5], selected, "Non-finite NaN confidence must be excluded")
            self.assertNotIn(ids[6], selected, "Non-finite infinity confidence must be excluded")
            self.assertTrue(all(row["license_review_status"] == "pending_manual_review" for row in rows))


if __name__ == "__main__":
    unittest.main()
