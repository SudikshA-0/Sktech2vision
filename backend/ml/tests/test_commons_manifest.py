"""Offline tests for the metadata-only Wikimedia Commons manifest adapter."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from email.message import Message
from pathlib import Path
from urllib.request import Request

from backend.ml.scripts import build_commons_manifest as commons


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def search_page(*records: tuple[int, str], continuation: dict | None = None) -> dict:
    result = {"query": {"search": [{"pageid": pageid, "title": title} for pageid, title in records]}}
    if continuation:
        result["continue"] = continuation
    return result


def info_page(pageid: int, title: str, *, sha1: str = "abc123", ext: dict | None = None) -> dict:
    return {
        "pageid": pageid,
        "title": title,
        "imageinfo": [{
            "url": f"https://upload.wikimedia.org/{pageid}.jpg",
            "descriptionurl": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
            "mime": "image/jpeg", "width": 640, "height": 480, "size": 10000,
            "sha1": sha1, "timestamp": "2026-01-02T03:04:05Z",
            "extmetadata": ext or {},
        }],
    }


class MockAPI:
    def __init__(self, search_responses: list[dict], info_pages: list[dict]):
        self.search_responses = list(search_responses)
        self.info_pages = info_pages
        self.calls: list[dict] = []

    def get_json(self, params: dict) -> dict:
        self.calls.append(params)
        if params.get("list") == "search":
            return self.search_responses.pop(0)
        requested = set(str(params.get("titles", "")).split("|"))
        pages = [page for page in self.info_pages if page["title"] in requested]
        return {"query": {"pages": pages}}


class CommonsManifestTests(unittest.TestCase):
    def test_category_contract_and_manifest_fields(self) -> None:
        self.assertEqual(commons.CATEGORIES, (
            "Person", "Dog", "Cat", "Car", "Bicycle", "Chair", "Bird", "Flower", "Tree", "Building"
        ))
        page = search_page((17, "File:Dog.jpg"))
        ext = {
            "Artist": {"value": "<a>Example creator</a>"},
            "ObjectName": {"value": "A dog"},
            "Credit": {"value": "Example creator / Wikimedia Commons"},
            "AttributionRequired": {"value": "true"},
            "LicenseShortName": {"value": "CC BY-SA 4.0"},
            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
        }
        client = MockAPI([page], [info_page(17, "File:Dog.jpg", ext=ext)])
        rows, summary = commons.build_manifest(client, ["Dog"], 1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(set(commons.MANIFEST_FIELDS).issubset(row))
        self.assertEqual(row["source_identifier"], "commons-17")
        self.assertEqual(row["source_path"], "Dog/commons-17.jpg")
        self.assertEqual(row["creator"], "Example creator")
        self.assertEqual(row["license_name"], "CC BY-SA 4.0")
        self.assertEqual(row["license_version"], "4.0")
        self.assertEqual(row["license_review_status"], "pending_manual_review")
        raw = json.loads(row["source_metadata_raw"])
        self.assertEqual(raw["imageinfo"]["extmetadata"]["LicenseUrl"]["value"], ext["LicenseUrl"]["value"])
        self.assertEqual(summary["selected_per_category"]["Dog"], 1)
        self.assertEqual(summary["shortfall_per_category"]["Dog"], 0)
        self.assertEqual(client.calls[0]["srnamespace"], "6")
        self.assertIn("filetype:bitmap", client.calls[0]["srsearch"])
        self.assertIn("-illustration", client.calls[0]["srsearch"])
        self.assertIn("photograph", client.calls[0]["srsearch"])

    def test_all_category_queries_prefer_photography_and_exclude_art_terms(self) -> None:
        for category, query in commons.SEARCH_TERMS.items():
            with self.subTest(category=category):
                self.assertIn("photograph", query)
                self.assertIn("filetype:bitmap", query)
                for term in ("poster", "diagram", "illustration", "logo", "painting", "cartoon"):
                    self.assertIn(f"-{term}", query)

    def test_explicit_non_photographs_are_filtered_without_license_filtering(self) -> None:
        search = search_page((41, "File:Dog poster.jpg"), (42, "File:Dog diagram.jpg"),
                             (43, "File:Dog photograph.jpg"))
        ext = {
            "ImageDescription": {"value": "Photograph of a dog outdoors"},
            "Categories": {"value": "Dogs photographed outdoors"},
            "LicenseShortName": {"value": "CC BY-SA 4.0"},
        }
        info = [info_page(41, "File:Dog poster.jpg"), info_page(42, "File:Dog diagram.jpg"),
                info_page(43, "File:Dog photograph.jpg", ext=ext)]
        client = MockAPI([search], info)
        rows, summary = commons.build_manifest(client, ["Dog"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["commons_page_id"], "43")
        self.assertEqual(rows[0]["license_name"], "CC BY-SA 4.0")
        self.assertEqual(rows[0]["license_review_status"], "pending_manual_review")
        self.assertEqual(summary["explicit_non_photograph_skipped_per_category"]["Dog"], 2)
        self.assertIn("ImageDescription", json.loads(rows[0]["source_metadata_raw"])["imageinfo"]["extmetadata"])

    def test_creator_cap_defers_repeats_but_fills_quota_when_needed(self) -> None:
        candidates = [
            {"creator_key": "alice", "photo_score": 3, "order": 0},
            {"creator_key": "alice", "photo_score": 2, "order": 1},
            {"creator_key": "bob", "photo_score": 1, "order": 2},
            {"creator_key": "", "photo_score": 1, "order": 3},
        ]
        selected, fallback_count = commons._select_creator_diverse(candidates, quota=3, max_per_creator=1)
        self.assertEqual([item["creator_key"] for item in selected], ["alice", "bob", ""])
        self.assertEqual(fallback_count, 0)

        only_one_creator = [
            {"creator_key": "alice", "photo_score": 2, "order": 0},
            {"creator_key": "alice", "photo_score": 1, "order": 1},
        ]
        selected, fallback_count = commons._select_creator_diverse(
            only_one_creator, quota=2, max_per_creator=1
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(fallback_count, 1)

    def test_metadata_subject_dedup_detects_same_creator_multiple_views(self) -> None:
        first = {"row": {"title": "Red Ford Mustang on street photograph"},
                 "creator_key": "same creator", "subject_tokens": {"red", "ford", "mustang", "street"}}
        same_subject = {"row": {"title": "Red Ford Mustang on street side view"},
                        "creator_key": "same creator", "subject_tokens": {"red", "ford", "mustang", "street"}}
        different_creator = {"row": {"title": "Red Ford Mustang on street side view"},
                             "creator_key": "another creator", "subject_tokens": {"red", "ford", "mustang", "street"}}
        self.assertTrue(commons._near_duplicate(same_subject, [first], "Car"))
        self.assertFalse(commons._near_duplicate(different_creator, [first], "Car"))

    def test_common_license_labels_are_not_candidate_filters_or_approvals(self) -> None:
        for pageid, license_name in enumerate(("CC BY-SA 4.0", "CC BY 4.0", "CC0 1.0", "Public domain"), start=51):
            with self.subTest(license=license_name):
                title = f"File:Candidate {pageid}.jpg"
                ext = {"LicenseShortName": {"value": license_name}}
                client = MockAPI([search_page((pageid, title))], [info_page(pageid, title, ext=ext)])
                rows, _ = commons.build_manifest(client, ["Flower"], 1)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["license_name"], license_name)
                self.assertEqual(rows[0]["license_review_status"], "pending_manual_review")

    def test_unknown_license_and_attribution_are_not_inferred(self) -> None:
        client = MockAPI([search_page((18, "File:Unknown.jpg"))], [info_page(18, "File:Unknown.jpg")])
        rows, _ = commons.build_manifest(client, ["Cat"], 1)
        row = rows[0]
        self.assertEqual(row["license_name"], "unknown")
        self.assertEqual(row["license_identifier"], "unknown")
        self.assertEqual(row["attribution_required"], "unknown")
        self.assertEqual(row["attribution_text"], "unknown")
        self.assertEqual(row["license_review_status"], "pending_manual_review")

    def test_missing_file_properties_are_explicitly_flagged(self) -> None:
        page = {"pageid": 19, "title": "File:Partial.jpg", "imageinfo": [{"mime": "image/jpeg"}]}
        client = MockAPI([search_page((19, "File:Partial.jpg"))], [page])
        rows, _ = commons.build_manifest(client, ["Chair"], 1)
        row = rows[0]
        self.assertEqual(row["original_width"], "unknown")
        self.assertEqual(row["original_size_bytes"], "unknown")
        self.assertEqual(row["sha1"], "unknown")
        self.assertIn("original_width_unknown", row["metadata_status"])
        self.assertIn("original_size_bytes_unknown", row["metadata_status"])
        self.assertIn("sha1_unknown", row["metadata_status"])
        self.assertEqual(row["license_review_status"], "pending_manual_review")

    def test_pagination_and_duplicate_sha1_are_handled(self) -> None:
        responses = [
            search_page((21, "File:First.jpg"), continuation={"continue": "-||", "sroffset": 1}),
            search_page((22, "File:Duplicate.jpg"), (23, "File:Third.jpg")),
        ]
        info = [info_page(21, "File:First.jpg", sha1="same"),
                info_page(22, "File:Duplicate.jpg", sha1="same"),
                info_page(23, "File:Third.jpg", sha1="different")]
        client = MockAPI(responses, info)
        rows, summary = commons.build_manifest(client, ["Bird"], 2)
        self.assertEqual([row["commons_page_id"] for row in rows], ["21", "23"])
        self.assertEqual(summary["duplicate_sha1_skipped_per_category"], {"Bird": 1})
        self.assertEqual(len([call for call in client.calls if call.get("list") == "search"]), 2)

    def test_same_page_id_is_emitted_once_and_records_categories(self) -> None:
        client = MockAPI(
            [search_page((31, "File:Shared.jpg")), search_page((31, "File:Shared.jpg"))],
            [info_page(31, "File:Shared.jpg")],
        )
        rows, _ = commons.build_manifest(client, ["Dog", "Cat"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["all_selected_categories"], "Cat;Dog")
        self.assertEqual(rows[0]["license_review_status"], "pending_manual_review")

    def test_retry_after_and_user_agent_are_respected(self) -> None:
        class FakeResponse:
            status = 429
            headers = {"Retry-After": "2"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"query":{}}'

        requests: list[Request] = []
        sleeps: list[float] = []

        def opener(request: Request, timeout: float):
            requests.append(request)
            self.assertEqual(timeout, 7)
            if len(requests) == 1:
                return FakeResponse()
            response = FakeResponse()
            response.status = 200
            return response

        client = commons.CommonsAPIClient("Sketch2Vision-tests (contact: test@example.invalid)", 7, 1,
                                          opener=opener, sleep=sleeps.append)
        self.assertEqual(client.get_json({"action": "query"}), {"query": {}})
        self.assertEqual(sleeps, [2.0])
        self.assertEqual(requests[0].get_header("User-agent"), "Sketch2Vision-tests (contact: test@example.invalid)")
        self.assertTrue(all(request.full_url.startswith(commons.API_URL + "?") for request in requests))

    def test_cli_help_is_available_without_network(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "backend/ml/scripts/build_commons_manifest.py"), "--help"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-csv", result.stdout)
        self.assertIn("--category", result.stdout)
        self.assertIn("--max-candidates-per-category", result.stdout)
        self.assertIn("--summary-json", result.stdout)
        self.assertIn("--max-per-creator-per-category", result.stdout)

    def test_summary_writer_serializes_and_protects_existing_output(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix=".commons-summary-test-", dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "summary.json"
            summary = {"selected_per_category": {"Dog": 2}, "license_review_status": "pending_manual_review"}
            commons.write_summary(path, summary)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), summary)
            with self.assertRaises(FileExistsError):
                commons.write_summary(path, summary)


if __name__ == "__main__":
    unittest.main()
