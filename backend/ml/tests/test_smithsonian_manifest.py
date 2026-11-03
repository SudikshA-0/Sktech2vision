"""Offline tests for the Smithsonian metadata-only adapter."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from backend.ml.scripts import build_smithsonian_manifest as manifest


def _record(item_id: str, rights: str = "CC0", *, title: str | None = None) -> dict:
    return {
        "id": item_id,
        "title": title or f"{item_id} test subject",
        "unitCode": "NMNH",
        "type": "Photograph",
        "url": f"https://www.si.edu/object/{item_id}",
        "content": {
            "descriptiveNonRepeating": {
                "record_ID": item_id,
                "record_link": f"https://www.si.edu/object/{item_id}",
            },
            "freetext": {"name": [{"label": "Photographer", "content": "Example Creator"}]},
            "online_media": {"media": [{
                "content": f"https://ids.si.edu/ids/deliveryService?id={item_id}",
                "usage": {"access": rights, "text": "Usage terms text"},
            }]},
        },
    }


class FakeClient:
    def __init__(self, by_query: dict[str, list[dict]]) -> None:
        self.by_query = by_query
        self.request_count = 0
        self.calls = []

    def search(self, query: str, start: int, rows: int) -> dict:
        self.request_count += 1
        self.calls.append((query, start, rows))
        records = self.by_query.get(query, [])
        return {"response": {"rows": records[start:start + rows]}}


class FakeHTTPResponse:
    status = 200

    def __init__(self, body: dict) -> None:
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


class SmithsonianManifestTests(unittest.TestCase):
    def test_categories_are_bounded_deduplicated_and_rights_separated(self) -> None:
        dog_query = manifest.CATEGORIES["Dog"][0]
        cat_query = manifest.CATEGORIES["Cat"][0]
        client = FakeClient({
            dog_query: [_record("A", "CC0"), _record("B", "Usage Conditions Apply")],
            cat_query: [_record("A", "CC0"), _record("C", "Not determined")],
        })
        rows, summary = manifest.collect_candidates(client, ["Dog", "Cat"], limit=5)
        self.assertEqual({row["item_id"] for row in rows}, {"A", "B", "C"})
        self.assertEqual(sum(manifest._cc0_is_explicit(r["rights_statement"], r["usage_terms"]) for r in rows), 1)
        self.assertEqual(summary["categories"]["Dog"]["eligible_cc0"], 1)
        self.assertEqual(summary["categories"]["Cat"]["eligible_cc0"], 1)
        self.assertEqual(summary["deduplication"]["duplicate_ids_collapsed"], 1)
        self.assertEqual(next(row for row in rows if row["item_id"] == "A")["matched_categories"], "Dog;Cat")
        self.assertTrue(all(r["license_review_status"] == "pending_manual_review" for r in rows))
        self.assertFalse(summary["license_approval_granted"])

    def test_unknown_rights_and_missing_metadata_go_to_review_without_inference(self) -> None:
        row = _record("U1", "")
        row["content"]["online_media"]["media"][0]["usage"] = {}
        row["content"]["freetext"] = {}
        client = FakeClient({manifest.CATEGORIES["Bird"][0]: [row]})
        records, summary = manifest.collect_candidates(client, ["Bird"], limit=3)
        self.assertEqual(records[0]["rights_statement"], "")
        self.assertEqual(records[0]["license_name"], "")
        self.assertEqual(records[0]["attribution_required"], "unknown")
        self.assertIn("creator", records[0]["missing_metadata_fields"])
        self.assertEqual(summary["total_records_with_non_cc0_or_unknown_rights"], 1)
        self.assertEqual(records[0]["license_review_status"], "pending_manual_review")

    def test_explicit_cc0_only_and_mentions_do_not_count(self) -> None:
        self.assertTrue(manifest._cc0_is_explicit("CC0", ""))
        self.assertTrue(manifest._cc0_is_explicit("https://creativecommons.org/publicdomain/zero/1.0/", ""))
        self.assertFalse(manifest._cc0_is_explicit("Not determined", "CC0 may apply"))
        self.assertFalse(manifest._cc0_is_explicit("Usage Conditions Apply", ""))

    def test_actual_edan_nested_media_rights_are_mapped_but_not_approved(self) -> None:
        record = {
            "id": "edanmdm-example_123",
            "title": "Dog portrait",
            "unitCode": "NMAH",
            "type": "edanmdm",
            "content": {
                "descriptiveNonRepeating": {
                    "record_ID": "example_123",
                    "record_link": "https://www.si.edu/object/example_123",
                    "title": {"content": "Dog portrait"},
                    "unit_code": "NMAH",
                    "metadata_usage": {"access": "CC0"},
                    "online_media": {"media": [{
                        "type": "Images",
                        "content": "https://ids.si.edu/ids/deliveryService?id=example_123",
                        "usage": {"access": "CC0"},
                    }]},
                },
                "freetext": {"name": [{"label": "Photographer", "content": "Example Creator"}]},
            },
        }
        query = manifest.CATEGORIES["Dog"][0]
        row = manifest._record_fields(record, "Dog", ["Dog"], manifest._safe_api_url(query, 0, 1))
        self.assertEqual(row["image_url"], "https://ids.si.edu/ids/deliveryService?id=example_123")
        self.assertEqual(row["rights_statement"], "CC0")
        self.assertEqual(row["license_name"], "CC0")
        self.assertTrue(manifest._eligible_cc0(row))
        self.assertEqual(row["license_review_status"], "pending_manual_review")

    def test_metadata_cc0_without_image_media_is_not_treated_as_image_rights(self) -> None:
        record = {
            "id": "edanmdm-example_text_only",
            "title": "Dog book record",
            "content": {"descriptiveNonRepeating": {
                "record_ID": "example_text_only",
                "metadata_usage": {"access": "CC0"},
            }},
        }
        row = manifest._record_fields(record, "Dog", ["Dog"], "https://api.si.edu/openaccess/api/v1.0/search")
        self.assertEqual(row["rights_statement"], "")
        self.assertEqual(row["image_url"], "")
        self.assertFalse(manifest._eligible_cc0(row))
        self.assertEqual(row["license_review_status"], "pending_manual_review")

    def test_outputs_are_separated_pending_and_do_not_expose_api_key(self) -> None:
        query = manifest.CATEGORIES["Dog"][0]
        client = FakeClient({query: [_record("CC", "CC0"), _record("UNK", "unknown")]})
        rows, summary = manifest.collect_candidates(client, ["Dog"], limit=5)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            eligible, review, summary_path = root / "eligible.csv", root / "review.csv", root / "summary.json"
            manifest.write_outputs(rows, summary, eligible, review, summary_path)
            self.assertEqual(len(eligible.read_text(encoding="utf-8-sig").splitlines()), 2)
            self.assertEqual(len(review.read_text(encoding="utf-8-sig").splitlines()), 2)
            content = eligible.read_text(encoding="utf-8-sig") + review.read_text(encoding="utf-8-sig")
            self.assertNotIn("api_key", content)
            self.assertIn("pending_manual_review", content)
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertFalse(saved["image_bytes_downloaded"])
            self.assertEqual(saved["eligible_cc0_view_count"], 1)
            with self.assertRaises(FileExistsError):
                manifest.write_outputs(rows, summary, eligible, review, summary_path)

    def test_client_requires_key_without_echoing_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "SMITHSONIAN_API_KEY"):
            manifest.SmithsonianClient("")

    def test_client_uses_official_search_endpoint_and_keeps_key_out_of_manifest_url(self) -> None:
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return FakeHTTPResponse({"responseCode": 200, "response": {"rows": [_record("ID1")]}})

        client = manifest.SmithsonianClient("test-secret-key", opener=opener)
        payload = client.search("dog photograph", 0, 5)
        params = parse_qs(urlsplit(captured["url"]).query)
        self.assertEqual(urlsplit(captured["url"]).scheme, "https")
        self.assertEqual(urlsplit(captured["url"]).netloc, "api.si.edu")
        self.assertEqual(params["api_key"], ["test-secret-key"])
        self.assertEqual(payload["response"]["rows"][0]["id"], "ID1")
        self.assertNotIn("test-secret-key", manifest._safe_api_url("dog photograph", 0, 5))

    def test_client_accepts_smithsonian_response_code_one(self) -> None:
        expected = {"status": 200, "responseCode": 1, "response": {"rowCount": 1, "rows": [_record("DOG1")]}}
        client = manifest.SmithsonianClient("key", opener=lambda request, timeout: FakeHTTPResponse(expected))
        result = client.search("dog photograph", 0, 5)
        self.assertEqual(result["response"]["rows"][0]["id"], "DOG1")

    def test_application_error_includes_http_status_and_safe_excerpt(self) -> None:
        payload = {"status": 200, "responseCode": 0, "responseMessage": "api_key=secret-key rejected", "response": {"rows": []}}
        client = manifest.SmithsonianClient("secret-key", opener=lambda request, timeout: FakeHTTPResponse(payload))
        with self.assertRaisesRegex(RuntimeError, r"HTTP 200, responseCode=0") as raised:
            client.search("dog photograph", 0, 5)
        self.assertIn("[REDACTED]", str(raised.exception))
        self.assertNotIn("secret-key", str(raised.exception))

    def test_http_error_reports_status_and_limited_redacted_body(self) -> None:
        body = b'{"error":"bad key secret-key","api_key":"secret-key"}'

        def opener(request, timeout):
            raise HTTPError(request.full_url, 403, "Forbidden", None, BytesIO(body))

        client = manifest.SmithsonianClient("secret-key", opener=opener)
        with self.assertRaisesRegex(RuntimeError, r"HTTP 403: Forbidden") as raised:
            client.search("cat photograph", 0, 5)
        self.assertNotIn("secret-key", str(raised.exception))
        self.assertIn("response excerpt", str(raised.exception))

    def test_malformed_response_schema_reports_status_excerpt(self) -> None:
        client = manifest.SmithsonianClient(
            "key", opener=lambda request, timeout: FakeHTTPResponse({"status": 200, "responseCode": 1, "response": []})
        )
        with self.assertRaisesRegex(RuntimeError, r"HTTP 200.*expected response.rows list"):
            client.search("bird photograph", 0, 5)

    def test_invalid_json_reports_status_and_bounded_excerpt(self) -> None:
        client = manifest.SmithsonianClient("key", opener=lambda request, timeout: FakeHTTPResponse(b"<html>not JSON"))
        with self.assertRaisesRegex(RuntimeError, r"HTTP 200 returned invalid JSON") as raised:
            client.search("chair photograph", 0, 5)
        self.assertIn("<html>not JSON", str(raised.exception))

    def test_missing_item_id_is_preserved_in_review_and_counted(self) -> None:
        record = _record("NO_ID", "CC0")
        del record["id"]
        del record["content"]["descriptiveNonRepeating"]["record_ID"]
        client = FakeClient({manifest.CATEGORIES["Tree"][0]: [record]})
        rows, summary = manifest.collect_candidates(client, ["Tree"], limit=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item_id"], "")
        self.assertEqual(summary["records_without_item_id"], 1)
        self.assertEqual(summary["total_unique_cc0_candidates"], 0)
        self.assertEqual(summary["total_records_with_non_cc0_or_unknown_rights"], 0)
        self.assertFalse(manifest._eligible_cc0(rows[0]))

    def test_category_and_limit_validation(self) -> None:
        client = FakeClient({})
        with self.assertRaisesRegex(ValueError, "unsupported category"):
            manifest.collect_candidates(client, ["Not a category"])
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            manifest.collect_candidates(client, ["Dog"], 51)


if __name__ == "__main__":
    unittest.main()
