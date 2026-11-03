"""Build a metadata-only Smithsonian Open Access candidate manifest.

The API is queried for collection metadata only. ``online_media`` URLs are
recorded but never requested. Explicit CC0 records go to the eligible-candidate
view; all other or ambiguous rights records go to a separate review view.
Neither output constitutes license approval: every row remains pending review.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://api.si.edu/openaccess/api/v1.0"
SEARCH_ENDPOINT = f"{API_BASE}/search"
API_DOCS_URL = "https://www.si.edu/openaccess/faq"
CATEGORIES = {
    "Person": ("person portrait photograph",),
    "Dog": ("dog photograph", "dogs photograph"),
    "Cat": ("cat photograph", "cats photograph"),
    "Car": ("car automobile photograph",),
    "Bicycle": ("bicycle photograph", "bicycles photograph"),
    "Chair": ("chair furniture photograph",),
    "Bird": ("bird photograph", "birds photograph"),
    "Flower": ("flower photograph", "flowers photograph"),
    "Tree": ("tree photograph", "trees photograph"),
    "Building": ("building architecture photograph",),
}
CC0_MARKERS = {"cc0", "cc0 1.0", "cc0 1.0 universal", "https://creativecommons.org/publicdomain/zero/1.0/"}
FIELDS = (
    "source_identifier", "source_path", "selected_category", "matched_categories",
    "item_id", "title", "creator", "unit", "object_type", "api_url",
    "item_page_url", "image_url", "rights_statement", "usage_terms",
    "category_relevance", "metadata_completeness", "missing_metadata_fields", "license_name",
    "license_identifier", "attribution_required", "attribution_text",
    "source_url", "source_page_url", "source_metadata_url",
    "license_review_status", "reviewer_notes", "source_metadata_raw",
)


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(text for item in value if (text := _as_text(item)))
    if isinstance(value, dict):
        # Smithsonian's freetext values commonly use {content: ...}.
        for key in ("content", "value", "name", "access", "text"):
            if key in value:
                text = _as_text(value[key])
                if text:
                    return text
        return ""
    return str(value).strip() if value is not None else ""


def _walk_values(value: Any, keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in keys:
                text = _as_text(child)
                if text and text not in found:
                    found.append(text)
            found.extend(item for item in _walk_values(child, keys) if item not in found)
    elif isinstance(value, list):
        for child in value:
            found.extend(item for item in _walk_values(child, keys) if item not in found)
    return found


def _id(record: dict[str, Any]) -> str:
    content = record.get("content") if isinstance(record.get("content"), dict) else {}
    descriptive = content.get("descriptiveNonRepeating") if isinstance(content.get("descriptiveNonRepeating"), dict) else {}
    for value in (record.get("id"), descriptive.get("record_ID")):
        candidate = _as_text(value)
        if candidate:
            return candidate
    return ""


def _cc0_is_explicit(rights: str, usage: str) -> bool:
    """Accept only an explicit CC0 marker in the rights/access value."""
    normalized = rights.strip().casefold().rstrip("/")
    # A generic use term or mention of CC0 in unrelated prose is not evidence.
    return normalized in {marker.rstrip("/") for marker in CC0_MARKERS}


def _record_fields(record: dict[str, Any], category: str, matched_categories: list[str], api_url: str) -> dict[str, Any]:
    content = record.get("content") if isinstance(record.get("content"), dict) else {}
    descriptive = content.get("descriptiveNonRepeating") if isinstance(content.get("descriptiveNonRepeating"), dict) else {}
    freetext = content.get("freetext") if isinstance(content.get("freetext"), dict) else {}
    # In EDAN records, online_media is nested under
    # content.descriptiveNonRepeating. Keep the top-level fallback for older
    # or alternate response records, but do not confuse metadata_usage.access
    # with rights on an image file.
    media_value = descriptive.get("online_media")
    if not isinstance(media_value, dict):
        media_value = content.get("online_media")
    media = media_value if isinstance(media_value, dict) else {}
    media_items = media.get("media", [])
    first_media = next((item for item in media_items if isinstance(item, dict)), {}) if isinstance(media_items, list) else {}
    usage_raw = first_media.get("usage")
    usage_meta = usage_raw if isinstance(usage_raw, dict) else {}

    item_id = _id(record)
    title = _as_text(record.get("title")) or _as_text(descriptive.get("title"))
    creator_values = _walk_values(freetext, {"name", "maker", "creator", "artist", "photographer"})
    creator = "; ".join(creator_values)
    unit = _as_text(record.get("unitCode")) or _as_text(record.get("unit"))
    object_type = _as_text(record.get("type")) or "; ".join(_walk_values(freetext, {"object_type", "objecttype"}))
    item_page_url = _as_text(descriptive.get("record_link")) or _as_text(record.get("url"))
    if item_page_url and item_page_url.startswith("http://"):
        item_page_url = "https://" + item_page_url[len("http://"):]
    image_url = _as_text(first_media.get("content")) or _as_text(first_media.get("url"))
    rights = (
        _as_text(usage_meta.get("access")) or (_as_text(usage_raw) if not isinstance(usage_raw, dict) else "")
        or _as_text(first_media.get("usage_access"))
        or _as_text(record.get("media_usage")) or _as_text(record.get("rights"))
    )
    usage_terms = _as_text(usage_meta.get("text")) or _as_text(usage_meta.get("usage")) or _as_text(first_media.get("usage_terms"))
    # Preserve explicit media usage text if the schema returns it as a scalar.
    if not usage_terms:
        usage_terms = _as_text(usage_meta)

    completeness_values = {
        "item_id": item_id, "title": title, "creator": creator, "unit": unit,
        "object_type": object_type, "item_page_url": item_page_url,
        "image_url": image_url, "rights_statement": rights, "usage_terms": usage_terms,
    }
    missing = [name for name, value in completeness_values.items() if not value]
    is_cc0 = _cc0_is_explicit(rights, usage_terms)
    source_id = item_id
    return {
        "source_identifier": source_id,
        # Extension and safe local filename are assigned during a later,
        # separately reviewed acquisition step; do not guess either here.
        "source_path": "",
        "selected_category": category,
        "matched_categories": ";".join(matched_categories),
        "item_id": item_id,
        "title": title,
        "creator": creator,
        "unit": unit,
        "object_type": object_type,
        "api_url": api_url,
        "item_page_url": item_page_url,
        "image_url": image_url,
        "rights_statement": rights,
        "usage_terms": usage_terms,
        "category_relevance": "unverified",
        "metadata_completeness": "complete" if not missing else "incomplete",
        "missing_metadata_fields": ";".join(missing),
        "license_name": "CC0" if is_cc0 else rights,
        "license_identifier": "https://creativecommons.org/publicdomain/zero/1.0/" if is_cc0 else "",
        "attribution_required": "false" if is_cc0 else "unknown",
        "attribution_text": "",
        "source_url": image_url,
        "source_page_url": item_page_url,
        "source_metadata_url": api_url,
        "license_review_status": "pending_manual_review",
        "reviewer_notes": "Explicit CC0 metadata is a candidate filter only; human review remains required." if is_cc0 else "Rights status is not explicitly CC0; verify the item record and media rights manually.",
        "source_metadata_raw": json.dumps(record, ensure_ascii=False, sort_keys=True),
    }


class SmithsonianClient:
    """Minimal API client; never requests any item media URL."""

    def __init__(self, api_key: str, *, timeout: float = 25.0,
                 opener: Callable[..., Any] | None = None) -> None:
        if not api_key.strip():
            raise ValueError("Smithsonian API key is required; set SMITHSONIAN_API_KEY")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.api_key = api_key
        self.timeout = timeout
        self.opener = opener or urlopen
        self.request_count = 0

    def search(self, query: str, start: int, rows: int) -> dict[str, Any]:
        params = {"q": query, "start": start, "rows": rows, "api_key": self.api_key}
        request = Request(SEARCH_ENDPOINT + "?" + urlencode(params), headers={
            "Accept": "application/json",
            "User-Agent": "Sketch2VisionMetadataResearch/1.0 (metadata-only; contact configured by operator)",
        })
        self.request_count += 1
        try:
            with self.opener(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                body = response.read()
        except HTTPError as exc:
            body = exc.read(2048)
            excerpt = _safe_response_excerpt(body, self.api_key)
            detail = f"; response excerpt: {excerpt}" if excerpt else ""
            raise RuntimeError(f"Smithsonian API HTTP {exc.code}: {exc.reason}{detail}") from None
        except URLError as exc:
            raise RuntimeError(f"Smithsonian API request failed: {exc.reason}") from exc
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            excerpt = _safe_response_excerpt(body, self.api_key)
            detail = f"; response excerpt: {excerpt}" if excerpt else "(empty response)"
            raise RuntimeError(f"Smithsonian API HTTP {status} returned invalid JSON {detail}") from None
        if not isinstance(payload, dict):
            excerpt = _safe_response_excerpt(body, self.api_key)
            raise RuntimeError(
                f"Smithsonian API HTTP {status} returned unexpected JSON type "
                f"{type(payload).__name__}; response excerpt: {excerpt or '(empty response)'}"
            )

        # EDAN/Open Access uses an application responseCode of 1 for success;
        # the HTTP status is separately 200. Older proxy responses may omit
        # responseCode, so in that case validate the actual response.rows shape.
        response_code = payload.get("responseCode")
        response_data = payload.get("response")
        rows_are_present = isinstance(response_data, dict) and isinstance(response_data.get("rows"), list)
        code_is_success = response_code in (1, "1", 200, "200")
        try:
            status_code = int(status)
        except (TypeError, ValueError):
            status_code = 0
        if not (200 <= status_code < 300) or (response_code is not None and not code_is_success) or not rows_are_present:
            excerpt = _safe_response_excerpt(body, self.api_key)
            code_text = f", responseCode={response_code!r}" if response_code is not None else ""
            raise RuntimeError(
                f"Smithsonian API HTTP {status}{code_text} returned an error or unexpected schema; "
                f"expected response.rows list; response excerpt: {excerpt or '(empty response)'}"
            )
        return payload


def _safe_response_excerpt(body: bytes, api_key: str, limit: int = 500) -> str:
    """Return a bounded diagnostic excerpt with credential forms redacted."""
    text = body.decode("utf-8", errors="replace")
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"(?i)(api_key\s*[=:]\s*)[^&\s,}\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _safe_api_url(query: str, start: int, rows: int) -> str:
    # Deliberately excludes api_key so credentials cannot enter manifest files.
    return SEARCH_ENDPOINT + "?" + urlencode({"q": query, "start": start, "rows": rows})


def _eligible_cc0(row: dict[str, Any]) -> bool:
    return bool(str(row.get("item_id", "")).strip()) and _explicit_cc0(row)


def _explicit_cc0(row: dict[str, Any]) -> bool:
    return _cc0_is_explicit(
        str(row.get("rights_statement", "")), str(row.get("usage_terms", ""))
    )


def collect_candidates(client: SmithsonianClient, categories: list[str], limit: int = 50) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if limit < 1 or limit > 50:
        raise ValueError("--max-candidates-per-category must be between 1 and 50")
    invalid = [category for category in categories if category not in CATEGORIES]
    if invalid:
        raise ValueError("unsupported category: " + ", ".join(invalid))

    by_id: dict[str, dict[str, Any]] = {}
    missing_id_records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    query_manifest: list[dict[str, Any]] = []
    for category in categories:
        raw_count = 0
        missing_id_count = 0
        category_unique_ids: list[str] = []
        # Search each category with one curated phrase; page only enough to get
        # the requested metadata candidate cap. No media URL is ever opened.
        query = CATEGORIES[category][0]
        start = 0
        while raw_count < limit:
            page_rows = min(50, limit - raw_count)
            payload = client.search(query, start, page_rows)
            response = payload.get("response", {})
            rows = response.get("rows", []) if isinstance(response, dict) else []
            if not isinstance(rows, list) or not rows:
                break
            for record in rows:
                if not isinstance(record, dict):
                    continue
                raw_count += 1
                item_id = _id(record)
                if item_id:
                    if item_id not in by_id:
                        by_id[item_id] = _record_fields(record, category, [category], _safe_api_url(query, start, page_rows))
                    else:
                        matched = by_id[item_id]["matched_categories"].split(";")
                        if category not in matched:
                            matched.append(category)
                            by_id[item_id]["matched_categories"] = ";".join(matched)
                    if item_id not in category_unique_ids:
                        category_unique_ids.append(item_id)
                else:
                    # Preserve malformed/incomplete records in the review view;
                    # without a stable ID they cannot be safely deduplicated.
                    missing_id_records.append(_record_fields(record, category, [category], _safe_api_url(query, start, page_rows)))
                    missing_id_count += 1
                if raw_count >= limit:
                    break
            query_manifest.append({"category": category, "query": query, "start": start, "rows": page_rows})
            start += len(rows)
            if len(rows) < page_rows:
                break
        counts[category] = {
            "search_records": raw_count,
            "unique_item_ids": len(category_unique_ids),
            "eligible_cc0": sum(1 for item_id in category_unique_ids if _eligible_cc0(by_id[item_id])),
            "uncertain_rights": sum(1 for item_id in category_unique_ids if not _eligible_cc0(by_id[item_id])),
            "records_without_item_id": missing_id_count,
            "shortfall_from_limit": max(0, limit - len(category_unique_ids)),
        }

    rows = list(by_id.values()) + missing_id_records
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Smithsonian Open Access API",
        "api_endpoint": SEARCH_ENDPOINT,
        "metadata_only": True,
        "image_bytes_downloaded": False,
        "license_approval_granted": False,
        "all_review_statuses": "pending_manual_review",
        "requested_max_per_category": limit,
        "api_request_count": client.request_count,
        "deduplication": {"key": "stable Smithsonian item id", "unique_records_with_id": len(by_id), "records_without_id_preserved_in_review": len(missing_id_records), "duplicate_ids_collapsed": max(0, sum(counts[c]["search_records"] for c in counts) - len(by_id) - len(missing_id_records))},
        "categories": counts,
        "total_unique_cc0_candidates": sum(1 for row in rows if _eligible_cc0(row)),
        "total_records_with_non_cc0_or_unknown_rights": sum(1 for row in rows if not _explicit_cc0(row)),
        "records_without_item_id": len(missing_id_records),
        "missing_metadata_counts": dict(Counter(field for row in rows for field in row["missing_metadata_fields"].split(";") if field)),
        "queries": query_manifest,
        "notes": ["CC0 is a candidate filter only, not an approval.", "Category relevance requires human review.", "No image URL was fetched."],
    }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], eligible_csv: Path,
                  review_csv: Path, summary_json: Path, *, overwrite: bool = False) -> None:
    paths = (eligible_csv, review_csv, summary_json)
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("output already exists; use --overwrite only when replacement is intentional")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in rows if _eligible_cc0(row)]
    review = [row for row in rows if not _eligible_cc0(row)]
    for path, records in ((eligible_csv, eligible), (review_csv, review)):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    summary = dict(summary)
    summary["eligible_cc0_view_count"] = len(eligible)
    summary["review_csv_count"] = len(review)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search Smithsonian metadata; never download image files or approve rights.")
    parser.add_argument("--eligible-csv", type=Path, required=True, help="CSV containing records with an explicit CC0 rights/access marker")
    parser.add_argument("--review-csv", type=Path, required=True, help="CSV containing missing, unknown, or non-CC0 rights records")
    parser.add_argument("--summary-json", type=Path, required=True, help="JSON with category counts and missing metadata totals")
    parser.add_argument("--category", action="append", choices=tuple(CATEGORIES), help="Category to query; repeatable (default: all 10)")
    parser.add_argument("--max-candidates-per-category", type=int, default=50, help="Maximum search records per category, 1-50 (default 50)")
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    api_key = os.environ.get("SMITHSONIAN_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SMITHSONIAN_API_KEY is not set; obtain an API key from api.data.gov and set it in the current shell.")
    try:
        client = SmithsonianClient(api_key, timeout=args.timeout)
        rows, summary = collect_candidates(client, args.category or list(CATEGORIES), args.max_candidates_per_category)
        write_outputs(rows, summary, args.eligible_csv, args.review_csv, args.summary_json, overwrite=args.overwrite)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({
        "unique_candidates": len(rows),
        "cc0_candidates": summary["total_unique_cc0_candidates"],
        "uncertain_rights_candidates": summary["total_records_with_non_cc0_or_unknown_rights"],
        "categories": summary["categories"],
        "api_request_count": client.request_count,
        "outputs": {"eligible_csv": str(args.eligible_csv), "review_csv": str(args.review_csv), "summary_json": str(args.summary_json)},
        "image_bytes_downloaded": False,
        "license_approval_granted": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
