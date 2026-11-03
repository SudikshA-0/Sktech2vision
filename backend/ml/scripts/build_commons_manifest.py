"""Build a metadata-only Wikimedia Commons candidate manifest.

Only MediaWiki JSON API responses are requested. This script never fetches
image URLs or downloads image bytes. Every record remains pending manual
license review.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import html
import json
import os
import re
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://commons.wikimedia.org/w/api.php"
DEFAULT_USER_AGENT = (
    "Sketch2VisionCommonsManifest/1.0 (academic metadata-only candidate research)"
)
CATEGORIES = (
    "Person", "Dog", "Cat", "Car", "Bicycle", "Chair", "Bird", "Flower", "Tree", "Building",
)
SEARCH_BASE_TERMS = {
    "Person": "person portrait",
    "Dog": "dog animal",
    "Cat": "cat animal",
    "Car": "car automobile",
    "Bicycle": "bicycle cycle",
    "Chair": "chair furniture",
    "Bird": "bird wildlife",
    "Flower": "flower plant",
    "Tree": "tree plant",
    "Building": "building architecture",
}
NEGATIVE_SEARCH_TERMS = (
    "poster posters diagram diagrams illustration illustrations logo logos painting paintings "
    "drawing drawings map maps icon icons infographic infographics cartoon cartoons screenshot "
    "screenshots rendering clipart vector artwork artworks"
)
SEARCH_TERMS = {
    category: f"{terms} photograph filetype:bitmap "
              + " ".join(f"-{term}" for term in NEGATIVE_SEARCH_TERMS.split())
    for category, terms in SEARCH_BASE_TERMS.items()
}
NON_PHOTOGRAPH_TERMS = (
    "poster", "diagram", "illustration", "logo", "painting", "drawing", "map", "icon",
    "infographic", "cartoon", "comic", "screenshot", "rendering", "clipart", "vector",
    "coat of arms", "sculpture", "stamp", "artwork",
)
PHOTO_TERMS = {"photograph", "photography", "photographed", "photo", "camera"}
SUBJECT_STOPWORDS = {
    "the", "and", "with", "from", "commons", "wikipedia", "file", "image", "photo",
    "photograph", "photography", "view", "front", "rear", "side", "left", "right",
    "jpg", "jpeg", "png", "original", "detail", "close", "up", "at", "in", "on",
    "of", "a", "an", "by", "near", "category",
}
EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png"}
LICENSE_VERSION_RE = re.compile(r"creativecommons\.org/licenses/[^/]+/(\d+(?:\.\d+)?)/?", re.I)
SHORT_LICENSE_VERSION_RE = re.compile(r"\b(\d+\.\d+)\s*$")
MANIFEST_FIELDS = (
    "source_identifier", "source_path", "selected_category", "all_selected_categories",
    "source_url", "source_page_url", "source_metadata_url", "creator", "title",
    "attribution_text", "attribution_required", "license_name", "license_version",
    "license_identifier", "commons_page_id", "file_page_title", "file_timestamp",
    "mime_type", "original_width", "original_height", "original_size_bytes", "sha1",
    "license_review_status", "reviewer_notes", "metadata_status", "source_metadata_raw",
)


def _retry_after_seconds(value: str | None, fallback: float = 5.0) -> float:
    if not value:
        return fallback
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError, OverflowError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return fallback


class CommonsAPIClient:
    """Small sequential API client with bounded 429/503 handling."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 20.0,
                 retries: int = 3, opener: Callable[..., Any] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not user_agent.strip():
            raise ValueError("User-Agent must be a non-empty identifying string")
        if timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries cannot be negative")
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.opener = opener or urlopen
        self.sleep = sleep
        self.request_count = 0

    def get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        request_url = API_URL + "?" + urlencode(params)
        request = Request(request_url, headers={
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        })
        for attempt in range(self.retries + 1):
            try:
                self.request_count += 1
                response = self.opener(request, timeout=self.timeout)
                status = getattr(response, "status", 200)
                headers = getattr(response, "headers", {})
                if status in {429, 503}:
                    if attempt >= self.retries:
                        raise RuntimeError(f"Commons API returned HTTP {status} after {self.retries} retries")
                    self.sleep(_retry_after_seconds(headers.get("Retry-After")))
                    continue
                with response:
                    payload = response.read()
                result = json.loads(payload.decode("utf-8"))
                if not isinstance(result, dict):
                    raise ValueError("Commons API returned a non-object JSON response")
                if isinstance(result.get("error"), dict):
                    error = result["error"]
                    raise RuntimeError(
                        f"Commons API error {error.get('code', 'unknown')}: "
                        f"{error.get('info', 'no details supplied')}"
                    )
                return result
            except HTTPError as exc:
                if exc.code not in {429, 503} or attempt >= self.retries:
                    raise RuntimeError(f"Commons API HTTP {exc.code}: {exc.reason}") from exc
                self.sleep(_retry_after_seconds(exc.headers.get("Retry-After")))
            except URLError as exc:
                raise RuntimeError(f"Commons API request failed: {exc.reason}") from exc
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Commons API returned invalid JSON: {exc}") from exc
        raise RuntimeError("Commons API retry loop ended unexpectedly")


def _text(value: Any) -> str:
    """Convert Commons extmetadata's HTML-formatted value to readable text."""
    if isinstance(value, dict):
        value = value.get("value", "")
    if value is None:
        return ""
    raw = str(value)
    # extmetadata uses simple HTML; discard tags without interpreting entities.
    return html.unescape(re.sub(r"<[^>]*>", " ", raw)).strip()


def _ext_value(extmetadata: dict[str, Any], key: str) -> str:
    return _text(extmetadata.get(key))


def _attribution_flag(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"true", "yes", "1", "required"}:
        return "true"
    if normalized in {"false", "no", "0", "not required", "not_required"}:
        return "false"
    return "unknown"


def _license_version(license_url: str, license_name: str) -> str:
    # Extract a version only from an explicit canonical URL or label. This does
    # not classify the template or grant any rights.
    match = LICENSE_VERSION_RE.search(license_url)
    if match:
        return match.group(1)
    match = SHORT_LICENSE_VERSION_RE.search(license_name)
    return match.group(1) if match else "unknown"


def _page_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    pages = response.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        return [value for value in pages.values() if isinstance(value, dict)]
    return [value for value in pages if isinstance(value, dict)] if isinstance(pages, list) else []


def _detail_api_url(file_page_title: str) -> str:
    params = {
        "action": "query", "format": "json", "formatversion": "2", "prop": "imageinfo",
        "titles": file_page_title,
        "iiprop": "url|size|mime|sha1|timestamp|extmetadata",
        "iiextmetadatafilter": "Artist|ObjectName|ImageDescription|Categories|Credit|Attribution|AttributionRequired|LicenseShortName|LicenseUrl|UsageTerms",
    }
    return API_URL + "?" + urlencode(params)


def search_category(client: CommonsAPIClient, category: str, wanted: int,
                    page_size: int = 50) -> Iterable[dict[str, Any]]:
    """Yield paginated Commons File-namespace bitmap search results."""
    params: dict[str, Any] = {
        "action": "query", "format": "json", "formatversion": "2", "list": "search",
        "srnamespace": "6", "srsearch": SEARCH_TERMS[category],
        "srlimit": min(50, max(1, page_size)),
    }
    returned = 0
    while returned < wanted:
        response = client.get_json(params)
        results = response.get("query", {}).get("search", [])
        if not isinstance(results, list) or not results:
            break
        for item in results:
            if isinstance(item, dict):
                yield item
                returned += 1
                if returned >= wanted:
                    break
        continuation = response.get("continue")
        if not isinstance(continuation, dict) or not continuation:
            break
        params.update(continuation)


def fetch_imageinfo(client: CommonsAPIClient, titles: list[str], batch_size: int = 50) -> dict[str, dict[str, Any]]:
    """Fetch imageinfo/extmetadata in bounded batches (no media URLs fetched)."""
    result: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(titles), batch_size):
        chunk = titles[offset:offset + batch_size]
        params = {
            "action": "query", "format": "json", "formatversion": "2", "prop": "imageinfo",
            "titles": "|".join(chunk),
            "iiprop": "url|size|mime|sha1|timestamp|extmetadata",
            "iiextmetadatafilter": "Artist|ObjectName|ImageDescription|Categories|Credit|Attribution|AttributionRequired|LicenseShortName|LicenseUrl|UsageTerms",
        }
        for page in _page_items(client.get_json(params)):
            title = str(page.get("title") or "")
            infos = page.get("imageinfo")
            if title and isinstance(infos, list) and infos and isinstance(infos[0], dict):
                result[title.casefold()] = {"page": page, "imageinfo": infos[0]}
            elif title:
                result[title.casefold()] = {"page": page, "imageinfo": {}}
    return result


def _candidate_row(category: str, page: dict[str, Any], imageinfo: dict[str, Any]) -> dict[str, str]:
    pageid = str(page.get("pageid") or "unknown")
    file_title = str(page.get("title") or "unknown")
    source_identifier = f"commons-{pageid}" if pageid != "unknown" else "unknown"
    mime_type = str(imageinfo.get("mime") or "unknown").casefold()
    extmetadata = imageinfo.get("extmetadata")
    if not isinstance(extmetadata, dict):
        extmetadata = {}
    license_name = _ext_value(extmetadata, "LicenseShortName") or _ext_value(extmetadata, "UsageTerms") or "unknown"
    license_url = _ext_value(extmetadata, "LicenseUrl") or "unknown"
    creator = _ext_value(extmetadata, "Artist") or "unknown"
    title = _ext_value(extmetadata, "ObjectName") or file_title.removeprefix("File:")
    attribution = _ext_value(extmetadata, "AttributionRequired")
    credit = _ext_value(extmetadata, "Credit") or _ext_value(extmetadata, "Attribution") or "unknown"
    extension = EXTENSIONS.get(mime_type, "")
    metadata_status: list[str] = []
    source_url = str(imageinfo.get("url") or "unknown")
    source_page_url = str(imageinfo.get("descriptionurl") or "unknown")
    if source_url == "unknown":
        metadata_status.append("missing_direct_image_url")
    if source_page_url == "unknown":
        metadata_status.append("missing_description_page_url")
    if creator == "unknown":
        metadata_status.append("creator_missing")
    if not license_name or license_name == "unknown":
        metadata_status.append("license_name_missing")
    if not license_url or license_url == "unknown":
        metadata_status.append("license_url_missing")
    if attribution == "":
        metadata_status.append("attribution_requirement_unknown")
    if not extension:
        metadata_status.append("mime_not_supported_by_prepare_dataset")
    raw = {"search_page": page, "imageinfo": imageinfo}
    width = imageinfo.get("width")
    height = imageinfo.get("height")
    size = imageinfo.get("size")
    for field_name, value in (("file_timestamp", imageinfo.get("timestamp")),
                              ("original_width", width), ("original_height", height),
                              ("original_size_bytes", size), ("sha1", imageinfo.get("sha1"))):
        if value is None or str(value).strip() == "":
            metadata_status.append(f"{field_name}_unknown")
    return {
        "source_identifier": source_identifier,
        "source_path": f"{category}/{source_identifier}{extension}" if extension else "unknown",
        "selected_category": category,
        "all_selected_categories": category,
        "source_url": source_url,
        "source_page_url": source_page_url,
        "source_metadata_url": _detail_api_url(file_title),
        "creator": creator,
        "title": title or "unknown",
        "attribution_text": credit,
        "attribution_required": _attribution_flag(attribution),
        "license_name": license_name,
        "license_version": _license_version(license_url, license_name),
        "license_identifier": license_url,
        "commons_page_id": pageid,
        "file_page_title": file_title,
        "file_timestamp": str(imageinfo.get("timestamp") or "unknown"),
        "mime_type": mime_type,
        "original_width": str(width) if width is not None else "unknown",
        "original_height": str(height) if height is not None else "unknown",
        "original_size_bytes": str(size) if size is not None else "unknown",
        "sha1": str(imageinfo.get("sha1") or "unknown"),
        "license_review_status": "pending_manual_review",
        "reviewer_notes": "",
        "metadata_status": ";".join(metadata_status) if metadata_status else "metadata_present_review_required",
        "source_metadata_raw": json.dumps(raw, ensure_ascii=False, sort_keys=True),
    }


def _metadata_text(page: dict[str, Any], imageinfo: dict[str, Any]) -> str:
    ext = imageinfo.get("extmetadata")
    ext = ext if isinstance(ext, dict) else {}
    fields = [
        str(page.get("title") or ""), str(page.get("snippet") or ""),
        _ext_value(ext, "ObjectName"), _ext_value(ext, "ImageDescription"),
        _ext_value(ext, "Categories"),
    ]
    return " ".join(fields).casefold()


def _non_photograph_reasons(page: dict[str, Any], imageinfo: dict[str, Any]) -> list[str]:
    text = _metadata_text(page, imageinfo)
    reasons = []
    for term in NON_PHOTOGRAPH_TERMS:
        parts = [re.escape(part) for part in term.split()]
        parts[-1] += r"s?"
        pattern = r"\b" + r"\s+".join(parts) + r"\b"
        if re.search(pattern, text):
            reasons.append(term.replace(" ", "_"))
    return reasons


def _photo_signal_score(page: dict[str, Any], imageinfo: dict[str, Any]) -> int:
    text = _metadata_text(page, imageinfo)
    score = sum(bool(re.search(r"\b" + re.escape(term) + r"\b", text)) for term in PHOTO_TERMS)
    # JPEG is a weak tie-break signal only; it does not establish that a file is a photo.
    score += int(str(imageinfo.get("mime", "")).casefold() == "image/jpeg")
    return score


def _subject_tokens(row: dict[str, str], category: str) -> set[str]:
    category_terms = {
        "person": {"person", "people", "portrait", "human"},
        "dog": {"dog", "dogs", "canine", "animal"},
        "cat": {"cat", "cats", "feline", "animal"},
        "car": {"car", "cars", "automobile", "vehicle"},
        "bicycle": {"bicycle", "bicycles", "bike", "cycle"},
        "chair": {"chair", "chairs", "furniture"},
        "bird": {"bird", "birds", "wildlife", "animal"},
        "flower": {"flower", "flowers", "plant"},
        "tree": {"tree", "trees", "plant"},
        "building": {"building", "buildings", "architecture"},
    }.get(category.casefold(), set())
    title = " ".join((row.get("title", ""), row.get("file_page_title", ""))).casefold()
    tokens = set(re.findall(r"[a-z0-9]+", title))
    return tokens - SUBJECT_STOPWORDS - category_terms


def _near_duplicate(candidate: dict[str, Any], previous: list[dict[str, Any]], category: str) -> bool:
    candidate_title = re.sub(r"[^a-z0-9]+", " ", candidate["row"].get("title", "").casefold()).strip()
    candidate_creator = candidate["creator_key"]
    candidate_tokens = _subject_tokens(candidate["row"], category)
    for existing in previous:
        existing_row = existing.get("row", existing)
        existing_title = re.sub(r"[^a-z0-9]+", " ", existing_row.get("title", "").casefold()).strip()
        existing_creator = existing.get("creator_key")
        existing_tokens = _subject_tokens(existing_row, category)
        if candidate_title and candidate_title == existing_title and (
            len(candidate_tokens) >= 2 or (candidate_creator and candidate_creator == existing_creator)
        ):
            return True
        if existing_creator and candidate_creator and existing_creator == candidate_creator:
            smaller = min(len(candidate_tokens), len(existing_tokens))
            if smaller >= 2 and len(candidate_tokens & existing_tokens) / smaller >= 0.8:
                return True
    return False


def _select_creator_diverse(candidates: list[dict[str, Any]], quota: int,
                            max_per_creator: int) -> tuple[list[dict[str, Any]], int]:
    """Apply a soft creator cap: defer repeats, then use them to fill any gap."""
    ranked = sorted(candidates, key=lambda item: (-item["photo_score"], item["order"]))
    chosen: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    creator_counts: Counter[str] = Counter()
    for candidate in ranked:
        creator = candidate["creator_key"]
        if max_per_creator <= 0 or not creator or creator_counts[creator] < max_per_creator:
            chosen.append(candidate)
            if creator:
                creator_counts[creator] += 1
        else:
            deferred.append(candidate)
    selected = chosen[:quota]
    if len(selected) < quota:
        selected.extend(deferred[:quota - len(selected)])
    selected_creator_counts: Counter[str] = Counter()
    fallbacks = 0
    for item in selected:
        creator = item["creator_key"]
        if creator:
            if max_per_creator > 0 and selected_creator_counts[creator] >= max_per_creator:
                fallbacks += 1
            selected_creator_counts[creator] += 1
    return selected, fallbacks


def build_manifest(client: CommonsAPIClient, categories: list[str], max_per_category: int,
                   max_per_creator: int = 3) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    rows_by_pageid: dict[str, dict[str, str]] = {}
    seen_sha1: set[str] = set()
    counts = {category: 0 for category in categories}
    previously_selected: list[dict[str, Any]] = []
    missing_imageinfo = Counter()
    duplicate_sha1 = Counter()
    unsupported_mime = Counter()
    malformed_page = Counter()
    duplicate_page_ids = Counter()
    eligible_counts = {category: 0 for category in categories}
    rejected_non_photograph = Counter()
    near_duplicate_subjects = Counter()
    creator_fallbacks = Counter()
    creator_diversity = {category: 0 for category in categories}

    for category in categories:
        # Paginate until the requested number of unique eligible files is
        # reached or Commons has no more results. Each result page is looked
        # up as a batch, so extra non-image/duplicate results do not cause an
        # avoidable shortfall. Search results are metadata only.
        local_ids: set[str] = set()
        search_results = iter(search_category(client, category, max_per_category * 20, page_size=50))
        candidate_target = max_per_category * 4
        candidates: list[dict[str, Any]] = []
        local_sha1: set[str] = set()
        order = 0
        while len(candidates) < candidate_target:
            batch = list(islice(search_results, 50))
            if not batch:
                break
            valid_pages: list[dict[str, Any]] = []
            for page in batch:
                pageid_raw = page.get("pageid")
                file_title = str(page.get("title") or "")
                pageid = str(pageid_raw) if isinstance(pageid_raw, int) and pageid_raw > 0 else ""
                if not pageid or not file_title:
                    malformed_page[category] += 1
                    continue
                if pageid in rows_by_pageid:
                    duplicate_page_ids[category] += 1
                    existing = rows_by_pageid[pageid]
                    labels = set(filter(None, existing["all_selected_categories"].split(";")))
                    labels.add(category)
                    existing["all_selected_categories"] = ";".join(sorted(labels, key=str.casefold))
                    continue
                if pageid in local_ids:
                    duplicate_page_ids[category] += 1
                    continue
                local_ids.add(pageid)
                valid_pages.append(page)
            infos = fetch_imageinfo(client, [str(page["title"]) for page in valid_pages])
            for page in valid_pages:
                pageid = str(page["pageid"])
                file_title = str(page["title"])
                record = infos.get(file_title.casefold())
                if not record or not record.get("imageinfo"):
                    missing_imageinfo[category] += 1
                    continue
                eligible_counts[category] += 1
                row = _candidate_row(category, page, record["imageinfo"])
                if row["mime_type"] not in EXTENSIONS:
                    unsupported_mime[category] += 1
                    continue
                reasons = _non_photograph_reasons(page, record["imageinfo"])
                if reasons:
                    rejected_non_photograph[category] += 1
                    continue
                sha1 = row["sha1"].casefold()
                if sha1 != "unknown" and (sha1 in seen_sha1 or sha1 in local_sha1):
                    duplicate_sha1[category] += 1
                    continue
                if sha1 != "unknown":
                    local_sha1.add(sha1)
                creator_key = re.sub(r"\s+", " ", row["creator"].casefold()).strip()
                candidate = {
                    "row": row,
                    "creator_key": "" if row["creator"] == "unknown" else creator_key,
                    "subject_tokens": _subject_tokens(row, category),
                    "photo_score": _photo_signal_score(page, record["imageinfo"]),
                    "order": order,
                }
                order += 1
                if _near_duplicate(candidate, candidates + previously_selected, category):
                    near_duplicate_subjects[category] += 1
                    continue
                candidates.append(candidate)

        selected, fallbacks = _select_creator_diverse(candidates, max_per_category, max_per_creator)
        creator_fallbacks[category] = fallbacks
        creator_diversity[category] = len({item["creator_key"] for item in selected if item["creator_key"]})
        for candidate in selected:
            row = candidate["row"]
            pageid = row["commons_page_id"]
            sha1 = row["sha1"].casefold()
            if sha1 != "unknown":
                seen_sha1.add(sha1)
            rows.append(row)
            rows_by_pageid[pageid] = row
            previously_selected.append(candidate)
            counts[category] += 1

    missing_metadata = Counter()
    license_distribution = Counter()
    dimensions_and_sizes: list[dict[str, str]] = []
    for row in rows:
        status = row["metadata_status"]
        if status != "metadata_present_review_required":
            missing_metadata.update(item for item in status.split(";") if item)
        license_distribution[row["license_name"]] += 1
        dimensions_and_sizes.append({
            "source_identifier": row["source_identifier"],
            "selected_category": row["selected_category"],
            "width": row["original_width"],
            "height": row["original_height"],
            "file_size_bytes": row["original_size_bytes"],
        })
    summary = {
        "requested_per_category": max_per_category,
        "requested_categories": categories,
        "max_per_creator_per_category": max_per_creator,
        "imageinfo_present_per_category": eligible_counts,
        "selected_per_category": counts,
        "shortfall_per_category": {category: max_per_category - counts[category] for category in categories},
        "missing_imageinfo_per_category": dict(missing_imageinfo),
        "duplicate_sha1_skipped_per_category": dict(duplicate_sha1),
        "duplicate_page_id_hits_per_category": dict(duplicate_page_ids),
        "unsupported_mime_skipped_per_category": dict(unsupported_mime),
        "malformed_search_result_per_category": dict(malformed_page),
        "explicit_non_photograph_skipped_per_category": dict(rejected_non_photograph),
        "near_duplicate_subject_skipped_per_category": dict(near_duplicate_subjects),
        "creator_diversity_count_per_category": creator_diversity,
        "creator_cap_fallbacks_per_category": dict(creator_fallbacks),
        "unique_records": len(rows),
        "license_distribution": dict(sorted(license_distribution.items())),
        "missing_metadata_counts": dict(sorted(missing_metadata.items())),
        "dimensions_and_file_sizes": dimensions_and_sizes,
        "license_review_status": "pending_manual_review",
        "image_bytes_downloaded": 0,
    }
    return rows, summary


def write_manifest(path: Path, rows: list[dict[str, str]], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, Any], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Summary output already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Wikimedia Commons metadata-only candidate manifest; never downloads image files."
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="candidate manifest destination")
    parser.add_argument("--summary-json", type=Path,
                        help="summary destination (default: <output-stem>.summary.json)")
    parser.add_argument("--category", action="append", choices=CATEGORIES,
                        help="category to search; repeat to select several (default: all ten)")
    parser.add_argument("--max-candidates-per-category", type=int, default=30,
                        help="maximum unique JPEG/PNG candidates for each category (default: 30)")
    parser.add_argument("--max-per-creator-per-category", type=int, default=3,
                        help="soft creator cap; repeats are deferred, then used to fill shortfalls (0 disables)")
    parser.add_argument("--timeout", type=float, default=20.0, help="API request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="bounded retries for HTTP 429/503")
    parser.add_argument("--user-agent",
                        help="full identifying User-Agent; if omitted, composed with SKETCH2VISION_CONTACT")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_candidates_per_category < 1:
        parser.error("--max-candidates-per-category must be positive")
    if args.max_per_creator_per_category < 0:
        parser.error("--max-per-creator-per-category cannot be negative")
    if args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be positive and --retries cannot be negative")
    categories = args.category or list(CATEGORIES)
    if len(set(categories)) != len(categories):
        parser.error("each --category may be specified only once")
    summary_path = args.summary_json or args.output_csv.with_name(args.output_csv.stem + ".summary.json")
    if not args.overwrite and (args.output_csv.exists() or summary_path.exists()):
        parser.error("manifest or summary output already exists; choose another path or pass --overwrite")
    user_agent = args.user_agent
    if user_agent is None:
        contact = os.environ.get("SKETCH2VISION_CONTACT", "").strip()
        if not contact:
            parser.error("set SKETCH2VISION_CONTACT to a real project contact or pass --user-agent with contact information")
        user_agent = f"{DEFAULT_USER_AGENT} (contact: {contact})"
    try:
        client = CommonsAPIClient(user_agent, args.timeout, args.retries)
        rows, summary = build_manifest(client, categories, args.max_candidates_per_category,
                                       args.max_per_creator_per_category)
        write_manifest(args.output_csv, rows, args.overwrite)
        summary["api_requests_including_retries"] = client.request_count
        write_summary(summary_path, summary, args.overwrite)
    except (OSError, RuntimeError, ValueError, csv.Error) as exc:
        print(f"Commons manifest generation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} unique Commons metadata rows to {args.output_csv}")
    print(f"Summary: {summary_path}")
    for category in categories:
        print(f"  {category}: selected={summary['selected_per_category'][category]}, "
              f"shortfall={summary['shortfall_per_category'][category]}")
    print("License status: pending_manual_review for every record; no rights approval implied.")
    print("Image bytes downloaded: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
