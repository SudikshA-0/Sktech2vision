"""Build a metadata-only Open Images candidate manifest.

This script reads local CSV metadata supplied by the operator. It never makes
network requests and never downloads image bytes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Iterable


LABEL_FIELDS = {"ImageID", "Source", "LabelName", "Confidence"}
MANIFEST_FIELDS = [
    "image_id", "open_images_subset", "selected_category", "all_selected_categories",
    "label_source", "label_confidence", "image_url", "landing_page_url",
    "original_size_reported", "original_width", "original_height", "rotation_degrees",
    "author", "author_profile_url", "title", "license_url", "license_name",
    "license_version", "attribution_text", "source_metadata_url",
    "license_review_status", "metadata_status", "source_identifier", "source_url",
    "source_page_url", "license_identifier", "attribution_required", "creator",
    "reviewer_notes", "provenance_schema_version", "record_type",
]
LICENSE_RE = re.compile(r"creativecommons\.org/licenses/([^/]+)/([0-9.]+)/?")
IMAGE_ID_RE = re.compile(r"^[0-9a-fA-F]{16}$")
LICENSE_NAMES = {
    "by": "CC BY",
    "by-sa": "CC BY-SA",
    "by-nd": "CC BY-ND",
    "by-nc": "CC BY-NC",
    "by-nc-sa": "CC BY-NC-SA",
    "by-nc-nd": "CC BY-NC-ND",
    "zero": "CC0",
}


def read_class_names(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                result[row[1].strip().casefold()] = row[0].strip()
    return result


def read_verified_candidates(path: Path, mids: set[str]) -> dict[str, dict[str, tuple[str, str]]]:
    grouped: dict[str, dict[str, tuple[str, str]]] = {mid: {} for mid in mids}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_fields(reader.fieldnames, LABEL_FIELDS, path)
        for row in reader:
            source = (row.get("Source") or "").strip()
            label = (row.get("LabelName") or "").strip()
            if source != "verification" or label not in mids:
                continue
            confidence_raw = (row.get("Confidence") or "").strip()
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError, OverflowError):
                # Malformed confidence values are not eligible positive labels.
                continue
            if not math.isfinite(confidence) or confidence != 1.0:
                continue

            image_id = (row.get("ImageID") or "").strip().lower()
            if not IMAGE_ID_RE.fullmatch(image_id):
                raise ValueError(f"{path}: malformed ImageID on line {reader.line_num}: {image_id!r}")
            if image_id in grouped[label]:
                raise ValueError(
                    f"{path}: duplicate positive verified label for ImageID {image_id} "
                    f"and category MID {label} (line {reader.line_num})"
                )
            grouped[label][image_id] = (source, confidence_raw)
    return grouped


def require_fields(actual: Iterable[str] | None, required: set[str], path: Path) -> None:
    missing = required - set(actual or [])
    if missing:
        raise ValueError(f"{path}: missing required CSV column(s): {', '.join(sorted(missing))}")


def license_parts(url: str) -> tuple[str, str]:
    match = LICENSE_RE.search(url)
    if not match or match.group(1).lower() not in LICENSE_NAMES:
        return "", ""
    return LICENSE_NAMES[match.group(1).lower()], match.group(2)


def load_image_info(path: Path, wanted_ids: set[str]) -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        # ImageID is the only join key required to read the file. Other
        # documented fields are optional in a partial/stale metadata export;
        # missing values are retained as blanks and flagged per manifest row.
        require_fields(reader.fieldnames, {"ImageID"}, path)
        for row in reader:
            image_id = (row.get("ImageID") or "").strip().lower()
            if not IMAGE_ID_RE.fullmatch(image_id):
                if image_id in wanted_ids:
                    raise ValueError(f"{path}: malformed ImageID in image-info CSV on line {reader.line_num}: {image_id!r}")
                continue
            if image_id in wanted_ids:
                if image_id in info:
                    raise ValueError(f"{path}: duplicate image-info record for selected ImageID {image_id} (line {reader.line_num})")
                info[image_id] = {key: (value or "").strip() for key, value in row.items() if key}
    return info


def assign_categories(
    candidate_ids: dict[str, list[str]], sources: dict[str, dict[str, tuple[str, str]]], quota: int
) -> dict[str, str]:
    """Maximum-cardinality seeded b-matching: each ID has one primary category."""
    owners: dict[str, str] = {}
    assigned: dict[str, set[str]] = {category: set() for category in candidate_ids}

    def augment(category: str, visited_categories: set[str], visited_ids: set[str]) -> bool:
        if category in visited_categories:
            return False
        visited_categories.add(category)
        for image_id in candidate_ids[category]:
            if image_id in visited_ids:
                continue
            visited_ids.add(image_id)
            previous = owners.get(image_id)
            if previous is None:
                owners[image_id] = category
                assigned[category].add(image_id)
                return True
            if len(assigned[previous]) < quota:
                # Moving an image to an under-quota category frees this ID.
                if augment(previous, visited_categories, visited_ids):
                    assigned[previous].remove(image_id)
                    owners[image_id] = category
                    assigned[category].add(image_id)
                    return True
            elif augment(previous, visited_categories, visited_ids):
                assigned[previous].remove(image_id)
                owners[image_id] = category
                assigned[category].add(image_id)
                return True
        return False

    # Iterate until no augmenting path exists. This computes a maximum
    # assignment rather than letting command-line category order consume IDs.
    progress = True
    while progress:
        progress = False
        for category in sorted(candidate_ids, key=lambda name: (len(candidate_ids[name]), name.casefold())):
            if len(assigned[category]) >= quota:
                continue
            if augment(category, set(), set()):
                progress = True
    return owners


def create_manifest(args: argparse.Namespace) -> dict[str, object]:
    class_names = read_class_names(args.class_descriptions_csv)
    categories: list[tuple[str, str]] = []
    for category in args.category:
        mid = class_names.get(category.casefold())
        if mid is None:
            raise ValueError(f"Category not found in class descriptions: {category!r}")
        categories.append((category, mid))
    if len({mid for _, mid in categories}) != len(categories):
        raise ValueError("Each --category must resolve to a distinct class MID")

    grouped = read_verified_candidates(args.labels_csv, {mid for _, mid in categories})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and (args.output_csv.exists() or args.summary_json.exists()):
        raise FileExistsError("Manifest output already exists; pass --overwrite to replace it.")

    all_candidate_ids = set().union(*(set(grouped[mid]) for _, mid in categories))
    image_info = load_image_info(args.image_info_csv, all_candidate_ids)
    rng = random.Random(args.seed)
    eligible_ids: dict[str, list[str]] = {}
    eligible_counts: dict[str, int] = {}
    missing_join_counts: dict[str, int] = {}
    for category, mid in categories:
        all_ids = sorted(grouped[mid])
        candidates = [image_id for image_id in all_ids if image_id in image_info]
        rng.shuffle(candidates)
        eligible_ids[category] = candidates
        eligible_counts[category] = len(candidates)
        missing_join_counts[category] = len(set(all_ids) - set(image_info))
    owners = assign_categories(eligible_ids, grouped, args.per_category)
    selected: dict[str, tuple[str, str, str]] = {}
    for image_id, category in owners.items():
        _, mid = next(item for item in categories if item[0] == category)
        source, confidence = grouped[mid][image_id]
        selected[image_id] = (category, source, confidence)
    category_counts = {category: sum(owner == category for owner in owners.values())
                       for category, _ in categories}
    all_matched: dict[str, list[str]] = {}
    for category, mid in categories:
        for image_id in grouped[mid]:
            all_matched.setdefault(image_id, []).append(category)

    known_sizes = 0
    missing_sizes = 0
    rows: list[dict[str, str]] = []
    for image_id in sorted(selected):
        category, source, confidence = selected[image_id]
        record = image_info.get(image_id, {})
        size_text = record.get("OriginalSize", "")
        try:
            parsed_size = int(size_text) if size_text else 0
            size = str(parsed_size) if parsed_size > 0 else ""
            known_sizes += int(size) if size else 0
        except ValueError:
            size = ""
        if not size:
            missing_sizes += 1
        rotation_raw = record.get("Rotation", "").strip()
        rotation = rotation_raw if rotation_raw in {"0", "90", "180", "270"} else ""
        license_url = record.get("License", "")
        license_name, license_version = license_parts(license_url)
        author = record.get("Author", "")
        title = record.get("Title", "")
        landing = record.get("OriginalLandingURL", "")
        issues = []
        if not record:
            raise ValueError(f"Selected ImageID {image_id} has no image-info record; refusing to write an incomplete manifest")
        if not record.get("OriginalURL"):
            issues.append("missing_image_url")
        if not size:
            issues.append("missing_or_invalid_original_size")
        if not rotation:
            issues.append("rotation_missing_or_invalid")
        if not re.fullmatch(r"[0-9a-f]{16}", image_id):
            issues.append("image_id_not_16_char_hex")
        if not record.get("Subset"):
            issues.append("missing_subset")
        elif record.get("Subset", "").casefold() != "train":
            issues.append("source_subset_not_train")
        issues.append("original_dimensions_not_in_documented_schema")
        if not license_url:
            issues.append("missing_license")
        if not author:
            issues.append("missing_author")
        if not record.get("AuthorProfileURL"):
            issues.append("missing_author_profile_url")
        if not title:
            issues.append("missing_title")
        if not landing:
            issues.append("missing_landing_page")
        if not license_name:
            issues.append("license_url_unrecognized")
        issues.append("attribution_requirement_requires_manual_review")
        # Open Images provides component metadata but no authoritative credit
        # wording. Never synthesize this into an attribution statement.
        rows.append({
            "image_id": image_id,
            "open_images_subset": record.get("Subset", ""),
            "selected_category": category,
            "all_selected_categories": ";".join(sorted(set(all_matched.get(image_id, [])), key=str.casefold)),
            "label_source": source,
            "label_confidence": confidence,
            "image_url": record.get("OriginalURL", ""),
            "landing_page_url": landing,
            "original_size_reported": size,
            "original_width": "",
            "original_height": "",
            "rotation_degrees": rotation,
            "author": author,
            "author_profile_url": record.get("AuthorProfileURL", ""),
            "title": title,
            "license_url": license_url,
            "license_name": license_name,
            "license_version": license_version,
            "attribution_text": "",
            "source_metadata_url": ";".join((
                f"labels={args.labels_source_url}",
                f"class_descriptions={args.class_descriptions_source_url}",
                f"image_info={args.image_info_source_url}",
            )),
            "license_review_status": "pending_manual_review",
            "metadata_status": ";".join(issues) if issues else "metadata_present_review_required",
            # These canonical fields let prepare_dataset.py carry each row into
            # source_metadata keyed by the generated pair ID. Image IDs remain
            # identifiers only; they do not imply permission or attribution text.
            "source_identifier": image_id,
            "source_url": record.get("OriginalURL", ""),
            "source_page_url": landing,
            "license_identifier": license_url,
            "attribution_required": "unknown",
            "creator": author,
            "reviewer_notes": "",
            "provenance_schema_version": "1",
            "record_type": "real",
        })

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    size_complete = missing_sizes == 0 and len(rows) > 0
    summary = {
        "manifest_csv": str(args.output_csv),
        "candidate_rows": len(rows),
        "unique_image_ids": len({row["image_id"] for row in rows}),
        "seed": args.seed,
        "per_category_target": args.per_category,
        "requested_candidate_total": args.per_category * len(categories),
        "eligible_per_category": eligible_counts,
        "selected_per_category": category_counts,
        "shortfall_per_category": {name: args.per_category - count for name, count in category_counts.items()},
        "missing_image_info_join_candidates_per_category": missing_join_counts,
        "missing_image_info_join_candidate_ids": len(all_candidate_ids - set(image_info)),
        "target_achieved": len(rows) == args.per_category * len(categories),
        "known_original_size_values_sum_assuming_bytes": known_sizes,
        "original_size_unit_confirmation": "Open Images docs call this download size but do not state units in the schema text; confirm units before using as a hard byte budget",
        "original_size_sum_complete": size_complete,
        "proposed_source_download_budget_bytes": args.budget_bytes,
        "known_size_sum_within_budget_assuming_bytes": known_sizes <= args.budget_bytes,
        "budget_check_final": "not cleared; confirm units, complete metadata, licensing, current source sizes, and post-review total before download",
        "missing_or_invalid_size_rows": missing_sizes,
        "source_metadata_url": ";".join((
            f"labels={args.labels_source_url}",
            f"class_descriptions={args.class_descriptions_source_url}",
            f"image_info={args.image_info_source_url}",
        )),
        "license_review_status": "not approved; manual per-image review required",
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Select Open Images candidates from local CSV metadata only; makes no network requests.")
    result.add_argument("--labels-csv", type=Path, required=True, help="human-verified image-level labels CSV")
    result.add_argument("--image-info-csv", type=Path, required=True, help="image-information CSV with URLs, licensing, OriginalSize, Rotation")
    result.add_argument("--class-descriptions-csv", type=Path, required=True, help="MID-to-display-name class descriptions CSV")
    result.add_argument("--output-csv", type=Path, required=True, help="candidate manifest destination")
    result.add_argument("--summary-json", type=Path, required=True, help="selection and size summary destination")
    result.add_argument("--labels-source-url", default="https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv")
    result.add_argument("--class-descriptions-source-url", default="https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv")
    result.add_argument("--image-info-source-url", default="https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv", help="source URL for image-information metadata; verify version/provenance")
    result.add_argument("--category", action="append", required=True, help="exact class display name; repeat once per category")
    result.add_argument("--per-category", type=int, default=30)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--budget-bytes", type=int, default=2_000_000_000, help="source-byte review budget (default: 2 GB decimal)")
    result.add_argument("--overwrite", action="store_true", help="replace existing manifest outputs")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.per_category < 1 or args.budget_bytes < 1:
        parser().error("--per-category and --budget-bytes must be positive")
    if len(args.category) != 10:
        parser().error("provide exactly 10 distinct --category values for the pilot")
    try:
        summary = create_manifest(args)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"Manifest generation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {summary['candidate_rows']} unique metadata rows to {args.output_csv}")
    print(f"Requested {summary['requested_candidate_total']}; selected {summary['candidate_rows']}; target achieved={summary['target_achieved']}")
    for category in args.category:
        print(f"  {category}: eligible={summary['eligible_per_category'][category]}, "
              f"selected={summary['selected_per_category'][category]}, "
              f"shortfall={summary['shortfall_per_category'][category]}")
    print(f"Missing image-info joins among category candidates: {summary['missing_image_info_join_candidate_ids']}")
    print(f"OriginalSize sum (known values, assuming bytes): {summary['known_original_size_values_sum_assuming_bytes']}; complete={summary['original_size_sum_complete']}")
    print("License status: pending manual review; no download permission implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
