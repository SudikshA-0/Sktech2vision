"""Read-only structural validation for prepared Pix2Pix training datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.ml.provenance import provenance_missing_fields


SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LICENSE_FIELDS = {
    "source_url": ("source_url", "source_page_url"),
    "license_name_or_identifier": ("license_name", "license_identifier", "license_url", "license"),
    "attribution_required": ("attribution_required",),
    "attribution": ("attribution_text", "attribution", "creator", "author"),
    "license_review_status": ("license_review_status", "license_status", "review_status"),
    "reviewer_notes": ("reviewer_notes",),
}
APPROVED_STATUSES = {"approved"}


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; use the existing backend environment with the ML optional requirements.") from exc
    return cv2


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _metadata_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize optional per-source metadata without inventing records."""
    records: dict[str, dict[str, Any]] = {}

    def add(source_id: Any, value: Any) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            return
        if isinstance(value, dict):
            records.setdefault(source_id, {}).update(value)
        elif isinstance(value, str):
            records.setdefault(source_id, {})["category"] = value
        elif isinstance(value, list):
            records.setdefault(source_id, {})["categories"] = value

    for key in ("source_metadata", "license_metadata", "licenses", "category_by_source_id", "categories"):
        value = manifest.get(key)
        if isinstance(value, dict):
            for source_id, record in value.items():
                add(source_id, record)
        elif isinstance(value, list):
            for record in value:
                if isinstance(record, dict):
                    add(record.get("source_id", record.get("image_id", record.get("id"))), record)
    return records


def _first_present(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def validate_dataset(root: Path, expected_size: tuple[int, int] | None = None) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    root = root.expanduser().resolve()
    manifest_path = root / "splits.json"

    def error(message: str) -> None:
        errors.append(message)
        print(f"[FATAL] {message}")

    def warning(message: str) -> None:
        warnings.append(message)
        print(f"[WARNING] {message}")

    if not root.is_dir():
        error(f"dataset directory does not exist: {root}")
        return _summary(errors, warnings, Counter(), 0)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        error(f"cannot read valid splits.json: {exc}")
        return _summary(errors, warnings, Counter(), 0)
    if not isinstance(manifest, dict):
        error("splits.json root must be a JSON object")
        return _summary(errors, warnings, Counter(), 0)
    dataset_kind = manifest.get("dataset_kind", "real")
    if dataset_kind not in {"real", "synthetic"}:
        error("splits.json dataset_kind must be 'real' or 'synthetic' when present")
        dataset_kind = "real"

    size = manifest.get("image_size")
    if not isinstance(size, dict) or not _is_positive_int(size.get("width")) or not _is_positive_int(size.get("height")):
        error("splits.json must declare positive integer image_size.width and image_size.height")
        declared_size: tuple[int, int] | None = None
    else:
        declared_size = (size["width"], size["height"])
    if expected_size is not None and declared_size is not None and declared_size != expected_size:
        error(f"manifest dimensions {declared_size[0]}x{declared_size[1]} do not match requested {expected_size[0]}x{expected_size[1]}")
    dimensions = expected_size or declared_size

    preprocessing = manifest.get("preprocessing")
    mode: str | None = None
    if not isinstance(preprocessing, dict):
        error("splits.json must include a preprocessing object with resize_mode")
    else:
        mode = preprocessing.get("resize_mode")
        if mode not in {"stretch", "pad"}:
            error("preprocessing.resize_mode must be explicitly 'stretch' or 'pad'")
        elif mode == "pad":
            required_pad = {
                "target_padding_rgb": [255, 255, 255],
                "edge_padding_grayscale": 0,
                "padding_alignment": "center",
            }
            for key, expected in required_pad.items():
                if preprocessing.get(key) != expected:
                    error(f"pad preprocessing requires {key}={expected!r}; found {preprocessing.get(key)!r}")

    source_ids = manifest.get("source_ids")
    valid_manifest_ids: dict[str, list[str]] = {split: [] for split in SPLITS}
    if not isinstance(source_ids, dict):
        error("splits.json must include source_ids as an object of split lists")
    else:
        for split in SPLITS:
            values = source_ids.get(split)
            if not isinstance(values, list):
                error(f"source_ids.{split} must be a list")
                continue
            seen_within: set[str] = set()
            for source_id in values:
                if not isinstance(source_id, str) or not source_id.strip():
                    error(f"source_ids.{split} contains an empty or non-string ID")
                    continue
                if source_id in seen_within:
                    error(f"duplicate source ID {source_id!r} in {split} manifest")
                else:
                    seen_within.add(source_id)
                    valid_manifest_ids[split].append(source_id)
        owner: dict[str, str] = {}
        for split in SPLITS:
            for source_id in valid_manifest_ids[split]:
                if source_id in owner:
                    error(f"source ID {source_id!r} leaks across {owner[source_id]} and {split}")
                else:
                    owner[source_id] = split

    try:
        cv2 = _load_cv2()
    except RuntimeError as exc:
        error(str(exc))
        return _summary(errors, warnings, Counter(), 0)

    actual_ids: dict[str, set[str]] = {split: set() for split in SPLITS}
    valid_images = 0
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            error(f"missing split directory: {split}/")
        split_pair_ids: dict[str, set[str]] = {"edges": set(), "targets": set()}
        for kind in ("edges", "targets"):
            folder = split_root / kind
            if not folder.is_dir():
                error(f"missing directory: {split}/{kind}/")
                continue
            for path in sorted(folder.iterdir()):
                if path.name == ".gitkeep":
                    continue
                if not path.is_file():
                    error(f"unexpected non-file entry: {split}/{kind}/{path.name}")
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    error(f"unsupported image/file extension: {split}/{kind}/{path.name}")
                    continue
                source_id = path.stem
                if source_id in split_pair_ids[kind]:
                    error(f"duplicate {kind} source ID {source_id!r} in {split}")
                split_pair_ids[kind].add(source_id)
                actual_ids[split].add(source_id)
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.size == 0:
                    error(f"unreadable image: {split}/{kind}/{path.name}")
                    continue
                if dimensions is None:
                    error(f"cannot validate dimensions for {split}/{kind}/{path.name}: expected dimensions are missing")
                elif (image.shape[1], image.shape[0]) != dimensions:
                    error(f"wrong dimensions for {split}/{kind}/{path.name}: got {image.shape[1]}x{image.shape[0]}, expected {dimensions[0]}x{dimensions[1]}")
                if kind == "edges":
                    channels_ok = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)
                    if not channels_ok:
                        error(f"edge map must be one-channel: {split}/{kind}/{path.name} has shape {image.shape}")
                else:
                    channels_ok = image.ndim == 3 and image.shape[2] == 3
                    if not channels_ok:
                        error(f"target must be three-channel color (RGB content): {split}/{kind}/{path.name} has shape {image.shape}")
                if channels_ok and (dimensions is None or (image.shape[1], image.shape[0]) == dimensions):
                    valid_images += 1

        edges = split_pair_ids["edges"]
        targets = split_pair_ids["targets"]
        for source_id in sorted(edges - targets):
            error(f"edge image has no matching target: {split}/{source_id}")
        for source_id in sorted(targets - edges):
            error(f"target has no matching edge image: {split}/{source_id}")
        declared = set(valid_manifest_ids[split])
        for source_id in sorted(actual_ids[split] - declared):
            error(f"unlisted source ID on disk: {split}/{source_id}")
        for source_id in sorted(declared - actual_ids[split]):
            error(f"manifest source ID has no files: {split}/{source_id}")

    # Also detect file-level leakage even if the manifest omitted an ID.
    file_owner: dict[str, str] = {}
    for split in SPLITS:
        for source_id in actual_ids[split]:
            if source_id in file_owner:
                error(f"file source ID {source_id!r} occurs in both {file_owner[source_id]} and {split}")
            else:
                file_owner[source_id] = split

    category_counts: Counter[str] = Counter()
    records = _metadata_records(manifest)
    category_records = 0
    for source_id in owner if isinstance(source_ids, dict) else ():
        record = records.get(source_id, {})
        category = record.get("selected_category", record.get("category"))
        categories = record.get("categories", record.get("all_selected_categories"))
        if category is not None and str(category).strip():
            category_counts[str(category).strip()] += 1
            category_records += 1
        elif isinstance(categories, list):
            for value in categories:
                if isinstance(value, str) and value.strip():
                    category_counts[value.strip()] += 1
                    category_records += 1
        elif isinstance(categories, str) and categories.strip():
            for value in categories.split(";"):
                if value.strip():
                    category_counts[value.strip()] += 1
                    category_records += 1
    if category_records:
        print("Category counts: " + ", ".join(f"{name}={category_counts[name]}" for name in sorted(category_counts, key=str.casefold)))

    if dataset_kind == "synthetic":
        non_synthetic = [source_id for source_id in owner if records.get(source_id, {}).get("record_type") != "synthetic"]
        if non_synthetic:
            error(f"synthetic dataset contains non-synthetic or unclassified provenance records: {len(non_synthetic)}")
        warning("synthetic data is explicitly classified and is not license-approved real data")
    else:
        license_counts: Counter[str] = Counter()
        for source_id in owner if isinstance(source_ids, dict) else ():
            record = records.get(source_id, {})
            missing = provenance_missing_fields(record)
            status_value = _first_present(record, LICENSE_FIELDS["license_review_status"])
            status = str(status_value).strip().casefold() if status_value is not None else "missing"
            if missing:
                warning(f"license metadata for {source_id!r} is missing: {', '.join(missing)}")
            if status in APPROVED_STATUSES:
                license_counts["explicitly_approved_status"] += 1
            elif status == "pending_manual_review":
                license_counts["pending_manual_review"] += 1
            else:
                license_counts["unresolved_or_missing_status"] += 1
        print("License review status: " + ", ".join(f"{name}={license_counts[name]}" for name in sorted(license_counts)))
        warning("structural validation is not license approval; missing and pending_manual_review provenance remains unapproved")

    skipped = manifest.get("skipped", [])
    if not isinstance(skipped, list):
        error("splits.json skipped field must be a list when present")
    elif skipped:
        warning(f"preparation manifest reports {len(skipped)} skipped source(s)")
    if not category_records:
        warning("no category metadata was available; category counts were not computed")

    split_counts = {split: len(actual_ids[split]) for split in SPLITS}
    return _summary(errors, warnings, Counter(split_counts), valid_images)


def _summary(errors: list[str], warnings: list[str], split_counts: Counter[str], valid_images: int) -> int:
    print("Counts: " + ", ".join(f"{split}={split_counts.get(split, 0)}" for split in SPLITS)
          + f"; readable_channel_and_dimension_valid_images={valid_images}")
    print(f"Summary: fatal_errors={len(errors)}, warnings={len(warnings)}")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation of prepared Pix2Pix edge/target datasets."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="prepared dataset root containing splits.json")
    parser.add_argument("--expected-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="optional dimensions to require in addition to splits.json")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    expected = tuple(args.expected_size) if args.expected_size is not None else None
    if expected is not None and any(value < 1 for value in expected):
        parser.error("--expected-size dimensions must be positive")
    try:
        return validate_dataset(args.dataset_dir, expected)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FATAL] validator error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
