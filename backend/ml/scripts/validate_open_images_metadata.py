"""Validate local Open Images metadata CSVs without network access.

The validator reads files only. It uses a temporary SQLite index under the
metadata directory for bounded-memory ID comparison and removes it on exit.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_METADATA_DIR = Path(r"C:\data\open-images-metadata")
FILES = {
    "labels": "oidv7-train-annotations-human-imagelabels.csv",
    "classes": "oidv7-class-descriptions.csv",
    "image_info": "oidv6-train-images-with-labels-with-rotation.csv",
}
OFFICIAL_URLS = {
    "labels": "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv",
    "classes": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv",
    "image_info": "https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv",
}
LABEL_COLUMNS = {"ImageID", "Source", "LabelName", "Confidence"}
IMAGE_INFO_COLUMNS = {
    "ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
    "AuthorProfileURL", "Author", "Title", "OriginalSize", "OriginalMD5",
    "Thumbnail300KURL", "Rotation",
}
ATTRIBUTION_COMPONENTS = (
    "Author", "AuthorProfileURL", "Title", "OriginalLandingURL", "License",
)
IMAGE_ID_RE = re.compile(r"^[0-9a-fA-F]{16}$")
UPSERT_ID = "INSERT INTO {table}(image_id, occurrences) VALUES (?, 1) ON CONFLICT(image_id) DO UPDATE SET occurrences=occurrences+1"


class FatalSchemaError(Exception):
    """An input cannot be interpreted safely under the documented schema."""


def missing_value(value: Any) -> bool:
    return value is None or not str(value).strip() or str(value).strip().casefold() in {"nan", "none"}


def read_dict_csv(path: Path, required: set[str]) -> tuple[csv.DictReader, Any]:
    handle = path.open("r", newline="", encoding="utf-8-sig")
    try:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
            handle.close()
            raise FatalSchemaError("duplicate column names in CSV header")
        fields = set(reader.fieldnames or [])
        absent = required - fields
        if absent:
            handle.close()
            raise FatalSchemaError(
                f"missing required column(s): {', '.join(sorted(absent))}; "
                f"found: {', '.join(reader.fieldnames or []) or '(no header)'}"
            )
        return reader, handle
    except csv.Error:
        handle.close()
        raise


def record_problem(stats: dict[str, Any], line: int, reason: str) -> None:
    stats["malformed_rows"] += 1
    if len(stats["malformed_examples"]) < 5:
        stats["malformed_examples"].append({"line": line, "reason": reason})


def flush_ids(connection: sqlite3.Connection, table: str, pending: list[tuple[str]]) -> None:
    if pending:
        connection.executemany(UPSERT_ID.format(table=table), pending)
        pending.clear()


def validate_labels(path: Path, connection: sqlite3.Connection) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": 0, "blank_image_id_rows": 0, "invalid_image_id_rows": 0,
        "missing_source_rows": 0, "missing_label_rows": 0,
        "missing_confidence_rows": 0, "malformed_rows": 0, "malformed_examples": [],
    }
    reader, handle = read_dict_csv(path, LABEL_COLUMNS)
    pending: list[tuple[str]] = []
    try:
        for row in reader:
            stats["rows"] += 1
            line = reader.line_num
            if None in row:
                record_problem(stats, line, "extra field(s) beyond the header")
            if any(row.get(name) is None for name in LABEL_COLUMNS):
                record_problem(stats, line, "row has fewer fields than required columns")
            image_id = (row.get("ImageID") or "").strip().lower()
            if not image_id:
                stats["blank_image_id_rows"] += 1
            else:
                if not IMAGE_ID_RE.fullmatch(image_id):
                    stats["invalid_image_id_rows"] += 1
                else:
                    pending.append((image_id,))
            if missing_value(row.get("Source")):
                stats["missing_source_rows"] += 1
            if missing_value(row.get("LabelName")):
                stats["missing_label_rows"] += 1
            if missing_value(row.get("Confidence")):
                stats["missing_confidence_rows"] += 1
            if stats["rows"] % 10_000 == 0:
                flush_ids(connection, "label_ids", pending)
                connection.commit()
        flush_ids(connection, "label_ids", pending)
        connection.commit()
    except csv.Error as exc:
        raise FatalSchemaError(f"malformed CSV syntax near line {reader.line_num}: {exc}") from exc
    finally:
        handle.close()
    return stats


def validate_image_info(path: Path, connection: sqlite3.Connection) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": 0, "blank_image_id_rows": 0, "invalid_image_id_rows": 0,
        "missing_field_values": {name: 0 for name in IMAGE_INFO_COLUMNS if name != "ImageID"},
        "invalid_rotation_rows": 0, "invalid_size_rows": 0,
        "incomplete_attribution_component_rows": 0,
        "original_size_raw_samples": [],
        "malformed_rows": 0, "malformed_examples": [],
    }
    reader, handle = read_dict_csv(path, {"ImageID"})
    fields = set(reader.fieldnames or [])
    stats["missing_columns"] = sorted(IMAGE_INFO_COLUMNS - fields)
    pending: list[tuple[str]] = []
    try:
        for row in reader:
            stats["rows"] += 1
            line = reader.line_num
            if None in row:
                record_problem(stats, line, "extra field(s) beyond the header")
            image_id = (row.get("ImageID") or "").strip().lower()
            if not image_id:
                stats["blank_image_id_rows"] += 1
            else:
                if not IMAGE_ID_RE.fullmatch(image_id):
                    stats["invalid_image_id_rows"] += 1
                else:
                    pending.append((image_id,))

            missing_fields = []
            for name in stats["missing_field_values"]:
                if missing_value(row.get(name)):
                    stats["missing_field_values"][name] += 1
                    missing_fields.append(name)
            if any(name in missing_fields for name in ATTRIBUTION_COMPONENTS):
                stats["incomplete_attribution_component_rows"] += 1

            rotation = (row.get("Rotation") or "").strip()
            if not missing_value(rotation) and rotation not in {"0", "90", "180", "270"}:
                stats["invalid_rotation_rows"] += 1
            size = (row.get("OriginalSize") or "").strip()
            if not missing_value(size):
                if len(stats["original_size_raw_samples"]) < 10:
                    stats["original_size_raw_samples"].append(size)
                try:
                    if int(size) <= 0:
                        stats["invalid_size_rows"] += 1
                except ValueError:
                    stats["invalid_size_rows"] += 1
            if stats["rows"] % 10_000 == 0:
                flush_ids(connection, "info_ids", pending)
                connection.commit()
        flush_ids(connection, "info_ids", pending)
        connection.commit()
    except csv.Error as exc:
        raise FatalSchemaError(f"malformed CSV syntax near line {reader.line_num}: {exc}") from exc
    finally:
        handle.close()
    return stats


def validate_classes(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": 0, "malformed_rows": 0, "malformed_examples": [],
        "blank_mid_rows": 0, "blank_display_name_rows": 0,
        "duplicate_mid_rows": 0,
    }
    seen_mids: set[str] = set()
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle, strict=True)
            first = True
            for row in reader:
                if first:
                    first = False
                    if len(row) >= 2 and row[0].strip().casefold() in {"labelname", "mid"} and row[1].strip().casefold() in {"displayname", "classname"}:
                        continue
                stats["rows"] += 1
                line = reader.line_num
                if len(row) < 2:
                    record_problem(stats, line, "expected at least MID and display-name columns")
                    continue
                mid, display = row[0].strip(), row[1].strip()
                if not mid:
                    stats["blank_mid_rows"] += 1
                elif mid in seen_mids:
                    stats["duplicate_mid_rows"] += 1
                else:
                    seen_mids.add(mid)
                if not display:
                    stats["blank_display_name_rows"] += 1
    except csv.Error as exc:
        raise FatalSchemaError(f"malformed CSV syntax near line {getattr(reader, 'line_num', '?')}: {exc}") from exc
    return stats


def duplicate_summary(connection: sqlite3.Connection, table: str) -> dict[str, int]:
    row = connection.execute(
        f"SELECT COUNT(*), COALESCE(SUM(occurrences-1), 0) FROM {table} WHERE occurrences > 1"
    ).fetchone()
    return {"duplicate_id_values": int(row[0]), "duplicate_id_rows": int(row[1])}


def comparison_summary(connection: sqlite3.Connection) -> dict[str, int]:
    labels_unique = int(connection.execute("SELECT COUNT(*) FROM label_ids").fetchone()[0])
    info_unique = int(connection.execute("SELECT COUNT(*) FROM info_ids").fetchone()[0])
    intersection = int(connection.execute(
        "SELECT COUNT(*) FROM label_ids l INNER JOIN info_ids i USING(image_id)"
    ).fetchone()[0])
    return {
        "label_unique_image_ids": labels_unique,
        "image_info_unique_image_ids": info_unique,
        "ids_in_both": intersection,
        "ids_only_in_labels": labels_unique - intersection,
        "ids_only_in_image_info": info_unique - intersection,
    }


def validate_selected_ids(path: Path, connection: sqlite3.Connection) -> dict[str, Any]:
    """Validate a candidate manifest's IDs and require a one-to-one info join."""
    stats = {"rows": 0, "blank_ids": 0, "malformed_ids": 0, "duplicate_ids": 0,
             "missing_image_info_ids": 0, "missing_examples": []}
    seen: set[str] = set()
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = set(reader.fieldnames or [])
            id_field = "image_id" if "image_id" in fields else "ImageID" if "ImageID" in fields else None
            if id_field is None:
                raise FatalSchemaError("selected manifest needs an image_id or ImageID column")
            for row in reader:
                stats["rows"] += 1
                value = (row.get(id_field) or "").strip().lower()
                if not value:
                    stats["blank_ids"] += 1
                    continue
                if not IMAGE_ID_RE.fullmatch(value):
                    stats["malformed_ids"] += 1
                    continue
                if value in seen:
                    stats["duplicate_ids"] += 1
                else:
                    seen.add(value)
                exists = connection.execute("SELECT 1 FROM info_ids WHERE image_id=?", (value,)).fetchone()
                if not exists:
                    stats["missing_image_info_ids"] += 1
                    if len(stats["missing_examples"]) < 10:
                        stats["missing_examples"].append(value)
    except csv.Error as exc:
        raise FatalSchemaError(f"malformed selected manifest CSV: {exc}") from exc
    return stats


def validate_directory(metadata_dir: Path, selected_ids_csv: Path | None = None) -> tuple[int, list[str]]:
    missing = [key for key, filename in FILES.items() if not (metadata_dir / filename).is_file()]
    print(f"Metadata directory: {metadata_dir}")
    if missing:
        print("MISSING FILES:")
        for key in missing:
            print(f"  {FILES[key]}\n    Official URL: {OFFICIAL_URLS[key]}")
        print("No network access attempted. Place the official CSVs here and rerun validation.")
    if not metadata_dir.is_dir():
        print("FATAL: metadata directory does not exist.")
        return 2, missing

    database_dir: tempfile.TemporaryDirectory[str] | None = None
    connection: sqlite3.Connection | None = None
    fatal_count = 0
    try:
        database_dir = tempfile.TemporaryDirectory(prefix=".s2v-metadata-validation-", dir=metadata_dir)
        connection = sqlite3.connect(Path(database_dir.name) / "ids.sqlite3")
        connection.execute("CREATE TABLE label_ids(image_id TEXT PRIMARY KEY, occurrences INTEGER NOT NULL)")
        connection.execute("CREATE TABLE info_ids(image_id TEXT PRIMARY KEY, occurrences INTEGER NOT NULL)")
        stats: dict[str, dict[str, Any]] = {}

        for key in ("labels", "classes", "image_info"):
            path = metadata_dir / FILES[key]
            if not path.is_file():
                continue
            print(f"\n{key.upper()}: {path.name}")
            try:
                if key == "labels":
                    stats[key] = validate_labels(path, connection)
                elif key == "image_info":
                    stats[key] = validate_image_info(path, connection)
                else:
                    stats[key] = validate_classes(path)
                print(f"  Rows: {stats[key]['rows']}")
                for field in ("blank_image_id_rows", "invalid_image_id_rows", "missing_source_rows", "missing_label_rows", "missing_confidence_rows", "blank_mid_rows", "blank_display_name_rows", "duplicate_mid_rows", "invalid_rotation_rows", "invalid_size_rows", "incomplete_attribution_component_rows"):
                    if field in stats[key] and stats[key][field]:
                        print(f"  WARNING {field}: {stats[key][field]}")
                if stats[key].get("missing_columns"):
                    print("  WARNING missing optional metadata columns: " + ", ".join(stats[key]["missing_columns"]))
                if key == "image_info":
                    print("  Missing/blank values by field:")
                    for field, count in stats[key]["missing_field_values"].items():
                        print(f"    {field}: {count}")
                    print("  OriginalSize raw-value samples (units not inferred): " + repr(stats[key]["original_size_raw_samples"]))
                    duplicates = duplicate_summary(connection, "info_ids")
                    if duplicates["duplicate_id_rows"]:
                        fatal_count += 1
                        print("  FATAL duplicate image-info ImageID records: " + str(duplicates))
                if key in {"labels", "image_info"}:
                    invalid = stats[key].get("invalid_image_id_rows", 0) + stats[key].get("blank_image_id_rows", 0)
                    if invalid:
                        fatal_count += 1
                        print(f"  FATAL invalid/blank ImageID rows: {invalid}; malformed IDs are not accepted")
                if stats[key]["malformed_rows"]:
                    print(f"  WARNING malformed rows (not silently discarded): {stats[key]['malformed_rows']}")
                    for sample in stats[key]["malformed_examples"]:
                        print(f"    line {sample['line']}: {sample['reason']}")
            except (OSError, UnicodeError, csv.Error, FatalSchemaError) as exc:
                fatal_count += 1
                print(f"  FATAL schema/read error: {exc}")

        if "labels" in stats and "image_info" in stats:
            print("\nIMAGE-ID COMPARISON:")
            print("  Labels duplicate IDs (expected for multi-label data): " + str(duplicate_summary(connection, "label_ids")))
            print("  Image-info duplicate IDs (review): " + str(duplicate_summary(connection, "info_ids")))
            comparison = comparison_summary(connection)
            print("  V7 labels vs linked image-info valid IDs: " + str(comparison))
            if comparison["ids_only_in_labels"]:
                fatal_count += 1
                print(f"  FATAL missing image-info joins for {comparison['ids_only_in_labels']} label image IDs")
            else:
                print("  Missing label-to-image-info joins: 0")
        if selected_ids_csv is not None and connection is not None and (metadata_dir / FILES["image_info"]).is_file():
            try:
                selected = validate_selected_ids(selected_ids_csv, connection)
                print(f"\nSELECTED MANIFEST JOIN CHECK: {selected_ids_csv}")
                print("  Counts: " + str({key: value for key, value in selected.items() if key != "missing_examples"}))
                if selected["missing_examples"]:
                    print("  Missing image-info ID examples: " + ", ".join(selected["missing_examples"]))
                if any(selected[key] for key in ("blank_ids", "malformed_ids", "duplicate_ids", "missing_image_info_ids")):
                    fatal_count += 1
                    print("  FATAL selected manifest must contain unique well-formed IDs with image-info joins")
            except (OSError, UnicodeError, csv.Error, FatalSchemaError) as exc:
                fatal_count += 1
                print(f"  FATAL selected-manifest validation error: {exc}")
        print("\nStorage estimate: UNRESOLVED. OriginalSize values are reported as source strings; this validator does not infer units or sum them as bytes.")
        print("License status: pending_manual_review. No attribution text is generated.")
        print(f"Result: {'FATAL ERRORS' if fatal_count else ('MISSING INPUTS' if missing else 'VALIDATION COMPLETE WITH WARNINGS AS LISTED')}")
        return (2 if fatal_count else (1 if missing else 0)), missing
    except OSError as exc:
        print(f"FATAL: could not create/read temporary ID index: {exc}")
        return 2, missing
    finally:
        if connection is not None:
            connection.close()
        if database_dir is not None:
            database_dir.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate local Open Images CSV metadata only; no network access or image reads."
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR,
        help=r"directory containing the three official metadata CSVs (default: C:\data\open-images-metadata)",
    )
    parser.add_argument("--selected-ids-csv", type=Path,
                        help="optional candidate manifest CSV; validates unique IDs and image-info joins")
    args = parser.parse_args()
    status, _ = validate_directory(args.metadata_dir.expanduser().resolve(),
                                   args.selected_ids_csv.expanduser().resolve() if args.selected_ids_csv else None)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
