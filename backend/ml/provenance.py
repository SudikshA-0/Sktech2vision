"""Source-to-pair provenance schema and conservative license-review gates."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath
from typing import Any


PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_FIELDS = (
    "source_id", "source_identifier", "source_path", "source_url", "source_page_url",
    "license_name", "license_identifier", "attribution_required", "attribution_text",
    "creator", "license_review_status", "reviewer_notes", "source_metadata_url",
    "selected_category", "source_metadata_raw", "record_type", "provenance_schema_version", "provenance_issues",
)
_APPROVED = "approved"
_PENDING = "pending_manual_review"


def _first(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _relative_key(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe or empty source_path in provenance CSV: {value!r}")
    return path.as_posix().casefold()


def _attribution_requirement(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"true", "yes", "1", "required"}:
        return "true"
    if normalized in {"false", "no", "0", "not_required"}:
        return "false"
    return "unknown"


def read_provenance_csv(path: Path) -> list[dict[str, str]]:
    """Read a reviewed/candidate CSV without inventing absent provenance values."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = set(reader.fieldnames or [])
        mapping_fields = {"source_path", "source_identifier", "image_id", "source_id"}
        if not fields.intersection(mapping_fields):
            raise ValueError(
                f"{path}: provenance CSV needs one mapping column: "
                "source_path, source_identifier, image_id, or source_id"
            )
        records: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}: row {line_number} has more values than the header")
            record = {key: (value or "").strip() for key, value in row.items() if key}
            if not (_first(record, ("source_path",)) or _first(record, ("source_identifier", "image_id", "source_id"))):
                raise ValueError(f"{path}: row {line_number} has no source_path or source identifier")
            records.append(record)
    return records


def _record_for_source(source_id: str, relative_path: str, raw: dict[str, str] | None,
                       dataset_kind: str) -> dict[str, Any]:
    raw = raw or {}
    issues: list[str] = []
    source_identifier = _first(raw, ("source_identifier", "image_id", "source_id")) or Path(relative_path).stem
    if dataset_kind == "synthetic":
        return {
            "source_id": source_id,
            "source_identifier": source_identifier,
            "source_path": relative_path,
            "source_url": "",
            "source_page_url": "",
            "license_name": "",
            "license_identifier": "",
            "attribution_required": "unknown",
            "attribution_text": "",
            "creator": "",
            "license_review_status": "not_applicable_synthetic",
            "reviewer_notes": "",
            "source_metadata_url": "",
            "selected_category": "",
            "source_metadata_raw": {},
            "record_type": "synthetic",
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "provenance_issues": [],
        }

    status_raw = _first(raw, ("license_review_status", "license_status", "review_status"))
    status = status_raw.casefold() if status_raw else _PENDING
    if not raw:
        issues.append("source_metadata_not_provided")
    if status not in {_APPROVED, _PENDING, "rejected"}:
        issues.append("unrecognized_license_review_status")
        status = _PENDING
    record = {
        "source_id": source_id,
        "source_identifier": source_identifier,
        "source_path": relative_path,
        "source_url": _first(raw, ("source_url", "image_url")),
        "source_page_url": _first(raw, ("source_page_url", "landing_page_url", "original_landing_url")),
        "license_name": _first(raw, ("license_name",)),
        "license_identifier": _first(raw, ("license_identifier", "license_url", "license_id")),
        "attribution_required": _attribution_requirement(_first(raw, ("attribution_required",))),
        "attribution_text": _first(raw, ("attribution_text", "attribution")),
        "creator": _first(raw, ("creator", "author")),
        "license_review_status": status,
        "reviewer_notes": _first(raw, ("reviewer_notes",)),
        "source_metadata_url": _first(raw, ("source_metadata_url", "metadata_source_url")),
        "selected_category": _first(raw, ("selected_category", "category")),
        "source_metadata_raw": dict(raw),
        "record_type": "real",
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "provenance_issues": issues,
    }
    for field, value in (
        ("source_url", record["source_url"] or record["source_page_url"]),
        ("license_name_or_identifier", record["license_name"] or record["license_identifier"]),
        ("attribution_requirement", record["attribution_required"] if record["attribution_required"] != "unknown" else ""),
        ("reviewer_notes", record["reviewer_notes"]),
    ):
        if not value:
            issues.append(f"missing_{field}")
    if record["attribution_required"] == "true" and not (record["attribution_text"] or record["creator"]):
        issues.append("missing_attribution")
    return record


def build_source_records(input_dir: Path, source_map: dict[str, Path], dataset_kind: str,
                         metadata_csv: Path | None = None) -> dict[str, dict[str, Any]]:
    """Map source metadata to generated pair IDs by safe relative path or image ID/stem."""
    if dataset_kind not in {"real", "synthetic"}:
        raise ValueError("dataset_kind must be 'real' or 'synthetic'")
    if dataset_kind == "synthetic" and metadata_csv is not None:
        raise ValueError("synthetic datasets cannot attach real-source provenance CSV records")
    raw_records = read_provenance_csv(metadata_csv) if metadata_csv is not None else []
    normalized_paths = [
        _relative_key(row["source_path"]) if row.get("source_path", "").strip() else None
        for row in raw_records
    ]
    image_keys = [
        _first(row, ("source_identifier", "image_id", "source_id")).casefold()
        for row in raw_records
    ]
    consumed_rows: set[int] = set()
    result: dict[str, dict[str, Any]] = {}
    for pair_id, source_path in source_map.items():
        relative = source_path.relative_to(input_dir).as_posix()
        relative_key = _relative_key(relative)
        stem = source_path.stem.casefold()
        matches = [index for index, row in enumerate(raw_records)
                   if normalized_paths[index] == relative_key or (image_keys[index] and image_keys[index] == stem)]
        if len(matches) > 1:
            raise ValueError(f"multiple provenance CSV rows map to source file {relative!r}")
        raw: dict[str, str] | None = None
        if matches:
            index = matches[0]
            if index in consumed_rows:
                raise ValueError(f"one provenance CSV row maps to multiple source files: {relative!r}")
            consumed_rows.add(index)
            raw = raw_records[index]
        result[pair_id] = _record_for_source(pair_id, relative, raw, dataset_kind)
    return result


def provenance_missing_fields(record: dict[str, Any]) -> list[str]:
    """Return review evidence gaps; an approval label alone is never sufficient."""
    missing: list[str] = []
    if not str(record.get("source_id", "")).strip():
        missing.append("source_id")
    if not str(record.get("source_identifier", "")).strip():
        missing.append("source_identifier")
    if not str(record.get("source_path", "")).strip():
        missing.append("source_path")
    if record.get("provenance_schema_version") != PROVENANCE_SCHEMA_VERSION:
        missing.append("provenance_schema_version")
    if not (str(record.get("source_url", "")).strip() or str(record.get("source_page_url", "")).strip()):
        missing.append("source_url")
    if not (str(record.get("license_name", "")).strip() or str(record.get("license_identifier", "")).strip()):
        missing.append("license_name_or_identifier")
    requirement = str(record.get("attribution_required", "unknown")).strip().casefold()
    if requirement not in {"true", "false"}:
        missing.append("attribution_required")
    if requirement == "true" and not (
        str(record.get("attribution_text", "")).strip() or str(record.get("creator", "")).strip()
    ):
        missing.append("attribution")
    if not str(record.get("reviewer_notes", "")).strip():
        missing.append("reviewer_notes")
    return missing


def training_provenance_findings(manifest: dict[str, Any], *, allow_synthetic_data: bool = False) -> list[str]:
    """Validate all split records; the synthetic opt-in can never pass real data."""
    dataset_kind = str(manifest.get("dataset_kind", "real")).strip().casefold()
    source_ids = manifest.get("source_ids")
    if dataset_kind not in {"real", "synthetic"}:
        return [f"unsupported dataset_kind {dataset_kind!r}; expected 'real' or 'synthetic'"]
    if not isinstance(source_ids, dict):
        return ["source_ids is missing; provenance cannot be matched to prepared pairs"]
    all_ids = [source_id for split in ("train", "validation", "test")
               for source_id in source_ids.get(split, []) if isinstance(source_id, str)]
    records = manifest.get("source_metadata")
    if not isinstance(records, dict):
        records = {}

    if dataset_kind == "synthetic":
        if not allow_synthetic_data:
            return ["dataset is explicitly marked synthetic; pass --allow-synthetic-data for synthetic fixtures only"]
        findings = []
        for source_id in all_ids:
            record = records.get(source_id)
            if (not isinstance(record, dict) or record.get("record_type") != "synthetic"
                    or record.get("source_id") != source_id or not str(record.get("source_path", "")).strip()
                    or not str(record.get("source_identifier", "")).strip()
                    or record.get("provenance_schema_version") != PROVENANCE_SCHEMA_VERSION):
                findings.append(f"{source_id}: synthetic override rejected because provenance record is not synthetic")
            elif record.get("license_review_status") == _APPROVED:
                findings.append(f"{source_id}: synthetic fixture must not be labeled license-approved")
            elif any(str(record.get(field, "")).strip() for field in
                     ("source_url", "source_page_url", "license_name", "license_identifier", "creator")):
                findings.append(f"{source_id}: synthetic override rejected because real-source provenance is present")
        return findings

    if allow_synthetic_data:
        return ["--allow-synthetic-data cannot override provenance for a real or unclassified dataset"]
    findings = []
    for source_id in all_ids:
        record = records.get(source_id)
        if not isinstance(record, dict):
            findings.append(f"{source_id}: provenance record missing")
            continue
        missing = provenance_missing_fields(record)
        status = str(record.get("license_review_status", "missing")).strip().casefold()
        mapping_missing = []
        if record.get("source_id") != source_id:
            mapping_missing.append("source_id_mapping")
        if not str(record.get("source_path", "")).strip():
            mapping_missing.append("source_path")
        if not str(record.get("source_identifier", "")).strip():
            mapping_missing.append("source_identifier")
        if record.get("provenance_schema_version") != PROVENANCE_SCHEMA_VERSION:
            mapping_missing.append("provenance_schema_version")
        if status != _APPROVED or missing or mapping_missing or record.get("record_type") != "real":
            reasons = [f"status={status}"]
            if missing:
                reasons.append("missing=" + ",".join(missing))
            if mapping_missing:
                reasons.append("invalid_mapping=" + ",".join(mapping_missing))
            if record.get("record_type") != "real":
                reasons.append("record_type_not_real")
            findings.append(f"{source_id}: " + "; ".join(reasons))
    return findings

