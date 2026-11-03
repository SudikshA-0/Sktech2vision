"""Prepare and validate aligned edge-map/original-image dataset pairs.

No images are downloaded. Install the optional dependency with:
    python -m pip install -r backend/ml/requirements-optional.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.provenance import PROVENANCE_SCHEMA_VERSION, build_source_records


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "validation", "test")


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to prepare images. Install it with "
            "`python -m pip install -r backend/ml/requirements-optional.txt`."
        ) from exc
    return cv2


def _source_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "image"
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"{stem}-{digest}"


def _split_sources(
    source_ids: list[str], seed: int, ratios: tuple[float, float, float]
) -> dict[str, list[str]]:
    shuffled = sorted(source_ids)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if not count:
        return {name: [] for name in SPLITS}

    exact = [count * ratio for ratio in ratios]
    sizes = [int(value) for value in exact]
    for index in sorted(range(3), key=lambda i: exact[i] - sizes[i], reverse=True)[
        : count - sum(sizes)
    ]:
        sizes[index] += 1

    # With at least three inputs, keep all three requested splits represented.
    if count >= 3:
        for empty_index in (1, 2):
            if sizes[empty_index] == 0:
                donor = max(range(3), key=lambda i: sizes[i])
                sizes[donor] -= 1
                sizes[empty_index] += 1

    result: dict[str, list[str]] = {}
    offset = 0
    for name, size in zip(SPLITS, sizes):
        result[name] = shuffled[offset : offset + size]
        offset += size
    return result


def _output_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _to_gray(cv2: Any, image: Any) -> Any:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported decoded image shape: {image.shape}")


def _prepare_pair(
    cv2: Any,
    source_path: Path,
    width: int,
    height: int,
    resize_mode: str,
    canny_low: int,
    canny_high: int,
) -> tuple[Any, Any, dict[str, Any]]:
    if resize_mode == "stretch":
        # Preserve the original decoding, resize, color and edge behavior.
        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError("OpenCV could not decode the image")
        source_height, source_width = image.shape[:2]
        if (source_width, source_height) != (width, height):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        gray = _to_gray(cv2, image)
        edges = cv2.Canny(gray, canny_low, canny_high)
        transform = {
            "source_dimensions_after_decode": {"width": source_width, "height": source_height},
            "resized_dimensions": {"width": width, "height": height},
            "padding": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "exif_orientation_applied": False,
        }
        return edges, image, transform

    if resize_mode != "pad":
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    # IMREAD_COLOR applies supported EXIF orientation before returning pixels.
    # IMREAD_UNCHANGED is intentionally not used here because it suppresses it.
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("OpenCV could not decode the image")
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, int(round(source_width * scale))))
    resized_height = max(1, min(height, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    horizontal_padding = width - resized_width
    vertical_padding = height - resized_height
    left = horizontal_padding // 2
    right = horizontal_padding - left
    top = vertical_padding // 2
    bottom = vertical_padding - top

    # OpenCV stores color arrays in BGR order; all-white is identical in RGB/BGR.
    target = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    gray = _to_gray(cv2, resized)
    content_edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.copyMakeBorder(
        content_edges, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    transform = {
        "source_dimensions_after_exif_orientation": {"width": source_width, "height": source_height},
        "resized_dimensions": {"width": resized_width, "height": resized_height},
        "padding": {"left": left, "top": top, "right": right, "bottom": bottom},
        "exif_orientation_applied": True,
    }
    return edges, target, transform


def prepare_dataset(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if (
        output_dir == input_dir
        or input_dir in output_dir.parents
        or output_dir in input_dir.parents
    ):
        raise ValueError(
            "Output directory must be separate from the input directory and must not contain it."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    replace_output = _output_has_files(output_dir)
    if replace_output and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
        )

    candidates = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if args.max_images is not None:
        candidates = candidates[: args.max_images]
    if not candidates:
        raise ValueError(f"No JPEG or PNG images found in {input_dir}")

    source_map: dict[str, Path] = {_source_id(path, input_dir): path for path in candidates}
    dataset_kind = getattr(args, "dataset_kind", "real")
    metadata_csv = getattr(args, "source_metadata_csv", None)
    # Load and validate provenance before an explicitly requested overwrite can
    # remove any previous prepared output.
    source_metadata = build_source_records(input_dir, source_map, dataset_kind, metadata_csv)

    cv2 = _load_cv2()
    if replace_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = _split_sources(
        list(source_map), args.seed, (args.train_ratio, args.validation_ratio, args.test_ratio)
    )
    skipped: list[dict[str, str]] = []
    transforms: dict[str, dict[str, Any]] = {}
    valid = 0
    print(f"Found {len(candidates)} candidate image(s). Preparing aligned pairs...")

    for split_name, source_ids in splits.items():
        edge_dir = output_dir / split_name / "edges"
        target_dir = output_dir / split_name / "targets"
        edge_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_id in source_ids:
            source_path = source_map[source_id]
            try:
                width, height = args.image_size
                resize_mode = getattr(args, "resize_mode", "stretch")
                edges, image, transform = _prepare_pair(
                    cv2, source_path, width, height, resize_mode,
                    args.canny_low, args.canny_high,
                )
                edge_path = edge_dir / f"{source_id}.png"
                target_path = target_dir / f"{source_id}.png"
                if not cv2.imwrite(str(edge_path), edges) or not cv2.imwrite(
                    str(target_path), image
                ):
                    raise OSError("OpenCV could not write the pair")
                transforms[source_id] = transform
                valid += 1
                print(f"[{split_name}] {source_path.relative_to(input_dir)} -> {source_id}.png")
            except Exception as exc:  # Report a bad source and continue with the rest.
                skipped.append({"source_id": source_id, "reason": str(exc)})
                for partial in (
                    output_dir / split_name / "edges" / f"{source_id}.png",
                    output_dir / split_name / "targets" / f"{source_id}.png",
                ):
                    partial.unlink(missing_ok=True)
                print(f"[skipped] {source_path}: {exc}", file=sys.stderr)

    split_ids = {
        split_name: [source_id for source_id in source_ids if source_id not in {
            item["source_id"] for item in skipped
        }]
        for split_name, source_ids in splits.items()
    }
    manifest = {
        "dataset_kind": dataset_kind,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "seed": args.seed,
        "image_size": {"width": args.image_size[0], "height": args.image_size[1]},
        "preprocessing": {
            "resize_mode": getattr(args, "resize_mode", "stretch"),
            "target_padding_rgb": [255, 255, 255] if getattr(args, "resize_mode", "stretch") == "pad" else None,
            "edge_padding_grayscale": 0 if getattr(args, "resize_mode", "stretch") == "pad" else None,
            "padding_alignment": "center" if getattr(args, "resize_mode", "stretch") == "pad" else None,
            "exif_orientation": "applied by OpenCV for pad mode; legacy stretch mode is unchanged",
        },
        "transforms": transforms,
        "ratios": {name: ratio for name, ratio in zip(SPLITS, (args.train_ratio, args.validation_ratio, args.test_ratio))},
        "canny_thresholds": {"low": args.canny_low, "high": args.canny_high},
        "source_ids": split_ids,
        "source_metadata": {
            source_id: source_metadata[source_id]
            for source_ids in split_ids.values()
            for source_id in source_ids
        },
        "skipped": skipped,
    }
    (output_dir / "splits.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Preparation complete: "
        + ", ".join(f"{name}={len(split_ids[name])}" for name in SPLITS)
        + f"; valid={valid}, skipped={len(skipped)}, invalid=0, output={output_dir}"
    )
    return 0


def validate_dataset(output_dir: Path) -> int:
    root = output_dir.expanduser().resolve()
    manifest_path = root / "splits.json"
    invalid_items: set[str] = set()
    valid_pairs_by_split: dict[str, set[str]] = {name: set() for name in SPLITS}
    skipped_count = 0
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_ids = manifest["source_ids"]
        skipped_count = len(manifest.get("skipped", []))
        if any(name not in source_ids or not isinstance(source_ids[name], list) for name in SPLITS):
            raise ValueError("manifest must contain train, validation, and test source_ids lists")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Invalid split manifest: {exc}", file=sys.stderr)
        print(f"Validation counts: valid=0, skipped={skipped_count}, invalid=1")
        return 1

    cv2 = _load_cv2()
    seen: dict[str, str] = {}
    for split_name in SPLITS:
        for source_id in source_ids[split_name]:
            previous = seen.get(source_id)
            if previous is not None:
                invalid_items.add(source_id)
                print(f"[invalid] source {source_id} occurs in {previous} and {split_name}")
            else:
                seen[source_id] = split_name

        edge_dir = root / split_name / "edges"
        target_dir = root / split_name / "targets"
        edge_files = {path.name: path for path in edge_dir.glob("*.png")} if edge_dir.is_dir() else {}
        target_files = {path.name: path for path in target_dir.glob("*.png")} if target_dir.is_dir() else {}
        for name in sorted(edge_files.keys() | target_files.keys()):
            edge_path = edge_files.get(name)
            target_path = target_files.get(name)
            source_id = name[:-4]
            if edge_path is None or target_path is None:
                invalid_items.add(source_id)
                print(f"[invalid] missing pair member for {split_name}/{name}")
                continue
            if source_id not in source_ids[split_name]:
                invalid_items.add(source_id)
                print(f"[invalid] pair {split_name}/{name} is not listed in splits.json")
                continue
            edge = cv2.imread(str(edge_path), cv2.IMREAD_UNCHANGED)
            target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
            if edge is None or target is None:
                invalid_items.add(source_id)
                print(f"[invalid] unreadable pair {split_name}/{name}")
            elif edge.shape[:2] != target.shape[:2]:
                invalid_items.add(source_id)
                print(f"[invalid] dimensions differ for {split_name}/{name}")
            else:
                valid_pairs_by_split[split_name].add(source_id)

        expected = {f"{source_id}.png" for source_id in source_ids[split_name]}
        if edge_files.keys() != expected or target_files.keys() != expected:
            missing = expected - (edge_files.keys() & target_files.keys())
            extra = (edge_files.keys() | target_files.keys()) - expected
            if missing or extra:
                invalid_items.update(name[:-4] for name in missing)
                invalid_items.update(name[:-4] for name in extra)
                print(f"[invalid] {split_name}: missing={len(missing)}, extra={len(extra)}")

    invalid = len(invalid_items)
    split_counts = {
        name: len(pair_ids - invalid_items)
        for name, pair_ids in valid_pairs_by_split.items()
    }
    valid = sum(split_counts.values())
    print(
        "Validation counts: "
        + ", ".join(f"{name}={split_counts[name]}" for name in SPLITS)
        + f"; valid={valid}, skipped={skipped_count}, invalid={invalid}"
    )
    return 1 if invalid else 0


def _positive_ratio(value: str) -> float:
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("ratio must be greater than 0 and less than 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and validate aligned Canny-edge/original-image dataset pairs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare pairs from local source images")
    prepare.add_argument("--input-dir", type=Path, required=True, help="local source image directory")
    prepare.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    prepare.add_argument(
        "--dataset-kind", choices=("real", "synthetic"), default="real",
        help="real is the safe default; mark synthetic only for generated test fixtures",
    )
    prepare.add_argument(
        "--source-metadata-csv", type=Path,
        help="optional reviewed/candidate provenance CSV keyed by source_path or image_id/source_identifier",
    )
    prepare.add_argument("--seed", type=int, default=42, help="deterministic split seed (default: 42)")
    prepare.add_argument("--max-images", type=int, help="limit candidates for a small experiment")
    prepare.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(256, 256),
        help="resize every target and edge pair to these dimensions (default: 256 256)",
    )
    prepare.add_argument(
        "--resize-mode",
        choices=("stretch", "pad"),
        default="stretch",
        help="stretch to fill (legacy default) or preserve aspect ratio with centered padding",
    )
    prepare.add_argument("--train-ratio", type=_positive_ratio, default=0.8)
    prepare.add_argument("--validation-ratio", type=_positive_ratio, default=0.1)
    prepare.add_argument("--test-ratio", type=_positive_ratio, default=0.1)
    prepare.add_argument("--canny-low", type=int, default=100)
    prepare.add_argument("--canny-high", type=int, default=200)
    prepare.add_argument("--overwrite", action="store_true", help="replace a non-empty output directory")
    validate = subparsers.add_parser("validate", help="check pairs, dimensions, and split leakage")
    validate.add_argument("--output-dir", type=Path, required=True, help="prepared dataset directory")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        if args.max_images is not None and args.max_images < 1:
            parser.error("--max-images must be at least 1")
        if any(dimension < 1 for dimension in args.image_size):
            parser.error("--image-size dimensions must be at least 1")
        if args.canny_low < 0 or args.canny_high <= args.canny_low:
            parser.error("Canny thresholds must satisfy 0 <= --canny-low < --canny-high")
        ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
        if abs(sum(ratios) - 1.0) > 1e-6:
            parser.error("train, validation, and test ratios must sum to 1")
        try:
            return prepare_dataset(args)
        except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    try:
        return validate_dataset(args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
