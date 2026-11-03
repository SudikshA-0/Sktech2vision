"""Command-line Pix2Pix training entry point, usable locally or in Colab."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # allow CLI help to work before optional ML dependencies exist
    np = None  # type: ignore[assignment]
try:
    import torch
except ImportError:  # the parser/help path remains usable without PyTorch installed
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.config import (
    PIX2PIX_ADAM_BETAS,
    PIX2PIX_BASE_CHANNELS,
    PIX2PIX_BATCH_SIZE,
    PIX2PIX_CHECKPOINT_INTERVAL,
    PIX2PIX_EPOCHS,
    PIX2PIX_IMAGE_SIZE,
    PIX2PIX_L1_WEIGHT,
    PIX2PIX_LEARNING_RATE,
    PIX2PIX_WORKERS,
    RANDOM_SEED,
)
from backend.ml.provenance import training_provenance_findings
from backend.ml.scripts.validate_training_dataset import validate_dataset

if torch is not None:
    from torch.utils.data import DataLoader
    from backend.ml.artifacts import append_jsonl, save_validation_grid
    from backend.ml.dataset import PairedEdgeImageDataset
    from backend.ml.evaluation import (
        capture_rng_state,
        dataset_fingerprint,
        evaluate_generator,
        load_checkpoint,
        restore_rng_state,
        save_checkpoint,
        write_run_metadata,
    )
    from backend.ml.models import build_pix2pix
    from backend.ml.training import train_one_epoch


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = PIX2PIX_IMAGE_SIZE[0]
    batch_size: int = PIX2PIX_BATCH_SIZE
    epochs: int = PIX2PIX_EPOCHS
    base_channels: int = PIX2PIX_BASE_CHANNELS
    learning_rate: float = PIX2PIX_LEARNING_RATE
    l1_weight: float = PIX2PIX_L1_WEIGHT
    workers: int = PIX2PIX_WORKERS
    seed: int = RANDOM_SEED
    checkpoint_interval: int = PIX2PIX_CHECKPOINT_INTERVAL
    output_dir: str = "backend/ml/outputs/pix2pix-run"
    device: str = "auto"

    def validate(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for training; install the project-approved PyTorch build in the runtime")
        if self.image_size < 128 or self.image_size & (self.image_size - 1):
            raise ValueError("image_size must be a power of two and at least 128")
        if self.image_size < 70 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("image size, batch size, and epochs must be positive and supported")
        if self.base_channels < 1 or self.workers < 0 or self.checkpoint_interval < 1:
            raise ValueError("base_channels/checkpoint_interval must be positive and workers non-negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.l1_weight) or self.l1_weight < 0:
            raise ValueError("l1_weight must be finite and non-negative")

    @property
    def num_downs(self) -> int:
        return int(math.log2(self.image_size))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_train_validation_loaders(dataset_dir: Path, *, image_size: int, batch_size: int,
                                   workers: int, seed: int, use_cuda: bool) -> tuple[DataLoader, DataLoader, torch.Generator]:
    """Construct only train and validation loaders; the test directory is never loaded here."""
    train_data = PairedEdgeImageDataset(dataset_dir, "train", (image_size, image_size))
    validation_data = PairedEdgeImageDataset(dataset_dir, "validation", (image_size, image_size))
    if not len(train_data) or not len(validation_data):
        raise ValueError("train and validation splits must both contain at least one pair")
    loader_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=use_cuda, worker_init_fn=seed_worker, generator=loader_generator,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=1, shuffle=False, num_workers=0, generator=validation_generator,
    )
    return train_loader, validation_loader, loader_generator


def validate_resume_config(saved: dict[str, Any], current: dict[str, Any], completed_epoch: int) -> None:
    """Reject data/optimization drift while permitting a longer epoch ceiling and new output/device."""
    ignored = {"epochs", "output_dir", "device"}
    differences = [key for key in current if key not in ignored and saved.get(key) != current[key]]
    if differences:
        raise ValueError("resume configuration differs for: " + ", ".join(differences))
    if int(current["epochs"]) < int(completed_epoch):
        raise ValueError("requested epochs is less than the checkpoint's completed epoch")


def _configure_runtime(requested: str) -> torch.device:
    if requested == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = requested
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python runtime")
    device = torch.device(selected)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        free, total = torch.cuda.mem_get_info(device)
        print(f"GPU memory: free={free / 1024**3:.2f} GiB / total={total / 1024**3:.2f} GiB")
    return device


def _checkpoint_payload(*, generator: torch.nn.Module, discriminator: torch.nn.Module,
                        optimizer_g: torch.optim.Optimizer, optimizer_d: torch.optim.Optimizer,
                        epoch: int, global_step: int, config: TrainConfig, fingerprint: str,
                        best_validation_l1: float, history: list[dict[str, Any]],
                        loader_generator: torch.Generator, runtime: dict[str, Any],
                        run_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "completed_epoch": epoch,
        "global_step": global_step,
        "config": asdict(config),
        "architecture": {"base_channels": config.base_channels, "num_downs": config.num_downs},
        "dataset_fingerprint": fingerprint,
        "best_validation_l1": best_validation_l1,
        "history": history,
        "rng_state": capture_rng_state(loader_generator),
        "runtime": runtime,
        "run_metadata": run_metadata,
    }


def run_training(dataset_dir: Path, config: TrainConfig, *, resume: Path | None = None,
                 allow_synthetic_data: bool = False) -> int:
    config.validate()
    seed_everything(config.seed)
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    # Structural validation is mandatory, read-only, and includes the held-out
    # test files only for integrity checking. Test examples are never loaded by
    # the train/validation DataLoaders or used for metrics/checkpoint selection.
    if validate_dataset(dataset_dir, (config.image_size, config.image_size)) != 0:
        raise ValueError("strict dataset structural validation failed; training was not started")

    manifest = json.loads((dataset_dir / "splits.json").read_text(encoding="utf-8"))
    source_ids = manifest["source_ids"]
    if not source_ids["test"]:
        raise ValueError("test split must contain at least one source ID, even though it is never used during training")
    training_ids = list(source_ids["train"]) + list(source_ids["validation"])
    provenance_findings = training_provenance_findings(manifest, allow_synthetic_data=allow_synthetic_data)
    if provenance_findings:
        sample = "; ".join(provenance_findings[:5])
        raise ValueError(
            f"training provenance gate rejected dataset ({len(provenance_findings)} issue(s)): {sample}"
        )
    if manifest.get("dataset_kind", "real") == "synthetic":
        print("[INFO] Explicit synthetic-data opt-in accepted; records remain not_applicable_synthetic, not license-approved.")

    train_loader, validation_loader, loader_generator = create_train_validation_loaders(
        dataset_dir, image_size=config.image_size, batch_size=config.batch_size,
        workers=config.workers, seed=config.seed, use_cuda=torch.cuda.is_available(),
    )
    train_data = train_loader.dataset
    fingerprint = dataset_fingerprint(dataset_dir, training_ids)
    device = _configure_runtime(config.device)
    generator, discriminator = build_pix2pix(base_channels=config.base_channels, num_downs=config.num_downs)
    generator.to(device)
    discriminator.to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=PIX2PIX_ADAM_BETAS)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=PIX2PIX_ADAM_BETAS)

    run_config = asdict(config)
    run_config["dataset_dir"] = str(dataset_dir)
    run_config["resize_mode"] = manifest["preprocessing"]["resize_mode"]
    run_config["preprocessing"] = manifest["preprocessing"]
    run_config["canny_thresholds"] = manifest.get("canny_thresholds")
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": _opencv_version(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": "enabled_warn_only",
    }
    run_metadata = {"config": run_config, "runtime": runtime, "dataset_fingerprint": fingerprint,
                    "dataset_kind": manifest.get("dataset_kind", "real"),
                    "synthetic_data_opt_in": bool(manifest.get("dataset_kind") == "synthetic" and allow_synthetic_data),
                    "fingerprint_scope": "splits.json plus train/validation pair bytes; test image bytes excluded"}
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise FileExistsError(f"output directory is non-empty; choose a new --output-dir or use --resume: {output_dir}")
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location=device)
        if checkpoint.get("dataset_fingerprint") != fingerprint:
            raise ValueError("resume checkpoint dataset fingerprint does not match current splits/train/validation data")
        saved_config = checkpoint.get("config", {})
        current_config = asdict(config)
        # Allow extending the epoch ceiling and moving the output directory/device;
        # data, architecture, optimization, and loader settings must match.
        validate_resume_config(saved_config, current_config, int(checkpoint["completed_epoch"]))
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation_l1 = float(checkpoint["best_validation_l1"])
        history = list(checkpoint.get("history", []))
        restore_rng_state(checkpoint["rng_state"], loader_generator)
        print(f"Resuming at epoch {start_epoch} from {resume}")
    else:
        start_epoch = 1
        global_step = 0
        best_validation_l1 = float("inf")
        history: list[dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, run_metadata)
    for epoch in range(start_epoch, config.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d,
                                  device, config.l1_weight)
        global_step += math.ceil(len(train_data) / config.batch_size)
        validation_l1 = evaluate_generator(generator, validation_loader, device)
        preview_path = output_dir / "samples" / f"validation_epoch_{epoch:04d}.png"
        save_validation_grid(generator, validation_loader, preview_path, device)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "generator_total": metrics.generator_total,
            "generator_adversarial": metrics.generator_adversarial,
            "generator_weighted_l1": metrics.generator_l1,
            "discriminator": metrics.discriminator,
            "validation_l1": validation_l1,
            "learning_rate_g": optimizer_g.param_groups[0]["lr"],
            "learning_rate_d": optimizer_d.param_groups[0]["lr"],
            "seconds": metrics.seconds,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "validation_grid": str(preview_path.relative_to(output_dir)),
        }
        history.append(record)
        append_jsonl(output_dir / "metrics.jsonl", record)
        print(f"epoch {epoch}/{config.epochs}: G={metrics.generator_total:.5f} D={metrics.discriminator:.5f} validation_L1={validation_l1:.6f}")
        payload = _checkpoint_payload(
            generator=generator, discriminator=discriminator, optimizer_g=optimizer_g, optimizer_d=optimizer_d,
            epoch=epoch, global_step=global_step, config=config, fingerprint=fingerprint,
            best_validation_l1=best_validation_l1, history=history, loader_generator=loader_generator, runtime=runtime,
            run_metadata=run_metadata,
        )
        if validation_l1 < best_validation_l1:
            best_validation_l1 = validation_l1
            payload["best_validation_l1"] = best_validation_l1
            save_checkpoint(output_dir / "best.pt", payload)
        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(output_dir / "latest.pt", payload)
    if start_epoch <= config.epochs:
        final_payload = _checkpoint_payload(
            generator=generator, discriminator=discriminator, optimizer_g=optimizer_g, optimizer_d=optimizer_d,
            epoch=config.epochs, global_step=global_step, config=config, fingerprint=fingerprint,
            best_validation_l1=best_validation_l1, history=history, loader_generator=loader_generator, runtime=runtime,
            run_metadata=run_metadata,
        )
        save_checkpoint(output_dir / "latest.pt", final_payload)
    print(f"Training complete. Best validation L1={best_validation_l1:.6f}; outputs={output_dir}")
    return 0


def _opencv_version() -> str | None:
    try:
        import cv2
        return cv2.__version__
    except ImportError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the paired Pix2Pix model on an already prepared dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="prepared dataset root containing splits.json")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/ml/outputs/pix2pix-run"))
    parser.add_argument("--image-size", type=int, default=PIX2PIX_IMAGE_SIZE[0], help="square prepared image size (default: 128)")
    parser.add_argument("--batch-size", type=int, default=PIX2PIX_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=PIX2PIX_EPOCHS)
    parser.add_argument("--base-channels", type=int, default=PIX2PIX_BASE_CHANNELS)
    parser.add_argument("--learning-rate", type=float, default=PIX2PIX_LEARNING_RATE)
    parser.add_argument("--l1-weight", type=float, default=PIX2PIX_L1_WEIGHT)
    parser.add_argument("--workers", type=int, default=PIX2PIX_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--checkpoint-interval", type=int, default=PIX2PIX_CHECKPOINT_INTERVAL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, help="resume from a latest checkpoint")
    parser.add_argument(
        "--allow-synthetic-data", action="store_true",
        help="allow only a dataset explicitly marked synthetic whose every source record is record_type=synthetic; never bypasses real-data review",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if torch is None or np is None:
        print("[FATAL] PyTorch and NumPy are required to train. Select/install the approved Colab ML runtime.", file=sys.stderr)
        return 2
    config = TrainConfig(
        image_size=args.image_size, batch_size=args.batch_size, epochs=args.epochs,
        base_channels=args.base_channels, learning_rate=args.learning_rate, l1_weight=args.l1_weight,
        workers=args.workers, seed=args.seed, checkpoint_interval=args.checkpoint_interval,
        output_dir=str(args.output_dir), device=args.device,
    )
    try:
        return run_training(args.dataset_dir, config, resume=args.resume,
                            allow_synthetic_data=args.allow_synthetic_data)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"[FATAL] training preflight/runtime error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

