"""Run a synthetic forward/backward smoke test for the Pix2Pix modules.

This is not a training loop: it performs no optimizer steps, reads no datasets,
and stores only a temporary pair of model state dictionaries when the optional
round-trip check is enabled.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile
from pathlib import Path

# CUDA deterministic GEMM configuration must be set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError as exc:  # No package installation or fallback is attempted.
    raise SystemExit(
        "PyTorch is required for this smoke test. Run it in a Colab runtime "
        "that already has a compatible PyTorch installation."
    ) from exc

from backend.ml.config import PIX2PIX_BASE_CHANNELS, PIX2PIX_IMAGE_SIZE
from backend.ml.losses import discriminator_loss, generator_loss
from backend.ml.models import build_pix2pix
from backend.ml.transforms import normalize_uint8_tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic Pix2Pix forward/backward smoke test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--size", type=int, default=PIX2PIX_IMAGE_SIZE[0], help="square synthetic input size (default: 128)")
    parser.add_argument("--base-channels", type=int, default=PIX2PIX_BASE_CHANNELS, help="model width (default: 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-checkpoint-roundtrip", action="store_true")
    return parser


def _select_device(requested: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested == "cpu" or not cuda_available:
        return torch.device("cpu")
    return torch.device("cuda")


def _check_gradients(module: torch.nn.Module, name: str) -> int:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError(f"{name} received no gradients")
    if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError(f"{name} has non-finite gradients")
    if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
        raise RuntimeError(f"{name} gradients are all zero")
    return len(gradients)


def _expected_patch_size(size: int) -> int:
    value = size
    for _ in range(3):
        value = (value + 2 - 4) // 2 + 1
    for _ in range(2):
        value = value + 2 - 4 + 1
    return value


def _round_trip_state_dicts(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    edges: torch.Tensor,
    device: torch.device,
    base_channels: int,
    num_downs: int,
) -> None:
    generator.eval()
    discriminator.eval()
    with torch.inference_mode():
        expected_image = generator(edges)
        expected_logits = discriminator(edges, expected_image)
    with tempfile.TemporaryDirectory(prefix="sketch2vision-model-smoke-") as temp_dir:
        checkpoint_path = Path(temp_dir) / "model-state-dicts.pt"
        torch.save({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }, checkpoint_path)
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:  # Compatibility with older Colab PyTorch versions.
            state = torch.load(checkpoint_path, map_location=device)
        restored_generator, restored_discriminator = build_pix2pix(
            base_channels=base_channels,
            num_downs=num_downs,
            initialize=False,
        )
        restored_generator.load_state_dict(state["generator"])
        restored_discriminator.load_state_dict(state["discriminator"])
        restored_generator.to(device).eval()
        restored_discriminator.to(device).eval()
        with torch.inference_mode():
            actual_image = restored_generator(edges)
            actual_logits = restored_discriminator(edges, actual_image)
        torch.testing.assert_close(actual_image, expected_image)
        torch.testing.assert_close(actual_logits, expected_logits)


def run_smoke_test(args: argparse.Namespace) -> int:
    if args.size < 128 or args.size & (args.size - 1):
        raise ValueError("--size must be a power of two of at least 128")
    if args.base_channels < 1:
        raise ValueError("--base-channels must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    device = _select_device(args.device)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU memory: free={free_bytes / 1024**3:.2f} GiB / total={total_bytes / 1024**3:.2f} GiB")
        torch.cuda.reset_peak_memory_stats(device)

    num_downs = int(math.log2(args.size))
    generator, discriminator = build_pix2pix(
        base_channels=args.base_channels,
        num_downs=num_downs,
    )
    generator.to(device).train()
    discriminator.to(device).train()

    cpu_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    edge_bytes = torch.randint(0, 2, (1, 1, args.size, args.size), generator=cpu_generator, dtype=torch.uint8) * 255
    target_bytes = torch.randint(0, 256, (1, 3, args.size, args.size), generator=cpu_generator, dtype=torch.uint8)
    edges = normalize_uint8_tensor(edge_bytes).to(device)
    targets = normalize_uint8_tensor(target_bytes).to(device)

    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)
    generated = generator(edges)
    expected_image_shape = (1, 3, args.size, args.size)
    if tuple(generated.shape) != expected_image_shape:
        raise RuntimeError(f"generator shape mismatch: {tuple(generated.shape)} != {expected_image_shape}")
    if not torch.isfinite(generated).all().item() or generated.min().item() < -1.0 or generated.max().item() > 1.0:
        raise RuntimeError("generator output is non-finite or outside [-1, 1]")

    real_logits = discriminator(edges, targets)
    detached_fake_logits = discriminator(edges, generated.detach())
    expected_patch_shape = (1, 1, _expected_patch_size(args.size), _expected_patch_size(args.size))
    if tuple(real_logits.shape) != expected_patch_shape:
        raise RuntimeError(f"discriminator shape mismatch: {tuple(real_logits.shape)} != {expected_patch_shape}")
    d_loss = discriminator_loss(real_logits, detached_fake_logits)
    if not torch.isfinite(d_loss).item():
        raise RuntimeError("discriminator loss is non-finite")
    d_loss.backward()
    d_gradient_count = _check_gradients(discriminator, "discriminator")
    if any(parameter.grad is not None for parameter in generator.parameters()):
        raise RuntimeError("detached fake discriminator pass unexpectedly produced generator gradients")

    discriminator.zero_grad(set_to_none=True)
    generator.zero_grad(set_to_none=True)
    generator_logits = discriminator(edges, generated)
    g_loss, g_adversarial, g_weighted_l1 = generator_loss(generator_logits, generated, targets)
    if not all(torch.isfinite(value).item() for value in (g_loss, g_adversarial, g_weighted_l1)):
        raise RuntimeError("generator loss contains a non-finite value")
    g_loss.backward()
    g_gradient_count = _check_gradients(generator, "generator")
    if not any(parameter.grad is not None for parameter in discriminator.parameters()):
        raise RuntimeError("discriminator did not receive gradients during the generator adversarial pass")

    print(f"Generator output: {tuple(generated.shape)}, range=[{generated.min().item():.4f}, {generated.max().item():.4f}]")
    print(f"Discriminator logits: {tuple(real_logits.shape)}")
    print(f"Losses: D={d_loss.item():.6f}, G={g_loss.item():.6f}, G_adv={g_adversarial.item():.6f}, weighted_L1={g_weighted_l1.item():.6f}")
    print(f"Gradient tensors: generator={g_gradient_count}, discriminator={d_gradient_count}")

    if not args.skip_checkpoint_roundtrip:
        _round_trip_state_dicts(
            generator, discriminator, edges, device, args.base_channels, num_downs
        )
        print("Model state-dict round trip: passed (temporary file cleaned up)")
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        print(f"Peak allocated GPU memory: {peak_bytes / 1024**3:.2f} GiB")
    print("Pix2Pix synthetic smoke test passed; no optimizer step or model training was performed.")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_smoke_test(args)
    except (RuntimeError, ValueError) as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
