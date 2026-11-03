"""Safe, isolated inference for the presentation-only CMP Facades checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .config import PIX2PIX_IMAGE_SIZE, PIX2PIX_NUM_DOWNSAMPLES
from .models import UNetGenerator
from .transforms import denormalize_to_uint8


DEMO_DATASET_NAME = "CMP Facades / Pix2Pix Facades demonstration subset"
MAX_SKETCH_BYTES = 10 * 1024 * 1024
MAX_SKETCH_PIXELS = 16_000_000


class InvalidSketchError(ValueError):
    """Raised when request bytes are not a supported, manageable image."""


class DemoCheckpointError(RuntimeError):
    """Raised when the supplied file is not the expected isolated demo checkpoint."""


def preprocess_sketch(image_bytes: bytes, *, image_size: int = PIX2PIX_IMAGE_SIZE[0]) -> torch.Tensor:
    """Convert a canvas PNG/JPEG to [1,1,H,W] Canny edges normalized to [-1,1].

    Geometry is preserved by centering the grayscale drawing on white before
    edge detection. Canny uses the same 50/150 thresholds as the Facades demo.
    The resulting condition has white edges on black, matching model inputs.
    """
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise InvalidSketchError("The sketch upload is empty.")
    if len(image_bytes) > MAX_SKETCH_BYTES:
        raise InvalidSketchError("Sketch image exceeds the 10 MB limit.")
    if image_size < 128 or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two of at least 128")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise InvalidSketchError("Could not decode the sketch. Upload a valid PNG or JPEG.")
    height, width = bgr.shape[:2]
    if width < 1 or height < 1 or width * height > MAX_SKETCH_PIXELS or max(width, height) > 8192:
        raise InvalidSketchError("Sketch image dimensions are too large to process.")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    side = max(width, height)
    square = np.full((side, side), 255, dtype=np.uint8)
    left = (side - width) // 2
    top = (side - height) // 2
    square[top:top + height, left:left + width] = gray
    edges = cv2.Canny(square, 50, 150)
    edges = cv2.resize(edges, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(edges.copy()).unsqueeze(0).unsqueeze(0).float().div_(127.5).sub_(1.0)
    return tensor


def load_demo_generator(checkpoint_path: str | Path, device: str | torch.device = "auto") -> tuple[UNetGenerator, torch.device, dict[str, Any]]:
    """Load only a weights-only checkpoint explicitly marked for the Facades demo."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Demo checkpoint was not found.")
    selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise DemoCheckpointError("CUDA was requested but is unavailable.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DemoCheckpointError("Checkpoint could not be read safely with weights-only loading.") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise DemoCheckpointError("Unsupported checkpoint format.")
    if checkpoint.get("demo_only") is not True or checkpoint.get("dataset") != DEMO_DATASET_NAME:
        raise DemoCheckpointError("Checkpoint is not marked as the isolated CMP Facades demo model.")
    config = checkpoint.get("model_config")
    state = checkpoint.get("generator")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise DemoCheckpointError("Checkpoint is missing generator weights or model configuration.")
    if config.get("image_size") != PIX2PIX_IMAGE_SIZE[0]:
        raise DemoCheckpointError("Checkpoint image size does not match the configured 128×128 model input.")
    base_channels = config.get("base_channels")
    if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 1:
        raise DemoCheckpointError("Checkpoint generator width is invalid.")
    generator = UNetGenerator(base_channels=base_channels, num_downs=PIX2PIX_NUM_DOWNSAMPLES)
    try:
        generator.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise DemoCheckpointError("Checkpoint generator weights do not match the Sketch2Vision U-Net architecture.") from exc
    generator.to(selected_device)
    generator.eval()
    return generator, selected_device, {"dataset": checkpoint["dataset"], "epoch": checkpoint.get("epoch"),
                                       "base_channels": base_channels, "image_size": config["image_size"]}


class FacadesDemoInference:
    """Long-lived loaded generator for FastAPI requests."""

    def __init__(self, generator: UNetGenerator, device: torch.device, metadata: dict[str, Any] | None = None) -> None:
        self.generator = generator.to(device).eval()
        self.device = device
        self.metadata = metadata or {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "auto") -> "FacadesDemoInference":
        generator, selected_device, metadata = load_demo_generator(path, device)
        return cls(generator, selected_device, metadata)

    def generate_png(self, image_bytes: bytes) -> bytes:
        sketch = preprocess_sketch(image_bytes).to(self.device)
        with torch.inference_mode():
            generated = self.generator(sketch)
        if not torch.isfinite(generated).all().item():
            raise RuntimeError("The generator returned non-finite image values.")
        rgb = denormalize_to_uint8(generated[0].cpu()).permute(1, 2, 0).numpy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("Could not encode the generated image as PNG.")
        return encoded.tobytes()

