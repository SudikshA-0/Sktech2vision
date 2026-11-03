"""Offline tests for demo inference preprocessing, loading, and FastAPI responses."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import HTTPException

from backend.app.main import describe_sketch_response, generate_image_response
from backend.app.providers.groq_sketch_description import GroqProviderError
from backend.ml.inference import (
    DEMO_DATASET_NAME,
    FacadesDemoInference,
    InvalidSketchError,
    load_demo_generator,
    preprocess_sketch,
)
from backend.ml.models import UNetGenerator


def png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("fixture PNG encoding failed")
    return encoded.tobytes()


class InferenceTests(unittest.TestCase):
    def test_preprocessing_produces_expected_edges_and_rejects_bad_input(self) -> None:
        sketch = np.full((160, 240, 3), 255, dtype=np.uint8)
        cv2.line(sketch, (30, 130), (120, 20), (0, 0, 0), 8)
        tensor = preprocess_sketch(png_bytes(sketch))
        self.assertEqual(tuple(tensor.shape), (1, 1, 128, 128))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(tensor.min().item(), -1)
        self.assertLessEqual(tensor.max().item(), 1)
        self.assertEqual(tensor.min().item(), -1.0)
        self.assertEqual(tensor.max().item(), 1.0)
        with self.assertRaises(InvalidSketchError):
            preprocess_sketch(b"not an image")
        with self.assertRaises(InvalidSketchError):
            preprocess_sketch(b"")

    def test_demo_checkpoint_loads_weights_only_and_requires_demo_marker(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / "best.pt"
            generator = UNetGenerator(base_channels=1)
            torch.save({
                "format_version": 1, "demo_only": True, "dataset": DEMO_DATASET_NAME,
                "generator": generator.state_dict(),
                "model_config": {"image_size": 128, "base_channels": 1},
            }, path)
            loaded, device, metadata = load_demo_generator(path, device="cpu")
            self.assertFalse(loaded.training)
            self.assertEqual(metadata["base_channels"], 1)
            self.assertEqual(device.type, "cpu")
            bad_path = Path(temp) / "not-demo.pt"
            torch.save({"format_version": 1, "demo_only": False}, bad_path)
            with self.assertRaisesRegex(RuntimeError, "isolated CMP Facades"):
                load_demo_generator(bad_path, device="cpu")

    def test_api_unavailable_invalid_type_and_png_response(self) -> None:
        with self.assertRaises(HTTPException) as unavailable:
            generate_image_response(png_bytes(np.full((128, 128, 3), 255, np.uint8)), "image/png", None)
        self.assertEqual(unavailable.exception.status_code, 503)
        self.assertIn("unavailable", unavailable.exception.detail)

        with self.assertRaises(HTTPException) as wrong_type:
            generate_image_response(b"bad", "text/plain", None)
        self.assertEqual(wrong_type.exception.status_code, 415)

        output_png = png_bytes(np.full((128, 128, 3), (20, 100, 220), np.uint8))

        class FakeService:
            def generate_png(self, body: bytes) -> bytes:
                self.assert_body = body
                return output_png

        service = FakeService()
        source = png_bytes(np.full((64, 96, 3), 255, np.uint8))
        response = generate_image_response(source, "image/png", service)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "image/png")
        self.assertEqual(service.assert_body, source)
        decoded = cv2.imdecode(np.frombuffer(response.body, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (128, 128, 3))

        class InvalidService:
            def generate_png(self, body: bytes) -> bytes:
                raise ValueError("Could not decode the sketch.")

        with self.assertRaises(HTTPException) as invalid_image:
            generate_image_response(b"not-an-image", "image/png", InvalidService())
        self.assertEqual(invalid_image.exception.status_code, 400)

    def test_description_api_passes_exact_sketch_and_returns_text(self) -> None:
        source = png_bytes(np.full((64, 96, 3), 255, np.uint8))

        class FakeProvider:
            def describe(self, image: bytes) -> str:
                self.received = image
                return "A simple house with a triangular roof."

        provider = FakeProvider()
        result = describe_sketch_response(source, "image/png", provider)
        self.assertEqual(result, {"description": "A simple house with a triangular roof."})
        self.assertEqual(provider.received, source)

    def test_description_api_reports_missing_provider_and_invalid_content_type(self) -> None:
        source = png_bytes(np.full((24, 24, 3), 255, np.uint8))
        with self.assertRaises(HTTPException) as unavailable:
            describe_sketch_response(source, "image/png", None)
        self.assertEqual(unavailable.exception.status_code, 503)
        self.assertIn("GROQ_API_KEY", unavailable.exception.detail)
        with self.assertRaises(HTTPException) as wrong_type:
            describe_sketch_response(source, "text/plain", object())
        self.assertEqual(wrong_type.exception.status_code, 415)

    def test_description_api_forwards_sanitized_groq_forbidden_message(self) -> None:
        source = png_bytes(np.full((24, 24, 3), 255, np.uint8))

        class ForbiddenProvider:
            def describe(self, _image: bytes) -> str:
                raise GroqProviderError(
                    'Groq request failed with HTTP 403: {"error":"model access denied"}', status_code=403,
                )

        with self.assertRaises(HTTPException) as forbidden:
            describe_sketch_response(source, "image/png", ForbiddenProvider())
        self.assertEqual(forbidden.exception.status_code, 403)
        self.assertIn("model access denied", forbidden.exception.detail)


if __name__ == "__main__":
    unittest.main()
