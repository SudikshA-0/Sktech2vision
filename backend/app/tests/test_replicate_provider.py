"""Offline tests for the hosted sketch-conditioned provider integration."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.app.providers.replicate_controlnet import (
    CONTROLNET_SCRIBBLE_VERSION,
    ProviderError,
    ReplicateControlNetScribble,
    prepare_sketch_data_uri,
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int = -1) -> bytes:
        return self.body


def image_bytes(seed: int) -> bytes:
    canvas = np.full((96, 128, 3), 255, np.uint8)
    if seed == 0:
        cv2.circle(canvas, (64, 48), 25, (0, 0, 0), 4)
    elif seed == 1:
        cv2.rectangle(canvas, (28, 25), (100, 75), (0, 0, 0), 4)
    else:
        cv2.line(canvas, (20, 70), (64, 20), (0, 0, 0), 4)
        cv2.line(canvas, (64, 20), (108, 70), (0, 0, 0), 4)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise AssertionError("PNG fixture encoding failed")
    return encoded.tobytes()


class ReplicateProviderTests(unittest.TestCase):
    def test_provider_sends_actual_image_and_prompt_for_object_examples(self) -> None:
        generated = np.full((64, 64, 3), (30, 100, 220), np.uint8)
        ok, output = cv2.imencode(".png", generated)
        self.assertTrue(ok)
        returned = output.tobytes()
        seen: list[dict] = []

        def opener(request, timeout=0):
            self.assertLessEqual(timeout, 45)
            if request.full_url.endswith("/v1/predictions"):
                payload = json.loads(request.data.decode("utf-8"))
                seen.append(payload)
                self.assertEqual(payload["version"], CONTROLNET_SCRIBBLE_VERSION)
                self.assertTrue(request.get_header("Authorization").startswith("Bearer "))
                return FakeResponse(json.dumps({
                    "status": "succeeded", "output": ["https://replicate.delivery/demo/result.png"],
                }).encode())
            if request.full_url == "https://replicate.delivery/demo/result.png":
                return FakeResponse(returned)
            raise AssertionError(f"unexpected provider URL: {request.full_url}")

        provider = ReplicateControlNetScribble("test-token-never-real", opener=opener)
        examples = [
            (image_bytes(0), "a red apple on a white background"),
            (image_bytes(1), "a small house"),
            (image_bytes(2), "a flower in a garden"),
        ]
        for sketch, prompt in examples:
            result = provider.generate_png(sketch, prompt)
            decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (64, 64, 3))
        self.assertEqual([item["input"]["prompt"] for item in seen], [prompt for _, prompt in examples])
        supplied = [
            base64.b64decode(item["input"]["image"].split(",", 1)[1])
            for item in seen
        ]
        self.assertEqual(len(set(supplied)), 3)
        self.assertTrue(all(item["input"]["image"].startswith("data:image/jpeg;base64,") for item in seen))

    def test_rate_limit_exposes_retry_after_without_token(self) -> None:
        def opener(request, timeout=0):
            raise urllib.error.HTTPError(
                request.full_url, 429, "rate limited", {"Retry-After": "12"}, None,
            )

        provider = ReplicateControlNetScribble("secret-test-token", opener=opener)
        with self.assertRaises(ProviderError) as caught:
            provider.generate_png(image_bytes(0), "apple")
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.retry_after, "12")
        self.assertNotIn("secret-test-token", str(caught.exception))

    def test_provider_requires_server_environment_token_and_rejects_untrusted_output(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderError) as caught:
                ReplicateControlNetScribble.from_environment()
        self.assertEqual(caught.exception.status_code, 503)
        with self.assertRaisesRegex(ProviderError, "untrusted host"):
            ReplicateControlNetScribble._validate_output_url("https://example.com/fake.png")

    def test_image_preparation_rejects_invalid_and_oversized_sketches(self) -> None:
        with self.assertRaisesRegex(ValueError, "decode"):
            prepare_sketch_data_uri(b"not an image")
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_sketch_data_uri(b"")


if __name__ == "__main__":
    unittest.main()
