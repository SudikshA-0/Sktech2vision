"""Replicate-hosted ControlNet Scribble inference provider.

Only receives caller-provided canvas images and prompts. Authentication stays
on the backend; model and output URLs are validated before use.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


REPLICATE_API = "https://api.replicate.com/v1/predictions"
CONTROLNET_SCRIBBLE_VERSION = "435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117"
MAX_IMAGE_DATA_URI_BYTES = 950_000
MAX_OUTPUT_BYTES = 20 * 1024 * 1024


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class PreparedSketch:
    data_uri: str
    mime_type: str


def prepare_sketch_data_uri(image_bytes: bytes) -> PreparedSketch:
    """Flatten and fit the actual submitted sketch into a safe Replicate data URI."""
    if not image_bytes:
        raise ValueError("The sketch image is empty.")
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError("Could not decode the sketch image. Upload or draw a valid PNG/JPEG.")
    height, width = decoded.shape[:2]
    if width * height > 16_000_000 or max(width, height) > 8192:
        raise ValueError("Sketch image dimensions are too large to process.")

    side = max(width, height)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    left, top = (side - width) // 2, (side - height) // 2
    square[top:top + height, left:left + width] = decoded
    scale = min(768 / side, 1.0)
    if scale < 1.0:
        new_side = max(128, int(round(side * scale)))
        square = cv2.resize(square, (new_side, new_side), interpolation=cv2.INTER_AREA)

    for quality in (92, 86, 78):
        ok, encoded = cv2.imencode(".jpg", square, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{payload}"
            if len(data_uri) <= MAX_IMAGE_DATA_URI_BYTES:
                return PreparedSketch(data_uri, "image/jpeg")
    raise ValueError("The sketch could not be compressed below the hosted provider's input limit.")


class ReplicateControlNetScribble:
    """Pinned Replicate ControlNet Scribble model using its HTTP prediction API."""

    def __init__(self, api_token: str, *, timeout_seconds: float = 45, max_wait_seconds: float = 180,
                 poll_interval_seconds: float = 2, opener: Any = urllib.request.urlopen) -> None:
        token = api_token.strip()
        if not token:
            raise ValueError("REPLICATE_API_TOKEN is required.")
        self._token = token
        self.timeout_seconds = timeout_seconds
        self.max_wait_seconds = max_wait_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._opener = opener

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "ReplicateControlNetScribble":
        token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
        if not token:
            raise ProviderError(
                "Hosted sketch generation is not configured. Set REPLICATE_API_TOKEN in backend/.env and restart the backend.",
                status_code=503,
            )
        return cls(token, **kwargs)

    def _request_json(self, url: str, *, payload: dict[str, Any] | None = None,
                      wait_header: str | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if wait_header:
            headers["Prefer"] = wait_header
        request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise ProviderError("The inference provider returned an oversized response.")
                value = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            excerpt = exc.read(2048).decode("utf-8", errors="replace")
            try:
                detail = json.loads(excerpt).get("detail") or json.loads(excerpt).get("error") or ""
            except (ValueError, AttributeError):
                detail = ""
            if exc.code == 401:
                raise ProviderError("Replicate rejected the API token. Check REPLICATE_API_TOKEN.", status_code=503) from exc
            if exc.code == 402:
                raise ProviderError("Replicate reports a billing or account-credit issue.", status_code=402) from exc
            if exc.code == 429:
                raise ProviderError("Replicate rate limit reached. Wait and retry.", status_code=429,
                                    retry_after=exc.headers.get("Retry-After")) from exc
            safe_detail = str(detail)[:300]
            message = f"Replicate request failed with HTTP {exc.code}."
            if safe_detail:
                message += f" {safe_detail}"
            raise ProviderError(message, status_code=502 if exc.code >= 500 else 400) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ProviderError("Could not reach Replicate or the request timed out. Try again shortly.", status_code=504) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderError("Replicate returned a malformed response.") from exc
        if not isinstance(value, dict):
            raise ProviderError("Replicate returned an unexpected response.")
        return value

    @staticmethod
    def _validate_output_url(url: Any) -> str:
        if not isinstance(url, str):
            raise ProviderError("The model did not return an image URL.")
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (hostname == "replicate.delivery" or hostname.endswith(".replicate.delivery")):
            raise ProviderError("The model returned an image URL from an untrusted host.")
        return url

    def _wait_for_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        status = prediction.get("status")
        get_url = prediction.get("urls", {}).get("get") if isinstance(prediction.get("urls"), dict) else None
        if status in {"succeeded", "failed", "canceled"}:
            return prediction
        if not isinstance(get_url, str) or not get_url.startswith("https://api.replicate.com/v1/predictions/"):
            raise ProviderError("Replicate returned an invalid prediction polling URL.")
        deadline = time.monotonic() + self.max_wait_seconds
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval_seconds)
            prediction = self._request_json(get_url)
            if prediction.get("status") in {"succeeded", "failed", "canceled"}:
                return prediction
        raise ProviderError("Image generation took too long. You can retry from the canvas.", status_code=504)

    def _download_output(self, url: str) -> bytes:
        trusted_url = self._validate_output_url(url)
        request = urllib.request.Request(trusted_url, headers={"User-Agent": "Sketch2Vision/1.0"})
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                image = response.read(MAX_OUTPUT_BYTES + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError("Could not retrieve the generated image from Replicate.", status_code=502) from exc
        if not image or len(image) > MAX_OUTPUT_BYTES:
            raise ProviderError("The generated image was empty or too large.")
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ProviderError("Replicate returned a file that is not a readable image.")
        ok, png = cv2.imencode(".png", decoded)
        if not ok:
            raise ProviderError("Could not convert the generated image to PNG.")
        return png.tobytes()

    def generate_png(self, image_bytes: bytes, prompt: str) -> bytes:
        prepared = prepare_sketch_data_uri(image_bytes)
        prompt = prompt.strip()[:500] if isinstance(prompt, str) else ""
        if not prompt:
            prompt = "a detailed realistic image based on the user's sketch"
        prediction = self._request_json(
            REPLICATE_API,
            payload={
                "version": CONTROLNET_SCRIBBLE_VERSION,
                "input": {
                    "image": prepared.data_uri,
                    "prompt": prompt,
                    "num_samples": "1",
                    "image_resolution": "512",
                    "ddim_steps": 20,
                    "scale": 9,
                    "a_prompt": "best quality, high detail, coherent composition",
                    "n_prompt": "lowres, blurry, worst quality, malformed, distorted",
                },
            },
            wait_header="wait=60",
        )
        prediction = self._wait_for_prediction(prediction)
        status = prediction.get("status")
        if status != "succeeded":
            safe_error = str(prediction.get("error") or "model prediction failed")[:300]
            raise ProviderError(f"Replicate image generation {status or 'failed'}: {safe_error}", status_code=502)
        output = prediction.get("output")
        image_url = output[0] if isinstance(output, list) and output else output
        return self._download_output(image_url)

