"""Groq vision API client for describing a user-provided sketch."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np


GROQ_CHAT_COMPLETIONS_API = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
MAX_REQUEST_IMAGE_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)


def _safe_error_excerpt(raw: bytes, api_key: str, limit: int = 1400) -> str:
    """Keep useful provider diagnostics while redacting credential-like values."""
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(text)

        def redact(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: "[REDACTED]" if re.search(r"api.?key|token|secret|authorization|credential", str(key), re.I)
                    else redact(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        text = json.dumps(redact(parsed), ensure_ascii=False, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        pass
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(api[_ -]?key|token|secret|authorization|credential)(\s*[:=]\s*)[^\s,;]+",
                  r"\1\2[REDACTED]", text)
    return text[:limit] or "<empty response body>"


class GroqProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def encode_sketch_data_uri(image_bytes: bytes) -> str:
    if not image_bytes:
        raise ValueError("The sketch image is empty.")
    if len(image_bytes) > MAX_REQUEST_IMAGE_BYTES:
        raise ValueError("Sketch image exceeds the 10 MB limit.")
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError("Could not decode the sketch image. Send a valid PNG or JPEG.")
    height, width = decoded.shape[:2]
    if width * height > 16_000_000 or max(width, height) > 8192:
        raise ValueError("Sketch image dimensions are too large to process.")
    scale = min(1024 / max(width, height), 1.0)
    if scale < 1.0:
        decoded = cv2.resize(decoded, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", decoded, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Could not prepare the sketch image for analysis.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class GroqSketchDescription:
    """Small stdlib HTTP client; API credentials remain server-side."""

    def __init__(self, api_key: str, *, model: str = DEFAULT_VISION_MODEL, timeout_seconds: float = 45,
                 opener: Any = urllib.request.urlopen) -> None:
        if not api_key.strip():
            raise ValueError("GROQ_API_KEY is required.")
        self._api_key = api_key.strip()
        self.model = model.strip() or DEFAULT_VISION_MODEL
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "GroqSketchDescription":
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise GroqProviderError(
                "Sketch description is not configured. Set GROQ_API_KEY in backend/.env and restart the backend.",
                status_code=503,
            )
        return cls(key, model=os.environ.get("GROQ_VISION_MODEL", DEFAULT_VISION_MODEL), **kwargs)

    def describe(self, image_bytes: bytes) -> str:
        image_uri = encode_sketch_data_uri(image_bytes)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You interpret sketch images that may contain drawings, handwriting, or both. Inspect the whole "
                        "image and determine whether it mainly shows an object, handwritten text/name, or a combination. "
                        "For a drawing, identify the most likely object from its overall silhouette and familiar visual "
                        "patterns. For handwriting, attempt to read the visible word or name; do not treat letters as an "
                        "abstract drawing. If some writing is unclear, state which part or letters are uncertain and do "
                        "not invent a complete word or name. If text and a drawing are both present, report both and their "
                        "visible relationship. Avoid vague shape labels when a recognizable object or text is present. "
                        "Give a concise, natural answer, naming the object or reporting the recognized text first. Briefly "
                        "describe only visible features; do not invent colors, materials, context, or other details."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Look for both recognizable objects and handwritten letters. Identify the most likely "
                                "object and read any legible handwritten word or name. If both occur, describe both. "
                                "Do not guess unclear letters; briefly mark the uncertain part. Keep the result concise."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                },
            ],
            "temperature": 0.2,
            "max_completion_tokens": 180,
            "stream": False,
        }
        request = urllib.request.Request(
            GROQ_CHAT_COMPLETIONS_API,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Sketch2Vision/1.0",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise GroqProviderError("Groq returned an oversized response.")
            result = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                response_body = exc.read(8192)
            except Exception:
                response_body = b""
            excerpt = _safe_error_excerpt(response_body, self._api_key)
            logger.warning("Groq API returned HTTP %s: %s", exc.code, excerpt)
            if exc.code == 401:
                raise GroqProviderError(
                    f"Groq rejected the API key. Check GROQ_API_KEY in backend/.env. Provider response: {excerpt}",
                    status_code=503,
                ) from exc
            if exc.code == 429:
                raise GroqProviderError(f"Groq rate limit reached. {excerpt}", status_code=429,
                                        retry_after=retry_after) from exc
            raise GroqProviderError(f"Groq request failed with HTTP {exc.code}: {excerpt}",
                                    status_code=exc.code if 400 <= exc.code < 500 else 502) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise GroqProviderError("Could not reach Groq or the request timed out. Try again shortly.", status_code=504) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GroqProviderError("Groq returned a malformed response.") from exc

        try:
            description = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqProviderError("Groq returned an unexpected response format.") from exc
        if not isinstance(description, str) or not description.strip():
            raise GroqProviderError("Groq returned an empty sketch description.")
        return description.strip()[:2000]
