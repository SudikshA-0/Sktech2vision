"""Offline tests for Groq sketch-description request handling."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.app.providers.groq_sketch_description import (
    DEFAULT_VISION_MODEL,
    GROQ_CHAT_COMPLETIONS_API,
    GroqProviderError,
    GroqSketchDescription,
    encode_sketch_data_uri,
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


def sample_png() -> bytes:
    canvas = np.full((80, 100, 3), 255, np.uint8)
    cv2.line(canvas, (20, 60), (50, 15), (0, 0, 0), 4)
    cv2.line(canvas, (50, 15), (80, 60), (0, 0, 0), 4)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise AssertionError("fixture PNG encoding failed")
    return encoded.tobytes()


def common_object_sketches() -> list[tuple[str, bytes, str]]:
    """Simple offline outlines; labels are used only for mocked responses."""
    images: list[tuple[str, np.ndarray, str]] = []

    tree = np.full((128, 128, 3), 255, np.uint8)
    cv2.line(tree, (64, 112), (64, 55), (0, 0, 0), 4)
    cv2.line(tree, (64, 83), (39, 60), (0, 0, 0), 3)
    cv2.line(tree, (64, 80), (87, 57), (0, 0, 0), 3)
    for center in ((42, 48), (64, 34), (87, 47), (52, 54), (76, 54)):
        cv2.circle(tree, center, 18, (0, 0, 0), 2)
    images.append(("tree", tree, "A tree with a rounded canopy and trunk."))

    house = np.full((128, 128, 3), 255, np.uint8)
    cv2.rectangle(house, (31, 56), (97, 108), (0, 0, 0), 3)
    cv2.polylines(house, [np.array([[25, 58], [64, 24], [103, 58]])], False, (0, 0, 0), 3)
    images.append(("house", house, "A small house with a peaked roof."))

    flower = np.full((128, 128, 3), 255, np.uint8)
    cv2.line(flower, (64, 63), (64, 117), (0, 0, 0), 3)
    cv2.circle(flower, (64, 45), 10, (0, 0, 0), 2)
    for center in ((64, 25), (84, 45), (64, 65), (44, 45)):
        cv2.circle(flower, center, 10, (0, 0, 0), 2)
    images.append(("flower", flower, "A flower with petals, a center, and a stem."))

    car = np.full((128, 128, 3), 255, np.uint8)
    cv2.polylines(car, [np.array([[19, 84], [27, 59], [47, 54], [61, 37], [89, 39], [105, 62], [112, 70], [112, 88], [19, 88]])], True, (0, 0, 0), 3)
    cv2.circle(car, (42, 88), 10, (0, 0, 0), 3)
    cv2.circle(car, (91, 88), 10, (0, 0, 0), 3)
    images.append(("car", car, "A car with two visible wheels and a sloped roof."))

    result = []
    for name, image, mocked_answer in images:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise AssertionError(f"fixture {name} PNG encoding failed")
        result.append((name, encoded.tobytes(), mocked_answer))
    return result


def text_and_mixed_sketches() -> list[tuple[str, bytes, str]]:
    """Offline OCR-oriented fixtures; response text is supplied by a mock only."""
    word = np.full((128, 256, 3), 255, np.uint8)
    cv2.putText(word, "Apple", (16, 84), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)

    name = np.full((128, 256, 3), 255, np.uint8)
    cv2.putText(name, "Sudo", (16, 84), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)

    unclear_word = np.full((128, 256, 3), 255, np.uint8)
    cv2.putText(unclear_word, "A...le", (16, 84), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)

    tree = np.full((256, 256, 3), 255, np.uint8)
    cv2.putText(tree, "Tree", (74, 35), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(tree, (128, 226), (128, 121), (0, 0, 0), 5)
    for center in ((128, 83), (91, 112), (165, 110), (106, 72), (150, 68)):
        cv2.circle(tree, center, 38, (0, 0, 0), 3)

    result = []
    fixtures = [
        ("handwritten_word", word, "The handwritten word appears to be 'Apple'."),
        ("handwritten_name", name, "The handwritten name appears to be 'Sudo'."),
        ("unclear_handwriting", unclear_word, "The handwritten word may read 'A...le'; the middle letters are unclear."),
        ("mixed_text_and_tree", tree, "This is a tree drawing, with the handwritten word 'Tree' above it."),
    ]
    for label, image, mocked_answer in fixtures:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise AssertionError(f"fixture {label} PNG encoding failed")
        result.append((label, encoded.tobytes(), mocked_answer))
    return result


class GroqSketchDescriptionTests(unittest.TestCase):
    def test_sends_actual_sketch_as_documented_multimodal_chat_completion(self) -> None:
        seen = []
        answers = iter(answer for _name, _image, answer in common_object_sketches())

        def opener(request, timeout=0):
            self.assertEqual(request.full_url, GROQ_CHAT_COMPLETIONS_API)
            self.assertLessEqual(timeout, 45)
            self.assertEqual(request.get_header("Authorization"), "Bearer mock-key")
            self.assertEqual(request.get_header("Content-type"), "application/json")
            self.assertEqual(request.get_header("User-agent"), "Sketch2Vision/1.0")
            payload = json.loads(request.data.decode("utf-8"))
            seen.append(payload)
            return FakeResponse(json.dumps({
                "choices": [{"message": {"content": next(answers)}}],
            }).encode("utf-8"))

        provider = GroqSketchDescription("mock-key", opener=opener)
        sketches = common_object_sketches()
        descriptions = [provider.describe(image) for _name, image, _answer in sketches]
        self.assertEqual(descriptions, [answer for _name, _image, answer in sketches])
        self.assertEqual(len(seen), 4)
        data_uris = []
        for payload in seen:
            self.assertEqual(payload["model"], DEFAULT_VISION_MODEL)
            system_prompt = payload["messages"][0]["content"].lower()
            user_prompt = payload["messages"][1]["content"][0]["text"].lower()
            self.assertIn("handwriting", system_prompt)
            self.assertIn("overall silhouette", system_prompt)
            self.assertIn("overall silhouette", system_prompt)
            self.assertIn("do not invent a complete word or name", system_prompt)
            self.assertIn("combination", system_prompt)
            self.assertIn("handwritten word or name", user_prompt)
            self.assertNotIn("tree", system_prompt + user_prompt)
            self.assertNotIn("apple", system_prompt + user_prompt)
            self.assertNotIn("sudo", system_prompt + user_prompt)
            content = payload["messages"][1]["content"]
            self.assertEqual(content[1]["type"], "image_url")
            image_data_uri = content[1]["image_url"]["url"]
            self.assertTrue(image_data_uri.startswith("data:image/jpeg;base64,"))
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(image_data_uri.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape[:2], (128, 128))
            data_uris.append(image_data_uri)
        self.assertEqual(len(set(data_uris)), 4)

    def test_mocked_text_name_mixed_and_uncertain_handwriting_responses(self) -> None:
        fixtures = text_and_mixed_sketches()
        answers = [answer for _label, _image, answer in fixtures]
        responses = iter(answers)
        seen_payloads = []

        def opener(request, timeout=0):
            payload = json.loads(request.data.decode("utf-8"))
            seen_payloads.append(payload)
            return FakeResponse(json.dumps({"choices": [{"message": {"content": next(responses)}}]}).encode())

        provider = GroqSketchDescription("mock-key", opener=opener)
        actual = [provider.describe(image) for _label, image, _answer in fixtures]
        self.assertEqual(actual, answers)
        self.assertEqual(len(seen_payloads), 4)
        for payload in seen_payloads:
            full_prompt = (payload["messages"][0]["content"] + " " +
                           payload["messages"][1]["content"][0]["text"]).lower()
            self.assertIn("handwritten", full_prompt)
            self.assertIn("uncertain", full_prompt)
            self.assertNotIn("apple", full_prompt)
            self.assertNotIn("sudo", full_prompt)
            self.assertNotIn("tree", full_prompt)

        self.assertIn("middle letters are unclear", actual[2])

    def test_missing_key_and_invalid_image_are_reported(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GroqProviderError) as caught:
                GroqSketchDescription.from_environment()
        self.assertEqual(caught.exception.status_code, 503)
        with self.assertRaisesRegex(ValueError, "decode"):
            encode_sketch_data_uri(b"not an image")

    def test_rate_limit_and_invalid_response_are_handled_without_key_leak(self) -> None:
        def rate_limited(request, timeout=0):
            raise urllib.error.HTTPError(request.full_url, 429, "limited", {"Retry-After": "8"}, None)

        provider = GroqSketchDescription("private-test-key", opener=rate_limited)
        with self.assertRaises(GroqProviderError) as caught:
            provider.describe(sample_png())
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.retry_after, "8")
        self.assertNotIn("private-test-key", str(caught.exception))

        malformed = GroqSketchDescription("mock-key", opener=lambda *_args, **_kwargs: FakeResponse(b"{}"))
        with self.assertRaisesRegex(GroqProviderError, "unexpected response"):
            malformed.describe(sample_png())

    def test_forbidden_response_body_is_logged_and_returned_with_secrets_redacted(self) -> None:
        secret = "private-key-value"
        body = json.dumps({
            "error": {
                "message": "Project is not permitted to use this model.",
                "api_key": secret,
            },
        }).encode("utf-8")

        def forbidden(request, timeout=0):
            raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, __import__("io").BytesIO(body))

        provider = GroqSketchDescription(secret, opener=forbidden)
        with self.assertLogs("backend.app.providers.groq_sketch_description", level="WARNING") as logs:
            with self.assertRaises(GroqProviderError) as caught:
                provider.describe(sample_png())
        self.assertEqual(caught.exception.status_code, 403)
        self.assertIn("Project is not permitted to use this model.", str(caught.exception))
        self.assertIn("Project is not permitted to use this model.", "\n".join(logs.output))
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, "\n".join(logs.output))
        self.assertIn("[REDACTED]", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
