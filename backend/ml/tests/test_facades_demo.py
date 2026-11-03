"""Offline tests for the isolated Facades demo adapter and training path."""

from __future__ import annotations

import importlib.util
import tempfile
import tarfile
import unittest
from pathlib import Path

import cv2
import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader

    from backend.ml.facades_demo import (
        FacadesPairDataset,
        extract_facades,
        predict_one,
        tensor_to_image,
        train_facades_demo,
    )
    from backend.ml.models import build_pix2pix


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for Facades demo tests")
class FacadesDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="facades-demo-test-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "source" / "facades"
        for split in ("train", "val", "test"):
            folder = self.source / split
            folder.mkdir(parents=True)
            # Side-by-side 128px semantic label map | 128px RGB photo.
            pair = np.zeros((128, 256, 3), dtype=np.uint8)
            pair[:, :128, :] = 0
            pair[24:104, 32:96, :] = (255, 255, 255)
            pair[:, 128:, :] = (40, 90, 180)  # RGB target after OpenCV conversion.
            bgr = cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
            self.assertTrue(cv2.imwrite(str(folder / "sample.jpg"), bgr))
        self.archive = self.root / "facades-test.tar.gz"
        with tarfile.open(self.archive, "w:gz") as tar:
            tar.add(self.source, arcname="facades")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_safe_extract_and_paired_loader_geometry_and_normalization(self) -> None:
        extracted = extract_facades(self.archive, self.root / "extracted")
        dataset = FacadesPairDataset(extracted, "train", image_size=128)
        example = dataset[0]
        self.assertEqual(example["source_id"], "sample")
        self.assertEqual(tuple(example["edges"].shape), (1, 128, 128))
        self.assertEqual(tuple(example["targets"].shape), (3, 128, 128))
        self.assertGreaterEqual(example["edges"].min().item(), -1)
        self.assertLessEqual(example["edges"].max().item(), 1)
        # RGB channel ordering survives the OpenCV BGR decode path.
        center = example["targets"][:, 64, 64]
        self.assertGreater(center[2].item(), center[0].item())

    def test_training_checkpoint_and_sample_inference(self) -> None:
        extracted = extract_facades(self.archive, self.root / "extracted")
        train = DataLoader(FacadesPairDataset(extracted, "train"), batch_size=1)
        validation = DataLoader(FacadesPairDataset(extracted, "val"), batch_size=1)
        generator, discriminator = build_pix2pix(base_channels=1)
        device = torch.device("cpu")
        edge = next(iter(validation))["edges"][0]
        initial = predict_one(generator, edge, device)
        self.assertEqual(tuple(initial.shape), (3, 128, 128))
        self.assertEqual(tensor_to_image(initial).shape, (128, 128, 3))
        history = train_facades_demo(
            generator, discriminator, train, validation, device, self.root / "run", epochs=1,
        )
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(list(history[0].values())).all())
        self.assertTrue((self.root / "run" / "latest.pt").is_file())
        self.assertTrue((self.root / "run" / "best.pt").is_file())

    def test_unsafe_tar_path_is_rejected(self) -> None:
        unsafe_archive = self.root / "unsafe.tar.gz"
        with tarfile.open(unsafe_archive, "w:gz") as tar:
            data = b"unsafe"
            import io

            info = tarfile.TarInfo("../escape.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, "unsafe archive"):
            extract_facades(unsafe_archive, self.root / "unsafe-output")


if __name__ == "__main__":
    unittest.main()
