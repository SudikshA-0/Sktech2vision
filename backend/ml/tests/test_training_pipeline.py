"""Small synthetic tests for paired loading and the Pix2Pix training pipeline."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from backend.ml.provenance import training_provenance_findings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from backend.ml.dataset import PairedEdgeImageDataset
    from backend.ml.evaluation import evaluate_generator, load_checkpoint, save_checkpoint
    from backend.ml.models import build_pix2pix
    from backend.ml.train import (
        TrainConfig,
        create_train_validation_loaders,
        run_training,
        validate_resume_config,
    )
    from backend.ml.training import train_one_epoch


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable; run in the documented Colab PyTorch environment")
class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="sketch2vision-train-tests-")
        self.root = Path(self.temp.name) / "dataset"
        for split in ("train", "validation", "test"):
            (self.root / split / "edges").mkdir(parents=True)
            (self.root / split / "targets").mkdir()
        self.write_pair("train", "train-item", 16)
        self.write_pair("validation", "validation-item", 32)
        # Deliberately corrupt test bytes: training/validation loader creation
        # and iteration must not decode or use the held-out test split.
        (self.root / "test" / "edges" / "sealed.png").write_bytes(b"sealed test fixture")
        (self.root / "test" / "targets" / "sealed.png").write_bytes(b"sealed test fixture")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_pair(self, split: str, source_id: str, intensity: int) -> None:
        edge = np.zeros((128, 128), dtype=np.uint8)
        edge[20:100, 64] = 255
        target = np.full((128, 128, 3), intensity, dtype=np.uint8)
        target[:, :, 2] = 240  # Red channel in OpenCV BGR format; loader converts it to RGB.
        self.assertTrue(cv2.imwrite(str(self.root / split / "edges" / f"{source_id}.png"), edge))
        self.assertTrue(cv2.imwrite(str(self.root / split / "targets" / f"{source_id}.png"), target))

    def test_paired_loader_shapes_normalization_rgb_and_no_resize(self) -> None:
        dataset = PairedEdgeImageDataset(self.root, "train", (128, 128))
        item = dataset[0]
        self.assertEqual(tuple(item["edges"].shape), (1, 128, 128))
        self.assertEqual(tuple(item["targets"].shape), (3, 128, 128))
        self.assertGreaterEqual(item["edges"].min().item(), -1)
        self.assertLessEqual(item["edges"].max().item(), 1)
        # OpenCV wrote BGR; loader's conversion makes the red value channel 0.
        self.assertGreater(item["targets"][0].mean().item(), item["targets"][2].mean().item())
        with self.assertRaisesRegex(ValueError, "expected 64x64"):
            PairedEdgeImageDataset(self.root, "train", (64, 64))[0]

    def test_train_validation_loaders_never_decode_test_split(self) -> None:
        train_loader, validation_loader, _ = create_train_validation_loaders(
            self.root, image_size=128, batch_size=1, workers=0, seed=9, use_cuda=False,
        )
        self.assertEqual(train_loader.dataset.split, "train")
        self.assertEqual(validation_loader.dataset.split, "validation")
        self.assertEqual(len(list(train_loader)), 1)
        self.assertEqual(len(list(validation_loader)), 1)

    def test_one_tiny_synthetic_training_step(self) -> None:
        generator, discriminator = build_pix2pix(base_channels=1, num_downs=7)
        device = torch.device("cpu")
        generator.to(device)
        discriminator.to(device)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        batch = next(iter(create_train_validation_loaders(
            self.root, image_size=128, batch_size=1, workers=0, seed=10, use_cuda=False,
        )[0]))
        before = next(generator.parameters()).detach().clone()
        metrics = train_one_epoch(generator, discriminator, [batch], optimizer_g, optimizer_d, device)
        self.assertEqual(metrics.examples, 1)
        self.assertTrue(np.isfinite([metrics.generator_total, metrics.generator_adversarial,
                                     metrics.generator_l1, metrics.discriminator]).all())
        self.assertFalse(torch.equal(before, next(generator.parameters()).detach()))
        self.assertTrue(all(parameter.requires_grad for parameter in discriminator.parameters()))

    def test_validation_l1_is_mean_pixel_error_and_restores_mode(self) -> None:
        class ZeroGenerator(nn.Module):
            def forward(self, edges: torch.Tensor) -> torch.Tensor:
                return torch.zeros((edges.shape[0], 3, edges.shape[2], edges.shape[3]), device=edges.device)

        generator = ZeroGenerator().train()
        batch = {"edges": torch.zeros(1, 1, 128, 128), "targets": torch.ones(1, 3, 128, 128)}
        score = evaluate_generator(generator, [batch], torch.device("cpu"))
        self.assertAlmostEqual(score, 1.0, delta=1e-7)
        self.assertTrue(generator.training)

    def test_checkpoint_round_trip_and_resume_config_policy(self) -> None:
        generator, discriminator = build_pix2pix(base_channels=1)
        path = Path(self.temp.name) / "latest.pt"
        save_checkpoint(path, {"format_version": 1, "generator": generator.state_dict(),
                               "discriminator": discriminator.state_dict()})
        loaded = load_checkpoint(path)
        clone, _ = build_pix2pix(base_channels=1)
        clone.load_state_dict(loaded["generator"])
        for left, right in zip(generator.parameters(), clone.parameters()):
            torch.testing.assert_close(left, right)
        saved = {"epochs": 5, "output_dir": "old", "device": "cuda", "image_size": 128,
                 "batch_size": 1, "seed": 42}
        current = {**saved, "epochs": 8, "output_dir": "new", "device": "cpu"}
        validate_resume_config(saved, current, completed_epoch=5)
        current["seed"] = 43
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_resume_config(saved, current, completed_epoch=5)

    def test_config_rejects_non_power_of_two_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            TrainConfig(image_size=192).validate()

    def test_epoch_boundary_resume_restores_and_continues(self) -> None:
        (self.root / "test" / "edges" / "sealed.png").unlink()
        (self.root / "test" / "targets" / "sealed.png").unlink()
        self.write_pair("test", "test-item", 48)
        ids = {"train": ["train-item"], "validation": ["validation-item"], "test": ["test-item"]}
        manifest = {
            "dataset_kind": "synthetic",
            "image_size": {"width": 128, "height": 128},
            "preprocessing": {"resize_mode": "stretch"},
            "source_ids": ids,
            "source_metadata": {
                source_id: {
                    "source_id": source_id, "source_identifier": source_id,
                    "source_path": f"{source_id}.png", "provenance_schema_version": 1,
                    "record_type": "synthetic", "license_review_status": "not_applicable_synthetic",
                }
                for source_id in ids["train"] + ids["validation"] + ids["test"]
            },
            "skipped": [],
        }
        (self.root / "splits.json").write_text(json.dumps(manifest), encoding="utf-8")
        output = Path(self.temp.name) / "run"
        first = TrainConfig(image_size=128, batch_size=1, epochs=1, base_channels=1,
                            workers=0, checkpoint_interval=1, output_dir=str(output), device="cpu")
        run_training(self.root, first, allow_synthetic_data=True)
        checkpoint_path = output / "latest.pt"
        first_state = load_checkpoint(checkpoint_path)
        self.assertEqual(first_state["completed_epoch"], 1)

        extended = TrainConfig(image_size=128, batch_size=1, epochs=2, base_channels=1,
                               workers=0, checkpoint_interval=1, output_dir=str(output), device="cpu")
        run_training(self.root, extended, resume=checkpoint_path, allow_synthetic_data=True)
        final_state = load_checkpoint(checkpoint_path)
        self.assertEqual(final_state["completed_epoch"], 2)
        self.assertIn("optimizer_g", final_state)
        self.assertIn("rng_state", final_state)


class LicenseGateTests(unittest.TestCase):
    @staticmethod
    def real_record(source_id: str, status: str = "approved") -> dict[str, object]:
        return {
            "source_id": source_id,
            "source_identifier": f"external-{source_id}",
            "source_path": f"{source_id}.jpg",
            "source_url": "https://images.example.test/source.jpg",
            "license_name": "CC BY",
            "license_identifier": "https://creativecommons.org/licenses/by/4.0/",
            "attribution_required": "false",
            "attribution_text": "",
            "creator": "",
            "license_review_status": status,
            "reviewer_notes": "Reviewed source terms and use case.",
            "record_type": "real",
            "provenance_schema_version": 1,
        }

    def manifest(self, records: dict[str, dict[str, object]]) -> dict[str, object]:
        return {
            "dataset_kind": "real",
            "source_ids": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            "source_metadata": records,
        }

    def test_missing_fields_and_missing_records_are_rejected(self) -> None:
        record = self.real_record("a")
        record["source_url"] = ""
        record["attribution_required"] = "unknown"
        record["reviewer_notes"] = ""
        findings = training_provenance_findings(self.manifest({"a": record}))
        self.assertTrue(any("a:" in item and "missing=source_url" in item for item in findings))
        self.assertTrue(any("b: provenance record missing" in item for item in findings))

    def test_pending_license_is_rejected(self) -> None:
        records = {source_id: self.real_record(source_id, "pending_manual_review") for source_id in ("a", "b", "c")}
        findings = training_provenance_findings(self.manifest(records))
        self.assertEqual(len(findings), 3)
        self.assertTrue(all("status=pending_manual_review" in item for item in findings))

    def test_complete_manually_approved_records_pass(self) -> None:
        records = {source_id: self.real_record(source_id) for source_id in ("a", "b", "c")}
        self.assertEqual(training_provenance_findings(self.manifest(records)), [])

    def test_synthetic_override_does_not_approve_or_bypass_real_data(self) -> None:
        ids = ("a", "b", "c")
        synthetic_records = {
            source_id: {"source_id": source_id, "source_path": f"{source_id}.png",
                        "source_identifier": source_id, "provenance_schema_version": 1,
                        "record_type": "synthetic", "license_review_status": "not_applicable_synthetic"}
            for source_id in ids
        }
        synthetic_manifest = self.manifest(synthetic_records)
        synthetic_manifest["dataset_kind"] = "synthetic"
        self.assertEqual(training_provenance_findings(synthetic_manifest, allow_synthetic_data=True), [])
        self.assertEqual(synthetic_records["a"]["license_review_status"], "not_applicable_synthetic")

        real_records = {source_id: self.real_record(source_id, "pending_manual_review") for source_id in ids}
        findings = training_provenance_findings(self.manifest(real_records), allow_synthetic_data=True)
        self.assertTrue(any("cannot override provenance for a real" in item for item in findings))


if __name__ == "__main__":
    unittest.main()
