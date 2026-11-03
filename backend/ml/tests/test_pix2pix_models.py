"""CPU-only synthetic tests for Pix2Pix modules; skipped if PyTorch is absent."""

from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from backend.ml.losses import discriminator_loss, generator_loss
    from backend.ml.models import PatchGANDiscriminator, UNetGenerator, build_pix2pix
    from backend.ml.transforms import denormalize_to_uint8, normalize_uint8_tensor


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; run these CPU tests in the documented Colab environment")
class Pix2PixModelTests(unittest.TestCase):
    def test_generator_shape_range_and_default_width(self) -> None:
        generator = UNetGenerator(base_channels=4)
        edges = torch.rand(1, 1, 128, 128, dtype=torch.float32) * 2 - 1
        generated = generator(edges)
        self.assertEqual(tuple(generated.shape), (1, 3, 128, 128))
        self.assertTrue(torch.isfinite(generated).all().item())
        self.assertGreaterEqual(generated.min().item(), -1.0)
        self.assertLessEqual(generated.max().item(), 1.0)
        self.assertEqual(UNetGenerator().base_channels, 32)

    def test_generator_rejects_invalid_channels_dtype_and_dimensions(self) -> None:
        generator = UNetGenerator(base_channels=4)
        with self.assertRaisesRegex(ValueError, r"shape \[B, 1, H, W\]"):
            generator(torch.zeros(1, 3, 128, 128))
        with self.assertRaisesRegex(TypeError, "floating point"):
            generator(torch.zeros(1, 1, 128, 128, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "divisible"):
            generator(torch.zeros(1, 1, 130, 128))

    def test_patchgan_shape_pair_and_concatenated_inputs(self) -> None:
        discriminator = PatchGANDiscriminator(base_channels=4).eval()
        edges = torch.zeros(1, 1, 128, 128)
        image = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            pair_logits = discriminator(edges, image)
            combined_logits = discriminator(torch.cat((edges, image), dim=1))
        self.assertEqual(tuple(pair_logits.shape), (1, 1, 14, 14))
        torch.testing.assert_close(pair_logits, combined_logits)

    def test_discriminator_rejects_invalid_pair(self) -> None:
        discriminator = PatchGANDiscriminator(base_channels=4)
        with self.assertRaisesRegex(ValueError, "same batch size"):
            discriminator(torch.zeros(1, 1, 128, 128), torch.zeros(2, 3, 128, 128))
        with self.assertRaisesRegex(ValueError, "at least 70"):
            discriminator(torch.zeros(1, 1, 64, 64), torch.zeros(1, 3, 64, 64))

    def test_loss_values_match_manual_baseline(self) -> None:
        real_logits = torch.zeros(1, 1, 2, 2)
        fake_logits = torch.zeros(1, 1, 2, 2)
        expected_bce = torch.log(torch.tensor(2.0)).item()
        d_loss = discriminator_loss(real_logits, fake_logits)
        self.assertAlmostEqual(d_loss.item(), expected_bce, delta=1e-5)

        generated = torch.zeros(1, 3, 4, 4)
        target = torch.ones_like(generated)
        total, adversarial, weighted_l1 = generator_loss(fake_logits, generated, target)
        self.assertAlmostEqual(adversarial.item(), expected_bce, delta=1e-5)
        self.assertAlmostEqual(weighted_l1.item(), 100.0, delta=1e-5)
        self.assertAlmostEqual(total.item(), 100.0 + expected_bce, delta=1e-5)

    def test_losses_reject_incompatible_shapes_and_weights(self) -> None:
        logits = torch.zeros(1, 1, 2, 2)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            discriminator_loss(logits, torch.zeros(1, 1, 3, 3))
        with self.assertRaisesRegex(ValueError, "identical rank-4"):
            generator_loss(logits, torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 5, 4))
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            generator_loss(logits, torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4), -1)

    def test_losses_reject_device_or_dtype_mismatches(self) -> None:
        logits = torch.zeros(1, 1, 2, 2)
        with self.assertRaisesRegex(ValueError, "same device and dtype"):
            discriminator_loss(logits, logits.double())
        with self.assertRaisesRegex(ValueError, "same device and dtype"):
            generator_loss(logits.double(), torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4))

    def test_gradients_reach_both_models(self) -> None:
        generator, discriminator = build_pix2pix(base_channels=4)
        edges = torch.rand(1, 1, 128, 128) * 2 - 1
        target = torch.rand(1, 3, 128, 128) * 2 - 1
        generated = generator(edges)
        fake_logits = discriminator(edges, generated)
        total, _, _ = generator_loss(fake_logits, generated, target)
        total.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in generator.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in discriminator.parameters()))

    def test_uint8_normalization_round_trip(self) -> None:
        source = torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
        normalized = normalize_uint8_tensor(source)
        self.assertEqual(normalized.dtype, torch.float32)
        torch.testing.assert_close(normalized, torch.tensor([[[[-1.0, 1.0]]]]))
        torch.testing.assert_close(denormalize_to_uint8(normalized), source)


if __name__ == "__main__":
    unittest.main()
