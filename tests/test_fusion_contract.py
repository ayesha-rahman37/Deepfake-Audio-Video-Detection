"""Unit tests for both fusion modules; uses only synthetic feature vectors."""
import unittest

import torch

from models.fusion import ConcatenationFusion, ReliabilityAwareFusion


class FusionContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.video = torch.randn(3, 512, requires_grad=True)
        self.audio = torch.randn(3, 256, requires_grad=True)
        self.quality = torch.rand(3, 4)

    def check_gradients(self, model, expected_feature_dim):
        fused, logits, weights = model(self.video, self.audio, self.quality)
        self.assertEqual(tuple(fused.shape), (3, expected_feature_dim))
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(weights.shape), (3, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(3), atol=1e-6))
        logits.sum().backward()
        self.assertIsNotNone(self.video.grad)
        self.assertIsNotNone(self.audio.grad)

    def test_concatenation_baseline(self):
        self.check_gradients(ConcatenationFusion(), 768)

    def test_reliability_fusion(self):
        self.check_gradients(ReliabilityAwareFusion(), 512)

    def test_batch_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ConcatenationFusion()(self.video, self.audio[:2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
