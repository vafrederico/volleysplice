from __future__ import annotations

import unittest

import numpy as np

from analysis.semantic_temporal_model import (
    HEAD_NAMES,
    SemanticTemporalNetwork,
    TemporalModelConfig,
    fit_audiovisual_scaler,
    positive_weights,
    standardize_audiovisual,
    trainable_parameter_count,
    weighted_temporal_loss,
)


@unittest.skipUnless(
    __import__("importlib.util").util.find_spec("torch") is not None,
    "PyTorch is not installed in this test interpreter",
)
class SemanticTemporalModelTests(unittest.TestCase):
    def test_initial_head_is_small_and_has_three_aligned_outputs(self) -> None:
        import torch

        config = TemporalModelConfig(
            dino_projection_dimension=4,
            audiovisual_dimension=3,
            hidden_dimension=16,
            dilations=(1, 2),
            groups=4,
            dropout=0.0,
        )
        model = SemanticTemporalNetwork(config)
        self.assertLess(trainable_parameter_count(model), 2_000_000)
        outputs = model(
            torch.randn(2, 12, 10, 384),
            torch.randn(2, 12, 3),
        )
        self.assertEqual(set(outputs), set(HEAD_NAMES))
        for value in outputs.values():
            self.assertEqual(tuple(value.shape), (2, 12))

    def test_fold_scaler_and_positive_weights_are_finite_and_local(self) -> None:
        values = [
            np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32),
            np.asarray([[4.0, 6.0], [6.0, 8.0]], dtype=np.float32),
        ]
        masks = [np.asarray([True, False]), np.asarray([True, True])]
        mean, scale = fit_audiovisual_scaler(values, masks)
        np.testing.assert_allclose(mean, np.asarray([10 / 3, 16 / 3], dtype=np.float32))
        self.assertTrue(np.all(scale > 0))
        standardized = standardize_audiovisual(values[0], mean, scale)
        self.assertTrue(np.isfinite(standardized).all())
        weights = positive_weights(
            {
                "live": [np.asarray([1, 0, 0], dtype=np.float32), np.asarray([0, 0, 1], dtype=np.float32)],
                "serve": [np.asarray([0, 0, 0], dtype=np.float32), np.asarray([0, 1, 0], dtype=np.float32)],
                "end": [np.asarray([0, 1, 0], dtype=np.float32), np.asarray([0, 0, 0], dtype=np.float32)],
            },
            [np.ones(3, dtype=bool), np.ones(3, dtype=bool)],
        )
        self.assertEqual(weights["live"], 2.0)
        self.assertEqual(weights["serve"], 5.0)
        self.assertEqual(weights["end"], 5.0)

    def test_declared_loss_masks_padding_and_backpropagates(self) -> None:
        import torch

        logits = {name: torch.zeros(2, 4, requires_grad=True) for name in HEAD_NAMES}
        targets = {name: torch.zeros(2, 4) for name in HEAD_NAMES}
        targets["live"][0, 0] = 1.0
        valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
        loss, pieces = weighted_temporal_loss(logits, targets, valid, {name: 2.0 for name in HEAD_NAMES})
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(set(pieces), set(HEAD_NAMES))
        loss.backward()
        for value in logits.values():
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))


if __name__ == "__main__":
    unittest.main()
