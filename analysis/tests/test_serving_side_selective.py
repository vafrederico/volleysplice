"""Independent loss-gradient and checkpoint checks for the selective head."""
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

try:
    import torch
    from analysis import serving_side_selective as study
    from analysis.serving_side_v2 import LogisticModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Research PyTorch environment required")
class SelectiveHeadTests(unittest.TestCase):
    def setUp(self):
        study.h.prior.configure_torch()
        torch.manual_seed(19)
        rng = np.random.default_rng(19)
        self.base = LogisticModel(
            np.zeros(237), rng.normal(0, .1, 237), np.full(237, 1.3),
            rng.normal(0, .02, 237), .1, .1)
        self.windows = torch.randn(4, 13, 3843)
        self.windows[..., 3840] = 1.
        self.features = torch.tensor(rng.normal(size=(4, 237)), dtype=torch.float32)

    def test_acceptance_gradient_rewards_correct_confidence_and_rejects_ambiguity(self):
        # Repeat for both sides so a swapped far/near truth cannot pass.
        for label in (0., 1.):
            for true_probability, direction in ((.995, -1), (.6, 1), (.005, 1)):
                with self.subTest(label=label, true_probability=true_probability):
                    near_probability = true_probability if label else 1 - true_probability
                    logit = torch.tensor([math.log(near_probability / (1 - near_probability))],
                                         dtype=torch.float64, requires_grad=True)
                    accept = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
                    residual = torch.zeros(1, dtype=torch.float64, requires_grad=True)
                    truth = torch.tensor([label], dtype=torch.float64)
                    loss = study.selective_loss(logit, accept, truth, residual, study.WARMUP + 1)
                    loss.sum().backward()
                    reward = .5 * true_probability + .5 * (1 - study.REVIEW_COST)
                    expected = -.25 * (true_probability - (1 - study.REVIEW_COST)) / reward
                    self.assertAlmostEqual(float(accept.grad[0]), expected, places=12)
                    self.assertEqual(int(torch.sign(accept.grad[0]).item()), direction)
                    # Gradient descent should move the side logit toward its truth.
                    self.assertLess(float(logit.grad[0]) * (2 * label - 1), 0.)

    def test_warmup_has_no_acceptance_gradient_or_optimizer_update(self):
        head = study.SelectiveHead(self.base).train()
        before = {k: v.detach().clone() for k, v in head.acceptance.state_dict().items()}
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.1)
        logits, accept, residual = head.outputs(self.windows, self.features)
        accept.retain_grad()
        truth = torch.tensor([0., 1., 0., 1.])
        loss = study.selective_loss(logits, accept, truth, residual, study.WARMUP).mean()
        loss.backward()
        self.assertIsNone(accept.grad)
        self.assertTrue(all(parameter.grad is None for parameter in head.acceptance.parameters()))
        self.assertGreater(float(head.correction[-1].weight.grad.abs().sum()), 0.)
        optimizer.step()
        for name, value in head.acceptance.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0., rtol=0.)

    def test_three_outputs_are_valid_in_training_and_evaluation(self):
        head = study.SelectiveHead(self.base)
        # Exercise nonconstant acceptance and nonzero correction, including dropout.
        torch.nn.init.normal_(head.acceptance[-1].weight, std=.03)
        torch.nn.init.normal_(head.correction[-1].weight, std=.03)
        for training in (True, False):
            with self.subTest(training=training):
                head.train(training)
                logits, acceptance_logits, residual = head.outputs(self.windows, self.features)
                q, a = logits.sigmoid(), acceptance_logits.sigmoid()
                probabilities = torch.stack((a * (1 - q), a * q, 1 - a), dim=1)
                self.assertEqual(tuple(probabilities.shape), (4, 3))
                self.assertTrue(bool(torch.isfinite(probabilities).all()))
                self.assertTrue(bool(((probabilities >= 0) & (probabilities <= 1)).all()))
                torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(4), atol=1e-7, rtol=0.)
                self.assertTrue(bool((residual.abs() <= 3.).all()))

    def test_initial_head_preserves_specialist_and_initial_acceptance(self):
        head = study.SelectiveHead(self.base).eval()
        result = study.predict(head, self.windows, self.features)
        probabilities = study.scores(result)
        np.testing.assert_allclose(probabilities[:, 0], self.base.predict_proba(self.features.numpy()),
                                   atol=2e-7, rtol=0.)
        np.testing.assert_allclose(probabilities[:, 1], .9, atol=2e-8, rtol=0.)
        np.testing.assert_array_equal(result[:, 2], 0.)

    def test_checkpoint_roundtrip_preserves_cpu_outputs_and_frozen_buffers(self):
        head = study.SelectiveHead(self.base).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        original_buffers = {name: value.clone() for name, value in head.named_buffers()}
        logits, acceptance, residual = head.outputs(self.windows, self.features)
        study.selective_loss(logits, acceptance, torch.tensor([0., 1., 0., 1.]),
                             residual, study.WARMUP + 1).mean().backward()
        self.assertGreater(float(head.acceptance[-1].weight.grad.abs().sum()), 0.)
        optimizer.step()
        for name, value in head.named_buffers():
            torch.testing.assert_close(value, original_buffers[name], atol=0., rtol=0.)
        expected = study.predict(head, self.windows, self.features)
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "synthetic-head.npz"
            np.savez_compressed(filename, **{name: value.detach().cpu().numpy()
                                            for name, value in head.state_dict().items()})
            restored = study.load_head(filename, "cpu")
            self.assertFalse(restored.training)
            self.assertEqual(set(restored.state_dict()), set(head.state_dict()))
            np.testing.assert_array_equal(study.predict(restored, self.windows, self.features), expected)

    def test_cost_sensitive_wrapper_checkpoint_and_score_roundtrip(self):
        head = study.CostSensitiveHead(self.base).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        logits, residual = head.outputs(self.windows, self.features)
        truth = torch.tensor([0., 1., 0., 1.])
        study.cost_sensitive_loss(logits, truth, residual).mean().backward()
        self.assertGreater(float(head.correction[-1].weight.grad.abs().sum()), 0.)
        optimizer.step()
        expected = study.predict(head, self.windows, self.features)
        self.assertEqual(expected.shape, (4, 4))
        transformed = study.scores(expected)
        raw = 1 / (1 + np.exp(-np.clip(expected[:, :2].astype(np.float64), -35., 35.)))
        np.testing.assert_array_equal(transformed[:, 1], raw.sum(axis=1))
        np.testing.assert_array_equal(transformed[:, 0], raw[:, 1] / raw.sum(axis=1))
        self.assertTrue(np.all(raw.sum(axis=1) < 1.))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic-cost-head.npz"
            np.savez_compressed(path, **{name: value.detach().cpu().numpy()
                                        for name, value in head.state_dict().items()})
            restored = study.load_head(path, "cpu")
            self.assertIsInstance(restored, study.CostSensitiveHead)
            self.assertFalse(restored.training)
            np.testing.assert_array_equal(study.predict(restored, self.windows, self.features), expected)
            np.testing.assert_array_equal(study.scores(study.predict(restored, self.windows, self.features)), transformed)


if __name__ == "__main__":
    unittest.main()
