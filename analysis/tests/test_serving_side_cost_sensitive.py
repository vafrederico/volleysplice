"""Small synthetic CPU checks for the fixed false-accept-cost objective."""
import math
import unittest

import numpy as np

try:
    import torch
    from analysis import serving_side_cost_sensitive as study
    from analysis.serving_side_v2 import LogisticModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Research PyTorch environment required")
class CostSensitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(19)
        self.weights = np.zeros(237)
        self.weights[:3] = [.7, -.5, .2]
        self.base = LogisticModel(np.zeros(237), np.zeros(237), np.ones(237), self.weights, .3, .1)
        self.windows = torch.randn(4, 13, 3843)
        self.windows[..., 3840] = 1.
        self.features = torch.randn(4, 237)

    def test_initial_outputs_match_exact_cost_transformation_of_base(self):
        model = study.CostSensitiveHead(self.base).eval()
        logits, residual = model.outputs(self.windows, self.features)
        base = self.features.numpy() @ self.weights + .3
        expected_logits = np.column_stack((-base, base)) - math.log(49.)
        np.testing.assert_allclose(logits.detach().numpy(), expected_logits, atol=4e-7)
        np.testing.assert_array_equal(residual.detach().numpy(), np.zeros((4, 2)))
        p = 1. / (1. + np.exp(-base))
        raw_near = p / (p + 49. * (1. - p))
        raw_far = (1. - p) / ((1. - p) + 49. * p)
        expected = np.column_stack((raw_near / (raw_near + raw_far), raw_near + raw_far))
        np.testing.assert_allclose(study.scores(logits.detach().numpy()), expected, atol=5e-8)

    def test_all_review_has_gradient_away_from_zero_acceptance(self):
        logits = torch.full((2, 2), -30., requires_grad=True)
        truth = torch.tensor([0., 1.])
        loss = study.cost_sensitive_loss(logits, truth, torch.zeros_like(logits))
        loss.sum().backward()
        self.assertLess(float(logits.grad[0, 0]), -.019)
        self.assertLess(float(logits.grad[1, 1]), -.019)
        self.assertGreater(float(logits.grad[0, 1]), 0.)
        self.assertGreater(float(logits.grad[1, 0]), 0.)
        improved = logits.detach().clone()
        improved[0, 0] += 1.
        improved[1, 1] += 1.
        self.assertTrue(bool(torch.all(study.cost_sensitive_loss(improved, truth, torch.zeros_like(improved)) < loss)))

    def test_false_accept_gradient_cost_is_forty_nine_to_one(self):
        logits = torch.zeros((1, 2), requires_grad=True)
        loss = study.cost_sensitive_loss(logits, torch.ones(1), torch.zeros_like(logits))
        loss.sum().backward()
        self.assertAlmostEqual(float(logits.grad[0, 0]), .49, places=6)
        self.assertAlmostEqual(float(logits.grad[0, 1]), -.01, places=6)

    def test_loss_matches_independent_scalar_formula(self):
        logits = torch.tensor([[-2., -4.], [-5., .2]], dtype=torch.float64)
        truth = torch.tensor([0., 1.], dtype=torch.float64)
        residual = torch.tensor([[1., -2.], [.5, .25]], dtype=torch.float64)
        actual = study.cost_sensitive_loss(logits, truth, residual).numpy()
        expected = []
        for values, label, correction in zip(logits.numpy(), truth.numpy(), residual.numpy()):
            total = 0.
            for value, target in zip(values, (1 - label, label)):
                probability = 1 / (1 + math.exp(-value))
                total += (1 if target else 49) * (-target * math.log(probability) - (1 - target) * math.log(1 - probability))
            expected.append(total / 50 + .001 * sum(correction ** 2) / 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    def test_bounded_logits_always_leave_nonnegative_review_mass(self):
        model = study.CostSensitiveHead(self.base).eval()
        with torch.no_grad():
            model.correction[-1].weight.fill_(100.)
            model.correction[-1].bias.fill_(100.)
        logits, residual = model.outputs(self.windows, self.features)
        self.assertTrue(bool(torch.all(residual.abs() <= 3.)))
        self.assertTrue(bool(torch.all(logits.sum(dim=1) <= 6 - 2 * math.log(49.) + 1e-6)))
        values = study.scores(logits.detach().numpy())
        self.assertTrue(np.all((values >= 0) & (values <= 1)))
        self.assertTrue(np.all(values[:, 1] < 1))

    def test_extreme_logits_and_gradients_are_finite(self):
        logits = torch.tensor([[-1000., 1000.], [1000., -1000.], [-1000., -1000.]], requires_grad=True)
        loss = study.cost_sensitive_loss(logits, torch.tensor([1., 0., 1.]), torch.zeros_like(logits))
        loss.sum().backward()
        self.assertTrue(bool(torch.isfinite(loss).all()))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        values = study.scores(logits.detach().numpy())
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all((values >= 0) & (values <= 1)))
        np.testing.assert_allclose(values[2], [.5, 2 / (1 + math.exp(35.))], atol=1e-20)

    def test_scores_reproduce_sigmoid_pair_and_reject_invalid_mass(self):
        values = np.asarray([[-3., -4.], [-1., -5.]])
        raw = 1 / (1 + np.exp(-values))
        output = study.scores(values)
        np.testing.assert_allclose(output[:, 1] * output[:, 0], raw[:, 1], rtol=1e-15)
        np.testing.assert_allclose(output[:, 1] * (1 - output[:, 0]), raw[:, 0], rtol=1e-15)
        for bad in (np.ones((1, 2)), np.asarray([[np.nan, 0.]]), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                study.scores(bad)

    def test_eval_other_rows_cannot_change_a_row_or_frozen_base_buffers(self):
        model = study.CostSensitiveHead(self.base).eval()
        with torch.no_grad():
            torch.nn.init.normal_(model.correction[-1].weight, std=.01)
        original = {name: value.clone() for name, value in model.named_buffers() if name.startswith("base_")}
        first = model(self.windows, self.features)[0].detach()
        windows = self.windows.clone()
        features = self.features.clone()
        windows[1:] = 1000.
        features[1:] = -1000.
        second = model(windows, features)[0].detach()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name, value in model.named_buffers():
            if name in original:
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertFalse(any(name.startswith("base_") for name, _ in model.named_parameters()))

    def test_specialist_only_and_masked_visual_inputs_are_invariant(self):
        for arm in ("specialist-only", "hybrid"):
            with self.subTest(arm=arm):
                model = study.CostSensitiveHead(self.base, arm).eval()
                with torch.no_grad():
                    torch.nn.init.normal_(model.correction[-1].weight, std=.01)
                windows = self.windows.clone()
                windows[:, 0, 3840] = 0.
                changed = windows.clone()
                changed[:, 0, :3840] = 999.
                if arm == "specialist-only":
                    changed[:] = 999.
                torch.testing.assert_close(model(windows, self.features), model(changed, self.features), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
