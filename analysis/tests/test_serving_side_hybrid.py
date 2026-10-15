"""Source-isolation, residual fusion, and nested-selection regression tests."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import tempfile
import unittest
import numpy as np

try:
    import torch
    from analysis import serving_side_hybrid as study
    from analysis.serving_side_v2 import fit_logistic
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Research PyTorch environment required")
class HybridTests(unittest.TestCase):
    def setUp(self):
        study.prior.configure_torch()
        torch.manual_seed(7)
        self.rows = [{"rowIndex": i, "sourceGroup": f"group-{i//6}", "label": i%2,
                      "environment": "synthetic"} for i in range(18)]
        self.features = np.random.default_rng(7).normal(size=(18, 237))
        self.windows = torch.randn(18, 13, 3843)
        self.windows[..., 3840] = 1.

    def base(self):
        return fit_logistic(self.features, np.asarray([r["label"] for r in self.rows]), l2=.1)

    def test_initial_hybrid_exactly_starts_at_fitted_specialist(self):
        base = self.base()
        head = study.HybridHead(base).eval()
        b, c = study.components(head, self.windows, torch.tensor(self.features, dtype=torch.float32))
        np.testing.assert_array_equal(c, 0.)
        np.testing.assert_allclose(study.probability(b,c,1.), base.predict_proba(self.features), atol=2e-7)

    def test_held_values_and_labels_never_fit_scaler_or_linear_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = study.fit_base(Path(tmp)/"first", self.rows, self.features, ("group-0",), "test")
            changed = [dict(r) for r in self.rows]
            values = self.features.copy(); values[:6] = 10000
            for r in changed[:6]:
                r["label"] = 1-r["label"]
            second = study.fit_base(Path(tmp)/"second", changed, values, ("group-0",), "test")
            self.assertEqual(first.to_dict(), second.to_dict())
            np.testing.assert_allclose(first.mean, self.features[6:].mean(axis=0), atol=0)

    def test_both_temporal_directions_and_specialist_features_reach_correction(self):
        head = study.HybridHead(self.base()).eval()
        torch.nn.init.normal_(head.correction[-1].weight, std=.01)
        windows = self.windows[:2].clone().requires_grad_()
        features = torch.tensor(self.features[:2], dtype=torch.float32, requires_grad=True)
        _, residual = head.components(windows, features)
        residual.sum().backward()
        self.assertGreater(float(windows.grad[:,0,:3840].abs().sum()), 0.)
        self.assertGreater(float(windows.grad[:,-1,:3840].abs().sum()), 0.)
        self.assertGreater(float(features.grad.abs().sum()), 0.)

    def test_specialist_only_is_invariant_to_embeddings(self):
        head = study.HybridHead(self.base(), "specialist-only").eval()
        torch.nn.init.normal_(head.correction[-1].weight, std=.01)
        fg = torch.tensor(self.features, dtype=torch.float32)
        first = study.components(head, self.windows, fg)
        second = study.components(head, self.windows*4., fg)
        np.testing.assert_array_equal(first, second)

    def test_masked_embeddings_cannot_influence_hybrid(self):
        head = study.HybridHead(self.base()).eval()
        torch.nn.init.normal_(head.correction[-1].weight, std=.01)
        windows = self.windows.clone(); windows[:,0,3840] = 0
        changed = windows.clone(); changed[:,0,:3840] = 500
        fg = torch.tensor(self.features, dtype=torch.float32)
        np.testing.assert_array_equal(study.components(head, windows, fg), study.components(head, changed, fg))

    def test_selection_prefers_zero_correction_on_exact_ties(self):
        y = np.array([r["label"] for r in self.rows])
        scores = {e: (np.where(y, 2., -2.), np.zeros(len(y))) for e in study.prior.EPOCHS}
        chosen = study.select(self.rows, scores)
        self.assertEqual((chosen["epoch"],chosen["multiplier"],chosen["threshold"]), (0,0.,.5))
        full = study.select(self.rows, scores, (1.,))
        self.assertEqual((full["epoch"],full["threshold"]), (5,.5))

    def test_selector_agrees_with_brute_force_metrics(self):
        rng = np.random.default_rng(45)
        for _ in range(3):
            scores = {e: (rng.normal(size=18), rng.normal(size=18)) for e in study.prior.EPOCHS}
            chosen = study.select(self.rows, scores)
            candidates = []
            for a in study.MULTIPLIERS:
                for e in study.prior.EPOCHS[:1] if a == 0 else study.prior.EPOCHS:
                    p = study.probability(*scores[e], a)
                    for t in study.prior.THRESHOLDS:
                        ev = study.prior.evaluate(self.rows, p >= t)
                        rank = (ev['sourceGroupMacroBalancedAccuracy'], ev['pooled']['balancedAccuracy'],
                                ev['pooled']['macroF1'], ev['worstSourceGroupBalancedAccuracy'],
                                -a, -(e if a else 0), -abs(t-.5), -t)
                        candidates.append((rank, e if a else 0, a, t))
            best = max(candidates)
            self.assertEqual((chosen['epoch'],chosen['multiplier'],chosen['threshold']), best[1:])

    def test_fit_checkpoint_reload_resume_and_corruption(self):
        fg = torch.tensor(self.features, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = dict(root=root, rows=self.rows, raw=self.windows, features=self.features, features_gpu=fg,
                          excluded=("group-0",), seed=7, arm="hybrid", contract="synthetic", epochs=(1,2))
            study.fit(**kwargs)
            study.fit(**kwargs)
            folder = root/"fits"/"hybrid"/"7"/"group-0"
            head = study.load_head(folder/"weights-2.npz", "hybrid", "cpu")
            b,c = study.components(head, self.windows[:6],fg[:6])
            with np.load(folder/"scores.npz") as data:
                np.testing.assert_array_equal(b, data['2-held-base'])
                np.testing.assert_array_equal(c, data['2-held-residual'])
            with (folder/"scores.npz").open('ab') as out:
                out.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                study.fit(**kwargs)


if __name__ == '__main__':
    unittest.main()
