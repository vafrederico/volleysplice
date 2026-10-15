import unittest

import numpy as np

try:
    import torch
    from analysis import serving_side_precision_iterations as study
    from analysis.serving_side_v2 import LogisticModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Research PyTorch environment required")
class PrecisionIterationsTests(unittest.TestCase):
    def test_centered_ignores_invalid_frames_and_static_offset(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(2, 13, 3843)).astype(np.float32)
        x[..., 3840] = 1.; x[:, 0, 3840] = 0.
        other = x.copy(); other[:, 1:, :3840] += 8.; other[:, 0, :3840] = 1e5
        a = study.temporal_transform(x, "centered")
        b = study.temporal_transform(other, "centered")
        np.testing.assert_allclose(a[..., :3840], b[..., :3840], atol=4e-6)
        np.testing.assert_allclose(a[:, 1:, :3840].mean(axis=1), 0., atol=1e-6)
        np.testing.assert_array_equal(a[..., 3840:], x[..., 3840:])

    def test_differences_mask_pairs_and_preserve_time(self):
        x = np.zeros((1, 13, 3843), np.float32)
        x[..., :3840] = np.arange(13)[None, :, None]*3
        x[..., 3840] = 1.; x[:, 4, 3840] = 0.
        x[..., 3841] = np.arange(13)
        y = study.temporal_transform(x, "differences")
        np.testing.assert_array_equal(y[0, [0, 4, 5], :3840], 0.)
        np.testing.assert_array_equal(y[0, [1, 2, 3, 6], :3840], 3.)
        np.testing.assert_array_equal(y[..., 3841], x[..., 3841])
        np.testing.assert_array_equal(y[0, [0, 4, 5], 3840], 0.)

    def test_fixed_changes_ignore_static_appearance(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(3, 13, 3843)).astype(np.float32)
        x[..., 3840] = 1.
        shifted = x.copy(); shifted[..., :3840] += rng.normal(size=(3, 1, 3840)).astype(np.float32)
        np.testing.assert_allclose(study.projected_changes(x), study.projected_changes(shifted), atol=3e-6)
        x[..., 3840] = 0.
        np.testing.assert_array_equal(study.projected_changes(x), 0.)

    def test_feature_arms_have_distinct_registered_dimensions(self):
        features = {"ranks": np.full((2, 237), .5), "quality": np.ones((2, 12)),
                    "court": np.ones((2, 82)), "changes": np.ones((2, 64)),
                    "interactionColumns": np.arange(6)}
        dims = {study.CONTROL: 237, "01-quality-mean": 241, "02-quality-phases": 249,
                "03-raw-court": 331, "04-embedding-changes": 313, "05-quality-interactions": 273}
        for arm, expected in dims.items():
            self.assertEqual(study.design(features, arm).shape, (2, expected))
        np.testing.assert_array_equal(study.design(features, "05-quality-interactions")[:, -24:], 0.)

    def test_correctness_feature_dimensions_and_direction(self):
        p = np.array([[.8, .7, .9], [.2, .3, .1], [.8, .4, .9]])
        q = np.ones((3, 12))
        e = study.evidence(p, q)
        self.assertEqual(e.shape, (3, 19))
        self.assertEqual(study.evidence(p, q, True).shape, (3, 110))
        np.testing.assert_allclose(e[0, 1:7], e[1, 1:7])
        self.assertEqual(e[0, 6], 1.)
        self.assertEqual(e[2, 6], 0.)
        self.assertEqual(study.evidence(p[:0], q[:0], True).shape, (0, 110))

    def test_gate_teachers_exclude_target_and_outer_groups(self):
        from unittest.mock import patch
        groups = list("abcdefgh")
        rows = [{"label": i % 2, "sourceGroup": g} for g in groups for i in range(4)]
        calls = []
        def teacher(fast, reg, rows, ranks, excluded):
            calls.append(set(excluded))
            return np.full((len(rows), 3), .6)
        reg = {"contract": {"sourceGroups": groups}}
        features = {"ranks": np.zeros((32, 237))}
        with patch.object(study, "teacher", teacher):
            train, held, p, out, provenance = study.gate_data(None, reg, rows, features, ("a", "b"))
        self.assertEqual(len(train), 24)
        self.assertEqual(len(held), 8)
        for item, excluded in zip(provenance, calls[:-1], strict=True):
            self.assertEqual(excluded, {"a", "b", item["group"]})
            self.assertTrue(all(rows[i]["sourceGroup"] == item["group"] for i in item["rows"]))
        self.assertEqual(calls[-1], {"a", "b"})

    def test_quality_head_frozen_skip_and_positive_gradients(self):
        torch.set_num_threads(2)
        base = LogisticModel(np.zeros(237), np.zeros(237), np.ones(237), np.zeros(237), .7, .1)
        model = study.QualityHead(base, np.arange(12), np.ones(12))
        x = torch.zeros((2, 13, 3843)); x[..., 3840] = 1.
        f = torch.zeros((2, 249))
        logits, residual = model.outputs(x, f)
        self.assertEqual(tuple(logits.shape), (2, 2))
        np.testing.assert_allclose(logits.detach().numpy(), [[-.7-np.log(49), .7-np.log(49)]]*2, atol=1e-6)
        loss = study.cost_sensitive_loss(logits, torch.tensor([0., 1.]), residual).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.correction[-1].weight.grad).all())
        self.assertNotIn("base_weights", dict(model.named_parameters()))

    def test_partition_contract(self):
        groups = [str(i) for i in range(8)]
        partitions = study.teacher_partitions(groups)
        self.assertEqual(len(partitions), 93)
        self.assertEqual(len(set(partitions)), 93)
        self.assertIn((), partitions)

    def test_saved_quality_head_preserves_scaling_and_outputs(self):
        from pathlib import Path
        import tempfile
        base = LogisticModel(np.zeros(237), np.zeros(237), np.ones(237), np.zeros(237), .7, .1)
        model = study.QualityHead(base, np.arange(12), np.full(12, 2.)).eval()
        x = torch.randn((3, 13, 3843)); x[..., 3840] = 1.
        f = torch.randn((3, 249))
        with torch.no_grad():
            model.correction[-1].bias[:] = torch.tensor([.2, -.3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"weights.npz"
            np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})
            restored = study.load_head(path)
            np.testing.assert_array_equal(study.prior.predict(model, x, f), study.prior.predict(restored, x, f))


if __name__ == "__main__":
    unittest.main()
