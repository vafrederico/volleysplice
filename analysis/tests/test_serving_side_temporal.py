"""Guard the data boundaries and source isolation of the new research head."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
    from analysis import serving_side_temporal as study
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Research PyTorch environment required")
class TemporalServingTests(unittest.TestCase):
    def setUp(self):
        study.configure_torch()
        torch.manual_seed(3407)

    def sequence(self):
        times = np.arange(0., 10., .5)
        tokens = np.broadcast_to(times[:, None, None], (len(times), 4, 960)).copy().astype(np.float32)
        return times, tokens

    def test_future_and_previous_frames_are_ordered(self):
        values = study.sample_window(*self.sequence(), 4., 10., [])
        np.testing.assert_array_equal(values[:, 0], np.arange(1., 7.01, .5))
        np.testing.assert_array_equal(values[:, 3840], 1.)

    def test_ignored_span_blocks_context_on_other_side(self):
        values = study.sample_window(*self.sequence(), 4., 10., [{"start": 4.75, "end": 5.25}])
        self.assertEqual(values[7, 0], 4.5)
        np.testing.assert_array_equal(values[8:, :3840], 0.)
        np.testing.assert_array_equal(values[8:, 3840], 0.)

    def test_past_ignore_and_video_edge_are_masked(self):
        values = study.sample_window(*self.sequence(), 2., 10., [{"start": .5, "end": 1.5}])
        np.testing.assert_array_equal(values[:5, 3840], 0.)
        self.assertEqual(values[5, 0], 1.5)
        self.assertEqual(values[6, 0], 2.)

    def test_anchor_in_ignore_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "excluded footage"):
            study.sample_window(*self.sequence(), 4., 10., [{"start": 3., "end": 5.}])

    def test_nearest_tie_uses_earlier_frame(self):
        values = study.sample_window(*self.sequence(), 4.25, 10., [])
        self.assertEqual(values[6, 0], 4.)
        self.assertEqual(values[6, 3842], -1.)

    def test_network_receives_gradient_from_both_directions(self):
        model = study.ServingSideTemporalNetwork().eval()
        values = torch.randn(2, 13, 3843, requires_grad=True)
        with torch.no_grad():
            values[..., 3840] = 1.
        model(values).sum().backward()
        self.assertGreater(float(values.grad[:, 0, :3840].abs().sum()), 0.)
        self.assertGreater(float(values.grad[:, -1, :3840].abs().sum()), 0.)

    def test_unavailable_pixels_cannot_influence_output(self):
        model = study.ServingSideTemporalNetwork().eval()
        raw = np.random.default_rng(1).normal(size=(2, 13, 3843)).astype(np.float32)
        raw[..., 3840] = 1.
        only = study.arm_values(raw, "anchor-only")
        changed = only.copy()
        changed[:, :6, :3840] = 123.
        changed[:, 7:, :3840] = -456.
        np.testing.assert_array_equal(study.predict(model, torch.from_numpy(only)),
                                      study.predict(model, torch.from_numpy(changed)))

    def test_fold_weights_never_read_held_labels(self):
        rows = [{"sourceGroup": group, "label": i % 2} for group in ("a", "b", "c") for i in range(4)]
        train, held = study.fold_indices(rows, ("a", "b"))
        self.assertEqual(train.tolist(), [8, 9, 10, 11])
        self.assertEqual(held.tolist(), list(range(8)))
        first = study.training_weights([rows[i] for i in train])
        for i in held:
            rows[i]["label"] = 1 - rows[i]["label"]
        np.testing.assert_array_equal(first, study.training_weights([rows[i] for i in train]))

    def test_checkpoint_selection_and_metrics(self):
        rows = [{"sourceGroup": group, "environment": "indoor", "label": label}
                for group in ("a", "b") for label in (0, 1)]
        scores = {e: np.asarray([.3, .7, .3, .7]) for e in study.EPOCHS}
        selected, _ = study.select_operating_point(rows, scores)
        self.assertEqual(selected["epoch"], 5)
        self.assertEqual(selected["threshold"], .5)
        self.assertEqual(selected["evaluation"]["sourceGroupMacroBalancedAccuracy"], 1.)
        with self.assertRaisesRegex(ValueError, "Invalid inner scores"):
            scores[5][0] = np.nan
            study.select_operating_point(rows, scores)

    def test_fit_reload_resume_and_corruption_detection(self):
        rows = [{"sourceGroup": group, "label": i % 2} for group in ("a", "b", "c") for i in range(4)]
        values = torch.randn(12, 13, 3843)
        values[..., 3840] = 1.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/"fit"
            first = study.fit(rows, values, ("a",), 3407, folder, "synthetic-contract", epochs=(1, 2))
            second = study.fit(rows, values, ("a",), 3407, folder, "synthetic-contract", epochs=(1, 2))
            np.testing.assert_array_equal(first[2], second[2])
            model = study.load_weights(folder/"weights-2.npz", "cpu")
            np.testing.assert_array_equal(first[2], study.predict(model, values[:4]))
            with (folder/"scores.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                study.fit(rows, values, ("a",), 3407, folder, "synthetic-contract", epochs=(1, 2))


if __name__ == "__main__":
    unittest.main()
