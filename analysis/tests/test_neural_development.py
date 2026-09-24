from __future__ import annotations
from analysis.private_ledger import private_value

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.neural_development import (
    Example,
    boundary_targets,
    choose_settings,
    fit_model,
    fit_scaler,
    load_examples,
    model_for,
    predict,
    segments,
    standardized,
    supervision_mask,
    training_chunks,
)
from analysis.schema import Interval, labels_for_times, mask_for_times


def example(identifier: str = "example", group: str = "group-a", length: int = 333) -> Example:
    times = np.arange(length, dtype=np.float64) / 4.0
    ignored = (Interval(25.0, 26.0),) if length > 104 else ()
    truth = (Interval(4.0, 8.0),)
    values = np.zeros((length, 104), dtype=np.float32)
    values[:, 0] = np.arange(length, dtype=np.float32) / length
    targets = np.stack((
        labels_for_times(times, truth),
        boundary_targets(times, [4.0]),
        boundary_targets(times, [8.0]),
    ), axis=1)
    return Example(identifier, group, length / 4.0, times, values, targets,
                   mask_for_times(times, ignored), truth, ignored, "indoor")


class NeuralDevelopmentDataTests(unittest.TestCase):
    def test_empty_or_entirely_ignored_training_cannot_fit_scaler(self) -> None:
        row = example()
        row.valid[:] = False
        for rows in ([], [row]):
            with self.subTest(count=len(rows)), self.assertRaisesRegex(ValueError, "valid training samples"):
                fit_scaler(rows)

    def test_scaler_uses_only_supplied_fitting_valid_rows_and_preserves_dino(self) -> None:
        fitting = example(length=3)
        fitting.values = np.zeros((3, 3944), dtype=np.float32)
        fitting.values[:, 0] = [0.0, 2.0, 1000.0]
        fitting.values[:, 104:] = 7.0
        fitting.valid = np.array([True, True, False])
        held = example("held", "held-group", length=3)
        held.values[:, 0] = 100.0
        mean, scale = fit_scaler([fitting])
        self.assertEqual(float(mean[0]), 1.0)
        self.assertEqual(float(scale[0]), 1.0)
        self.assertTrue(np.all(scale > 0))
        transformed = standardized(fitting, mean, scale, "dino_tcn")
        np.testing.assert_array_equal(transformed[:, 0], [-1.0, 1.0, 10.0])
        np.testing.assert_array_equal(transformed[:, 104:], fitting.values[:, 104:])
        self.assertEqual(standardized(held, mean, scale, "tcn").shape[1], 104)
        np.testing.assert_array_equal(mean[1:], np.zeros(103, dtype=np.float32))

    def test_boundary_masks_censor_ignored_halo_without_dropping_live_supervision(self) -> None:
        row = example()
        masks = supervision_mask(row)
        np.testing.assert_array_equal(masks[:, 0], row.valid)
        halo = (row.times >= 24.0) & (row.times <= 27.0)
        self.assertTrue(np.all(masks[halo, 1:] == 0))
        self.assertTrue(np.all(masks[~halo, 1:] == 1))
        # A boundary hidden inside ignored time must not supervise its pulse tail.
        hidden_boundary = boundary_targets(row.times, [25.5])
        self.assertEqual(float(np.sum(hidden_boundary * masks[:, 1])), 0.0)
        self.assertEqual(float(masks[96, 0]), 1.0)  # Live at 24 s is still valid.

    def test_chunk_masks_cover_each_valid_tick_once_without_ignored_inputs(self) -> None:
        row = example()
        chunks = training_chunks([row], np.zeros(104), np.ones(104), "tcn")
        observed = np.zeros((len(row.times), 3), dtype=np.float32)
        for values, targets, mask, weight in chunks:
            self.assertEqual(values.shape[0], targets.shape[0])
            self.assertEqual(mask.shape, targets.shape)
            self.assertGreater(weight, 0)
            ticks = np.rint(values[:, 0] * len(row.times)).astype(np.int64)
            self.assertTrue(np.all(row.valid[ticks]))
            np.testing.assert_array_equal(targets, row.targets[ticks])
            observed[ticks] += mask
        np.testing.assert_array_equal(observed, supervision_mask(row))

    def test_sampling_weights_give_equal_mass_to_groups_with_unequal_recordings(self) -> None:
        first = example("first", "a", 33)
        second = example("second", "a", 400)
        third = example("third", "b", 550)
        rows = [first, second, third]
        all_chunks = training_chunks(rows, np.zeros(104), np.ones(104), "tcn")
        counts = [len(training_chunks([row], np.zeros(104), np.ones(104), "tcn")) for row in rows]
        masses = []
        cursor = 0
        for count in counts:
            masses.append(sum(chunk[3] for chunk in all_chunks[cursor:cursor + count]))
            cursor += count
        self.assertAlmostEqual(masses[0], 0.5)
        self.assertAlmostEqual(masses[1], 0.5)
        self.assertAlmostEqual(masses[2], 1.0)

    def test_protected_and_nonconsenting_rows_fail_before_cache_access(self) -> None:
        cases = (
            ("train", private_value('source-group-008'), {"train": True}),
            ("test", "other", {"train": True}),
            ("train", "other", {"train": False}),
            ("validation", "other", {}),
        )
        for split, group, consent in cases:
            record = SimpleNamespace(id="refused", split=split, source_group=group, consent=consent)
            with self.subTest(split=split, group=group, consent=consent), \
                    patch("analysis.neural_development.load_manifest", return_value=SimpleNamespace(recordings=[record])), \
                    patch("analysis.neural_development.file_sha256") as cache_access, \
                    self.assertRaises(ValueError):
                load_examples(Path("unused.json"), with_dino=False)
            cache_access.assert_not_called()

    def test_checkpoint_ties_choose_earliest_epoch_and_no_boundary_refinement(self) -> None:
        predictions = {30: {}, 5: {}, 10: {}}
        with patch("analysis.neural_development.primary_score", return_value=0.8):
            epoch, settings, score = choose_settings([example()], predictions)
        self.assertEqual(epoch, 5)
        self.assertFalse(settings["boundary"])
        self.assertEqual(score, 0.8)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class NeuralDevelopmentInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        import torch

        torch.set_num_threads(cls.previous_threads)

    def test_training_chunk_interiors_match_inference_for_each_compact_model(self) -> None:
        import torch

        row = example()
        mean = np.zeros(104, dtype=np.float32)
        scale = np.ones(104, dtype=np.float32)
        for kind in ("linear", "mlp", "tcn"):
            with self.subTest(kind=kind), torch.inference_mode():
                torch.manual_seed(17)
                model = model_for(kind).eval()
                inferred = predict(model, row, mean, scale, kind, "cpu")
                np.testing.assert_array_equal(inferred[~row.valid], np.zeros((4, 3)))
                for inputs, _, mask, _ in training_chunks([row], mean, scale, kind):
                    local = torch.sigmoid(model(torch.from_numpy(inputs[None])))[0].numpy()
                    ticks = np.rint(inputs[:, 0] * len(row.times)).astype(np.int64)
                    selected = mask[:, 0].astype(bool)
                    np.testing.assert_allclose(local[selected], inferred[ticks[selected]], rtol=1e-5, atol=1e-6)

    def test_fit_refuses_shared_source_group_before_filesystem_or_model_access(self) -> None:
        train = [example("train", "same-match")]
        validation = [example("validation", "same-match")]
        with patch("analysis.neural_development.model_for") as build_model, \
                patch.object(Path, "exists") as access_files, \
                self.assertRaisesRegex(ValueError, "disjoint source groups"):
            fit_model(train, validation, "tcn", 7, (1,), Path("unused"), "cpu", "unused")
        build_model.assert_not_called()
        access_files.assert_not_called()

    def test_inference_cannot_use_ignored_values_or_the_next_segment(self) -> None:
        import torch

        row = example()
        model = model_for("tcn").eval()
        mean = np.zeros(104, dtype=np.float32)
        scale = np.ones(104, dtype=np.float32)
        initial = predict(model, row, mean, scale, "tcn", "cpu")
        row.values[100:] = 1000.0
        changed = predict(model, row, mean, scale, "tcn", "cpu")
        first_segment = segments(row.valid)[0]
        np.testing.assert_array_equal(initial[slice(*first_segment)], changed[slice(*first_segment)])


if __name__ == "__main__":
    unittest.main()
