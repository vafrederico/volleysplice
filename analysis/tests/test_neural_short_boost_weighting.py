from __future__ import annotations

import hashlib
import importlib.util
import unittest
from dataclasses import replace

import numpy as np

from analysis import neural_event_weighting as previous_weighting
from analysis import neural_expanded_development as frozen
from analysis import neural_short_boost_weighting as weighted
from analysis.schema import Interval
from analysis.tests.test_neural_event_weighting import make_row


class ShortBoostWeightsTests(unittest.TestCase):
    def test_fixed_short_boost_and_global_control_match_positive_mass_only(self):
        row = make_row()  # Four short positive ticks and16 long positive ticks.
        mask, targets = row.mask.copy(), row.example.targets.copy()
        baseline, b = weighted.live_event_weights(row, "baseline")
        short, s = weighted.live_event_weights(row, "short_boost")
        global_control, g = weighted.live_event_weights(row, "global_control")
        eligible = (row.mask[:, 0] > 0) & (row.example.targets[:, 0] > 0)
        self.assertTrue(np.all(baseline == 1))
        self.assertEqual((s["positiveSupervisedTicks"], s["shortPositiveSupervisedTicks"]), (20, 4))
        self.assertEqual([event["multiplier"] for event in s["events"]], [2., 1.])
        self.assertEqual(s["weightedPositiveMass"], 24.)
        self.assertEqual(b["weightedPositiveMass"], 20.)
        self.assertEqual(g["globalPositiveMultiplierFloat32"], float(np.float32(1.2)))
        self.assertTrue(np.all(global_control[eligible] == np.float32(1.2)))
        self.assertLessEqual(abs(g["weightedPositiveMass"] - s["weightedPositiveMass"]), g["positiveMassAbsoluteTolerance"])
        for vector in (baseline, short, global_control):
            self.assertEqual(vector.dtype, np.float32)
            self.assertTrue(np.all(vector[~eligible] == 1))
            self.assertTrue(np.all((vector >= 1) & (vector <= 2)))
        self.assertEqual(s["liveMultiplierSha256"], hashlib.sha256(short.astype("<f4").tobytes()).hexdigest())
        np.testing.assert_array_equal(row.mask, mask)
        np.testing.assert_array_equal(row.example.targets, targets)

    def test_three_seconds_inclusive_uses_original_duration_not_surviving_pieces(self):
        truth = (Interval(1., 7.), Interval(9., 12.), Interval(14., 17.00000001))
        row = make_row(truth, (Interval(2., 6.), Interval(10., 11.)), length=80)
        vector, audit = weighted.live_event_weights(row, "short_boost")
        self.assertEqual([r["isShortOriginalEvent"] for r in audit["events"]], [False, True, False])
        self.assertEqual([r["multiplier"] for r in audit["events"]], [1., 2., 1.])
        self.assertEqual([r["positiveSupervisedTicks"] for r in audit["events"]], [8, 8, 12])
        self.assertEqual(audit["shortOriginalEventCount"], 1)
        self.assertTrue(np.all(vector[~row.example.valid] == 1))

    def test_one_surviving_draft_tick_is_only_doubled_and_censored_events_remain_unknown(self):
        row = make_row((Interval(2., 4.25), Interval(8., 28.), Interval(29., 30.)), tier="draft", length=128)
        vector, audit = weighted.live_event_weights(row, "short_boost")
        self.assertEqual([r["positiveSupervisedTicks"] for r in audit["events"]], [1, 72, 0])
        self.assertEqual([r["multiplier"] for r in audit["events"]], [2., 1., None])
        self.assertEqual(audit["eligibleShortEventCount"], 1)
        self.assertEqual(audit["zeroSupervisedShortEventCount"], 1)
        self.assertEqual(audit["weightedPositiveMass"], 74.)
        self.assertEqual(vector.max(), 2.)
        unknown = row.mask[:, 0] == 0
        self.assertTrue(np.all(vector[unknown] == 1))
        censored = (row.example.times >= 29) & (row.example.times < 30)
        self.assertTrue(np.all(row.example.targets[censored, 0] == 1))
        self.assertTrue(np.all(row.mask[censored, 0] == 0))
        _, control = weighted.live_event_weights(row, "global_control")
        self.assertLessEqual(abs(control["weightedPositiveMass"]-74), control["positiveMassAbsoluteTolerance"])

    def test_zero_positives_and_no_short_events_have_identity_in_all_arms(self):
        rows = (make_row((), tier="exact"), make_row((Interval(2., 3.),), tier="draft"),
                make_row(tier="coverage"), make_row((Interval(2., 8.),)))
        for row in rows:
            for mode in weighted.WEIGHTING_MODES:
                with self.subTest(tier=row.tier, mode=mode):
                    vector, audit = weighted.live_event_weights(row, mode)
                    self.assertTrue(np.all(vector == 1))
                    self.assertEqual(audit["shortPositiveSupervisedTicks"], 0)
                    self.assertEqual(audit["globalPositiveMultiplierFloat32"], 1.)

    def test_all_positive_events_short_makes_control_identical_to_short_arm(self):
        row = make_row((Interval(2., 3.), Interval(5., 8.)))
        short, audit = weighted.live_event_weights(row, "short_boost")
        control, _ = weighted.live_event_weights(row, "global_control")
        np.testing.assert_array_equal(short, control)
        self.assertEqual(audit["shortPositiveSupervisedTicks"], audit["positiveSupervisedTicks"])

    def test_frozen_event_validator_and_mode_guards_remain_active(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            weighted.live_event_weights(make_row((Interval(2., 5.), Interval(4., 7.))))
        row = make_row()
        row.example.targets[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "unique original event"):
            weighted.live_event_weights(row)
        row = make_row(ignored=(Interval(4., 5.),))
        row.mask[~row.example.valid, 0] = 1
        with self.assertRaisesRegex(ValueError, "ignored/invalid"):
            weighted.live_event_weights(row)
        with self.assertRaisesRegex(ValueError, "unknown"):
            weighted.live_event_weights(make_row(), "tuned")


class ShortBoostChunkTests(unittest.TestCase):
    def setUp(self):
        self.rows = [make_row((Interval(1., 8.), Interval(40., 42.), Interval(74., 75.)),
                              (Interval(34., 37.),), length=611)]
        self.rows.append(frozen.auxiliary_supervision(replace(self.rows[0].example, id="draft"), "draft"))
        self.mean, self.scale = np.zeros(104, np.float32), np.ones(104, np.float32)
        self.scale[1:] = 2
        self.mean[1:] = -3
        self.dino_rows = []
        for row in self.rows:
            tokens = np.random.default_rng(37).normal(100, 3, (len(row.example.times), 3840)).astype(np.float32)
            self.dino_rows.append(replace(row, example=replace(row.example, values=np.concatenate((row.example.values, tokens), axis=1))))

    def test_all_modes_preserve_first_four_fields_and_all_seed_sampling_for_both_architectures(self):
        for kind, rows in (("tcn", self.rows), ("dino_tcn", self.dino_rows)):
            baseline = frozen.make_chunks(rows, self.mean, self.scale, kind)
            for mode in weighted.WEIGHTING_MODES:
                with self.subTest(kind=kind, mode=mode):
                    chunks = weighted.make_weighted_chunks(rows, self.mean, self.scale, kind, mode)
                    self.assertEqual(len(chunks), len(baseline))
                    for old, new in zip(baseline, chunks):
                        for i in range(3):
                            np.testing.assert_array_equal(old[i], new[i])
                        self.assertEqual(old[3], new[3])
                        self.assertEqual(len(new), 5)
                    np.testing.assert_array_equal(frozen.sampling_weights(chunks), frozen.sampling_weights(baseline))
                    for seed in (3407, 1729, 20260918):
                        self.assertEqual(frozen.epoch_batches(chunks, np.random.default_rng(seed)),
                                         frozen.epoch_batches(baseline, np.random.default_rng(seed)))

    def test_dino_retains_raw_tokens_and_same_real_halos_as_compact(self):
        av = weighted.make_weighted_chunks(self.rows, self.mean, self.scale, "tcn")
        dino = weighted.make_weighted_chunks(self.dino_rows, self.mean, self.scale, "dino_tcn")
        self.assertTrue(any(len(c[0]) == 252 for c in dino))
        self.assertEqual([len(c[0]) for c in av], [len(c[0]) for c in dino])
        for compact, fusion in zip(av, dino):
            np.testing.assert_array_equal(compact[0], fusion[0][:, :104])
            for field in (1, 2, 4):
                np.testing.assert_array_equal(compact[field], fusion[field])
            self.assertEqual(compact[3], fusion[3])
            self.assertEqual(fusion[0].shape[1], 3944)
            indices = np.rint(fusion[0][:, 0]*611).astype(int)
            np.testing.assert_array_equal(fusion[0][:, 104:], self.dino_rows[0].example.values[indices, 104:])
            self.assertGreater(float(fusion[0][:, 104:].min()), 10)  # Tokens were not clipped/scaled.
            self.assertTrue(np.all(self.rows[0].example.valid[indices]))
            self.assertTrue(np.all(np.diff(indices) == 1))

    def test_core_masks_cover_each_supervised_tick_once_without_duplicating_boost_mass(self):
        for kind, rows in (("tcn", self.rows), ("dino_tcn", self.dino_rows)):
            for row in rows:
                for mode in weighted.WEIGHTING_MODES:
                    vector, _ = weighted.live_event_weights(row, mode)
                    observed = np.zeros((len(vector), 4), np.float32)
                    observed_weight = np.zeros(len(vector), np.float64)
                    chunks = weighted.make_weighted_chunks([row], self.mean, self.scale, kind, mode)
                    for values, targets, mask, _, live in chunks:
                        indices = np.rint(values[:, 0]*len(vector)).astype(int)
                        np.testing.assert_array_equal(live, vector[indices])
                        observed[indices] += mask
                        observed_weight[indices] += live * mask[:, 0]
                        np.testing.assert_array_equal(targets, row.example.targets[indices])
                    np.testing.assert_array_equal(observed, row.mask)
                    np.testing.assert_array_equal(observed_weight, vector*row.mask[:, 0])

    def test_wrong_representation_and_unknown_arms_fail(self):
        with self.assertRaisesRegex(ValueError, "representation"):
            weighted.make_weighted_chunks(self.rows, self.mean, self.scale, "dino_tcn")
        with self.assertRaisesRegex(ValueError, "architecture"):
            weighted.make_weighted_chunks(self.rows, self.mean, self.scale, "linear")
        with self.assertRaisesRegex(ValueError, "unknown"):
            weighted.make_weighted_chunks(self.rows, self.mean, self.scale, "tcn", "per_rally")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class ShortBoostLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def test_imported_loss_changes_only_live_gradient_and_keeps_original_denominators(self):
        import torch
        self.assertIs(weighted.batch_loss, previous_weighting.batch_loss)
        row = make_row()
        x = np.arange(len(row.example.times)*4, dtype=np.float32).reshape(-1, 4)/100
        model = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(4))
        positive = torch.tensor([2., 3., 4., 5.])
        heads = (1., .5, .5, .25)
        gradients = []
        for mode in weighted.WEIGHTING_MODES:
            vector, _ = weighted.live_event_weights(row, mode)
            chunk = (x, row.example.targets, row.mask, 1., vector)
            model.zero_grad()
            loss = weighted.batch_loss(model, [chunk], [0], positive, heads, "cpu")
            logits = model(torch.from_numpy(x))
            elements = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(row.example.targets), pos_weight=positive, reduction="none")
            multipliers = torch.ones_like(elements)
            multipliers[:, 0] = torch.from_numpy(vector)
            mask = torch.from_numpy(row.mask)
            expected = ((elements*mask*multipliers).sum(0)/mask.sum(0).clamp_min(1)*torch.tensor(heads)).sum()
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            loss.backward()
            gradients.append(model.weight.grad.clone())
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient[1:], gradients[0][1:], rtol=0, atol=0)
            self.assertFalse(torch.equal(gradient[0], gradients[0][0]))
        self.assertFalse(torch.equal(gradients[1][0], gradients[2][0]))

    def test_baseline_stochastic_loss_matches_frozen_and_other_arms_consume_same_rng(self):
        import torch
        rows = [make_row(length=173), make_row(length=233)]
        model = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(104, 4))
        mean, scale = np.zeros(104, np.float32), np.ones(104, np.float32)
        original = frozen.make_chunks(rows, mean, scale, "tcn")
        indexes = list(range(len(original)))
        outcomes = []
        for implementation, chunks in [(frozen.batch_loss, original), *[
                (weighted.batch_loss, weighted.make_weighted_chunks(rows, mean, scale, "tcn", mode))
                for mode in weighted.WEIGHTING_MODES]]:
            torch.manual_seed(71)
            model.zero_grad()
            loss = implementation(model, chunks, indexes, torch.tensor([2., 3., 4., 5.]), (1., .5, .5, .25), "cpu")
            loss.backward()
            outcomes.append((loss.detach(), [p.grad.clone() for p in model.parameters()], torch.get_rng_state().clone()))
        torch.testing.assert_close(outcomes[0][0], outcomes[1][0], rtol=0, atol=0)
        for left, right in zip(outcomes[0][1], outcomes[1][1]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertTrue(all(torch.equal(outcomes[0][2], row[2]) for row in outcomes[1:]))


if __name__ == "__main__":
    unittest.main()
