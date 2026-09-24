from __future__ import annotations

import hashlib
import importlib.util
import unittest
from dataclasses import replace

import numpy as np

from analysis import neural_event_weighting as weighted
from analysis import neural_expanded_development as old
from analysis.neural_development import Example, boundary_targets
from analysis.schema import Interval, labels_for_times, mask_for_times


def make_row(truth=(Interval(2., 3.), Interval(5., 9.)), ignored=(), tier="exact", length=64):
    times = np.arange(length, dtype=np.float64)/4 + .125
    values = np.zeros((length, 104), np.float32)
    values[:, 0] = np.arange(length, dtype=np.float32)/length
    targets = np.column_stack((labels_for_times(times, truth),
                               boundary_targets(times, [v.start for v in truth]),
                               boundary_targets(times, [v.end for v in truth])))
    example = Example("synthetic", "group", length/4, times, values, targets,
                      mask_for_times(times, ignored), truth, ignored, "grass")
    return old.exact_supervision(example) if tier == "exact" else old.auxiliary_supervision(example, tier, truth)


class EventWeightTests(unittest.TestCase):
    def test_equal_mass_each_event_preserves_total_and_negative_weights(self):
        row = make_row()
        original_mask, original_targets = row.mask.copy(), row.example.targets.copy()
        vector, diagnostics = weighted.live_event_weights(row)
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(diagnostics["positiveSupervisedTicks"], 20)
        self.assertEqual(diagnostics["eligibleEventCount"], 2)
        self.assertEqual([event["positiveSupervisedTicks"] for event in diagnostics["events"]], [4, 16])
        self.assertEqual([event["multiplier"] for event in diagnostics["events"]], [2.5, .625])
        self.assertEqual([event["weightedPositiveMass"] for event in diagnostics["events"]], [10., 10.])
        self.assertEqual(diagnostics["weightedPositiveMass"], 20.)
        self.assertTrue(np.all(vector[row.example.targets[:, 0] == 0] == 1))
        np.testing.assert_array_equal(row.mask, original_mask)
        np.testing.assert_array_equal(row.example.targets, original_targets)
        self.assertEqual(diagnostics["liveMultiplierSha256"], hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest())

    def test_ignored_split_retains_original_event_identity(self):
        row = make_row((Interval(1., 7.), Interval(9., 10.)), (Interval(3., 5.),))
        vector, diagnostics = weighted.live_event_weights(row)
        self.assertEqual(diagnostics["originalEventCount"], 2)
        self.assertEqual(diagnostics["eligibleEventCount"], 2)
        self.assertEqual([event["positiveSupervisedTicks"] for event in diagnostics["events"]], [16, 4])
        self.assertEqual(diagnostics["events"][0]["invalidTicksInsideEvent"], 8)
        self.assertTrue(np.all(vector[~row.example.valid] == 1))
        first_visible = ((row.example.times >= 1) & (row.example.times < 7) & row.example.valid)
        self.assertTrue(np.all(vector[first_visible] == .625))
        self.assertTrue(np.all(row.mask[~row.example.valid] == 0))

    def test_fully_censored_draft_is_not_a_negative_or_a_new_event(self):
        row = make_row((Interval(2., 3.), Interval(5., 9.)), tier="draft")
        vector, diagnostics = weighted.live_event_weights(row)
        short = (row.example.times >= 2) & (row.example.times < 3)
        self.assertEqual(diagnostics["zeroSupervisedEventCount"], 1)
        self.assertIsNone(diagnostics["events"][0]["multiplier"])
        self.assertEqual(diagnostics["eligibleEventCount"], 1)
        self.assertTrue(np.all(row.example.targets[short, 0] == 1))
        self.assertTrue(np.all(row.mask[short, 0] == 0))
        self.assertTrue(np.all(vector[short] == 1))

    def test_one_tick_draft_weight_is_uncapped(self):
        row = make_row((Interval(2., 4.25), Interval(8., 28.)), tier="draft", length=128)
        vector, diagnostics = weighted.live_event_weights(row)
        counts = [event["positiveSupervisedTicks"] for event in diagnostics["events"]]
        self.assertEqual(counts, [1, 72])
        self.assertEqual(diagnostics["maximumPositiveMultiplier"], 36.5)
        self.assertEqual(float(vector.max()), 36.5)
        self.assertEqual(diagnostics["events"][0]["weightedPositiveMass"], 36.5)

    def test_no_positive_supervision_has_identity_weights(self):
        for row in (make_row((), tier="exact"), make_row((Interval(2., 3.),), tier="draft"), make_row(tier="coverage")):
            with self.subTest(tier=row.tier):
                vector, diagnostics = weighted.live_event_weights(row)
                self.assertTrue(np.all(vector == 1))
                self.assertEqual(diagnostics["eligibleEventCount"], 0)
                self.assertEqual(diagnostics["weightedPositiveMass"], 0)
                self.assertIsNone(diagnostics["maximumPositiveMultiplier"])

    def test_touching_half_open_events_remain_distinct(self):
        row = make_row((Interval(2., 3.), Interval(3., 5.)))
        _, diagnostics = weighted.live_event_weights(row)
        self.assertEqual([event["positiveSupervisedTicks"] for event in diagnostics["events"]], [4, 8])
        self.assertEqual(diagnostics["eligibleEventCount"], 2)

    def test_overlapping_event_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            weighted.live_event_weights(make_row((Interval(2., 5.), Interval(4., 7.))))

    def test_positive_tick_without_event_is_rejected(self):
        row = make_row()
        row.example.targets[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "unique original event"):
            weighted.live_event_weights(row)

    def test_supervision_in_ignored_time_is_rejected(self):
        row = make_row(ignored=(Interval(4., 5.),))
        row.mask[~row.example.valid, 0] = 1
        with self.assertRaisesRegex(ValueError, "ignored/invalid"):
            weighted.live_event_weights(row)

    def test_nonbinary_live_targets_and_masks_are_rejected(self):
        for field in ("targets", "mask"):
            row = make_row()
            (row.example.targets if field == "targets" else row.mask)[0, 0] = .5
            with self.assertRaisesRegex(ValueError, "binary"):
                weighted.live_event_weights(row)


class ChunkCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [make_row((Interval(1., 8.), Interval(40., 49.), Interval(74., 75.)),
                              (Interval(34., 37.),), length=411)]
        self.rows.append(old.auxiliary_supervision(replace(self.rows[0].example, id="draft"), "draft"))
        self.mean = np.zeros(104, np.float32)
        self.scale = np.ones(104, np.float32)

    def test_original_fields_order_sampling_and_rng_remain_identical(self):
        state = np.random.get_state()
        baseline = old.make_chunks(self.rows, self.mean, self.scale, "tcn")
        chunks = weighted.make_weighted_chunks(self.rows, self.mean, self.scale, "tcn")
        self.assertEqual(len(chunks), len(baseline))
        for left, right in zip(baseline, chunks):
            for i in range(3):
                np.testing.assert_array_equal(left[i], right[i])
            self.assertEqual(left[3], right[3])
            self.assertEqual(len(right), 5)
        np.testing.assert_array_equal(old.sampling_weights(chunks), old.sampling_weights(baseline))
        for seed in (3407, 1729, 20260918):
            self.assertEqual(old.epoch_batches(chunks, np.random.default_rng(seed)),
                             old.epoch_batches(baseline, np.random.default_rng(seed)))
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_context_does_not_duplicate_event_mass(self):
        for row in self.rows:
            weights, _ = weighted.live_event_weights(row)
            chunks = weighted.make_weighted_chunks([row], self.mean, self.scale, "tcn")
            observed = np.zeros(len(weights), np.float64)
            for values, _, mask, _, live in chunks:
                indices = np.rint(values[:, 0]*len(weights)).astype(int)
                np.testing.assert_array_equal(live, weights[indices])
                observed[indices] += mask[:, 0]*live
            np.testing.assert_array_equal(observed, row.mask[:, 0]*weights)

    def test_uniform_control_is_identity_and_unknown_mode_fails(self):
        chunks = weighted.make_weighted_chunks(self.rows, self.mean, self.scale, "tcn", "uniform")
        self.assertTrue(all(np.all(chunk[4] == 1) for chunk in chunks))
        with self.assertRaisesRegex(ValueError, "unknown live weighting"):
            weighted.make_weighted_chunks(self.rows, self.mean, self.scale, "tcn", "best")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch absent")
class WeightedLossTests(unittest.TestCase):
    def test_uniform_loss_gradient_and_stochastic_calls_match_frozen(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(51)
        chunks = []
        for length in (4, 7, 4):
            x = rng.normal(size=(length, 104)).astype(np.float32)
            y = rng.uniform(size=(length, 4)).astype(np.float32)
            mask = rng.integers(0, 2, size=(length, 4)).astype(np.float32)
            chunks.append((x, y, mask, 1., np.ones(length, np.float32)))
        model = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(104, 4))
        gradients, losses, states = [], [], []
        for implementation, pool in ((old.batch_loss, [c[:4] for c in chunks]), (weighted.batch_loss, chunks)):
            torch.manual_seed(99)
            model.zero_grad()
            loss = implementation(model, pool, [0, 1, 2, 0], torch.tensor([2., 3., 4., 5.]), (1., .5, .5, .25), "cpu")
            loss.backward()
            losses.append(loss.detach())
            gradients.append([p.grad.clone() for p in model.parameters()])
            states.append(torch.random.get_rng_state())
        self.assertTrue(torch.equal(losses[0], losses[1]))
        self.assertTrue(torch.equal(states[0], states[1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*gradients)))

    def test_live_weight_changes_only_live_loss_with_unweighted_denominator(self):
        import torch
        torch.set_num_threads(1)
        row = make_row()
        weights, _ = weighted.live_event_weights(row)
        x = np.arange(len(weights)*4, dtype=np.float32).reshape(-1, 4)/100
        chunk = (x, row.example.targets, row.mask, 1., weights)
        model = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(4))
        pos_weight, head_weight = torch.tensor([2., 3., 4., 5.]), (1., .5, .5, .25)
        loss = weighted.batch_loss(model, [chunk], [0], pos_weight, head_weight, "cpu")
        logits = model(torch.from_numpy(x))[None]
        elements = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(row.example.targets)[None], pos_weight=pos_weight, reduction="none")
        multipliers = torch.ones_like(elements)
        multipliers[0, :, 0] = torch.from_numpy(weights)
        mask = torch.from_numpy(row.mask)[None]
        expected = ((elements*mask*multipliers).sum((0, 1))/mask.sum((0, 1)).clamp_min(1)*torch.tensor(head_weight)).sum()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        new_gradient = model.weight.grad.clone()
        model.zero_grad()
        old.batch_loss(model, [chunk[:4]], [0], pos_weight, head_weight, "cpu").backward()
        torch.testing.assert_close(new_gradient[1:], model.weight.grad[1:], rtol=0, atol=0)
        self.assertFalse(torch.equal(new_gradient[0], model.weight.grad[0]))

    def test_masked_tick_amplification_cannot_change_loss_or_gradient(self):
        import torch
        x = np.ones((4, 1), np.float32)
        targets, mask = np.ones((4, 4), np.float32), np.zeros((4, 4), np.float32)
        mask[0, 0] = 1
        model = torch.nn.Linear(1, 4)
        observed = []
        for weights in (np.ones(4, np.float32), np.array([1, 1000, 1000, 1000], np.float32)):
            model.zero_grad()
            loss = weighted.batch_loss(model, [(x, targets, mask, 1., weights)], [0], torch.ones(4), (1, 1, 1, 1), "cpu")
            loss.backward()
            observed.append((loss.detach(), model.weight.grad.clone()))
        self.assertTrue(torch.equal(observed[0][0], observed[1][0]))
        self.assertTrue(torch.equal(observed[0][1], observed[1][1]))


if __name__ == "__main__":
    unittest.main()
