from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from analysis import neural_keep_rescue as rescue
from analysis.crop_evaluation import pad_and_merge_intervals, subtract_intervals
from analysis.neural_evaluation import evaluate_predictions
from analysis.schema import Interval, mask_for_times


def recording(duration=40., ignored=()):
    times = np.arange(round(duration * 4), dtype=np.float64) / 4 + .125
    # No label, target, feature or architecture field exists for the decoder to
    # read. Both four-head model families use this same probability interface.
    return SimpleNamespace(times=times, duration=duration, ignored=tuple(ignored),
                           valid=mask_for_times(times, ignored))


def scores_for(e, keep=(), live=()):
    result = np.zeros((len(e.times), 4), np.float32)
    for channel, ranges in ((0, live), (3, keep)):
        for start, end in ranges:
            result[(e.times >= start) & (e.times < end), channel] = 1
    result[~e.valid] = 0
    return result


SETTINGS = {'smoothing': .5, 'enter': .5, 'minimum': .25, 'boundary': False}


class KeepRescueTests(unittest.TestCase):
    def proposals(self, e, scores, threshold=.5, smoothing=.5):
        return rescue.keep_core_proposals(e.times, scores[:, 3], e.valid, e.duration, threshold=threshold,
                                          smoothing_seconds=smoothing, ignored_intervals=e.ignored)

    def test_noop_is_exact_baseline_and_does_not_mutate_inputs_or_settings(self):
        e = recording(ignored=(Interval(17., 18.),))
        scores = scores_for(e, keep=((6., 12.),), live=((2., 4.), (22., 26.)))
        scores[:, 1:3] = .7
        scores[~e.valid] = 0
        before = scores.copy()
        for boundary in (False, True):
            settings = {**SETTINGS, 'boundary': boundary}
            saved = deepcopy(settings)
            expected = rescue.baseline.decode(e, scores, settings)
            actual = rescue.decode_with_keep_rescue(e, scores, settings, keep_threshold=None)
            self.assertEqual(actual, expected)
            self.assertEqual([i.to_dict() for i in actual], [i.to_dict() for i in expected])
            self.assertEqual(settings, saved)
        np.testing.assert_array_equal(scores, before)
        self.assertEqual(rescue.KEEP_RESCUE_OPTIONS, (None, .35, .5, .65, .8))

    def test_smoothing_threshold_equality_uses_frozen_sample_edge_convention(self):
        e = recording()
        scores = scores_for(e, keep=((6., 12.),))
        # Two-sample mean is left-padded. Half-height boundary ticks qualify at
        # .5, producing keep[6,12.25], then core[8,10.25].
        self.assertEqual(self.proposals(e, scores), (Interval(8., 10.25),))
        # At .65 the two half-height edge ticks are absent.
        self.assertEqual(self.proposals(e, scores, .65), (Interval(8.25, 10.),))
        self.assertEqual(self.proposals(e, scores, .5, 1.), (Interval(8., 10.25),))

    def test_eroded_predicted_duration_is_strictly_positive_and_at_most_three(self):
        e = recording()
        for end, expected in ((9.75, ()), (10., (Interval(8., 8.25),)),
                              (12.75, (Interval(8., 11.),)), (13., ())):
            with self.subTest(end=end):
                self.assertEqual(self.proposals(e, scores_for(e, keep=((6., end),))), expected)

    def test_frame_quantized_nominal4hz_times_keep_actual_endpoint_convention(self):
        e = recording()
        # Existing feature extraction emits selected frame_index/fps, rather
        # than replacing timestamps with an idealized quarter-second grid.
        e.times = np.rint(np.arange(len(e.times), dtype=np.float64) / 4 * 30) / 30
        self.assertFalse(np.allclose(np.diff(e.times), .25))
        scores = np.zeros((len(e.times), 4), np.float32)
        scores[25:49, 3] = 1.
        proposals = self.proposals(e, scores)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].start, float(e.times[25]) - .125 + 2.)
        self.assertEqual(proposals[0].end, float(e.times[49]) + .125 - 2.)

    def test_video_and_valid_segment_edge_components_are_discarded_before_erosion(self):
        e = recording(ignored=(Interval(18., 20.),))
        for pulse in ((0., 6.), (34., 40.), (12., 18.), (20., 26.)):
            with self.subTest(pulse=pulse):
                self.assertEqual(self.proposals(e, scores_for(e, keep=(pulse,))), ())
        self.assertEqual(self.proposals(e, scores_for(e, keep=((22., 28.),))), (Interval(24., 26.25),))

    def test_exact_ignored_subtick_hole_and_mere_touch_are_rejected(self):
        for ignored in (Interval(7.05, 7.1), Interval(12.25, 12.3), Interval(5.95, 6.)):
            e = recording(ignored=(ignored,))
            # All these holes/contact ranges can be missed by tick-center
            # validity, but continuous component support must still be rejected.
            self.assertTrue(e.valid.all())
            self.assertEqual(self.proposals(e, scores_for(e, keep=((6., 12.),))), ())
        e = recording(ignored=(Interval(12.26, 12.3),))
        self.assertEqual(self.proposals(e, scores_for(e, keep=((6., 12.),))), (Interval(8., 10.25),))

    def test_smoothing_does_not_use_scores_from_invalid_segments(self):
        e = recording(ignored=(Interval(18., 20.),))
        scores = scores_for(e, keep=((10., 15.), (23., 28.)))
        original = self.proposals(e, scores, smoothing=1.)
        scores[~e.valid, 3] = 1.
        self.assertEqual(self.proposals(e, scores, smoothing=1.), original)
        all_invalid = recording()
        all_invalid.valid[:] = False
        self.assertEqual(self.proposals(all_invalid, scores_for(all_invalid, keep=((6., 12.),))), ())

    def test_enabled_union_retains_positive_gaps_and_never_removes_baseline_time(self):
        e = recording()
        scores = scores_for(e, keep=((6., 12.),), live=((4., 6.),))
        original = rescue.decode_with_keep_rescue(e, scores, SETTINGS)
        changed = rescue.decode_with_keep_rescue(e, scores, SETTINGS, keep_threshold=.5)
        self.assertEqual(original, [Interval(4., 6.25)])
        self.assertEqual(changed, [Interval(4., 6.25), Interval(8., 10.25)])
        self.assertEqual(subtract_intervals(original, changed), ())
        # Only the canonical export operation fills this strictly<3s gap.
        self.assertEqual(pad_and_merge_intervals(changed, e.duration, 0., 3.), (Interval(4., 10.25),))
        for live in (((4., 8.),), ((9., 12.),)):
            overlapping = scores_for(e, keep=((6., 12.),), live=live)
            base = rescue.baseline.decode(e, overlapping, SETTINGS)
            augmented = rescue.decode_with_keep_rescue(e, overlapping, SETTINGS, keep_threshold=.5)
            self.assertEqual(len(augmented), 1)
            self.assertEqual(subtract_intervals(base, augmented), ())

    def test_no_proposals_or_wholly_covered_proposals_preserve_baseline(self):
        e = recording()
        for keep in ((), ((6., 12.),)):
            scores = scores_for(e, keep=keep, live=((3., 15.),))
            self.assertEqual(rescue.decode_with_keep_rescue(e, scores, SETTINGS, keep_threshold=.5),
                             rescue.baseline.decode(e, scores, SETTINGS))

    def test_same_core_is_evaluated_at_all_padding_cases_without_double_padding(self):
        e = recording()
        scores = scores_for(e, keep=((6., 12.),))
        cores = rescue.decode_with_keep_rescue(e, scores, SETTINGS, keep_threshold=.5)
        self.assertEqual(cores, [Interval(8., 10.25)])
        row = {'id': 'synthetic', 'sourceGroup': 'synthetic', 'durationSeconds': e.duration,
               'rallies': [Interval(8., 10.25)], 'ignoredIntervals': e.ignored, 'predictions': cores}
        evaluation = evaluate_predictions([row])
        self.assertEqual([p['paddedModelExportSeconds'] for p in evaluation['padding']], [2.25, 4.25, 6.25, 8.25])
        self.assertEqual(evaluation['primary']['paddedModelExportSeconds'], 6.25)
        self.assertEqual(cores, [Interval(8., 10.25)])

    def test_false_rescue_can_hurt_event_f1_and_precision_despite_monotonic_recall(self):
        e = recording()
        scores = scores_for(e, keep=((6., 12.),), live=((4., 6.),))
        base = rescue.decode_with_keep_rescue(e, scores, SETTINGS)
        augmented = rescue.decode_with_keep_rescue(e, scores, SETTINGS, keep_threshold=.5)
        common = {'id': 'synthetic', 'sourceGroup': 'synthetic', 'durationSeconds': e.duration,
                  'rallies': base, 'ignoredIntervals': e.ignored}
        old, new = [evaluate_predictions([{**common, 'predictions': intervals}]) for intervals in (base, augmented)]
        self.assertLess(new['guardrails']['eventF1'], old['guardrails']['eventF1'])
        self.assertLess(new['primary']['P_pad'], old['primary']['P_pad'])
        for before, after in zip(old['padding'], new['padding']):
            self.assertGreaterEqual(after['R_core'], before['R_core'])

    def test_joined_keep_coverage_is_not_an_inverse_of_original_rally_boundaries(self):
        e = recording()
        original = (Interval(8., 8.25), Interval(10., 10.25))
        keep = pad_and_merge_intervals(original, e.duration, 2., 3.)
        self.assertEqual(keep, (Interval(6., 12.25),))
        scores = scores_for(e, keep=tuple((i.start, i.end) for i in keep))
        proposals = self.proposals(e, scores)
        self.assertEqual(len(proposals), 1)
        self.assertNotEqual(proposals, original)
        self.assertLess(proposals[0].start, 9.)
        self.assertGreater(proposals[0].end, 9.)  # Includes the unknown original dead gap.

    def test_invalid_alignment_grid_threshold_and_scores_are_rejected(self):
        e = recording()
        scores = scores_for(e, keep=((6., 12.),))
        with self.assertRaisesRegex(ValueError, 'aligned'):
            rescue.decode_with_keep_rescue(e, scores[:, :3], SETTINGS)
        changed = scores.copy()
        changed[0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            rescue.decode_with_keep_rescue(e, changed, SETTINGS, keep_threshold=.5)
        for threshold in (True, 0., 1.1, float('nan')):
            with self.assertRaisesRegex(ValueError, 'threshold'):
                self.proposals(e, scores, threshold)
        with self.assertRaisesRegex(ValueError, 'smoothing'):
            self.proposals(e, scores, smoothing=.75)
        e.times[10] += .15
        with self.assertRaisesRegex(ValueError, '4Hz'):
            self.proposals(e, scores)


if __name__ == '__main__':
    unittest.main()
