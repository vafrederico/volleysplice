from __future__ import annotations

import itertools
import json
import unittest

import numpy as np

from analysis.neural_rally_identity_metrics import (
    _maximum_weight_pairs,
    _timestamp_matches,
    evaluate_rally_identities,
)


def record(predictions, *, truth=None, ignored=(), identity="a", group="g"):
    return {"id": identity, "sourceGroup": group, "durationSeconds": 100,
            "rallies": [[10, 20]] if truth is None else truth,
            "predictions": predictions, "ignoredIntervals": ignored}


def score(predictions, **kwargs):
    return evaluate_rally_identities([record(predictions, **kwargs)])["pooled"]


def brute_match(weights, eligible):
    """Enumerate every matching independently, including unmatched rows."""
    best = (0, 0.0)
    for choices in itertools.product(range(-1, weights.shape[1]), repeat=weights.shape[0]):
        assigned = [choice for choice in choices if choice >= 0]
        if len(set(assigned)) != len(assigned):
            continue
        if any(choice >= 0 and not eligible[row, choice] for row, choice in enumerate(choices)):
            continue
        candidate = (len(assigned), sum(weights[row, choice] for row, choice in enumerate(choices) if choice >= 0))
        best = max(best, candidate)
    return best


class RallyIdentityMetricsTests(unittest.TestCase):
    def test_perfect_touching_identities_are_not_unioned(self):
        result = score([[10, 15], [15, 20]], truth=[[10, 15], [15, 20]])
        self.assertEqual(result["predictedRallies"], 2)
        self.assertEqual(result["matchedRallies"], 2)
        self.assertEqual(result["eventF1"], 1)
        self.assertEqual(result["mergedPredictions"], 0)
        self.assertEqual(result["splitTrueRallies"], 0)

    def test_single_merged_prediction_cannot_match_two_rallies(self):
        result = score([[10, 30]], truth=[[10, 20], [20, 30]])
        self.assertEqual(result["matchedRallies"], 1)
        self.assertEqual(result["eventPrecision"], 1)
        self.assertEqual(result["eventRecall"], 0.5)
        self.assertEqual(result["mergedPredictions"], 1)
        self.assertEqual(result["completeMisses"], 0)
        self.assertEqual(result["startLocalization"]["0.25"]["recall"], 0.5)

    def test_split_predictions_have_extra_identity_false_positive(self):
        result = score([[10, 15], [15, 20]])
        self.assertEqual(result["matchedRallies"], 1)
        self.assertEqual(result["falsePositiveRallies"], 1)
        self.assertEqual(result["splitTrueRallies"], 1)
        self.assertEqual(result["eventPrecision"], 0.5)

    def test_overlapping_duplicate_predictions_stay_separate(self):
        result = score([[10, 20], [10, 20]])
        self.assertEqual(result["predictedRallies"], 2)
        self.assertEqual(result["eventF1"], 2 / 3)
        self.assertEqual(result["startLocalization"]["0.5"]["precision"], 0.5)

    def test_masked_prediction_fragments_retain_parent_identity(self):
        result = score([[10, 30]], truth=[[10, 18], [22, 30]], ignored=[[18, 22]])
        self.assertEqual(result["predictedRallies"], 1)
        self.assertEqual(result["matchedRallies"], 1)
        self.assertEqual(result["mergedPredictions"], 1)

    def test_censored_gold_not_split_into_easier_short_rallies(self):
        result = score([[10, 20]], ignored=[[14, 16]])
        self.assertEqual(result["originalTrueRallies"], 1)
        self.assertEqual(result["ignoredTouchedTrueRallies"], 1)
        self.assertEqual(result["trueRallies"], 0)
        self.assertEqual(result["entirelyMaskedPredictions"], 1)
        self.assertIsNone(result["eventF1"])

    def test_ignored_false_positive_not_counted(self):
        result = score([[10, 20], [50, 60]], ignored=[[45, 65]])
        self.assertEqual(result["originalPredictedRallies"], 2)
        self.assertEqual(result["predictedRallies"], 1)
        self.assertEqual(result["eventF1"], 1)

    def test_boundary_in_mask_does_not_become_synthetic_observed_start(self):
        result = score([[5, 20]], ignored=[[0, 10]])
        self.assertEqual(result["eventF1"], 1)
        self.assertEqual(result["startLocalization"]["2"]["predicted"], 0)
        self.assertEqual(result["matchedEventBoundaries"]["2"]["startCorrect"], 0)
        self.assertEqual(result["matchedStartError"]["count"], 0)

    def test_end_at_ignored_start_and_start_at_ignored_end_are_observed(self):
        result = score([[10, 20], [30, 40]], truth=[[10, 20], [30, 40]], ignored=[[20, 30]])
        self.assertEqual(result["endLocalization"]["0.25"]["matched"], 2)
        self.assertEqual(result["startLocalization"]["0.25"]["matched"], 2)

    def test_unconditional_boundary_recall_counts_missed_events(self):
        result = score([[10, 20]], truth=[[10, 20], [30, 40]])
        self.assertEqual(result["matchedStartError"]["maeSeconds"], 0)
        self.assertEqual(result["matchedEventBoundaries"]["0.25"]["startRecall"], 0.5)
        self.assertEqual(result["completeMisses"], 1)

    def test_start_timestamp_proxy_does_not_imply_correct_event_end(self):
        result = score([[10, 11]])
        self.assertEqual(result["eventF1"], 0)
        self.assertEqual(result["startLocalization"]["0.25"]["f1"], 1)
        self.assertEqual(result["matchedEventBoundaries"]["0.25"]["startRecall"], 0)

    def test_synthetic_permission_edge_not_claimed_as_observed_contact(self):
        result = score([{"start": 10, "end": 20, "startObserved": False, "endObserved": True}])
        self.assertEqual(result["eventF1"], 1)
        self.assertEqual(result["startLocalization"]["1"]["f1"], 1)
        self.assertEqual(result["observedStartLocalization"]["1"]["recall"], 0)
        self.assertEqual(result["observedStartLocalization"]["1"]["true"], 1)
        self.assertEqual(result["observedEndLocalization"]["1"]["f1"], 1)
        self.assertEqual(result["unobservedPredictionStarts"], 1)

    def test_missing_observation_flags_default_to_baseline_observed(self):
        result = score([[10, 20]])
        self.assertEqual(result["observedStartLocalization"], result["startLocalization"])
        self.assertEqual(result["observedEndLocalization"], result["endLocalization"])

    def test_boundary_tolerances_inclusive_and_report_both_ends(self):
        result = score([[10.5, 20.5]])
        self.assertEqual(result["startLocalization"]["0.25"]["matched"], 0)
        self.assertEqual(result["startLocalization"]["0.5"]["matched"], 1)
        self.assertEqual(result["matchedEventBoundaries"]["0.5"]["bothCorrect"], 1)
        self.assertEqual(result["matchedEndError"]["biasSeconds"], 0.5)

    def test_touch_is_not_overlap_and_complete_miss_is_not_iou_failure(self):
        result = score([[0, 10], [19.9, 25]], truth=[[10, 20], [30, 40]])
        self.assertEqual(result["matchedRallies"], 0)
        self.assertEqual(result["completeMisses"], 1)

    def test_material_split_sensitivity_excludes_tiny_overlap(self):
        result = score([[10, 20], [19.99, 25]])
        self.assertEqual(result["splitTrueRallies"], 1)
        self.assertEqual(result["splitTrueRalliesMaterial"], 0)

    def test_counts_pool_before_rates(self):
        result = evaluate_rally_identities([
            record([[10, 20], [30, 40]], truth=[[10, 20], [30, 40]], identity="one", group="one"),
            record([], identity="two", group="two"),
        ])
        self.assertAlmostEqual(result["pooled"]["eventRecall"], 2 / 3)
        self.assertEqual(result["pooled"]["eventF1"], 0.8)
        self.assertEqual(result["sourceGroups"]["two"]["eventF1"], 0)

    def test_no_predictions_has_zero_detection_not_perfect_precision(self):
        result = score([])
        self.assertEqual(result["eventPrecision"], 0)
        self.assertEqual(result["completeMisses"], 1)

    def test_no_eligible_gold_has_null_quality_and_preserves_false_alarm_counts(self):
        result = score([[10, 20]], truth=[])
        self.assertIsNone(result["eventF1"])
        self.assertEqual(result["falsePositiveRallies"], 1)
        self.assertEqual(result["eventPrecision"], 0)
        json.dumps(result, allow_nan=False)

    def test_original_indexes_preserved_after_sort_and_mask(self):
        result = evaluate_rally_identities([record([[30, 40], [10, 20], [70, 80]], truth=[[10, 20], [30, 40]], ignored=[[70, 80]])])
        self.assertEqual(result["recordings"][0]["matches"], [
            {"truthIndex": 0, "predictionIndex": 1, "iou": 1.0},
            {"truthIndex": 1, "predictionIndex": 0, "iou": 1.0},
        ])

    def test_matching_cardinality_beats_larger_single_iou(self):
        weights = np.asarray([[1.0, 0.51], [0.5, 0.0]])
        eligible = weights >= 0.5
        self.assertEqual(_maximum_weight_pairs(weights, eligible), [(0, 1), (1, 0)])

    def test_general_matching_has_no_chronological_restriction(self):
        weights = np.asarray([[0.1, 0.9], [0.8, 0.2]])
        self.assertEqual(_maximum_weight_pairs(weights, weights >= 0.5), [(0, 1), (1, 0)])

    def test_matching_random_matrices_against_exhaustive_independent_oracle(self):
        rng = np.random.default_rng(41129)
        for shape in ((1, 4), (4, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 4)):
            for _ in range(20):
                weights = rng.integers(0, 11, size=shape) / 10
                eligible = rng.random(shape) > 0.4
                pairs = _maximum_weight_pairs(weights, eligible)
                brute = brute_match(weights, eligible)
                self.assertEqual(len(pairs), brute[0])
                self.assertAlmostEqual(sum(weights[i, j] for i, j in pairs), brute[1])

    def test_timestamp_matching_cardinality_against_brute_oracle(self):
        rng = np.random.default_rng(9109)
        for _ in range(80):
            actual = sorted(rng.integers(0, 20, size=4) / 4)
            predicted = sorted(rng.integers(0, 20, size=4) / 4)
            distances = np.abs(np.asarray(actual)[:, None] - np.asarray(predicted)[None, :])
            for tolerance in (0.25, 0.5, 1, 2):
                brute = brute_match(np.zeros((4, 4)), distances <= tolerance)
                self.assertEqual(_timestamp_matches(actual, predicted, tolerance), brute[0])

    def test_invalid_gold_overlap_duplicate_record_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            score([], truth=[[10, 20], [19, 30]])
        with self.assertRaises(ValueError):
            evaluate_rally_identities([record([]), record([])])
        with self.assertRaises(ValueError):
            score([[10, float("inf")]])
        with self.assertRaises(ValueError):
            score([[True, 10]])
        with self.assertRaises(ValueError):
            score([{"start": 10, "end": 20, "startObserved": 1}])
        with self.assertRaises(ValueError):
            evaluate_rally_identities([])


if __name__ == "__main__":
    unittest.main()
