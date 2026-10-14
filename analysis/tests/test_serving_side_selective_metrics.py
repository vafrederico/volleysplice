"""Independent checks of precision constraints, review accounting, and ranking."""
import unittest

import numpy as np

from analysis import serving_side_selective_metrics as study


def make_rows(sizes=(24, 24, 24, 24)):
    return [{"rowIndex": i, "sourceGroup": f"group-{group}", "label": local % 2}
            for i, (group, local) in enumerate((group, local) for group, size in enumerate(sizes) for local in range(size))]


def perfect(rows, confidence=.9):
    return np.asarray([[confidence if row["label"] else 1 - confidence, 1.] for row in rows])


def brute_select(rows, candidates, target):
    """Literal full-grid implementation independent of all production helpers."""
    grid = [value / 100 for value in range(100)] + [.995, .999, .9999, 1., 2.]
    truth = np.asarray([row["label"] for row in rows])
    group_masks = [np.asarray([row["sourceGroup"] == group for row in rows])
                   for group in sorted({row["sourceGroup"] for row in rows})]
    best = None
    feasible_count = 0
    for epoch, values in candidates.items():
        route = values[:, 0] >= .5
        score = values[:, 1] * np.where(route, values[:, 0], 1 - values[:, 0])
        for far_cutoff in grid:
            far = (~route) & (score > 0) & (score >= far_cutoff)
            if far.sum() < 20 or sum(bool((far & group).any()) for group in group_masks) < 4:
                continue
            far_precision = ((truth == 0) & far).sum() / far.sum()
            if far_precision < target:
                continue
            for near_cutoff in grid:
                near = route & (score > 0) & (score >= near_cutoff)
                if near.sum() < 20 or sum(bool((near & group).any()) for group in group_masks) < 4:
                    continue
                near_precision = ((truth == 1) & near).sum() / near.sum()
                if near_precision < target:
                    continue
                accepted = near | far
                correct = (near & (truth == 1)) | (far & (truth == 0))
                group_precision = [(correct & group).sum() / (accepted & group).sum()
                                   for group in group_masks if (accepted & group).any()]
                if sum(group_precision) / len(group_precision) < target:
                    continue
                feasible_count += 1
                macro_coverage = np.mean([(accepted & group).sum() / group.sum() for group in group_masks])
                rank = (float(macro_coverage), float(accepted.mean()), float(min(near_precision, far_precision)),
                        -int((accepted & ~correct).sum()), -epoch, far_cutoff + near_cutoff,
                        min(far_cutoff, near_cutoff), far_cutoff, near_cutoff)
                if best is None or rank > best[0]:
                    best = (rank, epoch, far_cutoff, near_cutoff)
    chosen = best[1:] if best else (min(candidates), 2., 2.)
    return chosen, feasible_count


class SelectiveMetricsTests(unittest.TestCase):
    def test_inclusive_cutoffs_and_fixed_near_route(self):
        scores = np.asarray([[.5, 1.], [.2, .5], [.8, .5], [.8, 0.], [0., 1.], [1., 1.]])
        np.testing.assert_array_equal(study.decisions(scores, .4, .4), [1, 0, 1, -1, 0, 1])
        np.testing.assert_array_equal(study.decisions(scores, 2., 2.), np.full(6, -1))
        np.testing.assert_array_equal(study.decisions(scores, 1., 1.), [-1, -1, -1, -1, 0, 1])
        self.assertEqual(study.decisions(np.asarray([[.8, 0.]]), 0., 0.)[0], -1)

    def test_zero_acceptance_veto_cannot_be_overridden_by_selector(self):
        rows = make_rows()
        scores = perfect(rows)
        scores[:2, 1] = 0.
        selected = study.select(rows, {1: scores}, .98)
        self.assertTrue(selected["feasible"])
        self.assertEqual(selected["innerEvaluation"]["pooled"]["accepted"], len(rows) - 2)
        self.assertEqual(selected["innerEvaluation"]["pooled"]["reviewed"], 2)

    def test_metrics_count_precision_review_and_wrong_guess_capture(self):
        rows = [{"label": value, "sourceGroup": "group-0" if i < 4 else "group-1"}
                for i, value in enumerate([0, 1, 0, 1, 0, 1])]
        result = study.evaluate(rows, np.asarray([0, 1, 1, -1, -1, -1]), np.asarray([0, 1, 1, 0, 1, 1]))
        pooled = result["pooled"]
        self.assertEqual((pooled["accepted"], pooled["correct"], pooled["errors"], pooled["reviewed"]), (3, 2, 1, 3))
        self.assertEqual(pooled["precision"], 2 / 3)
        self.assertEqual((pooled["nearPrecision"], pooled["farPrecision"]), (.5, 1.))
        self.assertEqual((pooled["coverage"], pooled["reviewFraction"]), (.5, .5))
        self.assertEqual(pooled["silentErrorFraction"], 1 / 6)
        self.assertEqual(pooled["correctAutoFraction"], 1 / 3)
        self.assertEqual(pooled["underlyingErrors"], 3)
        self.assertEqual(pooled["reviewedUnderlyingErrors"], 2)
        self.assertEqual(pooled["underlyingErrorReviewCapture"], 2 / 3)
        self.assertEqual(result["sourceGroupMacroCoverage"], .375)
        self.assertEqual(result["sourceGroupMacroPrecision"], 2 / 3)
        self.assertEqual(result["worstSupportedGroupPrecision"], 2 / 3)
        self.assertEqual(result["perLabelCoverage"], {"far": 2 / 3, "near": 1 / 3})

    def test_no_acceptance_has_null_precision_not_perfect_precision(self):
        rows = make_rows()
        result = study.evaluate(rows, np.full(len(rows), -1), np.asarray([row["label"] for row in rows]))
        self.assertIsNone(result["pooled"]["precision"])
        self.assertIsNone(result["pooled"]["nearPrecision"])
        self.assertIsNone(result["pooled"]["farPrecision"])
        self.assertIsNone(result["pooled"]["precisionLowerWilson95DESCRIPTIVE"])
        self.assertIsNone(result["sourceGroupMacroPrecision"])
        self.assertIsNone(result["worstSupportedGroupPrecision"])
        self.assertIsNone(result["pooled"]["underlyingErrorReviewCapture"])
        self.assertEqual(result["sourceGroupMacroCoverage"], 0.)
        self.assertEqual(result["pooled"]["reviewFraction"], 1.)

    def test_earlier_epoch_then_conservative_cutoffs_break_ties(self):
        rows = make_rows()
        chosen = study.select(rows, {5: perfect(rows, .9), 1: perfect(rows, .8)}, .98)
        self.assertTrue(chosen["feasible"])
        self.assertEqual((chosen["epoch"], chosen["farCutoff"], chosen["nearCutoff"]), (1, .8, .8))
        self.assertEqual(chosen["innerEvaluation"]["pooled"]["accepted"], len(rows))
        self.assertEqual(chosen["candidateCount"], 2 * 105 ** 2)

    def test_source_coverage_requirement_is_per_predicted_side(self):
        rows = make_rows((30, 30, 30))
        chosen = study.select(rows, {2: perfect(rows)}, .95)
        self.assertFalse(chosen["feasible"])
        self.assertEqual((chosen["epoch"], chosen["farCutoff"], chosen["nearCutoff"]), (2, 2., 2.))
        self.assertIsNone(chosen["innerEvaluation"]["pooled"]["precision"])
        rows = make_rows()
        values = perfect(rows)
        # The fourth group has no near routes, even though total support is ample.
        values[-24:, 0] = .1
        self.assertFalse(study.select(rows, {2: values}, .5)["feasible"])

    def test_class_imbalance_cannot_hide_bad_near_precision(self):
        rows = []
        values = []
        for group in range(4):
            for i in range(105):
                # Five near routes per group, only three of them correct.
                rows.append({"label": int(i >= 102), "sourceGroup": f"group-{group}"})
                values.append([.9 if i >= 100 else .1, 1.])
        result = study.select(rows, {1: np.asarray(values)}, .95)
        self.assertFalse(result["feasible"])

    def test_group_macro_precision_constrains_large_good_group(self):
        rows = make_rows((400, 12, 12, 12))
        values = perfect(rows, 1.)
        values[:, 1] = .9
        values[400:406, 0] = 1 - values[400:406, 0]
        guesses = (values[:, 0] >= .5).astype(int)
        evaluation = study.evaluate(rows, guesses)
        self.assertGreater(evaluation["pooled"]["nearPrecision"], .98)
        self.assertGreater(evaluation["pooled"]["farPrecision"], .98)
        self.assertLess(evaluation["sourceGroupMacroPrecision"], .95)
        self.assertFalse(study.select(rows, {1: values}, .95)["feasible"])

    def test_group_macro_coverage_ranks_before_pooled_coverage(self):
        rows = make_rows((160, 20, 20, 20))
        labels = np.asarray([row["label"] for row in rows])
        first_mask = np.asarray([i < 160 or (i - 160) % 20 < 10 for i in range(len(rows))])
        second_mask = np.asarray([i < 80 or i >= 160 for i in range(len(rows))])
        candidates = {}
        for epoch, mask in ((1, first_mask), (2, second_mask)):
            routed = np.where(mask, labels, 1 - labels)
            candidates[epoch] = np.column_stack((routed, mask.astype(float)))
        selected = study.select(rows, candidates, 1.)
        self.assertEqual(selected["epoch"], 2)
        self.assertEqual(selected["innerEvaluation"]["pooled"]["accepted"], 140)
        self.assertEqual(selected["innerEvaluation"]["sourceGroupMacroCoverage"], .875)

    def test_minimum_side_support_is_twenty(self):
        rows = make_rows((10, 10, 10, 10))
        self.assertTrue(study.select(rows, {1: perfect(rows)}, 1.)["feasible"])
        rows.pop()
        self.assertFalse(study.select(rows, {1: perfect(rows)}, 1.)["feasible"])

    def test_vectorized_selector_matches_independent_full_grid(self):
        rows = make_rows((24, 30, 26, 28, 20))
        labels = np.asarray([row["label"] for row in rows])
        for seed, target in ((7, .9), (17, .95), (27, .99)):
            with self.subTest(seed=seed, target=target):
                rng = np.random.default_rng(seed)
                candidates = {}
                for epoch in (1, 5):
                    guessed = labels.copy()
                    errors = rng.choice(len(rows), 8, replace=False)
                    guessed[errors] = 1 - guessed[errors]
                    certainty = rng.choice([.55, .65, .75, .85, .95, .999], len(rows))
                    accept = rng.choice([.45, .7, .9, 1.], len(rows))
                    accept[errors[:5]] = .3
                    candidates[epoch] = np.column_stack((np.where(guessed, certainty, 1 - certainty), accept))
                expected, feasible_count = brute_select(rows, candidates, target)
                actual = study.select(rows, candidates, target)
                self.assertEqual((actual["epoch"], actual["farCutoff"], actual["nearCutoff"]), expected)
                self.assertEqual(actual["feasibleCandidateCount"], feasible_count)

    def test_target_scores_and_labels_fail_closed(self):
        rows = make_rows()
        valid = perfect(rows)
        for target in (0., 1.1, -1., float("nan"), float("inf"), "0.98"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                study.select(rows, {1: valid}, target)
        for broken in (valid[:-1], np.zeros((len(rows), 3)), valid * 2, valid * np.nan):
            with self.assertRaises(ValueError):
                study.select(rows, {1: broken}, .98)
        with self.assertRaises(ValueError):
            study.select(rows, {}, .98)
        with self.assertRaises(ValueError):
            study.select(rows, {1.5: valid}, .98)
        with self.assertRaises(ValueError):
            study.select([], {1: np.zeros((0, 2))}, .98)
        with self.assertRaises(ValueError):
            study.evaluate([{"label": 2, "sourceGroup": "group-0"}], np.asarray([0]))
        with self.assertRaises(ValueError):
            study.evaluate(rows, np.full(len(rows), 2))
        with self.assertRaises(ValueError):
            study.decisions(valid, -.1, .5)

    def test_empty_evaluation_is_explicit(self):
        result = study.evaluate([], np.asarray([], dtype=int))
        self.assertEqual(result["pooled"]["rows"], 0)
        self.assertIsNone(result["pooled"]["precision"])
        self.assertIsNone(result["pooled"]["coverage"])
        self.assertIsNone(result["sourceGroupMacroCoverage"])


if __name__ == "__main__":
    unittest.main()
