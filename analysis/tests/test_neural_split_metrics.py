from __future__ import annotations

import json
import unittest

from analysis.neural_split_metrics import evaluate_split_proposals, match_proposals, split_targets


def record(*, gold=((10, 20), (30, 40)), parents=((8, 42),), proposals=(), ignored=(), predictions=None,
           identity='one', group='g'):
    row = {'id': identity, 'sourceGroup': group, 'durationSeconds': 100.,
           'rallies': [{'start': a, 'end': b} for a, b in gold],
           'productionEvents': [{'id': f'P{i}', 'start': a, 'end': b} for i, (a, b) in enumerate(parents)],
           'ignoredIntervals': [list(x) for x in ignored],
           'splitProposals': [{'parentId': f'P{parent}', 'time': time} for parent, time in proposals]}
    if predictions is not None:
        row['predictions'] = [list(x) for x in predictions]
    return row


def evaluate(**kwargs):
    return evaluate_split_proposals([record(**kwargs)])


def score(**kwargs):
    return evaluate(**kwargs)['pooled']


class SplitMetricsTests(unittest.TestCase):
    def test_one_parent_two_gold_has_only_secondary_start_target(self):
        result = evaluate(proposals=((0, 30),))
        row = result['recordings'][0]
        self.assertEqual([x['time'] for x in row['targets']], [30])
        self.assertEqual(row['splitLocalization']['1']['f1'], 1)
        self.assertEqual(row['baselineMergedPredictionsMaterial'], 1)

    def test_first_gold_start_is_boundary_correction_not_split(self):
        result = score(proposals=((0, 10),))['splitLocalization']['1']
        self.assertEqual(result['falsePositive'], 1)
        self.assertEqual(result['falseNegative'], 1)

    def test_first_sliver_does_not_create_material_split_target(self):
        result = score(parents=((19.8, 42),), proposals=((0, 30),))
        self.assertEqual(result['splitTargets'], 0)
        self.assertEqual(result['splitLocalization']['1']['falsePositive'], 1)
        self.assertIsNone(result['splitLocalization']['1']['f1'])

    def test_false_split_inside_one_rally(self):
        result = score(gold=((10, 40),), proposals=((0, 25),))
        self.assertEqual(result['splitTargets'], 0)
        self.assertEqual(result['splitLocalization']['1']['falsePositive'], 1)
        self.assertIsNone(result['splitLocalization']['1']['recall'])

    def test_near_edge_target_stays_in_denominator(self):
        result = score(parents=((8, 31),))
        self.assertEqual(result['splitTargets'], 1)
        self.assertEqual(result['edgeInaccessibleSplitTargets'], 1)
        self.assertEqual(result['splitLocalization']['1']['recall'], 0)

    def test_edge_guard_is_strict(self):
        result = score(parents=((8, 32),), proposals=((0, 30),))
        self.assertEqual(result['accessibleSplitTargets'], 0)
        self.assertEqual(result['splitLocalization']['1']['matched'], 1)

    def test_duplicate_proposals_are_false_positives(self):
        result = score(proposals=((0, 30), (0, 30)))['splitLocalization']['1']
        self.assertEqual(result['matched'], 1)
        self.assertEqual(result['falsePositive'], 1)
        self.assertEqual(result['precision'], .5)

    def test_two_targets_use_cardinality_then_minimum_total_distance(self):
        row = record(gold=((10, 20), (30, 30.5), (31.5, 40)), proposals=((0, 30.8), (0, 29.1)))
        result = match_proposals(row, row['splitProposals'])
        self.assertEqual(result['pairs'], [(1, 0), (0, 1)])

    def test_nearest_duplicate_proposal_selected(self):
        row = record(proposals=((0, 30.8), (0, 30.2)))
        result = match_proposals(row, row['splitProposals'])
        self.assertEqual(result['pairs'], [(1, 0)])

    def test_tolerance_inclusive_and_sensitivity(self):
        values = score(proposals=((0, 31),))['splitLocalization']
        self.assertEqual(values['0.5']['matched'], 0)
        self.assertEqual(values['1']['matched'], 1)
        self.assertEqual(values['2']['matched'], 1)

    def test_split_targets_do_not_bridge_ignored_gap(self):
        result = score(ignored=((21, 29),), proposals=((0, 30),))
        self.assertEqual(result['splitTargets'], 0)
        self.assertEqual(result['splitLocalization']['1']['falsePositive'], 1)

    def test_nearby_proposal_cannot_match_target_across_ignored_gap(self):
        result = score(gold=((10, 20), (30, 30.2), (30.4, 32)),
                       ignored=((29.9, 30),), proposals=((0, 29.8),))
        self.assertEqual(result['splitTargets'], 1)
        self.assertEqual(result['splitLocalization']['2']['matched'], 0)

    def test_ignored_touched_gold_censored_entirely(self):
        result = score(ignored=((34, 36),), proposals=((0, 30),))
        self.assertEqual(result['ignoredTouchedTrueRallies'], 1)
        self.assertEqual(result['maskedSplitProposals'], 1)
        self.assertEqual(result['splitTargets'], 0)
        self.assertEqual(result['splitProposals'], 0)

    def test_masked_proposals_preserve_original_indexes(self):
        row = record(gold=((10, 15), (20, 25), (30, 40)), ignored=((8, 9),),
                     proposals=((0, 8.5), (0, 20), (0, 30)))
        result = match_proposals(row, row['splitProposals'])
        self.assertEqual(result['pairs'], [(1, 0), (2, 1)])

    def test_proposals_cannot_match_across_parent_identity(self):
        result = score(parents=((8, 42), (28, 45)), proposals=((1, 30),))
        self.assertEqual(result['splitTargets'], 1)
        self.assertEqual(result['splitLocalization']['1']['matched'], 0)

    def test_partial_second_gold_with_internal_start_is_target(self):
        result = score(parents=((8, 31),))
        self.assertEqual(result['splitTargets'], 1)

    def test_gold_start_outside_parent_not_target(self):
        result = score(gold=((10, 20), (30, 40)), parents=((31, 42),))
        self.assertEqual(result['splitTargets'], 0)

    def test_touching_gold_is_valid_split(self):
        result = score(gold=((10, 20), (20, 30)), parents=((10, 30),), proposals=((0, 20),))
        self.assertEqual(result['splitLocalization']['1']['f1'], 1)

    def test_overlapping_parents_report_parent_targets_and_unique_gold(self):
        result = score(parents=((8, 42), (9, 41)), proposals=((0, 30), (1, 30)))
        self.assertEqual(result['splitTargets'], 2)
        self.assertEqual(result['uniqueTargetTrueRallies'], 1)
        self.assertEqual(result['splitLocalization']['1']['f1'], 1)

    def test_partition_preserves_coverage_and_resolves_material_merge(self):
        result = score(proposals=((0, 30),), predictions=((8, 30), (30, 42)))
        self.assertEqual(result['rawSelectedSecondsLostFromBaseline'], 0)
        self.assertEqual(result['rawCoreSecondsLostFromBaseline'], 0)
        self.assertEqual(result['additionalCompleteMisses'], 0)
        self.assertEqual(result['resultMergedPredictionsMaterial'], 0)

    def test_loss_new_retained_recovered_complete_misses(self):
        result = score(gold=((10, 20), (30, 40), (50, 60)), parents=((8, 22),),
                       predictions=((29, 41),))
        self.assertEqual(result['baselineCompleteMisses'], 2)
        self.assertEqual(result['resultCompleteMisses'], 2)
        self.assertEqual(result['retainedCompleteMisses'], 1)
        self.assertEqual(result['additionalCompleteMisses'], 1)
        self.assertEqual(result['recoveredCompleteMisses'], 1)
        self.assertEqual(result['rawCoreSecondsLostFromBaseline'], 10)
        self.assertEqual(result['rawCoreSecondsAddedToBaseline'], 10)

    def test_partial_loss_counted_even_without_new_complete_miss(self):
        result = score(gold=((10, 20),), parents=((8, 22),), predictions=((15, 22),))
        self.assertEqual(result['additionalCompleteMisses'], 0)
        self.assertEqual(result['rawCoreSecondsLostFromBaseline'], 5)

    def test_empty_result_still_scores_all_complete_misses(self):
        result = score(predictions=())
        self.assertEqual(result['resultCompleteMisses'], 2)
        self.assertEqual(result['additionalCompleteMisses'], 2)

    def test_material_split_diagnostic_excludes_sliver(self):
        result = score(gold=((10, 20),), parents=((8, 22),), predictions=((8, 19.99), (19.99, 22)))
        self.assertEqual(result['resultSplitTrueRallies'], 1)
        self.assertEqual(result['resultSplitTrueRalliesMaterial'], 0)

    def test_pool_counts_not_mean_per_recording_rates(self):
        result = evaluate_split_proposals([
            record(proposals=((0, 30),), identity='a'),
            record(gold=((10, 15), (20, 25), (30, 40)), identity='b', group='other'),
        ])
        self.assertEqual(result['pooled']['splitLocalization']['1']['recall'], 1 / 3)
        self.assertEqual(result['pooled']['splitLocalization']['1']['f1'], .5)
        self.assertEqual(result['sourceGroups']['other']['splitLocalization']['1']['recall'], 0)
        json.dumps(result, allow_nan=False)

    def test_invalid_and_edge_proposals_fail_loudly(self):
        for proposals in (((0, 8),), ((0, 42),), ((1, 30),), ((0, float('nan')),)):
            with self.subTest(proposals=proposals), self.assertRaises(ValueError):
                score(proposals=proposals)

    def test_invalid_overlapping_gold_rejected(self):
        with self.assertRaises(ValueError):
            score(gold=((10, 30), (20, 40)))

    def test_invalid_guard_and_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            split_targets(record(), edge_guard_seconds=-1)
        with self.assertRaises(ValueError):
            match_proposals(record(), [], tolerance=0)

    def test_duplicate_record_ids_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_split_proposals([record(), record()])


if __name__ == '__main__':
    unittest.main()
