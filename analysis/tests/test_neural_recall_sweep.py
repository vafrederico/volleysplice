import copy
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_recall_sweep as sweep
from analysis.neural_expanded_development import decoder_candidates
from analysis.schema import Interval


def example(identifier='a', group='A', policy='exact-rallies'):
    times = np.arange(.125, 20, .25)
    return sweep.SweepExample(identifier, group, 20., times, np.ones(len(times), bool),
                              (Interval(5., 10.),), (), policy)


def scores(e):
    p = np.zeros((len(e.times), 4), np.float32)
    p[(e.times >= 5) & (e.times < 10), 0] = .95
    return {epoch: {e.id: p.copy()} for epoch in sweep.EPOCHS}


def candidates():
    return [{'epoch': epoch, 'decoder': decoder, 'innerR_core': .89, 'innerF1_padP_coreR': .9}
            for epoch in sweep.EPOCHS for decoder in decoder_candidates()]


class RecallSweepTests(unittest.TestCase):
    def test_proxy_selection_requires_explicit_opt_in_and_is_not_gold_accuracy(self):
        e = example(policy='export-rally-proxy-selection')
        with self.assertRaises(ValueError):
            sweep.build_candidate_table([e], scores(e))
        table = sweep.build_candidate_table([e], scores(e), selection_policy='export-rally-proxy-selection')
        self.assertEqual(len(table), 192)
        self.assertTrue(all(c['selectionLabelPolicy'] == 'export-rally-proxy-selection'
                            and c['selectionMetricsAreGoldAccuracy'] is False for c in table))
        self.assertEqual(sweep.select_floors(table)[0]['selected']['selectionLabelPolicy'], 'export-rally-proxy-selection')

    def test_every_integer_floor_and_exact100_has_no_epsilon_fallback(self):
        table = candidates()
        table[0].update(innerR_core=.95, innerF1_padP_coreR=.94)
        table[1].update(innerR_core=np.nextafter(1., 0.), innerF1_padP_coreR=.85)
        selected = sweep.select_floors(table)
        self.assertEqual([x['floorPercent'] for x in selected], list(range(90, 101)))
        self.assertEqual(selected[5]['selected'], table[0])
        self.assertEqual(selected[6]['selected'], table[1])
        self.assertFalse(selected[-1]['feasible'])
        self.assertIsNone(selected[-1]['selected'])
        table[2].update(innerR_core=1., innerF1_padP_coreR=.8)
        self.assertEqual(sweep.select_floors(table)[-1]['selected'], table[2])

    def test_f1_ordered_selection_and_grid_order_are_preserved(self):
        table = candidates()
        table[0].update(innerR_core=.99, innerF1_padP_coreR=.9)
        table[1].update(innerR_core=1., innerF1_padP_coreR=.9)
        self.assertEqual(sweep.select_floors(table)[9]['selected'], table[0])
        table.reverse()
        with self.assertRaises(ValueError):
            sweep.select_floors(table)

    def test_selected_decoding_is_cached_across_floors_and_validity_unchanged(self):
        e = example(); before = e.valid.copy(); table = candidates()
        table[0].update(innerR_core=1., innerF1_padP_coreR=.9)
        decisions = sweep.select_floors(table)
        original = sweep.base.decode
        with patch.object(sweep.base, 'decode', wraps=original) as decoder:
            result = sweep.evaluate_selected([e], scores(e), decisions)
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(len(result['operatingPoints']), 1)
        np.testing.assert_array_equal(before, e.valid)
        self.assertTrue(all(r['status'] == 'available' for r in result['floors']))

    def test_missing_checkpoints_are_separate_from_infeasible_decisions(self):
        e = example(); table = candidates(); table[0].update(innerR_core=.99)
        result = sweep.evaluate_selected([e], {}, sweep.select_floors(table))
        self.assertEqual(result['floors'][0]['status'], 'missing-selected-checkpoint')
        self.assertEqual(result['floors'][-1]['status'], 'infeasible-inner-recall')

    def test_no_partial_seed_mean_or_cross_variant_scope_mean(self):
        e = example(); table = candidates(); table[0].update(innerR_core=.99)
        fold = sweep.evaluate_selected([e], scores(e), sweep.select_floors(table))
        full = sweep.pool_fold_results([fold], fold['expectedGold'])
        cell = {'model': 'test', 'variant': 'v1', 'selectionDesign': 'nested', 'panelId': 'original', 'seed': 1, 'result': full}
        second = copy.deepcopy(cell); second['seed'] = 2
        second['result']['floors'][0].update(completeEvaluationScope=False, evaluation=None)
        result = sweep.summarize_seed_cells([cell, second], [1, 2])
        self.assertIsNone(result['floors'][0]['meanPadding'])
        self.assertIsNotNone(result['floors'][1]['meanPadding'])
        self.assertIsNone(result['floors'][-1]['meanPadding'])
        second['variant'] = 'v2'
        with self.assertRaises(ValueError):
            sweep.summarize_seed_cells([cell, second], [1, 2])

    def test_outer_pool_rejects_duplicate_recordings(self):
        e = example(); table = candidates(); table[0].update(innerR_core=1.)
        fold = sweep.evaluate_selected([e], scores(e), sweep.select_floors(table))
        with self.assertRaises(ValueError):
            sweep.pool_fold_results([fold, fold], fold['expectedGold'])

    def test_reviewed_exports_are_fixed_human_union_not_invented_core_events(self):
        row = {'id': 'a', 'sourceGroup': 'A', 'durationSeconds': 20.,
               'humanExportIntervals': [{'start': 5., 'end': 10.}],
               'predictions': [{'start': 5., 'end': 10.}], 'ignoredIntervals': [{'start': 6., 'end': 7.}]}
        result = sweep.evaluate_export_panel([row])
        self.assertFalse(result['eventMetricsAvailable'])
        self.assertNotIn('R_core', result['primary'])
        self.assertEqual(result['primary']['humanExportSeconds'], 4.)
        self.assertEqual(result['primary']['modelExportSeconds'], 8.)
        self.assertEqual(result['primary']['P_export'], .5)
        self.assertEqual(result['primary']['R_export'], 1.)
        with self.assertRaises(ValueError):
            sweep.build_candidate_table([example(policy='reviewed-export')], scores(example()))

    def test_export_gap_exact3_not_joined_and_ignored_not_rejoined(self):
        row = {'id': 'a', 'sourceGroup': 'A', 'durationSeconds': 20.,
               'humanExportIntervals': [{'start': 1., 'end': 3.}, {'start': 6., 'end': 8.}],
               'predictions': [{'start': 1., 'end': 3.}, {'start': 6., 'end': 8.}],
               'ignoredIntervals': [{'start': 2., 'end': 2.5}]}
        result = sweep.evaluate_export_panel([row])
        self.assertEqual(result['padding'][0]['modelExportSeconds'], 3.5)
        self.assertEqual(result['padding'][0]['humanExportSeconds'], 3.5)
        row['predictions'][1]['start'] = 5.999
        self.assertAlmostEqual(sweep.evaluate_export_panel([row])['padding'][0]['modelExportSeconds'], 6.5)

    def test_build_decodes192_once_not_once_per_recall_floor(self):
        e = example(); original = sweep.base.decode
        with patch.object(sweep.base, 'decode', wraps=original) as decoder:
            table = sweep.build_candidate_table([e], scores(e))
            sweep.select_floors(table)
        self.assertEqual(decoder.call_count, 192)

    def test_reviewed_draft_is_symmetric_time_only_and_cannot_select(self):
        e = example(policy='reviewed-draft')
        rows = sweep.panel_rows([e], {e.id: [Interval(5., 10.)]}, 'reviewed-draft')
        result = sweep.evaluate_rows(rows, 'reviewed-draft')
        self.assertEqual(result['primary']['humanExportSeconds'], 9.)
        self.assertEqual(result['primary']['P_reviewed'], 1.)
        self.assertEqual(result['primary']['R_reviewed'], 1.)
        self.assertNotIn('guardrails', result)
        self.assertNotIn('R_core', result['primary'])
        self.assertFalse(result['eventMetricsAvailable'])
        with self.assertRaises(ValueError):
            sweep.build_candidate_table([e], scores(e))


if __name__ == '__main__':
    unittest.main()
