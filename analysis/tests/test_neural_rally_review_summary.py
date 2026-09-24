from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

from analysis.neural_production_combinations import duration_rows
from analysis.neural_rally_identity_metrics import evaluate_rally_identities


PATH = Path(__file__).resolve().parents[2]/'scripts/summarize-neural-rally-review-proposals.py'
SPEC = importlib.util.spec_from_file_location('rally_review_summary', PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def records(predictions):
    return [
        {'id': 'large', 'sourceGroup': 'large', 'durationSeconds': 100,
         'rallies': [[10, 20], [30, 40]], 'predictions': predictions[0], 'ignoredIntervals': []},
        {'id': 'small', 'sourceGroup': 'small', 'durationSeconds': 100,
         'rallies': [[60, 61]], 'predictions': predictions[1], 'ignoredIntervals': []},
    ]


def evaluation(predictions):
    rows = records(predictions)
    return {'durationMetrics': duration_rows(rows), 'identityMetrics': evaluate_rally_identities(rows)}


def fixture():
    config = {'id': 'production--compact_boost--local_events--evidence', 'mode': 'production',
              'model': 'compact_boost', 'inventory': 'local_events', 'ranker': 'evidence'}
    contract = {'seeds': [1, 2, 3], 'sourceGroups': ['large', 'small'], 'paddingCases': [0, 1, 2, 3],
                'budgetFractions': [.05, .1, .2, .4], 'configurations': [config]}
    base = evaluation(([[10, 20]], []))
    predicted = (([[10, 20], [30, 40]], []), ([[0, 50]], [[60, 61]]), ([[10, 20], [30, 40]], [[60, 61]]))
    cells, refs = [], {}
    for seed, predictions in enumerate(predicted, 1):
        outcomes = []
        for fraction in contract['budgetFractions']:
            work = {key: 0. for key in summary.WORKLOAD_FIELDS}
            work.update(budgetSeconds=200*fraction, reviewSeconds=100*fraction,
                        reviewFractionOfVideo=fraction/2, budgetUtilization=.5)
            group_work = {group: {**work, 'budgetSeconds': 100*fraction, 'reviewSeconds': 50*fraction}
                          for group in contract['sourceGroups']}
            outcomes.append({'budgetFraction': fraction, **evaluation(predictions), 'workload': work,
                             'workloadBySourceGroup': group_work})
        cells.append({'configuration': config, 'seed': seed, 'automatic': copy.deepcopy(base), 'outcomes': outcomes})
        refs[(config['id'], seed)] = {'path': f'fixture-{seed}.json', 'sha256': 'synthetic', 'sizeBytes': 1}
    return contract, cells, refs


class RallyReviewSummaryTests(unittest.TestCase):
    def test_duration_group_pool_uses_counts_not_video_mean(self):
        rows = evaluation(([[10, 20], [30, 40]], []))['durationMetrics'][0]['perRecording']
        pooled = summary.pool_duration(rows)
        self.assertAlmostEqual(pooled['R_core'], 20/21)
        self.assertNotAlmostEqual(pooled['R_core'], .5)
        self.assertEqual(pooled['P_pad'], 1)

    def test_seed_pooled_rates_averaged_not_repooled_across_seeds(self):
        contract, cells, refs = fixture()
        report = summary.aggregate(contract, cells, refs)
        arm = report['arms'][0]
        samples = [summary.at(cell['outcomes'][0]['durationMetrics']) for cell in cells]
        expected = sum(row['F1_padP_coreR'] for row in samples)/3
        self.assertAlmostEqual(arm['primary']['F1_padP_coreR'], expected)
        all_seed_repool = summary.pool_duration([record for cell in cells
                                               for record in summary.at(cell['outcomes'][0]['durationMetrics'])['perRecording']])
        self.assertNotAlmostEqual(expected, all_seed_repool['F1_padP_coreR'])

    def test_event_counts_pool_within_seed_then_seed_f1_mean(self):
        contract, cells, refs = fixture()
        arm = summary.aggregate(contract, cells, refs)['arms'][0]
        values = [cell['outcomes'][0]['identityMetrics']['pooled']['eventF1'] for cell in cells]
        self.assertAlmostEqual(arm['identity']['eventF1'], sum(values)/3)
        self.assertEqual(arm['identitySeedStatistics']['eventF1']['availableSeeds'], 3)
        self.assertEqual(arm['identitySeedStatistics']['eventF1']['min'], min(values))

    def test_every_budget_padding_scope_and_seed_ref_retained(self):
        contract, cells, refs = fixture()
        report = summary.aggregate(contract, cells, refs)
        self.assertEqual(len(report['arms']), 4)
        self.assertEqual([row['budgetFraction'] for row in report['arms']], [.05, .1, .2, .4])
        for arm in report['arms']:
            self.assertEqual([pad['paddingSecondsBeforeAndAfter'] for pad in arm['padding']], [0, 1, 2, 3])
            self.assertEqual(set(arm['sourceGroups']), {'large', 'small'})
            self.assertEqual([row['seed'] for row in arm['seedResults']], [1, 2, 3])
            self.assertAlmostEqual(arm['workload']['unusedBudgetSeconds'], 100*arm['budgetFraction'])
            self.assertEqual(len(arm['identity']['observedStartLocalization']), 4)

    def test_repeated_review_arms_do_not_inflate_baseline_seeds(self):
        contract, cells, refs = fixture()
        other = {**contract['configurations'][0], 'id': 'other', 'ranker': 'chronological'}
        contract['configurations'].append(other)
        for original in list(cells):
            cell = copy.deepcopy(original)
            cell['configuration'] = other
            cells.append(cell)
            refs[(other['id'], cell['seed'])] = {'path': 'other', 'sha256': 'synthetic', 'sizeBytes': 1}
        report = summary.aggregate(contract, cells, refs)
        self.assertEqual(len(report['automaticBaselines']), 1)
        stats = report['automaticBaselines'][0]['identitySeedStatistics']['eventF1']
        self.assertEqual(stats['totalSeeds'], 3)

    def test_changed_automatic_baseline_across_arms_rejected(self):
        contract, cells, refs = fixture()
        other = {**contract['configurations'][0], 'id': 'other'}
        contract['configurations'].append(other)
        cell = copy.deepcopy(cells[0]); cell['configuration'] = other
        cell['automatic']['identityMetrics']['pooled']['eventF1'] = .99
        cells.append(cell)
        with self.assertRaisesRegex(ValueError, 'baseline changed'):
            summary.aggregate(contract, cells, refs)

    def test_null_conditional_metrics_report_available_seed_count(self):
        stats = summary.tree_statistics([{'conditional': None}, {'conditional': 2.}, {'conditional': 4.}])
        self.assertEqual(summary.tree_means(stats)['conditional'], 3)
        self.assertEqual(stats['conditional']['availableSeeds'], 2)
        self.assertEqual(stats['conditional']['totalSeeds'], 3)

    def test_null_all_seeds_remains_unavailable(self):
        stats = summary.statistics([None, None, None])
        self.assertIsNone(stats['mean'])
        self.assertIsNone(stats['seedPopulationStddev'])
        self.assertEqual(stats['availableSeeds'], 0)

    def test_guardrail_screen_requires_all_four_conditions(self):
        contract, cells, refs = fixture()
        base = summary.aggregate(contract, cells, refs)['automaticBaselines'][0]
        perfect = copy.deepcopy(base)
        summary.add_deltas_and_screen(perfect, base)
        self.assertTrue(perfect['guardrailScreen']['passed'])
        for key in ('eventF1', 'completeMisses', 'observedF1', 'observedRecall'):
            worse = copy.deepcopy(base)
            if key == 'observedF1':
                worse['identity']['observedStartLocalization']['1']['f1'] -= .01
            elif key == 'observedRecall':
                worse['identity']['observedStartLocalization']['1']['recall'] -= .01
            else:
                worse['identity'][key] += 1 if key == 'completeMisses' else -.01
            summary.add_deltas_and_screen(worse, base)
            self.assertFalse(worse['guardrailScreen']['passed'])

    def test_missing_or_duplicated_seed_budget_rejected(self):
        contract, cells, refs = fixture()
        with self.assertRaises(ValueError):
            summary.aggregate(contract, cells[:-1], refs)
        with self.assertRaises(ValueError):
            summary.aggregate(contract, [*cells, cells[0]], refs)
        cells[0]['outcomes'].append(cells[0]['outcomes'][0])
        with self.assertRaises(ValueError):
            summary.aggregate(contract, cells, refs)

    def test_document_contains_all_arms_and_explicit_non_score_claim(self):
        contract, cells, refs = fixture()
        report = summary.aggregate(contract, cells, refs)
        report.update(contractSha256='synthetic', report={'sha256': 'synthetic'}, summaryPath='synthetic.json')
        document = summary.render_document(report)
        for arm in report['arms']:
            self.assertIn(arm['id'], document)
        self.assertIn('ROOT EXECUTIVE FINDINGS START', document)
        self.assertIn('This does not evaluate serving side, point winner or reconstructed scores.', document)
        self.assertIn('All four padding cases', document)


if __name__ == '__main__':
    unittest.main()
