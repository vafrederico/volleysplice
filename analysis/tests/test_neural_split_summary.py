from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

from analysis import neural_production_combinations as iv
from analysis.neural_rally_identity_metrics import evaluate_rally_identities
from analysis.neural_split_metrics import evaluate_split_proposals


PATH = Path(__file__).resolve().parents[2]/'scripts/summarize-neural-split-advisor.py'
SPEC = importlib.util.spec_from_file_location('split_summary_under_test', PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixture():
    records = []
    for name, group in [('a', 'g1'), ('b', 'g2')]:
        parents = [{'id': 'p', 'start': 8., 'end': 42.}]
        records.append({'id': name, 'sourceGroup': group, 'durationSeconds': 100.,
                        'rallies': [[10., 20.], [30., 40.]], 'productionEvents': parents,
                        'predictions': parents, 'splitProposals': [], 'ignoredIntervals': []})
    baseline = {'durationMetrics': iv.duration_rows(records), 'identityMetrics': evaluate_rally_identities(records),
                'splitMetrics': evaluate_split_proposals(records),
                'durationAudit': {'passed': True}, 'identityAudit': {'passed': True}}
    policies = ['event_starts', 'head_evidence', 'corroborated']
    configurations = [{'id': f'{p}--{inv}', 'policy': p, 'inventory': inv}
                      for p in policies for inv in ('split_only', 'combined')]
    configurations.append({'id': 'none--cleanup_only', 'policy': 'none', 'inventory': 'cleanup_only'})
    contract = {'seeds': [3407, 1729, 20260918], 'splitPolicies': policies,
                'reviewInventories': configurations, 'rankers': ['chronological', 'evidence'],
                'budgetFractions': [.05, .1, .2, .4]}
    cells = []
    for seed in contract['seeds']:
        automatic = [{**deepcopy(baseline), 'id': 'automatic--'+p, 'policy': p} for p in policies]
        reviewed = []
        for name, config in summary.expected_reviewed(contract).items():
            per = [{'id': r['id'], 'sourceGroup': r['sourceGroup'], **{k: 0 for k in summary.WORKLOAD_FIELDS}} for r in records]
            reviewed.append({**deepcopy(baseline), 'id': name, **config,
                             'workload': {k: 0 for k in summary.WORKLOAD_FIELDS}, 'perRecording': per})
        plans = [{'id': r['id'], 'cleanupYield': {'flaggedParents': 1, 'whollyFalseParents': 0,
                   'realOrMixedParents': 1, 'realRalliesTouched': 2}} for r in records]
        cells.append({'seed': seed, 'baseline': deepcopy(baseline), 'automatic': automatic, 'reviewed': reviewed, 'plans': plans})
    return contract, cells


class SplitSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.cells = fixture()

    def test_full_matrix_aggregates_all_scopes(self):
        result = summary.aggregate(self.contract, self.cells)
        self.assertEqual(result['automaticOutcomes'], 9)
        self.assertEqual(result['reviewedOutcomes'], 168)
        self.assertEqual(result['automaticArms'], 3)
        self.assertEqual(result['reviewedArms'], 56)
        self.assertEqual(set(result['baseline']['sourceGroups']), {'g1', 'g2'})
        self.assertEqual(set(result['baseline']['recordings']), {'a', 'b'})
        self.assertEqual(result['baseline']['split']['splitTargets'], 2)
        self.assertEqual(result['cleanupYield']['pooled']['metrics']['realRalliesTouched'], 4)
        self.assertTrue(result['verification']['everyExportPaddingEqualsProduction'])
        json.dumps(result, allow_nan=False)

    def test_numeric_means_ranges_ignore_unavailable_not_false_zero(self):
        tree = summary.tree_stats([{'x': None, 'nested': {'n': 1}}, {'x': None, 'nested': {'n': 2}}, {'x': None, 'nested': {'n': 6}}])
        self.assertIsNone(tree['x']['mean'])
        self.assertEqual(tree['x']['availableSeeds'], 0)
        self.assertEqual(tree['nested']['n']['mean'], 3)
        self.assertEqual(tree['nested']['n']['min'], 1)
        self.assertEqual(tree['nested']['n']['max'], 6)

    def test_scope_duration_pools_counts_before_rates(self):
        rows = [{k: 0. for k in summary.SUM_FIELDS} for _ in range(2)]
        rows[0].update(paddedModelExportSeconds=10, paddedIntersectionSeconds=10,
                       coreHumanSeconds=10, coreIntersectionSeconds=10)
        rows[1].update(paddedModelExportSeconds=90, paddedIntersectionSeconds=0,
                       coreHumanSeconds=90, coreIntersectionSeconds=0)
        pooled = summary.pool_duration(rows)
        self.assertEqual(pooled['P_pad'], .1)
        self.assertEqual(pooled['R_core'], .1)
        self.assertAlmostEqual(pooled['F1_padP_coreR'], .1)

    def test_missing_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, 'seed'):
            summary.aggregate(self.contract, self.cells[:2])

    def test_duplicate_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, 'seed'):
            summary.aggregate(self.contract, [self.cells[0], self.cells[0], self.cells[2]])

    def test_two_seed_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, 'three'):
            summary.aggregate({**self.contract, 'seeds': self.contract['seeds'][:2]}, self.cells[:2])

    def test_missing_review_arm_rejected(self):
        cells = deepcopy(self.cells); cells[0]['reviewed'].pop()
        with self.assertRaisesRegex(ValueError, 'Reviewed scope'):
            summary.aggregate(self.contract, cells)

    def test_overwritten_review_id_rejected(self):
        cells = deepcopy(self.cells); cells[0]['reviewed'][0]['id'] = 'event_starts--split_only'
        with self.assertRaisesRegex(ValueError, 'Reviewed scope'):
            summary.aggregate(self.contract, cells)

    def test_changed_padding_export_rejected(self):
        cells = deepcopy(self.cells)
        cells[0]['automatic'][0]['durationMetrics'][3]['paddedModelExportSeconds'] += 1
        with self.assertRaisesRegex(ValueError, 'Frozen exports'):
            summary.aggregate(self.contract, cells)

    def test_new_complete_miss_rejected(self):
        cells = deepcopy(self.cells)
        cells[0]['automatic'][0]['splitMetrics']['pooled']['additionalCompleteMisses'] = 1
        with self.assertRaisesRegex(ValueError, 'New complete'):
            summary.aggregate(self.contract, cells)

    def test_partial_core_loss_rejected(self):
        cells = deepcopy(self.cells)
        cells[0]['reviewed'][0]['splitMetrics']['pooled']['rawCoreSecondsLostFromBaseline'] = .1
        with self.assertRaisesRegex(ValueError, 'Raw rally coverage'):
            summary.aggregate(self.contract, cells)

    def test_automatic_dead_time_removal_rejected(self):
        cells = deepcopy(self.cells)
        cells[0]['automatic'][0]['splitMetrics']['pooled']['rawSelectedSecondsLostFromBaseline'] = 1
        with self.assertRaisesRegex(ValueError, 'raw occupancy'):
            summary.aggregate(self.contract, cells)

    def test_workload_total_mismatch_rejected(self):
        cells = deepcopy(self.cells)
        cells[0]['reviewed'][0]['workload']['reviewSeconds'] = 1
        with self.assertRaisesRegex(ValueError, 'Workload total'):
            summary.aggregate(self.contract, cells)

    def test_failed_upstream_audit_rejected(self):
        cells = deepcopy(self.cells)
        cells[0]['automatic'][0]['identityAudit']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'audit failed'):
            summary.aggregate(self.contract, cells)

    def test_markdown_includes_all_arms_and_limits(self):
        result = summary.aggregate(self.contract, self.cells)
        document = summary.markdown(result)
        self.assertIn('all four padding cases', document)
        self.assertIn('2 parent-specific genuine additional-start targets', document)
        self.assertIn('not serving-side, point-winner or reconstructed-score accuracy', document)
        for arm in result['reviewed']:
            self.assertIn(arm['id'], document)
        self.assertIn('none--cleanup_only--evidence--budget-40', document)

    def test_statistics_reject_nan(self):
        with self.assertRaises(ValueError):
            summary.stats([1, float('nan'), 2])


if __name__ == '__main__':
    unittest.main()
