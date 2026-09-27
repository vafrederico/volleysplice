from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/summarize-neural-event-balanced.py'
SPEC = importlib.util.spec_from_file_location('event_balanced_summary_tested', SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def result(coverage, seed):
    rallies = []
    for index, retained in enumerate(coverage):
        duration = 2 if index < 5 else 8
        rallies.append({'recordingId': 'x', 'truthIndex': index, 'start': index*20,
                        'end': index*20+duration, 'tags': [], 'evaluableCoreSeconds': duration,
                        'retainedCoreSeconds': duration*retained, 'coverage': retained,
                        'fullyCovered': retained == 1, 'completelyLost': retained == 0,
                        'partiallyLost': 0 < retained < 1})
    return {'seed': seed, 'cohort': 'exact', 'selections': [
        {'recallEligibilityFloor': .95, 'innerR_core': .96, 'recallEligibilityPassed': True}],
        'evaluation': {'primary': {'F1_padP_coreR': .9, 'R_core': .94},
                       'guardrails': {'eventF1': .8, **{scope: {'rallies': rallies} for scope in summary.SCOPES}}}}


class EventBalancedSummaryTests(unittest.TestCase):
    def pairs(self):
        baseline = [result([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], seed) for seed in (1, 2, 3)]
        candidate = [result([0, 0, 0, 1, 1, 0, 1, 1, 1, 1], seed) for seed in (1, 2, 3)]
        return candidate, baseline

    def test_recovery_pass_does_not_add_absolute_outer_floor(self):
        candidate, baseline = self.pairs()
        report = summary.recovery_screen(candidate, baseline, summary.RECOVERY_CONTRACT)
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['checks']), 10)
        self.assertFalse(report['allOuterSeedRecallAtLeast095Diagnostic'])
        self.assertFalse(report['productionPromotionAllowed'])

    def test_converting_complete_losses_to_many_partial_losses_fails(self):
        candidate, baseline = self.pairs()
        for row in candidate:
            for scope in summary.SCOPES:
                # Both scopes intentionally share these synthetic original identities.
                rallies = row['evaluation']['guardrails'][scope]['rallies']
                for index in (4, 6):
                    rallies[index].update(fullyCovered=False, partiallyLost=True, coverage=.9,
                                          retainedCoreSeconds=rallies[index]['evaluableCoreSeconds']*.9)
        report = summary.recovery_screen(candidate, baseline, summary.RECOVERY_CONTRACT)
        self.assertFalse(report['passed'])
        self.assertFalse(report['checks']['meanIncompleteLossesDoNotIncrease'])

    def test_short_seed_consistency_and_long_recall_are_separate_checks(self):
        candidate, baseline = self.pairs()
        candidate[1] = result([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 2)
        candidate[2] = result([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 3)
        report = summary.recovery_screen(candidate, baseline, summary.RECOVERY_CONTRACT)
        self.assertTrue(report['checks']['atLeastTwoSeedsReduceCompleteLosses'])
        self.assertFalse(report['checks']['atLeastTwoSeedsReduceShortCompleteLosses'])
        candidate, baseline = self.pairs()
        for row in candidate:
            r = row['evaluation']['guardrails']['primaryExportCoverage']['rallies'][9]
            r.update(fullyCovered=False, partiallyLost=True, coverage=.9, retainedCoreSeconds=7.2)
        report = summary.recovery_screen(candidate, baseline, summary.RECOVERY_CONTRACT)
        self.assertFalse(report['checks']['meanLongCoreRecallRegressionWithinLimit'])

    def test_zero_baseline_does_not_divide_by_zero_or_count_ties(self):
        rows = [result([1]*10, seed) for seed in (1, 2, 3)]
        report = summary.recovery_screen(rows, rows, summary.RECOVERY_CONTRACT)
        self.assertTrue(report['checks']['meanCompleteLossesReducedAtLeast20Percent'])
        self.assertFalse(report['checks']['atLeastTwoSeedsReduceCompleteLosses'])
        self.assertFalse(report['passed'])
        with self.assertRaisesRegex(ValueError, 'pairing'):
            summary.recovery_screen(rows, rows[::-1], summary.RECOVERY_CONTRACT)

    def test_historical_exposure_checks_coverage_stream_and_common_prefix(self):
        row = {'trainIds': ['x'], 'auxiliaryIds': {'draft': ['d'], 'coverage': ['c']}, 'validationIds': ['y'],
               'scalerTrainIds': ['x'], 'positiveWeight': [1]*4, 'supervisedCounts': {}, 'parameters': 29700,
               'history': [{'epoch': i, 'optimizerSteps': i*5, 'exposureSha256': {'exact': f'e{i}', 'draft': f'd{i}', 'coverage': f'c{i}'}}
                           for i in (1, 2)]}
        historical = copy.deepcopy(row)
        historical['history'].pop()
        self.assertEqual(summary.audit_historical_exposure(row, historical), 1)
        historical['history'][0]['exposureSha256']['coverage'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'sample exposure'):
            summary.audit_historical_exposure(row, historical)

    def test_weight_reconstruction_preserves_original_event_across_ignored_hole(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cache.npz'
            times = np.arange(0, 10, .25)
            np.savez(cache, times=times)
            row = {'id': 'x', 'sourceGroup': 'g', 'featureCaches': {'audiovisual': {'path': str(cache)}},
                   'rallies': [{'start': 1., 'end': 3.}, {'start': 5., 'end': 9.}],
                   'ignoredIntervals': [{'start': 6., 'end': 7.}]}
            actual = summary.expected_live_weighting(row, 'exact')
            self.assertEqual(actual['eligibleEventCount'], 2)
            self.assertEqual(actual['positiveSupervisedTicks'], 20)
            self.assertEqual(actual['events'][1]['positiveSupervisedTicks'], 12)
            self.assertAlmostEqual(actual['events'][0]['weightedPositiveMass'], 10)
            self.assertAlmostEqual(actual['events'][1]['weightedPositiveMass'], 10, places=5)
            expected = np.ones(len(times), dtype='<f4')
            expected[(times >= 1) & (times < 3)] = 1.25
            expected[((times >= 5) & (times < 6)) | ((times >= 7) & (times < 9))] = 20/24
            self.assertEqual(actual['liveMultiplierSha256'], hashlib.sha256(expected.tobytes()).hexdigest())
            draft = summary.expected_live_weighting(row, 'draft')
            self.assertEqual(draft['events'][0]['positiveSupervisedTicks'], 0)
            self.assertIsNone(draft['events'][0]['multiplier'])


if __name__ == '__main__':
    unittest.main()
