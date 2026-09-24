"""Mutations that the independent calibration gate must reject."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from analysis import neural_recall_sweep as sweep
from analysis.schema import Interval

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('selection_audit', ROOT/'scripts/audit-neural-generalization-selection.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
helper = audit.module('independent_selection_test_oracle', ROOT/'scripts/audit-neural-short-boost-intervals.py')


def example():
    times = np.arange(0, 20., .25)
    row = sweep.SweepExample('video', 'held-group', 20., times, np.ones(len(times), bool),
          (Interval(3., 6.), Interval(9., 12.), Interval(17., 19.)), (Interval(5., 5.5),))
    values = np.zeros((len(times), 4), np.float32)
    values[:, 0] = .1
    values[(times>=3)&(times<6), 0] = .8
    values[(times>=9)&(times<12), 0] = .8
    values[:, 3] = values[:, 0]
    return row, {epoch: {'video': values.copy()} for epoch in sweep.EPOCHS}


class GeneralizationSelectionAuditTest(unittest.TestCase):
    def test_initial_student_registration_is_in_distilled_run_directory(self):
        root = Path('/nas/recall-distillation')
        self.assertEqual(audit.initial_student_registration_path(root),
                         root/'distilled-mobile-v1/preregistration.json')

    def test_all_192_metrics_replayed_and_changed_metric_rejected(self):
        row, scores = example()
        candidates = sweep.build_candidate_table([row], scores)
        selection = {'candidates': candidates, 'floors': sweep.select_floors(candidates)}
        measured = audit.audit_candidates(selection, [row], scores, helper)
        self.assertEqual(len(measured), 192)
        self.assertTrue(all('P_pad' in m for m in measured))
        changed = copy.deepcopy(selection)
        changed['candidates'][0]['innerR_core'] += .01
        with self.assertRaisesRegex(ValueError, 'candidate recall'):
            audit.audit_candidates(changed, [row], scores, helper)

    def test_100_floor_no_tolerance_and_stable_tie(self):
        row, scores = example()
        candidates = sweep.build_candidate_table([row], scores)
        for candidate in candidates:
            candidate['innerR_core'] = np.nextafter(1., 0.)
            candidate['innerF1_padP_coreR'] = .8
        selection = {'candidates': candidates, 'floors': sweep.select_floors(candidates)}
        audit.audit_floors(selection)
        self.assertIsNone(selection['floors'][-1]['selected'])
        forged = copy.deepcopy(selection)
        forged['floors'][-1].update(selected=candidates[0], feasible=True, eligibleCandidateCount=192)
        with self.assertRaisesRegex(ValueError, 'Strict recall'):
            audit.audit_floors(forged)
        forged = copy.deepcopy(selection)
        # Equal metrics cannot change the deterministic candidate-order tie.
        forged['floors'][0]['selected'] = candidates[1]
        with self.assertRaisesRegex(ValueError, 'Strict recall'):
            audit.audit_floors(forged)

    def test_auxiliary_same_group_cannot_enter_calibration(self):
        def source(key, group, tier):
            return {'id': key, 'sourceGroup': group, 'protected': False, 'environment': 'grass',
                    'consent': {'train': True}, 'eligibleRoles': ['fit', 'calibrate'],
                    'labelTier': tier, 'scoringPolicy': 'exact-core'}
        rows = [source('train', 'a', 'exact'), source('aux', 'held', 'draft'), source('cal', 'held', 'exact')]
        task = {'trainIds': ['train', 'aux'], 'calibrationIds': ['cal'], 'commonEvaluationGroups': []}
        with self.assertRaisesRegex(ValueError, 'Source group leaks'):
            audit.check_membership(task, rows)
        rows[1]['sourceGroup'] = 'b'
        audit.check_membership(task, rows)
        rows[1]['protected'] = True
        with self.assertRaisesRegex(ValueError, 'authorization'):
            audit.check_membership(task, rows)

    def test_saved_proxy_calibration_keeps_full_timeline_valid(self):
        # This test verifies the independent constructor without feature I/O.
        class FakeEvidence:
            def bind(self, reference):
                return reference['path']
        from unittest.mock import patch
        class Archive(dict):
            def __enter__(self): return self
            def __exit__(self, *unused): return False
        row = {'id': 'proxy', 'sourceGroup': 'exports', 'durationSeconds': 3., 'contentSha256': 'a'*64,
               'rallies': [{'start': .75, 'end': 2.}], 'ignoredIntervals': [{'start': 0., 'end': 1.}],
               'experimentalSupervision': {'independentSemanticGold': False}}
        features = {'records': [{'recordingId': 'proxy', 'sourceGroup': 'exports', 'contentSha256': 'a'*64,
                                'features': {'audiovisual': {'path': 'unused'}}}]}
        with patch.object(audit.np, 'load', return_value=Archive(times=np.arange(0,3.,.25))):
            result, = audit.calibration_examples({'calibrationIds': ['proxy']}, {'records': [row]}, features, FakeEvidence())
        self.assertTrue(result.valid.all())
        self.assertEqual(result.truth[0].start, .75)
        self.assertEqual(result.label_policy, 'export-rally-proxy-selection')
        self.assertEqual(result.ignored[0].end, 1.)


if __name__ == '__main__':
    unittest.main()
