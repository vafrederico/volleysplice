import importlib.util
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[2]/'scripts'/'prepare-neural-human-review-probabilities.py'
SPEC = importlib.util.spec_from_file_location('review_probability_input', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProbabilityInputTests(unittest.TestCase):
    def test_rejects_retimed_shape_float64_and_nonprobability_scores(self):
        times = np.array([0., .266666666666, .5], np.float64)
        MODULE.validate_scores(np.zeros((3, 4), np.float32), times)
        for value in (np.zeros((4, 4), np.float32), np.zeros((3, 4), np.float64),
                      np.full((3, 4), np.nan, np.float32), np.full((3, 4), 1.01, np.float32)):
            with self.assertRaises(ValueError):
                MODULE.validate_scores(value, times)

    def test_rejects_outer_group_or_recording_leakage(self):
        rows = {'a': {'sourceGroup': 'held'}, 'b': {'sourceGroup': 'train'}}
        valid = {'validationGroups': ['held'], 'trainGroups': ['train'], 'auxiliaryGroups': {'draft': ['aux']},
                 'validationIds': ['a'], 'trainIds': ['b'], 'auxiliaryIds': {'draft': ['c']}}
        self.assertEqual(MODULE.validate_membership(valid, 'held', rows), {'a'})
        for field, value in [('trainGroups', ['held']), ('auxiliaryGroups', {'draft': ['held']}),
                             ('trainIds', ['a']), ('auxiliaryIds', {'draft': ['a']}), ('validationIds', ['b'])]:
            with self.assertRaises(ValueError):
                MODULE.validate_membership({**valid, field: value}, 'held', rows)

    def test_exact_audited_payload_required(self):
        cell = {'cohort': 'reviewed_export', 'kind': 'tcn', 'lossArm': 'short_boost', 'seed': 3407, 'predictions': [1]}
        MODULE.verify_reference_result(cell, {'results': [cell]})
        for rows in ([], [cell, cell], [{**cell, 'predictions': [2]}]):
            with self.assertRaises(ValueError):
                MODULE.verify_reference_result(cell, {'results': rows})

    def test_semantic_label_identity_retains_tags_but_ignores_free_text(self):
        full = [{'start': 1, 'end': 2, 'tags': ['short'], 'notes': 'description'}]
        normalized = [{'start': 1., 'end': 2., 'tags': ['short']}]
        self.assertEqual(MODULE.interval_identity(full, include_tags=True),
                         MODULE.interval_identity(normalized, include_tags=True))
        self.assertNotEqual(MODULE.interval_identity(full, include_tags=True),
                            MODULE.interval_identity([{'start': 1, 'end': 2, 'tags': []}], include_tags=True))
        self.assertEqual(MODULE.interval_identity([{'start': 1, 'end': 2, 'reason': 'ignored'}], include_tags=False),
                         MODULE.interval_identity([{'start': 1., 'end': 2.}], include_tags=False))


if __name__ == '__main__':
    unittest.main()
