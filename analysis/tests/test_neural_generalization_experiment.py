"""Leakage and label-blind fitting probe checks for the new experiment runner."""
import unittest

from analysis.neural_generalization_experiment import MODELS, sentinel, validate_membership


class MembershipTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {'id': 'train', 'sourceGroup': 'match-a', 'labelTier': 'exact', 'protected': False, 'environment': 'indoor'},
            {'id': 'validation', 'sourceGroup': 'match-b', 'labelTier': 'exact', 'protected': False, 'environment': 'indoor'},
            {'id': 'export', 'sourceGroup': 'match-a', 'labelTier': 'coverage', 'protected': False, 'environment': 'indoor'},
            {'id': 'test', 'sourceGroup': 'match-c', 'labelTier': 'exact', 'protected': True, 'environment': 'indoor'},
        ]
        for row in self.rows:
            row.update(eligibleRoles=['fit', 'calibrate', 'evaluate', 'infer'], consent={'train': True})
        self.task = {'trainIds': ['train', 'export'], 'calibrationIds': ['validation'],
                     'commonEvaluationGroups': ['match-c']}

    def test_source_related_export_cannot_enter_calibration(self):
        self.rows[1]['sourceGroup'] = 'match-a'
        with self.assertRaisesRegex(ValueError, 'source leakage'):
            validate_membership(self.task, self.rows)

    def test_protected_or_common_panel_cannot_fit(self):
        self.task['trainIds'].append('test')
        with self.assertRaisesRegex(ValueError, 'Protected'):
            validate_membership(self.task, self.rows)
        self.rows[-1]['protected'] = False
        with self.assertRaisesRegex(ValueError, 'Common evaluation'):
            validate_membership(self.task, self.rows)

    def test_export_only_reference_cannot_calibrate_core_recall(self):
        self.rows[1]['labelTier'] = 'coverage'
        with self.assertRaisesRegex(ValueError, 'exact rally gold'):
            validate_membership(self.task, self.rows)

    def test_valid_source_partition_retains_order(self):
        mapping = validate_membership(self.task, self.rows)
        self.assertEqual(list(mapping), [r['id'] for r in self.rows])

    def test_eval_only_exact_label_cannot_calibrate(self):
        self.rows[1]['eligibleRoles'] = ['evaluate', 'infer']
        with self.assertRaisesRegex(ValueError, 'Calibration role'):
            validate_membership(self.task, self.rows)

    def test_probe_has_no_gold_or_masks_from_a_recording(self):
        for config in MODELS.values():
            probe = sentinel(config)
            self.assertEqual(probe.values.shape, (1, config.input_dimension))
            self.assertTrue(probe.valid.all())
            self.assertFalse(probe.values.any())
            self.assertEqual(probe.truth, ())
            self.assertEqual(probe.ignored, ())
            self.assertFalse(probe.targets.any())


if __name__ == '__main__':
    unittest.main()
