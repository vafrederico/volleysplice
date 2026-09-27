"""Reject superficially similar training recipes and preserve first ownership."""
import copy
from pathlib import Path
import unittest

from analysis.neural_generalization_reuse import first_owners, fit_directory, training_recipe


class GeneralizationReuseTest(unittest.TestCase):
    def setUp(self):
        self.records = [{'id': 'a', 'labelTier': 'exact', 'sourceGroup': 'g1'},
                        {'id': 'b', 'labelTier': 'draft', 'sourceGroup': 'g2'},
                        {'id': 'c', 'labelTier': 'coverage', 'sourceGroup': 'g3'},
                        {'id': 'validation', 'labelTier': 'exact', 'sourceGroup': 'g4'}]
        self.configs = {'av-tcn': {'family': 'av', 'head': 'tcn'}}
        self.task = {'taskId': 'first', 'registration': {'path': '/nas/registration', 'sha256': 'a'*64},
            'registrationSha256': 'b'*64, 'manifest': {'path': '/nas/manifest', 'sha256': 'c'*64},
            'features': {'path': '/nas/features', 'sha256': 'd'*64}, 'model': 'av-tcn',
            'seed': 3407, 'epochs': [5, 15, 30, 60], 'lossArm': 'short_boost',
            'trainIds': ['a', 'b', 'c'], 'calibrationIds': ['validation'], 'variant': 'expanded-medium', 'splitSeed': 3407}

    def recipe(self, task=None, records=None):
        return training_recipe(task or self.task, records or self.records, self.configs)

    def test_calibration_changes_do_not_change_training_but_first_order_owns(self):
        second = {**self.task, 'taskId': 'second', 'calibrationIds': [], 'variant': 'expanded-wide-validation', 'splitSeed': 1729}
        self.assertEqual(self.recipe(), self.recipe(second))
        actual = first_owners([self.task, second], self.records, self.configs)
        self.assertEqual([r[2] for r in actual], ['first', 'first'])
        self.assertEqual([r[2] for r in first_owners([second, self.task], self.records, self.configs)], ['second', 'second'])

    def test_changed_order_seed_source_features_loss_epoch_or_tier_never_reuses(self):
        mutations = {'trainIds': ['b', 'a', 'c'], 'seed': 1729, 'epochs': [60], 'lossArm': 'baseline',
            'registrationSha256': 'e'*64, 'manifest': {'path': '/other', 'sha256': 'c'*64},
            'features': {'path': '/nas/features', 'sha256': 'f'*64}}
        for key, value in mutations.items():
            with self.subTest(key=key): self.assertNotEqual(self.recipe(), self.recipe({**self.task, key: value}))
        changed = copy.deepcopy(self.records); changed[1]['labelTier'] = 'exact'
        self.assertNotEqual(self.recipe(), self.recipe(records=changed))
        changed = copy.deepcopy(self.records); changed[-1]['sourceGroup'] = 'changed-excluded-group'
        self.assertNotEqual(self.recipe(), self.recipe(records=changed))

    def test_duplicate_unknown_or_unsupported_training_members_fail(self):
        for ids in ([], ['a', 'a'], ['missing']):
            with self.assertRaises(ValueError): self.recipe({**self.task, 'trainIds': ids})
        changed = copy.deepcopy(self.records); changed[0]['labelTier'] = 'unscored'
        with self.assertRaises(ValueError): self.recipe(records=changed)

    def test_target_path_keeps_logical_task_identity(self):
        self.assertEqual(fit_directory(Path('/nas/reg'), self.task), Path('/nas/reg/fits/expanded-medium/av-tcn/split-3407'))
        self.assertEqual(fit_directory(Path('/nas/reg'), {**self.task, 'variant': 'original-corpus'}), Path('/nas/reg/fits/av-tcn/seed-3407'))


if __name__ == '__main__': unittest.main()
