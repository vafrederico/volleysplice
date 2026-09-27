"""Actual registrar helper tests: masking, provenance and paired source movement."""
from analysis.private_ledger import private_value
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


proxy = load('register-neural-export-proxy-experiments')
original = load('register-neural-generalization')


def fixture():
    rows, candidates = [], []
    for group, count in [(private_value('source-group-004'), 7), (private_value('source-group-006'), 11)]:
        for index in range(count):
            rid = f'{group}-{index}'
            source = {'id': rid, 'sourceGroup': group, 'environment': 'grass', 'video': rid + '.mp4',
                      'contentSha256': 'a'*64, 'roi': None, 'durationSeconds': 20.,
                      'rallies': [{'id': 'a', 'start': 2., 'end': 6.}, {'id': 'b', 'start': 12., 'end': 14.}],
                      'ignoredIntervals': [{'start': 0., 'end': 4.}], 'gameWindow': {'start': 1., 'end': 19.},
                      'sourceFeedback': {'path': rid+'.json', 'sha256': 'b'*64}, 'derivation': {'endpointsChanged': False}}
            candidates.append(source)
            rows.append({**copy.deepcopy(source), 'labelTier': 'coverage', 'scoringPolicy': 'export-coverage',
                         'protected': False, 'eligibleRoles': ['fit', 'evaluate', 'infer'], 'consent': {'train': True},
                         'ignoredIntervals': [{'start': 8., 'end': 9.}], 'keepTargets': [{'start': 0., 'end': 17.}]})
    exact = []
    for group in ['a', 'b', 'c', 'd']:
        row = {'id': group, 'sourceGroup': group, 'environment': 'indoor', 'protected': False,
               'labelTier': 'exact', 'scoringPolicy': 'exact-core', 'rallies': [{'start': 4., 'end': 8.}],
               'eligibleRoles': ['fit', 'calibrate', 'evaluate', 'infer'], 'consent': {'train': True}}
        exact.append(row); rows.append(row)
    for rid, group, protected in [('protected', private_value('source-group-008'), True), ('september', private_value('source-group-001'), False)]:
        rows.append({'id': rid, 'sourceGroup': group, 'environment': 'indoor', 'protected': protected,
                     'labelTier': 'exact' if protected else 'coverage', 'eligibleRoles': ['evaluate', 'infer'],
                     'consent': {'train': False}})
    return {'records': rows}, {'records': candidates}, {'exactRows': exact, 'draftRows': [], 'coverageRows': rows[:7]}


class ProxyRegistrationTest(unittest.TestCase):
    def test_original_gold_untouched_and_ignored_masks_union_without_new_boundaries(self):
        base, source, _ = fixture(); before = copy.deepcopy(base)
        derived = proxy.proxy_inputs(base, source)
        self.assertEqual(base, before)
        self.assertEqual(derived['records'][18:], base['records'][18:])
        row = derived['records'][0]
        self.assertEqual(row['rallies'], source['records'][0]['rallies'])
        self.assertEqual(row['ignoredIntervals'], [{'start': 0., 'end': 4.}, {'start': 8., 'end': 9.}, {'start': 19., 'end': 20.}])
        self.assertEqual(row['rallies'][0]['start'], 2.)
        self.assertFalse(row['experimentalSupervision']['independentSemanticGold'])
        self.assertEqual(row['scoringPolicy'], 'exact-core')
        self.assertIn('calibrate', row['eligibleRoles'])

    def test_media_mismatch_rejected(self):
        base, source, _ = fixture(); source['records'][0]['video'] = 'different-source.mp4'
        with self.assertRaisesRegex(ValueError, 'media coordinates'):
            proxy.proxy_inputs(base, source)

    def test_overlapping_saved_cores_rejected(self):
        base, source, _ = fixture(); source['records'][0]['rallies'][1]['start'] = 5.
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            proxy.proxy_inputs(base, source)

    def test_protected_override_rejected(self):
        base, source, _ = fixture(); base['records'][0]['protected'] = True
        with self.assertRaisesRegex(ValueError, 'nonprotected'):
            proxy.proxy_inputs(base, source)

    def test_two_arms_pair_draws_and_move_whole_group_alternately(self):
        base, source, old = fixture()
        common = [private_value('source-group-008'), private_value('source-group-001')]
        plans = [p for p in original.split_plans(base['records'], old, set(common)) if p['variant'] == 'expanded-medium']
        derived = proxy.proxy_inputs(base, source)
        tasks = proxy.proxy_tasks({'commonEvaluationGroups': common}, plans, derived['records'])
        self.assertEqual(len(tasks), 4 * 2 * len(proxy.MODELS))
        index = {r['id']: r for r in derived['records']}
        for draw_index, plan in enumerate(plans):
            for model in proxy.MODELS:
                a, b = [t for t in tasks if t['splitSeed'] == plan['splitSeed'] and t['model'] == model]
                selected_group = [private_value('source-group-004'), private_value('source-group-006')][draw_index % 2]
                moved = {r['id'] for r in derived['records'] if r['sourceGroup'] == selected_group}
                self.assertEqual(set(b['trainIds']), set(a['trainIds']) - moved)
                self.assertEqual(set(b['calibrationIds']), set(a['calibrationIds']) | moved)
                self.assertEqual(b['selectionExportSourceGroup'], selected_group)
                self.assertEqual(a['selectionLabelPolicy'], 'exact-rallies')
                self.assertEqual(b['selectionLabelPolicy'], 'exact-and-export-rally-proxy')
                self.assertEqual(a['originalExactTrainGroups'], b['originalExactTrainGroups'])
                for task in [a, b]:
                    fit_groups = {index[r]['sourceGroup'] for r in task['trainIds']}
                    cal_groups = {index[r]['sourceGroup'] for r in task['calibrationIds']}
                    self.assertFalse(fit_groups & cal_groups)
                    self.assertFalse((fit_groups | cal_groups) & set(common))


if __name__ == '__main__':
    unittest.main()
