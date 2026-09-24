import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('combination_runner_tests',
    Path(__file__).resolve().parents[2]/'scripts/run-neural-production-combinations.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class RunnerTests(unittest.TestCase):
    def test_exact_registered_partition(self):
        rows = r.configurations()
        self.assertEqual(len(rows), 107)
        self.assertEqual(len({x['id'] for x in rows}), 107)
        self.assertEqual(sum(1 if x['family'] == 'production' else 3 for x in rows), 303)
        self.assertEqual(sum(x['family'] == 'production' for x in rows), 9)
        for anchor in r.ANCHORS:
            self.assertEqual(sum(x['anchor'] == anchor and x['family'] != 'production' for x in rows), 29)

    def fixture(self):
        record = {'id': 'r', 'sourceGroup': 'g', 'durationSeconds': 30, 'rallies': [], 'ignoredIntervals': []}
        production = {'r': {'cores': {
            'productionDefault': [[1, 4]], 'shippedUnion': [[1, 4], [8, 10]],
            'shippedPrevious': [[1, 4]], 'shippedV2': [[8, 10]],
            'refitUnion': [[1, 4]], 'refitPrevious': [[1, 4]], 'refitV2': []}}}
        neural = {'records': [record], 'predictions': {'n': {'1': {'r': []}, '2': {'r': [[1, 4]]}},
                                                      'd': {'1': {'r': [[8, 9]]}, '2': {'r': [[20, 21]]}}}}
        return production, neural

    def test_refit_anchor_uses_refit_source_tags(self):
        p, n = self.fixture()
        recipe = {'family': 'guarded_trim', 'anchor': 'refitUnion', 'neuralIds': ['n']}
        self.assertEqual(r.predictions_for(recipe, 1, p, n)['r'], [])
        recipe['anchor'] = 'productionDefault'
        self.assertEqual(r.predictions_for(recipe, 1, p, n)['r'], [[1., 4.]])

    def test_pairing_uses_requested_same_seed(self):
        p, n = self.fixture()
        recipe = {'family': 'nn_union', 'anchor': None, 'neuralIds': ['n', 'd']}
        self.assertEqual(r.predictions_for(recipe, 1, p, n)['r'], [[8., 9.]])
        self.assertEqual(r.predictions_for(recipe, 2, p, n)['r'], [[1., 4.], [20., 21.]])

    def test_standalone_preserves_touching_event_boundaries(self):
        p, n = self.fixture()
        n['predictions']['n']['1']['r'] = [[1, 2], [2, 3]]
        recipe = {'family': 'neural', 'anchor': None, 'neuralIds': ['n']}
        self.assertEqual(r.predictions_for(recipe, 1, p, n)['r'], [[1., 2.], [2., 3.]])

    def test_gold_metadata_tolerance_does_not_drop_tags(self):
        a = {'id': 'r', 'sourceGroup': 'g', 'rallies': [{'start': 1, 'end': 2, 'tags': ['ace'], 'note': 'x'}]}
        b = {'id': 'r', 'sourceGroup': 'g', 'rallies': [{'start': 1, 'end': 2, 'tags': ['ace']}]}
        self.assertTrue(r.same_labels(a, b))
        b['rallies'][0]['tags'] = []
        self.assertFalse(r.same_labels(a, b))

    def test_source_report_binding_detects_changed_predictions(self):
        result = {'cohort': 'exact', 'kind': 'tcn', 'lossArm': 'baseline', 'seed': 1,
                  'predictions': [[1, 2]], 'primary': .9}
        r.verify_reference_result(result, {'results': [result]})
        with self.assertRaises(ValueError):
            r.verify_reference_result({**result, 'predictions': [[2, 3]]}, {'results': [result]})
        with self.assertRaises(ValueError):
            r.verify_reference_result(result, {'results': [result, result]})

    def test_binding_requires_exact_path_and_hash(self):
        ref = {'path': 'x', 'sha256': 'a'}
        self.assertTrue(r.contains_binding({'inputs': [ref]}, ref))
        self.assertFalse(r.contains_binding({'inputs': [{'path': 'x', 'sha256': 'b'}]}, ref))


if __name__ == '__main__':
    unittest.main()
