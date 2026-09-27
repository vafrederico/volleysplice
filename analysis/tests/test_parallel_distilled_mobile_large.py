import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/parallel-distilled-mobile-large.py'
SPEC = importlib.util.spec_from_file_location('parallel_distilled_large', SCRIPT)
parallel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel)


class ParallelDistilledLargeTests(unittest.TestCase):
    def test_static_shards_are_disjoint_and_cover_only_remaining_tasks(self):
        assigned = parallel.assigned_tasks(24, [0, 1, 2, 3, 4], 2)
        self.assertEqual(assigned, [list(range(5, 24, 2)), list(range(6, 24, 2))])
        self.assertFalse(set(assigned[0]) & set(assigned[1]))
        self.assertEqual(sorted(assigned[0] + assigned[1]), list(range(5, 24)))

    def test_third_worker_and_duplicate_completed_membership_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'two workers'):
            parallel.assigned_tasks(24, [], 3)
        with self.assertRaisesRegex(ValueError, 'completed task'):
            parallel.assigned_tasks(24, [0, 0], 2)

    def test_task_view_preserves_exact_task_and_other_contract_fields(self):
        plan = {'tasks': [{'variant': 'a', 'seed': 3407, 'trainIds': ['x']},
                          {'variant': 'b', 'seed': 3407, 'trainIds': ['y']}],
                'sourceCode': {'unchanged': {'sha256': 'a' * 64}}, 'floorsPercent': [99]}
        original = copy.deepcopy(plan)
        selected = parallel.filtered_plan(plan, 1)
        self.assertEqual(selected['tasks'], [original['tasks'][1]])
        self.assertEqual(selected['sourceCode'], original['sourceCode'])
        self.assertEqual(selected['floorsPercent'], [99])
        selected['tasks'][0]['trainIds'].append('cannot-mutate-original')
        self.assertEqual(plan, original)

    def test_unknown_task_indices_fail_closed(self):
        for index in (-1, 1, True):
            with self.assertRaisesRegex(ValueError, 'Unknown task'):
                parallel.filtered_plan({'tasks': [{}]}, index)

    def receipt(self):
        return dict(kind=parallel.EXECUTION_KIND, workerCount=2,
                    plan={'sha256': 'same'}, executor={'sha256': 'same'},
                    validator={'sha256': 'same'}, output=str(Path('synthetic-root').resolve()),
                    execution=str(Path('synthetic-execution').resolve()), registeredTaskCount=4,
                    initialCompletedTaskIndexes=[0], assignments=[[1, 3], [2]])

    def test_execution_contract_rejects_overlap_and_changed_plan(self):
        root, execution = Path('synthetic-root'), Path('synthetic-execution')
        with patch.object(parallel, 'identity', return_value={'sha256': 'same'}):
            receipt = self.receipt()
            parallel.require_execution(receipt, root, execution)
            receipt['assignments'] = [[1, 3], [1, 2]]
            with self.assertRaisesRegex(ValueError, 'ownership'):
                parallel.require_execution(receipt, root, execution)
            receipt = self.receipt()
            receipt['plan'] = {'sha256': 'changed'}
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                parallel.require_execution(receipt, root, execution)

    def test_no_pending_work_has_empty_disjoint_assignments(self):
        self.assertEqual(parallel.assigned_tasks(4, [0, 1, 2, 3], 2), [[], []])

    def test_only_domain_excursion_is_corrected_and_raw_value_preserved(self):
        raw = [{'epoch': 15, 'decoder': {'enter': .2},
                'innerR_core': math.nextafter(1., math.inf), 'innerF1_padP_coreR': .74}]
        bounded, corrections = parallel.bounded_candidates(raw)
        self.assertEqual(bounded[0]['innerR_core'], 1.)
        self.assertEqual(raw[0]['innerR_core'], math.nextafter(1., math.inf))
        self.assertEqual(corrections[0]['field'], 'innerR_core')
        self.assertEqual(corrections[0]['excursion'], math.ulp(1.))

    def test_inside_domain_and_near99_values_are_unchanged(self):
        values = [0., math.nextafter(1., 0.), math.nextafter(.99, 0.), .99, 1.]
        raw = [{'epoch': 15, 'decoder': {}, 'innerR_core': value, 'innerF1_padP_coreR': value}
               for value in values]
        bounded, corrections = parallel.bounded_candidates(raw)
        self.assertEqual(bounded, raw)
        self.assertEqual(corrections, [])

    def test_large_excursions_and_nonfinite_metrics_fail_closed(self):
        for value in (1. + 2 * parallel.ROUNDING_TOLERANCE, -2 * parallel.ROUNDING_TOLERANCE,
                      float('nan'), float('inf')):
            raw = [{'epoch': 15, 'decoder': {}, 'innerR_core': value, 'innerF1_padP_coreR': .7}]
            with self.assertRaises(ValueError):
                parallel.bounded_candidates(raw)


if __name__ == '__main__':
    unittest.main()
