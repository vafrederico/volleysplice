import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.neural_selection_serialization import load_module
from analysis.neural_selection_identity_cache import IdentityCache

REPO = Path(__file__).resolve().parents[2]
workflow = load_module('cache_v2_routing_tests', REPO/'scripts/generalization-selection-cache-execution-v2.py')


class CacheDispatchTests(unittest.TestCase):
    def test_all_real_frozen_entrypoints_bind_before_registration(self):
        checks = workflow.validate_dispatch()
        by_action = {r['action']: r for r in checks}
        self.assertEqual(by_action['select']['callable'], 'select_task_with_correction')
        self.assertEqual(list(inspect.signature(workflow.body.old.select_task_with_correction).parameters),
            ['plan_path', 'task_path', 'fit_directory', 'output', 'historical_directory'])
        self.assertEqual(list(inspect.signature(workflow.duration.audit_selection).parameters), ['plan_path', 'selection_path', 'output'])
        self.assertEqual(list(inspect.signature(workflow.correction.audit).parameters), ['selection_path', 'selection_audit_path', 'output'])
        self.assertEqual(set(by_action), {'select', 'ordinary', 'correction', 'freeze-selections', 'evaluate',
            'audit-results', 'global-gate', 'publication-preflight'})
        for row in checks:
            self.assertTrue(Path(row['source']['path']).is_file())
            self.assertEqual(len(row['source']['sha256']), 64)

    def test_select_stage_routes_five_arguments_and_writes_disclosed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            plan_path = folder/'plan.json'; plan_path.write_text('{}')
            task = folder/'task.json'; task.write_text('{}')
            source_plan = folder/'correction.json'; source_plan.write_text('{}')
            protocol = folder/'protocol.json'; protocol.write_text('{}')
            job = {'task': workflow.io.identity(task), 'fitDirectory': str(folder/'fit')}
            plan = {'correctionPlan': workflow.io.identity(source_plan), 'historicalProtocol': workflow.io.identity(protocol)}
            output, receipt = folder/'selection.json', folder/'execution.json'
            called = []
            def saved_function(plan_path, task_path, fit_directory, output, historical_directory=None):
                # Signature is first checked against the actual frozen function.
                called.append((plan_path, task_path, fit_directory, output, historical_directory))
                workflow.io.write_new(output, {'syntheticRoutingOnly': True})
            actual = workflow.body.old.select_task_with_correction
            self.assertEqual(inspect.signature(actual), inspect.signature(saved_function))
            with patch.object(workflow, 'stage_inputs', return_value=[job['task']]), \
                 patch.object(workflow.body.old, 'select_task_with_correction', side_effect=saved_function):
                result = workflow.execute_stage(plan_path, plan, IdentityCache(), job, 'qualification-select', output, receipt)
            self.assertEqual(called, [(source_plan, task, folder/'fit', output, folder)])
            self.assertEqual(result, workflow.io.identity(receipt))
            saved = workflow.io.read(receipt)
            self.assertTrue(saved['functionBindingsRestored'])
            self.assertEqual(saved['output'], workflow.io.identity(output))


if __name__ == '__main__': unittest.main()
