"""File-only operational finalization tests; never opens research outcomes."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO/'scripts/finalize-neural-transfer-sequential.py'
SPEC = importlib.util.spec_from_file_location('synthetic_transfer_finalizer', SCRIPT)
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n', encoding='utf-8')


class FinalizationTests(unittest.TestCase):
    def test_scheduler_imports_no_analysis_or_numeric_modules(self):
        code = ('import importlib.util, sys; '
                f's=importlib.util.spec_from_file_location("tested_finalizer", {str(SCRIPT)!r}); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'assert not any(n == "torch" or n == "numpy" or n == "analysis" '
                'or n.startswith(("torch.", "numpy.", "analysis.")) for n in sys.modules)')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_report_matches_actual_frozen_expression_and_serialization(self):
        contract = {'manifestSha256': 'manifest', 'evaluationPopulation': {'records': 8},
                    'groups': ['a', 'b', 'c', 'd']}
        results = [{'synthetic': index} for index in range(54)]
        source = (REPO/'analysis/neural_short_boost_transfer.py').read_text(encoding='utf-8')
        actual = F.faithful_report(contract, 'contract', results, source)
        expression = F.frozen_report_expression(source)
        expected = eval(compile(ast.fix_missing_locations(expression), '<test-original>', 'eval'),
                        {}, {'contract': contract, 'digest': 'contract', 'results': results})
        self.assertEqual(json.dumps(actual, indent=2, allow_nan=False)+'\n',
                         json.dumps(expected, indent=2, allow_nan=False)+'\n')
        self.assertIs(actual['results'], results)
        self.assertFalse(actual['protectedTestOpened'])
        self.assertFalse(actual['productionPromotionAllowed'])

    def test_report_rejects_schema_drift_or_ambiguous_frozen_assignment(self):
        contract = {'manifestSha256': 'm', 'evaluationPopulation': {'records': 8}, 'groups': []}
        altered = 'def run_study():\n    report = {"schemaVersion": 2}\n'
        with self.assertRaisesRegex(ValueError, 'differs'):
            F.faithful_report(contract, 'c', [], altered)
        ambiguous = 'def run_study():\n    report = {}\n    report = {}\n'
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            F.frozen_report_expression(ambiguous)

    def test_gate_compares_complete_artifact_inventory_and_counts(self):
        inventory = {**F.COUNTS, 'fits': [{'completed': {'sha256': 'same'}}], 'resultFiles': [],
                     'completedMetadataSetSha256': 'same'}
        gate = {**deepcopy(inventory), 'kind': 'old-gate', 'createdAt': 'prior'}
        with patch.object(F.D, 'finalization_inventory', return_value=inventory):
            self.assertEqual(F.verify_gate(gate, {}), inventory)
            changed = deepcopy(gate)
            changed['fits'][0]['completed']['sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'differs'):
                F.verify_gate(changed, {})
            changed = deepcopy(gate)
            changed['NPZArtifacts'] = 5615
            with self.assertRaisesRegex(ValueError, 'counts'):
                F.verify_gate(changed, {})

    def job_fixture(self, root, reused=False):
        job = ['exact', 'tcn', 'baseline', 3407] if reused else ['reviewed_export', 'dino_tcn', 'short_boost', 1729]
        contract = {'referenceStudy': {'path': str(root/'reference')}}
        cohort, kind, arm, seed = job
        base = (root/'reference/fits'/cohort/kind/str(seed) if reused else
                root/'study/fits'/cohort/kind/arm/str(seed))
        fits = []
        for outer in range(4):
            for name in ('inner-0', 'inner-1', 'inner-2', 'refit'):
                path = base/f'outer-{outer}'/name/'completed.json'
                save(path, {'synthetic': True})
                fits.append({'completed': F.identity(path), 'reused': reused})
        result = F.D.result_path(root/'study', job)
        save(result, {'synthetic': True})
        return job, contract, {'fits': fits, 'resultFiles': [F.identity(result)]}

    def test_per_job_gate_requires_all16_fresh_destinations_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, contract, gate = self.job_fixture(root)
            with patch.object(F, 'ROOT', root):
                F.verify_job_gate(gate, contract, job)
                missing = deepcopy(gate)
                missing['fits'].pop()
                with self.assertRaisesRegex(ValueError, 'Missing completed'):
                    F.verify_job_gate(missing, contract, job)
                changed = Path(gate['fits'][0]['completed']['path'])
                changed.write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    F.verify_job_gate(gate, contract, job)

    def test_per_job_gate_preserves_historical_fit_root_and_result_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, contract, gate = self.job_fixture(root, reused=True)
            with patch.object(F, 'ROOT', root):
                F.verify_job_gate(gate, contract, job)
                self.assertFalse((root/'study/fits').exists())
                Path(gate['resultFiles'][0]['path']).write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    F.verify_job_gate(gate, contract, job)

    def test_receipts_require_complete_grid_success_and_unchanged_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root/'finalization'
            leaf.mkdir()
            jobs = [['exact', 'tcn', 'baseline', seed] for seed in (3407, 1729)]
            plan = {'jobs': jobs, 'commonCommand': ['python', '-m', 'frozen']}
            gate = {'resultFiles': []}
            for index, job in enumerate(jobs):
                result = F.D.result_path(root/'study', job)
                save(result, {'synthetic': index})
                gate['resultFiles'].append(F.identity(result))
                log = leaf/f'job-{index:02d}.log'
                log.write_text('synthetic completion\n', encoding='utf-8')
                save(leaf/f'job-{index:02d}-completed.json', {
                    'index': index, 'job': job, 'command': F.job_command(plan, job),
                    'exitCode': 0, 'trainingPermitted': False, 'result': F.identity(result), 'log': F.identity(log)})
            with patch.object(F, 'ROOT', root):
                self.assertEqual(len(F.verify_receipts(leaf, plan, gate)), 2)
                last = leaf/'job-01-completed.json'
                receipt = F.read(last)
                receipt['exitCode'] = 1
                save(last, receipt)
                with self.assertRaisesRegex(ValueError, 'differs'):
                    F.verify_receipts(leaf, plan, gate)
                receipt['exitCode'] = 0
                save(last, receipt)
                (leaf/'job-01.log').write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    F.verify_receipts(leaf, plan, gate)
                last.unlink()
                with self.assertRaisesRegex(ValueError, 'Not all54'):
                    F.verify_receipts(leaf, plan, gate)

    def test_failed_job_preserves_evidence_and_cannot_reach_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root/'finalization'
            leaf.mkdir()
            original = root/'original-result.json'
            save(original, {'immutable': True})
            before = original.read_bytes()
            plan = {'jobs': [['exact', 'tcn', 'baseline', 3407]],
                    'commonCommand': ['python', '-m', 'frozen']}
            save(leaf/'plan.json', plan)
            with (patch.object(F, 'ROOT', root),
                  patch.object(F, 'validate_plan', return_value=(plan, {}, {})),
                  patch.object(F, 'verify_gate'), patch.object(F, 'verify_job_gate'),
                  patch.object(F, 'verify_receipts') as receipts,
                  patch.object(F.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['synthetic'])) as launch,
                  redirect_stdout(io.StringIO())):
                with self.assertRaises(subprocess.CalledProcessError):
                    F.run(leaf)
            self.assertEqual(launch.call_count, 1)
            receipts.assert_not_called()
            self.assertEqual(original.read_bytes(), before)
            failed = F.read(leaf/'driver-failed.json')
            self.assertEqual(failed['status'], 'failed')
            self.assertFalse(failed['trainingPermitted'])
            self.assertTrue((leaf/'driver-started.json').exists())
            self.assertTrue((leaf/'job-00.log').exists())
            self.assertFalse((leaf/'job-00-completed.json').exists())
            self.assertFalse((leaf/'aggregation-gate.json').exists())
            self.assertFalse((leaf/'driver-completed.json').exists())


if __name__ == '__main__':
    unittest.main()
