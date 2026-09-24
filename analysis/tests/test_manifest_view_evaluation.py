import argparse
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO/'scripts'/filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


dispatch = load('manifest_view_evaluation_tests', 'run-neural-manifest-view-evaluation.py')
final = load('manifest_view_finalization_tests', 'finalize-neural-generalization-v4.py')


def put(path, value):
    path.write_text(json.dumps(value))
    return dispatch.ref(path)


class ManifestViewEvaluationTests(unittest.TestCase):
    def test_exact_scope_includes_reused_trailing_cells_without_original_progress(self):
        expected = ['cell-'+str(i) for i in range(270)]
        reference = {'path': '/plan', 'sha256': 'a'*64}
        # The final two cells were reused after the last newly evaluated student.
        # Frozen progress would remain268; the independent scope counts all270.
        complete = {'kind': 'manifest-view-evaluation-completed-v1', 'passed': True,
            'plan': reference, 'scope': 'all', 'count': 270, 'completedCellIds': expected}
        dispatch.require_complete_scope(complete, reference, 'all', expected)
        for changed in ({**complete, 'count': 268}, {**complete, 'completedCellIds': expected[:-1]},
                        {**complete, 'completedCellIds': expected[::-1]}, {**complete, 'scope': 'nonstudent'}):
            with self.assertRaises(ValueError):
                dispatch.require_complete_scope(changed, reference, 'all', expected)

    def test_frozen_complete_gate_does_not_allow_a_missing_adapter_companion(self):
        job = {'fitDirectory': '/fit', 'precision': 'fp32'}
        worker = SimpleNamespace(completed=lambda *_: True, evaluation_call=lambda *_: (None, ['original-argv'], None))
        adapter = SimpleNamespace(verify_execution=lambda *_a, **_kw: (_ for _ in ()).throw(FileNotFoundError('companion')))
        plan = {'viewPlan': {'path': '/view'}, 'viewQualification': {'path': '/qualification'}}
        with patch.object(dispatch, 'view', return_value=adapter), patch.object(dispatch, 'outputs', return_value={}):
            with self.assertRaises(FileNotFoundError): dispatch.check_cell(worker, {}, {}, {}, job, plan)

    def test_original_incomplete_cell_never_consults_a_companion(self):
        worker = SimpleNamespace(completed=lambda *_: False)
        with patch.object(dispatch, 'view', side_effect=AssertionError('must not accept sidecar alone')):
            self.assertFalse(dispatch.check_cell(worker, {}, {}, {}, {}, {}))

    def recovery_fixture(self, root):
        queue = root/'queue'; queue.mkdir()
        fit = root/'fit'; fit.mkdir(); start = fit/'evaluation-fp32-evaluation-start.json'
        start.write_bytes(b'{"preserve": "exact failure bytes"}\n')
        plan_path = root/'plan.json'; put(plan_path, {'registered': True})
        grant_path = root/'grant.json'; put(grant_path, {'authorized': True})
        plan = {'failedObservedExit': put(root/'old-exit.json', {'exitCode': 1}),
            'failedProcess': put(root/'old-process.json', {'identity': {'pid': 999999999}}),
            'failedStart': dispatch.ref(start), 'archivePath': str(root/'archive.json'),
            'recoveryPath': str(root/'recovery.json')}
        original = {'jobs': [{'fitDirectory': str(fit), 'precision': 'fp32'}]}
        helper = SimpleNamespace(zero_outputs=lambda jobs: self.assertFalse(start.exists()))
        return plan_path, grant_path, plan, original, helper, queue, start

    def test_recovery_moves_only_the_pinned_start_with_byte_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pp, gp, plan, original, helper, queue, start = self.recovery_fixture(root)
            original_bytes = start.read_bytes()
            with patch.object(dispatch, 'N', root), patch.object(dispatch, 'ORIGINAL', queue), \
                    patch.object(dispatch, 'verify_plan', return_value=(plan, helper, None, original)), \
                    patch.object(dispatch, 'grant', return_value={}):
                dispatch.recover(argparse.Namespace(plan=pp, grant=gp))
            self.assertFalse(start.exists()); self.assertEqual(Path(plan['archivePath']).read_bytes(), original_bytes)
            dispatch.verify_recovery(pp, plan)

    def test_recovery_rejects_any_other_partial_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pp, gp, plan, original, helper, queue, start = self.recovery_fixture(root)
            (start.parent/'evaluation-fp32.json').write_text('{}')
            with patch.object(dispatch, 'N', root), patch.object(dispatch, 'ORIGINAL', queue), \
                    patch.object(dispatch, 'verify_plan', return_value=(plan, helper, None, original)), \
                    patch.object(dispatch, 'grant', return_value={}):
                with self.assertRaises(ValueError): dispatch.recover(argparse.Namespace(plan=pp, grant=gp))
            self.assertTrue(start.exists()); self.assertFalse(Path(plan['archivePath']).exists())

    def test_supervisor_records_actual_zero_even_when_completion_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'execution'
            pp = root/'plan.json'; gp = root/'grant.json'; put(pp, {}); put(gp, {})
            plan = {'outputRoots': {'all': str(output)}, 'allCellIds': ['one', 'two']}
            def popen(*_a, **_kw):
                put(output/'completed.json', {'kind': 'wrong'})
                return SimpleNamespace(pid=999999999, wait=lambda: 0)
            with patch.object(dispatch, 'verify_plan', return_value=(plan, None, None, None)), \
                    patch.object(dispatch, 'grant', return_value={}), patch.object(dispatch, 'process_identity', return_value={'pid': 999999999}), \
                    patch.object(dispatch.subprocess, 'Popen', side_effect=popen):
                with self.assertRaises(ValueError): dispatch.supervise(argparse.Namespace(plan=pp, grant=gp, scope='all'))
            actual = dispatch.read(output/'observed-exit.json')
            self.assertEqual(actual['exitCode'], 0)
            self.assertFalse(actual['scopeComplete']); self.assertIsNotNone(actual['completionValidationError'])

    def test_missing_exit_or_missing_ledger_cannot_prove_scope_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pp = root/'plan.json'; put(pp, {})
            plan = {'outputRoots': {'all': str(root)}, 'allCellIds': ['one']}
            put(root/'completed.json', {'kind': 'manifest-view-evaluation-completed-v1', 'passed': True,
                'plan': dispatch.ref(pp), 'scope': 'all', 'count': 1, 'completedCellIds': ['one']})
            with self.assertRaises(FileNotFoundError): dispatch.verify_scope_exit(pp, plan, 'all')

    def test_final_index_requires_bijection_order_and_exact_hashes(self):
        refs = [{'path': '/result/'+str(i), 'sha256': str(i)} for i in range(270)]
        index = {'evaluations': [{'result': r} for r in refs]+[{'result': {'path': '/production', 'sha256': 'p'}}]}
        final.require_index_membership(index, refs)
        for mode in ('missing', 'duplicate', 'reordered', 'changed'):
            altered = copy.deepcopy(index)
            if mode == 'missing': altered['evaluations'].pop(0)
            if mode == 'duplicate': altered['evaluations'][1] = altered['evaluations'][0]
            if mode == 'reordered': altered['evaluations'][0], altered['evaluations'][1] = altered['evaluations'][1], altered['evaluations'][0]
            if mode == 'changed': altered['evaluations'][0]['result']['sha256'] = 'changed'
            with self.assertRaises(ValueError): final.require_index_membership(altered, refs)

    def test_final_audit_preserves_all_delegated_fields_and_rejects_unlisted_changes(self):
        delegated = {'kind': 'independent-generalization-finalization-audit-v3', 'passed': True,
                     'reportContentSha256': 'a'*64, 'metrics': {'recall': .99}}
        outer = {**copy.deepcopy(delegated), 'kind': final.KIND, **{k: {} for k in final.EXTRA}}
        final.restore_delegated(outer, delegated)
        for key, value in (('passed', False), ('reportContentSha256', 'b'*64), ('metrics', {'recall': 1.}), ('extra', True)):
            with self.assertRaises(ValueError): final.restore_delegated({**outer, key: value}, delegated)

    def test_report_adds_metadata_but_preserves_all_numeric_and_download_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pp = root/'plan.json'; ap = root/'audit.json'; ip = root/'index.json'
            for path in (pp, ap, ip): put(path, {})
            content = {'inventory': [{'id': 'video'}], 'scopes': {'scope': ['video']},
                       'series': [{'recall': .99, 'precision': .9}], 'tasks': [{'id': 'model'}]}
            delegated_audit = put(root/'delegated-audit.json', {'passed': True})
            gate = {'all270': True}; execution = {'path': '/audit-execution', 'sha256': 'a'*64}
            audit = {'delegatedManifestViewAudit': delegated_audit, 'manifestViewPublicationGates': gate,
                     'manifestViewFinalAuditExecution': execution, 'reportContentSha256': final.io.canonical(content)}
            plan = {'previousFinalizationPlan': {'path': '/v3-plan'}, 'viewPlan': {'path': '/view'},
                    'viewQualification': {'path': '/qualification'}}
            def report(args): put(args.output, {**content, 'metadata': {'auditPassed': True, 'audit': delegated_audit}})
            @contextmanager
            def installed(*_a, **_kw): yield {'passed': True}
            output = root/'report.json'
            with patch.object(final, 'verify_plan', return_value=(plan, SimpleNamespace(report=report), {}, None)), \
                    patch.object(final, 'verify_final_audit', return_value=audit), patch.object(final, 'preflight', return_value=gate), \
                    patch.object(final.view, 'installed', side_effect=installed), patch.object(final.view, 'verify_execution', return_value={}):
                final.report(argparse.Namespace(plan=pp, audit=ap, index=ip, output=output))
            value = dispatch.read(output)
            self.assertEqual({k: value[k] for k in content}, content)
            self.assertEqual(value['metadata']['audit'], dispatch.ref(ap))
            self.assertIn('originalManifestReadView', value['metadata'])
            companion = dispatch.read(final.companion_path(output))
            self.assertEqual(companion['outputs']['report'], dispatch.ref(output))


if __name__ == '__main__': unittest.main()
