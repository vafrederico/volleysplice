"""Test command routing and mandatory provenance without fitting or inference."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('student_stream_v3_tests',REPO/'scripts/run-neural-student-panel-queue-v3.py')
body=importlib.util.module_from_spec(spec);spec.loader.exec_module(body)


def ref(path):return {'path':str(path),'sha256':'fixed'}


def fixture(mode='cached-required'):
    row={'taskId':'student/one','task':ref('/task.json'),'fitDirectory':'/fit',
         'physicalOwnerTaskId':'student/one','reusePlan':None}
    item={'taskId':row['taskId'],'task':row['task'],'fitDirectory':'/fit','mode':mode,
        'numericAuditPath':'/fit/inference-numerical-audit-fp32.json',
        'numericAudit':ref('/fit/inference-numerical-audit-fp32.json') if mode=='preexisting-uncached' else None,
        'preexistingCompletion':ref('/old-completed.json') if mode=='preexisting-uncached' else None,
        'executionCompanion':None if mode=='preexisting-uncached' else '/fit/student-inference-cache-execution-fp32.json'}
    plan={'tasks':[row],'numericAuditClassification':[item],'panel':ref('/panel.json'),
          'numericExecutionPlan':ref('/cache-plan.json')}
    return plan,row,item


class ClassificationTests(unittest.TestCase):
    def test_exact_unique_classification_required(self):
        plan,row,item=fixture();self.assertEqual(body.classified(plan,row),item)
        for key,value in [('task',ref('/different')),('fitDirectory','/other'),
                          ('numericAuditPath','/other.json'),('mode','optional')]:
            p=copy.deepcopy(plan);p['numericAuditClassification'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):body.classified(p,row)
        for rows in ([],[item,item]):
            p={**plan,'numericAuditClassification':rows}
            with self.assertRaises(ValueError):body.classified(p,row)

    def test_new_numeric_without_companion_cannot_pass(self):
        plan,row,_=fixture()
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'is_file',return_value=False):
            with self.assertRaises(ValueError):body.companion_gate(plan,row)

    def test_new_numeric_companion_checks_exact_all_bindings(self):
        plan,row,item=fixture();api=SimpleNamespace(verify_execution=MagicMock())
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'is_file',return_value=True),\
             patch.object(body,'identity',side_effect=ref),patch.object(body,'adapter',return_value=api):
            self.assertEqual(body.companion_gate(plan,row),ref(item['executionCompanion']))
        api.verify_execution.assert_called_once_with(Path(item['executionCompanion']),
            plan_ref=plan['numericExecutionPlan'],task_ref=row['task'],panel_ref=plan['panel'],
            fit_audit_ref=ref('/fit/fit-numerical-audit.json'),output_path=Path(item['numericAuditPath']),
            command=body.numeric_command(plan,row))

    def test_preexisting_gate_must_stay_exact_and_uncached(self):
        plan,row,item=fixture('preexisting-uncached')
        exists=lambda path:Path(path).name=='inference-numerical-audit-fp32.json'
        with patch.object(Path,'exists',exists),patch.object(body,'identity',side_effect=ref),\
             patch.object(body,'verified',side_effect=lambda r:Path(r['path'])):
            self.assertIsNone(body.companion_gate(plan,row))
            item['numericAudit']=ref('/wrong')
            with self.assertRaises(ValueError):body.companion_gate(plan,row)

    def test_companion_without_result_is_partial(self):
        plan,row,_=fixture()
        exists=lambda path:Path(path).name=='student-inference-cache-execution-fp32.json'
        with patch.object(Path,'exists',exists):
            with self.assertRaises(ValueError):body.companion_gate(plan,row)

    def test_partial_archive_prevents_registration_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit=Path(tmp)/'fit';out=fit/'inference'/'fixed'/'fp32';out.mkdir(parents=True)
            (out/'partial.npz').write_bytes(b'partial')
            row={'taskId':'one','task':ref('/one.json'),'fitDirectory':str(fit)}
            base={'tasks':[row],'panel':ref('/panel')}
            with patch.object(body.V2.V1,'gate_helpers',return_value=(None,None,None)):
                with self.assertRaisesRegex(ValueError,'Partial unreceipted'):
                    body.snapshot_classification(base,Path(tmp)/'old',0)


class DispatchTests(unittest.TestCase):
    def test_only_exact_inference_numeric_command_uses_cache(self):
        plan,row,item=fixture();original=MagicMock();reuse=SimpleNamespace(run_commands=original)
        cache=SimpleNamespace(execute=MagicMock());numeric=body.numeric_command(plan,row)
        infer=[numeric[0],str(REPO/'scripts/run-neural-generalization.py'),'infer','--task','/task.json']
        reuse_audit=[numeric[0],str(REPO/'scripts/audit-neural-generalization-reuse.py'),'--phase','inference']
        returned=body.install_dispatch(plan,reuse,cache);self.assertIs(returned,original)
        with patch.object(body,'identity',side_effect=ref),patch.object(body,'companion_gate') as gate:
            reuse.run_commands([infer,numeric,reuse_audit],Path('/log'),row['taskId'])
        self.assertEqual(original.call_args_list[0].args,([infer],Path('/log'),row['taskId']))
        self.assertEqual(original.call_args_list[1].args,([reuse_audit],Path('/log'),row['taskId']))
        cache.execute.assert_called_once_with(numeric,plan_ref=plan['numericExecutionPlan'],task_ref=row['task'],
            panel_ref=plan['panel'],fit_audit_ref=ref('/fit/fit-numerical-audit.json'),
            receipt_path=Path(item['executionCompanion']),log_path=Path('/log'))
        gate.assert_called_once_with(plan,row)

    def test_other_numeric_phase_is_not_intercepted(self):
        plan,row,_=fixture();original=MagicMock();reuse=SimpleNamespace(run_commands=original)
        cache=SimpleNamespace(execute=MagicMock());command=body.numeric_command(plan,row)
        command[command.index('--phase')+1]='fit';body.install_dispatch(plan,reuse,cache)
        reuse.run_commands([command],Path('/log'),row['taskId'])
        original.assert_called_once_with([command],Path('/log'),row['taskId']);cache.execute.assert_not_called()

    def test_mutated_inference_command_rejected_before_cache(self):
        for flag in ('--task','--fit','--panel','--precision','--fit-audit','--output'):
            plan,row,_=fixture();reuse=SimpleNamespace(run_commands=MagicMock());cache=SimpleNamespace(execute=MagicMock())
            body.install_dispatch(plan,reuse,cache);command=body.numeric_command(plan,row)
            command[command.index(flag)+1]='changed'
            with self.subTest(flag=flag),self.assertRaises(ValueError):reuse.run_commands([command],Path('/log'),row['taskId'])
            cache.execute.assert_not_called()

    def test_preexisting_or_unknown_task_cannot_enter_adapter(self):
        for mode,task_id in [('preexisting-uncached','student/one'),('cached-required','other')]:
            plan,row,_=fixture(mode);reuse=SimpleNamespace(run_commands=MagicMock());cache=SimpleNamespace(execute=MagicMock())
            body.install_dispatch(plan,reuse,cache)
            with self.assertRaises(ValueError):reuse.run_commands([body.numeric_command(plan,row)],Path('/log'),task_id)
            cache.execute.assert_not_called()

    def test_companion_precedes_acceptance_of_existing_inference(self):
        plan,row,_=fixture()
        with patch.object(body,'companion_gate',side_effect=ValueError('missing')) as gate,\
             patch.object(body.V2,'inference_complete') as original:
            with self.assertRaises(ValueError):body.inference_complete(plan,row,{},None,{})
        gate.assert_called_once();original.assert_not_called()


if __name__=='__main__':unittest.main()
