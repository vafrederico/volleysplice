"""Independent readiness/resource mutations; never execute inference or signals."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('student_stream_tests',REPO/'scripts/run-neural-student-panel-queue-v2.py')
body=importlib.util.module_from_spec(spec);spec.loader.exec_module(body)


def ref(path):return {'path':str(path),'sha256':'fixed'}


def row(key,owner=None):
    owner=owner or key
    return {'taskId':key,'physicalOwnerTaskId':owner,'task':ref('/tasks/'+key+'.json'),
        'physicalOwnerTask':ref('/tasks/'+owner+'.json'),'fitDirectory':'/fits/'+key,'reusePlan':ref('/reuse.json')}


class StreamingReadinessTests(unittest.TestCase):
    def test_owner_waits_every_registered_fit_descendant_only(self):
        owner=row('owner');child=row('copy','owner');other=row('other');second=row('copy2','owner')
        plan={'tasks':[owner,child,other,second]}
        self.assertEqual(body.dependency_rows(plan,owner),[owner,child,second])
        self.assertEqual(body.dependency_rows(plan,child),[child])
        with self.assertRaises(ValueError):body.dependency_rows({'tasks':[child,owner]},owner)
        with self.assertRaises(ValueError):body.dependency_rows({'tasks':[other]},owner)

    def test_each_resource_floor_is_inclusive_and_independently_required(self):
        baseline={'linuxAvailableBytes':10*body.GIB,'windowsPhysicalFreeBytes':3*body.GIB,
                  'windowsCFreeBytes':20*body.GIB,'gpuFreeMiB':5120}
        self.assertTrue(body.resource_passes(baseline))
        for key in baseline:
            with self.subTest(key=key):self.assertFalse(body.resource_passes({**baseline,key:baseline[key]-1}))

    def fixture(self,copied=False):
        r=row('copy','owner') if copied else row('owner')
        folder=Path(r['fitDirectory']);task={'taskId':r['taskId'],'model':'distilled-mobile-tcn'}
        numeric={'taskId':r['taskId'],'checkpointEpochs':[5,15,30,60],
                 'completed':ref('/temporal.json'),'student':{'passed':True}}
        copied_gate={'plan':r['reusePlan'],'sourceTask':r['physicalOwnerTask'],
                     'targetNumericalAudit':ref(folder/'fit-numerical-audit.json')}
        documents={r['task']['path']:task,str(folder/'fit-result.json'):{'temporal':numeric['completed']},
                   str(folder/'selection.json'):{'task':r['task']}}
        correction=SimpleNamespace(correction_gate=MagicMock(),verify_closure=MagicMock())
        reuse=SimpleNamespace(check_gate=MagicMock(side_effect=lambda path,*_:copied_gate if Path(path).name=='reuse-audit.json' else numeric))
        helpers=(correction,reuse,MagicMock(return_value={'verified':True}))
        return r,numeric,copied_gate,documents,helpers

    def preflight(self,fixture,ready=True):
        r,numeric,copied,documents,helpers=fixture
        with patch.object(body,'ready_documents',return_value=ready),patch.object(body,'verified',side_effect=lambda r:Path(r['path'])),\
             patch.object(body,'identity',side_effect=ref),patch.object(body,'read',side_effect=lambda p:documents[str(p)]):
            return body.task_preflight({},r,helpers)

    def test_missing_task_gate_does_not_reach_any_numeric_dispatch(self):
        f=self.fixture();self.assertIsNone(self.preflight(f,ready=False))
        f[-1][1].check_gate.assert_not_called();f[-1][0].correction_gate.assert_not_called()

    def test_complete_task_binds_numeric_evidence_and_all_selection_gates(self):
        f=self.fixture();result=self.preflight(f)
        self.assertEqual(result['task'],f[0]['task'])
        f[-1][0].verify_closure.assert_any_call(f[1],{})
        f[-1][0].correction_gate.assert_called_once();f[-1][2].assert_called_once()

    def test_wrong_epoch_or_student_fit_cannot_be_ready(self):
        for field,value in [('checkpointEpochs',[60]),('taskId','another'),('student',None),('completed',ref('/wrong'))]:
            f=self.fixture();f[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.preflight(f)

    def test_copied_task_requires_its_exact_plan_owner_and_numeric_gate(self):
        f=self.fixture(True);self.assertIsNotNone(self.preflight(f)['fitReuseAudit'])
        for key in ('plan','sourceTask','targetNumericalAudit'):
            f=self.fixture(True);f[2][key]=ref('/wrong')
            with self.subTest(key=key),self.assertRaises(ValueError):self.preflight(f)

    def test_unfinished_inference_is_never_completion(self):
        f=self.fixture()
        with patch.object(Path,'exists',return_value=False):
            self.assertIsNone(body.inference_complete({},f[0],{},f[-1]))
        f[-1][1].check_gate.assert_not_called()


class StreamingSlotTests(unittest.TestCase):
    def fixture(self):
        plan={'source':ref('/source'),'resourceQualification':ref('/qualification')}
        bindings={'plan':ref('/plan'),'slotProof':ref('/slot'),'predecessorIdentity':ref('/identity')}
        predecessor={'pid':97282,'bootId':'same-boot'}
        receipt={'slot':'root-nonstudent-fit','requiredExitedPids':[97282],'released':True,
                 'allDescendantsExited':True,'exitCode':0,'processIdentity':bindings['predecessorIdentity']}
        grant={'kind':'streaming-student-panel-execution-grant-v2','source':plan['source'],**bindings,
            'resourceQualification':plan['resourceQualification'],'slotMode':'exited','requiredExitedPids':[97282],
            'cpuAffinity':[2,3],'threads':2,'nice':10,'maxLongLivedGpuWorkers':3,'maxImageWorkers':2,
            'imageEncoderConcurrency':1,'numericalRecipesChanged':False,'evaluationMetricsAllowed':False,
            'activeWorkerInventoryComplete':True,'activeWorkerInventory':[]}
        return plan,grant,receipt,predecessor,bindings

    def validate(self,values,exists=False):
        return body.validate_slot(*values,exists=lambda pid:exists,boot_id='same-boot')

    def test_successful_exit_is_required_and_three_worker_contract_cannot_relax(self):
        self.assertEqual(self.validate(self.fixture())['slotMode'],'exited')
        with self.assertRaises(ValueError):self.validate(self.fixture(),exists=True)
        for key,value in [('exitCode',1),('allDescendantsExited',False),('released',False),('requiredExitedPids',[]),('processIdentity',ref('/other'))]:
            v=self.fixture();v[2][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(v)
        for key,value in [('maxLongLivedGpuWorkers',4),('maxImageWorkers',3),('evaluationMetricsAllowed',True),('activeWorkerInventoryComplete',False)]:
            v=self.fixture();v[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(v)

    def test_wrong_predecessor_and_boot_are_not_a_released_slot(self):
        for key,value in [('pid',123),('bootId','another')]:
            v=self.fixture();v[3][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(v)

    def test_worker_inventory_cannot_hide_fourth_gpu_or_third_image_worker(self):
        for active in ([{'usesImageEncoder':False}]*3,[{'usesImageEncoder':True}]*2):
            v=self.fixture();v[1]['activeWorkerInventory']=active
            with self.assertRaises(ValueError):self.validate(v)

    def recovery_fixture(self):
        plan,grant,_,_,bindings=self.fixture()
        pids=[97282,107405,107249,104269]
        predecessor={'kind':'precision-raw-prerequisite-process-identities-v1','bootId':'prior-boot',
            'processes':[{'pid':pid,'startTimeTicks':123,'bootId':'prior-boot','command':'frozen-worker'}
                         for pid in (97282,107405,107249)]}
        receipt={'kind':'generalization-interruption-recovery-observation-v1','bootId':'same-boot',
            'priorObservedExitCode':None,'allListedOwnedProcessesAbsent':True,'newEvaluationOutcomesOpened':False,
            'frozenNumericalArtifactsModified':False,'knownProcessIdentityRecord':bindings['predecessorIdentity'],
            'priorProgressSnapshot':ref('/prior-progress'),'ownedProcessObservations':[{'pid':p,'exists':False} for p in pids]}
        grant.update({'slotMode':'recovery','requiredExitedPids':[], 'requiredAbsentPriorPids':pids,
                      'priorObservedExitCode':None,'priorSuccessfulExitClaimed':False})
        return plan,grant,receipt,predecessor,bindings

    def recovery_validate(self,values,exists=False):
        with patch.object(body,'verified',side_effect=lambda reference:Path(reference['path'])):
            return body.validate_slot(*values,mode='recovery',exists=lambda pid:exists,boot_id='same-boot')

    def test_recovery_accepts_verified_absence_without_inventing_exit_zero(self):
        result=self.recovery_validate(self.recovery_fixture())
        self.assertEqual(result['slotMode'],'recovery')
        self.assertIsNone(result['validation']['priorObservedExitCode'])
        self.assertTrue(result['validation']['freshAbsenceRechecked'])

    def test_recovery_rejects_changed_observation_and_successful_exit_claims(self):
        for field,value in [('priorObservedExitCode',0),('allListedOwnedProcessesAbsent',False),
                ('bootId','different-current-boot'),('newEvaluationOutcomesOpened',True),
                ('frozenNumericalArtifactsModified',True),('knownProcessIdentityRecord',ref('/different'))]:
            values=self.recovery_fixture();values[2][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.recovery_validate(values)
        for field,value in [('priorObservedExitCode',0),('priorSuccessfulExitClaimed',True),('requiredExitedPids',[97282])]:
            values=self.recovery_fixture();values[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.recovery_validate(values)

    def test_recovery_rejects_live_or_unobserved_predecessors(self):
        with self.assertRaises(ValueError):self.recovery_validate(self.recovery_fixture(),exists=True)
        values=self.recovery_fixture();values[2]['ownedProcessObservations'].pop()
        with self.assertRaises(ValueError):self.recovery_validate(values)
        values=self.recovery_fixture();values[2]['ownedProcessObservations'][0]['exists']=True
        with self.assertRaises(ValueError):self.recovery_validate(values)
        values=self.recovery_fixture();values[3]['processes'][0]['startTimeTicks']=0
        with self.assertRaises(ValueError):self.recovery_validate(values)


if __name__=='__main__':unittest.main()
