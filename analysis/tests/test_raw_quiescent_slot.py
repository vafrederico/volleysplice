import copy
import importlib.util
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('quiescent_test', REPO/'scripts/manage-neural-raw-quiescent-slot.py')
body = importlib.util.module_from_spec(spec); spec.loader.exec_module(body)


def owner():
    return {'pid':107405, 'startTimeTicks':123, 'bootId':'boot', 'command':'owned command', 'state':'S'}


class QuiescentSlotTests(unittest.TestCase):
    def test_descendants_include_grandchildren(self):
        self.assertEqual(body.descendant_ids(7,{7:1,8:7,9:8,10:9,11:1}),[8,9,10])

    def test_process_identity_mutations_are_rejected(self):
        for key in body.IDENTITY_KEYS:
            changed=owner();changed[key]='different'
            with self.subTest(key=key),self.assertRaises(ValueError):body.same_process(changed,owner())

    def test_stop_precedes_quiescent_check_and_uses_only_owned_pid(self):
        calls=[];states=[owner(),{**owner(),'state':'T'}]
        result=body.stop_and_verify(owner(),lambda pid:states.pop(0),
            lambda pid:(calls.append(('children',pid)) or []),lambda pid,sig:calls.append(('signal',pid,sig)))
        self.assertEqual(calls,[('signal',107405,signal.SIGSTOP),('children',107405)])
        self.assertEqual(result['state'],'T')

    def test_child_spawn_race_fails_after_stop_without_resume_or_termination(self):
        calls=[];states=[owner(),{**owner(),'state':'T'}]
        with self.assertRaisesRegex(ValueError,'zero descendants'):
            body.stop_and_verify(owner(),lambda pid:states.pop(0),lambda pid:[1234],lambda pid,sig:calls.append((pid,sig)))
        self.assertEqual(calls,[(107405,signal.SIGSTOP)])

    def test_not_stopped_or_wrong_owner_is_not_a_slot(self):
        for actual,children in [(owner(),[]),({**owner(),'state':'T','startTimeTicks':124},[]),({**owner(),'state':'T'},[1])]:
            with self.assertRaises(ValueError):body.validate_stopped(owner(),actual,children)

    def test_unknown_wsl_gpu_pid_is_not_fabricated(self):
        self.assertEqual(body.known_gpu_pids({'returnCode':0,'stdout':'N/A, python, N/A\n'}),set())
        self.assertEqual(body.known_gpu_pids({'returnCode':0,'stdout':'107405, python, 22 MiB\n'}),{107405})

    def test_pause_header_rejects_exit_claim_narrowing_and_live_state(self):
        plan={'parent':owner(),'wrapper':{**owner(),'pid':107249}}
        proof={'kind':'cached-raw-quiescent-slot-pause-v1','passed':True,'alivePaused':True,
            'releasedForTemporaryReplacement':True,'predecessorExited':False,'proxyLaunchAbsent':True,
            'source':{'path':'source','sha256':'hash'},'processIdentity':plan['parent'],'wrapperIdentity':plan['wrapper'],
            'firstThreeStageProofs':[{'stage':stage,'checks':[{}]*count} for stage,count in
                [('proxy-av',16),('original-nonstudent',15),('random-nonstudent',48)]],
            'missingProxyPrerequisites':[{}],'afterStop':{'process':{**owner(),'state':'T'},'descendants':[]}}
        with patch.object(body,'identity',return_value=proof['source']):
            body.validate_pause_header(proof,plan)
            mutations=[('predecessorExited',True),('alivePaused',False),('missingProxyPrerequisites',[]),('firstThreeStageProofs',proof['firstThreeStageProofs'][:2])]
            for key,value in mutations:
                changed=copy.deepcopy(proof);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):body.validate_pause_header(changed,plan)
            changed=copy.deepcopy(proof);changed['afterStop']['descendants']=[9]
            with self.assertRaises(ValueError):body.validate_pause_header(changed,plan)

    def test_resume_requires_correct_successful_exited_fit_and_no_descendants(self):
        ref={'path':'identities','sha256':'hash'}
        plan={'fitPredecessor':{'bootId':'boot'},'processIdentities':ref}
        proof={'passed':True,'exitCode':0,'exitedPids':[97282],'bootId':'boot','processIdentities':ref,'allDescendantsExited':True}
        body.validate_fit_exit(proof,plan,lambda pid:False)
        for key,value in [('exitCode',1),('exitedPids',[]),('bootId','other'),('allDescendantsExited',False),('processIdentities',{})]:
            with self.subTest(key=key),self.assertRaises(ValueError):body.validate_fit_exit({**proof,key:value},plan,lambda pid:False)
        with self.assertRaises(ValueError):body.validate_fit_exit(proof,plan,lambda pid:True)


if __name__=='__main__':unittest.main()
