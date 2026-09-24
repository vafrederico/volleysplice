import copy
from contextlib import ExitStack
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2]/'scripts/run-neural-nonstudent-evaluation.py'
spec = importlib.util.spec_from_file_location('nonstudent_schedule_test', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NonstudentScheduleTests(unittest.TestCase):
    def fixture(self):
        jobs, tasks = [], {}
        for model in ('av-tcn', 'mobile-tcn', 'av-transformer', 'dino-tcn', 'dino-transformer', 'distilled-mobile-tcn'):
            for draw in range(27):
                key = model+'/'+str(draw)
                tasks[key] = {'model': model}
                for precision in ('fp32', 'fp16', 'int8') if model.startswith('dino-') else ('fp32',):
                    jobs.append({'cellId': key+'/'+precision, 'precision': precision,
                                 'task': {'path': key, 'sha256': key}})
        return jobs, lambda r: tasks[r['sha256']]

    def test_exact_registered_partition_and_stable_order(self):
        jobs, load = self.fixture()
        included, excluded = m.partition(jobs, load)
        self.assertEqual(len(included), 243)
        self.assertEqual(len(excluded), 27)
        self.assertTrue(all(load(j['task'])['model'] != 'distilled-mobile-tcn' for j in jobs if j['cellId'] in included))
        self.assertEqual(included, [j['cellId'] for j in jobs if j['cellId'] not in excluded])

    def test_missing_duplicate_student_and_precision_scope_fail_closed(self):
        jobs, load = self.fixture()
        for altered in (jobs[:-1], jobs[:-1]+[jobs[0]]):
            with self.assertRaises(ValueError):
                m.partition(altered, load)
        with self.assertRaises(ValueError):
            m.partition(jobs, lambda r: {'model': 'av-tcn'})
        changed = copy.deepcopy(jobs)
        changed[0]['precision'] = 'int8'
        with self.assertRaisesRegex(ValueError, 'precision'):
            m.partition(changed, load)

    def test_partial_start_receipt_prevents_dispatch_without_reading_outcomes(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            m.zero_outputs([{'fitDirectory': str(folder/'not-yet-trained'), 'precision': 'fp32'}])
            jobs = [{'fitDirectory': str(folder), 'precision': 'fp32'}]
            m.zero_outputs(jobs)
            (folder/'evaluation-fp32-evaluation-start.json').write_text('preserved incomplete')
            with self.assertRaisesRegex(ValueError, 'partial output'):
                m.zero_outputs(jobs)
            self.assertEqual((folder/'evaluation-fp32-evaluation-start.json').read_text(), 'preserved incomplete')

    def stop_fixture(self, root):
        path = root/'plan.json'; path.write_text('{}')
        old_launch = {'path': 'original-launch', 'sha256': 'a'*64}
        old_process = {'path': 'original-process', 'sha256': 'b'*64}
        plan = {'originalIdentity': {'pid': 999999999, 'processStartTimeTicks': 123, 'bootId': 'boot'},
                'originalLaunch': old_launch, 'originalProcess': old_process}
        actual = root/'exit.json'
        actual.write_text(json.dumps({'kind': 'observed-cpu-worker-exit-v1', 'exitCode': 0,
                                      'launch': old_launch, 'process': old_process}))
        events = root/'events.jsonl'
        events.write_text(json.dumps({'stage': 'worker-stop', 'completed': 0})+'\n')
        stop = {'kind': 'observed-idle-evaluator-stop-v1', 'plan': m.ref(path),
                'originalIdentity': plan['originalIdentity'], 'noEvaluationOutputs': True,
                'observedExit': m.ref(actual), 'eventsSnapshot': m.ref(events)}
        return path, plan, stop

    def test_stop_requires_actual_zero_exit_associated_with_original_process(self):
        with tempfile.TemporaryDirectory() as name:
            path, plan, stop = self.stop_fixture(Path(name))
            m.verify_stop_value(path, plan, stop)
            actual_path = Path(stop['observedExit']['path'])
            actual = m.read(actual_path)
            for change in ({'exitCode': None}, {'exitCode': -15}, {'process': {'path': 'other', 'sha256': 'x'}}):
                actual_path.write_text(json.dumps({**actual, **change}))
                stop['observedExit'] = m.ref(actual_path)
                with self.assertRaisesRegex(ValueError, 'Actual original'):
                    m.verify_stop_value(path, plan, stop)

    def test_stop_rejects_prior_metrics_and_changed_start_time(self):
        with tempfile.TemporaryDirectory() as name:
            path, plan, stop = self.stop_fixture(Path(name))
            changed = copy.deepcopy(stop)
            changed['originalIdentity']['processStartTimeTicks'] += 1
            with self.assertRaisesRegex(ValueError, 'idle-stop'):
                m.verify_stop_value(path, plan, changed)
            events = Path(stop['eventsSnapshot']['path'])
            events.write_text(json.dumps({'stage': 'evaluation-start'})+'\n'+json.dumps({'stage': 'worker-stop', 'completed': 0})+'\n')
            stop['eventsSnapshot'] = m.ref(events)
            with self.assertRaisesRegex(ValueError, 'outcomes were opened'):
                m.verify_stop_value(path, plan, stop)

    def test_no_signal_without_explicit_root_grant(self):
        with patch.object(m.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'Root review'):
                m.stop_original(type('Args', (), {'permit_stop': False})())
            kill.assert_not_called()

    def test_no_dispatch_without_explicit_root_grant(self):
        with patch.object(m, 'load_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'Root review'):
                m.run(type('Args', (), {'permit_evaluation': False})())
            worker.assert_not_called()

    def run_fixture(self, root, stack):
        worker = SimpleNamespace(global_ready=lambda _: None)
        args = SimpleNamespace(permit_evaluation=True, plan=root/'plan.json',
                               original_stop=root/'stop.json', output_root=root/'execution')
        stack.enter_context(patch.object(m, 'verify_plan', return_value=({}, worker, {})))
        stack.enter_context(patch.object(m, 'verify_stop'))
        stack.enter_context(patch.object(m, 'verify_original'))
        stack.enter_context(patch.object(m, 'ORIGINAL', root))
        stack.enter_context(patch.object(m, 'N', root))
        stack.enter_context(patch.object(m.os, 'sched_getaffinity', return_value={18, 19}))
        stack.enter_context(patch.object(m.os, 'getpriority', return_value=10))
        stack.enter_context(patch.dict(m.os.environ, {'CUDA_VISIBLE_DEVICES': '',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
            'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(root)+'/synthetic-runtime'}))
        (root/'synthetic-runtime').mkdir()
        return args

    def test_same_original_flock_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            root = Path(name)
            args = self.run_fixture(root, stack)
            with (root/'worker.lock').open('a') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    m.run(args)
            self.assertFalse((args.output_root/'started.json').exists())

    def test_missing_global162_gate_cannot_begin_execution(self):
        with tempfile.TemporaryDirectory() as name, ExitStack() as stack:
            args = self.run_fixture(Path(name), stack)
            with self.assertRaisesRegex(ValueError, 'All162 global freeze'):
                m.run(args)
            self.assertFalse((args.output_root/'started.json').exists())


if __name__ == '__main__':
    unittest.main()
