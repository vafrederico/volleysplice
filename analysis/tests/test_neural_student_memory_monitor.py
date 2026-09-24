"""PID identity and phase attribution checks; no process launch or model work."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('student_memory', ROOT/'scripts/run-neural-student-reuse-profiled.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class StudentMemoryMonitorTest(unittest.TestCase):
    def test_proc_identity_handles_parentheses_inside_command_name(self):
        fields = ['S', '123'] + ['0']*17 + ['456789'] + ['0']*4
        self.assertEqual(monitor.stat_identity('456 (python (worker)) '+' '.join(fields)), (123, 456789))

    def test_linux_kib_are_bytes_and_cuda_is_not_accepted_as_rss(self):
        value = monitor.memory_status('VmRSS:\t1024 kB\nVmHWM:\t2048 kB\nVmSwap:\t3 kB\nCuda: 999 kB\n')
        self.assertEqual(value, {'VmRSSBytes': 1048576, 'VmHWMBytes': 2097152, 'VmSwapBytes': 3072})
        with self.assertRaises(ValueError):
            monitor.memory_status('VmRSS: 1024 MB')

    def test_fit_phase_markers_distinguish_loading_extraction_and_temporal(self):
        command = ['python', '/repo/scripts/run-neural-generalization.py', 'fit', '--output', '/nas/fit']
        existing = set()
        with patch.object(Path, 'exists', lambda p: str(p) in existing):
            self.assertEqual(monitor.artifact_phase(command), 'student-input-loading-or-training')
            existing.add('/nas/fit/student/completed.json')
            self.assertEqual(monitor.artifact_phase(command), 'student-feature-extraction-or-transition')
            existing.add('/nas/fit/temporal')
            self.assertEqual(monitor.artifact_phase(command), 'temporal-input-preparation-or-training')
            existing.add('/nas/fit/temporal/completed.json')
            self.assertEqual(monitor.artifact_phase(command), 'fit-finalization')
        self.assertEqual(monitor.artifact_phase(['python', '/repo/scripts/audit-neural-generalization-numerics.py']),
                         'independent-numerical-audit')

    def test_dropped_or_recycled_parent_cannot_attribute_its_child_to_queue(self):
        rows = [{'pid': 10, 'parentPid': 1}, {'pid': 20, 'parentPid': 10},
                {'pid': 40, 'parentPid': 30}, {'pid': 50, 'parentPid': 40}]
        self.assertEqual([row['pid'] for row in monitor.connected_rows(rows, 10)], [10, 20])
        self.assertEqual(monitor.connected_rows(rows[1:], 10), [])

    def test_grant_source_and_task_bindings_fail_closed(self):
        expected = {'source': {'sha256': 'frozen'}, 'requiredExitedPids': [10, 20], 'cpuAffinity': [10, 11]}
        monitor.validate_grant({**expected, 'createdAt': 'metadata'}, expected)
        for key, replacement in [('source', {'sha256': 'other'}), ('requiredExitedPids', [10]), ('cpuAffinity', [4, 5])]:
            with self.assertRaisesRegex(ValueError, key):
                monitor.validate_grant({**expected, key: replacement}, expected)


if __name__ == '__main__':
    unittest.main()
