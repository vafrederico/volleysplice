"""The recovery accepts actual new exits, never infers old successful exits."""
import copy
import importlib.util
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[2]
PATH = REPO/'scripts/audit-neural-precision-recovery-slot.py'
spec = importlib.util.spec_from_file_location('precision_recovery_test', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.interruption = {'path': '/nas/interruption.json', 'sha256': 'a'*64}
        self.receipt = {'kind': 'recovered-study-slot-observed-exit-v1', 'passed': True,
            'exitCode': 0, 'precisionSlot': 'fp16', 'bootId': 'new-boot',
            'interruption': self.interruption, 'exitedPids': [123, 124],
            'processes': [{'pid': pid, 'bootId': 'new-boot', 'startTimeTicks': 12345,
                           'command': 'python recovery.py'} for pid in (123, 124)],
            'releasedCpuAffinity': [12, 13]}

    def validate(self, value, absent=lambda pid: True):
        return module.validate_release(value, 'fp16', 'new-boot', self.interruption, absent)

    def test_current_exit(self):
        self.assertEqual(self.validate(self.receipt), ([123, 124], [12, 13]))

    def test_exit_and_scope_tampering(self):
        for key, value in [('kind', 'old-exit'), ('passed', False), ('exitCode', None),
                           ('exitCode', 1), ('precisionSlot', 'int8'), ('bootId', 'old-boot'),
                           ('interruption', {}), ('releasedCpuAffinity', [12, 14]),
                           ('exitedPids', []), ('exitedPids', [123, 123])]:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.receipt); changed[key] = value
                with self.assertRaises(AssertionError): self.validate(changed)

    def test_live_or_misidentified_process(self):
        with self.assertRaises(AssertionError): self.validate(self.receipt, lambda pid: pid != 123)
        for key, value in [('pid', 999), ('bootId', 'old-boot'), ('startTimeTicks', 0), ('command', '')]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.receipt); changed['processes'][0][key] = value
                with self.assertRaises(AssertionError): self.validate(changed)

    def test_frozen_numerical_preflight_is_preserved(self):
        if not module.OLD_SOURCE.exists():
            self.skipTest('NAS reference is only available on the study host')
        self.assertEqual(module.digest(module.OLD_SOURCE)['sha256'], module.OLD_SOURCE_SHA)
        old, new = module.OLD_SOURCE.read_text(), PATH.read_text()
        start, end = '    sys.path.insert(0,str(REPO))', '    previous=None'
        self.assertEqual(old[old.index(start):old.index(end)], new[new.index(start):new.index(end)])


if __name__ == '__main__':
    unittest.main()
