"""Pure resource gate tests; never query Windows or flush caches."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('host_memory_watch', ROOT/'scripts/watch-generalization-host-memory.py')
watch = importlib.util.module_from_spec(spec); spec.loader.exec_module(watch)


class HostMemoryWatchTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = {'windowsFreePhysicalBytes': 1.5*watch.GIB,
                         'linuxCachedBytes': 5*watch.GIB, 'linuxAvailableBytes': 7*watch.GIB,
                         'windowsCFreeBytes': 0}

    def decide(self, snapshot=None, **kwargs):
        return watch.decision(self.snapshot if snapshot is None else snapshot,
                              **{'seconds_since_attempt':180, 'elapsed_seconds':180, **kwargs})

    def test_all_strict_pressure_gates_are_required(self):
        self.assertEqual(self.decide()['action'], 'flush')
        for key, value in [('windowsFreePhysicalBytes', 2*watch.GIB),
                           ('linuxCachedBytes', 4*watch.GIB), ('linuxAvailableBytes', 6*watch.GIB)]:
            self.assertEqual(self.decide({**self.snapshot, key:value})['action'], 'observe')
        self.assertEqual(self.decide({**self.snapshot, 'windowsFreePhysicalBytes':3*watch.GIB})['action'], 'observe')

    def test_startup_and_between_attempt_cooldown_is_not_relaxed(self):
        for seconds in (0, 179, 179.999):
            self.assertEqual(self.decide(seconds_since_attempt=seconds)['action'], 'observe')
        self.assertEqual(self.decide(seconds_since_attempt=180)['action'], 'flush')

    def test_stop_complete_and_twelve_hour_limit_override_pressure(self):
        self.assertEqual(self.decide(stop_requested=True)['action'], 'stop')
        self.assertEqual(self.decide(progress_status='complete')['action'], 'stop')
        self.assertEqual(self.decide(elapsed_seconds=12*60*60)['action'], 'stop')
        self.assertEqual(self.decide(elapsed_seconds=12*60*60-1)['action'], 'flush')

    def test_missing_nonfinite_and_negative_samples_fail_closed(self):
        self.assertEqual(self.decide({})['action'], 'observe')
        for value in (None, True, float('nan'), float('inf'), -1):
            self.assertEqual(self.decide({**self.snapshot, 'windowsFreePhysicalBytes':value})['action'], 'observe')

    def test_only_exact_cache_command_or_readonly_identity_probe_is_allowed(self):
        command = watch.root_command('Ubuntu', watch.FLUSH_COMMAND)
        self.assertEqual(command[-1], 'sync; echo 3 > /proc/sys/vm/drop_caches')
        self.assertEqual(command[1:5], ['--distribution','Ubuntu','--user','root'])
        self.assertEqual(watch.root_command('Ubuntu', 'id -u')[-1], 'id -u')
        with self.assertRaisesRegex(ValueError, 'Unregistered privileged command'):
            watch.root_command('Ubuntu', 'service restart')


if __name__ == '__main__':
    unittest.main()
