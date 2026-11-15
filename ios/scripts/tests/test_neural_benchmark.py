import importlib.util
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).parents[1] / 'run-neural-benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkRunnerTests(unittest.TestCase):
    def run_case(self, platform, success=True):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'private-source.mp4'
            source.touch()
            plan = root / 'original.xctestrun'
            target = 'VolleySpliceMacTests' if platform == 'macos' else 'VolleySpliceTests'
            plan.write_bytes(plistlib.dumps({target: {}, '__xctestrun_metadata__': {'FormatVersion': 1}}))
            original = plan.read_bytes()
            output = root / 'report.json'
            args = ['benchmark', '--platform', platform, '--xctestrun', str(plan),
                    '--source', str(source), '--source-index', 'recording-1', '--output', str(output)]
            if platform == 'simulator': args += ['--destination', 'test-device']

            def execute(command, stdout, **kwargs):
                temporary = Path(command[command.index('-xctestrun') + 1])
                self.assertEqual(temporary.parent, plan.parent)
                generated = plistlib.loads(temporary.read_bytes())
                self.assertEqual(set(generated), {target, '__xctestrun_metadata__'})
                self.assertEqual(generated[target]['EnvironmentVariables']['VOLLEYCUT_IOS_BENCHMARK_SOURCE'], str(source))
                self.assertEqual(command[command.index('-destination') + 1],
                                 'platform=macOS,arch=' + benchmark.platform.machine() if platform == 'macos' else 'id=test-device')
                for model in ['legacy', 'distilled-large-recall-v1', 'distilled-large-f1-v1']:
                    stdout.write('NEURAL_PIPELINE_BENCHMARK ' + json.dumps({'modelId': model}) + '\n')
                return subprocess.CompletedProcess(command, 0 if success else 1)

            with patch.object(sys, 'argv', args), patch.object(benchmark.subprocess, 'run', side_effect=execute), patch('builtins.print'):
                if success:
                    benchmark.main()
                    report = json.loads(output.read_text())
                    self.assertEqual(report['platform'], platform)
                    self.assertNotIn(str(source), output.read_text())
                else:
                    with self.assertRaises(RuntimeError): benchmark.main()
                    self.assertFalse(output.exists())
            self.assertEqual(plan.read_bytes(), original)
            self.assertEqual(list(root.glob('*.xctestrun')), [plan])

    def test_simulator_target_is_preserved(self): self.run_case('simulator')
    def test_native_target_and_destination(self): self.run_case('macos')
    def test_xctest_failure_cannot_publish_success(self): self.run_case('macos', success=False)


if __name__ == '__main__': unittest.main()
