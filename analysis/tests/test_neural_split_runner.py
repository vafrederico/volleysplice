"""Synthetic end-to-end qualification of the frozen-run adapter and JSON output."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('synthetic_split_runner', REPO/'scripts/run-neural-split-advisor.py')
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class SplitRunnerTests(unittest.TestCase):
    def test_full_synthetic_seed_audits_and_serializes_every_arm(self):
        """Only registration loading is stubbed; actual NPZ/evaluation/audits/write run."""
        seed = 3407
        recording = {
            'id': 'synthetic', 'sourceGroup': 'fixture', 'durationSeconds': 200.,
            'ignoredIntervals': [{'start': 110., 'end': 112.}],
            'rallies': [{'start': 10., 'end': 16.}, {'start': 20., 'end': 27.},
                        {'start': 80., 'end': 86.}],
            'productionEvents': [{'id': 'merged', 'start': 10., 'end': 30.},
                                 {'id': 'false', 'start': 50., 'end': 54.},
                                 {'id': 'real', 'start': 80., 'end': 86.}],
        }
        neural = [{'id': 'n1', 'start': 10., 'end': 16.},
                  {'id': 'n2', 'start': 20.25, 'end': 27.},
                  {'id': 'n3', 'start': 80., 'end': 86.}]
        times = np.arange(0., 200., .25, dtype=np.float64)
        scores = np.zeros((len(times), 4), dtype=np.float32)
        scores[:, 0] = .8
        scores[times == 20.25, 1] = .9
        scores[times == 20., 2] = .3
        scores[times == 25., 1] = .8  # Spurious internal start must be rejected by review.
        scores[times == 24.75, 2] = .3
        data = {'records': [recording], 'entries': [
            {'recordingId': 'synthetic', 'modelId': 'compact_boost', 'seed': seed,
             'timesKey': 'times', 'scoresKey': 'scores', 'events': neural}]}
        with tempfile.TemporaryDirectory(prefix='volleycut-split-synthetic-') as directory:
            root = Path(directory)
            npz = root/'probabilities.npz'
            np.savez(npz, times=times, scores=scores)
            registration = {'sha256': 'synthetic-contract-only',
                            'contract': {'probabilities': {'path': str(npz)}}}
            with patch.object(runner, 'load', return_value=(registration, data)), redirect_stdout(io.StringIO()):
                receipt = runner.execute_seed((str(root), seed))
            self.assertEqual((receipt['automatic'], receipt['reviewed']), (3, 56))
            self.assertGreater(receipt['result']['sizeBytes'], 0)
            result = json.loads((root/'results'/f'{seed}.json').read_text(encoding='utf-8'))
            self.assertEqual(len({row['id'] for row in result['reviewed']}), 56)
            all_outcomes = [result['baseline'], *result['automatic'], *result['reviewed']]
            baseline_durations = result['baseline']['durationMetrics']
            for row in all_outcomes:
                self.assertEqual(row['durationMetrics'], baseline_durations)
                self.assertTrue(row['durationAudit']['passed'])
                self.assertTrue(row['separateExportAudit']['passed'])
                self.assertTrue(row['identityAudit']['passed'])
                self.assertEqual(row['splitMetrics']['pooled']['additionalCompleteMisses'], 0)
                self.assertEqual(row['splitMetrics']['pooled']['rawCoreSecondsLostFromBaseline'], 0.)
            self.assertTrue(any(row['workload']['acceptedSplitCount'] > 0 for row in result['reviewed']))
            self.assertTrue(any(row['workload']['rejectedSplitCount'] > 0 for row in result['reviewed']))
            self.assertTrue(any(row['workload']['removedFalseParents'] > 0 for row in result['reviewed']))
            for row in result['reviewed']:
                for recording_row in row['perRecording']:
                    self.assertTrue(recording_row['humanAudit']['passed'])
                    self.assertTrue(recording_row['queueAudit']['passed'])


if __name__ == '__main__':
    unittest.main()
