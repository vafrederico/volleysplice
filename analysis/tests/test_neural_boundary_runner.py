"""Synthetic end-to-end qualification for typed boundary evaluation adapters."""
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
SPEC = importlib.util.spec_from_file_location('synthetic_boundary_runner', REPO/'scripts/run-neural-typed-boundaries.py')
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class BoundaryRunnerTests(unittest.TestCase):
    def test_full_synthetic_seed_writes_audited_matrix_and_distinguishes_raw_coverage(self):
        """Stub only registered input loading; actual inference rules/audits/I/O run."""
        seed = 3407
        rec = {'id': 'synthetic', 'sourceGroup': 'fixture', 'durationSeconds': 400.,
               'ignoredIntervals': [{'start': 300., 'end': 302.}],
               'productionEvents': [{'id': 'wrong-first', 'start': 10., 'end': 40.},
                                    {'id': 'missed-second', 'start': 60., 'end': 90.},
                                    {'id': 'unsupported-false', 'start': 120., 'end': 126.},
                                    {'id': 'end-only', 'start': 150., 'end': 165.}],
               'rallies': [{'start': 12., 'end': 18.}, {'start': 25., 'end': 31.},
                           {'start': 62., 'end': 68.}, {'start': 76., 'end': 84.},
                           {'start': 150., 'end': 164.}]}
        # First compact event in the first parent is really its SECOND rally.
        # In the second parent compact never proposes the second rally at all.
        neural = [{'id': 'n1', 'start': 25.25, 'end': 29.},
                  {'id': 'n2', 'start': 62.25, 'end': 66.},
                  {'id': 'n3', 'start': 150., 'end': 162.}]
        times = np.arange(0., 400., .25, dtype=np.float64)
        scores = np.zeros((len(times), 4), dtype=np.float32)
        scores[:, 0] = .8
        for time, head, value in [(25.25, 1, .9), (30., 2, .8),
                                  (62.25, 1, .85), (67., 2, .8),
                                  (150., 1, .9), (164., 2, .8)]:
            scores[times == time, head] = value
        data = {'records': [rec], 'entries': [{'recordingId': 'synthetic', 'modelId': 'compact_boost',
                'seed': seed, 'timesKey': 'times', 'scoresKey': 'scores', 'events': neural}]}
        with tempfile.TemporaryDirectory(prefix='volleycut-boundary-synthetic-') as directory:
            root = Path(directory)
            npz = root/'probabilities.npz'
            np.savez(npz, times=times, scores=scores)
            registration = {'sha256': 'synthetic-boundary-contract',
                            'contract': {'probabilities': {'path': str(npz)}}}
            with patch.object(runner, 'load', return_value=(registration, data)), redirect_stdout(io.StringIO()):
                receipt = runner.execute_seed((str(root), seed))
            self.assertEqual((receipt['automatic'], receipt['reviewed']), (4, 64))
            self.assertGreater(receipt['result']['sizeBytes'], 0)
            result = json.loads((root/'results'/f'{seed}.json').read_text(encoding='utf-8'))
            self.assertEqual(len({r['id'] for r in result['reviewed']}), 64)
            baseline_duration = result['baseline']['durationMetrics']
            for out in [result['baseline'], *result['automatic'], *result['reviewed']]:
                self.assertEqual(out['durationMetrics'], baseline_duration)
                for field in ('fixedExportAudit', 'identityAudit', 'rawCoverageAudit', 'typedMetricAudit'):
                    self.assertTrue(out[field]['passed'])
            automatic = {r['policy']: r for r in result['automatic']}
            for out in automatic.values():
                self.assertGreater(out['typedMetrics']['pooled']['typeDiagnostics']['1']['initialProposedForAdditional'], 0)
            separate = automatic['separate_ends']
            self.assertGreater(separate['rawCoverageMetrics']['pooled']['rawCoreSecondsLostFromBaseline'], 0.)
            self.assertGreater(separate['rawCoverageMetrics']['pooled']['additionalCompleteMisses'], 0)
            # Fixed padded export recall cannot be substituted for changed raw-core recall.
            self.assertEqual(separate['durationMetrics'][2]['R_core'], baseline_duration[2]['R_core'])
            reviewed = {r['id']: r for r in result['reviewed']}
            limited = reviewed['separate_ends--proposal_confirmation--chronological--budget-40']
            full = reviewed['separate_ends--full_parent--chronological--budget-40']
            self.assertGreater(limited['rawCoverageMetrics']['pooled']['rawCoreSecondsLostFromBaseline'], 0.)
            self.assertGreater(limited['rawCoverageMetrics']['pooled']['additionalCompleteMisses'], 0)
            self.assertEqual(full['rawCoverageMetrics']['pooled']['rawCoreSecondsLostFromBaseline'], 0.)
            self.assertEqual(full['rawCoverageMetrics']['pooled']['additionalCompleteMisses'], 0)
            self.assertGreater(full['identityMetrics']['pooled']['eventF1'], limited['identityMetrics']['pooled']['eventF1'])
            self.assertGreater(limited['perRecording'][0]['correctedTypeCount'], 0)
            self.assertEqual(set(limited['perRecording'][0]['selectedParentIds']), set(full['perRecording'][0]['selectedParentIds']))
            # Full-parent review can recover the unproposed second rally inside a reviewed parent.
            self.assertTrue(any(row['start'] == 76. and row['end'] == 84. for row in full['perRecording'][0]['events']))
            self.assertFalse(any(row['start'] == 76. for row in limited['perRecording'][0]['events']))
            for out in result['reviewed']:
                for row in out['perRecording']:
                    self.assertTrue(row['humanAudit']['passed'])
                    self.assertTrue(row['queueAudit']['passed'])


if __name__ == '__main__':
    unittest.main()
