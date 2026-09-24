"""Synthetic end-to-end adapter qualification; no real outcome scoring."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rally_runner_test', ROOT/'scripts/run-neural-rally-review-proposals.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerIntegrationTests(unittest.TestCase):
    def test_complete_registered_cell_with_mapping_gold_and_all_scopes(self):
        times = np.arange(0., 100., .25)
        scores = np.full((len(times), 4), .1, dtype=np.float32)
        scores[(times >= 10) & (times < 20), 0] = .8
        records, entries, arrays = [], [], {}
        for i in range(2):
            rid = f'r{i}'
            records.append({'id': rid, 'sourceGroup': f'g{i}', 'durationSeconds': 100.,
                            'rallies': [{'start': 10., 'end': 20., 'tags': ['manual']}],
                            'ignoredIntervals': [{'start': 60., 'end': 65.}],
                            'productionEvents': [{'start': 8., 'end': 22., 'id': 'production'}]})
            entries.append({'modelId': 'compact_boost', 'seed': 3407, 'recordingId': rid,
                            'scoresKey': rid+'scores', 'timesKey': rid+'times',
                            'events': [{'start': 10., 'end': 20., 'id': 'neural'}]})
            arrays[rid+'times'] = times; arrays[rid+'scores'] = scores
        auditor = runner.module('integration_rally_auditor', 'scripts/audit-neural-rally-review-proposals.py')
        candidates = runner.module('integration_candidate_auditor', 'scripts/audit-neural-rally-review-candidates.py')
        accounting = runner.module('integration_accounting_auditor', 'scripts/audit-neural-combination-accounting.py')
        with tempfile.TemporaryDirectory() as folder:
            runner._STATE = (Path(folder), {'sha256': 'synthetic', 'contract': {'sourceGroups': ['g0', 'g1']}},
                             {'records': records, 'entries': entries}, arrays, auditor, candidates, accounting)
            for inventory in runner.review.INVENTORIES:
                config = {'id': inventory, 'mode': 'production', 'model': 'compact_boost',
                          'inventory': inventory, 'ranker': 'evidence'}
                result_ref = runner.execute((config, 3407))
                result = json.loads(Path(result_ref['path']).read_text())
                self.assertEqual(len(result['outcomes']), 4)
                for row in result['outcomes']:
                    self.assertTrue(row['durationAudit']['passed'])
                    self.assertTrue(row['identityAudit']['passed'])
                    self.assertEqual(row['identityMetrics']['pooled']['trueRallies'], 2)
                    self.assertEqual(row['workload']['playbackTrueRallies'],
                                     sum(x['playbackTrueRallies'] for x in row['recordings']))


if __name__ == '__main__':
    unittest.main()
