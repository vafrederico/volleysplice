"""Fail-closed input binding and canonical export checks for the product replay."""
from analysis.private_ledger import private_value
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/prepare-neural-production-comparison.py'
spec = importlib.util.spec_from_file_location('production_comparison_input', SCRIPT)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def scope():
    rows = [{'id':f'r{i}', 'sourceGroup':f'g{i//2}', 'environment':'grass',
             'split':'train', 'consent':{'train':True},
             'rallies':[{'start':1,'end':2}] * (42 if i == 0 else 40)} for i in range(8)]
    return {'recordings':rows}, {'exactRows':copy.deepcopy(rows)}


class ProductionComparisonInputTests(unittest.TestCase):
    def test_gold_and_ignored_revision_must_match_entire_rows(self):
        exact, manifest = scope()
        self.assertEqual(len(adapter.verify_exact_scope(exact, manifest)), 8)
        manifest['exactRows'][0]['ignoredIntervals'] = [{'start':0,'end':1}]
        with self.assertRaisesRegex(ValueError, 'gold/ignored/input'):
            adapter.verify_exact_scope(exact, manifest)

    def test_protected_and_beach_fail_even_if_manifest_matches(self):
        for key,value in [('sourceGroup',private_value('source-group-008')),('environment','beach')]:
            exact, _ = scope()
            for row in exact['recordings'][:2]:
                row[key] = value
            with self.assertRaisesRegex(ValueError,'protected or beach'):
                adapter.verify_exact_scope(exact, {'exactRows':copy.deepcopy(exact['recordings'])})

    def test_identity_checks_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.json'
            path.write_text('one')
            first=adapter.identity(path)
            path.write_text('two')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                adapter.identity(path,first['sha256'])

    def test_endpoint_parity_rejects_count_and_shift(self):
        base=[{'start':1.0,'end':2.0}]
        self.assertEqual(adapter.verify_endpoints(base,base),0)
        with self.assertRaises(ValueError):
            adapter.verify_endpoints(base,[])
        with self.assertRaises(ValueError):
            adapter.verify_endpoints(base,[{'start':1.0,'end':2.001}])

    def test_canonical_strict_gap_and_ignored_barrier(self):
        core=[{'start':1,'end':2},{'start':5,'end':6},{'start':8.999,'end':10}]
        self.assertEqual(adapter.canonical_exports(core,[],12,0),
                         [{'start':1.0,'end':2.0},{'start':5.0,'end':10.0}])
        self.assertEqual(adapter.canonical_exports([{'start':2,'end':4}],
                          [{'start':3,'end':3.5}],10,2),
                         [{'start':0.0,'end':3.0},{'start':3.5,'end':6.0}])

    def test_export_difference_is_time_union_not_event_count(self):
        a=[{'start':1,'end':5},{'start':4,'end':7}]
        b=[{'start':2,'end':6}]
        self.assertEqual(adapter.difference_seconds(a,b),2)
        self.assertEqual(adapter.difference_seconds(b,a),0)


if __name__ == '__main__':
    unittest.main()
