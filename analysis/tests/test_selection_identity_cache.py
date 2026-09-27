import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

from analysis import neural_selection_identity_cache as cachelib
from analysis.neural_selection_serialization import load_module


class IdentityCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'fixture.bin'
        self.path.write_bytes(bytes(range(256))*64)

    def test_uncached_parity_aliases_and_only_one_cold_read(self):
        expected = {'path': str(self.path), 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()}
        originals = cachelib.io.identity, cachelib.aliases.identity, cachelib.inputs.sha
        cache = cachelib.IdentityCache(); cache.begin()
        with cachelib.installed(cache):
            self.assertEqual(cachelib.io.identity(self.path), expected)
            self.assertEqual(cachelib.aliases.identity(self.path), expected)
            self.assertEqual(cachelib.inputs.sha(self.path), expected['sha256'])
            self.assertEqual(cachelib.inputs.verified(expected), self.path)
        result = cache.finish()
        self.assertEqual(result['statistics']['coldHashCount'], 1)
        self.assertEqual(result['statistics']['cacheHitCount'], 3)
        self.assertEqual((cachelib.io.identity, cachelib.aliases.identity, cachelib.inputs.sha), originals)
        self.assertEqual(cachelib.inputs._verified, {})

    def test_across_stage_reuse_and_path_representation_preserved(self):
        cache = cachelib.IdentityCache()
        for index in range(2):
            cache.begin()
            with cachelib.installed(cache): result = cachelib.io.identity(self.path)
            stats = cache.finish()['statistics']
            self.assertEqual(stats['coldHashCount'], int(index == 0))
            self.assertEqual(stats['cacheHitCount'], int(index == 1))
            self.assertEqual(result['path'], str(self.path))

    def test_preserved_size_mtime_content_mutation_fails_via_ctime(self):
        cache = cachelib.IdentityCache(); cache.begin(); cache.digest(self.path); cache.finish()
        before = self.path.stat()
        self.path.write_bytes(b'x'*before.st_size)
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.path.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertNotEqual(self.path.stat().st_ctime_ns, before.st_ctime_ns)
        cache.begin()
        with self.assertRaisesRegex(ValueError, 'cold rehash'): cache.digest(self.path)

    def test_metadata_only_rewrite_is_also_rejected(self):
        cache = cachelib.IdentityCache(); cache.begin(); cache.digest(self.path); cache.finish()
        before = self.path.stat(); os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns+1_000_000))
        cache.begin()
        with self.assertRaisesRegex(ValueError, 'metadata changed'): cache.digest(self.path)

    def test_cold_read_race_and_stage_end_mutation_fail(self):
        actual = cachelib.signature(self.path)
        with patch.object(cachelib, 'signature', side_effect=[actual, {**actual, 'ctimeNs': actual['ctimeNs']+1}]):
            with self.assertRaisesRegex(ValueError, 'while cold hashing'): cachelib.cold_sha(self.path)
        cache = cachelib.IdentityCache(); cache.begin(); cache.digest(self.path)
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'stage end'): cache.finish()

    def test_numerical_replacement_rejected_and_hashes_restored_on_error(self):
        cache = cachelib.IdentityCache(); cache.begin()
        original, canonical = cachelib.io.identity, cachelib.io.canonical
        try:
            with self.assertRaisesRegex(ValueError, 'non-hash'):
                with cachelib.installed(cache): cachelib.io.canonical = lambda value: 'changed'
        finally: cachelib.io.canonical = canonical
        self.assertIs(cachelib.io.identity, original)

    def test_later_preflight_requires_full_cold_closure_and_unchanged_ctime(self):
        workflow = load_module('cache_workflow_test', Path(__file__).resolve().parents[2]/'scripts/generalization-selection-cache-execution.py')
        fake = {'path': 'identity', 'sha256': 'f'*64}
        entry = {'path': str(self.path.resolve()), 'sha256': 'a'*64, 'stat': cachelib.signature(self.path)}
        gate = {'kind': 'independent-selection-identity-cache-cold-audit-v1', 'passed': True,
            'allColdHashesPassed': True, 'taskCount': 18, 'stageCount': 54, 'cachedStageCount': 51,
            'preexistingStageCount': 3, 'auditor': fake, 'plan': fake, 'coldFiles': [entry], 'files': [entry],
            'closurePolicy': cachelib.POLICY['coldClosurePolicy'], 'allImmutableStatsPassed': True,
            'completeStageAndDeclaredEvidenceClosureVerified': True, 'byteExactQualificationParity': True,
            'evidence': [{'path': entry['path'], 'sha256': entry['sha256']}]}
        def run(value, current=None):
            with patch.object(workflow.io, 'read', return_value=value), patch.object(workflow.io, 'identity', return_value=fake), \
                 patch.object(workflow, 'verify_plan', return_value={}), patch.object(workflow, 'verified', return_value='plan'), \
                 patch.object(workflow, 'signature', return_value=current or entry['stat']):
                return workflow.cold_gate('gate')
        self.assertEqual(run(gate), gate)
        changed = {**entry['stat'], 'ctimeNs': entry['stat']['ctimeNs']+1}
        with self.assertRaisesRegex(ValueError, 'metadata changed'): run(gate, changed)
        missing = deepcopy(gate); missing['coldFiles'] = []
        with self.assertRaisesRegex(ValueError, 'omits an evidence'): run(missing)


if __name__ == '__main__': unittest.main()
