from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[2]/'scripts/audit-neural-feature-index-continuity.py'
SPEC = importlib.util.spec_from_file_location('feature_index_continuity', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContinuityTest(unittest.TestCase):
    def inputs(self):
        core_ref, complete_ref = {'path': 'core', 'sha256': 'a'}, {'path': 'complete', 'sha256': 'b'}
        core = {'kind': 'neural-generalization-features-v1', 'publicationMode': 'core',
            'publicationScope': 'all', 'stagePlan': {'sha256': 'stage'},
            'records': [{'recordingId': str(i), 'sourceGroup': 'g', 'contentSha256': 'source',
                         'features': {'audiovisual': {'sha256': 'av'},
                                      'dino': {'fp32': {'sha256': 'fp32'}, 'fp16': {'sha256': 'fp16'}}}}
                        for i in range(42)]}
        complete = deepcopy(core)
        complete['publicationMode'] = 'complete'
        for row in complete['records']:
            row['features']['dino']['int8'] = {'sha256': 'int8'}
        audit = {'kind': 'independent-expansion-feature-audit-v1', 'passed': True, 'mode': 'core',
            'scope': 'all', 'auditedRecordingCount': 42, 'features': core_ref, 'stagePlan': core['stagePlan'],
            'inputs': {'sha256': 'gold'}, 'records': [{'id': str(i), 'teacherFrames': 0} for i in range(42)]}
        full_audit = deepcopy(audit)
        full_audit.update(mode='complete', features=complete_ref)
        return core, complete, audit, full_audit, core_ref, complete_ref

    def test_only_new_precision_passes_without_mutation(self):
        values = self.inputs()
        before = deepcopy(values)
        added = MODULE.compare_publications(*values)
        self.assertEqual(len(added), 42)
        self.assertTrue(all(row['addedPrecisions'] == ['int8'] for row in added))
        self.assertEqual(values, before)

    def test_existing_reference_change_rejected(self):
        for feature in ('audiovisual', 'fp32', 'fp16'):
            with self.subTest(feature=feature):
                values = self.inputs()
                features = values[1]['records'][0]['features']
                target = features if feature == 'audiovisual' else features['dino']
                target[feature]['sha256'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'changed'):
                    MODULE.compare_publications(*values)

    def test_source_scope_or_duplicate_rejected(self):
        for duplicate in (True, False):
            values = self.inputs()
            values[1]['records'][0]['recordingId'] = '1' if duplicate else 'new'
            with self.assertRaises(ValueError):
                MODULE.compare_publications(*values)

    def test_failed_or_rebound_audit_rejected(self):
        for field, changed in (('passed', False), ('features', {'path': 'other'}), ('inputs', {'sha256': 'newgold'})):
            values = self.inputs()
            values[3][field] = changed
            with self.assertRaises(ValueError):
                MODULE.compare_publications(*values)

    def test_publication_and_audit_record_order_preserved(self):
        for index in (1, 3):
            values = self.inputs()
            values[index]['records'].reverse()
            with self.assertRaisesRegex(ValueError, 'order changed'):
                MODULE.compare_publications(*values)


if __name__ == '__main__':
    unittest.main()
