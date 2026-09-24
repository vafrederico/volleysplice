from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis import neural_short_boost_weighting as implementation
from analysis.schema import Interval
from analysis.tests.test_neural_event_weighting import make_row

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/audit-neural-short-boost-tensors.py'
SPEC = importlib.util.spec_from_file_location('independent_short_boost_tensor_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def record_for(row):
    source = row.example
    result = {'id': source.id, 'sourceGroup': source.group,
              'rallies': [r.to_dict() for r in source.truth],
              'ignoredIntervals': [r.to_dict() for r in source.ignored]}
    if row.tier == 'coverage':
        result['gameWindow'] = {'start': 0, 'end': source.duration}
        result['keepTargets'] = result['rallies']
    return result


class TensorAuditFormulaTests(unittest.TestCase):
    def test_reference_identity_accepts_verified_optional_size_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_bytes(b'{"example":true}\n')
            enriched = audit.identity(path)
            compact = {key: enriched[key] for key in ('path', 'sha256')}
            audit.verify_reference_identity(enriched, compact)
            audit.verify_reference_identity(compact, enriched)
            audit.verify_reference_identity(enriched, enriched)

    def test_reference_identity_rejects_changed_path_hash_size_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_bytes(b'{"example":true}\n')
            enriched = audit.identity(path)
            compact = {key: enriched[key] for key in ('path', 'sha256')}
            mutations = ({'path': str(path.with_name('different.json'))},
                         {'sha256': '0'*64}, {'sizeBytes': enriched['sizeBytes']+1},
                         {'unrecognized': True})
            for mutation in mutations:
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    audit.verify_reference_identity({**enriched, **mutation}, compact)
            with self.assertRaises(ValueError):
                audit.verify_reference_identity(enriched, {**compact, 'sizeBytes': enriched['sizeBytes']+1})
            path.write_bytes(b'{"example":false}\n')
            with self.assertRaises(ValueError):
                audit.verify_reference_identity(enriched, compact)

    @staticmethod
    def pts_fixture(indexes=(0, 1, 2, 3)):
        times = np.arange(4, dtype=np.float64)/4
        ticks = np.array([0, 1, 3, 5, 7], np.int64)
        pts = ticks.astype(np.float64)/8
        selected = np.asarray(indexes, np.int64)
        hashes = np.asarray([f'{i:064x}' for i in indexes])
        arrays = {'times': times, 'packetPtsTicks': ticks, 'presentationTimes': pts,
                  'selectedOrdinals': selected, 'selectedPresentationTimes': pts[selected],
                  'observedPresentationTimes': pts[selected].copy(), 'selectedFrameSha256': hashes}
        selection = {'videoDurationSeconds': 1., 'videoFrameCount': 5, 'sampleCount': 4, 'timeBase': '1/8',
                     'maximumFrameSelectionErrorSeconds': float(np.max(np.abs(pts[selected]-times))),
                     'maximumDecodedPtsErrorSeconds': 0.,
                     'frameHashesSha256': hashlib.sha256('\n'.join(hashes).encode()).hexdigest()}
        for key, values, dtype in (('gridSha256', times, '<f8'), ('presentationTimesSha256', pts, '<f8'),
                                   ('selectedOrdinalsSha256', selected, '<i8'), ('selectedPresentationTimesSha256', pts[selected], '<f8')):
            selection[key] = hashlib.sha256(values.astype(dtype).tobytes()).hexdigest()
        return arrays, selection

    def test_media_pts_selection_independently_checks_nearest_frames_and_earlier_ties(self):
        arrays, selection = self.pts_fixture()
        result = audit.validate_pts_vectors(arrays, selection, arrays['times'], arrays['times'])
        self.assertEqual(result['maximumSelectionErrorSeconds'], .125)
        self.assertTrue(result['independentNearestFrameSelectionVerified'])
        # Each later choice is equally close, but violates the original tie rule.
        arrays, selection = self.pts_fixture((0, 2, 3, 4))
        with self.assertRaisesRegex(ValueError, 'earlier-tie'):
            audit.validate_pts_vectors(arrays, selection, arrays['times'], arrays['times'])

    def test_media_pts_selection_rejects_clock_corruption_and_stale_pixel_identity(self):
        arrays, selection = self.pts_fixture()
        with self.assertRaisesRegex(ValueError, 'media grids'):
            audit.validate_pts_vectors(arrays, selection, arrays['times']+.001, arrays['times'])
        changed = copy.deepcopy(arrays)
        changed['observedPresentationTimes'][2] += .01
        with self.assertRaisesRegex(ValueError, 'PTS error'):
            audit.validate_pts_vectors(changed, selection, arrays['times'], arrays['times'])
        changed = copy.deepcopy(arrays)
        changed['selectedFrameSha256'][1] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'pixel identity'):
            audit.validate_pts_vectors(changed, selection, arrays['times'], arrays['times'])
        changed = copy.deepcopy(selection)
        changed['timeBase'] = '1/9'
        with self.assertRaisesRegex(ValueError, 'packet time base'):
            audit.validate_pts_vectors(arrays, changed, arrays['times'], arrays['times'])

    def test_pinned_scaler_verifier_independently_handles_ties_masks_and_exact_scope(self):
        verifier = audit.independent_verifier()
        absolute = next(iter(verifier.ABSOLUTE_FEATURE_NAMES))
        raw = np.array([[2, 4], [2, 3], [4, 2], [8, 1]], np.float32)
        actual = verifier.independent_percentile_normalization(raw, ('ranked', absolute))
        np.testing.assert_array_equal(actual[:, 0], np.array([1/6, 1/6, 2/3, 1], np.float32))
        np.testing.assert_array_equal(actual[:, 1], raw[:, 1])
        inputs = {'gold': {'tier': 'exact', 'valid': np.array([True, False, True]),
                           'normalized': np.array([[0, 2, 7], [99, 99, 99], [2, 6, 7]], np.float32)}}
        mean, scale, ticks = verifier.independent_scaler(['gold'], inputs)
        np.testing.assert_array_equal(mean, np.array([1, 4, 7], np.float32))
        np.testing.assert_array_equal(scale, np.array([1, 2, 1e-4], np.float32))
        self.assertEqual(ticks, 2)
        inputs['gold']['tier'] = 'draft'
        with self.assertRaisesRegex(ValueError, 'Auxiliary'):
            verifier.independent_scaler(['gold'], inputs)

    def test_independent_four_head_reconstruction_matches_fixed_annotation_contract(self):
        for tier in ('exact', 'draft', 'coverage'):
            row = make_row((Interval(2., 4.25), Interval(8., 28.), Interval(29., 30.)),
                           (Interval(14., 17.),), tier=tier, length=128)
            actual = audit.independent_supervision(record_for(row), tier, row.example.times, row.example.duration)
            np.testing.assert_array_equal(actual['targets'], row.example.targets)
            np.testing.assert_array_equal(actual['mask'], row.mask)
            np.testing.assert_array_equal(actual['valid'], row.example.valid)
            self.assertEqual(actual['counts']['valid'], row.mask.sum(0).astype(int).tolist())
        self.assertEqual(audit.merge([(0, 1), (4, 5)], 3), [(0, 1), (4, 5)])
        self.assertEqual(audit.merge([(0, 1), (3.99, 5)], 3), [(0, 5)])
        self.assertEqual(audit.subtract([(0, 5)], [(2, 3)]), [(0, 2), (3, 5)])

    def test_all_weight_diagnostics_reconstruct_exactly_without_weighting_helpers(self):
        sources = [make_row((Interval(1., 7.), Interval(9., 12.)), (Interval(2., 6.), Interval(10., 11.))),
                   make_row((Interval(2., 4.25), Interval(8., 28.), Interval(29., 30.)), tier='draft', length=128),
                   make_row(tier='coverage'), make_row(())]
        for row in sources:
            record = record_for(row)
            independent = audit.independent_supervision(record, row.tier, row.example.times, row.example.duration)
            for arm in ('baseline', 'global_control', 'short_boost'):
                with self.subTest(tier=row.tier, arm=arm):
                    expected = audit.independent_weights(record, row.tier, row.example.times, independent, arm)
                    _, actual = implementation.live_event_weights(row, arm)
                    self.assertEqual(expected, actual)
        draft = sources[1]
        result = audit.independent_weights(record_for(draft), 'draft', draft.example.times,
            audit.independent_supervision(record_for(draft), 'draft', draft.example.times, draft.example.duration), 'short_boost')
        self.assertEqual(result['events'][0]['positiveSupervisedTicks'], 1)
        self.assertEqual(result['events'][0]['multiplier'], 2)
        self.assertIsNone(result['events'][2]['multiplier'])

    def test_membership_excludes_both_groups_in_every_training_tier(self):
        tiers = {tier: [{'id': f'{tier}-{g}', 'sourceGroup': g} for g in ('a', 'b', 'c', 'd', 'aux')]
                 for tier in ('exact', 'draft', 'coverage')}
        tiers['exact'] = tiers['exact'][:-1]
        inner = audit.expected_membership(tiers, 'reviewed_export', 'a', 'b')
        self.assertEqual(inner['trainIds'], ['exact-c', 'exact-d'])
        self.assertEqual(inner['validationIds'], ['exact-b'])
        self.assertEqual(inner['auxiliaryIds'], {'draft': ['draft-c', 'draft-d', 'draft-aux'],
                                               'coverage': ['coverage-c', 'coverage-d', 'coverage-aux']})
        outer = audit.expected_membership(tiers, 'draft', 'a', 'a')
        self.assertEqual(outer['trainIds'], ['exact-b', 'exact-c', 'exact-d'])
        self.assertEqual(outer['auxiliaryIds'], {'draft': ['draft-b', 'draft-c', 'draft-d', 'draft-aux']})
        self.assertEqual(audit.expected_membership(tiers, 'exact', 'a', 'a')['auxiliaryIds'], {})

    def test_origin_paths_cannot_relabel_fresh_or_historical_fits(self):
        reference = {'path': '/old/study', 'reportSha256': 'report', 'contractSha256': 'contract'}
        reused = {'cohort': 'exact', 'kind': 'tcn', 'lossArm': 'baseline', 'seed': 3407,
                  'origin': {'type': 'reused-reference', 'studyPath': '/old/study', 'reportSha256': 'report',
                             'referenceContractSha256': 'contract', 'fitRoot': '/old/study/fits/exact/tcn/3407'}}
        path, historical = audit.validate_origin(reused, Path('/new/study'), reference)
        self.assertTrue(historical)
        self.assertEqual(path, Path('/old/study/fits/exact/tcn/3407'))
        fresh = {'cohort': 'draft', 'kind': 'dino_tcn', 'lossArm': 'baseline', 'seed': 1729,
                 'origin': {'type': 'trained', 'fitRoot': '/new/study/fits/draft/dino_tcn/baseline/1729'}}
        self.assertFalse(audit.validate_origin(fresh, Path('/new/study'), reference)[1])
        corrected = {'cohort': 'reviewed_export', 'kind': 'tcn', 'lossArm': 'baseline', 'seed': 3407,
                     'origin': {'type': 'trained', 'fitRoot': '/new/study/fits/reviewed_export/tcn/baseline/3407'}}
        self.assertFalse(audit.validate_origin(corrected, Path('/new/study'), reference)[1])
        historical_coverage = copy.deepcopy(corrected)
        historical_coverage['origin'] = {**reused['origin'], 'fitRoot': '/old/study/fits/reviewed_export/tcn/3407'}
        with self.assertRaisesRegex(ValueError, 'fitRoot'):
            audit.validate_origin(historical_coverage, Path('/new/study'), reference)
        for changed in (copy.deepcopy(reused), copy.deepcopy(fresh)):
            changed['origin']['fitRoot'] += '/other'
            with self.assertRaisesRegex(ValueError, 'fitRoot'):
                audit.validate_origin(changed, Path('/new/study'), reference)
        reused['origin']['referenceContractSha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            audit.validate_origin(reused, Path('/new/study'), reference)

    def test_pts_revision_changes_only_coverage_av_and_preserves_all_annotations(self):
        original = {'exactRows': [{'id': 'exact'}], 'draftRows': [{'id': 'draft'}],
                    'exactManifest': {'path': 'gold', 'sha256': 'gold-hash'}, 'protectedSourceGroups': ['test'],
                    'coverageRows': [{'id': f'pixel-{i}', 'sourceGroup': 'pixel', 'gameWindow': {'start': 1, 'end': 9},
                                      'keepTargets': [{'start': 2, 'end': 5}], 'rallies': [], 'ignoredIntervals': [],
                                      'featureCaches': {'audiovisual': {'path': f'old-{i}', 'sha256': f'old-{i}'}}}
                                     for i in range(7)]}
        repaired = copy.deepcopy(original)
        for index, value in enumerate(repaired['coverageRows']):
            value['featureCaches']['audiovisual'] = {'path': f'new-{index}', 'sha256': f'new-{index}'}
        self.assertEqual(len(audit.validate_dataset_revision(repaired, original)), 7)
        changes = [copy.deepcopy(repaired) for _ in range(5)]
        changes[0]['exactRows'][0]['id'] = 'changed'
        changes[1]['coverageRows'][0]['keepTargets'][0]['end'] = 6
        changes[2]['coverageRows'][0]['featureCaches'] = copy.deepcopy(original['coverageRows'][0]['featureCaches'])
        changes[3]['coverageRows'].reverse()
        changes[4]['draftRows'][0]['id'] = 'changed'
        for altered in changes:
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                audit.validate_dataset_revision(altered, original)

    def test_exposure_compares_all_streams_and_only_common_epoch_prefix(self):
        baseline = {'trainIds': ['x'], 'auxiliaryIds': {'draft': ['d'], 'coverage': ['c']}, 'validationIds': ['y'],
                    'scalerTrainIds': ['x'], 'positiveWeight': [1]*4, 'supervisedCounts': {},
                    'history': [{'epoch': i, 'optimizerSteps': 5*i, 'exposureSha256': {'exact': str(i), 'draft': str(i+1), 'coverage': str(i+2)}}
                                for i in (1, 2)]}
        candidate = copy.deepcopy(baseline)
        candidate['history'].pop()
        self.assertEqual(audit.audit_exposure(candidate, baseline), 1)
        candidate['history'][0]['exposureSha256']['coverage'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'sampling exposure'):
            audit.audit_exposure(candidate, baseline)
        candidate = copy.deepcopy(baseline)
        candidate['history'][0]['optimizerSteps'] += 1
        with self.assertRaisesRegex(ValueError, 'optimizer steps'):
            audit.audit_exposure(candidate, baseline)

    def test_actual_final_audit_refuses_missing_report_before_data_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'completed report'):
                audit.audit(Path(temporary)/'missing-manifest.json', Path(temporary)/'study')

    def test_dino_recording_roi_backbone_and_label_associations_are_bound(self):
        source = {'id': 'record', 'sourceGroup': 'group', 'contentSha256': 'video',
                  'featureCaches': {'audiovisual': {'sha256': 'av'}},
                  'roi': {'x': .1, 'y': .2, 'width': .6, 'height': .7}}
        manifest = {'extractorConfigSha256': 'config', 'semanticBackbone': {'architecture': 'vits14'}}
        metadata = {'recordingId': 'record', 'recordingContentSha256': 'video',
                    'extractorConfigSha256': 'config', 'roi': [.1, .2, .6, .7],
                    'labelsUsed': False, 'completed': True, 'backbone': {'architecture': 'vits14'}}
        entry = {'recordingId': 'record', 'tier': 'exact', 'sourceGroup': 'group', 'passed': True,
                 'audiovisualSha256': 'av', 'sourceVideoVerified': {'sha256': 'video'}, 'dinoSha256': 'cache'}
        wrapper = {'recordingId': 'record', 'cacheMetadata': copy.deepcopy(metadata), 'cache': {'sha256': 'cache'}}
        audit.validate_dino_association(entry, manifest, source, 'exact', metadata, wrapper)
        for field, value in (('recordingId', 'other'), ('labelsUsed', True), ('roi', None),
                             ('backbone', {'architecture': 'other'}), ('recordingContentSha256', 'other')):
            changed = copy.deepcopy(metadata)
            changed[field] = value
            changed_wrapper = {**wrapper, 'cacheMetadata': changed}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'metadata association'):
                audit.validate_dino_association(entry, manifest, source, 'exact', changed, changed_wrapper)
        entry['audiovisualSha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'source association'):
            audit.validate_dino_association(entry, manifest, source, 'exact', metadata, wrapper)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch is not installed')
class TensorAuditCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.architectures = audit.parameter_contract()

    def fixture(self, folder, kind):
        from analysis.transfer_temporal_model import model_for
        model = model_for(kind)
        state = {'model::'+name: tensor.detach().numpy() for name, tensor in model.state_dict().items()}
        state.update(mean=np.zeros(104, np.float32), scale=np.ones(104, np.float32))
        predictions = {'held': np.full((5, 4), .5, np.float32)}
        predictions['held'][2] = 0
        inputs = {'held': {'tier': 'exact', 'times': np.arange(5)/4+.125, 'valid': np.array([1, 1, 0, 1, 1], bool)}}
        return state, predictions, inputs

    def save(self, folder, state, predictions):
        np.savez(folder/'weights-5.npz', **state)
        np.savez(folder/'predictions-5.npz', **predictions)
        return {'validationIds': ['held'], 'artifacts': {name: audit.digest(folder/name)
                for name in ('weights-5.npz', 'predictions-5.npz')}}

    def test_both_actual_model_tensor_schemas_and_exact_scalers_pass(self):
        self.assertEqual(self.architectures['tcn']['parameterCount'], 29700)
        self.assertEqual(self.architectures['dino_tcn']['parameterCount'], 46868)
        for kind in self.architectures:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                state, predictions, inputs = self.fixture(folder, kind)
                meta = self.save(folder, state, predictions)
                result = audit.audit_checkpoint(folder, 5, meta, self.architectures[kind],
                                                 np.zeros(104, np.float32), np.ones(104, np.float32), inputs)
                self.assertTrue(result['scalerExactlyEqual'])
                self.assertEqual(result['predictions'][0]['ignoredTicks'], 1)

    def test_tampered_tensors_scalers_probabilities_and_ignored_ticks_are_rejected(self):
        corruptions = ('state_nan', 'state_shape', 'state_missing', 'mean_shift', 'scale_floor',
                       'prediction_nan', 'prediction_dtype', 'prediction_shape', 'prediction_bound', 'ignored_nonzero')
        for corruption in corruptions:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                state, predictions, inputs = self.fixture(folder, 'dino_tcn')
                key = 'model::token_projection.weight'
                if corruption == 'state_nan': state[key][0, 0] = np.nan
                elif corruption == 'state_shape': state[key] = state[key][:-1]
                elif corruption == 'state_missing': state.pop(key)
                elif corruption == 'mean_shift': state['mean'][0] = 1e-8
                elif corruption == 'scale_floor': state['scale'][0] = 1e-5
                elif corruption == 'prediction_nan': predictions['held'][0, 0] = np.nan
                elif corruption == 'prediction_dtype': predictions['held'] = predictions['held'].astype(np.float64)
                elif corruption == 'prediction_shape': predictions['held'] = predictions['held'][:, :3]
                elif corruption == 'prediction_bound': predictions['held'][0, 0] = 1.01
                elif corruption == 'ignored_nonzero': predictions['held'][2, 0] = .1
                meta = self.save(folder, state, predictions)
                with self.assertRaises(ValueError):
                    audit.audit_checkpoint(folder, 5, meta, self.architectures['dino_tcn'],
                                            np.zeros(104, np.float32), np.ones(104, np.float32), inputs)

    def test_hash_tamper_rejected_before_tensor_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            state, predictions, inputs = self.fixture(folder, 'tcn')
            meta = self.save(folder, state, predictions)
            path = folder/'weights-5.npz'
            path.write_bytes(path.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError, 'Changed checkpoint'):
                audit.audit_checkpoint(folder, 5, meta, self.architectures['tcn'],
                                        np.zeros(104, np.float32), np.ones(104, np.float32), inputs)


if __name__ == '__main__':
    unittest.main()
