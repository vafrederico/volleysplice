from __future__ import annotations
from analysis.private_ledger import private_value

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/summarize-neural-short-boost-transfer.py'
SPEC = importlib.util.spec_from_file_location('short_boost_summary_tested', SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def evaluation():
    return summary.evaluate_predictions([{'id': 'x', 'sourceGroup': 'g', 'durationSeconds': 30,
                                          'rallies': [{'start': 2, 'end': 4}, {'start': 10, 'end': 18}],
                                          'predictions': [{'start': 2, 'end': 4}, {'start': 10, 'end': 18}]}])


def pts_fixture():
    times = np.arange(4, dtype=np.float64)/4
    ticks = np.array([0, 1, 3, 5, 7], dtype=np.int64)
    pts = ticks.astype(np.float64)/8
    indexes = np.arange(4, dtype=np.int64)
    frames = np.array([letter*64 for letter in 'abcd'])
    arrays = {'times': times, 'packetPtsTicks': ticks, 'presentationTimes': pts, 'selectedOrdinals': indexes,
              'selectedPresentationTimes': pts[indexes], 'observedPresentationTimes': pts[indexes].copy(),
              'selectedFrameSha256': frames}
    selection = {'decoder': summary.PTS_DECODER, 'sampleCount': 4, 'videoFrameCount': 5, 'timeBase': '1/8',
                 'videoDurationSeconds': 1., 'maximumFrameSelectionErrorSeconds': .125, 'maximumDecodedPtsErrorSeconds': 0.,
                 'audio': {'formulaUnchanged': True, 'sampleRate': 16000, 'startOffsetSeconds': .01, 'alignmentSamples': 160,
                           'stream': {'start_time': '.01', 'sample_rate': '48000', 'firstDecodedFramePtsSeconds': .01}}}
    for field, name, dtype in [('gridSha256', 'times', '<f8'), ('presentationTimesSha256', 'presentationTimes', '<f8'),
                               ('selectedOrdinalsSha256', 'selectedOrdinals', '<i8'),
                               ('selectedPresentationTimesSha256', 'selectedPresentationTimes', '<f8')]:
        selection[field] = hashlib.sha256(arrays[name].astype(dtype).tobytes()).hexdigest()
    selection['frameHashesSha256'] = hashlib.sha256('\n'.join(frames).encode()).hexdigest()
    return selection, arrays


class ShortBoostTransferSummaryTests(unittest.TestCase):
    def test_pts_audit_uses_media_clock_and_earlier_ties(self):
        selection, arrays = pts_fixture()
        checked = summary.verify_pts_arrays(selection, arrays, arrays['times'], arrays['times'])
        self.assertEqual(checked['samples'], 4)
        self.assertEqual(checked['maximumSelectionErrorSeconds'], .125)
        changed = copy.deepcopy(arrays)
        changed['selectedOrdinals'][1] = 2
        with self.assertRaisesRegex(ValueError, 'nearest with earlier ties'):
            summary.verify_pts_arrays(selection, changed, arrays['times'], arrays['times'])

    def test_pts_audit_rejects_modality_clock_and_audio_drift(self):
        selection, arrays = pts_fixture()
        shifted = arrays['times'].copy()
        shifted[-1] -= .01
        with self.assertRaisesRegex(ValueError, 'share exact media4Hz'):
            summary.verify_pts_arrays(selection, arrays, shifted, arrays['times'])
        changed = copy.deepcopy(arrays)
        changed['observedPresentationTimes'][1] += 3e-6
        with self.assertRaisesRegex(ValueError, 'PTS errors'):
            summary.verify_pts_arrays(selection, changed, arrays['times'], arrays['times'])
        selection['audio']['stream']['firstDecodedFramePtsSeconds'] = .02
        with self.assertRaisesRegex(ValueError, 'Audio first decoded frame'):
            summary.verify_pts_arrays(selection, arrays, arrays['times'], arrays['times'])

    def test_feature_amendment_changes_only_seven_coverage_cache_objects(self):
        original = {'exactRows': [{'id': 'gold', 'rallies': [{'start': 1, 'end': 2}]}], 'draftRows': [],
                    'exactManifest': {'path': 'gold.json', 'sha256': 'gold-sha'},
                    'coverageRows': [{'id': f'pixel-{index}', 'sourceGroup': 'pixel-session',
                                      'ignoredIntervals': [{'start': 0, 'end': 1}],
                                      'featureCaches': {'audiovisual': {'path': f'old-{index}', 'sha256': f'old-sha-{index}',
                                                                     'names': list(range(104)), 'featureNames': list(range(104))}}}
                                     for index in range(7)]}
        amended = copy.deepcopy(original)
        amended.update(originalManifest={}, protocolAmendment={}, featureRevision={})
        for index, row in enumerate(amended['coverageRows']):
            row['featureCaches']['audiovisual'].update(path=f'new-{index}', sha256=f'new-sha-{index}')
        self.assertEqual(len(summary.audit_feature_revision(original, amended)['changedCoverageRecordings']), 7)
        changed = copy.deepcopy(amended)
        changed['coverageRows'][0]['ignoredIntervals'][0]['end'] = 2
        with self.assertRaisesRegex(ValueError, 'labels/source identity'):
            summary.audit_feature_revision(original, changed)
        changed = copy.deepcopy(amended)
        changed['exactRows'][0]['rallies'][0]['end'] = 3
        with self.assertRaisesRegex(ValueError, 'exactRows'):
            summary.audit_feature_revision(original, changed)

    def test_coverage_baseline_must_have_fresh_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            reference = {'path': str(study / 'old'), 'reportSha256': 'report', 'contractSha256': 'contract'}
            row = {'cohort': 'reviewed_export', 'kind': 'tcn', 'architecture': 'tcn', 'lossArm': 'baseline', 'seed': 3407,
                   'origin': {'type': 'reused-reference'}}
            with self.assertRaisesRegex(ValueError, 'Fresh fit origin'):
                summary.validate_origin(row, study, reference, {})
            row['origin'] = {'type': 'trained', 'fitRoot': str(study / 'fits/reviewed_export/tcn/baseline/3407')}
            self.assertEqual(summary.validate_origin(row, study, reference, {}), Path(row['origin']['fitRoot']))

    def test_unchanged_baseline_payload_and_old_fit_provenance_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            reference = {'path': str(study / 'old'), 'reportSha256': 'report', 'contractSha256': 'old-contract'}
            old = {'cohort': 'draft', 'kind': 'tcn', 'seed': 3407, 'selections': [{'epoch': 5}]}
            row = {**copy.deepcopy(old), 'architecture': 'tcn', 'lossArm': 'baseline', 'contractSha256': 'new-contract',
                   'origin': {'type': 'reused-reference', 'studyPath': reference['path'], 'reportSha256': 'report',
                              'referenceContractSha256': 'old-contract', 'fitRoot': str(study / 'old/fits/draft/tcn/3407')}}
            historical = {('draft', 'tcn', 3407): old}
            summary.validate_origin(row, study, reference, historical)
            changed = copy.deepcopy(row)
            changed['selections'][0]['epoch'] = 15
            with self.assertRaisesRegex(ValueError, 'payload changed'):
                summary.validate_origin(changed, study, reference, historical)
            (study / 'fits/draft/tcn/baseline/3407').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'fabricated local fits'):
                summary.validate_origin(row, study, reference, historical)

    def test_alignment_uses_earlier_sample_at_exact_tie(self):
        actual = summary.alignment_diagnostics(np.array([.125, .375]), np.array([0., .25, .5]))
        self.assertEqual(actual['tieCount'], 2)
        self.assertEqual(actual['maximumErrorSeconds'], .125)
        self.assertEqual(actual['nearestIndexesSha256'], hashlib.sha256(np.array([0, 1], dtype='<i8').tobytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'half an AV tick'):
            summary.alignment_diagnostics(np.array([.13]), np.array([0., .5]))

    def test_only_explicit_nas_prefix_is_canonicalized(self):
        self.assertEqual(summary.canonical_nas_path(private_value('private-reference-0075')), private_value('private-reference-0072'))
        self.assertEqual(summary.canonical_nas_path(private_value('private-reference-0073')), private_value('private-reference-0073'))
        self.assertNotEqual(summary.canonical_nas_path(private_value('private-reference-0076')), private_value('private-reference-0074'))

    def test_exact_three_seconds_is_short_long_weights_stay_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.npz'
            times = np.arange(0, 15, .25)
            np.savez(path, times=times)
            row = {'id': 'x', 'sourceGroup': 'g', 'featureCaches': {'audiovisual': {'path': str(path)}},
                   'rallies': [{'start': 1., 'end': 4.}, {'start': 6., 'end': 10.}],
                   'ignoredIntervals': [{'start': 2., 'end': 3.}]}
            boost, stats = summary.expected_loss_weights(row, 'exact', 'short_boost')
            control, global_stats = summary.expected_loss_weights(row, 'exact', 'global_control')
            self.assertEqual(stats['shortPositiveSupervisedTicks'], 8)
            self.assertEqual(stats['longPositiveSupervisedTicks'], 16)
            self.assertTrue(np.all(boost[(times >= 6) & (times < 10)] == 1))
            self.assertTrue(np.all(boost[(times >= 2) & (times < 3)] == 1))
            self.assertEqual(stats['weightedPositiveMass'], 32)
            self.assertAlmostEqual(global_stats['weightedPositiveMass'], 32, places=5)
            self.assertTrue(np.all(control[(times >= 6) & (times < 10)] > 1))

    def test_draft_single_surviving_tick_does_not_turn_long_event_short(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.npz'
            np.savez(path, times=np.arange(0, 7, .25))
            row = {'id': 'x', 'sourceGroup': 'g', 'featureCaches': {'audiovisual': {'path': str(path)}},
                   'rallies': [{'start': 1., 'end': 5.}], 'ignoredIntervals': [{'start': 3.3, 'end': 3.5}]}
            weights, stats = summary.expected_loss_weights(row, 'draft', 'short_boost')
            self.assertEqual(stats['positiveSupervisedTicks'], 1)
            self.assertEqual(stats['shortPositiveSupervisedTicks'], 0)
            self.assertTrue(np.all(weights == 1))

    def test_zero_positive_coverage_control_is_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.npz'
            np.savez(path, times=np.arange(0, 4, .25))
            row = {'id': 'x', 'sourceGroup': 'g', 'featureCaches': {'audiovisual': {'path': str(path)}},
                   'gameWindow': {'start': 0, 'end': 4}, 'rallies': [], 'ignoredIntervals': []}
            weights, stats = summary.expected_loss_weights(row, 'coverage', 'global_control')
            self.assertEqual(stats['idealGlobalPositiveMultiplier'], 1)
            self.assertEqual(stats['weightedPositiveMass'], 0)
            self.assertTrue(np.all(weights == 1))

    def test_difference_in_differences_uses_effects_not_raw_architecture_gap(self):
        dc, db, cc, cb = (evaluation() for _ in range(4))
        for row, score in ((dc, .93), (db, .90), (cc, .83), (cb, .80)):
            row['primary']['F1_padP_coreR'] = score
        report = summary.difference_in_differences(dc, db, cc, cb)
        self.assertAlmostEqual(report['dinoWithinArchitectureEffect']['F1_padP_coreR'], .03)
        self.assertAlmostEqual(report['compactWithinArchitectureEffect']['F1_padP_coreR'], .03)
        self.assertAlmostEqual(report['differenceInDifferences']['F1_padP_coreR'], 0)

    def test_group_difference_in_differences_checks_group_gold(self):
        group = evaluation()['sourceGroups']['g']
        differing = copy.deepcopy(group)
        differing['objective'] -= .1
        differing['primary']['F1_padP_coreR'] -= .1
        report = summary.difference_in_differences(group, differing, group, group)
        self.assertAlmostEqual(report['differenceInDifferences']['F1_padP_coreR'], .1)
        changed = copy.deepcopy(group)
        changed['guardrails']['primaryExportCoverage']['rallies'][0]['start'] += .1
        with self.assertRaisesRegex(ValueError, 'event identities'):
            summary.difference_in_differences(group, changed, group, group)

    def test_replication_needs_both_controls_in_both_architectures_without_positive_did(self):
        rows = [{'cohort': 'exact', 'kind': kind, 'candidateArm': 'short_boost', 'referenceArm': reference,
                 'retentionRecoveryScreen': {'screenPassedAndInnerFeasible': True}}
                for kind in summary.KINDS for reference in ('baseline', 'global_control')]
        report = summary.replication_screen(rows, 'exact')
        self.assertTrue(report['passed'])
        self.assertFalse(report['positiveDifferenceInDifferencesRequired'])
        changed = copy.deepcopy(rows)
        changed[-1]['retentionRecoveryScreen']['screenPassedAndInnerFeasible'] = False
        self.assertFalse(summary.replication_screen(changed, 'exact')['passed'])
        with self.assertRaisesRegex(ValueError, 'four'):
            summary.replication_screen(rows[:-1], 'exact')


if __name__ == '__main__':
    unittest.main()
