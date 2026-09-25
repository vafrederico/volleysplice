import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/evaluate-distilled-mobile-large-panels.py'
SPEC = importlib.util.spec_from_file_location('distilled_large_panels', SCRIPT)
panels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panels)


def source():
    return dict(id='synthetic-record', sourceGroup='synthetic-group', contentSha256='a' * 64,
                durationSeconds=30., environment='indoor', scoringPolicy='exact-core',
                tier='completed-exact', rallies=[dict(start=5., end=7.)], ignoredIntervals=[])


def row(field='rallies'):
    return dict(id='synthetic-record', sourceGroup='synthetic-group', durationSeconds=30.,
                **{field: [dict(start=5., end=7.), dict(start=15., end=17.)]},
                ignoredIntervals=[], predictions=[dict(start=15., end=17.)])


class DistilledLargePanelTests(unittest.TestCase):
    def test_inventory_revision_replaces_historical_training_labels(self):
        old, latest = source(), source()
        latest['rallies'] = [dict(start=10., end=12.)]
        latest['ignoredIntervals'] = [dict(start=0., end=2.)]
        gold = panels.authoritative_gold(old, latest)
        self.assertEqual(gold['rallies'], latest['rallies'])
        self.assertEqual(gold['ignoredIntervals'], latest['ignoredIntervals'])

    def test_source_identity_and_training_proxy_labels_cannot_be_scored(self):
        for key, value in (('contentSha256', 'b' * 64), ('sourceGroup', 'another-group')):
            latest = source()
            latest[key] = value
            with self.assertRaisesRegex(ValueError, 'identity'):
                panels.authoritative_gold(source(), latest)
        latest = source()
        latest['experimentalSupervision'] = {'kind': 'export-proxy'}
        with self.assertRaisesRegex(ValueError, 'proxies'):
            panels.authoritative_gold(source(), latest)

    def test_unvalidated_or_partial_labels_are_not_promoted(self):
        unscored = source()
        unscored.update(scoringPolicy='none', tier='human-partially-reviewed-draft')
        self.assertIsNone(panels.authoritative_gold(unscored, unscored))
        wrongly_exact = {**unscored, 'scoringPolicy': 'exact-core'}
        with self.assertRaisesRegex(ValueError, 'quality'):
            panels.authoritative_gold(wrongly_exact, unscored)

    def test_older_unhashed_inventory_requires_full_manifest_media_binding(self):
        record, latest = source(), source()
        latest['contentSha256'] = None
        with self.assertRaisesRegex(ValueError, 'bound media'):
            panels.authoritative_gold(record, latest)
        evidence = {'path': 'synthetic-inventory', 'sha256': 'b' * 64}
        record.update(inventoryEvidence=evidence, video='synthetic-video',
                      mediaIdentity={'sizeBytes': 1234})
        latest.update(video='synthetic-video', mediaIdentity={'sizeBytes': 1234})
        self.assertEqual(panels.authoritative_gold(record, latest, evidence)['contentSha256'], 'a' * 64)
        latest['mediaIdentity'] = {'sizeBytes': 999}
        with self.assertRaisesRegex(ValueError, 'bound media'):
            panels.authoritative_gold(record, latest, evidence)

    def test_exact_panel_counts_wholly_missed_original_rallies(self):
        result = panels.evaluate_panel([row()], 'exact-rallies', 'all')
        self.assertEqual(result['foundRallies'], 1)
        self.assertEqual(result['whollyMissedSavedHumanRallies'], 1)
        self.assertEqual(result['recordings'][0]['whollyMissedSavedHumanRallies'], 1)
        self.assertEqual([value['paddingSecondsBeforeAndAfter'] for value in result['evaluation']['padding']], [0, 1, 2, 3])

    def test_export_panel_does_not_pad_human_export_or_claim_rally_core(self):
        sample = row('humanExportIntervals')
        sample['humanExportIntervals'] = [dict(start=15., end=17.)]
        result = panels.evaluate_panel([sample], 'reviewed-export', 'all')
        primary = result['evaluation']['primary']
        self.assertEqual(primary['humanExportSeconds'], 2.)
        self.assertAlmostEqual(primary['P_export'], 1 / 3)
        self.assertEqual(primary['R_export'], 1.)
        self.assertNotIn('R_core', primary)
        self.assertIsNone(result['whollyMissedSavedHumanRallies'])
        self.assertFalse(result['evaluation']['eventMetricsAvailable'])

    def test_draft_panel_has_only_reviewed_interval_metrics(self):
        result = panels.evaluate_panel([row('reviewedLiveIntervals')], 'reviewed-draft', 'all')
        self.assertIn('R_reviewed', result['evaluation']['primary'])
        self.assertNotIn('R_core', result['evaluation']['primary'])
        self.assertIsNone(result['whollyMissedSavedHumanRallies'])
        self.assertFalse(result['evaluation']['eventMetricsAvailable'])

    def test_unknown_exposure_is_not_clean_and_empty_scope_is_not_zero_error(self):
        self.assertFalse(panels.production_clean({}))
        self.assertFalse(panels.production_clean({'productionExposure': {'rallyPipeline': {'primaryTrainingClean': 1}}}))
        self.assertTrue(panels.production_clean({'productionExposure': {'rallyPipeline': {'primaryTrainingClean': True}}}))
        result = panels.evaluate_panel([], 'exact-rallies', 'no-production-training')
        self.assertEqual(result['status'], 'empty')
        self.assertIsNone(result['evaluation'])
        self.assertIsNone(result['whollyMissedSavedHumanRallies'])

    def test_only_exact_beach_labels_are_accepted(self):
        beach = {**source(), 'environment': 'beach'}
        self.assertEqual(panels.authoritative_gold(beach, beach)['scoringPolicy'], 'exact-core')
        beach['tier'] = 'unvalidated-candidate'
        with self.assertRaisesRegex(ValueError, 'exact saved'):
            panels.authoritative_gold(beach, beach)

    def test_label_using_prediction_is_rejected_before_decoding(self):
        record = source()
        receipt = dict(recordingId=record['id'], contentSha256=record['contentSha256'],
                       durationSeconds=record['durationSeconds'], plan={'sha256': 'plan'},
                       labelsUsed=True, ignoredIntervalsUsed=False)
        chosen = dict(setting={'epoch': 5, 'decoder': {}})
        with patch.object(panels, 'read', return_value=receipt), \
                patch.object(panels, 'identity', return_value={'sha256': 'plan'}), \
                patch.object(panels, 'decode') as decode:
            with self.assertRaisesRegex(ValueError, 'label isolation'):
                panels.prediction(Path('synthetic'), Path('synthetic/fit'), record, chosen, {}, {})
            decode.assert_not_called()

    def test_saved_boundary_replay_uses_catalog_duration_after_av_validation(self):
        record = {**source(), 'durationSeconds': 2., 'featureOrigin': 'synthetic'}
        times = np.arange(.125, 2., .25)
        scores = np.ones((len(times), 4), np.float32)
        decoder = dict(smoothing=.5, enter=.5, minimum=.25, boundary=False)
        example = panels.inputs.base.Example(record['id'], record['sourceGroup'],
            1.999956, times, np.zeros((len(times), 104)), np.zeros((len(times), 3)),
            np.ones(len(times), bool), (), (), 'indoor')
        self.assertEqual(panels.decode(example, scores, decoder)[-1].end, 1.999956)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'fit'
            (folder / 'inference').mkdir(parents=True)
            output = folder / 'inference' / (record['id'] + '.npz')
            np.savez(output, times=times, epoch_5=scores)
            receipt = dict(recordingId=record['id'], contentSha256=record['contentSha256'],
                durationSeconds=2., plan={'sha256': 'plan'}, labelsUsed=False,
                ignoredIntervalsUsed=False, weights={'5': {}}, studentWeights={},
                decoders={'5': decoder}, output={'path': str(output)},
                imageInput={'path': str(root / 'image.json')}, audiovisual={},
                decodedRallies={'5': [dict(start=0., end=2.)]})
            image = dict(id=record['id'], sourceGroup=record['sourceGroup'],
                contract={'source': {'contentSha256': record['contentSha256']}})
            chosen = dict(setting={'epoch': 5, 'decoder': decoder})
            with patch.object(panels, 'read', side_effect=lambda path:
                    image if Path(path).name == 'image.json' else receipt), \
                    patch.object(panels, 'identity', return_value={'sha256': 'plan'}), \
                    patch.object(panels.inputs, 'verified', side_effect=lambda ref: Path(ref['path'])), \
                    patch.object(panels.inputs, 'example_from_row', return_value=example):
                _, rallies, _ = panels.prediction(root, folder, record, chosen, {}, {})
                self.assertEqual(rallies, [dict(start=0., end=2.)])
                receipt['decodedRallies']['5'][0]['end'] = 1.99
                with self.assertRaisesRegex(ValueError, 'Saved rally boundaries'):
                    panels.prediction(root, folder, record, chosen, {}, {})


if __name__ == '__main__':
    unittest.main()
