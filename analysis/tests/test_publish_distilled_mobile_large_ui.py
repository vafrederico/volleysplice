"""Publication integrity and draft-isolation tests using synthetic recordings."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('distilled_large_publisher', REPO/'scripts/publish-distilled-mobile-large-ui.py')
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
RALLIES = [dict(start=3., end=6.)]
DECODER = dict(smoothing=.5, enter=.5, minimum=.25, boundary=False)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(publisher.encoded(value))


class DistilledLargePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.experiment = self.root/'experiment'
        self.index = self.root/'comparison/index.json'
        self.rows = [dict(id=f'recording-{i:03d}', sourceGroup=f'source-group-{i:03d}',
                         durationSeconds=10., environment=environment, contentSha256='b'*64)
                     for i, environment in ((1, 'indoor'), (2, 'beach'))]
        save(self.experiment/'plan.json', dict(floorsPercent=[99], paddingSeconds=[0, 1, 2, 3]))
        self.plan = publisher.identity(self.experiment/'plan.json')
        evaluator = self.experiment/'evaluator.py'
        evaluator.write_text('# synthetic frozen evaluator', encoding='utf-8')
        self.evaluator = publisher.identity(evaluator)
        save(self.experiment/'catalog-manifest.json', dict(records=self.rows))
        candidates = [dict(variant='registered-variant', draw=draw, seed=draw+100, floorPercent=99,
                           setting=dict(epoch=5, decoder=DECODER, innerR_core=.995),
                           evaluation=dict(primary=dict(P_pad=precision, R_core=recall, F1_padP_coreR=f1)))
                      for draw, precision, recall, f1 in ((7, .94, .97, .955), (9, .86, .998, .924))]
        self.evaluation = dict(plan=self.plan, sourceCode=self.evaluator, candidates=candidates,
                               selected=[dict(candidates[0], mode='f1'), dict(candidates[1], mode='recall')])
        save(self.experiment/'evaluation.json', self.evaluation)
        catalog = dict(schemaVersion=1, kind='volleycut-neural-comparison-index',
                       models=[dict(modelId='existing')], recordings=[])
        for row in self.rows:
            entry = dict(id=row['id'], name=row['id'], file=f"recordings/{row['id']}.json",
                         tier='completed-exact', modelIds=['existing'])
            catalog['recordings'].append(entry)
            reference = dict(modelId='existing', rallies=[dict(start=1., end=2.)],
                             research=dict(signals=dict(times=[1.5], live=[.9]), provenance={}))
            if row['environment'] == 'beach':
                reference['research']['provenance']['uiDraftRevision'] = 'e'*64
            save(self.index.parent/entry['file'], dict(schemaVersion=1, recordings=[dict(
                recordingId=row['id'], durationSeconds=10., contentSha256='a'*64,
                videoFilename=row['id']+'.mp4', references=[reference])]))
            for chosen in candidates:
                fit = self.experiment/'fits'/chosen['variant']/f"split-{chosen['draw']}"
                weights, student = fit/'temporal/weights-5.npz', fit/'student/weights.npz'
                for artifact in (weights, student):
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_bytes(b'frozen-checkpoint')
                save(fit/'selection.json', dict(plan=self.plan,
                    task=dict(variant=chosen['variant'], splitSeed=chosen['draw'], seed=chosen['seed']),
                    studentWeights=publisher.identity(student),
                    floors=[dict(floorPercent=99, feasible=True, selected=chosen['setting'])]))
                save(fit/'student/completed.json', dict(weights=publisher.identity(student)))
                save(fit/'temporal/completed.json', dict(artifacts={weights.name: publisher.sha(weights)}))
                score_path = fit/'inference'/(row['id']+'.npz')
                score_path.parent.mkdir(parents=True, exist_ok=True)
                times = np.arange(.125, 10., .25)
                scores = np.zeros((len(times), 4), np.float32)
                scores[(times >= 3) & (times < 6), 0] = .95
                np.savez_compressed(score_path, times=times, epoch_5=scores)
                save(score_path.with_suffix('.json'), dict(
                    recordingId=row['id'], labelsUsed=False, ignoredIntervalsUsed=False, plan=self.plan,
                    sourceGroup=row['sourceGroup'], sourceCode=self.evaluator,
                    contentSha256=row['contentSha256'], durationSeconds=row['durationSeconds'],
                    output=publisher.identity(score_path), weights={'5': publisher.identity(weights)},
                    studentWeights=publisher.identity(student), decoders={'5': DECODER},
                    decodedRallies={'5': RALLIES}))
        save(self.index, catalog)
        self.originals = {path: path.read_bytes() for path in self.index.parent.rglob('*.json')}
        self.decode_patch = patch.object(publisher, 'decode_scores', return_value=copy.deepcopy(RALLIES))
        self.decode_patch.start()
        self.addCleanup(self.decode_patch.stop)

    def prepare(self):
        return publisher.prepare(self.experiment, self.index, expected_recordings=2)

    def receipt(self, row=0):
        return self.experiment/'fits/registered-variant/split-7/inference'/(self.rows[row]['id']+'.json')

    def assert_originals(self):
        for path, original in self.originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_all_recordings_and_both_modes_preserve_models_and_drafts(self):
        prepared = self.prepare()
        self.assert_originals()
        publisher.publish(self.experiment, self.index, prepared)
        index = publisher.read(self.index)
        self.assertEqual([model['modelId'] for model in index['models']], ['existing', *publisher.MODEL_IDS])
        self.assertEqual([model['seed'] for model in prepared['models']], [7, 9])
        for number, entry in enumerate(index['recordings']):
            path = self.index.parent/entry['file']
            refs = publisher.read(path)['recordings'][0]['references']
            prior = refs[0]
            expected_revision = hashlib.sha256(self.originals[path]).hexdigest() if number == 0 else 'e'*64
            self.assertEqual(prior['research']['provenance']['uiDraftRevision'], expected_revision)
            self.assertEqual(prior['rallies'], [dict(start=1., end=2.)])
            self.assertEqual(entry['modelIds'], ['existing', *publisher.MODEL_IDS])
            for reference in refs[1:]:
                self.assertEqual(reference['rallies'], RALLIES)
                self.assertEqual(set(reference['research']['signals']), {'times', *publisher.HEADS})
                self.assertNotIn('source', reference['research']['provenance'])
        before = {path: path.read_bytes() for path in self.index.parent.rglob('*.json')}
        publisher.publish(self.experiment, self.index, self.prepare())
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_missing_beach_inference_fails_before_any_publication(self):
        self.receipt(1).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assert_originals()

    def test_catalog_cannot_silently_omit_a_ui_video(self):
        save(self.experiment/'catalog-manifest.json', dict(records=self.rows[:1]))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.prepare()
        self.assert_originals()

    def test_hash_recording_label_decoder_and_saved_prediction_guards(self):
        path = self.receipt()
        original = publisher.read(path)
        mutations = [dict(output={**original['output'], 'sha256': '0'*64}),
                     dict(recordingId='different-recording'), dict(labelsUsed=True),
                     dict(sourceGroup='different-source-group'),
                     dict(sourceCode={**self.evaluator, 'sha256': '0'*64}),
                     dict(contentSha256='f'*64), dict(durationSeconds=11.), dict(ignoredIntervalsUsed=None),
                     dict(decoders={'5': {**DECODER, 'enter': .65}}),
                     dict(decodedRallies={'5': []})]
        for mutation in mutations:
            with self.subTest(mutation=next(iter(mutation))):
                save(path, {**original, **mutation})
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assert_originals()
        save(path, original)

    def test_valid_but_foreign_student_receipt_is_rejected(self):
        path = self.receipt()
        receipt = publisher.read(path)
        other = self.experiment/'fits/registered-variant/split-9/student/weights.npz'
        receipt['studentWeights'] = publisher.identity(other)
        save(path, receipt)
        with self.assertRaisesRegex(ValueError, 'Student checkpoint owner differs'):
            self.prepare()
        self.assert_originals()

    def test_selected_fit_checkpoint_and_decoder_ownership_are_required(self):
        fit = self.receipt().parent.parent
        for filename, mutate in (
            ('selection.json', lambda value: value['floors'][0]['selected'].update(epoch=15)),
            ('temporal/completed.json', lambda value: value['artifacts'].update({'weights-5.npz': '0'*64})),
        ):
            path = fit/filename
            old = path.read_bytes()
            value = publisher.read(path)
            mutate(value)
            save(path, value)
            with self.assertRaises(ValueError):
                self.prepare()
            self.assert_originals()
            path.write_bytes(old)

    def test_corrupt_student_checkpoint_is_not_published(self):
        receipt = publisher.read(self.receipt())
        Path(receipt['studentWeights']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.prepare()
        self.assert_originals()

    def test_rank_or_recall_floor_changes_are_rejected(self):
        for mutation in ('rank', 'floor'):
            evaluation = copy.deepcopy(self.evaluation)
            if mutation == 'rank':
                evaluation['selected'][0] = dict(evaluation['candidates'][1], mode='f1')
            else:
                evaluation['candidates'][0]['setting']['innerR_core'] = .989
                evaluation['selected'][0]['setting']['innerR_core'] = .989
            save(self.experiment/'evaluation.json', evaluation)
            with self.assertRaises(ValueError):
                self.prepare()
            self.assert_originals()

    def test_same_fit_can_have_two_separate_ui_ids(self):
        evaluation = copy.deepcopy(self.evaluation)
        evaluation['candidates'] = evaluation['candidates'][:1]
        evaluation['selected'] = [dict(evaluation['candidates'][0], mode=mode) for mode in ('f1', 'recall')]
        save(self.experiment/'evaluation.json', evaluation)
        prepared = self.prepare()
        self.assertEqual(len({model['modelId'] for model in prepared['models']}), 2)
        self.assertEqual({model['seed'] for model in prepared['models']}, {7})

    def test_concurrent_update_is_preserved_and_publication_is_refused(self):
        prepared = self.prepare()
        path = prepared['changes'][0][0]
        newer = publisher.read(path)
        newer['externalRevision'] = 1
        save(path, newer)
        with self.assertRaisesRegex(ValueError, 'changed during preparation'):
            publisher.publish(self.experiment, self.index, prepared)
        self.assertEqual(publisher.read(path), newer)
        self.assertEqual(self.index.read_bytes(), self.originals[self.index])

    def test_io_failure_rolls_back_recordings_and_index(self):
        prepared = self.prepare()
        replace = publisher.replace_bytes
        calls = 0

        def fail_second(path, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('synthetic interrupted publication')
            replace(path, value)

        with patch.object(publisher, 'replace_bytes', side_effect=fail_second):
            with self.assertRaises(OSError):
                publisher.publish(self.experiment, self.index, prepared)
        self.assert_originals()


if __name__ == '__main__':
    unittest.main()
