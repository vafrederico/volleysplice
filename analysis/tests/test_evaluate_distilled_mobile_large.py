"""Adversarial provenance/resume and label-isolation checks without real videos."""
from contextlib import ExitStack
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('distilled_large_evaluation', REPO / 'scripts/evaluate-distilled-mobile-large.py')
evaluation = importlib.util.module_from_spec(SPEC)
with patch('analysis.private_ledger.private_value', side_effect=lambda key: '/synthetic/' + key):
    SPEC.loader.exec_module(evaluation)
evaluation.DEVICE = 'cpu'
DECODER = dict(smoothing=.5, enter=.5, minimum=.25, boundary=False)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class DistilledLargeEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.row = dict(id='recording-test', sourceGroup='source-test', contentSha256='a' * 64,
                        durationSeconds=4., environment='indoor', featureOrigin='synthetic',
                        rallies=[dict(start=1., end=2.)], ignoredIntervals=[dict(start=0., end=1.)])
        self.task = dict(variant='test-variant', splitSeed=7, seed=3407)
        self.folder = evaluation.experiment.fit_folder(self.root, self.task)
        self.setting = dict(epoch=5, decoder=DECODER)
        self.selection = dict(floors=[dict(feasible=True, floorPercent=99, selected=self.setting)])
        weight_path = self.folder / 'temporal/weights-5.npz'
        student_path = self.folder / 'student/weights.npz'
        for path in (weight_path, student_path, self.root / 'checkpoint.pth', self.root / 'av.npz'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic-artifact')
        self.weights = evaluation.identity(weight_path)
        self.metadata = dict(weights=evaluation.identity(student_path))
        self.completed = dict(artifacts={weight_path.name: self.weights['sha256']})
        self.plan = dict(initialCheckpoint=evaluation.identity(self.root / 'checkpoint.pth'))
        save(self.root / 'plan.json', self.plan)
        save(self.root / 'image.json', dict(id=self.row['id'], sourceGroup=self.row['sourceGroup'],
             contract=dict(source=dict(contentSha256=self.row['contentSha256']))))
        self.entries = {self.row['id']: dict(audiovisual=evaluation.identity(self.root / 'av.npz'),
                                            imageInput=evaluation.identity(self.root / 'image.json'))}
        self.times = np.arange(.125, 4., .25)
        self.scores = np.zeros((len(self.times), 4), np.float32)
        self.scores[:, 0] = .95
        self.output = self.folder / 'inference' / (self.row['id'] + '.npz')
        self.receipt_path = self.output.with_suffix('.json')
        self.arrays = dict(times=self.times, epoch_5=self.scores)

    def receipt(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output, **self.arrays)
        return dict(recordingId=self.row['id'], sourceGroup=self.row['sourceGroup'],
            contentSha256=self.row['contentSha256'], durationSeconds=self.row['durationSeconds'],
            plan=evaluation.identity(self.root / 'plan.json'), sourceCode=evaluation.identity(evaluation.__file__),
            weights={'5': self.weights}, studentWeights=self.metadata['weights'], decoders={'5': DECODER},
            labelsUsed=False, ignoredIntervalsUsed=False, **self.entries[self.row['id']],
            output=evaluation.identity(self.output), decodedRallies={'5': evaluation.decoded_arrays(self.row, self.arrays, 5, DECODER)})

    def inference(self, *, row=None):
        def run():
            return evaluation.infer(self.root, self.plan, self.task, [self.row if row is None else row], self.entries)

        with ExitStack() as local:
            local.enter_context(patch.object(evaluation, 'validated_fit', return_value=(
                self.folder, self.selection, self.completed, self.metadata)))
            local.enter_context(patch.object(evaluation, 'load_checkpoint', return_value=('model', 0., 1.)))
            local.enter_context(patch.object(evaluation.student, 'load_encoder', return_value='encoder'))
            local.enter_context(patch.object(evaluation.experiment, 'status'))
            return run()

    def test_resume_rejects_old_inference_code_even_when_saved_scores_hash_correctly(self):
        receipt = self.receipt()
        receipt['sourceCode']['sha256'] = '0' * 64
        save(self.receipt_path, receipt)
        with self.assertRaisesRegex(AssertionError, 'resume identity differs'):
            self.inference()

    def test_resume_rejects_a_saved_rally_list_that_disagrees_with_its_scores(self):
        receipt = self.receipt()
        receipt['decodedRallies']['5'] = []
        save(self.receipt_path, receipt)
        with self.assertRaisesRegex(AssertionError, 'Resumed rallies differ'):
            self.inference()

    def test_resume_and_decode_are_invariant_to_human_labels_and_ignored_spans(self):
        save(self.receipt_path, self.receipt())
        changed = {**self.row, 'rallies': [], 'ignoredIntervals': [dict(start=0., end=4.)],
                   'gameWindow': dict(start=1., end=2.)}
        with patch.object(evaluation, 'predict', side_effect=AssertionError('cached inference should not rerun')):
            first, second = self.inference(), self.inference(row=changed)
        np.testing.assert_array_equal(first[self.row['id']]['epoch_5'], second[self.row['id']]['epoch_5'])
        self.assertEqual(evaluation.decoded_arrays(self.row, self.arrays, 5, DECODER),
                         evaluation.decoded_arrays(changed, self.arrays, 5, DECODER))
        self.assertEqual(evaluation.decoded_arrays(changed, self.arrays, 5, DECODER), [dict(start=0., end=4.)])

    def test_resume_rejects_invalid_probability_arrays_after_matching_output_hash(self):
        receipt = self.receipt()
        self.arrays['epoch_5'] = np.full_like(self.scores, np.nan)
        np.savez_compressed(self.output, **self.arrays)
        receipt['output'] = evaluation.identity(self.output)
        save(self.receipt_path, receipt)
        with self.assertRaisesRegex(AssertionError, 'Invalid inference probabilities'):
            self.inference()

    def test_fresh_inference_strips_labels_before_feature_example_construction(self):
        allowed = {'id', 'sourceGroup', 'durationSeconds', 'environment', 'featureOrigin'}

        def blind_example(row, features, *, inference):
            self.assertEqual(set(row), allowed)
            self.assertIs(inference, True)
            return SimpleNamespace(id=row['id'], group=row['sourceGroup'], duration=4., times=self.times,
                                   valid=np.ones(len(self.times), bool), truth=(), ignored=())

        with patch.object(evaluation.inputs, 'example_from_row', side_effect=blind_example) as loader, \
             patch.object(evaluation.experiment, 'attach', side_effect=lambda example, _: example), \
             patch.object(evaluation.student, 'extract_student_record'), \
             patch.object(evaluation, 'predict', return_value=self.scores):
            result = self.inference()
        loader.assert_called_once()
        np.testing.assert_array_equal(result[self.row['id']]['epoch_5'], self.scores)
        receipt = evaluation.read(self.receipt_path)
        self.assertFalse(receipt['labelsUsed'])
        self.assertFalse(receipt['ignoredIntervalsUsed'])
        self.assertEqual(receipt['decodedRallies']['5'], [dict(start=0., end=4.)])

    def test_validated_fit_rejects_valid_student_weight_file_owned_by_another_fit(self):
        task = {**self.task, 'epochs': [5], 'trainIds': ['training-record'], 'calibrationIds': [],
                'commonEvaluationGroups': ['common-source'], 'manifest': {'path': str(self.root / 'manifest.json')}}
        folder = evaluation.experiment.fit_folder(self.root, task)
        plan_ref = evaluation.identity(self.root / 'plan.json')
        digest = evaluation.sweep.canonical({'plan': plan_ref, 'task': task})
        foreign = self.root / 'different-fit/student/weights.npz'
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(b'valid-but-different-owner')
        weights = evaluation.identity(foreign)
        records = [dict(id='training-record', sourceGroup='training-source', labelTier='exact',
                        eligibleRoles=['fit'], consent={'train': True}, protected=False, environment='indoor')]
        selection = dict(plan=plan_ref, task=task, contractSha256=digest, config=asdict(evaluation.experiment.CONFIG),
                         floors=[], candidates=[], studentWeights=weights)
        completed = dict(contractSha256=digest, seed=3407, epochs=[5], kind='mobile_tcn', lossArm='short_boost',
            model={**asdict(evaluation.experiment.CONFIG), 'inputDimension': 3952},
            trainIds=['training-record'], auxiliaryIds={'draft': [], 'coverage': []},
            scalerTrainIds=['training-record'], validationIds=[], artifacts={})
        encoder = dict(contractSha256=digest, seed=3407, trainIds=['training-record'],
                       trainGroups=['training-source'], weights=weights, recipe=evaluation.student.RECIPE)
        documents = {str(folder / 'selection.json'): selection, str(folder / 'temporal/completed.json'): completed,
                     str(folder / 'student/completed.json'): encoder, str(self.root / 'manifest.json'): {'records': records}}
        with patch.object(evaluation, 'read', side_effect=lambda path: documents[str(path)]), \
             patch.object(evaluation.inputs, 'verified', side_effect=lambda reference: Path(reference['path'])), \
             patch.object(evaluation.sweep, 'select_floors', return_value=[]):
            with self.assertRaisesRegex(AssertionError, 'Student weight owner differs'):
                evaluation.validated_fit(self.root, task)

    def test_common_selection_recomputes_and_refuses_stale_saved_choice(self):
        row = {**self.row, 'scoringPolicy': 'exact-core'}
        plan = {**self.plan, 'tasks': [self.task]}
        with patch.object(evaluation, 'COMMON_GROUPS', {row['sourceGroup']}), \
             patch.object(evaluation, 'validated_fit', return_value=(
                 self.folder, self.selection, self.completed, self.metadata)), \
             patch.object(evaluation, 'infer', return_value={row['id']: self.arrays}) as inference, \
             patch.object(evaluation.experiment, 'status'):
            first = evaluation.select(self.root, plan, [row], self.entries)
            stale = {**first, 'selected': []}
            save(self.root / 'evaluation.json', stale)
            with self.assertRaisesRegex(AssertionError, 'Frozen selection differs'):
                evaluation.select(self.root, plan, [row], self.entries)
        self.assertEqual(inference.call_count, 2)
        self.assertEqual(evaluation.read(self.root / 'evaluation.json'), stale)


if __name__ == '__main__':
    unittest.main()
