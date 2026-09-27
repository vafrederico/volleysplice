#!/usr/bin/env python3
"""Prove generalized loader parity against immutable legacy inputs; no fitting."""
from pathlib import Path
import gc
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_generalization_inputs as current
from analysis import neural_short_boost_transfer as original
from analysis.neural_context_development import identity, read, write_immutable

NAS = Path(private_value('private-reference-0057'))
OUTPUT = NAS/'2026-09-23-recall-sweep-generalization/features-v1/reference-qualification-v1'
SOURCE = NAS/'2026-09-19-short-boost-transfer/manifest-pts-v1.json'
DINO = SOURCE.with_name('dino-manifest.json')
MOBILE = NAS/'2026-09-22-recognition/mobile-features-v1/training-index.json'
DISTILL = NAS/'2026-09-23-recall-distillation'


def compare(left, right):
    checked = []
    for tier in current.TIERS:
        assert len(left[tier]) == len(right[tier])
        for a, b in zip(left[tier], right[tier], strict=True):
            assert a.tier == b.tier == tier and a.example.id == b.example.id
            for key in ('times', 'values', 'targets', 'valid'):
                assert np.array_equal(getattr(a.example, key), getattr(b.example, key)), (a.example.id, key)
            assert np.array_equal(a.mask, b.mask), (a.example.id, 'mask')
            for key in ('group', 'duration', 'truth', 'ignored', 'environment'):
                assert getattr(a.example, key) == getattr(b.example, key), (a.example.id, key)
            checked.append(a.example.id)
    return checked


def main():
    from analysis.neural_recognition_inputs import attach_features
    from analysis.recognition_temporal_model import RecognitionConfig
    from analysis.neural_mobile_distillation import attach_rows
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert Path('/mnt/freenas').is_mount()
    source = read(SOURCE)
    records = []
    for tier, key in (('exact', 'exactRows'), ('draft', 'draftRows'), ('coverage', 'coverageRows')):
        for r in source[key]:
            records.append({**r, 'labelTier': tier, 'scoringPolicy': current.POLICIES[tier],
                'eligibleRoles': ['fit', 'calibrate', 'evaluate', 'infer'], 'protected': False,
                'featureOrigin': 'opencv-media-pts-av104' if tier == 'coverage' else 'opencv-av104'})
    manifest = {'kind': 'neural-generalization-inputs-v1', 'source': identity(SOURCE), 'records': records}
    write_immutable(OUTPUT/'inputs.json', manifest)
    dino = {r['recordingId']: r for r in read(DINO)['records']}
    mobile = {r['id']: r for r in read(MOBILE)['records']}
    entries = []
    for row in records:
        key = row['id']
        features = {'audiovisual': row['featureCaches']['audiovisual'],
                    'dino': {'fp32': {'path': dino[key]['dinoPath'], 'sha256': dino[key]['dinoSha256']}},
                    'mobile': {'path': mobile[key]['cachePath'], 'sha256': mobile[key]['cacheSha256']},
                    'imageInput': identity(DISTILL/'distillation-images-v1'/key/'receipt.json'),
                    'teacherTargets': identity(DISTILL/'distillation-teacher-v1'/(key+'.json'))}
        entries.append({'recordingId': key, 'sourceGroup': row['sourceGroup'],
                        'contentSha256': row['contentSha256'], 'features': features})
    index = {'kind': 'neural-generalization-features-v1', 'records': entries}
    write_immutable(OUTPUT/'features.json', index)
    manifest_path, index_path = OUTPUT/'inputs.json', OUTPUT/'features.json'
    checks = {}
    for family in ('av', 'dino', 'mobile'):
        legacy = original.load_data(SOURCE, DINO if family == 'dino' else None, with_dino=family == 'dino')
        if family == 'mobile':
            legacy = attach_features(legacy, MOBILE, RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8), SOURCE)
        now = current.load_data(manifest_path, index_path, family=family)
        checks[family] = compare(legacy, now)
        print(json.dumps({'family': family, 'records': len(checks[family]), 'bitExact': True}), flush=True)
        del legacy, now
        gc.collect()
    # A fixed completed fold exercises the student's cache receipt adapter.
    feature_path = DISTILL/'distilled-mobile-v1/fits/3407/inner-0-1/features.json'
    receipts = {r['id']: r for r in read(feature_path)['records']}
    legacy = original.load_data(SOURCE, None, with_dino=False)
    legacy = {tier: attach_rows([r for r in rows if r.example.id in receipts], receipts) for tier, rows in legacy.items()}
    now = current.load_data(manifest_path, index_path, family='distilled', student_features=receipts,
                            recording_ids=[r['id'] for r in records if r['id'] in receipts])
    checks['distilled'] = compare(legacy, now)
    del legacy, now
    gc.collect()
    inferred = current.load_inference_examples(manifest_path, index_path)
    assert all(e.valid.all() and not e.truth and not e.ignored and not e.targets.any() for e in inferred)
    exact_ids = {r['id'] for r in records if r['labelTier'] == 'exact'}
    exact = [e for e in inferred if e.id in exact_ids]
    scored = current.make_examples_for_evaluation(exact, manifest_path)
    assert all(a.valid is b.valid and a.values is b.values and a.times is b.times for a, b in zip(exact, scored, strict=True))
    images, teachers = current.load_images_teachers(manifest_path, index_path)
    assert len(images) == len(teachers) == 18
    for row in records:
        assert images[row['id']]['arrays'] == read(DISTILL/'distillation-images-v1'/row['id']/'receipt.json')['arrays']
    inference_images, inference_teachers = current.load_images_teachers(manifest_path, index_path, for_training=False)
    assert set(inference_images) == set(images) and not inference_teachers
    result = {'kind': 'generalized-loader-legacy-parity-v1', 'passed': True, 'checks': checks,
              'source': identity(SOURCE), 'inputs': identity(manifest_path), 'features': identity(index_path),
              'code': [identity(__file__), identity(REPO/'analysis/neural_generalization_inputs.py')],
              'supervisedValuesTargetsMasksBitExact': True, 'inferenceAllTicksValidAndLabelsEmpty': True,
              'evaluationAttachmentPreservesInferenceArraysAndMask': True,
              'original18ImageTeacherBindingsPassed': True, 'inferenceCannotReadTeacherTargets': True}
    write_immutable(OUTPUT/'report.json', result)
    print(json.dumps({'passed': True, 'report': identity(OUTPUT/'report.json')}), flush=True)


if __name__ == '__main__':
    main()
