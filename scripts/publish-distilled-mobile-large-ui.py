"""Publish two frozen distilled Large selections without changing human labels.

Inputs are explicit private locations, resolved by the caller through its ledger
or ignored environment. The experiment contains evaluation.json, plan.json,
catalog-manifest.json and per-fit inference receipts. Every recording is prepared
and verified before any comparison file is replaced. No inference or fitting is
performed here; decoding is checked against the saved inference receipt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

MODEL = 'distilled-mobile-large-tcn'
MODEL_IDS = ('neural-distilled-mobile-large-tcn-fp32',
             'neural-distilled-mobile-large-tcn-fp32-high-recall')
HEADS = ('live', 'serve', 'end', 'keep')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    return dict(path=str(Path(path)), sha256=sha(path))


def verify_identity(reference, expected_path=None):
    require(isinstance(reference, dict) and isinstance(reference.get('path'), str)
            and re.fullmatch('[a-f0-9]{64}', reference.get('sha256', '')),
            'Missing artifact identity')
    path = Path(reference['path'])
    if expected_path is not None:
        require(path.resolve() == Path(expected_path).resolve(), 'Artifact points to a different file')
    require(path.is_file() and sha(path) == reference['sha256'], 'Artifact hash differs')
    return path


def selected_models(evaluation):
    """Recheck the prior common-unseen F1/within-variant recall selection rule."""
    selected = evaluation['selected']
    require(len(selected) == 2 and {row['mode'] for row in selected} == {'f1', 'recall'},
            'Exactly one F1 and one recall selection are required')
    candidates = [row for row in evaluation['candidates'] if row['floorPercent'] == 99]
    require(candidates, 'No feasible target-99 candidates')
    f1 = min(candidates, key=lambda row: (-row['evaluation']['primary']['F1_padP_coreR'],
                                        row['variant'], row['draw']))
    recall = min((row for row in candidates if row['variant'] == f1['variant']),
                 key=lambda row: (-row['evaluation']['primary']['R_core'],
                                  -row['evaluation']['primary']['F1_padP_coreR'], row['draw']))
    result = []
    for mode, expected, model_id in zip(('f1', 'recall'), (f1, recall), MODEL_IDS):
        chosen = next(row for row in selected if row['mode'] == mode)
        require({key: value for key, value in chosen.items() if key != 'mode'} == expected,
                'Selection differs from the registered common-unseen rule')
        require(re.fullmatch('[a-zA-Z0-9_-]+', chosen['variant'])
                and isinstance(chosen['draw'], int) and isinstance(chosen['seed'], int),
                'Invalid selected fit identity')
        setting, metrics = chosen['setting'], chosen['evaluation']['primary']
        require(setting['innerR_core'] >= .99, 'Selected calibration recall is below 99%')
        result.append(dict(modelId=model_id, model=MODEL, precision='fp32',
            modelLabel=f"Distilled MobileNetV3 Large-TCN · highest {'recall' if mode == 'recall' else 'F1'} · target 99%",
            variant=chosen['variant'], seed=chosen['draw'], trainingSeed=chosen['seed'],
            recallTargetPercent=99, selectionMode=mode, commonUnseenUsedForUiSelection=True,
            selectionPanel='common-unseen/exact-rallies/all', selectionF1=metrics['F1_padP_coreR'],
            selectionRecall=metrics['R_core'], selectionPrecision=metrics['P_pad'],
            epoch=setting['epoch'], decoder=setting['decoder']))
    return result


def decode_scores(row, times, scores, decoder):
    from analysis.neural_development import decode
    example = SimpleNamespace(id=row['id'], group=row['sourceGroup'], duration=row['durationSeconds'],
                              times=times, valid=np.ones(len(times), bool), truth=(), ignored=())
    return [dict(start=value.start, end=value.end) for value in decode(example, scores, decoder)]


def make_reference(experiment, row, model, plan_identity, evaluation_hash, evaluator_identity):
    fit = experiment/'fits'/model['variant']/f"split-{model['seed']}"
    selection = read(fit/'selection.json')
    owner = selection['task']
    require(selection['plan'] == plan_identity and owner['variant'] == model['variant']
            and owner.get('splitSeed', owner['seed']) == model['seed']
            and owner['seed'] == model['trainingSeed'], 'Selected fit ownership differs')
    floors = [floor for floor in selection['floors'] if floor['floorPercent'] == 99]
    require(len(floors) == 1 and floors[0]['feasible']
            and floors[0]['selected']['epoch'] == model['epoch']
            and floors[0]['selected']['decoder'] == model['decoder']
            and floors[0]['selected']['innerR_core'] >= .99, 'Selected fit operating point differs')
    completed = read(fit/'temporal/completed.json')
    student = read(fit/'student/completed.json')
    source = fit/'inference'/(row['id']+'.npz')
    receipt = read(source.with_suffix('.json'))
    require(receipt.get('labelsUsed') is False and receipt.get('recordingId') == row['id'],
            'Inference recording or label-blind provenance differs')
    require(receipt.get('sourceGroup') == row['sourceGroup']
            and receipt.get('sourceCode') == evaluator_identity, 'Inference source group or evaluator differs')
    require(receipt.get('ignoredIntervalsUsed') is False, 'Ignored labels reached inference')
    require(receipt['contentSha256'] == row['contentSha256']
            and receipt['durationSeconds'] == row['durationSeconds'], 'Inference source identity differs')
    require(receipt['plan'] == plan_identity, 'Inference belongs to a different experiment plan')
    verify_identity(receipt['output'], source)
    epoch = str(model['epoch'])
    weights = receipt['weights'][epoch]
    verify_identity(weights, fit/'temporal'/f"weights-{epoch}.npz")
    require(completed['artifacts'][f'weights-{epoch}.npz'] == weights['sha256'], 'Temporal checkpoint owner differs')
    require(receipt['studentWeights'] == student['weights'] == selection['studentWeights'],
            'Student checkpoint owner differs')
    verify_identity(receipt['studentWeights'], fit/'student/weights.npz')
    require(receipt['decoders'][epoch] == model['decoder'], 'Saved inference decoder differs')
    with np.load(source, allow_pickle=False) as data:
        times = data['times'].copy()
        scores = data[f'epoch_{epoch}'].copy()
    require(times.ndim == 1 and len(times) > 0 and np.isfinite(times).all()
            and np.all(np.diff(times) > 0) and times[0] >= 0
            and times[-1] <= row['durationSeconds'], 'Invalid inference timeline')
    require(scores.shape == (len(times), 4) and np.isfinite(scores).all()
            and scores.min() >= 0 and scores.max() <= 1, 'Invalid four-head inference probabilities')
    rallies = decode_scores(row, times, scores, model['decoder'])
    require(rallies == receipt['decodedRallies'][epoch], 'Published rallies differ from saved inference')
    description = (f"DINO-distilled MobileNetV3 Large at 224px/2Hz with FP32 TCN; {model['variant']} "
                   f"draw {model['seed']}, epoch {model['epoch']}. Calibration target 99% retained-play "
                   "recall at 2s padding; not guaranteed recall for this video. "
                   "Common-unseen is UI selection data. Beach recordings were excluded from fitting, "
                   "calibration and selection. Core rally boundaries remain separate; export joins gaps "
                   "strictly under 3s. Serve/start is timing only, not a serving-side prediction.")
    signals = dict(times=times.tolist(), **{head: scores[:, index].tolist() for index, head in enumerate(HEADS)})
    provenance = dict(**model, labelsUsed=False, ignoredIntervalsUsed=False,
                      sourceSha256=receipt['output']['sha256'], weightsSha256=weights['sha256'],
                      studentWeightsSha256=receipt['studentWeights']['sha256'],
                      evaluatorSha256=evaluator_identity['sha256'],
                      selectedEvaluationSha256=evaluation_hash)
    # A per-reference revision isolates new model drafts from unrelated catalog updates.
    provenance['uiDraftRevision'] = hashlib.sha256(encoded(dict(
        rallies=rallies, signals=signals, epoch=model['epoch'], decoder=model['decoder'],
        weightsSha256=weights['sha256'], studentWeightsSha256=receipt['studentWeights']['sha256']))).hexdigest()
    return dict(modelId=model['modelId'], modelLabel=model['modelLabel'], description=description,
                rallies=rallies, exportRallies=rallies, exportPolicy='model-predictions', research=dict(
                    recommendation=description, signals=signals, boundaryFlags=[], reviewRegions=[],
                    queue=dict(budgetFraction=0, reviewSeconds=0, selectedParentCount=0), provenance=provenance))


def prepare(experiment, index_path, catalog_manifest=None, expected_recordings=44):
    experiment, index_path = Path(experiment), Path(index_path)
    catalog_manifest = Path(catalog_manifest) if catalog_manifest else experiment/'catalog-manifest.json'
    evaluation_path = experiment/'evaluation.json'
    evaluation, plan = read(evaluation_path), identity(experiment/'plan.json')
    require(evaluation['plan'] == plan, 'Selection belongs to a different experiment plan')
    verify_identity(evaluation['sourceCode'])
    models = selected_models(evaluation)
    original_index = index_path.read_bytes()
    index = json.loads(original_index)
    require(index['schemaVersion'] == 1 and index['kind'] == 'volleycut-neural-comparison-index',
            'Invalid existing comparison index')
    records = read(catalog_manifest)['records']
    rows = {row['id']: row for row in records}
    require(len(rows) == len(records) == expected_recordings, 'Incomplete or duplicate inference catalog')
    require(len({entry['id'] for entry in index['recordings']}) == len(index['recordings']),
            'Duplicate UI recording')
    require(set(rows) == {entry['id'] for entry in index['recordings']},
            'Inference must cover every UI recording, including beach')
    new_ids = set(MODEL_IDS)
    changes, counts = [], []
    for entry in index['recordings']:
        require(re.fullmatch('[a-zA-Z0-9_-]+', entry['id'])
                and entry['file'] == f"recordings/{entry['id']}.json", 'Invalid comparison recording path')
        row = rows[entry['id']]
        path = index_path.parent/entry['file']
        original = path.read_bytes()
        document = json.loads(original)
        require(len(document['recordings']) == 1, 'Ambiguous recording reference document')
        record = document['recordings'][0]
        require(record['recordingId'] == row['id']
                and abs(record['durationSeconds']-row['durationSeconds']) < .11,
                'Inference and UI recording identities differ')
        refs = [copy.deepcopy(ref) for ref in record['references'] if ref['modelId'] not in new_ids]
        revision = hashlib.sha256(original).hexdigest()
        for ref in refs:
            ref.setdefault('research', {}).setdefault('provenance', {}).setdefault('uiDraftRevision', revision)
        new_refs = [make_reference(experiment, row, model, plan, sha(evaluation_path), evaluation['sourceCode']) for model in models]
        record['references'] = refs+new_refs
        entry['modelIds'] = [ref['modelId'] for ref in record['references']]
        require(len(set(entry['modelIds'])) == len(entry['modelIds']), 'Duplicate UI model reference')
        changes.append((path, original, encoded(document)))
        counts.append(dict(recordingId=row['id'], rallyCounts={ref['modelId']: len(ref['rallies']) for ref in new_refs}))
    index['models'] = [model for model in index['models'] if model['modelId'] not in new_ids]+models
    require(len({model['modelId'] for model in index['models']}) == len(index['models']), 'Duplicate UI model catalog entry')
    return dict(models=models, counts=counts, changes=changes, index=index, originalIndex=original_index,
                evaluationSha256=sha(evaluation_path), catalogSha256=sha(catalog_manifest))


def replace_bytes(path, value):
    temporary = path.with_name('.'+path.name+'.distilled-large.tmp')
    try:
        temporary.write_bytes(value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(experiment, index_path, prepared):
    """Index-last replacement with backups and rollback of only our own writes."""
    index_path, experiment = Path(index_path), Path(experiment)
    require(index_path.read_bytes() == prepared['originalIndex'], 'Comparison index changed during preparation')
    for path, old, _ in prepared['changes']:
        require(path.read_bytes() == old, 'Recording changed during preparation')
    backup = experiment/'ui-publication-before'/hashlib.sha256(prepared['originalIndex']).hexdigest()
    backup.mkdir(parents=True, exist_ok=True)
    if not (backup/'index.json').exists():
        (backup/'index.json').write_bytes(prepared['originalIndex'])
    written = []
    try:
        for path, old, new in prepared['changes']:
            require(path.read_bytes() == old, 'Recording changed during publication')
            backup_path = backup/path.relative_to(index_path.parent)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                backup_path.write_bytes(old)
            replace_bytes(path, new)
            written.append((path, old, new))
        require(index_path.read_bytes() == prepared['originalIndex'], 'Comparison index changed during publication')
        new_index = encoded(prepared['index'])
        replace_bytes(index_path, new_index)
        written.append((index_path, prepared['originalIndex'], new_index))
        receipt = dict(kind='distilled-mobile-large-ui-publication-v1', models=prepared['models'],
                       recordings=prepared['counts'], selectedEvaluationSha256=prepared['evaluationSha256'],
                       catalogSha256=prepared['catalogSha256'], labelsUsedForInference=False,
                       existingModelsPreserved=True, humanLabelsChanged=False)
        replace_bytes(experiment/'ui-publication.json', encoded(receipt))
    except Exception:
        for path, old, new in reversed(written):
            # Never undo somebody else's concurrent write during rollback.
            if path.read_bytes() == new:
                path.write_bytes(old)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--catalog-manifest', type=Path)
    parser.add_argument('--expected-recordings', type=int, default=44)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    prepared = prepare(args.experiment, args.index, args.catalog_manifest, args.expected_recordings)
    if not args.validate_only:
        publish(args.experiment, args.index, prepared)
    print(json.dumps(dict(validatedRecordings=len(prepared['changes']), published=not args.validate_only,
                          addedModels=list(MODEL_IDS))), flush=True)


if __name__ == '__main__':
    main()
