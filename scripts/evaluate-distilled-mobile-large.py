"""Freeze strict-99 Large-student selections, then infer every catalog recording."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import neural_generalization_inputs as inputs
from analysis import neural_mobile_large_distillation as student
from analysis import neural_recall_sweep as sweep
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_development import decode
from analysis.neural_evaluation import evaluate_predictions
from analysis.neural_generalization_experiment import load_checkpoint
from analysis.neural_generalization_results import COMMON_GROUPS, manifest_example
from analysis.neural_recognition_fit import predict

spec = importlib.util.spec_from_file_location('distilled_large', Path(__file__).with_name('experiment-distilled-mobile-large.py'))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def validated_fit(root, task):
    folder = experiment.fit_folder(root, task)
    plan_ref = identity(root / 'plan.json')
    digest = sweep.canonical({'plan': plan_ref, 'task': task})
    selection = read(folder / 'selection.json')
    assert selection['plan'] == plan_ref and selection['task'] == task and selection['contractSha256'] == digest
    assert selection['config'] == asdict(experiment.CONFIG)
    assert selection['floors'] == sweep.select_floors(selection['candidates'], floors=(99,))
    completed, encoder = read(folder / 'temporal/completed.json'), read(folder / 'student/completed.json')
    assert completed['contractSha256'] == encoder['contractSha256'] == digest
    assert completed['seed'] == encoder['seed'] == task['seed'] and completed['epochs'] == task['epochs']
    assert completed['kind'] == 'mobile_tcn' and completed['lossArm'] == 'short_boost'
    assert all(completed['model'][k] == v for k, v in asdict(experiment.CONFIG).items())
    assert completed['model']['inputDimension'] == 3952
    records = read(inputs.verified(task['manifest']))['records']
    by_id = experiment.validate_membership(task, records)
    tiers = {tier: [key for key in task['trainIds'] if by_id[key]['labelTier'] == tier] for tier in ('exact', 'draft', 'coverage')}
    assert completed['trainIds'] == tiers['exact'] and completed['auxiliaryIds'] == {k: tiers[k] for k in ('draft', 'coverage')}
    assert completed['scalerTrainIds'] == tiers['exact'] and completed['validationIds'] == task['calibrationIds']
    assert encoder['trainIds'] == [key for tier in ('exact', 'draft', 'coverage') for key in tiers[tier]]
    assert set(encoder['trainGroups']) == {by_id[key]['sourceGroup'] for key in task['trainIds']}
    assert not set(encoder['trainGroups']) & set(task['commonEvaluationGroups'])
    assert encoder['weights'] == selection['studentWeights'] and encoder['recipe'] == student.RECIPE
    assert Path(encoder['weights']['path']).resolve() == (folder / 'student/weights.npz').resolve(), 'Student weight owner differs'
    inputs.verified(encoder['weights'])
    for filename, checksum in completed['artifacts'].items():
        inputs.verified(dict(path=str(folder / 'temporal' / filename), sha256=checksum))
    return folder, selection, completed, encoder


def decoded_arrays(row, arrays, epoch, decoder):
    """Validate fresh/resumed scores and decode with every source tick valid."""
    assert set(arrays) == {'times', f'epoch_{epoch}'}, 'Inference score keys differ'
    times, scores = arrays['times'], arrays[f'epoch_{epoch}']
    assert times.ndim == 1 and len(times) and np.isfinite(times).all() \
        and np.all(np.diff(times) > 0) and times[0] >= 0 \
        and times[-1] <= row['durationSeconds'], 'Invalid inference timeline'
    assert scores.shape == (len(times), 4) and np.isfinite(scores).all() \
        and scores.min() >= 0 and scores.max() <= 1, 'Invalid inference probabilities'
    example = SimpleNamespace(id=row['id'], group=row['sourceGroup'], duration=row['durationSeconds'],
        times=times, valid=np.ones(len(times), bool), truth=(), ignored=())
    return [dict(start=value.start, end=value.end) for value in decode(example, scores, decoder)]


def infer(root, plan, task, records, entries):
    folder, selection, completed, metadata = validated_fit(root, task)
    floor = selection['floors'][0]
    assert floor['feasible'] and floor['floorPercent'] == 99
    setting = floor['selected']
    epoch, decoder = setting['epoch'], setting['decoder']
    weight_path = folder / 'temporal' / f'weights-{epoch}.npz'
    weights = dict(path=str(weight_path), sha256=completed['artifacts'][weight_path.name])
    inputs.verified(weights)
    temporal, mean, scale = load_checkpoint(weight_path, experiment.CONFIG, DEVICE)
    encoder = student.load_encoder(metadata, DEVICE, inputs.verified(plan['initialCheckpoint']))
    results = {}
    source_code = identity(__file__)
    for index, row in enumerate(records):
        output = folder / 'inference' / (row['id'] + '.npz')
        receipt_path = output.with_suffix('.json')
        expected = dict(recordingId=row['id'], sourceGroup=row['sourceGroup'], contentSha256=row['contentSha256'],
            durationSeconds=row['durationSeconds'], plan=identity(root / 'plan.json'),
            sourceCode=source_code,
            weights={str(epoch): weights}, studentWeights=metadata['weights'],
            decoders={str(epoch): decoder}, labelsUsed=False, ignoredIntervalsUsed=False,
            audiovisual=entries[row['id']]['audiovisual'], imageInput=entries[row['id']]['imageInput'])
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert all(receipt.get(k) == v for k, v in expected.items()), 'Inference resume identity differs'
            assert Path(receipt['output']['path']).resolve() == output.resolve()
            inputs.verified(receipt['output'])
            with np.load(output, allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            assert decoded_arrays(row, arrays, epoch, decoder) == receipt['decodedRallies'][str(epoch)], \
                'Resumed rallies differ from saved inference'
            results[row['id']] = arrays
            continue
        image = read(inputs.verified(entries[row['id']]['imageInput']))
        assert image['id'] == row['id'] and image['sourceGroup'] == row['sourceGroup']
        assert image['contract']['source']['contentSha256'] == row['contentSha256']
        student.extract_student_record(encoder, metadata, image, folder / 'student-features', DEVICE)
        allowed = ('id', 'sourceGroup', 'durationSeconds', 'environment', 'featureOrigin')
        example = inputs.example_from_row({k: row[k] for k in allowed}, entries[row['id']], inference=True)
        example = experiment.attach(example, folder)
        assert not example.truth and not example.ignored and example.valid.all()
        scores = predict(temporal, example, mean, scale, experiment.CONFIG, DEVICE)
        assert scores.shape == (len(example.times), 4) and np.isfinite(scores).all()
        arrays = dict(times=example.times, **{f'epoch_{epoch}': scores})
        rallies = decoded_arrays(row, arrays, epoch, decoder)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        write_immutable(receipt_path, dict(**expected, output=identity(output), decodedRallies={str(epoch): rallies}))
        results[row['id']] = arrays
        experiment.status(root, 'frozen-inference', variant=task['variant'], draw=task.get('splitSeed', task['seed']),
            completedVideos=index+1, totalVideos=len(records))
    del encoder, temporal
    return results


def select(root, plan, records, entries):
    common = [r for r in records if r['sourceGroup'] in COMMON_GROUPS and r['scoringPolicy'] == 'exact-core']
    assert common and all(r['environment'] != 'beach' for r in common)
    candidates = []
    for index, task in enumerate(plan['tasks']):
        _, selection, _, _ = validated_fit(root, task)
        floor = selection['floors'][0]
        if not floor['feasible']:
            continue
        setting = floor['selected']
        scores = infer(root, plan, task, common, entries)
        examples = [manifest_example(row, scores[row['id']]['times']) for row in common]
        evaluation = evaluate_predictions([e.row(decode(e, scores[e.id][f"epoch_{setting['epoch']}"], setting['decoder'])) for e in examples])
        candidates.append(dict(variant=task['variant'], draw=task.get('splitSeed', task['seed']), seed=task['seed'],
            floorPercent=99, setting=setting, evaluation=evaluation))
        experiment.status(root, 'common-selection', completedFits=index+1, totalFits=len(plan['tasks']))
    assert candidates, 'No feasible strict99 fit; lower recall targets must not be substituted'
    f1 = min(candidates, key=lambda r: (-r['evaluation']['primary']['F1_padP_coreR'], r['variant'], r['draw']))
    recall = min([r for r in candidates if r['variant'] == f1['variant']],
        key=lambda r: (-r['evaluation']['primary']['R_core'], -r['evaluation']['primary']['F1_padP_coreR'], r['draw']))
    result = dict(kind='distilled-mobile-large-common-selection-v1', plan=identity(root / 'plan.json'),
        sourceCode=identity(__file__), commonUnseenIsSelectionData=True,
        commonExactPanelRecordingCount=len(common), commonExactPanelSourceGroupCount=len({r['sourceGroup'] for r in common}),
        candidates=candidates, selected=[dict(f1, mode='f1'), dict(recall, mode='recall')])
    destination = root / 'evaluation.json'
    if destination.exists():
        assert read(destination) == result, 'Frozen selection differs'
    else:
        write_immutable(destination, result)
    return result


def all_video(root, plan, result, records, entries, beach_path):
    beach = read(beach_path)
    assert len(beach['records']) == 2 and beach['labelsUsed'] is False
    beach_rows = beach['records']
    for row in beach_rows:
        assert row['environment'] == 'beach' and row['sourceGroup'] not in {r['sourceGroup'] for r in records}
        entries[row['id']] = row['features']
    catalog = records + beach_rows
    assert len(catalog) == 44 and len({r['id'] for r in catalog}) == 44
    destination = root / 'catalog-manifest.json'
    document = dict(kind='distilled-large-inference-catalog-v1', records=catalog)
    if destination.exists():
        assert read(destination) == document
    else:
        write_immutable(destination, document)
    for chosen in result['selected']:
        task = next(t for t in plan['tasks'] if t['variant'] == chosen['variant'] and t.get('splitSeed', t['seed']) == chosen['draw'])
        infer(root, plan, task, catalog, entries)
    experiment.status(root, 'all-video-inference-complete', recordings=len(catalog), selections=len(result['selected']))


def main():
    import torch
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--beach-inputs', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    DEVICE = args.device
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    plan = experiment.load_plan(args.experiment)
    records = inputs.manifest_rows(inputs.verified(plan['inferenceManifest']))[1]
    entries = inputs.feature_entries(records, inputs.verified(plan['features']))
    result = select(args.experiment, plan, records, entries)
    if args.beach_inputs:
        all_video(args.experiment, plan, result, records, entries, args.beach_inputs)


if __name__ == '__main__':
    main()
