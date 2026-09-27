"""Matched DINO-distilled MobileNetV3-Large TCN fits at strict 99% calibration.

Input locations are explicit private arguments. Source membership is inherited
from the frozen Large substitution plan, without changing historical artifacts.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import mobile_visual_features as mobile
from analysis import neural_generalization_inputs as inputs
from analysis import neural_recall_sweep as sweep
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_experiment import validate_membership, ordered_rows
from analysis.neural_recognition_fit import fit_model
from analysis.recognition_temporal_model import RecognitionConfig

CONFIG = RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8, token_dimension=960)
REPO = Path(__file__).resolve().parents[1]
CODE = ('scripts/experiment-distilled-mobile-large.py', 'analysis/neural_mobile_large_distillation.py',
        'analysis/neural_mobile_distillation.py', 'analysis/neural_recognition_fit.py',
        'analysis/recognition_temporal_model.py', 'analysis/neural_generalization_inputs.py',
        'analysis/neural_recall_sweep.py', 'analysis/neural_generalization_experiment.py',
        'analysis/neural_generalization_results.py', 'analysis/neural_evaluation.py',
        'analysis/neural_development.py', 'analysis/neural_expanded_development.py',
        'analysis/neural_short_boost_weighting.py', 'analysis/neural_context_fit.py',
        'analysis/mobile_visual_features.py', 'analysis/neural_recall_operating_point.py',
        'analysis/expanded_temporal_model.py', 'analysis/crop_evaluation.py', 'analysis/metrics.py',
        'analysis/decoder.py', 'analysis/features.py', 'analysis/neural_event_weighting.py',
        'analysis/schema.py', 'analysis/config.py')


def status(root, phase, **details):
    value = dict(phase=phase, **details)
    temporary = root / '.progress.tmp'
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(root / 'progress.json')
    print(json.dumps(value), flush=True)


def register(a):
    parent = read(a.parent_plan)
    assert parent['kind'] == 'mobile-large-substitution-v1' and len(parent['tasks']) == 24
    checkpoint = identity(a.checkpoint)
    assert checkpoint['sha256'].startswith('8738ca79'), 'Unexpected ImageNet Large V1 checkpoint'
    tasks = []
    for original in parent['tasks']:
        task = {**original, 'floorsPercent': [99], 'config': asdict(CONFIG)}
        records = read(inputs.verified(task['manifest']))['records']
        validate_membership(task, records)
        tasks.append(task)
    plan = dict(kind='distilled-mobile-large-v1', parentPlan=identity(a.parent_plan), tasks=tasks,
        initialCheckpoint=checkpoint, sourceCode={name: identity(REPO / name) for name in CODE},
        config=asdict(CONFIG), precision='fp32', savedEmbeddingPrecision='float16',
        distillation=dict(epochs=8, batchSize=16, learningRate=1e-4, weightDecay=1e-4,
            teacher='cached-frozen-dino', projector=[960, 384], teacherTokens=10,
            batchNormStatisticsFrozen=True, teachingFramesPerRecording=128,
            teacherInputsRestrictedToTrainingMembership=True),
        floorsPercent=[99], temporalEpochs=list(sweep.EPOCHS), lossArm='short_boost',
        targetPaddingSeconds=2, paddingSeconds=[0, 1, 2, 3], joinGapSeconds=3,
        inferenceManifest=parent['inferenceManifest'], features=parent['features'],
        selection='Strict99 calibration only; maximum common-unseen exact-label F1; maximum recall draw within winning variant, F1 tie-break',
        commonUnseenIsUiSelectionData=True, beachTrainingCalibrationSelection=False,
        trainingLabels='Same registered exact/draft/export-proxy memberships as frozen Large')
    a.output.mkdir(parents=True, exist_ok=True)
    write_immutable(a.output / 'plan.json', plan)
    status(a.output, 'registered', totalFits=len(tasks))


def load_plan(root):
    plan = read(root / 'plan.json')
    assert plan['kind'] == 'distilled-mobile-large-v1' and plan['floorsPercent'] == [99]
    assert plan['config'] == asdict(CONFIG)
    for ref in plan['sourceCode'].values():
        inputs.verified(ref)
    inputs.verified(plan['initialCheckpoint'])
    return plan


def fit_folder(root, task):
    return root / 'fits' / task['variant'] / f"split-{task.get('splitSeed', task['seed'])}"


def attach(example, folder):
    receipt = read(folder / 'student-features' / (example.id + '.json'))
    student = read(folder / 'student' / 'completed.json')
    assert receipt['id'] == example.id and receipt['sourceGroup'] == example.group
    assert receipt['encoder'] == student['weights'] and receipt['contractSha256'] == student['contractSha256']
    assert Path(receipt['output']['path']).resolve() == (folder / 'student-features' / (example.id + '.npz')).resolve()
    with np.load(inputs.verified(receipt['output']), allow_pickle=False) as z:
        times, tokens, quality, pts = (z[k].copy() for k in ('timestamps', 'tokens', 'quality', 'selected_presentation_times'))
    assert tokens.shape == (len(times), 4, 960)
    cache = mobile.MobileVisualCache(Path(receipt['output']['path']), times, tokens.astype(np.float32), quality, pts, {})
    aligned = mobile.align_mobile_features(cache, example.times)
    assert np.all(aligned['available'] == 1) and example.values.shape[1] == 104
    extra = np.concatenate((aligned['tokens'].reshape(len(example.times), -1), aligned['quality'],
        aligned['feature_age_seconds'][:, None], aligned['available'][:, None]), axis=1)
    values = np.concatenate((example.values, extra), axis=1).astype(np.float32)
    assert values.shape == (len(example.times), CONFIG.input_dimension) and np.isfinite(values).all()
    return replace(example, values=values)


def fit(a):
    import torch
    from analysis import neural_mobile_large_distillation as student
    from analysis.neural_generalization_results import selection_examples
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    plan = load_plan(a.output)
    plan_ref = identity(a.output / 'plan.json')
    start = time.monotonic()
    for number, task in enumerate(plan['tasks']):
        folder = fit_folder(a.output, task)
        digest = sweep.canonical({'plan': plan_ref, 'task': task})
        draw = task.get('splitSeed', task['seed'])
        selection_path = folder / 'selection.json'
        if selection_path.exists():
            saved = read(selection_path)
            assert saved['plan'] == plan_ref and saved['task'] == task and saved['contractSha256'] == digest
            inputs.verified(saved['studentWeights'])
            assert saved['floors'] == sweep.select_floors(saved['candidates'], floors=(99,))
            continue
        manifest, features = inputs.verified(task['manifest']), inputs.verified(task['features'])
        records = read(manifest)['records']
        by_id = validate_membership(task, records)
        data = ordered_rows(inputs.load_data(manifest, features, recording_ids=task['trainIds'], family='av'), task['trainIds'])
        training_rows = [r for tier in ('exact', 'draft', 'coverage') for r in data[tier]]
        groups = {r.example.group for r in training_rows}
        excluded = {r['sourceGroup'] for r in records} - groups
        status(a.output, 'distilling', completedFits=number, totalFits=len(plan['tasks']), variant=task['variant'], draw=draw)
        images, teachers = inputs.load_images_teachers(manifest, features, recording_ids=task['trainIds'], for_training=True)
        metadata = student.fit_student(training_rows, excluded, images, teachers, task['seed'], folder / 'student',
            digest, a.device, inputs.verified(plan['initialCheckpoint']))
        del teachers
        identifiers = list(dict.fromkeys(task['trainIds'] + task['calibrationIds']))
        images, _ = inputs.load_images_teachers(manifest, features, recording_ids=identifiers, for_training=False)
        encoder = student.load_encoder(metadata, a.device, inputs.verified(plan['initialCheckpoint']))
        for i, key in enumerate(identifiers):
            student.extract_student_record(encoder, metadata, images[key], folder / 'student-features', a.device)
            status(a.output, 'student-features', completedFits=number, totalFits=len(plan['tasks']),
                variant=task['variant'], draw=draw, completedVideos=i+1, totalVideos=len(identifiers))
        del encoder
        for tier in data:
            data[tier] = [replace(r, example=attach(r.example, folder)) for r in data[tier]]
        validation = [attach(e, folder) for e in inputs.load_inference_examples(manifest, features, family='av', recording_ids=task['calibrationIds'])]
        status(a.output, 'temporal-fit', completedFits=number, totalFits=len(plan['tasks']), variant=task['variant'], draw=draw)
        scores = fit_model(data['exact'], {k: data[k] for k in ('draft', 'coverage')}, validation, CONFIG,
            task['seed'], sweep.EPOCHS, folder / 'temporal', a.device, digest, 'short_boost')
        del data, training_rows, validation
        status(a.output, 'calibrating', completedFits=number, totalFits=len(plan['tasks']), variant=task['variant'], draw=draw)
        examples = selection_examples(task, manifest, features)
        policy = 'export-rally-proxy-selection' if task.get('selectionLabelPolicy') == 'exact-and-export-rally-proxy' else 'exact-rallies'
        candidates = sweep.build_candidate_table(examples, scores, selection_policy=policy)
        write_immutable(selection_path, dict(task=task, config=asdict(CONFIG), plan=plan_ref, contractSha256=digest,
            studentWeights=metadata['weights'], candidates=candidates, floors=sweep.select_floors(candidates, floors=(99,))))
        status(a.output, 'fit-complete', completedFits=number+1, totalFits=len(plan['tasks']), variant=task['variant'], draw=draw,
            feasible99=read(selection_path)['floors'][0]['feasible'], elapsedSeconds=time.monotonic()-start)
    status(a.output, 'fits-complete', completedFits=len(plan['tasks']), totalFits=len(plan['tasks']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit'))
    parser.add_argument('--parent-plan', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    globals()[args.phase](args)


if __name__ == '__main__':
    main()
