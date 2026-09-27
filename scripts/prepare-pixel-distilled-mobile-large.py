"""Export selected Large students and qualify FP32 native benchmark graphs.

Private inputs and outputs resolve through the external ledger. Each selection
has a separate directory with the existing mobile-large runtime file names;
the contract and pipeline metadata identify the actual distilled student.
No DINO encoder or training-only projection is exported. No fitting occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx
import onnxruntime as ort
import torch

from analysis import mobile_visual_features as mobile
from analysis import neural_generalization_inputs as inputs
from analysis import neural_mobile_large_distillation as student
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_experiment import load_checkpoint
from analysis.neural_recognition_fit import standardized
from analysis.private_ledger import private_value

spec = importlib.util.spec_from_file_location(
    'distilled_native_evaluator', Path(__file__).with_name('evaluate-distilled-mobile-large.py'))
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

FAMILY = 'mobile-large'
MODEL_ID = 'dino-distilled-mobilenet-v3-large-tcn'


class RegionalEncoder(torch.nn.Module):
    """Only the student's spatial trunk and four content-relative pools."""
    def __init__(self, encoder):
        super().__init__()
        self.features = encoder

    def forward(self, image, pool_weights):
        return torch.einsum('bchw,brhw->brc', self.features(image), pool_weights)


def recording_id(index, allowed_ids):
    platform = 'WINDOWS' if os.name == 'nt' else 'POSIX'
    location = os.environ.get('VOLLEYCUT_PRIVATE_LEDGER_' + platform) or os.environ.get('VOLLEYCUT_PRIVATE_LEDGER')
    if not location:
        raise ValueError('Private ledger is required')
    matches = [key for key, value in read(location)['originalToAlias'].items()
               if value == index and key in allowed_ids]
    if len(matches) != 1:
        raise ValueError('Recording ledger index must resolve uniquely')
    return matches[0]


def selected_task(plan, evaluation, mode):
    selected = [row for row in evaluation['selected'] if row['mode'] == mode]
    if len(selected) != 1:
        raise ValueError('Selection mode is missing or duplicated')
    chosen = selected[0]
    if chosen['floorPercent'] != 99 or chosen['setting']['innerR_core'] < .99:
        raise ValueError('Selected model did not qualify at strict target99')
    matches = [task for task in plan['tasks'] if task['variant'] == chosen['variant']
               and task.get('splitSeed', task['seed']) == chosen['draw']]
    if len(matches) != 1:
        raise ValueError('Selected task must resolve uniquely')
    return chosen, matches[0]


def checked_geometry(image):
    source_plan = read(inputs.verified(image['contract']['plan']))
    sources = [row for row in source_plan['records'] if row['id'] == image['id']]
    if len(sources) != 1 or sources[0]['roi'] != dict(x=0, y=0, width=1, height=1):
        raise ValueError('Native benchmark requires registered full-frame source inputs')
    with np.load(inputs.verified(image['arrays']['timing']), allow_pickle=False) as timing:
        boxes = timing['boxes'].copy()
        times = timing['times'].copy()
    if boxes.shape != (len(times), 4) or not len(times):
        raise ValueError('Invalid cached frame geometry')
    np.testing.assert_array_equal(boxes, np.broadcast_to(boxes[0], boxes.shape))
    return boxes, times


def annotate_graph(path, mode, weights_sha, role):
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, dict(modelIdentity=MODEL_ID, selectionMode=mode,
        weightsSha256=weights_sha, role=role, precision='fp32', trainingProjectorIncluded='false'))
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def error_summary(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    return dict(maxAbsoluteError=float(np.max(np.abs(actual - expected))),
                rmse=float(np.sqrt(np.mean((actual - expected) ** 2))))


def export_one(root, destination, plan, evaluation, mode, row, entry, record_index):
    chosen, task = selected_task(plan, evaluation, mode)
    folder, selection, completed, metadata = evaluator.validated_fit(root, task)
    floor = selection['floors'][0]
    if not floor['feasible'] or floor['selected'] != chosen['setting']:
        raise ValueError('Frozen selection differs from qualified fit')
    if evaluation['plan'] != identity(root / 'plan.json'):
        raise ValueError('Frozen evaluation belongs to another plan')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing graph directory')

    image = read(inputs.verified(entry['imageInput']))
    if image['id'] != row['id'] or image['contract']['source']['contentSha256'] != row['contentSha256']:
        raise ValueError('Source image association differs')
    boxes, times = checked_geometry(image)
    pixels = np.load(inputs.verified(image['arrays']['images224']), mmap_mode='r', allow_pickle=False)
    if pixels.shape != (len(times), 3, 224, 224) or pixels.dtype != np.uint8:
        raise ValueError('Expected registered uint8 CHW224 cached video frames')
    indexes = np.unique(np.linspace(0, len(times) - 1, 16, dtype=int))
    pool = mobile.regional_pool_weights(boxes[:1], 7, 7)
    encoder = RegionalEncoder(student.load_encoder(metadata, 'cpu', inputs.verified(plan['initialCheckpoint']))).eval()
    # The graph owns only the student's encoder state. The projector never
    # enters the wrapper, and classifier/DINO weights are not loaded here.
    if sum(p.numel() for p in encoder.parameters()) != metadata['encoderParameters']:
        raise ValueError('Unexpected parameters in inference encoder')
    destination.mkdir(parents=True)
    encoder_path = destination / (FAMILY + '-encoder-fp32.onnx')
    first = student.normalize_pixels(pixels[indexes[:1]], 'cpu')
    torch.onnx.export(encoder, (first, torch.from_numpy(pool)), str(encoder_path),
        input_names=['image', 'pool_weights'], output_names=['tokens'], opset_version=17, dynamo=False)
    annotate_graph(encoder_path, mode, metadata['weights']['sha256'], 'regional-encoder')
    runtime = session(encoder_path)
    encoder_checks = []
    with torch.inference_mode():
        for index in indexes:
            tensor = student.normalize_pixels(pixels[index:index + 1], 'cpu')
            expected = encoder(tensor, torch.from_numpy(pool)).numpy()
            actual = runtime.run(None, {'image': tensor.numpy(), 'pool_weights': pool})[0]
            if actual.shape != (1, 4, 960):
                raise ValueError('Encoder output differs from Large regional token contract')
            encoder_checks.append(dict(frameIndex=int(index), timeSeconds=float(times[index]),
                                       **error_summary(actual, expected)))
    del encoder, runtime
    pool_path = destination / (FAMILY + '-encoder-pool_weights.f32')
    pool.astype('<f4').tofile(pool_path)

    epoch = chosen['setting']['epoch']
    checkpoint = folder / 'temporal' / f'weights-{epoch}.npz'
    checkpoint_ref = identity(checkpoint)
    if checkpoint_ref['sha256'] != completed['artifacts'][checkpoint.name]:
        raise ValueError('Temporal checkpoint differs from fit receipt')
    config = evaluator.experiment.CONFIG
    temporal, mean, scale = load_checkpoint(checkpoint, config, 'cpu')
    example = inputs.example_from_row(row, entry, inference=True)
    example = evaluator.experiment.attach(example, folder)
    values = standardized(example, mean, scale, config)
    if values.shape[1] != 3952 or not np.isfinite(values).all():
        raise ValueError('Invalid real fused/standardized temporal inputs')
    graph_path = destination / (FAMILY + '-tcn-dynamic-fp32.onnx')
    torch.onnx.export(temporal, torch.from_numpy(values[None, :252]), str(graph_path),
        input_names=['features'], output_names=['logits'],
        dynamic_axes={'features': {1: 'ticks'}, 'logits': {1: 'ticks'}}, opset_version=17, dynamo=False)
    annotate_graph(graph_path, mode, checkpoint_ref['sha256'], 'temporal-head')
    runtime = session(graph_path)
    temporal_checks = []
    with torch.inference_mode():
        for ticks in (1, 62, 128, 190, 252):
            for start in sorted({0, max(0, (len(values) - ticks) // 2), max(0, len(values) - ticks)}):
                tensor = np.ascontiguousarray(values[None, start:start + ticks])
                expected = temporal(torch.from_numpy(tensor)).numpy()
                actual = runtime.run(None, {'features': tensor})[0]
                temporal_checks.append(dict(ticks=ticks, startTick=start, **error_summary(actual, expected)))
    pipeline = dict(mean=mean.tolist(), scale=scale.tolist(), decoder=chosen['setting']['decoder'],
        weightsSha256=checkpoint_ref['sha256'], encoderWeightsSha256=metadata['weights']['sha256'],
        epoch=epoch, family=FAMILY, modelIdentity=MODEL_ID, selectionMode=mode, draw=chosen['draw'],
        recallTargetPercent=99, tokenDimension=3840, config=asdict(config), dynamicParity=temporal_checks,
        trainingProjectorIncluded=False, dinoRequiredForInference=False)
    pipeline_path = destination / (FAMILY + '-pipeline.json')
    write_immutable(pipeline_path, pipeline)
    hashes = {path.name: identity(path)['sha256'] for path in (encoder_path, graph_path, pipeline_path, pool_path)}
    contract = dict(recordingIndex=record_index, roi=dict(x=0, y=0, width=1, height=1),
        contentBox=boxes[0].tolist(), hashes=hashes, modelIdentity=MODEL_ID, selectionMode=mode,
        studentWeightsSha256=metadata['weights']['sha256'], temporalWeightsSha256=checkpoint_ref['sha256'],
        scope='Selected distilled Large FP32 graphs; original full-frame geometry; no training or calibration.')
    write_immutable(destination / 'input-contract.json', contract)
    qualification = dict(kind='distilled-large-native-fp32-qualification-v1', passed=True,
        modelIdentity=MODEL_ID, selectionMode=mode, draw=chosen['draw'], epoch=epoch,
        recordingIndex=record_index, labelsUsed=False, trainingPerformed=False, calibrationPerformed=False,
        trainingProjectorIncluded=False, dinoRequiredForInference=False,
        encoderParameters=metadata['encoderParameters'], encoderSha256=hashes[encoder_path.name],
        temporalSha256=hashes[graph_path.name], studentWeightsSha256=metadata['weights']['sha256'],
        temporalWeightsSha256=checkpoint_ref['sha256'], planSha256=identity(root / 'plan.json')['sha256'],
        evaluationSha256=identity(root / 'evaluation.json')['sha256'], sourceSha256=identity(__file__)['sha256'],
        encoderBytes=encoder_path.stat().st_size, temporalBytes=graph_path.stat().st_size,
        encoderChecks=encoder_checks, temporalChecks=temporal_checks,
        scope='CPU PyTorch to ONNX parity on registered real video frames and fused standardized tensors; device extraction parity is separate.')
    write_immutable(destination / 'qualification.json', qualification)
    print(json.dumps({key: qualification[key] for key in ('passed', 'selectionMode', 'draw', 'epoch',
        'encoderBytes', 'temporalBytes', 'encoderParameters')}), flush=True)
    return qualification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-index', default='private-reference-0222')
    parser.add_argument('--output-index', required=True)
    parser.add_argument('--recording-index', default='recording-044')
    parser.add_argument('--mode', choices=('f1', 'recall', 'both'), default='both')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root, output = Path(private_value(args.experiment_index)), Path(private_value(args.output_index))
    plan = evaluator.experiment.load_plan(root)
    evaluation = read(root / 'evaluation.json')
    _, records = inputs.manifest_rows(inputs.verified(plan['inferenceManifest']))
    key = recording_id(args.recording_index, {row['id'] for row in records})
    matches = [row for row in records if row['id'] == key]
    if len(matches) != 1:
        raise ValueError('Benchmark recording must occur exactly once in inference manifest')
    row = matches[0]
    entry = inputs.feature_entries([row], inputs.verified(plan['features']))[key]
    for mode in ('f1', 'recall') if args.mode == 'both' else (args.mode,):
        export_one(root, output / ('graphs-' + mode), plan, evaluation, mode, row, entry, args.recording_index)


if __name__ == '__main__':
    main()
