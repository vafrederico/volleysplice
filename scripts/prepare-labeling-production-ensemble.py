#!/usr/bin/env python3
"""Fresh production-only inference for a previously validated labeling source.

Produces metadata already supported by the labeling catalog. It never edits a
catalog, environment, human labels, research references, or previous artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), REPO/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_documents(recording_id, replay, core_metadata_path):
    """Use gated removed whole parents, never raw suppression suggestions."""
    bounds = lambda rows: {(float(r['start']), float(r['end'])) for r in rows}
    union, kept, removed = (bounds(replay['union']),
        bounds(replay['variants']['aggressive']['core']),
        bounds(replay['variants']['aggressive']['removedCore']))
    if len(union) != len(replay['union']) or kept & removed or union != kept | removed:
        raise ValueError('Aggressive suppression is not a whole-parent partition of ensemble')
    if bounds(replay['unionWithConfidence']) != union:
        raise ValueError('Ensemble identity intervals differ from union')
    core = {'recordingId': recording_id,
        'decodedRanges': {'all-labels-v2': replay['v2'], 'previous-production': replay['previous']},
        'labelsUsedAsInferenceInputs': False, 'llmLabelingUsed': False}
    evaluation = {'recordingId': recording_id, 'labelsUsedAsInferenceInputs': False,
        'llmLabelingUsed': False, 'predictedEnsembleRanges': replay['unionWithConfidence'],
        'coreInput': {'metadataPath': str(core_metadata_path)},
        'suppression': {'modelId': 'current-production-aggressive-whole-rally',
            'policy': 'aggressive', 'scope': 'whole-rally',
            'decodedIntervals': replay['variants']['aggressive']['removedCore'],
            'intervalSemantics': 'Actual automatically removed whole ensemble cores after policy gating.'}}
    return core, evaluation


def compare_replays(current, previous, production):
    endpoints = {key: production.verify_endpoints(current[key], previous[key])
                 for key in ('previous', 'v2', 'union')}
    for key in ('core', 'removedCore'):
        endpoints['aggressive_' + key] = production.verify_endpoints(
            current['variants']['aggressive'][key], previous['variants']['aggressive'][key])
    if current['probabilities'].keys() != previous['probabilities'].keys():
        raise ValueError('Probability head names changed')
    probability_errors = {}
    for key, values in current['probabilities'].items():
        left, right = np.asarray(values, dtype=np.float64), np.asarray(previous['probabilities'][key], dtype=np.float64)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Probability array shape/finite-value mismatch')
        probability_errors[key] = float(np.max(np.abs(left-right))) if len(left) else 0.
    return {'endpointMaximumErrorSeconds': endpoints,
        'probabilityMaximumAbsoluteErrors': probability_errors,
        'allEndpointsExactlyEqual': all(x == 0 for x in endpoints.values()),
        'allProbabilitiesExactlyEqual': all(x == 0 for x in probability_errors.values())}


def run(args):
    from analysis.config import FeatureConfig
    from analysis.features import contextualize
    common = script('prepare-labeling-neural-review.py')
    production = script('prepare-neural-production-comparison.py')
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads((args.study_root/'inference-registration.json').read_text())
    feedback_path = Path(registration['feedback']['path'])
    if common.identity(feedback_path) != registration['feedback']:
        raise ValueError('Original feature/source feedback changed')
    # Only source metadata and native feature arrays enter model input. The
    # validator does not inspect feedback corrections or any human rally labels.
    feedback = json.loads(feedback_path.read_text())
    sequence = common.feature_input(feedback)
    with np.load(args.study_root/'native-features.npz', allow_pickle=False) as old:
        if not (np.array_equal(sequence.times, old['times'])
                and np.array_equal(sequence.values, old['values'])
                and list(sequence.names) == old['names'].tolist()):
            raise ValueError('Feedback features differ from the validated native cache')
    ignored = registration['operationalIgnoredIntervalsForAdviserOnly']
    runtime_ids = {}
    for key, (filename, _) in production.ASSETS.items():
        path = REPO/'prod/public/runtime'/filename
        runtime_ids[key] = common.identity(path)
        if runtime_ids[key] != registration['browserRuntimes'][key]:
            raise ValueError('Production runtime differs from prior validated replay')
    config = FeatureConfig.from_dict(json.loads(Path(runtime_ids['previous']['path']).read_text())['featureConfig'])
    contextual, names = contextualize(sequence, config)
    if any(list(names) != json.loads(Path(r['path']).read_text())['featureNames'] for r in runtime_ids.values()):
        raise ValueError('Native features do not match production runtime signatures')
    prior_ids = [common.identity(p) for p in sorted(args.study_root.rglob('*'))
                 if p.is_file() and p.suffix in ('.json', '.npz') and output not in p.parents]
    sources = [common.identity(p) for p in sorted((REPO/'prod/src/lib').rglob('*.ts'))]
    sources += [common.identity(p) for p in (Path(__file__), REPO/'analysis/features.py',
        REPO/'analysis/config.py', REPO/'scripts/prepare-labeling-neural-review.py',
        REPO/'scripts/prepare-neural-production-comparison.py')]
    node_version = subprocess.check_output([str(args.node), '--version'], text=True).strip()
    provenance = {'schemaVersion': 1, 'createdAt': datetime.now(timezone.utc).isoformat(),
        'kind': 'volleycut-labeling-production-ensemble-inference', 'recordingId': args.recording_id,
        'feedback': registration['feedback'], 'sourceContentSha256': registration['rawContentSha256'],
        'runtimeAssets': runtime_ids, 'sourceCode': sources, 'priorArtifacts': prior_ids,
        'nodeExecutable': str(args.node), 'nodeVersion': node_version,
        'inputScope': 'All full-source native feature ticks at original float64 timestamps.',
        'featureRows': len(sequence.times), 'featureColumns': 104,
        'featureRuntime': registration['nativeFeatureRuntime'],
        'operationalIgnoredIntervals': ignored,
        'ignoredUse': 'Editor export materialization only; never masked model input.',
        'labelsUsedAsInferenceInputs': False, 'llmLabelingUsed': False,
        'manualSuppressionOverridesUsed': False, 'trainingPerformed': False,
        'servingSideInferred': False, 'sideSwitchInferred': False,
        'exports': {'paddingCases': [0, 1, 2, 3], 'joinGapSeconds': 3}}
    common.write_json(output/'registration.json', provenance)
    # Windows Node must import Windows file URLs even when launched from WSL.
    # Native Linux Node uses the ordinary POSIX URI instead.
    repo_url = production.node_repo_url(REPO) if str(args.node).lower().endswith('.exe') else REPO.as_uri()
    payload = {'repoUrl': repo_url, 'id': args.recording_id, 'duration': sequence.metadata.duration,
        'times': sequence.times.tolist(), 'contextual': contextual.ravel().tolist(), 'ignoredIntervals': ignored}
    print('Fresh current-production TypeScript inference', flush=True)
    result = subprocess.run([str(args.node), '--input-type=module', '-e', production.NODE_REPLAY],
        input=json.dumps(payload, allow_nan=False), capture_output=True, text=True, check=True)
    replay = json.loads(result.stdout)
    previous = json.loads((args.study_root/'production-replay.json').read_text())
    comparison = compare_replays(replay, previous, production)
    if not comparison['allEndpointsExactlyEqual'] or not comparison['allProbabilitiesExactlyEqual']:
        raise ValueError('Fresh replay unexpectedly differs from identical prior native inputs/runtime')
    common.write_json(output/'production-replay.json', replay)
    common.write_json(output/'production-signals.json', {'recordingId': args.recording_id,
        'times': sequence.times.tolist(), 'probabilities': replay['probabilities'],
        'labelsUsedAsInferenceInputs': False})
    with (output/'production-signals.npz').open('xb') as stream:
        np.savez_compressed(stream, times=sequence.times,
            **{k: np.asarray(v, dtype=np.float32) for k, v in replay['probabilities'].items()})
    core, evaluation = metadata_documents(args.recording_id, replay, output/'core-inference.json')
    evaluation['provenance'] = {'registrationPath': str(output/'registration.json'),
        'signalsPath': str(output/'production-signals.json'), 'freshInference': True}
    common.write_json(output/'core-inference.json', core)
    common.write_json(output/'ensemble-inference.json', evaluation)
    def valid_count(rows):
        return sum(bool(production.canonical_exports([row], ignored, sequence.metadata.duration, 0))
                   for row in rows)
    counts = {}
    for name, rows in (('ensemble', replay['union']), ('allLabelsV2', replay['v2']),
            ('previousProduction', replay['previous']), ('aggressiveKept', replay['variants']['aggressive']['core']),
            ('aggressiveRemoved', replay['variants']['aggressive']['removedCore'])):
        counts[name] = {'fullFile': len(rows), 'touchValidTimeline': valid_count(rows)}
    for original in prior_ids + sources:
        if common.identity(original['path']) != original:
            raise ValueError('Existing artifact or inference source changed during fresh replay')
    receipt = {'recordingId': args.recording_id, 'freshInferencePerformed': True,
        'labelsUsedAsInferenceInputs': False, 'llmLabelingUsed': False,
        'manualSuppressionOverridesUsed': False, 'priorArtifactsUnchanged': True,
        'priorArtifactsVerified': len(prior_ids), 'sourceFilesVerified': len(sources),
        'counts': counts, 'comparisonToPreviousReplay': comparison,
        'evaluation': common.identity(output/'ensemble-inference.json'),
        'core': common.identity(output/'core-inference.json'),
        'registration': common.identity(output/'registration.json'),
        'signals': common.identity(output/'production-signals.json'),
        'suppressionUiSemantics': 'Subtract actual gated removed whole ensemble cores; current aggressive default retained.',
        'productionDefaultProof': replay['defaultProof']}
    common.write_json(output/'receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recording-id', required=True)
    parser.add_argument('--node', type=Path, default=Path('/mnt/c/Program Files/nodejs/node.exe'))
    run(parser.parse_args())
