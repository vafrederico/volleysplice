#!/usr/bin/env python3
"""Fresh frozen serving-side inference at unchanged production ensemble starts.

Reuses validated native raw specialist features. Human serve corrections, human
rally labels and suppression overrides are never used as anchors or input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MODEL_FILE = 'serving-side-85bc3325fbd4.json'

NODE_PROGRAM = r'''
import fs from 'node:fs';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const base=input.repoUrl;
const core=await import(base+'/prod/src/lib/on-device/model.ts');
const ensemble=await import(base+'/prod/src/lib/on-device/ensemble.ts');
const serving=await import(base+'/prod/src/lib/on-device/serving-side-model.ts');
const asset=name=>JSON.parse(fs.readFileSync(new URL(base+'/prod/public/runtime/'+name),'utf8'));
const times=Float64Array.from(input.times), contextual=Float32Array.from(input.contextual);
const all=core.runOnDeviceModel(core.loadOnDeviceModelBundle(asset('model-1ca43e38eefc.json')),times,contextual,input.duration);
const previous=core.runOnDeviceModel(core.loadOnDeviceModelBundle(asset('model-9c92b8e9333f.json')),times,contextual,input.duration);
const tag=(prefix,rows)=>rows.map((r,i)=>({...r,id:prefix+String(i+1).padStart(3,'0'),included:true}));
const intervals=ensemble.mergeProductionModelIntervals(tag('AV2-',all.rallies),tag('PP-',previous.rallies));
const runtime=serving.parseServingSideRuntime(asset('serving-side-85bc3325fbd4.json'));
const cache={...input.cache,features:{...input.cache.features,values:Float64Array.from(input.raw)}};
if(!serving.isReusableServingSideOutput(cache,intervals))throw Error('Specialist cache model/anchor identity mismatch');
const raw=cache.features.values,rows=intervals.length,columns=runtime.featureNames.length;
const ranked=serving.tiedPercentileRanks(raw,rows,columns);
const allServe={modelId:'model-1ca43e38eefc',probabilities:all.probabilities.serve,detections:all.serves};
const previousServe={modelId:'model-9c92b8e9333f',probabilities:previous.probabilities.serve,detections:previous.serves};
const candidates=intervals.map((row,i)=>serving.composeServingSideVerdict(
  row,ranked.subarray(i*columns,(i+1)*columns),times,allServe,previousServe,runtime));
const output={modelId:runtime.modelId,modelFingerprint:runtime.fingerprint,
 featureVersion:runtime.featureVersion,anchorContract:serving.SERVING_SIDE_ANCHOR_CONTRACT,
 features:{rows,columns},candidates};
process.stdout.write(JSON.stringify({output,intervals,ranked:Array.from(ranked),
  cacheReusable:true,featureNames:runtime.featureNames,
  thresholds:{sideThreshold:runtime.sideThreshold,reviewBand:runtime.reviewBand,gate:runtime.gate},
  serveOutputs:{allLabelsV2:{...allServe,probabilities:Array.from(allServe.probabilities)},
    previousProduction:{...previousServe,probabilities:Array.from(previousServe.probabilities)}}}));
'''


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), REPO/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_input(feedback, common):
    # initialInference is the immutable pre-edit model analysis. No field from
    # corrections or finalExportProvenance is accessed anywhere in this runner.
    cached = feedback['initialInference']['servingSide']
    features = cached['features']
    rows, columns = int(features['rows']), int(features['columns'])
    if columns != 237 or rows != len(cached['candidates']):
        raise ValueError('Native serving-side cache dimensions changed')
    raw = common.numeric_array(features['values'], 'float64', (rows, columns))
    return cached, raw


def run(args):
    from analysis.config import FeatureConfig
    from analysis.features import contextualize
    common = script('prepare-labeling-neural-review.py')
    production = script('prepare-neural-production-comparison.py')
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((args.study_root/'inference-registration.json').read_text())
    feedback_path = Path(prior['feedback']['path'])
    if common.identity(feedback_path) != prior['feedback']:
        raise ValueError('Bound native feedback source changed')
    feedback = json.loads(feedback_path.read_text())
    sequence = common.feature_input(feedback)
    cached, raw = cache_input(feedback, common)
    previous_replay = json.loads((args.study_root/'production-ensemble-v1/production-replay.json').read_text())
    original_eval_path = args.study_root/'production-ensemble-v1/ensemble-inference.json'
    original_evaluation = json.loads(original_eval_path.read_text())
    if original_evaluation['recordingId'] != args.recording_id or 'servingSide' in original_evaluation:
        raise ValueError('Unexpected base production metadata')
    runtime_path = REPO/'prod/public/runtime'/MODEL_FILE
    runtime_identity = common.identity(runtime_path)
    runtime = json.loads(runtime_path.read_text())
    for key, (filename, _) in production.ASSETS.items():
        if common.identity(REPO/'prod/public/runtime'/filename) != prior['browserRuntimes'][key]:
            raise ValueError('Production model runtime changed')
    config = FeatureConfig.from_dict(json.loads(Path(prior['browserRuntimes']['previous']['path']).read_text())['featureConfig'])
    contextual, names = contextualize(sequence, config)
    if list(names) != json.loads(Path(prior['browserRuntimes']['previous']['path']).read_text())['featureNames']:
        raise ValueError('Production feature signature changed')
    prior_ids = [common.identity(p) for p in sorted(args.study_root.rglob('*'))
        if p.is_file() and p.suffix in ('.json', '.npz') and output not in p.parents]
    sources = [common.identity(p) for p in sorted((REPO/'prod/src/lib').rglob('*.ts'))]
    sources += [common.identity(p) for p in (Path(__file__), REPO/'analysis/features.py',
        REPO/'scripts/prepare-labeling-neural-review.py', REPO/'scripts/prepare-neural-production-comparison.py')]
    registration = {'kind': 'volleycut-labeling-serving-side-inference-v1', 'schemaVersion': 1,
        'createdAt': datetime.now(timezone.utc).isoformat(), 'recordingId': args.recording_id,
        'feedback': prior['feedback'], 'sourceContentSha256': prior['rawContentSha256'],
        'servingSideRuntime': runtime_identity, 'coreRuntimes': prior['browserRuntimes'],
        'baseEnsembleMetadata': common.identity(original_eval_path),
        'rawFeatureSource': 'Immutable initialInference.servingSide.features, native float64 raw features.',
        'rawFeatureRows': raw.shape[0], 'rawFeatureColumns': raw.shape[1],
        'cachedModelIdentity': {k: cached[k] for k in ('modelId', 'modelFingerprint', 'featureVersion', 'anchorContract')},
        'anchorSource': 'Fresh unedited, unsuppressed checked-in production ensemble interval starts.',
        'normalizationPopulation': 'All59 original candidates; ignored and suppressed candidates are not removed before ranks.',
        'ignoredIntervals': prior['operationalIgnoredIntervalsForAdviserOnly'],
        'ignoredAndSuppressionUse': 'Display/accounting scopes only, after complete candidate inference.',
        'humanRallyLabelsUsed': False, 'humanServeMarkersUsed': False,
        'manualSuppressionOverridesUsed': False, 'labelsUsedAsInferenceInputs': False,
        'llmLabelingUsed': False, 'sideSwitchInferred': False, 'trainingPerformed': False,
        'frameExtractionPerformed': False, 'freshModelInferencePerformed': True,
        'cachedOutputComparison': {'nearProbabilityAbsoluteTolerance': 1e-12,
            'otherCandidateFields': 'Exact equality including every verdict, reason and evidence field.',
            'reason': 'Prior attempt found2 nearProbability differences at <=1.11e-16 between browser and Node arithmetic.'},
        'sourceCode': sources, 'priorArtifacts': prior_ids,
        'nodeExecutable': str(args.node), 'nodeVersion': subprocess.check_output([str(args.node), '--version'], text=True).strip()}
    common.write_json(output/'registration.json', registration)
    repo_url = production.node_repo_url(REPO) if str(args.node).lower().endswith('.exe') else REPO.as_uri()
    # Cache predictions are used only by the explicit model/anchor cache guard;
    # fresh candidates derive from raw specialist features and fresh serve heads.
    payload = {'repoUrl': repo_url, 'times': sequence.times.tolist(), 'contextual': contextual.ravel().tolist(),
        'duration': sequence.metadata.duration, 'cache': {**cached, 'features': {'rows': raw.shape[0], 'columns': raw.shape[1]}},
        'raw': raw.ravel().tolist()}
    print('Replaying production serves and frozen serving-side classifier/gate', flush=True)
    process = subprocess.run([str(args.node), '--input-type=module', '-e', NODE_PROGRAM],
        input=json.dumps(payload, allow_nan=False), capture_output=True, text=True, check=True)
    result = json.loads(process.stdout)
    if result['intervals'] != previous_replay['unionWithConfidence']:
        raise ValueError('Fresh production ensemble anchor identity changed')
    for name, key in (('allLabelsV2', 'v2_serve'), ('previousProduction', 'previous_serve')):
        if result['serveOutputs'][name]['probabilities'] != previous_replay['probabilities'][key]:
            raise ValueError('Fresh production serve head differs from previous verified replay')
    serving = {**result['output'], 'features': cached['features']}
    probability_errors = [abs(a['nearProbability']-b['nearProbability'])
                          for a, b in zip(serving['candidates'], cached['candidates'], strict=True)]
    other_fields_equal = all({k: v for k, v in a.items() if k != 'nearProbability'}
                            == {k: v for k, v in b.items() if k != 'nearProbability'}
                            for a, b in zip(serving['candidates'], cached['candidates'], strict=True))
    if not other_fields_equal or max(probability_errors, default=0.) > 1e-12:
        raise ValueError('Fresh serving-side decisions/evidence differ, or probability difference exceeds1e-12')
    common.write_json(output/'serving-side-output.json', serving)
    common.write_json(output/'serve-head-evidence.json', {'recordingId': args.recording_id,
        'times': sequence.times.tolist(), 'serveOutputs': result['serveOutputs'], 'thresholds': result['thresholds']})
    with (output/'serving-side-features.npz').open('xb') as stream:
        np.savez_compressed(stream, raw=raw, ranked=np.asarray(result['ranked'], np.float64).reshape(raw.shape),
            anchors=np.asarray([c['anchor'] for c in serving['candidates']], np.float64), names=result['featureNames'])
    evaluation = {**original_evaluation, 'servingSide': serving}
    if {k: v for k, v in evaluation.items() if k != 'servingSide'} != original_evaluation:
        raise ValueError('Serving attachment altered production ensemble metadata')
    common.write_json(output/'ensemble-inference.json', evaluation)
    ignored = prior['operationalIgnoredIntervalsForAdviserOnly']
    kept = {(x['start'], x['end']) for x in previous_replay['variants']['aggressive']['core']}
    candidates = serving['candidates']
    valid = [c for c in candidates if not any(x['start'] <= c['anchor'] < x['end'] for x in ignored)]
    retained = [c for c in valid if (c['interval']['start'], c['interval']['end']) in kept]
    count = lambda rows: {key: sum(c['verdict'] == key for c in rows) for key in ('near', 'far', 'review', 'not-serve')}
    receipt = {'recordingId': args.recording_id, 'freshServingSideInferencePerformed': True,
        'servingSideModelId': runtime['modelId'], 'servingSideRuntime': runtime_identity,
        'nativeCacheReuseValidated': result['cacheReusable'], 'rawFeatureShape': list(raw.shape),
        'anchorContract': serving['anchorContract'], 'allCandidateAnchorsExactlyMatchProduction': True,
        'freshServeHeadsExactlyMatchPriorReplay': True,
        'freshCandidateDecisionsAndEvidenceExactlyMatchNativeInitialOutput': other_fields_equal,
        'freshNearProbabilityMaximumAbsoluteError': max(probability_errors, default=0.),
        'freshNearProbabilityTolerance': 1e-12,
        'freshNearProbabilityNonzeroErrorCount': sum(x != 0 for x in probability_errors),
        'allOtherEnsembleMetadataUnchanged': True, 'humanMarkersUsedAsAnchors': False,
        'labelsUsedAsInferenceInputs': False, 'manualSuppressionOverridesUsed': False,
        'verdictCounts': {'fullFile': count(candidates), 'outsideIgnoredByAnchor': count(valid),
                          'currentAggressiveOutsideIgnoredByAnchor': count(retained)},
        'reviewReasonCountsOutsideIgnored': dict(Counter(reason for c in valid for reason in c['reviewReasons'])),
        'notServeSemantics': 'Not a score-tracking serve; retain verdict and evidence, never display as near/far serve.',
        'reviewSemantics': 'Side-score ambiguity and/or both-rally-model recovery without serve-head threshold.',
        'servingSide': common.identity(output/'serving-side-output.json'),
        'evaluation': common.identity(output/'ensemble-inference.json'),
        'registration': common.identity(output/'registration.json'),
        'priorArtifactsUnchanged': True, 'priorArtifactsVerified': len(prior_ids)}
    for original in prior_ids + sources + [runtime_identity]:
        if common.identity(original['path']) != original:
            raise ValueError('Previous artifact or runtime/source changed during serving inference')
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
