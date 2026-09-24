#!/usr/bin/env python3
"""Bind exact-eight production references and replay current product suppression.

No training, feature extraction, model selection, or protected-test access. The
checked-in browser TypeScript performs inference, policy gating, and real editor
export materialization. Canonical core-based exports are stored separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
DATA = Path(private_value('private-reference-0057'))
BASELINE_SHA = '4c7dc5d0b23cc1e9d749fb93e4b21f1ec20b4a28fe46a90976037643e161baf0'
MANIFEST_SHA = '817c01a8809d93d6d9d23bc273a3eb4a9e3bf756cb54abe84aa337bd5f25196a'
EXACT_SHA = 'd582e7ae2f75136bf0419f97466e36d4929004a8474c90146b20530c8f311c16'
ASSETS = {
    'previous': ('model-9c92b8e9333f.json', 'd8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d'),
    'v2': ('model-1ca43e38eefc.json', 'd2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f'),
    'suppression': ('suppression-39eddf581639.json', 'ef0ad4eb93fa61ce1d403f083d91f7578cf9ff0f31fac797fde9ab8b73f42794'),
}
PREDICTION_KEYS = {
    'shippedPrevious': 'shipped-previous', 'shippedV2': 'shipped-all-labels',
    'shippedUnion': 'shipped-union', 'refitPrevious': 'refit-previous',
    'refitV2': 'refit-all-labels', 'refitUnion': 'refit-union',
}

# Only native inference input (features/runtime) reaches the neural-free model
# functions. Ignored ranges are attached later, exclusively to the editor seed.
NODE_REPLAY = r'''
import fs from 'node:fs';
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const base = input.repoUrl;
const model = await import(base + '/prod/src/lib/on-device/model.ts');
const suppression = await import(base + '/prod/src/lib/on-device/suppression-model.ts');
const policy = await import(base + '/prod/src/lib/on-device/suppression-policy.ts');
const ensemble = await import(base + '/prod/src/lib/on-device/ensemble.ts');
const editor = await import(base + '/prod/src/lib/cut-draft.ts');
const readAsset = name => JSON.parse(fs.readFileSync(new URL(base + '/prod/public/runtime/' + name), 'utf8'));
const times = Float64Array.from(input.times);
const contextual = Float32Array.from(input.contextual);
const previous = model.runOnDeviceModel(model.loadOnDeviceModelBundle(readAsset('model-9c92b8e9333f.json')), times, contextual, input.duration);
const v2 = model.runOnDeviceModel(model.loadOnDeviceModelBundle(readAsset('model-1ca43e38eefc.json')), times, contextual, input.duration);
const tags = (prefix, rows) => rows.map((r,i)=>({...r,id:prefix+String(i+1).padStart(3,'0'),included:true}));
const components = {allLabelsV2:tags('AV2-',v2.rallies),previousProduction:tags('PP-',previous.rallies)};
const union = ensemble.mergeProductionModelIntervals(components.allLabelsV2,components.previousProduction);
const sup = suppression.runSuppressionModel(suppression.loadSuppressionModelBundle(readAsset('suppression-39eddf581639.json')),times,contextual,input.duration);
const suggestions = policy.buildSuppressionSuggestions(components,tags('S',sup.intervals),input.duration);
const seed={analysisId:input.id,recordingId:input.id,duration:input.duration,rallies:union,ignoredIntervals:input.ignoredIntervals,scoreTrackingEnabled:false};
const fresh=editor.createCutDraft(seed);
if(fresh.selectedSuppressionPolicy!=='aggressive'||editor.DEFAULT_SUPPRESSION_SCOPE!=='whole-rally')throw Error('fresh product default changed');
const variants={};
const clean = rows => rows.map(r=>({start:r.start,end:r.end}));
const overlap=(a,b)=>a.start<b.end&&b.start<a.end;
for(const name of ['none','conservative','balanced','aggressive']){
  const active=policy.suggestionsForPolicy(suggestions.suggestions,name);
  const removed=union.filter(r=>active.some(s=>overlap(r,s)));
  const kept=union.filter(r=>!removed.includes(r));
  const exactExportsByPadding={};
  const barriersByPadding={};
  for(const pad of [0,1,2,3]){
    const draft={...editor.applyPaddingToCachedCuts(fresh,pad,pad,input.duration),selectedSuppressionPolicy:name};
    exactExportsByPadding[String(pad)]=clean(editor.materializeFinalCutIntervals(draft,suggestions).intervals);
    barriersByPadding[String(pad)]=draft.cuts.filter(c=>removed.some(r=>r.id===c.id)).map(c=>({start:c.keepStart,end:c.keepEnd}));
  }
  variants[name]={core:clean(kept),removedCore:clean(removed),suppressionMasks:clean(active),exactExportsByPadding,barriersByPadding};
}
const probabilities={};
for(const [name,result] of [['previous',previous],['v2',v2]])for(const [head,values]of Object.entries(result.probabilities))probabilities[name+'_'+head]=Array.from(values);
probabilities.suppression=Array.from(sup.probabilities);
process.stdout.write(JSON.stringify({previous:clean(previous.rallies),v2:clean(v2.rallies),union:clean(union),components,unionWithConfidence:union,suppressionDecoded:clean(sup.intervals),suggestions,variants,probabilities,defaultProof:{selectedPolicy:fresh.selectedSuppressionPolicy,scope:editor.DEFAULT_SUPPRESSION_SCOPE,initialDecision:'suppressed',joinGapSeconds:fresh.joinGapSeconds,beforePaddingSeconds:fresh.beforePaddingSeconds,afterPaddingSeconds:fresh.afterPaddingSeconds}}));
'''


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def identity(path: Path, expected: str | None = None) -> dict:
    result = {'path': str(path), 'sha256': sha(path), 'sizeBytes': path.stat().st_size}
    if expected is not None and result['sha256'] != expected:
        raise ValueError(f'identity mismatch: {path}')
    return result


def verify_exact_scope(exact: dict, manifest: dict) -> list[dict]:
    rows = exact['recordings']
    if len(rows) != 8 or len({r['id'] for r in rows}) != 8:
        raise ValueError('expected eight unique exact recordings')
    if rows != manifest['exactRows']:
        raise ValueError('gold/ignored/input rows differ from repaired study')
    if len({r['sourceGroup'] for r in rows}) != 4 or sum(len(r['rallies']) for r in rows) != 322:
        raise ValueError('exact group/rally scope changed')
    for row in rows:
        if row['sourceGroup'] == private_value('source-group-008') or row['environment'] == 'beach':
            raise ValueError('protected or beach row in production comparison')
        if row['split'] not in ('train', 'validation') or row['consent'].get('train') is not True:
            raise ValueError('unapproved development row')
    return rows


def endpoints(rows: list[dict]) -> list[dict]:
    result = [{'start': float(r['start']), 'end': float(r['end'])} for r in rows]
    if any(not math.isfinite(r[k]) for r in result for k in ('start', 'end')):
        raise ValueError('nonfinite prediction')
    if any(r['end'] <= r['start'] for r in result):
        raise ValueError('empty or reversed prediction')
    return result


def verify_endpoints(actual: list[dict], expected: list[dict], tolerance: float = 1e-9) -> float:
    a, b = endpoints(actual), endpoints(expected)
    if len(a) != len(b):
        raise ValueError('runtime replay interval count differs from cached baseline')
    delta = max((abs(x[k] - y[k]) for x, y in zip(a, b, strict=True) for k in ('start', 'end')), default=0)
    if delta > tolerance:
        raise ValueError(f'runtime replay endpoint mismatch: {delta}')
    return delta


def canonical_exports(core: list[dict], ignored: list[dict], duration: float, pad: float) -> list[dict]:
    from analysis.crop_evaluation import pad_and_merge_intervals, subtract_intervals
    from analysis.schema import Interval
    intervals = lambda values: tuple(Interval(float(r['start']), float(r['end'])) for r in values)
    result = subtract_intervals(pad_and_merge_intervals(intervals(core), duration, pad, 3.0), intervals(ignored))
    return [r.to_dict() for r in result]


def difference_seconds(a: list[dict], b: list[dict]) -> float:
    from analysis.crop_evaluation import subtract_intervals
    from analysis.schema import Interval
    left = [Interval(r['start'], r['end']) for r in a]
    right = [Interval(r['start'], r['end']) for r in b]
    return sum(r.end-r.start for r in subtract_intervals(left, right))


def node_repo_url(repo: Path) -> str:
    value = str(repo.resolve())
    if value.startswith('/mnt/') and len(value) > 7 and value[6] == '/':
        value = value[5].upper() + ':' + value[6:]
    return 'file:///' + value.replace('\\', '/').lstrip('/')


def prepare(args: argparse.Namespace) -> dict:
    import numpy as np
    from analysis.config import FeatureConfig
    from analysis.features import FeatureSequence, VideoMetadata, contextualize
    from analysis.model import load_model

    destination = args.output_root / 'production-input.json'
    if destination.exists():
        raise FileExistsError(f'immutable output already exists: {destination}')
    baseline_path = DATA / '2026-09-19-expanded/production-baseline/baseline.json'
    manifest_path = DATA / '2026-09-19-short-boost-transfer/manifest-pts-v1.json'
    baseline_id, manifest_id = identity(baseline_path, BASELINE_SHA), identity(manifest_path, MANIFEST_SHA)
    baseline, manifest = (json.loads(p.read_text()) for p in (baseline_path, manifest_path))
    exact_path = Path(baseline['manifestPath'])
    exact_id = identity(exact_path, EXACT_SHA)
    exact = json.loads(exact_path.read_text())
    if baseline['manifestSha256'] != EXACT_SHA or manifest['exactManifest']['sha256'] != EXACT_SHA:
        raise ValueError('upstream exact-manifest binding mismatch')
    rows = verify_exact_scope(exact, manifest)
    if any(set(predictions) != {r['id'] for r in rows} for predictions in baseline['predictions'].values()):
        raise ValueError('baseline prediction scope mismatch')
    runtimes, runtime_ids = {}, {}
    for name, (filename, expected) in ASSETS.items():
        path = REPO / 'prod/public/runtime' / filename
        runtime_ids[name] = identity(path, expected)
        runtimes[name] = json.loads(path.read_text())
    android_ids = {}
    for name in ('previous', 'v2'):
        path = REPO / 'android/app/src/main/assets' / ASSETS[name][0]
        android_ids[name] = identity(path, ASSETS[name][1])
    android_supp_path = REPO / 'android/app/src/main/assets/suppression-overlap-exclusion-retrained.json'
    android_ids['suppression'] = identity(android_supp_path, '02274d0f17b89cd54ea24da7d1665a6475e48dc0f554d06092e9ef892332d4f1')
    android_supp = json.loads(android_supp_path.read_text())
    for key in ('mean', 'scale', 'weights', 'bias'):
        if android_supp['head'][key] != runtimes['suppression']['head'][key]:
            raise ValueError('Android/browser suppression weights differ')
    if android_supp['decoder'] != runtimes['suppression']['head']['decoder']:
        raise ValueError('Android/browser suppression decoder differs')
    refits = []
    for fold in baseline['folds']:
        held = fold['heldOutSourceGroup']
        for bundle, heads in fold['bundles'].items():
            for role, info in heads.items():
                path = Path(info['path'])
                model = load_model(path)
                training = model.training_summary
                if model.artifact_sha256 != info['artifactSha256'] or held in training['trainingSourceGroups']:
                    raise ValueError('source-held artifact identity or isolation changed')
                if training['heldLabelsUsedForFittingOrSelection'] is not False:
                    raise ValueError('refit held-label exposure changed')
                refits.append({'heldGroup': held, 'bundle': bundle, 'role': role,
                               'model': identity(path/'model.json'), 'weights': identity(path/'weights.npz'),
                               'artifactSha256': model.artifact_sha256})
    source_paths = [
        'scripts/prepare-neural-production-comparison.py', 'analysis/features.py',
        'analysis/crop_evaluation.py', 'analysis/neural_production_baseline.py',
        'prod/src/lib/on-device/model.ts', 'prod/src/lib/on-device/ensemble.ts',
        'prod/src/lib/on-device/suppression-model.ts', 'prod/src/lib/on-device/suppression-policy.ts',
        'prod/src/lib/cut-draft.ts', 'prod/src/lib/score-tracking.ts',
        'android/app/src/main/java/com/volleycut/nativeanalysis/EditorModels.kt',
        'android/app/src/main/java/com/volleycut/nativeanalysis/ProductionEnsemble.java',
        'android/app/src/main/java/com/volleycut/nativeanalysis/SuppressionPolicyEngine.java',
    ]
    sources = [identity(REPO / name) for name in source_paths]
    config = FeatureConfig.from_dict(runtimes['previous']['featureConfig'])
    output_rows = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    probability_dir = args.output_root / 'production-probabilities'
    probability_dir.mkdir(exist_ok=False)
    node_version = subprocess.check_output([str(args.node), '--version'], text=True).strip()
    for row in rows:
        print('REPLAY ' + row['id'], flush=True)
        cache = row['featureCaches']['audiovisual']
        cache_id = identity(Path(cache['path']), cache['sha256'])
        with np.load(cache['path'], allow_pickle=False) as data:
            sequence = FeatureSequence(data['times'].astype(np.float64), data['values'].astype(np.float32),
                tuple(str(v) for v in data['names']), VideoMetadata(**json.loads(str(data['metadata_json'].item()))))
        if round(sequence.metadata.duration, 6) != row['durationSeconds']:
            raise ValueError('duration changed')
        values, names = contextualize(sequence, config)
        if any(list(names) != runtime['featureNames'] for runtime in runtimes.values()):
            raise ValueError('runtime/cache feature signature mismatch')
        payload = {'repoUrl': node_repo_url(REPO), 'id': row['id'], 'duration': sequence.metadata.duration,
                   'times': sequence.times.tolist(), 'contextual': values.ravel().tolist(),
                   'ignoredIntervals': row['ignoredIntervals']}
        result = subprocess.run([str(args.node), '--input-type=module', '-e', NODE_REPLAY],
                                input=json.dumps(payload, allow_nan=False), text=True, capture_output=True, check=True)
        replay = json.loads(result.stdout)
        parity = {}
        for key, cached in (('previous', 'shipped-previous'), ('v2', 'shipped-all-labels'), ('union', 'shipped-union')):
            parity[key] = verify_endpoints(replay[key], baseline['predictions'][cached][row['id']])
        probabilities = {k: np.asarray(v, dtype=np.float32) for k,v in replay.pop('probabilities').items()}
        probability_path = probability_dir / (row['id'] + '.npz')
        with probability_path.open('xb') as stream:
            np.savez_compressed(stream, times=sequence.times, **probabilities)
        for variant in replay['variants'].values():
            canonical, export_parity = {}, {}
            for pad in (0, 1, 2, 3):
                actual = variant['exactExportsByPadding'][str(pad)]
                standard = canonical_exports(variant['core'], row['ignoredIntervals'], row['durationSeconds'], pad)
                canonical[str(pad)] = standard
                left, right = difference_seconds(actual, standard), difference_seconds(standard, actual)
                export_parity[str(pad)] = {'exactOnlySeconds':left, 'canonicalOnlySeconds':right,
                                           'symmetricDifferenceSeconds':left+right,
                                           'equalWithinOneMicrosecond':left+right <= 1e-6}
            variant['canonicalExportsByPadding'] = canonical
            variant['exportParityByPadding'] = export_parity
        out = {k: row[k] for k in ('id','sourceGroup','durationSeconds','environment','rallies','ignoredIntervals')}
        out['featureMetadataDurationSeconds'] = sequence.metadata.duration
        out.update({key:endpoints(baseline['predictions'][old][row['id']]) for key,old in PREDICTION_KEYS.items()})
        out['currentDefault'] = replay['variants']['aggressive']['core']
        out['cores'] = {key:out[key] for key in PREDICTION_KEYS}
        export_keys = {'shippedUnion':'none','productionDefault':'aggressive',
                       'productionConservative':'conservative','productionBalanced':'balanced',
                       'productionAggressive':'aggressive'}
        for key, policy_name in export_keys.items():
            out['cores'][key] = replay['variants'][policy_name]['core']
        out['productExportsByPadding'] = {key:replay['variants'][policy_name]['exactExportsByPadding']
                                         for key,policy_name in export_keys.items()}
        out['suppressionPolicies'] = replay['variants']
        out['productReplay'] = {k:v for k,v in replay.items() if k != 'variants'}
        out['featureCache'] = cache_id
        out['runtimeRawReplayMaximumEndpointErrorSeconds'] = parity
        out['probabilities'] = identity(probability_path)
        output_rows.append(out)
    for source in sources:
        identity(Path(source['path']), source['sha256'])
    report = {
        'schemaVersion':1, 'kind':'volleycut-neural-production-comparison-input-v1',
        'createdAt':datetime.now(timezone.utc).isoformat(), 'protectedTestOpened':False,
        'productionChanged':False, 'trainingPerformed':False, 'featureExtractionPerformed':False,
        'primaryPaddingSeconds':2.0, 'joinGapSeconds':3.0, 'paddingCases':[0,1,2,3],
        'baseline':baseline_id, 'repairedManifest':manifest_id, 'exactManifest':exact_id,
        'browserRuntimes':runtime_ids, 'androidRuntimes':android_ids,
        'refittedArtifactVerification':refits, 'sourceCode':sources,
        'nodeVersion':node_version, 'numpyVersion':np.__version__,
        'nodeReplayProgramSha256':hashlib.sha256(NODE_REPLAY.encode()).hexdigest(),
        'currentDefault':{'policy':'aggressive','scope':'whole-rally','decision':'suppress',
            'basis':'checked-in browser fresh draft and Android EditorDraft defaults; persisted user projects can differ',
            'rawEnsemble':'previous-production + all-labels-v2 overlap union',
            'singleModelConfidenceCeiling':0.49,'singleModelConfidenceScale':0.6},
        'metricInterpretation':{
            'primary':'Canonical pad/clip/merge/strict gap <3 seconds then ignored subtraction on core predictions.',
            'actualApp':'exactExportsByPadding uses checked-in browser editor, including millisecond rounding and suppression barriers; report separately where different.',
            'policies':'All three fixed policies use one-model-only eligibility; whole-rally suppression matches fresh product default scope.',
        },
        'caveats':[
            'Shipped production weights have historical training exposure to all eight exact recordings; results are retrospective product replay, not held-out accuracy.',
            'Neural outer-fold predictions are source-group-held; direct shipped-vs-neural comparisons have asymmetric training exposure.',
            'Source-held refit weights/scalers exclude each outer group, but historical decoder settings and epoch caps retain historical selection exposure.',
            'Refits use historical allowlists intersected with these eight exact videos; later historical training additions and all auxiliary neural labels are excluded.',
            'Suppression weights are shipped, historically exposed weights, with no refitted suppression claim.',
            'Browser TypeScript inference is replayed on existing audited Python feature caches; raw intervals must match prior baseline to 1e-9 seconds. This does not retest browser/native feature extraction from raw video.',
            'Browser and Android core bundles are byte-identical; suppression parameters/decoder are identical despite packaging differences. Android native float execution is not freshly replayed here.',
            'Stored user drafts can select different policies, initial behavior, manual edits, padding, and overrides. Current default describes a fresh unedited analysis only.',
            'The older canonical vault note saying suppression is disabled by default is historical; current checked-in code defaults to aggressive whole-rally suppression.',
        ],
        'recordings':output_rows,
    }
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(destination),'sha256':sha(destination),'recordings':len(output_rows),
                      'refittedHeadsVerified':len(refits),'nodeVersion':node_version}),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DATA/'2026-09-19-production-combinations')
    parser.add_argument('--node',type=Path,default=Path('/mnt/c/Program Files/nodejs/node.exe'))
    prepare(parser.parse_args())
