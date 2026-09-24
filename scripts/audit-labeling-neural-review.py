#!/usr/bin/env python3
"""Focused independent audit of source-time research labeling references."""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0082'))


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), REPO/'scripts'/name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def read(path): return json.loads(Path(path).read_text())
def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'sizeBytes': path.stat().st_size}
def require(value, message):
    if not value: raise ValueError(message)


def main():
    import torch
    from analysis.features import ABSOLUTE_FEATURE_NAMES
    from analysis.transfer_temporal_model import model_for
    a = load('audit-neural-boundary-advisor.py'); scalar = a.scalar
    producer = load('prepare-labeling-neural-review.py')
    registration = read(ROOT/'inference-registration.json'); receipt = read(ROOT/'inference-receipt.json')
    feedback = read(registration['feedback']['path']); prepared = read(ROOT/'feedback-import/import-receipt.json')
    boundary = read(ROOT/'boundary-adviser.json'); replay = read(ROOT/'production-replay.json')
    manifest = read(ROOT/'research-references.json'); compact = read(ROOT/'compact-events.json')
    for entry in [registration['feedback'], registration['weights'], registration['fit'], registration['trainingManifest'],
                  *registration['browserRuntimes'].values(), *registration['sources'], receipt['manifest']]:
        require(ref(entry['path']) == entry, 'Bound input/source hash mismatch: '+entry['path'])
    require(registration['rawContentSha256'] == prepared['sourceVideo']['sha256'], 'Raw source lineage mismatch')
    require(registration['sampledFingerprint'] == prepared['sourceSampledFingerprint'], 'Sampled source lineage mismatch')
    with np.load(ROOT/'native-features.npz', allow_pickle=False) as cache:
        times, values, names = cache['times'], cache['values'], cache['names']
    with np.load(ROOT/'compact-probabilities.npz', allow_pickle=False) as cache:
        signal_times, scores = cache['times'], cache['scores']
    features = feedback['features']; n = features['rows']
    raw_times = np.frombuffer(base64.b64decode(features['timestamps']['data']), dtype='<f8')
    raw_values = np.frombuffer(base64.b64decode(features['values']['data']), dtype='<f4').reshape(n, 104)
    require(times.dtype == np.float64 and np.array_equal(times, raw_times), 'Native source timestamps changed')
    require(np.array_equal(values, raw_values) and list(names) == features['names'], 'Native source features changed')
    require(np.array_equal(times, signal_times) and scores.shape == (n, 4), 'Prediction timeline mismatch')
    require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'Invalid probabilities')
    # The whitelist must work with no corrections, labels, ranges or prior model output.
    whitelisted = producer.feature_input({'source': feedback['source'], 'features': feedback['features']})
    require(np.array_equal(whitelisted.times, times) and np.array_equal(whitelisted.values, values), 'Input whitelist differs')
    record = boundary['record']; parents = record['productionEvents']
    require(record['rallies'] == [], 'Human labels passed to boundary planning')
    expected_ignored = [{'start': r['start'], 'end': r['end']} for r in feedback['corrections']['ignoredIntervals']]
    require(record['ignoredIntervals'] == expected_ignored == registration['operationalIgnoredIntervalsForAdviserOnly'], 'Ignored scope mismatch')
    require(not registration['humanCorrectionsUsedForInference'] and not registration['trainingPerformed']
            and not registration['humanOracleUsed'], 'Unexpected fit/human operation')
    training = read(registration['fit']['path']); inventory = read(registration['trainingManifest']['path'])
    train_ids = training['trainIds'] + sum(training['auxiliaryIds'].values(), [])
    train_groups = training['trainGroups'] + sum(training['auxiliaryGroups'].values(), [])
    require(record['id'] not in train_ids and record['sourceGroup'] not in train_groups, 'Training ID/group leakage')
    require(all(registration['rawContentSha256'] != r.get(k) for section in ('exactRows', 'draftRows', 'coverageRows')
                for r in inventory[section] for k in ('contentSha256', 'sourceContentSha256')), 'Training content leakage')
    require((registration['seed'], registration['outerFoldIndex'], registration['epoch']) == (3407, 0, 60), 'Unpinned checkpoint')

    # Independently rank and scale native features, then replay the saved network
    # on CPU. This does not use the producer's feature transform or predict loop.
    ranked = np.empty_like(values)
    for column in range(104):
        order = np.argsort(values[:, column], kind='stable')
        cursor = 0
        while cursor < n:
            end = cursor + 1
            while end < n and values[order[end], column] == values[order[cursor], column]:
                end += 1
            ranked[order[cursor:end], column] = (cursor+end-1)/(2*(n-1))
            cursor = end
    for i, name in enumerate(names):
        if name in ABSOLUTE_FEATURE_NAMES: ranked[:, i] = values[:, i]
    model = model_for('tcn'); torch.set_num_threads(1)
    with np.load(registration['weights']['path'], allow_pickle=False) as saved:
        mean, scale = saved['mean'], saved['scale']
        require(mean.shape == scale.shape == (104,) and np.isfinite(mean).all() and np.all(scale > 0), 'Invalid frozen scaler')
        model.load_state_dict({key[7:]: torch.from_numpy(saved[key].copy()) for key in saved.files if key.startswith('model::')}, strict=True)
    require(sum(p.numel() for p in model.parameters()) == registration['modelParameters'] == 29700, 'Parameter count mismatch')
    scaled = np.clip((ranked-mean)/scale, -10., 10.)
    independent = np.zeros_like(scores); model.eval()
    with torch.inference_mode():
        for core in range(0, n, 128):
            finish = min(n, core+128); left = max(0, core-62); right = min(n, finish+62)
            chunk = torch.sigmoid(model(torch.from_numpy(scaled[left:right][None])))[0].numpy()
            independent[core:finish] = chunk[core-left:finish-left]
    error = float(np.max(np.abs(independent-scores)))
    require(error < 2e-5, 'Frozen network/scaler CPU replay differs')

    default = replay['variants']['aggressive']; union = replay['union']; masks = default['suppressionMasks']
    kept = [row for row in union if not any(row['start'] < m['end'] and m['start'] < row['end'] for m in masks)]
    require(kept == default['core'] and scalar.pairs(parents) == scalar.pairs(kept), 'Automatic default suppression differs')
    require(replay['defaultProof']['selectedPolicy'] == 'aggressive' and replay['defaultProof']['scope'] == 'whole-rally', 'Wrong production default')
    require(scalar.pairs(union) == scalar.pairs(feedback['initialInference']['ranges']), 'Original unsuppressed parity differs')
    plan_audit = a.audit_plan(record, parents, compact['events'], times, scores, 'head_refined', boundary['plan'])
    queue_audit = scalar.audit_jobs_queues(record, parents, boundary['plan']['proposals'], [], 'split_only', 'evidence',
                                          boundary['jobs'], [boundary['queue']], budgets=(.1,))
    require(boundary['queue']['selectedProposalIds'] == boundary['queue']['selectedSplitIds'], 'Queue alias mismatch')
    require(boundary['queue']['reviewSeconds'] <= .1*(record['durationSeconds']-151.528966), 'Queue exceeds valid-video budget')

    references = manifest['recordings'][0]['references']; require(len(references) == 2, 'Display layer count differs')
    for layer in references:
        require(layer['exportPolicy'] == 'fixed-production' and scalar.pairs(layer['exportRallies']) == scalar.pairs(parents), 'Display export changed')
    require(scalar.pairs(references[0]['rallies']) == scalar.pairs(parents), 'Production display differs')
    require(scalar.pairs(references[1]['rallies']) == scalar.pairs(boundary['plan']['events']), 'Preview display differs')
    research = references[0]['research']; signals = research['signals']
    require(np.array_equal(np.asarray(signals['times']), times), 'Displayed source timestamps differ')
    for i, head in enumerate(('live', 'serve', 'end', 'keep')):
        require(np.array_equal(np.asarray(signals[head], dtype=np.float32), scores[:, i]), 'Displayed head differs: '+head)
    flags = [{k: row[k] for k in ('id', 'kind', 'time', 'parentId', 'priority')} for row in boundary['plan']['proposals']]
    require(flags == research['boundaryFlags'], 'Displayed flags differ')
    selected = set(boundary['queue']['selectedParentIds'])
    require({r['parentId'] for r in research['reviewRegions'] if r['recommended']} == selected, 'Displayed recommendations differ')
    pads = []
    for padding in (0, 1, 2, 3):
        export = scalar.export(parents, record, padding)
        for layer in references:
            require(scalar.export(layer['exportRallies'], record, padding) == export, 'Padding override changed')
        exact = scalar.pairs(default['exactExportsByPadding'][str(padding)])
        pads.append({'paddingSeconds': padding, 'canonicalFixedExportSeconds': scalar.seconds(export),
            'appMaterializedExportSeconds': scalar.seconds(exact),
            'canonicalOnlySeconds': scalar.seconds(scalar.difference(export, exact)),
            'appOnlySeconds': scalar.seconds(scalar.difference(exact, export))})
    # Keep earlier registered numerical work unchanged.
    frozen = read(ROOT.parent/'2026-09-19-compact-typed-boundaries/registration.json')['contract']
    for entry in [*frozen['sources'].values(), *frozen['sourceCopies'].values()]:
        require(ref(entry['path']) == entry, 'Earlier frozen source changed')
    artifacts = ['inference-registration.json', 'inference-receipt.json', 'native-features.npz', 'compact-probabilities.npz',
                 'production-replay.json', 'compact-events.json', 'boundary-adviser.json', 'research-references.json']
    output = {'passed': True, 'createdAt': datetime.now(timezone.utc).isoformat(), 'auditor': ref(Path(__file__)),
        'artifacts': {name: ref(ROOT/name) for name in artifacts}, 'sourceFramesConverted': False,
        'rawFeaturesAndSourceTimestampsExact': True, 'importedHumanLabelsUsed': False,
        'frozenCheckpointAndScalerVerified': True, 'independentCpuReplayMaxAbsoluteError': error,
        'trainingIdentityAndContentAbsent': True, 'automaticProductionSeparateFromManualFeedback': True,
        'planAudit': plan_audit, 'queueAudit': queue_audit, 'displaySignalsExact': True,
        'fixedProductionExportsAllPaddingCases': pads, 'frozenPreviousSourcesUnchanged': len(frozen['sources']),
        'limits': ['Native feature extraction itself was not rerun.',
            'Full source content hash and sampled fingerprint inherit the independently completed import receipt.',
            'Canonical source-second exports and app millisecond materialization differ by rounding, listed above.',
            'This audit verifies inference plumbing and proposal provenance, not model accuracy or human review performance.']}
    destination = ROOT/'inference-audit-v1.json'
    with destination.open('x') as stream: json.dump(output, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': ref(destination), 'cpuMaxError': error, 'plan': plan_audit, 'queue': queue_audit}))


if __name__ == '__main__': main()
