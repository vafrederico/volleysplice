#!/usr/bin/env python3
"""Qualify new blind production inference against the frozen exact-eight replay."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis.source_exposure_inventory import identity, read, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--replay', type=Path, required=True)
parser.add_argument('--historical', type=Path, default=Path(private_value('private-reference-0081')))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
fresh, historical = read(args.replay), read(args.historical)
protocol_ref = fresh['protocol']; assert sha256(Path(protocol_ref['path'])) == protocol_ref['sha256']
protocol = read(Path(protocol_ref['path']))
assert protocol['labelsUsedForInference'] is False and protocol['trainingPerformed'] is False
assert protocol['programSha256'] == historical['nodeReplayProgramSha256']
for reference in protocol['sourceCode'].values():
    assert sha256(Path(reference['path'])) == reference['sha256']
for name, reference in protocol['assets'].items():
    assert reference['sha256'] == historical['browserRuntimes'][name]['sha256']
manifest_ref = protocol['manifest']; assert sha256(Path(manifest_ref['path'])) == manifest_ref['sha256']
manifest = read(Path(manifest_ref['path']))
assert manifest['labelsStrippedBeforeReplay'] is True
for row in manifest['records']:
    assert row['rallies'] == [] and row['ignoredIntervals'] == [] and 'gameWindow' not in row
    assert row['eligibleRoles'] == ['infer'] and row['consent']['train'] is False
old = {r['id']: r for r in historical['recordings']}
assert len(old) == 8 and {r['id'] for r in fresh['records']} == set(old)
checks = []
for record in fresh['records']:
    previous = old[record['id']]
    assert record['labelsUsed'] is False
    assert record['featureCache']['sha256'] == previous['featureCache']['sha256']
    comparisons = {'productionDefault': (record['predictions']['productionDefault'], previous['cores']['productionDefault']),
                   'productionUnion': (record['predictions']['productionUnion'], previous['cores']['shippedUnion']),
                   'previousHead': (record['productReplay']['previous'], previous['shippedPrevious']),
                   'v2Head': (record['productReplay']['v2'], previous['shippedV2'])}
    deltas = {}
    for name, (left, right) in comparisons.items():
        assert len(left) == len(right), (record['id'], name, 'count')
        delta = max((abs(a[k]-b[k]) for a, b in zip(left, right) for k in ('start', 'end')), default=0.)
        assert delta <= 1e-9, (record['id'], name, delta)
        deltas[name] = {'intervals': len(left), 'maximumEndpointErrorSeconds': delta}
    for reference in (record['probabilities'], previous['probabilities']):
        assert sha256(Path(reference['path'])) == reference['sha256']
    with np.load(record['probabilities']['path'], allow_pickle=False) as actual, np.load(previous['probabilities']['path'], allow_pickle=False) as expected:
        assert set(actual.files) == set(expected.files)
        for key in actual.files:
            assert actual[key].shape == expected[key].shape and np.array_equal(actual[key], expected[key]), (record['id'], key)
        probability_heads = sorted(set(actual.files)-{'times'})
        ticks = len(actual['times'])
    checks.append({'id': record['id'], 'sourceGroup': record['sourceGroup'], 'predictions': deltas,
                   'probabilityHeadsBitExact': probability_heads, 'ticks': ticks,
                   'ignoredLabelsProvidedToRuntime': False})
result = {'kind': 'blind-generalization-production-parity-v1', 'passed': True, 'trainingPerformed': False,
          'gpuUsed': False, 'protectedSourcesOpened': False, 'replay': identity(args.replay),
          'historical': identity(args.historical), 'manifest': manifest_ref, 'protocol': protocol_ref,
          'auditor': identity(Path(__file__).resolve()), 'records': checks,
          'maximumEndpointErrorSeconds': max(v['maximumEndpointErrorSeconds'] for r in checks for v in r['predictions'].values()),
          'allProbabilityArraysBitExact': True,
          'scope': 'Original eight recordings, identical AV caches and pinned production TS program/assets; labels/ignored/gameWindow stripped before new inference.',
          'limitations': ['No raw-media feature parity or new-video performance claim.',
                          'Canonical evaluation must apply padding/ignored masks after inference. Editor barrier-based exports are not substituted for canonical metrics.']}
with args.output.open('x', encoding='utf-8') as handle:
    json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
print(json.dumps({'audit': identity(args.output), 'passed': True, 'records': len(checks),
                  'maximumEndpointErrorSeconds': result['maximumEndpointErrorSeconds'], 'allProbabilityArraysBitExact': True}, indent=2))
