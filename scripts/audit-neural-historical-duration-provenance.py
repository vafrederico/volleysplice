#!/usr/bin/env python3
"""Independently prove rounded manifest durations preserve historical cache timing."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_recall_sweep_adapters import bind


def audit(root, output):
    io.require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(),
               'Provenance receipt needs a new NAS output')
    evidence = {}
    protocol_ref = io.identity(root / 'protocol.json')
    protocol = io.read(bind(protocol_ref, evidence))
    stored = io.read(bind(protocol['records'], evidence))['records']
    source = io.read(bind(protocol['source'], evidence))
    original = {row['id']: row for row in source['exactRows']}
    io.require(len(stored) == len(original) == 8 and {row['id'] for row in stored} == set(original),
               'Original eight-source inventory differs')
    checks = []
    for row in stored:
        gold = original[row['id']]
        cache = gold['featureCaches']['audiovisual']
        cache_ref = {key: cache[key] for key in ('path', 'sha256')}
        with np.load(bind(cache_ref, evidence), allow_pickle=False) as payload:
            metadata = json.loads(str(payload['metadata_json'].item()))
            times = payload['times'].copy()
        duration = metadata['duration']
        io.require(type(duration) in (int, float) and np.isfinite(duration) and duration > 0,
                   'Invalid cached media duration')
        io.require(row['durationSeconds'] == duration == cache['metadata']['duration']
                   == metadata['frame_count'] / metadata['fps'], 'Exact cache/frame timing differs')
        io.require(round(duration, 6) == gold['durationSeconds'], 'Manifest is not the exact six-decimal rounding')
        io.require(np.array_equal(times, np.asarray(row['timestamps'])), 'Historical stored timestamps differ')
        io.require(row['sourceGroup'] == gold['sourceGroup'], 'Source group differs')
        for key in ('rallies', 'ignoredIntervals'):
            normalized = [{'start': item['start'], 'end': item['end'],
                           **({'tags': item['tags']} if item.get('tags') else {})} for item in gold[key]]
            io.require(row[key] == normalized, 'Original interval values/tags differ')
        mask = np.ones(len(times), dtype=bool)
        for item in gold['ignoredIntervals']:
            mask &= ~((times >= item['start']) & (times < item['end']))
        io.require(np.array_equal(mask, row['valid']), 'Ignored-segment validity differs')
        checks.append({'id': row['id'], 'sourceGroup': row['sourceGroup'], 'audiovisualCache': cache_ref,
            'storedMetricDurationSeconds': row['durationSeconds'], 'cacheMetadataDurationSeconds': duration,
            'manifestDurationSeconds': gold['durationSeconds'], 'frameCount': metadata['frame_count'],
            'fps': metadata['fps'], 'manifestMinusStoredSeconds': gold['durationSeconds'] - duration,
            'storedEqualsBothCacheMetadataDurations': True, 'storedEqualsFrameCountDivFps': True,
            'manifestEqualsExactSixDecimalRound': True, 'timestampCount': len(times),
            'timestampsExactlyCache': True, 'sourceGroupAndIntervalValuesAndTagsExact': True,
            'validMaskExactlyIgnoredSubtraction': True})
    for reference in list(evidence.values()):
        bind(reference)
    result = {'kind': 'independent-historical-duration-provenance-audit-v1', 'passed': True,
        'protocol': protocol_ref, 'records': protocol['records'], 'source': protocol['source'],
        'recordingCount': 8, 'roundedManifestDifferenceCount': sum(r['manifestMinusStoredSeconds'] != 0 for r in checks),
        'checks': checks, 'auditor': io.identity(__file__), 'evidence': list(evidence.values()),
        'metricDurationChanged': False, 'selectionChanged': False, 'trainingPerformed': False,
        'gpuUsed': False, 'rawVideoOpened': False,
        'interpretation': 'Stored historical metric duration remains the exact cached media duration. '
            'The source manifest contains its six-decimal display rounding; no tolerance, '
            'resampling, label revision, decoder change or interval adjustment is applied.'}
    io.write_new(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.study, args.output)
    print(io.identity(args.output), flush=True)
