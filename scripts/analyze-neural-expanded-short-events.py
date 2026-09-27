#!/usr/bin/env python3
"""Descriptive short-event audit of frozen exact labels and optional final outputs.

No fitting, ranking, threshold changes, or protected data access. Dataset mode
needs no model results. Result mode requires all frozen study outputs and replays
the canonical evaluator. Each invocation creates a new immutable output leaf.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0070'))
from analysis.neural_evaluation import evaluate_predictions

HELPER = REPO / 'scripts/summarize-neural-development.py'
spec = importlib.util.spec_from_file_location('short_event_helpers', HELPER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
SLICES = ('all', 'duration_le_2s', 'duration_le_3s', 'duration_2_to_3s',
          'duration_gt_3s', 'ace', 'service_fault')
SCOPES = ('coreCoverage', 'primaryExportCoverage')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    return {'path': str(path), 'sha256': helpers.digest(path)}


def belongs(row, name):
    duration = row['end'] - row['start']
    return {'all': True, 'duration_le_2s': duration <= 2,
            'duration_le_3s': duration <= 3, 'duration_2_to_3s': 2 < duration <= 3,
            'duration_gt_3s': duration > 3, 'ace': 'ace' in row.get('tags', []),
            'service_fault': 'service-fault' in row.get('tags', [])}[name]


def histogram(rows, key):
    counts = Counter(str(row[key]) if row[key] < 9 else '>=9' for row in rows)
    return {str(value): counts[str(value)] for value in range(9)} | {'>=9': counts['>=9']}


def dataset_stats(rows):
    valid = [row for row in rows if row['evaluableCoreSeconds'] > 0]
    total = sum(row['evaluableCoreSeconds'] for row in valid)
    slices = {}
    for name in SLICES:
        members = [row for row in valid if belongs(row, name)]
        seconds = sum(row['evaluableCoreSeconds'] for row in members)
        slices[name] = {
            'evaluableRallies': len(members),
            'fractionOfEvaluableRallies': len(members) / len(valid) if valid else None,
            'evaluableCoreSeconds': seconds,
            'fractionOfEvaluableCoreSeconds': seconds / total if total else None,
            'minimumValidFeatureTicks': min((row['validFeatureTicks'] for row in members), default=None),
            'validFeatureTickHistogram': histogram(members, 'validFeatureTicks'),
        }
    return {'originalRallies': len(rows), 'evaluableRallies': len(valid),
            'fullyIgnoredRallies': len(rows) - len(valid), 'evaluableCoreSeconds': total,
            'ignoredTouchedRallies': sum(row['ignoredTouched'] for row in rows),
            'rawFeatureTickHistogram': histogram(rows, 'rawFeatureTicks'),
            'validFeatureTickHistogram': histogram(valid, 'validFeatureTicks'),
            'minimumValidFeatureTicks': min((row['validFeatureTicks'] for row in valid), default=None),
            'slices': slices}


def read_dataset(manifest_path):
    expanded_path = manifest_path.parent / 'manifest.json'
    audit_path = manifest_path.parent / 'dataset-audit.json'
    exact, expanded, audit = (helpers.load(path) for path in (manifest_path, expanded_path, audit_path))
    require(helpers.digest(manifest_path) == audit['exactManifestSha256'] == expanded['exactManifest']['sha256'],
            'Exact manifest hash mismatch')
    require(helpers.digest(expanded_path) == audit['manifestSha256'], 'Expanded manifest hash mismatch')
    require(exact['recordings'] == expanded['exactRows'], 'Exact rows differ between frozen manifests')
    records = exact['recordings']
    require(len({row['id'] for row in records}) == len(records), 'Duplicate recording identity')
    require(not {row['sourceGroup'] for row in records}.intersection(exact['protectedSourceGroups']), 'Protected group')
    require(all(row['environment'] in ('grass', 'indoor') and row['consent']['train'] for row in records),
            'Unexpected environment or consent')
    evaluation_input = []
    for row in records:
        require(row['rallies'] == sorted(row['rallies'], key=lambda x: (x['start'], x['end'])),
                'Unsorted gold would change original rally indices')
        duration = row['featureCaches']['audiovisual']['metadata']['duration']
        require(all(0 <= rally['start'] < rally['end'] <= duration for rally in row['rallies']), 'Gold outside video')
        evaluation_input.append({'id': row['id'], 'sourceGroup': row['sourceGroup'], 'durationSeconds': duration,
                                 'rallies': row['rallies'], 'ignoredIntervals': row.get('ignoredIntervals', []),
                                 'predictions': row['rallies']})
    # Self-coverage supplies canonical ignored subtraction and event identities.
    truth_eval = evaluate_predictions(evaluation_input, primary_padding_seconds=2, join_gap_seconds=3)
    coverage = {(r['recordingId'], r['truthIndex']): r
                for r in truth_eval['guardrails']['coreCoverage']['rallies']}
    events, cache_identities = [], []
    for record in records:
        cache = record['featureCaches']['audiovisual']
        cache_path = Path(cache['path'])
        require(helpers.digest(cache_path) == cache['sha256'], f"Cache hash changed: {record['id']}")
        cache_identities.append(identity(cache_path))
        with np.load(cache_path, allow_pickle=False) as data:
            times = np.asarray(data['times'], dtype=np.float64)
        require(times.ndim == 1 and np.isfinite(times).all() and (np.diff(times) > 0).all(), 'Invalid feature times')
        valid = np.ones(len(times), dtype=bool)
        for ignored in record.get('ignoredIntervals', []):
            valid &= ~((times >= ignored['start']) & (times < ignored['end']))
        for index, rally in enumerate(record['rallies']):
            ticks = (times >= rally['start']) & (times < rally['end'])
            canonical = coverage.get((record['id'], index))
            seconds = canonical['evaluableCoreSeconds'] if canonical else 0.
            events.append({'recordingId': record['id'], 'sourceGroup': record['sourceGroup'], 'truthIndex': index,
                           'start': rally['start'], 'end': rally['end'], 'tags': rally.get('tags', []),
                           'durationSeconds': rally['end'] - rally['start'], 'evaluableCoreSeconds': seconds,
                           'rawFeatureTicks': int(ticks.sum()), 'validFeatureTicks': int((ticks & valid).sum()),
                           'ignoredTouched': any(rally['start'] < x['end'] and x['start'] < rally['end']
                                                 for x in record.get('ignoredIntervals', []))})
    stats = dataset_stats(events)
    require(math.isclose(stats['evaluableCoreSeconds'], truth_eval['primary']['coreHumanSeconds'], abs_tol=1e-8),
            'Rally core seconds differ from pooled canonical union')
    groups = sorted({row['sourceGroup'] for row in records})
    output = {'recordingCount': len(records), 'sourceGroupCount': len(groups), **stats,
              'sourceGroups': {group: dataset_stats([r for r in events if r['sourceGroup'] == group]) for group in groups},
              'recordings': {row['id']: dataset_stats([r for r in events if r['recordingId'] == row['id']]) for row in records},
              'durationOnlyIllustrations': {
                  name: {'R_coreIfAllSliceCoreMissedAndAllOtherCoreRetained':
                         1 - stats['slices'][name]['fractionOfEvaluableCoreSeconds'],
                         'role': 'algebraic coverage illustration, not an observed prediction or simulated decoder'}
                  for name in ('duration_le_2s', 'duration_le_3s')}, 'rallies': events}
    provenance = {'exactManifest': identity(manifest_path), 'expandedManifest': identity(expanded_path),
                  'datasetAudit': identity(audit_path), 'featureCaches': cache_identities}
    return exact, output, provenance


def loss_stats(rows):
    seconds = sum(row['evaluableCoreSeconds'] for row in rows)
    retained = sum(row['retainedCoreSeconds'] for row in rows)
    return {'evaluableRallies': len(rows), 'completeRallyLosses': sum(row['completelyLost'] for row in rows),
            'partialRallyLosses': sum(row['partiallyLost'] for row in rows),
            'fullyCoveredRallies': sum(row['fullyCovered'] for row in rows),
            'completeRallyLossRate': sum(row['completelyLost'] for row in rows) / len(rows) if rows else None,
            'anyRallyLossRate': sum(not row['fullyCovered'] for row in rows) / len(rows) if rows else None,
            'evaluableCoreSeconds': seconds, 'retainedCoreSeconds': retained,
            'coreRecall': retained / seconds if seconds else None,
            'lossIdentities': [row for row in rows if not row['fullyCovered']]}


def read_results(report_path, exact, dataset, provenance):
    registration_path = report_path.parent / 'preregistration.json'
    registration = helpers.load(registration_path)
    contract = registration['contract']
    canonical_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(canonical_hash == registration['sha256'], 'Invalid preregistration hash')
    report = helpers.load(report_path)
    require(report['contractSha256'] == canonical_hash and report['manifestSha256'] == contract['manifestSha256']
            == provenance['expandedManifest']['sha256'], 'Report provenance mismatch')
    require(report['status'] == 'completed-expanded-development-screen' and not report['protectedTestOpened'],
            'Requires completed development-only report')
    require(contract['primaryMetric'] == 'F1_padP_coreR' and contract['targetPaddingSeconds'] == 2
            and contract['joinGapSeconds'] == 3 and contract['paddingSweep'] == [0, 1, 2, 3], 'Metric revision mismatch')
    require(report['records'] == dataset['recordingCount'] and sorted(report['sourceGroups']) == sorted(dataset['sourceGroups'])
            == sorted(contract['groups']), 'Scope mismatch')
    expected = {(c, k, s) for c in contract['cohorts'] for k in contract['kinds'] for s in contract['seeds']}
    results = report['results']
    require(len(results) == len(expected) and {(r['cohort'], r['kind'], r['seed']) for r in results} == expected,
            'Missing or duplicate result')
    gold = {(r['recordingId'], r['truthIndex']): r for r in dataset['rallies'] if r['evaluableCoreSeconds'] > 0}
    per_seed = []
    for result in results:
        key = f"{result['cohort']}/{result['kind']}/{result['seed']}"
        helpers.assert_result_revision(exact, result['predictions'], key)
        evaluation = evaluate_predictions(result['predictions'], primary_padding_seconds=2, join_gap_seconds=3)
        require(evaluation == result['evaluation'], f'{key}: stored evaluation differs from canonical replay')
        scopes = {}
        for scope in SCOPES:
            coverage = evaluation['guardrails'][scope]['rallies']
            require({(r['recordingId'], r['truthIndex']) for r in coverage} == set(gold), 'Coverage identity mismatch')
            for row in coverage:
                original = gold[(row['recordingId'], row['truthIndex'])]
                require(all(row[field] == original[field] for field in ('start', 'end', 'tags', 'evaluableCoreSeconds')),
                        'Coverage target revision mismatch')
            scopes[scope] = {name: loss_stats([r for r in coverage if belongs(r, name)]) for name in SLICES}
        per_seed.append({'cohort': result['cohort'], 'kind': result['kind'], 'seed': result['seed'],
                         'primary': evaluation['primary'], 'eventF1': evaluation['guardrails']['eventF1'], 'slices': scopes})
    means = []
    for cohort in contract['cohorts']:
        for kind in contract['kinds']:
            selected = [r for r in per_seed if r['cohort'] == cohort and r['kind'] == kind]
            scopes = {}
            for scope in SCOPES:
                scopes[scope] = {}
                for name in SLICES:
                    values = [r['slices'][scope][name] for r in selected]
                    scopes[scope][name] = {field: (statistics.fmean(r[field] for r in values)
                                                    if all(r[field] is not None for r in values) else None)
                                          for field in values[0] if field != 'lossIdentities'}
            means.append({'cohort': cohort, 'kind': kind, 'seedCount': len(selected), 'meanSeparateSeedSlices': scopes})
    return {'report': identity(report_path), 'preregistration': identity(registration_path),
            'canonicalReplays': len(results), 'perSeed': per_seed, 'meanSeparateSeedRuns': means}


def markdown(output):
    data = output['dataset']
    lines = ['# Short-event diagnostic', '',
             'Descriptive analysis of frozen exact development labels. No new ranking rule, gate, or training change.', '',
             f"{data['recordingCount']} recordings, {data['sourceGroupCount']} source groups, "
             f"{data['evaluableRallies']} evaluable points, {data['evaluableCoreSeconds']:.3f} evaluable core seconds.", '',
             '| Slice | Points | Point share | Core seconds | Core-time share |',
             '|---|---:|---:|---:|---:|']
    for name in SLICES[1:]:
        row = data['slices'][name]
        lines.append(f"| {name} | {row['evaluableRallies']} | {row['fractionOfEvaluableRallies']:.2%} | "
                     f"{row['evaluableCoreSeconds']:.3f} | {row['fractionOfEvaluableCoreSeconds']:.2%} |")
    lines += ['', 'The <=2s and <=3s slices overlap; ace/fault tags may overlap duration slices. Tags are inherited, '
              'not newly reviewed; absence of a tag does not certify the opposite outcome.', '',
              f"Every evaluable point has at least {data['minimumValidFeatureTicks']} valid feature ticks. "
              f"Tick counts: `{json.dumps(data['validFeatureTickHistogram'])}`. "
              'Ticks are half-open [start,end), with ignored ticks removed; this does not measure feature informativeness.', '']
    for name, row in data['durationOnlyIllustrations'].items():
        lines.append(f"If all {name} core time were missed and every other core second retained, duration recall would "
                     f"be {row['R_coreIfAllSliceCoreMissedAndAllOtherCoreRetained']:.6f}. "
                     'This is an algebraic illustration, not an observed model output.')
    lines += ['', 'Duration recall weights seconds rather than points. These statistics alone do not identify '
              'whether failures arise from optimization, decoding, representation, or annotation; they do not establish '
              'a need for new features.', '']
    if 'results' in output:
        lines += ['All result metrics were replayed with the same gold/ignored revision. The following are mean counts '
                  'across separate seed runs, not independent samples or confidence intervals. Raw prediction coverage '
                  'and exact loss identities are also recorded in JSON.', '',
                  '| Cohort | Model | Slice | Mean complete losses | Mean partial losses | Mean slice core recall |',
                  '|---|---|---|---:|---:|---:|']
        for row in output['results']['meanSeparateSeedRuns']:
            for name in ('duration_le_2s', 'duration_le_3s', 'duration_gt_3s', 'ace', 'service_fault'):
                metric = row['meanSeparateSeedSlices']['primaryExportCoverage'][name]
                recall = f"{metric['coreRecall']:.4f}" if metric['coreRecall'] is not None else 'n/a'
                lines.append(f"| {row['cohort']} | {row['kind']} | {name} | {metric['completeRallyLosses']:.2f} | "
                             f"{metric['partialRallyLosses']:.2f} | {recall} |")
        lines += ['', 'The table uses fixed 2s export padding and the canonical strict <3s short-gap join, followed by '
                  'ignored subtraction without rejoining. A complete loss retains <=1e-9 core seconds; a partial loss '
                  'retains some but not all core time. It is not an event-matching score. Any future event-balanced loss '
                  'or short-positive oversampling experiment would require a separately frozen proposal; none ran here.']
    else:
        lines += ['No model predictions were opened for this dataset-only artifact. Final loss slices can be produced '
                  'later in a separate output directory once the full report is complete.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'exact-manifest.json')
    parser.add_argument('--report', type=Path, help='Optional completed report; omitted for dataset-only mode')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwritten')
    args = parser.parse_args()
    require(not args.output.exists(), f'Output already exists: {args.output}')
    exact, dataset, provenance = read_dataset(args.manifest)
    output = {'schemaVersion': 1, 'kind': 'expanded-short-event-descriptive-diagnostic',
              'createdAt': datetime.now(timezone.utc).isoformat(), 'productionPromotionAllowed': False,
              'rankingOrTrainingChanged': False, 'source': provenance,
              'code': [identity(Path(__file__)), identity(HELPER), identity(REPO / 'analysis/neural_evaluation.py'),
                       identity(REPO / 'analysis/crop_evaluation.py')],
              'sliceDefinitions': {'duration': 'original half-open gold end minus start; seconds',
                                   'coreTime': 'remaining canonical core seconds after ignored subtraction',
                                   'outcomes': 'inherited ace and service-fault tags; no new annotation'},
              'dataset': dataset}
    if args.report:
        output['results'] = read_results(args.report, exact, dataset, provenance)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, content in (('report.json', json.dumps(output, indent=2, allow_nan=False) + '\n'),
                              ('report.md', markdown(output))):
        with (args.output / filename).open('x', encoding='utf-8') as stream:
            stream.write(content)
    print(json.dumps({'output': str(args.output), 'reportSha256': helpers.digest(args.output / 'report.json'),
                      'evaluableRallies': dataset['evaluableRallies'], 'minimumValidFeatureTicks': dataset['minimumValidFeatureTicks'],
                      'slices': {name: dataset['slices'][name] for name in ('duration_le_2s', 'duration_le_3s')},
                      'resultCount': output.get('results', {}).get('canonicalReplays', 0)}, indent=2))


if __name__ == '__main__':
    main()
