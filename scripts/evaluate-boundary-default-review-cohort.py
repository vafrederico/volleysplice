#!/usr/bin/env python3
"""Frozen development replay: apply all boundaries, veto harmful parent deletions.

This is an ideal-human simulation, not training or model/threshold selection.
Queue generation uses only production and frozen automatic event boundaries.
Gold is inspected only inside queued removed core to simulate the veto decision.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_production_combinations as iv
from analysis.neural_boundary_metrics import _coverage
from analysis.neural_rally_identity_metrics import evaluate_rally_identities

PRIOR = Path(private_value('private-reference-0079'))
OUTPUT = Path(private_value('private-reference-0080'))
SEEDS = (3407, 1729, 20260918)


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sizeBytes': path.stat().st_size}


def positive_overlap(a, b):
    return bool(iv.intersection(a, b))


def apply_policy(record, automatic):
    ignored = record.get('ignoredIntervals', [])
    removed = iv.difference(iv.difference(record['productionEvents'], automatic), ignored)
    parents = {p['id']: p for p in record['productionEvents']}
    queued = {pid for pid, parent in parents.items() if positive_overlap([parent], removed)}
    vetoed = {pid for pid in queued
              if positive_overlap(iv.intersection([parents[pid]], removed), record['rallies'])}
    output = [dict(e) for e in automatic if e['parentId'] not in vetoed]
    output.extend(dict(parents[pid]) for pid in sorted(vetoed))
    output.sort(key=lambda e: (e['start'], e['end'], e['id']))
    return output, removed, queued, vetoed


def self_check():
    parents = [{'id': 'p', 'start': 10., 'end': 30.}]
    auto = [{'id': 'a', 'parentId': 'p', 'start': 12., 'end': 28.}]
    record = {'durationSeconds': 40., 'productionEvents': parents, 'rallies': []}
    out, removed, queued, vetoed = apply_policy(record, auto)
    assert out == auto and queued == {'p'} and not vetoed
    assert iv.serial(removed) == [[10., 12.], [28., 30.]]
    record['rallies'] = [{'start': 11., 'end': 20.}]
    assert apply_policy(record, auto)[0] == parents
    record['ignoredIntervals'] = [{'start': 10., 'end': 12.}]
    assert apply_policy(record, auto)[0] == auto
    split = [{'id': 'a', 'parentId': 'p', 'start': 10., 'end': 20.},
             {'id': 'b', 'parentId': 'p', 'start': 20., 'end': 30.}]
    assert not apply_policy(record, split)[2]
    assert len(apply_policy(record, split)[0]) == 2


def score(records):
    identity = evaluate_rally_identities(records)
    raw = [_coverage(r) for r in records]
    raw_totals = {key: sum(row[key] for row in raw) for key in (
        'coreHumanSeconds', 'baselineRawCoreCoveredSeconds', 'resultRawCoreCoveredSeconds',
        'rawCoreSecondsLostFromBaseline', 'rawCoreSecondsAddedToBaseline',
        'rawSelectedSecondsLostFromBaseline', 'rawSelectedSecondsAddedToBaseline',
        'baselineCompleteMisses', 'resultCompleteMisses', 'additionalCompleteMisses')}
    raw_totals['resultRawCoreRecall'] = raw_totals['resultRawCoreCoveredSeconds'] / raw_totals['coreHumanSeconds']
    return {'identityMetrics': identity, 'rawCoreMetrics': raw_totals,
            'durationMetrics': iv.duration_rows(records)}


def workload(record, plan, removed, queued, vetoed):
    parents = record['productionEvents']
    automatic = plan['events']
    ignored = record.get('ignoredIntervals', [])
    playback = iv.difference(iv.dilate(removed, 2., record['durationSeconds']), ignored)
    gold = record['rallies']
    automatic_by_parent = {p['id']: [e for e in automatic if e['parentId'] == p['id']] for p in parents}
    changed = {p['id'] for p in parents
               if [(e['start'], e['end']) for e in automatic_by_parent[p['id']]] != [(p['start'], p['end'])]}
    extra = [p for p in plan['proposals'] if p['kind'] == 'additional_start']
    target_export_removed = iv.difference(iv.export(parents, record, 2), iv.export(automatic, record, 2))
    export_trigger_parents = {p['id'] for p in parents
        if positive_overlap(iv.export([p], record, 2), target_export_removed)}
    return {
        'removedCoreWindows': iv.serial(removed), 'removedCoreWindowCount': len(removed),
        'removedCoreSeconds': iv.duration(removed),
        'removedGoldCoreSeconds': iv.duration(iv.intersection(removed, gold)),
        'removedNonGoldCoreSeconds': iv.duration(iv.difference(removed, gold)),
        'reviewedParentCount': len(queued), 'reviewedParentIds': sorted(queued),
        'vetoedParentCount': len(vetoed), 'vetoedParentIds': sorted(vetoed),
        'queuedGoldRallyCount': sum(positive_overlap([g], removed) for g in gold),
        'playbackGoldRallyCount': sum(positive_overlap([g], playback) for g in gold),
        'playbackWindows': iv.serial(playback), 'playbackWindowCount': len(playback),
        'playbackSeconds': iv.duration(playback),
        'changedParentCount': len(changed),
        'changedParentsWithoutCoreRemovalTrigger': sorted(changed - queued),
        'changedParentsWithoutCoreRemovalTriggerCount': len(changed - queued),
        'changedParentsWithoutExportRemovalTrigger': sorted(changed - export_trigger_parents),
        'changedParentsWithoutExportRemovalTriggerCount': len(changed - export_trigger_parents),
        'addedStartFlagCount': len(extra),
        'addedStartFlagsWithoutCoreRemovalTrigger': [p['id'] for p in extra if p['parentId'] not in queued],
        'addedStartFlagsWithoutCoreRemovalTriggerCount': sum(p['parentId'] not in queued for p in extra),
        'addedStartFlagsWithoutExportRemovalTrigger': [p['id'] for p in extra if p['parentId'] not in export_trigger_parents],
        'addedStartFlagsWithoutExportRemovalTriggerCount': sum(p['parentId'] not in export_trigger_parents for p in extra),
        'targetExportRemovedWindows': iv.serial(target_export_removed),
        'targetExportRemovedSeconds': iv.duration(target_export_removed),
    }


def seed_result(seed, records):
    source = read(PRIOR / 'results' / f'{seed}.json')
    plans = {p['id']: p['policies']['head_refined'] for p in source['plans']}
    arms = {'production': [], 'automatic_applied_to_export': [], 'parent_veto_removed_core': []}
    per_record = []
    for record in records:
        plan = plans[record['id']]
        automatic = plan['events']
        output, removed, queued, vetoed = apply_policy(record, automatic)
        arms['production'].append({**record, 'predictions': record['productionEvents']})
        arms['automatic_applied_to_export'].append({**record, 'predictions': automatic})
        arms['parent_veto_removed_core'].append({**record, 'predictions': output})
        work = workload(record, plan, removed, queued, vetoed)
        per_record.append({'id': record['id'], 'sourceGroup': record['sourceGroup'],
            'resultEvents': output, 'workload': work})
        assert _coverage({**record, 'predictions': output})['rawCoreSecondsLostFromBaseline'] < 1e-8
        assert _coverage({**record, 'predictions': output})['additionalCompleteMisses'] == 0
    scores = {name: score(rows) for name, rows in arms.items()}
    for name, prior_name in [('production', None), ('automatic_applied_to_export', 'automatic--head_refined')]:
        old = source['baseline'] if prior_name is None else next(a for a in source['automatic'] if a['id'] == prior_name)
        for key in ('eventPrecision', 'eventRecall', 'eventF1', 'trueRallies', 'predictedRallies', 'matchedRallies'):
            assert abs(scores[name]['identityMetrics']['pooled'][key] - old['identityMetrics']['pooled'][key]) < 1e-12
    numeric = [k for k, value in per_record[0]['workload'].items() if isinstance(value, (int, float))]
    pooled_work = {k: sum(row['workload'][k] for row in per_record) for k in numeric}
    return {'seed': seed, 'arms': scores, 'workload': pooled_work, 'perRecording': per_record}


def summarize(results):
    def stats(values):
        return {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
    output = {'workload': {k: stats([r['workload'][k] for r in results]) for k in results[0]['workload']}, 'arms': {}}
    for arm in results[0]['arms']:
        output['arms'][arm] = {
            'identity': {k: stats([r['arms'][arm]['identityMetrics']['pooled'][k] for r in results]) for k in (
                'eventPrecision', 'eventRecall', 'eventF1', 'trueRallies', 'predictedRallies', 'matchedRallies',
                'completeMisses', 'mergedPredictionsMaterial', 'splitTrueRalliesMaterial')},
            'rawCore': {k: stats([r['arms'][arm]['rawCoreMetrics'][k] for r in results]) for k in results[0]['arms'][arm]['rawCoreMetrics']},
            'padding': [{
                'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3,
                **{k: stats([r['arms'][arm]['durationMetrics'][pad][k] for r in results]) for k in (
                    *iv.SUM_FIELDS, 'P_pad', 'R_core', 'F1_padP_coreR', 'exportDurationDifferenceSeconds')}
            } for pad in (0, 1, 2, 3)],
        }
    return output


def main():
    self_check()
    registration = read(PRIOR / 'registration.json')
    contract = registration['contract']
    for identity in [contract['input'], *contract['sources'].values(), *contract['sourceCopies'].values()]:
        assert ref(Path(identity['path']))['sha256'] == identity['sha256'], identity['path']
    data = read(PRIOR / 'input.json')
    assert data['protectedTestOpened'] is False and data['recordingCount'] == 8
    assert sum(len(r['rallies']) for r in data['records']) == 322
    assert tuple(data['seeds']) == SEEDS
    results = [seed_result(seed, data['records']) for seed in SEEDS]
    result = {
        'schemaVersion': 1, 'createdAt': datetime.now(timezone.utc).isoformat(),
        'policy': 'Apply all frozen head_refined events. Review production-core minus automatic-core outside ignored time. Perfect human vetoes the entire parent edit iff any queued removed core overlaps human core; keep all other automatic edits. Recompute export from resulting event cores.',
        'targetPaddingSecondsEachSide': 2, 'sensitivityPaddingSecondsEachSide': [0, 1, 2, 3],
        'joinGapSeconds': 3, 'joinComparison': 'strictly-less-than', 'reviewContextSecondsEachSide': 2,
        'reviewContextJoinSeconds': 0, 'scope': 'Existing held-source-group development predictions; no protected test',
        'recordingCount': 8, 'sourceGroupCount': 4, 'goldRallyCount': 322, 'seeds': SEEDS,
        'trainingPerformed': False, 'thresholdSelectionPerformed': False, 'goldUsedToGenerateQueue': False,
        'identityRule': 'Frozen cardinality-first one-to-one IoU>=0.5 event matching; never union event identities.',
        'limitations': ['Perfect human decisions; playback at1x is not measured labor.',
            'Gold determines parent-veto decisions only inside queued removals. Reverting the whole parent can preserve production false positives and merge errors.',
            'Zero newly lost core does not guarantee correct rally identity, starts, ends or scoring.',
            'No removed-core trigger exists for coverage-preserving splits. Export-only queues have additional padding/gap-joining blind spots.',
            'Seed metrics pool recordings first; summary gives means and seed range, not confidence intervals.'],
        'inputs': [ref(PRIOR / 'input.json'), ref(PRIOR / 'registration.json'),
            *[ref(PRIOR / 'results' / f'{seed}.json') for seed in SEEDS]],
        'sourcesVerifiedAgainstPriorContract': len(contract['sources']), 'source': ref(Path(__file__)),
        'summary': summarize(results), 'results': results,
        'checks': {'syntheticPolicyCasesPassed': True, 'frozenAutomaticAndBaselineEventMetricsExact': True,
                   'allSeedsNoNewCoreLossOrCompleteMisses': True},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'results-v1.json'
    with path.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': ref(path), 'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
