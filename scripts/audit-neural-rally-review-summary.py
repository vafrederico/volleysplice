#!/usr/bin/env python3
"""Independent completeness and pooling audit after the fixed review run.

This file imports neither the runner, numerical study nor the summarizer. It
opens real outcomes only after both the complete report and summary exist.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import fsum, isclose, isfinite, sqrt
from pathlib import Path


ROOT = Path(private_value('private-reference-0092'))
TIME_TOTALS = ('paddedModelExportSeconds', 'paddedHumanExportSeconds', 'evaluableVideoSeconds',
               'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds',
               'missedCoreSeconds', 'coreHumanSeconds', 'paddedIntersectionSeconds', 'coreIntersectionSeconds')
TIME_FIELDS = (*TIME_TOTALS, 'P_pad', 'R_core', 'F1_padP_coreR', 'exportDurationDifferenceSeconds')
WORKLOAD = ('budgetSeconds', 'reviewSeconds', 'editSeconds', 'reviewClips', 'decisionRegions',
            'proposalsSelected', 'proposalsAvailable', 'censoredStarts', 'censoredEnds',
            'unobservedStarts', 'unobservedEnds', 'touchedRalliesWithUneditableBoundary',
            'reviewedTrueRallies', 'playbackTrueRallies')
COUNT_FIELDS = ('originalTrueRallies', 'ignoredTouchedTrueRallies', 'originalPredictedRallies',
                'entirelyMaskedPredictions', 'trueRallies', 'predictedRallies', 'matchedRallies',
                'completeMisses', 'mergedPredictions', 'splitTrueRallies',
                'mergedPredictionsMaterial', 'splitTrueRalliesMaterial',
                'unobservedPredictionStarts', 'unobservedPredictionEnds')
LOCAL_FIELDS = ('startLocalization', 'endLocalization', 'observedStartLocalization', 'observedEndLocalization')
TOLERANCES = ('0.25', '0.5', '1', '2')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(reference):
    require(sha(reference['path']) == reference['sha256'], 'Bound artifact changed: ' + reference['path'])
    return read(reference['path'])


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and isfinite(actual)
            and isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7),
            f'{label}: expected {expected!r}, received {actual!r}')


def compare(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': not an object')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            compare(actual[key], value, label + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list length differs')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{label}[{i}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        close(actual, expected, label)
    else:
        require(actual == expected, label + ': differs')


def finish_time(totals):
    result = dict(totals)
    p = result['paddedIntersectionSeconds'] / result['paddedModelExportSeconds'] if result['paddedModelExportSeconds'] else 0.
    r = result['coreIntersectionSeconds'] / result['coreHumanSeconds'] if result['coreHumanSeconds'] else 0.
    result.update(P_pad=p, R_core=r, F1_padP_coreR=2*p*r/(p+r) if p+r else 0.,
                  exportDurationDifferenceSeconds=result['paddedModelExportSeconds']-result['paddedHumanExportSeconds'])
    return result


def pooled_time(rows):
    return finish_time({key: fsum(x[key] for x in rows) for key in TIME_TOTALS})


def audit_time(rows, record_ids):
    require(len(rows) == 4 and {x['paddingSecondsBeforeAndAfter'] for x in rows} == {0, 1, 2, 3},
            'Four fixed padding cases required')
    for row in rows:
        require(row['joinGapSeconds'] == 3, 'Short-gap join threshold differs')
        require(len(row['perRecording']) == len(record_ids)
                and {x['id'] for x in row['perRecording']} == set(record_ids), 'Duration recording scope differs')
        compare(row, pooled_time(row['perRecording']), 'pooled duration')
        for part in [row, *row['perRecording']]:
            require(all(isfinite(part[k]) and part[k] >= -1e-8 for k in TIME_TOTALS), 'Invalid duration count')
            compare(part, finish_time({key: part[key] for key in TIME_TOTALS}), 'duration rates')
            close(part['paddedModelExportSeconds'], part['paddedIntersectionSeconds'] + part['incorrectExportSeconds'], 'model partition')
            close(part['paddedHumanExportSeconds'], part['paddedIntersectionSeconds'] + part['incorrectlyRemovedSeconds'], 'human partition')
            close(part['coreHumanSeconds'], part['coreIntersectionSeconds'] + part['missedCoreSeconds'], 'core partition')
            close(part['evaluableVideoSeconds'], part['paddedModelExportSeconds'] + part['correctlyRemovedSeconds']
                  + part['incorrectlyRemovedSeconds'], 'universe partition')


def rates(matched, predicted, true):
    return {'true': true, 'predicted': predicted, 'matched': matched,
            'falsePositive': predicted-matched, 'falseNegative': true-matched,
            'precision': matched/predicted if predicted else (0. if true else None),
            'recall': matched/true if true else None, 'f1': 2*matched/(predicted+true) if true else None}


def identity_totals(rows):
    totals = {key: sum(x[key] for x in rows) for key in COUNT_FIELDS}
    event = rates(totals['matchedRallies'], totals['predictedRallies'], totals['trueRallies'])
    totals.update(eventPrecision=event['precision'], eventRecall=event['recall'], eventF1=event['f1'],
                  falsePositiveRallies=event['falsePositive'], falseNegativeRallies=event['falseNegative'])
    for field in LOCAL_FIELDS:
        totals[field] = {}
        for tolerance in TOLERANCES:
            counts = {name: sum(row[field][tolerance][name] for row in rows) for name in ('matched', 'predicted', 'true')}
            totals[field][tolerance] = rates(**counts)
    return totals


def audit_identities(value, records):
    rows = value['recordings']
    require(len(rows) == len(records) and {r['id'] for r in rows} == set(records), 'Identity recording scope differs')
    compare(value['pooled'], identity_totals(rows), 'pooled event identities')
    groups = {x['sourceGroup'] for x in records.values()}
    require(set(value['sourceGroups']) == groups, 'Identity group scope differs')
    for group in groups:
        subset = [x for x in rows if x['sourceGroup'] == group]
        compare(value['sourceGroups'][group], identity_totals(subset), 'source-group event identities')
    for row in rows:
        require(all(isinstance(row[k], int) and row[k] >= 0 for k in COUNT_FIELDS), 'Invalid identity count')
        compare(row, identity_totals([row]), 'recording identity rates')
        require(row['matchedRallies'] <= min(row['trueRallies'], row['predictedRallies']), 'Impossible matching cardinality')
        require(len(row['matches']) == row['matchedRallies'], 'Stored matching count differs')


def audit_evaluation(value, records):
    require(value['durationAudit']['passed'] and value['durationAudit']['scopeCount'] == 13
            and value['durationAudit']['paddingCases'] == 4, 'Duration audit absent/failed')
    require(value['identityAudit']['passed'] and value['identityAudit']['recordingsAudited'] == 8
            and value['identityAudit']['sourceGroupsAudited'] == 4, 'Identity audit absent/failed')
    audit_time(value['durationMetrics'], records)
    audit_identities(value['identityMetrics'], records)


def workload_totals(rows, video_seconds):
    result = {key: fsum(row[key] for row in rows) for key in WORKLOAD}
    result.update(reviewFractionOfVideo=result['reviewSeconds']/video_seconds,
                  budgetUtilization=result['reviewSeconds']/result['budgetSeconds'])
    return result


def audit_cell(cell, records, config, seed, contract_sha):
    require(cell['contractSha256'] == contract_sha and cell['configuration'] == config and cell['seed'] == seed,
            'Cell provenance or configuration differs')
    audit_evaluation(cell['automatic'], records)
    require(len(cell['plans']) == 8 and {x['id'] for x in cell['plans']} == set(records), 'Plan inventory differs')
    by_plan = {x['id']: x for x in cell['plans']}
    for plan in cell['plans']:
        require(plan['candidateAudit']['passed'] and plan['queueAudit']['passed'], 'Candidate/queue audit absent/failed')
        require(plan['queueAudit']['budgetsAudited'] == 4 and plan['queueAudit']['labelDataRead'] is False,
                'Queue audit scope differs')
        previous_ids = set()
        for queue, fraction in zip(plan['queues'], (.05, .1, .2, .4)):
            require(queue['budgetFraction'] == fraction, 'Plan budget order differs')
            ids = queue['selectedIds']
            require(len(ids) == len(set(ids)) and previous_ids <= set(ids), 'Queue selections not unique/nested')
            previous_ids = set(ids)
            require(queue['reviewSeconds'] <= queue['budgetSeconds'] + 1e-8, 'Queue exceeds budget')
    require(len(cell['outcomes']) == 4 and [x['budgetFraction'] for x in cell['outcomes']] == [.05, .1, .2, .4],
            'Outcome budget inventory differs')
    for index, outcome in enumerate(cell['outcomes']):
        audit_evaluation(outcome, records)
        rows = outcome['recordings']
        require(len(rows) == 8 and {x['id'] for x in rows} == set(records), 'Outcome recording scope differs')
        for row in rows:
            compare(row, by_plan[row['id']]['queues'][index], 'Outcome queue exact replay')
            require(row['editorAudit']['passed'], 'Event editor audit absent/failed')
            require(row['reviewedTrueRallies'] <= row['playbackTrueRallies'] <= len(records[row['id']]['rallies']),
                    'Distinct rally workload counts impossible')
        target = next(x for x in outcome['durationMetrics'] if x['paddingSecondsBeforeAndAfter'] == 2)
        compare(outcome['workload'], workload_totals(rows, target['evaluableVideoSeconds']), 'Pooled workload')
        groups = {x['sourceGroup'] for x in records.values()}
        require(set(outcome['workloadBySourceGroup']) == groups, 'Workload group inventory differs')
        for group in groups:
            selected = [r for r in rows if r['sourceGroup'] == group]
            seconds = fsum(x['evaluableVideoSeconds'] for x in target['perRecording'] if x['sourceGroup'] == group)
            compare(outcome['workloadBySourceGroup'][group], workload_totals(selected, seconds), 'Group workload')


STAT_FIELDS = {'mean', 'min', 'max', 'seedPopulationStddev', 'availableSeeds', 'totalSeeds'}
DELTA_FIELDS = ('eventPrecision', 'eventRecall', 'eventF1', 'trueRallies', 'predictedRallies', 'matchedRallies',
                'falsePositiveRallies', 'falseNegativeRallies', 'completeMisses', 'mergedPredictions',
                'splitTrueRallies', 'mergedPredictionsMaterial', 'splitTrueRalliesMaterial',
                'unobservedPredictionStarts', 'unobservedPredictionEnds')


def statistics(values):
    available = [float(x) for x in values if x is not None]
    average = fsum(available)/len(available) if available else None
    return {'mean': average, 'min': min(available) if available else None,
            'max': max(available) if available else None,
            'seedPopulationStddev': sqrt(fsum((x-average)**2 for x in available)/len(available)) if available else None,
            'availableSeeds': len(available), 'totalSeeds': len(values)}


def numeric_tree(value):
    if isinstance(value, dict):
        return {key: numeric_tree(child) for key, child in value.items()
                if isinstance(child, dict) or child is None
                or isinstance(child, (int, float)) and not isinstance(child, bool)}
    return value


def tree_statistics(rows):
    require(len(rows) == 3, 'Exactly three independent seed summaries expected')
    if isinstance(rows[0], dict):
        require(all(set(x) == set(rows[0]) for x in rows), 'Numeric seed trees differ')
        return {key: tree_statistics([row[key] for row in rows]) for key in rows[0]}
    return statistics(rows)


def means(tree):
    if set(tree) == STAT_FIELDS:
        return tree['mean']
    return {key: means(child) for key, child in tree.items()}


def summary_evaluations(evaluations, group=None):
    padding = []
    for pad in (0, 1, 2, 3):
        rows = []
        for value in evaluations:
            row = next(x for x in value['durationMetrics'] if x['paddingSecondsBeforeAndAfter'] == pad)
            rows.append({key: row[key] for key in TIME_FIELDS} if group is None else
                        pooled_time([x for x in row['perRecording'] if x['sourceGroup'] == group]))
        stats = tree_statistics(rows)
        padding.append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3,
                        'metrics': means(stats), 'seedStatistics': stats})
    identities = [numeric_tree(x['identityMetrics']['pooled'] if group is None else
                               x['identityMetrics']['sourceGroups'][group]) for x in evaluations]
    identity_stats = tree_statistics(identities)
    return {'padding': padding, 'primary': padding[2]['metrics'],
            'identity': means(identity_stats), 'identitySeedStatistics': identity_stats}


def add_workload(expected, outcomes, group):
    samples = []
    for outcome in outcomes:
        source = outcome['workload'] if group is None else outcome['workloadBySourceGroup'][group]
        sample = {key: source[key] for key in (*WORKLOAD, 'reviewFractionOfVideo', 'budgetUtilization')}
        sample['unusedBudgetSeconds'] = sample['budgetSeconds']-sample['reviewSeconds']
        samples.append(sample)
    expected['workloadSeedStatistics'] = tree_statistics(samples)
    expected['workload'] = means(expected['workloadSeedStatistics'])


def add_deltas(expected, baseline):
    def delta(left, right):
        return left-right if left is not None and right is not None else None
    identity, original = expected['identity'], baseline['identity']
    changes = {key: delta(identity[key], original[key]) for key in DELTA_FIELDS}
    for field, prefix in (('observedStartLocalization', 'observedStart1s'), ('observedEndLocalization', 'observedEnd1s'),
                          ('startLocalization', 'allStart1s'), ('endLocalization', 'allEnd1s')):
        for key in ('precision', 'recall', 'f1', 'matched', 'predicted'):
            changes[prefix+key[0].upper()+key[1:]] = delta(identity[field]['1'][key], original[field]['1'][key])
    expected['identityDeltas'] = changes
    expected['primaryDeltas'] = {key: delta(expected['primary'][key], baseline['primary'][key]) for key in TIME_FIELDS}
    criteria = {'eventF1NonWorsening': changes['eventF1'] is not None and changes['eventF1'] >= -1e-12,
                'observedStart1sF1NonWorsening': changes['observedStart1sF1'] is not None and changes['observedStart1sF1'] >= -1e-12,
                'observedStart1sRecallNonWorsening': changes['observedStart1sRecall'] is not None and changes['observedStart1sRecall'] >= -1e-12,
                'completeMissesNonIncreasing': changes['completeMisses'] <= 1e-12}
    expected['guardrailScreen'] = {'passed': all(criteria.values()), 'criteria': criteria}


def audit_summary(contract, summary, cells, references):
    by_config, baseline_samples = defaultdict(dict), defaultdict(dict)
    for cell in cells:
        config, seed = cell['configuration'], cell['seed']
        require(seed not in by_config[config['id']], 'Duplicate configuration seed')
        by_config[config['id']][seed] = cell
        name = 'production' if config['mode'] == 'production' else config['model']
        old = baseline_samples[name].get(seed)
        require(old is None or old == cell['automatic'], 'Automatic baseline differs across policy cells')
        baseline_samples[name][seed] = cell['automatic']
    expected_arm_count = len(contract['configurations'])*len(contract['budgetFractions'])
    require(len(summary['automaticBaselines']) == len(baseline_samples)
            and len(summary['arms']) == expected_arm_count, 'Summary inventory incomplete')
    emitted_baselines = {x['id']: x for x in summary['automaticBaselines']}
    require(set(emitted_baselines) == set(baseline_samples), 'Baseline identity scope differs')
    expected_baselines = {}
    for name, seed_map in baseline_samples.items():
        require(set(seed_map) == set(contract['seeds']), 'Automatic seed scope differs')
        evaluations = [seed_map[seed] for seed in contract['seeds']]
        expected = summary_evaluations(evaluations)
        expected['sourceGroups'] = {g: summary_evaluations(evaluations, g) for g in contract['sourceGroups']}
        compare(emitted_baselines[name], expected, 'automatic ' + name)
        expected_baselines[name] = expected
    emitted_arms = {x['id']: x for x in summary['arms']}
    require(len(emitted_arms) == expected_arm_count, 'Duplicate summary arm')
    expected_arms = []
    for config in contract['configurations']:
        seed_map = by_config[config['id']]
        require(set(seed_map) == set(contract['seeds']), 'Configuration seed scope differs')
        name = 'production' if config['mode'] == 'production' else config['model']
        for fraction in contract['budgetFractions']:
            outcomes = [next(x for x in seed_map[seed]['outcomes'] if x['budgetFraction'] == fraction)
                        for seed in contract['seeds']]
            identifier = f"{config['id']}--budget-{round(fraction*100):02d}"
            expected = {**config, 'id': identifier, 'configurationId': config['id'],
                        'budgetFraction': fraction, 'automaticBaselineId': name,
                        **summary_evaluations(outcomes)}
            add_workload(expected, outcomes, None)
            add_deltas(expected, expected_baselines[name])
            expected['sourceGroups'] = {}
            for group in contract['sourceGroups']:
                scoped = summary_evaluations(outcomes, group)
                add_workload(scoped, outcomes, group)
                add_deltas(scoped, expected_baselines[name]['sourceGroups'][group])
                expected['sourceGroups'][group] = scoped
            expected['seedResults'] = [{**references[(config['id'], seed)], 'seed': seed, 'budgetFraction': fraction}
                                       for seed in contract['seeds']]
            require(identifier in emitted_arms, 'Missing summary arm ' + identifier)
            compare(emitted_arms[identifier], expected, 'arm ' + identifier)
            expected_arms.append(expected)
    require(set(emitted_arms) == {x['id'] for x in expected_arms}, 'Extra summary arm')
    rankings = {}
    for fraction in contract['budgetFractions']:
        for mode in ('production', 'individual'):
            rows = [x for x in expected_arms if x['budgetFraction'] == fraction and x['mode'] == mode]
            rows.sort(key=lambda x: (-x['primary']['F1_padP_coreR'], x['workload']['reviewSeconds'], x['id']))
            rankings[f'{mode}--budget-{round(fraction*100):02d}'] = [x['id'] for x in rows]
    require(summary['rankingsByDeclaredBudgetAndMode'] == rankings, 'Primary metric rankings differ')
    return {'automaticBaselinesAudited': len(expected_baselines), 'armsAudited': expected_arm_count,
            'sourceGroupArmSummariesAudited': expected_arm_count*len(contract['sourceGroups']),
            'paddingCasesAudited': 4, 'rankingListsAudited': 8, 'guardrailScreensAudited': 720,
            'seedStatisticsReconstructed': True}


def main(root):
    require((root/'report.json').exists() and (root/'summary.json').exists(),
            'Do not read outcomes before report and summary both exist')
    registration, report, summary = [read(root/name) for name in ('registration.json', 'report.json', 'summary.json')]
    contract = registration['contract']
    require(canonical(contract) == registration['sha256'], 'Registration hash differs')
    require(report['contractSha256'] == registration['sha256'] == summary['contractSha256'], 'Report/summary contract differs')
    require(report['status'] == 'completed-fixed-rally-review-proposals' and report['configurationSeedRuns'] == 108
            and report['outcomeCells'] == 432 and len(report['results']) == 108, 'Complete fixed matrix required')
    require(report['candidatePlansAudited'] == 864 and report['editedRecordingsAudited'] == 3456, 'Audit scope counters differ')
    for reference in [*contract['sources'].values(), contract['input'], contract['probabilities'], contract['protocol'],
                      contract['qualification'], contract['goldSemantics'], summary['sourceScript'], summary['sourceTests']]:
        require(sha(reference['path']) == reference['sha256'], 'Frozen source/input bytes changed: ' + reference['path'])
    require(summary['registration'] == identity(root/'registration.json') and summary['report'] == identity(root/'report.json'),
            'Summary report bindings differ')
    require(summary['primaryMetric'] == 'F1_padP_coreR' and summary['targetPaddingSeconds'] == 2
            and summary['joinGapSeconds'] == 3, 'Summary primary metric differs')
    require(len(summary['automaticBaselines']) == 4 and len(summary['arms']) == 144, 'Summary fixed matrix differs')
    data = bound(contract['input'])
    records = {x['id']: x for x in data['records']}
    require(len(records) == 8 and sum(len(x['rallies']) for x in records.values()) == 322, 'Recording/gold scope differs')
    configs = {x['id']: x for x in contract['configurations']}
    expected_keys = {(key, seed) for key in configs for seed in contract['seeds']}
    require(len(expected_keys) == 108, 'Registered matrix differs')
    cells, references = [], {}
    for reference in report['results']:
        cell = bound(reference)
        key = cell['configuration']['id'], cell['seed']
        require(key in expected_keys and key not in references, 'Unexpected/duplicate result cell')
        audit_cell(cell, records, configs[key[0]], key[1], registration['sha256'])
        references[key] = reference
        cells.append(cell)
    require(set(references) == expected_keys, 'Fixed cell matrix incomplete')
    require({str(p) for p in (root/'results').glob('*.json')} == {x['path'] for x in report['results']},
            'Unreported result JSON files exist')
    counters = audit_summary(contract, summary, cells, references)
    receipt = {'passed': True, 'kind': 'independent-rally-review-summary-audit-v1',
               'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': registration['sha256'],
               'registration': identity(root/'registration.json'), 'report': identity(root/'report.json'),
               'summary': identity(root/'summary.json'), 'implementation': summary['sourceScript'],
               'sourceScript': identity(Path(__file__)),
               'tests': identity(Path(__file__).resolve().parents[1]/'analysis/tests/test_neural_rally_review_summary_audit.py'),
               'configurationSeedRunsAudited': 108, 'outcomeCellsAudited': 432,
               'candidatePlansAudited': 864, 'editedRecordingsAudited': 3456,
               'numericalReconstructionAuditsRequiredPassed': True, **counters}
    destination = root/'summary-audit-v1.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': identity(destination), **counters}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    arguments = parser.parse_args()
    main(arguments.root)
