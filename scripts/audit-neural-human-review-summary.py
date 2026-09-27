#!/usr/bin/env python3
"""Independently check complete perfect-human-review pooling and interpretation.

Does not import the runner, review implementation or summarizer. Reads outcomes
only after both the complete report and summary exist. No selection or training.
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


ROOT = Path(private_value('private-reference-0091'))
TOTALS = ('paddedModelExportSeconds', 'paddedHumanExportSeconds', 'evaluableVideoSeconds',
          'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds',
          'missedCoreSeconds', 'coreHumanSeconds', 'paddedIntersectionSeconds', 'coreIntersectionSeconds')
METRICS = (*TOTALS, 'P_pad', 'R_core', 'F1_padP_coreR', 'exportDurationDifferenceSeconds')
WORKLOAD = ('rawDecisionSeconds', 'reviewSeconds', 'decisionSeconds', 'reviewClips',
            'flaggedCandidates', 'flaggedPositiveCandidates', 'flaggedNegativeCandidates',
            'reviewedTrueRallies', 'playbackTrueRallies', 'binaryKeptCandidates', 'binaryDroppedCandidates',
            'mixedCandidates', 'completeRallyLossesBinary', 'partialRallyLossesBinary',
            'completeRallyLossesBoundary', 'partialRallyLossesBoundary')
LEGACY = ('reviewSeconds', 'disputedSeconds', 'unwantedExportFlaggedSeconds', 'wantedExportFlaggedSeconds',
          'missedHumanExportFlaggedSeconds', 'missedCoreFlaggedSeconds', 'reviewClips',
          'decisionSegments', 'reviewedTrueRallies', 'playbackTrueRallies')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(ref):
    require(sha(ref['path']) == ref['sha256'], 'Changed bound artifact: '+ref['path'])
    return read(ref['path'])


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and isfinite(actual)
            and isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7),
            f'{label}: expected {expected!r}, received {actual!r}')


def finish(totals):
    row = dict(totals)
    p = row['paddedIntersectionSeconds']/row['paddedModelExportSeconds'] if row['paddedModelExportSeconds'] else 0.
    r = row['coreIntersectionSeconds']/row['coreHumanSeconds'] if row['coreHumanSeconds'] else 0.
    row.update(P_pad=p, R_core=r, F1_padP_coreR=2*p*r/(p+r) if p+r else 0.,
               exportDurationDifferenceSeconds=row['paddedModelExportSeconds']-row['paddedHumanExportSeconds'])
    return row


def pooled(rows):
    return finish({k: fsum(row[k] for row in rows) for k in TOTALS})


def compare_values(actual, expected, fields, label):
    for field in fields:
        require(field in actual, label+': missing '+field)
        close(actual[field], expected[field], label+'.'+field)


def validate_metrics(row, label):
    require(row['joinGapSeconds'] == 3, label+': gap threshold differs')
    for key in TOTALS:
        require(isfinite(row[key]) and row[key] >= -1e-8, label+': negative/nonfinite '+key)
    compare_values(row, finish(row), METRICS, label)
    close(row['paddedModelExportSeconds'], row['paddedIntersectionSeconds']+row['incorrectExportSeconds'], label+'.export partition')
    close(row['paddedHumanExportSeconds'], row['paddedIntersectionSeconds']+row['incorrectlyRemovedSeconds'], label+'.human partition')
    close(row['coreHumanSeconds'], row['coreIntersectionSeconds']+row['missedCoreSeconds'], label+'.core partition')
    close(row['evaluableVideoSeconds'], row['paddedModelExportSeconds']+row['correctlyRemovedSeconds']+
          row['incorrectlyRemovedSeconds'], label+'.universe partition')
    compare_values(row, pooled(row['perRecording']), METRICS, label+'.pooled')
    for rec in row['perRecording']:
        compare_values(rec, finish(rec), METRICS, label+'.'+rec['id'])


def by_pad(rows):
    require(all(r['paddingSecondsBeforeAndAfter'] in (0, 1, 2, 3) and
                not isinstance(r['paddingSecondsBeforeAndAfter'], bool) for r in rows),
            'Padding value differs from declared cases')
    result = {int(row['paddingSecondsBeforeAndAfter']): row for row in rows}
    require(set(result) == {0, 1, 2, 3} and len(rows) == 4, 'Padding inventory differs')
    return result


def mean_values(rows, fields):
    require(len(rows) == 3, 'Exactly three seed cells must be averaged')
    return {field: fsum(row[field] for row in rows)/3 for field in fields}


def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def spans(rows):
    return [(float(r['start']), float(r['end'])) if isinstance(r, dict) else (float(r[0]), float(r[1])) for r in rows]


def subtract_holes(interval, holes):
    parts = [interval]
    for a, b in sorted(spans(holes)):
        next_parts = []
        for start, end in parts:
            if b <= start or a >= end:
                next_parts.append((start, end))
            else:
                if start < a:
                    next_parts.append((start, a))
                if b < end:
                    next_parts.append((b, end))
        parts = next_parts
    return parts


def true_rallies(record, intervals):
    ranges = spans(intervals)
    return [index for index, rally in enumerate(spans(record['rallies']))
            if any(overlap(part, target) for part in subtract_holes(rally, record.get('ignoredIntervals', []))
                   for target in ranges)]


def audit_cell(cell, records, contract_hash):
    require(cell['contractSha256'] == contract_hash, 'Result contract differs')
    require(cell['independentHumanReview']['passed'] and cell['independentBinaryAccounting']['passed']
            and cell['independentBoundaryAccounting']['passed'], 'Prior independent reconstruction failed')
    automatic, binary, boundary, workload = [by_pad(cell[k]) for k in (
        'baseDurationMetrics', 'binaryDurationMetrics', 'boundaryDurationMetrics', 'workload')]
    require({r['id'] for r in cell['recordings']} == set(records) and len(cell['recordings']) == 8, 'Cell recording inventory differs')
    cfg = cell['configuration']
    for pad in range(4):
        for name, row in [('automatic', automatic[pad]), ('binary', binary[pad]), ('boundary', boundary[pad])]:
            validate_metrics(row, cfg['id']+'.'+name+f'.pad{pad}')
            require({r['id'] for r in row['perRecording']} == set(records), 'Metric recording scope differs')
        expected = {key: fsum(by_pad(r['padding'])[pad][key] for r in cell['recordings']) for key in WORKLOAD}
        expected['reviewFractionOfVideo'] = expected['reviewSeconds']/automatic[pad]['evaluableVideoSeconds']
        compare_values(workload[pad], expected, (*WORKLOAD, 'reviewFractionOfVideo'), 'workload pooling')
        require(boundary[pad]['P_pad']+1e-10 >= automatic[pad]['P_pad'] and
                boundary[pad]['R_core']+1e-10 >= automatic[pad]['R_core'], 'Boundary correction lowered precision/recall')
        require(expected['reviewedTrueRallies'] <= expected['playbackTrueRallies'] <= 322,
                'Distinct rally totals are not bounded')
        if cfg['family'] == 'combination' and not cfg['policy'].startswith('bidirectional'):
            close(boundary[pad]['R_core'], automatic[pad]['R_core'], 'Suppression-only boundary recall')
        if cfg['policy'] == 'all_candidates':
            for field in ('P_pad', 'R_core', 'F1_padP_coreR'):
                close(boundary[pad][field], 1, 'Full-review boundary reference')
            close(binary[pad]['R_core'], 1, 'Full-review binary recall reference')
            close(expected['reviewSeconds'], automatic[pad]['evaluableVideoSeconds'], 'Full-review playback coverage')
        for recording in cell['recordings']:
            rec = records[recording['id']]
            require(recording['sourceGroup'] == rec['sourceGroup'], 'Recording source group differs')
            row = by_pad(recording['padding'])[pad]
            candidates = recording['candidates']
            flagged = [r for r in candidates if r['flagged']]
            require(len({r['id'] for r in candidates}) == len(candidates), 'Duplicate candidate ID')
            require(recording['selected'] == [r['id'] for r in flagged], 'Selected candidate identity differs')
            close(row['flaggedCandidates'], len(flagged), 'Flagged candidate count')
            close(row['flaggedPositiveCandidates'], sum(r['kind'] == 'positive' for r in flagged), 'Positive candidate count')
            close(row['flaggedNegativeCandidates'], sum(r['kind'] == 'negative' for r in flagged), 'Negative candidate count')
            close(row['binaryKeptCandidates']+row['binaryDroppedCandidates'], len(flagged), 'Binary decision partition')
            require(0 <= row['mixedCandidates'] <= row['binaryKeptCandidates'], 'Mixed candidate count is not bounded')
            reviewed = true_rallies(rec, flagged)
            playback = true_rallies(rec, row['playbackIntervals'])
            require(row['reviewedTrueRallyIds'] == reviewed and row['playbackTrueRallyIds'] == playback,
                    'Independent distinct-rally reconstruction differs')
            close(row['reviewedTrueRallies'], len(reviewed), 'Reviewed distinct rally count')
            close(row['playbackTrueRallies'], len(playback), 'Playback distinct rally count')
            require(set(reviewed) <= set(playback), 'Raw reviewed rallies missing from playback')
    return {'automatic': automatic, 'binary': binary, 'boundary': boundary, 'workload': workload}


def expected_padding(cells, group=None):
    result = []
    for pad in range(4):
        row = {'paddingSecondsBeforeAndAfter': pad}
        for role, field in [('automatic', 'baseDurationMetrics'), ('binary', 'binaryDurationMetrics'),
                            ('boundary', 'boundaryDurationMetrics')]:
            seeds = [by_pad(cell[field])[pad] for cell in cells]
            if group is not None:
                seeds = [pooled([r for r in item['perRecording'] if r['sourceGroup'] == group]) for item in seeds]
            row[role] = mean_values(seeds, METRICS)
        workloads = [by_pad(cell['workload'])[pad] for cell in cells]
        if group is not None:
            workloads = []
            for cell in cells:
                values = {key: fsum(by_pad(r['padding'])[pad][key] for r in cell['recordings']
                                    if r['sourceGroup'] == group) for key in WORKLOAD}
                universe = fsum(r['evaluableVideoSeconds'] for r in by_pad(cell['baseDurationMetrics'])[pad]['perRecording']
                                if r['sourceGroup'] == group)
                values['reviewFractionOfVideo'] = values['reviewSeconds']/universe
                workloads.append(values)
        row['workload'] = mean_values(workloads, (*WORKLOAD, 'reviewFractionOfVideo'))
        result.append(row)
    return result


def compare_padding(actual, expected, label, legacy=False):
    a, b = by_pad(actual), by_pad(expected)
    roles = ('automatic', 'boundary') if legacy else ('automatic', 'binary', 'boundary')
    for pad in range(4):
        for role in roles:
            compare_values(a[pad][role], b[pad][role], METRICS, label+f'.pad{pad}.'+role)
        fields = LEGACY if legacy else (*WORKLOAD, 'reviewFractionOfVideo')
        compare_values(a[pad]['workload'], b[pad]['workload'], fields, label+f'.pad{pad}.workload')


def seed_statistics(rows, fields):
    result = {}
    for field in fields:
        values = [r[field] for r in rows]
        average = fsum(values)/len(values)
        result[field] = {'mean': average, 'min': min(values), 'max': max(values),
                         'seedPopulationStddev': sqrt(fsum((v-average)**2 for v in values)/len(values))}
    return result


def check_pareto(summary):
    checks = 0
    for scope, scenarios in summary['paretoByScope'].items():
        rows = [r for r in summary['configurations'] if
                (r['family'] == 'individual' if scope == 'individual' else r['anchor'] == scope)]
        require(len(rows) == 30, 'Pareto comparison scope differs')
        for scenario in ('binary', 'boundary'):
            candidates = {r['id']: (r['primary'][scenario]['F1_padP_coreR'],
                                     r['primary']['workload']['reviewSeconds']) for r in rows}
            dominated = set()
            keys = list(candidates)
            for i, left in enumerate(keys):
                for right in keys[i+1:]:
                    lf, lt = candidates[left]
                    rf, rt = candidates[right]
                    if lf >= rf and lt <= rt and (lf > rf or lt < rt):
                        dominated.add(right)
                    if rf >= lf and rt <= lt and (rf > lf or rt < lt):
                        dominated.add(left)
            expected = set(keys)-dominated
            require(set(scenarios[scenario]) == expected and len(scenarios[scenario]) == len(expected),
                    'Pareto frontier differs for '+scope+'.'+scenario)
            checks += 1
    require(checks == 8, 'Pareto scope inventory differs')
    return checks


def audit_legacy(cell, records):
    old = bound(cell['source'])
    require(all(cell[k] == old[k] for k in ('anchor', 'neuralId', 'mode', 'seed')),
            'Legacy source configuration identity differs')
    require(cell['automaticDurationMetrics'] == old['automaticDurationMetrics'] and
            cell['boundaryDurationMetrics'] == old['oracleDurationMetrics'], 'Legacy source scores changed')
    current, original = by_pad(cell['padding']), by_pad(old['queue'])
    for pad in range(4):
        for key in LEGACY[:7]:
            close(current[pad][key], original[pad][key], 'Legacy original workload')
        per = {r['id']: r for r in current[pad]['perRecording']}
        require(set(per) == set(records), 'Legacy recording scope differs')
        for old_row in original[pad]['perRecording']:
            row = per[old_row['id']]
            expected = true_rallies(records[row['id']], old_row['disputedIntervals'])
            playback = true_rallies(records[row['id']], old_row['reviewIntervals'])
            require(row['reviewedTrueRallyIds'] == expected and row['playbackTrueRallyIds'] == playback,
                    'Legacy independent distinct rally count differs')
            close(row['reviewedTrueRallies'], len(expected), 'Legacy reviewed rally count')
            close(row['playbackTrueRallies'], len(playback), 'Legacy playback rally count')
            close(row['decisionSegments'], len(old_row['disputedIntervals']), 'Legacy decision segment count')
        for key in LEGACY[7:]:
            close(current[pad][key], fsum(r[key] for r in per.values()), 'Legacy pooled count')


def uncertainty_nested(cells):
    solo = {(c['configuration']['neuralId'], c['seed'], c['configuration']['policy']): c
            for c in cells if c['configuration']['family'] == 'individual'}
    checks = 0
    for model in sorted({k[0] for k in solo}):
        for seed in (3407, 1729, 20260918):
            levels = [solo[model, seed, policy] for policy in ('uncertain_narrow', 'uncertain_medium', 'uncertain_wide')]
            rows = [{r['id']: r for r in level['recordings']} for level in levels]
            for rid in rows[0]:
                inventories = [[(x['id'], x['start'], x['end'], x['kind'], x['score']) for x in level[rid]['candidates']]
                               for level in rows]
                require(inventories[0] == inventories[1] == inventories[2], 'Uncertainty levels change candidate inventory/scores')
                selected = [set(level[rid]['selected']) for level in rows]
                require(selected[0] <= selected[1] <= selected[2], 'Uncertainty flag sets not nested')
                checks += 1
    return checks


def negative_grid_slivers(cells):
    occurrences, unique = 0, {}
    minimum = None
    for cell in cells:
        if cell['configuration']['family'] != 'individual':
            continue
        for record in cell['recordings']:
            for candidate in record['candidates']:
                if candidate['kind'] != 'negative':
                    continue
                length = candidate['end']-candidate['start']
                minimum = length if minimum is None else min(minimum, length)
                if length <= 1e-9:
                    occurrences += 1
                    key = (cell['configuration']['neuralId'], cell['seed'], record['id'], candidate['id'])
                    unique[key] = {'modelId': key[0], 'seed': key[1], 'recordingId': key[2], 'candidateId': key[3],
                                   'start': candidate['start'], 'end': candidate['end'], 'durationSeconds': length}
    return {'thresholdSeconds': 1e-9, 'policyCandidateOccurrences': occurrences,
            'uniqueModelSeedRecordingCandidates': len(unique), 'minimumNegativeCandidateSeconds': minimum,
            'examples': list(unique.values())[:20]}


def audit(root, summary_path, document=None):
    require((root/'report.json').exists() and summary_path.exists(), 'Wait for full completion report and summary')
    report, summary, registration = read(root/'report.json'), read(summary_path), read(root/'registration.json')
    require(report['status'] == 'completed-perfect-human-review', 'Study not complete; outcomes must remain unread')
    contract = registration['contract']
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(digest == registration['sha256'] == report['contractSha256'], 'Registration/report identity differs')
    require(summary['contractSha256'] == digest and summary['passed'] and summary['targetPaddingSeconds'] == 2
            and summary['joinGapSeconds'] == 3 and summary['paddingCases'] == [0, 1, 2, 3], 'Summary contract differs')
    require(summary['artifacts']['report'] == identity(root/'report.json') and
            summary['artifacts']['registration'] == identity(root/'registration.json'), 'Summary source binding differs')
    for ref in contract['sources'].values():
        require(sha(ref['path']) == ref['sha256'], 'Registered numerical source changed')
    require(len(report['results']) == 360 and len(report['legacyResults']) == 90 and report['policyCount'] == 120,
            'Full outcome inventory incomplete')
    records = {r['id']: r for r in bound(contract['neuralInput'])['records']}
    require(len(records) == 8 and sum(len(r['rallies']) for r in records.values()) == 322, 'Exact evaluation scope differs')
    cells = [bound(ref) for ref in report['results']]
    expected_configs = {r['id']: r for r in contract['configurations']}
    groups = defaultdict(list)
    for cell in cells:
        require(cell['configuration'] == expected_configs[cell['configuration']['id']], 'Cell recipe differs from registration')
        audit_cell(cell, records, digest)
        groups[cell['configuration']['id']].append(cell)
    require(set(groups) == {c['id'] for c in contract['configurations']} and len(groups) == 120, 'Policy inventory differs')
    summaries = {r['id']: r for r in summary['configurations']}
    require(set(summaries) == set(groups) and len(summary['configurations']) == 120, 'Summary policy inventory differs')
    grouped_checks = statistics_checks = 0
    for config_id, seeds in groups.items():
        require(len(seeds) == 3 and {c['seed'] for c in seeds} == {3407, 1729, 20260918}, 'Seed scope differs')
        row = summaries[config_id]
        require(all(row[k] == seeds[0]['configuration'][k] for k in ('id', 'family', 'anchor', 'neuralId', 'policy')),
                'Summary configuration metadata differs')
        expected = expected_padding(seeds)
        compare_padding(row['padding'], expected, config_id)
        require(row['primary'] == by_pad(row['padding'])[2], 'Primary padding differs')
        require({(r['path'], r['sha256']) for r in row['seedResults']} ==
                {(r['path'], r['sha256']) for r in report['results'] if Path(r['path']).name.startswith(config_id+'--')},
                'Summary seed artifact binding differs')
        close(row['binaryEventF1'], fsum(c['binaryEvaluation']['guardrails']['eventF1'] for c in seeds)/3,
              'Binary event F1 mean')
        for pad in range(4):
            actual_pad = by_pad(row['padding'])[pad]
            for role, source_field, fields in [('automatic', 'baseDurationMetrics', METRICS),
                ('binary', 'binaryDurationMetrics', METRICS), ('boundary', 'boundaryDurationMetrics', METRICS),
                ('workload', 'workload', (*WORKLOAD, 'reviewFractionOfVideo'))]:
                expected_stats = seed_statistics([by_pad(c[source_field])[pad] for c in seeds], fields)
                actual_stats = actual_pad[role+'SeedStatistics']
                require(set(actual_stats) == set(expected_stats), 'Seed-statistic field inventory differs')
                for field in fields:
                    compare_values(actual_stats[field], expected_stats[field],
                                   ('mean', 'min', 'max', 'seedPopulationStddev'), 'Seed statistics')
                    statistics_checks += 1
        require(set(row['sourceGroups']) == set(contract['sourceGroups']), 'Summary source-group scope differs')
        for group in contract['sourceGroups']:
            compare_padding(row['sourceGroups'][group], expected_padding(seeds, group), config_id+'.'+group)
            grouped_checks += 1
    legacy_cells = [bound(ref) for ref in report['legacyResults']]
    legacy_groups = defaultdict(list)
    for cell in legacy_cells:
        audit_legacy(cell, records)
        legacy_groups[cell['anchor'], cell['neuralId'], cell['mode']].append(cell)
    legacy_summaries = {(r['anchor'], r['neuralId'], r['mode']): r for r in summary['legacy']}
    require(set(legacy_groups) == set(legacy_summaries) and len(legacy_groups) == 30 and len(summary['legacy']) == 30,
            'Legacy summary inventory differs')
    for key, seeds in legacy_groups.items():
        require(len(seeds) == 3 and {c['seed'] for c in seeds} == {3407, 1729, 20260918}, 'Legacy seed scope differs')
        expected = []
        for pad in range(4):
            row = {'paddingSecondsBeforeAndAfter': pad}
            for role, field in [('automatic', 'automaticDurationMetrics'), ('boundary', 'boundaryDurationMetrics')]:
                row[role] = mean_values([by_pad(c[field])[pad] for c in seeds], METRICS)
            workloads = []
            for cell in seeds:
                values = dict(by_pad(cell['padding'])[pad])
                values['reviewFractionOfVideo'] = values['reviewSeconds']/by_pad(cell['automaticDurationMetrics'])[pad]['evaluableVideoSeconds']
                workloads.append(values)
            row['workload'] = mean_values(workloads, (*LEGACY, 'reviewFractionOfVideo'))
            expected.append(row)
        compare_padding(legacy_summaries[key]['padding'], expected, str(key), legacy=True)
        require(legacy_summaries[key]['primary'] == by_pad(legacy_summaries[key]['padding'])[2],
                'Legacy primary padding differs')
    for role, field in [('binary', 'rankedBinaryIds'), ('boundary', 'rankedBoundaryIds')]:
        ranking = summary[field]
        require(set(ranking) == set(summaries) and len(ranking) == 120, 'Ranking inventory differs')
        scores = [summaries[k]['primary'][role]['F1_padP_coreR'] for k in ranking]
        require(all(a+1e-12 >= b for a, b in zip(scores, scores[1:])), 'Ranking is not descending fixed-padding F1')
    baseline_keys = {r['anchor'] or r['neuralId'] for r in summaries.values()}
    require(set(summary['baselines']) == baseline_keys and len(baseline_keys) == 8, 'Baseline summary scope differs')
    for row in summaries.values():
        compare_values(summary['baselines'][row['anchor'] or row['neuralId']], row['primary']['automatic'], METRICS,
                       'Automatic baseline consistency')
    pareto_checks = check_pareto(summary)
    checks = uncertainty_nested(cells)
    slivers = negative_grid_slivers(cells)
    receipt = {'kind': 'independent-perfect-human-review-summary-audit-v1', 'passed': True,
        'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': digest,
        'report': identity(root/'report.json'), 'summary': identity(summary_path),
        'registration': identity(root/'registration.json'), 'sourceScript': identity(Path(__file__).resolve()),
        'implementation': identity(Path(__file__).resolve()),
        'tests': identity(Path(__file__).resolve().parents[1]/'analysis/tests/test_neural_human_review_summary_audit.py'),
        'counts': {'resultCells': 360, 'policies': 120, 'legacyCells': 90, 'legacyPolicies': 30,
                   'paddingCases': 4, 'sourceGroupSummaryChecks': grouped_checks, 'uncertaintyNestingChecks': checks,
                   'seedStatisticChecks': statistics_checks, 'paretoFrontiers': pareto_checks,
                   'seedCount': 3, 'recordings': 8, 'rallies': 322},
        'checks': ['Independent pooled numerators/denominators and harmonic F1', 'Three-seed averaging without replication inflation',
                   'All policy and legacy workload fields', 'All four source-group summaries',
                   'Independent distinct-rally counts from flagged/played intervals', 'Uncertainty flag nesting',
                   'Fixed-padding descending F1 rankings and within-scope Pareto frontiers',
                   'Seed min/max/population standard deviations', 'Full-review and suppression-only oracle invariants'],
        'negativeGridSliverCheck': slivers,
        'failures': [], 'protectedTestOpened': False, 'newFits': 0}
    if document is not None:
        receipt['finalDocument'] = identity(document)
    destination = root/'summary-audit-v1.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': identity(destination), 'counts': receipt['counts']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--document', type=Path)
    args = parser.parse_args()
    audit(args.root, args.summary or args.root/'summary.json', args.document)
