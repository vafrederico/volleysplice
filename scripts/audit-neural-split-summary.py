#!/usr/bin/env python3
"""Independently verify frozen split-adviser aggregation and report tables.

No summarizer, neural implementation, or metric implementation is imported.
Seed statistics are reconstructed directly from the three raw result cells.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path


ROOT = Path(private_value('private-reference-0093'))
REPO = Path(__file__).resolve().parents[1]
DOC = REPO/'docs/research/neural-split-advisor-results-2026-09-19.md'
SUM_FIELDS = ('paddedModelExportSeconds', 'paddedHumanExportSeconds', 'evaluableVideoSeconds',
              'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds',
              'missedCoreSeconds', 'coreHumanSeconds', 'paddedIntersectionSeconds', 'coreIntersectionSeconds')
WORKLOAD_FIELDS = ('reviewSeconds', 'budgetSeconds', 'unusedBudgetSeconds', 'reviewJobs', 'jobsAvailable',
                  'reviewClips', 'editRegions', 'proposalsReviewed', 'acceptedSplitCount', 'rejectedSplitCount',
                  'removedFalseParents', 'eventTimelineRemovedSeconds', 'eventTimelineAddedSeconds',
                  'reviewedTrueRallies', 'playbackTrueRallies')
STAT_KEYS = {'mean', 'min', 'max', 'seedPopulationStddev', 'availableSeeds', 'totalSeeds'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': Path(path).stat().st_size}


def verify_ref(item):
    require(sha(item['path']) == item['sha256'], 'Bound bytes changed: ' + item['path'])
    require(Path(item['path']).stat().st_size == item['sizeBytes'], 'Bound size changed: ' + item['path'])


def compare(expected, actual, path='value'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), path + ': key set differs')
        for key in expected:
            compare(expected[key], actual[key], path + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), path + ': list length differs')
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f'{path}[{i}]')
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(expected == actual, path + ': value differs')
    else:
        require(isinstance(actual, Real) and not isinstance(actual, bool) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-8),
                f'{path}: expected {expected!r}, got {actual!r}')


def numeric_tree(value):
    if isinstance(value, dict):
        return {k: numeric_tree(v) for k, v in value.items()
                if isinstance(v, dict) or v is None or isinstance(v, Real) and not isinstance(v, bool)}
    return value


def flatten(value, prefix=()):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            output.update(flatten(item, (*prefix, key)))
        return output
    return {prefix: value}


def at(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def verify_statistics(samples, means, statistics, label):
    leaves = [flatten(numeric_tree(s)) for s in samples]
    expected_keys = set(leaves[0])
    require(all(set(x) == expected_keys for x in leaves), label + ': input seed shapes differ')
    require(set(flatten(means)) == expected_keys, label + ': output mean shape differs')
    count = 0
    for path in sorted(expected_keys):
        raw = [leaf[path] for leaf in leaves]
        valid = [float(x) for x in raw if x is not None]
        require(all(math.isfinite(x) for x in valid), 'Nonfinite raw seed data')
        center = math.fsum(valid) / len(valid) if valid else None
        expected = {'mean': center, 'min': min(valid) if valid else None,
                    'max': max(valid) if valid else None,
                    'seedPopulationStddev': math.sqrt(math.fsum((x - center) ** 2 for x in valid) / len(valid)) if valid else None,
                    'availableSeeds': len(valid), 'totalSeeds': len(raw)}
        compare(expected, at(statistics, path), label + '.' + '.'.join(path) + '.statistics')
        compare(center, at(means, path), label + '.' + '.'.join(path) + '.mean')
        count += 1
    # Recursive shape validation includes empty dictionaries and unexpected leaves.
    def shape(sample, result):
        if isinstance(sample, dict):
            require(isinstance(result, dict) and set(result) == set(sample), label + ': statistic shape differs')
            for key in sample:
                shape(sample[key], result[key])
        else:
            require(set(result) == STAT_KEYS, label + ': statistic leaf differs')
    shape(numeric_tree(samples[0]), statistics)
    return count


def pooled_duration(rows):
    require(bool(rows), 'Empty duration scope')
    values = {key: math.fsum(row[key] for row in rows) for key in SUM_FIELDS}
    p = values['paddedIntersectionSeconds'] / values['paddedModelExportSeconds'] if values['paddedModelExportSeconds'] else 0.
    r = values['coreIntersectionSeconds'] / values['coreHumanSeconds'] if values['coreHumanSeconds'] else 0.
    values.update(P_pad=p, R_core=r, F1_padP_coreR=2*p*r/(p+r) if p+r else 0.,
                  exportDurationDifferenceSeconds=values['paddedModelExportSeconds'] - values['paddedHumanExportSeconds'])
    return values


def verify_scope(raw, result, record_ids, kind, scope_key=None):
    leaves = 0
    require([x['paddingSecondsBeforeAndAfter'] for x in result['padding']] == [0, 1, 2, 3], 'Summary pad scope differs')
    for pad, output in enumerate(result['padding']):
        require(output['joinGapSeconds'] == 3, 'Summary join threshold differs')
        samples = []
        for item in raw:
            rows = [r for r in item['durationMetrics'] if r['paddingSecondsBeforeAndAfter'] == pad]
            require(len(rows) == 1 and rows[0]['joinGapSeconds'] == 3, 'Raw padding contract differs')
            chosen = [r for r in rows[0]['perRecording'] if r['id'] in record_ids]
            require({r['id'] for r in chosen} == set(record_ids), 'Raw duration recording scope differs')
            sample = pooled_duration(chosen)
            if kind == 'pooled':
                compare(sample, {k: rows[0][k] for k in sample}, 'Raw pooled duration')
            samples.append(sample)
        leaves += verify_statistics(samples, output['metrics'], output['seedStatistics'], f'{kind}:{scope_key}:pad{pad}')
    compare(result['padding'][2]['metrics'], result['primary'], 'Primary duration row')
    for source, name in (('identityMetrics', 'identity'), ('splitMetrics', 'split')):
        samples = []
        for item in raw:
            metric = item[source]
            if kind == 'pooled':
                sample = metric['pooled']
            elif kind == 'group':
                sample = metric['sourceGroups'][scope_key]
            else:
                found = [row for row in metric['recordings'] if row['id'] == scope_key]
                require(len(found) == 1, 'Raw recording metric scope differs')
                sample = found[0]
            samples.append(sample)
        leaves += verify_statistics(samples, result[name], result[name + 'SeedStatistics'], f'{kind}:{scope_key}:{name}')
    samples = []
    for item in raw:
        if 'workload' not in item:
            sample = {key: 0 for key in WORKLOAD_FIELDS}
        else:
            selected = [r for r in item['perRecording'] if r['id'] in record_ids]
            require({r['id'] for r in selected} == set(record_ids), 'Workload recording scope differs')
            sample = {key: math.fsum(row[key] for row in selected) for key in WORKLOAD_FIELDS}
            if kind == 'pooled':
                compare(sample, item['workload'], 'Raw pooled workload')
        samples.append(sample)
    leaves += verify_statistics(samples, result['workload'], result['workloadSeedStatistics'], f'{kind}:{scope_key}:workload')
    return leaves


def assert_audits(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith('Audit') and isinstance(item, dict):
                if 'passed' in item:
                    require(item['passed'], 'Raw audit did not pass: ' + key)
                else:
                    require(all(x.get('passed') for x in item.values()), 'Raw nested audit did not pass')
            assert_audits(item)
    elif isinstance(value, list):
        for item in value:
            assert_audits(item)


def audit_aggregate(contract, cells, summary):
    seeds = contract['seeds']
    require(len(set(seeds)) == len(seeds) == 3, 'Expected three registered seeds')
    require(sorted(c['seed'] for c in cells) == sorted(seeds), 'Result seeds differ')
    require(summary['seeds'] == seeds, 'Summary seeds differ')
    records = sorted(r['id'] for r in cells[0]['baseline']['identityMetrics']['recordings'])
    groups = sorted(cells[0]['baseline']['identityMetrics']['sourceGroups'])
    group_ids = {g: [r['id'] for r in cells[0]['baseline']['identityMetrics']['recordings'] if r['sourceGroup'] == g] for g in groups}
    require(summary['recordings'] == records and summary['sourceGroups'] == groups, 'Summary scope differs')
    expected_meta = {'production': {'kind': 'baseline'}}
    expected_meta.update({'automatic--' + p: {'kind': 'automatic', 'policy': p} for p in contract['splitPolicies']})
    for config in contract['reviewInventories']:
        for ranker in contract['rankers']:
            for fraction in contract['budgetFractions']:
                identity = f"{config['id']}--{ranker}--budget-{round(fraction*100):02d}"
                expected_meta[identity] = {'kind': 'reviewed', 'policy': config['policy'], 'inventory': config['inventory'],
                                          'ranker': ranker, 'budgetFraction': fraction}
    rows = [summary['baseline'], *summary['automatic'], *summary['reviewed']]
    require(len(rows) == len(expected_meta) and {r['id'] for r in rows} == set(expected_meta), 'Aggregate arm scope differs')
    raw_by_arm = {key: [] for key in expected_meta}
    for cell in cells:
        assert_audits(cell)
        require(cell['baseline'] == cells[0]['baseline'], 'Production differs across seeds')
        auto = cell['automatic']; reviewed = cell['reviewed']
        require(len(auto) == len(contract['splitPolicies']), 'Missing/duplicate automatic arm')
        require(len(reviewed) == len(expected_meta) - len(auto) - 1, 'Missing/duplicate review arm')
        raw_rows = [('production', cell['baseline']), *[(r['id'], r) for r in [*auto, *reviewed]]]
        require(len({key for key, _ in raw_rows}) == len(expected_meta), 'Duplicate raw arm')
        for identity, row in raw_rows:
            require(identity in raw_by_arm, 'Unknown raw arm')
            raw_by_arm[identity].append(row)
            require(row['durationMetrics'] == cell['baseline']['durationMetrics'], 'Export differs from production')
            require(row['splitMetrics']['pooled']['additionalCompleteMisses'] == 0, 'Additional complete miss')
            require(abs(row['splitMetrics']['pooled']['rawCoreSecondsLostFromBaseline']) < 1e-9, 'True core lost')
            if identity.startswith('automatic--'):
                require(abs(row['splitMetrics']['pooled']['rawSelectedSecondsLostFromBaseline']) < 1e-9
                        and abs(row['splitMetrics']['pooled']['rawSelectedSecondsAddedToBaseline']) < 1e-9,
                        'Automatic occupancy changed')
    leaf_count = 0
    for row in rows:
        identity = row['id']; meta = expected_meta[identity]
        compare(meta, {k: row[k] for k in meta}, identity + '.metadata')
        require(set(row['sourceGroups']) == set(groups) and set(row['recordings']) == set(records), 'Arm slice scope differs')
        raw = raw_by_arm[identity]
        leaf_count += verify_scope(raw, row, records, 'pooled')
        for group in groups:
            leaf_count += verify_scope(raw, row['sourceGroups'][group], group_ids[group], 'group', group)
        for recording in records:
            leaf_count += verify_scope(raw, row['recordings'][recording], [recording], 'recording', recording)
    for ids, output, label in [(records, summary['cleanupYield']['pooled'], 'pooled'),
                               *[(group_ids[g], summary['cleanupYield']['sourceGroups'][g], g) for g in groups],
                               *[([r], summary['cleanupYield']['recordings'][r], r) for r in records]]:
        samples = []
        for cell in cells:
            yields = [p['cleanupYield'] for p in cell['plans'] if p['id'] in ids]
            require(len(yields) == len(ids), 'Cleanup yield scope differs')
            fields = ['flaggedParents', 'whollyFalseParents', 'realOrMixedParents']
            if all('realRalliesTouched' in item for item in yields):
                fields.append('realRalliesTouched')
            total = {k: math.fsum(item[k] for item in yields) for k in fields}
            total['whollyFalsePrecision'] = total['whollyFalseParents'] / total['flaggedParents'] if total['flaggedParents'] else None
            samples.append(total)
        leaf_count += verify_statistics(samples, output['metrics'], output['seedStatistics'], 'cleanup.' + label)
    compare({'everyExportPaddingEqualsProduction': True, 'automaticRawOccupancyPreserved': True,
             'zeroAdditionalCompleteMisses': True, 'zeroRawCoreLoss': True, 'seedCountsPooledBeforeAveraging': True},
            summary['verification'], 'verification')
    compare({'metric': 'F1_padP_coreR', 'targetPaddingSeconds': 2,
             'result': 'all arms tie exactly with production; no export winner'}, summary['primaryRanking'], 'ranking')
    for key, value in {'automaticOutcomes': 9, 'reviewedOutcomes': 168, 'automaticArms': 3, 'reviewedArms': 56}.items():
        compare(value, summary[key], key)
    return {'passed': True, 'seedCells': len(cells), 'comparisonArms': len(rows) - 1, 'baselineArms': 1,
            'scopeSummaries': len(rows) * (1 + len(groups) + len(records)),
            'paddingRows': len(rows) * (1 + len(groups) + len(records)) * 4,
            'seedStatisticLeaves': leaf_count}


def display(value, scale=1.):
    return 'n/a' if value is None else f'{value * scale:.2f}'


def range_display(value, scale=1.):
    center = display(value['mean'], scale)
    return (center if value['mean'] is None or value['min'] == value['max'] else
            f"{center} [{display(value['min'], scale)}, {display(value['max'], scale)}]")


def table_lines(headers, rows):
    line = lambda items: '| ' + ' | '.join(str(v) for v in items) + ' |'
    return [line(headers), line(['---'] * len(headers)), *[line(row) for row in rows]]


QUALITY_HEADERS = ['Arm', 'Rally P %', 'Rally R %', 'Rally F1 %', 'Observed start R @1s %',
                   'Observed start F1 @1s %', 'Complete misses', 'Additional misses', 'Review min', 'Real rallies reviewed']


def quality_row(name, row):
    i, s, w = row['identitySeedStatistics'], row['splitSeedStatistics'], row['workloadSeedStatistics']
    return [name, *[range_display(i[k], 100.) for k in ('eventPrecision', 'eventRecall', 'eventF1')],
            range_display(i['observedStartLocalization']['1']['recall'], 100.),
            range_display(i['observedStartLocalization']['1']['f1'], 100.),
            range_display(i['completeMisses']), range_display(s['additionalCompleteMisses']),
            range_display(w['reviewSeconds'], 1/60), range_display(w['reviewedTrueRallies'])]


def audit_tables(summary, document):
    base = summary['baseline']; automatic = summary['automatic']; reviewed = summary['reviewed']
    all_arms = [base, *automatic, *reviewed]
    expected = table_lines(['Padding each side (s)', 'P_pad %', 'R_core %', 'F1_padP_coreR %',
                           'Model export min', 'Human export min', 'Model-human min', 'Correct removed min',
                           'Incorrect removed min', 'Incorrect export min'],
        [[r['paddingSecondsBeforeAndAfter'], *[display(r['metrics'][k], 100.) for k in ('P_pad', 'R_core', 'F1_padP_coreR')],
          *[display(r['metrics'][k], 1/60) for k in ('paddedModelExportSeconds', 'paddedHumanExportSeconds',
           'exportDurationDifferenceSeconds', 'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds')]]
         for r in base['padding']])
    expected += table_lines(QUALITY_HEADERS, [quality_row(a['id'], a) for a in [base, *automatic]])
    expected += table_lines(['Policy', 'Tolerance (s)', 'Proposals', 'Matched', 'Spurious', 'Missed targets',
                             'Split P %', 'Split R %', 'Split F1 %'],
        [[a['policy'], tolerance,
          *[range_display(a['splitSeedStatistics']['splitLocalization'][tolerance][k]) for k in ('predicted', 'matched', 'falsePositive', 'falseNegative')],
          *[range_display(a['splitSeedStatistics']['splitLocalization'][tolerance][k], 100.) for k in ('precision', 'recall', 'f1')]]
         for a in automatic for tolerance in ('0.5', '1', '2')])
    expected += table_lines(['Scope', 'Flagged parents', 'Wholly false', 'Real/mixed', 'Wholly-false precision %', 'Real rallies touched'],
        [[label, *[range_display(row['seedStatistics'][k]) for k in ('flaggedParents', 'whollyFalseParents', 'realOrMixedParents')],
          range_display(row['seedStatistics']['whollyFalsePrecision'], 100.),
          range_display(row['seedStatistics']['realRalliesTouched']) if 'realRalliesTouched' in row['seedStatistics'] else 'not recorded']
         for label, row in [('pooled', summary['cleanupYield']['pooled']), *summary['cleanupYield']['sourceGroups'].items()]])
    expected += table_lines(QUALITY_HEADERS, [quality_row(a['id'], a) for a in reviewed])
    expected += table_lines(['Arm', 'Parent jobs', 'Playback clips', 'Split proposals reviewed', 'Accepted', 'Rejected',
                            'False parents removed', 'Event timeline removed min', 'Unused budget min'],
        [[a['id'], *[range_display(a['workloadSeedStatistics'][k]) for k in ('reviewJobs', 'reviewClips', 'proposalsReviewed',
          'acceptedSplitCount', 'rejectedSplitCount', 'removedFalseParents')],
          *[range_display(a['workloadSeedStatistics'][k], 1/60) for k in ('eventTimelineRemovedSeconds', 'unusedBudgetSeconds')]] for a in reviewed])
    expected += table_lines(['Arm', 'Confirmed splits', 'Unrecovered split targets', 'Confirmed split R %',
                            'Confirmed split F1 %', 'Material merged events', 'Material split gold rallies', 'Raw core loss (s)'],
        [[a['id'], *[range_display(a['splitSeedStatistics']['splitLocalization']['1'][k]) for k in ('matched', 'falseNegative')],
          *[range_display(a['splitSeedStatistics']['splitLocalization']['1'][k], 100.) for k in ('recall', 'f1')],
          range_display(a['identitySeedStatistics']['mergedPredictionsMaterial']),
          range_display(a['identitySeedStatistics']['splitTrueRalliesMaterial']),
          range_display(a['splitSeedStatistics']['rawCoreSecondsLostFromBaseline'])] for a in reviewed])
    for group in summary['sourceGroups']:
        expected += table_lines(QUALITY_HEADERS, [quality_row(a['id'], a['sourceGroups'][group]) for a in all_arms])
    for recording in summary['recordings']:
        expected += table_lines(QUALITY_HEADERS, [quality_row(a['id'], a['recordings'][recording]) for a in all_arms])
    require('## Export invariance and accounting' in document, 'Generated report body missing')
    body = document.split('## Export invariance and accounting', 1)[1]
    emitted = [line for line in body.splitlines() if line.startswith('|')]
    require(len(emitted) == len(expected), 'Generated table row count differs')
    for i, (want, actual) in enumerate(zip(expected, emitted)):
        require(want == actual, f'Generated table row {i} differs: expected {want!r}, got {actual!r}')
    return {'passed': True, 'tableLinesVerified': len(expected), 'allGeneratedNumericTablesCompared': True,
            'executiveProseAndTablesBeforeGeneratedBodyExcluded': True}


def run(root, document_path, receipt_path):
    require((root/'report.json').exists() and (root/'summary.json').exists(), 'Matrix or summary is incomplete')
    registration = read(root/'registration.json'); contract = registration['contract']
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(canonical == registration['sha256'], 'Registration hash differs')
    bound = [*contract['sources'].values(), *contract['sourceCopies'].values(), contract['input'],
             contract['probabilities'], contract['protocol'], contract['qualification'], contract['priorRegistration']]
    for item in bound:
        verify_ref(item)
    report = read(root/'report.json')
    require(report['passed'] and report['contractSha256'] == canonical, 'Matrix report failed or contract differs')
    require(report['automaticOutcomes'] == 9 and report['reviewedOutcomes'] == 168, 'Matrix size differs')
    cells = []
    for row in report['seeds']:
        verify_ref(row['result']); cells.append(read(row['result']['path']))
        require(cells[-1]['contractSha256'] == canonical and cells[-1]['seed'] == row['seed'], 'Cell identity differs')
    summary = read(root/'summary.json')
    require(summary['contractSha256'] == canonical, 'Summary contract differs')
    for item in [summary['registration'], summary['report'], summary['summarizer'], *summary['inputs']]:
        verify_ref(item)
    aggregate = audit_aggregate(contract, cells, summary)
    tables = audit_tables(summary, document_path.read_text(encoding='utf-8'))
    # Recheck after reading to bind a stable snapshot of all frozen dependencies.
    for item in bound:
        verify_ref(item)
    receipt = {'passed': True, 'createdAt': datetime.now(timezone.utc).isoformat(),
               'contractSha256': canonical, 'registration': ref(root/'registration.json'),
               'report': ref(root/'report.json'), 'summary': ref(root/'summary.json'),
               'document': ref(document_path), 'auditor': ref(Path(__file__)),
               'frozenSourcesVerified': len(contract['sources']), 'frozenSourceCopiesVerified': len(contract['sourceCopies']),
               'aggregateAudit': aggregate, 'generatedTableAudit': tables}
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': ref(receipt_path), **aggregate, **tables}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--doc', type=Path, default=DOC)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    run(args.root, args.doc, args.receipt or args.root/'summary-audit-v1.json')
