#!/usr/bin/env python3
"""Summarize every frozen production-preserving split-adviser outcome."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from statistics import mean, pstdev


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0093'))
DOC = REPO/'docs/research/neural-split-advisor-results-2026-09-19.md'
SUM_FIELDS = ('paddedModelExportSeconds', 'paddedHumanExportSeconds', 'evaluableVideoSeconds',
              'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds',
              'missedCoreSeconds', 'coreHumanSeconds', 'paddedIntersectionSeconds', 'coreIntersectionSeconds')
DURATION_FIELDS = ('P_pad', 'R_core', 'F1_padP_coreR', *SUM_FIELDS, 'exportDurationDifferenceSeconds')
WORKLOAD_FIELDS = ('reviewSeconds', 'budgetSeconds', 'unusedBudgetSeconds', 'reviewJobs', 'jobsAvailable',
                  'reviewClips', 'editRegions', 'proposalsReviewed', 'acceptedSplitCount', 'rejectedSplitCount',
                  'removedFalseParents', 'eventTimelineRemovedSeconds', 'eventTimelineAddedSeconds',
                  'reviewedTrueRallies', 'playbackTrueRallies')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path):
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': Path(path).stat().st_size}


def bound(ref):
    require(sha(ref['path']) == ref['sha256'], 'Bound artifact changed: '+ref['path'])
    require(Path(ref['path']).stat().st_size == ref['sizeBytes'], 'Bound artifact size changed')
    return read(ref['path'])


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stats(values):
    require(all(v is None or isinstance(v, Real) and not isinstance(v, bool) and isfinite(v) for v in values), 'Invalid numeric statistic')
    found = [float(v) for v in values if v is not None]
    return {'mean': mean(found) if found else None, 'min': min(found) if found else None,
            'max': max(found) if found else None, 'seedPopulationStddev': pstdev(found) if found else None,
            'availableSeeds': len(found), 'totalSeeds': len(values)}


def numeric(value):
    if isinstance(value, dict):
        return {k: numeric(v) for k, v in value.items()
                if isinstance(v, dict) or v is None or isinstance(v, Real) and not isinstance(v, bool)}
    require(value is None or isinstance(value, Real) and not isinstance(value, bool), 'Non-numeric leaf')
    return value


def tree_stats(rows):
    require(bool(rows), 'No samples')
    if isinstance(rows[0], dict):
        require(all(isinstance(r, dict) and set(r) == set(rows[0]) for r in rows), 'Metric shape differs between seeds')
        return {key: tree_stats([r[key] for r in rows]) for key in rows[0]}
    return stats(rows)


def means(tree):
    if set(tree) == {'mean', 'min', 'max', 'seedPopulationStddev', 'availableSeeds', 'totalSeeds'}:
        return tree['mean']
    return {key: means(value) for key, value in tree.items()}


def pool_duration(rows):
    require(bool(rows), 'Empty duration scope')
    output = {key: sum(row[key] for row in rows) for key in SUM_FIELDS}
    p = output['paddedIntersectionSeconds']/output['paddedModelExportSeconds'] if output['paddedModelExportSeconds'] else 0.
    r = output['coreIntersectionSeconds']/output['coreHumanSeconds'] if output['coreHumanSeconds'] else 0.
    output.update(P_pad=p, R_core=r, F1_padP_coreR=2*p*r/(p+r) if p+r else 0.,
                  exportDurationDifferenceSeconds=output['paddedModelExportSeconds']-output['paddedHumanExportSeconds'])
    return output


def _metric_scope(evaluation, name, group=None, recording=None):
    metric = evaluation[name]
    if recording is not None:
        rows = [r for r in metric['recordings'] if r['id'] == recording]
        require(len(rows) == 1, 'Recording metric missing or repeated')
        return numeric(rows[0])
    return numeric(metric['sourceGroups'][group] if group is not None else metric['pooled'])


def summarize_evaluations(evaluations, group=None, recording=None):
    output = {'padding': []}
    for pad in (0, 1, 2, 3):
        samples = []
        for e in evaluations:
            rows = [r for r in e['durationMetrics'] if r['paddingSecondsBeforeAndAfter'] == pad]
            require(len(rows) == 1 and rows[0]['joinGapSeconds'] == 3, 'Padding/join scope differs')
            row = rows[0]
            if group is None and recording is None:
                samples.append({key: row[key] for key in DURATION_FIELDS})
            else:
                selected = [r for r in row['perRecording'] if (recording is None or r['id'] == recording)
                            and (group is None or r['sourceGroup'] == group)]
                samples.append(pool_duration(selected))
        statistics = tree_stats(samples)
        output['padding'].append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3,
                                  'metrics': means(statistics), 'seedStatistics': statistics})
    output['primary'] = output['padding'][2]['metrics']
    for source, name in (('identityMetrics', 'identity'), ('splitMetrics', 'split')):
        statistics = tree_stats([_metric_scope(e, source, group, recording) for e in evaluations])
        output[name] = means(statistics); output[name+'SeedStatistics'] = statistics
    samples = []
    for e in evaluations:
        rows = e.get('perRecording', [])
        if 'workload' not in e:
            samples.append({key: 0 for key in WORKLOAD_FIELDS})
        elif group is None and recording is None:
            samples.append({key: e['workload'][key] for key in WORKLOAD_FIELDS})
        else:
            selected = [r for r in rows if (recording is None or r['id'] == recording)
                        and (group is None or r['sourceGroup'] == group)]
            require(bool(selected), 'Missing workload scope')
            samples.append({key: sum(r[key] for r in selected) for key in WORKLOAD_FIELDS})
    statistics = tree_stats(samples)
    output['workload'] = means(statistics); output['workloadSeedStatistics'] = statistics
    return output


def _check_evaluation(evaluation):
    require(evaluation['durationAudit']['passed'] and evaluation['identityAudit']['passed'], 'Outcome audit failed')
    for key, value in evaluation.items():
        if key.endswith('Audit') and isinstance(value, dict) and 'passed' in value:
            require(value['passed'], 'Outcome auxiliary audit failed: '+key)
    require([r['paddingSecondsBeforeAndAfter'] for r in evaluation['durationMetrics']] == [0, 1, 2, 3], 'Padding cases differ')
    require(all(r['joinGapSeconds'] == 3 for r in evaluation['durationMetrics']), 'Join contract differs')


def expected_reviewed(contract):
    return {f"{c['id']}--{ranker}--budget-{round(100*fraction):02d}":
            {'policy': c['policy'], 'inventory': c['inventory'], 'ranker': ranker, 'budgetFraction': fraction}
            for c in contract['reviewInventories'] for ranker in contract['rankers'] for fraction in contract['budgetFractions']}


def aggregate(contract, cells):
    """Validate fixed scope and aggregate already-pooled seed metrics."""
    seeds = contract['seeds']
    require(len(set(seeds)) == len(seeds) == 3, 'Exactly three distinct registered seeds required')
    require(sorted(c['seed'] for c in cells) == sorted(seeds), 'Missing/duplicate/unregistered seed')
    cells = sorted(cells, key=lambda c: seeds.index(c['seed']))
    expected = expected_reviewed(contract)
    expected_auto = {'automatic--'+p for p in contract['splitPolicies']}
    automatic = {}; reviewed = {}
    first_base = cells[0]['baseline']
    for cell in cells:
        _check_evaluation(cell['baseline'])
        require(cell['baseline'] == first_base, 'Production baseline differs across seeds')
        require(len(cell['automatic']) == len(expected_auto) and {r['id'] for r in cell['automatic']} == expected_auto, 'Automatic scope differs')
        require(len(cell['reviewed']) == len(expected) and {r['id'] for r in cell['reviewed']} == set(expected), 'Reviewed scope differs')
        for kind, storage in (('automatic', automatic), ('reviewed', reviewed)):
            for row in cell[kind]:
                _check_evaluation(row)
                require(row['durationMetrics'] == cell['baseline']['durationMetrics'], 'Frozen exports changed')
                require(row['splitMetrics']['pooled']['additionalCompleteMisses'] == 0, 'New complete rally miss')
                require(abs(row['splitMetrics']['pooled']['rawCoreSecondsLostFromBaseline']) <= 1e-9, 'Raw rally coverage lost')
                if kind == 'automatic':
                    require(row['policy'] == row['id'].removeprefix('automatic--'), 'Automatic policy mismatch')
                    require(abs(row['splitMetrics']['pooled']['rawSelectedSecondsLostFromBaseline']) <= 1e-9
                            and abs(row['splitMetrics']['pooled']['rawSelectedSecondsAddedToBaseline']) <= 1e-9, 'Automatic partition changed raw occupancy')
                else:
                    require(all(row[key] == value for key, value in expected[row['id']].items()), 'Reviewed arm metadata mismatch')
                    for key in WORKLOAD_FIELDS:
                        require(abs(row['workload'][key] - sum(p[key] for p in row['perRecording'])) <= 1e-7, 'Workload total mismatch: '+key)
                storage.setdefault(row['id'], []).append(row)
    groups = sorted(first_base['identityMetrics']['sourceGroups'])
    recordings = sorted(r['id'] for r in first_base['identityMetrics']['recordings'])
    group_by_recording = {r['id']: r['sourceGroup'] for r in first_base['identityMetrics']['recordings']}
    def arm(identity, evaluations, metadata):
        return {'id': identity, **metadata, **summarize_evaluations(evaluations),
                'sourceGroups': {g: summarize_evaluations(evaluations, group=g) for g in groups},
                'recordings': {r: summarize_evaluations(evaluations, recording=r) for r in recordings}}
    baseline = arm('production', [c['baseline'] for c in cells], {'kind': 'baseline'})
    automatic_rows = [arm(k, automatic[k], {'kind': 'automatic', 'policy': automatic[k][0]['policy']}) for k in sorted(automatic)]
    reviewed_rows = [arm(k, reviewed[k], {'kind': 'reviewed', **expected[k]}) for k in sorted(reviewed)]
    def cleanup_scope(record_ids):
        samples = []
        for cell in cells:
            yields = [p['cleanupYield'] for p in cell['plans'] if p['id'] in record_ids]
            require(len(yields) == len(record_ids), 'Cleanup recording scope differs')
            fields = ['flaggedParents', 'whollyFalseParents', 'realOrMixedParents']
            if all('realRalliesTouched' in y for y in yields):
                fields.append('realRalliesTouched')
            total = {k: sum(y[k] for y in yields) for k in fields}
            total['whollyFalsePrecision'] = total['whollyFalseParents']/total['flaggedParents'] if total['flaggedParents'] else None
            samples.append(total)
        statistics = tree_stats(samples)
        return {'metrics': means(statistics), 'seedStatistics': statistics}
    return {'kind': 'production-preserving-compact-split-adviser-summary-v1',
            'seeds': seeds, 'sourceGroups': groups, 'recordings': recordings,
            'automaticOutcomes': sum(len(c['automatic']) for c in cells),
            'reviewedOutcomes': sum(len(c['reviewed']) for c in cells),
            'automaticArms': len(automatic_rows), 'reviewedArms': len(reviewed_rows),
            'primaryRanking': {'metric': 'F1_padP_coreR', 'targetPaddingSeconds': 2,
                               'result': 'all arms tie exactly with production; no export winner'},
            'verification': {'everyExportPaddingEqualsProduction': True,
                             'automaticRawOccupancyPreserved': True, 'zeroAdditionalCompleteMisses': True,
                             'zeroRawCoreLoss': True, 'seedCountsPooledBeforeAveraging': True},
            'baseline': baseline, 'automatic': automatic_rows, 'reviewed': reviewed_rows,
            'cleanupYield': {'pooled': cleanup_scope(recordings),
                             'sourceGroups': {g: cleanup_scope([r for r in recordings if group_by_recording[r] == g]) for g in groups},
                             'recordings': {r: cleanup_scope([r]) for r in recordings}}}


def fmt(value, *, percent=False, seconds_to_minutes=False):
    if value is None:
        return 'n/a'
    factor = 100 if percent else 1/60 if seconds_to_minutes else 1
    return f'{value*factor:.2f}'


def ranged(stat, *, percent=False, seconds_to_minutes=False):
    center = fmt(stat['mean'], percent=percent, seconds_to_minutes=seconds_to_minutes)
    if stat['mean'] is None:
        return center
    lo = fmt(stat['min'], percent=percent, seconds_to_minutes=seconds_to_minutes)
    hi = fmt(stat['max'], percent=percent, seconds_to_minutes=seconds_to_minutes)
    return center if stat['min'] == stat['max'] else f'{center} [{lo}, {hi}]'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def _quality_row(label, arm):
    i, s, w = arm['identitySeedStatistics'], arm['splitSeedStatistics'], arm['workloadSeedStatistics']
    return [label, ranged(i['eventPrecision'], percent=True), ranged(i['eventRecall'], percent=True),
            ranged(i['eventF1'], percent=True), ranged(i['observedStartLocalization']['1']['recall'], percent=True),
            ranged(i['observedStartLocalization']['1']['f1'], percent=True), ranged(i['completeMisses']),
            ranged(s['additionalCompleteMisses']), ranged(w['reviewSeconds'], seconds_to_minutes=True),
            ranged(w['reviewedTrueRallies'])]


QUALITY_HEADERS = ['Arm', 'Rally P %', 'Rally R %', 'Rally F1 %', 'Observed start R @1s %',
                   'Observed start F1 @1s %', 'Complete misses', 'Additional misses', 'Review min', 'Real rallies reviewed']


def markdown(summary):
    base = summary['baseline']; split = base['split']; identity = base['identity']
    all_rows = [base, *summary['automatic'], *summary['reviewed']]
    lines = ['# Production-preserving compact split adviser: results', '',
             'This experiment keeps original production export footage fixed and evaluates compact neural guidance for separate rally identities. All arms tie on the required primary export metric `F1_padP_coreR`; the event diagnostics below do not replace that ranking.', '',
             f"Scope: {len(summary['recordings'])} development recordings, {len(summary['sourceGroups'])} source groups, {identity['trueRallies']:.0f} eligible gold rallies; {summary['automaticOutcomes']} automatic outcomes and {summary['reviewedOutcomes']} restricted ideal-human outcomes across {len(summary['seeds'])} seeds. No new detector training, learned confidence calibration, protected test, or production change.", '',
             'Values are means across seeds after pooling counts within each seed; brackets show the seed minimum and maximum, not confidence intervals. Source groups and recordings remain the same footage across seeds. Review minutes are union playback duration at 1x including context, not measured human labor.', '',
             f"There are {split['splitTargets']:.0f} parent-specific genuine additional-start targets ({split['uniqueTargetTrueRallies']:.0f} distinct gold rallies), of which {split['accessibleSplitTargets']:.0f} satisfy the two-second edge guard and {split['edgeInaccessibleSplitTargets']:.0f} do not. The full target count remains the split-recall denominator. This is a narrow split benchmark, not all-rally or score accuracy.", '',
             'Automatic rows apply all suggested internal cuts. Restricted review accepts only proposed starts matched within one second and snaps those accepted starts to gold; it does not discover unproposed starts or adjust original endpoints. Cleanup can remove a wholly false parent from the event timeline only after perfect human classification; exported footage still stays selected. Mixed parents remain. Synthetic cut ends are not observed dead-ball boundaries.', '',
             '## Export invariance and accounting', '',
             'Every automatic and reviewed arm equals this production table exactly at all four padding cases. Padding applies symmetrically; overlapping/touching ranges merge and positive gaps strictly below three seconds join. Ignored intervals are subtracted without rejoining. Correct removed time means omitted time outside the wanted human export; incorrect removed time means wanted human export omitted. All duration columns are minutes.', '',
             table(['Padding each side (s)', 'P_pad %', 'R_core %', 'F1_padP_coreR %', 'Model export min', 'Human export min', 'Model-human min', 'Correct removed min', 'Incorrect removed min', 'Incorrect export min'],
                   [[p['paddingSecondsBeforeAndAfter'], *[fmt(p['metrics'][k], percent=True) for k in ('P_pad','R_core','F1_padP_coreR')],
                     *[fmt(p['metrics'][k], seconds_to_minutes=True) for k in ('paddedModelExportSeconds','paddedHumanExportSeconds','exportDurationDifferenceSeconds','correctlyRemovedSeconds','incorrectlyRemovedSeconds','incorrectExportSeconds')]] for p in base['padding']]), '',
             '## Automatic split policies', '', table(QUALITY_HEADERS, [_quality_row(a['id'], a) for a in [base, *summary['automatic']]]), '',
             'Automatic split precision/recall/F1 below score proposed timestamps against secondary starts within the same original parent and valid component. A duplicate or unmatched proposal is spurious. A first rally start is not a split target. Parent partitions preserve raw occupancy exactly.', '',
             table(['Policy', 'Tolerance (s)', 'Proposals', 'Matched', 'Spurious', 'Missed targets', 'Split P %', 'Split R %', 'Split F1 %'],
                   [[a['policy'], tol, *[ranged(a['splitSeedStatistics']['splitLocalization'][tol][k]) for k in ('predicted','matched','falsePositive','falseNegative')],
                     *[ranged(a['splitSeedStatistics']['splitLocalization'][tol][k], percent=True) for k in ('precision','recall','f1')]] for a in summary['automatic'] for tol in ('0.5','1','2')]), '',
             '## Cleanup flag yield before review', '',
             'Cleanup means compact supports less than half of an original parent after two-second neural dilation. These are heuristic flags, not calibrated probabilities. Wholly-false precision is the fraction whose event record can be removed with zero gold-core overlap; real/mixed flags must be retained in this restricted simulation.', '',
             table(['Scope', 'Flagged parents', 'Wholly false', 'Real/mixed', 'Wholly-false precision %', 'Real rallies touched'],
                   [[name, *[ranged(c['seedStatistics'][k]) for k in ('flaggedParents','whollyFalseParents','realOrMixedParents')], ranged(c['seedStatistics']['whollyFalsePrecision'], percent=True),
                     ranged(c['seedStatistics']['realRalliesTouched']) if 'realRalliesTouched' in c['seedStatistics'] else 'not recorded']
                    for name,c in [('pooled', summary['cleanupYield']['pooled']), *summary['cleanupYield']['sourceGroups'].items()]]), '',
             '## All restricted review configurations', '',
             'Arm names give split policy, review inventory, ordering and per-video budget cap. The actual reviewed minutes can be below the cap because a whole parent does not fit or the queue is exhausted. `none--cleanup_only` isolates cleanup without split advice. All rows retain the original production export.', '',
             table(QUALITY_HEADERS, [_quality_row(a['id'], a) for a in summary['reviewed']]), '',
             '### Review actions and workload', '',
             table(['Arm', 'Parent jobs', 'Playback clips', 'Split proposals reviewed', 'Accepted', 'Rejected', 'False parents removed', 'Event timeline removed min', 'Unused budget min'],
                   [[a['id'], *[ranged(a['workloadSeedStatistics'][k]) for k in ('reviewJobs','reviewClips','proposalsReviewed','acceptedSplitCount','rejectedSplitCount','removedFalseParents')],
                     *[ranged(a['workloadSeedStatistics'][k], seconds_to_minutes=True) for k in ('eventTimelineRemovedSeconds','unusedBudgetSeconds')]] for a in summary['reviewed']]), '',
             '### Confirmed split diagnostics', '',
             'These rows score accepted, gold-snapped splits only. Their perfect precision is an assumption of the human oracle, not model precision; automatic proposal precision appears above. Accepted snaps are exact, so 0.5/1/2-second scores coincide. Full sensitivity metrics are retained in summary JSON.', '',
             table(['Arm', 'Confirmed splits', 'Unrecovered split targets', 'Confirmed split R %', 'Confirmed split F1 %', 'Material merged events', 'Material split gold rallies', 'Raw core loss (s)'],
                   [[a['id'], ranged(a['splitSeedStatistics']['splitLocalization']['1']['matched']), ranged(a['splitSeedStatistics']['splitLocalization']['1']['falseNegative']),
                     ranged(a['splitSeedStatistics']['splitLocalization']['1']['recall'], percent=True), ranged(a['splitSeedStatistics']['splitLocalization']['1']['f1'], percent=True),
                     ranged(a['identitySeedStatistics']['mergedPredictionsMaterial']), ranged(a['identitySeedStatistics']['splitTrueRalliesMaterial']),
                     ranged(a['splitSeedStatistics']['rawCoreSecondsLostFromBaseline'])] for a in summary['reviewed']]), '',
             '## Source-group sensitivity', '',
             'Group metrics pool recording counts before rates. The full JSON additionally retains every numeric field, every padding case, and seed ranges within each scope.', '']
    for group in summary['sourceGroups']:
        lines.extend(['### '+group, '', table(QUALITY_HEADERS, [_quality_row(a['id'], a['sourceGroups'][group]) for a in all_rows]), ''])
    lines.extend(['## Recording sensitivity', '', 'The same automatic and restricted-review comparisons are retained per recording; these are diagnostic slices, not extra independent test sets.', ''])
    for recording in summary['recordings']:
        lines.extend(['### '+recording, '', table(QUALITY_HEADERS, [_quality_row(a['id'], a['recordings'][recording]) for a in all_rows]), ''])
    lines.extend(['## Interpretation limits and provenance', '',
                  'Unchanged export recall does not imply correct rally counts or safe automatic score updates. A false split can create a false scoring event despite retaining every frame. Observed start localization is a serve-contact timing proxy, not serving-side, point-winner or reconstructed-score accuracy. Dead time remains inside partitioned event records because no automatic endpoint trimming is performed.', '',
                  'These development recordings informed earlier model/decoder choices. Results require fresh-footage validation and actual human review measurements before a product decision. Low-confidence cleanup flags alone leave both export and event metrics unchanged until a human acts.', '',
                  'The summary validates exact four-padding export equality, zero additional complete misses, zero raw-core loss, and automatic raw-occupancy preservation for every arm. Upstream candidate, partition, queue, human-action, identity and duration audits remain bound in the original result files. Complete numeric means/ranges, scope summaries and input references are in `summary.json` alongside the registered experiment.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--doc', type=Path, default=DOC)
    args = parser.parse_args()
    registration = read(args.root/'registration.json'); contract = registration['contract']
    require(canonical(contract) == registration['sha256'], 'Registration changed')
    for ref in [*contract['sources'].values(), *contract['sourceCopies'].values(), contract['input'], contract['probabilities'], contract['protocol'], contract['qualification']]:
        require(sha(ref['path']) == ref['sha256'], 'Frozen dependency changed: '+ref['path'])
    report = read(args.root/'report.json')
    require(report['passed'] and report['contractSha256'] == registration['sha256'], 'Execution incomplete or wrong contract')
    cells = [bound(x['result']) for x in report['seeds']]
    require(all(c['contractSha256'] == registration['sha256'] for c in cells), 'Cell contract differs')
    summary = aggregate(contract, cells)
    require(summary['automaticOutcomes'] == contract['automaticOutcomes'] and summary['reviewedOutcomes'] == contract['reviewedOutcomes'], 'Matrix totals differ')
    summary.update(createdAt=datetime.now(timezone.utc).isoformat(), contractSha256=registration['sha256'],
                   registration=reference(args.root/'registration.json'), report=reference(args.root/'report.json'),
                   inputs=[x['result'] for x in report['seeds']], summarizer=reference(Path(__file__)))
    with (args.root/'summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, allow_nan=False); f.write('\n')
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    with args.doc.open('x', encoding='utf-8') as f:
        f.write(markdown(summary))
    print(json.dumps({'passed': True, 'summary': reference(args.root/'summary.json'), 'document': reference(args.doc),
                      'automaticArms': summary['automaticArms'], 'reviewedArms': summary['reviewedArms']}))


if __name__ == '__main__':
    main()
