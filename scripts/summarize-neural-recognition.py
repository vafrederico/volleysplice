#!/usr/bin/env python3
"""Summarize complete audited recognition arms against the frozen controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value

from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_evaluation import evaluate_predictions
from analysis.schema import Interval
from analysis.crop_evaluation import subtract_intervals

ROOT = Path(private_value('private-reference-0057'))
SEEDS = (3407, 1729, 20260918)
REFERENCE = ROOT/'2026-09-19-short-boost-transfer/study'
PRODUCTION = ROOT/'2026-09-19-production-combinations/results/productionDefault--fixed.json'


def duration(intervals):
    return sum(row.end-row.start for row in intervals)


def metrics(result):
    evaluation = result['evaluation']
    p, g = evaluation['primary'], evaluation['guardrails']
    coverage = g['primaryExportCoverage']
    rallies = coverage['rallies']
    short = [r for r in rallies if r['end']-r['start'] <= 3.]
    long = [r for r in rallies if r['end']-r['start'] > 3.]
    return {**{k: p[k] for k in ('P_pad', 'R_core', 'F1_padP_coreR', 'paddedModelExportSeconds',
                'paddedHumanExportSeconds', 'exportDurationDifferenceSeconds')},
            'eventPrecision': g['eventPrecision'], 'eventRecall': g['eventRecall'], 'eventF1': g['eventF1'],
            'completeLosses': coverage['completeRallyLosses'],
            'incompleteLosses': coverage['completeRallyLosses']+coverage['partialRallyLosses'],
            'shortCompleteLosses': sum(r['completelyLost'] for r in short),
            'longR_core': sum(r['retainedCoreSeconds'] for r in long)/sum(r['evaluableCoreSeconds'] for r in long),
            'incorrectExportSeconds': p['paddedModelExportSeconds']-p['paddedPrecisionIntersectionSeconds'],
            'wantedExportOmittedSeconds': p['paddedHumanExportSeconds']-p['paddedPrecisionIntersectionSeconds'],
            'missedCoreSeconds': p['coreHumanSeconds']-p['coreRecallIntersectionSeconds']}


def group_summary(results):
    keys = metrics(results[0])
    source_metrics = {group: [metrics({'evaluation': r['evaluation']['sourceGroups'][group]}) for r in results]
                      for group in results[0]['evaluation']['sourceGroups']}
    return {'mean': {k: statistics.mean(metrics(r)[k] for r in results) for k in keys},
            'range': {k: [min(metrics(r)[k] for r in results), max(metrics(r)[k] for r in results)] for k in keys},
            'seeds': [{**metrics(r), 'seed': r['seed']} for r in results],
            'padding': [{k: statistics.mean(r['evaluation']['padding'][i][k] for r in results)
                for k in ('paddingSecondsBeforeAndAfter', 'joinGapSeconds', 'P_pad', 'R_core', 'F1_padP_coreR',
                          'paddedModelExportSeconds', 'paddedHumanExportSeconds', 'exportDurationDifferenceSeconds')}
                for i in range(4)],
            'sourceGroups': {group: {k: statistics.mean(row[k] for row in rows) for k in keys}
                             for group, rows in source_metrics.items()}}


def compare(new, control):
    rows = []
    for seed in SEEDS:
        n = metrics(next(r for r in new if r['seed'] == seed))
        c = metrics(next(r for r in control if r['seed'] == seed))
        recovery = (n['completeLosses'] < c['completeLosses']
                    and n['shortCompleteLosses'] < c['shortCompleteLosses']
                    and n['incompleteLosses'] <= c['incompleteLosses']
                    and n['R_core'] >= c['R_core']-1e-12
                    and n['longR_core'] >= c['longR_core']-1e-12
                    and n['F1_padP_coreR'] >= c['F1_padP_coreR']-.005)
        f1 = (n['F1_padP_coreR'] > c['F1_padP_coreR']
              and n['completeLosses'] <= c['completeLosses'] and n['R_core'] >= c['R_core']-1e-12)
        rows.append({'seed': seed, 'delta': {key: n[key]-c[key] for key in n},
                     'recoveryDirectionPassed': recovery, 'f1DirectionPassed': f1})
    n, c = group_summary(new)['mean'], group_summary(control)['mean']
    pooled_recovery = (n['completeLosses'] <= .8*c['completeLosses'] and n['shortCompleteLosses'] <= .8*c['shortCompleteLosses']
                      and n['incompleteLosses'] <= c['incompleteLosses'] and n['R_core'] >= c['R_core']-1e-12
                      and n['longR_core'] >= c['longR_core']-1e-12 and n['F1_padP_coreR'] >= c['F1_padP_coreR']-.005)
    pooled_f1 = (n['F1_padP_coreR'] >= c['F1_padP_coreR']+.02 and n['completeLosses'] <= c['completeLosses']
                 and n['R_core'] >= c['R_core']-1e-12)
    return {'meanDelta': {key: n[key]-c[key] for key in n}, 'seeds': rows,
            'seedDirectionRule': (
                'Recovery direction: strictly fewer complete losses and short complete losses; no increase in '
                'incomplete losses, no decrease in core or long-rally core recall, and F1 decline at most 0.005. '
                'F1 direction: strictly higher F1, no increase in complete losses, and no decrease in core recall. '
                'Each screen requires its direction in at least two seeds. The 20% recovery reductions and '
                '0.02 F1 gain thresholds apply to seed means, with the same respective guards.'),
            'recoveryPassed': pooled_recovery and sum(r['recoveryDirectionPassed'] for r in rows) >= 2,
            'f1Passed': pooled_f1 and sum(r['f1DirectionPassed'] for r in rows) >= 2}


def gold_identity(rows):
    return {r['id']: {k: r[k] for k in ('sourceGroup', 'durationSeconds', 'rallies', 'ignoredIntervals')} for r in rows}


def frozen_controls(reference, reference_report):
    """Bind each historical payload to exactly one cell in the frozen report."""
    controls = {}
    for family, kind in (('av', 'tcn'), ('dino', 'dino_tcn')):
        controls[family] = []
        for seed in SEEDS:
            entries = [row for row in reference_report['results']
                       if row.get('cohort') == 'reviewed_export' and row.get('kind') == kind
                       and row.get('lossArm') == 'short_boost' and row.get('seed') == seed]
            if len(entries) != 1:
                raise ValueError(f'Historical reference needs exactly one control cell: {kind}/{seed}')
            payload = read(reference/f'result-reviewed_export-{kind}-short_boost-{seed}.json')
            if payload != entries[0]:
                raise ValueError(f'Historical control payload differs from reference report: {kind}/{seed}')
            controls[family].append(payload)
    return controls


def production_complementarity(candidates, production):
    """Describe same-rally coverage differences, without proposing a fusion rule."""
    baseline = {(r['recordingId'], r['truthIndex']): r for r in
                production['evaluation']['guardrails']['primaryExportCoverage']['rallies']}
    rows = []
    count_keys = ('productionMissesRetainedAny', 'productionMissesRetainedPartly',
                  'productionMissesRetainedFully', 'newCompleteLosses', 'ralliesWithWorseCoverage')
    for candidate in candidates:
        coverage = {(r['recordingId'], r['truthIndex']): r for r in
                    candidate['evaluation']['guardrails']['primaryExportCoverage']['rallies']}
        if set(coverage) != set(baseline):
            raise ValueError('Production complementarity rally identities differ')

        def details(keys):
            return [{**{field: baseline[key][field] for field in
                        ('recordingId', 'truthIndex', 'start', 'end', 'evaluableCoreSeconds')},
                     'productionRetainedCoreSeconds': baseline[key]['retainedCoreSeconds'],
                     'candidateRetainedCoreSeconds': coverage[key]['retainedCoreSeconds']}
                    for key in sorted(keys)]

        partly = [key for key, value in coverage.items()
                  if baseline[key]['completelyLost'] and value['partiallyLost']]
        fully = [key for key, value in coverage.items()
                 if baseline[key]['completelyLost'] and value['fullyCovered']]
        introduced = [key for key, value in coverage.items()
                      if value['completelyLost'] and not baseline[key]['completelyLost']]
        rows.append({'seed': candidate['seed'], 'productionMissesRetainedAny': len(partly)+len(fully),
                     'productionMissesRetainedPartly': len(partly), 'productionMissesRetainedFully': len(fully),
                     'productionMissesRetainedPartlyIds': details(partly),
                     'productionMissesRetainedFullyIds': details(fully),
                     'newCompleteLosses': len(introduced), 'newCompleteLossIds': details(introduced),
                     'ralliesWithWorseCoverage': sum(value['retainedCoreSeconds']+1e-9 < baseline[key]['retainedCoreSeconds']
                                                    for key, value in coverage.items())})
    return {'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
            'interpretation': 'Descriptive paired export coverage only; partial retention is not full rally recovery. '
                              'No combination policy, threshold selection, or fusion accuracy is evaluated.',
            'productionCompleteMisses': sum(value['completelyLost'] for value in baseline.values()),
            'mean': {key: statistics.mean(row[key] for row in rows) for key in count_keys}, 'seeds': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference_report_identity = identity(REFERENCE/'report.json')
    controls = frozen_controls(REFERENCE, read(REFERENCE/'report.json'))
    gold = controls['av'][0]['predictions']
    gold_signature = gold_identity(gold)
    for rows in controls.values():
        for row in rows:
            if gold_identity(row['predictions']) != gold_signature or evaluate_predictions(row['predictions']) != row['evaluation']:
                raise ValueError('Historical control scope/evaluation differs on replay')
    production = read(PRODUCTION)
    prod_rows = [{**row, 'predictions': production['predictions'][row['id']]} for row in gold]
    prod = {'seed': None, 'evaluation': evaluate_predictions(prod_rows), 'predictions': prod_rows}
    if abs(prod['evaluation']['primary']['F1_padP_coreR'] - production['evaluation']['primary']['F1_padP_coreR']) > 1e-10:
        raise ValueError('Production baseline scope differs')
    results = {'production_default': [prod], 'av_tcn_short_boost': controls['av'], 'dino_tcn_short_boost': controls['dino']}
    references = [identity(PRODUCTION), identity(REFERENCE/'preregistration.json'), reference_report_identity]
    references += [identity(REFERENCE/f'result-reviewed_export-{kind}-short_boost-{seed}.json')
                   for kind in ('tcn', 'dino_tcn') for seed in SEEDS]
    comparisons = {}
    for study in args.study:
        report, reg = read(study/'report.json'), read(study/'preregistration.json')
        if reg['contract'].get('referenceReport') != reference_report_identity:
            raise ValueError('Study reference report binding differs from current historical reference')
        if report['status'] != 'completed-recognition-development' or report['contractSha256'] != reg['sha256']:
            raise ValueError('Study incomplete or identity differs')
        audit = read(study/'audit.json')
        if (audit.get('passed') is not True or audit.get('contractSha256') != reg['sha256']
                or audit.get('kind') != 'independent-recognition-audit-v1'
                or audit.get('report') != identity(study/'report.json')
                or audit.get('registration') != identity(study/'preregistration.json')
                or audit.get('counts') != {'seeds': 3, 'physicalFits': 30, 'logicalInnerViews': 36}):
            raise ValueError('Study needs passing independent audit')
        rows = report['results']
        if sorted(r['seed'] for r in rows) != sorted(SEEDS) or any(gold_identity(r['predictions']) != gold_signature for r in rows):
            raise ValueError('Seed/gold/source/ignored comparison scope differs')
        for row in rows:
            measured = evaluate_predictions(row['predictions'])
            if measured != row['evaluation']:
                raise ValueError('Published evaluation differs from replay')
        config = reg['contract']['config']
        key = f"{config['family']}_{config['head']}"
        if key in results:
            raise ValueError('Duplicate study key')
        results[key] = rows
        control = 'dino' if config['family'] == 'dino' else 'av'
        comparisons[key] = compare(rows, controls[control])
        comparisons[key]['control'] = f'{control}_tcn_short_boost'
        references.extend((identity(study/'report.json'), identity(study/'audit.json'), identity(study/'preregistration.json')))
    complementarity = {key: production_complementarity(rows, prod) for key, rows in results.items()
                       if key != 'production_default'}
    for key, comparison in comparisons.items():
        comparison['productionCoverage'] = complementarity[key]['seeds']
        comparison['productionCoverageMean'] = complementarity[key]['mean']
    summary = {key: group_summary(rows) for key, rows in results.items()}
    evaluable_seconds = sum(duration(subtract_intervals((Interval(0, row['durationSeconds']),),
                            tuple(Interval(v['start'], v['end']) for v in row['ignoredIntervals']))) for row in gold)
    for value in summary.values():
        for metrics_row in [value['mean'], *value['seeds']]:
            metrics_row['correctlyRemovedSeconds'] = evaluable_seconds - metrics_row['paddedHumanExportSeconds'] - metrics_row['incorrectExportSeconds']
    result = {'kind': 'recognition-comparison-summary-v1', 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
              'aggregation': 'Pool durations within each seed, then mean seeds; never concatenate repeated recordings',
              'references': references, 'models': summary, 'comparisons': comparisons,
              'productionComplementarity': complementarity,
              'protectedTestOpened': False, 'productionChanged': False}
    write_immutable(args.output, result)
    lines = ['# Recognition comparison', '', 'Development only: 8 recordings / 4 source groups / 322 rallies. '
             'Target padding 2 seconds each side; join positive gaps strictly below 3 seconds. '
             'Neural rows average three seeds after pooling recordings per seed. Production has historical exposure.', '',
             '| Model | P_pad % | R_core % | F1_padP_coreR % | Event P % | Event R % | Event F1 % | Complete misses | Export min | Correctly removed min | Incorrect export min | Wanted export omitted min |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    ranked = sorted(summary, key=lambda k: summary[k]['mean']['F1_padP_coreR'], reverse=True)
    for key in ranked:
        m = summary[key]['mean']
        values = [f'{m[k]*100:.2f}' for k in ('P_pad', 'R_core', 'F1_padP_coreR', 'eventPrecision', 'eventRecall', 'eventF1')]
        values += [f"{m['completeLosses']:.2f}"]
        values += [f'{m[k]/60:.2f}' for k in ('paddedModelExportSeconds', 'correctlyRemovedSeconds', 'incorrectExportSeconds', 'wantedExportOmittedSeconds')]
        lines.append('| '+key+' | '+' | '.join(values)+' |')
    lines += ['', '## Paired screens', '', '| Candidate | Control | F1 delta pp | Core recall delta pp | Complete misses delta | Recovery screen | F1 screen |',
              '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for key, row in comparisons.items():
        d = row['meanDelta']
        lines.append(f"| {key} | {row['control']} | {100*d['F1_padP_coreR']:+.2f} | {100*d['R_core']:+.2f} | {d['completeLosses']:+.2f} | {row['recoveryPassed']} | {row['f1Passed']} |")
    lines += ['', '## Production complementarity', '',
              'Descriptive coverage at 2-second padding, averaged across seeds. Partial retention is not full recovery; '
              'these counts do not measure any combined system. Exact rally IDs and retained seconds are in the JSON.', '',
              '| Candidate | Production complete misses | Of those: partly retained | Of those: fully retained | New complete misses versus production |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key, row in complementarity.items():
        m = row['mean']
        lines.append(f"| {key} | {row['productionCompleteMisses']} | {m['productionMissesRetainedPartly']:.2f} | {m['productionMissesRetainedFully']:.2f} | {m['newCompleteLosses']:.2f} |")
    lines += ['', '## Padding sensitivity', '', '| Model | Padding s | P_pad % | R_core % | F1_padP_coreR % | Model export s | Human export s | Delta s |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key in ranked:
        for p in summary[key]['padding']:
            lines.append(f"| {key} | {p['paddingSecondsBeforeAndAfter']:.0f} | {100*p['P_pad']:.2f} | {100*p['R_core']:.2f} | {100*p['F1_padP_coreR']:.2f} | {p['paddedModelExportSeconds']:.3f} | {p['paddedHumanExportSeconds']:.3f} | {p['exportDurationDifferenceSeconds']:+.3f} |")
    markdown = args.output.with_suffix('.md')
    if markdown.exists():
        raise FileExistsError(markdown)
    markdown.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summary': str(args.output), 'markdown': str(markdown), 'models': ranked}))


if __name__ == '__main__':
    main()
