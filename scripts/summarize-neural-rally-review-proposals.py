#!/usr/bin/env python3
"""Summarize every completed, audited rally-preserving review arm without selection."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from numbers import Real
from pathlib import Path
from statistics import mean, pstdev


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0092'))
DOC = REPO/'docs/research/neural-rally-review-proposals-results-2026-09-19.md'
DURATION_SUM_FIELDS = (
    'paddedModelExportSeconds', 'paddedHumanExportSeconds', 'evaluableVideoSeconds',
    'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds',
    'missedCoreSeconds', 'coreHumanSeconds', 'paddedIntersectionSeconds', 'coreIntersectionSeconds',
)
DURATION_FIELDS = ('P_pad', 'R_core', 'F1_padP_coreR', *DURATION_SUM_FIELDS, 'exportDurationDifferenceSeconds')
WORKLOAD_FIELDS = (
    'budgetSeconds', 'reviewSeconds', 'editSeconds', 'reviewClips', 'decisionRegions',
    'proposalsSelected', 'proposalsAvailable', 'censoredStarts', 'censoredEnds',
    'unobservedStarts', 'unobservedEnds', 'touchedRalliesWithUneditableBoundary',
    'reviewedTrueRallies', 'playbackTrueRallies', 'reviewFractionOfVideo', 'budgetUtilization',
)
IDENTITY_DELTA_FIELDS = (
    'eventPrecision', 'eventRecall', 'eventF1', 'trueRallies', 'predictedRallies', 'matchedRallies',
    'falsePositiveRallies', 'falseNegativeRallies', 'completeMisses', 'mergedPredictions',
    'splitTrueRallies', 'mergedPredictionsMaterial', 'splitTrueRalliesMaterial',
    'unobservedPredictionStarts', 'unobservedPredictionEnds',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(reference):
    require(sha(reference['path']) == reference['sha256'], 'Bound artifact changed: '+reference['path'])
    require(Path(reference['path']).stat().st_size == reference['sizeBytes'], 'Bound artifact size differs')
    return read(reference['path'])


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def at(rows, padding=2):
    selected = [row for row in rows if row['paddingSecondsBeforeAndAfter'] == padding]
    require(len(selected) == 1, 'Padding case missing or repeated')
    return selected[0]


def pool_duration(rows):
    """Pool per-recording sufficient counts before computing each seed's rates."""
    values = {key: sum(row[key] for row in rows) for key in DURATION_SUM_FIELDS}
    p = values['paddedIntersectionSeconds']/values['paddedModelExportSeconds'] if values['paddedModelExportSeconds'] else 0.
    r = values['coreIntersectionSeconds']/values['coreHumanSeconds'] if values['coreHumanSeconds'] else 0.
    values.update(P_pad=p, R_core=r, F1_padP_coreR=2*p*r/(p+r) if p+r else 0.,
                  exportDurationDifferenceSeconds=values['paddedModelExportSeconds']-values['paddedHumanExportSeconds'])
    return values


def statistics(values):
    available = [float(value) for value in values if value is not None]
    require(all(value is None or isinstance(value, Real) and not isinstance(value, bool) for value in values), 'Non-numeric statistic')
    return {'mean': mean(available) if available else None,
            'min': min(available) if available else None, 'max': max(available) if available else None,
            'seedPopulationStddev': pstdev(available) if available else None,
            'availableSeeds': len(available), 'totalSeeds': len(values)}


def numeric_tree(value):
    """Retain reported numeric summaries, excluding raw error vectors/matches."""
    if isinstance(value, dict):
        output = {key: numeric_tree(child) for key, child in value.items()
                  if isinstance(child, dict) or child is None or isinstance(child, Real) and not isinstance(child, bool)}
        return output
    require(value is None or isinstance(value, Real) and not isinstance(value, bool), 'Unexpected tree leaf')
    return value


def tree_statistics(rows):
    require(bool(rows), 'Empty seed statistics')
    if isinstance(rows[0], dict):
        keys = set(rows[0])
        require(all(isinstance(row, dict) and set(row) == keys for row in rows), 'Seed metric shape differs')
        return {key: tree_statistics([row[key] for row in rows]) for key in rows[0]}
    return statistics(rows)


def tree_means(stats):
    if set(stats) == {'mean', 'min', 'max', 'seedPopulationStddev', 'availableSeeds', 'totalSeeds'}:
        return stats['mean']
    return {key: tree_means(child) for key, child in stats.items()}


def summarize_evaluations(evaluations, padding_cases, group=None):
    padding = []
    for pad in padding_cases:
        samples = []
        for evaluation in evaluations:
            raw = at(evaluation['durationMetrics'], pad)
            samples.append(pool_duration([row for row in raw['perRecording'] if row['sourceGroup'] == group])
                           if group is not None else {key: raw[key] for key in DURATION_FIELDS})
        stats = tree_statistics(samples)
        padding.append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3,
                        'metrics': tree_means(stats), 'seedStatistics': stats})
    identities = [numeric_tree(evaluation['identityMetrics']['pooled'] if group is None
                               else evaluation['identityMetrics']['sourceGroups'][group]) for evaluation in evaluations]
    identity_stats = tree_statistics(identities)
    return {'primary': at(padding)['metrics'], 'padding': padding,
            'identity': tree_means(identity_stats), 'identitySeedStatistics': identity_stats}


def delta(value, baseline):
    return value-baseline if value is not None and baseline is not None else None


def add_deltas_and_screen(output, baseline):
    identity, original = output['identity'], baseline['identity']
    changes = {key: delta(identity[key], original[key]) for key in IDENTITY_DELTA_FIELDS}
    for scope, prefix in (('observedStartLocalization', 'observedStart1s'), ('observedEndLocalization', 'observedEnd1s'),
                          ('startLocalization', 'allStart1s'), ('endLocalization', 'allEnd1s')):
        for key in ('precision', 'recall', 'f1', 'matched', 'predicted'):
            changes[prefix+key[0].upper()+key[1:]] = delta(identity[scope]['1'][key], original[scope]['1'][key])
    output['identityDeltas'] = changes
    output['primaryDeltas'] = {key: delta(output['primary'][key], baseline['primary'][key]) for key in DURATION_FIELDS}
    criteria = {
        'eventF1NonWorsening': changes['eventF1'] is not None and changes['eventF1'] >= -1e-12,
        'observedStart1sF1NonWorsening': changes['observedStart1sF1'] is not None and changes['observedStart1sF1'] >= -1e-12,
        'observedStart1sRecallNonWorsening': changes['observedStart1sRecall'] is not None and changes['observedStart1sRecall'] >= -1e-12,
        'completeMissesNonIncreasing': changes['completeMisses'] <= 1e-12,
    }
    output['guardrailScreen'] = {'passed': all(criteria.values()), 'criteria': criteria,
                                'interpretation': 'Descriptive non-worsening screen against the same automatic baseline; not score-tracking validation'}


def check_evaluation(evaluation):
    require(evaluation['durationAudit']['passed'] and evaluation['durationAudit']['scopeCount'] == 13
            and evaluation['durationAudit']['paddingCases'] == 4, 'Duration audit incomplete')
    require(evaluation['identityAudit']['passed'], 'Identity audit failed')
    require([row['paddingSecondsBeforeAndAfter'] for row in evaluation['durationMetrics']] == [0, 1, 2, 3], 'Padding scope differs')
    require(all(row['joinGapSeconds'] == 3 for row in evaluation['durationMetrics']), 'Join contract differs')


def aggregate(contract, cells, references):
    """Aggregate completed cells. Metadata/hashes are validated by main()."""
    by_config = defaultdict(dict)
    baseline_samples = defaultdict(dict)
    for cell in cells:
        config, seed = cell['configuration'], cell['seed']
        require(seed in contract['seeds'], 'Unexpected seed')
        require(seed not in by_config[config['id']], 'Repeated configuration/seed')
        by_config[config['id']][seed] = cell
        baseline_id = 'production' if config['mode'] == 'production' else config['model']
        prior = baseline_samples[baseline_id].get(seed)
        require(prior is None or prior == cell['automatic'], 'Automatic baseline changed across review arms')
        baseline_samples[baseline_id][seed] = cell['automatic']
    baselines = []
    for name, rows in sorted(baseline_samples.items()):
        require(set(rows) == set(contract['seeds']), 'Baseline seed scope differs')
        ordered = [rows[seed] for seed in contract['seeds']]
        summary = summarize_evaluations(ordered, contract['paddingCases'])
        baselines.append({'id': name, **summary, 'sourceGroups': {
            group: summarize_evaluations(ordered, contract['paddingCases'], group) for group in contract['sourceGroups']}})
    baseline_lookup = {row['id']: row for row in baselines}
    require(set(by_config) == {config['id'] for config in contract['configurations']}, 'Configuration scope differs')
    arms = []
    for config in contract['configurations']:
        rows = by_config[config['id']]
        require(set(rows) == set(contract['seeds']), 'Configuration seed scope differs')
        baseline_id = 'production' if config['mode'] == 'production' else config['model']
        for fraction in contract['budgetFractions']:
            outcomes = []
            for seed in contract['seeds']:
                matches = [row for row in rows[seed]['outcomes'] if row['budgetFraction'] == fraction]
                require(len(matches) == 1, 'Budget outcome missing or duplicated')
                outcomes.append(matches[0])
            arm = {**config, 'id': f"{config['id']}--budget-{round(fraction*100):02d}",
                   'configurationId': config['id'], 'budgetFraction': fraction, 'automaticBaselineId': baseline_id,
                   **summarize_evaluations(outcomes, contract['paddingCases'])}
            work = [{key: row['workload'][key] for key in WORKLOAD_FIELDS} for row in outcomes]
            for row in work:
                row['unusedBudgetSeconds'] = row['budgetSeconds']-row['reviewSeconds']
            arm['workloadSeedStatistics'] = tree_statistics(work)
            arm['workload'] = tree_means(arm['workloadSeedStatistics'])
            arm['sourceGroups'] = {}
            for group in contract['sourceGroups']:
                group_summary = summarize_evaluations(outcomes, contract['paddingCases'], group)
                group_work = [{key: row['workloadBySourceGroup'][group][key] for key in WORKLOAD_FIELDS} for row in outcomes]
                for row in group_work:
                    row['unusedBudgetSeconds'] = row['budgetSeconds']-row['reviewSeconds']
                group_summary['workloadSeedStatistics'] = tree_statistics(group_work)
                group_summary['workload'] = tree_means(group_summary['workloadSeedStatistics'])
                add_deltas_and_screen(group_summary, baseline_lookup[baseline_id]['sourceGroups'][group])
                arm['sourceGroups'][group] = group_summary
            add_deltas_and_screen(arm, baseline_lookup[baseline_id])
            arm['seedResults'] = [{**references[(config['id'], seed)], 'seed': seed, 'budgetFraction': fraction}
                                  for seed in contract['seeds']]
            arms.append(arm)
    rankings = {}
    for fraction in contract['budgetFractions']:
        for mode in ('production', 'individual'):
            selected = [row for row in arms if row['budgetFraction'] == fraction and row['mode'] == mode]
            selected.sort(key=lambda row: (-row['primary']['F1_padP_coreR'], row['workload']['reviewSeconds'], row['id']))
            rankings[f'{mode}--budget-{round(fraction*100):02d}'] = [row['id'] for row in selected]
    return {'automaticBaselines': baselines, 'arms': arms, 'rankingsByDeclaredBudgetAndMode': rankings}


def percent(value):
    return 'n/a' if value is None else f'{100*value:.2f}'


def number(value):
    return 'n/a' if value is None else f'{value:.2f}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def quality_row(row):
    p, e = row['primary'], row['identity']
    s = e['observedStartLocalization']['1']
    w = row.get('workload', {})
    return [row['id'], *[percent(p[key]) for key in ('P_pad', 'R_core', 'F1_padP_coreR')],
            *[percent(e[key]) for key in ('eventPrecision', 'eventRecall', 'eventF1')],
            *[percent(s[key]) for key in ('precision', 'recall', 'f1')],
            number(w.get('reviewSeconds', 0)/60), number(w.get('proposalsSelected', 0)),
            number(w.get('reviewedTrueRallies', 0)), 'yes' if row.get('guardrailScreen', {}).get('passed') else ('baseline' if not w else 'no')]


QUALITY_HEADERS = ['Arm', 'P_pad %', 'R_core %', 'F1_padP_coreR %', 'Event P %', 'Event R %', 'Event F1 %',
                   'Observed start P @1s %', 'Observed start R @1s %', 'Observed start F1 @1s %',
                   'Playback min', 'Proposals selected', 'True rallies in edit regions', 'Non-worsening screen']


def render_document(summary):
    lines = [
        '# Rally-preserving review proposal experiment', '',
        '<!-- ROOT EXECUTIVE FINDINGS START -->',
        '<!-- ROOT EXECUTIVE FINDINGS END -->', '',
        'This is a fixed development comparison on eight exact-label grass/indoor videos, four source groups and 322 rallies. '
        'Every listed value averages three seed results after pooling recording-level numerators and denominators within each seed. '
        'Seed replicas do not increase the video count. The neural detectors were held fixed; no learned review head was trained.', '',
        'Export quality uses `F1_padP_coreR` at the declared two-second symmetric padding and strict positive gap <3-second joining. '
        'The other three padding cases are sensitivity results. Raw event identities remain separate even when exports join. '
        'Event P/R/F1 uses general one-to-one IoU>=0.5 matching. Observed start localization at one second is a serve-contact proxy; '
        'synthetic edit-window edges are excluded. This does not evaluate serving side, point winner or reconstructed scores.', '',
        'The human perfectly corrects occupancy and start/end markers only inside selected edit windows, with no oracle endpoints imported '
        'from outside those windows. Playback includes extra context and is charged as union footage at 1x, not measured reviewer labor. '
        'A proposal is a decision range and is not necessarily a rally; true-rally counts deduplicate labeled rally identities intersecting edit regions.', '',
        'Inventory `legacy` uses prior whole-component/five-second flags; `local_events` uses localized disagreement, split/merge and sustained-live '
        'proposals; `local_heads` additionally uses trained start/end heads. `chronological` orders by time and `evidence` uses fixed label-blind '
        'evidence per standalone playback cost. Review caps are per recording and unused budget is reported. '
        'The descriptive screen requires non-worsening mean event F1, observed-start F1 and recall, and no increase in complete rally misses '
        'against the same automatic baseline. Passing is not score-tracking validation.', '',
        '## Automatic baselines', '', table(QUALITY_HEADERS, [quality_row(row) for row in summary['automaticBaselines']]), '',
        '## Every fixed arm at the target padding', '',
    ]
    by_id = {row['id']: row for row in summary['arms']}
    for key, ids in summary['rankingsByDeclaredBudgetAndMode'].items():
        lines.extend([f'### {key}', '', 'Sorted by `F1_padP_coreR` at the same declared cap; compare actual playback as well.', '',
                      table(QUALITY_HEADERS, [quality_row(by_id[identifier]) for identifier in ids]), ''])
    lines += ['## Rally separation and boundary censoring', '',
              'Merge/split counts first use any positive overlap. Material variants require at least min(0.5 seconds, 10% of gold rally duration). '
              'Complete misses have no raw overlap at all; they differ from IoU-based unmatched events. Synthetic/censored boundaries can leave '
              'partial event objects, so event F1 need not improve monotonically as more video is reviewed.', '',
              table(['Arm', 'Raw predictions', 'Complete misses', 'Merged predictions', 'Split gold rallies', 'Material merges', 'Material splits',
                     'Censored starts', 'Censored ends', 'Unobserved starts', 'Unobserved ends', 'Touched rallies with uneditable boundary'],
                    [[row['id'], *[number(row['identity'][key]) for key in ('predictedRallies', 'completeMisses', 'mergedPredictions',
                       'splitTrueRallies', 'mergedPredictionsMaterial', 'splitTrueRalliesMaterial')],
                      *[number(row['workload'][key]) for key in ('censoredStarts', 'censoredEnds', 'unobservedStarts',
                        'unobservedEnds', 'touchedRalliesWithUneditableBoundary')]] for row in summary['arms']]), '',
              '## Changes against the same automatic baseline', '',
              'Quality deltas are percentage points. Miss/merge/split deltas are counts, where a reduction is better.', '',
              table(['Arm', 'Export F1 delta pp', 'Event F1 delta pp', 'Event recall delta pp', 'Observed start F1 delta pp',
                     'Observed start recall delta pp', 'Complete misses delta', 'Material merges delta', 'Material splits delta'],
                    [[row['id'], percent(row['primaryDeltas']['F1_padP_coreR']),
                      *[percent(row['identityDeltas'][key]) for key in ('eventF1', 'eventRecall', 'observedStart1sF1', 'observedStart1sRecall')],
                      *[number(row['identityDeltas'][key]) for key in ('completeMisses', 'mergedPredictionsMaterial', 'splitTrueRalliesMaterial')]]
                     for row in summary['arms']]), '',
              '## Workload and unused budget', '',
              table(['Arm', 'Cap %', 'Actual footage %', 'Playback min', 'Unused cap min', 'Edit min', 'Selected proposals',
                     'Available proposals', 'Edit regions', 'Playback clips', 'True rallies in edit regions', 'True rallies visible'],
                    [[row['id'], percent(row['budgetFraction']), percent(row['workload']['reviewFractionOfVideo']),
                      *[number(row['workload'][key]/60) for key in ('reviewSeconds', 'unusedBudgetSeconds', 'editSeconds')],
                      *[number(row['workload'][key]) for key in ('proposalsSelected', 'proposalsAvailable', 'decisionRegions', 'reviewClips',
                                                               'reviewedTrueRallies', 'playbackTrueRallies')]] for row in summary['arms']]), '',
              '## All four padding cases and export accounting', '',
              'Correctly removed time is valid unwanted footage excluded from export. Incorrectly removed time is wanted padded-human export omitted. '
              'Incorrect export is unwanted footage retained. Model-minus-human duration is signed.', '',
              table(['Arm', 'Pad s', 'P_pad %', 'R_core %', 'F1_padP_coreR %', 'Model export min', 'Human export min', 'Difference min',
                     'Correctly removed min', 'Incorrectly removed min', 'Incorrect export min', 'Missed core s'],
                    [[row['id'], pad['paddingSecondsBeforeAndAfter'],
                      *[percent(pad['metrics'][key]) for key in ('P_pad', 'R_core', 'F1_padP_coreR')],
                      *[number(pad['metrics'][key]/60) for key in ('paddedModelExportSeconds', 'paddedHumanExportSeconds',
                        'exportDurationDifferenceSeconds', 'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds')],
                      number(pad['metrics']['missedCoreSeconds'])]
                     for row in [*summary['automaticBaselines'], *summary['arms']] for pad in row['padding']]), '',
              '## Source-group results at target padding', '',
              table(['Arm', 'Source group', 'P_pad %', 'R_core %', 'F1_padP_coreR %', 'Event F1 %', 'Observed start P @1s %',
                     'Observed start R @1s %', 'Observed start F1 @1s %', 'Playback min', 'True rallies reviewed'],
                    [[row['id'], group, *[percent(data['primary'][key]) for key in ('P_pad', 'R_core', 'F1_padP_coreR')],
                      percent(data['identity']['eventF1']),
                      *[percent(data['identity']['observedStartLocalization']['1'][key]) for key in ('precision', 'recall', 'f1')],
                      number(data['workload']['reviewSeconds']/60), number(data['workload']['reviewedTrueRallies'])]
                     for row in summary['arms'] for group, data in row['sourceGroups'].items()]), '',
              '## Provenance and limitations', '',
              f"Registered contract: `{summary['contractSha256']}`. Complete report SHA-256: `{summary['report']['sha256']}`.", '',
              'Every completed configuration/seed file was hash verified and required passed candidate, queue, editor, duration and identity audits before '
              'aggregation. The JSON summary retains seed mean/min/max/population standard deviation, all 0.25/0.5/1/2-second ordinary and observed '
              'boundary metrics, conditional matched-boundary errors, source-group details and references to every recording-level result.', '',
              'Production weights historically saw these recordings, and neural decoder/model choices are development-exposed. The simulated perfect '
              'reviewer is an optimistic assumption. These results are not fresh protected-test quality, measured reviewer speed, device inference '
              'performance, serving-side accuracy or score reconstruction. Production remains unchanged.', '',
              f"Machine-readable summary: `{summary['summaryPath']}`", '',
    ]
    return '\n'.join(lines)


def main(root, document):
    registration = read(root/'registration.json')
    contract = registration['contract']
    require(canonical(contract) == registration['sha256'], 'Registration hash differs')
    report = read(root/'report.json')
    require(report['status'] == 'completed-fixed-rally-review-proposals' and report['contractSha256'] == registration['sha256'], 'Run incomplete')
    require(report['configurationSeedRuns'] == 108 and report['outcomeCells'] == 432 and len(report['results']) == 108, 'Report matrix incomplete')
    require(report['candidatePlansAudited'] == 864 and report['editedRecordingsAudited'] == 3456, 'Audit totals incomplete')
    for reference in [*contract['sources'].values(), contract['input'], contract['protocol'], contract['qualification'], contract['goldSemantics']]:
        require(sha(reference['path']) == reference['sha256'], 'Registered bytes changed: '+reference['path'])
    require(sha(contract['probabilities']['path']) == contract['probabilities']['sha256'], 'Registered probability bytes changed')
    cells, references = [], {}
    for reference in report['results']:
        cell = bound(reference)
        require(cell['contractSha256'] == registration['sha256'], 'Cell contract differs')
        require(len(cell['plans']) == 8 and len(cell['outcomes']) == 4, 'Cell scope differs')
        require(all(plan['candidateAudit']['passed'] and plan['queueAudit']['passed'] for plan in cell['plans']), 'Candidate/queue audit failed')
        check_evaluation(cell['automatic'])
        for outcome in cell['outcomes']:
            check_evaluation(outcome)
            require(len(outcome['recordings']) == 8 and all(row['editorAudit']['passed'] for row in outcome['recordings']), 'Editor audit incomplete')
        cells.append(cell)
        references[(cell['configuration']['id'], cell['seed'])] = reference
    result = aggregate(contract, cells, references)
    require(len(result['arms']) == 144 and len(result['automaticBaselines']) == 4, 'Aggregated scope differs')
    summary = {'kind': 'fixed-rally-review-proposals-summary-v1', 'createdAt': datetime.now(timezone.utc).isoformat(),
               'contractSha256': registration['sha256'], 'registration': ref(root/'registration.json'), 'report': ref(root/'report.json'),
               'sourceScript': ref(Path(__file__)), 'sourceTests': ref(REPO/'analysis/tests/test_neural_rally_review_summary.py'),
               'summaryPath': str(root/'summary.json'),
               'primaryMetric': contract['primaryMetric'], 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
               'aggregation': 'Pool recording counts within each seed, then average seed metrics; 3 seeds are not extra videos',
               'guardrailScreenDefinition': 'Mean event F1, observed start@1s F1 and recall nondecreasing; mean complete misses nonincreasing vs same automatic baseline',
               'seedStatisticNullRule': 'Unavailable conditional metrics average available seeds only; availableSeeds and totalSeeds reported',
               'metricContract': cells[0]['automatic']['identityMetrics']['metricContract'], **result}
    with (root/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False); stream.write('\n')
    with document.open('x', encoding='utf-8') as stream:
        stream.write(render_document(summary))
    print(json.dumps({'completed': True, 'summary': ref(root/'summary.json'), 'document': ref(document),
                      'arms': len(result['arms']), 'baselines': len(result['automaticBaselines'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--document', type=Path, default=DOC)
    args = parser.parse_args()
    main(args.root, args.document)
