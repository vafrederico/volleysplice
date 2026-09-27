#!/usr/bin/env python3
"""Independent aggregate, report and candidate/output distinction audit."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0079'))
DOC = REPO/'docs/research/neural-typed-boundary-results-2026-09-19.md'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, REPO/path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


a = module('scripts/audit-neural-split-summary.py', 'independent_seed_statistics')
b = module('scripts/audit-neural-boundary-advisor.py', 'independent_boundary_primitives')
require, compare, read, ref = a.require, a.compare, a.read, a.ref
WORKLOAD = ('reviewSeconds', 'budgetSeconds', 'unusedBudgetSeconds', 'reviewJobs', 'jobsAvailable', 'reviewClips',
            'editRegions', 'boundaryFlagsReviewed', 'candidatesReviewed', 'reviewedTrueRallies', 'playbackTrueRallies',
            'eventTimelineRemovedSeconds', 'eventTimelineAddedSeconds')
ACTIONS = ('reviewedCandidates', 'acceptedCount', 'rejectedCount', 'correctedTypeCount', 'editedParents')
COVERAGE = ('coreHumanSeconds', 'baselineRawCoreCoveredSeconds', 'resultRawCoreCoveredSeconds',
            'baselineRawCoreRecall', 'resultRawCoreRecall', 'rawCoreSecondsLostFromBaseline',
            'rawCoreSecondsAddedToBaseline', 'rawSelectedSecondsLostFromBaseline', 'rawSelectedSecondsAddedToBaseline',
            'baselineCompleteMisses', 'resultCompleteMisses', 'additionalCompleteMisses', 'retainedCompleteMisses', 'recoveredCompleteMisses')


def raw_scope(metric, kind, key):
    if kind == 'pooled':
        return metric['pooled']
    if kind == 'group':
        return metric['sourceGroups'][key]
    found = [row for row in metric['recordings'] if row['id'] == key]
    require(len(found) == 1, 'Metric recording scope differs')
    return found[0]


def verify_scope(evaluations, summary, ids, kind, key=None):
    leaves = 0
    require([r['paddingSecondsBeforeAndAfter'] for r in summary['padding']] == [0, 1, 2, 3], 'Summary padding differs')
    for pad, output in enumerate(summary['padding']):
        samples = []
        require(output['joinGapSeconds'] == 3, 'Summary gap join differs')
        for e in evaluations:
            found = [r for r in e['durationMetrics'] if r['paddingSecondsBeforeAndAfter'] == pad]
            require(len(found) == 1 and found[0]['joinGapSeconds'] == 3, 'Raw pad configuration differs')
            selected = [r for r in found[0]['perRecording'] if r['id'] in ids]
            require({r['id'] for r in selected} == set(ids), 'Duration scope differs')
            sample = a.pooled_duration(selected)
            if kind == 'pooled':
                compare(sample, {k: found[0][k] for k in sample}, 'Raw duration pooling')
            samples.append(sample)
        leaves += a.verify_statistics(samples, output['metrics'], output['seedStatistics'], f'{kind}:{key}:pad{pad}')
    compare(summary['padding'][2]['metrics'], summary['primary'], 'Primary padded row')
    for source, name in [('identityMetrics', 'identity'), ('typedMetrics', 'typed'), ('rawCoverageMetrics', 'legacyIdentityCoverage')]:
        samples = [raw_scope(e[source], kind, key) for e in evaluations]
        leaves += a.verify_statistics(samples, summary[name], summary[name + 'SeedStatistics'], f'{kind}:{key}:{name}')
    compare({k: summary['typed'][k] for k in COVERAGE}, summary['coverage'], 'Physical coverage alias')
    compare({k: summary['typedSeedStatistics'][k] for k in COVERAGE}, summary['coverageSeedStatistics'], 'Coverage statistics alias')
    work, actions = [], []
    for evaluation in evaluations:
        if 'workload' not in evaluation:
            work.append({k: 0 for k in WORKLOAD}); actions.append({k: None for k in ACTIONS}); continue
        selected = [r for r in evaluation['perRecording'] if r['id'] in ids]
        require({r['id'] for r in selected} == set(ids), 'Review workload scope differs')
        sample = {k: math.fsum(r['workload'][k] for r in selected) for k in WORKLOAD}
        if kind == 'pooled':
            compare(sample, evaluation['workload'], 'Raw workload pooling')
        work.append(sample)
        actions.append({k: math.fsum(r[k] for r in selected) if all(k in r for r in selected) else None for k in ACTIONS})
    leaves += a.verify_statistics(work, summary['workload'], summary['workloadSeedStatistics'], f'{kind}:{key}:workload')
    leaves += a.verify_statistics(actions, summary['humanActions'], summary['humanActionsSeedStatistics'], f'{kind}:{key}:actions')
    return leaves


def audit_raw_semantics(contract, cells, data):
    source = {r['id']: r for r in data['records']}
    auto_count = review_count = candidate_rows = 0
    for cell in cells:
        a.assert_audits(cell)
        require(cell['baseline'] == cells[0]['baseline'], 'Baseline differs across seeds')
        plans = {p['id']: p['policies'] for p in cell['plans']}
        require(set(plans) == set(source), 'Plan recording inventory differs')
        for row in [*cell['automatic'], *cell['reviewed']]:
            require(row['durationMetrics'] == cell['baseline']['durationMetrics'], 'Fixed export changed')
            automatic = row['id'].startswith('automatic--')
            per = {} if automatic else {r['id']: r for r in row['perRecording']}
            for metric in row['typedMetrics']['recordings']:
                rid = metric['id']; plan = plans[rid][row['policy']]
                if automatic:
                    candidates, events = plan['eventCandidates'], plan['events']
                else:
                    selected = set(per[rid]['selectedParentIds'])
                    candidates = [c for c in plan['eventCandidates'] if c['parentId'] in selected]
                    events = per[rid]['events']
                    require(len(candidates) == per[rid]['workload']['candidatesReviewed'], 'Reviewed candidate count differs')
                expected = b.normalized_candidates(source[rid], candidates)
                compare(expected, metric['proposals'], 'Original candidate metric provenance')
                coverage = b.raw_coverage(source[rid], events)
                for key in COVERAGE:
                    if key == 'retainedCompleteMisses':
                        value = coverage['resultCompleteMisses'] - coverage['additionalCompleteMisses']
                    else:
                        value = coverage[key]
                    compare(value, metric[key], 'Raw physical coverage.' + key)
                if not automatic and row['humanMode'] == 'full_parent':
                    require(coverage['rawCoreSecondsLostFromBaseline'] <= 1e-8
                            and coverage['additionalCompleteMisses'] == 0, 'Full-parent review lost core/rally')
                    require(all(k not in per[rid] for k in ACTIONS), 'Full-parent falsely claims proposal acceptance')
                candidate_rows += 1
            if automatic:
                auto_count += 1
            else:
                review_count += 1
                counterpart = next(r for r in cell['reviewed'] if r['policy'] == row['policy']
                    and r['ranker'] == row['ranker'] and r['budgetFraction'] == row['budgetFraction']
                    and r['humanMode'] != row['humanMode'])
                for first, second in zip(row['perRecording'], counterpart['perRecording']):
                    require(first['id'] == second['id'] and first['selectedParentIds'] == second['selectedParentIds'],
                            'Human-mode comparison changed review selections')
    require(auto_count == contract['automaticOutcomes'] == 12, 'Automatic outcome count differs')
    require(review_count == contract['reviewedOutcomes'] == 192, 'Reviewed outcome count differs')
    return {'passed': True, 'automaticOutcomes': auto_count, 'reviewedOutcomes': review_count,
            'candidateProvenanceAndPhysicalCoverageRows': candidate_rows,
            'reviewedTypedMetricsAreOriginalCandidates': True, 'humanModesHaveIdenticalReviewSelections': True,
            'fixedExportsAllFourPads': True, 'onlyFullParentModeRequiresZeroCoreLoss': True}


def audit_aggregate(contract, cells, summary):
    require(len(contract['seeds']) == len(set(contract['seeds'])) == 3, 'Seed contract differs')
    require(sorted(c['seed'] for c in cells) == sorted(contract['seeds']), 'Raw seeds differ')
    require(summary['seeds'] == contract['seeds'], 'Summary seed order differs')
    expected = {'production': {'kind': 'baseline'}}
    expected.update({'automatic--' + p: {'kind': 'automatic', 'policy': p} for p in contract['policies']})
    for policy in contract['policies']:
        for mode in contract['humanModes']:
            for ranker in contract['rankers']:
                for fraction in contract['budgetFractions']:
                    identity = f'{policy}--{mode}--{ranker}--budget-{round(100*fraction):02d}'
                    expected[identity] = {'kind': 'reviewed', 'policy': policy, 'humanMode': mode,
                                          'ranker': ranker, 'budgetFraction': fraction}
    arms = [summary['baseline'], *summary['automatic'], *summary['reviewed']]
    require(len(arms) == len(expected) == 69 and {x['id'] for x in arms} == set(expected), 'Aggregate arm scope differs')
    raw_by_arm = {key: [] for key in expected}
    for cell in cells:
        rows = [('production', cell['baseline']), *[(r['id'], r) for r in [*cell['automatic'], *cell['reviewed']]]]
        require(len(rows) == len(expected) and {key for key, _ in rows} == set(expected), 'Raw cell arm scope differs')
        for key, row in rows:
            raw_by_arm[key].append(row)
            for field, value in expected[key].items():
                if field != 'kind':
                    require(row[field] == value, 'Raw arm metadata differs')
    baseline = cells[0]['baseline']
    groups = sorted(baseline['identityMetrics']['sourceGroups'])
    recording_ids = sorted(r['id'] for r in baseline['identityMetrics']['recordings'])
    require(summary['sourceGroups'] == groups and summary['recordings'] == recording_ids, 'Summary scope differs')
    group_ids = {g: [r['id'] for r in baseline['identityMetrics']['recordings'] if r['sourceGroup'] == g] for g in groups}
    leaves = 0
    for arm in arms:
        identity = arm['id']; meta = expected[identity]
        compare(meta, {key: arm[key] for key in meta}, identity + '.metadata')
        require(set(arm['sourceGroups']) == set(groups) and set(arm['recordings']) == set(recording_ids), 'Arm slice scope differs')
        rows = raw_by_arm[identity]
        leaves += verify_scope(rows, arm, recording_ids, 'pooled')
        for group in groups:
            leaves += verify_scope(rows, arm['sourceGroups'][group], group_ids[group], 'group', group)
        for recording in recording_ids:
            leaves += verify_scope(rows, arm['recordings'][recording], [recording], 'recording', recording)
    compare({'metric': 'F1_padP_coreR', 'targetPaddingSeconds': 2, 'result': 'all arms tie exactly with production'},
            summary['primaryRanking'], 'Primary ranking')
    compare({'fixedExportAllFourPads': True, 'allRegisteredArms': True, 'fullParentCorePreserved': True,
             'automaticAndProposalOnlyCoreLossAllowedAndReported': True}, summary['verification'], 'Verification claims')
    for key, value in {'automaticOutcomes': 12, 'reviewedOutcomes': 192, 'automaticArms': 4,
                       'reviewedArms': 64, 'allArmsIncludingBaseline': 69}.items():
        compare(value, summary[key], key)
    require('NOT corrected human output' in summary['typedMetricInterpretation'], 'Candidate metric distinction missing')
    require('including valid portions of ignored-touched gold' in summary['physicalCoverageInterpretation'], 'Physical core scope missing')
    return {'passed': True, 'aggregateArms': len(arms), 'seedCells': len(cells),
            'scopeSummaries': len(arms)*(1 + len(groups) + len(recording_ids)),
            'paddingRows': len(arms)*(1 + len(groups) + len(recording_ids))*4, 'seedStatisticLeaves': leaves}


QUALITY = ['Arm', 'Rally P %', 'Rally R %', 'Rally F1 %', 'Observed start R @1s %', 'Observed end R @1s %',
           'Raw core R %', 'Raw core lost s', 'Complete misses', 'Additional misses']


def quality_row(label, arm, review=False):
    identity, coverage, work = arm['identitySeedStatistics'], arm['coverageSeedStatistics'], arm['workloadSeedStatistics']
    rd = a.range_display
    row = [label, *[rd(identity[k], 100.) for k in ('eventPrecision', 'eventRecall', 'eventF1')],
           rd(identity['observedStartLocalization']['1']['recall'], 100.),
           rd(identity['observedEndLocalization']['1']['recall'], 100.), rd(coverage['resultRawCoreRecall'], 100.),
           rd(coverage['rawCoreSecondsLostFromBaseline']), rd(identity['completeMisses']), rd(coverage['additionalCompleteMisses'])]
    if review:
        row += [rd(work['reviewSeconds'], 1/60), rd(work['reviewJobs']), rd(work['reviewedTrueRallies'])]
    return row


def audit_tables(summary, document):
    base = summary['baseline']; automatic = summary['automatic']; reviewed = summary['reviewed']; rd = a.range_display
    expected = a.table_lines(['Pad each side s', 'P_pad %', 'R_core %', 'F1_padP_coreR %', 'Model export min',
                             'Human export min', 'Difference min', 'Correct removed min', 'Incorrect removed min', 'Incorrect export min'],
        [[p['paddingSecondsBeforeAndAfter'], *[a.display(p['metrics'][k], 100.) for k in ('P_pad', 'R_core', 'F1_padP_coreR')],
          *[a.display(p['metrics'][k], 1/60) for k in ('paddedModelExportSeconds', 'paddedHumanExportSeconds', 'exportDurationDifferenceSeconds',
           'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds')]] for p in base['padding']])
    expected += a.table_lines(QUALITY, [quality_row(row['id'], row) for row in [base, *automatic]])
    expected += a.table_lines(['Policy', 'Type', 'Tolerance s', 'True targets', 'Candidates with observed start', 'Matched', 'Typed P %', 'Typed R %', 'Typed F1 %'],
        [[row['policy'], kind, tolerance,
          *[rd(row['typedSeedStatistics']['byType'][kind][tolerance][k]) for k in ('true', 'predicted', 'matched')],
          *[rd(row['typedSeedStatistics']['byType'][kind][tolerance][k], 100.) for k in ('precision', 'recall', 'f1')]]
         for row in automatic for kind in ('initial_start', 'additional_start') for tolerance in ('0.5', '1', '2')])
    expected += a.table_lines(['Policy', 'Wrong type @1s', 'Initial proposed for additional', 'Additional proposed for initial',
                             'Joint start/end F1 @1s %', 'Unobserved starts', 'Unobserved ends'],
        [[row['policy'], *[rd(row['typedSeedStatistics']['typeDiagnostics']['1'][k]) for k in
          ('wrongType', 'initialProposedForAdditional', 'additionalProposedForInitial')],
          rd(row['typedSeedStatistics']['samePairBoundaries']['1']['f1'], 100.),
          rd(row['typedSeedStatistics']['unobservedCandidateStarts']), rd(row['typedSeedStatistics']['unobservedCandidateEnds'])] for row in automatic])
    for mode in ('proposal_confirmation', 'full_parent'):
        expected += a.table_lines(QUALITY + ['Review min', 'Parent jobs', 'Real rallies reviewed'],
                                 [quality_row(row['id'], row, True) for row in reviewed if row['humanMode'] == mode])
    expected += a.table_lines(['Arm', 'Flags reviewed', 'Candidates reviewed', 'Accepted candidates', 'Rejected candidates',
                              'Corrected types', 'Original wrong types @1s', 'Original joint F1 @1s %', 'Unused budget min'],
        [[row['id'], rd(row['workloadSeedStatistics']['boundaryFlagsReviewed']), rd(row['workloadSeedStatistics']['candidatesReviewed']),
          *[rd(row['humanActionsSeedStatistics'][k]) for k in ('acceptedCount', 'rejectedCount', 'correctedTypeCount')],
          rd(row['typedSeedStatistics']['typeDiagnostics']['1']['wrongType']),
          rd(row['typedSeedStatistics']['samePairBoundaries']['1']['f1'], 100.),
          rd(row['workloadSeedStatistics']['unusedBudgetSeconds'], 1/60)] for row in reviewed])
    for group in summary['sourceGroups']:
        selected = [base, *automatic, *[row for row in reviewed if row['ranker'] == 'evidence' and row['budgetFraction'] == .1]]
        expected += a.table_lines(QUALITY + ['Review min', 'Parent jobs', 'Real rallies reviewed'],
                                 [quality_row(row['id'], row['sourceGroups'][group], True) for row in selected])
    require('## Fixed export accounting' in document, 'Generated report body missing')
    body = document.split('## Fixed export accounting', 1)[1]
    actual = [line for line in body.splitlines() if line.startswith('|')]
    require(len(actual) == len(expected), 'Generated table row count differs')
    for i, (wanted, emitted) in enumerate(zip(expected, actual)):
        require(wanted == emitted, f'Generated table row {i} differs')
    for phrase in ('before any human correction', 'not gold-corrected output', 'may remove an unproposed rally',
                   'not proposal-only confirmation', 'not measured human labor', 'development estimates'):
        require(phrase in document, 'Report interpretation limit missing: ' + phrase)
    return {'passed': True, 'generatedTableLinesVerified': len(expected), 'candidateAndHumanMetricDistinctionPresent': True,
            'fullParentAndProposalConfirmationSeparated': True, 'executiveProseExcludedFromThisAudit': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--doc', type=Path, default=DOC); args = parser.parse_args()
    root = args.root
    require((root/'report.json').exists() and (root/'summary.json').exists(), 'Completed report and summary required')
    reg = read(root/'registration.json'); contract = reg['contract']
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(canonical == reg['sha256'], 'Registration changed')
    references = [*contract['sources'].values(), *contract['sourceCopies'].values(), contract['input'],
                  contract['probabilities'], contract['protocol'], contract['qualification'], contract['priorRegistration']]
    for item in references:
        a.verify_ref(item)
    report = read(root/'report.json'); require(report['passed'] and report['contractSha256'] == canonical, 'Run not complete')
    cells = []
    for row in report['seeds']:
        a.verify_ref(row['result']); cells.append(read(row['result']['path']))
        require(cells[-1]['contractSha256'] == canonical and cells[-1]['seed'] == row['seed'], 'Raw cell identity differs')
    summary = read(root/'summary.json'); require(summary['contractSha256'] == canonical, 'Summary contract differs')
    for item in [summary['registration'], summary['report'], summary['summarizer'], summary['utilitySource'], *summary['inputs']]:
        a.verify_ref(item)
    semantics = audit_raw_semantics(contract, cells, read(contract['input']['path']))
    aggregation = audit_aggregate(contract, cells, summary)
    tables = audit_tables(summary, args.doc.read_text(encoding='utf-8'))
    for item in references:
        a.verify_ref(item)
    receipt = {'passed': True, 'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': canonical,
        'registration': ref(root/'registration.json'), 'report': ref(root/'report.json'), 'summary': ref(root/'summary.json'),
        'document': ref(args.doc), 'auditor': ref(Path(__file__)),
        'independentUtilities': [ref(REPO/'scripts/audit-neural-split-summary.py'), ref(REPO/'scripts/audit-neural-boundary-advisor.py')],
        'frozenSourcesVerified': len(contract['sources']), 'frozenSourceCopiesVerified': len(contract['sourceCopies']),
        'rawSemanticsAudit': semantics, 'aggregateAudit': aggregation, 'generatedTableAudit': tables}
    path = root/'summary-audit-v1.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': ref(path), **aggregation, **tables}))


if __name__ == '__main__':
    main()
