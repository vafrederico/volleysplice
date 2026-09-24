#!/usr/bin/env python3
"""Independent interval arithmetic and strict-selection audit for recall sweeps.

Shares frozen decoding and input loading, not candidate metric arithmetic or
selection logic. Event matching additionally replays the canonical evaluator.
No training, probability regeneration or GPU work.
"""
from pathlib import Path
import argparse
import importlib.util
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as io
from analysis import neural_development as base, neural_expanded_development as expanded
from analysis.neural_evaluation import evaluate_predictions
from analysis.neural_recall_sweep_adapters import bind, load_score_shards

HELPER_SHA = '8ed1ed75ebc2511c9603642c0b09ba5888554910072a6e1d1c5490a65864b81b'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def audit_export(helper, rows, evaluation):
    io.require(evaluation['eventMetricsAvailable'] is False and evaluation['rallyCoreMetricsAvailable'] is False
               and 'guardrails' not in evaluation and 'R_core' not in evaluation['primary'], 'Export labels misrepresented as core/event truth')
    expected_padding = []
    for pad in (0, 1, 2, 3):
        sums = dict(modelExportSeconds=0., humanExportSeconds=0., intersectionSeconds=0., evaluableVideoSeconds=0.)
        for row in rows:
            parsed = helper.parse_record({**row, 'rallies': row['humanExportIntervals']})
            ignored = helper.ranges(parsed['ignoredIntervals'])
            human = helper.boolean_intervals(helper.ranges(parsed['rallies']), ignored, 'difference')
            model = helper.export_union(parsed, 'predictions', pad)
            sums['modelExportSeconds'] += helper.duration(model)
            sums['humanExportSeconds'] += helper.duration(human)
            sums['intersectionSeconds'] += helper.duration(helper.boolean_intervals(model, human, 'intersection'))
            sums['evaluableVideoSeconds'] += helper.duration(helper.boolean_intervals([(0., parsed['duration'])], ignored, 'difference'))
        p = sums['intersectionSeconds'] / sums['modelExportSeconds'] if sums['modelExportSeconds'] else 0.
        r = sums['intersectionSeconds'] / sums['humanExportSeconds']
        expected = {**sums, 'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3., 'humanPaddingSeconds': 0,
                    'humanJoinGapSeconds': 0, 'P_export': p, 'R_export': r, 'F1_export': 2*p*r/(p+r) if p+r else 0.,
                    'exportDurationDifferenceSeconds': sums['modelExportSeconds']-sums['humanExportSeconds'],
                    'incorrectExportSeconds': sums['modelExportSeconds']-sums['intersectionSeconds'],
                    'wantedExportOmittedSeconds': sums['humanExportSeconds']-sums['intersectionSeconds'],
                    'correctlyRemovedSeconds': sums['evaluableVideoSeconds']-sums['humanExportSeconds']-sums['modelExportSeconds']+sums['intersectionSeconds']}
        io.require(set(expected) == set(evaluation['padding'][pad]), 'Export metric schema differs')
        for key, value in expected.items():
            helper.close(value, evaluation['padding'][pad][key], 'export/' + key)
        expected_padding.append(expected)
    io.require(len(evaluation['padding']) == 4 and evaluation['primary'] == evaluation['padding'][2], 'Export padding/primary differs')
    io.require(evaluation['recordingCount'] == len(rows) and evaluation['sourceGroupCount'] == len({r['sourceGroup'] for r in rows}),
               'Export metric source inventory differs')


def check_evaluation(helper, rows, evaluation, policy):
    if policy == 'exact-rallies':
        helper.compare_scope([helper.parse_record(row) for row in rows], evaluation, 'sweep')
        io.require(evaluate_predictions(rows) == evaluation, 'Full canonical event/source metric replay differs')
    elif policy == 'reviewed-export':
        audit_export(helper, rows, evaluation)
    else:
        io.require(policy == 'reviewed-draft' and evaluation['eventMetricsAvailable'] is False
                   and evaluation['rallyCoreMetricsAvailable'] is False and 'guardrails' not in evaluation,
                   'Reviewed-draft labels misrepresented as exact rally events')
        parsed = [helper.parse_record({**row, 'rallies': row['reviewedLiveIntervals']}) for row in rows]
        universe = sum(helper.duration(helper.boolean_intervals([(0., r['duration'])], helper.ranges(r['ignoredIntervals']), 'difference')) for r in parsed)
        for pad in (0, 1, 2, 3):
            metric = helper.pooled_metric(parsed, pad)
            incorrect = metric['paddedModelExportSeconds'] - metric['paddedPrecisionIntersectionSeconds']
            expected = {'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3.,
                'P_reviewed': metric['P_pad'], 'R_reviewed': metric['R_core'], 'F1_reviewed': metric['F1_padP_coreR'],
                'modelExportSeconds': metric['paddedModelExportSeconds'], 'humanExportSeconds': metric['paddedHumanExportSeconds'],
                'reviewedLiveSeconds': metric['coreHumanSeconds'], 'retainedReviewedLiveSeconds': metric['coreRecallIntersectionSeconds'],
                'exportDurationDifferenceSeconds': metric['exportDurationDifferenceSeconds'], 'incorrectExportSeconds': incorrect,
                'wantedExportOmittedSeconds': metric['paddedHumanExportSeconds'] - metric['paddedPrecisionIntersectionSeconds'],
                'missedReviewedLiveSeconds': metric['coreHumanSeconds'] - metric['coreRecallIntersectionSeconds'],
                'correctlyRemovedSeconds': universe - metric['paddedHumanExportSeconds'] - incorrect}
            io.require(set(expected) == set(evaluation['padding'][pad]), 'Draft metric schema differs')
            for key, value in expected.items():
                helper.close(value, evaluation['padding'][pad][key], 'draft/' + key)
        io.require(len(evaluation['padding']) == 4 and evaluation['primary'] == evaluation['padding'][2], 'Draft padding differs')


def audit(report_path, output):
    evidence = {}
    report = io.read(bind(io.identity(report_path), evidence))
    io.require(report['kind'] == 'recall-floor-sweep-results-v1' and report['floors'] == list(range(90, 101))
               and report['heldPanelsUsedForSelection'] is False and report['trainingPerformed'] is False
               and report['protectedTestOpened'] is False, 'Unexpected sweep result contract')
    runner = module('sweep_bundle_reader', REPO / 'scripts/evaluate-neural-recall-sweep.py')
    bundle, examples, provenance = runner.load_bundle(bind(report['bundle'], evidence), evidence)
    helper_path = REPO / 'scripts/audit-neural-short-boost-intervals.py'
    bind({'path': str(helper_path), 'sha256': HELPER_SHA}, evidence)
    helper = module('independent_sweep_intervals', helper_path)
    for ref in report['references']:
        bind(ref, evidence)
    io.require(report['protocol'] == bundle['protocol'] and len(report['cells']) == len(bundle['cells']), 'Bundle/result scope differs')
    counts = dict(cells=0, candidateMetrics=0, floorSelections=0, distinctPanelPoints=0, pooledPanels=0)
    for spec, cell in zip(bundle['cells'], report['cells']):
        io.require(all(spec[k] == cell[k] for k in ('model', 'variant', 'seed', 'selectionDesign'))
                   and len(spec['folds']) == len(cell['folds']), 'Cell identity differs')
        panel_specs = {p['panelId']: p for p in spec['panels']}
        io.require(set(cell['panels']) == set(panel_specs), 'Panel inventory differs')
        for original, saved in zip(spec['folds'], cell['folds']):
            io.require(original['foldId'] == saved['foldId'], 'Fold identity differs')
            selection = runner.subset(examples, original['selectionRecordingIds'])
            io.require(all(e.label_policy == 'exact-rallies' and provenance[e.id] == 'manually-reviewed' for e in selection),
                       'Invalid calibration gold provenance')
            scores = load_score_shards(selection, original['selectionScores'], evidence=evidence)
            grid = [(epoch, decoder) for epoch in (5, 15, 30, 60) for decoder in expanded.decoder_candidates()]
            io.require([(c['epoch'], c['decoder']) for c in saved['candidates']] == grid, 'Candidate order/grid changed')
            for candidate in saved['candidates']:
                raw = io.serial_rows([e.row(base.decode(e, scores[candidate['epoch']][e.id], candidate['decoder'])) for e in selection])
                metric = helper.pooled_metric([helper.parse_record(row) for row in raw], 2.)
                helper.close(candidate['innerR_core'], metric['R_core'], 'candidate/R_core')
                helper.close(candidate['innerF1_padP_coreR'], metric['F1_padP_coreR'], 'candidate/F1')
                counts['candidateMetrics'] += 1
            io.require([d['floorPercent'] for d in saved['decisions']] == list(range(90, 101)), 'Missing recall floor')
            for decision in saved['decisions']:
                floor = decision['floorPercent'] / 100.
                # Use the exact stored IEEE operating metric after independent
                # interval validation: never add tolerance below the100% floor.
                eligible = [c for c in saved['candidates'] if c['innerR_core'] >= floor]
                io.require(decision['recallEligibilityFloor'] == floor and decision['feasible'] == bool(eligible)
                           and decision['eligibleCandidateCount'] == len(eligible) and decision['candidateCount'] == 192
                           and decision['maximumInnerRecall'] == max(c['innerR_core'] for c in saved['candidates'])
                           and decision['selected'] == (max(eligible, key=lambda c: c['innerF1_padP_coreR']) if eligible else None),
                           'Strict-floor selection/fallback differs')
                counts['floorSelections'] += 1
            io.require(set(saved['panels']) == {p['panelId'] for p in original['panels']}, 'Fold panel inventory differs')
            for panel in original['panels']:
                name = panel['panelId']; policy = panel_specs[name]['labelPolicy']; measured = saved['panels'][name]
                held = runner.subset(examples, panel['recordingIds'])
                io.require(all(provenance[e.id] == panel_specs[name]['reviewStatus'] for e in held), 'Panel review provenance differs')
                io.require(panel_specs[name]['sourcePolicy'] != 'source-held'
                           or not {e.group for e in held} & {e.group for e in selection}, 'Panel leaked into calibration sources')
                held_scores = load_score_shards(held, panel['scores'], evidence=evidence,
                    source_policy=panel_specs[name]['sourcePolicy'], require_all_epochs=False)
                expected_keys = set()
                io.require(len(measured['floors']) == 11 and measured['labelPolicy'] == policy, 'Panel floor/policy differs')
                for decision, selected in zip(saved['decisions'], measured['floors']):
                    chosen = decision['selected']
                    status = ('infeasible-inner-recall' if chosen is None else
                              'missing-selected-checkpoint' if chosen['epoch'] not in held_scores else 'available')
                    key = io.canonical({'epoch': chosen['epoch'], 'decoder': chosen['decoder']}) if status == 'available' else None
                    io.require(selected == {'floorPercent': decision['floorPercent'], 'selection': decision,
                                            'operatingPointKey': key, 'status': status}, 'Selected panel point differs')
                    if key:
                        expected_keys.add(key)
                io.require(set(measured['operatingPoints']) == expected_keys, 'Unselected or missing panel point')
                for key, point in measured['operatingPoints'].items():
                    decoded = {e.id: base.decode(e, held_scores[point['epoch']][e.id], point['decoder']) for e in held}
                    rows = io.panel_rows(held, decoded, policy)
                    io.require(rows == point['predictions'] and key == io.canonical({'epoch': point['epoch'], 'decoder': point['decoder']}),
                               'Saved panel boundaries changed')
                    check_evaluation(helper, rows, point['evaluation'], policy)
                    counts['distinctPanelPoints'] += 1
                gold = io.gold_signature(io.panel_rows(held, {e.id: [] for e in held}, policy), policy)
                io.require(measured['expectedGold'] == gold, 'Fold gold/source signature differs')
        for name, spec_panel in panel_specs.items():
            panel = cell['panels'][name]
            io.require(all(panel[k] == v for k, v in spec_panel.items()), 'Panel metadata differs')
            scope = runner.subset(examples, spec_panel['expectedRecordingIds']); policy = spec_panel['labelPolicy']
            expected_gold = io.gold_signature(io.panel_rows(scope, {e.id: [] for e in scope}, policy), policy)
            io.require(panel['result']['expectedGoldSha256'] == io.canonical(expected_gold), 'Pooled gold identity differs')
            fold_views = [f['panels'][name] for f in cell['folds'] if name in f['panels']]
            ids = [key for f in fold_views for key in f['expectedGold']]
            io.require(len(ids) == len(set(ids)) and set(ids) == set(expected_gold), 'Overlapping/incomplete declared outer scope')
            for i, floor in enumerate(panel['result']['floors']):
                statuses = [f['floors'][i]['status'] for f in fold_views]
                rows = [row for f in fold_views if f['floors'][i]['status'] == 'available'
                        for row in f['operatingPoints'][f['floors'][i]['operatingPointKey']]['predictions']]
                complete = all(s == 'available' for s in statuses)
                io.require(floor['floorPercent'] == 90+i and floor['foldStatuses'] == statuses and floor['predictions'] == rows
                           and floor['scopeRecordingIds'] == [r['id'] for r in rows] and floor['completeEvaluationScope'] == complete
                           and floor['partialScopeNotRankable'] == (not complete) and (floor['evaluation'] is not None) == complete,
                           'Partial-scope aggregation differs')
                if complete:
                    check_evaluation(helper, rows, floor['evaluation'], policy)
                    counts['pooledPanels'] += 1
        counts['cells'] += 1
    io.require(len(report['comparisons']) == len(bundle['comparisons']), 'Comparison inventory differs')
    for spec, comparison in zip(bundle['comparisons'], report['comparisons']):
        io.require(all(comparison[k] == v for k, v in spec.items()), 'Comparison identity differs')
        selected = [c for c in report['cells'] if all(c[k] == spec[k] for k in ('model', 'variant', 'selectionDesign'))]
        io.require(len(selected) == len(spec['seeds']) and {c['seed'] for c in selected} == set(spec['seeds']), 'Mean seed scope differs')
        selected.sort(key=lambda c: spec['seeds'].index(c['seed']))
        panels = [c['panels'][spec['panelId']]['result'] for c in selected]
        io.require(len({(p['expectedGoldSha256'], p['labelPolicy']) for p in panels}) == 1, 'Mean combines different gold/source policies')
        for i, row in enumerate(comparison['summary']['floors']):
            cells = [p['floors'][i] for p in panels]
            complete = all(c['completeEvaluationScope'] for c in cells)
            io.require(row['floorPercent'] == 90+i and row['allRegisteredSeedsComplete'] == complete
                       and row['completeSeeds'] == [c['seed'] for c, m in zip(selected, cells) if m['completeEvaluationScope']]
                       and (row['meanPadding'] is not None) == complete, 'Mean hides an infeasible seed')
            if complete:
                for j, padding in enumerate(row['meanPadding']):
                    io.require(padding['paddingSecondsBeforeAndAfter'] == j and padding['joinGapSeconds'] == 3., 'Mean padding differs')
                    for key in set(padding)-{'paddingSecondsBeforeAndAfter', 'joinGapSeconds'}:
                        helper.close(padding[key], statistics.mean(c['evaluation']['padding'][j][key] for c in cells), 'mean/' + key)
    for reference in list(evidence.values()):
        bind(reference)
    io.write_new(output, {'kind': 'independent-recall-floor-sweep-audit-v1', 'passed': True,
        'report': io.identity(report_path), 'bundle': report['bundle'], 'counts': counts,
        'auditor': io.identity(__file__), 'independentIntervalArithmetic': io.identity(helper_path),
        'references': list(evidence.values()), 'trainingPerformed': False, 'gpuUsed': False,
        'scope': 'Every candidate interval metric; exact strict IEEE floor/ordered F1 choice; every distinct held decoded point; all padding and original-rally coverage; full canonical event replay; no checkpoint neural forward replay.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    io.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')), 'Audit output must use NAS')
    audit(args.report, args.output)
