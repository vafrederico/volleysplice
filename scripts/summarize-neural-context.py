#!/usr/bin/env python3
"""Reporting helpers for a prospective matched temporal-context experiment.

The original profile is an immutable reference, not a newly trained replicate.
No hypothesis or cohort is selected by this module.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import argparse
from datetime import datetime, timezone
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transfer = import_script('context_transfer_helpers', 'summarize-neural-short-boost-transfer.py')
interval = import_script('context_interval_helpers', 'audit-neural-short-boost-intervals.py')
expanded, balanced, helpers = transfer.expanded, transfer.balanced, transfer.helpers
require, mean = transfer.require, transfer.mean
CONTEXTS = ('original', 'short')


def result_key(row):
    return row['kind'], row['context'], row['seed']


def verify_grid(results, contract):
    expected = {(kind, context, seed) for kind in contract['kinds']
                for context in CONTEXTS for seed in contract['seeds']}
    keys = [result_key(row) for row in results]
    require(set(keys) == expected and len(keys) == len(expected), 'Incomplete or duplicate context grid')
    require(all(row['cohort'] == contract['cohort'] and row['lossArm'] == contract['lossArm']
                for row in results), 'Context comparison changes cohort or loss arm')
    require(contract['targetPaddingSeconds'] == 2 and contract['joinGapSeconds'] == 3
            and contract['paddingSweep'] == [0, 1, 2, 3]
            and contract['primaryMetric'] == 'F1_padP_coreR', 'Metric contract differs')
    return {result_key(row): row for row in results}


def replay_metrics(result):
    """Re-evaluate saved intervals and independently sweep all time endpoints."""
    rows = result['predictions']
    replay = transfer.evaluate_predictions(rows)
    require(replay == result['evaluation'], 'Context canonical evaluation replay differs')
    records = [interval.parse_record(row) for row in rows]
    scopes = {'pooled': interval.compare_scope(records, replay, 'context/pooled')}
    for group, evaluation in replay['sourceGroups'].items():
        selected = [row for row in records if row['sourceGroup'] == group]
        scopes[group] = interval.compare_scope(selected, evaluation, 'context/'+group)
    recording_scopes = {row['id']: interval.compare_scope(
        [interval.parse_record(row)], transfer.evaluate_predictions([row]), 'context/recording/'+row['id'])
        for row in rows}
    return {'independentScopes': scopes,
            'independentRecordingScopes': recording_scopes,
            'independentPaddingRows': 4 * (1 + len(replay['sourceGroups']) + len(rows))}


def summarize_results(results, contract):
    by_key = verify_grid(results, contract)
    aggregates, per_seed, comparisons, losses = [], [], [], []
    for kind in contract['kinds']:
        for context in CONTEXTS:
            rows = []
            for seed in contract['seeds']:
                result = by_key[(kind, context, seed)]
                evaluation = result['evaluation']
                compact = {'kind': kind, 'context': context, 'seed': seed,
                           'cohort': contract['cohort'], 'lossArm': contract['lossArm'],
                           'origin': result['origin'], **expanded.compact_evaluation(evaluation),
                           'slices': balanced.event_slices(evaluation),
                           'sourceGroups': {group: {**expanded.compact_evaluation(value),
                                                   'slices': balanced.event_slices(value)}
                                            for group, value in evaluation['sourceGroups'].items()},
                           'selections': result['selections']}
                rows.append(compact)
                per_seed.append(compact)
            aggregates.append({'kind': kind, 'context': context, 'seedCount': len(rows),
                               'cohort': contract['cohort'], 'lossArm': contract['lossArm'],
                               'meanSeedPrimary': expanded.metric_means([r['primary'] for r in rows]),
                               'meanSeedPadding': [{'paddingSecondsBeforeAndAfter': pad,
                                   **expanded.metric_means([r['padding'][pad] for r in rows])} for pad in range(4)],
                               'meanSeedEventF1': mean(r['eventF1'] for r in rows),
                               'meanSeedSlices': balanced.mean_slices([r['slices'] for r in rows])})
        candidates = [by_key[(kind, 'short', seed)] for seed in contract['seeds']]
        references = [by_key[(kind, 'original', seed)] for seed in contract['seeds']]
        pairs = []
        for candidate, reference in zip(candidates, references):
            a, b = candidate['evaluation'], reference['evaluation']
            pairs.append({'seed': candidate['seed'], **expanded.paired_evaluation(a, b),
                          'sliceDelta': balanced.slice_delta(balanced.event_slices(a), balanced.event_slices(b)),
                          'pointRecovery': {name: transfer.recovery_effect(a, b, name)
                                            for name in balanced.short.SLICES}})
            losses.append({'kind': kind, 'seed': candidate['seed'],
                           'candidate': balanced.event_slices(a, identities=True),
                           'reference': balanced.event_slices(b, identities=True),
                           'paired': {scope: helpers.coverage_comparison(a, b, scope)
                                      for scope in balanced.SCOPES}})
        comparisons.append({'kind': kind, 'name': f'{kind}/short-minus-original', 'perSeed': pairs,
                            'meanSeedPrimaryDelta': expanded.metric_means([r['primaryDelta'] for r in pairs]),
                            'meanSeedEventF1Delta': mean(r['eventF1Delta'] for r in pairs),
                            'meanSeedSliceDelta': balanced.mean_slices([r['sliceDelta'] for r in pairs]),
                            'sourceGroupMeanPairedDelta': {group: {
                                field: mean(r['sourceGroups'][group][field] for r in pairs)
                                for field in ('F1_padP_coreRDelta', 'R_coreDelta', 'paddedModelExportSecondsDelta')}
                                for group in contract['groups']},
                            'standardF1Screen': expanded.feasibility(candidates, pairs, .95),
                            'retentionRecoveryScreen': balanced.recovery_screen(
                                candidates, references, contract['retentionRecoveryScreen'])})
    effects = []
    for seed in contract['seeds']:
        evaluations = [by_key[(kind, context, seed)]['evaluation']
                       for kind, context in (('dino_tcn', 'short'), ('dino_tcn', 'original'),
                                             ('tcn', 'short'), ('tcn', 'original'))]
        effects.append({'seed': seed, **transfer.difference_in_differences(*evaluations),
                        'sourceGroups': {group: transfer.difference_in_differences(
                            *(e['sourceGroups'][group] for e in evaluations)) for group in contract['groups']}})
    fields = effects[0]['differenceInDifferences']
    replication = {screen: all(row[screen]['screenPassedAndInnerFeasible'] for row in comparisons)
                   for screen in ('standardF1Screen', 'retentionRecoveryScreen')}
    return {'aggregates': aggregates, 'perSeed': per_seed, 'comparisons': comparisons,
            'transfer': {'perSeed': effects,
                         'meanSeedDifferenceInDifferences': {field: expanded.mean_available(
                             row['differenceInDifferences'][field] for row in effects) for field in fields},
                         'replication': replication,
                         'positiveInteractionRequired': False,
                         'interpretation': 'Within-architecture effects; zero interaction can mean equal benefits. '
                                           'F1 and recovery replication are separate descriptive screens.'}}, losses


def markdown(summary):
    lines = ['# Temporal-context comparison', '',
             'Primary scores use fixed 2s symmetric padding, strictly <3s gap joining and ignored subtraction. '
             'Each seed pools recordings before scoring; displayed means average the three seed scores. '
             'The original profile reuses audited reference results. No protected test or production promotion.', '',
             '| Architecture | Context | F1_padP_coreR | R_core | P_pad | Event F1 |',
             '|---|---|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        p = row['meanSeedPrimary']
        lines.append(f"| {row['kind']} | {row['context']} | {p['F1_padP_coreR']:.6f} | "
                     f"{p['R_core']:.6f} | {p['P_pad']:.6f} | {row['meanSeedEventF1']:.6f} |")
    lines += ['', '| Architecture | F1 change | Recall change | F1 screen | Recovery screen |',
              '|---|---:|---:|---|---|']
    for row in summary['comparisons']:
        p = row['meanSeedPrimaryDelta']
        lines.append(f"| {row['kind']} | {p['F1_padP_coreR']:+.6f} | {p['R_core']:+.6f} | "
                     f"{row['standardF1Screen']['screenPassedAndInnerFeasible']} | "
                     f"{row['retentionRecoveryScreen']['screenPassedAndInnerFeasible']} |")
    lines += ['', '| Architecture | Context | Slice | Complete losses | Partial losses | Core recall |',
              '|---|---|---|---:|---:|---:|']
    for row in summary['aggregates']:
        for name in ('all', 'duration_le_3s', 'duration_gt_3s', 'ace', 'service_fault'):
            value = row['meanSeedSlices']['primaryExportCoverage'][name]
            recall = f"{value['coreRecall']:.6f}" if value['coreRecall'] is not None else 'n/a'
            lines.append(f"| {row['kind']} | {row['context']} | {name} | {value['completeRallyLosses']:.2f} | "
                         f"{value['partialRallyLosses']:.2f} | {recall} |")
    lines += ['', '| Architecture | Context | Padding | P_pad | R_core | F1_padP_coreR | Model seconds | Human seconds | Difference |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        for p in row['meanSeedPadding']:
            lines.append(f"| {row['kind']} | {row['context']} | {p['paddingSecondsBeforeAndAfter']} | "
                         f"{p['P_pad']:.6f} | {p['R_core']:.6f} | {p['F1_padP_coreR']:.6f} | "
                         f"{p['paddedModelExportSeconds']:.3f} | {p['paddedHumanExportSeconds']:.3f} | "
                         f"{p['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ['', 'Per-seed and source-group effects, inner-selection feasibility, loss identities and paired '
              'difference-in-differences are retained in the JSON artifacts. Context refers to the temporal '
              'network only; existing audiovisual feature windows are unchanged.']
    return '\n'.join(lines) + '\n'


def audit_selections(result, data, manifest, contract, contract_hash):
    """Replay only the permitted inner view and the final held-group cuts."""
    from analysis import neural_context_development as runner

    examples = {row.example.id: row.example for row in data['exact']}
    owners, planned_views = runner.fold_graph(contract, contract_hash, manifest, result['kind'], result['seed'])
    refs = result['logicalInnerViews']
    require(len(refs) == 12 and len({ref['path'] for ref in refs}) == 12, 'Logical view inventory differs')
    views = [runner.read_verified(ref) for ref in refs]
    require(len({(v['outerIndex'], v['innerIndex']) for v in views}) == 12, 'Duplicate logical fold roles')
    for planned in planned_views:
        matches = [v for v in views if (v['outerIndex'], v['innerIndex']) == (planned['outerIndex'], planned['innerIndex'])]
        require(len(matches) == 1 and all(matches[0][key] == value for key, value in planned.items()), 'Logical view graph differs')
    selections = {s['heldSourceGroup']: s for s in result['selections']}
    require(set(selections) == set(contract['groups']) and len(result['selections']) == 4, 'Outer selections differ')
    saved_rows = {row['id']: row for row in result['predictions']}
    require(set(saved_rows) == set(examples) and len(saved_rows) == len(result['predictions']), 'Outer prediction population differs')
    checks = []
    for outer_index, outer in enumerate(contract['groups']):
        choice = selections[outer]
        require(choice['epoch'] in contract['checkpointEpochs'] and choice['decoder'] in contract['decoderCandidates'],
                'Selected checkpoint or decoder is outside registration')
        expanded.selection_feasible(choice, .95)
        fitting = [e for e in examples.values() if e.group != outer]
        all_scores = {epoch: {} for epoch in contract['checkpointEpochs']}
        inner_views = [v for v in views if v['outerIndex'] == outer_index]
        require(len(inner_views) == 3, 'Wrong number of inner logical folds')
        for view in inner_views:
            for epoch in contract['checkpointEpochs']:
                selected = runner.load_logical_predictions(view, owners[view['ownerId']], examples, epoch)
                require(not set(all_scores[epoch]) & set(selected), 'Duplicated inner prediction row')
                all_scores[epoch].update(selected)
        require(all(set(scores) == {e.id for e in fitting} for scores in all_scores.values()),
                'Inner selection sees wrong or outer rows')
        selected_again = runner.expanded.choose_settings(fitting, all_scores)
        require(selected_again == {key: value for key, value in choice.items() if key != 'heldSourceGroup'},
                'Saved selection is not the eligible-grid F1 winner')
        scores = all_scores[choice['epoch']]
        inner = runner.evaluate_predictions([e.row(runner.expanded.base.decode(e, scores[e.id], choice['decoder']))
                                             for e in fitting])
        require(abs(inner['primary']['F1_padP_coreR'] - choice['innerF1_padP_coreR']) <= 1e-12
                and abs(inner['primary']['R_core'] - choice['innerR_core']) <= 1e-12,
                'Selected inner score differs from restricted logical replay')
        folder = Path(result['origin']['fitRoot'])/f'outer-{outer_index}'/'refit'
        meta = runner.read(folder/'completed.json')
        filename = f"predictions-{choice['epoch']}.npz"
        require(runner.digest(folder/filename) == meta['artifacts'][filename], 'Outer refit probabilities changed')
        held = [e for e in examples.values() if e.group == outer]
        with np.load(folder/filename, allow_pickle=False) as cache:
            require(set(cache.files) == {e.id for e in held}, 'Outer refit prediction inventory differs')
            for e in held:
                probabilities = cache[e.id]
                require(probabilities.shape == (len(e.times), 4) and probabilities.dtype == np.float32
                        and np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
                        and np.all(probabilities[~e.valid] == 0), 'Invalid outer probabilities')
                cuts = [value.to_dict() for value in runner.expanded.base.decode(e, probabilities, choice['decoder'])]
                require(cuts == saved_rows[e.id]['predictions'], 'Saved outer cuts differ from refit replay')
        checks.append({'heldSourceGroup': outer, 'epoch': choice['epoch'],
                       'innerF1_padP_coreR': inner['primary']['F1_padP_coreR'], 'innerR_core': inner['primary']['R_core'],
                       'refitCompletion': runner.identity(folder/'completed.json'),
                       'refitProbabilities': runner.identity(folder/filename)})
    return checks


def summarize(study, tensor_audit_path):
    from analysis import neural_context_development as runner

    registration, manifest, reference = runner.validate_registration(study/'preregistration.json')
    contract, contract_hash = registration['contract'], registration['sha256']
    report = runner.read(study/'report.json')
    require(report['status'] == 'completed-context-development' and report['contractSha256'] == contract_hash
            and report['manifestSha256'] == contract['manifestSha256']
            and report['sourceGroups'] == contract['groups'] and report['records'] == 8
            and report['protectedTestOpened'] is False and report['productionPromotionAllowed'] is False,
            'Context study is not complete or changes evaluation scope')
    audit = runner.read(tensor_audit_path)
    require(audit['kind'] == 'independent-context-tensor-audit-v1' and audit['passed'] is True
            and audit['contractSha256'] == contract_hash
            and runner.contains_reference(audit, runner.identity(study/'report.json')), 'Missing bound independent tensor audit')
    bound = [runner.identity(path) for path in (study/'preregistration.json', study/'report.json', tensor_audit_path,
                                                Path(__file__), Path(contract['manifest']['path']),
                                                Path(manifest['exactManifest']['path']))]
    exact = runner.read_verified(manifest['exactManifest'])
    require(exact['recordings'] == manifest['exactRows'], 'Gold revision differs')
    results = report['results']
    verify_grid(results, contract)
    reference_by_key = {(r['kind'], r['seed']): r for r in reference['results']
                        if r['cohort'] == contract['cohort'] and r['lossArm'] == contract['lossArm']}
    data = runner.expanded.load_data(Path(contract['manifest']['path']))
    metric_checks, selection_checks = [], []
    for row in results:
        require(row['contractSha256'] == contract_hash, 'Result has wrong contract')
        helpers.assert_result_revision(exact, row['predictions'], str(result_key(row)))
        if row['context'] == 'original':
            original = reference_by_key[(row['kind'], row['seed'])]
            require(row == runner.reference_result(original, contract['referenceStudy'], contract_hash),
                    'Reused original profile payload changed')
        else:
            expected_root = study/'fits'/contract['cohort']/row['kind']/contract['lossArm']/'short'/str(row['seed'])
            require(row['origin'] == {'type': 'trained-context', 'fitRoot': str(expected_root),
                                      'innerOwnership': 'six-unordered-exclusions'}, 'Fresh context origin differs')
            selection_checks.append({'kind': row['kind'], 'seed': row['seed'],
                                     'selections': audit_selections(row, data, manifest, contract, contract_hash)})
        metric_checks.append({'kind': row['kind'], 'context': row['context'], 'seed': row['seed'], **replay_metrics(row)})
    values, losses = summarize_results(results, contract)
    require(sum(len(row['selections']) for row in selection_checks) == 24
            and sum(row['independentPaddingRows'] for row in metric_checks) == 624, 'Context audit count differs')
    for item in bound:
        require(runner.digest(item['path']) == item['sha256'], 'Context audit input changed')
    for name, sha in contract['code'].items():
        require(runner.digest(REPO/'analysis'/name) == sha, 'Registered source changed during summary')
    summary = {'kind': 'neural-context-summary-v1', 'status': 'completed-context-development-audit', 'passed': True,
               'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': contract_hash,
               'primaryMetric': 'F1_padP_coreR', 'primaryPaddingSeconds': 2, 'joinGapSeconds': 3,
               'protectedTestOpened': False, 'productionPromotionAllowed': False,
               'scope': {'cohort': contract['cohort'], 'lossArm': contract['lossArm'], 'exactRecordings': 8,
                         'exactSourceGroups': contract['groups'], 'seeds': contract['seeds'], 'resultCells': 12},
               'inputs': bound, 'referenceStudy': contract['referenceStudy'],
                       'audit': {'counts': {'canonicalMetricReplays': 12, 'reusedPayloadsVerified': 6,
                                    'freshSelectedInnerCandidatesReplayed': 24, 'freshSelectedRefitsReplayed': 24,
                                    'fullInnerGridSelectionsReplayed': 24,
                                    'independentPaddingScopeRows': 624},
                         'tensorAudit': runner.identity(tensor_audit_path), 'metricReplays': metric_checks,
                         'selectedCandidateReplays': selection_checks},
               'limitations': ['Adaptive development follow-up on previously inspected groups; no protected test.',
                               'Six original-profile cells retain preceding-study provenance; they are not new independent replicates.',
                               'Shared physical owners exclude both held groups in all tiers. Logical views remain distinct; fit reuse is computational only.',
                               'Parameter count and training recipe remain fixed; the temporal receptive field changes. Existing AV feature windows and whole-record transforms remain.',
                               'Profiles select settings independently with the same inner rule; effects describe training plus selection.',
                               'Three seeds and four source groups support descriptive comparisons, not a population significance claim.',
                               'F1 and recovery replication are separate screens. Neither permits production promotion.'],
               **values}
    return summary, {'scope': summary['scope'], 'perSeed': losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--tensor-audit', type=Path, required=True)
    args = parser.parse_args()
    names = ('summary.json', 'summary.md', 'loss-identities.json')
    require(not any((args.study/name).exists() for name in names), 'Refusing to overwrite completed context summary')
    summary, losses = summarize(args.study, args.tensor_audit)
    values = (json.dumps(summary, indent=2, allow_nan=False)+'\n', markdown(summary),
              json.dumps(losses, indent=2, allow_nan=False)+'\n')
    for name, value in zip(names, values):
        with (args.study/name).open('x', encoding='utf-8') as handle:
            handle.write(value)
    print(json.dumps({'status': summary['status'], 'counts': summary['audit']['counts'],
                      'artifacts': {name: transfer.identity(args.study/name) for name in names}}, indent=2))


if __name__ == '__main__':
    main()
