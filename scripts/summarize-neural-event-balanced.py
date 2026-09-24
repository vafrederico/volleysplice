#!/usr/bin/env python3
"""Independent audit of event-balanced TCNs against frozen unweighted runs.

This script never fits, ranks a new candidate, or changes a decoder. Both output
sets are replayed under the existing metric contract; recovery is a separately
declared descriptive screen, not a production authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0088'))
EXPANDED = Path(private_value('private-reference-0070'))
from analysis.neural_evaluation import evaluate_predictions


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


expanded = import_script('event_balanced_expanded_helpers', 'summarize-neural-expanded.py')
short = import_script('event_balanced_short_helpers', 'analyze-neural-expanded-short-events.py')
helpers = expanded.helpers
require, identity, mean = expanded.require, expanded.identity, expanded.mean
COHORTS = expanded.COHORTS
SCOPES = expanded.SCOPES
RECOVERY_CONTRACT = {
    'meanF1MinimumDelta': -.005, 'meanRCoreMinimumDelta': -.005,
    'maximumCompleteLossRatio': .8, 'maximumShortCompleteLossRatio': .8,
    'maximumIncompleteLossDelta': 0, 'maximumShortIncompleteLossDelta': 0,
    'completeLossReductionSeedCount': 2, 'meanEventF1MinimumDelta': -.01,
    'shortCompleteLossReductionSeedCount': 2, 'meanLongRCoreMinimumDelta': -.005,
    'shortDurationSeconds': 3, 'coverageScope': 'primaryExportCoverage',
}


def event_slices(evaluation, *, identities=False):
    output = {}
    for scope in SCOPES:
        output[scope] = {}
        for name in short.SLICES:
            row = short.loss_stats([r for r in evaluation['guardrails'][scope]['rallies'] if short.belongs(r, name)])
            if not identities:
                row.pop('lossIdentities')
            row['incompleteRallyLosses'] = row['completeRallyLosses'] + row['partialRallyLosses']
            output[scope][name] = row
    return output


def slice_delta(candidate, baseline):
    output = {}
    for scope in SCOPES:
        output[scope] = {}
        for name in short.SLICES:
            first, second = candidate[scope][name], baseline[scope][name]
            require(first['evaluableRallies'] == second['evaluableRallies']
                    and first['evaluableCoreSeconds'] == second['evaluableCoreSeconds'], 'Slice gold revisions differ')
            output[scope][name] = {key: (first[key] - second[key] if first[key] is not None and second[key] is not None else None)
                                   for key in first if key != 'lossIdentities'}
    return output


def mean_slices(rows):
    return {scope: {name: {key: expanded.mean_available(row[scope][name][key] for row in rows)
                           for key in rows[0][scope][name] if key != 'lossIdentities'}
                    for name in short.SLICES} for scope in SCOPES}


def recovery_screen(candidate_rows, baseline_rows, contract):
    """Point counts use padded exports; seed ties are not retention recovery."""
    require(contract == RECOVERY_CONTRACT, 'Unrecognized retention recovery contract; do not silently change gates')
    require(len(candidate_rows) == len(baseline_rows) == 3, 'Recovery screen requires the three fixed paired seeds')
    require(all(c['seed'] == b['seed'] and c['cohort'] == b['cohort'] for c, b in zip(candidate_rows, baseline_rows)),
            'Recovery screen has mismatched seed/cohort pairing')
    candidate_slices = [event_slices(r['evaluation']) for r in candidate_rows]
    baseline_slices = [event_slices(r['evaluation']) for r in baseline_rows]
    scope = contract['coverageScope']
    all_counts = lambda rows, key: [r[scope]['all'][key] for r in rows]
    short_counts = lambda rows, key: [r[scope]['duration_le_3s'][key] for r in rows]
    ac, ab = all_counts(candidate_slices, 'completeRallyLosses'), all_counts(baseline_slices, 'completeRallyLosses')
    sc, sb = short_counts(candidate_slices, 'completeRallyLosses'), short_counts(baseline_slices, 'completeRallyLosses')
    incomplete_delta = mean(all_counts(candidate_slices, 'incompleteRallyLosses')) - mean(all_counts(baseline_slices, 'incompleteRallyLosses'))
    short_incomplete_delta = mean(short_counts(candidate_slices, 'incompleteRallyLosses')) - mean(short_counts(baseline_slices, 'incompleteRallyLosses'))
    primary_delta = {key: mean(c['evaluation']['primary'][key] - b['evaluation']['primary'][key]
                               for c, b in zip(candidate_rows, baseline_rows)) for key in ('F1_padP_coreR', 'R_core')}
    events = [(c['evaluation']['guardrails']['eventF1'], b['evaluation']['guardrails']['eventF1'])
              for c, b in zip(candidate_rows, baseline_rows)]
    require(all(c is not None and b is not None for c, b in events), 'Recovery screen needs evaluable event F1')
    event_delta = mean(c - b for c, b in events)
    improved = sum(c < b for c, b in zip(ac, ab))
    short_improved = sum(c < b for c, b in zip(sc, sb))
    long_recalls = [(c[scope]['duration_gt_3s']['coreRecall'], b[scope]['duration_gt_3s']['coreRecall'])
                    for c, b in zip(candidate_slices, baseline_slices)]
    require(all(c is not None and b is not None for c, b in long_recalls), 'Long-rally slice recall is unavailable')
    long_delta = mean(c - b for c, b in long_recalls)
    checks = {
        'meanF1RegressionWithinLimit': primary_delta['F1_padP_coreR'] >= contract['meanF1MinimumDelta'] - 1e-12,
        'meanCoreRecallRegressionWithinLimit': primary_delta['R_core'] >= contract['meanRCoreMinimumDelta'] - 1e-12,
        'meanCompleteLossesReducedAtLeast20Percent': mean(ac) <= contract['maximumCompleteLossRatio'] * mean(ab) + 1e-12,
        'meanShortCompleteLossesReducedAtLeast20Percent': mean(sc) <= contract['maximumShortCompleteLossRatio'] * mean(sb) + 1e-12,
        'meanIncompleteLossesDoNotIncrease': incomplete_delta <= contract['maximumIncompleteLossDelta'] + 1e-12,
        'meanShortIncompleteLossesDoNotIncrease': short_incomplete_delta <= contract['maximumShortIncompleteLossDelta'] + 1e-12,
        'atLeastTwoSeedsReduceCompleteLosses': improved >= contract['completeLossReductionSeedCount'],
        'atLeastTwoSeedsReduceShortCompleteLosses': short_improved >= contract['shortCompleteLossReductionSeedCount'],
        'meanLongCoreRecallRegressionWithinLimit': long_delta >= contract['meanLongRCoreMinimumDelta'] - 1e-12,
        'meanEventF1RegressionWithinLimit': event_delta >= contract['meanEventF1MinimumDelta'] - 1e-12,
    }
    inner_feasible = all(expanded.selection_feasible(s, .95) for r in candidate_rows for s in r['selections'])
    return {'passed': all(checks.values()), 'checks': checks, 'completeLossReductionSeeds': improved,
            'shortCompleteLossReductionSeeds': short_improved, 'meanLongCoreRecallDelta': long_delta,
            'meanCandidateCompleteLosses': mean(ac), 'meanBaselineCompleteLosses': mean(ab),
            'meanCandidateShortCompleteLosses': mean(sc), 'meanBaselineShortCompleteLosses': mean(sb),
            'meanIncompleteLossDelta': incomplete_delta, 'meanShortIncompleteLossDelta': short_incomplete_delta,
            'meanEventF1Delta': event_delta, 'meanPrimaryDelta': primary_delta,
            'allCandidateInnerSelectionsFeasible': inner_feasible,
            'screenPassedAndInnerFeasible': all(checks.values()) and inner_feasible,
            'allOuterSeedRecallAtLeast095Diagnostic': all(r['evaluation']['primary']['R_core'] >= .95 for r in candidate_rows),
            'productionPromotionAllowed': False,
            'role': 'separate preregistered retention-recovery screen; no source-group or absolute outer recall gate'}


def audit_historical_exposure(weighted, baseline):
    """Check every shared epoch and every active sample stream, not just exact."""
    for key in ('trainIds', 'auxiliaryIds', 'validationIds', 'scalerTrainIds', 'positiveWeight', 'supervisedCounts', 'parameters'):
        require(weighted[key] == baseline[key], f'Weighted versus historical {key} differs')
    first, second = weighted['history'], baseline['history']
    require(first and second, 'Missing historical exposure history')
    for left, right in zip(first, second):
        require(left['epoch'] == right['epoch'] and left['optimizerSteps'] == right['optimizerSteps'],
                'Historical optimizer-step exposure differs')
        require(left['exposureSha256'] == right['exposureSha256'], 'Historical exact/auxiliary sample exposure differs')
        require(set(left['exposureSha256']) == {'exact', *weighted['auxiliaryIds']}, 'Missing sample stream')
    return min(len(first), len(second))


def summarize_results(results, references, contract):
    baseline = {(r['cohort'], r['kind'], r['seed']): r for r in references}
    per_seed, aggregates, comparisons, losses = [], [], [], []
    for cohort in COHORTS:
        weighted_rows = [next(r for r in results if r['cohort'] == cohort and r['seed'] == seed) for seed in contract['seeds']]
        historical_rows = [baseline[(cohort, 'tcn', seed)] for seed in contract['seeds']]
        pairs = []
        for candidate, reference in zip(weighted_rows, historical_rows):
            evaluation, previous = candidate['evaluation'], reference['evaluation']
            helpers.assert_comparable(evaluation, previous)
            slices, old_slices = event_slices(evaluation), event_slices(previous)
            pair = {'seed': candidate['seed'], **expanded.paired_evaluation(evaluation, previous),
                    'sliceDelta': slice_delta(slices, old_slices)}
            pairs.append(pair)
            per_seed.append({'cohort': cohort, 'kind': 'tcn', 'seed': candidate['seed'],
                             **expanded.compact_evaluation(evaluation), 'slices': slices,
                             'historicalTCN': {**expanded.compact_evaluation(previous), 'slices': old_slices},
                             'sourceGroups': {group: {'weighted': expanded.compact_evaluation(value),
                                                     'historicalTCN': expanded.compact_evaluation(previous['sourceGroups'][group]),
                                                     'weightedSlices': event_slices(value),
                                                     'historicalSlices': event_slices(previous['sourceGroups'][group]),
                                                     'sliceDelta': slice_delta(event_slices(value), event_slices(previous['sourceGroups'][group]))}
                                              for group, value in evaluation['sourceGroups'].items()},
                             'selections': candidate['selections'], 'pairedHistoricalTCN': pair})
            refs = {'sameCohortHistoricalTCN': previous,
                    'sameCohortHistoricalLinear': baseline[(cohort, 'linear', candidate['seed'])]['evaluation'],
                    'exactHistoricalTCN': baseline[('exact', 'tcn', candidate['seed'])]['evaluation']}
            losses.append({'cohort': cohort, 'kind': 'tcn', 'seed': candidate['seed'],
                           'slices': event_slices(evaluation, identities=True),
                           'pairedReferences': {name: {scope: helpers.coverage_comparison(evaluation, ref, scope)
                                                        for scope in SCOPES} for name, ref in refs.items()}})
        these = [r for r in per_seed if r['cohort'] == cohort]
        aggregates.append({'cohort': cohort, 'kind': 'tcn', 'seedCount': len(these),
                           'meanSeedPrimary': expanded.metric_means([r['primary'] for r in these]),
                           'meanSeedPadding': [{'paddingSecondsBeforeAndAfter': pad,
                                                **expanded.metric_means([r['padding'][pad] for r in these])} for pad in range(4)],
                           'meanSeedHistoricalPrimary': expanded.metric_means([r['historicalTCN']['primary'] for r in these]),
                           'meanSeedHistoricalPadding': [{'paddingSecondsBeforeAndAfter': pad,
                                                          **expanded.metric_means([r['historicalTCN']['padding'][pad] for r in these])} for pad in range(4)],
                           'meanSeedEventF1': mean(r['eventF1'] for r in these),
                           'meanSeedSlices': mean_slices([r['slices'] for r in these]),
                           'meanSeedHistoricalSlices': mean_slices([r['historicalTCN']['slices'] for r in these])})
        comparisons.append({'name': f'{cohort}/weighted-minus-unweighted-TCN',
                            'meanSeedPrimaryDelta': expanded.metric_means([r['primaryDelta'] for r in pairs]),
                            'meanSeedSliceDelta': mean_slices([r['sliceDelta'] for r in pairs]),
                            'sourceGroupMeanPairedDelta': {group: {
                                field: mean(r['sourceGroups'][group][field] for r in pairs)
                                for field in ('F1_padP_coreRDelta', 'R_coreDelta', 'paddedModelExportSecondsDelta')}
                                for group in contract['groups']},
                            'perSeed': pairs,
                            'standardF1Screen': expanded.feasibility(weighted_rows, pairs, .95),
                            'retentionRecoveryScreen': recovery_screen(weighted_rows, historical_rows, contract['retentionRecoveryScreen'])})
    return per_seed, aggregates, comparisons, losses


def expected_live_weighting(row, tier):
    """Reconstruct loss multipliers from original gold, without weighting code."""
    with np.load(row['featureCaches']['audiovisual']['path'], allow_pickle=False) as data:
        times = np.asarray(data['times'], dtype=np.float64)
    valid = np.ones(len(times), dtype=bool)
    ignored = row.get('ignoredIntervals', [])
    for interval in ignored:
        valid[(times >= interval['start']) & (times < interval['end'])] = False
    if tier == 'coverage':
        valid &= (times >= row['gameWindow']['start']) & (times < row['gameWindow']['end'])
    mask = valid.copy() if tier in ('exact', 'draft') else np.zeros(len(times), dtype=bool)
    truth = row.get('rallies', [])
    if tier == 'draft':
        for interval in [*truth, *ignored]:
            for edge in ('start', 'end'):
                mask[np.abs(times - interval[edge]) <= 1.] = False
    inside = [(times >= interval['start']) & (times < interval['end']) for interval in truth]
    positive = np.zeros(len(times), dtype=bool)
    for selected in inside:
        require(not np.any(positive & selected), 'Overlapping original event targets')
        positive |= selected
    if tier == 'coverage':
        positive[:] = False
    eligible = positive & mask & valid
    counts = [int((selected & eligible).sum()) for selected in inside]
    population, nonempty = sum(counts), sum(count > 0 for count in counts)
    weights = np.ones(len(times), dtype='<f4')
    events, multipliers = [], []
    for index, (interval, selected, count) in enumerate(zip(truth, inside, counts)):
        multiplier = population / (nonempty * count) if count else None
        if count:
            weights[selected & eligible] = multiplier
            multipliers.append(multiplier)
        events.append({'eventIndex': index, 'start': float(interval['start']), 'end': float(interval['end']),
                       'tags': interval.get('tags', []), 'positiveSupervisedTicks': count,
                       'invalidTicksInsideEvent': int((selected & ~valid).sum()),
                       'maskedValidTicksInsideEvent': int((selected & valid & ~mask).sum()),
                       'multiplier': multiplier,
                       'weightedPositiveMass': float(weights[selected & eligible].astype(np.float64).sum())})
    levels = (0, 25, 50, 75, 90, 95, 99, 100)
    percentiles = lambda values: {str(q): float(v) for q, v in zip(levels, np.percentile(values, levels))} if len(values) else {}
    return {'method': 'per-record-original-event-positive-mass-v1', 'recordingId': row['id'],
            'sourceGroup': row['sourceGroup'], 'tier': tier, 'originalEventCount': len(truth),
            'eligibleEventCount': nonempty, 'zeroSupervisedEventCount': len(truth) - nonempty,
            'positiveSupervisedTicks': population, 'weightedPositiveMass': float(weights[eligible].astype(np.float64).sum()),
            'negativeSupervisedTicks': int((valid & mask & ~positive).sum()),
            'minimumPositiveMultiplier': float(weights[eligible].min()) if population else None,
            'maximumPositiveMultiplier': float(weights[eligible].max()) if population else None,
            'liveMultiplierSha256': hashlib.sha256(weights.tobytes(order='C')).hexdigest(),
            'liveMultiplierHashEncoding': 'little-endian float32, C-order, one value per original cache tick',
            'eventMultiplierPercentiles': percentiles(multipliers),
            'positiveTickMultiplierPercentiles': percentiles(weights[eligible]), 'events': events}


def registration(path):
    value = helpers.load(path)
    canonical = json.dumps(value['contract'], sort_keys=True, separators=(',', ':')).encode()
    require(hashlib.sha256(canonical).hexdigest() == value['sha256'], f'Invalid preregistration: {path}')
    return value


def read_inputs(study, manifest_path, *, progress=False):
    reg = registration(study / 'preregistration.json')
    contract = reg['contract']
    require(contract['cohorts'] == list(COHORTS) and contract['kinds'] == ['tcn'], 'Unexpected weighted study population')
    require(contract['retentionRecoveryScreen'] == RECOVERY_CONTRACT, 'Unknown recovery screen')
    ref_info = contract['referenceStudy']
    reference_path = Path(ref_info['path'])
    reference_registration_path = reference_path / 'preregistration.json'
    reference_report_path, reference_summary_path = reference_path / 'report.json', reference_path / 'summary.json'
    for path, sha in ((reference_registration_path, ref_info['preregistrationFileSha256']),
                      (reference_report_path, ref_info['reportSha256']), (reference_summary_path, ref_info['summarySha256'])):
        require(helpers.digest(path) == sha, f'Historical reference changed: {path}')
    old_reg = registration(reference_registration_path)
    require(old_reg['sha256'] == ref_info['contractSha256'], 'Historical contract mismatch')
    old_contract = old_reg['contract']
    for field in ('manifestSha256', 'cohorts', 'seeds', 'checkpointEpochs', 'groups', 'primaryMetric',
                  'targetPaddingSeconds', 'joinGapSeconds', 'paddingSweep', 'decoderCandidates', 'selection', 'screen'):
        require(contract[field] == old_contract[field], f'Weighted comparison changed {field}')
    require({k: v for k, v in contract['training'].items() if k != 'loss'}
            == {k: v for k, v in old_contract['training'].items() if k != 'loss'}, 'Other training recipe fields changed')
    require(contract['environment'] == old_contract['environment'], 'Training environment differs from reference')
    require(contract['liveLossWeighting']['mode'] == 'per_rally' and contract['liveLossWeighting']['scope'] == ['exact', 'draft']
            and contract['liveLossWeighting']['cap'] is None, 'Unexpected live weighting scope or cap')
    require(len(contract['seeds']) == 3 and len(contract['groups']) == 4, 'Unexpected declared study size')
    require(contract['primaryMetric'] == 'F1_padP_coreR' and contract['targetPaddingSeconds'] == 2
            and contract['joinGapSeconds'] == 3 and contract['paddingSweep'] == [0, 1, 2, 3], 'Unexpected metric contract')
    manifest = helpers.load(manifest_path)
    require(helpers.digest(manifest_path) == contract['manifestSha256'], 'Expanded manifest changed')
    exact_path = Path(manifest['exactManifest']['path'])
    require(helpers.digest(exact_path) == manifest['exactManifest']['sha256'], 'Exact manifest changed')
    exact = helpers.load(exact_path)
    require(exact['recordings'] == manifest['exactRows'], 'Exact gold revision mismatch')
    rows = manifest['exactRows'] + manifest['draftRows'] + manifest['coverageRows']
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate tier recording')
    require(not {r['sourceGroup'] for r in rows}.intersection(manifest['protectedSourceGroups']), 'Protected group')
    require(all(r['environment'] in ('grass', 'indoor') and r['consent']['train'] for r in rows), 'Invalid scope/consent')
    old_report = helpers.load(reference_report_path)
    old_summary = helpers.load(reference_summary_path)
    require(old_summary['status'] == 'completed-expanded-development-audit'
            and old_summary['contractSha256'] == old_reg['sha256'], 'Historical summary is not the audited reference')
    require(old_report['contractSha256'] == old_reg['sha256'] and old_report['manifestSha256'] == contract['manifestSha256'],
            'Historical report provenance differs')
    report_path = study / 'report.json'
    report = helpers.load(report_path) if report_path.exists() else {'results': [helpers.load(p) for p in sorted(study.glob('result-*.json'))]}
    expected = {(cohort, 'tcn', seed) for cohort in COHORTS for seed in contract['seeds']}
    observed = {expanded.result_key(r) for r in report['results']}
    require(len(observed) == len(report['results']) and observed <= expected, 'Unexpected weighted result identity')
    if not progress:
        require(report_path.exists() and observed == expected, 'Final summary requires all nine weighted results')
        require(report['contractSha256'] == reg['sha256'] and report['manifestSha256'] == contract['manifestSha256'],
                'Weighted report provenance differs')
        require(report['status'] == 'completed-event-balanced-development', 'Weighted report is not completed')
    for current in [old_report, *([report] if report_path.exists() else [])]:
        require(not current['protectedTestOpened'] and not current['productionPromotionAllowed'], 'Unexpected promotion/test status')
        require(current['records'] == len(exact['recordings']) and sorted(current['sourceGroups']) == contract['groups'], 'Result scope differs')
    expected_old = {(c, k, s) for c in COHORTS for k in ('linear', 'tcn') for s in contract['seeds']}
    require(len(old_report['results']) == len(expected_old) and {expanded.result_key(r) for r in old_report['results']} == expected_old,
            'Historical results incomplete')
    inputs = {'preregistration': identity(study / 'preregistration.json'), 'manifest': identity(manifest_path),
              'exactManifest': identity(exact_path), 'referenceRegistration': identity(reference_registration_path),
              'referenceReport': identity(reference_report_path), 'referenceSummary': identity(reference_summary_path),
              'summaryCode': [identity(Path(__file__)), identity(REPO / 'scripts/summarize-neural-expanded.py'),
                              identity(REPO / 'scripts/analyze-neural-expanded-short-events.py'), identity(expanded.HELPER_PATH)]}
    if report_path.exists():
        inputs['report'] = identity(report_path)
    return reg, old_reg, manifest, exact, report, old_report, reference_path, inputs


def audit_fit_files(study, reference_study, manifest, reg, old_reg, results, references):
    contract, groups = reg['contract'], reg['contract']['groups']
    tiers = {'exact': manifest['exactRows'], 'draft': manifest['draftRows'], 'coverage': manifest['coverageRows']}
    by_id = {r['id']: r for rows in tiers.values() for r in rows}
    supervision = {r['id']: expanded.expected_supervision(r, tier) for tier, rows in tiers.items() for r in rows}
    weighting = {r['id']: expected_live_weighting(r, tier) for tier, rows in tiers.items() for r in rows}
    selected = {(r['cohort'], r['seed'], s['heldSourceGroup']): s['epoch'] for r in results for s in r['selections']}
    old_selected = {(r['cohort'], r['seed'], s['heldSourceGroup']): s['epoch'] for r in references
                    if r['kind'] == 'tcn' for s in r['selections']}
    counts = Counter()
    completed_inputs, metadata, paths, runtime = [], {}, {}, {}
    for cohort in COHORTS:
        active = () if cohort == 'exact' else ('draft',) if cohort == 'draft' else ('draft', 'coverage')
        for seed in contract['seeds']:
            for outer_index, outer in enumerate(groups):
                folds = [(f'inner-{i}', group) for i, group in enumerate(g for g in groups if g != outer)] + [('refit', outer)]
                for fold, validation in folds:
                    excluded = {outer, validation}
                    train_ids = [r['id'] for r in tiers['exact'] if r['sourceGroup'] not in excluded]
                    auxiliary = {tier: [r['id'] for r in tiers[tier] if r['sourceGroup'] not in excluded] for tier in active}
                    validation_ids = [r['id'] for r in tiers['exact'] if r['sourceGroup'] == validation]
                    relative = Path('fits') / cohort / 'tcn' / str(seed) / f'outer-{outer_index}' / fold
                    pair = []
                    for label, root, registered, chosen in (('weighted', study, reg, selected),
                                                          ('historical', reference_study, old_reg, old_selected)):
                        path = root / relative / 'completed.json'
                        meta = helpers.load(path)
                        completed_inputs.append(identity(path))
                        require(meta['contractSha256'] == registered['sha256'] and meta['kind'] == 'tcn' and meta['seed'] == seed,
                                f'Fit identity mismatch: {path}')
                        require(meta['trainIds'] == train_ids and meta['scalerTrainIds'] == train_ids
                                and meta['auxiliaryIds'] == auxiliary and meta['validationIds'] == validation_ids, f'Fold leakage: {path}')
                        require(meta['trainGroups'] == sorted({by_id[i]['sourceGroup'] for i in train_ids})
                                and meta['validationGroups'] == [validation]
                                and meta['auxiliaryGroups'] == {tier: sorted({by_id[i]['sourceGroup'] for i in ids}) for tier, ids in auxiliary.items()},
                                f'Fit group metadata mismatch: {path}')
                        epochs = [chosen[(cohort, seed, outer)]] if fold == 'refit' else contract['checkpointEpochs']
                        require(meta['epochs'] == epochs, f'Checkpoint/refit epochs differ: {path}')
                        ids_by_tier = {'exact': train_ids, **auxiliary}
                        require(set(meta['supervisedCounts']) == set(ids_by_tier), 'Unexpected supervision tier')
                        for tier, ids in ids_by_tier.items():
                            for field in ('valid', 'positiveMass'):
                                expected = np.sum([supervision[i][field] for i in ids], axis=0) if ids else np.zeros(4)
                                require(np.allclose(meta['supervisedCounts'][tier][field], expected, rtol=1e-6, atol=1e-4),
                                        f'Supervision counts differ from original gold: {path}')
                        if label == 'weighted':
                            require(meta['liveLossWeighting']['mode'] == 'per_rally'
                                    and meta['liveLossWeighting']['rowDiagnosticsApply'] is True, 'Weighted fit is not event-balanced')
                            require(meta['liveLossWeighting']['rows'] == {tier: [weighting[i] for i in ids] for tier, ids in ids_by_tier.items()},
                                    f'Live weight arrays/masses differ from independent reconstruction: {path}')
                            counts['fitWeightDiagnosticsVerified'] += 1
                        require([r['epoch'] for r in meta['history']] == list(range(1, max(epochs) + 1)), 'Missing epoch history')
                        require(meta['optimizerSteps'] == meta['history'][-1]['optimizerSteps']
                                and meta['exposureSha256'] == meta['history'][-1]['exposureSha256'], 'Final exposure metadata differs')
                        names = {f'{stem}-{epoch}.npz' for epoch in epochs for stem in ('weights', 'predictions')}
                        require(set(meta['artifacts']) == names, 'Missing checkpoint artifacts')
                        for name, sha in meta['artifacts'].items():
                            require(helpers.digest(path.parent / name) == sha, f'Checkpoint changed: {path.parent / name}')
                            counts[f'{label}ArtifactHashesVerified'] += 1
                        counts[f'{label}FitMembershipsVerified'] += 1
                        pair.append(meta)
                        if label == 'weighted':
                            key = (cohort, seed, outer_index, fold)
                            metadata[key], paths[key] = meta, path.parent
                            bucket = runtime.setdefault(cohort, {'fitCount': 0, 'summedFitWallSeconds': 0., 'optimizerSteps': 0,
                                                                 'maximumPerProcessAllocatedCudaBytes': 0})
                            bucket['fitCount'] += 1
                            bucket['summedFitWallSeconds'] += meta['wallSeconds']
                            bucket['optimizerSteps'] += meta['optimizerSteps']
                            bucket['maximumPerProcessAllocatedCudaBytes'] = max(bucket['maximumPerProcessAllocatedCudaBytes'], meta['peakAllocatedCudaBytes'] or 0)
                    counts['historicalPairedEpochPrefixesVerified'] += audit_historical_exposure(*pair)
                    counts['historicalPairedFitPrefixesVerified'] += 1
                    # Every new scaler checkpoint must equal its fixed historical scaler.
                    with np.load(reference_study / relative / f"weights-{pair[1]['epochs'][0]}.npz", allow_pickle=False) as historical:
                        for epoch in pair[0]['epochs']:
                            with np.load(study / relative / f'weights-{epoch}.npz', allow_pickle=False) as weighted:
                                require(np.array_equal(weighted['mean'], historical['mean'])
                                        and np.array_equal(weighted['scale'], historical['scale']), 'Historical scaler tensor mismatch')
                                counts['historicalPairedScalerCheckpointsVerified'] += 1
    for seed in contract['seeds']:
        for outer_index in range(len(groups)):
            for fold in [*(f'inner-{i}' for i in range(len(groups)-1)), 'refit']:
                for a, b in (('exact', 'draft'), ('draft', 'reviewed_export')):
                    left, right = metadata[(a, seed, outer_index, fold)], metadata[(b, seed, outer_index, fold)]
                    counts['cohortPairedEpochPrefixesVerified'] += expanded.audit_exposure_pair(left, right, compare_draft=a == 'draft')
                    counts['cohortPairedFitPrefixesVerified'] += 1
    counts['weightVectorsIndependentlyReconstructed'] = len(weighting)
    for name in ('weighted', 'historical'):
        require(counts[f'{name}FitMembershipsVerified'] == 144 and counts[f'{name}ArtifactHashesVerified'] == 936,
                f'Unexpected {name} fit/checkpoint inventory')
    diagnostics_path = study / 'weight-diagnostics.json'
    diagnostics = helpers.load(diagnostics_path)
    require(diagnostics['contractSha256'] == reg['sha256'] and diagnostics['manifestSha256'] == contract['manifestSha256'],
            'Global weight diagnostics provenance differs')
    require(diagnostics['rows'] == {tier: [weighting[r['id']] for r in rows] for tier, rows in tiers.items()},
            'Global weight diagnostics differ from independent reconstruction')
    return {'counts': dict(counts), 'completedMetadata': completed_inputs,
            'weightDiagnosticsArtifact': identity(diagnostics_path),
            'weightingByRecording': weighting, 'trainingRuntime': runtime,
            'runtimeScope': 'Summed fit durations may overlap; CUDA peak is per process. Training runtime is not phone/browser inference latency.'}


def summarize(study, manifest_path, *, progress=False):
    reg, old_reg, manifest, exact, report, old_report, reference_path, inputs = read_inputs(study, manifest_path, progress=progress)
    contract, results, references = reg['contract'], report['results'], old_report['results']
    if progress:
        return {'status': 'complete-report-present' if (study / 'report.json').exists() else 'in-progress',
                'completedResults': len(results), 'expectedResults': 9,
                'results': [{'cohort': r['cohort'], 'seed': r['seed'], 'F1_padP_coreR': r['evaluation']['objective'],
                             'R_core': r['evaluation']['primary']['R_core']} for r in results],
                'note': 'Progress only: no final audits or screen conclusions.'}, None
    inputs['frozenSources'] = []
    for registered in (old_reg, reg):
        for name, sha in registered['contract']['code'].items():
            path = REPO / 'analysis' / name
            require(helpers.digest(path) == sha, f'Frozen module changed: {path}')
            info = identity(path)
            if info not in inputs['frozenSources']:
                inputs['frozenSources'].append(info)
    inputs['featureCaches'] = []
    for row in manifest['exactRows'] + manifest['draftRows'] + manifest['coverageRows']:
        cache = row['featureCaches']['audiovisual']
        require(helpers.digest(Path(cache['path'])) == cache['sha256'], f"Feature cache changed: {row['id']}")
        inputs['featureCaches'].append(identity(Path(cache['path'])))
    inputs['resultFiles'] = []
    for rows, root in ((results, study), (references, reference_path)):
        for row in rows:
            path = root / f"result-{row['cohort']}-{row['kind']}-{row['seed']}.json"
            require(helpers.load(path) == row, f'Result differs from report: {path}')
            if root == study:
                require(row['contractSha256'] == reg['sha256'], 'Weighted result contract mismatch')
            inputs['resultFiles'].append(identity(path))
            helpers.assert_result_revision(exact, row['predictions'], str(expanded.result_key(row)))
            require(evaluate_predictions(row['predictions'], primary_padding_seconds=2., join_gap_seconds=3.) == row['evaluation'],
                    f'Canonical metric replay mismatch: {path}')
            require(sorted(s['heldSourceGroup'] for s in row['selections']) == contract['groups'], 'Missing/duplicate outer selections')
            for selection in row['selections']:
                require(selection['epoch'] in contract['checkpointEpochs'] and selection['decoder'] in contract['decoderCandidates'],
                        'Selection outside frozen grid')
                expanded.selection_feasible(selection, .95)
    audit = audit_fit_files(study, reference_path, manifest, reg, old_reg, results, references)
    audit['counts'].update({'exactResultTargetRevisionsVerified': len(results) + len(references),
                            'canonicalEvaluationsReplayedExactly': len(results) + len(references),
                            'frozenSourcesVerified': len(inputs['frozenSources']), 'featureCachesVerified': len(inputs['featureCaches']),
                            'weightedSelectedInnerCandidatesReplayed': expanded.audit_selected_inner_candidates(study, exact, contract, results)})
    per_seed, aggregates, comparisons, losses = summarize_results(results, references, contract)
    choices = [selection for row in results for selection in row['selections']]
    audit['selectionCounts'] = {'epochs': dict(Counter(s['epoch'] for s in choices)),
                                'thresholds': dict(Counter(s['decoder']['enter'] for s in choices)),
                                'boundaryEnabled': sum(s['decoder']['boundary'] for s in choices),
                                'infeasible': sum(not s['recallEligibilityPassed'] for s in choices)}
    summary = {'schemaVersion': 1, 'kind': 'neural-event-balanced-summary-v1',
               'createdAt': datetime.now(timezone.utc).isoformat(), 'status': 'completed-event-balanced-audit',
               'protectedTestOpened': False, 'productionPromotionAllowed': False,
               'contractSha256': reg['sha256'], 'manifestSha256': contract['manifestSha256'],
               'scope': {'exactRecordings': len(exact['recordings']), 'exactRallies': sum(len(r['rallies']) for r in exact['recordings']),
                         'exactSourceGroups': contract['groups'], 'seeds': contract['seeds'], 'auxiliaryRows': 10},
               'primaryMetric': 'F1_padP_coreR', 'primaryPaddingSeconds': 2, 'joinGapSeconds': 3,
               'aggregation': 'Within each seed, pool recording numerators and denominators. Means summarize separate seed runs, not independent samples or confidence intervals.',
               'retentionRecoveryScreenContract': contract['retentionRecoveryScreen'],
               'limitations': [
                   'Adaptive development evidence on previously inspected groups, not a protected test or production promotion.',
                   'Per-rally weighting changes positive live-loss allocation only; masks, sampling and the original valid-tick denominator remain fixed. Preservation of full-recording positive mass does not imply identical sampled minibatch mass.',
                   'Historical TCNs are paired by cohort and seed; each run selects settings using the same inner-only rule. Refit checkpoint lengths may differ; matched sample exposure is verified over their common prefix.',
                   'Recovery is a separately preregistered descriptive screen. Neither it nor the historical F1 screen changes inner selection, primary ranking, or product padding.',
                   'Point loss uses original rally identities and canonical 2s padded export coverage. Duration slices overlap; tags are inherited and may be incomplete.',
                   'All source-group metrics are reported. Recovery has no source-group improvement gate; outer .95 recall is descriptive only.',
                   'Complete and partial losses must be considered together. A partial loss becoming complete is not an improvement.',
                   'Runtime describes desktop training. It does not establish phone/browser inference performance.',
               ], 'inputs': inputs, 'audit': audit, 'aggregates': aggregates, 'perSeed': per_seed, 'comparisons': comparisons}
    return summary, {'scope': summary['scope'], 'perSeed': losses}


def markdown(summary):
    lines = ['# Event-balanced neural development summary', '',
             'Weighted TCNs are paired with the immutable unweighted TCN of the same cohort and seed. '
             'All evaluate the same exact labels. Primary padding is fixed at 2s, with strictly <3s gap joining '
             'followed by ignored subtraction. Means describe separate seed runs; no production promotion.', '',
             '| Cohort | Weighted F1 | Reference F1 | Weighted core recall | Reference core recall | Event F1 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        new, old = row['meanSeedPrimary'], row['meanSeedHistoricalPrimary']
        lines.append(f"| {row['cohort']} | {new['F1_padP_coreR']:.6f} | {old['F1_padP_coreR']:.6f} | "
                     f"{new['R_core']:.6f} | {old['R_core']:.6f} | {row['meanSeedEventF1']:.6f} |")
    lines += ['', '| Cohort | F1 delta | R_core delta | Standard F1 screen | Recovery screen | Inner feasible |',
              '|---|---:|---:|---|---|---|']
    for row in summary['comparisons']:
        delta, recovery = row['meanSeedPrimaryDelta'], row['retentionRecoveryScreen']
        lines.append(f"| {row['name']} | {delta['F1_padP_coreR']:+.6f} | {delta['R_core']:+.6f} | "
                     f"{row['standardF1Screen']['passed']} | {recovery['passed']} | {recovery['allCandidateInnerSelectionsFeasible']} |")
    lines += ['', 'Recovery requires no more than .005 mean F1/core-recall regression, at least 20% fewer complete '
              'losses overall and among <=3s points, no increase in incomplete points in either slice, strict complete-loss '
              'reductions in at least two seeds for both slices, no more than .005 >3s core-recall regression and .01 event-F1 '
              'regression. Source groups and absolute outer recall are descriptive, not extra gates.', '',
              '| Cohort | Slice | Weighted complete / partial | Reference complete / partial | Weighted slice recall |',
              '|---|---|---:|---:|---:|']
    for row in summary['aggregates']:
        for name in ('all', 'duration_le_2s', 'duration_le_3s', 'duration_gt_3s', 'ace', 'service_fault'):
            new = row['meanSeedSlices']['primaryExportCoverage'][name]
            old = row['meanSeedHistoricalSlices']['primaryExportCoverage'][name]
            recall = f"{new['coreRecall']:.6f}" if new['coreRecall'] is not None else 'n/a'
            lines.append(f"| {row['cohort']} | {name} | {new['completeRallyLosses']:.2f} / {new['partialRallyLosses']:.2f} | "
                         f"{old['completeRallyLosses']:.2f} / {old['partialRallyLosses']:.2f} | {recall} |")
    lines += ['', 'All padding cases use the same predictions; only the declared 2s case is primary.', '',
              '| Cohort | Run | Padding | P_pad | R_core | F1_padP_coreR | Model seconds | Human seconds | Difference |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        for label, key in (('weighted', 'meanSeedPadding'), ('historical', 'meanSeedHistoricalPadding')):
            for metric in row[key]:
                lines.append(f"| {row['cohort']} | {label} | {metric['paddingSecondsBeforeAndAfter']} | {metric['P_pad']:.6f} | "
                             f"{metric['R_core']:.6f} | {metric['F1_padP_coreR']:.6f} | {metric['paddedModelExportSeconds']:.3f} | "
                             f"{metric['paddedHumanExportSeconds']:.3f} | {metric['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ['', 'Audit counts:', '', '```json', json.dumps(summary['audit']['counts'], indent=2), '```', '',
              'Per-seed/group details, weight diagnostics and exact loss identities are in the JSON artifacts.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'study')
    parser.add_argument('--manifest', type=Path, default=EXPANDED / 'manifest.json')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    filenames = ('summary.json', 'summary.md', 'loss-identities.json')
    if not args.progress:
        require(not any((args.study / name).exists() for name in filenames), 'Refusing to overwrite existing summary artifacts')
    summary, losses = summarize(args.study, args.manifest, progress=args.progress)
    if args.progress:
        print(json.dumps(summary, indent=2, allow_nan=False))
        return
    summary['lossIdentitiesArtifact'] = str(args.study / 'loss-identities.json')
    for name, content in (('summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n'),
                          ('summary.md', markdown(summary)), ('loss-identities.json', json.dumps(losses, indent=2, allow_nan=False) + '\n')):
        with (args.study / name).open('x', encoding='utf-8') as stream:
            stream.write(content)
    print(json.dumps({'status': summary['status'], 'output': str(args.study), 'audit': summary['audit']['counts'],
                      'artifacts': {name: identity(args.study / name) for name in filenames}}, indent=2))


if __name__ == '__main__':
    main()
