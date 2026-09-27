#!/usr/bin/env python3
"""Audit bounded short-event loss boosts and their transfer across TCN inputs.

Comparisons remain within architecture. Difference-in-differences describes
effect modification; positive interaction is not required for replication.
Historical compact baselines retain explicit origin metadata and original fits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0084'))
EXPANDED = Path(private_value('private-reference-0070'))
from analysis.neural_evaluation import evaluate_predictions


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


balanced = import_script('boost_transfer_event_helpers', 'summarize-neural-event-balanced.py')
expanded = balanced.expanded
helpers = balanced.helpers
require, identity, mean = balanced.require, balanced.identity, balanced.mean
COHORTS = expanded.COHORTS
KINDS = ('tcn', 'dino_tcn')
ARMS = ('baseline', 'short_boost', 'global_control')
CONTRASTS = (('short_boost', 'baseline'), ('short_boost', 'global_control'), ('global_control', 'baseline'))
SCOPES = balanced.SCOPES
PTS_DECODER = 'sequential-opencv-nearest-media-pts-v1'


def key(row):
    return row['cohort'], row['kind'], row['lossArm'], row['seed']


def expected_loss_weights(row, tier, arm):
    """Independent reconstruction from frozen timestamps and original intervals."""
    require(arm in ARMS and tier in ('exact', 'draft', 'coverage'), 'Unknown loss arm/tier')
    with np.load(row['featureCaches']['audiovisual']['path'], allow_pickle=False) as data:
        times = np.asarray(data['times'], dtype=np.float64)
    valid = np.ones(len(times), dtype=bool)
    ignored = row.get('ignoredIntervals', [])
    for item in ignored:
        valid[(times >= item['start']) & (times < item['end'])] = False
    if tier == 'coverage':
        valid &= (times >= row['gameWindow']['start']) & (times < row['gameWindow']['end'])
    supervised = valid.copy() if tier != 'coverage' else np.zeros(len(times), dtype=bool)
    truth = row.get('rallies', [])
    if tier == 'draft':
        for item in [*truth, *ignored]:
            for edge in ('start', 'end'):
                supervised[np.abs(times - item[edge]) <= 1.] = False
    positive = np.zeros(len(times), dtype=bool)
    short = np.zeros(len(times), dtype=bool)
    events = []
    for index, item in enumerate(truth):
        inside = (times >= item['start']) & (times < item['end'])
        ticks = inside & supervised
        require(not np.any(positive & ticks), 'Original positive events overlap')
        positive |= ticks
        is_short = item['end'] - item['start'] <= 3.
        if is_short:
            short |= ticks
        events.append({'eventIndex': index, 'start': float(item['start']), 'end': float(item['end']),
                       'tags': item.get('tags', []), 'originalDurationSeconds': item['end']-item['start'],
                       'isShortOriginalEvent': is_short, 'positiveSupervisedTicks': int(ticks.sum()),
                       'invalidTicksInsideEvent': int((inside & ~valid).sum()),
                       'maskedValidTicksInsideEvent': int((inside & valid & ~supervised).sum())})
    p, s = int(positive.sum()), int(short.sum())
    factor = 1. + s / p if p else 1.
    weights = np.ones(len(times), dtype='<f4')
    if arm == 'short_boost':
        weights[short] = 2.
    elif arm == 'global_control':
        weights[positive] = factor
    weighted = float(weights[positive].sum(dtype=np.float64))
    for event in events:
        selected = positive & (times >= event['start']) & (times < event['end'])
        event['multiplier'] = float(weights[selected][0]) if selected.any() else None
        event['weightedPositiveMass'] = float(weights[selected].sum(dtype=np.float64))
    expected_mass = p + (s if arm != 'baseline' else 0)
    return weights, {'recordingId': row['id'], 'sourceGroup': row['sourceGroup'], 'tier': tier,
                     'method': 'fixed-short-rally-live-boost-with-record-matched-control-v1', 'mode': arm,
                     'shortDurationSeconds': 3., 'shortPositiveMultiplier': 2.,
                     'originalEventCount': len(events), 'eligibleEventCount': sum(e['positiveSupervisedTicks'] > 0 for e in events),
                     'zeroSupervisedEventCount': sum(e['positiveSupervisedTicks'] == 0 for e in events),
                     'negativeSupervisedTicks': int((valid & supervised & ~positive).sum()),
                     'positiveSupervisedTicks': p, 'shortPositiveSupervisedTicks': s,
                     'longPositiveSupervisedTicks': p-s, 'idealGlobalPositiveMultiplier': factor,
                     'globalPositiveMultiplierFloat32': float(np.float32(factor)),
                     'shortOriginalEventCount': sum(e['isShortOriginalEvent'] for e in events),
                     'eligibleShortEventCount': sum(e['isShortOriginalEvent'] and e['positiveSupervisedTicks'] > 0 for e in events),
                     'zeroSupervisedShortEventCount': sum(e['isShortOriginalEvent'] and e['positiveSupervisedTicks'] == 0 for e in events),
                     'weightedPositiveMass': weighted, 'expectedPositiveMass': expected_mass,
                     'positiveMassRoundingError': weighted-expected_mass,
                     'positiveMassAbsoluteTolerance': p * float(np.finfo(np.float32).eps),
                     'minimumPositiveMultiplier': float(weights[positive].min()) if p else None,
                     'maximumPositiveMultiplier': float(weights[positive].max()) if p else None,
                     'liveMultiplierSha256': hashlib.sha256(weights.tobytes(order='C')).hexdigest(),
                     'liveMultiplierHashEncoding': 'little-endian float32, C-order, one value per original cache tick',
                     'events': events}


def primary_delta(candidate, reference):
    helpers.assert_comparable(candidate, reference)
    return expanded.paired_evaluation(candidate, reference)


def recovery_effect(candidate, reference, slice_name='all'):
    """Positive values mean fewer candidate losses; recall gains keep their sign."""
    a = balanced.event_slices(candidate)['primaryExportCoverage'][slice_name]
    b = balanced.event_slices(reference)['primaryExportCoverage'][slice_name]
    return {'completeLossRecovery': b['completeRallyLosses'] - a['completeRallyLosses'],
            'incompleteLossRecovery': b['incompleteRallyLosses'] - a['incompleteRallyLosses'],
            'coreRecallGain': a['coreRecall'] - b['coreRecall'] if a['coreRecall'] is not None and b['coreRecall'] is not None else None,
            'baselineCompleteLosses': b['completeRallyLosses'], 'candidateCompleteLosses': a['completeRallyLosses'],
            'fractionOfBaselineCompleteLossesRecovered': ((b['completeRallyLosses'] - a['completeRallyLosses']) / b['completeRallyLosses']
                                                        if b['completeRallyLosses'] else None)}


def contrast(cohort, kind, arm, reference_arm, by_key, contract):
    candidates = [by_key[(cohort, kind, arm, seed)] for seed in contract['seeds']]
    references = [by_key[(cohort, kind, reference_arm, seed)] for seed in contract['seeds']]
    pairs = []
    for candidate, reference in zip(candidates, references):
        a, b = candidate['evaluation'], reference['evaluation']
        pairs.append({'seed': candidate['seed'], **primary_delta(a, b),
                      'sliceDelta': balanced.slice_delta(balanced.event_slices(a), balanced.event_slices(b)),
                      'sourceGroupSlices': {group: balanced.slice_delta(balanced.event_slices(a['sourceGroups'][group]),
                                                                        balanced.event_slices(b['sourceGroups'][group]))
                                            for group in contract['groups']},
                      'pointRecovery': {name: recovery_effect(a, b, name) for name in balanced.short.SLICES}})
    return {'cohort': cohort, 'kind': kind, 'candidateArm': arm, 'referenceArm': reference_arm,
            'name': f'{cohort}/{kind}/{arm}-minus-{reference_arm}', 'perSeed': pairs,
            'meanSeedPrimaryDelta': expanded.metric_means([r['primaryDelta'] for r in pairs]),
            'meanSeedEventF1Delta': mean(r['eventF1Delta'] for r in pairs),
            'meanSeedSliceDelta': balanced.mean_slices([r['sliceDelta'] for r in pairs]),
            'sourceGroupMeanPairedDelta': {group: {field: mean(r['sourceGroups'][group][field] for r in pairs)
                                                 for field in ('F1_padP_coreRDelta', 'R_coreDelta', 'paddedModelExportSecondsDelta')}
                                         for group in contract['groups']},
            'standardF1Screen': expanded.feasibility(candidates, pairs, .95),
            'retentionRecoveryScreen': balanced.recovery_screen(candidates, references, contract['retentionRecoveryScreen'])}


def effect_metrics(candidate, reference):
    assert_effect_comparable(candidate, reference)
    values = {name: candidate['primary'][name] - reference['primary'][name]
              for name in ('F1_padP_coreR', 'R_core', 'P_pad', 'paddedModelExportSeconds')}
    values['eventF1'] = candidate['guardrails']['eventF1'] - reference['guardrails']['eventF1']
    for name in balanced.short.SLICES:
        recovery = recovery_effect(candidate, reference, name)
        for field in ('completeLossRecovery', 'incompleteLossRecovery', 'coreRecallGain'):
            values[f'{name}/{field}'] = recovery[field]
    return values


def assert_effect_comparable(left, right):
    """Support canonical pooled reports and their nested source-group summaries.

    Group summaries omit the top-level recording/metric-contract envelopes.
    Their parent predictions undergo exact gold/ignored revision verification
    before aggregation; also require identical group gold and padding here.
    """
    if 'metricContract' in left or 'metricContract' in right:
        helpers.assert_comparable(left, right)
        return
    require((left['primary']['paddingSecondsBeforeAndAfter'], left['primary']['joinGapSeconds'])
            == (right['primary']['paddingSecondsBeforeAndAfter'], right['primary']['joinGapSeconds']), 'Group primary configuration differs')
    require(len(left['padding']) == len(right['padding']) == 4, 'Group padding cases differ')
    for a, b in zip(left['padding'], right['padding']):
        require((a['paddingSecondsBeforeAndAfter'], a['joinGapSeconds'])
                == (b['paddingSecondsBeforeAndAfter'], b['joinGapSeconds']), 'Group padding contract differs')
        gold = lambda value: {(r['id'], r['coreHumanSeconds'], r['paddedHumanExportSeconds']) for r in value['recordings']}
        require(gold(a) == gold(b), 'Group recording/gold scope differs')
    gold_events = lambda value: {(r['recordingId'], r['truthIndex'], r['start'], r['end'], r['evaluableCoreSeconds'], tuple(r['tags']))
                                 for r in value['guardrails']['primaryExportCoverage']['rallies']}
    require(gold_events(left) == gold_events(right), 'Group event identities differ')


def difference_in_differences(dino_candidate, dino_reference, compact_candidate, compact_reference):
    """DINO within-arm effect minus compact within-arm effect; no raw score gap."""
    assert_effect_comparable(dino_reference, compact_reference)
    dino = effect_metrics(dino_candidate, dino_reference)
    compact = effect_metrics(compact_candidate, compact_reference)
    return {'dinoWithinArchitectureEffect': dino, 'compactWithinArchitectureEffect': compact,
            'differenceInDifferences': {key: dino[key] - compact[key] if dino[key] is not None and compact[key] is not None else None
                                       for key in dino}}


def replication_screen(comparisons, cohort):
    relevant = [row for row in comparisons if row['cohort'] == cohort and row['candidateArm'] == 'short_boost']
    require({(r['kind'], r['referenceArm']) for r in relevant} == {(kind, ref) for kind in KINDS for ref in ('baseline', 'global_control')}
            and len(relevant) == 4, 'Replication needs exactly four within-architecture short-boost comparisons')
    checks = {f"{r['kind']}/versus-{r['referenceArm']}": r['retentionRecoveryScreen']['screenPassedAndInnerFeasible'] for r in relevant}
    return {'passed': all(checks.values()), 'checks': checks, 'positiveDifferenceInDifferencesRequired': False,
            'productionPromotionAllowed': False,
            'role': 'replication requires feasible retention recovery versus both own controls in both architectures; descriptive DiD is not a gate'}


def transfer_summary(cohort, by_key, comparisons, contract):
    contrasts = []
    for reference_arm in ('baseline', 'global_control'):
        seeds = []
        for seed in contract['seeds']:
            lookup = lambda kind, arm: by_key[(cohort, kind, arm, seed)]['evaluation']
            evaluations = (lookup('dino_tcn', 'short_boost'), lookup('dino_tcn', reference_arm),
                           lookup('tcn', 'short_boost'), lookup('tcn', reference_arm))
            seeds.append({'seed': seed, **difference_in_differences(*evaluations),
                          'sourceGroups': {group: difference_in_differences(*(e['sourceGroups'][group] for e in evaluations))
                                           for group in contract['groups']},
                          'baselinePointLosses': {kind: {name: recovery_effect(lookup(kind, 'short_boost'), lookup(kind, reference_arm), name)
                                                         for name in balanced.short.SLICES} for kind in KINDS}})
        fields = seeds[0]['differenceInDifferences']
        contrasts.append({'referenceArm': reference_arm, 'perSeed': seeds,
                          'meanSeedDifferenceInDifferences': {field: expanded.mean_available(r['differenceInDifferences'][field] for r in seeds) for field in fields},
                          'sourceGroupMeanDifferenceInDifferences': {
                              group: {field: expanded.mean_available(r['sourceGroups'][group]['differenceInDifferences'][field] for r in seeds)
                                      for field in fields} for group in contract['groups']}})
    return {'cohort': cohort, 'replicationScreen': replication_screen(comparisons, cohort), 'contrasts': contrasts,
            'interpretation': 'Paired within-architecture effects only. Positive recovery means fewer lost points. Three seeds and four groups are descriptive, not independent population evidence. Zero interaction can accompany equal benefit in both architectures.'}


def summarize_results(results, contract):
    by_key = {key(row): row for row in results}
    per_seed, aggregates, losses = [], [], []
    for cohort in COHORTS:
        for kind in KINDS:
            for arm in ARMS:
                rows = []
                for seed in contract['seeds']:
                    result = by_key[(cohort, kind, arm, seed)]
                    evaluation = result['evaluation']
                    compact = {'cohort': cohort, 'kind': kind, 'lossArm': arm, 'seed': seed,
                               'origin': result['origin'], **expanded.compact_evaluation(evaluation),
                               'slices': balanced.event_slices(evaluation),
                               'sourceGroups': {group: {**expanded.compact_evaluation(value), 'slices': balanced.event_slices(value)}
                                                for group, value in evaluation['sourceGroups'].items()},
                               'selections': result['selections']}
                    per_seed.append(compact)
                    rows.append(compact)
                    refs = {ref: by_key[(cohort, kind, ref, seed)]['evaluation'] for ref in ('baseline', 'global_control')}
                    losses.append({'cohort': cohort, 'kind': kind, 'lossArm': arm, 'seed': seed,
                                   'slices': balanced.event_slices(evaluation, identities=True),
                                   'pairedReferences': {ref: {scope: helpers.coverage_comparison(evaluation, old, scope)
                                                             for scope in SCOPES} for ref, old in refs.items()}})
                aggregates.append({'cohort': cohort, 'kind': kind, 'lossArm': arm, 'seedCount': len(rows),
                                   'meanSeedPrimary': expanded.metric_means([r['primary'] for r in rows]),
                                   'meanSeedPadding': [{'paddingSecondsBeforeAndAfter': pad,
                                                        **expanded.metric_means([r['padding'][pad] for r in rows])} for pad in range(4)],
                                   'meanSeedEventF1': mean(r['eventF1'] for r in rows),
                                   'meanSeedSlices': balanced.mean_slices([r['slices'] for r in rows])})
    comparisons = [contrast(cohort, kind, arm, reference, by_key, contract)
                   for cohort in COHORTS for kind in KINDS for arm, reference in CONTRASTS]
    transfer = [transfer_summary(cohort, by_key, comparisons, contract) for cohort in COHORTS]
    return per_seed, aggregates, comparisons, transfer, losses


def validate_origin(result, study, reference, historical_by_key):
    cohort, kind, arm, seed = key(result)
    require(result['architecture'] == kind, 'Architecture alias differs from kind')
    origin = result['origin']
    if kind == 'tcn' and arm == 'baseline' and cohort in ('exact', 'draft'):
        expected = {'type': 'reused-reference', 'studyPath': reference['path'],
                    'reportSha256': reference['reportSha256'], 'referenceContractSha256': reference['contractSha256'],
                    'fitRoot': str(Path(reference['path']) / 'fits' / cohort / kind / str(seed))}
        require(origin == expected, 'Historical baseline origin differs from immutable reference')
        historical = historical_by_key[(cohort, kind, seed)]
        require(all(result[field] == value for field, value in historical.items()), 'Reused baseline payload changed')
        require(not (study / 'fits' / cohort / kind / arm / str(seed)).exists(), 'Reused baseline has fabricated local fits')
    else:
        require(origin == {'type': 'trained', 'fitRoot': str(study / 'fits' / cohort / kind / arm / str(seed))},
                'Fresh fit origin does not match canonical result identity')
    return Path(origin['fitRoot'])


def audit_feature_revision(original, amended):
    """Only seven training-only coverage AV caches may change before fitting."""
    require(set(amended) == set(original) | {'originalManifest', 'protocolAmendment', 'featureRevision'},
            'Unexpected amended manifest fields')
    for field, value in original.items():
        if field != 'coverageRows':
            require(amended[field] == value, f'Feature repair changed frozen manifest field: {field}')
    old_rows, new_rows = original['coverageRows'], amended['coverageRows']
    require(len(old_rows) == len(new_rows) == 7 and [r['id'] for r in old_rows] == [r['id'] for r in new_rows],
            'Feature repair changed coverage recording identities')
    changed = []
    for old, new in zip(old_rows, new_rows):
        require({k: v for k, v in old.items() if k != 'featureCaches'}
                == {k: v for k, v in new.items() if k != 'featureCaches'}, 'Feature repair changed labels/source identity')
        require(set(old['featureCaches']) == set(new['featureCaches']) == {'audiovisual'}, 'Unexpected coverage feature families')
        before, after = old['featureCaches']['audiovisual'], new['featureCaches']['audiovisual']
        require(before['path'] != after['path'] and before['sha256'] != after['sha256'], 'Coverage cache was not replaced immutably')
        for field in ('featureNames', 'names'):
            require(before[field] == after[field] and len(after[field]) == 104, 'AV feature schema changed')
        changed.append({'recordingId': old['id'], 'sourceGroup': old['sourceGroup'],
                        'originalCache': {'path': before['path'], 'sha256': before['sha256']},
                        'repairedCache': {'path': after['path'], 'sha256': after['sha256']}})
    return {'changedCoverageRecordings': changed, 'exactAndDraftRowsUnchanged': True,
            'exactEvaluationManifestUnchanged': True, 'labelsGroupsAndIgnoredIntervalsUnchanged': True,
            'historicalBaselineReuse': 'Only exact/draft compact TCN baselines; reviewed-export compact baseline is freshly trained.'}


def read_inputs(study, manifest_path, *, progress=False):
    reg = balanced.registration(study / 'preregistration.json')
    contract = reg['contract']
    require(contract['cohorts'] == list(COHORTS) and set(contract['kinds']) == set(KINDS)
            and set(contract['lossArms']) == set(ARMS), 'Unexpected study cells')
    require(contract['retentionRecoveryScreen'] == balanced.RECOVERY_CONTRACT, 'Recovery screen changed')
    transfer = contract['transferScreen']
    require(set(transfer['requireRecoveryPass']) == {'short_boost-minus-baseline', 'short_boost-minus-global_control'}
            and set(transfer['architectures']) == set(KINDS) and transfer['sameCohortRequired'] is True
            and transfer['innerSelectionsMustBeFeasible'] is True
            and transfer['promotionAllowed'] is False and 'descriptive' in transfer['differenceInDifferences'],
            'Unexpected transfer screen')
    weighting = contract['liveLossWeighting']
    require(weighting['shortDurationSeconds'] == 3 and weighting['shortPositiveMultiplier'] == 2
            and weighting['scope'] == ['exact', 'draft'], 'Boost definition changed')
    reference = contract['referenceStudy']
    require(reference['baselineReuseCohorts'] == ['exact', 'draft'], 'Historical baseline reuse scope changed')
    old_root = Path(reference['path'])
    for filename, field in (('preregistration.json', 'preregistrationFileSha256'), ('report.json', 'reportSha256'), ('summary.json', 'summarySha256')):
        require(helpers.digest(old_root / filename) == reference[field], f'Historical reference changed: {filename}')
    old_reg = balanced.registration(old_root / 'preregistration.json')
    require(old_reg['sha256'] == reference['contractSha256'], 'Historical contract mismatch')
    for field in ('cohorts', 'seeds', 'checkpointEpochs', 'groups', 'primaryMetric',
                  'targetPaddingSeconds', 'joinGapSeconds', 'paddingSweep', 'decoderCandidates', 'selection', 'screen', 'environment'):
        require(contract[field] == old_reg['contract'][field], f'Comparison changed {field}')
    require({k: v for k, v in contract['training'].items() if k != 'loss'}
            == {k: v for k, v in old_reg['contract']['training'].items() if k != 'loss'}, 'Other training recipe changed')
    require(contract['primaryMetric'] == 'F1_padP_coreR' and contract['targetPaddingSeconds'] == 2
            and contract['joinGapSeconds'] == 3 and contract['paddingSweep'] == [0, 1, 2, 3], 'Metric contract changed')
    require(len(contract['groups']) == 4 and len(contract['seeds']) == 3, 'Unexpected experiment size')
    require(helpers.digest(manifest_path) == contract['manifestSha256'], 'Expanded manifest changed')
    manifest = helpers.load(manifest_path)
    original_path = Path(contract['originalManifest']['path'])
    require(helpers.digest(original_path) == contract['originalManifest']['sha256'] == old_reg['contract']['manifestSha256'],
            'Original historical manifest changed')
    feature_revision = audit_feature_revision(helpers.load(original_path), manifest)
    require(manifest['originalManifest'] == contract['originalManifest']
            and manifest['protocolAmendment'] == contract['protocolAmendment']
            and manifest['featureRevision'] == contract['featureRevision'], 'Pretraining feature-amendment binding differs')
    require(set(contract['featureRevision']['recordingIds']) == {r['id'] for r in manifest['coverageRows']},
            'Feature repair scope differs from seven coverage recordings')
    exact_path = Path(manifest['exactManifest']['path'])
    require(helpers.digest(exact_path) == manifest['exactManifest']['sha256'], 'Exact manifest changed')
    exact = helpers.load(exact_path)
    require(exact['recordings'] == manifest['exactRows'], 'Exact evaluation gold differs')
    rows = manifest['exactRows'] + manifest['draftRows'] + manifest['coverageRows']
    require(len(rows) == len({r['id'] for r in rows}) == 18, 'Duplicate/missing original recordings')
    require(not {r['sourceGroup'] for r in rows}.intersection(manifest['protectedSourceGroups']), 'Protected group included')
    require(all(r['environment'] in ('grass', 'indoor') and r['consent']['train'] for r in rows), 'Scope or consent differs')
    require(contract['protocolSnapshot']['sha256'] == 'f8a75a3d6a6d61fd12efca6790e942906bac3a9cb7c9e00ec2c79fd65d7a0826',
            'Initial pre-candidate protocol identity differs')
    require(contract['protocolAmendment']['sha256'] == '6e6d9733abe7c3b020c64afe3b0df1653a2d62dd565162bdd1912beef6b06d1a',
            'Pre-candidate PTS amendment identity differs')
    for field in ('dinoManifest', 'preflight', 'protocolSnapshot', 'protocolAmendment'):
        require(helpers.digest(Path(contract[field]['path'])) == contract[field]['sha256'], f'{field} changed')
    amendment = helpers.load(Path(contract['protocolAmendment']['path']))
    require(amendment['originalManifest'] == contract['originalManifest']
            and amendment['originalProtocol'] == contract['protocolSnapshot']
            and amendment['featureRevision'] == contract['featureRevision']
            and amendment['compactBaselineReuseCohorts'] == reference['baselineReuseCohorts'],
            'Amendment does not bind the original dataset/protocol and declared feature repair')
    require(amendment['candidateTrainingStarted'] is False and amendment['protectedTestOpened'] is False
            and amendment['lossAndEvaluationRulesChanged'] is False, 'Amendment was not declared as pretraining data-only repair')
    require(helpers.digest(Path(amendment['document']['path'])) == amendment['document']['sha256'], 'Amendment document changed')
    preflight = helpers.load(Path(contract['preflight']['path']))
    require(preflight['passed'] is True and preflight['code'] == contract['code'], 'Preflight does not bind registered sources')
    require(preflight['manifest']['sha256'] == contract['manifestSha256']
            and preflight['dinoManifest']['sha256'] == contract['dinoManifest']['sha256'], 'Preflight input binding changed')
    old_report = helpers.load(old_root / 'report.json')
    old_summary = helpers.load(old_root / 'summary.json')
    require(old_report['contractSha256'] == old_reg['sha256'] and old_report['manifestSha256'] == contract['originalManifest']['sha256']
            and old_summary['status'] == 'completed-expanded-development-audit' and old_summary['contractSha256'] == old_reg['sha256'],
            'Historical audited reference differs')
    old_by_key = {expanded.result_key(row): row for row in old_report['results']}
    require(len(old_by_key) == len(old_report['results']) == 18, 'Historical result inventory differs')
    report_path = study / 'report.json'
    report = helpers.load(report_path) if report_path.exists() else {'results': [helpers.load(p) for p in sorted(study.glob('result-*.json'))]}
    expected = {(c, k, a, s) for c in COHORTS for k in KINDS for a in ARMS for s in contract['seeds']}
    observed = {key(r) for r in report['results']}
    require(len(observed) == len(report['results']) and observed <= expected, 'Duplicate/unexpected result')
    if not progress:
        require(report_path.exists() and observed == expected, 'Final summary requires all54 result cells')
        require(report['status'] == 'completed-short-boost-transfer-development', 'Report is not complete')
    if report_path.exists():
        require(report['contractSha256'] == reg['sha256'] and report['manifestSha256'] == contract['manifestSha256'], 'Report provenance differs')
        require(report['records'] == len(exact['recordings']) and report['sourceGroups'] == contract['groups'], 'Report evaluation population differs')
        require(not report['protectedTestOpened'] and not report['productionPromotionAllowed'], 'Unexpected protected/promotion status')
    for row in report['results']:
        require(row['contractSha256'] == reg['sha256'], 'Result contract differs')
        validate_origin(row, study, reference, old_by_key)
    inputs = {'registration': identity(study / 'preregistration.json'), 'manifest': identity(manifest_path),
              'originalManifest': identity(original_path), 'featureRevisionAudit': feature_revision,
              'exactManifest': identity(exact_path), 'referenceRegistration': identity(old_root / 'preregistration.json'),
              'referenceReport': identity(old_root / 'report.json'), 'referenceSummary': identity(old_root / 'summary.json'),
              'dinoManifest': identity(Path(contract['dinoManifest']['path'])), 'preflight': identity(Path(contract['preflight']['path'])),
              'protocolSnapshot': identity(Path(contract['protocolSnapshot']['path'])),
              'protocolAmendment': identity(Path(contract['protocolAmendment']['path'])),
              'protocolAmendmentDocument': identity(Path(amendment['document']['path'])),
              'summarySources': [identity(Path(__file__)), identity(REPO / 'scripts/summarize-neural-event-balanced.py'),
                                 identity(REPO / 'scripts/summarize-neural-expanded.py'), identity(expanded.HELPER_PATH),
                                 identity(REPO / 'scripts/analyze-neural-expanded-short-events.py')]}
    if report_path.exists():
        inputs['report'] = identity(report_path)
    return reg, old_reg, manifest, exact, report, inputs


def alignment_diagnostics(av_times, dino_times):
    require(av_times.ndim == dino_times.ndim == 1 and len(av_times) and len(dino_times)
            and np.isfinite(av_times).all() and np.isfinite(dino_times).all()
            and (np.diff(av_times) > 0).all() and (np.diff(dino_times) > 0).all(), 'Invalid feature timeline')
    after = np.minimum(np.searchsorted(dino_times, av_times), len(dino_times)-1)
    before = np.maximum(after-1, 0)
    before_error, after_error = np.abs(dino_times[before]-av_times), np.abs(dino_times[after]-av_times)
    indexes = np.where(before_error <= after_error, before, after).astype('<i8')
    errors = np.abs(dino_times[indexes]-av_times)
    require(errors.max() <= .125+1e-8, 'DINO alignment exceeds half an AV tick')
    return {'rule': 'nearest DINO timestamp to each actual AV tick; earlier timestamp wins ties',
            'toleranceSeconds': .125, 'maximumErrorSeconds': float(errors.max()), 'meanErrorSeconds': float(errors.mean()),
            'avTicks': len(av_times), 'dinoTicks': len(dino_times),
            'tieCount': int(((before != after) & (before_error == after_error)).sum()),
            'avTimesSha256': hashlib.sha256(av_times.astype('<f8').tobytes()).hexdigest(),
            'dinoTimesSha256': hashlib.sha256(dino_times.astype('<f8').tobytes()).hexdigest(),
            'nearestIndexesSha256': hashlib.sha256(indexes.tobytes()).hexdigest()}


def canonical_nas_path(value):
    """Only the two explicitly verified mounts of the same NAS are aliases."""
    value = str(value)
    prefix = private_value('private-reference-0104')
    return private_value('private-reference-0105') + value[len(prefix):] if value.startswith(prefix) else value


def same_file_reference(left, right):
    return canonical_nas_path(left['path']) == canonical_nas_path(right['path']) and left['sha256'] == right['sha256']


def verify_pts_arrays(selection, arrays, av_times, dino_times):
    """Independently verify media-clock selection, without decoding video again."""
    fields = {'times', 'packetPtsTicks', 'presentationTimes', 'selectedOrdinals',
              'selectedPresentationTimes', 'observedPresentationTimes', 'selectedFrameSha256'}
    require(set(arrays) == fields, 'PTS selection array inventory differs')
    values = {key: np.asarray(arrays[key]) for key in fields}
    for key, value in values.items():
        require(value.ndim == 1 and len(value), 'Empty/non-vector PTS selection array')
        if key != 'selectedFrameSha256':
            require(np.isfinite(value).all(), 'Non-finite PTS selection')
    for key in ('times', 'presentationTimes', 'selectedPresentationTimes', 'observedPresentationTimes'):
        require(values[key].dtype == np.float64, 'Media time arrays must retain float64')
    for key in ('packetPtsTicks', 'selectedOrdinals'):
        require(values[key].dtype == np.int64, 'Media packet/ordinal arrays must retain int64')
    times, ticks, pts, indexes = (values[key] for key in ('times', 'packetPtsTicks', 'presentationTimes', 'selectedOrdinals'))
    require(selection['decoder'] == PTS_DECODER and selection['sampleCount'] == len(times)
            and selection['videoFrameCount'] == len(pts) == len(ticks), 'PTS count/decoder metadata differs')
    require(np.all(np.diff(ticks) > 0) and np.all(np.diff(pts) > 0) and abs(pts[0]) <= 1e-6,
            'Media timestamps must increase from the verified zero origin')
    numerator, denominator = (int(value) for value in selection['timeBase'].split('/'))
    require(numerator > 0 and denominator > 0 and np.array_equal(pts, ticks.astype(np.float64)*numerator/denominator),
            'Presentation timestamps differ from exact packet clock')
    duration = selection['videoDurationSeconds']
    require(np.isfinite(duration) and duration > 0, 'Invalid media duration')
    grid = np.arange(int(np.ceil(duration*4-1e-9)), dtype=np.float64)/4
    grid = grid[grid < duration]
    require(np.array_equal(times, grid) and np.array_equal(times, av_times) and np.array_equal(times, dino_times),
            'Repaired AV/DINO grids do not share exact media4Hz ticks')
    after = np.minimum(np.searchsorted(pts, times), len(pts)-1)
    before = np.maximum(after-1, 0)
    expected = np.where(np.abs(times-pts[before]) <= np.abs(pts[after]-times), before, after)
    require(np.array_equal(indexes, expected) and np.all(np.diff(indexes) > 0),
            'Selected media frame is not nearest with earlier ties')
    selected, observed = values['selectedPresentationTimes'], values['observedPresentationTimes']
    require(np.array_equal(selected, pts[indexes]) and observed.shape == times.shape, 'Selected/observed media PTS differ in shape or identity')
    error, decoded_error = float(np.abs(selected-times).max()), float(np.abs(observed-selected).max())
    require(error <= .125+1e-9 and decoded_error <= 2e-6
            and error == selection['maximumFrameSelectionErrorSeconds']
            and decoded_error == selection['maximumDecodedPtsErrorSeconds'], 'Media selection/decoded PTS errors exceed contract or metadata')
    frame_hashes = values['selectedFrameSha256'].tolist()
    require(len(frame_hashes) == len(times) and all(isinstance(h, str) and len(h) == 64 and set(h) <= set('0123456789abcdef') for h in frame_hashes),
            'Invalid selected-frame hashes')
    hashes = {'gridSha256': hashlib.sha256(times.astype('<f8').tobytes()).hexdigest(),
              'presentationTimesSha256': hashlib.sha256(pts.astype('<f8').tobytes()).hexdigest(),
              'selectedOrdinalsSha256': hashlib.sha256(indexes.astype('<i8').tobytes()).hexdigest(),
              'selectedPresentationTimesSha256': hashlib.sha256(selected.astype('<f8').tobytes()).hexdigest(),
              'frameHashesSha256': hashlib.sha256('\n'.join(frame_hashes).encode()).hexdigest()}
    require(all(selection[key] == value for key, value in hashes.items()), 'PTS vector/frame hashes differ')
    audio = selection['audio']
    require(audio['formulaUnchanged'] is True and audio['sampleRate'] == 16000
            and abs(audio['startOffsetSeconds']) <= .25
            and audio['startOffsetSeconds'] == float(audio['stream']['start_time'])
            and audio['alignmentSamples'] == round(audio['startOffsetSeconds']*audio['sampleRate']), 'Audio-origin alignment changed')
    require(abs(audio['stream']['firstDecodedFramePtsSeconds']-audio['startOffsetSeconds'])
            <= 1/float(audio['stream']['sample_rate']), 'Audio first decoded frame differs from verified origin')
    return {'samples': len(times), 'mediaFrames': len(pts), 'maximumSelectionErrorSeconds': error,
            'maximumDecodedPtsErrorSeconds': decoded_error, 'verifiedArrayHashes': hashes}


def audit_pts_selection(entry, row, metadata, av_times, dino_times, av_binding, repair_reference):
    binding = metadata['decoderSelection']
    require(binding == av_binding == row['featureCaches']['audiovisual']['decoderSelection']
            and binding['decoder'] == PTS_DECODER, 'AV and DINO do not bind the same selected frames')
    require(binding['selection']['path'] == entry['decoderSelectionPath']
            and binding['selection']['sha256'] == entry['decoderSelectionSha256'], 'DINO selection sidecar association differs')
    artifacts = []
    for field in ('selection', 'arrays'):
        item = binding[field]
        require(helpers.digest(Path(item['path'])) == item['sha256'] and Path(item['path']).stat().st_size == item['sizeBytes'],
                'PTS selection sidecar/array identity changed')
        artifacts.append(identity(Path(item['path'])))
    selection = helpers.load(Path(binding['selection']['path']))
    require(selection['recordingId'] == row['id'] and selection['source'] == entry['sourceVideoVerified']
            and selection['arrays'] == binding['arrays'] and same_file_reference(selection['repairPlan'], repair_reference),
            'PTS sidecar source, array or repair-plan association differs')
    with np.load(binding['arrays']['path'], allow_pickle=False) as data:
        check = verify_pts_arrays(selection, {key: data[key] for key in data.files}, av_times, dino_times)
    return {'recordingId': row['id'], **check, 'artifacts': artifacts}


def audit_extraction_dataset(dino_path, dino, manifest, contract):
    path = dino_path.parent / 'dataset-audit-pts-v1.json'
    audit = helpers.load(path)
    require(audit['passed'] is True and audit['records'] == 18 and audit['repairedCoveragePairs'] == 7
            and audit['unchangedExactDraftPairs'] == 11, 'Final extraction audit is incomplete')
    require(same_file_reference(audit['dinoManifest'], contract['dinoManifest'])
            and audit['manifest']['path'] == dino['expandedManifestPath']
            and audit['manifest']['sha256'] == contract['manifestSha256']
            and same_file_reference(audit['originalManifest'], contract['originalManifest'])
            and audit['repairPlan'] == dino['repairPlan'], 'Final extraction-audit input identity differs')
    retained_reference = audit['retainedInputAudit']
    retained_path = Path(retained_reference['path'])
    require(helpers.digest(retained_path) == retained_reference['sha256']
            == 'b22d1f31a0df7ddd3112007dfad4e5f144c3b6f00d5d5314733f73697209f5ea', 'Immutable retained-input decoder audit changed')
    retained = helpers.load(retained_path)
    require(retained['passed'] is True and retained['unchangedProxies'] == 11, 'Retained eleven input decoder check failed')
    retained_by_id = {r['recordingId']: r for r in retained['records']}
    require(len(retained_by_id) == len(retained['records']), 'Duplicate retained decoder-audit recording')
    for row in manifest['exactRows'] + manifest['draftRows']:
        check = retained_by_id[row['id']]
        require(same_file_reference(check['AV'], row['featureCaches']['audiovisual'])
                and np.isfinite(check['maximumAbsoluteClockErrorSeconds'])
                and 0 <= check['maximumAbsoluteClockErrorSeconds'] <= .125, 'Retained exact/draft AV source clock differs')
    records = {r['recordingId']: r for r in dino['records'] if r['tier'] == 'coverage'}
    coverage = {r['id']: r for r in manifest['coverageRows']}
    checked, artifacts = set(), []
    require(len(audit['recordAudits']) == 7, 'Missing repaired pair audits')
    for reference in audit['recordAudits']:
        record_path = Path(reference['path'])
        require(helpers.digest(record_path) == reference['sha256'], 'Repaired record audit changed')
        pair = helpers.load(record_path)
        rid = pair['record']['recordingId']
        require(rid in records and rid not in checked and pair['passed'] is True
                and pair['record'] == records[rid] and pair['audiovisual'] == coverage[rid]['featureCaches']['audiovisual'],
                'Repaired record audit differs from published AV/DINO association')
        require(pair['decoderSelection']['path'] == records[rid]['decoderSelectionPath']
                and pair['decoderSelection']['sha256'] == records[rid]['decoderSelectionSha256'], 'Record audit selection association differs')
        checked.add(rid)
        artifacts.append(identity(record_path))
    return {'datasetAudit': identity(path), 'retainedInputAudit': identity(retained_path), 'recordAudits': artifacts,
            'unchangedExactDraftInputsVerified': len(manifest['exactRows']) + len(manifest['draftRows']),
            'repairedCoveragePairsVerified': len(checked)}


def audit_dino_inputs(manifest, contract):
    path = Path(contract['dinoManifest']['path'])
    dino = helpers.load(path)
    require(dino['expandedManifestSha256'] == contract['manifestSha256'], 'DINO manifest targets a different dataset')
    plan_path = Path(dino['extractionPlan']['path'])
    require(helpers.digest(plan_path) == dino['extractionPlan']['sha256'], 'DINO extraction plan changed')
    plan = helpers.load(plan_path)
    require(plan['expandedManifestSha256'] == contract['originalManifest']['sha256'] and plan['labelsUsedForExtraction'] is False,
            'DINO extraction has different scope or used labels')
    require(len(plan['reproductionProbes']) == 6 and all(p['reproductionBitExact'] and p['maximumStoredTokenDifference'] == 0
                                                       for p in plan['reproductionProbes']), 'Historical extractor reproduction was not verified')
    for field in ('originalManifest', 'protocolAmendment', 'featureRevision'):
        require(dino[field] == contract[field], f'DINO {field} differs from repaired input contract')
    repair_reference = dino['repairPlan']
    repair_path = Path(repair_reference['path'])
    require(helpers.digest(repair_path) == repair_reference['sha256'], 'Media PTS repair plan changed')
    repair = helpers.load(repair_path)
    require(same_file_reference(repair['originalManifest'], contract['originalManifest'])
            and repair['originalExtractionPlan'] == dino['extractionPlan']
            and same_file_reference(repair['protocolAmendment'], contract['protocolAmendment'])
            and repair['recordingIds'] == contract['featureRevision']['recordingIds']
            and repair['decoder'] == PTS_DECODER, 'Repair plan scope/protocol differs')
    require(repair['extractorConfigSha256'] == dino['extractorConfigSha256'] == plan['extractorConfigSha256']
            and repair['semanticBackbone'] == dino['semanticBackbone'] == plan['semanticBackbone'], 'Repair changed frozen DINO semantics')
    require(dino['sourceCode'] == repair['sourceCode']
            and set(repair['sourceCode']) == set(plan['sourceCode']) | {'prepare-neural-transfer-pts.py'}
            and all(repair['sourceCode'][key] == value for key, value in plan['sourceCode'].items()),
            'DINO source provenance differs from original extractor plus explicit PTS repair')
    for entry in dino['sourceCode'].values():
        require(helpers.digest(Path(entry['path'])) == entry['sha256'], 'Extraction source changed')
    rows = {r['id']: (tier, r) for tier, name in (('exact', 'exactRows'), ('draft', 'draftRows'), ('coverage', 'coverageRows')) for r in manifest[name]}
    require(len(dino['records']) == len({r['recordingId'] for r in dino['records']}) == len(rows)
            and {r['recordingId'] for r in dino['records']} == set(rows), 'DINO recording identity mismatch')
    artifacts, alignments, pts_selections = [], {}, []
    for entry in dino['records']:
        tier, row = rows[entry['recordingId']]
        av = row['featureCaches']['audiovisual']
        require(entry['tier'] == tier and entry['sourceGroup'] == row['sourceGroup']
                and entry['audiovisualPath'] == av['path'] and entry['audiovisualSha256'] == av['sha256'], 'DINO AV/group association changed')
        require(entry['rawSourceContentSha256'] == row.get('sourceContentSha256', row['contentSha256']), 'DINO raw lineage differs')
        for field, sha_field in (('dinoPath', 'dinoSha256'), ('metadataPath', 'metadataSha256')):
            require(helpers.digest(Path(entry[field])) == entry[sha_field], 'DINO cache/sidecar changed')
            artifacts.append(identity(Path(entry[field])))
        sidecar = helpers.load(Path(entry['metadataPath']))
        require(sidecar['extractionPlan'] == dino['extractionPlan'], 'DINO sidecar original extractor-plan binding differs')
        require(sidecar['recordingId'] == row['id'] and sidecar['sourceVideoVerified'] == entry['sourceVideoVerified'], 'DINO sidecar recording association differs')
        require(sidecar['sourceVideoVerified']['sha256'] == row['contentSha256']
                and canonical_nas_path(sidecar['sourceVideoVerified']['path']) == canonical_nas_path(row['video']), 'DINO input video identity differs')
        require(sidecar['cache']['sha256'] == entry['dinoSha256'], 'Sidecar binds a different DINO cache')
        with np.load(entry['dinoPath'], allow_pickle=False) as cache:
            times = cache['timestamps'].astype(np.float64)
            tokens = cache['tokens']
            metadata = json.loads(str(cache['metadata_json'].item()))
            require(tokens.dtype == np.float16 and tokens.shape == (len(times), 10, 384) and np.isfinite(tokens).all(), 'Invalid DINO tokens')
            require(list(tokens.shape) == entry['shape'] and entry['storedDtype'] == 'float16', 'DINO shape metadata differs')
        require(metadata == sidecar['cacheMetadata'], 'Embedded/sidecar DINO metadata differs')
        require(metadata['recordingId'] == row['id'] and metadata['recordingContentSha256'] == row['contentSha256']
                and canonical_nas_path(metadata['sourceVideoPath']) == canonical_nas_path(row['video']), 'Embedded DINO source video differs')
        roi = row.get('roi')
        expected_roi = [roi[k] for k in ('x', 'y', 'width', 'height')] if isinstance(roi, dict) else roi
        require(metadata['roi'] == expected_roi, 'DINO court ROI differs from frozen recording')
        require(metadata['extractorConfigSha256'] == dino['extractorConfigSha256'] == plan['extractorConfigSha256']
                and metadata['labelsUsed'] is False and metadata['completed'] is True, 'DINO extractor contract differs')
        semantic = {key: value for key, value in metadata['backbone'].items() if key not in ('repository', 'checkpoint')}
        require(semantic == dino['semanticBackbone'] == plan['semanticBackbone'], 'DINO backbone weights/semantics differ')
        with np.load(av['path'], allow_pickle=False) as cache:
            av_times = cache['times'].astype(np.float64)
            if tier == 'coverage':
                require(str(cache['video_decoder'].item()) == PTS_DECODER, 'Repaired AV decoder metadata differs')
                av_binding = json.loads(str(cache['decoder_selection_json'].item()))
        if tier == 'coverage':
            require(sidecar['repairPlan'] == repair_reference, 'DINO sidecar repair-plan binding differs')
            pts_selections.append(audit_pts_selection(entry, row, metadata, av_times, times, av_binding, repair_reference))
        else:
            require('decoderSelection' not in metadata and 'decoderSelectionPath' not in entry
                    and 'decoderSelection' not in av, 'Unchanged exact/draft decoder identity unexpectedly revised')
        alignment = alignment_diagnostics(av_times, times)
        require(alignment == entry['nearestAlignment'] == sidecar['nearestAlignment'], 'DINO nearest alignment differs')
        alignments[row['id']] = alignment
    require(len(pts_selections) == 7, 'Seven repaired media PTS pairs are required')
    return {'recordsVerified': len(rows), 'cacheAndSidecarHashesVerified': len(artifacts), 'artifacts': artifacts,
            'nearestAlignmentByRecording': alignments, 'extractionPlan': identity(plan_path),
            'mediaPtsRepairPlan': identity(repair_path), 'mediaPtsSelections': pts_selections,
            'extractionDatasetAudit': audit_extraction_dataset(path, dino, manifest, contract),
            'sourceVideoBindingScope': 'Embedded and sidecar identities matched to frozen source hashes. MP4 bytes were hashed by extraction audit and are not rehashed by this summary.'}


def exposure_prefix(left, right, *, same_architecture=True):
    for field in ('trainIds', 'auxiliaryIds', 'validationIds', 'scalerTrainIds', 'positiveWeight', 'supervisedCounts'):
        require(left[field] == right[field], f'Matched experiment changed {field}')
    if same_architecture:
        require(left['parameters'] == right['parameters'], 'Capacity differs across loss arms')
    require(left['history'] and right['history'], 'Missing training history')
    streams = {'exact', *left['auxiliaryIds']}
    for a, b in zip(left['history'], right['history']):
        require(a['epoch'] == b['epoch'] and a['optimizerSteps'] == b['optimizerSteps'], 'Optimizer exposure differs')
        require(a['exposureSha256'] == b['exposureSha256'] and set(a['exposureSha256']) == streams,
                'Exact/auxiliary sample streams differ')
    return min(len(left['history']), len(right['history']))


def audit_fits(study, manifest, reg, old_reg, results):
    contract = reg['contract']
    groups = contract['groups']
    tiers = {'exact': manifest['exactRows'], 'draft': manifest['draftRows'], 'coverage': manifest['coverageRows']}
    by_id = {row['id']: row for rows in tiers.values() for row in rows}
    supervision = {row['id']: expanded.expected_supervision(row, tier) for tier, rows in tiers.items() for row in rows}
    diagnostics = {arm: {row['id']: expected_loss_weights(row, tier, arm)[1]
                        for tier, rows in tiers.items() for row in rows} for arm in ARMS}
    global_diagnostics = helpers.load(study / 'weight-diagnostics.json')
    expected_global = {arm: {'mode': arm, 'rows': {tier: [diagnostics[arm][r['id']] for r in rows] for tier, rows in tiers.items()}}
                       for arm in ARMS}
    require(global_diagnostics == {'contractSha256': reg['sha256'], 'arms': expected_global}, 'Registered global weighting diagnostics differ')
    for identifier in by_id:
        a, b = diagnostics['short_boost'][identifier], diagnostics['global_control'][identifier]
        require(abs(a['weightedPositiveMass']-b['weightedPositiveMass']) <= b['positiveMassAbsoluteTolerance'] + 1e-10,
                'Positive mass differs beyond float32 storage rounding')
        require(all(e['multiplier'] in (None, 1.) for e in a['events'] if not e['isShortOriginalEvent']),
                'Short boost changes a long original event')
    metadata, paths, completed_inputs, runtime = {}, {}, [], {}
    counts = Counter()
    for result in results:
        cohort, kind, arm, seed = key(result)
        reused = result['origin']['type'] == 'reused-reference'
        label = 'reused' if reused else 'fresh'
        root = Path(result['origin']['fitRoot'])
        choices = {choice['heldSourceGroup']: choice['epoch'] for choice in result['selections']}
        active = () if cohort == 'exact' else ('draft',) if cohort == 'draft' else ('draft', 'coverage')
        for outer_index, outer in enumerate(groups):
            folds = [(f'inner-{i}', group) for i, group in enumerate(g for g in groups if g != outer)] + [('refit', outer)]
            for fold, validation in folds:
                path = root / f'outer-{outer_index}' / fold / 'completed.json'
                meta = helpers.load(path)
                excluded = {outer, validation}
                train = [r['id'] for r in tiers['exact'] if r['sourceGroup'] not in excluded]
                auxiliary = {tier: [r['id'] for r in tiers[tier] if r['sourceGroup'] not in excluded] for tier in active}
                valid = [r['id'] for r in tiers['exact'] if r['sourceGroup'] == validation]
                ids = {'exact': train, **auxiliary}
                require(meta['contractSha256'] == (old_reg['sha256'] if reused else reg['sha256'])
                        and meta['kind'] == kind and meta['seed'] == seed, f'Fit identity differs: {path}')
                require(meta['trainIds'] == train and meta['scalerTrainIds'] == train
                        and meta['auxiliaryIds'] == auxiliary and meta['validationIds'] == valid, f'Fold leakage: {path}')
                require(meta['trainGroups'] == sorted({by_id[i]['sourceGroup'] for i in train})
                        and meta['validationGroups'] == [validation]
                        and meta['auxiliaryGroups'] == {tier: sorted({by_id[i]['sourceGroup'] for i in values}) for tier, values in auxiliary.items()},
                        'Fit group metadata differs')
                epochs = [choices[outer]] if fold == 'refit' else contract['checkpointEpochs']
                require(meta['epochs'] == epochs, 'Selected refit/checkpoint epochs differ')
                require(set(meta['supervisedCounts']) == set(ids), 'Unexpected supervision tier')
                for tier, identifiers in ids.items():
                    for field in ('valid', 'positiveMass'):
                        expected = np.sum([supervision[i][field] for i in identifiers], axis=0) if identifiers else np.zeros(4)
                        require(np.allclose(meta['supervisedCounts'][tier][field], expected, rtol=1e-6, atol=1e-4), 'Supervision differs from original labels')
                require(meta['parameters'] == contract['models'][kind]['parameters'], 'Parameter count differs from registered architecture')
                if not reused:
                    require(meta['architecture'] == kind and meta['lossArm'] == arm
                            and meta['model'] == contract['models'][kind], 'Fresh architecture/loss metadata differs')
                    expected = {'mode': arm, 'rows': {tier: [diagnostics[arm][i] for i in identifiers] for tier, identifiers in ids.items()}}
                    require(meta['liveLossWeighting'] == expected, 'Per-fit live multipliers differ from independent reconstruction')
                    counts['freshFitWeightDiagnosticsVerified'] += 1
                require([r['epoch'] for r in meta['history']] == list(range(1, max(epochs)+1)), 'Incomplete epoch history')
                require(meta['optimizerSteps'] == meta['history'][-1]['optimizerSteps']
                        and meta['exposureSha256'] == meta['history'][-1]['exposureSha256'], 'Final exposure metadata differs')
                expected_names = {f'{stem}-{epoch}.npz' for epoch in epochs for stem in ('weights', 'predictions')}
                require(set(meta['artifacts']) == expected_names, 'Missing/unexpected artifacts')
                for name, sha in meta['artifacts'].items():
                    require(helpers.digest(path.parent / name) == sha, f'Artifact hash changed: {path.parent / name}')
                    counts[f'{label}ArtifactHashesVerified'] += 1
                fit_key = (cohort, kind, arm, seed, outer_index, fold)
                metadata[fit_key], paths[fit_key] = meta, path.parent
                completed_inputs.append(identity(path))
                counts[f'{label}FitMembershipsVerified'] += 1
                if not reused:
                    bucket = runtime.setdefault(f'{cohort}/{kind}/{arm}', {'fitCount': 0, 'summedFitWallSeconds': 0., 'optimizerSteps': 0,
                                                                           'maximumPerProcessAllocatedCudaBytes': 0})
                    bucket['fitCount'] += 1
                    bucket['summedFitWallSeconds'] += meta['wallSeconds']
                    bucket['optimizerSteps'] += meta['optimizerSteps']
                    bucket['maximumPerProcessAllocatedCudaBytes'] = max(bucket['maximumPerProcessAllocatedCudaBytes'], meta['peakAllocatedCudaBytes'] or 0)
    require(counts['freshFitMembershipsVerified'] == 768 and counts['reusedFitMembershipsVerified'] == 96
            and counts['freshArtifactHashesVerified'] == 4992 and counts['reusedArtifactHashesVerified'] == 624,
            'Full54-cell fit/artifact inventory is incomplete')
    for cohort in COHORTS:
        for seed in contract['seeds']:
            for outer_index in range(len(groups)):
                for fold in [*(f'inner-{i}' for i in range(len(groups)-1)), 'refit']:
                    suffix = (seed, outer_index, fold)
                    # Exact training IDs are identical across cohorts, so this
                    # remains a historical anchor even when C is freshly fit.
                    historical = metadata[('exact', 'tcn', 'baseline', *suffix)]
                    old_path = paths[('exact', 'tcn', 'baseline', *suffix)]
                    with np.load(old_path / f"weights-{historical['epochs'][0]}.npz", allow_pickle=False) as previous:
                        old_mean, old_scale = previous['mean'], previous['scale']
                        for kind in KINDS:
                            for arm in ARMS:
                                fit_key = (cohort, kind, arm, *suffix)
                                meta, path = metadata[fit_key], paths[fit_key]
                                for epoch in meta['epochs']:
                                    with np.load(path / f'weights-{epoch}.npz', allow_pickle=False) as current:
                                        require(np.array_equal(old_mean, current['mean']) and np.array_equal(old_scale, current['scale']),
                                                'AV scaler differs from historical fold')
                                    counts['historicalAVScalerCheckpointsVerified'] += 1
                            for arm, ref in CONTRASTS:
                                counts['armPairedEpochPrefixesVerified'] += exposure_prefix(metadata[(cohort, kind, arm, *suffix)], metadata[(cohort, kind, ref, *suffix)])
                                counts['armPairedFitPrefixesVerified'] += 1
                        for arm in ARMS:
                            counts['architecturePairedEpochPrefixesVerified'] += exposure_prefix(
                                metadata[(cohort, 'dino_tcn', arm, *suffix)], metadata[(cohort, 'tcn', arm, *suffix)], same_architecture=False)
                            counts['architecturePairedFitPrefixesVerified'] += 1
    for kind in KINDS:
        for arm in ARMS:
            for seed in contract['seeds']:
                for outer_index in range(len(groups)):
                    for fold in [*(f'inner-{i}' for i in range(len(groups)-1)), 'refit']:
                        for first, second in (('exact', 'draft'), ('draft', 'reviewed_export')):
                            suffix = (kind, arm, seed, outer_index, fold)
                            counts['cohortPairedEpochPrefixesVerified'] += expanded.audit_exposure_pair(
                                metadata[(first, *suffix)], metadata[(second, *suffix)], compare_draft=first == 'draft')
                            counts['cohortPairedFitPrefixesVerified'] += 1
    counts['independentWeightVectorsVerified'] = len(ARMS) * len(by_id)
    return {'counts': dict(counts), 'completedMetadata': completed_inputs, 'weightDiagnostics': diagnostics,
            'weightDiagnosticsArtifact': identity(study / 'weight-diagnostics.json'), 'trainingRuntime': runtime,
            'runtimeScope': 'Fresh fits only; reused historical fits are not counted as new work. Summed fit durations may overlap; peak CUDA allocation is per process. These are training costs, not feature-extraction or phone/browser inference measurements.'}


def audit_selected_inner_candidates(exact, contract, results):
    from types import SimpleNamespace
    from analysis.crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
    from analysis.neural_development import decode
    from analysis.schema import Interval, mask_for_times
    examples = {}
    for row in exact['recordings']:
        with np.load(row['featureCaches']['audiovisual']['path'], allow_pickle=False) as data:
            times = data['times'].astype(np.float64)
            duration = float(json.loads(str(data['metadata_json'].item()))['duration'])
        convert = lambda rows: tuple(Interval(r['start'], r['end'], tuple(r.get('tags', []))) for r in rows)
        ignored = convert(row.get('ignoredIntervals', []))
        examples[row['id']] = SimpleNamespace(id=row['id'], group=row['sourceGroup'], times=times, duration=duration,
                                              truth=convert(row['rallies']), ignored=ignored, valid=mask_for_times(times, ignored))
    count = 0
    for result in results:
        root = Path(result['origin']['fitRoot'])
        for choice in result['selections']:
            outer = choice['heldSourceGroup']
            index = contract['groups'].index(outer)
            probabilities = {}
            for inner_index, inner in enumerate(g for g in contract['groups'] if g != outer):
                path = root / f'outer-{index}' / f'inner-{inner_index}' / f"predictions-{choice['epoch']}.npz"
                with np.load(path, allow_pickle=False) as cache:
                    require(set(cache.files) == {r.id for r in examples.values() if r.group == inner}, 'Inner validation scope differs')
                    for identifier in cache.files:
                        scores = cache[identifier]
                        require(scores.shape == (len(examples[identifier].times), 4) and np.isfinite(scores).all()
                                and (scores >= 0).all() and (scores <= 1).all(), 'Invalid selected inner scores')
                        probabilities[identifier] = scores
            rows = [RecordingIntervals(e.id, 'development', e.duration, e.truth,
                                       tuple(decode(e, probabilities[e.id], choice['decoder'])), e.ignored)
                    for e in examples.values() if e.group != outer]
            score = evaluate_f1_pad_p_core_r(rows, [2.], 3.)[0]
            require(abs(score['F1_padP_coreR']-choice['innerF1_padP_coreR']) <= 1e-12
                    and abs(score['R_core']-choice['innerR_core']) <= 1e-12, 'Selected inner score differs from artifact replay')
            count += 1
    return count


def summarize(study, manifest_path, *, progress=False):
    reg, old_reg, manifest, exact, report, inputs = read_inputs(study, manifest_path, progress=progress)
    contract, results = reg['contract'], report['results']
    if progress:
        return {'status': 'complete-report-present' if (study / 'report.json').exists() else 'in-progress',
                'completedResultCells': len(results), 'expectedResultCells': 54,
                'reusedBaselineCells': sum(r['origin']['type'] == 'reused-reference' for r in results),
                'note': 'Progress only; no final audit, recovery or replication conclusions.'}, None
    inputs['frozenSources'] = []
    for name, sha in contract['code'].items():
        path = REPO / 'analysis' / name
        require(helpers.digest(path) == sha, f'Registered source changed: {name}')
        inputs['frozenSources'].append(identity(path))
    inputs['avCaches'] = []
    for row in manifest['exactRows'] + manifest['draftRows'] + manifest['coverageRows']:
        cache = row['featureCaches']['audiovisual']
        require(helpers.digest(Path(cache['path'])) == cache['sha256'], f"AV cache changed: {row['id']}")
        inputs['avCaches'].append(identity(Path(cache['path'])))
    dino_audit = audit_dino_inputs(manifest, contract)
    inputs['resultFiles'] = []
    for row in results:
        path = study / f"result-{row['cohort']}-{row['kind']}-{row['lossArm']}-{row['seed']}.json"
        require(helpers.load(path) == row, 'Result file differs from report')
        inputs['resultFiles'].append(identity(path))
        helpers.assert_result_revision(exact, row['predictions'], str(key(row)))
        require(evaluate_predictions(row['predictions'], primary_padding_seconds=2., join_gap_seconds=3.) == row['evaluation'],
                f'Canonical metric replay differs: {key(row)}')
        require(sorted(s['heldSourceGroup'] for s in row['selections']) == contract['groups'], 'Outer selections missing/duplicated')
        for choice in row['selections']:
            require(choice['epoch'] in contract['checkpointEpochs'] and choice['decoder'] in contract['decoderCandidates'], 'Selection outside fixed grid')
            expanded.selection_feasible(choice, .95)
    audit = audit_fits(study, manifest, reg, old_reg, results)
    audit['dinoInputs'] = dino_audit
    audit['counts'].update({'canonicalMetricReplays': len(results), 'exactGoldAndIgnoredRevisionsVerified': len(results),
                            'reusedBaselineResultsVerified': sum(r['origin']['type'] == 'reused-reference' for r in results),
                            'registeredSourcesVerified': len(inputs['frozenSources']), 'avCacheHashesVerified': len(inputs['avCaches']),
                            'dinoCacheAndSidecarHashesVerified': dino_audit['cacheAndSidecarHashesVerified'],
                            'dinoSourceAndTimelineAssociationsVerified': dino_audit['recordsVerified'],
                            'selectedInnerCandidatesReplayed': audit_selected_inner_candidates(exact, contract, results)})
    per_seed, aggregates, comparisons, transfer, losses = summarize_results(results, contract)
    choices = [choice for row in results for choice in row['selections']]
    audit['selectionCounts'] = {'epochs': dict(Counter(c['epoch'] for c in choices)),
                                'thresholds': dict(Counter(c['decoder']['enter'] for c in choices)),
                                'boundaryEnabled': sum(c['decoder']['boundary'] for c in choices),
                                'infeasible': sum(not c['recallEligibilityPassed'] for c in choices)}
    summary = {'schemaVersion': 1, 'kind': 'neural-short-boost-transfer-summary-v1',
               'createdAt': datetime.now(timezone.utc).isoformat(), 'status': 'completed-short-boost-transfer-audit',
               'contractSha256': reg['sha256'], 'manifestSha256': contract['manifestSha256'],
               'protectedTestOpened': False, 'productionPromotionAllowed': False,
               'primaryMetric': 'F1_padP_coreR', 'primaryPaddingSeconds': 2, 'joinGapSeconds': 3,
               'scope': {'exactRecordings': len(exact['recordings']), 'exactRallies': sum(len(r['rallies']) for r in exact['recordings']),
                         'exactSourceGroups': contract['groups'], 'seeds': contract['seeds'], 'resultCells': len(results)},
               'aggregation': 'Pool recording numerators/denominators within each seed. Means summarize separate seed runs, not independent population samples or confidence intervals.',
               'retentionRecoveryScreenContract': contract['retentionRecoveryScreen'], 'transferScreenContract': contract['transferScreen'],
               'limitations': [
                   'Seven Pixel coverage feature caches were repaired to actual media timestamps before candidate training. Exact/draft labels and features are unchanged; reviewed-export compact baselines are freshly fit and cannot reuse the historical coverage cohort.',
                   'Adaptive development comparison on previously inspected source groups; no protected test or production promotion.',
                   'Six exact/draft compact baseline results retain immutable historical fits and provenance; they are not newly trained replicates. All reviewed-export cells use repaired feature inputs.',
                   'Global positive-mass control matches each full recording within float32 storage rounding, not every sampled minibatch.',
                   'Original event duration defines the short slice. Ignored holes and draft uncertainty masks do not redefine original duration or create new events.',
                   'Both architectures select settings independently with the same inner rule. Effects describe the training and selection procedure rather than fixed-decoder logits.',
                   'Raw cross-architecture score gains are not evidence that short-rally weighting transfers. Replication uses the four declared within-architecture recovery comparisons per cohort; difference-in-differences is descriptive.',
                   'DINO adds pretrained tokens and a projection network/capacity. This study does not isolate pretrained representation from all architecture changes.',
                   'Lower baseline point loss leaves less room for absolute recovery. Zero baseline loss is reported without dividing by zero; seed ties are not recovery.',
                   'No positive interaction, source-group improvement, or absolute outer recall requirement is added to the frozen recovery/transfer screens.',
                   'Runtime excludes frozen DINO extraction and is not a phone/browser inference benchmark.',
               ], 'inputs': inputs, 'audit': audit, 'aggregates': aggregates, 'perSeed': per_seed,
               'comparisons': comparisons, 'transfer': transfer}
    return summary, {'scope': summary['scope'], 'perSeed': losses}


def markdown(summary):
    lines = ['# Short-boost transfer summary', '',
             'Primary scores use fixed 2s symmetric padding, strictly <3s gap joining and ignored subtraction. '
             'All models evaluate the same exact labels. Means describe three separate seed runs. '
             'Six exact/draft compact baselines reuse verified historical results. The reviewed-export compact baseline is freshly trained after the pretraining Pixel timestamp repair; no production promotion.', '',
             '| Cohort | Architecture | Loss arm | F1 | R_core | P_pad | Event F1 |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        metric = row['meanSeedPrimary']
        lines.append(f"| {row['cohort']} | {row['kind']} | {row['lossArm']} | {metric['F1_padP_coreR']:.6f} | "
                     f"{metric['R_core']:.6f} | {metric['P_pad']:.6f} | {row['meanSeedEventF1']:.6f} |")
    lines += ['', '| Within-architecture comparison | F1 delta | R_core delta | F1 screen | Recovery screen |',
              '|---|---:|---:|---|---|']
    for row in summary['comparisons']:
        metric = row['meanSeedPrimaryDelta']
        lines.append(f"| {row['name']} | {metric['F1_padP_coreR']:+.6f} | {metric['R_core']:+.6f} | "
                     f"{row['standardF1Screen']['passed']} | {row['retentionRecoveryScreen']['passed']} |")
    lines += ['', 'Replication requires feasible recovery versus both own controls in both architectures, within the same cohort. '
              'A positive difference-in-differences is not required: equal benefit in both architectures gives zero interaction.', '',
              '| Cohort | Replication screen | Reference arm | F1 DiD | Complete-point recovery DiD | <=3s recovery DiD |',
              '|---|---|---|---:|---:|---:|']
    for row in summary['transfer']:
        for effect in row['contrasts']:
            metric = effect['meanSeedDifferenceInDifferences']
            lines.append(f"| {row['cohort']} | {row['replicationScreen']['passed']} | {effect['referenceArm']} | "
                         f"{metric['F1_padP_coreR']:+.6f} | {metric['all/completeLossRecovery']:+.2f} | {metric['duration_le_3s/completeLossRecovery']:+.2f} |")
    lines += ['', 'DiD subtracts the compact within-architecture effect from the DINO within-architecture effect. '
              'Positive point recovery means fewer complete losses; raw architecture-score differences are not transfer evidence.', '',
              '| Cohort | Architecture | Arm | Slice | Mean complete losses | Mean partial losses | Slice core recall |',
              '|---|---|---|---|---:|---:|---:|']
    for row in summary['aggregates']:
        for name in ('all', 'duration_le_2s', 'duration_le_3s', 'duration_gt_3s', 'ace', 'service_fault'):
            value = row['meanSeedSlices']['primaryExportCoverage'][name]
            recall = f"{value['coreRecall']:.6f}" if value['coreRecall'] is not None else 'n/a'
            lines.append(f"| {row['cohort']} | {row['kind']} | {row['lossArm']} | {name} | "
                         f"{value['completeRallyLosses']:.2f} | {value['partialRallyLosses']:.2f} | {recall} |")
    lines += ['', 'Padding sensitivity uses the same fixed predictions in all four cases; only 2s is primary.', '',
              '| Cohort | Architecture | Arm | Padding | P_pad | R_core | F1_padP_coreR | Model seconds | Human seconds | Difference |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        for metric in row['meanSeedPadding']:
            lines.append(f"| {row['cohort']} | {row['kind']} | {row['lossArm']} | {metric['paddingSecondsBeforeAndAfter']} | "
                         f"{metric['P_pad']:.6f} | {metric['R_core']:.6f} | {metric['F1_padP_coreR']:.6f} | "
                         f"{metric['paddedModelExportSeconds']:.3f} | {metric['paddedHumanExportSeconds']:.3f} | {metric['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ['', 'Audit counts:', '', '```json', json.dumps(summary['audit']['counts'], indent=2), '```', '',
              'Per-seed/group effects, baseline loss floors, source hashes and exact rally-loss identities are in the JSON artifacts.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'study')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'manifest-pts-v1.json')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    names = ('summary.json', 'summary.md', 'loss-identities.json')
    if not args.progress:
        require(not any((args.study / name).exists() for name in names), 'Refusing to overwrite summary artifacts')
    summary, losses = summarize(args.study, args.manifest, progress=args.progress)
    if args.progress:
        print(json.dumps(summary, indent=2, allow_nan=False))
        return
    summary['lossIdentitiesArtifact'] = str(args.study / 'loss-identities.json')
    for name, content in (('summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n'),
                          ('summary.md', markdown(summary)), ('loss-identities.json', json.dumps(losses, indent=2, allow_nan=False) + '\n')):
        with (args.study / name).open('x', encoding='utf-8') as stream:
            stream.write(content)
    print(json.dumps({'status': summary['status'], 'audit': summary['audit']['counts'],
                      'artifacts': {name: identity(args.study / name) for name in names}}, indent=2))


if __name__ == '__main__':
    main()
