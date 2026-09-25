"""Create identifier-free summaries of frozen distilled Large selections.

Private experiment locations are explicit CLI inputs. Public outputs contain
allowlisted aggregate measurements; recording names, source groups, per-rally
diagnostics, input paths and private provenance are never copied through.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

METRIC_FIELDS = (
    'paddingSecondsBeforeAndAfter', 'joinGapSeconds', 'P_pad', 'R_core', 'F1_padP_coreR',
    'paddedPrecisionIntersectionSeconds', 'paddedModelExportSeconds', 'coreRecallIntersectionSeconds',
    'coreHumanSeconds', 'paddedHumanExportSeconds', 'exportDurationDifferenceSeconds',
    'inputCropCount', 'outputCropCount', 'recordingCount',
)
COVERAGE_FIELDS = (
    'originalRallies', 'evaluableRallies', 'fullyIgnoredRallies', 'completeRallyLosses',
    'partialRallyLosses', 'fullyCoveredRallies', 'evaluableCoreSeconds', 'retainedCoreSeconds', 'coreRecall',
)
EVENT_FIELDS = ('trueRallies', 'predictedRallies', 'matchedRallies', 'eventPrecision', 'eventRecall', 'eventF1')
NONEXACT_FIELDS = ('paddingSecondsBeforeAndAfter', 'joinGapSeconds', 'humanPaddingSeconds',
    'humanJoinGapSeconds', 'modelExportSeconds', 'humanExportSeconds', 'intersectionSeconds',
    'evaluableVideoSeconds', 'exportDurationDifferenceSeconds', 'incorrectExportSeconds',
    'wantedExportOmittedSeconds', 'correctlyRemovedSeconds', 'reviewedLiveSeconds',
    'retainedReviewedLiveSeconds', 'missedReviewedLiveSeconds')
POLICIES = ('exact-rallies', 'reviewed-draft', 'reviewed-export')
SCOPES = ('all', 'no-production-training', 'no-production-training-or-calibration', 'beach', 'non-beach')
FIT_ROLES = ('training', 'training-related-source', 'calibration', 'calibration-related-source',
             'common-panel-ui-selection', 'unseen-by-this-fit-and-selection')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def numeric_fields(value, names):
    result = {}
    for key in names:
        if key not in value:
            continue
        item = value[key]
        require(item is None or (not isinstance(item, bool) and isinstance(item, (int, float))
                                and math.isfinite(item)), 'Invalid aggregate measurement')
        result[key] = item
    return result


def metric(value):
    result = numeric_fields(value, METRIC_FIELDS)
    require(set(METRIC_FIELDS[:11]) <= set(result), 'Incomplete padded/core measurements')
    require(result['paddingSecondsBeforeAndAfter'] in (0, 1, 2, 3)
            and result['joinGapSeconds'] == 3, 'Unexpected padding or gap-join contract')
    return result


def aggregate(evaluation):
    padding = [metric(value) for value in evaluation['padding']]
    require([value['paddingSecondsBeforeAndAfter'] for value in padding] == [0, 1, 2, 3],
            'All four symmetric padding cases are required')
    primary = metric(evaluation['primary'])
    require(primary == padding[2], 'Primary comparison must use fixed 2s padding')
    guards = evaluation['guardrails']
    coverage = numeric_fields(guards['primaryExportCoverage'], COVERAGE_FIELDS)
    require(set(COVERAGE_FIELDS) <= set(coverage), 'Incomplete whole-rally coverage diagnostics')
    return dict(recordingCount=evaluation['recordingCount'], sourceGroupCount=evaluation['sourceGroupCount'],
                primary=primary, padding=padding, primaryExportCoverage=coverage,
                whollyMissedSavedHumanRallies=coverage['completeRallyLosses'],
                eventMetrics=numeric_fields(guards, EVENT_FIELDS))


def candidate(value, fit_number, variant_number):
    setting = value['setting']
    decoder = setting['decoder']
    require(set(decoder) == {'smoothing', 'enter', 'minimum', 'boundary'}
            and isinstance(decoder['boundary'], bool), 'Unexpected decoder metadata')
    decoder_values = numeric_fields(decoder, ('smoothing', 'enter', 'minimum'))
    require(len(decoder_values) == 3, 'Incomplete decoder metadata')
    require(all(isinstance(value[key], int) and not isinstance(value[key], bool)
                for key in ('draw', 'seed', 'floorPercent'))
            and value['floorPercent'] == 99, 'Report only covers strict target-99 fits')
    require(setting['innerR_core'] >= .99, 'Candidate did not achieve its calibration recall floor')
    require(isinstance(setting['epoch'], int) and not isinstance(setting['epoch'], bool), 'Invalid checkpoint epoch')
    return dict(fitIndex=f'fit-{fit_number:03d}', variantIndex=f'variant-{variant_number:03d}',
                draw=value['draw'], trainingSeed=value['seed'], recallTargetPercent=99,
                epoch=setting['epoch'], decoder=dict(**decoder_values, boundary=decoder['boundary']),
                calibration=numeric_fields(setting, ('innerR_core', 'innerF1_padP_coreR')),
                **aggregate(value['evaluation']))


def selections(evaluation):
    candidates = [value for value in evaluation['candidates'] if value['floorPercent'] == 99]
    require(candidates, 'No feasible target-99 options')
    require(len({(value['variant'], value['draw']) for value in candidates}) == len(candidates),
            'Duplicate fit candidate')
    candidates.sort(key=lambda value: (value['variant'], value['draw']))
    variants = sorted({value['variant'] for value in candidates})
    public = [candidate(value, number+1, variants.index(value['variant'])+1)
              for number, value in enumerate(candidates)]
    f1 = min(candidates, key=lambda value: (-value['evaluation']['primary']['F1_padP_coreR'],
                                           value['variant'], value['draw']))
    recall = min((value for value in candidates if value['variant'] == f1['variant']),
                 key=lambda value: (-value['evaluation']['primary']['R_core'],
                                    -value['evaluation']['primary']['F1_padP_coreR'], value['draw']))
    require(len(evaluation['selected']) == 2 and {row['mode'] for row in evaluation['selected']} == {'f1', 'recall'},
            'Expected the two requested selection modes')
    selected = []
    for mode, expected in (('f1', f1), ('recall', recall)):
        chosen = next(value for value in evaluation['selected'] if value['mode'] == mode)
        require({key: value for key, value in chosen.items() if key != 'mode'} == expected,
                'Published selection differs from the requested ranking rule')
        selected.append(dict(mode=mode, **public[candidates.index(expected)]))
    return public, selected


def verified_manifest(plan):
    reference = plan['inferenceManifest']
    path = Path(reference['path'])
    require(hashlib.sha256(path.read_bytes()).hexdigest() == reference['sha256'], 'Inference manifest changed')
    rows = read(path)['records']
    require(len({value['id'] for value in rows}) == len(rows), 'Duplicate manifest recording')
    return {value['id']: value for value in rows}


def common_scope(evaluation, manifest):
    """Private comparison signature; never include its values in public output."""
    signature = {}
    for row in evaluation['recordings']:
        source = manifest[row['id']]
        require(source['sourceGroup'] == row['sourceGroup'], 'Common-panel source identity differs')
        require(source['scoringPolicy'] == 'exact-core', 'Common comparison requires exact human rallies')
        signature[row['id']] = {key: source[key] for key in
            ('sourceGroup', 'durationSeconds', 'rallies', 'ignoredIntervals', 'scoringPolicy')}
    require(len(signature) == evaluation['recordingCount'], 'Incomplete common-panel membership')
    return signature


def baseline_comparison(experiment, evaluation, baseline_experiment):
    baseline = read(baseline_experiment/'evaluation.json')
    _, selected = selections(baseline)
    current_manifest = verified_manifest(read(experiment/'plan.json'))
    previous_manifest = verified_manifest(read(baseline_experiment/'plan.json'))
    scope = common_scope(evaluation['selected'][0]['evaluation'], current_manifest)
    for value in evaluation['selected']:
        require(common_scope(value['evaluation'], current_manifest) == scope, 'Selected scopes differ')
    for value in baseline['selected']:
        require(common_scope(value['evaluation'], previous_manifest) == scope,
                'Baseline common panel, gold labels or ignored intervals differ')
    return dict(model='Frozen ImageNet MobileNetV3-Large + TCN', precision='FP32',
                sameCommonExactScopeAndGoldVerified=True, selected=selected)


def all_video_report(document):
    models = document['models']
    require(len(models) == 2 and {model['mode'] for model in models} == {'f1', 'recall'},
            'All-video evaluation must include both frozen selections')
    result = []
    for model in models:
        panels = []
        require(len({(panel['labelPolicy'], panel['scope']) for panel in model['panels']}) == len(model['panels']),
                'Duplicate all-video evaluation panel')
        for panel in model['panels']:
            policy, scope = panel['labelPolicy'], panel['scope']
            require(policy in POLICIES and scope in SCOPES, 'Unknown evaluation panel policy/scope')
            evaluation = panel['evaluation']
            if evaluation is None:
                require(panel.get('status') == 'empty', 'Unavailable panel is not marked empty')
                panels.append(dict(labelPolicy=policy, scope=scope, status='empty', recordingCount=0))
                continue
            if policy == 'exact-rallies':
                summary = aggregate(evaluation)
            else:
                suffix = 'export' if policy == 'reviewed-export' else 'reviewed'
                score_keys = tuple(f'{prefix}_{suffix}' for prefix in ('P', 'R', 'F1'))
                padding = [numeric_fields(value, NONEXACT_FIELDS+score_keys) for value in evaluation['padding']]
                require([value['paddingSecondsBeforeAndAfter'] for value in padding] == [0, 1, 2, 3]
                        and all(value['joinGapSeconds'] == 3 and all(key in value for key in score_keys) for value in padding),
                        'Incomplete nonexact padding sensitivity')
                require(numeric_fields(evaluation['primary'], NONEXACT_FIELDS+score_keys) == padding[2],
                        'Nonexact primary padding differs')
                require(evaluation['rallyCoreMetricsAvailable'] is False and evaluation['eventMetricsAvailable'] is False,
                        'Nonexact labels cannot establish true rally cores or event accuracy')
                if policy == 'reviewed-export':
                    require(all(value.get('humanPaddingSeconds') == 0 and value.get('humanJoinGapSeconds') == 0
                                for value in padding), 'Reviewed exports must remain fixed human intervals')
                summary = dict(recordingCount=evaluation['recordingCount'], sourceGroupCount=evaluation['sourceGroupCount'],
                               primary=padding[2], padding=padding, rallyCoreMetricsAvailable=False,
                               eventMetricsAvailable=False, whollyMissedSavedHumanRallies=None)
            counts = numeric_fields(panel, ('foundRallies',))
            require('foundRallies' in counts and isinstance(counts['foundRallies'], int)
                    and counts['foundRallies'] >= 0, 'Missing found-rally count')
            panels.append(dict(labelPolicy=policy, scope=scope, status='available', **counts, **summary))
        roles = numeric_fields(model.get('fitRoleCounts', {}), FIT_ROLES)
        result.append(dict(mode=model['mode'], panels=panels, fitRoleCounts=roles,
                           **numeric_fields(model, ('scoredRecordingCount', 'unscoredRecordingCount',
                                                     'missingProductionExposureCount'))))
    return dict(models=result, allScopeExcludesBeach=True,
                **numeric_fields(document, ('inferenceRecordingCount',)),
                interpretation='Exact, draft and export panels are separate; all and no-production-training scopes exclude beach. '
                               'These panels include fitting/calibration recordings and are not an untouched generalization test. '
                               'No-production-training excludes known production fitting exposure; production calibration exposure can remain.')


def report(experiment, baseline_experiment=None, all_video_path=None):
    experiment = Path(experiment)
    evaluation, plan = read(experiment/'evaluation.json'), read(experiment/'plan.json')
    require(Path(evaluation['plan']['path']).resolve() == (experiment/'plan.json').resolve()
            and evaluation['plan']['sha256'] == hashlib.sha256((experiment/'plan.json').read_bytes()).hexdigest(),
            'Selection belongs to another experiment plan')
    require(plan['floorsPercent'] == [99] and plan['targetPaddingSeconds'] == 2
            and plan['paddingSeconds'] == [0, 1, 2, 3] and plan['joinGapSeconds'] == 3,
            'Experiment contract differs from strict-99 plan')
    candidates, selected = selections(evaluation)
    require(all(row['recordingCount'] == evaluation['commonExactPanelRecordingCount']
                and row['sourceGroupCount'] == evaluation['commonExactPanelSourceGroupCount'] for row in candidates),
            'Common-panel population counts differ across candidates')
    registered = {(task['variant'], task.get('splitSeed', task['seed'])) for task in plan['tasks']}
    require(len(registered) == len(plan['tasks']), 'Duplicate registered fit')
    require({(row['variant'], row['draw']) for row in evaluation['candidates']} <= registered,
            'Evaluation contains an unregistered fit')
    value = dict(schemaVersion=1, kind='distilled-mobile-large-selection-report',
        model='DINO-distilled MobileNetV3-Large + TCN', precision='FP32',
        architecture=dict(backbone='MobileNetV3-Large', inputPixels=224, samplingHz=2,
                          regionalPools=4, embeddingValuesPerPool=960, audiovisualFeatureCount=104,
                          qualityAgeAvailabilityValues=8, temporalInputDimension=3952,
                          teacher='Cached frozen DINO training targets only',
                          teacherRequiredAtInference=False, distillationProjectionUsedAtInference=False),
        recallTargetPercent=99, primaryPaddingSeconds=2, symmetricPaddingCasesSeconds=[0, 1, 2, 3],
        joinGapSeconds=3, joinRule='Positive gaps strictly below 3 seconds; overlaps and touching merge',
        ignoredRule='Subtract ignored intervals after padding/joining; never rejoin across ignored spans',
        metricAggregation='Pool intersections and durations across recordings before computing metrics',
        registeredFitCount=len(registered), feasible99FitCount=len(candidates),
        commonUnseenUsedForSelection=True,
        commonExactPanelRecordingCount=evaluation['commonExactPanelRecordingCount'],
        commonExactPanelSourceGroupCount=evaluation['commonExactPanelSourceGroupCount'],
        selectionRule='Strict 99% calibration only; maximum common-unseen exact-label F1_padP_coreR; '
                      'highest-recall draw within the winning variant, breaking ties with F1_padP_coreR',
        whollyMissedDefinition='A nonignored saved-human rally core with no overlap with the padded, joined model export',
        candidates=candidates, selected=selected, allVideoEvaluation=None)
    if baseline_experiment is not None:
        value['baseline'] = baseline_comparison(experiment, evaluation, Path(baseline_experiment))
        value['baselineDeltas'] = []
        for current in selected:
            old = next(row for row in value['baseline']['selected'] if row['mode'] == current['mode'])
            value['baselineDeltas'].append(dict(mode=current['mode'],
                **{key: current['primary'][key]-old['primary'][key] for key in
                   ('P_pad', 'R_core', 'F1_padP_coreR', 'paddedModelExportSeconds')},
                whollyMissedSavedHumanRallies=current['whollyMissedSavedHumanRallies']-old['whollyMissedSavedHumanRallies']))
    panel_path = Path(all_video_path) if all_video_path else experiment/'all-video-evaluation.json'
    if panel_path.exists():
        document = read(panel_path)
        expected_hash = hashlib.sha256((experiment/'evaluation.json').read_bytes()).hexdigest()
        actual_hash = document.get('selectedEvaluationSha256') or document.get('selections', {}).get('sha256')
        require(actual_hash == expected_hash, 'All-video panels belong to another selected run')
        value['allVideoEvaluation'] = all_video_report(document)
    elif all_video_path is not None:
        raise FileNotFoundError('Requested all-video evaluation is missing')
    return value


def percent(value):
    return f'{value:.2%}'


def markdown(report):
    lines = ['# DINO-distilled MobileNetV3-Large + TCN', '',
        'The student uses four 960-value regional MobileNetV3-Large embeddings at 224px/2Hz, '
        'AV104 features and eight quality/age/availability values. The FP32 TCN consumes 3,952 values per tick. '
        'Cached frozen DINO features supervise student training; DINO and the training projection are absent at inference.', '',
        f"{report['feasible99FitCount']} of {report['registeredFitCount']} registered fits achieved the strict 99% calibration bar. "
        'The target refers to retained human-core play after 2s export padding. It is not a guarantee of 99% recall on another video.', '',
        report['selectionRule']+'. Common-unseen is already selection data, not an untouched test set. '
        f"Its exact-label panel contains {report['commonExactPanelRecordingCount']} recording(s) from "
        f"{report['commonExactPanelSourceGroupCount']} source group(s); this small panel limits generalization claims.", '',
        'Target product padding is fixed at 2s before and after. Positive gaps strictly under 3s join; '
        'ignored intervals are subtracted afterward without rejoining. Metrics pool intersections and durations.', '',
        '| Model | Selection | Fit | Draw | Epoch | P_pad | R_core | F1_padP_coreR | Wholly missed human rallies |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    groups = [('Distilled Large', report['selected'])]
    if 'baseline' in report:
        groups.append(('Frozen Large', report['baseline']['selected']))
    for name, selected in groups:
        for row in selected:
            m = row['primary']
            lines.append(f"| {name} | Highest {row['mode']} | {row['fitIndex']} | {row['draw']} | {row['epoch']} | "
                         f"{percent(m['P_pad'])} | {percent(m['R_core'])} | {percent(m['F1_padP_coreR'])} | "
                         f"{row['whollyMissedSavedHumanRallies']} |")
    lines += ['', 'Wholly missed means no retained nonignored core from a saved-human rally after padding and gap joining. '
              'This event count differs from duration-weighted retained-play recall.', '',
              '| Model | Selection | Padding each side | P_pad | R_core | F1_padP_coreR | Model export (s) | Human export (s) | Difference (s) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, selected in groups:
        for row in selected:
            for m in row['padding']:
                lines.append(f"| {name} | {row['mode']} | {m['paddingSecondsBeforeAndAfter']:.0f} | {percent(m['P_pad'])} | "
                             f"{percent(m['R_core'])} | {percent(m['F1_padP_coreR'])} | {m['paddedModelExportSeconds']:.3f} | "
                             f"{m['paddedHumanExportSeconds']:.3f} | {m['exportDurationDifferenceSeconds']:+.3f} |")
    if 'baseline' in report:
        lines += ['', 'Frozen-Large comparisons use the same common exact-label recordings, human rally boundaries, '
                  'ignored intervals and duration revision, verified before reporting. Each family retains its own frozen selections.']
    if report['allVideoEvaluation'] is not None:
        lines += ['', '## Frozen selections across all videos', '', report['allVideoEvaluation']['interpretation'], '',
                  '| Selection | Videos with predictions | Videos with scored labels | Videos with counts only |',
                  '|---|---:|---:|---:|']
        for model in report['allVideoEvaluation']['models']:
            scored = model.get('scoredRecordingCount')
            unscored = model.get('unscoredRecordingCount')
            if scored is not None and unscored is not None:
                lines.append(f"| {model['mode']} | {scored+unscored} | {scored} | {unscored} |")
        lines += ['', 'Videos without an eligible scoring-label policy still receive predictions; their counts do not establish accuracy.', '',
                  'P/R/F1 below use padded-precision/core-recall for exact labels, reviewed-live overlap for drafts, '
                  'and fixed human-export overlap for reviewed exports. Their different denominators are not pooled together.', '',
                  '| Selection | Labels | Scope | Videos | Found rallies | P | R | F1 | Wholly missed human rallies |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for model in report['allVideoEvaluation']['models']:
            for panel in model['panels']:
                if panel['status'] == 'empty':
                    continue
                policy = panel['labelPolicy']
                keys = ('P_pad', 'R_core', 'F1_padP_coreR') if policy == 'exact-rallies' else (
                    ('P_export', 'R_export', 'F1_export') if policy == 'reviewed-export' else ('P_reviewed', 'R_reviewed', 'F1_reviewed'))
                m = panel['primary']
                missed = panel['whollyMissedSavedHumanRallies']
                lines.append(f"| {model['mode']} | {policy} | {panel['scope']} | {panel['recordingCount']} | {panel['foundRallies']} | "
                             f"{percent(m[keys[0]])} | {percent(m[keys[1]])} | {percent(m[keys[2]])} | {missed if missed is not None else 'N/A'} |")
        lines += ['', '| Selection | Labels | Scope | Padding | P | R | F1 | Model export (s) | Human export (s) | Difference (s) |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for model in report['allVideoEvaluation']['models']:
            for panel in model['panels']:
                if panel['status'] == 'empty':
                    continue
                policy = panel['labelPolicy']
                keys = ('P_pad', 'R_core', 'F1_padP_coreR') if policy == 'exact-rallies' else (
                    ('P_export', 'R_export', 'F1_export') if policy == 'reviewed-export' else ('P_reviewed', 'R_reviewed', 'F1_reviewed'))
                model_key = 'paddedModelExportSeconds' if policy == 'exact-rallies' else 'modelExportSeconds'
                human_key = 'paddedHumanExportSeconds' if policy == 'exact-rallies' else 'humanExportSeconds'
                for m in panel['padding']:
                    lines.append(f"| {model['mode']} | {policy} | {panel['scope']} | {m['paddingSecondsBeforeAndAfter']:.0f}s | "
                                 f"{percent(m[keys[0]])} | {percent(m[keys[1]])} | {percent(m[keys[2]])} | "
                                 f"{m[model_key]:.3f} | {m[human_key]:.3f} | {m['exportDurationDifferenceSeconds']:+.3f} |")
    lines += ['', 'The JSON includes every feasible target-99 fit with all four padding cases and aggregate event/coverage diagnostics. '
              'Fit and variant indexes are report-local and disclose no recording identifiers. '
              'Exact human rallies, reviewed drafts and reviewed export coverage must remain separate evaluation panels; '
              'export coverage does not establish rally boundaries or wholly missed rally counts.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--baseline-experiment', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--all-video-evaluation', type=Path)
    args = parser.parse_args()
    value = report(args.experiment, args.baseline_experiment, args.all_video_evaluation)
    destination = args.output or args.experiment
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'selection-report.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (destination/'selection-report.md').write_text(markdown(value), encoding='utf-8')
    print(json.dumps(dict(feasible99FitCount=value['feasible99FitCount'], selected=[dict(
        mode=row['mode'], **{key: row['primary'][key] for key in ('P_pad', 'R_core', 'F1_padP_coreR')},
        whollyMissedSavedHumanRallies=row['whollyMissedSavedHumanRallies']) for row in value['selected']])))


if __name__ == '__main__':
    main()
