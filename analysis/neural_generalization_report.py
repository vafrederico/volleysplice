"""Flatten audited operating points and summarize complete matched split draws."""
from __future__ import annotations

from collections import defaultdict
import statistics
from pathlib import Path

from . import neural_recall_sweep as sweep
from .neural_generalization_results import FILTERS, MODEL_NAMES, is_clean, metric_rows
from .neural_generalization_inputs import verified

METRICS = ('precisionValue', 'recallValue', 'f1Value', 'exportSeconds', 'humanExportSeconds',
    'exportDurationDifferenceSeconds', 'correctlyRemovedSeconds', 'incorrectExportSeconds',
    'wantedExportOmittedSeconds', 'missedCoreSeconds', 'missedReviewedLiveSeconds',
    'completeRallyLosses', 'partialRallyLosses', 'eventPrecision', 'eventRecall', 'eventF1',
    'startBoundaryMaeSeconds', 'endBoundaryMaeSeconds')
PANEL_KEYS = ('panelId', 'labelPolicy', 'productionFilter')


def scope_id(ids, scopes):
    ordered = sorted(ids)
    key = sweep.canonical(ordered)[:20]
    sweep.require(key not in scopes or scopes[key] == ordered, 'Scope hash collision')
    scopes[key] = ordered
    return key


def point_rows(result, scopes):
    """Each model/draw/floor keeps its declared full population, even if infeasible."""
    series = []
    base = {k: result[k] for k in ('model', 'variant', 'draw', 'precision')}
    for floor in result['floors']:
        chosen = result['operatingPoints'].get(floor['operatingPointKey'])
        point_panels = {tuple(p[k] for k in PANEL_KEYS): p for p in chosen['panels']} if chosen else {}
        for panel in result['panels']:
            panel_key = tuple(panel[k] for k in PANEL_KEYS)
            value = point_panels.get(panel_key)
            sweep.require(value is None or value['recordingIds'] == panel['recordingIds'], 'Selected point population drifted')
            status = floor['status'] if panel['recordingIds'] else 'empty-scope'
            if value is not None:
                status = value['status']
            metrics = {m['paddingSeconds']: m for m in value['padding']} if value and value['padding'] else {}
            for pad in sweep.PADDINGS:
                metric = metrics.get(pad, {})
                decision = floor.get('selection') or {}
                selected = decision.get('selected') or {}
                series.append({**base, **{k: panel[k] for k in PANEL_KEYS},
                    'floorPercent': floor['floorPercent'], 'paddingSeconds': pad,
                    'status': status, 'scopeId': scope_id(panel['recordingIds'], scopes),
                    'aggregationScope': 'one model fit; pooled recording durations',
                    'operatingPointKey': floor['operatingPointKey'],
                    'calibrationRecall': selected.get('innerR_core'),
                    'calibrationF1': selected.get('innerF1_padP_coreR'),
                    'maximumCalibrationRecall': decision.get('maximumInnerRecall'),
                    'eligibleCandidateCount': decision.get('eligibleCandidateCount'),
                    **{key: metric.get(key) for key in METRICS}})
    return series


def historical_rows(report, inventory, scopes, precision='fp32'):
    series = []
    reverse = {v: k for k, v in MODEL_NAMES.items()}
    for cell in report['cells']:
        for panel_id, panel in cell['panels'].items():
            sweep.require(panel['labelPolicy'] == 'exact-rallies', 'Unexpected historical label policy')
            ids = panel['expectedRecordingIds']
            cache = {}
            for floor in panel['result']['floors']:
                for production_filter in FILTERS:
                    selected_ids = [key for key in ids if is_clean(inventory[key], production_filter)]
                    key = sweep.canonical({'predictions': floor['predictions'], 'ids': selected_ids})
                    available = bool(selected_ids) and floor['completeEvaluationScope']
                    if available and key not in cache:
                        rows = [r for r in floor['predictions'] if r['id'] in selected_ids]
                        sweep.require({r['id'] for r in rows} == set(selected_ids), 'Historical scope incomplete')
                        cache[key] = metric_rows(rows, 'exact-rallies')
                    for pad in sweep.PADDINGS:
                        metric = cache[key][pad] if available else {}
                        series.append({'model': reverse[cell['model']], 'variant': 'historical-nested',
                            'draw': cell['seed'], 'precision': precision, 'panelId': panel_id,
                            'labelPolicy': 'exact-rallies', 'productionFilter': production_filter,
                            'floorPercent': floor['floorPercent'], 'paddingSeconds': pad,
                            'status': 'available' if available else 'empty-scope' if not selected_ids else 'infeasible-inner-recall',
                            'scopeId': scope_id(selected_ids, scopes),
                            'aggregationScope': 'one seed; pooled disjoint source-held outer folds',
                            **{name: metric.get(name) for name in METRICS}})
    return series


def summarize_draws(series, expected_draws):
    """Never average favorable subsets or unlike recording populations."""
    groups = defaultdict(list)
    identity = ('model', 'variant', 'precision', 'panelId', 'labelPolicy', 'productionFilter', 'floorPercent', 'paddingSeconds')
    for row in series:
        if row['variant'] != 'fixed-production':
            groups[tuple(row[k] for k in identity)].append(row)
    summaries = []
    for key, rows in groups.items():
        values = dict(zip(identity, key))
        required = expected_draws[(values['variant'], values['model'], values['precision'])]
        actual = [r['draw'] for r in rows]
        sweep.require(len(actual) == len(set(actual)) and set(actual) == set(required), 'Registered draw population incomplete')
        same_scope = len({r['scopeId'] for r in rows}) == 1
        if not same_scope:
            # These per-task-unseen panels intentionally vary with source splits;
            # the individual rows remain available, with no synthetic mean.
            continue
        complete = all(r['status'] == 'available' for r in rows)
        summary = {**values, 'draw': 'all registered draws', 'scopeId': rows[0]['scopeId'],
            'status': 'available' if complete else 'empty-scope' if all(r['status'] == 'empty-scope' for r in rows) else 'incomplete-registered-draws',
            'aggregationScope': 'mean across every registered draw; each draw pools the same recordings',
            'registeredDraws': list(required), 'completeDraws': [r['draw'] for r in rows if r['status'] == 'available']}
        summary.update({name: statistics.mean(r[name] for r in rows)
                        if complete and all(r.get(name) is not None for r in rows) else None for name in METRICS})
        for name in ('precisionValue', 'recallValue', 'f1Value'):
            summary[name+'Range'] = [min(r[name] for r in rows), max(r[name] for r in rows)] if complete else None
        summaries.append(summary)
    return summaries


def task_membership(result, all_records):
    if 'task' not in result:
        fit, calibration = [], []
        for row in all_records:
            p = row['productionExposure']['rallyPipeline']
            if p['trainingOrRelated']:
                fit.append(row['id'])
            if p['directCalibrationHeads'] or p['sameGroupCalibrationHeads']:
                calibration.append(row['id'])
        return {'id': result['taskId'], 'model': result['model'], 'variant': result['variant'],
            'draw': result['draw'], 'precision': result['precision'],
            'memberships': {'fit': fit, 'calibrate': calibration,
                'evaluate': sorted({key for p in result['panels'] for key in p['recordingIds']}),
                'infer': [r['id'] for r in all_records if r['environment'] != 'beach']},
            'roleInterpretation': 'Production fit/calibration includes source-related aliases, not only direct fitting files'}
    task = sweep.read(verified(result['task']))
    selection = sweep.read(verified(result['selection']))
    fit_directory = Path(result['selection']['path']).parent
    fit = sweep.read(fit_directory/'fit-result.json')
    completed = sweep.read(verified(fit['temporal']))
    source_rows = {r['id']: r for r in sweep.read(verified(task['manifest']))['records']}
    reuse = sweep.read(verified(fit['trainingReuse'])) if fit.get('trainingReuse') else None
    execution = {'trainingPerformedForThisTask': reuse is None,
        'physicalOwnerTaskId': sweep.read(verified(reuse['sourceTask']))['taskId'] if reuse else task['taskId'],
        'reusedAllFixedCheckpointEpochs': list(sweep.EPOCHS) if reuse else [],
        'calibrationAndSelectionAreTaskLocal': True,
        'reuseReceipt': fit.get('trainingReuse'), 'reusePlan': reuse['plan'] if reuse else None}
    return {'id': task['taskId'], 'model': task['model'], 'variant': task['variant'],
        'draw': result['draw'], 'precision': result['precision'],
        'selectionLabelPolicy': selection['selectionLabelPolicy'],
        'trainingExecution': execution,
        'trainingStatistics': {'recordings': len(task['trainIds']),
            'sourceGroups': len({source_rows[k]['sourceGroup'] for k in task['trainIds']}),
            'approximateRallyProxyRecordings': sum(bool(source_rows[k].get('experimentalSupervision')) for k in task['trainIds']),
            'optimizerSteps': completed['optimizerSteps'], 'epochs': completed['epochs'],
            'trainingBudgetInterpretation': 'Fixed epochs, not fixed optimizer-update count; moving coverage to four-head supervision changes sampling and scaling too.'},
        'memberships': {'fit': task['trainIds'], 'calibrate': selection['selectionRecordingIds'],
            'evaluate': sorted({key for p in result['panels'] for key in p['recordingIds']}),
            'infer': [r['id'] for r in all_records if r['environment'] != 'beach']}}


def build_report(result_index_path, output, audit_path=None):
    """The index lists every expected task explicitly; missing files fail closed."""
    index = sweep.read(result_index_path)
    sweep.require(index['kind'] == 'registered-generalization-result-index-v1', 'Unknown result index')
    inventory_document = sweep.read(verified(index['inventory']))
    inventory = {r['id']: r for r in inventory_document['records']}
    scopes, series, tasks, references = {}, [], [], []
    expected = {}
    seen = set()
    for entry in index['evaluations']:
        reference = entry['result']
        document = sweep.read(verified(reference))
        results = document['results'] if document['kind'] == 'fixed-production-generalization-evaluation-v1' else [document]
        for result in results:
            signature = (result['taskId'], result['precision'])
            sweep.require(signature not in seen, 'Duplicate result task/precision')
            seen.add(signature)
            series.extend(point_rows(result, scopes))
            tasks.append(task_membership(result, inventory_document['records']))
            if result['variant'] != 'fixed-production':
                key = (result['variant'], result['model'], result['precision'])
                declared = entry['expectedDraws']
                sweep.require(key not in expected or expected[key] == declared, 'Draw declaration differs')
                expected[key] = declared
        references.append(reference)
    for entry in index.get('historical', []):
        report = sweep.read(verified(entry['result']))
        series.extend(historical_rows(report, inventory, scopes, entry.get('precision', 'fp32')))
        for cell in report['cells']:
            reverse = {v: k for k, v in MODEL_NAMES.items()}
            expected[('historical-nested', reverse[cell['model']], entry.get('precision', 'fp32'))] = [3407, 1729, 20260918]
        references.append(entry['result'])
    summaries = summarize_draws(series, expected)
    metadata = {**index.get('metadata', {}), 'auditPassed': False,
        'targetPaddingSeconds': 2, 'paddingCasesSeconds': [0, 1, 2, 3], 'joinGapSeconds': 3,
        'rankingMetric': 'F1_padP_coreR at ±2s on declared calibration sources only',
        'recallFloorsPercent': list(sweep.FLOORS), 'sourceIndex': sweep.identity(result_index_path),
        'resultReferences': references, 'aggregationPolicy': 'Pool recording durations within a draw; mean only all registered complete draws on the identical population.',
        'defaultPanelId': 'common-unseen', 'defaultLabelPolicy': 'exact-rallies',
        'defaultProductionFilter': 'all', 'defaultPaddingSeconds': 2,
        'defaultScenario': 'original-corpus', 'defaultDraw': 'all registered draws', 'defaultPrecision': 'fp32',
        'precisionTransferPolicy': 'FP16/INT8 change DINO embedding extraction only. Temporal heads and all floor-specific decoder/checkpoint selections remain fixed from FP32. Calibration recall describes FP32 eligibility; transfers are not recalibrated.',
        'panelInterpretation': {'all-labeled': 'Includes neural fitting and calibration sources as well as held footage.',
            'common-unseen': 'Same reserved source groups for every new neural fit; excluded from fitting, student teaching and calibration.',
            'outside-original-sources': 'Independent of original-corpus fits; expanded variants deliberately train on some of these sources.',
            'task-unseen-sources': 'Excludes this task\'s fitting and calibration source groups; membership can differ across split draws.'},
        'productionFilterInterpretation': 'Production exposure filters do not remove neural training sources; use a declared unseen panel for generalization.',
        'curveMetricInterpretation': {'exact-rallies': 'Time-based P_pad and R_core: matching padded-human export divided by model export, and retained human rally-core seconds divided by all human rally-core seconds. Event recall/F1 and lost-rally counts are separate.',
            'reviewed-draft': 'Time-based precision and recall against reviewed live intervals; not independent exact semantic-rally boundaries.',
            'reviewed-export': 'Precision and recall against the fixed desired human export union; not semantic rally-core or event recall.'},
        'protectedVideosAreNoLongerUntouchedFutureTests': True,
        'reviewedExportProxiesAreNotIndependentExactEvaluationGold': True,
        'strictProductionUnexposedExactGoldAvailable': False,
        'mobileDeviceQualified': False,
        'detectorArms': 'YOLOX-Nano, SSDLite, MediaPipe and NanoDet have no qualified trained rally head; no rally recall sweep is available for those configurations.'}
    if audit_path:
        audit = sweep.read(audit_path)
        sweep.require(audit['passed'] and audit['resultIndex'] == sweep.identity(result_index_path), 'Final report audit differs')
        metadata.update(auditPassed=True, audit=sweep.identity(audit_path))
    visible = series+summaries
    metadata['emptyScopeRowsRetainedInDetailedEvaluations'] = sum(r['status'] == 'empty-scope' for r in visible)
    visible = [{k: v for k, v in r.items() if v is not None and k != 'operatingPointKey'}
               for r in visible if r['status'] != 'empty-scope']
    result = {'metadata': metadata, 'inventory': inventory_document['records'], 'scopes': scopes,
              'series': visible, 'tasks': tasks}
    if audit_path:
        content = {key: result[key] for key in ('inventory', 'scopes', 'series', 'tasks')}
        sweep.require(audit['reportContentSha256'] == sweep.canonical(content), 'Final report data differs from independently audited payload')
    sweep.write_new(output, result)
    return {'path': str(output), 'rows': len(result['series']), 'tasks': len(tasks), 'auditPassed': metadata['auditPassed']}
