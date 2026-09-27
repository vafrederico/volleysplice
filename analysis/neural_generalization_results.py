"""Selection and evaluation for the registered source-composition study.

Selection is a separate immutable artifact. External panel outcomes never
choose a checkpoint, decoder, recall floor, split, or precision setting.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

from pathlib import Path
from dataclasses import asdict
import numpy as np

from . import neural_recall_sweep as sweep
from . import neural_development as base
from .crop_evaluation import subtract_intervals
from .neural_evaluation import evaluate_predictions
from .neural_generalization_experiment import load_task, MODELS
from .neural_generalization_inputs import load_inference_examples, make_examples_for_evaluation, manifest_rows, feature_entries, verified as verify
from .neural_recall_sweep_adapters import load_score_shards, shard_from_fit
from .schema import Interval

MODEL_NAMES = {'av-tcn': 'av_tcn_short_boost', 'dino-tcn': 'dino_tcn_short_boost',
    'mobile-tcn': 'mobile_tcn', 'distilled-mobile-tcn': 'distilled_mobile_tcn',
    'av-transformer': 'av_transformer', 'dino-transformer': 'dino_transformer'}
POLICY_NAMES = {'exact-core': 'exact-rallies', 'draft-reviewed': 'reviewed-draft', 'export-coverage': 'reviewed-export'}
FILTERS = ('all', 'no-production-training', 'no-production-training-or-calibration')
COMMON_GROUPS = frozenset((private_value('source-group-001'), private_value('source-group-008')))


def own_code():
    root = Path(__file__).resolve().parents[1]
    names = ('analysis/neural_generalization_results.py', 'analysis/neural_recall_sweep.py',
             'analysis/neural_recall_sweep_adapters.py', 'analysis/neural_evaluation.py',
             'analysis/crop_evaluation.py', 'analysis/neural_development.py',
             'analysis/neural_expanded_development.py', 'analysis/neural_recall_operating_point.py')
    return {name: sweep.identity(root/name) for name in names}


def selection_examples(task, manifest, features):
    blind = load_inference_examples(manifest, features, family='av', recording_ids=task['calibrationIds'])
    gold = make_examples_for_evaluation(blind, manifest, scoring_policy='exact-core')
    by_id = {r['id']: r for r in manifest_rows(manifest)[1]}
    return [sweep.SweepExample(e.id, e.group, e.duration, e.times, e.valid, e.truth, e.ignored,
            'export-rally-proxy-selection' if by_id[e.id].get('experimentalSupervision') else 'exact-rallies') for e in gold]


def bound_fit(task_path, task, directory):
    directory = Path(directory)
    fit = sweep.read(directory/'fit-result.json')
    sweep.require(fit['task'] == sweep.identity(task_path) and fit['taskId'] == task['taskId']
                  and fit['config'] == asdict(MODELS[task['model']]), 'Fit owner/model differs')
    sweep.require(Path(fit['temporal']['path']) == directory/'temporal/completed.json', 'Fit temporal path differs')
    completed = sweep.read(verify(fit['temporal']))
    sweep.require(completed['contractSha256'] == task['registrationSha256'] and completed['seed'] == task['seed']
                  and completed['epochs'] == task['epochs'], 'Completed fit contract/seed differs')
    actual = completed['trainIds'] + [k for values in completed['auxiliaryIds'].values() for k in values]
    sweep.require(len(actual) == len(set(actual)) and set(actual) == set(task['trainIds']), 'Fit membership differs')
    expected_calibration = task['calibrationIds'] or ['__fit_output_probe__']
    sweep.require(completed['validationIds'] == expected_calibration, 'Fit calibration population differs')
    if task['model'] == 'distilled-mobile-tcn':
        sweep.require(fit['student'] == sweep.identity(directory/'student/completed.json'), 'Student fit owner differs')
        student = sweep.read(verify(fit['student']))
        sweep.require(student['contractSha256'] == task['registrationSha256'] and student['seed'] == task['seed']
                      and set(student['trainIds']) == set(task['trainIds']), 'Student fitting lineage differs')
        verify(student['weights'])
    return fit, completed


def select_task(task_path, fit_directory, output, historical_directory=None):
    task, manifest, features, _ = load_task(task_path)
    output, fit_directory = Path(output), Path(fit_directory)
    sweep.require(not output.exists(), 'Selection is already frozen')
    fit, _ = bound_fit(task_path, task, fit_directory)
    evidence = {str(fit_directory/'fit-result.json'): sweep.identity(fit_directory/'fit-result.json'),
                fit['temporal']['path']: fit['temporal']}
    if task['variant'] == 'original-corpus':
        sweep.require(historical_directory is not None, 'Original-corpus selection requires source-held original OOF scores')
        folder = Path(historical_directory)
        protocol = sweep.read(folder/'protocol.json')
        plan = sweep.read(folder/'refit-plan.json')
        refit_audit = sweep.read(folder/'audit-refits.json')
        sweep.require(refit_audit['passed'] and refit_audit['kind'] == 'independent-historical-all-outer-epochs-audit-v1'
            and refit_audit['protocol'] == sweep.identity(folder/'protocol.json')
            and refit_audit['plan'] == sweep.identity(folder/'refit-plan.json')
            and refit_audit['counts']['outerFits'] == 72, 'Historical independent refit audit absent')
        sweep.require(plan['protocol'] == sweep.identity(folder/'protocol.json')
            and sweep.canonical(plan['tasks']) == protocol['refitTasksSha256'], 'Historical plan differs')
        for reference in [*protocol['code'], *refit_audit['references']]:
            verify(reference)
        verify(protocol['records'])
        examples = sweep.examples_from_rows(sweep.read(protocol['records']['path'])['records'])
        layouts = sweep.read(folder/'layouts.json')
        sweep.require(layouts['protocol'] == sweep.identity(folder/'protocol.json'), 'Historical protocol differs')
        sweep.require(sweep.canonical(layouts['layouts']) == protocol['layoutsSha256'], 'Historical layouts changed')
        layout, = [r for r in layouts['layouts'] if r['model'] == MODEL_NAMES[task['model']] and r['seed'] == task['seed']]
        shards = []
        for fold in layout['folds']:
            owner = folder/'refits'/layout['model']/str(task['seed'])/f'outer-{fold["outerIndex"]}'
            parity = sweep.read(owner/'parity.json')
            sweep.require(parity['passed'] and parity['plan'] == sweep.identity(folder/'refit-plan.json'), 'Historical refit parity absent')
            expected_task, = [r for r in plan['tasks'] if r['model'] == layout['model']
                and r['seed'] == task['seed'] and r['outerIndex'] == fold['outerIndex']]
            sweep.require(parity['task'] == expected_task and parity['completed'] == sweep.identity(owner/'temporal/completed.json'),
                          'Historical parity task/completed owner differs')
            audited, = [r for r in refit_audit['checks'] if r['task'] == expected_task]
            sweep.require(audited['parity'] == sweep.identity(owner/'parity.json') and audited['completed'] == parity['completed']
                          and audited['allEpochArraysChecked'], 'Historical owner differs from independent audit')
            ids = [e.id for e in examples if e.group == fold['heldSourceGroup']]
            for epoch in sweep.EPOCHS:
                shards.append(shard_from_fit(owner/'temporal', epoch, ids,
                    additional_training_owners=parity['additionalTrainingOwners'], excluded_groups=[fold['heldSourceGroup']]))
        policy = 'exact-rallies'
        inference_policy = 'historical-valid-segments; original eight OOF only'
        evidence[str(folder/'protocol.json')] = sweep.identity(folder/'protocol.json')
        for name in ('audit-refits.json', 'layouts.json', 'refit-plan.json', 'records.json'):
            evidence[str(folder/name)] = sweep.identity(folder/name)
    else:
        examples = selection_examples(task, manifest, features)
        sweep.require(not ({e.group for e in examples} & COMMON_GROUPS), 'Common evaluation group reached selection')
        extra = [sweep.identity(fit_directory/'student/completed.json')] if task['model'] == 'distilled-mobile-tcn' else []
        shards = [shard_from_fit(fit_directory/'temporal', epoch, [e.id for e in examples],
                    additional_training_owners=extra, excluded_groups=[e.group for e in examples]) for epoch in sweep.EPOCHS]
        policy = 'export-rally-proxy-selection' if task.get('selectionLabelPolicy') == 'exact-and-export-rally-proxy' else 'exact-rallies'
        inference_policy = 'blind-full-timeline; all ticks valid before attaching selection labels'
    scores = load_score_shards(examples, shards, evidence=evidence)
    candidates = sweep.build_candidate_table(examples, scores, selection_policy=policy)
    result = {'kind': 'frozen-generalization-selection-v1', 'task': sweep.identity(task_path),
        'taskId': task['taskId'], 'selectionDesign': task['selectionDesign'],
        'selectionLabelPolicy': policy, 'selectionMetricsAreGoldAccuracy': policy == 'exact-rallies',
        'inferencePolicy': inference_policy, 'selectionRecordingIds': [e.id for e in examples],
        'selectionSourceGroups': sorted({e.group for e in examples}), 'scoreShards': shards,
        'evidence': list(evidence.values()), 'code': own_code(),
        'selectionGoldSha256': sweep.canonical([{**sweep.serial_rows([e.row([])])[0],
            'times': e.times.tolist(), 'valid': e.valid.tolist(), 'labelPolicy': e.label_policy} for e in examples]),
        'candidates': candidates, 'floors': sweep.select_floors(candidates),
        'targetPaddingSeconds': 2, 'paddingSeconds': [0, 1, 2, 3], 'joinGapSeconds': 3,
        'externalEvaluationOutcomesUsed': False, 'precisionPolicy': 'FP32 choices transferred unchanged to FP16/INT8'}
    sweep.write_new(output, result)
    return result


def manifest_example(row, times):
    """Attach metric truth; every inference/decoder tick stays valid."""
    parse = lambda values: tuple(Interval(r['start'], r['end'], tuple(r.get('tags', ()))) for r in values)
    policy = POLICY_NAMES[row['scoringPolicy']]
    truth = parse(row['keepTargets'] if policy == 'reviewed-export' else row.get('rallies', []))
    ignored = parse(row.get('ignoredIntervals', []))
    if policy == 'reviewed-export':
        window = row['gameWindow']
        if window['start'] > 0:
            ignored += (Interval(0., window['start']),)
        if window['end'] < row['durationSeconds']:
            ignored += (Interval(window['end'], row['durationSeconds']),)
    return sweep.SweepExample(row['id'], row['sourceGroup'], row['durationSeconds'], times,
                              np.ones(len(times), bool), truth, ignored, policy)


def inference_scores(folder, records, task_reference, panel_reference, precision, panel, fit_directory, completed):
    """Read scores only after the operating points have been selected."""
    examples, scores, evidence = [], {e: {} for e in sweep.EPOCHS}, []
    for row in records:
        receipt_path = Path(folder)/(row['id']+'.json')
        receipt = sweep.read(receipt_path)
        sweep.require(receipt['task'] == task_reference and receipt['panel'] == panel_reference
            and receipt['id'] == row['id'] and receipt['precision'] == precision
            and receipt['manifest'] == panel['manifest'] and receipt['features'] == panel['features']
            and receipt['epochs'] == list(sweep.EPOCHS)
            and receipt['labelsUsed'] is False and receipt['ignoredLabelsUsedForContextOrDecoding'] is False,
            'Inference lineage or label isolation differs')
        path = verify(receipt['output'])
        sweep.require(set(receipt['weights']) == {str(e) for e in sweep.EPOCHS}, 'Inference weights incomplete')
        for epoch in sweep.EPOCHS:
            reference = receipt['weights'][str(epoch)]
            name = f'weights-{epoch}.npz'
            sweep.require(Path(reference['path']) == Path(fit_directory)/'temporal'/name
                          and reference['sha256'] == completed['artifacts'][name], 'Inference checkpoint owner differs')
            verify(reference)
        evidence.append(sweep.identity(receipt_path))
        with np.load(path, allow_pickle=False) as payload:
            sweep.require(set(payload.files) == {'times', *[f'epoch_{e}' for e in sweep.EPOCHS]}, 'Inference archive differs')
            times = payload['times'].copy()
            sweep.require(times.ndim == 1 and len(times) and np.isfinite(times).all()
                          and np.all(np.diff(times) > 0), 'Invalid inference timeline')
            blind, = load_inference_examples(panel['manifest']['path'], panel['features']['path'], family='av', recording_ids=[row['id']])
            sweep.require(np.array_equal(times, blind.times), 'Inference timestamps differ from bound AV grid')
            examples.append(manifest_example(row, times))
            for epoch in sweep.EPOCHS:
                scores[epoch][row['id']] = payload[f'epoch_{epoch}'].copy()
    sweep.validate_scores(examples, scores)
    return examples, scores, evidence


def is_clean(record, production_filter):
    sweep.require(production_filter in FILTERS, 'Unknown production exposure filter')
    if production_filter == 'all':
        return True
    exposure = record['productionExposure']['rallyPipeline']
    return exposure['primaryTrainingClean'] if production_filter == 'no-production-training' else exposure['strictNoFitOrCalibration']


def panel_members(records, inventory, task, original_groups, used_groups=None):
    """Panels are fixed from provenance, independent of scores or outcomes."""
    by_id = {r['id']: r for r in records}
    if used_groups is None:
        source_rows = sweep.read(task['manifest']['path'])['records']
        groups = {r['id']: r['sourceGroup'] for r in source_rows}
        used = {groups[k] for k in task['trainIds'] + task.get('calibrationIds', [])}
    else:
        used = set(used_groups)
    predicates = {'all-labeled': lambda r: True,
        'common-unseen': lambda r: r['sourceGroup'] in COMMON_GROUPS,
        'outside-original-sources': lambda r: r['sourceGroup'] not in original_groups,
        'task-unseen-sources': lambda r: r['sourceGroup'] not in used}
    result = []
    for name, condition in predicates.items():
        for policy in sweep.POLICIES:
            for production_filter in FILTERS:
                ids = [key for key, r in by_id.items() if POLICY_NAMES[r['scoringPolicy']] == policy
                       and condition(r) and is_clean(inventory[key], production_filter)]
                result.append({'panelId': name, 'labelPolicy': policy, 'productionFilter': production_filter,
                               'recordingIds': ids})
    return result


def metric_rows(rows, policy):
    evaluation = sweep.evaluate_rows(rows, policy)
    if policy != 'exact-rallies':
        names = ('P_export', 'R_export', 'F1_export') if policy == 'reviewed-export' else ('P_reviewed', 'R_reviewed', 'F1_reviewed')
        return [{**m, 'paddingSeconds': m['paddingSecondsBeforeAndAfter'],
            'precisionValue': m[names[0]], 'recallValue': m[names[1]], 'f1Value': m[names[2]],
            'exportSeconds': m['modelExportSeconds'], 'missedCoreSeconds': None,
            'completeRallyLosses': None, 'eventF1': None} for m in evaluation['padding']]
    universe = sum(sum(v.end-v.start for v in subtract_intervals((Interval(0., r['durationSeconds']),),
        tuple(Interval(v['start'], v['end']) for v in r['ignoredIntervals']))) for r in rows)
    result = []
    for metric in evaluation['padding']:
        pad = metric['paddingSecondsBeforeAndAfter']
        details = evaluation if pad == 2 else evaluate_predictions(rows, primary_padding_seconds=pad)
        incorrect = metric['paddedModelExportSeconds'] - metric['paddedPrecisionIntersectionSeconds']
        result.append({**metric, 'paddingSeconds': pad, 'precisionValue': metric['P_pad'],
            'recallValue': metric['R_core'], 'f1Value': metric['F1_padP_coreR'],
            'exportSeconds': metric['paddedModelExportSeconds'], 'humanExportSeconds': metric['paddedHumanExportSeconds'],
            'incorrectExportSeconds': incorrect, 'wantedExportOmittedSeconds': metric['paddedHumanExportSeconds']-metric['paddedPrecisionIntersectionSeconds'],
            'correctlyRemovedSeconds': universe-metric['paddedHumanExportSeconds']-incorrect,
            'missedCoreSeconds': metric['coreHumanSeconds']-metric['coreRecallIntersectionSeconds'],
            'completeRallyLosses': details['guardrails']['primaryExportCoverage']['completeRallyLosses'],
            'partialRallyLosses': details['guardrails']['primaryExportCoverage']['partialRallyLosses'],
            **{name: details['guardrails'][name] for name in ('eventPrecision', 'eventRecall', 'eventF1',
                                                            'startBoundaryMaeSeconds', 'endBoundaryMaeSeconds')}})
    return result


def evaluate_task(task_path, fit_directory, selection_path, panel_path, inventory_path, original_manifest,
                  output, precision='fp32', selection_audit_path=None):
    task, _, _, _ = load_task(task_path)
    _, completed = bound_fit(task_path, task, fit_directory)
    selection = sweep.read(selection_path)
    sweep.require(selection['task'] == sweep.identity(task_path) and selection['externalEvaluationOutcomesUsed'] is False,
                  'Wrong selection owner or external outcomes used')
    for reference in selection['code'].values():
        verify(reference)
    for reference in selection['evidence']:
        verify(reference)
    sweep.require(selection_audit_path is not None, 'Independent selection audit required before external evaluation')
    audit = sweep.read(selection_audit_path)
    sweep.require(audit['passed'] and audit['selection'] == sweep.identity(selection_path), 'Selection audit absent or changed')
    panel = sweep.read(panel_path)
    sweep.require(panel['kind'] == 'frozen-independent-inference-panel-v1', 'Wrong independent inference panel')
    manifest = verify(panel['manifest'])
    verify(panel['features'])
    _, all_records = manifest_rows(manifest)
    records = [r for r in all_records if 'evaluate' in r['eligibleRoles'] and r['scoringPolicy'] != 'none']
    sweep.require(not any(r.get('experimentalSupervision') for r in records), 'Proxy fitting labels reached independent evaluation gold')
    inventory_doc = sweep.read(inventory_path)
    inventory = {r['id']: r for r in inventory_doc['records']}
    original_groups = {r['sourceGroup'] for r in sweep.read(original_manifest)['records']}
    folders = Path(fit_directory)/'inference'/sweep.identity(panel_path)['sha256'][:16]/precision
    examples, scores, evidence = inference_scores(folders, records, sweep.identity(task_path), sweep.identity(panel_path), precision,
                                                panel, fit_directory, completed)
    panels = panel_members(records, inventory, task, original_groups)
    all_points, floors = {}, []
    for decision in selection['floors']:
        chosen = decision['selected']
        key = sweep.operating_point_key(chosen) if chosen else None
        if key and key not in all_points:
            decoded = {e.id: base.decode(e, scores[chosen['epoch']][e.id], chosen['decoder']) for e in examples}
            point = {'epoch': chosen['epoch'], 'decoder': chosen['decoder'], 'rowsByPolicy': {}, 'panels': []}
            for policy in sweep.POLICIES:
                selected_examples = [e for e in examples if e.label_policy == policy]
                point['rowsByPolicy'][policy] = sweep.panel_rows(selected_examples, decoded, policy) if selected_examples else []
            for p in panels:
                rows = [r for r in point['rowsByPolicy'][p['labelPolicy']] if r['id'] in p['recordingIds']]
                point['panels'].append({**p, 'status': 'available' if rows else 'empty-scope',
                    'padding': metric_rows(rows, p['labelPolicy']) if rows else None})
            all_points[key] = point
        floors.append({'floorPercent': decision['floorPercent'], 'status': 'available' if chosen else 'infeasible-inner-recall',
                       'operatingPointKey': key, 'selection': decision})
    result = {'kind': 'neural-generalization-task-evaluation-v1', 'task': sweep.identity(task_path),
        'taskId': task['taskId'], 'model': task['model'], 'variant': task['variant'],
        'draw': task.get('splitSeed', task['seed']), 'precision': precision,
        'selection': sweep.identity(selection_path), 'panel': sweep.identity(panel_path),
        'selectionAudit': sweep.identity(selection_audit_path),
        'inventory': sweep.identity(inventory_path), 'originalManifest': sweep.identity(original_manifest),
        'inferenceReceipts': evidence, 'code': own_code(), 'panels': panels,
        'operatingPoints': all_points, 'floors': floors,
        'inferencePolicy': 'All timeline ticks valid; ignored labels applied only after decoding for metric subtraction',
        'precisionPolicy': 'FP32 selected operating points transferred unchanged; no requalification' if precision != 'fp32' else 'FP32',
        'protectedSourcesEvaluated': [r['id'] for r in records if r['protected']],
        'productionPromotionAllowed': False}
    sweep.write_new(output, result)
    return result


def evaluate_production(production_directory, parity_audit_path, panel_path, inventory_path, original_manifest, output):
    """The shipped policy is a fixed comparator, not a recalibrated curve."""
    production_directory = Path(production_directory)
    protocol = sweep.read(production_directory/'protocol.json')
    panel = sweep.read(panel_path)
    sweep.require(protocol['manifest'] == panel['manifest'] and protocol['features'] == panel['features']
                  and protocol['labelsUsedForInference'] is False, 'Production panel or inference policy differs')
    verify(panel['features']); manifest = verify(panel['manifest'])
    for reference in [*protocol['sourceCode'].values(), *protocol['assets'].values()]:
        verify(reference)
    parity = sweep.read(parity_audit_path)
    sweep.require(parity['passed'] and parity['kind'] == 'blind-generalization-production-parity-v1',
                  'Independent production replay qualification absent')
    qualified_protocol = sweep.read(verify(parity['protocol']))
    for key in ('assets', 'sourceCode', 'programSha256', 'modelIds', 'productionPolicy'):
        sweep.require(protocol[key] == qualified_protocol[key], 'Production replay differs from independently qualified implementation')
    _, population = manifest_rows(manifest)
    records = [r for r in population if 'evaluate' in r['eligibleRoles'] and r['scoringPolicy'] != 'none']
    features = feature_entries(records, panel['features']['path'])
    inventory = {r['id']: r for r in sweep.read(inventory_path)['records']}
    original_groups = {r['sourceGroup'] for r in sweep.read(original_manifest)['records']}
    used_groups = {r['sourceGroup'] for r in records if not is_clean(inventory[r['id']], 'no-production-training-or-calibration')}
    panels = panel_members(records, inventory, None, original_groups, used_groups)
    evidence, decoded, examples = [], {k: {} for k in protocol['modelIds']}, []
    for row in records:
        receipt_path = production_directory/(row['id']+'.json')
        receipt = sweep.read(receipt_path)
        sweep.require(receipt['protocol'] == sweep.identity(production_directory/'protocol.json')
            and receipt['id'] == row['id'] and receipt['sourceGroup'] == row['sourceGroup'] and receipt['labelsUsed'] is False,
            'Production recording lineage differs')
        sweep.require(receipt['featureCache'] == features[row['id']]['audiovisual'], 'Production audiovisual input differs')
        with np.load(verify(receipt['probabilities']), allow_pickle=False) as payload:
            times = payload['times'].copy()
        blind, = load_inference_examples(manifest, panel['features']['path'], family='av', recording_ids=[row['id']])
        sweep.require(np.array_equal(times, blind.times), 'Production input timeline differs')
        examples.append(manifest_example(row, times))
        evidence.append(sweep.identity(receipt_path))
        for model in decoded:
            decoded[model][row['id']] = [Interval(v['start'], v['end']) for v in receipt['predictions'][model]]
    results = []
    for model, values in decoded.items():
        point = {'epoch': None, 'decoder': 'fixed shipped product policy', 'rowsByPolicy': {}, 'panels': []}
        for policy in sweep.POLICIES:
            selected = [e for e in examples if e.label_policy == policy]
            point['rowsByPolicy'][policy] = sweep.panel_rows(selected, values, policy) if selected else []
        for p in panels:
            rows = [r for r in point['rowsByPolicy'][p['labelPolicy']] if r['id'] in p['recordingIds']]
            point['panels'].append({**p, 'status': 'available' if rows else 'empty-scope',
                                   'padding': metric_rows(rows, p['labelPolicy']) if rows else None})
        results.append({'kind': 'neural-generalization-task-evaluation-v1', 'taskId': 'fixed-production/'+model,
            'model': model, 'variant': 'fixed-production', 'draw': 'fixed', 'precision': 'shipped',
            'protocol': sweep.identity(production_directory/'protocol.json'), 'productionParityAudit': sweep.identity(parity_audit_path),
            'panel': sweep.identity(panel_path), 'inventory': sweep.identity(inventory_path),
            'originalManifest': sweep.identity(original_manifest), 'inferenceReceipts': evidence, 'code': own_code(),
            'panels': panels, 'operatingPoints': {'fixed': point},
            'floors': [{'floorPercent': f, 'status': 'available', 'operatingPointKey': 'fixed',
                       'selection': None, 'fixedComparatorNoFloorQualification': True} for f in sweep.FLOORS],
            'productionPromotionAllowed': False})
    sweep.write_new(output, {'kind': 'fixed-production-generalization-evaluation-v1', 'results': results})
    return results
