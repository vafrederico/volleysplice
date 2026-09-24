"""Additive, proof-gated correction of impossible recall overshoots only.

The registered decoder, metric implementation, candidate ordering and selector
remain unchanged. In-range values (including all values below one) are preserved.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
from numbers import Real
from pathlib import Path

from . import neural_generalization_results as frozen
from . import neural_recall_sweep as sweep
from .crop_evaluation import pad_and_merge_intervals, subtract_intervals
from .neural_generalization_results import (bound_fit, selection_examples, load_task,
    COMMON_GROUPS, MODEL_NAMES, verify, shard_from_fit, load_score_shards)

REPO = Path(__file__).resolve().parents[1]
POLICY = 'full-containment-recall-overshoot-v1'
MAX_RECALL_ULPS = 4
POLICY_CONTRACT = {
    'policy': POLICY, 'maxRecallUlps': MAX_RECALL_ULPS,
    'correctedField': 'innerR_core', 'correctedValue': 1.0,
    'proof': 'exact empty core-minus-retained after registered padding/join/ignored subtraction',
    'validValuesUnchanged': True, 'belowOneUnchanged': True,
    'f1AndCandidateOrderUnchanged': True, 'strictFloorComparisonUnchanged': True,
    'f1OvershootsFailClosed': True, 'zeroCorrectionCasesRequireGate': True,
    'externalOutcomesUsed': False,
}


def fraction_json(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def exact_duration(intervals):
    return sum((Fraction.from_float(float(v.end)) - Fraction.from_float(float(v.start))
                for v in intervals), Fraction())


def containment_proof(examples, scores, candidate):
    """Use existing interval operations; the companion audit has its own sweep."""
    rows = []
    total_core, total_intersection = Fraction(), Fraction()
    for example in examples:
        decoded = frozen.base.decode(example, scores[candidate['epoch']][example.id], candidate['decoder'])
        core = subtract_intervals(example.truth, example.ignored)
        retained = subtract_intervals(pad_and_merge_intervals(decoded, example.duration, 2., 3.), example.ignored)
        missing = subtract_intervals(core, retained)
        sweep.require(not missing, 'Recall overshoot has nonempty omitted core; no correction permitted')
        intersection = subtract_intervals(core, missing)
        core_exact, intersection_exact = exact_duration(core), exact_duration(intersection)
        sweep.require(core_exact > 0 and core_exact == intersection_exact, 'Exact containment measure differs')
        serialize = lambda values: [[v.start, v.end] for v in values]
        rows.append({'id': example.id, 'coreUnion': serialize(core), 'retainedUnion': serialize(retained),
            'missingCoreIntervals': serialize(missing), 'coreExact': fraction_json(core_exact),
            'intersectionExact': fraction_json(intersection_exact)})
        total_core += core_exact; total_intersection += intersection_exact
    sweep.require(rows and total_core == total_intersection, 'Empty or invalid containment population')
    return {'recordings': rows, 'pooledCoreExact': fraction_json(total_core),
            'pooledIntersectionExact': fraction_json(total_intersection), 'fullCoreContainment': True}


def correct_candidates(raw_candidates, proof_builder):
    normalized, corrections = deepcopy(raw_candidates), []
    for index, candidate in enumerate(raw_candidates):
        recall, f1 = candidate['innerR_core'], candidate['innerF1_padP_coreR']
        sweep.require(all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v)
                          for v in (recall, f1)), 'Invalid candidate metric type/value')
        sweep.require(0 <= f1 <= 1, 'F1 outside [0,1]; this amendment cannot correct F1')
        sweep.require(0 <= recall <= 1. + MAX_RECALL_ULPS * math.ulp(1.), 'Recall outside narrow correction envelope')
        if recall > 1:
            proof = proof_builder(index, candidate)
            sweep.require(proof['fullCoreContainment'] is True and proof['recordings']
                and all(not row['missingCoreIntervals'] for row in proof['recordings'])
                and proof['pooledCoreExact'] == proof['pooledIntersectionExact'], 'Containment proof is incomplete')
            normalized[index]['innerR_core'] = 1.0
            corrections.append({'candidateIndex': index, 'rawRecall': recall, 'correctedRecall': 1.0, 'proof': proof})
    return normalized, corrections


def verify_plan(path):
    plan = sweep.read(path)
    sweep.require(plan['kind'] == 'generalization-selection-containment-amendment-v1'
        and plan['policy'] == POLICY_CONTRACT, 'Unknown containment correction plan')
    for ref in [plan['protocol'], plan['diagnostic'], *plan['registrations'], *plan['code'].values()]:
        verify(ref)
    sweep.require(plan['code']['analysis/neural_selection_containment_correction.py'] == sweep.identity(__file__),
                  'Correction implementation differs from plan')
    return plan


def own_code():
    return {**frozen.own_code(), 'analysis/neural_selection_containment_correction.py': sweep.identity(__file__)}


# Original selection orchestration retained; only candidate normalization is additive.
def select_task_with_correction(plan_path, task_path, fit_directory, output, historical_directory=None):
    plan = verify_plan(plan_path)
    task, manifest, features, _ = load_task(task_path)
    sweep.require(task['registration'] in plan['registrations'], 'Task is outside amendment registrations')
    output, fit_directory = Path(output), Path(fit_directory)
    sweep.require(output.resolve().is_relative_to('/mnt/freenas'), 'Selection output must use NAS')
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
    raw_candidates = sweep.build_candidate_table(examples, scores, selection_policy=policy)
    candidates, corrections = correct_candidates(raw_candidates,
        lambda index, candidate: containment_proof(examples, scores, candidate))
    evidence[str(Path(plan_path))] = sweep.identity(plan_path)
    result = {'kind': 'frozen-generalization-selection-v1', 'task': sweep.identity(task_path),
        'taskId': task['taskId'], 'selectionDesign': task['selectionDesign'],
        'selectionLabelPolicy': policy, 'selectionMetricsAreGoldAccuracy': policy == 'exact-rallies',
        'inferencePolicy': inference_policy, 'selectionRecordingIds': [e.id for e in examples],
        'selectionSourceGroups': sorted({e.group for e in examples}), 'scoreShards': shards,
        'evidence': list(evidence.values()), 'code': own_code(),
        'selectionGoldSha256': sweep.canonical([{**sweep.serial_rows([e.row([])])[0],
            'times': e.times.tolist(), 'valid': e.valid.tolist(), 'labelPolicy': e.label_policy} for e in examples]),
        'rawCandidates': raw_candidates,
        'candidateMetricCorrection': {'plan': sweep.identity(plan_path), 'policy': POLICY,
            'maxRecallUlps': MAX_RECALL_ULPS, 'corrections': corrections},
        'candidates': candidates, 'floors': sweep.select_floors(candidates),
        'targetPaddingSeconds': 2, 'paddingSeconds': [0, 1, 2, 3], 'joinGapSeconds': 3,
        'externalEvaluationOutcomesUsed': False, 'precisionPolicy': 'FP32 choices transferred unchanged to FP16/INT8'}
    sweep.write_new(output, result)
    return result

