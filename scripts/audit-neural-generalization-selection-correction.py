#!/usr/bin/env python3
"""Independent proof gate for a narrowly corrected calibration recall ratio.

The historical selector, candidate arithmetic, and fit recipes remain unchanged.
Only an out-of-domain recall above one, bounded by four ULPs and independently
proven to have no missing core interval, may become exactly one. Valid F1 and
recall fields, candidate order, and strict floor comparisons remain untouched.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as io
from analysis import neural_generalization_results as results
from analysis import neural_development as decoder
from analysis.neural_generalization_experiment import load_task

POLICY = 'full-containment-recall-overshoot-v1'
MAX_ULPS = 4
POLICY_CONTRACT = {
    'policy': POLICY, 'maxRecallUlps': MAX_ULPS,
    'correctedField': 'innerR_core', 'correctedValue': 1.0,
    'proof': 'exact empty core-minus-retained after registered padding/join/ignored subtraction',
    'validValuesUnchanged': True, 'belowOneUnchanged': True,
    'f1AndCandidateOrderUnchanged': True, 'strictFloorComparisonUnchanged': True,
    'f1OvershootsFailClosed': True, 'zeroCorrectionCasesRequireGate': True,
    'externalOutcomesUsed': False,
}


def exact_json(left, right):
    return io.canonical(left) == io.canonical(right)


def module(path):
    spec = importlib.util.spec_from_file_location('independent_base_selection_gate', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def atomic(left, right=(), operation='union'):
    """Boolean interval union via atomic endpoints, independent of crop code."""
    io.require(operation in ('union', 'intersection', 'difference'), 'Unknown interval operation')
    events = {}
    for side, collection in enumerate((left, right)):
        for start, end in collection:
            io.require(math.isfinite(start) and math.isfinite(end) and start <= end, 'Invalid interval endpoint')
            if start == end:
                continue
            events.setdefault(start, [0, 0])[side] += 1
            events.setdefault(end, [0, 0])[side] -= 1
    points, active, output = sorted(events), [0, 0], []
    for start, end in zip(points, points[1:]):
        active[0] += events[start][0]
        active[1] += events[start][1]
        keep = ((active[0] > 0 or active[1] > 0) if operation == 'union' else
                (active[0] > 0 and active[1] > 0) if operation == 'intersection' else
                (active[0] > 0 and active[1] == 0))
        if keep:
            if output and output[-1][1] == start:
                output[-1][1] = end
            else:
                output.append([start, end])
    return output


def retained(intervals, duration, ignored):
    clipped = [[max(0., item.start - 2.), min(duration, item.end + 2.)] for item in intervals]
    union = atomic([span for span in clipped if span[0] < span[1]])
    joined = []
    for start, end in union:
        if joined and 0 < start - joined[-1][1] < 3.:
            joined[-1][1] = end
        else:
            joined.append([start, end])
    return atomic(joined, ignored, 'difference')


def exact_measure(intervals):
    return sum((Fraction.from_float(float(end)) - Fraction.from_float(float(start))
                for start, end in intervals), Fraction(0))


def rational(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def coverage_proof(examples, scores, candidate):
    records, pooled_core, pooled_intersection = [], Fraction(0), Fraction(0)
    for example in examples:
        ignored = [[span.start, span.end] for span in example.ignored]
        core = atomic([[span.start, span.end] for span in example.truth], ignored, 'difference')
        raw = decoder.decode(example, scores[candidate['epoch']][example.id], candidate['decoder'])
        kept = retained(raw, example.duration, ignored)
        missing = atomic(core, kept, 'difference')
        intersection = atomic(core, kept, 'intersection')
        core_exact, intersection_exact = exact_measure(core), exact_measure(intersection)
        io.require(core_exact > 0, 'No evaluable core in correction proof')
        pooled_core += core_exact
        pooled_intersection += intersection_exact
        records.append({'id': example.id, 'coreUnion': core, 'retainedUnion': kept,
                        'missingCoreIntervals': missing, 'coreExact': rational(core_exact),
                        'intersectionExact': rational(intersection_exact)})
    return {'recordings': records, 'pooledCoreExact': rational(pooled_core),
            'pooledIntersectionExact': rational(pooled_intersection),
            'fullCoreContainment': bool(records) and all(not row['missingCoreIntervals'] for row in records)
                                   and pooled_core == pooled_intersection}


def check_tables(raw, corrected, corrections, prove):
    """Require exact field preservation and independent correction proofs."""
    io.require(len(raw) == len(corrected) == 192, 'Correction candidate grid is incomplete')
    expected, proofs = [], []
    for index, candidate in enumerate(raw):
        recall, f1 = candidate['innerR_core'], candidate['innerF1_padP_coreR']
        io.require(math.isfinite(recall) and recall >= 0 and math.isfinite(f1) and 0 <= f1 <= 1,
                   'Unsupported invalid metric; no F1 or negative-recall repair is authorized')
        updated = dict(candidate)
        if recall > 1:
            io.require(recall <= 1. + MAX_ULPS * math.ulp(1.), 'Recall overshoot exceeds frozen ULP envelope')
            proof = prove(candidate)
            io.require(proof['fullCoreContainment'] is True and proof['recordings']
                       and all(row['missingCoreIntervals'] == [] for row in proof['recordings'])
                       and proof['pooledCoreExact'] == proof['pooledIntersectionExact'],
                       'A positive missing interval cannot be promoted to full recall')
            updated['innerR_core'] = 1.
            proofs.append({'candidateIndex': index, 'rawRecall': recall, 'correctedRecall': 1., 'proof': proof})
        expected.append(updated)
    io.require(exact_json(corrected, expected), 'Candidate fields/order changed beyond permitted recall corrections')
    io.require(exact_json(corrections, proofs), 'Correction inventory or exact containment proof differs')
    return proofs


def audit(selection_path, selection_audit_path, output):
    output = Path(output)
    io.require(str(output.resolve()).startswith(private_value('private-reference-0060')) and not output.exists(), 'New direct-NAS gate required')
    old = module(REPO / 'scripts/audit-neural-generalization-selection.py')
    evidence = old.Evidence()
    selection_ref = evidence.capture(selection_path)
    selection = evidence.document(selection_ref)
    ordinary_ref = evidence.capture(selection_audit_path)
    ordinary = evidence.document(ordinary_ref)
    io.require(ordinary['kind'] == 'independent-generalization-selection-audit-v1' and ordinary['passed'] is True
               and ordinary['selection'] == selection_ref and ordinary['task'] == selection['task']
               and ordinary['taskId'] == selection['taskId']
               and ordinary['auditor'] == io.identity(REPO / 'scripts/audit-neural-generalization-selection.py'),
               'Current ordinary selection audit is required')
    evidence.closure(ordinary['references'])
    correction = selection['candidateMetricCorrection']
    io.require(correction['policy'] == POLICY and correction['maxRecallUlps'] == MAX_ULPS,
               'Correction policy differs from the fixed four-ULP rule')
    plan = evidence.document(correction['plan'])
    # The additive plan schema is checked before publication alongside its code.
    io.require(plan['kind'] == 'generalization-selection-containment-amendment-v1', 'Unknown correction plan')
    io.require(exact_json(plan['policy'], POLICY_CONTRACT), 'Correction plan rule differs')
    evidence.closure(plan)
    io.require(io.identity(__file__) in list(plan['code'].values()), 'Independent correction auditor is not frozen in plan')
    task_path = evidence.bind(selection['task'])
    task, manifest, features, _ = load_task(task_path)
    io.require(task['registration'] in plan['registrations'], 'Task registration is outside this amendment')
    if task['variant'] == 'original-corpus':
        protocol_ref, = [r for r in selection['evidence'] if Path(r['path']).name == 'protocol.json'
                        and io.read(r['path']).get('kind') == 'historical-recall-floor-sweep-v1']
        protocol = evidence.document(protocol_ref)
        records = evidence.document(protocol['records'])
        examples = io.examples_from_rows(records['records'])
    else:
        examples = results.selection_examples(task, manifest, features)
    io.require(selection['selectionRecordingIds'] == [e.id for e in examples]
               and selection['selectionSourceGroups'] == sorted({e.group for e in examples}), 'Calibration scope differs')
    gold = [{**io.serial_rows([e.row([])])[0], 'times': e.times.tolist(), 'valid': e.valid.tolist(),
             'labelPolicy': e.label_policy} for e in examples]
    io.require(selection['selectionGoldSha256'] == io.canonical(gold), 'Calibration labels/timestamps/validity changed')
    scores = old.scores_from_shards(examples, selection['scoreShards'], evidence)
    raw = io.build_candidate_table(examples, scores, selection_policy=selection['selectionLabelPolicy'])
    io.require(exact_json(selection['rawCandidates'], raw), 'Raw candidates differ from the unchanged frozen arithmetic')
    proofs = check_tables(raw, selection['candidates'], correction['corrections'],
                          lambda candidate: coverage_proof(examples, scores, candidate))
    old.audit_floors(selection)
    for ref in list(evidence.references.values()):
        evidence.bind(ref)
    result = {'kind': 'independent-generalization-selection-correction-audit-v1', 'passed': True,
              'selection': selection_ref, 'selectionAudit': ordinary_ref, 'plan': correction['plan'],
              'task': selection['task'], 'taskId': task['taskId'], 'normalizationPolicy': POLICY,
              'maxRecallUlps': MAX_ULPS, 'checkedCandidateCount': len(raw), 'correctionCount': len(proofs),
              'indices': [row['candidateIndex'] for row in proofs], 'corrections': proofs,
              'rawCandidateReplayExact': True, 'allValidFieldsAndF1Unchanged': True,
              'strictFloorsAndStableTieOrderVerified': True, 'independentExactContainmentProof': True,
              'auditor': io.identity(__file__), 'evidence': list(evidence.references.values()),
              'externalOutcomesRead': False, 'trainingPerformed': False, 'gpuUsed': False}
    io.write_new(output, result)
    print(json.dumps({'passed': True, 'correctionCount': len(proofs), 'output': io.identity(output)}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    io.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')) and not args.output.exists(), 'New direct-NAS gate required')
    audit(args.selection, args.selection_audit, args.output)
