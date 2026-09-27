#!/usr/bin/env python3
"""Independent exact-value proof gate for the explicitly adapted auditor execution."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_experiment import load_task

POLICY = {'kind': 'lossless-endpoint-float-serialization-v1',
    'replacedModuleSymbol': 'calibration_examples', 'changedFields': ['rallies.*.start', 'rallies.*.end',
        'ignoredIntervals.*.start', 'ignoredIntervals.*.end'], 'exactRationalEqualityRequired': True,
    'otherFieldsTypesAndOrderUnchanged': True, 'clippingOrMergingAllowed': False,
    'historicalBranchAdapted': False, 'selectionBytesChanged': False,
    'delegatedAuditorFieldIsBodyIdentityOnly': True, 'independentCompanionRequired': True}


def normalize_gold(before):
    after, changes = deepcopy(before), []
    for record_index, row in enumerate(before):
        for field in ('rallies', 'ignoredIntervals'):
            for interval_index, interval in enumerate(row[field]):
                for endpoint in ('start', 'end'):
                    original = interval[endpoint]
                    io.require(type(original) in (float, int) and math.isfinite(original), 'Invalid original endpoint')
                    cast = float(original)
                    rational = Fraction(original)
                    io.require(rational == Fraction.from_float(cast), 'Float serialization loses exact endpoint value')
                    after[record_index][field][interval_index][endpoint] = cast
                    if type(original) is not float:
                        changes.append({'path': f'/{record_index}/{field}/{interval_index}/{endpoint}',
                            'before': original, 'after': cast, 'beforeType': type(original).__name__, 'afterType': 'float',
                            'exactValue': {'numerator': rational.numerator, 'denominator': rational.denominator}})
    return after, changes


def check_proof(proof, before, selected_gold_hash):
    after, changes = normalize_gold(before)
    io.require(proof['replacedSymbols'] == ['calibration_examples']
        and proof['otherFunctionBindingsUnchanged'] is True and proof['selectionBytesUnchanged'] is True,
        'Only the declared serialization function may be adapted')
    io.require(io.canonical(proof['beforeGold']) == io.canonical(before)
        and io.canonical(proof['afterGold']) == io.canonical(after)
        and proof['beforeGoldSha256'] == io.canonical(before)
        and proof['afterGoldSha256'] == io.canonical(after) == selected_gold_hash
        and io.canonical(proof['changes']) == io.canonical(changes),
        'Endpoint transformation, unchanged fields/order, or gold identities differ')
    return changes


def audit(plan_path, selection_path, base_path, proof_path, output):
    output = Path(output)
    io.require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(), 'New NAS gate required')
    source = REPO/'scripts/audit-neural-generalization-selection.py'
    spec = importlib.util.spec_from_file_location('unadapted_auditor_for_serialization_proof', source)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    evidence = old.Evidence()
    plan_ref = evidence.capture(plan_path); plan = evidence.document(plan_ref)
    io.require(plan['kind'] == 'generalization-selection-serialization-amendment-v1'
        and io.canonical(plan['policy']) == io.canonical(POLICY), 'Serialization policy changed')
    evidence.closure(plan)
    io.require(plan['code']['scripts/audit-neural-selection-serialization.py'] == io.identity(__file__), 'Independent gate code is not bound')
    selection_ref = evidence.capture(selection_path); selection = evidence.document(selection_ref)
    base_ref = evidence.capture(base_path); base = evidence.document(base_ref)
    proof_ref = evidence.capture(proof_path); proof = evidence.document(proof_ref)
    io.require(base['kind'] == 'independent-generalization-selection-audit-v1' and base['passed'] is True
        and base['selection'] == selection_ref and base['task'] == selection['task']
        and base['auditor'] == io.identity(source) and base['counts']['candidates'] == 192
        and 'executionAdapter' not in base, 'Delegated frozen auditor base receipt differs')
    evidence.closure(base['references'])
    io.require(proof['kind'] == 'lossless-auditor-endpoint-serialization-proof-v1'
        and proof['plan'] == plan_ref and proof['selection'] == selection_ref and proof['task'] == selection['task']
        and proof['executionAdapter'] == plan['code']['analysis/neural_selection_serialization.py']
        and proof['delegatedAuditor'] == io.identity(source), 'Execution proof lineage differs')
    task, manifest, features, _ = load_task(evidence.bind(selection['task']))
    correction = evidence.document(plan['correctionPlan'])
    io.require(selection['candidateMetricCorrection']['plan'] == plan['correctionPlan']
        and task['registration'] in correction['registrations'], 'Selection correction plan or registration scope differs')
    io.require(task['variant'] != 'original-corpus', 'Historical branch must not be adapted')
    # This is the original function in a fresh, unmodified auditor module.
    examples = old.calibration_examples(task, io.read(manifest), io.read(features), evidence)
    before = [{**io.serial_rows([e.row([])])[0], 'times': e.times.tolist(), 'valid': e.valid.tolist(),
               'labelPolicy': e.label_policy} for e in examples]
    changes = check_proof(proof, before, selection['selectionGoldSha256'])
    old.audit_floors(selection)
    for ref in list(evidence.references.values()): evidence.bind(ref)
    result = {'kind': 'independent-selection-serialization-audit-v1', 'passed': True,
        'plan': plan_ref, 'selection': selection_ref, 'baseAudit': base_ref, 'proof': proof_ref,
        'task': selection['task'], 'changedEndpointTypes': len(changes), 'zeroDifferenceAdaptation': not changes,
        'everyEndpointExactRationalValuePreserved': True, 'allOtherFieldsTypesAndOrderPreserved': True,
        'onlyRegisteredFunctionReplacement': True, 'auditor': io.identity(__file__),
        'evidence': list(evidence.references.values()), 'externalOutcomesRead': False, 'trainingPerformed': False}
    io.write_new(output, result)
    return result
