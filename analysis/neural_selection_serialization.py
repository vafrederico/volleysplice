"""Explicit execution adapter for lossless endpoint serialization in a frozen audit."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import importlib.util
import inspect
import math
from pathlib import Path

from . import neural_recall_sweep as io
from .schema import Interval
from .neural_generalization_inputs import verified

REPO = Path(__file__).resolve().parents[1]
POLICY = {'kind': 'lossless-endpoint-float-serialization-v1',
    'replacedModuleSymbol': 'calibration_examples', 'changedFields': ['rallies.*.start', 'rallies.*.end',
        'ignoredIntervals.*.start', 'ignoredIntervals.*.end'], 'exactRationalEqualityRequired': True,
    'otherFieldsTypesAndOrderUnchanged': True, 'clippingOrMergingAllowed': False,
    'historicalBranchAdapted': False, 'selectionBytesChanged': False,
    'delegatedAuditorFieldIsBodyIdentityOnly': True, 'independentCompanionRequired': True}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def gold(examples):
    return [{**io.serial_rows([e.row([])])[0], 'times': e.times.tolist(), 'valid': e.valid.tolist(),
             'labelPolicy': e.label_policy} for e in examples]


def normalize_examples(examples):
    result = []
    for example in examples:
        def intervals(values):
            output = []
            for value in values:
                endpoints = []
                for endpoint in (value.start, value.end):
                    io.require(type(endpoint) in (float, int) and math.isfinite(endpoint), 'Invalid endpoint type/value')
                    cast = float(endpoint)
                    io.require(Fraction(endpoint) == Fraction.from_float(cast), 'Endpoint float cast changes exact value')
                    endpoints.append(cast)
                output.append(Interval(*endpoints, value.tags))
            return tuple(output)
        result.append(replace(example, truth=intervals(example.truth), ignored=intervals(example.ignored)))
    return result


def field_changes(before, after):
    changes = []
    for index, (left, right) in enumerate(zip(before, after, strict=True)):
        for field in ('rallies', 'ignoredIntervals'):
            for interval_index, (one, two) in enumerate(zip(left[field], right[field], strict=True)):
                for endpoint in ('start', 'end'):
                    value, cast = one[endpoint], two[endpoint]
                    if type(value) is not type(cast):
                        rational = Fraction(value)
                        changes.append({'path': f'/{index}/{field}/{interval_index}/{endpoint}',
                            'before': value, 'after': cast, 'beforeType': type(value).__name__, 'afterType': type(cast).__name__,
                            'exactValue': {'numerator': rational.numerator, 'denominator': rational.denominator}})
    return changes


def verify_plan(path):
    plan = io.read(path)
    io.require(plan['kind'] == 'generalization-selection-serialization-amendment-v1'
        and plan['policy'] == POLICY, 'Serialization amendment differs')
    for ref in [plan['protocol'], plan['diagnostic'], plan['correctionPlan'], *plan['code'].values()]: verified(ref)
    io.require(plan['code']['analysis/neural_selection_serialization.py'] == io.identity(__file__), 'Execution adapter source changed')
    return plan


def audit_selection(plan_path, selection_path, output):
    plan = verify_plan(plan_path); output = Path(output)
    io.require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(), 'New NAS audit required')
    selection_ref = io.identity(selection_path); selection = io.read(selection_path)
    io.require(selection['candidateMetricCorrection']['plan'] == plan['correctionPlan'], 'Selection correction plan differs')
    task = io.read(verified(selection['task']))
    io.require(task['variant'] != 'original-corpus', 'Historical serialization/masks are never adapted')
    fit_ref, = [r for r in selection['evidence'] if Path(r['path']).name == 'fit-result.json']
    folder = Path(verified(fit_ref)).parent
    delegated_path = output.with_name(output.stem+'-delegated-base.json')
    proof_path = output.with_name(output.stem+'-serialization-proof.json')
    companion_path = output.with_name(output.stem+'-serialization-gate.json')
    io.require(not any(p.exists() for p in (delegated_path, proof_path, companion_path)), 'Adapter sidecar already exists')
    old = load_module('explicit_isolated_selection_auditor_body', REPO/'scripts/audit-neural-generalization-selection.py')
    original = old.calibration_examples
    functions = {name: value for name, value in vars(old).items() if inspect.isfunction(value)}
    captured = []

    def calibration_examples(task_value, manifest, features, evidence):
        before_examples = original(task_value, manifest, features, evidence)
        after_examples = normalize_examples(before_examples)
        captured.append({'before': gold(before_examples), 'after': gold(after_examples)})
        return after_examples

    # This is the sole deliberate runtime replacement, registered in POLICY and
    # disclosed in the final gate. It occurs only in this isolated module object.
    old.calibration_examples = calibration_examples
    io.require(all(vars(old)[name] is value for name, value in functions.items() if name != 'calibration_examples'),
               'Unexpected auditor function replacement')
    old.audit(selection['task']['path'], folder, selection_path, delegated_path)
    io.require(len(captured) == 1 and all(vars(old)[name] is value for name, value in functions.items()
        if name != 'calibration_examples') and old.calibration_examples is calibration_examples,
        'Auditor execution replacement scope differs')
    before, after = captured[0]['before'], captured[0]['after']
    io.require(io.canonical(after) == selection['selectionGoldSha256'] and io.identity(selection_path) == selection_ref,
               'Normalized gold or immutable selection differs')
    proof = {'kind': 'lossless-auditor-endpoint-serialization-proof-v1', 'plan': io.identity(plan_path),
        'selection': selection_ref, 'task': selection['task'], 'executionAdapter': io.identity(__file__),
        'delegatedAuditor': io.identity(REPO/'scripts/audit-neural-generalization-selection.py'),
        'replacedSymbols': ['calibration_examples'], 'otherFunctionBindingsUnchanged': True,
        'beforeGold': before, 'afterGold': after, 'beforeGoldSha256': io.canonical(before),
        'afterGoldSha256': io.canonical(after), 'changes': field_changes(before, after),
        'selectionBytesUnchanged': True, 'externalOutcomesRead': False, 'trainingPerformed': False}
    io.write_new(proof_path, proof)
    companion = load_module('independent_endpoint_serialization_gate', REPO/'scripts/audit-neural-selection-serialization.py')
    companion.audit(plan_path, selection_path, delegated_path, proof_path, companion_path)
    base = io.read(delegated_path)
    execution = {'source': io.identity(__file__), 'plan': io.identity(plan_path), 'proof': io.identity(proof_path),
        'serializationAudit': io.identity(companion_path), 'baseAudit': io.identity(delegated_path),
        'replacedSymbols': ['calibration_examples']}
    refs = [execution[k] for k in ('source', 'plan', 'proof', 'serializationAudit', 'baseAudit')]
    io.write_new(output, {**base, 'executionAuditor': io.identity(__file__), 'executionAdapter': execution,
        'executionScope': 'The auditor field identifies the frozen delegated body for existing consumers. Actual execution uses the explicitly registered lossless endpoint-serialization adapter; its independent companion and full proof are mandatory.',
        'references': [*base['references'], *refs]})
    return io.read(output)


def verify_execution(ordinary_path):
    ordinary = io.read(ordinary_path)
    io.require(ordinary['kind'] == 'independent-generalization-selection-audit-v1' and ordinary['passed'] is True
        and ordinary['auditor'] == io.identity(REPO/'scripts/audit-neural-generalization-selection.py'), 'Ordinary audit body differs')
    if 'executionAdapter' not in ordinary:
        io.require('executionAuditor' not in ordinary and 'executionScope' not in ordinary, 'Undeclared execution adapter')
        verify_unmodified_gold(ordinary)
        return {'ordinaryAudit': io.identity(ordinary_path), 'execution': 'unmodified-frozen-auditor', 'serializationAudit': None}
    execution = ordinary['executionAdapter']; verify_plan(verified(execution['plan']))
    for key in ('source', 'plan', 'proof', 'serializationAudit', 'baseAudit'): verified(execution[key])
    io.require(execution['source'] == io.identity(__file__) and ordinary['executionAuditor'] == execution['source']
        and execution['replacedSymbols'] == ['calibration_examples'], 'Actual execution adapter differs')
    base = io.read(execution['baseAudit']['path']); gate = io.read(execution['serializationAudit']['path'])
    io.require(gate['kind'] == 'independent-selection-serialization-audit-v1' and gate['passed'] is True
        and gate['selection'] == ordinary['selection'] and gate['plan'] == execution['plan']
        and gate['proof'] == execution['proof'] and gate['baseAudit'] == execution['baseAudit']
        and gate['auditor'] == io.identity(REPO/'scripts/audit-neural-selection-serialization.py'), 'Independent serialization gate differs')
    for reference in gate['evidence']: verified(reference)
    stripped = {k: v for k, v in ordinary.items() if k not in ('executionAuditor', 'executionAdapter', 'executionScope')}
    stripped['references'] = base['references']
    io.require(io.canonical(stripped) == io.canonical(base) and ordinary['references'] == [*base['references'],
        *[execution[k] for k in ('source', 'plan', 'proof', 'serializationAudit', 'baseAudit')]], 'Adapter changed delegated audit fields')
    return {'ordinaryAudit': io.identity(ordinary_path), 'execution': 'explicit-lossless-serialization-adapter',
            'serializationAudit': execution['serializationAudit'], 'plan': execution['plan']}


def verify_unmodified_gold(ordinary):
    """A delegated adapted base cannot be passed off as an unmodified audit."""
    from .neural_generalization_experiment import load_task
    selection = io.read(verified(ordinary['selection']))
    io.require(ordinary['task'] == selection['task'], 'Unmodified audit task differs')
    task, manifest, features, _ = load_task(verified(selection['task']))
    if task['variant'] == 'original-corpus':
        protocol_ref, = [r for r in selection['evidence'] if Path(r['path']).name == 'protocol.json'
            and io.read(verified(r)).get('kind') == 'historical-recall-floor-sweep-v1']
        protocol = io.read(verified(protocol_ref))
        examples = io.examples_from_rows(io.read(verified(protocol['records']))['records'])
    else:
        original = load_module('unchanged_selection_auditor_gold_check', REPO/'scripts/audit-neural-generalization-selection.py')
        examples = original.calibration_examples(task, io.read(manifest), io.read(features), original.Evidence())
    io.require(io.canonical(gold(examples)) == selection['selectionGoldSha256'],
        'Unmodified audit gold serialization differs; an execution adapter may have been stripped')
