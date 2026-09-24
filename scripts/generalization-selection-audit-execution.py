#!/usr/bin/env python3
"""Composite publication workflow for explicitly disclosed selection-audit execution."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_inputs import verified
from analysis.neural_selection_serialization import load_module
from analysis import neural_selection_serialization as serialization
from analysis.neural_selection_duration_view import POLICY, verify_plan, audit_selection, verify_execution

old = load_module('frozen_containment_workflow', REPO/'scripts/generalization-selection-containment.py')
GLOBAL_KIND = 'all-registered-selections-frozen-before-external-evaluation-v1'
JOINT_KIND = 'independent-generalization-result-execution-audit-v1'


def register(args):
    prior = serialization.verify_plan(args.serialization_plan)
    protocol = io.read(args.historical_protocol)
    provenance = io.read(args.duration_provenance)
    io.require(provenance['passed'] is True and provenance['protocol'] == io.identity(args.historical_protocol)
        and provenance['source'] == protocol['source'] and provenance['records'] == protocol['records'],
        'Independent duration provenance association differs')
    names = ('analysis/neural_selection_duration_view.py', 'scripts/audit-neural-selection-duration-view.py',
        'scripts/generalization-selection-audit-execution.py', 'scripts/audit-neural-historical-duration-provenance.py',
        'analysis/tests/test_selection_duration_view.py', 'analysis/tests/test_selection_duration_view_audit.py')
    io.write_new(args.output, {'kind': 'generalization-selection-duration-comparison-amendment-v1',
        'createdAtUTC': datetime.now(timezone.utc).isoformat(), 'policy': POLICY,
        'protocol': io.identity(args.protocol), 'historicalProtocol': io.identity(args.historical_protocol),
        'durationProvenance': io.identity(args.duration_provenance),
        'serializationPlan': io.identity(args.serialization_plan), 'correctionPlan': prior['correctionPlan'],
        'code': {**prior['code'], **{name: io.identity(REPO/name) for name in names}},
        'allExternalEvaluationRequiresGlobal162SelectionGate': True,
        'mandatoryPublicationWorkflow': 'scripts/generalization-selection-audit-execution.py'})


def freeze_selections(args):
    plan = verify_plan(args.plan); correction = io.read(verified(plan['correctionPlan']))
    registrations = [io.identity(p) for p in args.registration]
    io.require(registrations == correction['registrations'], 'Global gate registration inventory/order differs')
    checks = []
    for registration_ref in registrations:
        registration = io.read(registration_ref['path']); directory = Path(registration_ref['path']).parent
        for task_id in registration['contract']['taskIds']:
            task_path = directory/'tasks'/(task_id.replace('/', '__')+'.json'); task = io.read(task_path)
            folder = directory/'fits'/task['model']/f'seed-{task["seed"]}' if task['variant'] == 'original-corpus' else (
                directory/'fits'/task['variant']/task['model']/f'split-{task["splitSeed"]}')
            selection, ordinary, companion = folder/'selection.json', folder/'selection-audit.json', folder/'selection-correction-audit.json'
            old.correction_gate(selection, ordinary, companion)
            execution = verify_execution(ordinary)
            if execution.get('plan'):
                expected_plan = io.identity(args.plan) if task['variant'] == 'original-corpus' else plan['serializationPlan']
                io.require(execution['plan'] == expected_plan, 'Declared execution amendment differs')
            io.require(io.read(selection)['task'] == io.identity(task_path), 'Global selection belongs to another task')
            checks.append({'task': io.identity(task_path), 'selection': io.identity(selection),
                'ordinaryAudit': io.identity(ordinary), 'correctionAudit': io.identity(companion), 'execution': execution})
    io.require(len(checks) == 162 and len({r['task']['sha256'] for r in checks}) == 162, 'All162 task selections required')
    io.write_new(args.output, {'kind': GLOBAL_KIND, 'passed': True, 'plan': io.identity(args.plan),
        'registrations': registrations, 'taskCount': 162, 'selections': checks,
        'generator': io.identity(__file__), 'externalMetricsOpened': False,
        'createdAtUTC': datetime.now(timezone.utc).isoformat()})


def global_gate(path):
    gate = io.read(path)
    io.require(gate['kind'] == GLOBAL_KIND and gate['passed'] is True and gate['taskCount'] == 162
        and len(gate['selections']) == 162 and gate['generator'] == io.identity(__file__)
        and gate['externalMetricsOpened'] is False, 'External evaluation requires the frozen global162 selection gate')
    plan = verify_plan(verified(gate['plan'])); old.verify_closure(gate)
    correction = io.read(verified(plan['correctionPlan']))
    io.require(gate['registrations'] == correction['registrations'], 'Global freeze registration scope changed')
    expected = []
    for reference in gate['registrations']:
        registration = io.read(verified(reference)); directory = Path(reference['path']).parent
        expected.extend(io.identity(directory/'tasks'/(key.replace('/', '__')+'.json'))
                        for key in registration['contract']['taskIds'])
    io.require([row['task'] for row in gate['selections']] == expected
        and len({row['task']['sha256'] for row in gate['selections']}) == 162,
        'Global freeze omits, duplicates or reorders registered task selections')
    return gate


def evaluate(args):
    frozen = global_gate(args.global_selection_gate)
    selected, = [r for r in frozen['selections'] if r['task'] == io.identity(args.task)]
    io.require(selected['selection'] == io.identity(args.selection) and selected['ordinaryAudit'] == io.identity(args.selection_audit)
        and selected['correctionAudit'] == io.identity(args.correction_audit), 'Task selection differs from global freeze')
    io.require(verify_execution(args.selection_audit) == selected['execution'], 'Audit execution differs from global freeze')
    start = args.output.with_name(args.output.stem+'-evaluation-start.json')
    io.require(not args.output.exists() and not start.exists(), 'Evaluation/start receipt already exists')
    started = {'kind': 'external-evaluation-after-global-selection-freeze-v1',
        'globalSelectionGate': io.identity(args.global_selection_gate), 'task': io.identity(args.task),
        'selection': selected, 'panel': io.identity(args.panel), 'precision': args.precision,
        'workflow': io.identity(__file__), 'createdAtUTC': datetime.now(timezone.utc).isoformat()}
    io.write_new(start, started)
    old.evaluate(args)
    io.require(global_gate(args.global_selection_gate) == frozen, 'Global selection freeze changed during evaluation')
    gate = args.output.with_name(args.output.stem+'-publication-gate.json')
    io.write_new(gate, {'kind': 'generalization-evaluation-publication-gate-v1', 'passed': True,
        'evaluation': io.identity(args.output), 'evaluationStart': io.identity(start),
        'globalSelectionGate': io.identity(args.global_selection_gate), 'auditExecution': selected['execution'],
        'task': selected['task'], 'panel': started['panel'], 'precision': args.precision, 'workflow': io.identity(__file__)})


def publication_preflight(index_path, global_path):
    frozen = global_gate(global_path); index = io.read(index_path); checks = []
    known = {r['task']['sha256']: r for r in frozen['selections']}
    for entry in index['evaluations']:
        result_path = verified(entry['result']); result = io.read(result_path)
        if result['kind'] == 'fixed-production-generalization-evaluation-v1': continue
        path = result_path.with_name(result_path.stem+'-publication-gate.json'); gate = io.read(path)
        selected = known[result['task']['sha256']]
        io.require(gate['kind'] == 'generalization-evaluation-publication-gate-v1' and gate['passed'] is True
            and gate['evaluation'] == entry['result'] and gate['globalSelectionGate'] == io.identity(global_path)
            and gate['workflow'] == io.identity(__file__) and gate['task'] == result['task']
            and gate['panel'] == result['panel'] and gate['precision'] == result['precision']
            and gate['auditExecution'] == selected['execution'], 'Evaluation publication preflight differs')
        io.require(verify_execution(verified(result['selectionAudit'])) == selected['execution'], 'Execution gate changed')
        start = io.read(verified(gate['evaluationStart']))
        io.require(start['globalSelectionGate'] == gate['globalSelectionGate'] and start['selection'] == selected
            and start['workflow'] == io.identity(__file__) and start['panel'] == gate['panel']
            and start['precision'] == gate['precision'], 'Evaluation started without the same global freeze')
        checks.append(io.identity(path))
    io.require(len(checks) == 270, 'Full270 publication gates required')
    return checks


def audit_results(args):
    checks = publication_preflight(args.index, args.global_selection_gate)
    base = args.output.with_name(args.output.stem+'-containment-base.json')
    io.require(not args.output.exists() and not base.exists(), 'Joint audit already exists')
    delegated_args = argparse.Namespace(**{**vars(args), 'output': base})
    old.audit_results(delegated_args)
    io.require(publication_preflight(args.index, args.global_selection_gate) == checks, 'Publication gates changed during full audit')
    result = io.read(base)
    io.write_new(args.output, {**result, 'kind': JOINT_KIND, 'executionAuditor': io.identity(__file__),
        'delegatedContainmentAudit': io.identity(base), 'globalSelectionGate': io.identity(args.global_selection_gate),
        'evaluationPublicationGates': checks})


def report(args):
    from analysis.neural_generalization_report import build_report
    audit = io.read(args.audit)
    io.require(audit['kind'] == JOINT_KIND and audit['passed'] is True and audit['executionAuditor'] == io.identity(__file__)
        and audit['resultIndex'] == io.identity(args.index), 'Final publication requires all explicit-execution/global-selection gates')
    old.verify_closure(audit)
    io.require(publication_preflight(args.index, verified(audit['globalSelectionGate'])) == audit['evaluationPublicationGates'],
               'Final publication gate inventory differs')
    base = io.read(verified(audit['delegatedContainmentAudit']))
    io.require(base['passed'] and base['kind'] == old.RESULT_GATE_KIND and base['auditor'] == io.identity(REPO/'scripts/generalization-selection-containment.py')
        and base['resultIndex'] == audit['resultIndex'] and base['reportContentSha256'] == audit['reportContentSha256'], 'Delegated full audit differs')
    return build_report(args.index, args.output, args.audit)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    reg = sub.add_parser('register')
    for key in ('serialization-plan', 'historical-protocol', 'duration-provenance', 'protocol', 'output'): reg.add_argument('--'+key, type=Path, required=True)
    aud = sub.add_parser('audit-selection')
    for key in ('plan', 'selection', 'output'): aud.add_argument('--'+key, type=Path, required=True)
    glob = sub.add_parser('freeze-selections')
    for key in ('plan', 'output'): glob.add_argument('--'+key, type=Path, required=True)
    glob.add_argument('--registration', type=Path, action='append', required=True)
    ev = sub.add_parser('evaluate')
    for key in ('task', 'fit', 'selection', 'selection-audit', 'correction-audit', 'panel', 'inventory', 'original-manifest', 'global-selection-gate', 'output'):
        ev.add_argument('--'+key, type=Path, required=True)
    ev.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), default='fp32')
    full = sub.add_parser('audit-results')
    for key in ('index', 'report', 'global-selection-gate', 'output'): full.add_argument('--'+key, type=Path, required=True)
    full.add_argument('--node', default='/mnt/c/Program Files/nodejs/node.exe')
    rep = sub.add_parser('report')
    for key in ('index', 'audit', 'output'): rep.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(); io.require(args.output.resolve().is_relative_to('/mnt/freenas'), 'Outputs must use NAS')
    if args.action == 'audit-selection':
        plan = verify_plan(args.plan)
        selected = io.read(args.selection); task = io.read(verified(selected['task']))
        if task['variant'] == 'original-corpus': audit_selection(args.plan, args.selection, args.output)
        else: serialization.audit_selection(verified(plan['serializationPlan']), args.selection, args.output)
    else: globals()[args.action.replace('-', '_')](args)
    print(json.dumps({'action': args.action, 'output': io.identity(args.output)}), flush=True)


if __name__ == '__main__': main()
