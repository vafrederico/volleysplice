#!/usr/bin/env python3
"""Additive selection/evaluation/publication entry points requiring containment gates.

No frozen metric, selector, evaluator, final auditor or report renderer is edited.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_inputs import verified
from analysis.neural_selection_containment_correction import POLICY, MAX_RECALL_ULPS, POLICY_CONTRACT, verify_plan, select_task_with_correction

AUDITOR = REPO/'scripts/audit-neural-generalization-selection-correction.py'
GATE_KIND = 'independent-generalization-selection-correction-audit-v1'
RESULT_GATE_KIND = 'independent-generalization-result-containment-audit-v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def verify_closure(value, cache=None):
    cache = {} if cache is None else cache
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            ref = {'path': value['path'], 'sha256': value['sha256']}
            path = Path(ref['path']); stat = path.stat()
            token = (ref['sha256'], stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
            if str(path) in cache:
                io.require(cache[str(path)] == token, 'Bound gate evidence changed')
            else:
                verified(ref); cache[str(path)] = token
        for child in value.values(): verify_closure(child, cache)
    elif isinstance(value, list):
        for child in value: verify_closure(child, cache)
    return cache


def correction_gate(selection_path, old_audit_path, gate_path, cache=None):
    selection = io.read(selection_path); gate = io.read(gate_path); old = io.read(old_audit_path)
    io.require('rawCandidates' in selection and 'candidateMetricCorrection' in selection,
               'Every new deployment task requires the containment amendment, including zero corrections')
    plan_ref = selection['candidateMetricCorrection']['plan']
    verify_plan(verified(plan_ref))
    io.require(gate['kind'] == GATE_KIND and gate['passed'] is True
        and gate['selection'] == io.identity(selection_path) and gate['selectionAudit'] == io.identity(old_audit_path)
        and gate['task'] == selection['task'] and gate['plan'] == plan_ref
        and gate['checkedCandidateCount'] == 192 and gate['normalizationPolicy'] == POLICY
        and gate['maxRecallUlps'] == MAX_RECALL_ULPS
        and gate['auditor'] == io.identity(AUDITOR), 'Companion correction gate is absent or stale')
    io.require(old['kind'] == 'independent-generalization-selection-audit-v1'
        and old['passed'] is True and old['selection'] == gate['selection'] and old['task'] == selection['task']
        and old['auditor'] == io.identity(REPO/'scripts/audit-neural-generalization-selection.py'),
        'Ordinary frozen selection gate is absent or stale')
    verify_closure(gate, cache)
    return gate


def register(args):
    sources = [
        'analysis/neural_selection_containment_correction.py',
        'scripts/generalization-selection-containment.py',
        'scripts/audit-neural-generalization-selection-correction.py',
        'analysis/tests/test_selection_containment_correction.py',
        'analysis/tests/test_generalization_selection_correction_audit.py',
        'analysis/neural_generalization_results.py', 'analysis/neural_recall_sweep.py',
        'analysis/neural_recall_operating_point.py', 'analysis/crop_evaluation.py',
        'scripts/audit-neural-generalization-selection.py',
        'scripts/audit-neural-generalization-results.py',
        'scripts/audit-neural-short-boost-intervals.py',
        'analysis/neural_generalization_report.py',
    ]
    pending, discovered = [REPO/name for name in sources], set()
    while pending:
        path = pending.pop()
        if path in discovered: continue
        discovered.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = []
            if isinstance(node, ast.ImportFrom):
                if node.level and path.parent.name == 'analysis':
                    names = [node.module.split('.')[0]] if node.module else [v.name for v in node.names]
                elif node.module == 'analysis': names = [v.name for v in node.names]
                elif node.module and node.module.startswith('analysis.'): names = [node.module.split('.')[1]]
            elif isinstance(node, ast.Import):
                names = [v.name.split('.')[1] for v in node.names if v.name.startswith('analysis.')]
            pending.extend(p for name in names if (p := REPO/'analysis'/(name+'.py')).exists())
    refs = [io.identity(path) for path in args.registration]
    io.require(len(refs) == 3 and len({r['sha256'] for r in refs}) == 3, 'All three distinct registrations required')
    io.write_new(args.output, {'kind': 'generalization-selection-containment-amendment-v1',
        'createdAtUTC': datetime.now(timezone.utc).isoformat(), 'policy': POLICY_CONTRACT,
        'protocol': io.identity(args.protocol), 'diagnostic': io.identity(args.diagnostic),
        'registrations': refs, 'code': {str(path.relative_to(REPO)): io.identity(path) for path in sorted(discovered)},
        'frozenScientificSourcesUnchanged': True, 'fittingRepeated': False,
        'historicalNestedSweepsChanged': False,
        'requiredPublicationEntryPoint': 'scripts/generalization-selection-containment.py report',
        'requiredEvaluationEntryPoint': 'scripts/generalization-selection-containment.py evaluate',
        'requiredFinalAuditEntryPoint': 'scripts/generalization-selection-containment.py audit-results'})


def bind_index(args):
    verify_plan(args.plan)
    original = io.read(args.index)
    io.require(original['kind'] == 'registered-generalization-result-index-v1'
        and 'selectionContainmentAmendment' not in original, 'Index is already amended or has another schema')
    io.write_new(args.output, {**original, 'baseResultIndex': io.identity(args.index),
        'selectionContainmentAmendment': io.identity(args.plan),
        'metadata': {**original['metadata'], 'selectionContainmentAmendment': io.identity(args.plan),
            'selectionMetricRoundoffPolicy': 'Only impossible recall overshoots above1 within4ULPs are normalized to1 after exact empty omitted-core proof. All raw candidates, valid metrics, F1 order and strict floors remain unchanged; every task has both independent selection gates.'}})


def evaluate(args):
    from analysis.neural_generalization_results import evaluate_task
    correction_gate(args.selection, args.selection_audit, args.correction_audit)
    io.require(not args.output.exists(), 'Evaluation already published')
    base = args.output.with_name(args.output.stem+'-frozen-evaluator-base.json')
    io.require(not base.exists(), 'Base evaluation exists; inspect interrupted publication before resuming')
    result = evaluate_task(args.task, args.fit, args.selection, args.panel, args.inventory,
        args.original_manifest, base, args.precision, args.selection_audit)
    # "base" is the unchanged evaluator workflow, not a different selection.
    # It used the same corrected, twice-audited selection. Keep it immutable.
    result = {**result, 'selectionCorrectionAudit': io.identity(args.correction_audit),
        'baseEvaluation': io.identity(base), 'containmentWorkflow': io.identity(__file__)}
    correction_gate(args.selection, args.selection_audit, args.correction_audit)
    io.write_new(args.output, result)


def preflight_index(index_path):
    index = io.read(index_path); checks, cache = {}, {}
    verify_closure(index, cache)
    plan_ref = index['selectionContainmentAmendment']; verify_plan(verified(plan_ref))
    io.require(index['metadata']['selectionContainmentAmendment'] == plan_ref, 'Index amendment metadata differs')
    base_index = io.read(verified(index['baseResultIndex']))
    restored = {k: v for k, v in index.items() if k not in ('selectionContainmentAmendment', 'baseResultIndex')}
    restored['metadata'] = {k: v for k, v in index['metadata'].items()
        if k not in ('selectionContainmentAmendment', 'selectionMetricRoundoffPolicy')}
    io.require(restored == base_index, 'Additive index wrapper changed original matrix/data')
    for entry in index['evaluations']:
        result = io.read(verified(entry['result']))
        if result['kind'] == 'fixed-production-generalization-evaluation-v1': continue
        selection_ref, ordinary_ref, companion_ref = result['selection'], result['selectionAudit'], result['selectionCorrectionAudit']
        io.require(result['containmentWorkflow'] == io.identity(__file__), 'Evaluation bypassed correction workflow')
        base = io.read(verified(result['baseEvaluation']))
        io.require({k: v for k, v in result.items() if k not in (
            'selectionCorrectionAudit', 'baseEvaluation', 'containmentWorkflow')} == base,
            'Delegated numerical evaluation changed in wrapper')
        if selection_ref['sha256'] not in checks:
            gate = correction_gate(verified(selection_ref), verified(ordinary_ref), verified(companion_ref), cache)
            io.require(gate['plan'] == plan_ref, 'Evaluation used a different correction amendment')
            checks[selection_ref['sha256']] = {'selection': selection_ref, 'selectionAudit': ordinary_ref,
                'correctionAudit': companion_ref, 'plan': gate['plan'], 'task': gate['task']}
        else:
            io.require(checks[selection_ref['sha256']]['correctionAudit'] == companion_ref
                and checks[selection_ref['sha256']]['selectionAudit'] == ordinary_ref, 'Precision arms use different gates')
    io.require(len(checks) == 162, 'Full task matrix lacks companion selection gates')
    return list(checks.values())


def audit_results(args):
    io.require(not args.output.exists(), 'Joint result audit already exists')
    checks = preflight_index(args.index)
    frozen = module('frozen_full_generalization_auditor', REPO/'scripts/audit-neural-generalization-results.py')
    base = args.output.with_name(args.output.stem+'-frozen-base.json')
    io.require(not base.exists(), 'Frozen full audit exists; inspect interrupted publication before resuming')
    frozen.audit(args.index, args.report, base, args.node)
    io.require(preflight_index(args.index) == checks, 'Companion gates changed during final audit')
    result = io.read(base)
    io.write_new(args.output, {**result, 'kind': RESULT_GATE_KIND, 'auditor': io.identity(__file__),
        'frozenFullAudit': io.identity(base), 'selectionCorrectionGates': checks,
        'scope': result['scope']+' Also requires the independent full-containment correction gate for all162 new deployment selections, including zero-correction cases.'})


def report(args):
    from analysis.neural_generalization_report import build_report
    audit = io.read(args.audit)
    io.require(audit['kind'] == RESULT_GATE_KIND and audit['passed'] is True
        and audit['auditor'] == io.identity(__file__) and audit['resultIndex'] == io.identity(args.index),
        'Publication requires the joint full-result and correction audit')
    verify_closure(audit)
    io.require(preflight_index(args.index) == audit['selectionCorrectionGates'], 'Publication correction gates differ')
    base = io.read(verified(audit['frozenFullAudit']))
    io.require(base['passed'] and base['resultIndex'] == audit['resultIndex']
        and base['reportContentSha256'] == audit['reportContentSha256'], 'Frozen full numerical audit differs')
    return build_report(args.index, args.output, args.audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    reg = sub.add_parser('register')
    for key in ('protocol', 'diagnostic', 'output'): reg.add_argument('--'+key, type=Path, required=True)
    reg.add_argument('--registration', type=Path, action='append', required=True)
    idx = sub.add_parser('bind-index')
    for key in ('index', 'plan', 'output'): idx.add_argument('--'+key, type=Path, required=True)
    sel = sub.add_parser('select')
    for key in ('plan', 'task', 'fit', 'output'): sel.add_argument('--'+key, type=Path, required=True)
    sel.add_argument('--historical', type=Path)
    ev = sub.add_parser('evaluate')
    for key in ('task', 'fit', 'selection', 'selection-audit', 'correction-audit', 'panel', 'inventory', 'original-manifest', 'output'):
        ev.add_argument('--'+key, type=Path, required=True)
    ev.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), default='fp32')
    aud = sub.add_parser('audit-results')
    for key in ('index', 'report', 'output'): aud.add_argument('--'+key, type=Path, required=True)
    aud.add_argument('--node', default='/mnt/c/Program Files/nodejs/node.exe')
    rep = sub.add_parser('report')
    for key in ('index', 'audit', 'output'): rep.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    io.require(args.output.resolve().is_relative_to('/mnt/freenas'), 'All outputs must use NAS')
    if args.action == 'select':
        select_task_with_correction(args.plan, args.task, args.fit, args.output, args.historical)
    else: globals()[args.action.replace('-', '_')](args)
    print(json.dumps({'action': args.action, 'output': io.identity(args.output)}), flush=True)


if __name__ == '__main__': main()
