#!/usr/bin/env python3
"""Add student-cache publication proof around the unchanged v2 finalizer."""
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io

V2_PATH = REPO/'scripts/finalize-neural-generalization-v2.py'
V2_SHA = 'a0e21277f640fc8d2e07afacb01ab1f8c11670bf44e7dbae8140eb37cb505602'
COLD_PATH = REPO/'scripts/audit-neural-student-inference-cache.py'
KIND = 'independent-generalization-finalization-audit-v3'
EXTRA = ('studentCacheFinalizer', 'studentCacheFinalizationPlan',
         'delegatedFinalizationAudit', 'studentCachePublicationGate')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def unchanged_v2():
    io.require(io.identity(V2_PATH)['sha256'] == V2_SHA, 'Frozen finalizer v2 changed')
    return module('student_cache_delegated_finalizer_v2', V2_PATH)


def register(args):
    previous = unchanged_v2()
    previous.verify_plan(args.previous_plan)
    queue = io.read(args.student_plan)
    io.require(queue['kind'] == 'registered-streaming-student-panel-queue-v3', 'V3 student plan required')
    names = ('scripts/finalize-neural-generalization-v3.py',
             'scripts/audit-neural-student-inference-cache.py',
             'analysis/tests/test_student_cache_publication.py',
             'docs/research/neural-student-inference-cache-publication-2026-09-23.md')
    io.write_new(args.output, {'kind': 'generalization-student-cache-finalization-plan-v3',
        'createdAtUTC': datetime.now(timezone.utc).isoformat(),
        'source': io.identity(__file__), 'previousFinalizer': io.identity(V2_PATH),
        'previousFinalizationPlan': io.identity(args.previous_plan),
        'studentQueuePlan': io.identity(args.student_plan),
        'code': {name: io.identity(REPO/name) for name in names},
        'numericalFunctionsReplaced': False, 'reportContentMustRemainByteEquivalent': True,
        'independentStudentColdGateRequiredBeforeAndAfterFinalAudit': True,
        'studentColdGateRequiredBeforeAndAfterReport': True,
        'oldNumericalAuditAndProductionSelectionGatesUnchanged': True})


def verify_plan(path):
    plan = io.read(path)
    io.require(plan['kind'] == 'generalization-student-cache-finalization-plan-v3'
        and plan['source'] == io.identity(__file__)
        and plan['previousFinalizer'] == io.identity(V2_PATH)
        and plan['numericalFunctionsReplaced'] is False
        and plan['reportContentMustRemainByteEquivalent'] is True
        and plan['independentStudentColdGateRequiredBeforeAndAfterFinalAudit'] is True
        and plan['studentColdGateRequiredBeforeAndAfterReport'] is True
        and plan['oldNumericalAuditAndProductionSelectionGatesUnchanged'] is True,
        'Student-cache finalization contract changed')
    for reference in [*plan['code'].values(), plan['previousFinalizationPlan'], plan['studentQueuePlan']]:
        io.require(io.identity(reference['path']) == reference, 'Finalization dependency changed')
    previous = unchanged_v2()
    previous.verify_plan(Path(plan['previousFinalizationPlan']['path']))
    return plan, previous


def preflight(plan, cold_path):
    cold = module('independent_student_cold_publication', COLD_PATH)
    return cold.verify_publication(cold_path, Path(plan['studentQueuePlan']['path']))


def restore_delegated(outer, delegated):
    restored = {key: value for key, value in outer.items() if key not in EXTRA}
    restored['kind'] = 'independent-generalization-finalization-audit-v1'
    io.require(io.canonical(restored) == io.canonical(delegated),
               'Student-cache wrapper changed the delegated final audit or report digest')


def audit_results(args):
    plan, previous = verify_plan(args.plan)
    before = preflight(plan, args.student_cold_audit)
    delegated_path = args.output.with_name(args.output.stem+'-finalization-v2.json')
    io.require(not args.output.exists() and not delegated_path.exists(), 'Existing/partial final audit requires inspection')
    delegated_args = argparse.Namespace(**vars(args))
    delegated_args.plan = Path(plan['previousFinalizationPlan']['path'])
    delegated_args.output = delegated_path
    previous.audit_results(delegated_args)
    io.require(preflight(plan, args.student_cold_audit) == before, 'Student evidence changed during final numerical audit')
    delegated = io.read(delegated_path)
    io.require(delegated['kind'] == previous.KIND and delegated['passed'] is True,
               'Complete unchanged v2 final audit required')
    result = {**delegated, 'kind': KIND, 'studentCacheFinalizer': io.identity(__file__),
        'studentCacheFinalizationPlan': io.identity(args.plan),
        'delegatedFinalizationAudit': io.identity(delegated_path), 'studentCachePublicationGate': before}
    restore_delegated(result, delegated)
    io.write_new(args.output, result)


def report(args):
    plan, previous = verify_plan(args.plan)
    audit = io.read(args.audit)
    io.require(audit['kind'] == KIND and audit['passed'] is True
        and audit['studentCacheFinalizer'] == io.identity(__file__)
        and audit['studentCacheFinalizationPlan'] == io.identity(args.plan)
        and audit['resultIndex'] == io.identity(args.index), 'Complete v3 final audit required')
    gate = audit['studentCachePublicationGate']
    before = preflight(plan, Path(gate['coldAudit']['path']))
    io.require(before == gate, 'Student cold publication gate changed')
    reference = audit['delegatedFinalizationAudit']
    io.require(io.identity(reference['path']) == reference, 'Delegated final audit changed')
    delegated = io.read(reference['path'])
    restore_delegated(audit, delegated)
    delegated_output = args.output.with_name(args.output.stem+'-finalization-v2.json')
    io.require(not args.output.exists() and not delegated_output.exists(), 'Existing/partial report requires inspection')
    previous.report(argparse.Namespace(plan=Path(plan['previousFinalizationPlan']['path']),
        index=args.index, audit=Path(reference['path']), output=delegated_output))
    gc.collect()
    value = io.read(delegated_output)
    io.require(value['metadata']['auditPassed'] is True
        and value['metadata']['audit'] == reference, 'Delegated report audit differs')
    content = {key: value[key] for key in ('inventory', 'scopes', 'series', 'tasks')}
    io.require(io.canonical(content) == audit['reportContentSha256'], 'Delegated report content changed')
    value['metadata']['audit'] = io.identity(args.audit)
    value['metadata']['studentInferenceCachePublication'] = before
    value['metadata']['studentInferenceCachePolicy'] = (
        'Disclosed process-local immutable-file hash reuse only; unchanged neural replay and metrics; '
        'all27 student inference audits and execution companions independently cold-verified before publication.')
    io.require(preflight(plan, Path(gate['coldAudit']['path'])) == before,
               'Student evidence changed during report construction')
    io.write_new(args.output, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    reg = sub.add_parser('register')
    for name in ('previous-plan', 'student-plan', 'output'):
        reg.add_argument('--'+name, type=Path, required=True)
    audit = sub.add_parser('audit-results')
    for name in ('plan', 'index', 'report', 'global-selection-gate', 'student-cold-audit', 'output'):
        audit.add_argument('--'+name, type=Path, required=True)
    audit.add_argument('--node', default='/mnt/c/Program Files/nodejs/node.exe')
    rep = sub.add_parser('report')
    for name in ('plan', 'index', 'audit', 'output'):
        rep.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    io.require(Path('/mnt/freenas').is_mount() and args.output.resolve().is_relative_to('/mnt/freenas'), 'Direct NAS output required')
    globals()[args.action.replace('-', '_')](args)
    print(json.dumps(io.identity(args.output)), flush=True)


if __name__ == '__main__':
    main()
