"""Disclose the qualified legacy-manifest view around frozen finalization."""
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
from analysis import neural_original_manifest_view as view

V3_PATH = REPO/'scripts/finalize-neural-generalization-v3.py'
V3_SHA = '97d2667ec3d85d9830fa4f3a17ba606beff9619a6a0c51528bdcb63adeefc961'
DISPATCH_PATH = REPO/'scripts/run-neural-manifest-view-evaluation.py'
NODE = '/mnt/c/Program Files/nodejs/node.exe'
KIND = 'independent-generalization-finalization-audit-v4'
EXTRA = ('manifestViewFinalizer', 'manifestViewFinalizationPlan', 'delegatedManifestViewAudit',
         'manifestViewPublicationGates', 'manifestViewFinalAuditExecution')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def checked(reference):
    io.require(io.identity(reference['path']) == reference, 'Finalization reference changed')
    return Path(reference['path'])


def unchanged_v3():
    io.require(io.identity(V3_PATH)['sha256'] == V3_SHA, 'Frozen v3 finalizer changed')
    return module('manifest_view_delegated_finalizer_v3', V3_PATH)


def dispatcher(plan):
    return module('manifest_view_publication_dispatcher', checked(plan['dispatcher']))


def register(args):
    previous = unchanged_v3(); previous.verify_plan(args.previous_plan)
    dispatch = module('manifest_view_register_dispatcher', DISPATCH_PATH)
    execution, _, _, _ = dispatch.verify_plan(args.dispatcher_plan)
    io.write_new(args.output, {'kind': 'manifest-view-finalization-plan-v4',
        'createdAtUTC': datetime.now(timezone.utc).isoformat(), 'source': io.identity(__file__),
        'previousFinalizer': io.identity(V3_PATH), 'previousFinalizationPlan': io.identity(args.previous_plan),
        'dispatcher': io.identity(DISPATCH_PATH), 'dispatcherPlan': io.identity(args.dispatcher_plan),
        'viewPlan': execution['viewPlan'], 'viewQualification': execution['viewQualification'],
        'nodeExecutable': NODE,
        'code': {name: io.identity(REPO/name) for name in (
            'scripts/finalize-neural-generalization-v4.py', 'analysis/tests/test_manifest_view_evaluation.py',
            'docs/research/neural-original-manifest-evaluation-recovery-2026-09-23.md')},
        'policy': {'unchangedV2ProductionAndV3FinalAuditReport': True, 'all270NeuralCompanionsRequired': True,
            'fullScopeActualExit0Required': True, 'productionAndFinalAuditCompanionsRequired': True,
            'unchangedIndexAndContainmentBinding': True, 'onlyReportMetadataAdded': True,
            'reportContentMustRemainByteEquivalent': True}})


def verify_plan(path):
    plan = io.read(path)
    io.require(plan['kind'] == 'manifest-view-finalization-plan-v4' and plan['source'] == io.identity(__file__)
        and plan['previousFinalizer'] == io.identity(V3_PATH) and plan['nodeExecutable'] == NODE
        and plan['policy'] == {'unchangedV2ProductionAndV3FinalAuditReport': True,
            'all270NeuralCompanionsRequired': True, 'fullScopeActualExit0Required': True,
            'productionAndFinalAuditCompanionsRequired': True, 'unchangedIndexAndContainmentBinding': True,
            'onlyReportMetadataAdded': True, 'reportContentMustRemainByteEquivalent': True}, 'Finalization v4 contract differs')
    for reference in [*plan['code'].values(), plan['previousFinalizationPlan'], plan['dispatcher'],
                      plan['dispatcherPlan'], plan['viewPlan'], plan['viewQualification']]: checked(reference)
    previous = unchanged_v3(); prior, production = previous.verify_plan(Path(plan['previousFinalizationPlan']['path']))
    execution, _, _, _ = dispatcher(plan).verify_plan(Path(plan['dispatcherPlan']['path']))
    io.require(execution['viewPlan'] == plan['viewPlan'] and execution['viewQualification'] == plan['viewQualification']
        and execution['finalizationV3'] == plan['previousFinalizationPlan'], 'Dispatcher/finalization input association differs')
    return plan, previous, prior, production


def invocation(source, action, values):
    argv = [sys.executable, str(source), action]
    for key, value in values.items(): argv.extend(['--'+key.replace('_', '-'), str(value)])
    return argv


def companion_path(output):
    return output.with_name(output.stem+'-manifest-view-execution.json')


def production_outputs(output):
    return {'evaluation': io.identity(output),
        'evaluationStart': io.identity(output.with_name(output.stem+'-evaluation-start.json')),
        'publicationGate': io.identity(output.with_name(output.stem+'-publication-gate.json'))}


def production_values(prior, args):
    return {'plan': Path(prior['previousFinalizationPlan']['path']),
        **{name: getattr(args, name) for name in ('global_selection_gate', 'production', 'parity_audit',
                                                 'inventory', 'original_manifest', 'output')}}


def evaluate_production(args):
    plan, previous, prior, production = verify_plan(args.plan)
    compatibility = view.verify_plan(Path(plan['viewPlan']['path']))
    io.require(io.identity(args.original_manifest) == compatibility['legacyManifest'], 'Production legacy input differs')
    values = production_values(prior, args)
    argv = invocation(previous.V2_PATH, 'evaluate-production', values)
    companion = companion_path(args.output)
    io.require(not companion.exists(), 'Preserve previous production companion')
    with view.installed(Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
            stage='production-evaluation', argv=argv,
            extra_bindings={'production.evaluate_production': (production, 'evaluate_production')}) as proof:
        production.evaluate_production(argparse.Namespace(**values))
    io.write_new(companion, {**proof, 'outputs': production_outputs(args.output)})
    view.verify_execution(companion, Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
                          stage='production-evaluation', argv=argv, outputs=production_outputs(args.output))


def production_companion(plan, prior, previous, output):
    refs = production_outputs(output); start = io.read(checked(refs['evaluationStart']))
    values = {'plan': Path(prior['previousFinalizationPlan']['path']),
        'global_selection_gate': checked(start['globalSelectionGate']),
        'production': checked(start['productionProtocol']).parent,
        'parity_audit': checked(start['parityAudit']), 'inventory': checked(start['inventory']),
        'original_manifest': checked(start['originalManifest']), 'output': output}
    io.require(start['plan'] == prior['previousFinalizationPlan'], 'Production used a different frozen v2 plan')
    view.verify_execution(companion_path(output), Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
        stage='production-evaluation', argv=invocation(previous.V2_PATH, 'evaluate-production', values), outputs=refs)
    return io.identity(companion_path(output))


def require_index_membership(index, expected):
    io.require(len(index['evaluations']) == 271, 'Complete270 neural plus one production index required')
    actual = [entry['result'] for entry in index['evaluations'][:-1]]
    io.require(len({row['path'] for row in actual}) == 270 and actual == expected,
               'Index neural cells/order differ from the original270 plan')


def preflight(plan, previous, prior, index_path):
    dispatch = dispatcher(plan)
    plan_path = Path(plan['dispatcherPlan']['path'])
    execution, helper, worker, original = dispatch.verify_plan(plan_path)
    closure = dispatch.verify_scope_exit(plan_path, execution, 'all')
    started = io.read(checked(closure['started'])); authorization = io.read(checked(started['grant']))
    student = dispatch.student_gate(execution, authorization)
    io.require(started['studentGate'] == student, 'Final student and prior243 execution proofs changed')
    helper.verify_original(worker, original); global_gate = worker.global_ready(original)
    io.require(global_gate is not None and global_gate['reference'] == execution['globalSelectionGate'], 'Same global162 freeze required')
    companions, expected = [], []
    for job in original['jobs']:
        panel = worker.panel_ready(original, job['precision'])
        io.require(panel is not None and dispatch.check_cell(worker, original, global_gate, panel, job, execution),
                   'Missing original/view complete-cell proof: '+job['cellId'])
        expected.append(dispatch.outputs(job)['evaluation'])
        companions.append({'cellId': job['cellId'], 'execution': io.identity(dispatch.companion_path(job))})
    index = io.read(index_path); require_index_membership(index, expected)
    production_path = checked(index['evaluations'][-1]['result'])
    return {'dispatcherPlan': plan['dispatcherPlan'], 'dispatcherCompletion': closure,
        'studentAndPrior243Publication': student, 'neuralExecutions': companions,
        'productionExecution': production_companion(plan, prior, previous, production_path),
        'viewPlan': plan['viewPlan'], 'viewQualification': plan['viewQualification']}


def restore_delegated(outer, delegated):
    restored = {key: value for key, value in outer.items() if key not in EXTRA}
    restored['kind'] = 'independent-generalization-finalization-audit-v3'
    io.require(io.canonical(restored) == io.canonical(delegated), 'Manifest-view wrapper changed the delegated audit payload')


def audit_values(plan, args, output):
    io.require(args.node == NODE, 'Registered audit Node executable required')
    return {'plan': Path(plan['previousFinalizationPlan']['path']), 'index': args.index, 'report': args.report,
        'global_selection_gate': args.global_selection_gate, 'student_cold_audit': args.student_cold_audit,
        'output': output, 'node': args.node}


def audit_results(args):
    plan, previous, prior, _ = verify_plan(args.plan)
    before = preflight(plan, previous, prior, args.index)
    delegated_path = args.output.with_name(args.output.stem+'-finalization-v3.json')
    companion = companion_path(args.output)
    io.require(not args.output.exists() and not delegated_path.exists() and not companion.exists(), 'Preserve existing/partial final audit')
    values = audit_values(plan, args, delegated_path); argv = invocation(V3_PATH, 'audit-results', values)
    with view.installed(Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
            stage='final-audit', argv=argv, extra_bindings={'finalizer.audit_results': (previous, 'audit_results')}) as proof:
        previous.audit_results(argparse.Namespace(**values))
    refs = {'delegatedFinalizationAudit': io.identity(delegated_path)}
    io.write_new(companion, {**proof, 'outputs': refs})
    view.verify_execution(companion, Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
                          stage='final-audit', argv=argv, outputs=refs)
    io.require(preflight(plan, previous, prior, args.index) == before, 'Execution companions changed during final audit')
    delegated = io.read(delegated_path)
    io.require(delegated['kind'] == previous.KIND and delegated['passed'] is True, 'Complete unchanged v3 audit required')
    result = {**delegated, 'kind': KIND, 'manifestViewFinalizer': io.identity(__file__),
        'manifestViewFinalizationPlan': io.identity(args.plan), 'delegatedManifestViewAudit': io.identity(delegated_path),
        'manifestViewPublicationGates': before, 'manifestViewFinalAuditExecution': io.identity(companion)}
    restore_delegated(result, delegated); io.write_new(args.output, result)


def verify_final_audit(plan_path, plan, audit_path, index_path):
    audit = io.read(audit_path)
    io.require(audit['kind'] == KIND and audit['passed'] is True
        and audit['manifestViewFinalizer'] == io.identity(__file__)
        and audit['manifestViewFinalizationPlan'] == io.identity(plan_path)
        and audit['resultIndex'] == io.identity(index_path), 'Complete v4 final audit required')
    delegated_path = checked(audit['delegatedManifestViewAudit'])
    restore_delegated(audit, io.read(delegated_path))
    args = argparse.Namespace(index=index_path, report=checked(audit['candidateReport']),
        global_selection_gate=checked(audit['globalSelectionGate']),
        student_cold_audit=checked(audit['studentCachePublicationGate']['coldAudit']), node=NODE)
    argv = invocation(V3_PATH, 'audit-results', audit_values(plan, args, delegated_path))
    view.verify_execution(checked(audit['manifestViewFinalAuditExecution']), Path(plan['viewPlan']['path']),
        Path(plan['viewQualification']['path']), stage='final-audit', argv=argv,
        outputs={'delegatedFinalizationAudit': audit['delegatedManifestViewAudit']})
    return audit


def report(args):
    plan, previous, prior, _ = verify_plan(args.plan)
    audit = verify_final_audit(args.plan, plan, args.audit, args.index)
    before = preflight(plan, previous, prior, args.index)
    io.require(audit['manifestViewPublicationGates'] == before, 'Final publication execution evidence changed')
    delegated_output = args.output.with_name(args.output.stem+'-finalization-v3.json')
    companion = companion_path(args.output)
    io.require(not args.output.exists() and not delegated_output.exists() and not companion.exists(), 'Preserve existing/partial report')
    values = {'plan': Path(plan['previousFinalizationPlan']['path']), 'index': args.index,
        'audit': Path(audit['delegatedManifestViewAudit']['path']), 'output': delegated_output}
    argv = invocation(V3_PATH, 'report', values)
    with view.installed(Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
            stage='report', argv=argv, extra_bindings={'finalizer.report': (previous, 'report')}) as proof:
        previous.report(argparse.Namespace(**values))
    gc.collect(); value = io.read(delegated_output)
    io.require(value['metadata']['auditPassed'] is True
        and value['metadata']['audit'] == audit['delegatedManifestViewAudit'], 'Delegated report audit differs')
    content = {key: value[key] for key in ('inventory', 'scopes', 'series', 'tasks')}
    io.require(io.canonical(content) == audit['reportContentSha256'], 'Delegated report content changed')
    value['metadata']['audit'] = io.identity(args.audit)
    value['metadata']['originalManifestReadView'] = {
        'plan': plan['viewPlan'], 'qualification': plan['viewQualification'],
        'finalAuditExecution': audit['manifestViewFinalAuditExecution'],
        'policy': 'Exact-path legacy JSON compatibility only: records aliases the unchanged 8 exact, 3 draft and 7 coverage rows. '
                  'All original bytes, 18 recordings, 7 source groups, selections, scores, metrics and index bindings are preserved; '
                  '270 neural cells, production and final numerical audit require disclosed execution companions.'}
    io.require(preflight(plan, previous, prior, args.index) == before, 'Publication evidence changed during report construction')
    io.write_new(args.output, value)
    refs = {'delegatedReport': io.identity(delegated_output), 'report': io.identity(args.output)}
    io.write_new(companion, {**proof, 'outputs': refs})
    view.verify_execution(companion, Path(plan['viewPlan']['path']), Path(plan['viewQualification']['path']),
                          stage='report', argv=argv, outputs=refs)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    for action, names in (
        ('register', ('previous-plan', 'dispatcher-plan', 'output')),
        ('evaluate-production', ('plan', 'global-selection-gate', 'production', 'parity-audit', 'inventory', 'original-manifest', 'output')),
        ('audit-results', ('plan', 'index', 'report', 'global-selection-gate', 'student-cold-audit', 'output')),
        ('report', ('plan', 'index', 'audit', 'output'))):
        command = sub.add_parser(action)
        for name in names: command.add_argument('--'+name, type=Path, required=True)
        if action == 'audit-results': command.add_argument('--node', default=NODE)
    args = parser.parse_args()
    io.require(Path('/mnt/freenas').is_mount() and args.output.resolve().is_relative_to('/mnt/freenas'), 'Direct NAS output required')
    globals()[args.action.replace('-', '_')](args)
    print(json.dumps(io.identity(args.output)), flush=True)


if __name__ == '__main__':
    main()
