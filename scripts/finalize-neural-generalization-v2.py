#!/usr/bin/env python3
"""Explicit finalization through frozen publication gates and the declared v2 audit."""
import argparse
from datetime import datetime, timezone
import inspect
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_inputs import verified
from analysis.neural_selection_serialization import load_module

cache = load_module('declared_cache_v2_finalization', REPO/'scripts/generalization-selection-cache-execution-v2.py')
execution = cache.body
containment = execution.old
KIND = 'independent-generalization-finalization-audit-v1'
EXTRA = ('finalizationAuditor', 'finalizationPlan', 'delegatedNumericalAudit', 'globalSelectionGate',
         'globalIdentityCacheGate', 'selectionCorrectionGates', 'evaluationPublicationGates', 'productionPublicationGate')


def now(): return datetime.now(timezone.utc).isoformat()


def production_continuity(reference):
    gate = io.read(verified(reference)); containment.verify_closure(gate)
    io.require(gate['kind'] == 'independent-production-panel-continuity-audit-v1' and gate['passed'] is True
        and gate['auditor'] == io.identity(REPO/'scripts/audit-neural-production-panel.py')
        and gate['recordingCount'] == 42 and gate['scoredRecordingCount'] == 34
        and gate['sameOrderedPopulationAndGold'] is True and gate['allAudiovisualReferencesAndBytesIdentical'] is True
        and gate['predictionOrOutcomeAccessed'] is False and len(gate['checks']) == 42,
        'Complete independent production AV continuity gate required')
    return gate


def verify_plan(path):
    plan = io.read(path)
    io.require(plan['kind'] == 'generalization-explicit-finalization-plan-v2'
        and plan['numericalFunctionsReplaced'] is False and plan['allExternalMetricsRequireGlobal162ColdGate'] is True,
        'Finalization execution contract differs')
    for reference in plan['code'].values(): verified(reference)
    io.require(plan['code']['scripts/finalize-neural-generalization-v2.py'] == io.identity(__file__), 'Finalizer changed')
    io.require(plan['numericalAuditor'] == plan['code']['scripts/audit-neural-generalization-results-v2.py']
        and plan['originalNumericalAuditor'] == plan['code']['scripts/audit-neural-generalization-results.py'],
        'Declared numerical auditor is outside the registered source closure')
    cache.verify_plan(verified(plan['cachePlan'])); verified(plan['protocol'])
    gate = production_continuity(plan['productionPanelAudit'])
    io.require(gate['panel'] == plan['productionPanel'] and gate['corePanel'] == plan['neuralCorePanel'], 'Production panel association differs')
    return plan


def register(args):
    prior = cache.verify_plan(args.cache_plan); gate = production_continuity(io.identity(args.production_panel_audit))
    names = ('scripts/finalize-neural-generalization-v2.py', 'scripts/audit-neural-generalization-results-v2.py',
        'scripts/register-neural-production-panel.py', 'scripts/audit-neural-production-panel.py',
        'analysis/tests/test_neural_production_panel.py', 'analysis/tests/test_neural_generalization_finalization.py',
        'analysis/tests/test_generalization_results_audit_v2.py')
    numerical = load_module('declared_v2_audit_signature', REPO/'scripts/audit-neural-generalization-results-v2.py')
    inspect.signature(numerical.audit).bind(None, None, None, None)
    io.write_new(args.output, {'kind': 'generalization-explicit-finalization-plan-v2', 'createdAtUTC': now(),
        'cachePlan': io.identity(args.cache_plan), 'productionPanelAudit': io.identity(args.production_panel_audit),
        'productionPanel': gate['panel'], 'neuralCorePanel': gate['corePanel'], 'protocol': io.identity(args.protocol.resolve()),
        'code': {**prior['code'], **{name: io.identity(REPO/name) for name in names}},
        'numericalAuditor': io.identity(REPO/'scripts/audit-neural-generalization-results-v2.py'),
        'originalNumericalAuditor': io.identity(REPO/'scripts/audit-neural-generalization-results.py'),
        'auditContractRevision': numerical.REVISION_POLICY, 'numericalFunctionsReplaced': False,
        'allExternalMetricsRequireGlobal162ColdGate': True, 'originalMetricAndDecoderSourcesUnchanged': True,
        'explicitDispatch': {'globalAndColdGate': 'cache-v2.global_cache_gate', 'all270ExecutionGates': 'execution.publication_preflight',
            'all162CorrectionGates': 'containment.preflight_index', 'numericalAudit': 'audit-neural-generalization-results-v2.audit',
            'report': 'neural_generalization_report.build_report'}})


def global_preflight(plan, global_path):
    frozen, cache_ref = cache.global_cache_gate(global_path)
    cache_gate = io.read(cache_ref['path'])
    io.require(cache_gate['plan'] == plan['cachePlan'], 'Global freeze uses another hash-cache plan')
    return {'globalSelectionGate': io.identity(global_path), 'globalIdentityCacheGate': cache_ref}


def evaluate_production(args):
    plan = verify_plan(args.plan); before = global_preflight(plan, args.global_selection_gate)
    continuity = production_continuity(plan['productionPanelAudit'])
    io.require(continuity['productionProtocol'] == io.identity(args.production/'protocol.json'), 'Production raw inference protocol changed')
    panel = io.read(verified(plan['productionPanel']))
    io.require(io.identity(args.inventory) == panel['inventory'], 'Production evaluation inventory differs')
    start = args.output.with_name(args.output.stem+'-evaluation-start.json')
    publication = args.output.with_name(args.output.stem+'-publication-gate.json')
    io.require(not args.output.exists() and not start.exists() and not publication.exists(), 'Existing/partial production evaluation requires inspection')
    io.write_new(start, {'kind': 'production-evaluation-after-global-selection-freeze-v1', **before,
        'plan': io.identity(args.plan), 'panelContinuity': plan['productionPanelAudit'], 'panel': plan['productionPanel'],
        'productionProtocol': continuity['productionProtocol'], 'parityAudit': io.identity(args.parity_audit),
        'inventory': io.identity(args.inventory), 'originalManifest': io.identity(args.original_manifest),
        'workflow': io.identity(__file__), 'createdAtUTC': now()})
    from analysis.neural_generalization_results import evaluate_production as frozen_evaluate
    frozen_evaluate(args.production, args.parity_audit, Path(plan['productionPanel']['path']),
        args.inventory, args.original_manifest, args.output)
    io.require(global_preflight(plan, args.global_selection_gate) == before, 'Global gates changed during production evaluation')
    production_continuity(plan['productionPanelAudit'])
    io.write_new(publication, {'kind': 'production-generalization-evaluation-publication-gate-v1', 'passed': True,
        **before, 'plan': io.identity(args.plan), 'panelContinuity': plan['productionPanelAudit'],
        'panel': plan['productionPanel'], 'evaluation': io.identity(args.output), 'evaluationStart': io.identity(start),
        'workflow': io.identity(__file__)})


def production_publication(index_path, plan, plan_ref, global_refs):
    index = io.read(index_path); found = []
    # The frozen indexer appends the single production document after270 neural
    # entries. The separate execution preflight requires all270 neural gates.
    io.require(len(index['evaluations']) == 271, 'Complete index population required')
    for entry in index['evaluations'][-1:]:
        document = io.read(verified(entry['result']))
        if document['kind'] != 'fixed-production-generalization-evaluation-v1': continue
        path = Path(entry['result']['path']); reference = io.identity(path.with_name(path.stem+'-publication-gate.json'))
        gate = io.read(reference['path']); containment.verify_closure(gate)
        io.require(gate['kind'] == 'production-generalization-evaluation-publication-gate-v1' and gate['passed'] is True
            and gate['evaluation'] == entry['result'] and gate['plan'] == plan_ref and gate['workflow'] == io.identity(__file__)
            and gate['panelContinuity'] == plan['productionPanelAudit'] and gate['panel'] == plan['productionPanel']
            and all(gate[k] == v for k,v in global_refs.items()), 'Production publication prerequisite differs')
        start = io.read(verified(gate['evaluationStart']))
        io.require(start['kind'] == 'production-evaluation-after-global-selection-freeze-v1'
            and start['plan'] == plan_ref and start['workflow'] == gate['workflow']
            and start['panel'] == gate['panel'] and start['panelContinuity'] == gate['panelContinuity']
            and all(start[k] == v for k,v in global_refs.items()), 'Production started without same global freeze')
        io.require(len(document['results']) == 2 and all(r['panel'] == gate['panel']
            and r['protocol'] == start['productionProtocol'] and r['productionParityAudit'] == start['parityAudit']
            and r['inventory'] == start['inventory'] and r['originalManifest'] == start['originalManifest']
            for r in document['results']), 'Production output source/gold bindings differ')
        found.append(reference)
    io.require(len(found) == 1, 'Exactly one fixed-production evaluation required')
    return found[0]


def final_preflight(plan_path, index_path, global_path):
    plan = verify_plan(plan_path); refs = global_preflight(plan, global_path)
    correction = containment.preflight_index(index_path)
    publications = execution.publication_preflight(index_path, global_path)
    production = production_publication(index_path, plan, io.identity(plan_path), refs)
    return {**refs, 'selectionCorrectionGates': correction, 'evaluationPublicationGates': publications,
        'productionPublicationGate': production}


def audit_results(args):
    plan = verify_plan(args.plan); checks = final_preflight(args.plan, args.index, args.global_selection_gate)
    base = args.output.with_name(args.output.stem+'-numerical-v2.json')
    io.require(not args.output.exists() and not base.exists(), 'Existing/partial final audit requires inspection')
    numerical = load_module('explicit_full_numerical_auditor_v2', verified(plan['numericalAuditor']))
    numerical.audit(args.index, args.report, base, args.node)
    io.require(final_preflight(args.plan, args.index, args.global_selection_gate) == checks, 'Publication dependencies changed during numerical audit')
    result = io.read(base)
    io.require(result['kind'] == 'independent-generalization-result-report-audit-v2' and result['passed'] is True
        and result['auditor'] == plan['numericalAuditor'] and result['originalAuditor'] == plan['originalNumericalAuditor']
        and result['auditContractRevision'] == plan['auditContractRevision'], 'Declared v2 numerical audit differs')
    io.write_new(args.output, {**result, 'kind': KIND, 'finalizationAuditor': io.identity(__file__),
        'finalizationPlan': io.identity(args.plan), 'delegatedNumericalAudit': io.identity(base), **checks})


def restore_numerical(audit, base):
    restored = {k:v for k,v in audit.items() if k not in EXTRA}
    restored['kind'] = 'independent-generalization-result-report-audit-v2'
    io.require(io.canonical(restored) == io.canonical(base), 'Finalizer changed delegated numerical audit or report digest')


def report(args):
    audit = io.read(args.audit); plan = verify_plan(args.plan)
    io.require(audit['kind'] == KIND and audit['passed'] is True and audit['finalizationAuditor'] == io.identity(__file__)
        and audit['finalizationPlan'] == io.identity(args.plan) and audit['resultIndex'] == io.identity(args.index), 'Complete explicit final audit required')
    containment.verify_closure(audit); base = io.read(verified(audit['delegatedNumericalAudit']))
    restore_numerical(audit, base)
    io.require(base['auditor'] == plan['numericalAuditor'] and base['originalAuditor'] == plan['originalNumericalAuditor']
        and base['auditContractRevision'] == plan['auditContractRevision'], 'Numerical audit contract changed')
    checks = final_preflight(args.plan, args.index, verified(audit['globalSelectionGate']))
    io.require(all(audit[k] == v for k,v in checks.items()), 'Final report gates differ')
    from analysis.neural_generalization_report import build_report
    return build_report(args.index, args.output, args.audit)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    reg = sub.add_parser('register')
    for key in ('cache-plan', 'production-panel-audit', 'protocol', 'output'): reg.add_argument('--'+key, type=Path, required=True)
    prod = sub.add_parser('evaluate-production')
    for key in ('plan', 'global-selection-gate', 'production', 'parity-audit', 'inventory', 'original-manifest', 'output'):
        prod.add_argument('--'+key, type=Path, required=True)
    aud = sub.add_parser('audit-results')
    for key in ('plan', 'index', 'report', 'global-selection-gate', 'output'): aud.add_argument('--'+key, type=Path, required=True)
    aud.add_argument('--node', default='/mnt/c/Program Files/nodejs/node.exe')
    rep = sub.add_parser('report')
    for key in ('plan', 'index', 'audit', 'output'): rep.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); io.require(Path('/mnt/freenas').is_mount() and a.output.resolve().is_relative_to('/mnt/freenas'), 'Direct NAS output required')
    globals()[a.action.replace('-', '_')](a); print(io.identity(a.output), flush=True)


if __name__ == '__main__': main()
