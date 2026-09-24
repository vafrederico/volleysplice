"""Disclosed comparison-only duration view for the frozen historical audit body."""
from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np

from . import neural_recall_sweep as io
from .neural_generalization_inputs import verified
from . import neural_selection_serialization as serialization

REPO = Path(__file__).resolve().parents[1]
POLICY = {'kind': 'historical-cache-duration-comparison-view-v1',
    'injectedDependency': 'Evidence.document', 'replacedModuleSymbols': [],
    'interceptedDocument': 'historicalProtocol.source',
    'changedFields': ['exactRows.*.durationSeconds'], 'recordingCount': 8,
    'changedFieldCount': 7, 'storedEqualsCacheMetadataAndFrameCountDivFps': True,
    'sourceEqualsExactSixDecimalRounding': True, 'otherValuesTypesAndOrderUnchanged': True,
    'selectionBytesChanged': False, 'metricDurationsChanged': False,
    'delegatedAuditorFieldIsBodyIdentityOnly': True, 'independentCompanionRequired': True}


def verify_plan(path):
    plan = io.read(path)
    io.require(plan['kind'] == 'generalization-selection-duration-comparison-amendment-v1'
        and plan['policy'] == POLICY, 'Duration comparison amendment differs')
    for key in ('serializationPlan', 'correctionPlan', 'historicalProtocol', 'durationProvenance', 'protocol'):
        verified(plan[key])
    prior = serialization.verify_plan(plan['serializationPlan']['path'])
    io.require(plan['correctionPlan'] == prior['correctionPlan'], 'Correction plan association differs')
    for ref in plan['code'].values(): verified(ref)
    io.require(plan['code']['analysis/neural_selection_duration_view.py'] == io.identity(__file__), 'Duration adapter changed')
    return plan


def comparison_view(source, records, cache_loader):
    """Build only a source-comparison view; retain original stored metric records."""
    view, changes = deepcopy(source), []
    by_id = {r['id']: r for r in records}
    io.require(len(by_id) == len(records) == len(source['exactRows'])
        and set(by_id) == {r['id'] for r in source['exactRows']}, 'Duration view source membership differs')
    for index, gold in enumerate(source['exactRows']):
        row = by_id[gold['id']]; cache = gold['featureCaches']['audiovisual']
        ref = {k: cache[k] for k in ('path', 'sha256')}
        metadata, times = cache_loader(ref)
        duration = metadata['duration']
        io.require(type(duration) in (float, int) and np.isfinite(duration) and duration > 0
            and type(gold['durationSeconds']) in (float, int), 'Invalid cache/source duration')
        io.require(row['durationSeconds'] == duration == cache['metadata']['duration']
            == metadata['frame_count'] / metadata['fps'], 'Exact cache and frame timing differs')
        io.require(gold['durationSeconds'] == round(duration, 6), 'Source is not exact six-decimal cache rounding')
        io.require(np.array_equal(times, np.asarray(row['timestamps'])), 'Stored timestamps differ from cache')
        io.require(row['sourceGroup'] == gold['sourceGroup'], 'Original source group differs')
        for field in ('rallies', 'ignoredIntervals'):
            normalized = [{'start': r['start'], 'end': r['end'], **({'tags': r['tags']} if r.get('tags') else {})}
                          for r in gold[field]]
            io.require(normalized == row[field], 'Original interval values/tags/order differ')
        mask = np.ones(len(times), bool)
        for r in gold['ignoredIntervals']: mask &= ~((times >= r['start']) & (times < r['end']))
        io.require(np.array_equal(mask, row['valid']), 'Historical valid mask differs')
        if gold['durationSeconds'] != row['durationSeconds']:
            value = row['durationSeconds']
            view['exactRows'][index]['durationSeconds'] = value
            changes.append({'path': f'/exactRows/{index}/durationSeconds', 'id': gold['id'],
                'before': gold['durationSeconds'], 'after': value, 'beforeType': type(gold['durationSeconds']).__name__,
                'afterType': type(value).__name__, 'audiovisualCache': ref})
    return view, changes


def audit_selection(plan_path, selection_path, output):
    plan = verify_plan(plan_path); output = Path(output)
    io.require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(), 'New NAS audit required')
    selection_ref = io.identity(selection_path); selection = io.read(selection_path)
    io.require(selection['candidateMetricCorrection']['plan'] == plan['correctionPlan'], 'Selection correction plan differs')
    task = io.read(verified(selection['task']))
    correction = io.read(verified(plan['correctionPlan']))
    io.require(task['variant'] == 'original-corpus' and task['registration'] in correction['registrations'], 'Original registered task required')
    fit_ref, = [r for r in selection['evidence'] if Path(r['path']).name == 'fit-result.json']
    folder = Path(verified(fit_ref)).parent
    historical = Path(plan['historicalProtocol']['path']).parent
    io.require(plan['historicalProtocol'] in selection['evidence'], 'Historical selection protocol differs')
    protocol = io.read(verified(plan['historicalProtocol']))
    provenance = io.read(verified(plan['durationProvenance']))
    io.require(provenance['kind'] == 'independent-historical-duration-provenance-audit-v1' and provenance['passed'] is True
        and provenance['protocol'] == plan['historicalProtocol'] and provenance['source'] == protocol['source']
        and provenance['records'] == protocol['records'] and provenance['recordingCount'] == 8
        and provenance['roundedManifestDifferenceCount'] == 7
        and provenance['auditor'] == io.identity(REPO/'scripts/audit-neural-historical-duration-provenance.py'), 'Duration provenance differs')
    for ref in provenance['evidence']: verified(ref)
    delegated_path = output.with_name(output.stem+'-duration-delegated-base.json')
    proof_path = output.with_name(output.stem+'-duration-proof.json')
    companion_path = output.with_name(output.stem+'-duration-gate.json')
    io.require(not any(p.exists() for p in (delegated_path, proof_path, companion_path)), 'Duration sidecar already exists')
    old = serialization.load_module('duration_comparison_frozen_auditor', REPO/'scripts/audit-neural-generalization-selection.py')
    functions = {k: v for k, v in vars(old).items() if inspect.isfunction(v)}
    methods = {k: v for k, v in vars(old.Evidence).items() if inspect.isfunction(v)}
    evidence = old.Evidence()
    source = evidence.document(protocol['source']); records = evidence.document(protocol['records'])['records']
    def cache_loader(ref):
        with np.load(evidence.bind(ref), allow_pickle=False) as payload:
            return json.loads(str(payload['metadata_json'].item())), payload['times'].copy()
    view, changes = comparison_view(source, records, cache_loader)
    io.require(len(records) == 8 and len(changes) == 7, 'Predeclared duration view scope differs')
    intercepted = []
    class ComparisonEvidence(old.Evidence):
        def document(self, reference):
            original = super().document(reference)
            if reference == protocol['source']:
                io.require(io.canonical(original) == io.canonical(source), 'Original source changed before comparison')
                intercepted.append(reference)
                return deepcopy(view)
            return original
    adapted_evidence = ComparisonEvidence()
    adapted_evidence.cache = evidence.cache
    old.audit(selection['task']['path'], folder, selection_path, delegated_path, historical=historical, evidence=adapted_evidence)
    io.require(intercepted == [protocol['source']]
        and all(vars(old)[k] is v for k, v in functions.items())
        and all(vars(old.Evidence)[k] is v for k, v in methods.items())
        and set(vars(ComparisonEvidence)) - {'__module__', '__doc__', '__qualname__'} == {'document'},
        'Evidence interception/function binding scope differs')
    io.require(io.identity(selection_path) == selection_ref, 'Selection bytes changed')
    proof = {'kind': 'historical-duration-comparison-view-proof-v1', 'plan': io.identity(plan_path),
        'selection': selection_ref, 'task': selection['task'], 'executionAdapter': io.identity(__file__),
        'delegatedAuditor': io.identity(REPO/'scripts/audit-neural-generalization-selection.py'),
        'historicalProtocol': plan['historicalProtocol'], 'source': protocol['source'], 'records': protocol['records'],
        'durationProvenance': plan['durationProvenance'], 'originalSourceSha256': io.canonical(source),
        'comparisonSourceSha256': io.canonical(view), 'comparisonSource': view, 'changes': changes,
        'interceptedDocuments': intercepted, 'replacedSymbols': [], 'evidenceOverride': 'Evidence.document',
        'otherFunctionBindingsUnchanged': True, 'otherEvidenceMethodsUnchanged': True,
        'selectionBytesUnchanged': True, 'metricDurationChanged': False, 'externalOutcomesRead': False, 'trainingPerformed': False}
    io.write_new(proof_path, proof)
    companion = serialization.load_module('independent_duration_comparison_gate', REPO/'scripts/audit-neural-selection-duration-view.py')
    companion.audit(plan_path, selection_path, delegated_path, proof_path, companion_path)
    base = io.read(delegated_path)
    execution = {'source': io.identity(__file__), 'plan': io.identity(plan_path), 'proof': io.identity(proof_path),
        'durationAudit': io.identity(companion_path), 'baseAudit': io.identity(delegated_path),
        'replacedSymbols': [], 'evidenceOverride': 'Evidence.document'}
    refs = [execution[k] for k in ('source', 'plan', 'proof', 'durationAudit', 'baseAudit')]
    io.write_new(output, {**base, 'executionAuditor': io.identity(__file__), 'executionAdapter': execution,
        'executionScope': 'The auditor field identifies the frozen delegated body for compatibility. The body ran against an explicitly derived source comparison view, NOT the unmodified source JSON: seven rounded display durations are replaced by independently proved original cache durations through Evidence.document only. Saved selection, label, timestamp and metric-duration values are unchanged; the independent duration-view gate is mandatory.',
        'references': [*base['references'], *refs]})
    return io.read(output)


def verify_execution(ordinary_path):
    ordinary = io.read(ordinary_path)
    task = io.read(verified(ordinary['task']))
    if task['variant'] != 'original-corpus': return serialization.verify_execution(ordinary_path)
    io.require(ordinary.get('executionAuditor') == io.identity(__file__) and 'executionAdapter' in ordinary,
               'Original historical comparison-view adapter is missing or stripped')
    execution = ordinary['executionAdapter']; plan = verify_plan(verified(execution['plan']))
    for key in ('source', 'plan', 'proof', 'durationAudit', 'baseAudit'): verified(execution[key])
    io.require(ordinary['kind'] == 'independent-generalization-selection-audit-v1' and ordinary['passed'] is True
        and ordinary['auditor'] == io.identity(REPO/'scripts/audit-neural-generalization-selection.py')
        and execution['source'] == io.identity(__file__) and execution['replacedSymbols'] == []
        and execution['evidenceOverride'] == 'Evidence.document', 'Historical comparison execution differs')
    base = io.read(execution['baseAudit']['path']); gate = io.read(execution['durationAudit']['path'])
    io.require(gate['kind'] == 'independent-selection-duration-view-audit-v1' and gate['passed'] is True
        and gate['selection'] == ordinary['selection'] and gate['task'] == ordinary['task']
        and gate['plan'] == execution['plan'] and gate['proof'] == execution['proof']
        and gate['baseAudit'] == execution['baseAudit'] and gate['historicalProtocol'] == plan['historicalProtocol']
        and gate['durationProvenance'] == plan['durationProvenance'] and gate['changedFieldCount'] == 7
        and gate['recordingCount'] == 8 and gate['auditor'] == io.identity(REPO/'scripts/audit-neural-selection-duration-view.py'),
        'Independent historical duration-view gate differs')
    for reference in gate['evidence']: verified(reference)
    stripped = {k: v for k, v in ordinary.items() if k not in ('executionAuditor', 'executionAdapter', 'executionScope')}
    stripped['references'] = base['references']
    io.require(io.canonical(stripped) == io.canonical(base) and ordinary['references'] == [*base['references'],
        *[execution[k] for k in ('source', 'plan', 'proof', 'durationAudit', 'baseAudit')]], 'Adapter changed delegated audit fields')
    return {'ordinaryAudit': io.identity(ordinary_path), 'execution': 'explicit-historical-duration-comparison-view',
        'durationAudit': execution['durationAudit'], 'plan': execution['plan']}
