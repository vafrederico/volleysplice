#!/usr/bin/env python3
"""Independent source/cache proof of an explicitly injected duration comparison view."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_generalization_experiment import load_task

POLICY = {'kind': 'historical-cache-duration-comparison-view-v1',
    'injectedDependency': 'Evidence.document', 'replacedModuleSymbols': [],
    'interceptedDocument': 'historicalProtocol.source',
    'changedFields': ['exactRows.*.durationSeconds'], 'recordingCount': 8,
    'changedFieldCount': 7, 'storedEqualsCacheMetadataAndFrameCountDivFps': True,
    'sourceEqualsExactSixDecimalRounding': True, 'otherValuesTypesAndOrderUnchanged': True,
    'selectionBytesChanged': False, 'metricDurationsChanged': False,
    'delegatedAuditorFieldIsBodyIdentityOnly': True, 'independentCompanionRequired': True}


def expected_view(original, records, caches):
    """Derive the sole permitted comparison document from independently read caches."""
    by_id = {row['id']: row for row in records}
    ids = [row['id'] for row in original['exactRows']]
    io.require(len(records) == len(by_id) == len(ids) == len(set(ids)) == 8
               and set(ids) == set(by_id) == set(caches), 'Original eight-source inventory differs')
    result, changes, checks = deepcopy(original), [], []
    for index, label in enumerate(original['exactRows']):
        key = label['id']; row = by_id[key]
        metadata, times, cache_ref = caches[key]
        declared = label['featureCaches']['audiovisual']
        io.require(cache_ref == {field: declared[field] for field in ('path', 'sha256')},
                   'Cache is not the declared source cache')
        duration = metadata['duration']
        io.require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0
                   and type(row['durationSeconds']) in (int, float)
                   and type(label['durationSeconds']) in (int, float)
                   and type(metadata['frame_count']) is int and metadata['frame_count'] > 0
                   and type(metadata['fps']) in (int, float) and math.isfinite(metadata['fps'])
                   and metadata['fps'] > 0, 'Invalid cache/source duration metadata')
        io.require(row['durationSeconds'] == duration == declared['metadata']['duration']
                   == metadata['frame_count'] / metadata['fps'], 'Exact cache/frame duration differs')
        io.require(label['durationSeconds'] == round(duration, 6), 'Manifest is not the exact six-decimal rounding')
        io.require(times.ndim == 1 and len(times) and np.isfinite(times).all()
                   and np.all(np.diff(times) > 0) and times[0] >= 0 and times[-1] <= duration
                   and np.array_equal(times, row['timestamps']), 'Original cached timestamp grid differs')
        io.require(row['sourceGroup'] == label['sourceGroup'], 'Original source group differs')
        for field in ('rallies', 'ignoredIntervals'):
            normalized = [{'start': r['start'], 'end': r['end'],
                           **({'tags': r['tags']} if r.get('tags') else {})} for r in label[field]]
            # The historical loader already serializes integer endpoints as
            # floats. This cross-document check preserves the frozen auditor's
            # exact value/tag/order semantics; the entire comparison view below
            # must still preserve every original source type outside durations.
            io.require(row[field] == normalized, 'Original interval values/order differ')
        valid = np.ones(len(times), bool)
        for interval in label['ignoredIntervals']:
            valid &= ~((times >= interval['start']) & (times < interval['end']))
        io.require(np.array_equal(valid, row['valid']), 'Historical ignored validity differs')
        if label['durationSeconds'] != row['durationSeconds']:
            result['exactRows'][index]['durationSeconds'] = row['durationSeconds']
            changes.append({'path': f'/exactRows/{index}/durationSeconds', 'id': key,
                'before': label['durationSeconds'], 'after': row['durationSeconds'],
                'beforeType': type(label['durationSeconds']).__name__,
                'afterType': type(row['durationSeconds']).__name__, 'audiovisualCache': cache_ref})
        checks.append({'id': key, 'sourceGroup': row['sourceGroup'], 'audiovisualCache': cache_ref,
            'storedMetricDurationSeconds': row['durationSeconds'], 'cacheMetadataDurationSeconds': duration,
            'manifestDurationSeconds': label['durationSeconds'], 'frameCount': metadata['frame_count'],
            'fps': metadata['fps'], 'manifestMinusStoredSeconds': label['durationSeconds'] - duration,
            'storedEqualsBothCacheMetadataDurations': True, 'storedEqualsFrameCountDivFps': True,
            'manifestEqualsExactSixDecimalRound': True, 'timestampCount': len(times),
            'timestampsExactlyCache': True, 'sourceGroupAndIntervalValuesAndTagsExact': True,
            'validMaskExactlyIgnoredSubtraction': True})
    io.require(len(changes) == 7, 'Expected exactly seven display-duration replacements')
    return result, changes, checks


def check_proof(proof, original, records, caches, source_ref):
    expected, changes, checks = expected_view(original, records, caches)
    io.require(proof['interceptedDocuments'] == [source_ref] and proof['replacedSymbols'] == []
               and proof['evidenceOverride'] == 'Evidence.document'
               and proof['otherFunctionBindingsUnchanged'] is True
               and proof['otherEvidenceMethodsUnchanged'] is True
               and proof['selectionBytesUnchanged'] is True and proof['metricDurationChanged'] is False
               and proof['externalOutcomesRead'] is False and proof['trainingPerformed'] is False,
               'Undeclared source interception or numerical mutation')
    io.require(proof['originalSourceSha256'] == io.canonical(original)
               and proof['comparisonSourceSha256'] == io.canonical(expected)
               and io.canonical(proof['comparisonSource']) == io.canonical(expected)
               and io.canonical(proof['changes']) == io.canonical(changes),
               'Comparison view changes values/types/order outside declared durations')
    return changes, checks


def audit(plan_path, selection_path, base_path, proof_path, output):
    output = Path(output)
    io.require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(), 'New NAS gate required')
    body_path = REPO / 'scripts/audit-neural-generalization-selection.py'
    spec = importlib.util.spec_from_file_location('unadapted_duration_view_auditor_body', body_path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    evidence = old.Evidence()
    plan_ref = evidence.capture(plan_path); plan = evidence.document(plan_ref)
    io.require(plan['kind'] == 'generalization-selection-duration-comparison-amendment-v1'
               and io.canonical(plan['policy']) == io.canonical(POLICY), 'Duration comparison policy differs')
    evidence.closure(plan)
    io.require(plan['code']['scripts/audit-neural-selection-duration-view.py'] == io.identity(__file__),
               'Independent companion source is not bound')
    serialization = evidence.document(plan['serializationPlan'])
    correction = evidence.document(plan['correctionPlan'])
    io.require(serialization['kind'] == 'generalization-selection-serialization-amendment-v1'
               and serialization['correctionPlan'] == plan['correctionPlan'], 'Prior amendment association differs')
    selection_ref = evidence.capture(selection_path); selection = evidence.document(selection_ref)
    task, _, _, _ = load_task(evidence.bind(selection['task']))
    io.require(task['variant'] == 'original-corpus' and task['registration'] in correction['registrations']
               and selection['candidateMetricCorrection']['plan'] == plan['correctionPlan']
               and plan['historicalProtocol'] in selection['evidence'], 'Selection scope or historical source differs')
    protocol = evidence.document(plan['historicalProtocol'])
    source = evidence.document(protocol['source'])
    records = evidence.document(protocol['records'])['records']
    provenance = evidence.document(plan['durationProvenance'])
    io.require(provenance['kind'] == 'independent-historical-duration-provenance-audit-v1'
               and provenance['passed'] is True and provenance['protocol'] == plan['historicalProtocol']
               and provenance['source'] == protocol['source'] and provenance['records'] == protocol['records']
               and provenance['recordingCount'] == 8 and provenance['roundedManifestDifferenceCount'] == 7
               and provenance['auditor'] == io.identity(REPO / 'scripts/audit-neural-historical-duration-provenance.py')
               and provenance['metricDurationChanged'] is False and provenance['selectionChanged'] is False,
               'Independent duration provenance differs')
    evidence.closure(provenance)
    base_ref = evidence.capture(base_path); base = evidence.document(base_ref)
    proof_ref = evidence.capture(proof_path); proof = evidence.document(proof_ref)
    io.require(base['kind'] == 'independent-generalization-selection-audit-v1' and base['passed'] is True
               and base['selection'] == selection_ref and base['task'] == selection['task']
               and base['auditor'] == io.identity(body_path) and base['counts']['candidates'] == 192
               and 'executionAdapter' not in base and 'executionAuditor' not in base,
               'Delegated frozen auditor receipt differs')
    evidence.closure(base['references'])
    io.require(proof['kind'] == 'historical-duration-comparison-view-proof-v1' and proof['plan'] == plan_ref
               and proof['selection'] == selection_ref and proof['task'] == selection['task']
               and proof['executionAdapter'] == plan['code']['analysis/neural_selection_duration_view.py']
               and proof['delegatedAuditor'] == io.identity(body_path)
               and proof['historicalProtocol'] == plan['historicalProtocol']
               and proof['durationProvenance'] == plan['durationProvenance']
               and proof['source'] == protocol['source'] and proof['records'] == protocol['records'],
               'Comparison proof source/selection identity differs')
    caches = {}
    for row in source['exactRows']:
        ref = {key: row['featureCaches']['audiovisual'][key] for key in ('path', 'sha256')}
        with np.load(evidence.bind(ref), allow_pickle=False) as payload:
            caches[row['id']] = (json.loads(str(payload['metadata_json'].item())), payload['times'].copy(), ref)
    changes, checks = check_proof(proof, source, records, caches, protocol['source'])
    io.require({row['id']: row for row in checks} == {row['id']: row for row in provenance['checks']}
               and len(provenance['checks']) == 8, 'Independent provenance receipt content differs')
    examples = io.examples_from_rows(records)
    original_gold = [{**io.serial_rows([e.row([])])[0], 'times': e.times.tolist(), 'valid': e.valid.tolist(),
                      'labelPolicy': e.label_policy} for e in examples]
    io.require(io.canonical(original_gold) == selection['selectionGoldSha256'], 'Original metric gold or duration changed')
    old.audit_floors(selection)
    for ref in list(evidence.references.values()):
        evidence.bind(ref)
    result = {'kind': 'independent-selection-duration-view-audit-v1', 'passed': True,
        'plan': plan_ref, 'selection': selection_ref, 'task': selection['task'], 'proof': proof_ref,
        'baseAudit': base_ref, 'historicalProtocol': plan['historicalProtocol'], 'source': protocol['source'],
        'records': protocol['records'], 'durationProvenance': plan['durationProvenance'],
        'changedFieldCount': len(changes), 'recordingCount': len(records),
        'allOtherValuesTypesAndOrderPreserved': True, 'oneDeclaredDocumentInterception': True,
        'originalMetricGoldSha256Unchanged': True, 'exactCacheFrameAndRoundingProof': True,
        'auditor': io.identity(__file__), 'evidence': list(evidence.references.values()),
        'externalOutcomesRead': False, 'trainingPerformed': False, 'gpuUsed': False}
    io.write_new(output, result)
    return result
