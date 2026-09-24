"""Disclosed, exact-path compatibility view for the frozen legacy manifest.

The legacy bytes, identities, model inputs and numerical callables are unchanged.
Only the shared JSON reader's return value for one registered document receives
the missing ``records`` alias. Every execution needs a separate companion.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path

from . import neural_recall_sweep as io

REPO = Path(__file__).resolve().parents[1]
TIERS = ('exactRows', 'draftRows', 'coverageRows')
PLAN_KIND = 'lossless-original-manifest-view-plan-v1'
QUAL_KIND = 'independent-original-manifest-view-qualification-v1'
EXEC_KIND = 'explicit-original-manifest-view-execution-v1'
POLICY = {
    'replacedSymbol': 'analysis.neural_recall_sweep.read',
    'exactRegisteredPathAndHashOnly': True,
    'addedField': 'records', 'concatenateInOrder': list(TIERS),
    'existingFieldsRowsTypesAndOrderUnchanged': True,
    'legacyBytesAndIdentityUnchanged': True,
    'numericalAndHashFunctionsReplaced': False,
    'labelsScoresSelectionsAndPanelMembershipUnchanged': True,
    'independentQualificationAndPerExecutionCompanionRequired': True,
}
STAGES = ('neural-evaluation', 'production-evaluation', 'final-audit', 'report')


def raw_read(path):
    """Do not recursively consult the temporary compatibility reader."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(reference):
    io.require(io.identity(reference['path']) == reference, 'Manifest-view reference changed')
    return Path(reference['path'])


def projection(document):
    io.require(document.get('kind') == 'neural-expanded-quality-tier-manifest-v1'
               and 'records' not in document, 'Expected the unmodified legacy tier manifest')
    io.require(all(isinstance(document.get(k), list) for k in TIERS), 'Missing legacy tier')
    rows = [row for key in TIERS for row in document[key]]
    io.require([len(document[k]) for k in TIERS] == [8, 3, 7]
               and len({r['id'] for r in rows}) == 18
               and all(isinstance(r.get('sourceGroup'), str) and r['sourceGroup'] for r in rows)
               and len({r['sourceGroup'] for r in rows}) == 7, 'Legacy18/seven-group scope changed')
    # A new outer dictionary and list; every existing field and row is retained.
    return {**document, 'records': rows}


def verify_plan(path):
    plan = raw_read(path)
    io.require(plan['kind'] == PLAN_KIND and plan['policy'] == POLICY
               and plan['adapter'] == io.identity(__file__), 'Manifest-view contract differs')
    for name in ('legacyManifest', 'normalizedManifest', 'inventory', 'correspondence',
                 'originalEvaluationPlan', 'panel', 'globalSelectionGate', 'failedExecution'):
        checked(plan[name])
    for reference in [*plan['code'].values(), *plan['frozenCode'].values(), *plan['registrations']]:
        checked(reference)
    legacy = raw_read(plan['legacyManifest']['path']); view = projection(legacy)
    proof = raw_read(plan['correspondence']['path'])
    io.require(proof['kind'] == 'independent-legacy-original-manifest-correspondence-v1'
               and proof['passed'] is True and proof['legacyManifest'] == plan['legacyManifest']
               and proof['normalizedManifest'] == plan['normalizedManifest']
               and proof['inventory'] == plan['inventory']
               and proof['recordingCount'] == 18 and proof['sourceGroupCount'] == 7,
               'Independent original-manifest correspondence is missing')
    ids = [r['id'] for r in view['records']]; groups = sorted({r['sourceGroup'] for r in view['records']})
    io.require(ids == proof['orderedRecordingIds'] == plan['orderedRecordingIds']
               and groups == proof['sourceGroups'] == plan['sourceGroups']
               and io.canonical(view['records']) == plan['recordsCanonicalSha256'],
               'Compatibility projection membership/order/rows changed')
    original_plan = raw_read(plan['originalEvaluationPlan']['path'])
    io.require(original_plan['originalManifest'] == plan['legacyManifest'],
               'Registered original-manifest argument changed')
    expected_tasks = list({r['task']['path']: r['task'] for r in original_plan['jobs']}.values())
    io.require(len(expected_tasks) == 162 and plan['tasks'] == expected_tasks,
               'Compatibility task inventory differs from original162')
    io.require(raw_read(plan['failedExecution']['path'])['exitCode'] == 1,
               'Preserved failed evaluation exit differs')
    return plan


def verify_qualification(path, plan_path):
    plan = verify_plan(plan_path); value = raw_read(path)
    io.require(value['kind'] == QUAL_KIND and value['passed'] is True
               and value['plan'] == io.identity(plan_path)
               and value['qualifier'] == plan['code']['scripts/qualify-neural-original-manifest-view.py']
               and value['adapter'] == plan['adapter'] and value['legacyManifest'] == plan['legacyManifest']
               and value['recordsCanonicalSha256'] == plan['recordsCanonicalSha256']
               and value['fullOriginalDocumentPreserved'] is True
               and value['all18RowsUnchanged'] is True and value['allSevenSourceGroupsPreserved'] is True
               and value['neuralTaskCount'] == 162 and value['productionComparatorCount'] == 2
               and value['allPanelMembershipsEqual'] is True and value['scoreArraysRead'] is False
               and value['accuracyMetricsComputed'] is False, 'Incomplete compatibility qualification')
    io.require(len(value['neuralMembershipChecks']) == 162
               and len({r['task']['sha256'] for r in value['neuralMembershipChecks']}) == 162
               and {(r['task']['path'], r['task']['sha256']) for r in value['neuralMembershipChecks']}
                   == {(r['path'], r['sha256']) for r in plan['tasks']}
               and all(r['equal'] is True for r in value['neuralMembershipChecks'])
               and value['productionMembershipCheck']['equal'] is True, 'Incomplete scope correspondence')
    return value


def numerical_bindings(extra=None):
    from . import neural_generalization_results as results
    from . import crop_evaluation
    from . import neural_evaluation
    values = {'io.identity': (io, 'identity'), 'io.canonical': (io, 'canonical'),
              'io.panel_rows': (io, 'panel_rows'), 'io.evaluate_rows': (io, 'evaluate_rows'),
              'results.evaluate_task': (results, 'evaluate_task'),
              'results.evaluate_production': (results, 'evaluate_production'),
              'results.panel_members': (results, 'panel_members'),
              'results.metric_rows': (results, 'metric_rows'),
              'results.inference_scores': (results, 'inference_scores'),
              'decoder.decode': (results.base, 'decode'),
              'crop.evaluate_f1_pad_p_core_r': (crop_evaluation, 'evaluate_f1_pad_p_core_r'),
              'events.evaluate_predictions': (neural_evaluation, 'evaluate_predictions')}
    io.require(not set(values).intersection(extra or {}), 'Duplicate protected binding name')
    values.update(extra or {})
    return {key: (owner, name, getattr(owner, name)) for key, (owner, name) in values.items()}


@contextmanager
def _installed(plan, plan_ref, qualification_ref, *, stage, argv, extra_bindings=None):
    """Internal implementation; public entry validates the frozen qualification."""
    io.require(stage in STAGES and isinstance(argv, list) and all(isinstance(s, str) for s in argv),
               'Declared compatibility stage/argv required')
    original_read = io.read
    io.require(getattr(original_read, '_original_manifest_view', False) is False,
               'Nested manifest-view installation is forbidden')
    bindings = numerical_bindings(extra_bindings)
    path = str(Path(plan['legacyManifest']['path']))
    original = raw_read(checked(plan['legacyManifest']))
    expected = projection(original)
    proof = {'kind': EXEC_KIND, 'passed': False, 'plan': plan_ref,
             'qualification': qualification_ref, 'adapter': io.identity(__file__),
             'policy': POLICY, 'stage': stage, 'equivalentArgv': list(argv),
             'legacyManifest': plan['legacyManifest'],
             'recordsCanonicalSha256': plan['recordsCanonicalSha256'],
             'readCount': 0, 'readSites': [], 'pid': os.getpid(),
             'startedAtUTC': datetime.now(timezone.utc).isoformat()}
    sites = set()
    def reader(requested):
        value = original_read(requested)
        if str(Path(requested)) != path:
            return value
        checked(plan['legacyManifest'])
        io.require(value == original, 'Legacy JSON changed while reading')
        result = projection(value)
        io.require(result == expected, 'Lossless manifest projection differs')
        proof['readCount'] += 1
        caller = inspect.currentframe().f_back
        sites.add((caller.f_code.co_filename, caller.f_code.co_name))
        return result
    reader._original_manifest_view = True
    io.read = reader
    succeeded = False
    try:
        yield proof
        succeeded = True
    finally:
        reader_intact = io.read is reader
        io.read = original_read
        unchanged = all(getattr(owner, name) is function for owner, name, function in bindings.values())
        proof.update(readerRestored=io.read is original_read, onlyDeclaredReaderReplaced=reader_intact,
                     numericalBindingsUnchanged=unchanged, guardedBindings=sorted(bindings),
                     evidenceBindingsReplaced=False, hashFunctionsReplaced=False,
                     readSites=[{'path': p, 'function': f} for p, f in sorted(sites)],
                     finishedAtUTC=datetime.now(timezone.utc).isoformat())
        io.require(reader_intact and unchanged, 'A protected function binding changed during compatibility execution')
        checked(plan['legacyManifest'])
        if succeeded:
            io.require(stage == 'report' or proof['readCount'] > 0, 'Required legacy read view was never used')
            proof['passed'] = True


@contextmanager
def installed(plan_path, qualification_path, *, stage, argv, extra_bindings=None):
    plan = verify_plan(plan_path); verify_qualification(qualification_path, plan_path)
    with _installed(plan, io.identity(plan_path), io.identity(qualification_path),
                    stage=stage, argv=argv, extra_bindings=extra_bindings) as proof:
        yield proof


def verify_execution(path, plan_path, qualification_path, *, stage, argv, outputs):
    plan = verify_plan(plan_path); verify_qualification(qualification_path, plan_path)
    value = raw_read(path)
    io.require(stage in STAGES and value['kind'] == EXEC_KIND and value['passed'] is True
               and value['plan'] == io.identity(plan_path)
               and value['qualification'] == io.identity(qualification_path)
               and value['adapter'] == plan['adapter'] and value['policy'] == POLICY
               and value['stage'] == stage and value['equivalentArgv'] == argv
               and value['legacyManifest'] == plan['legacyManifest']
               and value['recordsCanonicalSha256'] == plan['recordsCanonicalSha256']
               and value['readerRestored'] is True and value['onlyDeclaredReaderReplaced'] is True
               and value['numericalBindingsUnchanged'] is True
               and value['evidenceBindingsReplaced'] is False and value['hashFunctionsReplaced'] is False
               and (stage == 'report' or value['readCount'] > 0)
               and set(numerical_bindings()).issubset(value['guardedBindings'])
               and value['outputs'] == outputs, 'Manifest-view execution companion differs')
    for reference in outputs.values(): checked(reference)
    return value
