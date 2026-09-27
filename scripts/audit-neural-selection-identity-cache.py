#!/usr/bin/env python3
"""Cold-hash touched files and named numerical evidence, without historical lineage expansion."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BINDINGS = ['analysis.neural_recall_sweep.identity', 'analysis.neural_recall_sweep_adapters.identity',
            'analysis.neural_generalization_inputs.sha']
POLICY = {'kind': 'explicit-process-local-immutable-identity-cache-v1', 'bindings': BINDINGS,
    'keyFields': ['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'], 'coldFirstRead': True,
    'prePostStatRaceCheck': True, 'changedMetadataFullRehashThenFail': True,
    'clearExistingVerifiedCacheBetweenStages': True, 'allNumericalFunctionsUnchanged': True,
    'stageExecutionSidecarsRequired': True, 'savedSelectionBytesUnchanged': True,
    'independentColdClosureBeforeGlobalFreezeAndPublication': True,
    'laterPublicationPreflight': 'rehash gate and source plan; exact full cold-closure stat signatures',
    'finalNumericalResultAuditUnchanged': True, 'persistentCacheAllowed': False,
    'coldClosurePolicy': 'actual-touched-plus-named-numerical-evidence-v1',
    'arbitraryReferencedJsonTraversal': False}
STAGES = ('select', 'ordinary', 'correction')
FILENAMES = ('selection.json', 'selection-audit.json', 'selection-correction-audit.json')
REFERENCE_FIELDS = {
    'frozen-generalization-selection-v1': ('task', 'evidence', 'scoreShards', 'code', 'candidateMetricCorrection'),
    'independent-generalization-selection-audit-v1': ('task', 'selection', 'auditor', 'references'),
    'independent-generalization-selection-correction-audit-v1': ('selection', 'selectionAudit', 'plan', 'task', 'auditor', 'evidence'),
    'independent-generalization-fit-numerical-audit-v1': ('task', 'fitResult', 'completed', 'initialStudentRegistration', 'student', 'evidence', 'auditor'),
    'independent-selection-duration-view-audit-v1': ('plan', 'selection', 'task', 'proof', 'baseAudit', 'historicalProtocol', 'source', 'records', 'durationProvenance', 'auditor', 'evidence'),
    'independent-selection-serialization-audit-v1': ('plan', 'selection', 'baseAudit', 'proof', 'task', 'auditor', 'evidence'),
    'independent-historical-all-outer-epochs-audit-v1': ('protocol', 'plan', 'auditor', 'references', 'checks')}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def signature(path):
    stat = Path(path).stat()
    return {'dev': stat.st_dev, 'ino': stat.st_ino, 'size': stat.st_size,
            'mtimeNs': stat.st_mtime_ns, 'ctimeNs': stat.st_ctime_ns}


def resolve_reference(value):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    resolved = (REPO / path).resolve()
    require(resolved.is_relative_to(REPO), 'Relative identity escapes the registered repository')
    return resolved


def cold_hash(path):
    before = signature(path); digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = signature(path)
    require(before == after, 'File changed during independent cold hash: ' + str(path))
    return digest.hexdigest(), after


def references(tree):
    if isinstance(tree, dict):
        if isinstance(tree.get('path'), str) and isinstance(tree.get('sha256'), str):
            yield tree
        for item in tree.values():
            yield from references(item)
    elif isinstance(tree, list):
        for item in tree:
            yield from references(item)


class ColdEvidence:
    """Every distinct file is physically rehashed once in this new audit process."""
    def __init__(self):
        self.checked = {}
        self.declared_documents = {}
        self.total_bytes = 0
        self.bytes_by_suffix = {}

    def bind(self, ref, expected_stat=None):
        path = resolve_reference(ref['path']); resolved = str(path)
        require(isinstance(ref['sha256'], str)
                and len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'),
                'Malformed absolute file identity')
        if resolved not in self.checked:
            digest, stat = cold_hash(path)
            self.checked[resolved] = {'path': resolved, 'sha256': digest, 'stat': stat}
            self.total_bytes += stat['size']
            suffix = path.suffix.lower()
            self.bytes_by_suffix[suffix] = self.bytes_by_suffix.get(suffix, 0) + stat['size']
        item = self.checked[resolved]
        require(signature(path) == item['stat'] and item['sha256'] == ref['sha256'],
                'Cold identity mismatch or file changed: ' + resolved)
        if expected_stat is not None:
            require(item['stat'] == expected_stat, 'Cached immutable metadata changed: ' + resolved)
        return path

    def capture(self, path):
        supplied = str(Path(path)); path = resolve_reference(supplied); key = str(path)
        if key not in self.checked:
            digest, stat = cold_hash(path)
            self.checked[key] = {'path': key, 'sha256': digest, 'stat': stat}
            self.total_bytes += stat['size']
            suffix = path.suffix.lower()
            self.bytes_by_suffix[suffix] = self.bytes_by_suffix.get(suffix, 0) + stat['size']
        row = self.checked[key]
        require(signature(path) == row['stat'], 'Captured file changed: ' + key)
        return {'path': supplied, 'sha256': row['sha256']}

    def document(self, ref):
        path = self.bind(ref)
        value = json.loads(path.read_text(encoding='utf-8'))
        self.bind(ref)
        return value

    def closure(self, tree):
        # These are explicit identity trees supplied by the registered callers.
        # Hash their referenced documents as bytes, but never recursively open
        # arbitrary historical manifests, reports or comparison-source JSON.
        for ref in references(tree):
            self.bind(ref)

    def finish(self):
        for path, item in self.checked.items():
            require(signature(path) == item['stat'], 'File changed before final cold gate: ' + path)


def bind_declared_document(evidence, ref, expected_kind):
    require(expected_kind in REFERENCE_FIELDS, 'Unsupported numerical evidence schema')
    value = evidence.document(ref)
    require(value.get('kind') == expected_kind, 'Numerical document schema differs')
    if expected_kind != 'frozen-generalization-selection-v1':
        require(value.get('passed') is True, 'Required numerical audit did not pass')
    fields = REFERENCE_FIELDS[expected_kind]
    require(all(field in value for field in fields), 'Missing required numerical evidence field')
    key = str(resolve_reference(ref['path']))
    declaration = {'document': ref, 'kind': expected_kind, 'fields': list(fields)}
    if key in evidence.declared_documents:
        require(evidence.declared_documents[key] == declaration, 'Conflicting evidence declaration')
        return value
    evidence.declared_documents[key] = declaration
    for field in fields:
        evidence.closure(value[field])
    if expected_kind == 'independent-generalization-selection-audit-v1' and 'executionAdapter' in value:
        adapter = value['executionAdapter']
        require('executionAuditor' in value and 'baseAudit' in adapter and 'proof' in adapter,
                'Undeclared numerical audit execution')
        evidence.closure({'executionAuditor': value['executionAuditor'], 'executionAdapter': adapter})
        if 'durationAudit' in adapter:
            bind_declared_document(evidence, adapter['durationAudit'], 'independent-selection-duration-view-audit-v1')
        elif 'serializationAudit' in adapter:
            bind_declared_document(evidence, adapter['serializationAudit'], 'independent-selection-serialization-audit-v1')
        else:
            raise ValueError('Unsupported numerical audit execution schema')
        bind_declared_document(evidence, adapter['baseAudit'], 'independent-generalization-selection-audit-v1')
    return value


def inventory_union(collections):
    result = {}
    for rows in collections:
        require(isinstance(rows, list) and rows and [r['path'] for r in rows] == sorted({r['path'] for r in rows}),
                'Stage file inventory is empty, duplicated or unsorted')
        for row in rows:
            require(set(row) == {'path', 'sha256', 'stat'} and str(Path(row['path']).resolve()) == row['path']
                    and set(row['stat']) == {'dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'}
                    and all(type(v) is int and v >= 0 for v in row['stat'].values()),
                    'Malformed cached file or immutable stat inventory')
            previous = result.setdefault(row['path'], row)
            require(canonical(previous) == canonical(row), 'Repeated cache identity/stat differs')
    return [result[key] for key in sorted(result)]


def validate_stage_inventory(plan, inventory):
    expected = [(task, stage) for task in plan['tasks'] for stage in task['stages']]
    require(len(plan['tasks']) == 18 and len(expected) == len(inventory['stages']) == 54,
            'Full original18 /54-stage inventory required')
    preexisting = 0
    for index, ((task, stage), item) in enumerate(zip(expected, inventory['stages'], strict=True)):
        require(item['task'] == task['task'] and item['stage'] == stage['stage']
                and item['output']['path'] == stage['output']
                and item['preexistingOutput'] == stage['preexistingOutput'],
                'Stage task/order/output differs from registration')
        if index < 3:
            require(stage['preexistingOutput'] is not None and item['execution'] is None
                    and item['output'] == stage['preexistingOutput'], 'First three immutable control stages required')
            preexisting += 1
        else:
            require(stage['preexistingOutput'] is None and item['execution'] is not None,
                    'Only the first task may have preexisting outputs')
    require(preexisting == 3, 'Preexisting stage count differs')


def validate_final_union(declared, collections):
    expected = inventory_union(collections)
    require(canonical(declared) == canonical(expected), 'Final file union is narrowed, expanded or altered')
    return expected


def validate_execution(execution, plan_ref, task_ref, stage, output_ref):
    require(execution['kind'] == 'explicit-selection-identity-cache-stage-v1' and execution['passed'] is True
            and execution['plan'] == plan_ref and execution['task'] == task_ref and execution['stage'] == stage
            and execution['output'] == output_ref and execution['bindings'] == BINDINGS
            and execution['functionBindingsRestored'] is True and execution['numericalFunctionsUnchanged'] is True
            and plan_ref in execution['inputs'] and task_ref in execution['inputs'],
            'Cache stage identity/bindings/numerical scope differs')
    files = inventory_union([execution['files']])
    stats = execution['statistics']
    require(set(stats) == {'coldHashCount', 'cacheHitCount', 'coldBytes', 'reusedBytes'}
            and all(type(v) is int and v >= 0 for v in stats.values())
            and stats['coldHashCount'] <= len(files) <= stats['coldHashCount'] + stats['cacheHitCount']
            and stats['coldBytes'] <= sum(row['stat']['size'] for row in files)
            and stats['coldBytes'] + stats['reusedBytes'] >= sum(row['stat']['size'] for row in files)
            and (stats['coldHashCount'] != 0 or stats['coldBytes'] == 0)
            and (stats['cacheHitCount'] != 0 or stats['reusedBytes'] == 0), 'Impossible cache statistics')
    require(isinstance(execution['startedAtUTC'], str) and isinstance(execution['finishedAtUTC'], str)
            and execution['startedAtUTC'] <= execution['finishedAtUTC'], 'Invalid stage time order')
    return files


def audit(plan_path, inventory_path, output):
    output = Path(output)
    require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(), 'New NAS cold audit required')
    evidence = ColdEvidence()
    plan_ref = evidence.capture(plan_path); plan = evidence.document(plan_ref)
    require(plan['kind'] == 'generalization-selection-identity-cache-amendment-v1'
            and canonical(plan['policy']) == canonical(POLICY), 'Registered hash-cache policy differs')
    evidence.closure(plan)
    require(plan['code']['scripts/audit-neural-selection-identity-cache.py'] == evidence.capture(__file__),
            'Independent cold auditor source is not registered')
    registration = evidence.document(plan['registration'])
    duration = evidence.document(plan['durationPlan'])
    require(duration['kind'] == 'generalization-selection-duration-comparison-amendment-v1'
            and duration['correctionPlan'] == plan['correctionPlan']
            and duration['historicalProtocol'] == plan['historicalProtocol'], 'Numerical amendment lineage differs')
    # Plans/registration are structured dependency declarations. Their direct
    # identities are required; arbitrary referenced metadata is not expanded.
    evidence.closure(registration)
    evidence.closure(duration)
    for ref in (duration['serializationPlan'], plan['correctionPlan']):
        evidence.closure(evidence.document(ref))
    historical = Path(plan['historicalProtocol']['path']).parent
    historical_refits = bind_declared_document(evidence, evidence.capture(historical/'audit-refits.json'),
        'independent-historical-all-outer-epochs-audit-v1')
    require(historical_refits['protocol'] == plan['historicalProtocol']
            and historical_refits['counts']['outerFits'] == 72
            and historical_refits['auditor'] == plan['code']['scripts/audit-neural-historical-refits.py'],
            'Historical numerical prerequisite scope differs')
    require([task['taskId'] for task in plan['tasks']] == registration['contract']['taskIds'],
            'Original task inventory/order differs from registration')
    model_seeds = set()
    directory = Path(plan['registration']['path']).parent
    for task_plan in plan['tasks']:
        task = evidence.document(task_plan['task'])
        expected_path = directory/'tasks'/(task['taskId'].replace('/', '__')+'.json')
        expected_fit = directory/'fits'/task['model']/f'seed-{task["seed"]}'
        require(task['variant'] == 'original-corpus' and task['taskId'] == task_plan['taskId']
                and task['registration'] == plan['registration'] and Path(task_plan['task']['path']) == expected_path
                and Path(task_plan['fitDirectory']) == expected_fit
                and [stage['stage'] for stage in task_plan['stages']] == list(STAGES)
                and [stage['output'] for stage in task_plan['stages']] == [str(expected_fit/name) for name in FILENAMES],
                'Original task/source/output ownership differs')
        model_seeds.add((task['model'], task['seed']))
        fit_gate = bind_declared_document(evidence, evidence.capture(expected_fit/'fit-numerical-audit.json'),
            'independent-generalization-fit-numerical-audit-v1')
        require(fit_gate['task'] == task_plan['task']
                and fit_gate['fitResult'] == evidence.capture(expected_fit/'fit-result.json')
                and fit_gate['checkpointEpochs'] == [5, 15, 30, 60]
                and fit_gate['auditor'] == plan['code']['scripts/audit-neural-generalization-numerics.py'],
                'Original fit numerical gate ownership differs')
    require(model_seeds == {(model, seed) for model in ('av-tcn', 'dino-tcn', 'mobile-tcn',
            'distilled-mobile-tcn', 'av-transformer', 'dino-transformer') for seed in (3407, 1729, 20260918)},
            'Original six-model/three-seed scope differs')
    inventory_ref = evidence.capture(inventory_path); inventory = evidence.document(inventory_ref)
    require(inventory['kind'] == 'selection-identity-cache-run-inventory-v1' and inventory['plan'] == plan_ref
            and inventory['source'] == plan['code']['scripts/generalization-selection-cache-execution.py'],
            'Final cached-run inventory lineage differs')
    validate_stage_inventory(plan, inventory)
    qualification_ref = inventory['qualification']; qualification = evidence.document(qualification_ref)
    q = plan['qualification']; first = plan['tasks'][0]
    require(q['task'] == first['task'] and q['referenceSelection'] == first['stages'][0]['preexistingOutput']
            and qualification['kind'] == 'selection-identity-cache-qualification-v1' and qualification['passed'] is True
            and qualification['plan'] == plan_ref and qualification['referenceSelection'] == q['referenceSelection']
            and qualification['selection']['path'] == q['output'] and qualification['byteExactSelectionParity'] is True,
            'Qualification control/task/output differs')
    reference = evidence.bind(qualification['referenceSelection']); selected = evidence.bind(qualification['selection'])
    require(reference.read_bytes() == selected.read_bytes(), 'Cached qualification selection bytes differ from cold control')
    qualification_execution = evidence.document(qualification['execution'])
    collections = [validate_execution(qualification_execution, plan_ref, q['task'], 'qualification-select', qualification['selection'])]
    evidence.closure(qualification_execution)
    evidence.closure(qualification)
    bind_declared_document(evidence, qualification['selection'], 'frozen-generalization-selection-v1')
    kinds = dict(zip(STAGES, ('frozen-generalization-selection-v1',
        'independent-generalization-selection-audit-v1', 'independent-generalization-selection-correction-audit-v1'), strict=True))
    for item in inventory['stages']:
        numerical = bind_declared_document(evidence, item['output'], kinds[item['stage']])
        require(numerical['task'] == item['task'], 'Numerical stage belongs to another registered task')
        if item['execution'] is not None:
            execution = evidence.document(item['execution'])
            collections.append(validate_execution(execution, plan_ref, item['task'], item['stage'], item['output']))
            evidence.closure(execution)
        evidence.closure(item)
    expected_files = validate_final_union(inventory['files'], collections)
    for row in expected_files:
        evidence.bind(row, row['stat'])
    evidence.closure(inventory)
    evidence.finish()
    result = {'kind': 'independent-selection-identity-cache-cold-audit-v1', 'passed': True,
        'plan': plan_ref, 'inventory': inventory_ref, 'qualification': qualification_ref,
        'taskCount': 18, 'stageCount': 54, 'cachedStageCount': 51, 'preexistingStageCount': 3,
        'files': expected_files, 'evidence': [{'path': row['path'], 'sha256': row['sha256']}
            for _, row in sorted(evidence.checked.items())],
        'coldFiles': [row for _, row in sorted(evidence.checked.items())],
        'allColdHashesPassed': True, 'allImmutableStatsPassed': True,
        'completeStageAndDeclaredEvidenceClosureVerified': True, 'byteExactQualificationParity': True,
        'closurePolicy': POLICY['coldClosurePolicy'], 'referenceFieldsBySchema': REFERENCE_FIELDS,
        'declaredDocuments': [row for _, row in sorted(evidence.declared_documents.items())],
        'informationalLineageScope': 'Containing JSON bytes are pinned. Arbitrary nested source, label, report, '
            'media and fixture references are not opened; all actual touched files and named numerical evidence are cold-hashed.',
        'coldHashedFileCount': len(evidence.checked), 'coldHashedBytes': evidence.total_bytes,
        'coldHashedBytesBySuffix': evidence.bytes_by_suffix,
        'auditor': evidence.capture(__file__), 'trainingPerformed': False, 'gpuUsed': False,
        'statisticsScope': 'Runtime hit counters checked for consistency only; every distinct file independently cold-hashed here.'}
    evidence.finish()
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'inventory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.plan, args.inventory, args.output)
