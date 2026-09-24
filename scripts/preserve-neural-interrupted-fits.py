#!/usr/bin/env python3
"""One explicitly authorized resource-recovery inventory and quarantine.

Only the three known incomplete directories can move. Completed checkpoints and
result files are read as bytes for integrity verification, never as model data.
No training modules are imported and no registered source is changed.
"""
from analysis.private_ledger import private_value
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0084'))
CONTRACT = '2f6b6bcc4dd423dfe2dc0365180f1451888b31bc2cec3e1b8f0d8dd58e630467'
PARTIALS = {
    'reviewed_export/dino_tcn/baseline/1729/outer-0/inner-1': {
        'predictions-5.npz': (108242, 'b4fd7c3ef7f0df57fb194b19a10f85af3e245457e94d2744b4d210f1f220f042'),
        'weights-5.npz': (181832, '464a0d961389891febea8dc3e976a482dbf7724c7eed51ccfcebfb488084b63e')},
    'reviewed_export/dino_tcn/baseline/20260918/outer-0/inner-0': {
        'predictions-15.npz': (163150, 'ebc3a41c4895ba7bc2aa93bdaf5cd7f2580817b97482fc08ab0e3b19ea9a44bc'),
        'predictions-5.npz': (158796, 'e293443bbc97703ea0f02934fdc2575c97cfa1791f75b5f659242f403cf07727'),
        'weights-15.npz': (182184, '5ad638b4c9cb9ce8bdb0d1573d9c35864600af00ff4fcabbe3f0931dda697f19'),
        'weights-5.npz': (181810, '4a90107a20621e2c81da6e94035db613ae173cc6c57e435db058e3adfbfc7c34')},
    'reviewed_export/dino_tcn/baseline/3407/outer-0/inner-1': {
        'predictions-15.npz': (111064, 'e20855476aa293b89fa4874024f1a745e067716cbd4a539aa2236f707edb003c'),
        'predictions-30.npz': (110199, '359e068f710010f1d061a6bba546dd505f43b204bbb7f30a009c44639ba7f13d'),
        'predictions-5.npz': (107773, '164f358f6149a6bfa81469eec8234007ab9ed79cb70f5d733c01603dfb21ef76'),
        'weights-15.npz': (182201, '1512f7410d91ffc7c05f9acc64b3dd36873b8bbbc6c747eedc71c6ab9b2488a2'),
        'weights-30.npz': (182572, 'ed1a7f32d792fb7596cbfc8e5d9a1c67ba0a82e8a5a53a1391dea5ccfe27e191'),
        'weights-5.npz': (181825, 'fff5b0d9d0f59c59b4223e36a861d619114f2fcd12ac7819ea1c080252fadab9')},
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(path):
    require(path.is_file() and not path.is_symlink(), f'Not a regular file: {path}')
    return {'path': str(path), 'sizeBytes': path.stat().st_size, 'sha256': digest(path)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bounded(path, root):
    result = path.resolve()
    require(result.is_relative_to(root.resolve()) and result != root.resolve(), f'Path escapes named root: {path}')
    require(not path.is_symlink(), f'Symlink forbidden: {path}')
    return result


def partial_files(folder, expected):
    require(folder.is_dir() and not folder.is_symlink(), f'Missing original partial: {folder}')
    files = sorted(folder.iterdir())
    require({p.name for p in files} == set(expected), f'Partial inventory changed: {folder}')
    rows = []
    for path in files:
        row = identity(path)
        require((row['sizeBytes'], row['sha256']) == expected[path.name], f'Partial bytes changed: {path}')
        rows.append(row)
    return rows


def main():
    root = ROOT.resolve()
    study, fits, recovery = root/'study', root/'study/fits', root/'runtime-recovery-v2'
    require(not recovery.exists(), 'Refuse an existing recovery leaf')
    registration = read(study/'preregistration.json')
    require(registration['sha256'] == CONTRACT == canonical(registration['contract']), 'Registered contract changed')
    require(not (study/'report.json').exists(), 'Interrupted study unexpectedly has a final report')
    for name, sha in registration['contract']['code'].items():
        require(Path(name).name == name and digest(REPO/'analysis'/name) == sha, 'Frozen source changed')
    leaves = {p for p in fits.rglob('*') if p.is_dir() and (p.name == 'refit' or p.name in ('inner-0', 'inner-1', 'inner-2'))}
    complete = sorted(p/'completed.json' for p in leaves if (p/'completed.json').is_file())
    partial = {p.relative_to(fits).as_posix() for p in leaves if not (p/'completed.json').exists()}
    require(len(complete) == 626 and partial == set(PARTIALS) and len(leaves) == 629, 'Stopped fit inventory changed')
    completed_before = [identity(p) for p in complete]
    result_files = sorted(study.glob('result-*.json'))
    require(len(result_files) == 45, 'Stopped result count changed')
    results_before = [identity(p) for p in result_files]
    original_execution = study/'execution-1789816744084500647.json'
    execution_before = identity(original_execution)
    require(read(original_execution)['status'] == 'running', 'Original execution record changed')
    moves = []
    for relative, expected in sorted(PARTIALS.items()):
        source = bounded(fits/relative, root)
        destination = bounded(recovery/'quarantined-fits'/relative, root)
        require(source.is_relative_to(fits.resolve()) and destination.is_relative_to((recovery/'quarantined-fits').resolve()), 'Move scope differs')
        require(not destination.exists(), 'Quarantine destination exists')
        moves.append({'relativeToStudyFits': relative, 'source': str(source), 'destination': str(destination),
                      'files': partial_files(source, expected)})
    copied = []
    for name in ('observation.json', 'temporary-memory-cap.json'):
        source = REPO/'data/reports/neural-wsl-recovery-v2'/name
        copied.append({'source': identity(source), 'destination': str(recovery/name)})
    plan = {'kind': 'neural-interrupted-fit-preservation-v2', 'createdAt': datetime.now(timezone.utc).isoformat(),
            'contractSha256': CONTRACT, 'script': identity(Path(__file__)), 'originalExecution': execution_before,
            'interruption': {'method': 'Ubuntu termination authorized by root after host-memory stall',
                             'terminatedAt': '2026-09-19T13:43:19Z', 'originalRecordWillRemainUnchanged': True},
            'scope': 'Exactly three incomplete leaves; no model-data decoding or quality inspection.',
            'completedFitsBefore': completed_before, 'completedMetadataSetSha256': canonical(completed_before),
            'resultFilesBefore': results_before, 'resultFileSetSha256': canonical(results_before),
            'moves': moves, 'copiedObservations': copied,
            'intendedResume': {'workers': 2, 'sameContractSha256': CONTRACT, 'fitRecipeChanged': False,
                               'incompleteFitsRestartFromEpochZero': True, 'compareSavedPartialCheckpointsAfterRegeneration': True}}
    recovery.mkdir(parents=False, exist_ok=False)
    write_new(recovery/'plan.json', plan)
    with (recovery/'recovery-source.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    for item in copied:
        with Path(item['destination']).open('xb') as stream:
            stream.write(Path(item['source']['path']).read_bytes())
        require(digest(Path(item['destination'])) == item['source']['sha256'], 'Copied observation differs')
    audited, artifacts_count, artifact_bytes = [], 0, 0
    try:
        for index, info in enumerate(completed_before, 1):
            path = Path(info['path'])
            require(identity(path) == info, 'Completed metadata changed during inventory')
            meta = read(path)
            require(meta['contractSha256'] == CONTRACT, 'Completed fit has different contract')
            expected = {f'{stem}-{epoch}.npz' for epoch in meta['epochs'] for stem in ('weights', 'predictions')}
            require(set(meta['artifacts']) == expected, 'Completed checkpoint inventory differs')
            require({p.name for p in path.parent.iterdir()} == expected | {'completed.json'}, 'Unexpected completed fit files')
            rows = []
            for name, sha in sorted(meta['artifacts'].items()):
                require(Path(name).name == name and name.endswith('.npz'), 'Invalid artifact filename')
                item = identity(path.parent/name)
                require(item['sha256'] == sha, f'Completed checkpoint hash mismatch: {item["path"]}')
                rows.append(item)
                artifact_bytes += item['sizeBytes']
            artifacts_count += len(rows)
            audited.append({'completed': info, 'artifacts': rows})
            if index % 50 == 0 or index == len(completed_before):
                print(json.dumps({'completedFitsVerified': index, 'artifactsVerified': artifacts_count}), flush=True)
        write_new(recovery/'completed-artifact-verification.json', {'kind': 'interrupted-study-completed-artifact-byte-audit-v2',
                  'passed': True, 'contractSha256': CONTRACT, 'completedFits': len(audited),
                  'completedMetadataSetSha256': canonical(completed_before), 'artifactCount': artifacts_count,
                  'artifactBytes': artifact_bytes, 'fits': audited, 'modelOrQualityValuesRead': False})
        for item in moves:
            source = bounded(Path(item['source']), root)
            destination = bounded(Path(item['destination']), root)
            partial_files(source, PARTIALS[item['relativeToStudyFits']])
            require(not destination.exists() and not (source/'completed.json').exists(), 'Move became unsafe')
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            partial_files(destination, PARTIALS[item['relativeToStudyFits']])
            require(not source.exists(), 'Original partial still exists after rename')
        require(sorted(fits.rglob('completed.json')) == complete, 'Completed inventory changed during quarantine')
        require([identity(p) for p in complete] == completed_before, 'Completed metadata changed during quarantine')
        require(sorted(study.glob('result-*.json')) == result_files and [identity(p) for p in result_files] == results_before, 'Result bytes changed during quarantine')
        require(identity(original_execution) == execution_before, 'Original execution record was changed')
        remaining = [p for p in fits.rglob('*') if p.is_dir() and (p.name == 'refit' or p.name in ('inner-0', 'inner-1', 'inner-2')) and not (p/'completed.json').exists()]
        require(not remaining, 'Unexpected incomplete fit remains')
        result = {'kind': 'neural-resource-recovery-quarantine-v2', 'passed': True,
                  'completedAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': CONTRACT,
                  'plan': identity(recovery/'plan.json'), 'source': identity(recovery/'recovery-source.py'),
                  'completedArtifactVerification': identity(recovery/'completed-artifact-verification.json'),
                  'completedFitsUnchanged': 626, 'completedArtifactsVerified': artifacts_count,
                  'resultCellsUnchanged': 45, 'completedMetadataSetSha256': canonical(completed_before),
                  'resultFileSetSha256': canonical(results_before), 'originalExecutionUnchanged': execution_before,
                  'quarantinedFitDirectories': 3, 'quarantinedFiles': 12,
                  'quarantined': [{'relativeToStudyFits': item['relativeToStudyFits'], 'path': item['destination'],
                                  'files': partial_files(Path(item['destination']), PARTIALS[item['relativeToStudyFits']])} for item in moves],
                  'copiedObservations': [identity(Path(item['destination'])) for item in copied],
                  'intendedResume': plan['intendedResume'], 'readyToResume': True}
        write_new(recovery/'recovery-verification.json', result)
        print(json.dumps({'readyToResume': True, 'completedFits': 626, 'artifactsVerified': artifacts_count,
                          'quarantinedDirectories': 3, 'quarantinedFiles': 12,
                          'report': identity(recovery/'recovery-verification.json')}, indent=2), flush=True)
    except BaseException as error:
        write_new(recovery/'failure.json', {'kind': 'neural-resource-recovery-failure-v2', 'contractSha256': CONTRACT,
                  'errorType': type(error).__name__, 'error': str(error), 'plan': identity(recovery/'plan.json')})
        raise


if __name__ == '__main__':
    main()
