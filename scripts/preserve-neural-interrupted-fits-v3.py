#!/usr/bin/env python3
"""Preserve exactly the two empty v3 interrupted fit leaves, then verify integrity."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v2_preservation_primitives', REPO/'scripts/preserve-neural-interrupted-fits.py')
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
require, identity, digest, read, write_new, canonical, bounded = (
    getattr(previous, name) for name in ('require', 'identity', 'digest', 'read', 'write_new', 'canonical', 'bounded'))
ROOT, CONTRACT = previous.ROOT, previous.CONTRACT
PARTIALS = ('reviewed_export/dino_tcn/baseline/1729/outer-3/refit',
            'reviewed_export/dino_tcn/baseline/20260918/outer-0/inner-0')


def main():
    root = ROOT.resolve()
    study, fits, recovery = root/'study', root/'study/fits', root/'runtime-recovery-v3'
    require(not recovery.exists(), 'Refuse existing recovery leaf')
    reg = read(study/'preregistration.json')
    require(reg['sha256'] == CONTRACT == canonical(reg['contract']), 'Contract changed')
    require(not (study/'report.json').exists(), 'Unexpected completed report')
    for name, sha in reg['contract']['code'].items():
        require(Path(name).name == name and digest(REPO/'analysis'/name) == sha, 'Frozen source changed')
    require(digest(Path(previous.__file__)) == digest(root/'runtime-recovery-v2/recovery-source.py'), 'V2 helper changed')
    leaves = {p for p in fits.rglob('*') if p.is_dir() and p.name in ('refit', 'inner-0', 'inner-1', 'inner-2')}
    completed = sorted(p/'completed.json' for p in leaves if (p/'completed.json').is_file())
    partial = {p.relative_to(fits).as_posix() for p in leaves if not (p/'completed.json').exists()}
    require(len(completed) == 655 and len(leaves) == 657 and partial == set(PARTIALS), 'Stopped inventory changed')
    completed_before = [identity(p) for p in completed]
    results = sorted(study.glob('result-*.json'))
    require(len(results) == 46, 'Result count changed')
    results_before = [identity(p) for p in results]
    execution = study/'execution-1789825795658602291.json'
    execution_before = identity(execution)
    require(read(execution)['status'] == 'running', 'Interrupted execution record changed')
    moves = []
    for relative in PARTIALS:
        source, destination = bounded(fits/relative, root), bounded(recovery/'quarantined-fits'/relative, root)
        require(source.is_relative_to(fits.resolve()) and destination.is_relative_to((recovery/'quarantined-fits').resolve()), 'Move escapes allowed trees')
        require(source.is_dir() and not list(source.iterdir()) and not destination.exists(), 'Expected empty source/new destination')
        moves.append({'relativeToStudyFits': relative, 'source': str(source), 'destination': str(destination), 'files': []})
    observation = REPO/'data/reports/neural-wsl-recovery-v3/observation.json'
    plan = {'kind': 'neural-interrupted-fit-preservation-v3', 'createdAt': datetime.now(timezone.utc).isoformat(),
            'contractSha256': CONTRACT, 'script': identity(Path(__file__)), 'primitives': identity(Path(previous.__file__)),
            'observation': identity(observation), 'interruptedExecution': execution_before,
            'completedFitsBefore': completed_before, 'completedMetadataSetSha256': canonical(completed_before),
            'resultFilesBefore': results_before, 'resultFileSetSha256': canonical(results_before), 'moves': moves,
            'intendedResume': {'trainingWorkers': 1, 'coordinator': 'stdlib-only sequential frozen --job CLI',
                               'finalizationWorkers': 1, 'sameContractSha256': CONTRACT, 'fitRecipeChanged': False},
            'v2EvidenceUnchanged': identity(root/'runtime-recovery-v2/recovery-verification.json'),
            'originalByteParityPlanUnchanged': identity(root/'runtime-recovery-v2/regeneration-parity-plan.json')}
    recovery.mkdir(exist_ok=False)
    write_new(recovery/'plan.json', plan)
    for source, name in ((Path(__file__), 'recovery-source.py'), (Path(previous.__file__), 'recovery-primitives-source.py'),
                         (observation, 'observation.json')):
        with (recovery/name).open('xb') as stream:
            stream.write(source.read_bytes())
        require(digest(recovery/name) == digest(source), 'Source copy differs')
    try:
        audits, total = [], 0
        for index, before in enumerate(completed_before, 1):
            path = Path(before['path'])
            require(identity(path) == before, 'Completed metadata changed')
            meta = read(path)
            require(meta['contractSha256'] == CONTRACT, 'Completed contract differs')
            names = {f'{stem}-{ep}.npz' for ep in meta['epochs'] for stem in ('weights', 'predictions')}
            require(set(meta['artifacts']) == names and {p.name for p in path.parent.iterdir()} == names | {'completed.json'}, 'Artifact inventory differs')
            artifacts = []
            for name, sha in sorted(meta['artifacts'].items()):
                row = identity(path.parent/name)
                require(row['sha256'] == sha, 'Completed artifact hash differs: '+row['path'])
                artifacts.append(row)
            total += len(artifacts)
            audits.append({'completed': before, 'artifacts': artifacts})
            if index % 100 == 0 or index == 655:
                print(json.dumps({'completedFitsVerified': index, 'artifactsVerified': total}), flush=True)
        write_new(recovery/'completed-artifact-verification.json', {'kind': 'completed-artifact-byte-audit-v3', 'passed': True,
                  'contractSha256': CONTRACT, 'completedFits': 655, 'artifactCount': total,
                  'completedMetadataSetSha256': canonical(completed_before), 'fits': audits, 'qualityValuesRead': False})
        for move in moves:
            source, destination = bounded(Path(move['source']), root), bounded(Path(move['destination']), root)
            require(not list(source.iterdir()) and not destination.exists(), 'Move became unsafe')
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            require(not source.exists() and destination.is_dir() and not list(destination.iterdir()), 'Empty directory preservation differs')
        require(sorted(fits.rglob('completed.json')) == completed and [identity(p) for p in completed] == completed_before, 'Completed fits changed')
        require(sorted(study.glob('result-*.json')) == results and [identity(p) for p in results] == results_before, 'Results changed')
        require(identity(execution) == execution_before, 'Interrupted execution was changed')
        for name in ('v2EvidenceUnchanged', 'originalByteParityPlanUnchanged'):
            require(identity(Path(plan[name]['path'])) == plan[name], 'V2 evidence changed')
        result = {'kind': 'neural-resource-recovery-quarantine-v3', 'passed': True, 'readyToResume': True,
                  'completedAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': CONTRACT,
                  'plan': identity(recovery/'plan.json'), 'source': identity(recovery/'recovery-source.py'),
                  'completedArtifactVerification': identity(recovery/'completed-artifact-verification.json'),
                  'completedFitsUnchanged': 655, 'completedArtifactsVerified': total, 'resultCellsUnchanged': 46,
                  'completedMetadataSetSha256': canonical(completed_before), 'resultFileSetSha256': canonical(results_before),
                  'interruptedExecutionUnchanged': execution_before, 'quarantinedFitDirectories': 2, 'quarantinedFiles': 0,
                  'moves': moves, 'copiedObservation': identity(recovery/'observation.json'), 'intendedResume': plan['intendedResume'],
                  'v2EvidenceUnchanged': plan['v2EvidenceUnchanged'], 'originalByteParityPlanUnchanged': plan['originalByteParityPlanUnchanged']}
        write_new(recovery/'recovery-verification.json', result)
        print(json.dumps({'readyToResume': True, 'completedFits': 655, 'artifactsVerified': total, 'report': identity(recovery/'recovery-verification.json')}, indent=2), flush=True)
    except BaseException as error:
        write_new(recovery/'failure.json', {'errorType': type(error).__name__, 'error': str(error), 'plan': identity(recovery/'plan.json')})
        raise


if __name__ == '__main__':
    main()
