#!/usr/bin/env python3
"""Read-only preflight for an explicitly authorized encoder scheduling handoff."""
from analysis.private_ledger import private_value
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def audit(folder, guard_path):
    refs = {}

    def bind(ref):
        require(reference(ref['path']) == ref, 'Handoff evidence hash changed')
        refs[ref['path']] = ref
        return Path(ref['path'])

    identity_ref, plan_ref, boundary_ref, result_ref = [reference(folder / name) for name in
        ('process-identity.json', 'plan.json', 'boundary.json', 'result.json')]
    process, plan, boundary, result = [read(bind(ref)) for ref in (identity_ref, plan_ref, boundary_ref, result_ref)]
    guard_ref = reference(guard_path)
    guard = read(bind(guard_ref))
    bind(guard['source']); bind(guard['protocol'])
    require(guard['kind'] == 'historical-student-resource-handoff-v1' and guard['beforeTask'] in (64, 72)
            and guard['numericalRecipeChanged'] is False, 'Unknown historical handoff')
    require(process['kind'] == 'verified-encoder-process-identity-v1' and process['uid'] == os.getuid()
            and process['startTimeTicks'] > 0 and process['clockTicksPerSecond'] == os.sysconf('SC_CLK_TCK'),
            'Encoder process identity is incomplete')
    require(process['bootId'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'Encoder identity came from a different WSL boot')
    require(process['command'][-2:] == ['scripts/run-generalization-gpu-queue.py', 'run']
            and all(process[key] == plan[key] for key in ('pid', 'command', 'queuePlan', 'executionGrant')),
            'Process identity differs from released encoder plan')
    require(result['released'] is True and result['plan'] == plan_ref and result['boundary'] == boundary_ref
            and result['unchangedPerRecordingOutputsReusedOnResume'] is True,
            'Encoder controller did not establish a completed release')
    require(boundary['plan'] == plan_ref and boundary['incompleteGpuRecords'] == [], 'Encoder boundary is incomplete')
    bind(plan['source']); bind(plan['queuePlan']); bind(plan['executionGrant'])
    require(not (Path('/proc') / str(process['pid'])).exists(), 'The released encoder PID still exists')
    feature_root = folder.parents[1]
    completed = []
    for arm in ('fp32', 'fp16', 'mobile'):
        for directory in sorted((feature_root / 'encoders' / arm).glob('*')):
            if not directory.is_dir():
                continue
            receipt = directory / 'receipt.json'
            require(receipt.exists(), 'A GPU record remains incomplete')
            ref = reference(receipt)
            value = read(bind(ref))
            require(value['encoderPlan'] == reference(feature_root / 'encoder-plan.json'), 'Encoder receipt plan differs')
            bind(value['encoderPlan'])
            completed.append(ref)
            require(arm != 'fp32' or (feature_root / 'encoders/mobile' / directory.name / 'receipt.json').exists(),
                    'A completed DINO record lacks its paired Mobile/teacher phase')
    require(completed and completed == boundary['completedReceipts'], 'Live completed GPU-record inventory differs from release boundary')
    for ref in refs.values():
        bind(ref)
    return {'kind': 'independent-encoder-handoff-preflight-v1', 'passed': True,
            'atUtc': datetime.now(timezone.utc).isoformat(), 'guard': guard_ref,
            'encoderProcessIdentity': identity_ref, 'encoderReleasePlan': plan_ref,
            'encoderBoundary': boundary_ref, 'encoderReleaseResult': result_ref,
            'encoderPidAbsent': True, 'sameBootVerified': True,
            'completeGpuRecordReceipts': len(completed), 'incompleteGpuRecords': [],
            'references': list(refs.values()), 'auditor': reference(__file__),
            'processSignalsSent': False, 'numericalArtifactsChanged': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-folder', type=Path, required=True)
    parser.add_argument('--guard', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(str(args.output.resolve()).startswith(private_value('private-reference-0060')), 'Preflight output must use NAS')
    result = audit(args.release_folder, args.guard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': True, 'output': reference(args.output),
                      'completeGpuRecordReceipts': result['completeGpuRecordReceipts']}), flush=True)


if __name__ == '__main__':
    main()
