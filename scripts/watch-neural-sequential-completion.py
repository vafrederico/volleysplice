#!/usr/bin/env python3
"""Watch the v3 sequential driver and finish the unchanged v2 byte-parity plan."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('previous_recovery_watch_primitives', REPO/'scripts/watch-neural-recovery-completion.py')
old = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old)
read, identity, digest, write_new = old.read, old.identity, old.digest, old.write_new
ROOT, CONTRACT = old.ROOT, old.CONTRACT
DRIVER_PLAN_SHA = 'd45c7c99e8b3c9b1cacb54ae0ce1d62130f8232132ba9ab907340a3812f59d2e'


def main():
    recovery, driver = ROOT/'runtime-recovery-v3', ROOT/'runtime-recovery-v3/sequential-driver'
    plan_path = ROOT/'runtime-recovery-v2/regeneration-parity-plan.json'
    assert digest(plan_path) == old.PLAN_SHA and digest(driver/'plan.json') == DRIVER_PLAN_SHA
    assert digest(Path(old.__file__)) == digest(ROOT/'runtime-recovery-v2/regeneration-watch-source.py')
    plan = read(plan_path)
    assert plan['contractSha256'] == CONTRACT and len(plan['files']) == 12
    completions = sorted({Path(r['fitCompletedPath']) for r in plan['files']})
    assert len(completions) == 3
    with (recovery/'completion-watch-source.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    watch_plan = {'kind': 'sequential-neural-completion-watch-v3', 'startedAt': datetime.now(timezone.utc).isoformat(),
                  'contractSha256': CONTRACT, 'originalParityPlan': identity(plan_path),
                  'driverPlan': identity(driver/'plan.json'), 'source': identity(recovery/'completion-watch-source.py'),
                  'primitives': identity(Path(old.__file__)), 'pollSeconds': 30,
                  'completionGate': 'driver-completed.json after frozen finalizer exits0 and successful report54+metadata-preservation checks',
                  'auditOrder': 'separate coordinators: summary, then tensor, then interval',
                  'v2EvidenceWillRemainUnchanged': True}
    write_new(recovery/'completion-watch-plan.json', watch_plan)
    checked, last_notice = False, 0.
    while True:
        if (driver/'driver-failed.json').exists():
            try:
                failure = read(driver/'driver-failed.json')
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            print(json.dumps({'completionWatcher': 'failed', 'driverFailure': identity(driver/'driver-failed.json'),
                              'errorType': failure.get('errorType'), 'error': failure.get('error')}), flush=True)
            raise SystemExit(2)
        regenerated = sum(p.is_file() for p in completions)
        if regenerated == 3 and not checked:
            rows = []
            for row in plan['files']:
                original, current = identity(Path(row['quarantined']['path'])), identity(Path(row['regeneratedPath']))
                assert original == row['quarantined']
                rows.append({'quarantined': original, 'regenerated': current,
                             'byteIdentical': (original['sha256'], original['sizeBytes']) == (current['sha256'], current['sizeBytes'])})
            assert all(read(p)['contractSha256'] == CONTRACT for p in completions)
            result = {'kind': 'neural-resource-regenerated-checkpoint-byte-parity-v3',
                      'passed': all(r['byteIdentical'] for r in rows), 'createdAt': datetime.now(timezone.utc).isoformat(),
                      'contractSha256': CONTRACT, 'originalPlan': identity(plan_path), 'watchPlan': identity(recovery/'completion-watch-plan.json'),
                      'regeneratedCompletedFits': [identity(p) for p in completions], 'fitDirectories': 3,
                      'comparedNPZFiles': 12, 'files': rows, 'noArraysOrQualityValuesRead': True,
                      'comparison': 'Exact NumPy NPZ ZIP bytes by file size and SHA256, unchanged prospective v2 plan.'}
            output = recovery/'regeneration-parity-result.json'
            write_new(output, result)
            print(json.dumps({'regenerationParityPassed': result['passed'], 'files': 12, 'artifact': identity(output)}), flush=True)
            if not result['passed']:
                raise SystemExit(3)
            checked = True
        if (driver/'driver-completed.json').exists():
            try:
                done = read(driver/'driver-completed.json')
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            assert checked and done['status'] == 'completed' and done['contractSha256'] == CONTRACT
            assert done['finalizationPerformedTraining'] is False and done['originalResultFilesUnchanged'] == 46
            assert identity(Path(done['report']['path'])) == done['report']
            assert identity(Path(done['finalExecution']['path'])) == done['finalExecution']
            report, execution = read(done['report']['path']), read(done['finalExecution']['path'])
            assert report['contractSha256'] == execution['contractSha256'] == CONTRACT
            assert report['status'] == 'completed-short-boost-transfer-development'
            assert execution['status'] == 'completed' and execution['workers'] == 1 and execution['reportSha256'] == done['report']['sha256']
            assert len(report['results']) == len({(r['cohort'], r['kind'], r['lossArm'], r['seed']) for r in report['results']}) == 54
            print(json.dumps({'completionWatcher': 'ready', 'driverCompletion': identity(driver/'driver-completed.json'),
                              'report': done['report'], 'execution': done['finalExecution'], 'cells': 54,
                              'at': datetime.now(timezone.utc).isoformat()}), flush=True)
            return
        if time.monotonic()-last_notice >= 300:
            print(json.dumps({'completionWatcher': 'waiting', 'driverStarted': (driver/'driver-started.json').exists(),
                              'reportExists': (ROOT/'study/report.json').exists(), 'regeneratedCompletedFits': regenerated,
                              'at': datetime.now(timezone.utc).isoformat()}), flush=True)
            last_notice = time.monotonic()
        time.sleep(30)


if __name__ == '__main__':
    main()
