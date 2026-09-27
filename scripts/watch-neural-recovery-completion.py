#!/usr/bin/env python3
"""Read-only completion watch plus prospectively fixed regenerated-byte audit.

No tensor arrays, intermediate quality values or partial result JSONs are read.
Final report contents are opened only after the resumed execution says completed.
This watcher never launches fitting or final auditors itself.
"""
from analysis.private_ledger import private_value
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(private_value('private-reference-0084'))
CONTRACT = '2f6b6bcc4dd423dfe2dc0365180f1451888b31bc2cec3e1b8f0d8dd58e630467'
EXECUTION = 'execution-1789825795658602291.json'
PLAN_SHA = 'bdf835bfb186561df552ccb18e0f889a273f735f8bad3fbdef6496fb4f9ad7e2'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {'path': str(path), 'sha256': digest(path), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    study, recovery = ROOT/'study', ROOT/'runtime-recovery-v2'
    execution = study/EXECUTION
    plan_path = recovery/'regeneration-parity-plan.json'
    assert digest(plan_path) == PLAN_SHA
    plan = read(plan_path)
    assert plan['contractSha256'] == CONTRACT and plan['beforeAnyRegeneratedFitCompleted'] is True
    assert len(plan['files']) == plan['expectedFiles'] == 12
    completions = sorted({Path(f['fitCompletedPath']) for f in plan['files']})
    assert len(completions) == plan['expectedDirectories'] == 3
    for row in plan['files']:
        assert Path(row['regeneratedPath']).resolve().is_relative_to((study/'fits').resolve())
        assert Path(row['quarantined']['path']).resolve().is_relative_to((recovery/'quarantined-fits').resolve())
    source = Path(__file__)
    source_copy = recovery/'regeneration-watch-source.py'
    with source_copy.open('xb') as stream:
        stream.write(source.read_bytes())
    write_new(recovery/'completion-watch.json', {'kind': 'resumed-neural-completion-watch-v2',
              'startedAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': CONTRACT,
              'executionPath': str(execution), 'expectedWorkers': 2, 'source': identity(source_copy),
              'byteParityPlan': identity(plan_path), 'pollSeconds': 30,
              'finalAuditScheduling': 'Separate coordinator; summary, tensor, interval sequentially after successful completed execution/report54.'})
    checked, last_notice = False, 0.
    while True:
        try:
            status = read(execution)
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        assert status['contractSha256'] == CONTRACT and status['workers'] == 2
        if status['status'] == 'failed':
            print(json.dumps({'completionWatcher': 'failed', 'execution': str(execution),
                              'errorType': status.get('errorType'), 'error': status.get('error')}), flush=True)
            raise SystemExit(2)
        regenerated = sum(p.is_file() for p in completions)
        if not checked and regenerated == 3:
            rows = []
            for row in plan['files']:
                old, new = Path(row['quarantined']['path']), Path(row['regeneratedPath'])
                old_identity, new_identity = identity(old), identity(new)
                assert old_identity == row['quarantined'], 'Preserved bytes changed'
                rows.append({'quarantined': old_identity, 'regenerated': new_identity,
                             'byteIdentical': (old_identity['sha256'], old_identity['sizeBytes']) ==
                                              (new_identity['sha256'], new_identity['sizeBytes'])})
            for path in completions:
                assert read(path)['contractSha256'] == CONTRACT
            result = {'kind': 'neural-resource-regenerated-checkpoint-byte-parity-v2',
                      'passed': all(row['byteIdentical'] for row in rows),
                      'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': CONTRACT,
                      'plan': identity(plan_path), 'watchSource': identity(source_copy),
                      'executionPath': str(execution), 'regeneratedCompletedFits': [identity(p) for p in completions],
                      'fitDirectories': 3, 'comparedNPZFiles': 12, 'files': rows,
                      'noArraysOrQualityValuesRead': True, 'comparison': 'Exact NumPy NPZ ZIP file bytes via size and SHA256.'}
            output = recovery/'regeneration-parity-result.json'
            write_new(output, result)
            print(json.dumps({'regenerationParityPassed': result['passed'], 'files': 12,
                              'artifact': identity(output)}), flush=True)
            if not result['passed']:
                raise SystemExit(3)
            checked = True
        if status['status'] == 'completed' and (study/'report.json').exists():
            report_path = study/'report.json'
            report, registration = read(report_path), read(study/'preregistration.json')
            c = registration['contract']
            expected = {(co, kind, arm, seed) for co in c['cohorts'] for kind in c['kinds']
                        for arm in c['lossArms'] for seed in c['seeds']}
            actual = {(r['cohort'], r['kind'], r['lossArm'], r['seed']) for r in report['results']}
            assert checked and report['status'] == 'completed-short-boost-transfer-development'
            assert report['contractSha256'] == registration['sha256'] == CONTRACT
            assert len(report['results']) == len(actual) == len(expected) == 54 and actual == expected
            assert status['reportSha256'] == digest(report_path)
            print(json.dumps({'completionWatcher': 'ready', 'execution': identity(execution),
                              'report': identity(report_path), 'cells': 54,
                              'at': datetime.now(timezone.utc).isoformat()}), flush=True)
            return
        if time.monotonic()-last_notice >= 300:
            print(json.dumps({'completionWatcher': 'waiting', 'executionStatus': status['status'],
                              'reportExists': (study/'report.json').exists(), 'regeneratedCompletedFits': regenerated,
                              'at': datetime.now(timezone.utc).isoformat()}), flush=True)
            last_notice = time.monotonic()
        time.sleep(30)


if __name__ == '__main__':
    main()
