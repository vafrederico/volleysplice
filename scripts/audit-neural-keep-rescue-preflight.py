#!/usr/bin/env python3
"""Qualify no-op parity and synthetic fold isolation; no candidate quality scan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_keep_rescue_development as runner
from analysis import neural_keep_rescue as rescue

TESTS = ('analysis.tests.test_neural_keep_rescue', 'analysis.tests.test_neural_keep_rescue_development')


def reference_binding(root, tensor_audit, interval_audit):
    registration = runner.read(root/'preregistration.json')
    return {'path': str(root), 'contractSha256': registration['sha256'],
            'preregistrationFileSha256': runner.digest(root/'preregistration.json'),
            'reportSha256': runner.digest(root/'report.json'), 'summarySha256': runner.digest(root/'summary.json'),
            'independentAudits': [runner.identity(path) for path in (tensor_audit, interval_audit)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--tensor-audit', type=Path, required=True)
    parser.add_argument('--interval-audit', type=Path, required=True)
    parser.add_argument('--cohorts', nargs='+', choices=runner.COHORTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runner.require(args.cohorts == [c for c in runner.COHORTS if c in args.cohorts], 'Cohorts must be unique and in canonical order')
    reference = reference_binding(args.reference, args.tensor_audit, args.interval_audit)
    reg, report, audited_fits = runner.completed_reference(reference)
    old = reg['contract']
    code = {**old['code'], **{name: runner.digest(REPO/'analysis'/name) for name in runner.NEW_SOURCES}}
    runner.require(all(runner.digest(REPO/'analysis'/name) == sha for name, sha in code.items()), 'Frozen source changed')
    manifest_path = Path(old['dinoManifest']['path']).parent/'manifest-pts-v1.json'
    manifest = runner.read(manifest_path)
    runner.require(runner.digest(manifest_path) == old['manifestSha256'], 'Manifest changed')
    plan = {'kind': 'keep-rescue-engineering-preflight-v1', 'createdAt': datetime.now(timezone.utc).isoformat(),
            'cohorts': args.cohorts, 'code': code, 'manifest': runner.identity(manifest_path),
            'referenceContractSha256': reg['sha256'], 'referenceReport': runner.identity(args.reference/'report.json'),
            'referenceAudits': reference['independentAudits'], 'script': runner.identity(Path(__file__)),
            'executionEnvironment': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'cpu'},
            'syntheticTests': [runner.identity(REPO/Path(*name.split('.')).with_suffix('.py')) for name in TESTS],
            'candidateQualityMetricsInspected': False,
            'procedure': 'Run synthetic no-op/geometry/selection-isolation suites, then decode only None '
                         'on all selected baseline outer probabilities; compare every saved core interval exactly. '
                         'Validate actual nominal timestamp cadence without evaluating any rescue threshold.'}
    args.output.mkdir(parents=True, exist_ok=False)
    runner.write_immutable(args.output/'plan.json', plan)
    command = [sys.executable, '-m', 'unittest', *TESTS, '-v']
    tests = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with (args.output/'tests.log').open('x', encoding='utf-8') as handle:
        handle.write(tests.stdout)
    runner.require(tests.returncode == 0, 'Keep-rescue synthetic qualification failed; see tests.log')
    examples = runner.load_evaluation_examples(manifest)
    cells = [r for r in report['results'] if r['cohort'] in args.cohorts and r['lossArm'] == 'baseline']
    runner.require(len(cells) == len(args.cohorts)*6, 'Baseline preflight population differs')
    checked, record_count = [], 0
    for baseline in cells:
        print('NO-OP', baseline['cohort'], baseline['kind'], baseline['seed'], flush=True)
        root, source_contract = runner.source_fit_root(baseline, old, args.reference)
        saved = {row['id']: row for row in baseline['predictions']}
        cell_checks = []
        for outer_index, outer in enumerate(old['groups']):
            fixed = next(row for row in baseline['selections'] if row['heldSourceGroup'] == outer)
            held = [e for e in examples if e.group == outer]
            scores, evidence = runner.load_bound_predictions(root/f'outer-{outer_index}'/'refit', fixed['epoch'], held,
                expected=runner.expected_membership(manifest, baseline['cohort'], {outer}, outer),
                kind=baseline['kind'], seed=baseline['seed'], contract_hash=source_contract, audited_fits=audited_fits)
            for e in held:
                proposals = rescue.keep_core_proposals(e.times, scores[e.id][:, 3], e.valid, e.duration,
                    threshold=None, smoothing_seconds=fixed['decoder']['smoothing'], ignored_intervals=e.ignored)
                runner.require(proposals == (), 'No-op generated proposals')
                cuts = rescue.decode_with_keep_rescue(e, scores[e.id], fixed['decoder'], keep_threshold=None)
                runner.require([cut.to_dict() for cut in cuts] == saved[e.id]['predictions'], 'No-op differs from saved baseline cuts')
                record_count += 1
            cell_checks.append({'heldSourceGroup': outer, 'baselineSelection': fixed, 'evidence': evidence})
        checked.append({'cohort': baseline['cohort'], 'kind': baseline['kind'], 'seed': baseline['seed'], 'folds': cell_checks})
    runner.require(record_count == len(cells)*8, 'No-op recording count differs')
    for name, sha in code.items():
        runner.require(runner.digest(REPO/'analysis'/name) == sha, 'Source changed during qualification')
    runner.require(runner.digest(manifest_path) == plan['manifest']['sha256'], 'Manifest changed during qualification')
    runner.completed_reference(reference)
    result = {**plan, 'passed': True, 'noOpReplayQualified': True, 'selectionIsolationQualified': True,
              'plan': runner.identity(args.output/'plan.json'), 'syntheticTestLog': runner.identity(args.output/'tests.log'),
              'testExitCode': tests.returncode, 'noOpRecordingsReplayed': record_count,
              'selectedRefitNPZs': len(cells)*4, 'checks': checked}
    runner.write_immutable(args.output/'report.json', result)
    print(json.dumps({'passed': True, 'noOpRecordingsReplayed': record_count,
                      'report': runner.identity(args.output/'report.json')}, indent=2))


if __name__ == '__main__':
    main()
