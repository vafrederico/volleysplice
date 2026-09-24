#!/usr/bin/env python3
"""Require exact historical95/99 control replay before new-floor interpretation."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as sweep
from analysis.neural_recall_sweep_adapters import NAS, bind


def by_recording(rows):
    return sorted(rows, key=lambda row: row['id'])


def compare_metrics(left, right):
    if isinstance(left, dict):
        sweep.require(isinstance(right, dict) and set(left) == set(right), 'Control metric schema differs')
        for key in left:
            compare_metrics(left[key], right[key])
    elif isinstance(left, list):
        sweep.require(isinstance(right, list) and len(left) == len(right), 'Control metric list differs')
        for key in ('id', 'recordingId'):
            if left and all(isinstance(row, dict) and key in row for row in left):
                left, right = sorted(left, key=lambda row: row[key]), sorted(right, key=lambda row: row[key])
                break
        for a, b in zip(left, right, strict=True):
            compare_metrics(a, b)
    elif isinstance(left, float):
        sweep.require(isinstance(right, (int, float)) and abs(left - right) <= 1e-10, 'Control numeric metric differs')
    else:
        sweep.require(left == right, 'Control metric value differs')


def audit(report_path):
    evidence = {}
    report_ref = sweep.identity(report_path)
    report = sweep.read(bind(report_ref, evidence))
    control_ref = {'path': str(NAS / '2026-09-23-recall-distillation/recall99-v1/report-complete.json'),
                   'sha256': '7443fa8e5409a84c2f1708ebaaef13b0333f0e850403a55608c70d6a5e28237f'}
    student_ref = {'path': str(NAS / '2026-09-23-recall-distillation/distilled-mobile-v1/report.json'),
                   'sha256': '0620fe81eee6a852112b4884b1b2f1c9c449d5fc8589ef5f5d9745dda60ac00a'}
    control = sweep.read(bind(control_ref, evidence))
    students = sweep.read(bind(student_ref, evidence))
    checks = []
    for cell in report['cells']:
        panel = cell['panels']['historical-nested-exact']['result']['floors']
        sweep.require([f['floorPercent'] for f in panel] == list(sweep.FLOORS), 'New sweep floor order differs')
        if cell['model'] == 'distilled_mobile_tcn':
            old = next(r for r in students['results'] if r['seed'] == cell['seed'])
            sweep.require(panel[9]['completeEvaluationScope'] == old['completeEvaluation']
                          and by_recording(panel[9]['predictions']) == by_recording(old['predictions']),
                          'Old distilled99 raw predictions/evaluation changed')
            compare_metrics(panel[9]['evaluation'], old['evaluation'])
            for fold, selected in zip(cell['folds'], old['selections'], strict=True):
                sweep.require(fold['foldId'] == selected['heldSourceGroup']
                              and fold['decisions'][9]['selected'] == selected['selected']
                              and fold['decisions'][9]['feasible'] == selected['feasible'], 'Distilled99 selection changed')
            checks.append({'model': cell['model'], 'seed': cell['seed'], 'old99RawIntervalsExact': True, 'old99AllMetricsMatched': True})
            continue
        old = next(r for r in control['results'] if (r['model'], r['seed']) == (cell['model'], cell['seed']))
        sweep.require(panel[5]['completeEvaluationScope'] and panel[5]['evaluation'] == old['original95']['evaluation'],
                      'Old95 complete-scope metrics changed')
        joint = old['modes']['joint']
        sweep.require(panel[9]['completeEvaluationScope'] == joint['completeEvaluationScope']
                      and by_recording(panel[9]['predictions']) == by_recording(joint['predictions']),
                      'Old99 raw predictions/evaluation changed')
        compare_metrics(panel[9]['evaluation'], joint['evaluation'])
        for fold, prior in zip(cell['folds'], old['folds'], strict=True):
            sweep.require(fold['foldId'] == prior['heldSourceGroup'] and fold['candidates'] == prior['candidates']
                          and fold['decisions'][9]['selected'] == prior['decisions']['joint']['selected'],
                          'Original candidate grid/strict99 decision changed')
        checks.append({'model': cell['model'], 'seed': cell['seed'], 'old95AllMetricsExact': True,
                       'old99RawIntervalsExact': True, 'old99AllMetricsMatched': True, 'originalCandidateMetricsExact': True})
    sweep.require(len(checks) == 18, 'Control replay model/seed scope incomplete')
    for ref in evidence.values():
        bind(ref)
    return {'kind': 'independent-historical-recall-control-replay-v1', 'passed': True,
            'report': report_ref, 'checks': checks, 'references': list(evidence.values()),
            'auditor': sweep.identity(Path(__file__)), 'trainingPerformed': False, 'gpuUsed': False,
            'scope': 'All15 old95 complete evaluations exact. All18 old99 raw partial/full interval rows and feasibility exact after recording-order normalization; every full metric matched within1e-10 for changed floating summation order. All60 older-family candidate grids exact.',
            'recordingOrderPolicy': 'Original99 supplemented missing folds after existing rows; new sweep uses source-fold order. Raw per-recording equality remains exact.',
            'metricAbsoluteTolerance': 1e-10}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sweep.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')) and not args.output.exists(), 'Audit needs new NAS output')
    sweep.write_new(args.output, audit(args.report))
