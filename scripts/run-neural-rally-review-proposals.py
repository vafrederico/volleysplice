#!/usr/bin/env python3
"""Freeze and execute budgeted rally-preserving review proposal comparisons."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
sys.dont_write_bytecode = True
from analysis import neural_rally_review_proposals as review
from analysis import neural_production_combinations as iv
from analysis.neural_rally_identity_metrics import evaluate_rally_identities

ROOT = Path(private_value('private-reference-0092'))
MODELS = ('compact_boost', 'dino_global', 'dino_boost')
SEEDS = (3407, 1729, 20260918)
SOURCES = (
    'analysis/neural_rally_review_proposals.py', 'analysis/neural_rally_identity_metrics.py',
    'scripts/run-neural-rally-review-proposals.py', 'scripts/prepare-neural-rally-review-input.py',
    'scripts/audit-neural-rally-review-proposals.py', 'scripts/audit-neural-rally-review-candidates.py',
    'analysis/tests/test_neural_rally_review_proposals.py', 'analysis/tests/test_neural_rally_identity_metrics.py',
    'analysis/tests/test_neural_rally_review_audit.py', 'analysis/tests/test_neural_rally_review_candidates_audit.py',
    'analysis/tests/test_neural_rally_review_input.py', 'analysis/tests/test_neural_rally_review_runner.py',
    'analysis/neural_human_review.py', 'analysis/neural_production_combinations.py',
    'scripts/audit-neural-combination-accounting.py', 'analysis/crop_evaluation.py', 'analysis/schema.py')
WORKLOAD = ('budgetSeconds', 'reviewSeconds', 'editSeconds', 'reviewClips', 'decisionRegions',
            'proposalsSelected', 'proposalsAvailable', 'censoredStarts', 'censoredEnds',
            'unobservedStarts', 'unobservedEnds', 'touchedRalliesWithUneditableBoundary',
            'reviewedTrueRallies', 'playbackTrueRallies')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': path.stat().st_size}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO/path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def configurations():
    return [{'id': f'{mode}--{model}--{inventory}--{ranker}', 'mode': mode, 'model': model,
             'inventory': inventory, 'ranker': ranker}
            for mode in ('production', 'individual') for model in MODELS
            for inventory in review.INVENTORIES for ranker in review.RANKERS]


def register(root):
    require(not (root/'registration.json').exists(), 'Registration already exists')
    data = read(root/'rally-review-input.json')
    require(data['models'] == list(MODELS) and data['seeds'] == list(SEEDS), 'Input model/seed scope differs')
    require(len(data['entries']) == 72 and len(data['records']) == 8, 'Input inventory differs')
    require(sum(len(r['rallies']) for r in data['records']) == 322, 'Gold count differs')
    require(sha(data['npz']['path']) == data['npz']['sha256'], 'Probability bytes changed')
    with np.load(data['npz']['path'], allow_pickle=False) as arrays:
        for entry in data['entries']:
            a, t = arrays[entry['scoresKey']], arrays[entry['timesKey']]
            require(a.dtype == np.float32 and t.dtype == np.float64 and a.shape == (len(t), 4), 'Aligned input shape/dtype differs')
            require(hashlib.sha256(a.tobytes()).hexdigest() == entry['scoresBytesSha256'], 'Score bytes differ')
            review.event_rows(entry['events'])
        for r in data['records']:
            review.event_rows(r['productionEvents']); review.event_rows(r['rallies'])
            require(hashlib.sha256(arrays[r['timesKey']].tobytes()).hexdigest() == r['timelineBytesSha256'], 'Timeline changed')
    require(sha(data['priorProbabilityInput']['path']) == data['priorProbabilityInput']['sha256'], 'Prior probability input changed')
    old_reg = read(Path(data['priorProbabilityInput']['path']).parent/'registration.json')
    require(canonical(old_reg['contract']) == old_reg['sha256'], 'Prior review contract changed')
    for identity in [*old_reg['contract']['sources'].values(), data['sourceScript']]:
        require(sha(identity['path']) == identity['sha256'], 'Previous source changed: '+identity['path'])
    protocol = REPO/'docs/research/neural-rally-review-proposals-protocol-2026-09-19.md'
    with (root/'protocol-initial.md').open('xb') as stream:
        stream.write(protocol.read_bytes())
    qualification = read(root/'qualification-v1.json')
    require(qualification['passed'], 'Qualification did not pass')
    failed_path = root.parent/'2026-09-19-rally-review-proposals'/'failed-attempt-v1.json'
    failed = read(failed_path)
    require(failed['completedResultFiles'] == 0 and not failed['outcomeTablesRead'], 'Initial attempt closure differs')
    contract = {'kind': 'fixed-budget-rally-review-proposals-v1', 'createdAt': datetime.now(timezone.utc).isoformat(),
                'models': list(MODELS), 'seeds': list(SEEDS), 'configurations': configurations(),
                'budgetFractions': list(review.BUDGETS), 'configurationSeedRuns': 108, 'outcomeCells': 432,
                'primaryMetric': 'F1_padP_coreR', 'targetPaddingSeconds': 2, 'paddingCases': [0, 1, 2, 3],
                'joinGapSeconds': 3, 'editMarginSeconds': 2, 'viewContextSeconds': 2,
                'newDetectorFits': 0, 'learnedReviewFits': 0, 'protectedTestOpened': False, 'productionChanged': False,
                'goldUsedForProposalsOrRanking': False, 'hypotheticalHumanOutcomes': True,
                'input': ref(root/'rally-review-input.json'), 'probabilities': data['npz'],
                'goldSemantics': ref(root/'gold-semantics-audit-v1.json'), 'qualification': ref(root/'qualification-v1.json'),
                'previousFailedAttempt': ref(failed_path),
                'protocol': ref(root/'protocol-initial.md'), 'sources': {p: ref(REPO/p) for p in SOURCES},
                'oldReviewContractSha256': old_reg['sha256'], 'sourceGroups': sorted({r['sourceGroup'] for r in data['records']}),
                'labelLimit': 'Rally starts follow serve-contact annotation policy; no complete independent serve marker, serving-side, winner or score benchmark',
                'guardrails': ['eventF1', 'observedStartLocalization@1s', 'completeMisses', 'mergedPredictions',
                               'splitTrueRallies', 'censoredStarts', 'censoredEnds']}
    write(root/'registration.json', {'sha256': canonical(contract), 'contract': contract})
    print(json.dumps({'registered': True, 'contractSha256': canonical(contract), 'outcomeCells': 432}), flush=True)


def load(root):
    reg = read(root/'registration.json'); c = reg['contract']
    require(canonical(c) == reg['sha256'], 'Contract changed')
    for identity in [*c['sources'].values(), c['input'], c['probabilities'], c['protocol'], c['qualification'], c['goldSemantics'], c['previousFailedAttempt']]:
        require(sha(identity['path']) == identity['sha256'], 'Registered dependency changed: '+identity['path'])
    return reg, read(c['input']['path'])


def duration_audit(records, rows, auditor):
    checks = [auditor.audit_duration_records(records, rows)]
    for subset in [[r for r in records if r['sourceGroup'] == g] for g in sorted({r['sourceGroup'] for r in records})] + [[r] for r in records]:
        checks.append(auditor.audit_duration_records(subset, iv.duration_rows(subset)))
    require(all(c['passed'] for c in checks), 'Duration audit failed')
    return {'passed': True, 'scopeCount': len(checks), 'paddingCases': 4}


def evaluate(records, auditor, accounting):
    durations = iv.duration_rows(records)
    identities = evaluate_rally_identities(records)
    return {'durationMetrics': durations, 'identityMetrics': identities,
            'durationAudit': duration_audit(records, durations, accounting),
            'identityAudit': auditor.audit_identity_metrics(records, identities)}


_STATE = None


def initialize(root_text):
    global _STATE
    root = Path(root_text)
    reg, data = load(root)
    with np.load(data['npz']['path'], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    _STATE = (root, reg, data, arrays,
              module('rally_review_auditor', 'scripts/audit-neural-rally-review-proposals.py'),
              module('rally_candidate_auditor', 'scripts/audit-neural-rally-review-candidates.py'),
              module('rally_accounting_auditor', 'scripts/audit-neural-combination-accounting.py'))


def execute(task):
    root, reg, data, arrays, auditor, candidate_auditor, accounting = _STATE
    config, seed = task
    entries = {(x['modelId'], x['seed'], x['recordingId']): x for x in data['entries']}
    records, plans, base_records = [], [], []
    for source in data['records']:
        rid = source['id']; entry = entries[config['model'], seed, rid]
        base = source['productionEvents'] if config['mode'] == 'production' else entry['events']
        rec = {k: source[k] for k in ('id', 'sourceGroup', 'durationSeconds', 'rallies', 'ignoredIntervals')}
        times, scores = arrays[entry['timesKey']], arrays[entry['scoresKey']]
        candidates = review.proposals(rec, base, entry['events'], times, scores, config['inventory'], config['mode'])
        check = candidate_auditor.audit_candidates(rec, base, entry['events'], times, scores,
                                                  config['inventory'], config['mode'], candidates)
        poisoned = {**rec, 'rallies': [], 'serveMarkers': [{'time': -1000000, 'invalid': True}]}
        require(review.proposals(poisoned, base, entry['events'], times, scores, config['inventory'], config['mode']) == candidates,
                'Labels influence proposal generation')
        queues = review.budget_queues(rec, candidates, config['ranker'])
        queue_check = auditor.audit_budget_queues(rec, candidates, config['ranker'], queues)
        plans.append({'id': rid, 'sourceGroup': rec['sourceGroup'], 'baseEvents': base,
                      'candidates': candidates, 'queues': queues, 'candidateAudit': check, 'queueAudit': queue_check})
        base_records.append({**rec, 'predictions': base})
        records.append(rec)
    base_eval = evaluate(base_records, auditor, accounting)
    outcomes = []
    for budget_index, fraction in enumerate(review.BUDGETS):
        per, edited_records = [], []
        for rec, plan in zip(records, plans):
            queue = plan['queues'][budget_index]
            edited = review.edit_events(rec, plan['baseEvents'], queue['editWindows'])
            check = auditor.audit_editor(rec, plan['baseEvents'], queue, edited)
            touched = len(review.legacy.touched_rallies(rec, queue['playbackWindows']))
            require(touched == sum(bool(auditor.intersection(auditor.intervals([g], rec['durationSeconds']),
                                                           queue['playbackWindows'])) for g in rec['rallies']),
                    'Playback true-rally count differs from independent intersection')
            per.append({'id': rec['id'], 'sourceGroup': rec['sourceGroup'], **queue, **edited,
                        'playbackTrueRallies': touched, 'editorAudit': check})
            edited_records.append({**rec, 'predictions': edited['events']})
        values = evaluate(edited_records, auditor, accounting)
        workload = {key: sum(row[key] for row in per) for key in WORKLOAD}
        workload['reviewFractionOfVideo'] = workload['reviewSeconds']/values['durationMetrics'][2]['evaluableVideoSeconds']
        workload['budgetUtilization'] = workload['reviewSeconds']/workload['budgetSeconds']
        groups = {}
        for group in reg['contract']['sourceGroups']:
            selected = [r for r in per if r['sourceGroup'] == group]
            totals = {key: sum(row[key] for row in selected) for key in WORKLOAD}
            denominator = sum(x['evaluableVideoSeconds'] for x in values['durationMetrics'][2]['perRecording'] if x['sourceGroup'] == group)
            totals.update(reviewFractionOfVideo=totals['reviewSeconds']/denominator,
                          budgetUtilization=totals['reviewSeconds']/totals['budgetSeconds'])
            groups[group] = totals
        outcomes.append({'budgetFraction': fraction, **values, 'workload': workload,
                         'workloadBySourceGroup': groups, 'recordings': per})
    result = {'configuration': config, 'seed': seed, 'contractSha256': reg['sha256'],
              'automatic': base_eval, 'plans': plans, 'outcomes': outcomes}
    path = root/'results'/f"{config['id']}--{seed}.json"
    write(path, result)
    return ref(path)


def run(root, workers):
    reg, _ = load(root)
    require(not (root/'report.json').exists(), 'Completed report exists')
    require(not list((root/'results').glob('*.json')), 'Partial results exist; investigate rather than silently overwrite')
    tasks = [(c, seed) for c in configurations() for seed in SEEDS]
    started = datetime.now(timezone.utc).isoformat(); start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(str(root),)) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            print(f'COMPLETED {len(results)}/108 configuration-seed runs', flush=True)
    require(load(root)[0] == reg, 'Registered source changed during execution')
    report = {'kind': reg['contract']['kind'], 'status': 'completed-fixed-rally-review-proposals',
              'contractSha256': reg['sha256'], 'startedAt': started, 'completedAt': datetime.now(timezone.utc).isoformat(),
              'wallSeconds': time.perf_counter()-start, 'workers': workers, 'configurationSeedRuns': 108,
              'outcomeCells': 432, 'candidatePlansAudited': 864, 'editedRecordingsAudited': 3456,
              'newDetectorFits': 0, 'learnedReviewFits': 0, 'protectedTestOpened': False, 'productionChanged': False,
              'results': sorted(results, key=lambda x: x['path'])}
    write(root/'report.json', report)
    print(json.dumps({'completed': True, 'report': ref(root/'report.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    register(args.root) if args.register else run(args.root, args.workers)
