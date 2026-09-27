#!/usr/bin/env python3
"""Freeze and execute the production-preserving compact split-adviser experiment."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
sys.dont_write_bytecode = True
import numpy as np
from analysis import neural_split_advisor as advisor
from analysis import neural_split_metrics as split_metrics
from analysis import neural_split_review as review
from analysis import neural_production_combinations as iv
from analysis.neural_rally_identity_metrics import evaluate_rally_identities

ROOT = Path(private_value('private-reference-0093'))
PRIOR = ROOT.parent/'2026-09-19-rally-review-proposals-v2'
SEEDS = (3407, 1729, 20260918)
NEW_SOURCES = ('analysis/neural_split_advisor.py', 'analysis/neural_split_metrics.py',
 'analysis/neural_split_review.py', 'scripts/run-neural-split-advisor.py',
 'scripts/audit_neural_split_advisor.py',
 'analysis/tests/test_neural_split_advisor.py', 'analysis/tests/test_neural_split_metrics.py',
 'analysis/tests/test_neural_split_review.py', 'analysis/tests/test_neural_split_audit.py',
 'analysis/tests/test_neural_split_runner.py',
 'scripts/summarize-neural-split-advisor.py', 'analysis/tests/test_neural_split_summary.py',
 'analysis/metrics.py', 'analysis/__init__.py', 'analysis/version.py')
TOTAL_FIELDS = ('reviewSeconds', 'budgetSeconds', 'unusedBudgetSeconds', 'reviewJobs', 'jobsAvailable',
 'reviewClips', 'editRegions', 'proposalsReviewed', 'acceptedSplitCount', 'rejectedSplitCount',
 'removedFalseParents', 'eventTimelineRemovedSeconds', 'eventTimelineAddedSeconds',
 'reviewedTrueRallies', 'playbackTrueRallies')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': Path(path).stat().st_size}


def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def require(value, message):
    if not value:
        raise ValueError(message)


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO/relative)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def register(root):
    require(not (root/'registration.json').exists(), 'Already registered')
    prior = read(PRIOR/'registration.json')
    require(canonical(prior['contract']) == prior['sha256'], 'Prior contract changed')
    c = prior['contract']
    for identity in [*c['sources'].values(), c['input'], c['probabilities']]:
        require(sha(identity['path']) == identity['sha256'], 'Prior dependency changed: '+identity['path'])
    q = read(root/'qualification.json')
    require(q['passed'], 'Qualification failed')
    data = read(c['input']['path'])
    require(len(data['records']) == 8 and sum(len(r['rallies']) for r in data['records']) == 322, 'Scope changed')
    source_names = sorted(set(c['sources']) | set(NEW_SOURCES))
    for name in source_names:
        dest = root/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        require(not dest.exists(), 'Snapshot already exists')
        shutil.copyfile(REPO/name, dest)
    for src, dest in [(c['input']['path'], root/'input.json'), (c['probabilities']['path'], root/'probabilities.npz'),
        (REPO/'docs/research/neural-split-advisor-protocol-2026-09-19.md', root/'protocol-initial.md')]:
        require(not dest.exists(), 'Input snapshot already exists'); shutil.copyfile(src, dest)
    contract = {'kind': 'production-preserving-compact-split-adviser-v1',
      'createdAt': datetime.now(timezone.utc).isoformat(), 'seeds': list(SEEDS),
      'model': 'compact_boost', 'splitPolicies': list(advisor.POLICIES),
      'reviewInventories': configurations(), 'rankers': list(review.RANKERS),
      'budgetFractions': list(review.BUDGETS), 'automaticOutcomes': 9, 'reviewedOutcomes': 168,
      'primaryMetric': 'F1_padP_coreR', 'targetPaddingSeconds': 2, 'paddingCases': [0, 1, 2, 3],
      'joinGapSeconds': 3, 'exportPolicy': 'immutable original production exports; event timeline separate',
      'input': ref(root/'input.json'), 'probabilities': ref(root/'probabilities.npz'),
      'protocol': ref(root/'protocol-initial.md'), 'qualification': ref(root/'qualification.json'),
      'sources': {p: ref(REPO/p) for p in source_names},
      'sourceCopies': {p: ref(root/'source'/p) for p in source_names},
      'priorRegistration': ref(PRIOR/'registration.json'), 'priorContractSha256': prior['sha256'],
      'newFits': 0, 'protectedTestOpened': False, 'productionChanged': False,
      'runtime': {'pythonVersion': sys.version, 'pythonExecutable': sys.executable, 'numpyVersion': np.__version__},
      'goldUsedForProposalsOrRanking': False, 'hypotheticalHumanOutcomes': True}
    write(root/'registration.json', {'sha256': canonical(contract), 'contract': contract})
    print(json.dumps({'registered': True, 'sha256': canonical(contract), 'sources': len(source_names)}), flush=True)


def configurations():
    return [{'id': f'{p}--{inv}', 'policy': p, 'inventory': inv}
            for p in advisor.POLICIES for inv in ('split_only', 'combined')] + [
            {'id': 'none--cleanup_only', 'policy': 'none', 'inventory': 'cleanup_only'}]


def load(root):
    reg = read(root/'registration.json'); c = reg['contract']
    require(canonical(c) == reg['sha256'], 'Contract changed')
    for item in [*c['sources'].values(), *c['sourceCopies'].values(), c['input'], c['probabilities'], c['protocol'], c['qualification']]:
        require(sha(item['path']) == item['sha256'], 'Frozen input/source changed: '+item['path'])
    return reg, read(c['input']['path'])


def evaluate(records, frozen_exports, identity_auditor, accounting, split_auditor):
    durations = iv.duration_rows(records, export_overrides=frozen_exports)
    identities = evaluate_rally_identities(records)
    proposal_metrics = split_metrics.evaluate_split_proposals(records)
    # Frozen-export accounting is audited against original production predictions.
    base = [{**r, 'predictions': r['productionEvents']} for r in records]
    duration_check = accounting.audit_duration_records(base, durations)
    identity_check = identity_auditor.audit_identity_metrics(records, identities)
    separate_export_check = split_auditor.audit_duration_records(records, durations, export_overrides=frozen_exports)
    split_check = split_auditor.audit_split_metrics(records, proposal_metrics)
    require(duration_check['passed'] and identity_check['passed'] and separate_export_check['passed'] and split_check['passed'], 'Independent metric audit failed')
    return {'durationMetrics': durations, 'identityMetrics': identities,
            'splitMetrics': proposal_metrics,
            'durationAudit': duration_check, 'separateExportAudit': separate_export_check,
            'identityAudit': identity_check, 'splitMetricAudit': split_check}


def execute_seed(args):
    root_text, seed = args; root = Path(root_text)
    reg, data = load(root)
    with np.load(reg['contract']['probabilities']['path'], allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    identity_auditor = load_module('split_run_identity_audit', 'scripts/audit-neural-rally-review-proposals.py')
    accounting = load_module('split_run_accounting', 'scripts/audit-neural-combination-accounting.py')
    split_auditor = load_module('split_run_audit', 'scripts/audit_neural_split_advisor.py')
    sources = data['records']
    entries = {x['recordingId']: x for x in data['entries'] if x['modelId'] == 'compact_boost' and x['seed'] == seed}
    frozen_exports = {r['id']: {str(p): iv.export(r['productionEvents'], r, p) for p in (0, 1, 2, 3)} for r in sources}
    plans = []
    for record in sources:
        entry = entries[record['id']]
        times, scores = arrays[entry['timesKey']], arrays[entry['scoresKey']]
        base, neural = record['productionEvents'], entry['events']
        cleanup = advisor.cleanup_proposals(record, base, neural)
        splits = {p: advisor.split_proposals(record, base, neural, times, scores, p) for p in advisor.POLICIES}
        poisoned = {**record, 'rallies': [], 'serveMarkers': [{'time': -12345}]}
        require(advisor.cleanup_proposals(poisoned, base, neural) == cleanup, 'Cleanup depends on gold')
        for p in advisor.POLICIES:
            require(advisor.split_proposals(poisoned, base, neural, times, scores, p) == splits[p], 'Splits depend on gold')
        candidate_check = {p: split_auditor.audit_candidates(record, base, neural, times, scores, p, splits[p]) for p in advisor.POLICIES}
        cleanup_check = split_auditor.audit_cleanup(record, base, neural, cleanup)
        require(all(v['passed'] for v in candidate_check.values()) and cleanup_check['passed'], 'Candidate audit failed')
        plans.append({'id': record['id'], 'cleanup': cleanup, 'splits': splits,
                      'cleanupYield': review.cleanup_yield(record, cleanup), 'candidateAudit': candidate_check, 'cleanupAudit': cleanup_check})
    baseline_records = [{**r, 'predictions': r['productionEvents'], 'splitProposals': []} for r in sources]
    baseline = evaluate(baseline_records, frozen_exports, identity_auditor, accounting, split_auditor)
    automatic = []
    for policy in advisor.POLICIES:
        records, per = [], []
        for r, plan in zip(sources, plans):
            proposals = plan['splits'][policy]
            events = advisor.apply_splits(r['productionEvents'], proposals)
            check = split_auditor.audit_apply_splits(r, r['productionEvents'], proposals, events)
            require(check['passed'], 'Automatic partition audit failed')
            records.append({**r, 'predictions': events, 'splitProposals': proposals})
            per.append({'id': r['id'], 'events': events, 'partitionAudit': check})
        automatic.append({'id': 'automatic--'+policy, 'policy': policy, 'perRecording': per,
                          **evaluate(records, frozen_exports, identity_auditor, accounting, split_auditor)})
    outcomes = []
    for config in configurations():
        for ranker in review.RANKERS:
            queue_plans = []
            for r, plan in zip(sources, plans):
                proposals = [] if config['policy'] == 'none' else plan['splits'][config['policy']]
                jobs = review.jobs(r, r['productionEvents'], proposals, plan['cleanup'], config['inventory'])
                queues = review.queues(r, jobs, ranker)
                qcheck = split_auditor.audit_jobs_queues(r, r['productionEvents'], proposals, plan['cleanup'],
                                                       config['inventory'], ranker, jobs, queues)
                require(qcheck['passed'], 'Queue audit failed')
                queue_plans.append({'jobs': jobs, 'queues': queues, 'audit': qcheck})
            for bi, fraction in enumerate(review.BUDGETS):
                records, per = [], []
                for r, plan, qp in zip(sources, plans, queue_plans):
                    proposals = [] if config['policy'] == 'none' else plan['splits'][config['policy']]
                    queue = qp['queues'][bi]
                    edited = review.human_edit(r, proposals, plan['cleanup'], queue, advisor, split_metrics)
                    hcheck = split_auditor.audit_human(r, proposals, plan['cleanup'], queue, edited)
                    require(hcheck['passed'], 'Human operation audit failed')
                    per.append({'id': r['id'], 'sourceGroup': r['sourceGroup'], **queue, **edited,
                                'humanAudit': hcheck, 'queueAudit': qp['audit']})
                    records.append({**r, 'predictions': edited['events'], 'splitProposals': edited['acceptedSplits']})
                workload = {k: sum(p[k] for p in per) for k in TOTAL_FIELDS}
                outcomes.append({**config, 'id': f"{config['id']}--{ranker}--budget-{round(100*fraction):02d}",
                  'ranker': ranker, 'budgetFraction': fraction,
                  'workload': workload, 'perRecording': per, 'queuePlans': queue_plans,
                  **evaluate(records, frozen_exports, identity_auditor, accounting, split_auditor)})
                print(json.dumps({'seed': seed, 'inventory': config['id'], 'ranker': ranker, 'budget': fraction, 'completed': len(outcomes)}), flush=True)
    result = {'contractSha256': reg['sha256'], 'seed': seed, 'baseline': baseline,
              'plans': plans, 'automatic': automatic, 'reviewed': outcomes}
    write(root/'results'/f'{seed}.json', result)
    return {'seed': seed, 'automatic': len(automatic), 'reviewed': len(outcomes), 'result': ref(root/'results'/f'{seed}.json')}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--register', action='store_true'); p.add_argument('--run', action='store_true')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if args.register:
        register(args.root)
    if args.run:
        reg, _ = load(args.root)
        require(not (args.root/'results').exists(), 'Results already exist; preserve original run')
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [future.result() for future in as_completed([pool.submit(execute_seed, (str(args.root), s)) for s in SEEDS])]
        require(sum(r['reviewed'] for r in rows) == 168, 'Incomplete matrix')
        write(args.root/'report.json', {'contractSha256': reg['sha256'], 'passed': True,
              'automaticOutcomes': 9, 'reviewedOutcomes': 168, 'seeds': sorted(rows, key=lambda r:r['seed']),
              'wallSeconds': time.monotonic()-started})


if __name__ == '__main__':
    main()
