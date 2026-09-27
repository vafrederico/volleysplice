#!/usr/bin/env python3
"""Require every registered task/precision before indexing final report inputs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_generalization_experiment import load_task
from analysis.neural_generalization_inputs import verified
from analysis.neural_recall_sweep import identity, read, require, write_new

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--registration-dir', type=Path, action='append', required=True)
p.add_argument('--reuse-plan', type=Path, action='append', required=True)
p.add_argument('--inventory', type=Path, required=True)
p.add_argument('--production-result', type=Path, required=True)
p.add_argument('--historical-fp32', type=Path, required=True)
p.add_argument('--historical-fp16', type=Path)
p.add_argument('--historical-int8', type=Path)
p.add_argument('--study-protocol', type=Path, required=True)
p.add_argument('--timing-amendment', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
entries, tasks = [], set()
plans = [read(path) for path in a.reuse_plan]
require(len(plans) == 2 and len({r['registration']['sha256'] for r in plans}) == 2,
        'Both randomized and export-proxy companion plans are required')
reuse_mappings = {}
for plan in plans:
    require(plan['kind'] == 'generalization-training-reuse-plan-v1', 'Unknown reuse plan')
    for key in ('registration', 'protocol', 'qualification'):
        verified(plan[key])
    for reference in plan['code'].values():
        verified(reference)
    for row in plan['tasks']:
        require(row['taskId'] not in reuse_mappings, 'Duplicate reuse mapping')
        reuse_mappings[row['taskId']] = row
execution_counts = {key: sum(plan['counts'][key] for plan in plans) for key in plans[0]['counts']}
require(execution_counts == {'logicalHeadFits': 144, 'physicalHeadFits': 102, 'reusedHeadFits': 42,
        'logicalStudentFits': 24, 'physicalStudentFits': 17}, 'Registered execution-sharing counts differ')
execution_counts['logicalHeadFits'] += 18
execution_counts['physicalHeadFits'] += 18
execution_counts['logicalStudentFits'] += 3
execution_counts['physicalStudentFits'] += 3
for directory in a.registration_dir:
    registration = read(directory/'registration.json')
    for task_id in registration['contract']['taskIds']:
        require(task_id not in tasks, 'Duplicate task across registrations')
        tasks.add(task_id)
        task_path = directory/'tasks'/(task_id.replace('/', '__')+'.json')
        task, _, _, _ = load_task(task_path)
        require((task_id in reuse_mappings) is (task['variant'] != 'original-corpus'),
                'Companion plan is absent or unexpectedly applies to the original corpus')
        fit = directory/'fits'/task['model']/f'seed-{task["seed"]}' if task['variant'] == 'original-corpus' else (
            directory/'fits'/task['variant']/task['model']/f'split-{task["splitSeed"]}')
        draws = [3407, 1729, 20260918] if task['variant'] == 'original-corpus' else [3407, 1729, 20260918, 20260923]
        for precision in ('fp32', 'fp16', 'int8') if task['model'] in ('dino-tcn', 'dino-transformer') else ('fp32',):
            path = fit/f'evaluation-{precision}.json'
            result = read(path)
            require(result['task'] == identity(task_path) and result['precision'] == precision,
                    'Evaluation task or precision differs')
            entries.append({'result': identity(path), 'expectedDraws': draws})
require(len(tasks) == 162 and len(entries) == 270, 'Full registered task/precision matrix is incomplete')
entries.append({'result': identity(a.production_result)})
historical = [{'result': identity(a.historical_fp32), 'precision': 'fp32'}]
require(a.historical_fp16 and a.historical_int8, 'Both historical precision transfers required')
historical.extend({'result': identity(path), 'precision': precision} for path, precision in (
    (a.historical_fp16, 'fp16'), (a.historical_int8, 'int8')))
write_new(a.output, {'kind': 'registered-generalization-result-index-v1', 'inventory': identity(a.inventory),
    'studyProtocol': identity(a.study_protocol), 'timingAmendment': identity(a.timing_amendment),
    'reusePlans': [identity(path) for path in a.reuse_plan], 'executionCounts': execution_counts,
    'registrations': [identity(d/'registration.json') for d in a.registration_dir], 'evaluations': entries,
    'historical': historical, 'indexer': identity(__file__),
    'metadata': {'title': 'Neural recall sweep and source-diversity experiments',
        'protocol': identity(a.study_protocol), 'timingAmendment': identity(a.timing_amendment),
        'neuralTaskCount': len(tasks), 'neuralTaskPrecisionCount': 270,
        'reusePlans': [identity(path) for path in a.reuse_plan], 'executionCounts': execution_counts,
        'executionSharing': '162 logical tasks use 120 distinct temporal-head fits and 20 distinct student encoders. Identical ordered training recipes share all four fixed checkpoints; each logical task retains its own calibration, selection and evaluation. Repeated split draws are correlated, not independent training replications. Historical outer-fold refits are additional and excluded from these counts.',
        'labelScope': '9 exact,6 reviewed continuous drafts,19 reviewed export coverage,8 unscored; beach excluded',
        'extraExportProxyTests': '674 approximate individual cores from18 reviewed training sources; never substituted for independent evaluation gold.',
        'knownSourceQualityCaveat': 'Aug16_164327879 rotates off court; outside-game-window/ignored footage remains excluded from evaluation and negatives.'}})
print(str(a.output))
