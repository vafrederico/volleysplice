#!/usr/bin/env python3
"""Copy audited precision-transfer aggregate fields into a small NAS artifact."""
import argparse
import gc
import importlib.util
import os
from pathlib import Path
import resource
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()
                if key not in ('rallies', 'recordings', 'sourceGroups')}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def summarize(root, output):
    io.require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(), 'New NAS output required')
    io.require(sorted(os.sched_getaffinity(0)) == [16, 17] and os.getpriority(os.PRIO_PROCESS, 0) >= 10
               and os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'Serial CPU resource scope differs')
    helper_path = REPO/'scripts/run-neural-historical-precision-cpu.py'
    spec = importlib.util.spec_from_file_location('frozen_precision_resource_guard', helper_path)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    preflight = guard.snapshot()
    io.require(preflight['startupFloorsPassed'], 'Frozen precision startup resource floors not reached')
    audit_ref = io.identity(root/'audit.json'); audit = io.read(audit_ref['path'])
    protocol_ref = io.identity(root/'protocol.json'); source_ref = io.identity(root/'report.json')
    io.require(audit['kind'] == 'independent-historical-dino-precision-floor-audit-v1'
        and audit['passed'] is True and audit['protocol'] == protocol_ref and audit['report'] == source_ref
        and audit['auditor'] == io.identity(REPO/'scripts/audit-neural-historical-precision-sweep.py'),
        'Passing current precision numerical audit differs')
    cells, comparisons, projections = [], [], []
    for arm in ('fp16', 'int8'):
        reference = io.identity(root/f'report-{arm}.json'); projections.append(reference)
        report = io.read(reference['path'])
        io.require(report['kind'] == 'audited-historical-precision-sweep-projection-v1'
            and report['precision'] == arm and report['sourceAudit'] == audit_ref and report['sourceReport'] == source_ref
            and report['protocol'] == protocol_ref and report['precisionSpecificSelection'] is False
            and report['code'] == io.identity(REPO/'scripts/project-neural-historical-precision-sweep.py'),
            'Projection scope or audit association differs')
        rows = []
        for cell in report['cells']:
            panel = cell['panels']['historical-nested-exact']; result = panel['result']
            floors = []
            for floor in result['floors']:
                complete = floor['completeEvaluationScope']
                io.require((floor['evaluation'] is not None) == complete, 'Incomplete scope has aggregate metrics')
                floors.append({key: floor[key] for key in ('floorPercent', 'completeEvaluationScope',
                    'partialScopeNotRankable', 'foldStatuses', 'scopeRecordingIds')} | {
                    'evaluation': compact(floor['evaluation']) if complete else None})
            rows.append({key: cell[key] for key in ('model', 'variant', 'seed', 'precision', 'selectionDesign')} | {
                'panel': {key: value for key, value in panel.items() if key != 'result'},
                'expectedGoldSha256': result['expectedGoldSha256'], 'floors': floors})
        io.require({(r['model'], r['seed']) for r in rows} == {
            (model, seed) for model in ('dino_tcn_short_boost', 'dino_transformer') for seed in (3407, 1729, 20260918)}
            and len(rows) == 6, 'Projection model/seed population differs')
        io.require(len(report['comparisons']) == 2 and {c['model'] for c in report['comparisons']}
            == {'dino_tcn_short_boost', 'dino_transformer'}, 'Precision comparison population differs')
        for comparison in report['comparisons']:
            matched = [r for r in rows if r['model'] == comparison['model']]
            io.require(len(matched) == 3, 'Precision mean lacks three registered seeds')
            for index, floor in enumerate(comparison['summary']['floors']):
                available = [row['floors'][index] for row in matched]
                complete = all(row['completeEvaluationScope'] for row in available)
                io.require((floor['meanPadding'] is not None) == complete, 'Precision mean hides an infeasible seed')
                if complete:
                    for pad, mean in enumerate(floor['meanPadding']):
                        for key, value in mean.items():
                            if key not in ('paddingSecondsBeforeAndAfter', 'joinGapSeconds'):
                                io.require(value == statistics.mean(row['evaluation']['padding'][pad][key]
                                    for row in available), 'Copied precision mean differs from seed metrics')
        cells.extend(rows); comparisons.extend(report['comparisons'])
        del report; gc.collect()
    result = {'kind': 'audited-historical-precision-scalar-summary-v1', 'protocol': protocol_ref,
        'report': source_ref, 'audit': audit_ref, 'projections': projections,
        'generator': io.identity(__file__), 'resourceGuard': io.identity(helper_path), 'resourcePreflight': preflight,
        'maximumProcessRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'cells': cells, 'comparisons': comparisons, 'primaryPaddingSecondsBeforeAndAfter': 2,
        'paddingCases': [0, 1, 2, 3], 'joinGapSeconds': 3, 'floors': list(range(90, 101)),
        'precisionSpecificSelection': False,
        'interpretation': 'FP32-selected epochs and decoders transfer unchanged to FP16/INT8 embeddings. '
            'These are deployment precision transfer diagnostics, not independently calibrated precision recall floors. '
            'Three-seed means are null unless all three seeds cover all eight original source-held recordings. '
            'Historical ignored-segment inference is unchanged; this is distinct from blind full-timeline deployment.',
        'projection': 'All stored aggregate evaluation fields copied exactly; detailed diagnostic arrays remain in bound reports.',
        'trainingPerformed': False, 'gpuUsed': False, 'newMetricsComputed': False, 'externalOutcomesOpened': False}
    for ref in (audit_ref, protocol_ref, source_ref, *projections):
        io.require(io.identity(ref['path']) == ref, 'Input changed during scalar projection')
    io.write_new(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); summarize(args.study, args.output)
    print(io.identity(args.output), flush=True)
