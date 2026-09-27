#!/usr/bin/env python3
"""Lossless scalar projection of the independently audited historical sweep."""
import argparse
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis.neural_recall_sweep_adapters import bind


def compact(value):
    """Retain every aggregate field, excluding per-rally/per-recording diagnostics."""
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()
                if key not in ('rallies', 'recordings', 'sourceGroups')}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def summarize(root, output):
    io.require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(),
               'Summary needs a new NAS output')
    reference = io.identity(root / 'report-complete.json')
    specs = (
        ('audit-complete.json', 'independent-recall-floor-sweep-audit-v1', 'audit-neural-recall-sweep.py'),
        ('audit-controls-complete.json', 'independent-historical-recall-control-replay-v1', 'audit-neural-historical-controls.py'),
        ('audit-refits.json', 'independent-historical-all-outer-epochs-audit-v1', 'audit-neural-historical-refits.py'),
        ('audit-owners.json', 'independent-historical-score-owner-provenance-audit-v1', 'audit-neural-historical-owners.py'))
    gates = [io.identity(root / name) for name, _, _ in specs]
    protocol = io.identity(root / 'protocol.json')
    for gate_ref, (name, kind, auditor) in zip(gates, specs, strict=True):
        gate = io.read(bind(gate_ref))
        io.require(gate['passed'] is True and gate['kind'] == kind
                   and gate['auditor'] == io.identity(REPO / 'scripts' / auditor),
                   'Historical audit kind or current auditor differs')
        if 'report' in gate:
            io.require(gate['report'] == reference, 'Audit binds another report')
        if name in ('audit-refits.json', 'audit-owners.json'):
            io.require(gate['protocol'] == protocol, 'Audit binds another protocol')
        if name == 'audit-refits.json':
            io.require(gate['plan'] == io.identity(root / 'refit-plan.json'), 'Refit plan differs')
    report = io.read(bind(reference))
    rows = []
    for cell in report['cells']:
        panel = cell['panels']['historical-nested-exact']
        values = []
        for i, floor in enumerate(panel['result']['floors']):
            complete = floor['completeEvaluationScope']
            io.require((floor['evaluation'] is not None) == complete,
                       'Partial scope has an aggregate evaluation')
            decisions = [fold['decisions'][i] for fold in cell['folds']]
            values.append({key: floor[key] for key in (
                'floorPercent', 'completeEvaluationScope', 'partialScopeNotRankable',
                'foldStatuses', 'scopeRecordingIds')} | {
                'eligibleFoldCount': sum(d['feasible'] for d in decisions),
                'expectedFoldCount': len(decisions),
                'selections': [{'heldSourceGroup': fold['foldId'],
                    'feasible': decision['feasible'], 'selected': decision['selected'],
                    'maximumInnerRecall': decision['maximumInnerRecall'],
                    'eligibleCandidateCount': decision['eligibleCandidateCount']}
                    for fold, decision in zip(cell['folds'], decisions, strict=True)],
                'evaluation': compact(floor['evaluation']) if complete else None})
        rows.append({key: cell[key] for key in ('model', 'variant', 'seed', 'selectionDesign')} | {
            'panel': {key: panel[key] for key in panel if key != 'result'},
            'expectedGoldSha256': panel['result']['expectedGoldSha256'], 'floors': values})
    io.require(len(rows) == 18 and len({(r['model'], r['seed']) for r in rows}) == 18,
               'Historical model/seed inventory differs')
    for comparison in report['comparisons']:
        matched = [r for r in rows if r['model'] == comparison['model']]
        io.require(len(matched) == 3, 'Mean lacks all three registered seeds')
        for index, floor in enumerate(comparison['summary']['floors']):
            available = [row['floors'][index] for row in matched]
            complete = all(row['completeEvaluationScope'] for row in available)
            io.require((floor['meanPadding'] is not None) == complete,
                       'Mean hides an infeasible seed')
            if complete:
                for pad, mean in enumerate(floor['meanPadding']):
                    for key, value in mean.items():
                        if key not in ('paddingSecondsBeforeAndAfter', 'joinGapSeconds'):
                            io.require(value == statistics.mean(row['evaluation']['padding'][pad][key]
                                       for row in available), 'Copied mean differs from seed metrics')
    result = {'kind': 'audited-historical-recall-sweep-scalar-summary-v1',
        'report': reference, 'protocol': protocol, 'audits': gates, 'generator': io.identity(__file__),
        'primaryPaddingSecondsBeforeAndAfter': 2, 'paddingCases': [0, 1, 2, 3],
        'joinGapSeconds': 3, 'floors': list(range(90, 101)), 'cells': rows,
        'comparisons': report['comparisons'],
        'interpretation': 'Floors constrain pooled inner calibration R_core, not held-source recall. '
            'Aggregate evaluation is null when any held-source fold is infeasible; '
            'three-seed means are null unless all three seeds cover the identical original eight recordings. '
            'Event and coreCoverage guardrails use unpadded predictions; primaryExportCoverage uses two-second padding. '
            'Historical ignored-segment inference is unchanged; this is distinct from blind full-timeline deployment evaluation.',
        'projection': 'All stored aggregate evaluation fields copied exactly; per-rally, per-recording, '
            'per-source diagnostics and raw prediction rows remain in the bound full report.',
        'trainingPerformed': False, 'gpuUsed': False, 'externalOutcomesOpened': False}
    for ref in (reference, *gates):
        bind(ref)
    io.write_new(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.study, args.output)
    print(io.identity(args.output), flush=True)
