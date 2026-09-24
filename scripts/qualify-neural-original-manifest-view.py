#!/usr/bin/env python3
"""Register/qualify the exact legacy-manifest alias without reading predictions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_recall_sweep as io
from analysis import neural_original_manifest_view as view
from analysis import neural_generalization_results as results


def register(args):
    proof = io.read(args.correspondence)
    io.require(proof['passed'] is True, 'Independent correspondence must pass first')
    original = io.read(args.evaluation_plan)
    legacy = io.read(view.checked(proof['legacyManifest']))
    projected = view.projection(legacy)
    names = ('analysis/neural_original_manifest_view.py',
             'scripts/qualify-neural-original-manifest-view.py',
             'analysis/tests/test_neural_original_manifest_view.py',
             'docs/research/neural-original-manifest-view-2026-09-23.md')
    io.write_new(args.output, {'kind': view.PLAN_KIND, 'policy': view.POLICY,
        'createdAtUTC': datetime.now(timezone.utc).isoformat(),
        'adapter': io.identity(REPO/names[0]), 'correspondence': io.identity(args.correspondence),
        'legacyManifest': proof['legacyManifest'], 'normalizedManifest': proof['normalizedManifest'],
        'inventory': proof['inventory'], 'originalEvaluationPlan': io.identity(args.evaluation_plan),
        'panel': io.identity(args.panel), 'globalSelectionGate': io.identity(args.global_selection_gate),
        'registrations': original['registrations'], 'frozenCode': original['code'],
        'tasks': list({r['task']['path']: r['task'] for r in original['jobs']}.values()),
        'orderedRecordingIds': [r['id'] for r in projected['records']],
        'sourceGroups': sorted({r['sourceGroup'] for r in projected['records']}),
        'recordsCanonicalSha256': io.canonical(projected['records']),
        'code': {name: io.identity(REPO/name) for name in names},
        'failedExecution': io.identity(args.failed_execution),
        'failedBeforeAnyAccuracyResult': True})


def qualify(args):
    plan = view.verify_plan(args.plan)
    original = view.raw_read(plan['legacyManifest']['path'])
    adapted = view.projection(original)
    # This independently constructs the required concatenation without using
    # the adapter's tier constant or projection implementation.
    expected = original['exactRows'] + original['draftRows'] + original['coverageRows']
    io.require(io.canonical(adapted['records']) == io.canonical(expected) and len(expected) == 18,
               'Full source rows/order were changed')
    io.require(io.canonical({k: v for k, v in adapted.items() if k != 'records'}) == io.canonical(original),
               'An existing legacy document field changed')
    normalized = view.raw_read(plan['normalizedManifest']['path'])['records']
    io.require(len(normalized) == 18 and len({r['id'] for r in normalized}) == 18
               and {r['id'] for r in normalized} == {r['id'] for r in expected}, 'Normalized18 membership differs')
    expected_groups = {r['sourceGroup'] for r in normalized}
    actual_groups = {r['sourceGroup'] for r in adapted['records']}
    io.require(actual_groups == expected_groups and len(actual_groups) == 7, 'Seven-group correspondence differs')
    panel = view.raw_read(plan['panel']['path'])
    manifest = view.raw_read(view.checked(panel['manifest']))
    records = [r for r in manifest['records'] if 'evaluate' in r['eligibleRoles'] and r['scoringPolicy'] != 'none']
    inventory = {r['id']: r for r in view.raw_read(plan['inventory']['path'])['records']}
    checks = []
    for registration_ref in plan['registrations']:
        registration = view.raw_read(view.checked(registration_ref))
        root = Path(registration_ref['path']).parent
        for task_id in registration['contract']['taskIds']:
            path = root/'tasks'/(task_id.replace('/', '__')+'.json'); task = view.raw_read(path)
            baseline = results.panel_members(records, inventory, task, expected_groups)
            projected = results.panel_members(records, inventory, task, actual_groups)
            io.require(baseline == projected, 'Neural panel membership changed')
            checks.append({'task': io.identity(path), 'equal': True,
                           'panelMembershipCanonicalSha256': io.canonical(baseline)})
    io.require(len(checks) == 162 and len({r['task']['sha256'] for r in checks}) == 162, 'Incomplete task qualification')
    used = {r['sourceGroup'] for r in records if not results.is_clean(inventory[r['id']], 'no-production-training-or-calibration')}
    baseline = results.panel_members(records, inventory, None, expected_groups, used)
    projected = results.panel_members(records, inventory, None, actual_groups, used)
    io.require(baseline == projected, 'Production panel membership changed')
    io.write_new(args.output, {'kind': view.QUAL_KIND, 'passed': True,
        'plan': io.identity(args.plan), 'qualifier': io.identity(__file__), 'adapter': plan['adapter'],
        'legacyManifest': plan['legacyManifest'], 'recordsCanonicalSha256': io.canonical(adapted['records']),
        'fullOriginalDocumentPreserved': True, 'all18RowsUnchanged': True,
        'allSevenSourceGroupsPreserved': True, 'neuralTaskCount': 162, 'productionComparatorCount': 2,
        'allPanelMembershipsEqual': True, 'neuralMembershipChecks': checks,
        'productionMembershipCheck': {'equal': True, 'comparators': ['productionDefault', 'productionUnion'],
                                    'panelMembershipCanonicalSha256': io.canonical(baseline)},
        'scoreArraysRead': False, 'accuracyMetricsComputed': False,
        'createdAtUTC': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    reg = sub.add_parser('register')
    for key in ('correspondence', 'evaluation-plan', 'panel', 'global-selection-gate', 'failed-execution', 'output'):
        reg.add_argument('--'+key, type=Path, required=True)
    qual = sub.add_parser('qualify')
    for key in ('plan', 'output'): qual.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    io.require(Path('/mnt/freenas').is_mount() and args.output.resolve().is_relative_to('/mnt/freenas'), 'NAS output required')
    globals()[args.action](args)
    print(json.dumps({'action': args.action, 'output': io.identity(args.output)}), flush=True)
