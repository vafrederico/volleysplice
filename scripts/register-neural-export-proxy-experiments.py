#!/usr/bin/env python3
"""Register the two user-requested approximate export-rally supervision arms."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value

from analysis.neural_context_development import canonical_hash, identity, read, write_immutable
from analysis.neural_generalization_experiment import EPOCHS, MODELS, require, validate_membership
from analysis.neural_generalization_inputs import manifest_rows
from analysis.schema import Interval
from analysis.crop_evaluation import pad_and_merge_intervals


def proxy_inputs(base_manifest, proxy_document):
    result = copy.deepcopy(base_manifest)
    candidates = {r['id']: r for r in proxy_document['records']}
    require(len(candidates) == 18, 'Expected eighteen reviewed export training sources')
    seen = set()
    for row in result['records']:
        if row['id'] not in candidates:
            continue
        source = candidates[row['id']]
        require(row['labelTier'] == 'coverage' and not row['protected'] and 'fit' in row['eligibleRoles'],
                'Export proxy must come from authorized nonprotected coverage')
        require(row['sourceGroup'] == source['sourceGroup'] and row['environment'] == 'grass', 'Proxy source association differs')
        require(row['sourceGroup'] in (private_value('source-group-004'), private_value('source-group-006')), 'Unexpected proxy source group')
        require(all(row.get(key) == source.get(key) for key in ('video', 'roi', 'durationSeconds')),
                'Proxy media coordinates differ from base inputs')
        require(source.get('contentSha256') is None or row['contentSha256'] == source['contentSha256'],
                'Proxy declared media hash differs from base inputs')
        rallies = source['rallies']
        require(rallies and all(r['end']-r['start'] >= .25 for r in rallies), 'Invalid proxy core duration')
        require(all(b['start'] >= a['end'] for a, b in zip(rallies, rallies[1:])), 'Overlapping saved proxy cores')
        # The fixed loader's exact tier supplies four supervised heads. The
        # experimental provenance remains explicit and never changes final gold.
        row['labelTier'] = 'exact'
        row['scoringPolicy'] = 'exact-core'
        row['rallies'] = copy.deepcopy(rallies)
        row['eligibleRoles'] = list(dict.fromkeys([*row['eligibleRoles'], 'calibrate']))
        ignored = [Interval(r['start'], r['end']) for r in row.get('ignoredIntervals', []) + source['ignoredIntervals']]
        for window in (row.get('gameWindow'), source.get('gameWindow')):
            if window:
                if window['start'] > 0:
                    ignored.append(Interval(0, window['start']))
                if window['end'] < row['durationSeconds']:
                    ignored.append(Interval(window['end'], row['durationSeconds']))
        row['ignoredIntervals'] = [r.to_dict() for r in pad_and_merge_intervals(ignored, row['durationSeconds'], 0, 0)]
        row['experimentalSupervision'] = {'kind': 'reviewed-export-rally-proxy', 'independentSemanticGold': False,
            'fourHeadsSupervised': True, 'approximateEndpoints': True, 'originalEvaluationPolicy': 'export-coverage',
            'sourceFeedback': source['sourceFeedback'], 'derivation': source['derivation']}
        seen.add(row['id'])
    require(seen == set(candidates), 'Proxy input population incomplete')
    result['kind'] = 'neural-generalization-inputs-v1'
    result['experimentalLabelPolicy'] = 'Approximate saved export cores for training/calibration only; authoritative evaluation labels unchanged.'
    return result


def proxy_tasks(base, plans, rows):
    """Pair the same four exact-group draws; move one whole export group."""
    by_id = {r['id']: r for r in rows}
    tasks = []
    for index, plan in enumerate(plans):
        for variant in ('export-rally-training', 'export-rally-selection'):
            train, calibration = list(plan['trainIds']), list(plan['calibrationIds'])
            selection_group = None
            if variant == 'export-rally-selection':
                selection_group = (private_value('source-group-004'), private_value('source-group-006'))[index % 2]
                moved = [identifier for identifier in train if by_id[identifier]['sourceGroup'] == selection_group]
                require(moved, 'Export selection group was not in paired training population')
                train = [identifier for identifier in train if identifier not in moved]
                calibration.extend(moved)
            for model in MODELS:
                task = {**base, **plan, 'variant': variant, 'trainIds': train, 'calibrationIds': calibration,
                        'model': model, 'seed': 3407, 'selectionExportSourceGroup': selection_group,
                        'originalExactTrainGroups': plan['exactTrainGroups'],
                        'trainingSourceGroups': sorted({by_id[k]['sourceGroup'] for k in train}),
                        'calibrationGroups': sorted({by_id[k]['sourceGroup'] for k in calibration}),
                        'selectionLabelPolicy': 'exact-and-export-rally-proxy' if selection_group else 'exact-rallies',
                        'selectionDesign': 'source-disjoint-reviewed-export-core-proxy-experiment',
                        'taskId': f'{variant}/{model}/split-{plan["splitSeed"]}'}
                task.pop('exactTrainGroups')
                validate_membership(task, rows)
                tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--proxies', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not (args.output/'registration.json').exists(), 'Proxy study is already registered')
    spec = importlib.util.spec_from_file_location('frozen_generalization_registration', REPO/'scripts/register-neural-generalization.py')
    original_registration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_registration)
    base_manifest, base_rows = manifest_rows(args.manifest)
    original, _ = original_registration.original_ids()
    original_registration.validate_original_rows(base_rows, original)
    require(read(args.qualification).get('passed') is True, 'Feature/input qualification absent')
    proxy_document = read(args.proxies)
    derived = proxy_inputs(base_manifest, proxy_document)
    common_groups = [private_value('source-group-001'), private_value('source-group-008')]
    plans = [p for p in original_registration.split_plans(base_rows, original, set(common_groups))
             if p['variant'] == 'expanded-medium']
    require(len(plans) == 4, 'Expected four balanced draws')
    args.output.mkdir(parents=True, exist_ok=True)
    derived.update(originalManifest=identity(args.manifest), proxySource=identity(args.proxies))
    manifest_path = args.output/'proxy-inputs.json'
    write_immutable(manifest_path, derived)
    _, rows = manifest_rows(manifest_path)
    base = {'manifest': identity(manifest_path), 'features': identity(args.features),
            'epochs': list(EPOCHS), 'commonEvaluationGroups': common_groups, 'lossArm': 'short_boost',
            'targetPaddingSeconds': 2, 'joinGapSeconds': 3, 'floorsPercent': list(range(90, 101)),
            'productionPromotionAllowed': False, 'proxySource': identity(args.proxies)}
    tasks = proxy_tasks(base, plans, rows)
    sources = original_registration.code_dependencies()
    sources['scripts/register-neural-export-proxy-experiments.py'] = identity(__file__)
    archive = {}
    for name, reference in sources.items():
        destination = args.output/'registered-sources'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(reference['path']).read_bytes())
        archive[name] = identity(destination)
    protocol_path = args.output/'protocol.md'
    protocol_path.write_bytes(args.protocol.read_bytes())
    contract = {'kind': 'reviewed-export-rally-proxy-training-selection-v1', 'protocol': identity(protocol_path),
        'manifest': identity(manifest_path), 'features': identity(args.features), 'proxySource': identity(args.proxies),
        'originalManifest': identity(args.manifest), 'qualification': identity(args.qualification),
        'code': sources, 'archivedCode': archive, 'taskDigests': [canonical_hash(t) for t in tasks],
        'taskIds': [t['taskId'] for t in tasks], 'trainingSeeds': [3407],
        'splitSeeds': list(original_registration.SPLIT_SEEDS), 'floorsPercent': list(range(90, 101)),
        'paddingSeconds': [0, 1, 2, 3], 'joinGapSeconds': 3, 'commonEvaluationGroups': common_groups,
        'selectionUsesAdditionalPanelOutcomes': False, 'approximateLabelsUsedOnlyForFitAndSelection': True,
        'independentEvaluationGoldChanged': False, 'infeasiblePolicy': 'null; no fallback; no partial-scope mean',
        'studentOrderPolicy': 'Existing tier-ordered numerical recipe retained; changed supervision tiers can change deterministic teaching-sample order.',
        'createdAt': datetime.now(timezone.utc).isoformat()}
    registration = {'sha256': canonical_hash(contract), 'contract': contract}
    path = args.output/'registration.json'
    write_immutable(path, registration)
    for task in tasks:
        write_immutable(args.output/'tasks'/(task['taskId'].replace('/', '__')+'.json'),
                        {**task, 'registration': identity(path), 'registrationSha256': registration['sha256']})
    print(json.dumps({'registration': str(path), 'sha256': registration['sha256'], 'tasks': len(tasks)}))


if __name__ == '__main__':
    main()
