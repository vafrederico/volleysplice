#!/usr/bin/env python3
"""Freeze original-corpus or balanced randomized source-composition tasks."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value

from analysis.neural_context_development import canonical_hash, identity, read, write_immutable
from analysis.neural_generalization_experiment import EPOCHS, MODELS, require, validate_membership
from analysis.neural_generalization_inputs import manifest_rows

SEEDS = (3407, 1729, 20260918)
SPLIT_SEEDS = (3407, 1729, 20260918, 20260923)
OLD = Path(private_value('private-reference-0095'))


def code_dependencies():
    """Bind the local import closure, without unrelated active research files."""
    pending = [REPO/name for name in ('scripts/register-neural-generalization.py',
               'scripts/run-neural-generalization.py', 'analysis/neural_generalization_experiment.py')]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                if node.level and path.parent.name == 'analysis':
                    names = [node.module.split('.')[0]] if node.module else [a.name for a in node.names]
                elif node.module and node.module.startswith('analysis.'):
                    names = [node.module.split('.')[1]]
            elif isinstance(node, ast.Import):
                names = [a.name.split('.')[1] for a in node.names if a.name.startswith('analysis.')]
            pending.extend(candidate for name in names if (candidate := REPO/'analysis'/(name+'.py')).exists()
                           and candidate not in found)
    return {str(path.relative_to(REPO)): identity(path) for path in sorted(found)}


def original_ids():
    source = read(OLD)
    return source, [r['id'] for key in ('exactRows', 'draftRows', 'coverageRows') for r in source[key]]


def validate_original_rows(rows, original):
    """The composition intervention must not silently revise old supervision."""
    by_id = {r['id']: r for r in rows}
    for key in ('exactRows', 'draftRows', 'coverageRows'):
        for old in original[key]:
            current = by_id[old['id']]
            for field in ('sourceGroup', 'contentSha256', 'environment', 'rallies', 'ignoredIntervals',
                          'keepTargets', 'gameWindow', 'annotation', 'targetContract'):
                require(current.get(field) == old.get(field), f'Original supervision changed: {old["id"]}/{field}')


def split_plans(rows, original, common_groups):
    """One balanced calibration source per draw; randomize the other sources."""
    old_ids = {r['id'] for key in ('exactRows', 'draftRows', 'coverageRows') for r in original[key]}
    groups = sorted({r['sourceGroup'] for r in original['exactRows']})
    require(len(groups) == 4, 'Expected four original exact training groups')
    trainable = [r for r in rows if 'fit' in r['eligibleRoles'] and not r['protected']
                 and r['sourceGroup'] not in common_groups]
    require({r['sourceGroup'] for r in trainable if r['labelTier'] == 'exact'} == set(groups),
            'Exact group population changed; freeze a revised split protocol before proceeding')
    result = []
    for held_group, split_seed in zip(groups, SPLIT_SEEDS, strict=True):
        remaining = [g for g in groups if g != held_group]
        order = np.random.default_rng(split_seed).permutation(remaining).tolist()
        for variant in ('original-medium', 'expanded-medium', 'expanded-large', 'expanded-wide-validation'):
            training_groups = order if variant == 'expanded-large' else order[:2]
            calibration_groups = [held_group] + (order[2:] if variant == 'expanded-wide-validation' else [])
            training = [r['id'] for r in trainable
                        if (r['sourceGroup'] in training_groups or r['sourceGroup'] not in groups)
                        and (variant != 'original-medium' or r['id'] in old_ids)]
            calibration = [r['id'] for r in rows if r['labelTier'] == 'exact' and r['sourceGroup'] in calibration_groups]
            result.append({'variant': variant, 'splitSeed': split_seed,
                           'designatedCalibrationGroup': held_group, 'randomRemainingGroupOrder': order,
                           'exactTrainGroups': training_groups, 'calibrationGroups': calibration_groups,
                           'trainIds': training, 'calibrationIds': calibration})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('original-corpus', 'randomized-variants'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--common-group', action='append', default=[])
    args = parser.parse_args()
    require(not (args.output/'registration.json').exists(), 'Registration already exists; use its frozen tasks')
    _, rows = manifest_rows(args.manifest)
    original, ids = original_ids()
    validate_original_rows(rows, original)
    qualification = read(args.qualification)
    require(qualification.get('passed') is True, 'Generalized-input qualification did not pass')
    base = {'manifest': identity(args.manifest), 'features': identity(args.features),
            'epochs': list(EPOCHS), 'commonEvaluationGroups': sorted(set(args.common_group)),
            'lossArm': 'short_boost', 'targetPaddingSeconds': 2, 'joinGapSeconds': 3,
            'floorsPercent': list(range(90, 101)), 'productionPromotionAllowed': False}
    plans = [{'variant': 'original-corpus', 'trainIds': ids, 'calibrationIds': [],
              'selectionDesign': 'original-eight-outer-oof-then-full-original-corpus-fit'}] if args.mode == 'original-corpus' else [
        {**plan, 'selectionDesign': 'disjoint-source-calibration-fixed-before-common-panel-outcomes'}
        for plan in split_plans(rows, original, set(args.common_group))]
    tasks = []
    for plan in plans:
        for model in MODELS:
            for seed in (SEEDS if args.mode == 'original-corpus' else (3407,)):
                suffix = f'split-{plan["splitSeed"]}' if 'splitSeed' in plan else f'seed-{seed}'
                task = {**base, **plan, 'model': model, 'seed': seed,
                        'taskId': f'{plan["variant"]}/{model}/{suffix}'}
                validate_membership(task, rows)
                tasks.append(task)
    sources = code_dependencies()
    args.output.mkdir(parents=True, exist_ok=True)
    archived = {}
    for name, reference in sources.items():
        destination = args.output/'registered-sources'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(reference['path']).read_bytes())
        archived[name] = identity(destination)
    protocol_copy = args.output/'protocol.md'
    protocol_copy.write_bytes(args.protocol.read_bytes())
    contract = {'kind': 'neural-source-composition-generalization-v1', 'mode': args.mode,
                'protocol': identity(protocol_copy), 'originalManifest': identity(OLD),
                'qualification': identity(args.qualification), 'manifest': base['manifest'], 'features': base['features'],
                'code': sources, 'archivedCode': archived, 'taskDigests': [canonical_hash(t) for t in tasks],
                'taskIds': [t['taskId'] for t in tasks], 'frozenPlans': plans,
                'trainingSeeds': list(SEEDS) if args.mode == 'original-corpus' else [3407],
                'splitSeeds': list(SPLIT_SEEDS) if args.mode != 'original-corpus' else [],
                'floorsPercent': list(range(90, 101)), 'paddingSeconds': [0, 1, 2, 3], 'joinGapSeconds': 3,
                'infeasiblePolicy': 'null; no fallback; no partial-scope mean',
                'selectionUsesAdditionalPanelOutcomes': False, 'createdAt': datetime.now(timezone.utc).isoformat()}
    registration = {'sha256': canonical_hash(contract), 'contract': contract}
    registration_path = args.output/'registration.json'
    write_immutable(registration_path, registration)
    for task in tasks:
        completed = {**task, 'registration': identity(registration_path), 'registrationSha256': registration['sha256']}
        write_immutable(args.output/'tasks'/(task['taskId'].replace('/', '__')+'.json'), completed)
    print(json.dumps({'registration': str(registration_path), 'sha256': registration['sha256'], 'tasks': len(tasks)}))


if __name__ == '__main__':
    main()
