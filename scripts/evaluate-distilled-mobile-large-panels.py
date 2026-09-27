"""Evaluate fixed Large-student predictions without changing either selection.

Gold labels and production exposure come from an explicit private inventory.
Exact, draft, export-only, and beach results remain separate; no new threshold,
epoch, split, or seed is selected here. Generated details belong outside Git.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis import neural_generalization_inputs as inputs
from analysis import neural_recall_sweep as sweep
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_development import decode
from analysis.neural_generalization_results import manifest_example


POLICIES = {'exact-core': 'exact-rallies', 'draft-reviewed': 'reviewed-draft',
            'export-coverage': 'reviewed-export'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authoritative_gold(record, inventory, inventory_reference=None):
    """Use the inventory revision, never original training or proxy labels."""
    require(record['id'] == inventory['id'] and record['sourceGroup'] == inventory['sourceGroup']
            and record['durationSeconds'] == inventory['durationSeconds']
            and record['environment'] == inventory['environment'], 'Scoring source identity differs')
    if inventory.get('contentSha256') is None:
        # Eleven inventory rows predate full source hashing. Their subsequently
        # qualified manifest pins this exact inventory and its physical-media
        # identity; absence of an old hash does not authorize an ID-only join.
        require(inventory_reference is not None
                and record.get('inventoryEvidence') == inventory_reference
                and record.get('video') == inventory.get('video') and bool(record.get('video'))
                and record.get('mediaIdentity') == inventory.get('mediaIdentity')
                and bool(record.get('mediaIdentity')),
                'Unhashed inventory source lacks bound media identity')
    else:
        require(record['contentSha256'] == inventory['contentSha256'], 'Scoring source identity differs')
    require(not inventory.get('experimentalSupervision'), 'Approximate training proxies cannot be evaluation gold')
    if record['environment'] == 'beach':
        require(inventory['tier'] == 'completed-exact', 'Beach evaluation requires exact saved labels')
        scoring_policy = 'exact-core'
    else:
        scoring_policy = record['scoringPolicy']
    if scoring_policy == 'none':
        return None
    require(scoring_policy in POLICIES, 'Unknown scoring policy')
    tiers = {'exact-core': 'completed-exact', 'draft-reviewed': 'human-continuously-reviewed-draft',
             'export-coverage': 'reviewed-export-coverage'}
    require(inventory['tier'] == tiers[scoring_policy], 'Inventory label quality differs from scoring policy')
    gold = {key: inventory[key] for key in
            ('id', 'sourceGroup', 'contentSha256', 'durationSeconds', 'environment')}
    gold['contentSha256'] = record['contentSha256']
    gold.update(scoringPolicy=scoring_policy, rallies=inventory.get('rallies', []),
                ignoredIntervals=inventory.get('ignoredIntervals', []))
    if scoring_policy == 'export-coverage':
        require(inventory.get('keepTargets') and inventory.get('gameWindow'),
                'Reviewed export coverage or valid game window is missing')
        gold.update(keepTargets=inventory['keepTargets'], gameWindow=inventory['gameWindow'])
    else:
        require(gold['rallies'], 'Scored recording has no human intervals')
    return gold


def production_clean(inventory):
    exposure = inventory.get('productionExposure', {}).get('rallyPipeline', {})
    # Missing/uncertain lineage must never be treated as known-unexposed.
    return exposure.get('primaryTrainingClean') is True


def evaluate_panel(rows, policy, scope):
    if not rows:
        return dict(labelPolicy=policy, scope=scope, status='empty', evaluation=None,
                    recordingCount=0, sourceGroupCount=0, foundRallies=0,
                    whollyMissedSavedHumanRallies=None, recordings=[])
    evaluation = sweep.evaluate_rows(rows, policy)
    exact = policy == 'exact-rallies'
    details = {row['id']: row for row in evaluation.get('recordings', [])}
    counts = []
    for row in rows:
        missing = (details[row['id']]['guardrails']['primaryExportCoverage']['completeRallyLosses']
                   if exact else None)
        counts.append(dict(id=row['id'], sourceGroup=row['sourceGroup'],
                           foundRallies=len(row['predictions']), whollyMissedSavedHumanRallies=missing))
    return dict(labelPolicy=policy, scope=scope, status='available', evaluation=evaluation,
                recordingCount=len(rows), sourceGroupCount=len({row['sourceGroup'] for row in rows}),
                foundRallies=sum(row['foundRallies'] for row in counts),
                whollyMissedSavedHumanRallies=(sum(row['whollyMissedSavedHumanRallies'] for row in counts)
                                             if exact else None), recordings=counts)


def role_for(record, task, training_rows):
    groups = {row['id']: row['sourceGroup'] for row in training_rows}
    train_groups = {groups[key] for key in task['trainIds']}
    calibration_groups = {groups[key] for key in task['calibrationIds']}
    if record['id'] in task['trainIds']:
        return 'training'
    if record['sourceGroup'] in train_groups:
        return 'training-related-source'
    if record['id'] in task['calibrationIds']:
        return 'calibration'
    if record['sourceGroup'] in calibration_groups:
        return 'calibration-related-source'
    if (record['sourceGroup'] in task['commonEvaluationGroups']
            and record.get('scoringPolicy') == 'exact-core'):
        return 'common-panel-ui-selection'
    return 'unseen-by-this-fit-and-selection'


def validate_choice(root, plan, selection_document, chosen):
    plan_ref = identity(root / 'plan.json')
    require(selection_document['plan'] == plan_ref and chosen['floorPercent'] == 99,
            'Selected model does not belong to the strict99 plan')
    matches = [task for task in plan['tasks'] if task['variant'] == chosen['variant']
               and task.get('splitSeed', task['seed']) == chosen['draw']]
    require(len(matches) == 1, 'Selected fit ownership is ambiguous')
    task = matches[0]
    folder = root / 'fits' / task['variant'] / f"split-{chosen['draw']}"
    saved = read(folder / 'selection.json')
    digest = sweep.canonical({'plan': plan_ref, 'task': task})
    require(saved['task'] == task and saved['plan'] == plan_ref and saved['contractSha256'] == digest,
            'Fit selection identity differs')
    require(len(saved['floors']) == 1 and saved['floors'][0]['floorPercent'] == 99
            and saved['floors'][0]['feasible'] is True
            and saved['floors'][0]['selected'] == chosen['setting'], 'Frozen operating point differs')
    completed = read(folder / 'temporal/completed.json')
    student = read(folder / 'student/completed.json')
    require(completed['contractSha256'] == student['contractSha256'] == digest
            and completed['seed'] == student['seed'] == chosen['seed'] == task['seed'],
            'Temporal or student ownership differs')
    require(completed['model']['inputDimension'] == 3952 and completed['lossArm'] == 'short_boost'
            and completed['kind'] == 'mobile_tcn', 'Selected model recipe differs')
    require(student['weights'] == saved['studentWeights'], 'Selected student weights differ')
    inputs.verified(student['weights'])
    epoch = str(chosen['setting']['epoch'])
    filename = f'weights-{epoch}.npz'
    weights = dict(path=str(folder / 'temporal' / filename), sha256=completed['artifacts'][filename])
    inputs.verified(weights)
    return task, folder, weights, student['weights']


def prediction(root, folder, record, chosen, weights, student_weights, features=None):
    """Verify and decode saved probabilities before accepting human labels."""
    output = folder / 'inference' / (record['id'] + '.npz')
    require(output.resolve().parent == (folder / 'inference').resolve(), 'Unsafe recording output identity')
    receipt_path = output.with_suffix('.json')
    receipt = read(receipt_path)
    epoch, decoder = str(chosen['setting']['epoch']), chosen['setting']['decoder']
    require(receipt['recordingId'] == record['id']
            and receipt['contentSha256'] == record['contentSha256']
            and receipt['durationSeconds'] == record['durationSeconds']
            and receipt['plan'] == identity(root / 'plan.json')
            and receipt['labelsUsed'] is False and receipt['ignoredIntervalsUsed'] is False,
            'Prediction source or label isolation differs')
    require(receipt['weights'] == {epoch: weights} and receipt['studentWeights'] == student_weights
            and receipt['decoders'] == {epoch: decoder}, 'Prediction checkpoint or decoder differs')
    if features is not None:
        require(receipt['audiovisual'] == features['audiovisual']
                and receipt['imageInput'] == features['imageInput'], 'Prediction feature ownership differs')
    require(Path(receipt['output']['path']).resolve() == output.resolve(), 'Prediction output owner differs')
    inputs.verified(receipt['output'])
    image = read(inputs.verified(receipt['imageInput']))
    require(image['id'] == record['id'] and image['sourceGroup'] == record['sourceGroup']
            and image['contract']['source']['contentSha256'] == record['contentSha256'],
            'Image source identity differs')
    with np.load(output, allow_pickle=False) as archive:
        require(set(archive.files) == {'times', f'epoch_{epoch}'}, 'Prediction tensor inventory differs')
        times, scores = archive['times'].copy(), archive[f'epoch_{epoch}'].copy()
    shell = {key: record[key] for key in ('id', 'sourceGroup', 'durationSeconds', 'environment', 'featureOrigin')}
    example = inputs.example_from_row(shell, {'audiovisual': receipt['audiovisual']}, inference=True)
    require(np.array_equal(times, example.times) and not example.truth and not example.ignored
            and example.valid.all(), 'Prediction AV timeline or label isolation differs')
    require(scores.shape == (len(times), 4) and np.isfinite(scores).all()
            and np.all((scores >= 0) & (scores <= 1)), 'Invalid saved probabilities')
    # The AV loader validates its metadata duration against this source, but
    # frozen inference clips decoded boundaries to the catalog duration. Replay
    # that exact contract; small metadata precision differences are not edits.
    example = replace(example, duration=record['durationSeconds'])
    rallies = [dict(start=value.start, end=value.end) for value in decode(example, scores, decoder)]
    require(receipt['decodedRallies'] == {epoch: rallies}, 'Saved rally boundaries differ from frozen decoding')
    return times, rallies, identity(receipt_path)


def run(root, inventory_path):
    plan = read(root / 'plan.json')
    selected = read(root / 'evaluation.json')
    catalog = read(root / 'catalog-manifest.json')
    require(plan['kind'] == 'distilled-mobile-large-v1' and plan['floorsPercent'] == [99],
            'Unknown experiment plan')
    require(selected['kind'] == 'distilled-mobile-large-common-selection-v1'
            and selected['commonUnseenIsSelectionData'] is True, 'Unknown frozen selection')
    require(catalog['kind'] == 'distilled-large-inference-catalog-v1', 'Unknown inference catalog')
    records = catalog['records']
    require(len(records) == len({row['id'] for row in records}) == 44, 'Expected all44 unique predictions')
    inventory_rows = read(inventory_path)['records']
    inventory = {row['id']: row for row in inventory_rows}
    require(len(inventory) == len(inventory_rows) and {row['id'] for row in records} <= inventory.keys(),
            'Inventory source membership differs')
    require({row['mode'] for row in selected['selected']} == {'f1', 'recall'}
            and len(selected['selected']) == 2, 'Both frozen UI selections are required')
    base_records = inputs.manifest_rows(inputs.verified(plan['inferenceManifest']))[1]
    feature_entries = inputs.feature_entries(base_records, inputs.verified(plan['features']))
    for record in records:
        if record['environment'] == 'beach':
            feature_entries[record['id']] = record['features']
    require(set(feature_entries) == {row['id'] for row in records}, 'Inference feature population differs')
    result = dict(kind='distilled-mobile-large-all-video-evaluation-v1', plan=identity(root / 'plan.json'),
                  selections=identity(root / 'evaluation.json'), catalog=identity(root / 'catalog-manifest.json'),
                  inventory=identity(inventory_path), sourceCode=identity(__file__),
                  inferenceRecordingCount=44, targetPaddingSeconds=2, paddingSeconds=[0, 1, 2, 3],
                  joinGapSeconds=3, labelsUsedForInference=False, selectionsChanged=False,
                  allAndNoProductionTrainingScopesExcludeBeach=True, commonUnseenIsSelectionData=True,
                  goldRevision='Explicit inventory; historical training labels and export proxies are not evaluation gold',
                  foundRalliesDefinition='All decoded core events, including ignored time; no padding or export-gap merging',
                  whollyMissedDefinition='Exact-label original rallies with zero nonignored core retained after2s padding and gaps strictly under3s joined',
                  productionFilterDefinition='Recorded rally-pipeline primaryTrainingClean=true; excludes direct and related-source fitting exposure, not necessarily calibration exposure',
                  models=[])
    for chosen in selected['selected']:
        task, folder, weights, student_weights = validate_choice(root, plan, selected, chosen)
        training_rows = read(inputs.verified(task['manifest']))['records']
        by_policy = {policy: [] for policy in POLICIES.values()}
        beach, all_counts, receipts = [], [], []
        for record in records:
            times, rallies, receipt = prediction(root, folder, record, chosen, weights, student_weights,
                                                feature_entries[record['id']])
            receipts.append(receipt)
            source = inventory[record['id']]
            gold = authoritative_gold(record, source, identity(inventory_path))
            policy = POLICIES[gold['scoringPolicy']] if gold is not None else None
            counts = dict(id=record['id'], sourceGroup=record['sourceGroup'],
                          environment=record['environment'], labelPolicy=policy, foundRallies=len(rallies),
                          fitRole=role_for(record, task, training_rows),
                          productionTrainingClean=production_clean(source),
                          productionExposure=source.get('productionExposure'), whollyMissedSavedHumanRallies=None)
            if gold is not None:
                example = manifest_example(gold, times)
                metric_row = sweep.serial_rows([example.row([])])[0]
                metric_row['predictions'] = rallies
                if policy != 'exact-rallies':
                    metric_row[sweep.GOLD_FIELDS[policy]] = metric_row.pop('rallies')
                (beach if record['environment'] == 'beach' else by_policy[policy]).append(metric_row)
            all_counts.append(counts)
        panels = []
        for policy, rows in by_policy.items():
            panels.append(evaluate_panel(rows, policy, 'all'))
            panels.append(evaluate_panel([row for row in rows if production_clean(inventory[row['id']])],
                                         policy, 'no-production-training'))
        panels.append(evaluate_panel(beach, 'exact-rallies', 'beach'))
        exact_counts = {row['id']: row['whollyMissedSavedHumanRallies'] for panel in panels
                        if panel['labelPolicy'] == 'exact-rallies' and panel['scope'] in ('all', 'beach')
                        for row in panel['recordings']}
        for row in all_counts:
            row['whollyMissedSavedHumanRallies'] = exact_counts.get(row['id'])
        result['models'].append(dict(mode=chosen['mode'], variant=chosen['variant'], draw=chosen['draw'],
            trainingSeed=chosen['seed'], recallTargetPercent=99, setting=chosen['setting'],
            panels=panels, recordings=all_counts, inferenceReceipts=receipts,
            fitRoleCounts=dict(Counter(row['fitRole'] for row in all_counts)),
            scoredRecordingCount=sum(row['labelPolicy'] is not None for row in all_counts),
            unscoredRecordingCount=sum(row['labelPolicy'] is None for row in all_counts),
            missingProductionExposureCount=sum(row['productionExposure'] is None for row in all_counts)))
    write_immutable(root / 'all-video-evaluation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.experiment, args.inventory)
    print(json.dumps(dict(phase='all-video-evaluation-complete', models=len(result['models']),
                          recordings=result['inferenceRecordingCount'])), flush=True)


if __name__ == '__main__':
    main()
