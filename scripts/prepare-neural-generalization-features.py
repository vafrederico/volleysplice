#!/usr/bin/env python3
"""Register and stage the bounded non-beach feature expansion on NAS."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_generalization_feature_stage as stage
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_inputs import require, POLICIES, verified

ROOT = stage.ROOT
PRIOR = stage.NAS/'2026-09-23-recall-distillation'
INVENTORY = ROOT.parent/'inventory-v1/inventory-v2.json'
ENGINEERING = private_value('recording-037')
PANEL_GROUPS = {private_value('source-group-008'), private_value('source-group-001')}


def normalize_rows(inventory):
    legacy = read(ROOT/'reference-qualification-v1/inputs.json')
    old = {r['id']: r for r in legacy['records']}
    tiers = {'completed-exact': 'exact', 'human-continuously-reviewed-draft': 'draft',
             'reviewed-export-coverage': 'coverage'}
    rows = []
    for row in inventory['records']:
        if row['environment'] == 'beach':
            continue
        receipt = read(ROOT/'source-identities-v1'/(row['id']+'.json'))
        value = copy.deepcopy(old.get(row['id'], row))
        require(receipt['video']['path'] == value['video'], 'Source hash registry uses another representation')
        value['contentSha256'] = receipt['video']['sha256']
        if row['id'] not in old:
            tier = tiers.get(row['tier'], 'unscored')
            value['labelTier'], value['scoringPolicy'] = tier, POLICIES[tier]
            value['featureOrigin'] = 'opencv-media-pts-av104'
            value['nativeFeatureEvidence'] = value.pop('featureCaches', {})
            value['featureCaches'] = {}
            roles = ['infer']
            if tier != 'unscored':
                roles.append('evaluate')
            if (row['eligibleForVariantTrainingBeforePanelReservation'] and not row['protected']
                    and row['sourceGroup'] not in PANEL_GROUPS):
                roles.append('fit')
                if tier == 'exact':
                    roles.append('calibrate')
            value['eligibleRoles'] = roles
            if tier == 'coverage':
                require(row['coverageReview']['exhaustiveKeptDiscardedReviewConfirmed'] is True,
                        'Reviewed export coverage authorization absent')
                value['annotation']['continuousVideoReviewed'] = True
                value['annotation']['independentEndpointGold'] = False
                value['annotation']['reviewScope'] = 'All kept/discarded export coverage confirmed by user; no exact endpoint certification'
                value['targetContract']['negativesOutsideKeepAuthorizedByFullManualReview'] = True
        value['inventoryEvidence'] = identity(INVENTORY)
        value['goldPolicy'] = 'Original18 labels unchanged' if row['id'] in old else 'Frozen current inventory labels by declared quality tier'
        rows.append(value)
    require(len(rows) == 42 and set(old) <= {r['id'] for r in rows}, 'Unexpected non-beach expansion inventory')
    return rows


def register():
    import cv2
    root = stage.environment(ROOT)
    inventory = read(INVENTORY)
    require(identity(INVENTORY)['sha256'] == 'fcb6263fbc2b1ce8374d0ca411e25eece6ee029ffebac134b5cc087e69293a85',
            'Inventory revision changed')
    rows = normalize_rows(inventory)
    manifest = {'kind': 'neural-generalization-inputs-v1', 'inventory': identity(INVENTORY),
        'original18': identity(ROOT/'reference-qualification-v1/inputs.json'),
        'commonEvaluationGroups': sorted(PANEL_GROUPS), 'records': rows}
    write_immutable(root/'inputs.json', manifest)
    existing = {r['recordingId']: r['features'] for r in read(ROOT/'reference-qualification-v1/features.json')['records']}
    precision_protocol = read(PRIOR/'dino-precision-v1/protocol.json')
    precise_ids = {r['source']['id'] for r in precision_protocol['records']}
    sources = []
    for row in rows:
        key = row['id']
        prior = copy.deepcopy(existing.get(key, {}))
        if key in precise_ids:
            for arm in ('fp16', 'int8'):
                receipt = read(PRIOR/'dino-precision-v1'/arm/key/'report.json')
                prior['dino'][arm] = receipt['cache']
        source_identity = read(root/'source-identities-v1'/(key+'.json'))
        if key in existing:
            sampler = 'media-pts' if row['labelTier'] == 'coverage' else 'historical-ordinal'
        else:
            sampler = 'media-pts'
        allowed = 'fit' in row['eligibleRoles'] and not row['protected'] and row['sourceGroup'] not in PANEL_GROUPS
        sources.append({'id': key, 'sourceGroup': row['sourceGroup'], 'environment': row['environment'],
            'durationSeconds': row['durationSeconds'], 'videoIdentity': source_identity['video'],
            'sourceIdentityReceipt': identity(root/'source-identities-v1'/(key+'.json')),
            'roi': row.get('roi'), 'protected': row['protected'], 'sampling': sampler, 'reuse': prior,
            'teaching': {'allowed': allowed,
                         'excludedIntervals': row.get('ignoredIntervals', []) if allowed else [],
                         'gameWindow': row.get('gameWindow') if allowed and row['labelTier'] == 'coverage' else None},
            'engineeringReplay': key == ENGINEERING})
    modules = ['neural_generalization_feature_stage.py', 'neural_generalization_inputs.py',
               'features.py', 'config.py', 'schema.py', 'mobile_visual_features.py',
               'dinov2_embeddings.py', 'distillation_image_inputs.py', 'neural_context_development.py']
    code = {name: identity(REPO/'analysis'/name) for name in modules}
    code[Path(__file__).name] = identity(__file__)
    plan = {'kind': 'generalization-label-blind-feature-plan-v1', 'labelsUsedForFeatures': False,
        'trainingValidityUsedForTeachingSelectionOnly': True, 'code': code,
        'parents': [identity(INVENTORY), identity(root/'inputs.json'),
                    identity(root/'reference-qualification-v1/report.json'),
                    identity(PRIOR/'dino-precision-v1/publication-v1/manifest.json')],
        'records': sources, 'engineeringRecord': ENGINEERING,
        'sampling': 'New inputs use exact4Hz nearest-media-PTS frames; mobile224 takes even2Hz ticks. Historical sources preserve original sampler and feature bytes.',
        'imagePreprocessing': 'Existing ROI, AV104 functions, RGB336 DINO letterbox, CHW224 MobileNet letterbox/quality, unchanged normalization/pooling.',
        'teacherPolicy': 'At most128 evenly spaced review-valid frames, only fit-eligible pool; no protected/Sep17 or other evaluation-only teacher targets.',
        'precisionPolicy': 'Same staged pixels; FP32/FP16 CUDA encoder plus nativeCPU batch1 dynamic mixed INT8 graph. CPU INT8 is not browser-qualified; no physical-phone claim.',
        'runtime': {'opencv': cv2.__version__, 'opencvBuildSha256': hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()},
        'storage': 'All new pixels/caches/temporary files and reports on direct NAS. OS swap remains external to experiment control.'}
    write_immutable(root/'stage-plan.json', plan)
    snapshots = root/'registered-stage-sources'
    snapshots.mkdir(exist_ok=True)
    for name, reference in code.items():
        target = snapshots/name
        if target.exists():
            require(target.read_bytes() == Path(reference['path']).read_bytes(), 'Snapshot changed')
        else:
            with target.open('xb') as stream:
                stream.write(Path(reference['path']).read_bytes())
    print(json.dumps({'plan': identity(root/'stage-plan.json'), 'inputs': identity(root/'inputs.json'),
                      'records': len(rows), 'sourceHours': sum(r['durationSeconds'] for r in rows)/3600,
                      'trainingSources': sum('fit' in r['eligibleRoles'] for r in rows)}), flush=True)


def qualify():
    receipt = stage.stage_record(ROOT/'stage-plan.json', ENGINEERING)
    source = receipt['lineage']['source']
    old = source['reuse']
    with np.load(verified(old['audiovisual']), allow_pickle=False) as a, np.load(verified(receipt['outputs']['audiovisual']), allow_pickle=False) as b:
        for key in ('times', 'values', 'names'):
            require(np.array_equal(a[key], b[key]), 'AV same-pixel formula replay differs: '+key)
    original_image = read(verified(old['imageInput']))
    staged_image = read(verified(receipt['outputs']['imageInput']))
    for key in ('images224', 'teaching336'):
        a = np.load(verified(original_image['arrays'][key]), mmap_mode='r')
        b = np.load(verified(staged_image['arrays'][key]), mmap_mode='r')
        require(a.shape == b.shape and np.array_equal(a, b), 'Staged image bytes differ: '+key)
    with np.load(verified(original_image['arrays']['timing']), allow_pickle=False) as a, np.load(verified(staged_image['arrays']['timing']), allow_pickle=False) as b:
        for key in ('times', 'selected_pts', 'quality', 'boxes', 'teaching_indexes'):
            require(np.array_equal(a[key], b[key]), 'Staged mobile timeline/quality differs: '+key)
    result = {'kind': 'new-PTS-wrapper-legacy-full-record-parity-v1', 'passed': True,
        'plan': identity(ROOT/'stage-plan.json'), 'stagingReceipt': identity(ROOT/'staged'/ENGINEERING/'receipt.json'),
        'recordingId': ENGINEERING, 'fullAv104TimesValuesNamesBitExact': True,
        'fullMobile224AndTeacher336PixelsBitExact': True, 'allMobileTimingQualityBoxesTeachingIndexesBitExact': True,
        'labelsUsedForFeatures': False, 'source': identity(__file__)}
    write_immutable(ROOT/'engineering-parity.json', result)
    print(json.dumps({'passed': True, 'report': identity(ROOT/'engineering-parity.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'qualify', 'stage'))
    parser.add_argument('--id', action='append', default=[])
    args = parser.parse_args()
    if args.phase == 'register':
        register()
    elif args.phase == 'qualify':
        qualify()
    else:
        require(read(ROOT/'engineering-parity.json')['passed'] is True, 'CPU staging parity must pass first')
        plan = stage.verify_plan(ROOT/'stage-plan.json')
        for source in plan['records']:
            if not args.id or source['id'] in args.id:
                stage.stage_record(ROOT/'stage-plan.json', source['id'])


if __name__ == '__main__':
    main()
