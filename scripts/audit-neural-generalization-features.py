#!/usr/bin/env python3
"""Independently audit published expansion features and source/teacher isolation."""
from pathlib import Path
import argparse
import gc
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_generalization_feature_stage as stage
from analysis import neural_generalization_inputs as inputs
from analysis.neural_context_development import identity, read, write_immutable
from analysis.distillation_image_inputs import teaching_indexes
from analysis.schema import Interval, mask_for_times

require, verified = inputs.require, inputs.verified


def audit(root, mode, scope='all'):
    root = stage.environment(root)
    audit_plan_path = root/'feature-audit-plan.json'
    audit_plan = read(audit_plan_path)
    require(audit_plan['source'] == identity(__file__) and audit_plan['inputs'] == identity(root/'inputs.json')
            and audit_plan['stagePlan'] == identity(root/'stage-plan.json'), 'Independent feature audit registration differs')
    verified(audit_plan['frozenSource'])
    index_path = root/f'features-{"fit-" if scope == "fit" else ""}{mode}.json'
    index = read(index_path)
    manifest, rows = inputs.manifest_rows(root/'inputs.json')
    plan = stage.verify_plan(root/'stage-plan.json')
    require(index['stagePlan'] == identity(root/'stage-plan.json') and index['publicationMode'] == mode
            and index['publicationScope'] == scope,
            'Feature publication does not bind this stage plan/mode')
    expected_count = 32 if scope == 'fit' else 42
    require(len(rows) == len(plan['records']) == 42 and len(index['records']) == expected_count, 'Wrong expansion scope')
    require(index['labelsUsedForFeatures'] is False and index['protectedImageryUsedForTraining'] is False,
            'Invalid feature isolation claim')
    selected_rows = [r for r in rows if scope == 'all' or 'fit' in r['eligibleRoles']]
    entries = inputs.feature_entries(selected_rows, index_path)
    by_id = {r['id']: r for r in rows}
    engineering = read(root/'engineering-parity.json')
    require(engineering['passed'] is True and engineering['plan'] == identity(root/'stage-plan.json'),
            'CPU parity gate is missing or stale')
    if mode != 'av':
        from analysis import neural_generalization_feature_inference as encoders
        encoders.verify_plan(root)
        gate = read(root/'encoder-engineering.json')
        require(gate['passed'] is True and gate['encoderPlan'] == identity(root/'encoder-plan.json')
                and index['encoderPlan'] == identity(root/'encoder-plan.json'), 'Encoder gate is missing or stale')
    audit_rows, amendments = [], {}
    scheduling = []
    affinity_root = root/'int8-affinity-amendment-v1'
    if (affinity_root/'applied.json').exists():
        applied_ref = identity(affinity_root/'applied.json')
        applied = read(applied_ref['path'])
        protocol = read(verified(applied['plan']))
        pilot = read(verified(applied['pilot']))
        require(protocol['encoderPlan']['sha256'] == identity(root/'encoder-plan.json')['sha256']
                and protocol['queuePlan']['sha256'] == identity(root/'int8-queue-plan.json')['sha256']
                and protocol['ortThreads'] == 2 and protocol['batch'] == 1
                and protocol['candidateAffinity'] == [6, 8]
                and pilot['plan'] == applied['plan'] and pilot['passed'] is True
                and pilot['all32OutputsBitExact'] is True and pilot['storedFloat16AlsoBitExact'] is True
                and applied['existingWorkerReused'] is True
                and applied['graphAndBatchAndOrtThreadCountUnchanged'] is True
                and all(value == [6, 8] for value in applied['threadsAfter'].values()),
                'CPU scheduling amendment lacks bit-exact qualification')
        verified(protocol['source']); verified(protocol['stagingReceipt']); verified(protocol['graph'])
        scheduling.append({'protocol': applied['plan'], 'qualification': applied['pilot'], 'applied': applied_ref})
    gpu_grant_path = root.parent/'execution-resource-amendment-v2.json'
    if gpu_grant_path.exists():
        gpu_grant = read(gpu_grant_path)
        queue_plan = read(verified(gpu_grant['encoderQueuePlan']))
        require(gpu_grant['kind'] == 'generalization-resource-scheduling-amendment-v2'
                and gpu_grant['encoderQueuePlan']['sha256'] == identity(root/'gpu-queue-plan.json')['sha256']
                and gpu_grant['studentEncoderConcurrentExecutionAllowed'] is False
                and gpu_grant['trainingRecipeChanged'] is False and gpu_grant['wallTimesAreNotBenchmarks'] is True
                and queue_plan['cpuAffinity'] == [10, 11]
                and queue_plan['maximumTorchGpuReservedBytes'] == int(1.75*1024**3)
                and queue_plan['minimumBeforeRecord']['gpuFreeBytes'] == gpu_grant['minimumGpuFreeBeforeNewRecordBytes'],
                'Full encoder queue resource grant differs from the registered queue')
        verified(queue_plan['source'])
        scheduling.append({'gpuExecutionGrant': identity(gpu_grant_path), 'queuePlan': gpu_grant['encoderQueuePlan']})
    for source in plan['records']:
        if scope == 'fit' and not source['teaching']['allowed']:
            continue
        key = source['id']
        row, features = by_id[key], entries[key]
        for name, reference in features.items():
            if name == 'dino':
                for arm_reference in reference.values():
                    verified(arm_reference)
            else:
                verified(reference)
            if name == 'teacherTargets':
                target_receipt = read(verified(reference))
                verified(target_receipt['output'])
                verified(target_receipt['registration'])
        for name, original in source.get('reuse', {}).items():
            if name == 'dino':
                require(all(features[name][arm] == reference for arm, reference in original.items()),
                        'Original DINO cache was substituted')
            else:
                require(features[name] == original, 'Original feature/teacher reference was substituted: '+name)
        receipt_path = root/'staged'/key/'receipt.json'
        entry = next(r for r in index['records'] if r['recordingId'] == key)
        if entry['stagingReceipt'] is not None:
            verified(entry['stagingReceipt'])
            require(entry['stagingReceipt'] == identity(receipt_path), 'Published staging receipt differs')
        else:
            require(scope == 'fit' and mode != 'complete' and source.get('reuse')
                    and features == source['reuse']
                    and entry['reusedOriginalFeatureRegistration'] == identity(root/'reference-qualification-v1/report.json'),
                    'Missing published stage binding is not an intact original qualified input')
        if receipt_path.exists():
            receipt = read(receipt_path)
        else:
            require(scope == 'fit' and mode != 'complete' and source.get('reuse')
                    and entry['stagingReceipt'] is None and features == source['reuse']
                    and entry['reusedOriginalFeatureRegistration'] == identity(root/'reference-qualification-v1/report.json'),
                    'Unstaged feature reference is not an intact original qualified input')
            receipt = {'lineage': {'plan': identity(root/'stage-plan.json'), 'source': source},
                       'id': key, 'sourceGroup': row['sourceGroup'], 'labelsUsed': False,
                       'outputs': {}, 'dinoRgbChunks': [], 'allFeaturesReused': True}
        require(receipt['lineage'] == {'plan': identity(root/'stage-plan.json'), 'source': source}
                and receipt['labelsUsed'] is False and receipt['id'] == key
                and receipt['sourceGroup'] == row['sourceGroup'], 'Staging receipt binding differs')
        amendment_ref = receipt.get('samplingAmendment')
        if amendment_ref is not None:
            amendment_path = verified(amendment_ref)
            amendment = read(amendment_path)
            require(amendment['kind'] == 'bounded-source-timing-amendment-v1'
                    and amendment['basePlan'] == identity(root/'stage-plan.json')
                    and amendment['maximumNearestErrorSeconds'] == .25
                    and amendment['labelsUsedForAmendment'] is False and amendment['labelMasksChanged'] is False
                    and key in amendment['affectedRecordings'], 'Unregistered source timing amendment')
            for reference in list(amendment['code'].values()) + amendment['diagnostics'] + amendment['strictCompletedSources']:
                verified(reference)
            gate_path = amendment_path.parent/'engineering-parity.json'
            amended_gate = read(gate_path)
            require(amended_gate['passed'] is True and amended_gate['protocol'] == amendment_ref
                    and amended_gate['allSelectedPixelsAndFeatureArraysBitExact'] is True,
                    'Timing amendment full unaffected-source parity failed')
            verified(amended_gate['originalReceipt']); verified(amended_gate['amendedReceipt'])
            publication_path = amendment_path.parent/'publications'/(key+'.json')
            publication = read(publication_path)
            require(publication['protocol'] == amendment_ref and publication['engineering'] == identity(gate_path)
                    and publication['publishedReceipt'] == identity(receipt_path)
                    and read(verified(publication['actualReceipt'])) == receipt, 'Amended staging publication differs')
            if publication['preservedFailedDirectory'] is not None:
                preserved = publication['preservedFailedDirectory']
                require(preserved['entriesBefore'] == [] and Path(preserved['preservedPath']).is_dir()
                        and not list(Path(preserved['preservedPath']).iterdir()), 'Failed source folder was not preserved intact')
            amendments[amendment_ref['path']] = {'protocol': amendment_ref, 'engineering': identity(gate_path)}
        source_receipt = read(verified(source['sourceIdentityReceipt']))
        require(source_receipt['video'] == source['videoIdentity']
                and source_receipt['video']['sha256'] == row['contentSha256'], 'Source identity chain differs')
        stat = Path(source['videoIdentity']['path']).stat()
        require(stat.st_size == source_receipt['sourceSizeBytes']
                and stat.st_mtime_ns == source_receipt['sourceMtimeNs'], 'Previously hashed source stat changed')
        require(source['protected'] == row['protected'] and source['environment'] == row['environment'],
                'Source population metadata differs')
        allowed = ('fit' in row['eligibleRoles'] and row.get('consent', {}).get('train') is True and not row['protected']
                   and row['sourceGroup'] not in manifest['commonEvaluationGroups'])
        require(source['teaching']['allowed'] == allowed, 'Teaching authorization differs')
        for reference in receipt['outputs'].values():
            verified(reference)
        lengths = []
        for reference in receipt['dinoRgbChunks']:
            pixels = np.load(verified(reference), mmap_mode='r')
            require(pixels.dtype == np.uint8 and pixels.ndim == 4 and pixels.shape[1:] == (336, 336, 3)
                    and 0 < len(pixels) <= 128, 'Staged RGB chunk shape/storage differs')
            require(int(Path(reference['path']).stem) == sum(lengths), 'Staged chunk coverage is discontinuous')
            lengths.append(len(pixels))
            del pixels
        selection = None
        if 'selection' in receipt['outputs']:
            with np.load(verified(receipt['outputs']['selection']), allow_pickle=False) as stored:
                selection = {name: stored[name] for name in ('times', 'ordinals', 'selected_pts', 'frame_sha256')}
            times = selection['times']
            require(times.dtype == np.float64 and len(times) == receipt['samples']
                    and np.isfinite(times).all() and np.all(np.diff(times) > 0), 'Staged timeline differs')
            require(all(value.shape == times.shape for value in selection.values()), 'Staged selections differ in length')
            difference = np.diff(selection['ordinals'])
            require(np.all(difference >= 0 if amendment_ref is not None else difference > 0) and np.isfinite(selection['selected_pts']).all()
                    and all(len(value) == 64 for value in selection['frame_sha256']), 'Invalid selected frame identity')
            require(not lengths or sum(lengths) == len(times), 'RGB chunks do not cover selected ticks')
            if source['sampling'] == 'media-pts':
                maximum = .25 if amendment_ref is not None else .125
                require(np.max(abs(times-selection['selected_pts'])) <= maximum+1e-9, 'Media PTS error exceeds contract')
            if amendment_ref is not None:
                errors = selection['selected_pts']-times
                bad = np.flatnonzero(abs(errors) > .125+1e-9)
                gap = receipt['sourceGapDiagnostics']
                expected_ticks = [{'tickIndex': int(i), 'time': float(times[i]),
                                   'selectedPts': float(selection['selected_pts'][i]),
                                   'ordinal': int(selection['ordinals'][i]), 'errorSeconds': float(errors[i])} for i in bad]
                require(gap['affectedTicks'] == expected_ticks
                        and gap['duplicateOrdinalTickIndexes'] == (np.flatnonzero(difference == 0)+1).tolist()
                        and gap['maximumNearestErrorSeconds'] == float(abs(errors).max()), 'Amended source timing diagnostics differ')
        else:
            require(receipt.get('allFeaturesReused') is True and not lengths, 'Missing staged selection receipt')
        example = inputs.load_inference_examples(root/'inputs.json', index_path, recording_ids=[key])[0]
        require(example.valid.all() and not example.truth and not example.ignored and not example.targets.any(),
                'Inference sees label-derived information')
        if 'audiovisual' not in source.get('reuse', {}):
            require(features['audiovisual'] == receipt['outputs']['audiovisual']
                    and row['featureOrigin'] == 'opencv-media-pts-av104', 'Native or unbound AV substituted')
            require(np.array_equal(example.times, selection['times']), 'New AV/staged timeline differs')
        if row['scoringPolicy'] != 'none':
            labeled = inputs.make_examples_for_evaluation([example], root/'inputs.json', scoring_policy=row['scoringPolicy'])[0]
            require(labeled.valid is example.valid and labeled.values is example.values and labeled.times is example.times,
                    'Gold attachment changed inference inputs')
        teacher_count = 0
        if 'imageInput' in features:
            images = read(verified(features['imageInput']))
            require(images['id'] == key and images['sourceGroup'] == row['sourceGroup']
                    and images['contract']['source']['contentSha256'] == row['contentSha256'], 'Image source association differs')
            for reference in images['arrays'].values():
                verified(reference)
            data = np.load(verified(images['arrays']['images224']), mmap_mode='r')
            require(data.dtype == np.uint8 and data.shape == (images['frames'], 3, 224, 224), 'Mobile input geometry differs')
            with np.load(verified(images['arrays']['timing']), allow_pickle=False) as timing:
                image_times, teacher_ticks = timing['times'], timing['teaching_indexes']
                require(teacher_ticks.tolist() == images['contract']['teachingIndexes'], 'Image contract teaching indexes differ')
                require(len(image_times) == len(data) and timing['boxes'].shape == (len(data), 4)
                        and timing['quality'].shape == (len(data), 6), 'Mobile timing/geometry differs')
                if 'imageInput' not in source.get('reuse', {}):
                    require(np.array_equal(image_times, example.times[::2])
                            and np.array_equal(timing['selected_pts'], selection['selected_pts'][::2]), 'Mobile even-tick selection differs')
                if allowed:
                    valid = mask_for_times(example.times, tuple(Interval(r['start'], r['end'])
                                           for r in source['teaching']['excludedIntervals']))
                    window = source['teaching']['gameWindow']
                    if window:
                        valid &= (example.times >= window['start']) & (example.times < window['end'])
                    expected_indexes = teaching_indexes(image_times, example.times, valid, maximum=128)
                    require(np.array_equal(teacher_ticks, expected_indexes), 'Teacher sampling differs from authorized valid interval mask')
                    require('teaching336' in images['arrays'], 'Authorized teacher imagery absent')
                    teaching = np.load(verified(images['arrays']['teaching336']), mmap_mode='r')
                    require(teaching.dtype == np.uint8 and teaching.shape == (len(expected_indexes), 3, 336, 336), 'Teacher image geometry differs')
                    if 'imageInput' not in source.get('reuse', {}):
                        require(np.array_equal(timing['selected_frame_sha256'], selection['frame_sha256'][::2]),
                                'Mobile frame hashes differ from even4Hz source frames')
                        loaded, current = None, None
                        for destination, tick in enumerate(expected_indexes):
                            position = int(tick)*2
                            block = position//128
                            if block != current:
                                loaded = np.load(verified(receipt['dinoRgbChunks'][block]), mmap_mode='r')
                                current = block
                            require(np.array_equal(teaching[destination], loaded[position % 128].transpose(2, 0, 1)),
                                    'Teacher pixels differ from the declared same4Hz DINO frame')
                        del loaded
                    teacher_count = len(expected_indexes)
                    del teaching
                else:
                    require(not len(teacher_ticks) and 'teaching336' not in images['arrays']
                            and 'teacherTargets' not in features, 'Evaluation-only teacher imagery or targets leaked')
            del data
        if mode != 'av':
            arms = ('fp32', 'fp16', 'int8') if mode == 'complete' else ('fp32',)
            for arm in arms:
                reference = features['dino'][arm]
                with np.load(verified(reference), allow_pickle=False) as data:
                    times, tokens = data['timestamps'], data['tokens']
                    require(times.dtype == np.float64 and tokens.dtype == np.float16
                            and tokens.shape == (len(times), 10, 384) and np.isfinite(tokens).all(), 'DINO output geometry differs')
                    if arm not in source.get('reuse', {}).get('dino', {}):
                        require(np.array_equal(times, selection['times']), 'New DINO timeline differs')
                        folder = root/'encoders'/arm/key
                        encoded = read(folder/'receipt.json')
                        lineage = {'id': key, 'sourceGroup': row['sourceGroup'], 'arm': arm,
                                   'encoderPlan': identity(root/'encoder-plan.json'), 'staged': identity(receipt_path)}
                        require(all(encoded[k] == v for k, v in lineage.items()) and encoded['output'] == reference,
                                'DINO output receipt binding differs')
                        require(len(encoded['chunks']) == len(receipt['dinoRgbChunks']), 'DINO chunk count differs')
                        cursor = 0
                        for chunk_index, (rgb_ref, chunk_ref) in enumerate(zip(receipt['dinoRgbChunks'], encoded['chunks'])):
                            chunk = np.load(verified(chunk_ref), mmap_mode='r')
                            require(chunk.dtype == np.float16 and chunk.shape == (lengths[chunk_index], 10, 384),
                                    'DINO chunk geometry differs from its paired source RGB chunk')
                            binding = read(Path(chunk_ref['path']).with_suffix('.json'))
                            require(binding == {'lineage': lineage, 'input': rgb_ref, 'output': chunk_ref}, 'DINO chunk input binding differs')
                            require(np.array_equal(chunk, tokens[cursor:cursor+len(chunk)]), 'DINO publication differs from frozen chunks')
                            cursor += len(chunk)
                            del chunk
                        require(cursor == len(times), 'DINO chunks leave uncovered ticks')
                del tokens
            frozen_mobile = inputs.mobile.load_mobile_visual_cache(verified(features['mobile']))
            require(frozen_mobile.metadata['recordingId'] == key
                    and frozen_mobile.metadata['recordingContentSha256'] == row['contentSha256'], 'Mobile content association differs')
            require(inputs.mobile.align_mobile_features(frozen_mobile, example.times)['available'].all(), 'Mobile cache leaves AV ticks uncovered')
            if 'mobile' not in source.get('reuse', {}):
                mobile_receipt = read(root/'encoders/mobile'/key/'receipt.json')
                lineage = {'id': key, 'sourceGroup': row['sourceGroup'],
                           'encoderPlan': identity(root/'encoder-plan.json'), 'imageInput': features['imageInput']}
                require(all(mobile_receipt[k] == value for k, value in lineage.items())
                        and mobile_receipt['output'] == features['mobile'] and mobile_receipt['labelsUsed'] is False
                        and frozen_mobile.metadata['registeredInputs'] == lineage,
                        'New Mobile cache/encoder receipt lineage differs')
                encoder_plan = read(root/'encoder-plan.json')
                require(frozen_mobile.metadata['identity']['backbone']['checkpointSha256']
                        == encoder_plan['mobileCheckpoint']['sha256'], 'Mobile checkpoint differs from pinned encoder plan')
            del frozen_mobile
            if allowed:
                teacher = read(verified(features['teacherTargets']))
                require(teacher['id'] == key and teacher['sourceGroup'] == row['sourceGroup']
                        and teacher['input'] == images['arrays']['teaching336'] and teacher['labelsUsed'] is False,
                        'Teacher target lineage differs')
                targets = np.load(verified(teacher['output']), mmap_mode='r')
                require(targets.dtype == np.float32 and targets.shape == (teacher_count, 10, 384)
                        and np.isfinite(targets).all(), 'Teacher target geometry differs')
                verified(teacher['registration'])
                if 'teacherTargets' not in source.get('reuse', {}):
                    fp32_receipt = read(root/'encoders/fp32'/key/'receipt.json')
                    require(teacher['registration'] == identity(root/'encoder-plan.json')
                            and fp32_receipt['teacherTargets'] == features['teacherTargets']
                            and fp32_receipt['encoderPlan'] == teacher['registration'],
                            'New teacher target encoder/FP32 receipt binding differs')
                    with np.load(verified(features['dino']['fp32']), allow_pickle=False) as fp32_cache:
                        require(np.array_equal(targets.astype(np.float16), fp32_cache['tokens'][teacher_ticks*2]),
                                'New FP32 teacher targets differ from the selected same-frame DINO cache rows')
                del targets
        audit_rows.append({'id': key, 'sourceGroup': row['sourceGroup'], 'labelTier': row['labelTier'],
                           'stagingReceipt': identity(receipt_path) if receipt_path.exists() else None,
                           'sourceHashReceipt': source['sourceIdentityReceipt'],
                           'inferenceTicks': len(example.times), 'teacherFrames': teacher_count,
                           'samplingAmendment': amendment_ref,
                           'protected': row['protected'], 'allAvailableHashesVerified': True})
        del example
        gc.collect()
        print(json.dumps({'featureAudit': key, 'mode': mode}), flush=True)
    require(len(audit_rows) == expected_count, 'Audited feature scope differs')
    report = {'kind': 'independent-expansion-feature-audit-v1', 'passed': True, 'mode': mode, 'scope': scope,
              'auditPlan': identity(audit_plan_path),
              'inputs': identity(root/'inputs.json'), 'features': identity(index_path),
              'stagePlan': identity(root/'stage-plan.json'), 'cpuEngineering': identity(root/'engineering-parity.json'),
              'encoderEngineering': identity(root/'encoder-engineering.json') if mode != 'av' else None,
              'samplingAmendments': list(amendments.values()),
              'runtimeSchedulingAmendments': scheduling,
              'records': audit_rows, 'auditedRecordingCount': expected_count, 'allPublishedSourceIdentitiesAndAvailableArrayHashesChecked': True,
              'sourceBytesRehashedAtRegistrationAndStatsUnchanged': True, 'protectedAndPanelTeacherTargetsAbsent': True,
              'nativeAndroidAvNotSubstituted': True, 'inferenceMaskAlwaysAllTrue': True,
              'pixelParityScope': 'Full fixed engineering recording; all source/cache inventories audited, other source pixels are not independently decoded again.',
              'source': identity(__file__)}
    output = root/f'audit-features-{"fit-" if scope == "fit" else ""}{mode}.json'
    write_immutable(output, report)
    print(json.dumps({'passed': True, 'report': identity(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=stage.ROOT)
    parser.add_argument('--mode', choices=('av', 'core', 'complete'), required=True)
    parser.add_argument('--scope', choices=('all', 'fit'), default='all')
    args = parser.parse_args()
    audit(args.root, args.mode, args.scope)
