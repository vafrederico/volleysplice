#!/usr/bin/env python3
"""Final independent interval audit for all54 short-boost transfer result cells.

Interval arithmetic below uses an endpoint sweep, never the shared padding,
joining, subtraction or metric helpers. Only reconstruction of raw decoded cuts
uses the explicitly permitted frozen neural decoder. No fitting or selection.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0084'))
MANIFEST = ROOT/'manifest-pts-v1.json'
PADDINGS = (0, 1, 2, 3)
SLICES = ('all', 'duration_le_2s', 'duration_le_3s', 'duration_2_to_3s', 'duration_gt_3s', 'ace', 'service_fault')
SUM_FIELDS = ('paddedPrecisionIntersectionSeconds', 'paddedModelExportSeconds',
              'coreRecallIntersectionSeconds', 'coreHumanSeconds', 'paddedHumanExportSeconds')
COUNT_FIELDS = ('originalRallies', 'evaluableRallies', 'fullyIgnoredRallies',
                'completeRallyLosses', 'partialRallyLosses', 'fullyCoveredRallies')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {'path': str(path), 'sha256': digest(path), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def number(value):
    require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value), 'Nonfinite or nonnumeric boundary')
    return float(value)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), f'{label}: {actual} != {expected}')


def verify_serialized_gold(stored, gold):
    """Check exact Interval.to_dict fields; annotations remain hash-bound separately."""
    expected = []
    for row in gold:
        start, end = number(row['start']), number(row['end'])
        require(start < end, 'Gold interval start must precede end')
        tags = row.get('tags', [])
        require(isinstance(tags, (list, tuple)) and all(isinstance(tag, str) and tag.strip() for tag in tags),
                'Invalid gold tags')
        item = {'start': start, 'end': end}
        if tags:
            item['tags'] = list(tags)
        expected.append(item)
    require(stored == expected, 'Serialized prediction gold revision differs')


def normalize_intervals(rows, duration):
    """Clip BEFORE padding, retaining original event tags and sorted identities."""
    result = []
    for row in rows:
        start, end = number(row['start']), number(row['end'])
        require(start < end, 'Interval start must precede end')
        tags = row.get('tags', [])
        require(isinstance(tags, (list, tuple)) and all(isinstance(t, str) and t.strip() for t in tags), 'Invalid tags')
        start, end = max(0., start), min(duration, end)
        if start < end:
            result.append({'start': start, 'end': end, 'tags': list(dict.fromkeys(tags))})
    return sorted(result, key=lambda r: (r['start'], r['end']))


def parse_record(row):
    duration = number(row['durationSeconds'])
    require(duration > 0, 'Duration must be positive')
    require(all(isinstance(row[k], str) and row[k] for k in ('id', 'sourceGroup')), 'Missing recording identity')
    parsed = {'id': row['id'], 'sourceGroup': row['sourceGroup'], 'duration': duration}
    for key in ('rallies', 'predictions', 'ignoredIntervals'):
        parsed[key] = normalize_intervals(row.get(key, []), duration)
    require(all(a['end'] <= b['start'] for a, b in zip(parsed['rallies'], parsed['rallies'][1:])), 'Original rallies overlap')
    return parsed


def ranges(rows):
    return [(r['start'], r['end']) for r in rows]


def boolean_intervals(left, right=(), operation='union'):
    """Atomic endpoint sweep with independent left/right activity counters."""
    require(operation in ('union', 'intersection', 'difference'), 'Unknown interval operation')
    changes = {}
    for side, collection in enumerate((left, right)):
        for start, end in collection:
            start, end = number(start), number(end)
            require(start <= end, 'Reversed interval')
            if start == end:
                continue
            changes.setdefault(start, [0, 0])[side] += 1
            changes.setdefault(end, [0, 0])[side] -= 1
    points = sorted(changes)
    active = [0, 0]
    output = []
    for index, start in enumerate(points[:-1]):
        active[0] += changes[start][0]
        active[1] += changes[start][1]
        keep = (active[0] > 0 or active[1] > 0) if operation == 'union' else (
            active[0] > 0 and active[1] > 0 if operation == 'intersection' else active[0] > 0 and active[1] == 0)
        if keep:
            end = points[index+1]
            if output and output[-1][1] == start:
                output[-1] = (output[-1][0], end)
            else:
                output.append((start, end))
    return output


def duration(intervals):
    return sum(end-start for start, end in intervals)


def padded_union(intervals, video_duration, padding, join_gap=3.):
    require(number(video_duration) > 0 and number(padding) >= 0 and number(join_gap) >= 0, 'Invalid padding settings')
    expanded = [(max(0., start-padding), min(video_duration, end+padding)) for start, end in intervals]
    union = boolean_intervals([(start, end) for start, end in expanded if start < end])
    result = []
    for start, end in union:
        if result and 0 < start-result[-1][1] < join_gap:
            result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return result


def export_union(record, key, padding):
    joined = padded_union(ranges(record[key]), record['duration'], padding, 3.)
    return boolean_intervals(joined, ranges(record['ignoredIntervals']), 'difference')


def recording_metric(record, padding):
    ignored = ranges(record['ignoredIntervals'])
    core = boolean_intervals(ranges(record['rallies']), ignored, 'difference')
    require(duration(core) > 0, 'Recording has no evaluable core gold time')
    model, human = (export_union(record, key, padding) for key in ('predictions', 'rallies'))
    return {'id': record['id'], 'paddedPrecisionIntersectionSeconds': duration(boolean_intervals(model, human, 'intersection')),
            'paddedModelExportSeconds': duration(model),
            'coreRecallIntersectionSeconds': duration(boolean_intervals(model, core, 'intersection')),
            'coreHumanSeconds': duration(core), 'paddedHumanExportSeconds': duration(human),
            'inputCropCount': len(record['predictions']), 'outputCropCount': len(model)}


def pooled_metric(records, padding):
    require(records and len({r['id'] for r in records}) == len(records), 'Empty/duplicate recording population')
    values = [recording_metric(record, padding) for record in records]
    sums = {field: sum(row[field] for row in values) for field in SUM_FIELDS}
    precision = sums['paddedPrecisionIntersectionSeconds']/sums['paddedModelExportSeconds'] if sums['paddedModelExportSeconds'] else 0.
    recall = sums['coreRecallIntersectionSeconds']/sums['coreHumanSeconds']
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return {**sums, 'P_pad': precision, 'R_core': recall, 'F1_padP_coreR': f1,
            'paddingSecondsBeforeAndAfter': float(padding), 'joinGapSeconds': 3.,
            'exportDurationDifferenceSeconds': sums['paddedModelExportSeconds']-sums['paddedHumanExportSeconds'],
            'inputCropCount': sum(r['inputCropCount'] for r in values),
            'outputCropCount': sum(r['outputCropCount'] for r in values), 'recordingCount': len(records),
            'recordings': [{'id': row['id'], **{key: row[key] for key in SUM_FIELDS}} for row in values]}


def event_coverage(records, scope):
    require(scope in ('coreCoverage', 'primaryExportCoverage'), 'Unknown coverage scope')
    events = []
    original = 0
    for record in records:
        ignored = ranges(record['ignoredIntervals'])
        output = (export_union(record, 'predictions', 2) if scope == 'primaryExportCoverage' else
                  boolean_intervals(ranges(record['predictions']), ignored, 'difference'))
        original += len(record['rallies'])
        for index, rally in enumerate(record['rallies']):
            core = boolean_intervals([(rally['start'], rally['end'])], ignored, 'difference')
            seconds = duration(core)
            if seconds <= 0:
                continue
            # Compute missed atomic spans independently; matching the tolerance
            # definition is necessary even when endpoint arithmetic rounds.
            missed = duration(boolean_intervals(core, output, 'difference'))
            retained = max(0., seconds-missed)
            full, lost = abs(missed) <= 1e-9, retained <= 1e-9
            events.append({'recordingId': record['id'], 'truthIndex': index,
                           'start': rally['start'], 'end': rally['end'], 'tags': rally['tags'],
                           'evaluableCoreSeconds': seconds, 'retainedCoreSeconds': retained,
                           'coverage': retained/seconds, 'fullyCovered': full,
                           'completelyLost': lost, 'partiallyLost': not full and not lost})
    seconds = sum(r['evaluableCoreSeconds'] for r in events)
    retained = sum(r['retainedCoreSeconds'] for r in events)
    return {'originalRallies': original, 'evaluableRallies': len(events), 'fullyIgnoredRallies': original-len(events),
            'completeRallyLosses': sum(r['completelyLost'] for r in events),
            'partialRallyLosses': sum(r['partiallyLost'] for r in events),
            'fullyCoveredRallies': sum(r['fullyCovered'] for r in events),
            'evaluableCoreSeconds': seconds, 'retainedCoreSeconds': retained,
            'coreRecall': retained/seconds if seconds else 0., 'rallies': events}


def belongs(row, name):
    length = row['end']-row['start']
    return {'all': True, 'duration_le_2s': length <= 2, 'duration_le_3s': length <= 3,
            'duration_2_to_3s': 2 < length <= 3, 'duration_gt_3s': length > 3,
            'ace': 'ace' in row['tags'], 'service_fault': 'service-fault' in row['tags']}[name]


def slices(coverage):
    result = {}
    for name in SLICES:
        rows = [row for row in coverage['rallies'] if belongs(row, name)]
        seconds = sum(r['evaluableCoreSeconds'] for r in rows)
        retained = sum(r['retainedCoreSeconds'] for r in rows)
        complete, partial = sum(r['completelyLost'] for r in rows), sum(r['partiallyLost'] for r in rows)
        result[name] = {'evaluableRallies': len(rows), 'evaluableCoreSeconds': seconds, 'retainedCoreSeconds': retained,
                        'completeRallyLosses': complete, 'partialRallyLosses': partial,
                        'incompleteRallyLosses': complete+partial, 'coreRecall': retained/seconds if seconds else None}
    return result


def compare_metric(actual, expected, label):
    require(set(actual) == set(expected), label+' metric schema differs')
    for key in actual:
        if key == 'recordings':
            by_id = {r['id']: r for r in expected[key]}
            require(len(by_id) == len(expected[key]) == len(actual[key]), label+' recording inventory differs')
            for row in actual[key]:
                require(row['id'] in by_id, label+' recording missing')
                for field in SUM_FIELDS:
                    close(row[field], by_id[row['id']][field], label+'/'+row['id']+'/'+field)
        elif key.endswith('Count'):
            require(actual[key] == expected[key], label+'/'+key+' count differs')
        else:
            close(actual[key], expected[key], label+'/'+key)


def compare_coverage(actual, expected, label):
    for field in COUNT_FIELDS:
        require(actual[field] == expected[field], label+'/'+field+' differs')
    for field in ('evaluableCoreSeconds', 'retainedCoreSeconds', 'coreRecall'):
        close(actual[field], expected[field], label+'/'+field)
    saved = {(r['recordingId'], r['truthIndex']): r for r in expected['rallies']}
    require(len(saved) == len(expected['rallies']) == len(actual['rallies']), label+' original-event inventory differs')
    for row in actual['rallies']:
        key = (row['recordingId'], row['truthIndex'])
        require(key in saved, label+' original event missing')
        other = saved[key]
        for field in ('start', 'end', 'tags', 'fullyCovered', 'completelyLost', 'partiallyLost'):
            require(row[field] == other[field], label+'/'+str(key)+'/'+field+' differs')
        for field in ('evaluableCoreSeconds', 'retainedCoreSeconds', 'coverage'):
            close(row[field], other[field], label+'/'+str(key)+'/'+field)
    expected_slices = slices(expected)
    actual_slices = slices(actual)
    for name in SLICES:
        for field in actual_slices[name]:
            left, right = actual_slices[name][field], expected_slices[name][field]
            if left is None or right is None:
                require(left is right, label+' slice evaluability differs')
            elif field.endswith('Rallies') or field.endswith('Losses'):
                require(left == right, label+' slice count differs')
            else:
                close(left, right, label+'/'+name+'/'+field)
    return actual_slices


def compare_scope(records, evaluation, label):
    padding = {r['paddingSecondsBeforeAndAfter']: r for r in evaluation['padding']}
    require(len(evaluation['padding']) == 4 and set(padding) == set(PADDINGS), label+' padding cases differ')
    actual_padding = [pooled_metric(records, pad) for pad in PADDINGS]
    for row in actual_padding:
        compare_metric(row, padding[row['paddingSecondsBeforeAndAfter']], label)
    compare_metric(actual_padding[2], evaluation['primary'], label+'/primary')
    close(evaluation['objective'], actual_padding[2]['F1_padP_coreR'], label+'/objective')
    coverages = {}
    for scope in ('coreCoverage', 'primaryExportCoverage'):
        coverage = event_coverage(records, scope)
        subslices = compare_coverage(coverage, evaluation['guardrails'][scope], label+'/'+scope)
        if scope == 'primaryExportCoverage':
            close(coverage['coreRecall'], actual_padding[2]['R_core'], label+'/coverage-vs-union recall')
        coverages[scope] = {key: value for key, value in coverage.items() if key != 'rallies'}
        coverages[scope]['slices'] = subslices
        coverages[scope]['originalEventRowsSha256'] = hashlib.sha256(
            json.dumps(coverage['rallies'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return {'padding': actual_padding, 'coverage': coverages}


def refit_origin(result, study, reference):
    cohort, kind, arm, seed = (result[k] for k in ('cohort', 'kind', 'lossArm', 'seed'))
    reused = cohort in ('exact', 'draft') and kind == 'tcn' and arm == 'baseline'
    expected_root = (Path(reference['path'])/'fits'/cohort/kind/str(seed) if reused else
                     study/'fits'/cohort/kind/arm/str(seed))
    require(Path(result['origin']['fitRoot']).resolve() == expected_root.resolve(), 'Unexpected refit origin path')
    require(result['origin']['type'] == ('reused-reference' if reused else 'trained'), 'Unexpected refit origin type')
    if reused:
        require(result['origin']['studyPath'] == reference['path']
                and result['origin']['reportSha256'] == reference['reportSha256']
                and result['origin']['referenceContractSha256'] == reference['contractSha256'], 'Reused refit provenance differs')
    return expected_root, reused


def refit_decode(result, study, contract, source_inputs):
    # This is the only shared numerical code called by the auditor. Interval
    # evaluation above does not call any shared metric/crop/schema helpers.
    sys.path.insert(0, str(REPO))
    from analysis.neural_development import decode

    cohort, kind, arm, seed = (result[k] for k in ('cohort', 'kind', 'lossArm', 'seed'))
    reference = contract['referenceStudy']
    expected_root, reused = refit_origin(result, study, reference)
    expected_contract = reference['contractSha256'] if reused else result['contractSha256']
    predicted = {r['id']: r for r in result['predictions']}
    require(len(predicted) == len(result['predictions']) == len(source_inputs), 'Prediction population differs')
    selected = {s['heldSourceGroup']: s for s in result['selections']}
    require(len(result['selections']) == len(selected) == len(contract['groups']) and set(selected) == set(contract['groups']), 'Missing/duplicate refit selection')
    artifacts = []
    for outer_index, group in enumerate(contract['groups']):
        setting = selected[group]
        require(setting['epoch'] in contract['checkpointEpochs'] and setting['decoder'] in contract['decoderCandidates'], 'Refit selection outside frozen grid')
        folder = expected_root/f'outer-{outer_index}'/'refit'
        completed_path = folder/'completed.json'
        completed = read(completed_path)
        ids = [rid for rid, row in source_inputs.items() if row['group'] == group]
        require(completed['contractSha256'] == expected_contract and completed['kind'] == kind
                and completed['seed'] == seed and completed['epochs'] == [setting['epoch']]
                and completed['validationIds'] == ids, 'Selected refit metadata differs')
        path = folder/f"predictions-{setting['epoch']}.npz"
        stored = identity(path)
        require(stored['sha256'] == completed['artifacts'][path.name], 'Selected refit prediction artifact changed')
        with np.load(path, allow_pickle=False) as data:
            require(len(data.files) == len(ids) and set(data.files) == set(ids), 'Refit probability recording keys differ')
            for rid in ids:
                source, scores = source_inputs[rid], data[rid]
                require(scores.shape == (len(source['times']), 4) and scores.dtype == np.float32
                        and np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
                        and np.all(scores[~source['valid']] == 0), 'Invalid selected refit probability trace')
                example = SimpleNamespace(times=source['times'], duration=source['duration'], valid=source['valid'])
                cuts = [row.to_dict() for row in decode(example, scores, setting['decoder'])]
                require(cuts == predicted[rid]['predictions'], 'Stored cuts differ from selected refit decoding: '+rid)
        artifacts.append({'predictions': stored, 'completed': identity(completed_path),
                          'heldSourceGroup': group, 'recordings': ids, 'epoch': setting['epoch'], 'decoder': setting['decoder']})
    return artifacts


def audit(manifest_path, study):
    require((study/'report.json').is_file(), 'Final interval audit requires completed report.json')
    locked = [identity(path) for path in (manifest_path, study/'preregistration.json', study/'report.json', Path(__file__))]
    reg, report, manifest = read(study/'preregistration.json'), read(study/'report.json'), read(manifest_path)
    contract = reg['contract']
    require(hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == reg['sha256'], 'Registration hash differs')
    require(contract['experiment'] == 'bounded-short-boost-dino-transfer-v1'
            and report['status'] == 'completed-short-boost-transfer-development'
            and report['contractSha256'] == reg['sha256'] and report['manifestSha256'] == contract['manifestSha256'] == digest(manifest_path), 'Report/manifest contract differs')
    require(not report['protectedTestOpened'] and not report['productionPromotionAllowed'], 'Unexpected protected/promotion scope')
    require(contract['primaryMetric'] == 'F1_padP_coreR' and contract['targetPaddingSeconds'] == 2
            and contract['joinGapSeconds'] == 3 and contract['paddingSweep'] == list(PADDINGS), 'Metric contract differs')
    require(len(contract['code']) == 16, 'Unexpected registered source count')
    for name, sha in contract['code'].items():
        require(digest(REPO/'analysis'/name) == sha, 'Frozen source changed: '+name)
    exact_path = Path(manifest['exactManifest']['path'])
    require(digest(exact_path) == manifest['exactManifest']['sha256'], 'Exact manifest changed')
    locked.append(identity(exact_path))
    exact = read(exact_path)
    require(exact['recordings'] == manifest['exactRows'], 'Exact annotation/feature revision differs')
    original_path = Path(contract['originalManifest']['path'])
    require(digest(original_path) == contract['originalManifest']['sha256'], 'Original manifest changed')
    original = read(original_path)
    require(original['exactRows'] == manifest['exactRows'] and original['draftRows'] == manifest['draftRows'],
            'Unchanged exact/draft dataset revision differs')
    require(manifest['originalManifest'] == contract['originalManifest']
            and manifest['protocolAmendment'] == contract['protocolAmendment']
            and manifest['featureRevision'] == contract['featureRevision'], 'Dataset amendment differs from registered revision')
    amendment_path = Path(contract['protocolAmendment']['path'])
    require(digest(amendment_path) == contract['protocolAmendment']['sha256'], 'PTS protocol amendment changed')
    locked.append(identity(original_path))
    locked.append(identity(amendment_path))
    source_rows = {r['id']: r for r in manifest['exactRows']}
    require(len(source_rows) == len(manifest['exactRows']) == report['records'] == 8, 'Exact recording count differs')
    require(report['sourceGroups'] == contract['groups'] == sorted({r['sourceGroup'] for r in source_rows.values()}), 'Exact source groups differ')
    require(all(r['environment'] in ('grass', 'indoor') and r['sourceGroup'] not in manifest['protectedSourceGroups']
                and r['consent']['train'] is True for r in source_rows.values()), 'Excluded source or missing consent')
    source_inputs = {}
    for rid, row in source_rows.items():
        entry = row['featureCaches']['audiovisual']
        path = Path(entry['path'])
        require(digest(path) == entry['sha256'], 'AV source timeline changed')
        with np.load(path, allow_pickle=False) as data:
            times = data['times'].astype(np.float64)
            video_duration = float(json.loads(str(data['metadata_json'].item()))['duration'])
        require(times.ndim == 1 and len(times) and np.isfinite(times).all() and np.all(np.diff(times) > 0), 'Invalid source timeline')
        valid = np.ones(len(times), bool)
        for ignored in row.get('ignoredIntervals', []):
            valid[(times >= ignored['start']) & (times < ignored['end'])] = False
        source_inputs[rid] = {'times': times, 'valid': valid, 'duration': video_duration, 'group': row['sourceGroup']}
    expected = {(c, k, a, s) for c in contract['cohorts'] for k in contract['kinds'] for a in contract['lossArms'] for s in contract['seeds']}
    keys = [(r['cohort'], r['kind'], r['lossArm'], r['seed']) for r in report['results']]
    require(len(keys) == len(set(keys)) == len(expected) == 54 and set(keys) == expected, 'Incomplete54-cell factorial report')
    counts, checked = Counter(), []
    for result in report['results']:
        cohort, kind, arm, seed = (result[k] for k in ('cohort', 'kind', 'lossArm', 'seed'))
        label = f'{cohort}/{kind}/{arm}/{seed}'
        require(result['contractSha256'] == reg['sha256'] and result['architecture'] == kind, 'Result identity differs')
        result_path = study/f'result-{cohort}-{kind}-{arm}-{seed}.json'
        require(read(result_path) == result, 'Cell result differs from report')
        predictions = result['predictions']
        require(len(predictions) == 8 and {r['id'] for r in predictions} == set(source_rows), 'Prediction recording scope differs')
        for prediction in predictions:
            gold = source_rows[prediction['id']]
            require(prediction['sourceGroup'] == gold['sourceGroup']
                    and prediction['durationSeconds'] == source_inputs[prediction['id']]['duration'], 'Prediction gold revision differs')
            verify_serialized_gold(prediction['rallies'], gold['rallies'])
            verify_serialized_gold(prediction['ignoredIntervals'], gold.get('ignoredIntervals', []))
        refits = refit_decode(result, study, contract, source_inputs)
        records = [parse_record(row) for row in predictions]
        evaluation = result['evaluation']
        global_metrics = compare_scope(records, evaluation, label)
        require(set(evaluation['sourceGroups']) == set(contract['groups']), 'Stored group evaluation inventory differs')
        for group in contract['groups']:
            compare_scope([r for r in records if r['sourceGroup'] == group], evaluation['sourceGroups'][group], label+'/'+group)
        single = {r['id']: r for r in evaluation['recordings']}
        require(len(evaluation['recordings']) == len(single) == 8 and set(single) == set(source_rows), 'Stored recording evaluation inventory differs')
        for record in records:
            compare_scope([record], single[record['id']], label+'/'+record['id'])
        counts['resultCells'] += 1
        counts['reusedResultCells' if cohort in ('exact', 'draft') and kind == 'tcn' and arm == 'baseline'
               else 'freshResultCells'] += 1
        counts['paddingScopeRows'] += 4*(1+len(contract['groups'])+len(records))
        counts['globalPaddingRows'] += 4
        counts['refitPredictionNPZ'] += len(refits)
        counts['decodedRecordingTraces'] += len(records)
        counts['originalRallyCoverageRowsPerScope'] += global_metrics['coverage']['primaryExportCoverage']['evaluableRallies']
        checked.append({'cohort': cohort, 'kind': kind, 'lossArm': arm, 'seed': seed,
                        'resultFile': identity(result_path), 'selectedRefits': refits, **global_metrics})
    require(counts['resultCells'] == 54 and counts['globalPaddingRows'] == 216 and counts['paddingScopeRows'] == 2808
            and counts['refitPredictionNPZ'] == 216 and counts['decodedRecordingTraces'] == 432
            and counts['reusedResultCells'] == 6 and counts['freshResultCells'] == 48, 'Final interval audit inventory differs')
    for name, sha in contract['code'].items():
        require(digest(REPO/'analysis'/name) == sha, 'Frozen source changed during interval audit')
    for source in locked:
        require(digest(Path(source['path'])) == source['sha256'], 'Audit input changed during interval reconstruction')
    return {'kind': 'independent-short-boost-transfer-interval-audit-v1', 'passed': True,
            'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': reg['sha256'],
            'registeredSourceHashesVerified': contract['code'], 'inputs': locked, 'counts': dict(counts),
            'numericTolerance': {'absolute': 1e-8, 'relative': 1e-10},
            'eventClassificationAbsoluteTolerance': 1e-9,
            'method': 'Independent endpoint sweeps for union/intersection/difference; preclip raw ranges, pad, union, strictly<3s gap joining, then ignored subtraction without rejoining. Pooled duration arithmetic at0/1/2/3s. Original-event complete/partial classifications and duration/tag slices at2s export and raw core coverage.',
            'sharedNumericalCode': 'Only frozen neural_development.decode for exact selected-refit probability-to-raw-cut replay; no shared crop/metric arithmetic.',
            'scope': 'Identical eight exact-label recordings across all54 results; no protected test or production promotion.',
            'limits': ['This independently evaluates stored selected probabilities/cuts; it does not rerun neural feature extraction or model forwards.',
                       'Event-IoU matching/boundary-F1 and inner candidate selection belong to the separate summary auditor.',
                       'Zero-padding ranking still applies the strict<3s join; raw coreCoverage does not.'],
            'results': checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root/'interval-audit-v1.json'
    require(not output.exists(), 'Refusing to overwrite immutable interval audit')
    result = audit(args.manifest, args.study or args.root/'study')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'counts': result['counts'], 'artifact': identity(output)}, indent=2))


if __name__ == '__main__':
    main()
