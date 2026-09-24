#!/usr/bin/env python3
"""Independent result/index/flat-report audit; CPU only, no model selection.

Requires every registered experiment and precision. Replays frozen decoding,
independent interval arithmetic, canonical event matching and fixed shipped TS.
The HTML may consume only the numerical payload digest certified here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as io
from analysis import neural_development as decoder
from analysis.neural_evaluation import evaluate_predictions
from analysis.neural_generalization_inputs import feature_entries, load_inference_examples
from analysis.schema import Interval

# This gate is intentionally separate from the numerical result/aggregation code.
import importlib.util
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result

selection_auditor = module('bound_generalization_selection_auditor', REPO/'scripts/audit-neural-generalization-selection.py')
Evidence = selection_auditor.Evidence
HELPER_SHA = selection_auditor.HELPER_SHA
COMMON = frozenset((private_value('source-group-001'), private_value('source-group-008')))
POLICY = {'exact-core': 'exact-rallies', 'draft-reviewed': 'reviewed-draft', 'export-coverage': 'reviewed-export'}
POLICIES = ('exact-rallies', 'reviewed-export', 'reviewed-draft')
FILTERS = ('all', 'no-production-training', 'no-production-training-or-calibration')
PANEL_KEYS = ('panelId', 'labelPolicy', 'productionFilter')
METRICS = ('precisionValue', 'recallValue', 'f1Value', 'exportSeconds', 'humanExportSeconds',
    'exportDurationDifferenceSeconds', 'correctlyRemovedSeconds', 'incorrectExportSeconds',
    'wantedExportOmittedSeconds', 'missedCoreSeconds', 'missedReviewedLiveSeconds',
    'completeRallyLosses', 'partialRallyLosses', 'eventPrecision', 'eventRecall', 'eventF1',
    'startBoundaryMaeSeconds', 'endBoundaryMaeSeconds')
ROW_KEY = ('model', 'variant', 'draw', 'precision', 'panelId', 'labelPolicy', 'productionFilter', 'floorPercent', 'paddingSeconds')
PANEL_CHECKS = set()
REUSE_MAPPINGS = {}

# V2 changes provenance crosschecks only. The original numerical oracle remains
# byte-for-byte represented by the unchanged functions/classes below.
ORIGINAL_AUDITOR_SHA = 'e4a874286b5a521e3a2c974ce9e6f1d4606086979961152af0ff5b67838535f1'
DURATION_PROOF_SHA = '52282c128dd2bd4ddf529d7f7f797836407f50e4f68ec66af9bfb8b9414082fe'
DURATION_AUDITOR_SHA = '5a3fb4dffffc15e0f4f40c72b872684d385227d7b33ffa060dedf126d6dff59e'
REVISION_POLICY = {'kind': 'final-audit-historical-provenance-crosschecks-v2',
    'historicalMetricAndControlAuditSchemasDistinct': True,
    'historicalMetricDuration': 'exact registered audiovisual cache duration',
    'inventoryDisplayDuration': 'exact six-decimal rounding of that same duration',
    'independentExistingDurationProofRequired': True,
    'numericalOracleFunctionsUnchanged': True, 'metricToleranceChanged': False,
    'predictionsOrLabelsChanged': False, 'metricsRewritten': False}


def historical_report_gates(directory, evidence):
    """Select the two named authorities; unrelated audit schemas cannot collide."""
    report = evidence.capture(directory/'report-complete.json')
    found = []
    for filename, kind, auditor in (
        ('audit-complete.json', 'independent-recall-floor-sweep-audit-v1', 'audit-neural-recall-sweep.py'),
        ('audit-controls-complete.json', 'independent-historical-recall-control-replay-v1', 'audit-neural-historical-controls.py')):
        path = directory/filename; gate = evidence.document(evidence.capture(path))
        io.require(gate['passed'] is True and gate['kind'] == kind and gate['report'] == report
            and gate['auditor'] == io.identity(REPO/'scripts'/auditor),
            'Historical metric/control authority differs: '+filename)
        evidence.closure(gate)
        found.append((path, gate))
    return report, found


def historical_duration_checks(protocol_ref, protocol, evidence, inventory):
    """Recheck the old timing proof against its exact cache; no metric is changed."""
    directory = Path(protocol_ref['path']).parent
    proof_ref = evidence.capture(directory/'audit-duration-provenance.json')
    io.require(proof_ref['sha256'] == DURATION_PROOF_SHA, 'Registered duration proof changed')
    proof = evidence.document(proof_ref)
    auditor = {'path': str(REPO/'scripts/audit-neural-historical-duration-provenance.py'),
               'sha256': DURATION_AUDITOR_SHA}
    evidence.bind(auditor)
    io.require(proof['kind'] == 'independent-historical-duration-provenance-audit-v1'
        and proof['passed'] is True and proof['protocol'] == protocol_ref
        and proof['records'] == protocol['records'] and proof['source'] == protocol['source']
        and proof['auditor'] == auditor and proof['recordingCount'] == 8
        and proof['roundedManifestDifferenceCount'] == 7
        and all(proof[key] is False for key in ('metricDurationChanged', 'selectionChanged',
            'trainingPerformed', 'gpuUsed', 'rawVideoOpened')), 'Historical duration proof scope differs')
    evidence.closure(proof)
    stored = evidence.document(protocol['records'])['records']
    original = evidence.document(protocol['source'])['exactRows']
    source = {r['id']: r for r in original}
    io.require(len(stored) == len(original) == len(source) == len(proof['checks']) == 8
        and [r['id'] for r in stored] == [r['id'] for r in proof['checks']]
        and {r['id'] for r in stored} == set(source), 'Historical timing population differs')
    checked, differences = {}, 0
    for row, check in zip(stored, proof['checks'], strict=True):
        gold = source[row['id']]; cache = gold['featureCaches']['audiovisual']
        reference = {key: cache[key] for key in ('path', 'sha256')}
        with np.load(evidence.bind(reference), allow_pickle=False) as payload:
            metadata = json.loads(str(payload['metadata_json'].item()))
            times = payload['times'].copy()
        duration = metadata['duration']
        io.require(type(duration) in (int, float) and np.isfinite(duration) and duration > 0
            and type(metadata['frame_count']) is int and metadata['frame_count'] > 0
            and type(metadata['fps']) in (int, float) and np.isfinite(metadata['fps']) and metadata['fps'] > 0,
            'Invalid historical media timing')
        io.require(duration == row['durationSeconds'] == cache['metadata']['duration']
            == metadata['frame_count']/metadata['fps'] and gold['durationSeconds'] == round(duration, 6),
            'Historical exact cache/frame or display rounding differs')
        io.require(row['sourceGroup'] == gold['sourceGroup'] and np.array_equal(times, row['timestamps']),
            'Historical source/timestamp proof differs')
        for field in ('rallies', 'ignoredIntervals'):
            normalized = [{'start': value['start'], 'end': value['end'],
                **({'tags': value['tags']} if value.get('tags') else {})} for value in gold[field]]
            io.require(row[field] == normalized, 'Historical timing proof label values/tags/order changed')
        valid = np.ones(len(times), dtype=bool)
        for value in gold['ignoredIntervals']:
            valid &= ~((times >= value['start']) & (times < value['end']))
        io.require(np.array_equal(valid, row['valid']), 'Historical timing proof validity differs')
        expected = {'id': row['id'], 'sourceGroup': row['sourceGroup'], 'audiovisualCache': reference,
            'storedMetricDurationSeconds': duration, 'cacheMetadataDurationSeconds': duration,
            'manifestDurationSeconds': gold['durationSeconds'], 'frameCount': metadata['frame_count'],
            'fps': metadata['fps'], 'manifestMinusStoredSeconds': gold['durationSeconds']-duration,
            'storedEqualsBothCacheMetadataDurations': True, 'storedEqualsFrameCountDivFps': True,
            'manifestEqualsExactSixDecimalRound': True, 'timestampCount': len(times),
            'timestampsExactlyCache': True, 'sourceGroupAndIntervalValuesAndTagsExact': True,
            'validMaskExactlyIgnoredSubtraction': True}
        io.require(check == expected, 'Stored independent duration proof differs from exact cache replay')
        validate_historical_duration(row, inventory[row['id']], check)
        checked[row['id']] = check
        differences += gold['durationSeconds'] != duration
    io.require(differences == 7, 'Historical rounded-duration population differs')
    return checked


def validate_historical_duration(raw, authoritative, check):
    io.require(raw['id'] == authoritative['id'] == check['id']
        and raw['sourceGroup'] == authoritative['sourceGroup'] == check['sourceGroup'],
        'Historical duration recording/source identity differs')
    stored, display = raw['durationSeconds'], authoritative['durationSeconds']
    io.require(type(stored) in (int, float) and type(display) in (int, float)
        and np.isfinite(stored) and np.isfinite(display)
        and stored == check['storedMetricDurationSeconds'] == check['cacheMetadataDurationSeconds']
        == check['frameCount']/check['fps']
        and display == check['manifestDurationSeconds'] == round(stored, 6),
        'Historical metric/display duration is not the proved exact cache/rounding pair')


def compare(actual, expected, label='value'):
    if isinstance(expected, dict):
        io.require(isinstance(actual, dict) and set(actual) == set(expected), label+' field inventory differs')
        for key in expected: compare(actual[key], expected[key], label+'/'+str(key))
    elif isinstance(expected, list):
        io.require(isinstance(actual, list) and len(actual) == len(expected), label+' list scope differs')
        for i, (a, b) in enumerate(zip(actual, expected)): compare(a, b, label+'/'+str(i))
    elif expected is None or isinstance(expected, (str, bool)):
        io.require(actual == expected and (not isinstance(expected, bool) or isinstance(actual, bool)), label+' differs')
    elif isinstance(expected, (float, int)):
        io.require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
                   and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-10, atol=1e-8), label+' numeric value differs')
    else:
        io.require(actual == expected, label+' differs')


def clean(row, kind):
    p = row['productionExposure']['rallyPipeline']
    fitting = bool(p['directTrainingHeads'] or p['sameSourceTrainingHeads'] or p['sameGroupTrainingHeads'])
    calibration = bool(p['directCalibrationHeads'] or p['sameGroupCalibrationHeads'])
    unknown = bool(p['unknownScopeHeads'])
    io.require(p['trainingOrRelated'] == fitting and p['primaryTrainingClean'] == (not fitting and not unknown)
               and p['strictNoFitOrCalibration'] == (not fitting and not calibration and not unknown), 'Production lineage summary differs')
    return True if kind == 'all' else not fitting and not unknown if kind == FILTERS[1] else not fitting and not calibration and not unknown


def independent_panels(records, inventory, original_groups, used_groups):
    conditions = [('all-labeled', lambda r: True), ('common-unseen', lambda r: r['sourceGroup'] in COMMON),
        ('outside-original-sources', lambda r: r['sourceGroup'] not in original_groups),
        ('task-unseen-sources', lambda r: r['sourceGroup'] not in used_groups)]
    return [{'panelId': name, 'labelPolicy': policy, 'productionFilter': production_filter,
             'recordingIds': [r['id'] for r in records if POLICY[r['scoringPolicy']] == policy
                              and condition(r) and clean(inventory[r['id']], production_filter)]}
            for name, condition in conditions for policy in POLICIES for production_filter in FILTERS]


def simple_intervals(values):
    return [{'start': r['start'], 'end': r['end'], **({'tags': r['tags']} if r.get('tags') else {})} for r in values]


def gold_rows(records, predictions, policy):
    field = {'exact-rallies': 'rallies', 'reviewed-draft': 'reviewedLiveIntervals', 'reviewed-export': 'humanExportIntervals'}[policy]
    result = []
    for row in records:
        if POLICY[row['scoringPolicy']] != policy: continue
        ignored = simple_intervals(row.get('ignoredIntervals', []))
        if policy == 'reviewed-export':
            window = row['gameWindow']
            if window['start'] > 0: ignored.append({'start': 0., 'end': window['start']})
            if window['end'] < row['durationSeconds']: ignored.append({'start': window['end'], 'end': row['durationSeconds']})
        result.append({'id': row['id'], 'sourceGroup': row['sourceGroup'], 'durationSeconds': row['durationSeconds'],
            field: simple_intervals(row['keepTargets'] if policy == 'reviewed-export' else row['rallies']),
            'ignoredIntervals': ignored, 'predictions': predictions[row['id']]})
    return result


def coverage_at(helper, parsed, padding):
    events, original = [], 0
    for row in parsed:
        ignored = helper.ranges(row['ignoredIntervals'])
        export = helper.export_union(row, 'predictions', padding)
        original += len(row['rallies'])
        for index, rally in enumerate(row['rallies']):
            core = helper.boolean_intervals([(rally['start'], rally['end'])], ignored, 'difference')
            seconds = helper.duration(core)
            if seconds <= 0: continue
            missed = helper.duration(helper.boolean_intervals(core, export, 'difference'))
            retained = max(0., seconds-missed)
            full, lost = abs(missed) <= 1e-9, retained <= 1e-9
            events.append({'recordingId': row['id'], 'truthIndex': index, 'start': rally['start'], 'end': rally['end'],
                'tags': rally['tags'], 'evaluableCoreSeconds': seconds, 'retainedCoreSeconds': retained,
                'coverage': retained/seconds, 'fullyCovered': full, 'completelyLost': lost, 'partiallyLost': not full and not lost})
    seconds = sum(e['evaluableCoreSeconds'] for e in events); retained = sum(e['retainedCoreSeconds'] for e in events)
    return {'originalRallies': original, 'evaluableRallies': len(events), 'fullyIgnoredRallies': original-len(events),
        'completeRallyLosses': sum(e['completelyLost'] for e in events), 'partialRallyLosses': sum(e['partiallyLost'] for e in events),
        'fullyCoveredRallies': sum(e['fullyCovered'] for e in events), 'evaluableCoreSeconds': seconds,
        'retainedCoreSeconds': retained, 'coreRecall': retained/seconds if seconds else 0., 'rallies': events}


def independent_metrics(helper, rows, policy):
    truth = {'exact-rallies': 'rallies', 'reviewed-draft': 'reviewedLiveIntervals', 'reviewed-export': 'humanExportIntervals'}[policy]
    parsed = [helper.parse_record({**r, 'rallies': r[truth]}) for r in rows]
    universe = sum(helper.duration(helper.boolean_intervals([(0., r['duration'])], helper.ranges(r['ignoredIntervals']), 'difference')) for r in parsed)
    outputs, guardrails = [], []
    for pad in (0, 1, 2, 3):
        if policy == 'reviewed-export':
            model, human, intersection = 0., 0., 0.
            for r in parsed:
                m = helper.export_union(r, 'predictions', pad)
                h = helper.boolean_intervals(helper.ranges(r['rallies']), helper.ranges(r['ignoredIntervals']), 'difference')
                model += helper.duration(m); human += helper.duration(h)
                intersection += helper.duration(helper.boolean_intervals(m, h, 'intersection'))
            p, recall = intersection/model if model else 0., intersection/human
            metric = {'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3., 'humanPaddingSeconds': 0, 'humanJoinGapSeconds': 0,
                'modelExportSeconds': model, 'humanExportSeconds': human, 'intersectionSeconds': intersection,
                'evaluableVideoSeconds': universe, 'P_export': p, 'R_export': recall,
                'F1_export': 2*p*recall/(p+recall) if p+recall else 0., 'exportDurationDifferenceSeconds': model-human,
                'incorrectExportSeconds': model-intersection, 'wantedExportOmittedSeconds': human-intersection,
                'correctlyRemovedSeconds': universe-human-model+intersection}
            outputs.append({**metric, 'paddingSeconds': pad, 'precisionValue': p, 'recallValue': recall,
                'f1Value': metric['F1_export'], 'exportSeconds': model, 'missedCoreSeconds': None,
                'completeRallyLosses': None, 'eventF1': None})
            continue
        metric = helper.pooled_metric(parsed, pad)
        p, recall, f1 = (metric[k] for k in ('P_pad', 'R_core', 'F1_padP_coreR'))
        model, human, intersection = (metric[k] for k in ('paddedModelExportSeconds', 'paddedHumanExportSeconds', 'paddedPrecisionIntersectionSeconds'))
        incorrect = model-intersection
        if policy == 'reviewed-draft':
            outputs.append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3.,
                'P_reviewed': p, 'R_reviewed': recall, 'F1_reviewed': f1,
                'modelExportSeconds': model, 'humanExportSeconds': human, 'reviewedLiveSeconds': metric['coreHumanSeconds'],
                'retainedReviewedLiveSeconds': metric['coreRecallIntersectionSeconds'],
                'exportDurationDifferenceSeconds': model-human, 'incorrectExportSeconds': incorrect,
                'wantedExportOmittedSeconds': human-intersection,
                'missedReviewedLiveSeconds': metric['coreHumanSeconds']-metric['coreRecallIntersectionSeconds'],
                'correctlyRemovedSeconds': universe-human-incorrect, 'paddingSeconds': pad,
                'precisionValue': p, 'recallValue': recall, 'f1Value': f1, 'exportSeconds': model,
                'missedCoreSeconds': None, 'completeRallyLosses': None, 'eventF1': None})
            continue
        canonical = evaluate_predictions(rows, primary_padding_seconds=pad)
        independent_coverage = coverage_at(helper, parsed, pad)
        helper.compare_coverage(independent_coverage, canonical['guardrails']['primaryExportCoverage'], 'original-rally coverage')
        if pad == 2: helper.compare_scope(parsed, canonical, 'canonical event/interval replay')
        guardrails.append({'paddingSeconds': pad, 'coverage': {k: v for k, v in independent_coverage.items() if k != 'rallies'},
            'slices': helper.slices(independent_coverage),
            'completelyLostOriginalRallies': [r for r in independent_coverage['rallies'] if r['completelyLost']],
            'partiallyLostOriginalRallies': [r for r in independent_coverage['rallies'] if r['partiallyLost']],
            'originalEventRowsSha256': io.canonical(independent_coverage['rallies'])})
        outputs.append({**metric, 'paddingSeconds': pad, 'precisionValue': p, 'recallValue': recall, 'f1Value': f1,
            'exportSeconds': model, 'humanExportSeconds': human, 'incorrectExportSeconds': incorrect,
            'wantedExportOmittedSeconds': human-intersection, 'correctlyRemovedSeconds': universe-human-incorrect,
            'missedCoreSeconds': metric['coreHumanSeconds']-metric['coreRecallIntersectionSeconds'],
            'completeRallyLosses': independent_coverage['completeRallyLosses'], 'partialRallyLosses': independent_coverage['partialRallyLosses'],
            **{k: canonical['guardrails'][k] for k in ('eventPrecision', 'eventRecall', 'eventF1', 'startBoundaryMaeSeconds', 'endBoundaryMaeSeconds')}})
    return outputs, guardrails


class MetricCache:
    def __init__(self, helper, output):
        self.helper, self.cache = helper, {}
        self.output = Path(output); self.stream = self.output.open('x', encoding='utf-8')
    def measure(self, rows, policy):
        key = io.canonical({'rows': rows, 'labelPolicy': policy})
        if key not in self.cache:
            metrics, guards = independent_metrics(self.helper, rows, policy)
            self.cache[key] = metrics
            if guards:
                self.stream.write(json.dumps({'metricInputSha256': key, 'recordingIds': [r['id'] for r in rows],
                    'padding': guards}, allow_nan=False, separators=(',', ':'))+'\n')
        return self.cache[key]
    def close(self): self.stream.close()


def audit_point(result, key, records, decoded, panels, metrics):
    point = result['operatingPoints'][key]
    rows_by_policy = {policy: gold_rows(records, decoded, policy) for policy in POLICIES}
    compare(point['rowsByPolicy'], rows_by_policy, 'point gold/raw decoded cuts')
    io.require(len(point['panels']) == len(panels), 'Point panel count differs')
    for expected, saved in zip(panels, point['panels']):
        rows = [r for r in rows_by_policy[expected['labelPolicy']] if r['id'] in expected['recordingIds']]
        compare(saved, {**expected, 'status': 'available' if rows else 'empty-scope',
            'padding': metrics.measure(rows, expected['labelPolicy']) if rows else None}, 'point metric/panel')


def check_gold(population, inventory, helper):
    io.require(len(population) == 42 and len({r['id'] for r in population}) == 42, 'Independent inference population differs')
    count = Counter(r['scoringPolicy'] for r in population)
    io.require(count == {'exact-core': 9, 'draft-reviewed': 6, 'export-coverage': 19, 'none': 8}, 'Authoritative label scope differs')
    for row in population:
        original = inventory[row['id']]
        io.require(not row.get('experimentalSupervision') and row['environment'] != 'beach'
                   and row['sourceGroup'] == original['sourceGroup'] and row['protected'] == original['protected'], 'Independent gold/source role changed')
        if row['scoringPolicy'] == 'none': continue
        field = 'keepTargets' if row['scoringPolicy'] == 'export-coverage' else 'rallies'
        compare(simple_intervals(row[field]), simple_intervals(original[field]), 'authoritative gold intervals')
        ignored = helper.ranges(row.get('ignoredIntervals', [])); old_ignored = helper.ranges(original.get('ignoredIntervals', []))
        if row['scoringPolicy'] == 'export-coverage':
            compare(row['gameWindow'], original['gameWindow'], 'authoritative reviewed window')
            for r, spans in ((row, ignored), (original, old_ignored)):
                w = r['gameWindow']
                if w['start'] > 0: spans.append((0., w['start']))
                if w['end'] < r['durationSeconds']: spans.append((w['end'], r['durationSeconds']))
        compare(helper.boolean_intervals(ignored), helper.boolean_intervals(old_ignored), 'authoritative ignored universe')


def panel_inputs(result, evidence, inventory, helper):
    panel = evidence.document(result['panel']); manifest = evidence.document(panel['manifest'])
    features = evidence.document(panel['features'])
    io.require(panel['kind'] == 'frozen-independent-inference-panel-v1', 'Independent panel contract differs')
    io.require(panel['inventory'] == result['inventory'] and panel['inferenceUsesLabels'] is False
               and panel['allInferenceTicksValid'] is True and set(panel['commonEvaluationGroups']) == COMMON,
               'Panel inventory or isolation contract differs')
    if result['panel']['sha256'] not in PANEL_CHECKS:
        feature_audit = evidence.document(panel['featureAudit'])
        io.require(feature_audit['kind'] == 'independent-expansion-feature-audit-v1' and feature_audit['passed']
                   and feature_audit['inputs'] == panel['manifest'] and feature_audit['features'] == panel['features']
                   and feature_audit['auditedRecordingCount'] == 42 and feature_audit['protectedAndPanelTeacherTargetsAbsent']
                   and feature_audit['inferenceMaskAlwaysAllTrue'], 'Independent all-video feature gate differs')
        evidence.closure(feature_audit); evidence.bind(panel['registrar'])
        entries = feature_entries(manifest['records'], panel['features']['path'])
        for row in manifest['records']:
            entry = entries[row['id']]; evidence.closure(entry)
            image = evidence.document(entry['imageInput']); evidence.closure(image['arrays'])
            if row['protected'] or row['sourceGroup'] in COMMON or 'fit' not in row['eligibleRoles']:
                io.require('teacherTargets' not in entry and 'teaching336' not in image['arrays'], 'External teacher imagery reached model inputs')
            if 'teacherTargets' in entry: evidence.closure(evidence.document(entry['teacherTargets']))
        PANEL_CHECKS.add(result['panel']['sha256'])
    check_gold(manifest['records'], inventory, helper)
    original = evidence.document(result['originalManifest'])
    original_groups = {r['sourceGroup'] for r in original['records']}
    evidence.closure(result['code'])
    return panel, manifest['records'], features, original_groups


def reuse_required(task, fit, mapping):
    if task['variant'] == 'original-corpus':
        io.require(mapping is None, 'Original corpus must remain a physical fit')
    else:
        io.require(mapping is not None, 'Expanded task lacks its frozen reuse mapping')
    expected = mapping is not None and mapping['taskId'] != mapping['physicalOwnerTaskId']
    io.require(('trainingReuse' in fit) == expected, 'Physical/reused fit status differs from registration')
    return expected


def audit_reuse_gates(task, fit, folder, result, fit_gate_ref, inference_gate_ref, evidence):
    entry = REUSE_MAPPINGS.get(task['taskId'])
    mapping = entry['mapping'] if entry else None
    if not reuse_required(task, fit, mapping):
        return None
    plan_ref = entry['plan']
    fit_ref = evidence.capture(folder/'fit-result.json')
    receipt = evidence.document(fit['trainingReuse'])
    io.require(fit['trainingReuse'] == evidence.capture(folder/'reuse-receipt.json')
        and receipt['plan'] == plan_ref and receipt['task'] == result['task']
        and receipt['sourceTask'] == mapping['physicalOwnerTask']
        and receipt['recipeSha256'] == mapping['recipeSha256'], 'Fit reuse receipt owner differs')
    audit_ref = evidence.capture(folder/'reuse-audit.json'); checked = evidence.document(audit_ref)
    auditor = io.identity(REPO/'scripts/audit-neural-generalization-reuse.py')
    io.require(checked['kind'] == 'independent-generalization-fit-reuse-audit-v1' and checked['passed'] is True
        and checked['plan'] == plan_ref and checked['receipt'] == fit['trainingReuse']
        and checked['task'] == result['task'] and checked['sourceTask'] == mapping['physicalOwnerTask']
        and checked['fitResult'] == fit_ref and checked['targetNumericalAudit'] == fit_gate_ref
        and checked['recipeSha256'] == mapping['recipeSha256'] and checked['auditor'] == auditor,
        'Independent fit reuse gate absent or changed')
    evidence.closure(checked)
    precision = result['precision']
    raw_ref = evidence.capture(folder/f'inference-reuse-receipt-{precision}.json')
    raw = evidence.document(raw_ref)
    io.require(raw['kind'] == 'generalization-raw-inference-reuse-receipt-v1'
        and raw['plan'] == plan_ref and raw['task'] == result['task']
        and raw['sourceTask'] == mapping['physicalOwnerTask'] and raw['panel'] == result['panel']
        and raw['precision'] == precision and raw['targetFitResult'] == fit_ref
        and raw['fitReuseAudit'] == audit_ref and raw['decodedPredictionsReused'] is False
        and raw['selectionOrMetricsReused'] is False and raw['rawScoreCopiesBitExact'] is True,
        'Raw inference reuse receipt scope differs')
    gate_ref = evidence.capture(folder/f'reuse-inference-audit-{precision}.json'); gate = evidence.document(gate_ref)
    io.require(gate['kind'] == 'independent-generalization-inference-reuse-audit-v1' and gate['passed'] is True
        and gate['plan'] == plan_ref and gate['receipt'] == raw_ref and gate['task'] == result['task']
        and gate['sourceTask'] == mapping['physicalOwnerTask'] and gate['panel'] == result['panel']
        and gate['precision'] == precision and gate['fitResult'] == fit_ref and gate['fitReuseAudit'] == audit_ref
        and gate['targetNumericalAudit'] == inference_gate_ref and gate['auditor'] == auditor
        and gate['allRawScoreBytesExactlyEqual'] is True, 'Independent raw inference reuse gate absent or changed')
    evidence.closure(gate)
    return {row['id']: row for row in raw['scoreCopies']}


def audit_neural(result, evidence, inventory, helper, metrics):
    task = evidence.document(result['task'])
    selection = evidence.document(result['selection'])
    audit = evidence.document(result['selectionAudit'])
    io.require(audit['kind'] == 'independent-generalization-selection-audit-v1' and audit['passed'] is True
               and audit['selection'] == result['selection'] and audit['task'] == result['task']
               and audit['auditor'] == io.identity(REPO/'scripts/audit-neural-generalization-selection.py'), 'Independent selection gate absent/changed')
    evidence.closure(audit)
    io.require(selection['task'] == result['task'] and result['taskId'] == task['taskId']
               and result['model'] == task['model'] and result['variant'] == task['variant']
               and result['draw'] == task.get('splitSeed', task['seed']), 'Evaluation owner differs')
    panel, population, features, original_groups = panel_inputs(result, evidence, inventory, helper)
    source = evidence.document(task['manifest']); by_id = {r['id']: r for r in source['records']}
    selection_auditor.check_membership(task, source['records'])
    used = {by_id[k]['sourceGroup'] for k in task['trainIds']+task['calibrationIds']}
    records = [r for r in population if 'evaluate' in r['eligibleRoles'] and r['scoringPolicy'] != 'none']
    panels = independent_panels(records, inventory, original_groups, used)
    compare(result['panels'], panels, 'source-held/exposure-filter panels')
    folder = Path(result['selection']['path']).parent
    fit = evidence.document(evidence.capture(folder/'fit-result.json'))
    owner = evidence.document(fit['temporal'])
    io.require(fit['task'] == result['task'], 'Inference fit owner changed')
    precision = result['precision']; io.require(precision in ('fp32', 'fp16', 'int8')
        and (precision == 'fp32' or task['model'] in ('dino-tcn', 'dino-transformer')), 'Invalid precision arm')
    target = folder/'inference'/result['panel']['sha256'][:16]/precision
    fit_gate_ref = evidence.capture(folder/'fit-numerical-audit.json')
    fit_gate = evidence.document(fit_gate_ref)
    numerical_auditor = io.identity(REPO/'scripts/audit-neural-generalization-numerics.py')
    io.require(fit_gate['kind'] == 'independent-generalization-fit-numerical-audit-v1' and fit_gate['passed']
               and fit_gate['task'] == result['task'] and fit_gate['fitResult'] == io.identity(folder/'fit-result.json')
               and fit_gate['completed'] == fit['temporal'] and fit_gate['auditor'] == numerical_auditor,
               'Independent fit/scaler/exposure numerical gate absent or changed')
    evidence.closure(fit_gate)
    inference_gate_ref = evidence.capture(folder/f'inference-numerical-audit-{precision}.json')
    inference_gate = evidence.document(inference_gate_ref)
    io.require(inference_gate['kind'] == 'independent-generalization-inference-numerical-audit-v1' and inference_gate['passed']
               and inference_gate['task'] == result['task'] and inference_gate['panel'] == result['panel']
               and inference_gate['fitAudit'] == fit_gate_ref and inference_gate['precision'] == precision
               and inference_gate['auditor'] == numerical_auditor, 'Independent sampled neural inference gate absent or changed')
    evidence.closure(inference_gate)
    reused = audit_reuse_gates(task, fit, folder, result, fit_gate_ref, inference_gate_ref, evidence)
    if reused is not None:
        io.require(set(reused) == {r['id'] for r in population}, 'Reused raw score population differs')
    receipts, scores, times = [], {}, {}
    all_receipts = []
    for row in population:
        receipt_ref = evidence.capture(target/(row['id']+'.json')); receipt = evidence.document(receipt_ref)
        all_receipts.append(receipt_ref)
        if reused is None:
            io.require('rawInferenceReuse' not in receipt, 'Physical owner cannot silently reuse another raw inference')
        else:
            copied = reused[row['id']]
            io.require(copied['targetReceipt'] == receipt_ref and copied['targetOutput'] == receipt['output']
                and copied['sourceOutput']['sha256'] == copied['targetOutput']['sha256']
                and receipt['rawInferenceReuse'] == {'plan': REUSE_MAPPINGS[task['taskId']]['plan'],
                    'sourceReceipt': copied['sourceReceipt'], 'sourceOutput': copied['sourceOutput']},
                'Displayed raw probabilities are not the audited owner copy')
        io.require(receipt['task'] == result['task'] and receipt['panel'] == result['panel'] and receipt['manifest'] == panel['manifest']
                   and receipt['features'] == panel['features'] and receipt['precision'] == precision and receipt['id'] == row['id']
                   and receipt['epochs'] == list(io.EPOCHS) and receipt['labelsUsed'] is False
                   and receipt['ignoredLabelsUsedForContextOrDecoding'] is False, 'Blind inference receipt differs')
        for epoch in io.EPOCHS:
            ref = receipt['weights'][str(epoch)]; name = f'weights-{epoch}.npz'
            io.require(Path(ref['path']) == folder/'temporal'/name and ref['sha256'] == owner['artifacts'][name], 'Inference used another checkpoint')
            evidence.bind(ref)
        with np.load(evidence.bind(receipt['output']), allow_pickle=False) as payload:
            io.require(set(payload.files) == {'times', *[f'epoch_{e}' for e in io.EPOCHS]}, 'Inference probability archive inventory differs')
            timeline = payload['times'].copy()
            blind, = load_inference_examples(panel['manifest']['path'], panel['features']['path'], recording_ids=[row['id']])
            io.require(np.array_equal(timeline, blind.times) and blind.valid.all() and not blind.truth and not blind.ignored, 'Blind inference grid differs')
            times[row['id']] = timeline
            scores[row['id']] = {e: payload[f'epoch_{e}'].copy() for e in io.EPOCHS}
            io.require(all(v.shape == (len(timeline), 4) and np.isfinite(v).all() and np.all((v>=0)&(v<=1)) for v in scores[row['id']].values()), 'Invalid probability tensor')
        if row['scoringPolicy'] != 'none': receipts.append(receipt_ref)
    compare(result['inferenceReceipts'], receipts, 'Scored inference receipt list')
    compare(inference_gate['inferenceReceipts'], all_receipts, 'All42 numerically qualified inference receipts')
    keys = set()
    for decision, floor in zip(selection['floors'], result['floors']):
        chosen = decision['selected']; key = io.canonical({'epoch': chosen['epoch'], 'decoder': chosen['decoder']}) if chosen else None
        compare(floor, {'floorPercent': decision['floorPercent'], 'status': 'available' if chosen else 'infeasible-inner-recall',
            'operatingPointKey': key, 'selection': decision}, 'Transferred frozen floor')
        if not key or key in keys: continue
        keys.add(key); decoded = {}
        for row in records:
            example = io.SweepExample(row['id'], row['sourceGroup'], row['durationSeconds'], times[row['id']],
                                      np.ones(len(times[row['id']]), bool), (), ())
            decoded[row['id']] = [r.to_dict() for r in decoder.decode(example, scores[row['id']][chosen['epoch']], chosen['decoder'])]
        io.require(result['operatingPoints'][key]['epoch'] == chosen['epoch'] and result['operatingPoints'][key]['decoder'] == chosen['decoder'], 'Point changed FP32 operating choice')
        audit_point(result, key, records, decoded, panels, metrics)
    io.require(len(result['floors']) == 11 and set(result['operatingPoints']) == keys, 'Extra/missing selected operating points')
    compare(result['protectedSourcesEvaluated'], [r['id'] for r in records if r['protected']], 'Protected evaluation disclosure')
    return task


def replay_production(protocol, evidence, population, panel, node):
    from analysis.config import FeatureConfig
    from analysis.features import FeatureSequence, VideoMetadata, contextualize
    replay = module('independent_fixed_product_program', REPO/'scripts/prepare-neural-production-comparison.py')
    io.require(protocol['programSha256'] == hashlib.sha256(replay.NODE_REPLAY.encode()).hexdigest(), 'Production TS program changed')
    evidence.closure(protocol['sourceCode']); evidence.closure(protocol['assets'])
    config = FeatureConfig.from_dict(evidence.document(protocol['assets']['previous'])['featureConfig'])
    entries = feature_entries(population, panel['features']['path'])
    folder = Path(evidence.capture(Path(protocol['_path']))['path']).parent
    decoded_by_id = {}
    for row in population:
        receipt = evidence.document(evidence.capture(folder/(row['id']+'.json')))
        io.require(receipt['id'] == row['id'] and receipt['sourceGroup'] == row['sourceGroup'] and receipt['labelsUsed'] is False
                   and receipt['protocol'] == io.identity(protocol['_path']) and receipt['featureCache'] == entries[row['id']]['audiovisual'], 'Production input association differs')
        with np.load(evidence.bind(receipt['featureCache']), allow_pickle=False) as cache:
            metadata = VideoMetadata(**json.loads(str(cache['metadata_json'].item())))
            sequence = FeatureSequence(cache['times'].astype(np.float64), cache['values'].astype(np.float32), tuple(str(n) for n in cache['names']), metadata)
        contextual, _ = contextualize(sequence, config)
        payload = {'repoUrl': replay.node_repo_url(REPO), 'id': row['id'], 'duration': metadata.duration,
            'times': sequence.times.tolist(), 'contextual': contextual.ravel().tolist(), 'ignoredIntervals': []}
        output = json.loads(subprocess.run([node, '--input-type=module', '-e', replay.NODE_REPLAY], input=json.dumps(payload, allow_nan=False),
                            text=True, capture_output=True, check=True).stdout)
        with np.load(evidence.bind(receipt['probabilities']), allow_pickle=False) as probabilities:
            io.require(np.array_equal(probabilities['times'], sequence.times) and set(probabilities.files) == {'times', *output['probabilities']}, 'Production score inventory/timeline differs')
            io.require(all(np.array_equal(probabilities[k], np.asarray(v, np.float32)) for k, v in output['probabilities'].items()), 'Production probability replay differs')
        expected = {'productionDefault': output['variants']['aggressive']['core'], 'productionUnion': output['variants']['none']['core']}
        compare(receipt['predictions'], expected, 'Fixed production boundaries')
        decoded_by_id[row['id']] = expected
        print(json.dumps({'productionAuditReplay': row['id']}), flush=True)
    return decoded_by_id


def audit_production(document, evidence, inventory, helper, metrics, node):
    results = document['results']; io.require([r['model'] for r in results] == ['productionDefault', 'productionUnion'], 'Production comparators differ')
    reference = results[0]
    panel, population, _, original_groups = panel_inputs(reference, evidence, inventory, helper)
    protocol = evidence.document(reference['protocol'])
    io.require(protocol['manifest'] == panel['manifest'] and protocol['features'] == panel['features'] and protocol['labelsUsedForInference'] is False, 'Production inference panel differs')
    parity = evidence.document(reference['productionParityAudit']); evidence.closure(parity)
    io.require(parity['passed'] and parity['kind'] == 'blind-generalization-production-parity-v1', 'Production parity gate absent')
    qualified = evidence.document(parity['protocol'])
    for key in ('assets', 'sourceCode', 'programSha256', 'modelIds', 'productionPolicy'):
        compare(protocol[key], qualified[key], 'Qualified production implementation')
    decoded = replay_production({**protocol, '_path': reference['protocol']['path']}, evidence, population, panel, node)
    records = [r for r in population if 'evaluate' in r['eligibleRoles'] and r['scoringPolicy'] != 'none']
    used = {r['sourceGroup'] for r in records if not clean(inventory[r['id']], FILTERS[2])}
    panels = independent_panels(records, inventory, original_groups, used)
    for result in results:
        compare(result['panels'], panels, 'Production exposure panel')
        io.require(result['protocol'] == reference['protocol'] and result['panel'] == reference['panel'] and result['inventory'] == reference['inventory']
                   and result['variant'] == 'fixed-production' and result['draw'] == 'fixed' and result['precision'] == 'shipped', 'Fixed comparator identity differs')
        compare(result['floors'], [{'floorPercent': f, 'status': 'available', 'operatingPointKey': 'fixed', 'selection': None,
                                   'fixedComparatorNoFloorQualification': True} for f in range(90, 101)], 'Fixed comparator floor display')
        io.require(set(result['operatingPoints']) == {'fixed'}, 'Production was recalibrated')
        audit_point(result, 'fixed', records, {r['id']: decoded[r['id']][result['model']] for r in records}, panels, metrics)
    return results


def scope(ids): return io.canonical(sorted(ids))[:20]


def flat_points(result):
    rows = []
    for floor in result['floors']:
        point = result['operatingPoints'].get(floor['operatingPointKey'])
        by_panel = {tuple(p[k] for k in PANEL_KEYS): p for p in point['panels']} if point else {}
        for panel in result['panels']:
            value = by_panel.get(tuple(panel[k] for k in PANEL_KEYS))
            status = value['status'] if value else floor['status'] if panel['recordingIds'] else 'empty-scope'
            decision = floor.get('selection') or {}; selected = decision.get('selected') or {}
            for pad in range(4):
                metric = next((m for m in (value['padding'] or []) if m['paddingSeconds'] == pad), {}) if value else {}
                rows.append({**{k: result[k] for k in ('model', 'variant', 'draw', 'precision')}, **{k: panel[k] for k in PANEL_KEYS},
                    'floorPercent': floor['floorPercent'], 'paddingSeconds': pad, 'status': status, 'scopeId': scope(panel['recordingIds']),
                    'aggregationScope': 'one model fit; pooled recording durations',
                    'calibrationRecall': selected.get('innerR_core'), 'calibrationF1': selected.get('innerF1_padP_coreR'),
                    'maximumCalibrationRecall': decision.get('maximumInnerRecall'), 'eligibleCandidateCount': decision.get('eligibleCandidateCount'),
                    **{k: metric.get(k) for k in METRICS}})
    return rows


def complete_draw_summaries(rows, expected_draws):
    grouped = defaultdict(list)
    identity = tuple(k for k in ROW_KEY if k != 'draw')
    for row in rows:
        if row['variant'] != 'fixed-production': grouped[tuple(row[k] for k in identity)].append(row)
    result = []
    for key, members in grouped.items():
        base = dict(zip(identity, key)); required = expected_draws[(base['variant'], base['model'], base['precision'])]
        io.require(len(members) == len(required) and {r['draw'] for r in members} == set(required), 'Registered draw is missing or duplicated')
        if len({r['scopeId'] for r in members}) != 1: continue
        complete = all(r['status'] == 'available' for r in members)
        row = {**base, 'draw': 'all registered draws', 'scopeId': members[0]['scopeId'],
            'status': 'available' if complete else 'empty-scope' if all(r['status'] == 'empty-scope' for r in members) else 'incomplete-registered-draws',
            'aggregationScope': 'mean across every registered draw; each draw pools the same recordings',
            'registeredDraws': list(required), 'completeDraws': [r['draw'] for r in members if r['status'] == 'available']}
        for name in METRICS:
            row[name] = statistics.mean(r[name] for r in members) if complete and all(r.get(name) is not None for r in members) else None
        for name in ('precisionValue', 'recallValue', 'f1Value'):
            row[name+'Range'] = [min(r[name] for r in members), max(r[name] for r in members)] if complete else None
        result.append(row)
    return result


def audit_historical(entry, evidence, inventory, metrics):
    path = evidence.bind(entry['result']); report = io.read(path)
    protocol = evidence.document(report['protocol']); evidence.closure(protocol)
    original_protocol_ref = protocol['historicalProtocol'] if 'historicalProtocol' in protocol else report['protocol']
    original_protocol = evidence.document(original_protocol_ref)
    original_ids = [r['id'] for r in evidence.document(original_protocol['records'])['records']]
    duration_checks = historical_duration_checks(original_protocol_ref, original_protocol, evidence, inventory)
    historical_dir = Path(original_protocol_ref['path']).parent
    for filename, kind, auditor in (
        ('audit-refits.json', 'independent-historical-all-outer-epochs-audit-v1', 'audit-neural-historical-refits.py'),
        ('audit-owners.json', 'independent-historical-score-owner-provenance-audit-v1', 'audit-neural-historical-owners.py')):
        gate = evidence.document(evidence.capture(historical_dir/filename))
        io.require(gate['passed'] and gate['kind'] == kind and gate['protocol'] == original_protocol_ref
                   and gate['auditor'] == io.identity(REPO/'scripts'/auditor), 'Historical fit/source numerical gate differs')
        if filename == 'audit-refits.json': io.require(gate['counts']['outerFits'] == 72, 'Historical refit scope incomplete')
        else: io.require(gate['logicalInnerViews'] == 216 and gate['physicalInnerOwners'] == 144, 'Historical inner-owner scope incomplete')
        evidence.closure(gate)
    # Historical numerical audits are separate, already exhaustive protocols.
    original_report_ref, candidates = historical_report_gates(historical_dir, evidence)
    if report.get('kind') == 'audited-historical-precision-sweep-projection-v1':
        original = evidence.document(report['sourceReport']); audit = evidence.document(report['sourceAudit'])
        io.require(audit['passed'] and audit['kind'] == 'independent-historical-dino-precision-floor-audit-v1'
                   and audit['report'] == report['sourceReport'] and audit['protocol'] == report['protocol'] == original['protocol']
                   and audit['auditor'] == io.identity(REPO/'scripts/audit-neural-historical-precision-sweep.py'), 'Precision projection source audit differs')
        evidence.bind(report['code']); evidence.closure(audit)
        arm = entry['precision']; io.require(report['precision'] == arm and report['precisionSpecificSelection'] is False, 'Projection precision changed')
        cells = [r for r in original['cells'] if r['precision'] == arm]
        io.require(len(cells) == len(report['cells']) == 6, 'Projection model/seed scope differs')
        for saved, source in zip(report['cells'], cells):
            io.require(all(saved[k] == source[k] for k in ('model', 'variant', 'seed', 'selectionDesign', 'precision'))
                       and saved['panels']['historical-nested-exact']['result'] == source['result'], 'Projected precision cell changed')
        candidates.append((Path(report['sourceAudit']['path']), audit))
    else:
        io.require(entry['result'] == original_report_ref, 'Historical FP32 report authority differs')
    io.require(candidates, 'Historical report lacks its independently passing numerical audit')
    for path, audit in candidates:
        evidence.capture(path); evidence.closure(audit)
    expected_models = set(selection_auditor.MODEL_NAMES.values()) if entry.get('precision', 'fp32') == 'fp32' else {'dino_tcn_short_boost', 'dino_transformer'}
    check_historical_scope(report, original_ids, expected_models, entry.get('precision', 'fp32'))
    reverse = {v: k for k, v in selection_auditor.MODEL_NAMES.items()}
    rows = []
    for cell in report['cells']:
        for panel_id, panel in cell['panels'].items():
            io.require(panel['labelPolicy'] == 'exact-rallies', 'Historical policy differs')
            for floor in panel['result']['floors']:
                for raw in floor['predictions']:
                    authoritative = inventory[raw['id']]
                    compare(raw['sourceGroup'], authoritative['sourceGroup'], 'Historical source identity')
                    validate_historical_duration(raw, authoritative, duration_checks[raw['id']])
                    compare(raw['rallies'], simple_intervals(authoritative['rallies']), 'Historical/common exact gold revision')
                    compare(raw['ignoredIntervals'], simple_intervals(authoritative.get('ignoredIntervals', [])), 'Historical/common ignored revision')
                for filt in FILTERS:
                    ids = [k for k in panel['expectedRecordingIds'] if clean(inventory[k], filt)]
                    complete = bool(ids) and floor['completeEvaluationScope']
                    raw = [r for r in floor['predictions'] if r['id'] in ids]
                    if complete: io.require({r['id'] for r in raw} == set(ids), 'Historical complete scope omits a video')
                    measured = metrics.measure(raw, 'exact-rallies') if complete else []
                    for pad in range(4):
                        metric = measured[pad] if complete else {}
                        rows.append({'model': reverse[cell['model']], 'variant': 'historical-nested', 'draw': cell['seed'],
                            'precision': entry.get('precision', 'fp32'), 'panelId': panel_id, 'labelPolicy': 'exact-rallies',
                            'productionFilter': filt, 'floorPercent': floor['floorPercent'], 'paddingSeconds': pad,
                            'status': 'available' if complete else 'empty-scope' if not ids else 'infeasible-inner-recall',
                            'scopeId': scope(ids), 'aggregationScope': 'one seed; pooled disjoint source-held outer folds',
                            **{k: metric.get(k) for k in METRICS}})
    return rows


def check_historical_scope(report, original_ids, expected_models, precision):
    io.require(len(original_ids) == len(set(original_ids)) == 8, 'Historical reference is not original eight')
    expected_cells = {(model, seed) for model in expected_models for seed in (3407, 1729, 20260918)}
    io.require(len(report['cells']) == len(expected_cells)
               and {(c['model'], c['seed']) for c in report['cells']} == expected_cells, 'Historical model/seed scope differs')
    for cell in report['cells']:
        io.require(set(cell['panels']) == {'historical-nested-exact'}, 'Historical declared panels differ')
        panel = cell['panels']['historical-nested-exact']
        expected = {'panelId': 'historical-nested-exact', 'labelPolicy': 'exact-rallies', 'reviewStatus': 'manually-reviewed',
                    'sourcePolicy': 'source-held', 'expectedRecordingIds': original_ids}
        io.require(all(panel[k] == v for k, v in expected.items()), 'Historical original-eight panel was narrowed or changed')
        io.require([f['floorPercent'] for f in panel['result']['floors']] == list(range(90, 101)), 'Historical floor scope differs')
        if precision != 'fp32':
            io.require(cell['precision'] == precision and cell['precisionSpecificSelection'] is False, 'Historical precision transfer differs')


def audit_flat(report, expected, expected_draws, inventory, results, evidence):
    compare(report['inventory'], list(inventory.values()), 'Flat inventory')
    for key, ids in report['scopes'].items():
        io.require(key == scope(ids) and ids == sorted(ids) and len(set(ids)) == len(ids), 'Compact scope identity differs')
    expected += complete_draw_summaries(expected, expected_draws)
    expected = [{k: v for k, v in r.items() if v is not None} for r in expected if r['status'] != 'empty-scope']
    by_key = {tuple(r[k] for k in ROW_KEY): r for r in report['series']}
    io.require(len(by_key) == len(report['series']) == len(expected), 'Flat operating-point inventory differs')
    for row in expected:
        key = tuple(row[k] for k in ROW_KEY)
        io.require(key in by_key, 'Missing flat operating point')
        compare(by_key[key], row, 'Flat operating point/complete mean')
        io.require(row['scopeId'] in report['scopes'], 'Flat scope array missing')
    saved_tasks = {(t['id'], t['precision']): t for t in report['tasks']}
    io.require(len(saved_tasks) == len(report['tasks']) == len(results), 'Task role inventory differs')
    for result in results:
        saved = saved_tasks[(result['taskId'], result['precision'])]
        io.require(all(saved[k] == result[k] for k in ('model', 'variant', 'draw', 'precision')), 'Task display identity differs')
        evaluated = sorted({key for panel in result['panels'] for key in panel['recordingIds']})
        inferred = [r['id'] for r in inventory.values() if r['environment'] != 'beach']
        if 'task' in result:
            task = evidence.document(result['task']); selection = evidence.document(result['selection'])
            train, cal = task['trainIds'], selection['selectionRecordingIds']
            io.require(saved['selectionLabelPolicy'] == selection['selectionLabelPolicy'], 'Displayed calibration policy differs')
            fit = evidence.document(evidence.capture(Path(result['selection']['path']).parent/'fit-result.json'))
            completed = evidence.document(fit['temporal'])
            source = {r['id']: r for r in evidence.document(task['manifest'])['records']}
            expected_statistics = {'recordings': len(train), 'sourceGroups': len({source[k]['sourceGroup'] for k in train}),
                'approximateRallyProxyRecordings': sum(bool(source[k].get('experimentalSupervision')) for k in train),
                'optimizerSteps': completed['optimizerSteps'], 'epochs': completed['epochs'],
                'trainingBudgetInterpretation': 'Fixed epochs, not fixed optimizer-update count; moving coverage to four-head supervision changes sampling and scaling too.'}
            compare(saved['trainingStatistics'], expected_statistics, 'Displayed training statistics')
            entry = REUSE_MAPPINGS.get(task['taskId'])
            mapping = entry['mapping'] if entry else None
            reused = reuse_required(task, fit, mapping)
            expected_execution = {'trainingPerformedForThisTask': not reused,
                'physicalOwnerTaskId': mapping['physicalOwnerTaskId'] if reused else task['taskId'],
                'reusedAllFixedCheckpointEpochs': list(io.EPOCHS) if reused else [],
                'calibrationAndSelectionAreTaskLocal': True, 'reuseReceipt': fit.get('trainingReuse'),
                'reusePlan': entry['plan'] if reused else None}
            compare(saved['trainingExecution'], expected_execution, 'Displayed physical/logical training ownership')
        else:
            train = [r['id'] for r in inventory.values() if r['productionExposure']['rallyPipeline']['trainingOrRelated']]
            cal = [r['id'] for r in inventory.values() if r['productionExposure']['rallyPipeline']['directCalibrationHeads'] or r['productionExposure']['rallyPipeline']['sameGroupCalibrationHeads']]
        compare(saved['memberships'], {'fit': train, 'calibrate': cal, 'evaluate': evaluated, 'infer': inferred}, 'Displayed task roles')
    return len(expected)


def task_display_evidence(result):
    """Release already-audited raw points while retaining task-display evidence."""
    return {key: result[key] for key in ('taskId', 'model', 'variant', 'draw', 'precision', 'panels', 'task', 'selection')
            if key in result}


def audit(index_path, report_path, output, node):
    PANEL_CHECKS.clear()
    REUSE_MAPPINGS.clear()
    output = Path(output); io.require(output.resolve().is_relative_to('/mnt/freenas'), 'All audit output must use NAS')
    io.require(not output.exists(), 'Audit output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(); index_ref = evidence.capture(index_path); index = evidence.document(index_ref)
    original_auditor = {'path': str(REPO/'scripts/audit-neural-generalization-results.py'), 'sha256': ORIGINAL_AUDITOR_SHA}
    evidence.bind(original_auditor)
    report_ref = evidence.capture(report_path); report = evidence.document(report_ref)
    io.require(index['kind'] == 'registered-generalization-result-index-v1' and report['metadata']['sourceIndex'] == index_ref, 'Final report input index differs')
    for index_key, metadata_key in (('studyProtocol', 'protocol'), ('timingAmendment', 'timingAmendment')):
        evidence.bind(index[index_key])
        io.require(index['metadata'][metadata_key] == index[index_key]
                   and report['metadata'][metadata_key] == index[index_key], 'Protocol/amendment metadata differs')
    inventory_doc = evidence.document(index['inventory']); inventory = {r['id']: r for r in inventory_doc['records']}
    inventory_audit_path = Path(index['inventory']['path']).parent/'audit-v2.json'
    inventory_audit = evidence.document(evidence.capture(inventory_audit_path))
    io.require(inventory_audit['passed'] and inventory_audit['inventory']['sha256'] == index['inventory']['sha256'], 'Inventory production exposure/gold audit differs')
    evidence.closure(inventory_audit)
    for path, sha in inventory_audit['dependencyHashes'].items(): evidence.bind({'path': path, 'sha256': sha})
    io.require(len(inventory) == 44 and len(index['evaluations']) == 271 and len(index['historical']) == 3, 'Full study population incomplete')
    helper_path = evidence.bind({'path': str(REPO/'scripts/audit-neural-short-boost-intervals.py'), 'sha256': HELPER_SHA})
    helper = module('independent_generalization_results_intervals', helper_path)
    evidence.capture(__file__); evidence.capture(REPO/'scripts/audit-neural-generalization-selection.py')
    evidence.capture(REPO/'analysis/neural_generalization_report.py'); evidence.bind(index['indexer'])
    registered = set()
    for reference in index['registrations']:
        registration = evidence.document(reference); io.require(io.canonical(registration['contract']) == registration['sha256'], 'Registration changed')
        evidence.closure(registration['contract'])
        io.require(not registered & set(registration['contract']['taskIds']), 'Task registered twice')
        registered.update(registration['contract']['taskIds'])
    io.require(len(registered) == 162, 'Registered task matrix incomplete')
    io.require(len(index['reusePlans']) == 2
        and index['metadata']['reusePlans'] == index['reusePlans']
        and report['metadata']['reusePlans'] == index['reusePlans'], 'Companion reuse plans absent or changed')
    reuse_auditor = module('independent_final_reuse_mapping', REPO/'scripts/audit-neural-generalization-reuse.py')
    reuse_counts = {'logicalHeadFits': 18, 'physicalHeadFits': 18, 'reusedHeadFits': 0,
                    'logicalStudentFits': 3, 'physicalStudentFits': 3}
    mapped_registrations = []
    for reference in index['reusePlans']:
        plan = evidence.document(reference)
        tasks, mappings = reuse_auditor.audit_mapping(plan, evidence)
        io.require(plan['registration'] in index['registrations'] and not set(mappings) & set(REUSE_MAPPINGS),
                   'Reuse plan is outside registrations or duplicates tasks')
        mapped_registrations.append(plan['registration'])
        for key, mapping in mappings.items():
            io.require(tasks[key]['variant'] != 'original-corpus', 'Original task improperly mapped for reuse')
            REUSE_MAPPINGS[key] = {'plan': reference, 'mapping': mapping}
        for key in reuse_counts: reuse_counts[key] += plan['counts'][key]
    io.require(len({r['sha256'] for r in mapped_registrations}) == 2 and len(REUSE_MAPPINGS) == 144
        and reuse_counts == {'logicalHeadFits': 162, 'physicalHeadFits': 120, 'reusedHeadFits': 42,
                             'logicalStudentFits': 27, 'physicalStudentFits': 20}, 'Logical/physical reuse accounting differs')
    compare(index['executionCounts'], reuse_counts, 'Indexed physical/logical execution counts')
    compare(index['metadata']['executionCounts'], reuse_counts, 'Index metadata execution counts')
    compare(report['metadata']['executionCounts'], reuse_counts, 'Report metadata execution counts')
    metrics = MetricCache(helper, output.with_name(output.stem+'-rally-guardrails.jsonl'))
    expected, expected_draws, results, seen = [], {}, [], set()
    try:
        for entry in index['evaluations']:
            document = evidence.document(entry['result'])
            if document['kind'] == 'fixed-production-generalization-evaluation-v1':
                evaluated = audit_production(document, evidence, inventory, helper, metrics, node)
            else:
                io.require(document['kind'] == 'neural-generalization-task-evaluation-v1', 'Unknown evaluation kind')
                task = audit_neural(document, evidence, inventory, helper, metrics)
                io.require(task['taskId'] in registered, 'Unregistered evaluation')
                draws = [3407, 1729, 20260918] if task['variant'] == 'original-corpus' else [3407, 1729, 20260918, 20260923]
                io.require(entry['expectedDraws'] == draws, 'Expected draw declaration reduced scope')
                expected_draws[(task['variant'], task['model'], document['precision'])] = draws
                evaluated = [document]
            for result in evaluated:
                signature = (result['taskId'], result['precision'])
                io.require(signature not in seen, 'Duplicate task/precision result'); seen.add(signature)
                io.require(result['inventory'] == index['inventory'], 'Evaluation used another production lineage')
                expected.extend(flat_points(result)); results.append(task_display_evidence(result))
            print(json.dumps({'auditedEvaluation': entry['result']['path'], 'done': len(results)}), flush=True)
        io.require({r['taskId'] for r in results if r['variant'] != 'fixed-production'} == registered, 'Registered task omitted')
        for entry in index['historical']:
            extra = audit_historical(entry, evidence, inventory, metrics); expected.extend(extra)
            for row in extra: expected_draws[(row['variant'], row['model'], row['precision'])] = [3407, 1729, 20260918]
        count = audit_flat(report, expected, expected_draws, inventory, results, evidence)
    finally:
        metrics.close()
    guardrails_ref = evidence.capture(metrics.output)
    for reference in list(evidence.references.values()): evidence.bind(reference)
    content = {k: report[k] for k in ('inventory', 'scopes', 'series', 'tasks')}
    result = {'kind': 'independent-generalization-result-report-audit-v2', 'passed': True,
        'originalAuditor': original_auditor, 'auditContractRevision': REVISION_POLICY,
        'resultIndex': index_ref, 'candidateReport': report_ref, 'reportContentSha256': io.canonical(content),
        'auditor': io.identity(__file__), 'guardrails': guardrails_ref, 'references': list(evidence.references.values()),
        'counts': {'registeredTasks': len(registered), 'taskPrecisionEvaluationsIncludingTwoProduction': len(results),
                   'flatRows': count, 'independentMetricScopes': len(metrics.cache)},
        'trainingPerformed': False, 'gpuUsed': False, 'externalOutcomesUsedForSelection': False,
        'scope': 'Every registered neural task/precision and all42 inference receipts; saved-score decoder replay and authoritative label tiers; all filtered panels/all4 padding interval metrics; canonical event matching and independent original-rally slices; all42 fixed-production TS forwards; complete same-population all3/all4 draw summaries. Neural network forward replay is bound through separate fit and all42 inference numerical audits.'}
    io.write_new(output, result); print(json.dumps({'passed': True, 'audit': io.identity(output)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--node', default='/mnt/c/Program Files/nodejs/node.exe')
    a = p.parse_args(); audit(a.index, a.report, a.output, a.node)
