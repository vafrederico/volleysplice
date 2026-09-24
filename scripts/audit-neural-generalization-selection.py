#!/usr/bin/env python3
"""Independent CPU audit of frozen generalization calibration selections.

Shares the registered decoder and score schema, not selection construction,
candidate arithmetic or floor selection. No external-panel scores are read.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as io
from analysis import neural_development as decoder
from analysis.neural_generalization_experiment import MODELS
from analysis.recognition_temporal_model import model_metadata
from analysis.schema import Interval

HELPER_SHA = '8ed1ed75ebc2511c9603642c0b09ba5888554910072a6e1d1c5490a65864b81b'
INITIAL_STUDENT_REGISTRATION = '20b452deb0b78758cb59ef8d3db775a636b4e571881702b38646d2744d2b14ac'
COMMON = frozenset((private_value('source-group-001'), private_value('source-group-008')))
MODEL_NAMES = {'av-tcn': 'av_tcn_short_boost', 'dino-tcn': 'dino_tcn_short_boost',
    'mobile-tcn': 'mobile_tcn', 'distilled-mobile-tcn': 'distilled_mobile_tcn',
    'av-transformer': 'av_transformer', 'dino-transformer': 'dino_transformer'}
EPOCHS = (5, 15, 30, 60)


class Evidence:
    """Rehash once per process; fail if a previously verified file changes."""
    def __init__(self):
        self.cache = {}
        self.references = {}

    def bind(self, reference):
        path = Path(reference['path'])
        stat = path.stat()
        signature = (reference['sha256'], stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
        if str(path) in self.cache:
            io.require(self.cache[str(path)] == signature, 'Previously audited artifact changed: '+str(path))
        else:
            io.require(io.identity(path)['sha256'] == reference['sha256'], 'Artifact hash changed: '+str(path))
            after = path.stat()
            io.require(signature[1:] == (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev), 'Artifact changed while hashing')
            self.cache[str(path)] = signature
        self.references[str(path)] = {'path': str(path), 'sha256': reference['sha256']}
        return path

    def capture(self, path):
        reference = io.identity(path)
        self.bind(reference)
        return reference

    def document(self, reference):
        return io.read(self.bind(reference))

    def closure(self, tree):
        if isinstance(tree, dict):
            if isinstance(tree.get('path'), str) and isinstance(tree.get('sha256'), str):
                self.bind(tree)
            for value in tree.values():
                self.closure(value)
        elif isinstance(tree, list):
            for value in tree:
                self.closure(value)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def groups(owner):
    return set(owner['trainGroups']) | {g for values in owner.get('auxiliaryGroups', {}).values() for g in values}


def static_closure(start):
    pending, found = [REPO / start], set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = []
            if isinstance(node, ast.ImportFrom):
                if node.level and path.parent.name == 'analysis':
                    names = [node.module.split('.')[0]] if node.module else [a.name for a in node.names]
                elif node.module and node.module.startswith('analysis.'):
                    names = [node.module.split('.')[1]]
            elif isinstance(node, ast.Import):
                names = [a.name.split('.')[1] for a in node.names if a.name.startswith('analysis.')]
            pending.extend(p for name in names if (p := REPO/'analysis'/(name+'.py')).exists() and p not in found)
    return {str(p.relative_to(REPO)) for p in found}


def check_membership(task, rows):
    by_id = {r['id']: r for r in rows}
    io.require(len(by_id) == len(rows), 'Duplicate source recording')
    train, cal = task['trainIds'], task['calibrationIds']
    io.require(train and len(set(train)) == len(train) and len(set(cal)) == len(cal)
               and not set(train) & set(cal) and set(train+cal) <= set(by_id), 'Invalid task membership')
    train_groups, cal_groups = ({by_id[k]['sourceGroup'] for k in ids} for ids in (train, cal))
    io.require(not train_groups & cal_groups, 'Source group leaks from fitting to calibration')
    io.require(not (train_groups | cal_groups) & (COMMON | set(task['commonEvaluationGroups'])), 'External common group used for fit/selection')
    for role, ids in (('fit', train), ('calibrate', cal)):
        for key in ids:
            row = by_id[key]
            io.require(not row['protected'] and row['environment'] != 'beach'
                       and row['consent']['train'] is True and role in row['eligibleRoles'], 'Source authorization differs')
            io.require(row['labelTier'] in ('exact', 'draft', 'coverage'), 'Unscored source in supervised task')
            if role == 'calibrate':
                io.require(row['labelTier'] == 'exact' and row['scoringPolicy'] == 'exact-core', 'Invalid calibration label tier')
    return by_id


def check_proxy_inputs(manifest, evidence, helper):
    if not any(r.get('experimentalSupervision') for r in manifest['records']):
        return
    original = evidence.document(manifest['originalManifest'])
    proxies = evidence.document(manifest['proxySource'])
    originals = {r['id']: r for r in original['records']}
    sources = {r['id']: r for r in proxies['records']}
    modified = {r['id'] for r in manifest['records'] if r.get('experimentalSupervision')}
    io.require(modified == set(sources) and len(modified) == 18, 'Proxy scope differs')
    for row in manifest['records']:
        old = originals[row['id']]
        if row['id'] not in modified:
            io.require(row == old, 'Unrelated gold changed in proxy manifest')
            continue
        source = sources[row['id']]
        io.require(old['labelTier'] == 'coverage' and not old['protected']
                   and row['rallies'] == source['rallies'] and row['experimentalSupervision']['independentSemanticGold'] is False,
                   'Saved approximate cores changed or misrepresented')
        for key in ('video', 'durationSeconds', 'roi', 'sourceGroup', 'contentSha256', 'keepTargets', 'gameWindow'):
            io.require(row.get(key) == old.get(key), 'Proxy changed media/coverage identity: '+key)
        io.require(all(r['end']-r['start'] >= .25 for r in row['rallies'])
                   and all(b['start'] >= a['end'] for a, b in zip(row['rallies'], row['rallies'][1:])), 'Proxy core duration or separation differs')
        spans = helper.ranges(old.get('ignoredIntervals', [])) + helper.ranges(source['ignoredIntervals'])
        for window in (old.get('gameWindow'), source.get('gameWindow')):
            if window:
                if window['start'] > 0:
                    spans.append((0., window['start']))
                if window['end'] < row['durationSeconds']:
                    spans.append((window['end'], row['durationSeconds']))
        clipped = [(max(0., a), min(row['durationSeconds'], b)) for a, b in spans if max(0., a) < min(row['durationSeconds'], b)]
        io.require(helper.boolean_intervals(helper.ranges(row['ignoredIntervals'])) == helper.boolean_intervals(clipped),
                   'Proxy lost ignored or outside-reviewed-window time')


def audit_task_owner(task_path, folder, selection, evidence, helper):
    task_ref = evidence.capture(task_path)
    task = io.read(task_path)
    io.require(selection['task'] == task_ref and selection['taskId'] == task['taskId'], 'Wrong selection task')
    registration = evidence.document(task['registration'])
    contract = registration['contract']
    io.require(registration['sha256'] == io.canonical(contract) == task['registrationSha256'], 'Registration contract changed')
    plain = {k: v for k, v in task.items() if k not in ('registration', 'registrationSha256')}
    io.require(io.canonical(plain) in contract['taskDigests'] and task['taskId'] in contract['taskIds'], 'Task not registered')
    io.require(task['epochs'] == list(EPOCHS) and task['lossArm'] == 'short_boost'
               and task['floorsPercent'] == list(range(90, 101)) and task['targetPaddingSeconds'] == 2
               and task['joinGapSeconds'] == 3 and task['productionPromotionAllowed'] is False,
               'Registered numerical scope differs')
    io.require(contract['manifest'] == task['manifest'] and contract['features'] == task['features']
               and contract['selectionUsesAdditionalPanelOutcomes'] is False, 'Registered input/panel contract differs')
    evidence.closure(contract)
    io.require(set(contract['code']) == set(contract['archivedCode']), 'Archived code inventory differs')
    for name, reference in contract['code'].items():
        io.require(reference['sha256'] == contract['archivedCode'][name]['sha256'], 'Archived registered source changed')
    declared = {**contract['code'], **selection['code']}
    needed = static_closure('analysis/neural_generalization_results.py')
    io.require(needed <= set(declared), 'Local selection import closure is not hash-bound: '+str(sorted(needed-set(declared))))
    for name, ref in declared.items():
        io.require(Path(ref['path']).resolve() == (REPO/name).resolve(), 'Code reference points outside current source')
        evidence.bind(ref)
    qualification = evidence.document(contract['qualification'])
    io.require(qualification['passed'] is True, 'Input engineering qualification absent')
    manifest = evidence.document(task['manifest'])
    features = evidence.document(task['features'])
    by_id = check_membership(task, manifest['records'])
    check_proxy_inputs(manifest, evidence, helper)
    feature_by_id = {r['recordingId']: r for r in features['records']}
    io.require(len(feature_by_id) == len(features['records']), 'Duplicate feature recording')
    for key in task['trainIds']+task['calibrationIds']:
        entry, row = feature_by_id[key], by_id[key]
        io.require(entry['sourceGroup'] == row['sourceGroup'] and entry['contentSha256'] == row['contentSha256'], 'Training feature source association differs')
        refs = {**row.get('featureCaches', {}), **entry['features']}
        evidence.bind(refs['audiovisual'])
        if task['model'].startswith('dino-'):
            evidence.bind(refs['dino']['fp32'])
        elif task['model'] == 'mobile-tcn':
            evidence.bind(refs['mobile'])
    fit_ref = evidence.capture(folder/'fit-result.json')
    fit = io.read(fit_ref['path'])
    io.require(fit['task'] == task_ref and fit['taskId'] == task['taskId']
               and fit['config'] == asdict(MODELS[task['model']])
               and fit['externalLabelsUsed'] is False and fit['selectionPerformed'] is False,
               'Fit owner/model/isolation differs')
    io.require(Path(fit['temporal']['path']) == folder/'temporal/completed.json', 'Wrong fit directory')
    owner = evidence.document(fit['temporal'])
    config = MODELS[task['model']]
    io.require(owner['kind'] == f'{config.family}_{config.head}' and owner['model'] == model_metadata(config),
               'Temporal checkpoint architecture differs from registered model')
    expected = {tier: [key for key in task['trainIds'] if by_id[key]['labelTier'] == tier] for tier in ('exact', 'draft', 'coverage')}
    io.require(owner['trainIds'] == expected['exact'] and owner['auxiliaryIds'] == {k: expected[k] for k in ('draft', 'coverage')}
               and owner['scalerTrainIds'] == expected['exact'], 'Temporal supervision/scaler membership differs')
    io.require(owner['trainGroups'] == sorted({by_id[k]['sourceGroup'] for k in expected['exact']})
               and owner['auxiliaryGroups'] == {tier: sorted({by_id[k]['sourceGroup'] for k in expected[tier]}) for tier in ('draft', 'coverage')},
               'Temporal group lineage differs')
    calibration = task['calibrationIds'] or ['__fit_output_probe__']
    calibration_groups = sorted({by_id[k]['sourceGroup'] for k in task['calibrationIds']}) if task['calibrationIds'] else ['__reserved_output_probe__']
    io.require(owner['validationIds'] == calibration and owner['validationGroups'] == calibration_groups
               and fit['calibrationPredictionIds'] == calibration
               and fit['syntheticOutputProbeOnly'] == (not bool(task['calibrationIds'])), 'Calibration/output probe population differs')
    io.require(owner['epochs'] == list(EPOCHS) and owner['seed'] == task['seed']
               and owner['contractSha256'] == task['registrationSha256'] and owner['lossArm'] == 'short_boost', 'Fit contract differs')
    io.require([r['epoch'] for r in owner['history']] == list(range(1, 61))
               and all(np.isfinite(r['loss']) for r in owner['history'])
               and all(a['optimizerSteps'] < b['optimizerSteps'] for a, b in zip(owner['history'], owner['history'][1:])), 'Incomplete/nonfinite fit history')
    expected_artifacts = {f'{kind}-{epoch}.npz' for epoch in EPOCHS for kind in ('weights', 'predictions')}
    io.require(set(owner['artifacts']) == expected_artifacts, 'Checkpoint/score inventory differs')
    for name, sha in owner['artifacts'].items():
        evidence.bind({'path': str(folder/'temporal'/name), 'sha256': sha})
    if task['model'] == 'distilled-mobile-tcn':
        audit_student(task, folder, fit, by_id, features, expected, evidence)
    else:
        io.require('student' not in fit, 'Unexpected encoder owner')
    return task, manifest, features, owner


def initial_student_registration_path(output_root):
    return Path(output_root)/'distilled-mobile-v1/preregistration.json'


def audit_student(task, folder, fit, by_id, features, expected, evidence):
    from analysis import neural_mobile_distillation as student_code
    initial_registration = evidence.document(evidence.capture(initial_student_registration_path(student_code.OUTPUT)))
    io.require(io.canonical(initial_registration['contract']) == initial_registration['sha256'] == INITIAL_STUDENT_REGISTRATION,
               'Original student registration changed')
    initial = initial_registration['contract']['mobileCheckpoint']
    io.require(Path(initial['path']) == student_code.CHECKPOINT, 'Student initial checkpoint path differs')
    evidence.bind(initial)
    io.require(Path(fit['student']['path']) == folder/'student/completed.json', 'Wrong student owner')
    student = evidence.document(fit['student'])
    ordered = [k for tier in ('exact', 'draft', 'coverage') for k in expected[tier]]
    train_groups = {by_id[k]['sourceGroup'] for k in ordered}
    io.require(student['trainIds'] == ordered and student['trainGroups'] == sorted(train_groups)
               and student['excludedGroups'] == sorted({r['sourceGroup'] for r in by_id.values()}-train_groups)
               and student['contractSha256'] == task['registrationSha256'] and student['seed'] == task['seed'], 'Student source exclusion/owner differs')
    io.require([h['epoch'] for h in student['history']] == list(range(1, 9))
               and all(np.isfinite(h['weightedMeanLoss']) for h in student['history'])
               and student['batchNormStatisticsFrozen'] is True, 'Student recipe/history metadata differs')
    evidence.bind(student['weights'])
    feature_by_id = {r['recordingId']: r for r in features['records']}
    membership = []
    for key in ordered:
        entry = {**by_id[key].get('featureCaches', {}), **feature_by_id[key]['features']}
        image = evidence.document(entry['imageInput'])
        teacher = evidence.document(entry['teacherTargets'])
        io.require(image['id'] == teacher['id'] == key and image['sourceGroup'] == teacher['sourceGroup'] == by_id[key]['sourceGroup']
                   and image['contract']['source']['contentSha256'] == by_id[key]['contentSha256']
                   and teacher['input'] == image['arrays']['teaching336'] and teacher['labelsUsed'] is False, 'Student image/teacher source association differs')
        evidence.closure(image['arrays']); evidence.bind(teacher['output']); evidence.bind(teacher['registration'])
        with np.load(evidence.bind(image['arrays']['timing']), allow_pickle=False) as timing:
            indexes = timing['teaching_indexes'].tolist()
        io.require(indexes == image['contract']['teachingIndexes'], 'Student teaching frame identities differ')
        membership.append({'id': key, 'group': by_id[key]['sourceGroup'], 'frameIndexes': indexes,
                           'teacher': teacher['output'], 'image': image['arrays']['images224']})
    io.require(student['membership'] == membership and student['trainingFrames'] == sum(len(r['frameIndexes']) for r in membership),
               'Student frame exposure membership differs')
    # Bind the actual encoder outputs supplied to the head, including calibration.
    for key in task['trainIds']+task['calibrationIds']:
        receipt = evidence.document(evidence.capture(folder/'student-features'/(key+'.json')))
        entry = {**by_id[key].get('featureCaches', {}), **feature_by_id[key]['features']}
        image = evidence.document(entry['imageInput'])
        io.require(receipt['encoder'] == student['weights'] and receipt['id'] == key
                   and receipt['sourceGroup'] == by_id[key]['sourceGroup']
                   and receipt['contractSha256'] == task['registrationSha256']
                   and receipt['imageInput'] == image['arrays'] and receipt['labelsUsed'] is False,
                   'Head received a different student encoder or source imagery')
        evidence.closure(receipt)


def calibration_examples(task, manifest, features, evidence):
    rows = {r['id']: r for r in manifest['records']}
    indexed = {r['recordingId']: r for r in features['records']}
    examples = []
    for key in task['calibrationIds']:
        row, feature = rows[key], indexed[key]
        io.require(feature['sourceGroup'] == row['sourceGroup'] and feature['contentSha256'] == row['contentSha256'], 'Calibration feature association differs')
        refs = {**row.get('featureCaches', {}), **feature['features']}
        with np.load(evidence.bind(refs['audiovisual']), allow_pickle=False) as payload:
            times = payload['times'].astype(np.float64)
            duration = float(json.loads(str(payload['metadata_json'].item()))['duration']) if 'metadata_json' in payload else float(row['durationSeconds'])
        io.require(times.ndim == 1 and len(times) and np.isfinite(times).all() and np.all(np.diff(times)>0), 'Invalid calibration timeline')
        io.require(abs(duration-row['durationSeconds']) < .11 and 0 <= times[0] and times[-1] <= duration, 'Calibration media duration differs')
        parse = lambda values: tuple(Interval(r['start'], r['end'], tuple(r.get('tags', ()))) for r in values)
        policy = 'export-rally-proxy-selection' if row.get('experimentalSupervision') else 'exact-rallies'
        examples.append(io.SweepExample(key, row['sourceGroup'], duration, times,
                        np.ones(len(times), bool), parse(row['rallies']), parse(row.get('ignoredIntervals', [])), policy))
    return examples


def make_shard(folder, epoch, ids, excluded, extra, evidence):
    reference = evidence.capture(folder/'completed.json')
    meta = evidence.document(reference)
    return {'epoch': epoch, 'archive': {'path': str(folder/f'predictions-{epoch}.npz'), 'sha256': meta['artifacts'][f'predictions-{epoch}.npz']},
        'archiveRecordingIds': meta['validationIds'], 'recordingIds': ids,
        'checkpoint': {'path': str(folder/f'weights-{epoch}.npz'), 'sha256': meta['artifacts'][f'weights-{epoch}.npz']},
        'trainingOwners': [reference, *extra], 'excludedSourceGroups': sorted(excluded)}


def historical_inputs(task, folder, evidence):
    protocol_ref = evidence.capture(folder/'protocol.json')
    protocol = evidence.document(protocol_ref)
    plan_ref = evidence.capture(folder/'refit-plan.json'); plan = evidence.document(plan_ref)
    audit = evidence.document(evidence.capture(folder/'audit-refits.json'))
    io.require(audit['passed'] and audit['kind'] == 'independent-historical-all-outer-epochs-audit-v1'
               and audit['protocol'] == protocol_ref and audit['plan'] == plan_ref and audit['counts']['outerFits'] == 72,
               'Historical audit scope differs')
    evidence.closure(audit)
    io.require(plan['protocol'] == protocol_ref and io.canonical(plan['tasks']) == protocol['refitTasksSha256'], 'Historical plan changed')
    evidence.closure(protocol)
    stored = evidence.document(protocol['records'])['records']
    examples = io.examples_from_rows(stored)
    io.require(len(examples) == 8 and len({e.group for e in examples}) == 4
               and all(e.label_policy == 'exact-rallies' for e in examples), 'Original calibration scope differs')
    original = evidence.document(protocol['source'])
    original_exact = {r['id']: r for r in original['exactRows']}
    io.require(set(original_exact) == {e.id for e in examples}, 'OOF exact IDs differ')
    for row in stored:
        gold = original_exact[row['id']]
        for key in ('sourceGroup', 'durationSeconds'):
            io.require(row[key] == gold[key], 'Original OOF gold revision differs')
        for key in ('rallies', 'ignoredIntervals'):
            normalized = [{'start': r['start'], 'end': r['end'], **({'tags': r['tags']} if r.get('tags') else {})} for r in gold[key]]
            io.require(row[key] == normalized, 'Original OOF interval revision differs')
        expected_valid = np.ones(len(row['timestamps']), bool)
        for interval in gold['ignoredIntervals']:
            times = np.asarray(row['timestamps'])
            expected_valid &= ~((times >= interval['start']) & (times < interval['end']))
        io.require(np.array_equal(row['valid'], expected_valid), 'Historical ignored-segment mask differs')
    layouts = evidence.document(evidence.capture(folder/'layouts.json'))
    io.require(layouts['protocol'] == protocol_ref and io.canonical(layouts['layouts']) == protocol['layoutsSha256'], 'Historical layouts changed')
    layout, = [r for r in layouts['layouts'] if r['model'] == MODEL_NAMES[task['model']] and r['seed'] == task['seed']]
    shards = []
    for fold in layout['folds']:
        owner = folder/'refits'/layout['model']/str(task['seed'])/f'outer-{fold["outerIndex"]}'
        parity_ref = evidence.capture(owner/'parity.json'); parity = evidence.document(parity_ref)
        planned, = [r for r in plan['tasks'] if r['model'] == layout['model'] and r['seed'] == task['seed'] and r['outerIndex'] == fold['outerIndex']]
        checked, = [r for r in audit['checks'] if r['task'] == planned]
        io.require(parity['passed'] and parity['task'] == planned and parity['plan'] == plan_ref
                   and parity['completed'] == evidence.capture(owner/'temporal/completed.json')
                   and checked['parity'] == parity_ref and checked['completed'] == parity['completed']
                   and checked['allEpochArraysChecked'], 'Historical audited refit ownership differs')
        ids = [e.id for e in examples if e.group == fold['heldSourceGroup']]
        for epoch in EPOCHS:
            shards.append(make_shard(owner/'temporal', epoch, ids, [fold['heldSourceGroup']], parity['additionalTrainingOwners'], evidence))
    return examples, shards


def scores_from_shards(examples, shards, evidence):
    by_id = {e.id: e for e in examples}
    scores = {epoch: {} for epoch in EPOCHS}
    for shard in shards:
        checkpoint = evidence.bind(shard['checkpoint'])
        archive = evidence.bind(shard['archive'])
        owners = [evidence.document(r) for r in shard['trainingOwners']]
        io.require(owners and shard['recordingIds'] and not set(scores[shard['epoch']]) & set(shard['recordingIds']), 'Missing/duplicate score owner')
        excluded = set(shard['excludedSourceGroups']) | {by_id[k].group for k in shard['recordingIds']}
        io.require(all(not groups(owner) & excluded for owner in owners), 'Calibration source reached encoder or temporal fitting')
        head = owners[0]
        io.require(head['validationIds'] == shard['archiveRecordingIds']
                   and head['artifacts'][archive.name] == shard['archive']['sha256']
                   and head['artifacts'][checkpoint.name] == shard['checkpoint']['sha256'], 'Checkpoint/archive belongs to a different head')
        with np.load(archive, allow_pickle=False) as payload:
            io.require(set(payload.files) == set(shard['archiveRecordingIds']), 'Validation archive membership differs')
            for key in shard['recordingIds']:
                values = payload[key].copy()
                io.require(values.shape == (len(by_id[key].times), 4) and np.issubdtype(values.dtype, np.floating)
                           and np.isfinite(values).all() and np.all((values>=0)&(values<=1)), 'Invalid calibration probability tensor')
                scores[shard['epoch']][key] = values
    io.require(all(set(v) == set(by_id) for v in scores.values()), 'Incomplete calibration scores')
    return scores


def audit_candidates(selection, examples, scores, helper):
    grid = [(epoch, {'smoothing': smooth, 'enter': enter, 'minimum': minimum, 'boundary': boundary})
            for epoch in EPOCHS for smooth in (.5, 1.) for enter in (.2, .35, .5, .65, .8, .9)
            for minimum in (.25, 1.) for boundary in (False, True)]
    io.require([(c['epoch'], c['decoder']) for c in selection['candidates']] == grid, 'Candidate order/grid changed')
    metrics = []
    for candidate in selection['candidates']:
        raw = io.serial_rows([e.row(decoder.decode(e, scores[candidate['epoch']][e.id], candidate['decoder'])) for e in examples])
        metric = helper.pooled_metric([helper.parse_record(r) for r in raw], 2.)
        helper.close(candidate['innerR_core'], metric['R_core'], 'candidate recall')
        helper.close(candidate['innerF1_padP_coreR'], metric['F1_padP_coreR'], 'candidate F1')
        if 'innerP_pad' in candidate:
            helper.close(candidate['innerP_pad'], metric['P_pad'], 'candidate precision')
        metrics.append({'epoch': candidate['epoch'], 'decoder': candidate['decoder'], 'P_pad': metric['P_pad'],
                        'R_core': metric['R_core'], 'F1_padP_coreR': metric['F1_padP_coreR']})
    audit_floors(selection)
    return metrics


def audit_floors(selection):
    io.require([r['floorPercent'] for r in selection['floors']] == list(range(90, 101)), 'Recall floor inventory differs')
    for decision in selection['floors']:
        floor = decision['floorPercent']/100.
        # Arithmetic was checked above; strict thresholding uses saved IEEE values
        # so no numerical tolerance quietly makes 99.999999% eligible for 100%.
        eligible = [c for c in selection['candidates'] if c['innerR_core'] >= floor]
        io.require(decision['recallEligibilityFloor'] == floor and decision['candidateCount'] == 192
                   and decision['feasible'] == bool(eligible) and decision['eligibleCandidateCount'] == len(eligible)
                   and decision['maximumInnerRecall'] == max(c['innerR_core'] for c in selection['candidates'])
                   and decision['selected'] == (max(eligible, key=lambda c: c['innerF1_padP_coreR']) if eligible else None),
                   'Strict recall eligibility, stable ordered tie or no-fallback selection differs')
def audit(task_path, folder, selection_path, output, historical=None, evidence=None):
    evidence = evidence or Evidence()
    evidence.references = {}
    output, folder = Path(output), Path(folder)
    io.require(output.resolve().is_relative_to('/mnt/freenas'), 'Audit output must use NAS')
    io.require(not output.exists(), 'Independent audit already exists')
    selection_ref = evidence.capture(selection_path); selection = evidence.document(selection_ref)
    io.require(selection['kind'] == 'frozen-generalization-selection-v1'
               and selection['externalEvaluationOutcomesUsed'] is False and selection['targetPaddingSeconds'] == 2
               and selection['paddingSeconds'] == [0, 1, 2, 3] and selection['joinGapSeconds'] == 3,
               'Selection contract differs')
    helper_path = evidence.bind({'path': str(REPO/'scripts/audit-neural-short-boost-intervals.py'), 'sha256': HELPER_SHA})
    helper = module('generalization_independent_interval_oracle', helper_path)
    task, manifest, features, owner = audit_task_owner(Path(task_path), folder, selection, evidence, helper)
    if task['variant'] == 'original-corpus':
        io.require(historical is not None and not task['calibrationIds'], 'Original corpus requires original OOF selection')
        examples, expected_shards = historical_inputs(task, Path(historical), evidence)
        policy = 'exact-rallies'; inference_policy = 'historical-valid-segments; original eight OOF only'
    else:
        examples = calibration_examples(task, manifest, features, evidence)
        extra = [evidence.capture(folder/'student/completed.json')] if task['model'] == 'distilled-mobile-tcn' else []
        expected_shards = [make_shard(folder/'temporal', epoch, [e.id for e in examples], [e.group for e in examples], extra, evidence) for epoch in EPOCHS]
        policy = 'export-rally-proxy-selection' if any(e.label_policy == 'export-rally-proxy-selection' for e in examples) else 'exact-rallies'
        io.require((task.get('selectionLabelPolicy') == 'exact-and-export-rally-proxy') == (policy != 'exact-rallies'), 'Proxy selection policy differs')
        inference_policy = 'blind-full-timeline; all ticks valid before attaching selection labels'
    io.require(selection['scoreShards'] == expected_shards and selection['selectionRecordingIds'] == [e.id for e in examples]
               and selection['selectionSourceGroups'] == sorted({e.group for e in examples})
               and selection['selectionLabelPolicy'] == policy and selection['selectionMetricsAreGoldAccuracy'] == (policy == 'exact-rallies')
               and selection['selectionDesign'] == task['selectionDesign'] and selection['inferencePolicy'] == inference_policy,
               'Selection data lineage/policy differs')
    gold = [{**io.serial_rows([e.row([])])[0], 'times': e.times.tolist(), 'valid': e.valid.tolist(), 'labelPolicy': e.label_policy} for e in examples]
    io.require(selection['selectionGoldSha256'] == io.canonical(gold), 'Selection gold/timestamps/masks changed')
    scores = scores_from_shards(examples, expected_shards, evidence)
    evidence.closure(selection['evidence'])
    saved_paths = {r['path'] for r in selection['evidence']}
    essential = {str(folder/'fit-result.json'), str(folder/'temporal/completed.json')}
    for shard in expected_shards:
        essential.update(r['path'] for r in [shard['archive'], shard['checkpoint'], *shard['trainingOwners']])
    io.require(essential <= saved_paths, 'Frozen selection omits required score/fit evidence')
    metrics = audit_candidates(selection, examples, scores, helper)
    for reference in list(evidence.references.values()):
        evidence.bind(reference)
    result = {'kind': 'independent-generalization-selection-audit-v1', 'passed': True, 'selection': selection_ref,
        'task': io.identity(task_path), 'taskId': task['taskId'], 'auditor': io.identity(__file__),
        'independentIntervalArithmetic': {'path': str(helper_path), 'sha256': HELPER_SHA},
        'selectionLabelPolicy': policy, 'selectionMetricsAreGoldAccuracy': policy == 'exact-rallies',
        'counts': {'candidates': len(metrics), 'floors': 11, 'recordings': len(examples), 'sourceGroups': len({e.group for e in examples})},
        'independentCandidateMetrics': metrics, 'references': list(evidence.references.values()),
        'externalPredictionScoresRead': False, 'externalLabelsUsedForSelection': False,
        'gpuUsed': False, 'trainingPerformed': False,
        'scope': 'All candidate precision/recall/F1 via independent endpoint arithmetic; exact strict floor and stable tie reselection; complete registration/code/score-owner evidence; source/head/encoder exclusions and student image/teacher memberships. No neural forward, scaler arithmetic, BN-buffer or optimizer-state replay in this selection gate.'}
    io.write_new(output, result)
    print(json.dumps({'passed': True, 'taskId': task['taskId'], 'audit': io.identity(output)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path); parser.add_argument('--fit', type=Path)
    parser.add_argument('--selection', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--jobs', type=Path, help='JSON array: task, fitDirectory, selection, output, historicalDirectory(optional)')
    args = parser.parse_args()
    evidence = Evidence()
    if args.jobs:
        jobs = io.read(args.jobs)
        for job in jobs:
            audit(job['task'], job['fitDirectory'], job['selection'], job['output'], job.get('historicalDirectory'), evidence)
    else:
        if not all((args.selection, args.output)):
            parser.error('Provide --jobs or --selection --output')
        selection = io.read(args.selection)
        args.task = args.task or Path(selection['task']['path'])
        if args.fit is None:
            fit_ref, = [r for r in selection['evidence'] if Path(r['path']).name == 'fit-result.json']
            args.fit = Path(fit_ref['path']).parent
        if args.historical is None and io.read(args.task)['variant'] == 'original-corpus':
            protocol_ref, = [r for r in selection['evidence'] if Path(r['path']).name == 'protocol.json'
                            and io.read(r['path']).get('kind') == 'historical-recall-floor-sweep-v1']
            args.historical = Path(protocol_ref['path']).parent
        audit(args.task, args.fit, args.selection, args.output, args.historical, evidence)


if __name__ == '__main__':
    main()
