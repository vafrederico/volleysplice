"""Strict recall-floor sweeps over frozen scores; no fitting or GPU execution.

Selection always uses exact rally labels. Independent reviewed-export panels
receive export-overlap diagnostics, never fabricated rally/core/event metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from . import neural_development as base
from . import neural_expanded_development as expanded
from .crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r, pad_and_merge_intervals, subtract_intervals
from .neural_evaluation import evaluate_predictions
from .neural_recall_operating_point import strict_selection, serial_rows
from .schema import Interval

FLOORS = tuple(range(90, 101))
EPOCHS = (5, 15, 30, 60)
POLICIES = ('exact-rallies', 'reviewed-export', 'reviewed-draft')
GOLD_FIELDS = {'exact-rallies': 'rallies', 'reviewed-export': 'humanExportIntervals', 'reviewed-draft': 'reviewedLiveIntervals'}
PADDINGS = (0, 1, 2, 3)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


@dataclass
class SweepExample:
    """Only the timing, validity and gold fields needed to decode saved scores."""
    id: str
    group: str
    duration: float
    times: np.ndarray
    valid: np.ndarray
    truth: tuple[Interval, ...]
    ignored: tuple[Interval, ...]
    label_policy: str = 'exact-rallies'

    def row(self, predictions):
        return {'id': self.id, 'sourceGroup': self.group, 'durationSeconds': self.duration,
                'rallies': self.truth, 'ignoredIntervals': self.ignored, 'predictions': predictions}


def examples_from_rows(rows):
    examples = []
    for row in rows:
        policy = row['labelPolicy']
        require(policy in POLICIES and row.get('environment') != 'beach', 'Unknown label policy or excluded beach recording')
        field = GOLD_FIELDS[policy]
        intervals = lambda values: tuple(Interval(v['start'], v['end'], tuple(v.get('tags', ()))) for v in values)
        times, valid = np.asarray(row['timestamps'], np.float64), np.asarray(row['valid'], bool)
        require(times.ndim == 1 and len(times) and valid.shape == times.shape and np.isfinite(times).all()
                and np.all(np.diff(times) > 0), 'Invalid timeline or validity mask')
        require(isinstance(row['id'], str) and row['id'] and isinstance(row['sourceGroup'], str)
                and row['sourceGroup'] and np.isfinite(row['durationSeconds']) and row['durationSeconds'] > 0,
                'Invalid recording identity/duration')
        examples.append(SweepExample(row['id'], row['sourceGroup'], row['durationSeconds'], times, valid,
                                     intervals(row[field]), intervals(row.get('ignoredIntervals', [])), policy))
    require(len({e.id for e in examples}) == len(examples), 'Duplicate recording id')
    return examples


def validate_scores(examples, scores_by_epoch, *, require_all_epochs=True):
    ids = {e.id for e in examples}
    require(examples and len(ids) == len(examples), 'Empty or duplicate score population')
    require(bool(scores_by_epoch) or not require_all_epochs, 'No saved checkpoint scores')
    require(set(scores_by_epoch) == set(EPOCHS) if require_all_epochs else set(scores_by_epoch) <= set(EPOCHS),
            'Checkpoint grid differs')
    for epoch, scores in scores_by_epoch.items():
        require(set(scores) == ids, f'Score population differs at epoch {epoch}')
        for e in examples:
            values = np.asarray(scores[e.id])
            require(values.shape == (len(e.times), 4) and np.issubdtype(values.dtype, np.floating)
                    and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), 'Invalid probability tensor')


def operating_point_key(candidate):
    return canonical({'epoch': candidate['epoch'], 'decoder': candidate['decoder']})


def build_candidate_table(examples, scores_by_epoch, *, selection_policy='exact-rallies'):
    """Decode each of 192 ordered candidates once for this selection population."""
    require(selection_policy in ('exact-rallies', 'export-rally-proxy-selection'), 'Unknown selection label policy')
    allowed = ('exact-rallies',) if selection_policy == 'exact-rallies' else ('exact-rallies', 'export-rally-proxy-selection')
    require(all(getattr(e, 'label_policy', 'exact-rallies') in allowed for e in examples),
            'Only exact rallies or explicitly authorized export-rally proxies may select an operating point')
    validate_scores(examples, scores_by_epoch)
    candidates = []
    for epoch in EPOCHS:
        for decoder in expanded.decoder_candidates():
            rows = [RecordingIntervals(e.id, 'development', e.duration, e.truth,
                    tuple(base.decode(e, scores_by_epoch[epoch][e.id], decoder)), e.ignored) for e in examples]
            metric = evaluate_f1_pad_p_core_r(rows, [2.], 3.)[0]
            candidate = {'epoch': epoch, 'decoder': decoder, 'innerR_core': metric['R_core'],
                         'innerF1_padP_coreR': metric['F1_padP_coreR']}
            if selection_policy != 'exact-rallies':
                candidate['selectionLabelPolicy'] = selection_policy
                candidate['selectionMetricsAreGoldAccuracy'] = False
            candidates.append(candidate)
    return candidates


def select_floors(candidates, floors=FLOORS):
    """No tolerance below the floor, including exact 100%; no fallback."""
    floors = tuple(floors)
    require(floors and len(set(floors)) == len(floors)
            and all(isinstance(f, int) and not isinstance(f, bool) and 90 <= f <= 100 for f in floors),
            'Floors must be distinct integer percentages from 90 through 100')
    expected = [(epoch, decoder) for epoch in EPOCHS for decoder in expanded.decoder_candidates()]
    require([(c['epoch'], c['decoder']) for c in candidates] == expected, 'Registered candidate order/grid differs')
    return [{'floorPercent': floor, **strict_selection(candidates, floor / 100.)} for floor in floors]


def panel_rows(examples, decoded, policy):
    require(policy in POLICIES and all(getattr(e, 'label_policy', policy) == policy for e in examples),
            'Mixed or mismatched panel label policies')
    rows = serial_rows([e.row(decoded[e.id]) for e in examples])
    if policy != 'exact-rallies':
        for row in rows:
            row[GOLD_FIELDS[policy]] = row.pop('rallies')
    return rows


def gold_signature(rows, policy):
    require(policy in POLICIES and len({r['id'] for r in rows}) == len(rows), 'Invalid gold policy/population')
    field = GOLD_FIELDS[policy]
    return {r['id']: {k: r[k] for k in ('sourceGroup', 'durationSeconds', field, 'ignoredIntervals')} for r in rows}


def _duration(values):
    return sum(v.end - v.start for v in values)


def evaluate_export_panel(rows):
    """Compare padded model export against fixed reviewed kept intervals.

    Human intervals already describe an export. Do not add synthetic human
    padding or join their gaps, and do not infer rally cores or event identities.
    """
    require(rows and len({r['id'] for r in rows}) == len(rows), 'Empty/duplicate export panel')
    padding = []
    for pad in PADDINGS:
        sums = dict(modelExportSeconds=0., humanExportSeconds=0., intersectionSeconds=0., evaluableVideoSeconds=0.)
        for row in rows:
            parse = lambda key: tuple(Interval(v['start'], v['end']) for v in row[key])
            ignored = parse('ignoredIntervals')
            human = subtract_intervals(pad_and_merge_intervals(parse('humanExportIntervals'), row['durationSeconds'], 0, 0), ignored)
            model = subtract_intervals(pad_and_merge_intervals(parse('predictions'), row['durationSeconds'], pad, 3.), ignored)
            sums['modelExportSeconds'] += _duration(model)
            sums['humanExportSeconds'] += _duration(human)
            sums['intersectionSeconds'] += _duration(model) - _duration(subtract_intervals(model, human))
            sums['evaluableVideoSeconds'] += _duration(subtract_intervals((Interval(0., row['durationSeconds']),), ignored))
        require(sums['humanExportSeconds'] > 0, 'Export panel has no evaluable human-kept time')
        p = sums['intersectionSeconds'] / sums['modelExportSeconds'] if sums['modelExportSeconds'] else 0.
        r = sums['intersectionSeconds'] / sums['humanExportSeconds']
        padding.append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3., 'humanPaddingSeconds': 0,
                        'humanJoinGapSeconds': 0, **sums, 'P_export': p, 'R_export': r,
                        'F1_export': 2 * p * r / (p + r) if p + r else 0.,
                        'exportDurationDifferenceSeconds': sums['modelExportSeconds'] - sums['humanExportSeconds'],
                        'incorrectExportSeconds': sums['modelExportSeconds'] - sums['intersectionSeconds'],
                        'wantedExportOmittedSeconds': sums['humanExportSeconds'] - sums['intersectionSeconds'],
                        'correctlyRemovedSeconds': sums['evaluableVideoSeconds'] - sums['humanExportSeconds']
                                                  - sums['modelExportSeconds'] + sums['intersectionSeconds']})
    return {'labelPolicy': 'reviewed-export', 'recordingCount': len(rows),
            'sourceGroupCount': len({r['sourceGroup'] for r in rows}), 'primary': padding[2], 'padding': padding,
            'rallyCoreMetricsAvailable': False, 'eventMetricsAvailable': False,
            'interpretation': 'Fixed reviewed-human export overlap only; no rally cores or event identities inferred.'}


def evaluate_rows(rows, policy):
    require(policy in POLICIES, 'Unknown evaluation policy')
    if policy == 'exact-rallies':
        return evaluate_predictions(rows)
    if policy == 'reviewed-export':
        return evaluate_export_panel(rows)
    records, universe = [], 0.
    for row in rows:
        parse = lambda key: tuple(Interval(v['start'], v['end']) for v in row[key])
        ignored = parse('ignoredIntervals')
        records.append(RecordingIntervals(row['id'], 'reviewed-draft', row['durationSeconds'],
                       parse('reviewedLiveIntervals'), parse('predictions'), ignored))
        universe += _duration(subtract_intervals((Interval(0., row['durationSeconds']),), ignored))
    metrics = evaluate_f1_pad_p_core_r(records, PADDINGS, 3.)
    padding = []
    for metric in metrics:
        incorrect = metric['paddedModelExportSeconds'] - metric['paddedPrecisionIntersectionSeconds']
        padding.append({'paddingSecondsBeforeAndAfter': metric['paddingSecondsBeforeAndAfter'], 'joinGapSeconds': 3.,
            'P_reviewed': metric['P_pad'], 'R_reviewed': metric['R_core'], 'F1_reviewed': metric['F1_padP_coreR'],
            'modelExportSeconds': metric['paddedModelExportSeconds'], 'humanExportSeconds': metric['paddedHumanExportSeconds'],
            'reviewedLiveSeconds': metric['coreHumanSeconds'], 'retainedReviewedLiveSeconds': metric['coreRecallIntersectionSeconds'],
            'exportDurationDifferenceSeconds': metric['exportDurationDifferenceSeconds'], 'incorrectExportSeconds': incorrect,
            'wantedExportOmittedSeconds': metric['paddedHumanExportSeconds'] - metric['paddedPrecisionIntersectionSeconds'],
            'missedReviewedLiveSeconds': metric['coreHumanSeconds'] - metric['coreRecallIntersectionSeconds'],
            'correctlyRemovedSeconds': universe - metric['paddedHumanExportSeconds'] - incorrect})
    return {'labelPolicy': policy, 'recordingCount': len(rows), 'sourceGroupCount': len({r['sourceGroup'] for r in rows}),
            'primary': padding[2], 'padding': padding, 'rallyCoreMetricsAvailable': False, 'eventMetricsAvailable': False,
            'interpretation': 'Approximate reviewed-live interval overlap; symmetric padding on model/reviewed labels. No exact rally cores, event identities, boundary errors or rally-loss claims.'}


def evaluate_selected(examples, scores_by_epoch, decisions, policy='exact-rallies'):
    """Decode/evaluate each distinct selected point once, independent of floor."""
    validate_scores(examples, scores_by_epoch, require_all_epochs=False)
    points, floors = {}, []
    for decision in decisions:
        selected = decision['selected']
        row = {'floorPercent': decision['floorPercent'], 'selection': decision, 'operatingPointKey': None}
        if selected is None:
            require(not decision['feasible'], 'Feasible decision lacks selected point')
            row['status'] = 'infeasible-inner-recall'
        elif selected['epoch'] not in scores_by_epoch:
            row['status'] = 'missing-selected-checkpoint'
        else:
            key = operating_point_key(selected)
            if key not in points:
                decoded = {e.id: base.decode(e, scores_by_epoch[selected['epoch']][e.id], selected['decoder']) for e in examples}
                rows = panel_rows(examples, decoded, policy)
                points[key] = {'epoch': selected['epoch'], 'decoder': selected['decoder'], 'predictions': rows,
                               'evaluation': evaluate_rows(rows, policy)}
            row.update(status='available', operatingPointKey=key)
        floors.append(row)
    return {'labelPolicy': policy, 'expectedGold': gold_signature(panel_rows(examples, {e.id: [] for e in examples}, policy), policy),
            'operatingPoints': points, 'floors': floors}


def pool_fold_results(folds, expected_gold, policy='exact-rallies'):
    """Pool disjoint folds only when every declared recording is available."""
    require(folds and expected_gold, 'Empty fold population')
    expected_floors = [f['floorPercent'] for f in folds[0]['floors']]
    require(all(f['labelPolicy'] == policy and [x['floorPercent'] for x in f['floors']] == expected_floors for f in folds),
            'Panel policy or floor order differs')
    inventory = {}
    for fold in folds:
        require(not set(inventory) & set(fold['expectedGold']), 'Overlapping outer evaluation folds')
        inventory.update(fold['expectedGold'])
    require(inventory == expected_gold, 'Declared complete panel gold/source scope differs')
    result = []
    for i, floor in enumerate(expected_floors):
        rows, statuses = [], []
        for fold in folds:
            chosen = fold['floors'][i]
            statuses.append(chosen['status'])
            if chosen['status'] == 'available':
                rows.extend(fold['operatingPoints'][chosen['operatingPointKey']]['predictions'])
        complete = all(status == 'available' for status in statuses)
        if complete:
            require(gold_signature(rows, policy) == expected_gold, 'Available complete gold scope differs')
        result.append({'floorPercent': floor, 'completeEvaluationScope': complete, 'foldStatuses': statuses,
                       'scopeRecordingIds': [r['id'] for r in rows], 'predictions': rows,
                       'evaluation': evaluate_rows(rows, policy) if complete else None,
                       'partialScopeNotRankable': not complete})
    return {'labelPolicy': policy, 'expectedGoldSha256': canonical(expected_gold), 'floors': result}


def summarize_seed_cells(cells, expected_seeds):
    """Aggregate only one declared variant/panel/design over every registered seed."""
    require([c['seed'] for c in cells] == list(expected_seeds) and len(set(expected_seeds)) == len(expected_seeds),
            'Missing, duplicate, or reordered registered seed')
    require(len({(c['model'], c['variant'], c['panelId'], c['selectionDesign']) for c in cells}) == 1,
            'Do not mix models, training variants, panels, or selection designs')
    first = cells[0]['result']
    require(all(c['result']['labelPolicy'] == first['labelPolicy'] and c['result']['expectedGoldSha256'] == first['expectedGoldSha256']
                and [r['floorPercent'] for r in c['result']['floors']] == [r['floorPercent'] for r in first['floors']] for c in cells),
            'Gold revision/policy/floor or source scope differs across seeds')
    exact = first['labelPolicy'] == 'exact-rallies'
    overlap_keys = ('P_export', 'R_export', 'F1_export') if first['labelPolicy'] == 'reviewed-export' else ('P_reviewed', 'R_reviewed', 'F1_reviewed')
    keys = (('P_pad', 'R_core', 'F1_padP_coreR', 'paddedModelExportSeconds', 'paddedHumanExportSeconds', 'exportDurationDifferenceSeconds')
            if exact else (*overlap_keys, 'modelExportSeconds', 'humanExportSeconds', 'exportDurationDifferenceSeconds',
                           'incorrectExportSeconds', 'wantedExportOmittedSeconds', 'correctlyRemovedSeconds'))
    output = []
    for i, row in enumerate(first['floors']):
        values = [c['result']['floors'][i] for c in cells]
        complete = all(v['completeEvaluationScope'] and v['evaluation'] is not None for v in values)
        output.append({'floorPercent': row['floorPercent'], 'allRegisteredSeedsComplete': complete,
                       'completeSeeds': [c['seed'] for c, v in zip(cells, values) if v['completeEvaluationScope']],
                       'meanPadding': [{**{k: statistics.mean(v['evaluation']['padding'][p][k] for v in values) for k in keys},
                                        'paddingSecondsBeforeAndAfter': p, 'joinGapSeconds': 3.} for p in PADDINGS] if complete else None})
    return {'floors': output, 'labelPolicy': first['labelPolicy'], 'expectedGoldSha256': first['expectedGoldSha256'],
            'aggregation': 'Pool recordings within each seed; mean only all declared complete seeds on identical scope. Never select a floor using held panels.'}
