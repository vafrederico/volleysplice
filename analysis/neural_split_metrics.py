"""Score internal rally-start split advice while preserving production coverage.

This is an evaluation-only module. Gold is used to define targets and score
proposals, never to create deployable proposals. A first gold rally inside a
production parent is a start-boundary correction, not a requested extra split.
"""
from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .crop_evaluation import subtract_intervals
from .neural_rally_identity_metrics import _maximum_weight_pairs
from .schema import Interval


SPLIT_TOLERANCES_SECONDS = (0.5, 1.0, 2.0)
PRIMARY_SPLIT_TOLERANCE_SECONDS = 1.0
DEFAULT_EDGE_GUARD_SECONDS = 2.0


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f'{name} must be finite and numeric')
    return float(value)


def _intervals(values: Iterable[Any], duration: float, name: str) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f'{name} must be an interval iterable')
    output = []
    for index, value in enumerate(values):
        if isinstance(value, Interval):
            start, end = value.start, value.end
        elif isinstance(value, Mapping):
            start, end = value.get('start'), value.get('end')
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            start, end = value
        else:
            raise ValueError(f'invalid {name}[{index}]')
        start, end = _number(start, 'start'), _number(end, 'end')
        if end <= start:
            raise ValueError(f'{name}[{index}] must have start < end')
        start, end = max(0., start), min(duration, end)
        if end > start:
            supplied_id = value.get('id') if isinstance(value, Mapping) else None
            identity = supplied_id if supplied_id is not None else f'{name}-{index}'
            if not isinstance(identity, str) or not identity:
                raise ValueError(f'{name} id must be a nonempty string')
            output.append({'index': index, 'id': identity, 'interval': Interval(start, end)})
    if len({x['id'] for x in output}) != len(output):
        raise ValueError(f'{name} IDs must be unique')
    return sorted(output, key=lambda x: (x['interval'].start, x['interval'].end, x['index']))


def _overlap(a: Interval, b: Interval) -> float:
    return max(0., min(a.end, b.end) - max(a.start, b.start))


def _material(gold: Interval) -> float:
    return min(.5, .1 * (gold.end - gold.start))


def _prepared(record: Mapping[str, Any]):
    duration = _number(record.get('durationSeconds'), 'durationSeconds')
    if duration <= 0:
        raise ValueError('durationSeconds must be positive')
    gold = _intervals(record['rallies'], duration, 'gold')
    if any(b['interval'].start < a['interval'].end for a, b in zip(gold, gold[1:])):
        raise ValueError('gold identities must not overlap')
    parents = _intervals(record['productionEvents'], duration, 'production')
    ignored = [x['interval'] for x in _intervals(record.get('ignoredIntervals', ()), duration, 'ignored')]
    censored = [x for x in gold if any(_overlap(x['interval'], mask) > 0 for mask in ignored)]
    censored_indexes = {x['index'] for x in censored}
    eligible = [x for x in gold if x['index'] not in censored_indexes]
    excluded = [*ignored, *(x['interval'] for x in censored)]
    for parent in parents:
        parent['support'] = subtract_intervals([parent['interval']], excluded)
    return duration, gold, eligible, parents, excluded, censored


def _targets(eligible, parents, edge_guard_seconds):
    output = []
    for parent in parents:
        # Masks split the target universe. A preceding rally across an ignored
        # span is not evidence of an additional observable rally in this span.
        for component_index, component in enumerate(parent['support']):
            covered = [g for g in eligible if _overlap(g['interval'], component) >= _material(g['interval'])]
            for gold in covered[1:]:
                start = gold['interval'].start
                if not component.start < start < component.end:
                    continue
                edge_distance = min(start - component.start, component.end - start)
                output.append({'id': f"{parent['id']}::gold-{gold['index']}",
                               'parentId': parent['id'], 'parentIndex': parent['index'],
                               'truthIndex': gold['index'], 'time': start, 'start': start,
                               'componentIndex': component_index,
                               'componentStart': component.start, 'componentEnd': component.end,
                               'edgeDistanceSeconds': edge_distance,
                               'accessible': edge_distance > edge_guard_seconds})
    return output


def split_targets(record: Mapping[str, Any], *, edge_guard_seconds: float = DEFAULT_EDGE_GUARD_SECONDS):
    """Return all genuine additional starts; near-edge targets stay in recall.

Within each unmasked component of each original production identity, count a
later gold start when at least one earlier gold rally materially overlaps that
component. Material means >=min(0.5 seconds,10% of that gold rally's duration).
Parent-specific targets remain distinct if production identities overlap.
"""
    edge_guard_seconds = _number(edge_guard_seconds, 'edge_guard_seconds')
    if edge_guard_seconds < 0:
        raise ValueError('edge_guard_seconds must be nonnegative')
    _, _, eligible, parents, _, _ = _prepared(record)
    return _targets(eligible, parents, edge_guard_seconds)


def _rates(matched: int, predicted: int, true: int):
    return {'true': true, 'predicted': predicted, 'matched': matched,
            'falsePositive': predicted - matched, 'falseNegative': true - matched,
            'precision': matched / predicted if predicted else (0. if true else None),
            'recall': matched / true if true else None,
            'f1': 2 * matched / (predicted + true) if true else None}


def _match(targets, proposals, tolerance):
    distances = np.asarray([[abs(t['time'] - p['time']) for p in proposals] for t in targets], dtype=float).reshape(len(targets), len(proposals))
    allowed = np.asarray([[t['parentId'] == p['parentId'] and t['componentIndex'] == p['componentIndex']
                           for p in proposals] for t in targets], dtype=bool).reshape(len(targets), len(proposals))
    return _maximum_weight_pairs(np.maximum(0., 1. - distances / tolerance), allowed & (distances <= tolerance))


def _diagnostics(eligible, events):
    overlaps = [[sum(_overlap(g['interval'], piece) for piece in event['support']) for event in events]
                for g in eligible]
    missed = [g['index'] for g, row in zip(eligible, overlaps) if not any(x > 0 for x in row)]
    material = [[value >= _material(g['interval']) for value in row] for g, row in zip(eligible, overlaps)]
    return {'completeMissIndexes': missed,
            'mergedPredictions': sum(sum(row[j] > 0 for row in overlaps) > 1 for j in range(len(events))),
            'mergedPredictionsMaterial': sum(sum(row[j] for row in material) > 1 for j in range(len(events))),
            'splitTrueRallies': sum(sum(x > 0 for x in row) > 1 for row in overlaps),
            'splitTrueRalliesMaterial': sum(sum(row) > 1 for row in material)}


def _score_record(record, edge_guard_seconds):
    for name in ('id', 'sourceGroup'):
        if not isinstance(record.get(name), str) or not record[name]:
            raise ValueError(f'{name} must be nonempty')
    duration, gold, eligible, parents, excluded, censored = _prepared(record)
    targets = _targets(eligible, parents, edge_guard_seconds)
    parent_map = {x['id']: x for x in parents}
    original_proposals = list(record.get('splitProposals', ()))
    proposals = []
    masked = 0
    for index, proposal in enumerate(original_proposals):
        if not isinstance(proposal, Mapping) or proposal.get('parentId') not in parent_map:
            raise ValueError('split proposal must name an existing parentId')
        parent = parent_map[proposal['parentId']]
        time = _number(proposal.get('time'), 'proposal time')
        if not parent['interval'].start < time < parent['interval'].end:
            raise ValueError('split proposal must be strictly internal to its parent')
        # Starts are observed from their right, as in identity boundary scoring.
        if any(span.start <= time < span.end for span in excluded):
            masked += 1
            continue
        component = next((i for i, span in enumerate(parent['support']) if span.start <= time < span.end), None)
        if component is None:
            raise ValueError('unmasked split proposal has no parent support')
        proposals.append({'index': index, 'id': proposal.get('id', f'proposal-{index}'),
                          'parentId': parent['id'], 'time': time, 'componentIndex': component})
    result_events = _intervals(record.get('predictions', record['productionEvents']), duration, 'prediction')
    for event in result_events:
        event['support'] = subtract_intervals([event['interval']], excluded)
    baseline = _diagnostics(eligible, parents)
    result = _diagnostics(eligible, result_events)
    baseline_misses, result_misses = set(baseline['completeMissIndexes']), set(result['completeMissIndexes'])
    baseline_union = subtract_intervals((span for p in parents for span in p['support']), ())
    result_union = subtract_intervals((span for p in result_events for span in p['support']), ())
    lost = subtract_intervals(baseline_union, result_union)
    added = subtract_intervals(result_union, baseline_union)
    row = {'id': record['id'], 'sourceGroup': record['sourceGroup'],
           'originalTrueRallies': len(gold), 'ignoredTouchedTrueRallies': len(censored),
           'eligibleTrueRallies': len(eligible), 'productionParents': len(parents),
           'splitTargets': len(targets), 'uniqueTargetTrueRallies': len({x['truthIndex'] for x in targets}),
           'accessibleSplitTargets': sum(x['accessible'] for x in targets),
           'edgeInaccessibleSplitTargets': sum(not x['accessible'] for x in targets),
           'originalSplitProposals': len(original_proposals), 'maskedSplitProposals': masked,
           'splitProposals': len(proposals),
           'baselineCompleteMisses': len(baseline_misses), 'resultCompleteMisses': len(result_misses),
           'retainedCompleteMisses': len(baseline_misses & result_misses),
           'additionalCompleteMisses': len(result_misses - baseline_misses),
           'recoveredCompleteMisses': len(baseline_misses - result_misses),
           'rawCoreSecondsLostFromBaseline': sum(_overlap(g['interval'], s) for g in eligible for s in lost),
           'rawCoreSecondsAddedToBaseline': sum(_overlap(g['interval'], s) for g in eligible for s in added),
           'rawSelectedSecondsLostFromBaseline': sum(s.end - s.start for s in lost),
           'rawSelectedSecondsAddedToBaseline': sum(s.end - s.start for s in added),
           'baselineDiagnostics': baseline, 'resultDiagnostics': result,
           'additionalCompleteMissIndexes': sorted(result_misses - baseline_misses),
           'recoveredCompleteMissIndexes': sorted(baseline_misses - result_misses),
           'targets': targets, 'proposals': proposals, 'splitLocalization': {}}
    for prefix, diagnostics in (('baseline', baseline), ('result', result)):
        for name in ('mergedPredictions', 'mergedPredictionsMaterial', 'splitTrueRallies', 'splitTrueRalliesMaterial'):
            row[prefix + name[0].upper() + name[1:]] = diagnostics[name]
    for tolerance in SPLIT_TOLERANCES_SECONDS:
        pairs = _match(targets, proposals, tolerance)
        values = _rates(len(pairs), len(proposals), len(targets))
        values['matchedAccessibleTargets'] = sum(targets[i]['accessible'] for i, _ in pairs)
        values['accessibleTargetRecall'] = values['matchedAccessibleTargets'] / row['accessibleSplitTargets'] if row['accessibleSplitTargets'] else None
        values['pairs'] = [{'targetId': targets[i]['id'], 'proposalIndex': proposals[j]['index'],
                            'errorSeconds': proposals[j]['time'] - targets[i]['time']} for i, j in pairs]
        row['splitLocalization'][format(tolerance, 'g')] = values
    return row


def match_proposals(record: Mapping[str, Any], proposals: Iterable[Mapping[str, Any]], *, tolerance: float = PRIMARY_SPLIT_TOLERANCE_SECONDS):
    """Gold oracle correspondence for hypothetical review, never inference.

Return {'pairs': [(original_proposal_index, target_index), ...], 'targets': [...]}.
Near-edge targets remain eligible. Masked proposals preserve the original index
of surviving proposals; duplicate proposals remain separate match candidates.
"""
    tolerance = _number(tolerance, 'tolerance')
    if tolerance <= 0:
        raise ValueError('tolerance must be positive')
    scored = _score_record({**record, 'splitProposals': list(proposals)}, DEFAULT_EDGE_GUARD_SECONDS)
    pairs = _match(scored['targets'], scored['proposals'], tolerance)
    return {'pairs': [(scored['proposals'][j]['index'], i) for i, j in pairs], 'targets': scored['targets']}


_COUNTS = ('originalTrueRallies', 'ignoredTouchedTrueRallies', 'eligibleTrueRallies', 'productionParents',
           'splitTargets', 'uniqueTargetTrueRallies', 'accessibleSplitTargets', 'edgeInaccessibleSplitTargets',
           'originalSplitProposals', 'maskedSplitProposals', 'splitProposals', 'baselineCompleteMisses',
           'resultCompleteMisses', 'retainedCompleteMisses', 'additionalCompleteMisses', 'recoveredCompleteMisses',
           'rawCoreSecondsLostFromBaseline', 'rawCoreSecondsAddedToBaseline',
           'rawSelectedSecondsLostFromBaseline', 'rawSelectedSecondsAddedToBaseline',
           *(prefix + name for prefix in ('baseline', 'result') for name in
             ('MergedPredictions', 'MergedPredictionsMaterial', 'SplitTrueRallies', 'SplitTrueRalliesMaterial')))


def _pool(rows):
    output = {name: sum(x[name] for x in rows) for name in _COUNTS}
    output['splitLocalization'] = {}
    for tolerance in SPLIT_TOLERANCES_SECONDS:
        key = format(tolerance, 'g')
        counts = {name: sum(x['splitLocalization'][key][name] for x in rows) for name in ('matched', 'predicted', 'true')}
        output['splitLocalization'][key] = _rates(**counts)
        matched_accessible = sum(x['splitLocalization'][key]['matchedAccessibleTargets'] for x in rows)
        output['splitLocalization'][key]['matchedAccessibleTargets'] = matched_accessible
        output['splitLocalization'][key]['accessibleTargetRecall'] = matched_accessible / output['accessibleSplitTargets'] if output['accessibleSplitTargets'] else None
    return output


def evaluate_split_proposals(records: Iterable[Mapping[str, Any]], *, edge_guard_seconds: float = DEFAULT_EDGE_GUARD_SECONDS):
    """Pool per-parent split timestamp metrics and independent coverage guards.

Rows require id, sourceGroup, durationSeconds, rallies and productionEvents;
splitProposals contains {parentId,time}, and predictions optionally holds the
resulting rally identities. No proposals means zero split recall. If no split
targets exist, recall/F1 are null and false-positive counts remain available.
"""
    edge_guard_seconds = _number(edge_guard_seconds, 'edge_guard_seconds')
    if edge_guard_seconds < 0:
        raise ValueError('edge_guard_seconds must be nonnegative')
    rows = [_score_record(record, edge_guard_seconds) for record in records]
    if not rows or len({x['id'] for x in rows}) != len(rows):
        raise ValueError('records must be nonempty with unique recording IDs')
    return {'metricContract': {
                'version': 'fixed-production-split-advice-v1',
                'primaryToleranceSeconds': PRIMARY_SPLIT_TOLERANCE_SECONDS,
                'tolerancesSeconds': list(SPLIT_TOLERANCES_SECONDS),
                'targetRule': 'Additional material-overlap gold starts strictly internal to the same evaluable production-parent component; first material-overlap gold excluded',
                'materialRule': 'overlap >= min(0.5 seconds, 10% gold duration), computed per valid parent component',
                'ignoredRule': 'exclude ignored-touched gold identities and their whole spans; targets and matches never bridge excluded components; proposals in masks excluded',
                'matchingRule': 'one-to-one within same parent and valid component; maximum cardinality then minimum total timestamp error',
                'nearEdgeRule': 'all genuine targets stay in recall denominator; accessible iff distance to both valid component edges strictly exceeds edge guard',
                'edgeGuardSeconds': edge_guard_seconds,
                'partialGoldRule': 'partially parent-covered gold eligible if material overlap and actual start internal; outside-start gold never a target',
                'duplicateRule': 'duplicate proposals remain distinct predictions and unmatched duplicates are false positives',
                'overlappingParentsRule': 'targets parent-specific; unique gold target count also reported',
                'coverageRule': 'complete miss means no positive overlap with any unmasked prediction; lost/added core seconds compare raw interval unions',
                'aggregation': 'pool counts across recordings before computing precision, recall and F1; separate seeds',
                'labelLimit': 'rally-start timing proxy only; no serving-side, point winner or score accuracy claim',
            },
            'pooled': _pool(rows),
            'sourceGroups': {g: _pool([x for x in rows if x['sourceGroup'] == g]) for g in sorted({x['sourceGroup'] for x in rows})},
            'recordings': rows}
