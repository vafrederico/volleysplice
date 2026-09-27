"""Label-blind compact advice that never changes production export occupancy.

The three fixed policies predict internal rally-start markers. Applying markers
partitions each production parent into touching children; it does not cut footage.
New child ends at partition points are bookkeeping, not observed dead-ball ends.
There is no label access, fitting, threshold selection, or filesystem access here.
"""
from __future__ import annotations

from math import isfinite

import numpy as np

from . import neural_human_review as legacy
from . import neural_production_combinations as iv

POLICIES = ('event_starts', 'head_evidence', 'corroborated')
EDGE_MARGIN_SECONDS = 2.
MIN_NEURAL_OVERLAP_SECONDS = .5
SERVE_THRESHOLD = .35
END_THRESHOLD = .25
LIVE_VALLEY_THRESHOLD = .35
PRECEDING_SECONDS = 3.
NMS_SECONDS = 1.
CORROBORATION_SECONDS = 1.
CLEANUP_SUPPORT_PADDING_SECONDS = 2.
CLEANUP_SUPPORT_FRACTION = .5


def event_rows(values, prefix='p'):
    """Normalize intervals without discarding source IDs or event metadata."""
    output = []
    for index, value in enumerate(values):
        row = dict(value) if isinstance(value, dict) else {}
        interval = iv.intervals([value])[0]
        row.update(start=interval.start, end=interval.end)
        row['id'] = str(row.get('id', f'{prefix}{index:05d}'))
        output.append(row)
    output.sort(key=lambda row: (row['start'], row['end'], row['id']))
    if len({row['id'] for row in output}) != len(output):
        raise ValueError('Event IDs must be unique')
    if any(left['end'] > right['start'] for left, right in zip(output, output[1:])):
        raise ValueError('Event interiors must not overlap')
    return output


def _inputs(record, base, neural, times, scores):
    duration = float(record['durationSeconds'])
    t, a = np.asarray(times, dtype=float), np.asarray(scores, dtype=float)
    if not isfinite(duration) or duration <= 0:
        raise ValueError('Invalid duration')
    if (t.ndim != 1 or a.shape != (len(t), 4) or not np.isfinite(t).all()
            or not np.isfinite(a).all() or not np.all(np.diff(t) > 0)
            or np.any(a < 0) or np.any(a > 1)
            or np.any(t < 0) or np.any(t > duration)):
        raise ValueError('Invalid aligned four-head probabilities/times')
    parents, children = event_rows(base), event_rows(neural, 'n')
    if any(row['start'] < 0 or row['end'] > duration for row in parents + children):
        raise ValueError('Events must remain within recording bounds')
    return parents, children, t, a, legacy.valid_time(record)


def _component(components, time):
    # An ignored interval's start is excluded and its end is valid.
    return next((x for x in components if x.start <= time < x.end), None)


def _inside(parent, time, component):
    return (component is not None
            and max(parent['start'], component.start) + EDGE_MARGIN_SECONDS < time
            < min(parent['end'], component.end) - EDGE_MARGIN_SECONDS)


def _nms(rows):
    selected = []
    for row in sorted(rows, key=lambda x: (-x['strength'], x['time'], str(x['evidence']))):
        if not any(row['component'] == old['component']
                   and abs(row['time'] - old['time']) < NMS_SECONDS for old in selected):
            selected.append(row)
    return sorted(selected, key=lambda x: x['time'])


def _event_candidates(parent, neural, t, a, components):
    output = []
    for index, event in enumerate(neural):
        time = event['start']
        component = _component(components, time)
        if not _inside(parent, time, component):
            continue
        start, end = max(parent['start'], component.start), min(parent['end'], component.end)
        overlap = lambda row: max(0., min(row['end'], end) - max(row['start'], start))
        if overlap(event) < MIN_NEURAL_OVERLAP_SECONDS:
            continue
        previous = next((row for row in reversed(neural[:index])
                         if overlap(row) >= MIN_NEURAL_OVERLAP_SECONDS), None)
        if previous is None:
            continue
        left, right = max(time, start), min(event['end'], end)
        values = a[(t >= left) & (t < right), 0]
        strength = float(np.mean(values)) if len(values) else 0.
        output.append({'time': time, 'strength': strength, 'component': (component.start, component.end),
                       'evidence': {'neuralEventId': event['id'], 'previousNeuralEventId': previous['id'],
                                    'neuralStart': time, 'previousNeuralEnd': previous['end'],
                                    'meanLive': strength}})
    return _nms(output)


def _local_maxima(t, a, components):
    """Earliest tick of an exact-score plateau strictly above outside neighbors."""
    result = []
    for component in components:
        ids = np.flatnonzero((t >= component.start) & (t < component.end))
        position = 0
        while position < len(ids):
            stop = position + 1
            value = float(a[ids[position], 1])
            while stop < len(ids) and float(a[ids[stop], 1]) == value:
                stop += 1
            before = float(a[ids[position - 1], 1]) if position else -float('inf')
            after = float(a[ids[stop], 1]) if stop < len(ids) else -float('inf')
            if value >= SERVE_THRESHOLD and value > before and value > after:
                result.append(int(ids[position]))
            position = stop
    return result


def _head_candidates(parent, t, a, components, maxima):
    result = []
    for index in maxima:
        time = float(t[index])
        component = _component(components, time)
        if not _inside(parent, time, component):
            continue
        start = max(parent['start'], component.start, time - PRECEDING_SECONDS)
        previous = a[(t >= start) & (t < time)]
        if not len(previous):
            continue
        maximum_end, minimum_live = float(np.max(previous[:, 2])), float(np.min(previous[:, 0]))
        if maximum_end >= END_THRESHOLD or minimum_live < LIVE_VALLEY_THRESHOLD:
            strength = float(a[index, 1])
            result.append({'time': time, 'strength': strength, 'component': (component.start, component.end),
                           'evidence': {'serveTime': time, 'serveScore': strength,
                                        'precedingMaxEnd': maximum_end, 'precedingMinLive': minimum_live}})
    return _nms(result)


def _corroborated(events, heads):
    pairs = [(abs(e['time'] - h['time']), -h['strength'], e['time'], h['time'], i, j)
             for i, e in enumerate(events) for j, h in enumerate(heads)
             if e['component'] == h['component'] and abs(e['time'] - h['time']) <= CORROBORATION_SECONDS]
    used_events, used_heads, output = set(), set(), []
    for _, _, _, _, i, j in sorted(pairs):
        if i in used_events or j in used_heads:
            continue
        used_events.add(i)
        used_heads.add(j)
        e, h = events[i], heads[j]
        output.append({'time': h['time'], 'strength': min(e['strength'], h['strength']),
                       'component': h['component'],
                       'evidence': {**e['evidence'], **h['evidence'],
                                    'corroborationDistanceSeconds': abs(e['time'] - h['time'])}})
    return _nms(output)


def split_proposals(record, base, neural, times, scores, policy):
    """Return fixed model-evidenced splits; labels in ``record`` are never read.

    ``start/end`` are local display bands (not edit permission). A review queue
    should use ``parentStart/parentEnd`` to show the entire production parent.
    Priority is heuristic mean-live/head evidence, not calibrated confidence.
    """
    if policy not in POLICIES:
        raise ValueError('Unknown split policy')
    parents, children, t, a, components = _inputs(record, base, neural, times, scores)
    maxima = _local_maxima(t, a, components) if policy != 'event_starts' else []
    output = []
    for parent in parents:
        events = _event_candidates(parent, children, t, a, components) if policy != 'head_evidence' else []
        heads = _head_candidates(parent, t, a, components, maxima) if policy != 'event_starts' else []
        rows = events if policy == 'event_starts' else heads if policy == 'head_evidence' else _corroborated(events, heads)
        for row in rows:
            time = row['time']
            component = _component(components, time)
            output.append({'id': f's{len(output):05d}', 'parentId': parent['id'],
                           'parentStart': parent['start'], 'parentEnd': parent['end'],
                           'time': time, 'start': max(parent['start'], component.start, time - 1.),
                           'end': min(parent['end'], component.end, time + 1.),
                           'priority': row['strength'], 'confidencePriority': row['strength'],
                           'kind': 'split', 'policy': policy, 'evidence': row['evidence']})
    return output


def cleanup_proposals(record, base, neural):
    """Flag whole parents having <50% valid raw support by neural events +/-2s.

    Support is intersected with each valid component before dilation, then clipped
    to that same component, without a positive-gap join. Ignored detections cannot
    support adjacent time. Fully ignored parents are ineligible. Flags alone never
    change any interval.
    """
    parents, children = event_rows(base), event_rows(neural, 'n')
    valid = legacy.valid_time(record)
    support = iv.union(*(iv.intersection(iv.dilate(iv.intersection(children, [component]),
                                                   CLEANUP_SUPPORT_PADDING_SECONDS,
                                                   record['durationSeconds']), [component])
                         for component in valid))
    output = []
    for parent in parents:
        occupied = iv.intersection([parent], valid)
        total = iv.duration(occupied)
        if total <= 0:
            continue
        supported = iv.duration(iv.intersection(occupied, support))
        fraction = supported / total
        if fraction < CLEANUP_SUPPORT_FRACTION:
            output.append({'id': f'c{len(output):05d}', 'parentId': parent['id'],
                           'parentStart': parent['start'], 'parentEnd': parent['end'],
                           'start': parent['start'], 'end': parent['end'], 'kind': 'cleanup',
                           'supportFraction': fraction, 'supportedSeconds': supported,
                           'parentValidSeconds': total, 'priority': 1. - fraction,
                           'confidencePriority': 1. - fraction})
    return output


def apply_splits(base, proposals):
    """Partition parents at proposed times, preserving exact occupancy/lineage.

    This function also accepts human-snapped proposals with their original IDs.
    It cannot remove parents or alter endpoints. ``startObserved`` marks a model
    or reviewed start proposal; partition-only ends are explicitly unobserved.
    """
    parents = event_rows(base)
    by_parent = {row['id']: {} for row in parents}
    parent_lookup = {row['id']: row for row in parents}
    seen_ids = set()
    for proposal in proposals:
        parent_id, time, proposal_id = str(proposal['parentId']), float(proposal['time']), str(proposal['id'])
        if parent_id not in by_parent:
            raise ValueError('Split proposal has unknown parent')
        parent = parent_lookup[parent_id]
        if not isfinite(time) or not parent['start'] < time < parent['end']:
            raise ValueError('Split must remain strictly inside its parent')
        if proposal_id in seen_ids:
            raise ValueError('Proposal IDs must be unique')
        seen_ids.add(proposal_id)
        by_parent[parent_id].setdefault(time, []).append(proposal_id)
    result = []
    for parent in parents:
        cuts = by_parent[parent['id']]
        times = [parent['start'], *sorted(cuts), parent['end']]
        for index, (start, end) in enumerate(zip(times, times[1:])):
            row = dict(parent)
            row.update(id=parent['id'] if not cuts else f"{parent['id']}::child:{index + 1}",
                       parentId=parent['id'], parentStart=parent['start'], parentEnd=parent['end'],
                       start=start, end=end, childIndex=index,
                       startObserved=bool(parent.get('startObserved', True)) if index == 0 else True,
                       endObserved=bool(parent.get('endObserved', True)) if index == len(times) - 2 else False,
                       startKind='inherited' if index == 0 else 'proposed-rally-start',
                       endKind='inherited' if index == len(times) - 2 else 'partition-only',
                       proposalIds=sorted(cuts.get(start, [])),
                       endProposalIds=sorted(cuts.get(end, [])))
            result.append(row)
    return result
