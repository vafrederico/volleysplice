"""Typed compact start advice and separate event ends; exports are external.

All candidate construction is label-blind. Intervals remain inside original
production parents but may omit raw production time. The caller must report that
loss separately from the immutable production export, and never claim invariance
of the new event timeline's core coverage.
"""
from __future__ import annotations

from math import isfinite
import numpy as np

from . import neural_human_review as legacy
from . import neural_production_combinations as iv
from .neural_split_advisor import event_rows

POLICIES = ('first_start', 'typed_starts', 'separate_ends', 'head_refined')
MIN_OVERLAP_SECONDS = .5
MIN_EVENT_SECONDS = .5
HEAD_RADIUS_SECONDS = 3.
HEAD_THRESHOLD = .25
ACTIONABLE_SHIFT_SECONDS = .25


def _inputs(record, base, neural, times, scores):
    duration = float(record['durationSeconds'])
    t, a = np.asarray(times, dtype=float), np.asarray(scores, dtype=float)
    if (not isfinite(duration) or duration <= 0 or t.ndim != 1 or a.shape != (len(t), 4)
            or not np.isfinite(t).all() or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1)
            or not np.all(np.diff(t) > 0) or np.any(t < 0) or np.any(t > duration)):
        raise ValueError('Invalid aligned four-head probabilities or timeline')
    parents, neural = event_rows(base), event_rows(neural, 'n')
    if any(row['start'] < 0 or row['end'] > duration for row in parents + neural):
        raise ValueError('Events must lie within the recording')
    return parents, neural, t, a, legacy.valid_time(record)


def _ignored_edge(record, time):
    return any(time in (span.start, span.end) for span in iv.intervals(record.get('ignoredIntervals', ())))


def _native_boundary(record, parent, original, clipped, kind):
    if _ignored_edge(record, clipped):
        return False, 'ignored-clipped'
    if clipped == original:
        return True, 'compact-' + kind
    return False, 'parent-clipped'


def _production_boundary(record, parent, time, kind):
    observed = (time == parent[kind] and not _ignored_edge(record, time)
                and bool(parent.get(kind + 'Observed', True)))
    return observed, 'production-' + kind if time == parent[kind] else 'ignored-clipped'


def _peak(t, a, head, anchor, low, high, valid_component):
    allowed = ((t >= max(low, anchor - HEAD_RADIUS_SECONDS, valid_component.start))
               & (t <= min(high, anchor + HEAD_RADIUS_SECONDS))
               & (t < valid_component.end) & (a[:, head] >= HEAD_THRESHOLD))
    ids = np.flatnonzero(allowed)
    if not len(ids):
        return None
    index = min(ids, key=lambda i: (-float(a[i, head]), abs(float(t[i]) - anchor), float(t[i])))
    return float(t[index]), float(a[index, head])


def _candidate(record, parent, component_id, component, event, neural_index, index, t, a):
    start, end = max(component.start, event['start']), min(component.end, event['end'])
    values = a[(t >= start) & (t < end), 0]
    priority = float(np.mean(values)) if len(values) else 0.
    start_observed, start_source = _native_boundary(record, parent, event['start'], start, 'start')
    end_observed, end_source = _native_boundary(record, parent, event['end'], end, 'end')
    identity = f"candidate:{component_id}:neural:{neural_index}"
    return {'id': identity, 'candidateId': identity, 'parentId': parent['id'],
            'parentStart': parent['start'], 'parentEnd': parent['end'],
            'componentId': component_id, 'componentStart': component.start, 'componentEnd': component.end,
            'type': 'initial_start' if index == 0 else 'additional_start',
            'start': start, 'end': end, 'startObserved': start_observed, 'endObserved': end_observed,
            'startSource': start_source, 'endSource': end_source,
            'sourceNeuralId': event['id'], 'sourceNeuralIndex': neural_index,
            'originalNeuralStart': event['start'], 'originalNeuralEnd': event['end'],
            'originalClippedStart': start, 'originalClippedEnd': end,
            'priority': priority, 'meanLive': priority, 'fallback': False,
            'startHeadScore': None, 'endHeadScore': None}


def _fallback(record, parent, component=None, component_id=None):
    row = dict(parent)
    start = parent['start'] if component is None else component.start
    end = parent['end'] if component is None else component.end
    start_observed, start_source = _production_boundary(record, parent, start, 'start')
    end_observed, end_source = _production_boundary(record, parent, end, 'end')
    row.update(id=parent['id'] if component is None else 'fallback:' + component_id,
               parentId=parent['id'], parentStart=parent['start'], parentEnd=parent['end'],
               componentId=component_id, componentStart=start, componentEnd=end,
               start=start, end=end, type='initial_start', candidateId=None, fallback=True,
               startObserved=start_observed, endObserved=end_observed,
               startSource=start_source, endSource=end_source)
    return row


def _refine(candidates, t, a, valid_component):
    refined = []
    for index, original in enumerate(candidates):
        row = dict(original)
        low = refined[-1]['end'] if refined else row['componentStart']
        start_peak = _peak(t, a, 1, row['originalClippedStart'], low,
                           row['originalClippedEnd'] - MIN_EVENT_SECONDS, valid_component)
        if start_peak is not None:
            row.update(start=start_peak[0], startHeadScore=start_peak[1],
                       startObserved=True, startSource='head-serve')
        high = candidates[index + 1]['originalClippedStart'] if index + 1 < len(candidates) else row['componentEnd']
        end_peak = _peak(t, a, 2, row['originalClippedEnd'], row['start'] + MIN_EVENT_SECONDS,
                         high, valid_component)
        if end_peak is not None:
            row.update(end=end_peak[0], endHeadScore=end_peak[1], endObserved=True, endSource='head-end')
        if row['end'] - row['start'] < MIN_EVENT_SECONDS - 1e-9:
            raise ValueError('Head refinement broke minimum event duration')
        if refined and refined[-1]['end'] > row['start']:
            raise ValueError('Head refinement produced overlapping events')
        refined.append(row)
    return refined


def _flags(candidates, policy):
    output = []
    for candidate in candidates:
        boundaries = []
        if candidate['type'] == 'additional_start' or abs(candidate['start'] - candidate['componentStart']) >= ACTIONABLE_SHIFT_SECONDS:
            boundaries.append((candidate['type'], 'start'))
        if (policy in ('separate_ends', 'head_refined') and candidate['endObserved']
                and abs(candidate['end'] - candidate['componentEnd']) >= ACTIONABLE_SHIFT_SECONDS):
            boundaries.append(('end', 'end'))
        for kind, boundary in boundaries:
            output.append({'id': candidate['id'] + ':' + kind, 'candidateId': candidate['id'],
                           'parentId': candidate['parentId'], 'componentId': candidate['componentId'],
                           'parentStart': candidate['parentStart'], 'parentEnd': candidate['parentEnd'],
                           'componentStart': candidate['componentStart'], 'componentEnd': candidate['componentEnd'],
                           'start': candidate['start'], 'end': candidate['end'], 'time': candidate[boundary],
                           'kind': kind, 'boundary': boundary, 'type': candidate['type'],
                           'observed': candidate[boundary + 'Observed'], 'source': candidate[boundary + 'Source'],
                           'priority': candidate['priority']})
    return output


def plan(record, base, neural, times, scores, policy):
    """Build typed intervals, actionable boundary flags, and explicit fallbacks.

    Each supported parent/valid component uses its first compact event as an
    initial-start correction and later events as additional rallies. This typing
    is a heuristic: it can mistake a missed first rally for an initial correction.
    Gold is never read. EventCandidates contains supported intervals only;
    ``events`` also contains retained unsupported production parents/components.
    """
    if policy not in POLICIES:
        raise ValueError('Unknown boundary adviser policy')
    parents, neural, t, a, valid = _inputs(record, base, neural, times, scores)
    events, candidates, decisions = [], [], []
    for parent in parents:
        components = []
        for valid_index, valid_component in enumerate(valid):
            pieces = iv.intersection([parent], [valid_component])
            if not pieces:
                continue
            component = pieces[0]
            component_id = f"{parent['id']}::valid:{valid_index}"
            associated = [(i, event) for i, event in enumerate(neural)
                          if min(event['end'], component.end) - max(event['start'], component.start) >= MIN_OVERLAP_SECONDS]
            components.append((component_id, component, valid_component, associated))
        supported = any(associated for _, _, _, associated in components)
        decision = {'parentId': parent['id'], 'supported': supported, 'fallback': not supported, 'components': []}
        if not supported:
            events.append(_fallback(record, parent))
        for component_id, component, valid_component, associated in components:
            decision['components'].append({'componentId': component_id, 'start': component.start, 'end': component.end,
                                           'associatedNeuralIds': [e['id'] for _, e in associated],
                                           'fallback': not bool(associated)})
            if not supported:
                continue
            if not associated:
                events.append(_fallback(record, parent, component, component_id))
                continue
            selected = associated[:1] if policy == 'first_start' else associated
            rows = [_candidate(record, parent, component_id, component, event, neural_index, i, t, a)
                    for i, (neural_index, event) in enumerate(selected)]
            if policy in ('first_start', 'typed_starts'):
                for i, row in enumerate(rows):
                    if i + 1 < len(rows):
                        row.update(end=rows[i + 1]['start'], endObserved=False, endSource='partition-only')
                    else:
                        observed, source = _production_boundary(record, parent, component.end, 'end')
                        row.update(end=component.end, endObserved=observed, endSource=source)
            elif policy == 'head_refined':
                rows = _refine(rows, t, a, valid_component)
            candidates.extend(rows)
            for row in rows:
                events.append({**parent, **row})
        decisions.append(decision)
    events.sort(key=lambda row: (row['start'], row['end'], row['id']))
    candidates.sort(key=lambda row: (row['start'], row['end'], row['id']))
    event_rows(events)
    return {'policy': policy, 'events': events, 'eventCandidates': candidates,
            'proposals': _flags(candidates, policy), 'parentDecisions': decisions}


propose = plan
