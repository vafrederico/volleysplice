#!/usr/bin/env python3
"""Independent reconstruction of label-blind rally review candidates.

Uses standard-library interval sweeps and scalar score access. Does not import
the candidate builder, its interval helpers, labels, queue builder or editor.
"""
from __future__ import annotations

from collections import Counter
from math import floor, fsum, isfinite


EPS = 1e-9


def require(condition, text):
    if not condition:
        raise ValueError(text)


def span(row):
    if isinstance(row, dict):
        return float(row['start']), float(row['end'])
    if hasattr(row, 'start'):
        return float(row.start), float(row.end)
    return float(row[0]), float(row[1])


def merged(values, gap=0.):
    output = []
    for start, end in sorted(span(x) for x in values):
        require(isfinite(start) and isfinite(end) and end > start, 'Invalid interval')
        if output and (start <= output[-1][1] or start-output[-1][1] < gap):
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output


def subtract(values, masks):
    output = []
    for start, end in merged(values):
        cursor = start
        for left, right in merged(masks):
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                output.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            output.append((cursor, end))
    return output


def intersect(left, right):
    result = []
    for a, b in merged(left):
        for c, d in merged(right):
            if min(b, d) > max(a, c):
                result.append((max(a, c), min(b, d)))
    return merged(result)


def seconds(values):
    return fsum(b-a for a, b in merged(values))


def expanded(values, padding, duration, join=0.):
    return merged([(max(0., a-padding), min(duration, b+padding))
                   for a, b in map(span, values) if min(duration, b+padding) > max(0., a-padding)], join)


def original_events(values):
    output = sorted(map(span, values))
    require(all(isfinite(a) and isfinite(b) and a < b for a, b in output), 'Invalid event')
    require(all(a[1] <= b[0] for a, b in zip(output, output[1:])), 'Overlapping source event interiors')
    return output


def reconstruct(record, base, neural, times, scores, inventory, mode):
    require(inventory in ('legacy', 'local_events', 'local_heads') and mode in ('production', 'individual'),
            'Unknown proposal configuration')
    # Deliberately read only these two recording fields. Gold is unavailable.
    duration = float(record['durationSeconds'])
    ignored = merged(record.get('ignoredIntervals', []))
    universe = subtract([(0., duration)], ignored)
    p, n = original_events(base), original_events(neural)
    t = [float(v) for v in times]
    a = [tuple(float(v) for v in row) for row in scores]
    require(len(a) == len(t) and all(len(row) == 4 for row in a), 'Four-head alignment differs')
    require(all(isfinite(v) and 0 <= v <= 1 for row in a for v in row), 'Invalid four-head scores')
    require(all(isfinite(v) for v in t) and all(x < y for x, y in zip(t, t[1:])), 'Invalid exact timestamps')
    allowed = [not any(left <= value < right for left, right in ignored) for value in t]
    pieces = {}

    def visible(values):
        return intersect(values, universe)

    def add(start, end, reason):
        if end-start < 1.:
            midpoint = (start+end)/2
            start, end = midpoint-.5, midpoint+.5
        start, end = max(0., start), min(duration, end)
        if end <= start:
            return
        for value in visible([(start, end)]):
            pieces.setdefault(value, set()).add(reason)

    def nearby_peak(center, head):
        local = [(left, right) for left, right in universe if left <= center <= right]
        ids = [i for i, value in enumerate(t) if center-3 <= value <= center+3 and allowed[i]
               and any(left <= value < right for left, right in local)]
        if not ids:
            return center, 0.
        chosen = sorted(ids, key=lambda i: (-a[i][head], abs(t[i]-center), t[i]))[0]
        return t[chosen], a[chosen][head]

    def masked_indices(start, end):
        return [i for i, value in enumerate(t) if start <= value < end and allowed[i]]

    if inventory == 'legacy':
        positives = visible(p)
        if mode == 'production':
            other = visible(n)
            triggers = []
            for left, right in ((positives, other), (other, positives)):
                support = expanded(right, 2., duration)
                triggers.extend(item for item in left if subtract([item], support))
            for candidate in merged([*positives, *other]):
                if intersect([candidate], triggers):
                    add(*candidate, 'legacy_positive' if intersect([candidate], positives) else 'legacy_negative')
        else:
            for candidate in positives:
                values = [a[i][0] for i in masked_indices(*candidate)]
                if not values or fsum(values)/len(values) < .8:
                    add(*candidate, 'legacy_positive')
            for left, right in subtract(universe, positives):
                while left < right:
                    stop = min(right, (floor(left/5)+1)*5.)
                    require(stop > left, 'Negative grid did not advance')
                    values = [a[i][0] for i in masked_indices(left, stop)]
                    if not values or max(values) > .2:
                        add(left, stop, 'legacy_negative')
                    left = stop
    else:
        if mode == 'production':
            for candidate in subtract(p, n):
                add(*candidate, 'production_only')
            for candidate in subtract(n, p):
                add(*candidate, 'neural_only')
            # Independent union-find on the bipartite event graph. Same-source
            # touching events stay separate; only cross-source positive overlap
            # creates an edge.
            parent = list(range(len(p)+len(n)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            active = set()
            for i, (left, right) in enumerate(p):
                for j, (start, end) in enumerate(n):
                    if min(right, end) > max(left, start):
                        parent[find(len(p)+j)] = find(i)
                        active.update((i, len(p)+j))
            groups = {}
            for index in sorted(active):
                groups.setdefault(find(index), []).append(index)
            for members in groups.values():
                parts = ([p[i] for i in members if i < len(p)], [n[i-len(p)] for i in members if i >= len(p)])
                if any(len(part) > 1 for part in parts):
                    for part in parts:
                        for left, right in zip(part, part[1:]):
                            add(left[1], right[0], 'split_merge')
        else:
            for candidate in p:
                values = [a[i][0] for i in masked_indices(*candidate)]
                if not values or fsum(values)/len(values) < .8:
                    add(*candidate, 'uncertain_positive')

        components = []
        if len(t) > 1:
            edges = [max(0., t[0]-(t[1]-t[0])/2),
                     *[(x+y)/2 for x, y in zip(t, t[1:])],
                     min(duration, t[-1]+(t[-1]-t[-2])/2)]
            low_ids = [i for i in range(len(t)) if allowed[i] and a[i][0] >= .15]
            runs = []
            for i in low_ids:
                if not runs or i != runs[-1][-1]+1:
                    runs.append([])
                runs[-1].append(i)
            for run in runs:
                for candidate in visible([(edges[run[0]], edges[run[-1]+1])]):
                    piece_values = [a[i][0] for i in masked_indices(*candidate)]
                    if candidate[1]-candidate[0] >= .5 and piece_values and max(piece_values) >= .30:
                        components.append(candidate)
        for start, end in components:
            if inventory == 'local_heads':
                ps, vs = nearby_peak(start, 1)
                pe, ve = nearby_peak(end, 2)
                if vs >= .25 and ps < end-.5:
                    start = ps
                if ve >= .25 and pe > start+.5:
                    end = pe
            missing = seconds(subtract([(start, end)], p))
            if missing >= .5 and missing >= .25*(end-start):
                add(start, end, 'missed_live')
        if inventory == 'local_heads':
            for left, right in [*p, *n]:
                for center, head, reason in ((left, 1, 'boundary_start'), (right, 2, 'boundary_end')):
                    point, value = nearby_peak(center, head)
                    if value >= .25 and abs(point-center) >= .25:
                        add(min(point, center), max(point, center), reason)
                ids = [i for i in range(len(t)) if left+2 < t[i] < right-2 and a[i][1] >= .35 and allowed[i]]
                chosen = []
                for i in sorted(ids, key=lambda j: (-a[j][1], t[j])):
                    if any(abs(t[i]-point) < 1 for point in chosen):
                        continue
                    local = [(left, right) for left, right in universe if left <= t[i] <= right]
                    preceding = [j for j in range(len(t)) if t[i]-3 <= t[j] < t[i] and allowed[j]
                                 and any(left <= t[j] < right for left, right in local)]
                    if preceding and (max(a[j][2] for j in preceding) >= .25 or min(a[j][0] for j in preceding) < .35):
                        chosen.append(t[i])
                        add(t[i]-1, t[i]+1, 'internal_start')

    output = []
    for index, ((start, end), reasons) in enumerate(sorted(pieces.items())):
        rows = [a[i] for i in masked_indices(start, end)]
        mean_live = fsum(row[0] for row in rows)/len(rows) if rows else .5
        max_live = max(row[0] for row in rows) if rows else .5
        boundary = max(max(row[1:3]) for row in rows) if rows else .5
        evidence = []
        for reason in reasons:
            if reason in ('production_only', 'uncertain_positive'):
                evidence.append(1-mean_live)
            elif reason in ('neural_only', 'missed_live', 'legacy_negative'):
                evidence.append(max_live)
            elif reason in ('split_merge', 'internal_start', 'boundary_start', 'boundary_end'):
                evidence.append(max(.5, boundary))
            else:
                evidence.append(max(1-mean_live, boundary))
        strength = max(evidence)
        windows = visible(expanded([(start, end)], 2., duration))
        view = subtract(expanded(windows, 2., duration, 3.), ignored)
        cost = seconds(view)
        priority = ((end-start)*strength + 4.*strength*any(r != 'production_only' for r in reasons))/max(cost, 1e-12)
        output.append({'id': f'q{index:05d}', 'start': start, 'end': end, 'reasons': sorted(reasons),
                       'meanLive': mean_live, 'maxLive': max_live, 'maxBoundary': boundary,
                       'priority': priority, 'standalonePlaybackSeconds': cost})
    return output


def audit_candidates(record, base, neural, times, scores, inventory, mode, candidates):
    expected = reconstruct(record, base, neural, times, scores, inventory, mode)
    require(len(expected) == len(candidates), 'Candidate inventory count differs')
    maximum_error = 0.
    for actual, wanted in zip(candidates, expected):
        require(set(actual) == set(wanted), 'Candidate payload fields differ')
        for key, value in wanted.items():
            if isinstance(value, float):
                error = abs(float(actual[key])-value)
                maximum_error = max(maximum_error, error)
                require(isfinite(float(actual[key])) and error <= EPS, f'Candidate {key} differs')
            else:
                require(actual[key] == value, f'Candidate {key} differs')
    return {'passed': True, 'candidateCount': len(expected),
            'reasonCounts': dict(sorted(Counter(reason for row in expected for reason in row['reasons']).items())),
            'sourceEventCounts': {'base': len(base), 'neural': len(neural)},
            'maximumNumericDifference': maximum_error,
            'coverage': ['independent full proposal reconstruction', 'original event partition and positive-overlap graph',
                         'actual midpoint time cells', 'ignored tick masking and local evidence barriers', 'four-head peak ties and NMS',
                         'candidate reasons and deterministic IDs', 'all priority features and standalone playback cost'],
            'goldFieldsRead': False}
