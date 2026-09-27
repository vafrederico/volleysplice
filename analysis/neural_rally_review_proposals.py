"""Label-blind review proposals and localized hypothetical event correction.

Event markers remain separate from interval unions used for playback/export.
No fitting, filesystem access, or labels enter proposal/queue construction.
"""
from __future__ import annotations

import numpy as np

from . import neural_production_combinations as iv
from . import neural_human_review as legacy

INVENTORIES = ('legacy', 'local_events', 'local_heads')
RANKERS = ('chronological', 'evidence')
BUDGETS = (.05, .10, .20, .40)
EDIT_MARGIN = 2.
CONTEXT = 2.


def event_rows(values):
    rows = [{'start': x.start, 'end': x.end} for x in iv.intervals(values)]
    rows.sort(key=lambda x: (x['start'], x['end']))
    if any(a['end'] > b['start'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Event editor requires nonoverlapping event interiors')
    return rows


def valid(record, spans):
    return iv.intersection(spans, legacy.valid_time(record))


def permission(record, proposals):
    return valid(record, iv.dilate(proposals, EDIT_MARGIN, record['durationSeconds']))


def playback(record, windows):
    return iv.export(windows, record, CONTEXT)


def contains(windows, time):
    return any(x.start <= time <= x.end for x in iv.intervals(windows))


def valid_ticks(record, times):
    mask = np.ones(len(times), dtype=bool)
    for x in iv.intervals(record.get('ignoredIntervals', [])):
        mask &= ~((times >= x.start) & (times < x.end))
    return mask


def local_ticks(record, times, center):
    mask = np.zeros(len(times), dtype=bool)
    for x in legacy.valid_time(record):
        if x.start <= center <= x.end:
            mask |= (times >= x.start) & (times < x.end)
    return mask & valid_ticks(record, times)


def live_components(record, times, scores):
    """Actual-time midpoint cells; low=.15, at least one high=.30, >=.5s."""
    t = np.asarray(times, dtype=float)
    a = np.asarray(scores, dtype=float)
    if len(t) < 2:
        return []
    edges = np.r_[max(0., t[0]-(t[1]-t[0])/2), (t[:-1]+t[1:])/2,
                  min(record['durationSeconds'], t[-1]+(t[-1]-t[-2])/2)]
    output, start = [], None
    good = valid_ticks(record, t)
    low = (a[:, 0] >= .15) & good
    for i in range(len(t)+1):
        if i < len(t) and low[i]:
            if start is None:
                start = i
        elif start is not None:
            for x in valid(record, [(edges[start], edges[i])]):
                piece = (t >= x.start) & (t < x.end) & good
                if x.end-x.start >= .5 and np.any(piece) and np.max(a[piece, 0]) >= .30:
                    output.append({'start': x.start, 'end': x.end})
            start = None
    return output


def peak(times, scores, center, head, radius=3., allowed=None):
    mask = (times >= center-radius) & (times <= center+radius)
    if allowed is not None:
        mask &= allowed
    ids = np.flatnonzero(mask)
    if not len(ids):
        return center, 0.
    # Nearest then earlier breaks equal-score ties deterministically.
    i = min(ids, key=lambda j: (-float(scores[j, head]), abs(float(times[j])-center), float(times[j])))
    return float(times[i]), float(scores[i, head])


def overlap_groups(base, neural):
    """Bipartite positive-overlap components, without coalescing source events."""
    p, n = event_rows(base), event_rows(neural)
    adjacency = {('p', i): set() for i in range(len(p))}
    adjacency.update({('n', i): set() for i in range(len(n))})
    for i, a in enumerate(p):
        for j, b in enumerate(n):
            if min(a['end'], b['end']) > max(a['start'], b['start']):
                adjacency['p', i].add(('n', j)); adjacency['n', j].add(('p', i))
    seen, result = set(), []
    for node in adjacency:
        if node in seen or not adjacency[node]:
            continue
        pending, group = [node], []
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item); group.append(item); pending.extend(sorted(adjacency[item]-seen))
        result.append(([p[i] for source, i in sorted(group) if source == 'p'],
                       [n[i] for source, i in sorted(group) if source == 'n']))
    return result


def proposals(record, base, neural, times, scores, inventory, mode):
    if inventory not in INVENTORIES or mode not in ('production', 'individual'):
        raise ValueError('Unknown proposal inventory/mode')
    p, n = event_rows(base), event_rows(neural)
    t, a = np.asarray(times, dtype=float), np.asarray(scores, dtype=float)
    if a.shape != (len(t), 4) or not np.isfinite(a).all() or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid aligned four-head scores')
    allowed = valid_ticks(record, t)
    items = {}

    def add(start, end, reason):
        if end-start < 1.:
            mid = (start+end)/2
            start, end = mid-.5, mid+.5
        if min(record['durationSeconds'], end) <= max(0., start):
            return
        for x in valid(record, [(max(0., start), min(record['durationSeconds'], end))]):
            key = (x.start, x.end)
            items.setdefault(key, set()).add(reason)

    if inventory == 'legacy':
        rows = (legacy.combination_plan(record, p, n, 'bidirectional_any') if mode == 'production'
                else legacy.individual_plan(record, p, t, a[:, 0], 'uncertain_medium'))
        for row in rows:
            if row['flagged']:
                add(row['start'], row['end'], 'legacy_'+row['kind'])
    else:
        if mode == 'production':
            for x in iv.difference(p, n):
                add(x.start, x.end, 'production_only')
            for x in iv.difference(n, p):
                add(x.start, x.end, 'neural_only')
            for pp, nn in overlap_groups(p, n):
                if len(pp) > 1 or len(nn) > 1:
                    for group in (pp, nn):
                        for left, right in zip(group, group[1:]):
                            add(left['end'], right['start'], 'split_merge')
        else:
            for x in p:
                values = a[(t >= x['start']) & (t < x['end']) & allowed, 0]
                if not len(values) or np.mean(values) < .8:
                    add(x['start'], x['end'], 'uncertain_positive')
        components = live_components(record, t, a)
        for x in components:
            start, end = x['start'], x['end']
            if inventory == 'local_heads':
                ps, vs = peak(t, a, start, 1, allowed=local_ticks(record, t, start))
                pe, ve = peak(t, a, end, 2, allowed=local_ticks(record, t, end))
                if vs >= .25 and ps < end-.5:
                    start = ps
                if ve >= .25 and pe > start+.5:
                    end = pe
            missing = iv.duration(iv.difference([(start, end)], p))
            if missing >= .5 and missing >= .25*(end-start):
                add(start, end, 'missed_live')
        if inventory == 'local_heads':
            for x in [*p, *n]:
                for key, head in (('start', 1), ('end', 2)):
                    point, value = peak(t, a, x[key], head, allowed=local_ticks(record, t, x[key]))
                    if value >= .25 and abs(point-x[key]) >= .25:
                        add(min(point, x[key]), max(point, x[key]), 'boundary_'+key)
                ids = np.flatnonzero((t > x['start']+2) & (t < x['end']-2) & (a[:, 1] >= .35) & allowed)
                chosen = []
                for i in sorted(ids, key=lambda j: (-float(a[j, 1]), float(t[j]))):
                    if any(abs(float(t[i])-v) < 1. for v in chosen):
                        continue
                    preceding = (t >= t[i]-3) & (t < t[i]) & local_ticks(record, t, float(t[i]))
                    if np.any(preceding) and (np.max(a[preceding, 2]) >= .25 or np.min(a[preceding, 0]) < .35):
                        chosen.append(float(t[i])); add(float(t[i])-1, float(t[i])+1, 'internal_start')
    output = []
    for i, ((start, end), reasons) in enumerate(sorted(items.items())):
        values = a[(t >= start) & (t < end) & allowed]
        mean_live = float(np.mean(values[:, 0])) if len(values) else .5
        max_live = float(np.max(values[:, 0])) if len(values) else .5
        boundary = float(np.max(values[:, 1:3])) if len(values) else .5
        evidence = []
        for reason in reasons:
            if reason in ('production_only', 'uncertain_positive'):
                evidence.append(1-mean_live)
            elif reason in ('neural_only', 'missed_live', 'legacy_negative'):
                evidence.append(max_live)
            elif reason in ('split_merge', 'internal_start') or reason.startswith('boundary_'):
                evidence.append(max(.5, boundary))
            else:
                evidence.append(max(1-mean_live, boundary))
        strength = max(evidence)
        event_risk = any(r != 'production_only' for r in reasons)
        cost = iv.duration(playback(record, permission(record, [(start, end)])))
        utility = ((end-start)*strength + 4.*strength*event_risk)/max(cost, 1e-12)
        output.append({'id': f'q{i:05d}', 'start': start, 'end': end, 'reasons': sorted(reasons),
                       'meanLive': mean_live, 'maxLive': max_live, 'maxBoundary': boundary,
                       'priority': utility, 'standalonePlaybackSeconds': cost})
    return output


def budget_queues(record, candidates, ranker):
    if ranker not in RANKERS:
        raise ValueError('Unknown ranker')
    ordered = sorted(candidates, key=lambda x: ((-x['priority'] if ranker == 'evidence' else x['start']),
                                               x['start'], x['end'], x['id']))
    selected, seen, result = [], set(), []
    total = iv.duration(legacy.valid_time(record))
    for fraction in BUDGETS:
        budget = fraction*total
        for row in ordered:
            if row['id'] in seen:
                continue
            trial = permission(record, [*selected, row])
            if iv.duration(playback(record, trial)) <= budget+1e-9:
                selected.append(row); seen.add(row['id'])
        windows = permission(record, selected)
        viewing = playback(record, windows)
        result.append({'budgetFraction': fraction, 'budgetSeconds': budget,
                       'selectedIds': [x['id'] for x in selected],
                       'editWindows': iv.serial(windows), 'playbackWindows': iv.serial(viewing),
                       'reviewSeconds': iv.duration(viewing), 'editSeconds': iv.duration(windows),
                       'reviewClips': len(viewing), 'decisionRegions': len(windows),
                       'proposalsSelected': len(selected), 'proposalsAvailable': len(candidates)})
    return result


def edit_events(record, base, windows):
    """Perfect raw occupancy + boundary edits confined to explicit windows.

Ignored base time remains untouched; canonical export/evaluation masks it later.
No true endpoint outside windows is imported. Censored edge proposals remain
imperfect events rather than receiving oracle endpoints outside the viewed task.
"""
    p = event_rows(base)
    w = valid(record, windows)
    gold = event_rows(record['rallies'])
    coverage = iv.union(iv.difference(p, w), iv.intersection(gold, w))
    markers = {x[k] for x in p for k in ('start', 'end') if not contains(w, x[k])}
    markers.update(x[k] for x in gold for k in ('start', 'end') if contains(w, x[k]))
    output = []
    for x in coverage:
        cuts = [x.start, *sorted(v for v in markers if x.start < v < x.end), x.end]
        for start, end in zip(cuts, cuts[1:]):
            output.append({'id': f'e{len(output):05d}', 'start': start, 'end': end,
                           'startObserved': any(g['start'] == start for g in gold) if contains(w, start)
                               else any(b['start'] == start for b in p),
                           'endObserved': any(g['end'] == end for g in gold) if contains(w, end)
                               else any(b['end'] == end for b in p)})
    edges = {v for x in w for v in (x.start, x.end)}
    censored_starts = sum(x['start'] in edges and any(g['start'] < x['start'] < g['end'] for g in gold) for x in output)
    censored_ends = sum(x['end'] in edges and any(g['start'] < x['end'] < g['end'] for g in gold) for x in output)
    unresolved = sum(bool(iv.intersection([g], w)) and
                     not (contains(w, g['start']) and contains(w, g['end'])) for g in gold)
    return {'events': output, 'censoredStarts': censored_starts, 'censoredEnds': censored_ends,
            'unobservedStarts': sum(not x['startObserved'] for x in output),
            'unobservedEnds': sum(not x['endObserved'] for x in output),
            'touchedRalliesWithUneditableBoundary': unresolved,
            'reviewedTrueRallies': len(legacy.touched_rallies(record, w)),
            'reviewedTrueRallyIds': legacy.touched_rallies(record, w)}
