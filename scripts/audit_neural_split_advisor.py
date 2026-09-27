#!/usr/bin/env python3
"""Independent scalar audits for production-preserving neural split advice.

This module imports no model, proposal, interval, or evaluation implementation.
The same endpoint sweep accounts for geometry, but event identity and parent
lineage are always checked before any union is made.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real


PADDINGS = (0, 1, 2, 3)
BUDGETS = (.05, .10, .20, .40)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value), 'Expected finite numeric value')
    return float(value)


def close(expected, actual, label):
    require(math.isclose(number(expected), number(actual), rel_tol=1e-10, abs_tol=1e-8),
            f'{label}: expected {expected!r}, got {actual!r}')


def check_tree(expected, actual, name='result'):
    if isinstance(expected, Mapping):
        require(isinstance(actual, Mapping), name + ': expected object')
        for key, value in expected.items():
            require(key in actual, name + ': missing ' + key)
            check_tree(value, actual[key], name + '.' + key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(expected) == len(actual),
                name + ': sequence length differs')
        for i, (a, b) in enumerate(zip(expected, actual)):
            check_tree(a, b, f'{name}[{i}]')
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(expected == actual, name + ': value differs')
    else:
        close(expected, actual, name)


def pairs(values, duration=math.inf):
    output = []
    for row in values:
        a, b = ((row['start'], row['end']) if isinstance(row, Mapping) else
                (row.start, row.end) if hasattr(row, 'start') else row)
        a, b = number(a), number(b)
        require(a < b, 'Nonpositive interval')
        a, b = max(0., a), min(duration, b)
        if a < b:
            output.append((a, b))
    return output


def sweep(groups, predicate):
    changes = {}
    for index, intervals in enumerate(groups):
        for a, b in intervals:
            require(a < b, 'Nonpositive sweep span')
            changes.setdefault(a, [0] * len(groups))[index] += 1
            changes.setdefault(b, [0] * len(groups))[index] -= 1
    active, result = [0] * len(groups), []
    points = sorted(changes)
    for index, point in enumerate(points[:-1]):
        active = [a + b for a, b in zip(active, changes[point])]
        require(all(x >= 0 for x in active), 'Negative sweep occupancy')
        if predicate([x > 0 for x in active]):
            end = points[index + 1]
            if result and result[-1][1] == point:
                result[-1] = (result[-1][0], end)
            else:
                result.append((point, end))
    return result


def union(*groups):
    return sweep(groups, any)


def intersection(left, right):
    return sweep((left, right), all)


def difference(left, right):
    return sweep((left, right), lambda bits: bits[0] and not bits[1])


def seconds(values):
    return math.fsum(b - a for a, b in union(values))


def export(values, record, padding):
    duration = number(record['durationSeconds'])
    padded = union([(max(0., a - padding), min(duration, b + padding))
                    for a, b in pairs(values, duration)])
    joined = []
    for a, b in padded:
        if joined and 0 < a - joined[-1][1] < 3.:
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    return difference(joined, pairs(record.get('ignoredIntervals', []), duration))


def _rates(row):
    p = row['paddedIntersectionSeconds'] / row['paddedModelExportSeconds'] if row['paddedModelExportSeconds'] else 0.
    r = row['coreIntersectionSeconds'] / row['coreHumanSeconds'] if row['coreHumanSeconds'] else 0.
    return {**row, 'P_pad': p, 'R_core': r, 'F1_padP_coreR': 2*p*r/(p+r) if p+r else 0.,
            'exportDurationDifferenceSeconds': row['paddedModelExportSeconds'] - row['paddedHumanExportSeconds']}


def duration_rows(records, export_overrides=None):
    """Independent four-padding pooled/per-recording footage accounting."""
    output = []
    require(len({x['id'] for x in records}) == len(records), 'Duplicate recording')
    for pad in PADDINGS:
        details = []
        for record in records:
            duration = number(record['durationSeconds'])
            ignored = pairs(record.get('ignoredIntervals', []), duration)
            valid = difference([(0., duration)], ignored)
            core = difference(pairs(record['rallies'], duration), ignored)
            human = export(record['rallies'], record, pad)
            model = (export(record['predictions'], record, pad) if export_overrides is None else
                     intersection(pairs(export_overrides[record['id']][str(pad)], duration), valid))
            counts = {'paddedModelExportSeconds': seconds(model), 'paddedHumanExportSeconds': seconds(human),
                      'evaluableVideoSeconds': seconds(valid), 'coreHumanSeconds': seconds(core),
                      'paddedIntersectionSeconds': seconds(intersection(model, human)),
                      'coreIntersectionSeconds': seconds(intersection(model, core)),
                      'correctlyRemovedSeconds': seconds(difference(valid, union(model, human))),
                      'incorrectlyRemovedSeconds': seconds(difference(human, model)),
                      'incorrectExportSeconds': seconds(difference(model, human)),
                      'missedCoreSeconds': seconds(difference(core, model))}
            close(seconds(valid), counts['paddedModelExportSeconds'] + counts['correctlyRemovedSeconds']
                  + counts['incorrectlyRemovedSeconds'], 'time partition')
            details.append({'id': record['id'], 'sourceGroup': record['sourceGroup'], **_rates(counts)})
        sums = {key: math.fsum(row[key] for row in details) for key in counts}
        output.append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3,
                       **_rates(sums), 'perRecording': details})
    return output


def audit_duration_records(records, actual, export_overrides=None):
    check_tree(duration_rows(records, export_overrides), actual, 'durationMetrics')
    return {'passed': True, 'recordings': len(records), 'paddingCases': 4,
            'pooledAndPerRecordingCompared': True}


def audit_coverage_lineage(record, parents, events):
    """Require every parent to have an exact, nonoverlapping ordered tiling."""
    duration = number(record['durationSeconds'])
    require(len({p['id'] for p in parents}) == len(parents), 'Duplicate parent ID')
    require(len({e['id'] for e in events}) == len(events), 'Duplicate output event ID')
    by_id = {p['id']: p for p in parents}
    require(all(e.get('parentId') in by_id for e in events), 'Unknown parent lineage')
    for parent in parents:
        children = [e for e in events if e['parentId'] == parent['id']]
        require(children, 'Parent omitted')
        ranges = pairs(children, duration)
        require(ranges == sorted(ranges), 'Children not chronological')
        require(ranges[0][0] == parent['start'] and ranges[-1][1] == parent['end'],
                'Parent edge changed')
        require(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])),
                'Children have a gap or overlap')
        for child in children:
            for key, value in parent.items():
                if key not in ('id', 'start', 'end', 'startObserved', 'endObserved', 'parentId', 'proposalIds'):
                    require(child.get(key) == value, f'Parent metadata changed: {key}')
        if len(children) == 1:
            require(children[0]['id'] == parent['id'], 'Unsplit parent identity changed')
        require(children[0].get('startObserved', True) == parent.get('startObserved', True),
                'Original start observation changed')
        require(children[-1].get('endObserved', True) == parent.get('endObserved', True),
                'Original end observation changed')
    require(union(pairs(parents, duration)) == union(pairs(events, duration)), 'Raw coverage changed')
    for pad in PADDINGS:
        require(export(parents, record, pad) == export(events, record, pad),
                f'Export coverage changed at padding {pad}')
    return {'passed': True, 'parentsAudited': len(parents), 'eventsAudited': len(events),
            'exactRawCoverage': True, 'paddingCases': 4, 'parentMetadataPreserved': True}


def _overlap(a, b):
    return max(0., min(a[1], b[1]) - max(a[0], b[0]))


def reconstruct_candidates(record, parents, neural, times, scores, policy):
    """Scalar reconstruction; neither labels nor detector helpers are read."""
    require(policy in ('event_starts', 'head_evidence', 'corroborated'), 'Unknown split policy')
    t = [number(x) for x in times]
    a = [[number(x) for x in row] for row in scores]
    require(len(a) == len(t) and all(len(row) == 4 for row in a), 'Invalid four-head scores')
    require(all(x < y for x, y in zip(t, t[1:])), 'Invalid timestamps')
    duration = number(record['durationSeconds'])
    components = difference([(0., duration)], pairs(record.get('ignoredIntervals', []), duration))
    neural = sorted(neural, key=lambda x: (x['start'], x['end'], x.get('id', '')))
    output = []
    for parent in parents:
        p = (parent['start'], parent['end'])
        events, heads = [], []
        for left, right in components:
            effective = (max(left, p[0]), min(right, p[1]))
            if effective[1] <= effective[0]:
                continue
            nn = [row for row in neural if _overlap((row['start'], row['end']), effective) >= .5]
            for previous, current in zip(nn, nn[1:]):
                point = float(current['start'])
                if not (effective[0] + 2. < point < effective[1] - 2.):
                    continue
                low, high = max(current['start'], effective[0]), min(current['end'], effective[1])
                values = [row[0] for tick, row in zip(t, a) if low <= tick < high]
                strength = math.fsum(values) / len(values) if values else 0.
                events.append({'time': point, 'strength': strength, 'component': (left, right),
                               'source': (previous.get('id', ''), current.get('id', ''))})
            ticks = [i for i, value in enumerate(t) if left <= value < right]
            j = 0
            while j < len(ticks):
                first = j
                strength = a[ticks[j]][1]
                while j + 1 < len(ticks) and a[ticks[j + 1]][1] == strength:
                    j += 1
                previous = a[ticks[first - 1]][1] if first else -math.inf
                following = a[ticks[j + 1]][1] if j + 1 < len(ticks) else -math.inf
                point = t[ticks[first]]
                if (strength >= .35 and strength > previous and strength > following
                        and effective[0] + 2. < point < effective[1] - 2.):
                    before = [a[i] for i in ticks if max(point - 3., p[0]) <= t[i] < point]
                    if before and (max(row[2] for row in before) >= .25
                                   or min(row[0] for row in before) < .35):
                        heads.append({'time': point, 'strength': strength, 'component': (left, right),
                                      'source': (ticks[first],)})
                j += 1
        def suppress(rows):
            kept = []
            for candidate in sorted(rows, key=lambda x: (-x['strength'], x['time'], str(x['source']))):
                if not any(candidate['component'] == old['component']
                           and abs(candidate['time'] - old['time']) < 1. for old in kept):
                    kept.append(candidate)
            return sorted(kept, key=lambda x: x['time'])
        events, heads = suppress(events), suppress(heads)
        if policy == 'event_starts':
            choices = events
        elif policy == 'head_evidence':
            choices = heads
        else:
            choices, used_events, used_heads = [], set(), set()
            options = [(abs(e['time'] - h['time']), -h['strength'], e['time'], h['time'], i, j)
                       for i, e in enumerate(events) for j, h in enumerate(heads)
                       if e['component'] == h['component'] and abs(e['time'] - h['time']) <= 1.]
            for _, _, _, _, i, j in sorted(options):
                if i not in used_events and j not in used_heads:
                    used_events.add(i); used_heads.add(j)
                    choices.append({'time': heads[j]['time'],
                                    'strength': min(events[i]['strength'], heads[j]['strength']),
                                    'component': heads[j]['component'],
                                    'source': (*events[i]['source'], *heads[j]['source'])})
        chosen = suppress(choices)
        output.extend({'parentId': parent['id'], 'time': item['time'], 'priority': item['strength']}
                      for item in sorted(chosen, key=lambda x: x['time']))
    return output


def audit_candidates(record, parents, neural, times, scores, policy, actual):
    expected = reconstruct_candidates(record, parents, neural, times, scores, policy)
    check_tree(expected, actual, 'splitCandidates')
    require(len({row['id'] for row in actual}) == len(actual), 'Duplicate split proposal ID')
    lookup = {row['id']: row for row in parents}
    for row in actual:
        parent = lookup[row['parentId']]
        require(row['parentStart'] == parent['start'] and row['parentEnd'] == parent['end'],
                'Split parent geometry differs')
        require(parent['start'] <= row['start'] <= row['time'] < row['end'] <= parent['end'],
                'Split display band outside parent')
        require(row['policy'] == policy, 'Split policy mismatch')
    return {'passed': True, 'candidatesAudited': len(actual), 'labelDataRead': False,
            'scalarReconstruction': True}


def cleanup_support(record, parents, neural):
    """Fraction covered by compact intervals dilated two seconds, no joining."""
    duration = number(record['durationSeconds'])
    ignored = pairs(record.get('ignoredIntervals', []), duration)
    components = difference([(0., duration)], ignored)
    support = []
    for left, right in components:
        inside = intersection(pairs(neural, duration), [(left, right)])
        support.extend((max(left, a - 2.), min(right, b + 2.)) for a, b in inside)
    neural_support = union(support)
    result = {}
    for parent in parents:
        parent_support = difference(pairs([parent], duration), ignored)
        total = seconds(parent_support)
        fraction = seconds(intersection(parent_support, neural_support)) / total if total else 1.
        result[parent['id']] = {'supportFraction': fraction, 'flagged': fraction < .5,
                               'priority': 1. - fraction}
    return result


def audit_cleanup(record, parents, neural, actual):
    support = cleanup_support(record, parents, neural)
    expected = []
    for parent in parents:
        values = support[parent['id']]
        if values['flagged']:
            valid = difference(pairs([parent]), pairs(record.get('ignoredIntervals', [])))
            total = seconds(valid)
            expected.append({'id': f'c{len(expected):05d}', 'parentId': parent['id'],
                             'parentStart': parent['start'], 'parentEnd': parent['end'],
                             'start': parent['start'], 'end': parent['end'], 'kind': 'cleanup',
                             'supportFraction': values['supportFraction'],
                             'supportedSeconds': values['supportFraction'] * total,
                             'parentValidSeconds': total, 'priority': values['priority'],
                             'confidencePriority': values['priority']})
    check_tree(expected, actual, 'cleanupCandidates')
    return {'passed': True, 'cleanupFlagsAudited': len(actual), 'labelDataRead': False}


def audit_jobs_queues(record, parents, splits, cleanup, inventory, ranker, jobs, queues,
                      budgets=BUDGETS):
    require(inventory in ('split_only', 'cleanup_only', 'combined'), 'Unknown review inventory')
    require(ranker in ('chronological', 'evidence'), 'Unknown queue ranker')
    grouped = {}
    parent_map = {row['id']: row for row in parents}
    flags = ([] if inventory == 'cleanup_only' else [(row, 'split') for row in splits])
    flags += ([] if inventory == 'split_only' else [(row, 'cleanup') for row in cleanup])
    for flag, reason in flags:
        pid = flag['parentId']; parent = parent_map[pid]
        row = grouped.setdefault(pid, {'id': 'parent:' + pid, 'parentId': pid,
            'start': parent['start'], 'end': parent['end'], 'splitIds': [], 'cleanupIds': [], 'priority': 0.})
        row[reason + 'Ids'].append(flag['id'])
        row['priority'] = max(row['priority'], flag['priority'])
    for row in grouped.values():
        row['splitIds'].sort(); row['cleanupIds'].sort()
        row['standalonePlaybackSeconds'] = seconds(export([row], record, 2))
        row['evidencePerSecond'] = row['priority'] / row['standalonePlaybackSeconds']
    expected_jobs = sorted(grouped.values(), key=lambda r: (r['start'], r['end'], r['id']))
    check_tree(expected_jobs, jobs, 'reviewJobs')
    ordered = sorted(expected_jobs, key=lambda r: ((-r['evidencePerSecond'],) if ranker == 'evidence' else ())
                     + (r['start'], r['end'], r['id']))
    valid = difference([(0., record['durationSeconds'])], pairs(record.get('ignoredIntervals', [])))
    chosen, ids = [], set()
    require(len(queues) == len(budgets), 'Queue budget count differs')
    for fraction, actual in zip(budgets, queues):
        limit = fraction * seconds(valid)
        for row in ordered:
            if row['id'] not in ids and seconds(export([*chosen, row], record, 2)) <= limit + 1e-9:
                chosen.append(row); ids.add(row['id'])
        view = export(chosen, record, 2)
        edit = difference(pairs(chosen), pairs(record.get('ignoredIntervals', [])))
        expected = {'budgetFraction': fraction, 'budgetSeconds': limit,
                    'reviewSeconds': seconds(view), 'unusedBudgetSeconds': limit - seconds(view),
                    'selectedJobIds': [row['id'] for row in chosen],
                    'selectedParentIds': [row['parentId'] for row in chosen],
                    'selectedSplitIds': sorted({i for row in chosen for i in row['splitIds']}),
                    'selectedCleanupIds': sorted({i for row in chosen for i in row['cleanupIds']}),
                    'reviewJobs': len(chosen), 'jobsAvailable': len(jobs),
                    'reviewClips': len(view), 'editRegions': len(edit)}
        check_tree(expected, actual, f'queue{fraction}')
        require(view == pairs(actual['playbackWindows']), 'Playback union differs')
        require(edit == pairs(actual['editWindows']), 'Edit union differs')
    return {'passed': True, 'jobsAudited': len(jobs), 'budgetsAudited': len(budgets),
            'labelDataRead': False, 'nestedSelections': True}


def split_events(parents, proposals):
    result = []
    for parent in parents:
        cuts = {}
        for row in proposals:
            if row['parentId'] == parent['id']:
                require(parent['start'] < row['time'] < parent['end'], 'Proposed start outside parent')
                cuts.setdefault(row['time'], []).append(row['id'])
        boundaries = [parent['start'], *sorted(cuts), parent['end']]
        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            result.append({**parent,
                'id': parent['id'] if not cuts else f"{parent['id']}::child:{i + 1}",
                'parentId': parent['id'], 'parentStart': parent['start'], 'parentEnd': parent['end'],
                'start': start, 'end': end, 'childIndex': i,
                'startObserved': bool(parent.get('startObserved', True)) if i == 0 else True,
                'endObserved': bool(parent.get('endObserved', True)) if i == len(boundaries) - 2 else False,
                'startKind': 'inherited' if i == 0 else 'proposed-rally-start',
                'endKind': 'inherited' if i == len(boundaries) - 2 else 'partition-only',
                'proposalIds': sorted(cuts.get(start, [])), 'endProposalIds': sorted(cuts.get(end, []))})
    return result


def audit_apply_splits(record, parents, proposals, actual):
    check_tree(split_events(parents, proposals), actual, 'splitEvents')
    return audit_coverage_lineage(record, parents, actual)


def split_targets(record):
    """Mask censored gold, then find secondary material truth per parent component."""
    duration = record['durationSeconds']
    ignored = pairs(record.get('ignoredIntervals', []), duration)
    gold = pairs(record['rallies'], duration)
    censored = [row for row in gold if intersection([row], ignored)]
    eligible = [(i, row) for i, row in enumerate(gold) if not intersection([row], ignored)]
    targets = []
    for parent_index, parent in enumerate(record['productionEvents']):
        components = difference(pairs([parent], duration), union(ignored, censored))
        for ci, component in enumerate(components):
            covered = [(i, row) for i, row in eligible
                       if _overlap(row, component) >= min(.5, .1*(row[1] - row[0]))]
            for i, row in covered[1:]:
                if component[0] < row[0] < component[1]:
                    edge = min(row[0] - component[0], component[1] - row[0])
                    targets.append({'id': f"{parent['id']}::gold-{i}", 'parentId': parent['id'],
                                    'parentIndex': parent_index, 'truthIndex': i, 'time': row[0],
                                    'componentIndex': ci, 'componentStart': component[0], 'componentEnd': component[1],
                                    'edgeDistanceSeconds': edge, 'accessible': edge > 2.})
    return targets


def _optimal_starts(proposals, targets, tolerance=1.):
    """Monotone dynamic programming maximizes count, then minimizes distance."""
    rows = [[(0, 0.) for _ in range(len(targets) + 1)] for _ in range(len(proposals) + 1)]
    for i, proposal in enumerate(proposals, 1):
        for j, target in enumerate(targets, 1):
            best = max(rows[i - 1][j], rows[i][j - 1])
            distance = abs(proposal['time'] - target['time'])
            if (distance <= tolerance and target['componentStart'] <= proposal['time'] < target['componentEnd']):
                previous = rows[i - 1][j - 1]
                best = max(best, (previous[0] + 1, previous[1] - distance))
            rows[i][j] = best
    count, negative_error = rows[-1][-1]
    return count, -negative_error


def audit_human(record, splits, cleanup, queue, actual):
    parents = record['productionEvents']
    targets = split_targets(record)
    selected = [row for row in splits if row['id'] in set(queue['selectedSplitIds'])]
    selected_map = {row['id']: row for row in selected}
    accepted = actual['acceptedSplits']
    require(len({row['id'] for row in accepted}) == len(accepted), 'Split accepted more than once')
    target_map = {(row['parentId'], row['truthIndex']): row for row in targets}
    seen_targets, error = set(), 0.
    for row in accepted:
        require(row['id'] in selected_map, 'Unreviewed split accepted')
        original = selected_map[row['id']]
        key = (row['parentId'], row['targetTruthIndex'])
        require(key in target_map and key not in seen_targets, 'Unknown/duplicate accepted split target')
        target = target_map[key]; seen_targets.add(key)
        check_tree({**original, 'time': target['time'], 'originalProposedTime': original['time'],
                    'humanConfirmed': True, 'targetTruthIndex': target['truthIndex']}, row, 'acceptedSplit')
        distance = abs(original['time'] - target['time'])
        require(distance <= 1. and target['componentStart'] <= original['time'] < target['componentEnd'],
                'Accepted split outside tolerance or component')
        error += distance
    optimal_count, optimal_error = 0, 0.
    for parent in parents:
        pp = sorted([row for row in selected if row['parentId'] == parent['id']], key=lambda r: r['time'])
        tt = sorted([row for row in targets if row['parentId'] == parent['id']], key=lambda r: r['time'])
        count, distance = _optimal_starts(pp, tt)
        optimal_count += count; optimal_error += distance
    require(len(accepted) == optimal_count, 'Human acceptance fails maximum cardinality')
    close(optimal_error, error, 'Human acceptance nearest total distance')
    gold = difference(pairs(record['rallies']), pairs(record.get('ignoredIntervals', [])))
    parent_map = {row['id']: row for row in parents}
    removed = sorted({row['parentId'] for row in cleanup if row['id'] in set(queue['selectedCleanupIds'])
                      and not intersection(pairs([parent_map[row['parentId']]]), gold)})
    expected = [row for row in split_events(parents, accepted) if row['parentId'] not in removed]
    check_tree(expected, actual['events'], 'humanEvents')
    check_tree(removed, actual['removedFalseParentIds'], 'removedFalseParentIds')
    retained = [row for row in parents if row['id'] not in removed]
    coverage_audit = audit_coverage_lineage(record, retained, actual['events'])
    before = difference(pairs(parents), pairs(record.get('ignoredIntervals', [])))
    after = difference(pairs(expected), pairs(record.get('ignoredIntervals', [])))
    require(intersection(before, gold) == intersection(after, gold), 'Human deleted true rally core')
    touched = lambda windows: sum(bool(intersection(difference([row], pairs(record.get('ignoredIntervals', []))),
                                                  pairs(windows))) for row in pairs(record['rallies']))
    counts = {'proposalsReviewed': len(selected), 'acceptedSplitCount': len(accepted),
              'rejectedSplitCount': len(selected) - len(accepted), 'removedFalseParents': len(removed),
              'eventTimelineRemovedSeconds': seconds(difference(before, after)),
              'eventTimelineAddedSeconds': seconds(difference(after, before)),
              'reviewedTrueRallies': touched(queue['editWindows']),
              'playbackTrueRallies': touched(queue['playbackWindows'])}
    check_tree(counts, actual, 'humanWorkload')
    return {'passed': True, 'proposalsReviewed': len(selected), 'acceptedSplitsAudited': len(accepted),
            'removedFalseParentsAudited': len(removed), 'noTrueCoreDeleted': True,
            'retainedParentLineageAudit': coverage_audit}


def _count_rates(matched, predicted, true):
    return {'true': true, 'predicted': predicted, 'matched': matched,
            'falsePositive': predicted - matched, 'falseNegative': true - matched,
            'precision': matched / predicted if predicted else (0. if true else None),
            'recall': matched / true if true else None,
            'f1': 2. * matched / (predicted + true) if true else None}


def _split_metric_record(record, actual):
    duration = record['durationSeconds']
    ignored = pairs(record.get('ignoredIntervals', []), duration)
    gold = pairs(record['rallies'], duration)
    censored = [(i, row) for i, row in enumerate(gold) if intersection([row], ignored)]
    eligible = [(i, row) for i, row in enumerate(gold) if not intersection([row], ignored)]
    excluded = union(ignored, [row for _, row in censored])
    parents = record['productionEvents']
    predictions = record.get('predictions', parents)
    base_support = [difference(pairs([row], duration), excluded) for row in parents]
    prediction_support = [difference(pairs([row], duration), excluded) for row in predictions]
    targets = split_targets(record)
    check_tree(targets, actual['targets'], 'splitMetricTargets')

    def diagnostics(support):
        overlaps = [[seconds(intersection([truth], item)) for item in support] for _, truth in eligible]
        material = [[overlap >= min(.5, .1*(truth[1] - truth[0])) for overlap in row]
                    for (_, truth), row in zip(eligible, overlaps)]
        return {'completeMissIndexes': [i for (i, _), row in zip(eligible, overlaps) if not any(x > 0 for x in row)],
                'mergedPredictions': sum(sum(row[j] > 0 for row in overlaps) > 1 for j in range(len(support))),
                'mergedPredictionsMaterial': sum(sum(row[j] for row in material) > 1 for j in range(len(support))),
                'splitTrueRallies': sum(sum(x > 0 for x in row) > 1 for row in overlaps),
                'splitTrueRalliesMaterial': sum(sum(row) > 1 for row in material)}

    baseline, result = diagnostics(base_support), diagnostics(prediction_support)
    check_tree(baseline, actual['baselineDiagnostics'], 'baselineDiagnostics')
    check_tree(result, actual['resultDiagnostics'], 'resultDiagnostics')
    base_misses, new_misses = set(baseline['completeMissIndexes']), set(result['completeMissIndexes'])
    base_union, result_union = union(*base_support), union(*prediction_support)
    lost, added = difference(base_union, result_union), difference(result_union, base_union)
    parent_map = {row['id']: (row, support) for row, support in zip(parents, base_support)}
    original_proposals = record.get('splitProposals', [])
    proposals = []
    for i, proposal in enumerate(original_proposals):
        parent, support = parent_map[proposal['parentId']]
        time = proposal['time']
        if any(a <= time < b for a, b in excluded):
            continue
        component = next(j for j, (a, b) in enumerate(support) if a <= time < b)
        proposals.append({'index': i, 'id': proposal.get('id', f'proposal-{i}'), 'parentId': parent['id'],
                          'time': time, 'componentIndex': component})
    check_tree(proposals, actual['proposals'], 'metricProposals')
    row = {'id': record['id'], 'sourceGroup': record['sourceGroup'],
           'originalTrueRallies': len(gold), 'ignoredTouchedTrueRallies': len(censored),
           'eligibleTrueRallies': len(eligible), 'productionParents': len(parents),
           'splitTargets': len(targets), 'uniqueTargetTrueRallies': len({x['truthIndex'] for x in targets}),
           'accessibleSplitTargets': sum(x['accessible'] for x in targets),
           'edgeInaccessibleSplitTargets': sum(not x['accessible'] for x in targets),
           'originalSplitProposals': len(original_proposals), 'maskedSplitProposals': len(original_proposals) - len(proposals),
           'splitProposals': len(proposals), 'baselineCompleteMisses': len(base_misses),
           'resultCompleteMisses': len(new_misses), 'retainedCompleteMisses': len(base_misses & new_misses),
           'additionalCompleteMisses': len(new_misses - base_misses),
           'recoveredCompleteMisses': len(base_misses - new_misses),
           'rawCoreSecondsLostFromBaseline': math.fsum(seconds(intersection([g], lost)) for _, g in eligible),
           'rawCoreSecondsAddedToBaseline': math.fsum(seconds(intersection([g], added)) for _, g in eligible),
           'rawSelectedSecondsLostFromBaseline': seconds(lost),
           'rawSelectedSecondsAddedToBaseline': seconds(added)}
    for prefix, diagnostic in (('baseline', baseline), ('result', result)):
        for name in ('mergedPredictions', 'mergedPredictionsMaterial', 'splitTrueRallies', 'splitTrueRalliesMaterial'):
            row[prefix + name[0].upper() + name[1:]] = diagnostic[name]
    check_tree(row, actual, 'splitRecordCounts')
    check_tree(sorted(new_misses - base_misses), actual['additionalCompleteMissIndexes'], 'additionalMissIds')
    check_tree(sorted(base_misses - new_misses), actual['recoveredCompleteMissIndexes'], 'recoveredMissIds')
    row['splitLocalization'] = {}
    for tolerance in (.5, 1., 2.):
        key = format(tolerance, 'g'); emitted = actual['splitLocalization'][key]
        optimal_count = 0; optimal_error = 0.
        for parent in parents:
            pp = sorted([p for p in proposals if p['parentId'] == parent['id']], key=lambda x: x['time'])
            tt = sorted([t for t in targets if t['parentId'] == parent['id']], key=lambda x: x['time'])
            count, error = _optimal_starts(pp, tt, tolerance)
            optimal_count += count; optimal_error += error
        rates = _count_rates(optimal_count, len(proposals), len(targets))
        check_tree(rates, emitted, f'splitLocalization{key}')
        target_map = {x['id']: x for x in targets}; prop_map = {x['index']: x for x in proposals}
        used_t, used_p = set(), set(); actual_error = 0.; accessible = 0
        for pair in emitted['pairs']:
            tid, pi = pair['targetId'], pair['proposalIndex']
            require(tid in target_map and pi in prop_map and tid not in used_t and pi not in used_p,
                    'Invalid duplicate split metric correspondence')
            target, proposal = target_map[tid], prop_map[pi]
            require(target['parentId'] == proposal['parentId']
                    and target['componentIndex'] == proposal['componentIndex'], 'Split match bridges parent or mask')
            error = proposal['time'] - target['time']
            require(abs(error) <= tolerance, 'Split match exceeds tolerance')
            close(error, pair['errorSeconds'], 'Matched timestamp error')
            used_t.add(tid); used_p.add(pi); actual_error += abs(error); accessible += target['accessible']
        require(len(used_t) == optimal_count, 'Split match cardinality differs')
        close(optimal_error, actual_error, 'Split matching total distance')
        rates.update(matchedAccessibleTargets=accessible,
                     accessibleTargetRecall=accessible / row['accessibleSplitTargets'] if row['accessibleSplitTargets'] else None)
        check_tree(rates, emitted, f'splitLocalization{key}')
        row['splitLocalization'][key] = rates
    return row


def audit_split_metrics(records, actual):
    """Independent scalar targets, DP timing matches, coverage and group pooling."""
    require(len(records) == len(actual['recordings']), 'Split recording count differs')
    rows = [_split_metric_record(record, emitted) for record, emitted in zip(records, actual['recordings'])]
    sum_fields = [key for key in rows[0] if key not in ('id', 'sourceGroup', 'splitLocalization')]
    def pooled(subset):
        result = {key: math.fsum(row[key] for row in subset) for key in sum_fields}
        result['splitLocalization'] = {}
        for key in ('0.5', '1', '2'):
            counts = {name: sum(row['splitLocalization'][key][name] for row in subset)
                      for name in ('matched', 'predicted', 'true')}
            rates = _count_rates(**counts)
            accessible = sum(row['splitLocalization'][key]['matchedAccessibleTargets'] for row in subset)
            rates.update(matchedAccessibleTargets=accessible,
                         accessibleTargetRecall=accessible / result['accessibleSplitTargets'] if result['accessibleSplitTargets'] else None)
            result['splitLocalization'][key] = rates
        return result
    check_tree(pooled(rows), actual['pooled'], 'splitPooled')
    groups = {row['sourceGroup'] for row in rows}
    require(set(actual['sourceGroups']) == groups, 'Split source-group inventory differs')
    for group in sorted(groups):
        check_tree(pooled([row for row in rows if row['sourceGroup'] == group]), actual['sourceGroups'][group],
                   'splitSourceGroup.' + group)
    check_tree({'primaryToleranceSeconds': 1., 'tolerancesSeconds': [.5, 1., 2.],
                'edgeGuardSeconds': 2.}, actual['metricContract'], 'splitMetricContract')
    return {'passed': True, 'recordingsAudited': len(records), 'sourceGroupsAudited': len(groups),
            'tolerancesAudited': 3, 'independentAlgorithm': 'scalar endpoint sweep and monotone cardinality-first DP'}
