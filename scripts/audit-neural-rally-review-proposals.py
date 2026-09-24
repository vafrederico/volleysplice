#!/usr/bin/env python3
"""Independent standard-library audit of event-preserving review proposals.

No implementation, inference, or metric modules are imported. Interval unions
represent footage, never the identity-preserving event list.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name='number'):
    require(isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value), f'{name}: expected finite number')
    return float(value)


def close(expected, actual, name):
    actual = number(actual, name)
    require(math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-8),
            f'{name}: expected {expected!r}, got {actual!r}')


def intervals(values, duration, name='intervals'):
    require(isinstance(values, (list, tuple)), f'{name}: expected interval list')
    rows = []
    for value in values:
        if isinstance(value, Mapping):
            a, b = value['start'], value['end']
        else:
            require(isinstance(value, (tuple, list)) and len(value) == 2,
                    f'{name}: expected a boundary pair')
            a, b = value
        a, b = number(a, name), number(b, name)
        require(b > a, f'{name}: nonpositive interval')
        a, b = max(0., a), min(duration, b)
        if a < b:
            rows.append((a, b))
    return rows


def setop(groups, predicate):
    changes = {}
    for channel, rows in enumerate(groups):
        for a, b in rows:
            require(b > a, 'Nonpositive sweep span')
            changes.setdefault(a, [0] * len(groups))[channel] += 1
            changes.setdefault(b, [0] * len(groups))[channel] -= 1
    points, active, result = sorted(changes), [0] * len(groups), []
    for index, point in enumerate(points[:-1]):
        active = [x + y for x, y in zip(active, changes[point])]
        require(all(x >= 0 for x in active), 'Negative sweep occupancy')
        if predicate(tuple(x > 0 for x in active)):
            end = points[index + 1]
            if result and result[-1][1] == point:
                result[-1] = (result[-1][0], end)
            else:
                result.append((point, end))
    return result


def union(*groups):
    return setop(groups, any)


def intersection(left, right):
    return setop((left, right), all)


def difference(left, right):
    return setop((left, right), lambda x: x[0] and not x[1])


def seconds(rows):
    return math.fsum(b - a for a, b in union(rows))


def scope(record):
    duration = number(record['durationSeconds'], 'durationSeconds')
    require(duration > 0, 'Nonpositive recording duration')
    ignored = union(intervals(record.get('ignoredIntervals', []), duration))
    return duration, ignored, difference([(0., duration)], ignored)


def export(rows, record, pad, join=3.):
    duration, ignored, _ = scope(record)
    padded = union([(max(0., a - pad), min(duration, b + pad)) for a, b in rows])
    joined = []
    for a, b in padded:
        if joined and 0 < a - joined[-1][1] < join:
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    return difference(joined, ignored)


def same_intervals(expected, actual, duration, name):
    actual = intervals(actual, duration, name)
    require(len(expected) == len(actual), f'{name}: interval count differs')
    for index, (want, got) in enumerate(zip(expected, actual)):
        close(want[0], got[0], f'{name}[{index}].start')
        close(want[1], got[1], f'{name}[{index}].end')


def interval_iou(first, second):
    common = max(0., min(first[1], second[1]) - max(first[0], second[0]))
    total = max(first[1], second[1]) - min(first[0], second[0])
    return common / total if total else 0.


def ordered_matches(truth, predictions, allowed, quality):
    """Maximum cardinality, then maximum quality, chronological matching."""
    # Each cell contains the complete immutable path, avoiding imported DP code.
    rows = [[(0, 0., ()) for _ in range(len(predictions) + 1)]
            for _ in range(len(truth) + 1)]
    for i, actual in enumerate(truth, 1):
        for j, prediction in enumerate(predictions, 1):
            upper, left = rows[i - 1][j], rows[i][j - 1]
            best = left if left[:2] > upper[:2] else upper
            if allowed(actual, prediction):
                previous = rows[i - 1][j - 1]
                candidate = (previous[0] + 1, previous[1] + quality(actual, prediction),
                             previous[2] + ((i - 1, j - 1),))
                if candidate[:2] > best[:2]:
                    best = candidate
            rows[i][j] = best
    return list(rows[-1][-1][2])


def event_matches(truth, predictions, minimum_iou=.5):
    return ordered_matches(truth, predictions,
                           lambda a, b: interval_iou(a, b) >= minimum_iou,
                           interval_iou)


def start_matches(truth, predictions, tolerance):
    return ordered_matches(truth, predictions,
                           lambda a, b: abs(a[0] - b[0]) <= tolerance,
                           lambda a, b: -abs(a[0] - b[0]))


def contains_closed(rows, point):
    return any(a <= point <= b for a, b in rows)


def reconstruct_events(record, base, window):
    """Local perfect occupancy and boundary editing, not whole-event access.

    Gold endpoints are imported only if their actual timestamps lie within the
    closed edit permission. Context footage is deliberately absent from the API.
    """
    duration, ignored, _ = scope(record)
    original = intervals(base, duration, 'base events')
    require(original == sorted(original), 'Base events must be sorted')
    require(all(a[1] <= b[0] for a, b in zip(original, original[1:])),
            'Base event identities overlap')
    gold = intervals(record.get('rallies', []), duration, 'gold events')
    require(gold == sorted(gold), 'Gold events must be sorted')
    require(all(a[1] <= b[0] for a, b in zip(gold, gold[1:])),
            'Gold event identities overlap')
    permission = difference(intervals(window, duration, 'edit permission'), ignored)
    # Raw occupancy inside ignored time is retained as immutable base material;
    # canonical evaluation/export masks it later. Permission never includes it.
    before = union(original)
    truth = union(gold)
    edited = union(difference(before, permission), intersection(truth, permission))
    base_markers = {x for event in original for x in event
                    if not contains_closed(permission, x)}
    gold_markers = {x for event in gold for x in event
                    if contains_closed(permission, x)}
    markers = sorted(base_markers | gold_markers)
    result = []
    for start, end in edited:
        edges = [start, *(x for x in markers if start < x < end), end]
        result.extend(zip(edges, edges[1:]))
    # This invariant checks geometry; event boundaries are independently checked
    # through the marker reconstruction rather than merging the output list.
    require(not difference(difference(edited, permission), before)
            and not difference(difference(before, permission), edited),
            'Oracle edited outside permission')
    return {'events': result, 'permission': permission,
            'baseMarkersOutsidePermission': sorted(base_markers),
            'goldMarkersInsidePermission': sorted(gold_markers)}


def audit_edit_events(record, base, window, emitted_events):
    expected = reconstruct_events(record, base, window)
    same_intervals(expected['events'], emitted_events, record['durationSeconds'], 'edited events')
    return {'passed': True, 'eventsAudited': len(expected['events']),
            'editIntervalsAudited': len(expected['permission']),
            'baseMarkersPreservedOutsidePermission': len(expected['baseMarkersOutsidePermission']),
            'goldMarkersImportedInsidePermission': len(expected['goldMarkersInsidePermission']),
            'goldMarkersImportedOutsidePermission': 0}


def review_permission(record, proposals):
    duration, ignored, _ = scope(record)
    return difference([(max(0., a - 2.), min(duration, b + 2.))
                       for a, b in proposals], ignored)


def candidate_playback(record, proposals):
    return export(review_permission(record, proposals), record, 2.)


def audit_budget_queues(record, candidates, ranker, emitted):
    """Replay nested greedy union-cost selection without human labels."""
    require(ranker in ('chronological', 'evidence'), 'Unknown queue ranker')
    duration, _, valid = scope(record)
    require(len({row['id'] for row in candidates}) == len(candidates),
            'Duplicate proposal identity')
    ordered = sorted(candidates, key=lambda x: ((-x['priority'] if ranker == 'evidence' else x['start']),
                                               x['start'], x['end'], x['id']))
    require(isinstance(emitted, (list, tuple)) and len(emitted) == 4,
            'Queue must contain four budgets')
    selected, selected_ids = [], set()
    for fraction, output in zip((.05, .10, .20, .40), emitted):
        budget = fraction * seconds(valid)
        for item in ordered:
            if item['id'] in selected_ids:
                continue
            trial = [*selected, (item['start'], item['end'])]
            if seconds(candidate_playback(record, trial)) <= budget + 1e-9:
                selected.append((item['start'], item['end']))
                selected_ids.add(item['id'])
        # Recover insertion order separately from the set; the previous budget's
        # order remains fixed even if later high-ranked candidates become cheap.
        # IDs follow the corresponding selected geometry, which is deduplicated
        # by the proposal builder.
        pair_ids = {(x['start'], x['end']): x['id'] for x in candidates}
        ids = [pair_ids[x] for x in selected]
        window = review_permission(record, selected)
        playback = export(window, record, 2.)
        want = {'budgetFraction': fraction, 'budgetSeconds': budget,
                'reviewSeconds': seconds(playback), 'editSeconds': seconds(window),
                'reviewClips': len(playback), 'decisionRegions': len(window),
                'proposalsSelected': len(selected), 'proposalsAvailable': len(candidates)}
        for key, value in want.items():
            close(value, output.get(key), f'queue {fraction} {key}')
        require(output.get('selectedIds') == ids, f'queue {fraction}: selected IDs differ')
        same_intervals(window, output.get('editWindows'), duration, f'queue {fraction} editWindows')
        same_intervals(playback, output.get('playbackWindows'), duration, f'queue {fraction} playbackWindows')
        require(seconds(playback) <= budget + 1e-9, 'Playback exceeds review budget')
    return {'passed': True, 'budgetsAudited': 4, 'labelDataRead': False,
            'candidateCount': len(candidates), 'nestedSelections': True}


def audit_editor(record, base, queue, emitted):
    window = queue['editWindows']
    output = emitted['events']
    result = audit_edit_events(record, base, window, output)
    duration, _, _ = scope(record)
    permission = review_permission(record, []) if not window else difference(
        intervals(window, duration), scope(record)[1])
    gold = intervals(record['rallies'], duration)
    events = intervals(output, duration)
    original = intervals(base, duration)
    observed_starts, observed_ends = [], []
    for row, (a, b) in zip(output, events):
        observed_start = any(c == a for c, _ in (gold if contains_closed(permission, a) else original))
        observed_end = any(d == b for _, d in (gold if contains_closed(permission, b) else original))
        require(row.get('startObserved') is observed_start, 'startObserved differs')
        require(row.get('endObserved') is observed_end, 'endObserved differs')
        observed_starts.append(observed_start); observed_ends.append(observed_end)
    edges = {x for row in permission for x in row}
    censored_starts = sum(a in edges and any(c < a < d for c, d in gold) for a, _ in events)
    censored_ends = sum(b in edges and any(c < b < d for c, d in gold) for _, b in events)
    ids = [i for i, item in enumerate(gold) if intersection([item], permission)]
    unresolved = sum(bool(intersection([row], permission)) and not
                     (contains_closed(permission, row[0]) and contains_closed(permission, row[1]))
                     for row in gold)
    for key, expected in {'censoredStarts': censored_starts, 'censoredEnds': censored_ends,
                          'unobservedStarts': observed_starts.count(False),
                          'unobservedEnds': observed_ends.count(False),
                          'touchedRalliesWithUneditableBoundary': unresolved,
                          'reviewedTrueRallies': len(ids)}.items():
        close(expected, emitted.get(key), key)
    require(emitted.get('reviewedTrueRallyIds') == ids, 'Reviewed true event IDs differ')
    require([x.get('id') for x in output] == [f'e{i:05d}' for i in range(len(output))],
            'Output event IDs differ')
    return result


def optimal_bipartite(edges, left_count, right_count):
    """Independent min-cost maximum flow, distinct from Hungarian assignment."""
    source, sink = left_count + right_count, left_count + right_count + 1
    graph = [[] for _ in range(sink + 1)]

    def edge(a, b, cost):
        forward = [b, 1, cost, len(graph[b])]
        reverse = [a, 0, -cost, len(graph[a])]
        graph[a].append(forward); graph[b].append(reverse)

    for i in range(left_count):
        edge(source, i, 0.)
    for j in range(right_count):
        edge(left_count + j, sink, 0.)
    for (i, j), weight in sorted(edges.items()):
        edge(i, left_count + j, -weight)
    matched = 0
    total_cost = 0.
    while True:
        distance, previous = [math.inf] * len(graph), [None] * len(graph)
        distance[source] = 0.
        for _ in range(len(graph) - 1):
            changed = False
            for a, rows in enumerate(graph):
                if not math.isfinite(distance[a]):
                    continue
                for ordinal, row in enumerate(rows):
                    b, capacity, cost, _ = row
                    if capacity and distance[a] + cost < distance[b] - 1e-14:
                        distance[b] = distance[a] + cost
                        previous[b] = (a, ordinal)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            return matched, -total_cost
        current = sink
        seen = set()
        while current != source:
            require(current not in seen, 'Residual shortest path contains a cycle')
            seen.add(current)
            a, ordinal = previous[current]
            row = graph[a][ordinal]
            row[1] -= 1
            graph[current][row[3]][1] += 1
            current = a
        matched += 1
        total_cost += distance[sink]


COUNT_FIELDS = ('originalTrueRallies', 'ignoredTouchedTrueRallies', 'originalPredictedRallies',
                'entirelyMaskedPredictions', 'trueRallies', 'predictedRallies', 'matchedRallies',
                'completeMisses', 'mergedPredictions', 'splitTrueRallies',
                'mergedPredictionsMaterial', 'splitTrueRalliesMaterial',
                'unobservedPredictionStarts', 'unobservedPredictionEnds')
LOCAL_FIELDS = ('startLocalization', 'endLocalization', 'observedStartLocalization',
                'observedEndLocalization')
TOLERANCES = (.25, .5, 1., 2.)


def rates(matched, predicted, true):
    return {'true': true, 'predicted': predicted, 'matched': matched,
            'falsePositive': predicted - matched, 'falseNegative': true - matched,
            'precision': matched / predicted if predicted else (0. if true else None),
            'recall': matched / true if true else None,
            'f1': 2 * matched / (predicted + true) if true else None}


def check_tree(expected, actual, name):
    if isinstance(expected, Mapping):
        require(isinstance(actual, Mapping), f'{name}: expected mapping')
        for key, value in expected.items():
            require(key in actual, f'{name}: missing {key}')
            check_tree(value, actual[key], name + '.' + key)
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)) and len(expected) == len(actual),
                f'{name}: sequence length differs')
        for i, (left, right) in enumerate(zip(expected, actual)):
            check_tree(left, right, f'{name}[{i}]')
    elif isinstance(expected, Real) and not isinstance(expected, bool):
        close(expected, actual, name)
    else:
        require(expected == actual, f'{name}: differs')


def percentile(values, fraction):
    values = sorted(values)
    point = (len(values) - 1) * fraction
    lower = math.floor(point)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (point - lower) * (values[upper] - values[lower])


def error_summary(values):
    if not values:
        return {'count': 0, 'maeSeconds': None, 'medianAbsoluteSeconds': None,
                'p95AbsoluteSeconds': None, 'biasSeconds': None}
    absolute = [abs(x) for x in values]
    return {'count': len(values), 'maeSeconds': math.fsum(absolute) / len(values),
            'medianAbsoluteSeconds': percentile(absolute, .5),
            'p95AbsoluteSeconds': percentile(absolute, .95),
            'biasSeconds': math.fsum(values) / len(values)}


def summarize_identity(rows):
    output = {key: sum(x[key] for x in rows) for key in COUNT_FIELDS}
    event_rates = rates(output['matchedRallies'], output['predictedRallies'], output['trueRallies'])
    output.update(eventPrecision=event_rates['precision'], eventRecall=event_rates['recall'],
                  eventF1=event_rates['f1'], falsePositiveRallies=event_rates['falsePositive'],
                  falseNegativeRallies=event_rates['falseNegative'])
    for side in ('Start', 'End'):
        errors = [value for row in rows for value in row[f'matched{side}ErrorsSeconds']]
        output[f'matched{side}ErrorsSeconds'] = errors
        output[f'matched{side}Error'] = error_summary(errors)
    for name in LOCAL_FIELDS:
        output[name] = {}
        for tolerance in TOLERANCES:
            key = format(tolerance, 'g')
            counts = {field: sum(row[name][key][field] for row in rows)
                      for field in ('matched', 'predicted', 'true')}
            output[name][key] = rates(**counts)
    output['matchedEventBoundaries'] = {}
    for tolerance in TOLERANCES:
        key = format(tolerance, 'g')
        counts = {field: sum(row['matchedEventBoundaries'][key][field] for row in rows)
                  for field in ('startCorrect', 'endCorrect', 'bothCorrect', 'eligibleTrueRallies')}
        counts.update({side + 'Recall': counts[side + 'Correct'] / counts['eligibleTrueRallies']
                       if counts['eligibleTrueRallies'] else None for side in ('start', 'end', 'both')})
        output['matchedEventBoundaries'][key] = counts
    return output


def audit_identity_record(record, emitted):
    duration, ignored, _ = scope(record)

    def identities(values):
        result = []
        for index, value in enumerate(values):
            clipped = intervals([value], duration)
            if clipped:
                result.append({'index': index, 'span': clipped[0],
                               'startObserved': value.get('startObserved', True) if isinstance(value, Mapping) else True,
                               'endObserved': value.get('endObserved', True) if isinstance(value, Mapping) else True})
        return sorted(result, key=lambda x: (*x['span'], x['index']))

    original_truth = identities(record['rallies'])
    original_predictions = identities(record['predictions'])
    censored = [x for x in original_truth if intersection([x['span']], ignored)]
    truth = [x for x in original_truth if x not in censored]
    excluded = union(ignored, [x['span'] for x in censored])
    predicted = []
    for item in original_predictions:
        support = difference([item['span']], excluded)
        if support:
            predicted.append({**item, 'support': support})
    shared, ious, material = {}, {}, {}
    for i, actual in enumerate(truth):
        for j, prediction in enumerate(predicted):
            common = seconds(intersection([actual['span']], prediction['support']))
            true_seconds = actual['span'][1] - actual['span'][0]
            shared[i, j] = common
            ious[i, j] = common / (true_seconds + seconds(prediction['support']) - common)
            material[i, j] = common >= min(.5, .1 * true_seconds)
    eligible = {edge: score for edge, score in ious.items() if score >= .5}
    optimum_count, optimum_weight = optimal_bipartite(eligible, len(truth), len(predicted))
    by_truth = {x['index']: i for i, x in enumerate(truth)}
    by_prediction = {x['index']: j for j, x in enumerate(predicted)}
    matches = []
    for match in emitted.get('matches', []):
        require(match['truthIndex'] in by_truth and match['predictionIndex'] in by_prediction,
                'Event match references masked/missing identity')
        edge = by_truth[match['truthIndex']], by_prediction[match['predictionIndex']]
        require(edge in eligible, 'Event match fails IoU threshold')
        close(eligible[edge], match['iou'], 'matched IoU')
        matches.append(edge)
    require(len({i for i, _ in matches}) == len(matches)
            and len({j for _, j in matches}) == len(matches), 'Event matching is not one-to-one')
    require(len(matches) == optimum_count, 'Event matching is not maximum cardinality')
    close(optimum_weight, math.fsum(ious[edge] for edge in matches), 'Maximum total matched IoU')
    available_starts = [not any(a <= x['span'][0] < b for a, b in excluded) for x in predicted]
    available_ends = [not any(a < x['span'][1] <= b for a, b in excluded) for x in predicted]
    start_errors = [predicted[j]['span'][0] - truth[i]['span'][0] for i, j in matches if available_starts[j]]
    end_errors = [predicted[j]['span'][1] - truth[i]['span'][1] for i, j in matches if available_ends[j]]
    row = {'originalTrueRallies': len(original_truth), 'ignoredTouchedTrueRallies': len(censored),
           'originalPredictedRallies': len(original_predictions),
           'entirelyMaskedPredictions': len(original_predictions) - len(predicted),
           'trueRallies': len(truth), 'predictedRallies': len(predicted), 'matchedRallies': len(matches),
           'completeMisses': sum(not any(shared[i, j] > 0 for j in range(len(predicted))) for i in range(len(truth))),
           'mergedPredictions': sum(sum(shared[i, j] > 0 for i in range(len(truth))) > 1 for j in range(len(predicted))),
           'splitTrueRallies': sum(sum(shared[i, j] > 0 for j in range(len(predicted))) > 1 for i in range(len(truth))),
           'mergedPredictionsMaterial': sum(sum(material[i, j] for i in range(len(truth))) > 1 for j in range(len(predicted))),
           'splitTrueRalliesMaterial': sum(sum(material[i, j] for j in range(len(predicted))) > 1 for i in range(len(truth))),
           'unobservedPredictionStarts': sum(not x['startObserved'] for x in predicted),
           'unobservedPredictionEnds': sum(not x['endObserved'] for x in predicted),
           'matchedStartErrorsSeconds': start_errors, 'matchedEndErrorsSeconds': end_errors,
           'matchedEventBoundaries': {}, **{name: {} for name in LOCAL_FIELDS}}
    for tolerance in TOLERANCES:
        key = format(tolerance, 'g')
        for name, coordinate, available, observed in (
                ('startLocalization', 0, available_starts, False),
                ('endLocalization', 1, available_ends, False),
                ('observedStartLocalization', 0, available_starts, True),
                ('observedEndLocalization', 1, available_ends, True)):
            actual = [x['span'][coordinate] for x in truth]
            predicted_times = [x['span'][coordinate] for j, x in enumerate(predicted)
                               if available[j] and (not observed or x['startObserved' if coordinate == 0 else 'endObserved'])]
            allowed = {(i, j): 0. for i, a in enumerate(actual) for j, b in enumerate(predicted_times)
                       if abs(a - b) <= tolerance}
            count, _ = optimal_bipartite(allowed, len(actual), len(predicted_times))
            row[name][key] = rates(count, len(predicted_times), len(actual))
        row['matchedEventBoundaries'][key] = {
            'startCorrect': sum(abs(x) <= tolerance for x in start_errors),
            'endCorrect': sum(abs(x) <= tolerance for x in end_errors),
            'bothCorrect': sum(available_starts[j] and available_ends[j]
                               and abs(predicted[j]['span'][0] - truth[i]['span'][0]) <= tolerance
                               and abs(predicted[j]['span'][1] - truth[i]['span'][1]) <= tolerance
                               for i, j in matches), 'eligibleTrueRallies': len(truth)}
    expected = summarize_identity([row])
    check_tree(expected, emitted, 'record ' + record['id'])
    return expected


def audit_identity_metrics(records, emitted):
    require(len(records) == len(emitted['recordings']), 'Event metric recording count differs')
    by_id = {x['id']: x for x in emitted['recordings']}
    require(len(by_id) == len(records) and set(by_id) == {x['id'] for x in records},
            'Event metric recording identities differ')
    rows = []
    for record in records:
        row = audit_identity_record(record, by_id[record['id']])
        rows.append({**row, 'sourceGroup': record['sourceGroup']})
    check_tree(summarize_identity(rows), emitted['pooled'], 'pooled identities')
    groups = sorted({x['sourceGroup'] for x in records})
    require(set(emitted['sourceGroups']) == set(groups), 'Event metric source groups differ')
    for group in groups:
        check_tree(summarize_identity([x for x in rows if x['sourceGroup'] == group]),
                   emitted['sourceGroups'][group], 'source group ' + group)
    return {'passed': True, 'recordingsAudited': len(records), 'sourceGroupsAudited': len(groups),
            'eventMatchingAudit': 'independent min-cost maximum flow',
            'localizationTolerancesAudited': len(TOLERANCES), 'observedBoundaryFlagsAudited': True}
