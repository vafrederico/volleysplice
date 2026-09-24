#!/usr/bin/env python3
"""Independent audits for typed rally boundaries with a separate fixed export.

The existing frozen split-study auditor supplies independently implemented
interval sweeps only. No boundary adviser or metric implementation is imported.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('boundary_scalar_intervals', REPO/'scripts/audit_neural_split_advisor.py')
scalar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scalar)
MATCH_SPEC = importlib.util.spec_from_file_location('boundary_independent_flow', REPO/'scripts/audit-neural-rally-review-proposals.py')
matching = importlib.util.module_from_spec(MATCH_SPEC)
MATCH_SPEC.loader.exec_module(matching)
require = scalar.require
check_tree = scalar.check_tree
close = scalar.close
pairs = scalar.pairs
union = scalar.union
intersection = scalar.intersection
difference = scalar.difference
seconds = scalar.seconds


def audit_fixed_export(records, actual_duration_metrics, fixed_exports):
    """Assert every final export equals production, independently of event cores."""
    require(set(fixed_exports) == {r['id'] for r in records}, 'Fixed export recording scope differs')
    for record in records:
        values = fixed_exports[record['id']]
        require(set(values) == {'0', '1', '2', '3'}, 'Fixed export padding scope differs')
        for pad in (0, 1, 2, 3):
            expected = scalar.export(record['productionEvents'], record, pad)
            require(pairs(values[str(pad)]) == expected, 'Export changed from original production')
    result = scalar.audit_duration_records(records, actual_duration_metrics, fixed_exports)
    return {**result, 'eventTimelineIndependentOfExport': True, 'fixedProductionExportsVerified': True}


def raw_coverage(record, predictions):
    """Raw timeline changes, all gold core and complete-miss identity guards.

Ignored-touched gold is separately counted; complete-event scoring excludes
those censored identities, while physical lost-core duration excludes ignored
seconds only. Both denominators are explicit to prevent export/core confusion.
"""
    duration = record['durationSeconds']
    ignored = pairs(record.get('ignoredIntervals', []), duration)
    gold = pairs(record['rallies'], duration)
    baseline = difference(pairs(record['productionEvents'], duration), ignored)
    result = difference(pairs(predictions, duration), ignored)
    core = difference(gold, ignored)
    lost, added = difference(baseline, result), difference(result, baseline)
    eligible = [(i, g) for i, g in enumerate(gold) if not intersection([g], ignored)]
    before_misses = {i for i, g in eligible if not intersection([g], baseline)}
    after_misses = {i for i, g in eligible if not intersection([g], result)}
    core_seconds = seconds(core)
    before_seconds, after_seconds = seconds(intersection(baseline, core)), seconds(intersection(result, core))
    return {'coreHumanSeconds': core_seconds, 'baselineRawCoreCoveredSeconds': before_seconds,
            'resultRawCoreCoveredSeconds': after_seconds,
            'baselineRawCoreRecall': before_seconds/core_seconds if core_seconds else None,
            'resultRawCoreRecall': after_seconds/core_seconds if core_seconds else None,
            'rawSelectedSecondsLostFromBaseline': seconds(lost),
            'rawSelectedSecondsAddedToBaseline': seconds(added),
            'rawCoreSecondsLostFromBaseline': seconds(intersection(lost, core)),
            'rawCoreSecondsAddedToBaseline': seconds(intersection(added, core)),
            'baselineCompleteMisses': len(before_misses), 'resultCompleteMisses': len(after_misses),
            'additionalCompleteMisses': len(after_misses - before_misses),
            'recoveredCompleteMisses': len(before_misses - after_misses),
            'additionalCompleteMissIndexes': sorted(after_misses - before_misses),
            'recoveredCompleteMissIndexes': sorted(before_misses - after_misses),
            'ignoredTouchedGoldIdentities': len(gold) - len(eligible)}


def feasible_peak(times, scores, head, anchor, low, high, component):
    """Scalar head lookup over feasible actual samples; no interpolation."""
    choices = [(float(row[head]), abs(float(time) - anchor), float(time))
               for time, row in zip(times, scores)
               if anchor - 3. <= time <= anchor + 3. and low <= time <= high
               and component[0] <= time < component[1] and row[head] >= .25]
    if not choices:
        return anchor, False
    value, _, time = min(choices, key=lambda x: (-x[0], x[1], x[2]))
    return time, True


def reconstruct_core_candidates(record, parents, neural, times, scores, policy):
    """Independent interval/observed-marker reconstruction from frozen policy."""
    require(policy in ('first_start', 'typed_starts', 'separate_ends', 'head_refined'), 'Unknown boundary policy')
    duration = record['durationSeconds']
    ignored = pairs(record.get('ignoredIntervals', []), duration)
    t = [float(x) for x in times]
    a = [[float(x) for x in row] for row in scores]
    require(len(t) == len(a) and all(len(row) == 4 for row in a), 'Invalid aligned score matrix')
    require(all(x < y for x, y in zip(t, t[1:])), 'Invalid actual timeline')
    require(all(math.isfinite(x) for row in a for x in row), 'Invalid score values')
    output = []
    for parent in parents:
        components = difference(pairs([parent], duration), ignored)
        for component_index, component in enumerate(components):
            associated = []
            for raw in neural:
                start, end = max(component[0], raw['start']), min(component[1], raw['end'])
                if end - start >= .5:
                    associated.append({'raw': raw, 'start': start, 'end': end,
                                       'startObserved': start == raw['start'], 'endObserved': end == raw['end']})
            if policy == 'first_start':
                associated = associated[:1]
            previous_end = component[0]
            for i, current in enumerate(associated):
                start, end = current['start'], current['end']
                observed_start, observed_end = current['startObserved'], current['endObserved']
                next_start = associated[i + 1]['start'] if i + 1 < len(associated) else component[1]
                if policy in ('first_start', 'typed_starts'):
                    end = next_start if policy == 'typed_starts' else component[1]
                    observed_end = end == parent['end'] and parent.get('endObserved', True)
                elif policy == 'head_refined':
                    start, from_head = feasible_peak(t, a, 1, current['start'],
                                                     max(component[0], previous_end), current['end'] - .5, component)
                    observed_start = True if from_head else current['startObserved']
                    end, from_head = feasible_peak(t, a, 2, current['end'], start + .5,
                                                   min(component[1], next_start), component)
                    observed_end = True if from_head else current['endObserved']
                previous_end = end
                output.append({'parentId': parent['id'], 'componentIndex': component_index,
                               'type': 'initial_start' if i == 0 else 'additional_start',
                               'start': start, 'end': end, 'startObserved': observed_start,
                               'endObserved': observed_end, 'neuralSourceId': current['raw']['id']})
    return output


TYPES = ('initial_start', 'additional_start')
TOLERANCES = (.5, 1., 2.)


def typed_prepared(record):
    ignored = pairs(record.get('ignoredIntervals', []), record['durationSeconds'])
    gold = pairs(record['rallies'], record['durationSeconds'])
    censored = [(i, row) for i, row in enumerate(gold) if intersection([row], ignored)]
    eligible = [(i, row) for i, row in enumerate(gold) if not intersection([row], ignored)]
    masks = union(ignored, [row for _, row in censored])
    parents = [(i, row, difference(pairs([row]), masks)) for i, row in enumerate(record['productionEvents'])]
    return gold, eligible, censored, masks, parents


def typed_targets(record):
    _, gold, _, _, parents = typed_prepared(record)
    result = []
    for pi, parent, support in parents:
        for ci, component in enumerate(support):
            covered = [(i, row) for i, row in gold
                       if scalar._overlap(row, component) >= min(.5, .1*(row[1] - row[0]))]
            for position, (i, row) in enumerate(covered):
                result.append({'id': f"{parent['id']}::c{ci}::gold{i}", 'parentId': parent['id'], 'parentIndex': pi,
                               'componentIndex': ci, 'componentStart': component[0], 'componentEnd': component[1],
                               'truthIndex': i, 'start': row[0], 'end': row[1],
                               'type': TYPES[0] if position == 0 else TYPES[1],
                               'startAccessible': component[0] <= row[0] < component[1],
                               'endAccessible': component[0] < row[1] <= component[1]})
    return result


def normalized_candidates(record, candidates):
    _, _, _, masks, parents = typed_prepared(record)
    parent_map = {parent['id']: (parent, support) for _, parent, support in parents}
    result = []
    for i, row in enumerate(candidates):
        parent, support = parent_map[row['parentId']]
        start, end = row['start'], row['end']
        require(parent['start'] <= start < end <= parent['end'], 'Boundary candidate outside parent')
        require(row['type'] in TYPES, 'Unknown boundary type')
        available_start = not any(a <= start < b for a, b in masks)
        available_end = not any(a < end <= b for a, b in masks)
        component = next((ci for ci, (a, b) in enumerate(support) if a <= start < b), None)
        result.append({'index': i, 'id': row.get('id', f'candidate-{i}'), 'parentId': row['parentId'],
                       'start': start, 'end': end, 'type': row['type'], 'componentIndex': component,
                       'startObserved': row.get('startObserved', True) and available_start,
                       'endObserved': row.get('endObserved', True) and available_end,
                       'startAvailable': available_start, 'endAvailable': available_end})
    return result


def optimal_matches(targets, candidates, tolerance, *, typed, material=False):
    edges = {}
    for i, target in enumerate(targets):
        for j, candidate in enumerate(candidates):
            error = abs(candidate['start'] - target['start'])
            eligible = (candidate['startObserved'] and candidate['parentId'] == target['parentId']
                        and candidate['componentIndex'] == target['componentIndex']
                        and (not typed or candidate['type'] == target['type']) and error <= tolerance)
            if material:
                overlap = max(0., min(target['end'], candidate['end']) - max(target['start'], candidate['start']))
                eligible = eligible and overlap >= min(.5, .1*(target['end'] - target['start']))
            if eligible:
                edges[i, j] = max(0., 1. - error/tolerance)
    count, quality = matching.optimal_bipartite(edges, len(targets), len(candidates))
    return count, quality, edges


def check_pairs(targets, candidates, pairs_emitted, tolerance, *, typed, material=False):
    count, quality, edges = optimal_matches(targets, candidates, tolerance, typed=typed, material=material)
    used_t, used_p, total_quality = set(), set(), 0.
    by_index = {p['index']: j for j, p in enumerate(candidates)}
    correspondence = []
    for pair in pairs_emitted:
        ti, pi = pair['targetIndex'], pair['proposalIndex']
        require(pi in by_index, 'Pair names missing candidate')
        pj = by_index[pi]
        require((ti, pj) in edges and ti not in used_t and pi not in used_p, 'Ineligible/duplicate boundary pair')
        used_t.add(ti); used_p.add(pi); total_quality += edges[ti, pj]
        target, proposal = targets[ti], candidates[pj]
        close(proposal['start'] - target['start'], pair['startErrorSeconds'], 'Boundary start error')
        if 'endErrorSeconds' in pair:
            close(proposal['end'] - target['end'], pair['endErrorSeconds'], 'Boundary end error')
            require(pair['endObserved'] == proposal['endObserved'], 'Pair end observed differs')
        correspondence.append((target, proposal))
    require(len(correspondence) == count, 'Boundary match cardinality differs')
    close(quality, total_quality, 'Boundary matching total quality')
    return correspondence


def _typed_record(record, actual):
    targets = typed_targets(record)
    candidates = normalized_candidates(record, record.get('boundaryProposals', []))
    check_tree(targets, actual['targets'], 'typed targets')
    check_tree(candidates, actual['proposals'], 'typed candidates')
    gold, eligible, censored, _, _ = typed_prepared(record)
    coverage = raw_coverage(record, record.get('predictions', record['productionEvents']))
    coverage['retainedCompleteMisses'] = coverage['resultCompleteMisses'] - coverage['additionalCompleteMisses']
    coverage = {k: v for k, v in coverage.items() if k not in ('recoveredCompleteMissIndexes', 'ignoredTouchedGoldIdentities')}
    observed = [row for row in candidates if row['startObserved']]
    counts = {'id': record['id'], 'sourceGroup': record['sourceGroup'],
              'originalTrueRallies': len(gold), 'eligibleTrueRallies': len(eligible), 'ignoredTouchedTrueRallies': len(censored),
              'targetCount': len(targets), 'uniqueTargetTrueRallies': len({t['truthIndex'] for t in targets}),
              'inaccessibleTargetStarts': sum(not t['startAccessible'] for t in targets),
              'inaccessibleTargetEnds': sum(not t['endAccessible'] for t in targets),
              'candidateCount': len(candidates), 'observedCandidateStarts': len(observed),
              'unobservedCandidateStarts': len(candidates) - len(observed),
              'observedCandidateEnds': sum(p['endObserved'] for p in candidates),
              'unobservedCandidateEnds': sum(not p['endObserved'] for p in candidates), **coverage}
    check_tree(counts, actual, 'typed raw counts')
    result = {**counts, 'typedStartLocalization': {}, 'untypedStartLocalization': {},
              'typeDiagnostics': {}, 'samePairBoundaries': {}, 'byType': {kind: {} for kind in TYPES}}
    for tolerance in TOLERANCES:
        key = format(tolerance, 'g')
        typed_pairs = check_pairs(targets, candidates, actual['samePairBoundaries'][key]['pairs'], tolerance, typed=True, material=True)
        untyped_pair_rows = actual['typeDiagnostics'][key].get('pairs')
        require(untyped_pair_rows is not None, 'Untyped pairing evidence required for independently verified wrong-type counts')
        untyped_pairs = check_pairs(targets, candidates, untyped_pair_rows, tolerance, typed=False, material=True)
        for field, matched in [('typedStartLocalization', len(typed_pairs)), ('untypedStartLocalization', len(untyped_pairs))]:
            result[field][key] = scalar._count_rates(matched, len(observed), len(targets))
            check_tree(result[field][key], actual[field][key], field + key)
        wrong = [(t, p) for t, p in untyped_pairs if t['type'] != p['type']]
        result['typeDiagnostics'][key] = {'wrongType': len(wrong),
            'initialProposedForAdditional': sum(p['type'] == TYPES[0] for _, p in wrong),
            'additionalProposedForInitial': sum(p['type'] == TYPES[1] for _, p in wrong), 'untypedMatches': len(untyped_pairs)}
        check_tree(result['typeDiagnostics'][key], actual['typeDiagnostics'][key], 'type diagnostics' + key)
        observed_ends = sum(p['endObserved'] for _, p in typed_pairs)
        correct = sum(p['endObserved'] and abs(p['end'] - t['end']) <= tolerance for t, p in typed_pairs)
        result['samePairBoundaries'][key] = {**scalar._count_rates(correct, len(observed), len(targets)),
            'startMatched': len(typed_pairs), 'startMatchedWithObservedEnd': observed_ends,
            'endCorrectGivenMatchedStart': correct, 'conditionalEndAccuracy': correct/observed_ends if observed_ends else None}
        check_tree(result['samePairBoundaries'][key], actual['samePairBoundaries'][key], 'same pair boundaries' + key)
        for kind in TYPES:
            result['byType'][kind][key] = scalar._count_rates(sum(t['type'] == kind for t, _ in typed_pairs),
                sum(p['type'] == kind for p in observed), sum(t['type'] == kind for t in targets))
            check_tree(result['byType'][kind][key], actual['byType'][kind][key], 'by type' + kind + key)
    return result


def audit_typed_metrics(records, actual):
    require(len(records) == len(actual['recordings']), 'Typed recording count differs')
    rows = [_typed_record(record, result) for record, result in zip(records, actual['recordings'])]
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))
               and k not in ('baselineRawCoreRecall', 'resultRawCoreRecall')]
    def pool(selected):
        output = {key: sum(row[key] for row in selected) for key in numeric}
        for name in ('baseline', 'result'):
            output[name + 'RawCoreRecall'] = (output[name + 'RawCoreCoveredSeconds']/output['coreHumanSeconds']
                                               if output['coreHumanSeconds'] else None)
        for field in ('typedStartLocalization', 'untypedStartLocalization', 'samePairBoundaries'):
            output[field] = {}
            for tolerance in TOLERANCES:
                key = format(tolerance, 'g')
                rates = scalar._count_rates(**{k: sum(row[field][key][k] for row in selected) for k in ('matched', 'predicted', 'true')})
                if field == 'samePairBoundaries':
                    rates.update({k: sum(row[field][key][k] for row in selected) for k in
                                  ('startMatched', 'startMatchedWithObservedEnd', 'endCorrectGivenMatchedStart')})
                    rates['conditionalEndAccuracy'] = (rates['endCorrectGivenMatchedStart']/rates['startMatchedWithObservedEnd']
                                                       if rates['startMatchedWithObservedEnd'] else None)
                output[field][key] = rates
        output['typeDiagnostics'] = {format(t, 'g'): {key: sum(r['typeDiagnostics'][format(t, 'g')][key] for r in selected)
            for key in ('wrongType', 'initialProposedForAdditional', 'additionalProposedForInitial', 'untypedMatches')} for t in TOLERANCES}
        output['byType'] = {kind: {format(t, 'g'): scalar._count_rates(**{key: sum(r['byType'][kind][format(t, 'g')][key] for r in selected)
            for key in ('matched', 'predicted', 'true')}) for t in TOLERANCES} for kind in TYPES}
        return output
    check_tree(pool(rows), actual['pooled'], 'typed pooled')
    groups = sorted({r['sourceGroup'] for r in rows})
    require(set(groups) == set(actual['sourceGroups']), 'Typed group scope differs')
    for group in groups:
        check_tree(pool([r for r in rows if r['sourceGroup'] == group]), actual['sourceGroups'][group], 'typed group' + group)
    check_tree({'types': list(TYPES), 'tolerancesSeconds': list(TOLERANCES), 'primaryToleranceSeconds': 1.},
               actual['metricContract'], 'typed contract')
    return {'passed': True, 'recordingsAudited': len(rows), 'sourceGroupsAudited': len(groups),
            'tolerancesAudited': len(TOLERANCES), 'matchingOracle': 'independent min-cost maximum flow',
            'physicalCoreLossAndIdentityMissesSeparate': True}


def reconstruct_plan(record, parents, neural, times, scores, policy):
    """Scalar policy replay with exact provenance and explicit component fallbacks."""
    require(policy in ('first_start', 'typed_starts', 'separate_ends', 'head_refined'), 'Unknown boundary policy')
    parents = sorted(parents, key=lambda p: (p['start'], p['end'], p['id']))
    neural = sorted(neural, key=lambda p: (p['start'], p['end'], p['id']))
    duration = record['durationSeconds']; ignored = pairs(record.get('ignoredIntervals', []), duration)
    components = difference([(0., duration)], ignored)
    ignored_edges = {v for interval in ignored for v in interval}
    t, a = [float(x) for x in times], [[float(v) for v in row] for row in scores]
    events, candidates, decisions = [], [], []
    def boundary(parent, point, kind):
        observed = point == parent[kind] and point not in ignored_edges and parent.get(kind + 'Observed', True)
        return observed, 'production-' + kind if point == parent[kind] else 'ignored-clipped'
    def fallback(parent, component=None, cid=None):
        start, end = (parent['start'], parent['end']) if component is None else component
        so, ss = boundary(parent, start, 'start'); eo, es = boundary(parent, end, 'end')
        return {**parent, 'id': parent['id'] if component is None else 'fallback:' + cid,
                'parentId': parent['id'], 'parentStart': parent['start'], 'parentEnd': parent['end'],
                'componentId': cid, 'componentStart': start, 'componentEnd': end, 'start': start, 'end': end,
                'type': 'initial_start', 'candidateId': None, 'fallback': True,
                'startObserved': so, 'endObserved': eo, 'startSource': ss, 'endSource': es}
    for parent in parents:
        pieces = []
        for global_index, global_component in enumerate(components):
            comp = (max(parent['start'], global_component[0]), min(parent['end'], global_component[1]))
            if comp[0] >= comp[1]:
                continue
            associated = [(i, row) for i, row in enumerate(neural) if scalar._overlap((row['start'], row['end']), comp) >= .5]
            pieces.append((f"{parent['id']}::valid:{global_index}", comp, global_component, associated))
        supported = any(associations for _, _, _, associations in pieces)
        decision = {'parentId': parent['id'], 'supported': supported, 'fallback': not supported, 'components': []}
        if not supported:
            events.append(fallback(parent))
        for cid, comp, global_component, associations in pieces:
            decision['components'].append({'componentId': cid, 'start': comp[0], 'end': comp[1],
                                           'associatedNeuralIds': [row['id'] for _, row in associations],
                                           'fallback': not bool(associations)})
            if not supported:
                continue
            if not associations:
                events.append(fallback(parent, comp, cid)); continue
            selected = associations[:1] if policy == 'first_start' else associations
            produced = []
            for position, (ni, raw) in enumerate(selected):
                start, end = max(comp[0], raw['start']), min(comp[1], raw['end'])
                values = [row[0] for tick, row in zip(t, a) if start <= tick < end]
                mean = math.fsum(values)/len(values) if values else 0.
                marker = lambda point, original, kind: ((False, 'ignored-clipped') if point in ignored_edges else
                    (True, 'compact-' + kind) if point == original else (False, 'parent-clipped'))
                so, ss = marker(start, raw['start'], 'start'); eo, es = marker(end, raw['end'], 'end')
                identifier = f'candidate:{cid}:neural:{ni}'
                row = {'id': identifier, 'candidateId': identifier, 'parentId': parent['id'],
                    'parentStart': parent['start'], 'parentEnd': parent['end'], 'componentId': cid,
                    'componentStart': comp[0], 'componentEnd': comp[1],
                    'type': TYPES[0] if position == 0 else TYPES[1], 'start': start, 'end': end,
                    'startObserved': so, 'endObserved': eo, 'startSource': ss, 'endSource': es,
                    'sourceNeuralId': raw['id'], 'sourceNeuralIndex': ni,
                    'originalNeuralStart': raw['start'], 'originalNeuralEnd': raw['end'],
                    'originalClippedStart': start, 'originalClippedEnd': end,
                    'priority': mean, 'meanLive': mean, 'fallback': False, 'startHeadScore': None, 'endHeadScore': None}
                next_start = max(comp[0], selected[position + 1][1]['start']) if position + 1 < len(selected) else comp[1]
                if policy in ('first_start', 'typed_starts'):
                    if position + 1 < len(selected):
                        row.update(end=next_start, endObserved=False, endSource='partition-only')
                    else:
                        observed, source = boundary(parent, comp[1], 'end')
                        row.update(end=comp[1], endObserved=observed, endSource=source)
                elif policy == 'head_refined':
                    lower = produced[-1]['end'] if produced else comp[0]
                    point, from_head = feasible_peak(t, a, 1, start, lower, end - .5, global_component)
                    if from_head:
                        row.update(start=point, startObserved=True, startSource='head-serve', startHeadScore=a[t.index(point)][1])
                    point, from_head = feasible_peak(t, a, 2, end, row['start'] + .5, next_start, global_component)
                    if from_head:
                        row.update(end=point, endObserved=True, endSource='head-end', endHeadScore=a[t.index(point)][2])
                produced.append(row)
            candidates.extend(produced)
            events.extend({**parent, **row} for row in produced)
        decisions.append(decision)
    sorter = lambda row: (row['start'], row['end'], row['id'])
    events.sort(key=sorter); candidates.sort(key=sorter)
    proposals = []
    for row in candidates:
        actions = []
        if row['type'] == TYPES[1] or abs(row['start'] - row['componentStart']) >= .25:
            actions.append((row['type'], 'start'))
        if policy in ('separate_ends', 'head_refined') and row['endObserved'] and abs(row['end'] - row['componentEnd']) >= .25:
            actions.append(('end', 'end'))
        for kind, endpoint in actions:
            proposals.append({'id': row['id'] + ':' + kind, 'candidateId': row['id'],
                'parentId': row['parentId'], 'componentId': row['componentId'],
                'parentStart': row['parentStart'], 'parentEnd': row['parentEnd'],
                'componentStart': row['componentStart'], 'componentEnd': row['componentEnd'],
                'start': row['start'], 'end': row['end'], 'time': row[endpoint], 'kind': kind,
                'boundary': endpoint, 'type': row['type'], 'observed': row[endpoint + 'Observed'],
                'source': row[endpoint + 'Source'], 'priority': row['priority']})
    return {'policy': policy, 'events': events, 'eventCandidates': candidates,
            'proposals': proposals, 'parentDecisions': decisions}


def audit_plan(record, parents, neural, times, scores, policy, actual):
    expected = reconstruct_plan(record, parents, neural, times, scores, policy)
    check_tree(expected, actual, 'boundary plan')
    for name in ('events', 'eventCandidates', 'proposals'):
        require(len({row['id'] for row in actual[name]}) == len(actual[name]), 'Duplicate ' + name + ' ID')
    ranges = pairs(actual['events'])
    require(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])), 'Overlapping event cores')
    parents_by_id = {row['id']: row for row in parents}
    for row in actual['events']:
        parent = parents_by_id[row['parentId']]
        require(parent['start'] <= row['start'] < row['end'] <= parent['end'], 'Event escaped original parent')
    return {'passed': True, 'parentsAudited': len(parents), 'candidatesAudited': len(actual['eventCandidates']),
            'eventsAudited': len(actual['events']), 'proposalsAudited': len(actual['proposals']),
            'scalarReconstruction': True, 'labelDataRead': False}


def audit_proposal_human(record, plan, selected_parent_ids, actual):
    parents = record['productionEvents']; candidates = plan['eventCandidates']
    selected = set(selected_parent_ids)
    require(selected <= {p['id'] for p in parents}, 'Unknown selected parent')
    correct_ends = plan['policy'] in ('separate_ends', 'head_refined')
    reviewed_indexes = [i for i, candidate in enumerate(candidates) if candidate['parentId'] in selected]
    reviewed = [candidates[i] for i in reviewed_indexes]
    normalized = normalized_candidates(record, reviewed)
    targets = typed_targets(record)
    accepted = actual['acceptedCandidates']
    pairs_emitted = []
    for row in accepted:
        require(row['candidateIndex'] in reviewed_indexes, 'Accepted unreviewed candidate')
        ci = row['candidateIndex']; candidate = candidates[ci]; ti = row['targetIndex']; target = targets[ti]
        check_tree({'candidateIndex': ci, 'candidateId': candidate['id'], 'parentId': candidate['parentId'],
                    'proposedType': candidate['type'], 'targetType': target['type'], 'targetTruthIndex': target['truthIndex'],
                    'targetIndex': ti, 'target': target, 'proposedStart': candidate['start'], 'proposedEnd': candidate['end'],
                    'typeCorrected': candidate['type'] != target['type']}, row, 'Human acceptance')
        pairs_emitted.append({'targetIndex': ti, 'proposalIndex': reviewed_indexes.index(ci),
                              'startErrorSeconds': candidate['start'] - target['start']})
    check_pairs(targets, normalized, pairs_emitted, 1., typed=False, material=True)
    accepted_indexes = {row['candidateIndex'] for row in accepted}
    rejected = [{'candidateIndex': i, 'candidateId': candidates[i]['id'], 'parentId': candidates[i]['parentId']}
                for i in reviewed_indexes if i not in accepted_indexes]
    check_tree(rejected, actual['rejectedCandidates'], 'Rejected candidates')
    output, edited_parent_ids = [], []
    for parent in parents:
        accepted_here = [row for row in accepted if row['parentId'] == parent['id']]
        if not accepted_here:
            output.append(dict(parent)); continue
        edited_parent_ids.append(parent['id'])
        components = difference(pairs([parent]), pairs(record.get('ignoredIntervals', [])))
        for ci, component in enumerate(components):
            local = sorted([row for row in accepted_here if component[0] <= row['proposedStart'] < component[1]],
                           key=lambda row: (max(component[0], row['target']['start']), row['candidateIndex']))
            rows = []
            if not any(row['targetType'] == TYPES[0] for row in local):
                rows.append({**parent, 'start': component[0], 'end': component[1],
                    'startObserved': component[0] == parent['start'] and parent.get('startObserved', True),
                    'endObserved': component[1] == parent['end'] and parent.get('endObserved', True),
                    'startKind': 'baseline_fallback', 'endKind': 'inherited', 'humanConfirmed': False})
            for row in local:
                target = row['target']; start = max(component[0], target['start'])
                end = min(component[1], target['end']) if correct_ends else component[1]
                rows.append({**parent, 'start': start, 'end': end, 'startObserved': target['start'] == start,
                    'endObserved': target['end'] == end if correct_ends else component[1] == parent['end'] and parent.get('endObserved', True),
                    'startKind': 'human_confirmed' if target['start'] == start else 'permission_edge',
                    'endKind': ('human_confirmed' if target['end'] == end else 'permission_edge') if correct_ends else 'inherited',
                    'humanConfirmed': True, 'candidateId': row['candidateId'],
                    'targetTruthIndex': row['targetTruthIndex'], 'type': row['targetType']})
            rows.sort(key=lambda row: (row['start'], row['end']))
            for left, right in zip(rows, rows[1:]):
                if left['end'] > right['start']:
                    left.update(end=right['start'], endObserved=False, endKind='synthetic_partition')
            for index, row in enumerate(rows):
                require(row['start'] < row['end'], 'Human created empty event')
                row.update(id=f"{parent['id']}::review-c{ci}-{index}", parentId=parent['id'],
                           parentStart=parent['start'], parentEnd=parent['end'], componentId=f"{parent['id']}::c{ci}")
                output.append(row)
    output.sort(key=lambda row: (row['start'], row['end'], row['id']))
    check_tree(output, actual['events'], 'Human event timeline')
    check_tree({'reviewedCandidates': len(reviewed), 'acceptedCount': len(accepted), 'rejectedCount': len(rejected),
                'correctedTypeCount': sum(row['typeCorrected'] for row in accepted),
                'editedParentIds': edited_parent_ids, 'editedParents': len(edited_parent_ids),
                'selectedParentIds': sorted(selected), 'correctEnds': correct_ends}, actual, 'Human counts')
    return {'passed': True, 'reviewedCandidatesAudited': len(reviewed), 'acceptedCandidatesAudited': len(accepted),
            'eventsAudited': len(output), 'parentFallbacksVerified': True,
            'unproposedGoldEndpointsImported': False, 'rawCoverage': raw_coverage(record, output)}


def audit_full_parent_human(record, selected_parent_ids, actual):
    selected = set(selected_parent_ids)
    require(selected <= {p['id'] for p in record['productionEvents']}, 'Unknown selected parent')
    output = []
    for parent in record['productionEvents']:
        if parent['id'] not in selected:
            output.append(dict(parent)); continue
        components = difference(pairs([parent]), pairs(record.get('ignoredIntervals', [])))
        for ci, component in enumerate(components):
            for gi, gold in enumerate(record['rallies']):
                start, end = max(component[0], gold['start']), min(component[1], gold['end'])
                if start < end:
                    output.append({**parent, 'id': f"full::{parent['id']}::gold:{gi}::component:{ci}",
                        'parentId': parent['id'], 'componentId': f"{parent['id']}::component:{ci}",
                        'start': start, 'end': end, 'startObserved': start == gold['start'], 'endObserved': end == gold['end'],
                        'startSource': 'human' if start == gold['start'] else 'permission-clipped',
                        'endSource': 'human' if end == gold['end'] else 'permission-clipped', 'humanGoldIndex': gi})
    output.sort(key=lambda row: (row['start'], row['end'], row['id']))
    check_tree(output, actual['events'], 'Full-parent human events')
    coverage = raw_coverage(record, output)
    require(coverage['rawCoreSecondsLostFromBaseline'] == 0 and coverage['additionalCompleteMisses'] == 0,
            'Full-parent human lost true core or rally identity')
    return {'passed': True, 'selectedParentsAudited': len(selected), 'eventsAudited': len(output),
            'localPermissionPreserved': True, 'unproposedRalliesMayBeRecovered': True, 'rawCoverage': coverage}


def audit_workload(record, plan, queue, edited, actual):
    selected = set(queue['selectedParentIds'])
    candidates = [row for row in plan['eventCandidates'] if row['parentId'] in selected]
    before = difference(pairs(record['productionEvents']), pairs(record.get('ignoredIntervals', [])))
    after = difference(pairs(edited['events']), pairs(record.get('ignoredIntervals', [])))
    touched = lambda windows: sum(bool(intersection(difference([gold], pairs(record.get('ignoredIntervals', []))), pairs(windows)))
                                  for gold in pairs(record['rallies']))
    expected = {key: queue[key] for key in ('reviewSeconds', 'budgetSeconds', 'unusedBudgetSeconds',
                'reviewJobs', 'jobsAvailable', 'reviewClips', 'editRegions', 'boundaryFlagsReviewed')}
    require(queue['selectedProposalIds'] == queue['selectedSplitIds'], 'Proposal aliases differ')
    require(queue['boundaryFlagsReviewed'] == len(queue['selectedProposalIds']), 'Boundary flag count differs')
    expected.update(candidatesReviewed=len(candidates), reviewedTrueRallies=touched(queue['editWindows']),
                    playbackTrueRallies=touched(queue['playbackWindows']),
                    eventTimelineRemovedSeconds=seconds(difference(before, after)),
                    eventTimelineAddedSeconds=seconds(difference(after, before)))
    check_tree(expected, actual, 'Boundary review workload')
    return {'passed': True, 'workloadFieldsAudited': len(expected), 'candidateAndTrueRallyWorkloadIndependent': True}
