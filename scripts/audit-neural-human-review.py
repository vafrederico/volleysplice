#!/usr/bin/env python3
"""Independent stdlib reconstruction of perfect-human review experiments.

This module deliberately imports no study, inference, or interval helpers.  A
candidate plan is checked independently of labels; a separate oracle audit then
uses labels to verify decisions, exports, queue geometry, and workload counts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real


PADS = (0, 1, 2, 3)
COMBO = ('guarded_trim', 'suppression_zero', 'suppression_half', 'suppression_any',
         'bidirectional_half', 'bidirectional_any')
SOLO = ('positive_uncertain', 'uncertain_narrow', 'uncertain_medium',
        'uncertain_wide', 'all_positive', 'all_candidates')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(isinstance(value, Real) and not isinstance(value, bool)
             and math.isfinite(value), f'{name}: expected finite number')
    return float(value)


def _close(a, b, name):
    b = _number(b, name)
    _require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8),
             f'{name}: independent {a!r}, emitted {b!r}')


def _read(values, duration, name='intervals'):
    _require(isinstance(values, (list, tuple)), f'{name}: expected interval list')
    rows = []
    for value in values:
        if isinstance(value, Mapping):
            _require('start' in value and 'end' in value, f'{name}: missing boundaries')
            a, b = value['start'], value['end']
        else:
            _require(isinstance(value, (list, tuple)) and len(value) == 2,
                     f'{name}: expected two boundaries')
            a, b = value
        a, b = _number(a, name), _number(b, name)
        _require(b > a, f'{name}: nonpositive interval')
        a, b = max(0., a), min(duration, b)
        if a < b:
            rows.append((a, b))
    return rows


def _setop(groups, predicate):
    """Classify atomic spans through an endpoint occupancy sweep."""
    changes = {}
    for channel, group in enumerate(groups):
        for a, b in group:
            changes.setdefault(a, [0] * len(groups))[channel] += 1
            changes.setdefault(b, [0] * len(groups))[channel] -= 1
    points, active, output = sorted(changes), [0] * len(groups), []
    for i, point in enumerate(points[:-1]):
        active = [count + delta for count, delta in zip(active, changes[point])]
        _require(all(count >= 0 for count in active), 'Invalid sweep activity')
        if predicate(tuple(count > 0 for count in active)):
            end = points[i + 1]
            if output and output[-1][1] == point:
                output[-1] = (output[-1][0], end)
            else:
                output.append((point, end))
    return output


def _union(*groups):
    return _setop(groups, any)


def _difference(a, b):
    return _setop((a, b), lambda x: x[0] and not x[1])


def _intersection(a, b):
    return _setop((a, b), all)


def _seconds(a):
    return math.fsum(end - start for start, end in _union(a))


def _dilate(rows, duration, pad):
    return _union([(max(0., a - pad), min(duration, b + pad)) for a, b in rows])


def _export(rows, ignored, duration, pad):
    joined = []
    for a, b in _dilate(rows, duration, pad):
        if joined and 0 < a - joined[-1][1] < 3:
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    return _difference(joined, ignored)


def _same_intervals(actual, expected, duration, name):
    actual = _read(actual, duration, name)
    _require(len(actual) == len(expected), f'{name}: interval count differs')
    for i, (a, b) in enumerate(zip(actual, expected)):
        _close(b[0], a[0], f'{name}[{i}].start')
        _close(b[1], a[1], f'{name}[{i}].end')


def _scope(record):
    duration = _number(record.get('durationSeconds'), 'durationSeconds')
    _require(duration > 0, 'Nonpositive recording duration')
    ignored = _union(_read(record.get('ignoredIntervals', ()), duration, 'ignoredIntervals'))
    valid = _difference([(0., duration)], ignored)
    return duration, ignored, valid


def _component_trigger(base, other, duration, mode):
    support, flagged = _dilate(other, duration, 2), []
    for component in base:
        overlap = _intersection([component], support)
        if mode == 'zero':
            select = not overlap
        elif mode == 'half':
            select = 2 * _seconds(overlap) < component[1] - component[0]
        else:
            _require(mode == 'any', 'Unknown trigger mode')
            select = bool(_difference([component], support))
        if select:
            flagged.append(component)
    return flagged


def _guarded_trigger(base, other, previous, v2, duration):
    entries = []
    for source, values in enumerate((previous, v2)):
        for a, b in values:
            entries.append((max(0., a - 2), min(duration, b + 2), source, (a, b)))
    groups = []
    for a, b, source, raw in sorted(entries):
        if groups and (a <= groups[-1]['end'] or a - groups[-1]['end'] < .5):
            group = groups[-1]
            group['end'] = max(group['end'], b)
            group['sources'].add(source)
            group['raw'].append(raw)
        else:
            groups.append({'end': b, 'sources': {source}, 'raw': [raw]})
    eligible = _union(*(g['raw'] for g in groups if len(g['sources']) == 1))
    return _difference(_intersection(eligible, base), _dilate(other, duration, 2))


def _candidate_score(component, ticks, probabilities, kind):
    values = [value for tick, value in zip(ticks, probabilities)
              if component[0] <= tick < component[1]]
    if not values:
        return None
    return math.fsum(values) / len(values) if kind == 'positive' else max(values)


def _compare_plan(expected, emitted):
    _require(isinstance(emitted, (list, tuple)) and len(emitted) == len(expected),
             'Candidate plan length differs')
    for index, (left, right) in enumerate(zip(expected, emitted)):
        _require(isinstance(right, Mapping), 'Candidate must be an object')
        for key in ('id', 'flagged', 'kind'):
            _require(right.get(key) == left[key], f'Candidate {index}: {key} differs')
        _require(isinstance(right['flagged'], bool), 'Candidate flag must be bool')
        for key in ('start', 'end'):
            _close(left[key], right.get(key), f'Candidate {index}: {key}')
        _require('score' in right, 'Candidate score missing')
        if left['score'] is None:
            _require(right['score'] is None, f'Candidate {index}: expected null score')
        else:
            _close(left['score'], right['score'], f'Candidate {index}: score')


def audit_candidate_plan(record, base, challenger, policy, emitted,
                         previous=None, v2=None, times=None, live=None):
    """Independently reconstruct label-blind inventory, scores, and flags.

    The implementation never reads record['rallies'].  Combo scores are null;
    solo positives use mean live probability and negatives use maximum live
    probability. Missing ticks conservatively flag uncertain candidates.
    """
    duration, ignored, valid = _scope(record)
    raw_base = _read(base, duration, 'base')
    base = _difference(raw_base, ignored)
    expected = []
    if policy in COMBO:
        raw_other = _read(challenger, duration, 'challenger')
        other = _difference(raw_other, ignored)
        bidirectional = policy.startswith('bidirectional_')
        components = _union(base, other) if bidirectional else base
        if policy == 'guarded_trim':
            _require(previous is not None and v2 is not None, 'Guarded plan needs both source tags')
            # Replay the existing guarded raw-core combiner before exclusions;
            # final eligible candidate geometry still excludes ignored spans.
            trigger = _difference(_guarded_trigger(raw_base, raw_other,
                                  _read(previous, duration), _read(v2, duration),
                                  duration), ignored)
        else:
            mode = policy.rsplit('_', 1)[1]
            trigger = _component_trigger(base, other, duration, mode)
            if bidirectional:
                trigger = _union(trigger, _component_trigger(other, base, duration, mode))
        for a, b in components:
            expected.append({'start': a, 'end': b, 'kind': 'positive' if
                             _intersection([(a, b)], base) else 'negative',
                             'flagged': bool(_intersection([(a, b)], trigger)), 'score': None})
    else:
        _require(policy in SOLO, 'Unknown candidate policy')
        _require(isinstance(times, Sequence) and isinstance(live, Sequence)
                 and len(times) == len(live), 'Matching tick/probability sequences required')
        ticks = [_number(value, 'tick') for value in times]
        probabilities = [_number(value, 'probability') for value in live]
        _require(all(0 <= value <= 1 for value in probabilities), 'Probability outside [0,1]')
        _require(all(a < b for a, b in zip(ticks, ticks[1:])), 'Tick times must increase')
        _require(all(0 <= value <= duration for value in ticks), 'Tick outside recording')
        negatives = _difference(valid, base)
        inventory = [(a, b, 'positive') for a, b in base]
        for cell in range(math.ceil(duration / 5)):
            for a, b in _intersection(negatives, [(5. * cell, min(duration, 5. * (cell + 1)))]):
                inventory.append((a, b, 'negative'))
        for a, b, kind in sorted(inventory):
            score = _candidate_score((a, b), ticks, probabilities, kind)
            if policy == 'all_candidates':
                flagged = True
            elif policy == 'all_positive':
                flagged = kind == 'positive'
            elif policy == 'positive_uncertain':
                flagged = kind == 'positive' and (score is None or score < .8)
            else:
                pos, neg = {'uncertain_narrow': (.7, .3), 'uncertain_medium': (.8, .2),
                            'uncertain_wide': (.9, .1)}[policy]
                flagged = score is None or (score < pos if kind == 'positive' else score > neg)
            expected.append({'start': a, 'end': b, 'kind': kind, 'score': score, 'flagged': flagged})
    for ordinal, candidate in enumerate(expected):
        candidate['id'] = f'c{ordinal:05d}'
    _compare_plan(expected, emitted)
    return {'passed': True, 'policy': policy, 'candidatesAudited': len(expected),
            'flagsAudited': len(expected), 'labelDataRead': False}


def _gold_ids(gold, region):
    return [index for index, rally in enumerate(gold) if _intersection(rally, region)]


def _losses(gold, export):
    complete, partial = 0, 0
    for rally in gold:
        if _seconds(rally) <= 1e-9:
            continue
        retained = _seconds(_intersection(rally, export))
        missing = _seconds(_difference(rally, export))
        complete += retained <= 1e-9
        partial += retained > 1e-9 and missing > 1e-9
    return complete, partial


def audit_record(record, base, result):
    """Verify a plan's perfect-human decisions and all four padding cases.

    Pair with ``audit_candidate_plan`` to also verify label-blind candidate
    construction. Gold rally identity is its original zero-based ordinal.
    """
    duration, ignored, valid = _scope(record)
    raw_base = _read(base, duration, 'base')
    base = _difference(raw_base, ignored)
    raw_gold = _read(record.get('rallies', ()), duration, 'rallies')
    gold = [_difference([rally], ignored) for rally in raw_gold]
    truth = _union(*gold)
    candidates = result.get('candidates')
    _require(isinstance(candidates, (list, tuple)), 'Missing candidates')
    selected, kept, dropped, selected_rows, kept_rows = [], [], [], [], []
    mixed, positives, negatives = 0, 0, 0
    previous_end = -math.inf
    for ordinal, candidate in enumerate(candidates):
        _require(candidate.get('id') == f'c{ordinal:05d}', 'Invalid candidate ordinal ID')
        _require(isinstance(candidate.get('flagged'), bool), 'Missing Boolean flag')
        row = _read([candidate], duration, 'candidate')
        _require(len(row) == 1 and row[0][0] >= previous_end, 'Overlapping/unsorted candidate inventory')
        _require(not _difference(row, valid), 'Candidate crosses ignored time')
        previous_end = row[0][1]
        kind = 'positive' if _intersection(row, base) else 'negative'
        _require(candidate.get('kind') == kind, 'Incorrect positive/negative candidate kind')
        if not candidate['flagged']:
            continue
        selected.append(candidate['id'])
        selected_rows.extend(row)
        positives += kind == 'positive'
        negatives += kind == 'negative'
        if _intersection(row, truth):
            kept.append(candidate['id'])
            kept_rows.extend(row)
            mixed += bool(_difference(row, truth))
        else:
            dropped.append(candidate['id'])
    _require(result.get('selected') == selected, 'Selected candidate IDs differ')
    selected_union = _union(selected_rows)
    binary = _union(_difference(base, selected_union), kept_rows)
    _same_intervals(result.get('binaryRaw'), binary, duration, 'binaryRaw')
    for key in ('binaryExports', 'boundaryExports'):
        _require(isinstance(result.get(key), Mapping) and set(result[key]) == {str(p) for p in PADS},
                 f'{key}: require exactly four padding cases')
    padding = result.get('padding')
    _require(isinstance(padding, (list, tuple)) and len(padding) == 4, 'Require four workload rows')
    rows = {}
    for row in padding:
        pad = row.get('paddingSecondsBeforeAndAfter')
        _require(pad in PADS and pad not in rows, 'Invalid/duplicate workload padding')
        rows[pad] = row
    reviewed_ids = _gold_ids(gold, selected_union)
    for pad in PADS:
        baseline = _export(raw_base, ignored, duration, pad)
        wanted = _export(raw_gold, ignored, duration, pad)
        window = _export(selected_union, ignored, duration, pad)
        playback = _export(window, ignored, duration, 2)
        binary_export = _export(binary, ignored, duration, pad)
        boundary = _union(_difference(baseline, window), _intersection(wanted, window))
        _same_intervals(result['binaryExports'][str(pad)], binary_export, duration, f'binaryExports[{pad}]')
        _same_intervals(result['boundaryExports'][str(pad)], boundary, duration, f'boundaryExports[{pad}]')
        row = rows[pad]
        if 'joinGapSeconds' in row:
            _close(3., row['joinGapSeconds'], f'pad={pad} joinGapSeconds')
        if 'rawDecisionSeconds' in row:
            _close(_seconds(selected_union), row['rawDecisionSeconds'],
                   f'pad={pad} rawDecisionSeconds')
        _same_intervals(row.get('decisionExportIntervals'), window, duration, f'decisionExportIntervals[{pad}]')
        _same_intervals(row.get('playbackIntervals'), playback, duration, f'playbackIntervals[{pad}]')
        playback_ids = _gold_ids(gold, playback)
        binary_complete, binary_partial = _losses(gold, binary_export)
        boundary_complete, boundary_partial = _losses(gold, boundary)
        expected = {'reviewSeconds': _seconds(playback), 'decisionSeconds': _seconds(window),
                    'reviewClips': len(playback), 'flaggedCandidates': len(selected),
                    'flaggedPositiveCandidates': positives, 'flaggedNegativeCandidates': negatives,
                    'reviewedTrueRallies': len(reviewed_ids), 'playbackTrueRallies': len(playback_ids),
                    'binaryKeptCandidates': len(kept), 'binaryDroppedCandidates': len(dropped),
                    'mixedCandidates': mixed, 'completeRallyLossesBinary': binary_complete,
                    'partialRallyLossesBinary': binary_partial,
                    'completeRallyLossesBoundary': boundary_complete,
                    'partialRallyLossesBoundary': boundary_partial}
        for key, value in expected.items():
            _close(value, row.get(key), f'pad={pad} {key}')
        for key, value in {'keptCandidateIds': kept, 'droppedCandidateIds': dropped,
                           'reviewedTrueRallyIds': reviewed_ids, 'playbackTrueRallyIds': playback_ids}.items():
            _require(row.get(key) == value, f'pad={pad} {key} differs')
    return {'passed': True, 'candidatesAudited': len(candidates),
            'selectedCandidatesAudited': len(selected), 'paddingCasesAudited': len(PADS),
            'exportRowsAudited': 2 * len(PADS), 'workloadRowsAudited': len(PADS)}
