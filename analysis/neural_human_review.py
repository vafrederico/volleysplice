"""Label-blind review plans and explicitly hypothetical perfect-human outcomes."""
from __future__ import annotations

from bisect import bisect_left
from math import fsum, floor

from . import neural_production_combinations as iv

COMBO_POLICIES = ('guarded_trim', 'suppression_zero', 'suppression_half',
                  'suppression_any', 'bidirectional_half', 'bidirectional_any')
SOLO_POLICIES = ('positive_uncertain', 'uncertain_narrow', 'uncertain_medium',
                 'uncertain_wide', 'all_positive', 'all_candidates')
THRESHOLDS = {'positive_uncertain': (.8, None), 'uncertain_narrow': (.7, .3),
              'uncertain_medium': (.8, .2), 'uncertain_wide': (.9, .1)}


def valid_time(record):
    return iv.difference([(0., record['durationSeconds'])], record.get('ignoredIntervals', ()))


def visible(values, record):
    return iv.intersection(values, valid_time(record))


def under_supported(component, support, mode):
    overlap = iv.intersection([component], support)
    if mode == 'zero':
        return not overlap
    if mode == 'half':
        return 2 * iv.duration(overlap) < component.end-component.start
    if mode == 'any':
        return bool(iv.difference([component], support))
    raise ValueError('Unknown support condition')


def combination_plan(record, base, challenger, policy, previous=(), v2=()):
    if policy not in COMBO_POLICIES:
        raise ValueError('Unknown combination review policy')
    p, n = visible(base, record), visible(challenger, record)
    if policy == 'guarded_trim':
        trimmed = iv.combine(base, challenger, previous, v2, record['durationSeconds'], 'guarded_trim')
        trigger = visible(iv.difference(base, trimmed), record)
        candidates = p
    else:
        mode = policy.rsplit('_', 1)[1]
        support_n = iv.dilate(n, 2, record['durationSeconds'])
        trigger = iv.union(*( [x] for x in p if under_supported(x, support_n, mode)))
        candidates = p
        if policy.startswith('bidirectional'):
            support_p = iv.dilate(p, 2, record['durationSeconds'])
            trigger = iv.union(trigger, *( [x] for x in n if under_supported(x, support_p, mode)))
            candidates = iv.union(p, n)
    return [{'id': f'c{i:05d}', 'start': x.start, 'end': x.end,
             'kind': 'positive' if iv.intersection([x], p) else 'negative',
             'flagged': bool(iv.intersection([x], trigger)), 'score': None}
            for i, x in enumerate(candidates)]


def individual_plan(record, base, times, live, policy):
    if policy not in SOLO_POLICIES or len(times) != len(live):
        raise ValueError('Invalid individual review input')
    if any(times[i] >= times[i+1] for i in range(len(times)-1)):
        raise ValueError('Score times must increase strictly')
    if any(not 0 <= float(x) <= 1 for x in live):
        raise ValueError('Invalid live probabilities')
    p = visible(base, record)
    negative = iv.difference(valid_time(record), p)
    candidates = [(x, 'positive') for x in p]
    for gap in negative:
        start = gap.start
        while start < gap.end:
            edge = (floor(start/5)+1)*5.
            end = min(gap.end, edge)
            if end <= start:
                raise ValueError('Grid did not advance')
            candidates.append((iv.Interval(start, end), 'negative'))
            start = end
    output = []
    for i, (x, kind) in enumerate(sorted(candidates, key=lambda row:(row[0].start,row[0].end,row[1]))):
        scores = live[bisect_left(times, x.start):bisect_left(times, x.end)]
        score = None if not len(scores) else (fsum(float(v) for v in scores)/len(scores)
                                              if kind == 'positive' else max(float(v) for v in scores))
        if policy == 'all_candidates':
            flag = True
        elif policy == 'all_positive':
            flag = kind == 'positive'
        else:
            positive_threshold, negative_threshold = THRESHOLDS[policy]
            flag = ((score is None or score < positive_threshold) if kind == 'positive' else
                    (negative_threshold is not None and (score is None or score > negative_threshold)))
        output.append({'id': f'c{i:05d}', 'start': x.start, 'end': x.end,
                       'kind': kind, 'flagged': flag, 'score': score})
    return output


def touched_rallies(record, spans):
    return [i for i, r in enumerate(record['rallies']) if iv.intersection(visible([r],record), spans)]


def losses(record, exported):
    complete = partial = 0
    for row in record['rallies']:
        core = visible([row], record)
        seconds = iv.duration(core)
        if seconds <= 1e-9:
            continue
        retained = iv.duration(iv.intersection(core, exported))
        if retained <= 1e-9:
            complete += 1
        elif seconds-retained > 1e-9:
            partial += 1
    return complete, partial


def human_outcomes(record, base, candidates):
    """Gold is first used here, after construction of every review flag.

    A correct binary decision keeps a whole candidate containing any real play.
    Boundary editing corrects only its final decision export, without re-padding.
    """
    selected = [r for r in candidates if r['flagged']]
    decision_raw = iv.union(selected)
    truth = visible(record['rallies'], record)
    kept = [r for r in selected if iv.intersection([r], truth)]
    dropped = [r for r in selected if not iv.intersection([r], truth)]
    mixed = [r for r in kept if iv.difference([r], truth)]
    binary_raw = iv.union(iv.difference(visible(base,record), decision_raw), kept)
    binary_exports, boundary_exports, padding = {}, {}, []
    reviewed = touched_rallies(record, decision_raw)
    for pad in iv.PADS:
        m = iv.export(base, record, pad)
        w = iv.export(decision_raw, record, pad)
        playback = iv.export(w, record, 2)
        hp = iv.export(record['rallies'], record, pad)
        binary = iv.export(binary_raw, record, pad)
        boundary = iv.union(iv.difference(m,w),iv.intersection(hp,w))
        bc,bp = losses(record,binary)
        ec,ep = losses(record,boundary)
        binary_exports[str(pad)] = iv.serial(binary)
        boundary_exports[str(pad)] = iv.serial(boundary)
        in_playback = touched_rallies(record, playback)
        padding.append({'paddingSecondsBeforeAndAfter':pad,'joinGapSeconds':3,
                        'decisionExportIntervals':iv.serial(w),'playbackIntervals':iv.serial(playback),
                        'rawDecisionSeconds':iv.duration(decision_raw),
                        'reviewSeconds':iv.duration(playback),'decisionSeconds':iv.duration(w),
                        'reviewClips':len(playback),'flaggedCandidates':len(selected),
                        'flaggedPositiveCandidates':sum(r['kind']=='positive' for r in selected),
                        'flaggedNegativeCandidates':sum(r['kind']=='negative' for r in selected),
                        'reviewedTrueRallies':len(reviewed),'playbackTrueRallies':len(in_playback),
                        'binaryKeptCandidates':len(kept),'binaryDroppedCandidates':len(dropped),
                        'mixedCandidates':len(mixed),'keptCandidateIds':[r['id'] for r in kept],
                        'droppedCandidateIds':[r['id'] for r in dropped],
                        'reviewedTrueRallyIds':reviewed,'playbackTrueRallyIds':in_playback,
                        'completeRallyLossesBinary':bc,'partialRallyLossesBinary':bp,
                        'completeRallyLossesBoundary':ec,'partialRallyLossesBoundary':ep})
    return {'candidates':candidates,'selected':[r['id'] for r in selected],
            'binaryRaw':iv.serial(binary_raw),'binaryExports':binary_exports,
            'boundaryExports':boundary_exports,'padding':padding}
