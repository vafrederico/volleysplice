"""Fixed, label-blind production/neural interval combinations and accounting.

No fitting, threshold selection, model loading or filesystem access occurs here.
Core combinations precede the canonical export transformation. Explicit export
overrides are reserved for app-fidelity measurements and hypothetical review.
"""
from __future__ import annotations

from math import isfinite

from .crop_evaluation import pad_and_merge_intervals, subtract_intervals
from .schema import Interval

PADS = (0, 1, 2, 3)
FAMILIES = ('union', 'intersection', 'tolerant_intersection',
            'guarded_trim', 'guarded_component_rejection')
SUM_FIELDS = ('paddedModelExportSeconds', 'paddedHumanExportSeconds',
              'evaluableVideoSeconds', 'correctlyRemovedSeconds',
              'incorrectlyRemovedSeconds', 'incorrectExportSeconds',
              'missedCoreSeconds', 'coreHumanSeconds',
              'paddedIntersectionSeconds', 'coreIntersectionSeconds')


def intervals(values):
    result = []
    for value in values:
        if isinstance(value, Interval):
            start, end = value.start, value.end
        elif isinstance(value, dict):
            start, end = value['start'], value['end']
        else:
            start, end = value
        if not (isfinite(start) and isfinite(end) and end > start):
            raise ValueError('Invalid interval')
        result.append(Interval(float(start), float(end)))
    return tuple(result)


def serial(values):
    return [[r.start, r.end] for r in values]


def union(*groups):
    return subtract_intervals((r for group in groups for r in intervals(group)), ())


def difference(left, right):
    return subtract_intervals(intervals(left), intervals(right))


def intersection(left, right):
    a, b = union(left), union(right)
    output, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start, end = max(a[i].start, b[j].start), min(a[i].end, b[j].end)
        if start < end:
            output.append(Interval(start, end))
        if a[i].end <= b[j].end:
            i += 1
        else:
            j += 1
    return union(output)


def duration(values):
    return sum(r.end-r.start for r in union(values))


def dilate(values, seconds, video_duration):
    # Agreement support is distinct from the final export's short-gap join.
    return pad_and_merge_intervals(intervals(values), video_duration, seconds, 0)


def agreement_components(previous, v2, video_duration):
    """Source-tagged 2-second agreement padding, strict <0.5-second join.

    Protection covers original raw intervals throughout a connected component,
    including nonoverlapping tails. Expanded agreement support is never exported.
    """
    expanded = []
    for source, values in enumerate((previous, v2)):
        for raw in intervals(values):
            expanded.append((max(0., raw.start-2), min(video_duration, raw.end+2), source, raw))
    groups = []
    for start, end, source, raw in sorted(expanded, key=lambda r: (r[0], r[1], r[2])):
        if end <= start:
            continue
        gap = start-groups[-1]['end'] if groups else float('inf')
        if groups and (gap <= 0 or gap < .5):
            group = groups[-1]
            group['end'] = max(group['end'], end)
            group['sources'].add(source)
            group['raw'].append(raw)
        else:
            groups.append({'start': start, 'end': end, 'sources': {source}, 'raw': [raw]})
    return [{'both': len(g['sources']) == 2, 'raw': union(g['raw'])} for g in groups]


def combine(production, neural, previous, v2, video_duration, family):
    p, n = union(production), union(neural)
    if family == 'union':
        return union(p, n)
    if family == 'intersection':
        return intersection(p, n)
    supported = dilate(n, 2, video_duration)
    if family == 'tolerant_intersection':
        return intersection(p, supported)
    if family not in ('guarded_trim', 'guarded_component_rejection'):
        raise ValueError('Unknown combination family')
    groups = agreement_components(previous, v2, video_duration)
    eligible = [intersection(g['raw'], p) for g in groups if not g['both']]
    if family == 'guarded_trim':
        cut = difference(union(*eligible), supported)
    else:
        cut = union(*(component for component in eligible
                      if not intersection(component, supported)))
    return difference(p, cut)


def majority(a, b, c):
    return union(intersection(a, b), intersection(a, c), intersection(b, c))


def export(values, record, pad):
    return difference(pad_and_merge_intervals(intervals(values), record['durationSeconds'], pad, 3),
                      record.get('ignoredIntervals', ()))


def finish_counts(row):
    p = row['paddedIntersectionSeconds']/row['paddedModelExportSeconds'] if row['paddedModelExportSeconds'] else 0.
    r = row['coreIntersectionSeconds']/row['coreHumanSeconds'] if row['coreHumanSeconds'] else 0.
    row.update(P_pad=p, R_core=r, F1_padP_coreR=2*p*r/(p+r) if p+r else 0.,
               exportDurationDifferenceSeconds=row['paddedModelExportSeconds']-row['paddedHumanExportSeconds'])
    if abs(row['paddedModelExportSeconds']+row['correctlyRemovedSeconds']+
           row['incorrectlyRemovedSeconds']-row['evaluableVideoSeconds']) > 1e-7:
        raise ValueError('Export/TN/FN accounting does not partition evaluable time')
    return row


def duration_rows(records, export_overrides=None):
    """Pool time numerators/denominators; never average per-recording scores."""
    records = list(records)
    if len({r['id'] for r in records}) != len(records) or not records:
        raise ValueError('Empty or duplicate recording scope')
    if export_overrides is not None:
        if set(export_overrides) != {r['id'] for r in records} or any(
                set(v) != {str(p) for p in PADS} for v in export_overrides.values()):
            raise ValueError('Export overrides must cover exactly all recordings and padding cases')
    rows = []
    for pad in PADS:
        pooled = {key: 0. for key in SUM_FIELDS}
        per_record = []
        for rec in records:
            ignored = rec.get('ignoredIntervals', ())
            valid = difference([(0, rec['durationSeconds'])], ignored)
            h = intersection(difference(rec['rallies'], ignored), valid)
            hp = export(rec['rallies'], rec, pad)
            m = (export(rec['predictions'], rec, pad) if export_overrides is None else
                 intersection(export_overrides[rec['id']][str(pad)], valid))
            values = finish_counts({
                'id': rec['id'], 'sourceGroup': rec['sourceGroup'],
                'paddedModelExportSeconds': duration(m), 'paddedHumanExportSeconds': duration(hp),
                'evaluableVideoSeconds': duration(valid),
                'correctlyRemovedSeconds': duration(difference(valid, union(m, hp))),
                'incorrectlyRemovedSeconds': duration(difference(hp, m)),
                'incorrectExportSeconds': duration(difference(m, hp)),
                'missedCoreSeconds': duration(difference(h, m)), 'coreHumanSeconds': duration(h),
                'paddedIntersectionSeconds': duration(intersection(m, hp)),
                'coreIntersectionSeconds': duration(intersection(m, h)),
            })
            for key in SUM_FIELDS:
                pooled[key] += values[key]
            per_record.append(values)
        pooled = finish_counts(pooled)
        rows.append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3,
                     **pooled, 'perRecording': per_record})
    return rows


def review_diagnostic(records, neural_predictions, mode, base_exports=None):
    """Label-blind disagreement queue; unchanged automatic output.

    A separate oracle corrects only disputed *export seconds*, using padded
    human export as truth. It is an optimistic bound, never an automatic result
    or measured human performance. No padding/rejoining follows oracle editing.
    """
    if mode not in ('suppression_review', 'bidirectional_review'):
        raise ValueError('Unknown review mode')
    oracle, queue = {r['id']: {} for r in records}, []
    for pad in PADS:
        counts = {k: 0. for k in ('reviewSeconds', 'disputedSeconds', 'unwantedExportFlaggedSeconds',
                                  'wantedExportFlaggedSeconds', 'missedHumanExportFlaggedSeconds',
                                  'missedCoreFlaggedSeconds')}
        count_clips, per_record = 0, []
        for rec in records:
            valid = difference([(0, rec['durationSeconds'])], rec.get('ignoredIntervals', ()))
            m = export(rec['predictions'], rec, pad) if base_exports is None else intersection(base_exports[rec['id']][str(pad)], valid)
            n = export(neural_predictions[rec['id']], rec, pad)
            proposed_remove = difference(m, n)
            disputed = proposed_remove if mode == 'suppression_review' else union(proposed_remove, difference(n, m))
            # Human playback context is workload only; it is not oracle edit permission.
            view = difference(pad_and_merge_intervals(disputed, rec['durationSeconds'], 2, 3),
                              rec.get('ignoredIntervals', ()))
            hp = export(rec['rallies'], rec, pad)
            h = intersection(rec['rallies'], valid)
            fixed = union(difference(m, disputed), intersection(hp, disputed))
            oracle[rec['id']][str(pad)] = serial(fixed)
            values = {
                'reviewSeconds': duration(view), 'disputedSeconds': duration(disputed),
                'unwantedExportFlaggedSeconds': duration(intersection(disputed, difference(m, hp))),
                'wantedExportFlaggedSeconds': duration(intersection(proposed_remove, hp)),
                'missedHumanExportFlaggedSeconds': duration(intersection(disputed, difference(hp, m))),
                'missedCoreFlaggedSeconds': duration(intersection(disputed, difference(h, m))),
            }
            for key in counts:
                counts[key] += values[key]
            count_clips += len(view)
            per_record.append({'id': rec['id'], **values, 'reviewClips': len(view),
                               'disputedIntervals': serial(disputed), 'reviewIntervals': serial(view)})
        queue.append({'paddingSecondsBeforeAndAfter': pad, **counts, 'reviewClips': count_clips,
                      'perRecording': per_record})
    return {'mode': mode, 'queue': queue, 'oracleExports': oracle,
            'oracleDurationMetrics': duration_rows(records, oracle),
            'oracleRole': 'Optimistic perfect correction of disputed export seconds only; not automatic or measured human performance.'}
