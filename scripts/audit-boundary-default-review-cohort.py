#!/usr/bin/env python3
"""Independent interval/policy audit; does not import the implementation's helpers."""
from analysis.private_ledger import private_value
import hashlib
import json
from pathlib import Path

PRIOR = Path(private_value('private-reference-0079'))
ROOT = Path(private_value('private-reference-0080'))


def read(p):
    return json.loads(p.read_text())


def rows(values):
    return [(v['start'], v['end']) if isinstance(v, dict) else tuple(v) for v in values]


def merge(values, gap=0):
    out = []
    for start, end in sorted(rows(values)):
        if end <= start:
            continue
        if out and (start <= out[-1][1] or start - out[-1][1] < gap):
            out[-1] = out[-1][0], max(end, out[-1][1])
        else:
            out.append((start, end))
    return out


def subtract(a, b):
    out = merge(a)
    for start, end in merge(b):
        out = [part for left, right in out for part in
               ([(left, right)] if right <= start or left >= end else
                ([(left, start)] if left < start else []) + ([(end, right)] if right > end else []))]
    return out


def intersect(a, b):
    return merge([(max(s, x), min(e, y)) for s, e in rows(a) for x, y in rows(b)
                  if max(s, x) < min(e, y)])


def duration(values):
    return sum(e-s for s, e in merge(values))


def exported(events, record, pad):
    return subtract(merge([(max(0, s-pad), min(record['durationSeconds'], e+pad))
                           for s, e in rows(events)], 3), record.get('ignoredIntervals', []))


def close(a, b):
    assert abs(a-b) < 1e-7, (a, b)


def main():
    data = read(PRIOR/'input.json')
    result = read(ROOT/'results-v1.json')
    records = {r['id']: r for r in data['records']}
    checked = 0
    for seed in result['results']:
        frozen = read(PRIOR/'results'/f"{seed['seed']}.json")
        plans = {p['id']: p['policies']['head_refined'] for p in frozen['plans']}
        for saved in seed['perRecording']:
            r = records[saved['id']]
            p = r['productionEvents']
            a = plans[r['id']]['events']
            ignored = r.get('ignoredIntervals', [])
            removed = subtract(subtract(p, a), ignored)
            assert rows(saved['workload']['removedCoreWindows']) == removed
            vetoed = {parent['id'] for parent in p if intersect(intersect([parent], removed), r['rallies'])}
            assert sorted(vetoed) == saved['workload']['vetoedParentIds']
            expected = [e for e in a if e['parentId'] not in vetoed] + [e for e in p if e['id'] in vetoed]
            expected.sort(key=lambda e:(e['start'], e['end'], e['id']))
            assert expected == saved['resultEvents']
            assert not intersect(subtract(subtract(p, expected), ignored), r['rallies'])
            playback = subtract(merge([(max(0, s-2), min(r['durationSeconds'], e+2)) for s,e in removed]), ignored)
            assert rows(saved['workload']['playbackWindows']) == playback
            for arm, events in [('production', p), ('automatic_applied_to_export', a), ('parent_veto_removed_core', expected)]:
                for pad in (0,1,2,3):
                    m, h = exported(events, r, pad), exported(r['rallies'], r, pad)
                    core = subtract(r['rallies'], ignored)
                    valid = subtract([(0, r['durationSeconds'])], ignored)
                    actual = next(row for row in seed['arms'][arm]['durationMetrics'][pad]['perRecording'] if row['id']==r['id'])
                    values = {
                        'paddedModelExportSeconds':duration(m), 'paddedHumanExportSeconds':duration(h),
                        'coreHumanSeconds':duration(core), 'coreIntersectionSeconds':duration(intersect(core,m)),
                        'paddedIntersectionSeconds':duration(intersect(h,m)), 'evaluableVideoSeconds':duration(valid),
                        'correctlyRemovedSeconds':duration(subtract(valid,merge(rows(m)+rows(h)))),
                        'incorrectlyRemovedSeconds':duration(subtract(h,m)), 'incorrectExportSeconds':duration(subtract(m,h)),
                        'missedCoreSeconds':duration(subtract(core,m)),
                        'exportDurationDifferenceSeconds':duration(m)-duration(h),
                    }
                    precision = values['paddedIntersectionSeconds']/values['paddedModelExportSeconds']
                    recall = values['coreIntersectionSeconds']/values['coreHumanSeconds']
                    values.update(P_pad=precision, R_core=recall, F1_padP_coreR=2*precision*recall/(precision+recall))
                    for key, value in values.items():
                        close(actual[key], value)
                        checked += 1
    report = {'passed':True, 'policyCases':24, 'exportCases':288, 'numericComparisons':checked,
              'independentImplementation':'No imports from analysis interval or policy modules.',
              'checks':['Removed core windows exact','Parent vetoes exact','Output event identities exact',
                        'No newly omitted production-covered gold core','Context playback windows exact',
                        'Every per-recording all4padding export metric independently reconstructed'],
              'inputSha256':hashlib.sha256((ROOT/'results-v1.json').read_bytes()).hexdigest(),
              'sourceSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (ROOT/'audit-v1.json').open('x') as handle:
        json.dump(report,handle,indent=2)
        handle.write('\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
