"""Whole-parent review for a split adviser; export footage is a separate output."""
from __future__ import annotations

from analysis import neural_production_combinations as iv

BUDGETS = (0.05, 0.1, 0.2, 0.4)
RANKERS = ('chronological', 'evidence')


def parents_by_id(base):
    rows = {str(x['id']): x for x in base}
    if len(rows) != len(base):
        raise ValueError('Duplicate parent IDs')
    return rows


def jobs(record, base, splits, cleanup, inventory):
    if inventory not in ('split_only', 'combined', 'cleanup_only'):
        raise ValueError('Unknown inventory')
    parents = parents_by_id(base)
    grouped = {}
    flags = ([] if inventory == 'cleanup_only' else [(x, 'split') for x in splits])
    flags += ([] if inventory == 'split_only' else [(x, 'cleanup') for x in cleanup])
    for flag, reason in flags:
        pid = str(flag['parentId'])
        if pid not in parents:
            raise ValueError('Unknown flagged parent')
        parent = parents[pid]
        row = grouped.setdefault(pid, {'id': 'parent:'+pid, 'parentId': pid,
            'start': float(parent['start']), 'end': float(parent['end']),
            'splitIds': [], 'cleanupIds': [], 'priority': 0.0})
        row[reason+'Ids'].append(flag['id'])
        row['priority'] = max(row['priority'], float(flag['priority']))
    for row in grouped.values():
        row['splitIds'].sort(); row['cleanupIds'].sort()
        cost = iv.duration(iv.export([row], record, 2))
        if cost <= 0:
            raise ValueError('Nonpositive job viewing cost')
        row['standalonePlaybackSeconds'] = cost
        row['evidencePerSecond'] = row['priority']/cost
    return sorted(grouped.values(), key=lambda r: (r['start'], r['end'], r['id']))


def queues(record, rows, ranker, budgets=BUDGETS):
    if ranker not in RANKERS:
        raise ValueError('Unknown ranker')
    order = sorted(rows, key=lambda r: ((-r['evidencePerSecond'],) if ranker == 'evidence' else ())
                   + (r['start'], r['end'], r['id']))
    valid_seconds = iv.duration(iv.difference([(0, record['durationSeconds'])], record.get('ignoredIntervals', [])))
    selected, out = [], []
    for fraction in budgets:
        selected_ids = {r['id'] for r in selected}
        limit = valid_seconds*fraction
        for row in order:
            if row['id'] in selected_ids:
                continue
            trial = selected+[row]
            if iv.duration(iv.export(trial, record, 2)) <= limit+1e-9:
                selected.append(row); selected_ids.add(row['id'])
        playback = iv.export(selected, record, 2)
        windows = iv.difference(selected, record.get('ignoredIntervals', []))
        out.append({'budgetFraction': fraction, 'budgetSeconds': limit,
                    'reviewSeconds': iv.duration(playback), 'unusedBudgetSeconds': limit-iv.duration(playback),
                    'selectedJobIds': [r['id'] for r in selected],
                    'selectedParentIds': [r['parentId'] for r in selected],
                    'selectedSplitIds': sorted({i for r in selected for i in r['splitIds']}),
                    'selectedCleanupIds': sorted({i for r in selected for i in r['cleanupIds']}),
                    'playbackWindows': iv.serial(playback), 'editWindows': iv.serial(windows),
                    'reviewJobs': len(selected), 'jobsAvailable': len(rows),
                    'reviewClips': len(playback), 'editRegions': len(windows)})
    return out


def touched(record, windows):
    return [i for i, gold in enumerate(record['rallies'])
            if iv.duration(iv.intersection(iv.difference([gold], record.get('ignoredIntervals', [])), windows)) > 0]


def cleanup_yield(record, cleanup):
    parents = parents_by_id(record['productionEvents'])
    gold = iv.difference(record['rallies'], record.get('ignoredIntervals', []))
    rows = []
    for flag in cleanup:
        parent = parents[str(flag['parentId'])]
        valid = iv.difference([parent], record.get('ignoredIntervals', []))
        core = iv.duration(iv.intersection(valid, gold))
        rows.append({'id': flag['id'], 'parentId': flag['parentId'],
                     'whollyFalse': core == 0, 'goldCoreSeconds': core,
                     'validSeconds': iv.duration(valid), 'supportFraction': flag['supportFraction']})
    return {'flags': rows, 'flaggedParents': len(rows),
            'realRalliesTouched': len(touched(record, [parents[str(f['parentId'])] for f in cleanup])),
            'whollyFalseParents': sum(r['whollyFalse'] for r in rows),
            'realOrMixedParents': sum(not r['whollyFalse'] for r in rows),
            'whollyFalsePrecision': sum(r['whollyFalse'] for r in rows)/len(rows) if rows else 0.0}


def human_edit(record, splits, cleanup, queue, advisor, metrics):
    """Only selected proposed splits and whole-false cleanup; no export changes."""
    selected_split_ids = set(queue['selectedSplitIds'])
    proposed = [x for x in splits if x['id'] in selected_split_ids]
    match = metrics.match_proposals(record, proposed, tolerance=1.0)
    # match_proposals returns targets and exact proposal/target index pairs.
    targets, pairs = match['targets'], match['pairs']
    accepted = []
    for pi, ti in pairs:
        candidate, target = proposed[pi], targets[ti]
        accepted.append({**candidate, 'time': float(target['start']),
                         'originalProposedTime': float(candidate['time']),
                         'humanConfirmed': True, 'targetTruthIndex': target['truthIndex']})
    parent_map = parents_by_id(record['productionEvents'])
    gold = iv.difference(record['rallies'], record.get('ignoredIntervals', []))
    removed = []
    selected_cleanup_ids = set(queue['selectedCleanupIds'])
    for flag in cleanup:
        pid = str(flag['parentId'])
        if flag['id'] in selected_cleanup_ids and iv.duration(iv.intersection([parent_map[pid]], gold)) == 0:
            removed.append(pid)
    events = advisor.apply_splits(record['productionEvents'], accepted)
    events = [e for e in events if str(e.get('parentId', e['id'])) not in set(removed)]
    old_union = iv.difference(record['productionEvents'], record.get('ignoredIntervals', []))
    new_union = iv.difference(events, record.get('ignoredIntervals', []))
    return {'events': events, 'acceptedSplits': accepted,
            'proposalsReviewed': len(proposed), 'acceptedSplitCount': len(accepted),
            'rejectedSplitCount': len(proposed)-len(accepted),
            'removedFalseParentIds': sorted(set(removed)), 'removedFalseParents': len(set(removed)),
            'eventTimelineRemovedSeconds': iv.duration(iv.difference(old_union, new_union)),
            'eventTimelineAddedSeconds': iv.duration(iv.difference(new_union, old_union)),
            'reviewedTrueRallies': len(touched(record, queue['editWindows'])),
            'playbackTrueRallies': len(touched(record, queue['playbackWindows']))}
