"""Whole-parent review queues and the explicit full-parent human ceiling."""
from analysis import neural_production_combinations as iv
from analysis import neural_split_review as old_review

BUDGETS = old_review.BUDGETS
RANKERS = old_review.RANKERS


def jobs(record, plan):
    # Existing qualified selector is generic interval work despite its original
    # splitIds names. Expose additional unambiguous proposal aliases in outputs.
    return old_review.jobs(record, record['productionEvents'], plan['proposals'], [], 'split_only')


def queues(record, rows, ranker, budgets=BUDGETS):
    values = old_review.queues(record, rows, ranker, budgets)
    return [{**q, 'selectedProposalIds': q['selectedSplitIds'],
             'boundaryFlagsReviewed': len(q['selectedSplitIds'])} for q in values]


def full_parent_edit(record, selected_parent_ids):
    selected = set(selected_parent_ids)
    if not selected <= {p['id'] for p in record['productionEvents']}:
        raise ValueError('Unknown selected parent')
    events = []
    for parent in record['productionEvents']:
        if parent['id'] not in selected:
            events.append(dict(parent))
            continue
        components = iv.intersection([parent], iv.difference([(0, record['durationSeconds'])], record.get('ignoredIntervals', [])))
        for ci, component in enumerate(components):
            for gi, gold in enumerate(record['rallies']):
                start = max(component.start, float(gold['start']))
                end = min(component.end, float(gold['end']))
                if start >= end:
                    continue
                events.append({**parent, 'id': f"full::{parent['id']}::gold:{gi}::component:{ci}",
                  'parentId': parent['id'], 'componentId': f"{parent['id']}::component:{ci}",
                  'start': start, 'end': end,
                  'startObserved': start == float(gold['start']), 'endObserved': end == float(gold['end']),
                  'startSource': 'human' if start == float(gold['start']) else 'permission-clipped',
                  'endSource': 'human' if end == float(gold['end']) else 'permission-clipped',
                  'humanGoldIndex': gi})
    return {'events': sorted(events, key=lambda x:(x['start'], x['end'], x['id']))}


def workload(record, plan, queue, edited):
    selected = set(queue['selectedParentIds'])
    candidates = [x for x in plan['eventCandidates'] if x['parentId'] in selected]
    before = iv.difference(record['productionEvents'], record.get('ignoredIntervals', []))
    after = iv.difference(edited['events'], record.get('ignoredIntervals', []))
    return {k: queue[k] for k in ('reviewSeconds', 'budgetSeconds', 'unusedBudgetSeconds',
              'reviewJobs', 'jobsAvailable', 'reviewClips', 'editRegions', 'boundaryFlagsReviewed')} | {
       'candidatesReviewed': len(candidates),
       'reviewedTrueRallies': len(old_review.touched(record, queue['editWindows'])),
       'playbackTrueRallies': len(old_review.touched(record, queue['playbackWindows'])),
       'eventTimelineRemovedSeconds': iv.duration(iv.difference(before, after)),
       'eventTimelineAddedSeconds': iv.duration(iv.difference(after, before))}
