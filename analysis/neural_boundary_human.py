"""Restricted ideal-human actions on proposed rally events inside selected parents."""
from __future__ import annotations

from . import neural_production_combinations as iv
from .neural_boundary_metrics import match_boundary_proposals


def human_edit(record, candidates, selected_parent_ids, *, correct_ends=False):
    """Accept only observed-start/material-overlap candidates; never invent rallies.

All model candidates in a selected parent may be reviewed. A parent with no
accepted candidate stays exactly original. An edited parent is represented by
valid-component pieces; any component without an accepted initial target keeps
its original baseline start as a fallback. Paired mode may remove unproposed
core time: physical coverage diagnostics must report that loss independently.
"""
    parents = {p['id']: p for p in record['productionEvents']}
    selected = set(selected_parent_ids)
    if not selected <= set(parents):
        raise ValueError('Selected parent IDs must exist')
    reviewed_indexes = [i for i,c in enumerate(candidates) if c['parentId'] in selected]
    reviewed = [candidates[i] for i in reviewed_indexes]
    match = match_boundary_proposals(record, reviewed, tolerance=1.)
    accepted = []
    for local_index, target_index in match['pairs']:
        candidate, target = reviewed[local_index], match['targets'][target_index]
        accepted.append({'candidateIndex': reviewed_indexes[local_index], 'candidateId': candidate['id'],
                         'parentId': candidate['parentId'], 'proposedType': candidate['type'],
                         'targetType': target['type'], 'targetTruthIndex': target['truthIndex'],
                         'targetIndex': target_index, 'target': target,
                         'proposedStart': candidate['start'], 'proposedEnd': candidate['end'],
                         'typeCorrected': candidate['type'] != target['type']})
    accepted_indexes = {x['candidateIndex'] for x in accepted}
    rejected = [{'candidateIndex': i, 'candidateId': candidates[i]['id'], 'parentId': candidates[i]['parentId']}
                for i in reviewed_indexes if i not in accepted_indexes]
    output = []
    edited_parent_ids = []
    for parent in record['productionEvents']:
        accepted_here = [a for a in accepted if a['parentId'] == parent['id']]
        if not accepted_here:
            output.append(dict(parent))
            continue
        edited_parent_ids.append(parent['id'])
        components = iv.difference([parent], record.get('ignoredIntervals', []))
        for ci, component in enumerate(components):
            local = [a for a in accepted_here if component.start <= a['proposedStart'] < component.end]
            local.sort(key=lambda a:(max(component.start,a['target']['start']),a['candidateIndex']))
            rows = []
            has_initial = any(a['targetType'] == 'initial_start' for a in local)
            if not has_initial:
                rows.append({**parent, 'start': component.start, 'end': component.end,
                             'startObserved': component.start == parent['start'] and parent.get('startObserved',True),
                             'endObserved': component.end == parent['end'] and parent.get('endObserved',True),
                             'startKind':'baseline_fallback', 'endKind':'inherited', 'humanConfirmed':False})
            for item in local:
                target = item['target']
                start = max(component.start,target['start'])
                end = min(component.end,target['end']) if correct_ends else component.end
                if end <= start:
                    raise ValueError('Accepted candidate created empty reviewed event')
                rows.append({**parent, 'start':start, 'end':end,
                             'startObserved':target['start'] == start,
                             'endObserved':(target['end'] == end if correct_ends else component.end == parent['end'] and parent.get('endObserved',True)),
                             'startKind':'human_confirmed' if target['start'] == start else 'permission_edge',
                             'endKind':('human_confirmed' if target['end'] == end else 'permission_edge') if correct_ends else 'inherited',
                             'humanConfirmed':True,'candidateId':item['candidateId'],
                             'targetTruthIndex':item['targetTruthIndex'], 'type':item['targetType']})
            rows.sort(key=lambda r:(r['start'],r['end']))
            for left,right in zip(rows,rows[1:]):
                if left['end'] > right['start']:
                    left.update(end=right['start'],endObserved=False,endKind='synthetic_partition')
            for index,row in enumerate(rows):
                if row['end'] <= row['start']:
                    raise ValueError('Reviewed partitions contain a zero-length event')
                row.update(id=f"{parent['id']}::review-c{ci}-{index}", parentId=parent['id'],
                           parentStart=parent['start'],parentEnd=parent['end'],componentId=f"{parent['id']}::c{ci}")
                output.append(row)
    output.sort(key=lambda r:(r['start'],r['end'],r['id']))
    return {'events':output, 'acceptedCandidates':accepted, 'rejectedCandidates':rejected,
            'reviewedCandidates':len(reviewed), 'acceptedCount':len(accepted), 'rejectedCount':len(rejected),
            'correctedTypeCount':sum(a['typeCorrected'] for a in accepted),
            'editedParentIds':edited_parent_ids, 'editedParents':len(edited_parent_ids),
            'selectedParentIds':sorted(selected), 'correctEnds':correct_ends}
