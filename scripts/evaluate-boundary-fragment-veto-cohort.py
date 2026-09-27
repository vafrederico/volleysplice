#!/usr/bin/env python3
"""Add binary whole-removal-fragment undo to the frozen development comparison."""
import copy
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    base = module('cohort_parent_policy', REPO/'scripts/evaluate-boundary-default-review-cohort.py')
    iv = base.iv
    old_path = base.OUTPUT/'results-v1.json'
    previous = base.read(old_path)
    assert base.ref(Path(previous['source']['path']))['sha256'] == previous['source']['sha256']
    for item in previous['inputs']:
        assert base.ref(Path(item['path']))['sha256'] == item['sha256']
    original_helper = REPO/'scripts/evaluate-labeling-boundary-default-review.py'
    snapshot = base.OUTPUT/'source-shared-restoration-v2.py'
    with snapshot.open('xb') as handle:
        handle.write(original_helper.read_bytes())
    helper = module('cohort_frozen_fragment_helper', snapshot)
    helper.self_check()
    data = base.read(base.PRIOR/'input.json')
    results = copy.deepcopy(previous['results'])
    for seed in results:
        source = base.read(base.PRIOR/'results'/f"{seed['seed']}.json")
        plans = {p['id']: p['policies']['head_refined'] for p in source['plans']}
        scored = []
        counters = Counter()
        for record, saved in zip(data['records'], seed['perRecording']):
            assert record['id'] == saved['id']
            plan = plans[record['id']]
            automatic = plan['events']
            ignored = record.get('ignoredIntervals', [])
            removed = iv.difference(iv.difference(record['productionEvents'], automatic), ignored)
            restore = tuple(fragment for fragment in removed if iv.intersection([fragment], record['rallies']))
            label_blind = {key:value for key,value in record.items() if key != 'rallies'}
            output, changes = helper.restore_core_fragments(label_blind, automatic, restore, removed,
                                                            boundary_mode='known-parent-only')
            assert output == helper.restore_core_fragments(record, automatic, restore, removed,
                boundary_mode='known-parent-only')[0], 'Gold must not influence event construction'
            expected_union = iv.union(automatic, restore)
            assert iv.duration(iv.difference(output, expected_union)) < 1e-8
            assert iv.duration(iv.difference(expected_union, output)) < 1e-8
            assert base._coverage({**record,'predictions':output})['rawCoreSecondsLostFromBaseline'] < 1e-8
            assert base._coverage({**record,'predictions':output})['additionalCompleteMisses'] == 0
            source_endpoints = {v for e in [*record['productionEvents'],*automatic] for v in (e['start'],e['end'])}
            source_endpoints.update(v for e in ignored for v in (e['start'],e['end']))
            assert all(e['start'] in source_endpoints and e['end'] in source_endpoints for e in output)
            operations = Counter(change['operation'] for change in changes)
            stats = {'granularRestoredFragmentCount':len(restore),
                'granularAcceptedRemovalFragmentCount':len(removed)-len(restore),
                'granularRestoredSeconds':iv.duration(restore),
                'granularRestoredNonGoldSeconds':iv.duration(iv.difference(restore,record['rallies'])),
                'granularExtendedEventCount':operations['extend-existing'],
                'granularMergedEventGroups':operations['merge-through-restored-play'],
                'granularIsolatedRestorationCount':operations['isolated-reviewed-fragment']}
            counters.update(stats)
            # Primary playback follows the earlier review queue's strict <3s
            # join. Preserve the no-gap-join workload as an explicit sensitivity.
            work = saved['workload']
            for key in ('playbackWindows','playbackWindowCount','playbackSeconds','playbackGoldRallyCount'):
                work[key+'NoGapJoin'] = work[key]
            playback = iv.export(removed, record, 2)
            work.update(playbackWindows=iv.serial(playback), playbackWindowCount=len(playback),
                playbackSeconds=iv.duration(playback),
                playbackGoldRallyCount=sum(bool(iv.intersection([g],playback)) for g in record['rallies']))
            global_export_removed = iv.difference(iv.export(record['productionEvents'],record,2),iv.export(automatic,record,2))
            trigger_parents = set()
            for parent in record['productionEvents']:
                own_auto = [e for e in automatic if e['parentId']==parent['id']]
                own_removed = iv.difference(iv.export([parent],record,2),iv.export(own_auto,record,2))
                if iv.intersection(own_removed,global_export_removed):
                    trigger_parents.add(parent['id'])
            changed = {parent['id'] for parent in record['productionEvents'] if
                [(e['start'],e['end']) for e in automatic if e['parentId']==parent['id']] != [(parent['start'],parent['end'])]}
            assert sorted(changed-trigger_parents)==saved['workload']['changedParentsWithoutExportRemovalTrigger']
            saved['granularVeto'] = {'restoredWindows':iv.serial(restore),'events':output,'changes':changes,**stats}
            scored.append({**record,'predictions':output})
        seed['arms']['granular_veto_removed_core'] = base.score(scored)
        for key in ('playbackWindowCount','playbackSeconds','playbackGoldRallyCount',
                    'playbackWindowCountNoGapJoin','playbackSecondsNoGapJoin','playbackGoldRallyCountNoGapJoin'):
            seed['workload'][key] = sum(saved['workload'][key] for saved in seed['perRecording'])
        seed['workload'].update(counters)
    result = {**previous, 'createdAt':datetime.now(timezone.utc).isoformat(), 'revision':2,
        'policy': 'All head_refined edits apply. Queue connected nonignored P_core minus A_core fragments. Perfect binary reviewer restores a WHOLE fragment iff any queued part overlaps human core. Extend adjacent candidates; merge only candidates actually linked by restored fragments; preserve other automatic event identities. No gold-derived boundary times.',
        'reviewContextJoinSeconds':3, 'reviewContextJoinComparison':'strictly-less-than',
        'reviewContextSensitivityJoinSeconds':0,
        'previousResult':base.ref(old_path), 'sharedRestorationSource':base.ref(snapshot),
        'sharedRestorationOriginalPath':str(original_helper), 'source':base.ref(Path(__file__)),
        'summary':base.summarize(results),'results':results,
        'checks':{**previous['checks'],'sharedRestorationSelfChecksPassed':True,
            'goldRemovedBeforeEventConstruction':True,'onlyPreexistingBoundaryCoordinatesUsed':True,
            'exactAutomaticUnionPlusWholeAcceptedFragments':True,
            'parentSpecificExportTriggerAttributionVerified':True},
    }
    result['limitations'] += ['Granular review restores an entire removed fragment based on a binary keep/remove decision. It does not permit gold-accurate temporal trimming or general rally reconstruction.']
    path=base.OUTPUT/'results-v2.json'
    with path.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    brief={'output':base.ref(path),'granular':{
        'identity':result['summary']['arms']['granular_veto_removed_core']['identity'],
        'targetExport':result['summary']['arms']['granular_veto_removed_core']['padding'][2],
        'workload':{k:v for k,v in result['summary']['workload'].items() if k.startswith('granular')}},
        'checks':result['checks']}
    print(json.dumps(brief,indent=2))


if __name__=='__main__':
    main()
