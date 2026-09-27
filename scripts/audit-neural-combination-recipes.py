#!/usr/bin/env python3
"""Independent standard-library reconstruction of the fixed combination recipes.

Imports no application/model/shared interval modules. Automatic predictions are
reconstructed from frozen inputs. Review disagreement, playback queues and
limited oracle edits are reconstructed separately, without reading model ranks.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(ref):
    require(sha(ref['path']) == ref['sha256'], 'Artifact changed: ' + ref['path'])
    return read(ref['path'])


def ranges(rows):
    output = []
    for row in rows:
        a, b = (row['start'], row['end']) if isinstance(row, dict) else row
        require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                    for x in (a, b)) and a < b, 'Invalid interval')
        output.append((float(a), float(b)))
    return output


def select(groups, predicate):
    """Independent occupancy sweep implementing arbitrary set predicates."""
    changes = {}
    for channel, group in enumerate(groups):
        for a, b in ranges(group):
            changes.setdefault(a, [0] * len(groups))[channel] += 1
            changes.setdefault(b, [0] * len(groups))[channel] -= 1
    points = sorted(changes)
    active, output = [0] * len(groups), []
    for i, a in enumerate(points[:-1]):
        active = [old + change for old, change in zip(active, changes[a], strict=True)]
        require(all(x >= 0 for x in active), 'Negative sweep activity')
        b = points[i + 1]
        if predicate(tuple(x > 0 for x in active)):
            if output and output[-1][1] == a:
                output[-1] = (output[-1][0], b)
            else:
                output.append((a, b))
    return output


def united(*groups):
    return select(groups, any)


def minus(a, b):
    return select((a, b), lambda x: x[0] and not x[1])


def overlap(a, b):
    return select((a, b), all)


def expand(rows, pad, duration):
    return united([(max(0.0, a-pad), min(duration, b+pad))
                   for a,b in ranges(rows) if min(duration,b+pad) > max(0.0,a-pad)])


def export(rows, rec, pad):
    joined=[]
    for a,b in expand(rows,pad,rec['durationSeconds']):
        if joined and a-joined[-1][1] < 3.0:
            joined[-1]=(joined[-1][0],b)
        else:
            joined.append((a,b))
    return minus(joined,rec['ignoredIntervals'])


def agreement(previous, v2, duration):
    """Connected graph of source-tagged padded ranges; strict .5s gap edge."""
    vertices=[]
    for source,rows in enumerate((previous,v2)):
        for raw in ranges(rows):
            vertices.append((max(0.,raw[0]-2),min(duration,raw[1]+2),source,raw))
    vertices.sort()
    owners=list(range(len(vertices)))
    def owner(i):
        while owners[i] != i:
            owners[i]=owners[owners[i]]
            i=owners[i]
        return i
    for i,(a,b,_,_) in enumerate(vertices):
        for j in range(i+1,len(vertices)):
            other=vertices[j]
            if other[0]-b >= .5:
                break
            owners[owner(j)]=owner(i)
    components={}
    for i,(_,_,source,raw) in enumerate(vertices):
        component=components.setdefault(owner(i),{'sources':set(),'raw':[]})
        component['sources'].add(source)
        component['raw'].append(raw)
    return [united(c['raw']) for c in components.values() if len(c['sources']) == 1]


def construction(family, p, ns, eligible, duration):
    if family in ('nn_union','three_union','union'):
        return united(p,*ns)
    if family in ('nn_intersection','intersection'):
        return overlap(p,ns[0])
    if family == 'majority':
        return select((p,*ns),lambda flags:sum(flags)>=2)
    support=expand(ns[0],2,duration)
    if family == 'tolerant_intersection':
        return overlap(p,support)
    retained_components=[overlap(part,p) for part in eligible]
    if family == 'guarded_trim':
        return minus(p,minus(united(*retained_components),support))
    require(family == 'guarded_component_rejection','Unknown family')
    rejected=[part for part in retained_components if not overlap(part,support)]
    return minus(p,united(*rejected))


def assert_ranges(actual, expected, label):
    a,b=ranges(actual),ranges(expected)
    require(len(a)==len(b),label+' interval count differs')
    require(all(abs(x-y)<=1e-9 for row1,row2 in zip(a,b,strict=True)
                for x,y in zip(row1,row2,strict=True)),label+' endpoints differ')


def seconds(rows):
    return math.fsum(b-a for a,b in united(rows))


def assert_number(actual,expected,label):
    require(math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-8),label+' duration differs')


def audit(output):
    output=Path(output)
    source_before=sha(__file__)
    report_before=sha(output/'report.json')
    report=read(output/'report.json')
    require(report['status']=='completed-fixed-combinations','Study is not complete')
    require(report['protectedTestOpened'] is False and report['productionChanged'] is False
            and report['newFits']==0,'Completed report scope differs')
    registration=read(output/'registration.json')
    contract=registration['contract']
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(signature==registration['sha256']==report['contractSha256'],'Contract hash differs')
    require(contract['automaticConfigurations']==107 and contract['automaticResultCells']==303
            and contract['reviewResultCells']==90,'Fixed inventory differs')
    production=bound(contract['productionInput'])
    neural=bound(contract['neuralInput'])
    for ref in neural['sourceBindings']:
        bound(ref)
    prod={r['id']:r for r in production['recordings']}
    records={r['id']:r for r in neural['records']}
    require(set(prod)==set(records) and len(records)==8,'Scope differs')
    recipes={r['id']:r for r in contract['configurations']}
    require(len(recipes)==107,'Duplicate recipe')
    require(contract['seeds']==[3407,1729,20260918],'Seed population/order differs')
    require(contract['anchors']==['productionDefault','shippedUnion','refitUnion'],'Anchor population differs')
    require(contract['participants']==['compact_boost','compact_keep','dino_global','dino_boost','dino_keep'],
            'Combination neural population differs')
    production_names=['shippedPrevious','shippedV2','shippedUnion','refitPrevious','refitV2',
                      'refitUnion','productionDefault','productionBalanced','productionConservative']
    neural_names=contract['participants']+['compact_baseline','dino_baseline']
    pairs={'best_f1_pair':['compact_boost','dino_global'],
           'recovery_pair':['compact_keep','dino_boost']}
    independent_recipes=[]
    def add(name,family,anchor,neural_ids):
        independent_recipes.append({'id':name,'family':family,'anchor':anchor,'neuralIds':neural_ids})
    for name in production_names:
        add(name,'production',name,[])
    for name in neural_names:
        add(name,'neural',None,[name])
    for name,pair in pairs.items():
        for family in ('nn_union','nn_intersection'):
            add(name+'--'+family,family,None,pair)
    for anchor in contract['anchors']:
        for name in contract['participants']:
            for family in ('union','intersection','tolerant_intersection','guarded_trim','guarded_component_rejection'):
                add(anchor+'--'+name+'--'+family,family,anchor,[name])
        for name,pair in pairs.items():
            for family in ('three_union','majority'):
                add(anchor+'--'+name+'--'+family,family,anchor,pair)
    require({r['id']:r for r in independent_recipes}==recipes,'Registered inventory differs from fixed protocol recipes')
    expected_cells={(name,None if r['family']=='production' else seed)
                    for name,r in recipes.items() for seed in ([None] if r['family']=='production' else contract['seeds'])}
    component_cache={}
    for rid,rec in records.items():
        for prefix in ('shipped','refit'):
            p=prod[rid]['cores']
            component_cache[rid,prefix]=agreement(p[prefix+'Previous'],p[prefix+'V2'],rec['durationSeconds'])
    seen,construction_rows=set(),0
    for ref in report['automaticResults']:
        result=bound(ref)
        recipe,seed=result['configuration'],result['seed']
        key=(recipe['id'],seed)
        require(key in expected_cells and key not in seen,'Unexpected/duplicate automatic cell')
        require(recipe==recipes[recipe['id']] and result['contractSha256']==signature,'Cell recipe binding differs')
        require(set(result['predictions'])==set(records),'Automatic recording scope differs')
        seen.add(key)
        for rid,rec in records.items():
            p=prod[rid]['cores']
            family,anchor=recipe['family'],recipe['anchor']
            ns=[neural['predictions'][name][str(seed)][rid] for name in recipe['neuralIds']]
            if family=='production':
                expected=p[anchor]
            elif family=='neural':
                expected=ns[0]
            elif family.startswith('nn_'):
                expected=construction(family,ns[0],ns[1:],[],rec['durationSeconds'])
            else:
                prefix='refit' if anchor=='refitUnion' else 'shipped'
                expected=construction(family,p[anchor],ns,component_cache[rid,prefix],rec['durationSeconds'])
            assert_ranges(result['predictions'][rid],expected,f'{recipe["id"]}/{seed}/{rid}')
            construction_rows+=1
    require(seen==expected_cells,'Missing automatic cell')
    expected_reviews={(anchor,nn,seed,mode) for anchor in contract['anchors'] for nn in contract['participants']
                      for seed in contract['seeds'] for mode in contract['reviewModes']}
    seen_reviews,review_rows=set(),0
    for ref in report['reviewResults']:
        result=bound(ref)
        anchor,nn,seed,mode=(result[k] for k in ('anchor','neuralId','seed','mode'))
        key=anchor,nn,seed,mode
        require(key in expected_reviews and key not in seen_reviews,'Unexpected/duplicate review cell')
        require(result['contractSha256']==signature,'Review contract differs')
        seen_reviews.add(key)
        require(sorted(r['paddingSecondsBeforeAndAfter'] for r in result['queue'])==[0,1,2,3],
                'Review padding population differs')
        for qrow in result['queue']:
            pad=qrow['paddingSecondsBeforeAndAfter']
            require(pad in (0,1,2,3),'Review padding differs')
            per={r['id']:r for r in qrow['perRecording']}
            require(set(per)==set(records),'Review recording scope differs')
            pooled={k:0. for k in ('reviewSeconds','disputedSeconds','unwantedExportFlaggedSeconds',
                                   'wantedExportFlaggedSeconds','missedHumanExportFlaggedSeconds','missedCoreFlaggedSeconds')}
            clips=0
            for rid,rec in records.items():
                m=export(prod[rid]['cores'][anchor],rec,pad)
                n=export(neural['predictions'][nn][str(seed)][rid],rec,pad)
                proposed=minus(m,n)
                disputed=proposed if mode=='suppression_review' else united(proposed,minus(n,m))
                view=export(disputed,rec,2)
                hp=export(rec['rallies'],rec,pad)
                hc=minus(rec['rallies'],rec['ignoredIntervals'])
                oracle=united(minus(m,disputed),overlap(hp,disputed))
                assert_ranges(per[rid]['disputedIntervals'],disputed,'Review disputed '+rid)
                assert_ranges(per[rid]['reviewIntervals'],view,'Review playback '+rid)
                assert_ranges(result['oracleExports'][rid][str(pad)],oracle,'Review oracle '+rid)
                values={'reviewSeconds':seconds(view),'disputedSeconds':seconds(disputed),
                        'unwantedExportFlaggedSeconds':seconds(overlap(disputed,minus(m,hp))),
                        'wantedExportFlaggedSeconds':seconds(overlap(proposed,hp)),
                        'missedHumanExportFlaggedSeconds':seconds(overlap(disputed,minus(hp,m))),
                        'missedCoreFlaggedSeconds':seconds(overlap(disputed,minus(hc,m)))}
                for field,value in values.items():
                    assert_number(per[rid][field],value,'Review '+field)
                    pooled[field]+=value
                require(per[rid]['reviewClips']==len(view),'Per-recording review clip count differs')
                clips+=len(view)
                review_rows+=1
            for field,value in pooled.items():
                assert_number(qrow[field],value,'Pooled review '+field)
            require(qrow['reviewClips']==clips,'Pooled review clip count differs')
    require(seen_reviews==expected_reviews,'Missing review cell')
    fidelity_rows=0
    require(sorted(r['variant'] for r in report['actualAppFidelity'])==
            sorted(('shippedUnion','productionDefault','productionBalanced','productionConservative')),
            'Actual-app variant population differs')
    for row in report['actualAppFidelity']:
        name=row['variant']
        require(name in ('shippedUnion','productionDefault','productionBalanced','productionConservative'),'Unexpected app variant')
        for rid in records:
            for pad in map(str,range(4)):
                assert_ranges(row['exports'][rid][pad],prod[rid]['productExportsByPadding'][name][pad],'App override identity')
                fidelity_rows+=1
    require(fidelity_rows==128,'Incomplete app fidelity overrides')
    require(sha(__file__)==source_before and sha(output/'report.json')==report_before,
            'Audit source/report changed during audit')
    result={'kind':'independent-production-combination-recipe-audit-v1','passed':True,
            'createdAt':datetime.now(timezone.utc).isoformat(),'contractSha256':signature,
            'report':{'path':str(output/'report.json'),'sha256':sha(output/'report.json')},
            'implementation':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
            'automaticCellsAudited':len(seen),'automaticRecordingConstructionsAudited':construction_rows,
            'reviewCellsAudited':len(seen_reviews),'reviewRecordingPaddingRowsAudited':review_rows,
            'actualAppOverrideRecordingPaddingRowsAudited':fidelity_rows,
            'method':'Independent endpoint occupancy sweep and source-tagged agreement connectivity graph; no shared interval/model/application imports.',
            'protectedTestOpened':False}
    path=output/'recipe-audit-v1.json'
    with path.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(private_value('private-reference-0083')))
    audit(parser.parse_args().output)
