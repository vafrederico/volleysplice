"""Typed rally-start advice and identity-bound end diagnostics, evaluation only."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from . import neural_production_combinations as iv
from .neural_rally_identity_metrics import _maximum_weight_pairs
from .neural_split_metrics import _prepared, _material, _overlap, _number


TYPES = ('initial_start', 'additional_start')
TOLERANCES = (.5, 1., 2.)


def boundary_targets(record):
    """All material-overlap identities, including inaccessible external endpoints."""
    _, _, gold, parents, _, _ = _prepared(record)
    targets = []
    for parent in parents:
        for ci, component in enumerate(parent['support']):
            covered = [g for g in gold if _overlap(g['interval'], component) >= _material(g['interval'])]
            for position, g in enumerate(covered):
                start, end = g['interval'].start, g['interval'].end
                targets.append({'id': f"{parent['id']}::c{ci}::gold{g['index']}",
                                'parentId': parent['id'], 'parentIndex': parent['index'],
                                'componentIndex': ci, 'componentStart': component.start, 'componentEnd': component.end,
                                'truthIndex': g['index'], 'start': start, 'end': end,
                                'type': TYPES[0] if position == 0 else TYPES[1],
                                'startAccessible': component.start <= start < component.end,
                                'endAccessible': component.start < end <= component.end})
    return targets


def _proposals(record, proposals):
    _, _, _, parents, excluded, _ = _prepared(record)
    parent_map = {p['id']: p for p in parents}
    rows = []
    for index, p in enumerate(proposals):
        if not isinstance(p, Mapping) or p.get('parentId') not in parent_map or p.get('type') not in TYPES:
            raise ValueError('Candidate needs existing parentId and initial_start/additional_start type')
        start, end = _number(p.get('start'), 'start'), _number(p.get('end'), 'end')
        parent = parent_map[p['parentId']]
        if not parent['interval'].start <= start < end <= parent['interval'].end:
            raise ValueError('Candidate must stay within original parent')
        so, eo = p.get('startObserved', True), p.get('endObserved', True)
        if not isinstance(so, bool) or not isinstance(eo, bool):
            raise ValueError('Observation flags must be booleans')
        start_available = not any(x.start <= start < x.end for x in excluded)
        end_available = not any(x.start < end <= x.end for x in excluded)
        component = next((i for i, x in enumerate(parent['support']) if x.start <= start < x.end), None)
        rows.append({'index': index, 'id': p.get('id', f'candidate-{index}'),
                     'parentId': p['parentId'], 'start': start, 'end': end, 'type': p['type'],
                     'componentIndex': component, 'startObserved': so and start_available,
                     'endObserved': eo and end_available, 'startAvailable': start_available,
                     'endAvailable': end_available})
    return rows


def _pairs(targets, proposals, tolerance, *, typed, material=False):
    errors = np.asarray([[abs(t['start']-p['start']) for p in proposals] for t in targets]).reshape(len(targets), len(proposals))
    eligible = np.zeros(errors.shape, dtype=bool)
    for i, target in enumerate(targets):
        for j, proposal in enumerate(proposals):
            overlap = max(0., min(target['end'], proposal['end'])-max(target['start'], proposal['start']))
            eligible[i,j] = (proposal['startObserved'] and proposal['parentId'] == target['parentId']
                             and proposal['componentIndex'] == target['componentIndex']
                             and (not typed or proposal['type'] == target['type'])
                             and errors[i,j] <= tolerance
                             and (not material or overlap >= min(.5, .1*(target['end']-target['start']))))
    return _maximum_weight_pairs(np.maximum(0., 1.-errors/tolerance), eligible)


def match_boundary_proposals(record, proposals, *, tolerance=1.):
    """Restricted human oracle, untyped matching; type may be corrected by review.

Observed start within tolerance and material interval overlap are both required.
Return original candidate index/target index pairs. Gold endpoint coordinates
outside target component are NOT permission to edit beyond that component.
"""
    tolerance = _number(tolerance, 'tolerance')
    if tolerance <= 0:
        raise ValueError('tolerance must be positive')
    targets, candidates = boundary_targets(record), _proposals(record, proposals)
    pairs = _pairs(targets, candidates, tolerance, typed=False, material=True)
    return {'targets': targets, 'pairs': [(candidates[j]['index'], i) for i,j in pairs]}


def _rates(matched, predicted, true):
    return {'true': true, 'predicted': predicted, 'matched': matched,
            'falsePositive': predicted-matched, 'falseNegative': true-matched,
            'precision': matched/predicted if predicted else (0. if true else None),
            'recall': matched/true if true else None, 'f1': 2*matched/(predicted+true) if true else None}


def _coverage(record):
    ignored = record.get('ignoredIntervals', [])
    valid = iv.difference([[0., record['durationSeconds']]], ignored)
    base = iv.intersection(record['productionEvents'], valid)
    result = iv.intersection(record.get('predictions', record['productionEvents']), valid)
    core = iv.intersection(record['rallies'], valid)
    _, _, eligible_gold, _, _, _ = _prepared(record)
    base_missed = {g['index'] for g in eligible_gold if not iv.intersection([g['interval']], base)}
    result_missed = {g['index'] for g in eligible_gold if not iv.intersection([g['interval']], result)}
    core_seconds = iv.duration(core)
    base_covered = iv.duration(iv.intersection(core,base))
    result_covered = iv.duration(iv.intersection(core,result))
    return {'coreHumanSeconds':core_seconds,'baselineRawCoreCoveredSeconds':base_covered,
            'resultRawCoreCoveredSeconds':result_covered,
            'baselineRawCoreRecall':base_covered/core_seconds if core_seconds else None,
            'resultRawCoreRecall':result_covered/core_seconds if core_seconds else None,
            'rawCoreSecondsLostFromBaseline': iv.duration(iv.intersection(core, iv.difference(base, result))),
            'rawCoreSecondsAddedToBaseline': iv.duration(iv.intersection(core, iv.difference(result, base))),
            'rawSelectedSecondsLostFromBaseline': iv.duration(iv.difference(base, result)),
            'rawSelectedSecondsAddedToBaseline': iv.duration(iv.difference(result, base)),
            'baselineCompleteMisses': len(base_missed), 'resultCompleteMisses': len(result_missed),
            'additionalCompleteMisses': len(result_missed-base_missed),
            'retainedCompleteMisses': len(result_missed & base_missed),
            'recoveredCompleteMisses': len(base_missed-result_missed),
            'additionalCompleteMissIndexes': sorted(result_missed-base_missed)}


def _record(record):
    for key in ('id','sourceGroup'):
        if not isinstance(record.get(key),str) or not record[key]:
            raise ValueError('Recording id and sourceGroup must be nonempty strings')
    targets = boundary_targets(record)
    proposals = _proposals(record, record.get('boundaryProposals', []))
    observed = [p for p in proposals if p['startObserved']]
    _, original_gold, eligible_gold, _, _, censored_gold = _prepared(record)
    output = {'id': record['id'], 'sourceGroup': record['sourceGroup'],
              'originalTrueRallies': len(original_gold), 'eligibleTrueRallies': len(eligible_gold),
              'ignoredTouchedTrueRallies': len(censored_gold), 'targets': targets, 'proposals': proposals,
              'targetCount': len(targets), 'uniqueTargetTrueRallies': len({t['truthIndex'] for t in targets}),
              'inaccessibleTargetStarts': sum(not t['startAccessible'] for t in targets),
              'inaccessibleTargetEnds': sum(not t['endAccessible'] for t in targets),
              'candidateCount': len(proposals), 'observedCandidateStarts': len(observed),
              'unobservedCandidateStarts': sum(not p['startObserved'] for p in proposals),
              'observedCandidateEnds': sum(p['endObserved'] for p in proposals),
              'unobservedCandidateEnds': sum(not p['endObserved'] for p in proposals),
              **_coverage(record), 'typedStartLocalization': {}, 'untypedStartLocalization': {},
              'typeDiagnostics': {}, 'samePairBoundaries': {}, 'byType': {kind: {} for kind in TYPES}}
    for tolerance in TOLERANCES:
        key = format(tolerance, 'g')
        pairs = _pairs(targets, proposals, tolerance, typed=True, material=True)
        untyped = _pairs(targets, proposals, tolerance, typed=False, material=True)
        output['typedStartLocalization'][key] = _rates(len(pairs),len(observed),len(targets))
        output['untypedStartLocalization'][key] = _rates(len(untyped),len(observed),len(targets))
        wrong = [(i,j) for i,j in untyped if targets[i]['type'] != proposals[j]['type']]
        output['typeDiagnostics'][key] = {'wrongType': len(wrong),
                'initialProposedForAdditional': sum(proposals[j]['type'] == TYPES[0] for _,j in wrong),
                'additionalProposedForInitial': sum(proposals[j]['type'] == TYPES[1] for _,j in wrong),
                'untypedMatches': len(untyped),
                'pairs': [{'targetIndex': i, 'proposalIndex': proposals[j]['index'],
                           'startErrorSeconds': proposals[j]['start']-targets[i]['start'],
                           'targetType': targets[i]['type'], 'proposalType': proposals[j]['type']}
                          for i,j in untyped]}
        end_correct = sum(proposals[j]['endObserved'] and abs(proposals[j]['end']-targets[i]['end']) <= tolerance for i,j in pairs)
        end_observed = sum(proposals[j]['endObserved'] for _,j in pairs)
        output['samePairBoundaries'][key] = {**_rates(end_correct,len(observed),len(targets)),
                'startMatched': len(pairs), 'startMatchedWithObservedEnd': end_observed,
                'endCorrectGivenMatchedStart': end_correct,
                'conditionalEndAccuracy': end_correct/end_observed if end_observed else None,
                'pairs': [{'targetIndex': i, 'proposalIndex': proposals[j]['index'],
                           'startErrorSeconds': proposals[j]['start']-targets[i]['start'],
                           'endErrorSeconds': proposals[j]['end']-targets[i]['end'],
                           'endObserved': proposals[j]['endObserved']} for i,j in pairs]}
        for kind in TYPES:
            kind_pairs = [(i,j) for i,j in pairs if targets[i]['type'] == kind]
            output['byType'][kind][key] = _rates(len(kind_pairs), sum(p['type'] == kind for p in observed), sum(t['type'] == kind for t in targets))
    return output


COUNT_FIELDS = ('originalTrueRallies','eligibleTrueRallies','ignoredTouchedTrueRallies','targetCount',
                'uniqueTargetTrueRallies','inaccessibleTargetStarts','inaccessibleTargetEnds','candidateCount',
                'observedCandidateStarts','unobservedCandidateStarts','observedCandidateEnds','unobservedCandidateEnds',
                'coreHumanSeconds','baselineRawCoreCoveredSeconds','resultRawCoreCoveredSeconds',
                'rawCoreSecondsLostFromBaseline','rawCoreSecondsAddedToBaseline','rawSelectedSecondsLostFromBaseline',
                'rawSelectedSecondsAddedToBaseline','baselineCompleteMisses','resultCompleteMisses',
                'additionalCompleteMisses','retainedCompleteMisses','recoveredCompleteMisses')


def _pool(rows):
    output = {key:sum(r[key] for r in rows) for key in COUNT_FIELDS}
    for prefix in ('baseline','result'):
        output[prefix+'RawCoreRecall'] = output[prefix+'RawCoreCoveredSeconds']/output['coreHumanSeconds'] if output['coreHumanSeconds'] else None
    for field in ('typedStartLocalization','untypedStartLocalization','samePairBoundaries'):
        output[field] = {}
        for tolerance in TOLERANCES:
            key = format(tolerance,'g')
            counts = {k:sum(r[field][key][k] for r in rows) for k in ('matched','predicted','true')}
            output[field][key] = _rates(**counts)
            if field == 'samePairBoundaries':
                for name in ('startMatched','startMatchedWithObservedEnd','endCorrectGivenMatchedStart'):
                    output[field][key][name] = sum(r[field][key][name] for r in rows)
                v = output[field][key]
                v['conditionalEndAccuracy'] = v['endCorrectGivenMatchedStart']/v['startMatchedWithObservedEnd'] if v['startMatchedWithObservedEnd'] else None
    output['typeDiagnostics'] = {format(t,'g'):{k:sum(r['typeDiagnostics'][format(t,'g')][k] for r in rows)
            for k in ('wrongType','initialProposedForAdditional','additionalProposedForInitial','untypedMatches')} for t in TOLERANCES}
    output['byType'] = {kind:{format(t,'g'):_rates(**{k:sum(r['byType'][kind][format(t,'g')][k] for r in rows)
                    for k in ('matched','predicted','true')}) for t in TOLERANCES} for kind in TYPES}
    return output


def evaluate_boundary_proposals(records: Iterable[Mapping[str,Any]]):
    rows = [_record(r) for r in records]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique recording IDs required')
    return {'metricContract': {
            'version':'typed-boundary-advice-v1','types':list(TYPES),'tolerancesSeconds':list(TOLERANCES),
            'primaryToleranceSeconds':1.,
            'targetRule':'first material-overlap gold in each parent valid component is initial_start; all later are additional_start; inaccessible endpoints remain in denominators',
            'ignoredRule':'exclude ignored-touched gold identities and their entire spans from identity/typed boundaries; physical raw-core loss uses ALL gold minus ignored',
            'matchingRule':'observed start markers only; same parent and valid component AND material candidate-gold interval overlap; typed matches require same type; maximum cardinality then minimum total start error',
            'wrongTypeRule':'type mismatch on a separate untyped maximum-cardinality minimum-start-error matching',
            'endRule':'end correctness uses exactly the typed start-matched pair at the same tolerance and requires observed end; no separate end assignment',
            'jointRatesRule':'joint correct count over all observed candidate starts/all gold targets; synthetic/unobserved ends cannot be correct',
            'oracleRule':'untyped one-to-one observed-start match within1s AND candidate-gold material interval overlap; no endpoint outside review permission is observed',
            'materialRule':'overlap>=min(0.5 seconds,10% gold duration)',
            'aggregation':'sum counts across recordings before rates; keep seeds separate',
            'limit':'rally boundary timing only, not serve side, winner or reconstructed score accuracy'},
            'pooled':_pool(rows), 'sourceGroups':{g:_pool([r for r in rows if r['sourceGroup']==g]) for g in sorted({r['sourceGroup'] for r in rows})},
            'recordings':rows}
