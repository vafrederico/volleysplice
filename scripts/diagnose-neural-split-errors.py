#!/usr/bin/env python3
"""Descriptive post-hoc taxonomy of frozen automatic split errors; no rerun."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
sys.dont_write_bytecode = True
from analysis.neural_split_metrics import _prepared, _material, _overlap
from analysis.neural_rally_identity_metrics import _maximum_weight_pairs
from analysis import neural_production_combinations as iv

ROOT = Path(private_value('private-reference-0093'))
COUNTS = ('proposals', 'matchedSecondaryStarts', 'unmatched', 'nearFirstMaterialStart',
          'nearOtherGoldStart', 'awayAllEligibleGoldStarts')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': Path(path).stat().st_size}


def require(value, message):
    if not value:
        raise ValueError(message)


def classify(record, frozen_split_record):
    _, _, gold, parents, _, _ = _prepared(record)
    parent_by_id = {p['id']: p for p in parents}
    metric = frozen_split_record['splitLocalization']['1']
    matched = {p['proposalIndex'] for p in metric['pairs']}
    proposals = frozen_split_record['proposals']
    counts = {key: 0 for key in COUNTS}
    counts.update(proposals=len(proposals), matchedSecondaryStarts=len(matched), unmatched=len(proposals)-len(matched))
    examples = []
    for proposal in proposals:
        if proposal['index'] in matched:
            continue
        parent = parent_by_id[proposal['parentId']]
        component = parent['support'][proposal['componentIndex']]
        material = [g for g in gold if _overlap(g['interval'], component) >= _material(g['interval'])]
        first = material[0] if material else None
        eligible_near = [g for g in gold if component.start <= g['interval'].start < component.end
                         and abs(g['interval'].start-proposal['time']) <= 1.0]
        if first is not None and first in eligible_near:
            category = 'nearFirstMaterialStart'
        elif eligible_near:
            category = 'nearOtherGoldStart'
        else:
            category = 'awayAllEligibleGoldStarts'
        counts[category] += 1
        examples.append({**proposal, 'category': category,
                         'parentInterval': [parent['interval'].start, parent['interval'].end],
                         'firstMaterialGold': {'truthIndex': first['index'], 'start': first['interval'].start,
                                               'end': first['interval'].end} if first else None,
                         'nearbyGold': [{'truthIndex': g['index'], 'start': g['interval'].start,
                                         'end': g['interval'].end, 'errorSeconds': proposal['time']-g['interval'].start}
                                        for g in eligible_near]})
    require(counts['matchedSecondaryStarts'] == metric['matched'], 'Frozen match counts differ')
    require(counts['unmatched'] == metric['falsePositive'], 'Frozen false proposal count differs')
    require(counts['unmatched'] == sum(counts[k] for k in COUNTS[3:]), 'Taxonomy does not partition unmatched proposals')
    return {'id': record['id'], 'sourceGroup': record['sourceGroup'], **counts, 'unmatchedProposals': examples}


def totals(rows):
    return {key: sum(row[key] for row in rows) for key in COUNTS}


def statistics(rows):
    return {key: {'mean': mean(r[key] for r in rows), 'min': min(r[key] for r in rows),
                  'max': max(r[key] for r in rows)} for key in COUNTS}


def interval_examples(data, cell):
    """First three accepted parents sorted by recording ID/parent ID, seed3407."""
    sources = {r['id']: r for r in data['records']}
    arm_id = 'head_evidence--split_only--evidence--budget-40'
    arm = next(a for a in cell['reviewed'] if a['id'] == arm_id)
    rows = []
    for edited in sorted(arm['perRecording'], key=lambda p: p['id']):
        record = sources[edited['id']]
        _, _, gold, parents, excluded, _ = _prepared(record)
        for pid in sorted({p['parentId'] for p in edited['acceptedSplits']}):
            parent = next(p for p in parents if p['id'] == pid)
            overlapping = [g for g in gold if sum(_overlap(g['interval'], span) for span in parent['support']) > 0]
            children = [e for e in edited['events'] if e['parentId'] == pid]
            children_support = [iv.difference([e], excluded) for e in children]
            def iou(g, support):
                shared = sum(_overlap(g['interval'], span) for span in support)
                denominator = g['interval'].end-g['interval'].start+iv.duration(support)-shared
                return shared/denominator if denominator else 0.
            original_iou = np.asarray([[iou(g, parent['support'])] for g in overlapping])
            children_iou = np.asarray([[iou(g, support) for support in children_support] for g in overlapping])
            original_matches = _maximum_weight_pairs(original_iou, original_iou >= .5)
            child_matches = _maximum_weight_pairs(children_iou, children_iou >= .5)
            valid_gold = [g['interval'] for g in overlapping]
            rows.append({'seed': cell['seed'], 'arm': arm_id, 'recordingId': record['id'], 'parentId': pid,
                         'originalParent': [parent['interval'].start, parent['interval'].end],
                         'goldRallies': [{'truthIndex': g['index'], 'start': g['interval'].start,
                                         'end': g['interval'].end} for g in overlapping],
                         'acceptedStarts': [{'originalProposedTime': p['originalProposedTime'], 'time': p['time'],
                                             'truthIndex': p['targetTruthIndex']}
                                            for p in edited['acceptedSplits'] if p['parentId'] == pid],
                         'children': [{'start': e['start'], 'end': e['end'], 'startObserved': e['startObserved'],
                                       'endObserved': e['endObserved'],
                                       'nonGoldSecondsRetained': iv.duration(iv.difference(support, valid_gold))}
                                      for e,support in zip(children, children_support)],
                         'originalIouByGold': original_iou.tolist(), 'childrenIouByGold': children_iou.tolist(),
                         'originalLocalMatchesAtIouHalf': len(original_matches),
                         'partitionLocalMatchesAtIouHalf': len(child_matches),
                         'localMatchCaveat': 'Illustrative parent-only matching; full-recording bipartite scores remain authoritative'})
            if len(rows) == 3:
                return rows
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    reg = read(args.root/'registration.json'); contract = reg['contract']
    require(hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == reg['sha256'], 'Registration changed')
    for identity in [*contract['sources'].values(), contract['input']]:
        require(sha(identity['path']) == identity['sha256'], 'Frozen dependency changed')
    report = read(args.root/'report.json')
    require(report['passed'] and report['contractSha256'] == reg['sha256'], 'Run is not complete')
    cells = []
    for row in report['seeds']:
        require(sha(row['result']['path']) == row['result']['sha256'], 'Frozen result changed')
        cells.append(read(row['result']['path']))
    data = read(contract['input']['path']); records = {r['id']: r for r in data['records']}
    rows = []
    for cell in cells:
        for arm in cell['automatic']:
            per = [classify(records[r['id']], r) for r in arm['splitMetrics']['recordings']]
            rows.append({'seed': cell['seed'], 'policy': arm['policy'], 'pooled': totals(per),
                         'sourceGroups': {g: totals([r for r in per if r['sourceGroup'] == g])
                                          for g in sorted({r['sourceGroup'] for r in per})},
                         'recordings': per})
    groups = sorted(rows[0]['sourceGroups'])
    aggregates = [{'policy': p, 'pooled': statistics([r['pooled'] for r in rows if r['policy'] == p]),
                   'sourceGroups': {g: statistics([r['sourceGroups'][g] for r in rows if r['policy'] == p]) for g in groups}}
                  for p in contract['splitPolicies']]
    result = {'kind': 'descriptive-post-hoc-frozen-split-error-diagnosis-v1',
              'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': reg['sha256'],
              'postHoc': True, 'newConfigurations': 0, 'policyOrThresholdChanges': False, 'outcomeRerun': False,
              'taxonomy': {'matchedSecondaryStarts': 'Exact one-to-one1s matches already reported by frozen split metric',
                           'nearFirstMaterialStart': 'Unmatched proposal within1s of first material-overlap eligible gold start in same parent/component; classified here first',
                           'nearOtherGoldStart': 'Remaining unmatched proposal within1s of another eligible gold start in same parent/component; can include duplicate/already-claimed or nonmaterial targets',
                           'awayAllEligibleGoldStarts': 'Remaining unmatched proposal farther than1s from every eligible gold start in same parent/component',
                           'limit': 'Near-first proposals may be start-correction advice but were applied as extra partitions; this diagnostic does not measure a corrected-start policy or prove useful splitting'},
              'registration': ref(args.root/'registration.json'), 'input': contract['input'],
              'resultReferences': [r['result'] for r in report['seeds']], 'script': ref(Path(__file__)),
              'policySeedRows': rows, 'aggregates': aggregates,
              'illustrativeAcceptedParents': interval_examples(data, next(c for c in cells if c['seed'] == 3407))}
    out = args.root/'split-error-diagnosis.json'
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'output': ref(out), 'aggregates': aggregates,
                      'illustrativeAcceptedParents': result['illustrativeAcceptedParents']}, indent=2))


if __name__ == '__main__':
    main()
