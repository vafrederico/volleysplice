#!/usr/bin/env python3
"""Independent post-hoc taxonomy, interval-example and prose-number checks."""
from __future__ import annotations
from analysis.private_ledger import private_value

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0093'))
DOC = REPO/'docs/research/neural-split-advisor-results-2026-09-19.md'


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, REPO/relative)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


scalar = load('scripts/audit_neural_split_advisor.py', 'scalar_split_audit')
aggregate = load('scripts/audit-neural-split-summary.py', 'aggregate_split_audit')
require = scalar.require
read = aggregate.read
compare = aggregate.compare
ref = aggregate.ref
COUNTS = ('proposals', 'matchedSecondaryStarts', 'unmatched', 'nearFirstMaterialStart',
          'nearOtherGoldStart', 'awayAllEligibleGoldStarts')


def prepared(record):
    ignored = scalar.pairs(record['ignoredIntervals'])
    gold = scalar.pairs(record['rallies'])
    censored = [g for g in gold if scalar.intersection([g], ignored)]
    eligible = [(i, g) for i, g in enumerate(gold) if not scalar.intersection([g], ignored)]
    masks = scalar.union(ignored, censored)
    parents = {p['id']: (p, scalar.difference(scalar.pairs([p]), masks)) for p in record['productionEvents']}
    return eligible, parents, masks


def classify(record, frozen):
    gold, parents, _ = prepared(record)
    matched = {p['proposalIndex'] for p in frozen['splitLocalization']['1']['pairs']}
    counts = {k: 0 for k in COUNTS}
    counts.update(proposals=len(frozen['proposals']), matchedSecondaryStarts=len(matched),
                  unmatched=len(frozen['proposals']) - len(matched))
    examples = []
    for proposal in frozen['proposals']:
        if proposal['index'] in matched:
            continue
        parent, support = parents[proposal['parentId']]
        component = support[proposal['componentIndex']]
        material = [(i, g) for i, g in gold if scalar._overlap(g, component) >= min(.5, .1*(g[1] - g[0]))]
        first = material[0] if material else None
        nearby = [(i, g) for i, g in gold if component[0] <= g[0] < component[1]
                  and abs(g[0] - proposal['time']) <= 1.]
        category = ('nearFirstMaterialStart' if first is not None and first in nearby else
                    'nearOtherGoldStart' if nearby else 'awayAllEligibleGoldStarts')
        counts[category] += 1
        examples.append({**proposal, 'category': category, 'parentInterval': [parent['start'], parent['end']],
            'firstMaterialGold': {'truthIndex': first[0], 'start': first[1][0], 'end': first[1][1]} if first else None,
            'nearbyGold': [{'truthIndex': i, 'start': g[0], 'end': g[1], 'errorSeconds': proposal['time'] - g[0]}
                           for i, g in nearby]})
    require(counts['unmatched'] == sum(counts[k] for k in COUNTS[3:]), 'Taxonomy does not partition')
    return {'id': record['id'], 'sourceGroup': record['sourceGroup'], **counts, 'unmatchedProposals': examples}


def count_matches(matrix):
    # Tiny illustrative parent matrices; exhaustive assignments are independent
    # of the experiment's Hungarian matcher and sufficient for cardinality.
    def step(i, used):
        if i == len(matrix):
            return 0
        return max([step(i + 1, used)] + [1 + step(i + 1, used | {j})
                     for j, value in enumerate(matrix[i]) if j not in used and value >= .5])
    return step(0, set())


def verify_examples(records, cell, emitted):
    arm_id = 'head_evidence--split_only--evidence--budget-40'
    arm = next(x for x in cell['reviewed'] if x['id'] == arm_id)
    expected = []
    for edited in sorted(arm['perRecording'], key=lambda p: p['id']):
        record = records[edited['id']]
        gold, parents, masks = prepared(record)
        for pid in sorted({p['parentId'] for p in edited['acceptedSplits']}):
            parent, support = parents[pid]
            overlap_gold = [(i, g) for i, g in gold if scalar.intersection([g], support)]
            children = [e for e in edited['events'] if e['parentId'] == pid]
            child_support = [scalar.difference(scalar.pairs([e]), masks) for e in children]
            def iou(truth, covered):
                common = scalar.seconds(scalar.intersection([truth], covered))
                total = truth[1] - truth[0] + scalar.seconds(covered) - common
                return common / total if total else 0.
            original = [[iou(g, support)] for _, g in overlap_gold]
            after = [[iou(g, covered) for covered in child_support] for _, g in overlap_gold]
            expected.append({'seed': cell['seed'], 'arm': arm_id, 'recordingId': record['id'], 'parentId': pid,
                'originalParent': [parent['start'], parent['end']],
                'goldRallies': [{'truthIndex': i, 'start': g[0], 'end': g[1]} for i, g in overlap_gold],
                'acceptedStarts': [{'originalProposedTime': p['originalProposedTime'], 'time': p['time'],
                                    'truthIndex': p['targetTruthIndex']} for p in edited['acceptedSplits'] if p['parentId'] == pid],
                'children': [{'start': e['start'], 'end': e['end'], 'startObserved': e['startObserved'],
                              'endObserved': e['endObserved'],
                              'nonGoldSecondsRetained': scalar.seconds(scalar.difference(covered, [g for _, g in overlap_gold]))}
                             for e, covered in zip(children, child_support)],
                'originalIouByGold': original, 'childrenIouByGold': after,
                'originalLocalMatchesAtIouHalf': count_matches(original),
                'partitionLocalMatchesAtIouHalf': count_matches(after),
                'localMatchCaveat': 'Illustrative parent-only matching; full-recording bipartite scores remain authoritative'})
            if len(expected) == 3:
                compare(expected, emitted, 'Illustrative parents')
                return len(expected)
    compare(expected, emitted, 'Illustrative parents')
    return len(expected)


def audit_diagnosis(records, cells, diagnosis):
    rows = []
    for cell in cells:
        for arm in cell['automatic']:
            per = [classify(records[row['id']], row) for row in arm['splitMetrics']['recordings']]
            total = lambda subset: {key: sum(row[key] for row in subset) for key in COUNTS}
            rows.append({'seed': cell['seed'], 'policy': arm['policy'], 'pooled': total(per),
                'sourceGroups': {g: total([r for r in per if r['sourceGroup'] == g]) for g in sorted({r['sourceGroup'] for r in per})},
                'recordings': per})
    compare(rows, diagnosis['policySeedRows'], 'Taxonomy rows')
    for policy in diagnosis['aggregates']:
        chosen = [r for r in rows if r['policy'] == policy['policy']]
        for source, output in [('pooled', policy['pooled']), *[(g, v) for g, v in policy['sourceGroups'].items()]]:
            samples = [r['pooled'] if source == 'pooled' else r['sourceGroups'][source] for r in chosen]
            expected = {k: {'mean': math.fsum(r[k] for r in samples) / len(samples),
                            'min': min(r[k] for r in samples), 'max': max(r[k] for r in samples)} for k in COUNTS}
            compare(expected, output, 'Taxonomy aggregate')
    examples = verify_examples(records, next(c for c in cells if c['seed'] == 3407), diagnosis['illustrativeAcceptedParents'])
    require(diagnosis['postHoc'] and diagnosis['newConfigurations'] == 0 and not diagnosis['policyOrThresholdChanges']
            and not diagnosis['outcomeRerun'], 'Diagnosis claims policy changes')
    return {'passed': True, 'policySeedRows': len(rows), 'recordingRows': sum(len(r['recordings']) for r in rows),
            'classifiedProposals': sum(r['pooled']['proposals'] for r in rows), 'illustrativeParents': examples,
            'independentExampleMatching': 'exhaustive assignment cardinality'}


def audit_prose(summary, diagnosis, document, readme):
    executive = document.split('<!-- EXECUTIVE FINDINGS START -->', 1)[1].split('<!-- EXECUTIVE FINDINGS END -->', 1)[0].replace('**', '')
    arms = {a['id']: a for a in summary['reviewed']}
    auto = {a['policy']: a for a in summary['automatic']}
    checked = []
    def claims(label, paragraph_start, values):
        paragraph = next(p for p in executive.split('\n\n') if p.startswith(paragraph_start))
        for value, suffix, decimals in values:
            rendered = f'{value:.{decimals}f}' + suffix
            require(rendered in paragraph, f'{label}: missing or wrong prose value {rendered!r}')
            checked.append({'claim': label, 'rendered': rendered, 'numericValue': value})
    clean5, clean10, clean40 = [arms['none--cleanup_only--evidence--budget-' + b] for b in ('05', '10', '40')]
    claims('cleanup5', 'The useful low-effort', [(clean5['workload']['reviewSeconds']/60, ' minutes', 2),
        (clean5['workload']['reviewJobs'], ' parent jobs', 2), (clean5['workload']['reviewedTrueRallies'], ' real rallies', 2),
        (clean5['workload']['removedFalseParents'], ' wholly false', 2), (summary['baseline']['identity']['eventF1']*100, '%', 2),
        (clean5['identity']['eventF1']*100, '%', 2)])
    claims('cleanup10', 'The useful low-effort', [(clean10['workload']['reviewSeconds']/60, ' minutes', 2),
        (clean10['workload']['reviewJobs'], ' jobs', 0), (clean10['workload']['reviewedTrueRallies'], ' real rallies', 2),
        (clean10['workload']['removedFalseParents'], ' false event records', 2),
        *[(clean10['identity'][key]*100, suffix, 2) for key, suffix in [('eventPrecision', '% event precision'), ('eventRecall', '% recall'), ('eventF1', '% F1')]]])
    claims('cleanup40', 'The useful low-effort', [(clean40['workload']['reviewSeconds']/60, ' minutes', 2),
        (clean40['workload']['removedFalseParents'], ' false event records', 0), (clean40['identity']['eventF1']*100, '% event F1', 2)])
    require(clean40['workload']['reviewJobs'] == clean40['workload']['jobsAvailable'], 'Full cleanup queue not exhausted')
    cy = summary['cleanupYield']['pooled']['metrics']
    claims('cleanupInventory', 'Across the complete cleanup', [(cy['flaggedParents'], ' parents', 2),
        (cy['whollyFalseParents'], ' wholly false', 0), (cy['realOrMixedParents'], ' real or mixed', 2),
        (cy['whollyFalsePrecision']*100, '% mean wholly-false precision', 2)])
    for policy, label in [('event_starts', 'events'), ('head_evidence', 'heads'), ('corroborated', 'corroborated')]:
        rates = auto[policy]['split']['splitLocalization']['1']
        claims('automatic_' + label, 'Automatic splitting', [(rates['predicted'], ' proposals', 0 if policy != 'corroborated' else 2),
            (rates['precision']*100, '%', 2), (rates['recall']*100, '%', 2), (auto[policy]['identity']['eventF1']*100, '%', 2)])
    cor = auto['corroborated']; rates = cor['split']['splitLocalization']['1']
    claims('automatic_corroborated_extra', 'Automatic splitting', [(rates['matched'], ' match', 2),
        (rates['falsePositive'], ' are spurious', 2), (rates['f1']*100, '% split F1', 2),
        ((cor['identity']['eventF1'] - summary['baseline']['identity']['eventF1'])*100, ' points', 2),
        (cor['identitySeedStatistics']['eventF1']['min']*100, '', 2), (cor['identitySeedStatistics']['eventF1']['max']*100, '% seed range', 2)])
    split = arms['corroborated--split_only--evidence--budget-40']
    claims('corroborated_review', 'A small separate', [(split['workload']['reviewSeconds']/60, ' minutes', 2),
        (split['workload']['reviewJobs'], ' jobs', 2), (split['workload']['acceptedSplitCount'], ' proposed boundaries', 2),
        (split['workload']['rejectedSplitCount'], ',', 2), (split['identity']['mergedPredictionsMaterial'], ' without', 2),
        (split['identity']['eventF1']*100, '%', 2), (split['identity']['observedStartLocalization']['1']['recall']*100, '%', 2),
        (summary['baseline']['identity']['observedStartLocalization']['1']['recall']*100, '%', 2)])
    require(split['workload']['reviewJobs'] == split['workload']['jobsAvailable'], 'Full corroborated queue not exhausted')
    require(split['workload']['reviewedTrueRallies'] == 9 and split['identity']['splitTrueRalliesMaterial'] == 4, 'Split prose count differs')
    combined = arms['corroborated--combined--evidence--budget-40']
    heads = arms['head_evidence--combined--evidence--budget-40']
    combined10 = arms['corroborated--combined--evidence--budget-10']
    claims('combined', 'Combining corroborated', [(combined['workload']['reviewSeconds']/60, ' minutes', 2),
        (combined['workload']['reviewJobs'], ' jobs', 2), (combined['workload']['reviewedTrueRallies'], ' real rallies', 2),
        *[(combined['identity'][key]*100, suffix, 2) for key, suffix in [('eventPrecision', '% event precision'), ('eventRecall', '% recall'), ('eventF1', '% F1')]],
        (combined['workload']['removedFalseParents'], ' false event records', 0), (combined['workload']['acceptedSplitCount'], ' splits', 2),
        (heads['workload']['reviewSeconds']/60, ' minutes', 2), (heads['identity']['eventF1']*100, '% event F1', 2),
        (combined10['workload']['reviewSeconds']/60, ' minutes', 2)])
    require(combined10['workload']['acceptedSplitCount'] == 0, 'Combined10 accepted splits differs')
    require(summary['baseline']['split']['splitTargets'] == 7 and summary['baseline']['identity']['completeMisses'] == 3, 'Target/miss count differs')
    for row in diagnosis['aggregates']:
        values = row['pooled']
        for key in ('matchedSecondaryStarts', 'nearFirstMaterialStart', 'awayAllEligibleGoldStarts'):
            rendered = f"{values[key]['mean']:.2f}"
            require(rendered in executive.split('### Post-hoc error diagnosis', 1)[1], 'Diagnosis prose count missing')
            checked.append({'claim': 'diagnosis_' + row['policy'] + '_' + key, 'rendered': rendered})
        require(values['nearOtherGoldStart']['max'] == 0, 'Other-start prose zero differs')
    example = diagnosis['illustrativeAcceptedParents'][0]
    example_paragraph = next(p for p in executive.split('\n\n') if p.startswith('A concrete example'))
    for value in [*example['originalParent'], *[x[k] for x in example['goldRallies'] for k in ('start', 'end')],
                  *[x[k] for x in example['children'] for k in ('start', 'end')],
                  example['childrenIouByGold'][0][0], example['childrenIouByGold'][1][1]]:
        require(f'{value:.3f}' in example_paragraph, 'Example prose value missing')
    require(example['originalLocalMatchesAtIouHalf'] == example['partitionLocalMatchesAtIouHalf'] == 0, 'Example match count differs')
    paragraph = next(p for p in readme.split('\n\n') if 'neural-split-advisor-results-2026-09-19.md' in p)
    for value in [summary['baseline']['identity']['eventF1']*100, clean10['identity']['eventF1']*100,
                  clean10['workload']['reviewSeconds']/60, cor['split']['splitLocalization']['1']['precision']*100,
                  split['workload']['reviewSeconds']/60, split['workload']['acceptedSplitCount']]:
        require(f'{value:.2f}' in paragraph, 'README value missing')
    for phrase in ('perfect-human operations', 'development-exposure limits', 'not calibrated correctness probabilities',
                   'only inserts starts', 'their video remains in the unchanged export', 'did not create or tune a new automatic policy'):
        require(phrase in executive, 'Interpretation limit missing: ' + phrase)
    return {'passed': True, 'numericClaimsChecked': checked, 'readmeClaimsChecked': 6,
            'developmentAndIdealHumanLimitationsPresent': True}


def main():
    registration = read(ROOT/'registration.json'); report = read(ROOT/'report.json')
    summary = read(ROOT/'summary.json'); diagnosis = read(ROOT/'split-error-diagnosis.json')
    cells = [read(row['result']['path']) for row in report['seeds']]
    inputs = read(registration['contract']['input']['path']); records = {r['id']: r for r in inputs['records']}
    for item in [diagnosis['registration'], diagnosis['input'], diagnosis['script'], *diagnosis['resultReferences']]:
        aggregate.verify_ref(item)
    diagnostic_receipt = audit_diagnosis(records, cells, diagnosis)
    document = DOC.read_text(encoding='utf-8'); readme_path = REPO/'docs/research/README.md'
    prose_receipt = audit_prose(summary, diagnosis, document, readme_path.read_text(encoding='utf-8'))
    table_receipt = aggregate.audit_tables(summary, document)
    for item in registration['contract']['sources'].values():
        aggregate.verify_ref(item)
    receipt = {'passed': True, 'createdAt': datetime.now(timezone.utc).isoformat(),
        'contractSha256': registration['sha256'], 'summary': ref(ROOT/'summary.json'), 'diagnosis': ref(ROOT/'split-error-diagnosis.json'),
        'document': ref(DOC), 'readme': ref(readme_path), 'auditor': ref(Path(__file__)),
        'diagnosisAudit': diagnostic_receipt, 'proseAudit': prose_receipt, 'tableAudit': table_receipt,
        'frozenSourcesUnchanged': len(registration['contract']['sources'])}
    path = ROOT/'interpretation-audit-v1.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': ref(path), 'diagnosisAudit': diagnostic_receipt,
                      'proseClaims': len(prose_receipt['numericClaimsChecked']), 'tableLines': table_receipt['tableLinesVerified']}))


if __name__ == '__main__':
    main()
