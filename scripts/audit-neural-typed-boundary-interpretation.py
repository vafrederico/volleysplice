#!/usr/bin/env python3
"""Bind the final narrative to independently audited typed-boundary results."""
from __future__ import annotations
from analysis.private_ledger import private_value

from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0079'))
DOC = REPO/'docs/research/neural-typed-boundary-results-2026-09-19.md'
README = REPO/'docs/research/README.md'
SPEC = importlib.util.spec_from_file_location('typed_boundary_summary_audit', REPO/'scripts/audit-neural-typed-boundary-summary.py')
audit = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(audit)
require, compare, read, ref = audit.require, audit.compare, audit.read, audit.ref


def audit_exec_tables(summary, executive):
    automatic = {row['policy']: row for row in summary['automatic']}
    reviews = {row['id']: row for row in summary['reviewed']}
    display = audit.a.display
    labels = [('Production', summary['baseline']), ('Correct first start only', automatic['first_start']),
              ('Correct first start and add subsequent starts', automatic['typed_starts']),
              ('Give each proposed rally its own end', automatic['separate_ends']),
              ('Refine those starts and ends with neural heads', automatic['head_refined'])]
    expected = audit.a.table_lines(['Event timeline', 'Rally precision %', 'Rally recall %', 'Rally F1 %',
             'Raw core recall %', 'Additional completely missed rallies', 'Core play removed from rally records (s)'],
        [[name, *[display(row['identity'][k], 100) for k in ('eventPrecision', 'eventRecall', 'eventF1')],
          display(row['coverage']['resultRawCoreRecall'], 100), display(row['coverage']['additionalCompleteMisses']),
          display(row['coverage']['rawCoreSecondsLostFromBaseline'])] for name, row in labels])
    rows = []
    for budget in ('05', '10', '20', '40'):
        proposed = reviews['head_refined--proposal_confirmation--evidence--budget-' + budget]
        full = reviews['head_refined--full_parent--evidence--budget-' + budget]
        rows.append([str(int(budget)) + '%', display(proposed['workload']['reviewSeconds'], 1/60),
            display(proposed['workload']['reviewJobs']), display(proposed['workload']['reviewedTrueRallies']),
            ' / '.join(display(proposed['identity'][key], 100) for key in ('eventPrecision', 'eventRecall', 'eventF1')),
            ' / '.join(display(full['identity'][key], 100) for key in ('eventPrecision', 'eventRecall', 'eventF1'))])
        for key in ('reviewSeconds', 'reviewJobs', 'reviewedTrueRallies'):
            compare(proposed['workload'][key], full['workload'][key], 'Paired human workload')
    expected += audit.a.table_lines(['Per-video cap', 'Playback min', 'Parent jobs', 'Real rallies reviewed',
                                    'Proposal-confirmation P / R / F1 %', 'Full-parent correction P / R / F1 %'], rows)
    actual = [line for line in executive.splitlines() if line.startswith('|')]
    compare(expected, actual, 'Executive numeric tables')
    return {'passed': True, 'executiveTableLinesVerified': len(expected)}


def audit_claims(summary, document, readme, qualification):
    executive = document.split('<!-- EXECUTIVE FINDINGS START -->', 1)[1].split('<!-- EXECUTIVE FINDINGS END -->', 1)[0].replace('**', '')
    auto = {row['policy']: row for row in summary['automatic']}
    reviews = {row['id']: row for row in summary['reviewed']}
    base = summary['baseline']; head = auto['head_refined']; ends = auto['separate_ends']
    checked = []
    def values(label, paragraph_start, specs):
        paragraph = next(p for p in executive.split('\n\n') if p.startswith(paragraph_start))
        for value, suffix, decimals in specs:
            rendered = f'{value:.{decimals}f}' + suffix
            require(rendered in paragraph, f'{label}: missing/wrong prose value {rendered!r}')
            checked.append({'claim': label, 'value': value, 'rendered': rendered})
    require(qualification['testsRun'] == 88 and qualification['passed'], 'Qualification claim differs')
    require(summary['automaticOutcomes'] == 12 and summary['reviewedOutcomes'] == 192, 'Outcome claims differ')
    require(len(summary['recordings']) == 8 and base['identity']['trueRallies'] == 322, 'Scope claims differ')
    values('headAutomatic', 'These are seed means', [
        (head['identitySeedStatistics']['eventF1']['min']*100, '%', 2), (head['identitySeedStatistics']['eventF1']['max']*100, '%', 2),
        (base['identity']['observedStartLocalization']['1']['recall']*100, '%', 2),
        (head['identity']['observedStartLocalization']['1']['recall']*100, '%', 2),
        (base['identity']['observedEndLocalization']['1']['recall']*100, '%', 2),
        (head['identity']['observedEndLocalization']['1']['recall']*100, '%', 2),
        (head['coverage']['rawCoreSecondsLostFromBaseline'], ' seconds', 2)])
    rates = head['typed']['byType']; unrefined = ends['typed']['byType']
    values('typedCandidates', 'First-start candidates', [
        *[(rates[kind]['1'][metric]*100, '%', 2) for kind in ('initial_start', 'additional_start') for metric in ('precision', 'recall')],
        (rates['additional_start']['1']['predicted'], ' observed candidates', 2),
        (rates['additional_start']['1']['matched'], ' matches', 2), (rates['initial_start']['1']['true'], ' initial targets', 0),
        (unrefined['initial_start']['1']['precision']*100, '%', 2), (unrefined['additional_start']['1']['precision']*100, '%', 2),
        (ends['typed']['samePairBoundaries']['1']['f1']*100, '%', 2), (head['typed']['samePairBoundaries']['1']['f1']*100, '%', 2)])
    require(rates['additional_start']['1']['true'] == 7, 'Seven additional targets claim differs')
    restricted40 = reviews['head_refined--proposal_confirmation--evidence--budget-40']
    values('restricted40loss', 'Proposal confirmation accepts', [(restricted40['coverage']['rawCoreSecondsLostFromBaseline'], ' seconds', 2)])
    require(restricted40['coverage']['additionalCompleteMisses'] == 2, 'Restricted40 additional misses differs')
    for budget in ('05', '10', '20'):
        arm = reviews['head_refined--proposal_confirmation--evidence--budget-' + budget]
        require(arm['coverageSeedStatistics']['rawCoreSecondsLostFromBaseline']['max'] == 0
                and arm['coverageSeedStatistics']['additionalCompleteMisses']['max'] == 0, 'Lower-budget no-loss claim differs')
    for arm in reviews.values():
        if arm['humanMode'] == 'full_parent':
            require(arm['coverageSeedStatistics']['rawCoreSecondsLostFromBaseline']['max'] == 0
                    and arm['coverageSeedStatistics']['additionalCompleteMisses']['max'] == 0, 'Full-parent no-loss claim differs')
    values('reviewPolicyComparison', 'Full-parent review explicitly', [
        (reviews[policy + '--proposal_confirmation--evidence--budget-10']['identity']['eventF1']*100, '%', 2)
        for policy in ('first_start', 'separate_ends', 'head_refined')])
    full10 = [r for r in reviews.values() if r['humanMode'] == 'full_parent' and r['ranker'] == 'evidence' and r['budgetFraction'] == .1]
    require(f"{min(r['identity']['eventF1'] for r in full10)*100:.2f}-{max(r['identity']['eventF1'] for r in full10)*100:.2f}%" in executive,
            'Full-parent policy F1 range differs')
    values('fixedExport', 'Every arm retains exactly', [
        *[(base['primary'][key]*100, '%', 2) for key in ('P_pad', 'R_core', 'F1_padP_coreR')],
        *[(base['primary'][key]/60, ' minutes', 2) for key in ('paddedModelExportSeconds', 'correctlyRemovedSeconds',
          'incorrectlyRemovedSeconds', 'incorrectExportSeconds')]])
    values('recallDefinitions', 'Raw core recall measures', [
        (base['coverage']['resultRawCoreRecall']*100, '%', 2), (base['padding'][0]['metrics']['R_core']*100, '%', 2),
        (base['typed']['targetCount'], ' parent-specific targets', 0), (base['typed']['inaccessibleTargetStarts'], ' inaccessible starts', 0),
        (base['typed']['inaccessibleTargetEnds'], ' inaccessible ends', 0)])
    require(base['identity']['completeMisses'] == 3, 'Three existing complete misses differs')
    text = next(p for p in readme.split('\n\n') if 'neural-typed-boundary-results-2026-09-19.md' in p)
    full = reviews['head_refined--full_parent--evidence--budget-10']
    for value in [base['identity']['eventF1']*100, head['identity']['eventF1']*100,
                  head['coverage']['rawCoreSecondsLostFromBaseline'], full['identity']['eventF1']*100,
                  full['workload']['reviewSeconds']/60, full['workload']['reviewJobs']]:
        require(f'{value:.2f}' in text, 'README metric differs')
    require(full['workload']['reviewedTrueRallies'] == 82 and '82 real rallies' in text, 'README reviewed count differs')
    for phrase in ('did not retrain the network or test DINO transfer', 'ideal human actions', 'not measured reviewer performance',
                   'not only actionable corrections', 'not calibrated correctness probabilities',
                   'cannot discover an unproposed rally', 'event F1 is a separate diagnostic'):
        require(phrase in executive, 'Required interpretation distinction absent: ' + phrase)
    return {'passed': True, 'numericClaims': checked, 'readmeNumericClaims': 7,
            'allFullParentConfigurationsPreserveCore': True, 'restrictedHumanLossesNotHidden': True}


def audit_restricted_loss_attribution(data, cells):
    records = {r['id']: r for r in data['records']}; scalar = audit.b.scalar; rows = []
    for cell in cells:
        arm = next(r for r in cell['reviewed'] if r['id'] == 'head_refined--proposal_confirmation--evidence--budget-40')
        removed_core = removed_missing_core = 0.; missed = 0
        for output, metric in zip(arm['perRecording'], arm['typedMetrics']['recordings']):
            record = records[output['id']]
            old = scalar.difference(scalar.pairs(record['productionEvents']), scalar.pairs(record['ignoredIntervals']))
            new = scalar.difference(scalar.pairs(output['events']), scalar.pairs(record['ignoredIntervals']))
            lost = scalar.difference(old, new)
            core = scalar.difference(scalar.pairs(record['rallies']), scalar.pairs(record['ignoredIntervals']))
            removed_core += scalar.seconds(scalar.intersection(lost, core))
            for index in metric['additionalCompleteMissIndexes']:
                missed += 1
                removed_missing_core += scalar.seconds(scalar.intersection(lost, scalar.pairs([record['rallies'][index]])))
        compare(removed_core, removed_missing_core, 'Restricted loss attributable to newly missed rallies')
        require(missed == 2, 'Restricted40 newly missed rally count differs')
        rows.append({'seed': cell['seed'], 'additionalCompleteMisses': missed,
                     'allLostCoreSeconds': removed_core, 'newlyMissedRallyCoreLostSeconds': removed_missing_core})
    return {'passed': True, 'seedRows': rows}


def audit_censoring(data, cells, diagnosis, resolution, document):
    """Reconstruct saved post-hoc cells from raw events, not diagnostic code."""
    import numpy as np
    records = {row['id']: row for row in data['records']}
    metadata = {}
    for record in records.values():
        audit.a.verify_ref(record['featureCache'])
        with np.load(record['featureCache']['path'], allow_pickle=False) as cache:
            fps = float(json.loads(str(cache['metadata_json']))['fps'])
        metadata[record['id']] = {'fps': fps, 'frameDurationSeconds': 1 / fps}
    compare(metadata, resolution['frameMetadata'], 'Frame metadata')
    compare([r['featureCache'] for r in records.values()], resolution['metadataSources'], 'Frame sources')
    for reference in [diagnosis['input'], diagnosis['script'], resolution['script'], resolution['sourceDiagnosis'],
                      *diagnosis['resultReferences']]:
        audit.a.verify_ref(reference)
    require(diagnosis['postHoc'] and diagnosis['newConfigurations'] == 0 and not diagnosis['outcomeRerun'], 'Diagnosis scope')
    require(resolution['postHoc'] and not resolution['frozenMetricsChanged'] and not resolution['outcomeRerun'], 'Resolution scope')
    rows, bins = [], []
    for cell in cells:
        base = cell['baseline']['identityMetrics']['pooled']
        for mode in ('proposal_confirmation', 'full_parent'):
            arm = next(r for r in cell['reviewed'] if r['id'] == f'head_refined--{mode}--evidence--budget-10')
            examples = []
            for output in arm['perRecording']:
                record = records[output['id']]
                _, eligible, _, masks, _ = audit.b.typed_prepared(record)
                eligible_ids = {i for i, _ in eligible}
                parents = {p['id']: p for p in record['productionEvents']}
                for event in output['events']:
                    if event.get('startObserved', True) or not audit.b.difference(audit.b.pairs([event]), masks):
                        continue
                    gi = event.get('targetTruthIndex', event.get('humanGoldIndex'))
                    require(gi is not None, 'Clipped event must have linked gold')
                    gold = record['rallies'][gi]
                    parent = parents[event.get('parentId', event['id'])]
                    require(event['start'] == parent['start'] > gold['start'], 'Unobserved start is not original-parent clipping')
                    error = event['start'] - gold['start']
                    examples.append({'recordingId': record['id'], 'eventId': event['id'], 'parentId': parent['id'],
                        'start': event['start'], 'end': event['end'], 'truthIndex': gi, 'goldStart': gold['start'],
                        'eligibleGold': gi in eligible_ids, 'startErrorSeconds': error, 'parentClipped': True,
                        'ignoredClipped': any(abs(x['end']-event['start']) < 1e-9 for x in record['ignoredIntervals']),
                        'withinOneSecondOfOwnGold': error <= 1,
                        'startSource': event.get('startSource', event.get('startKind'))})
            identity = arm['identityMetrics']['pooled']
            ordinary = identity['startLocalization']['1']; observed = identity['observedStartLocalization']['1']
            row = {'seed': cell['seed'], 'mode': mode, 'armId': arm['id'], 'trueRallies': identity['trueRallies'],
                'baselineOrdinaryMatched1s': base['startLocalization']['1']['matched'],
                'baselineObservedMatched1s': base['observedStartLocalization']['1']['matched'],
                'ordinaryMatched1s': ordinary['matched'], 'observedMatched1s': observed['matched'],
                'ordinaryRecall1s': ordinary['recall'], 'observedRecall1s': observed['recall'],
                'ordinaryMatchGain1s': ordinary['matched']-base['startLocalization']['1']['matched'],
                'observedMatchChange1s': observed['matched']-base['observedStartLocalization']['1']['matched'],
                'correctMatchesExcludedByObservationFlag1s': ordinary['matched']-observed['matched'],
                'unobservedStartCount': len(examples), 'parentClippedStartCount': len(examples),
                'ignoredClippedStartCount': sum(x['ignoredClipped'] for x in examples),
                'clippedWithin1sOfOwnGoldCount': sum(x['withinOneSecondOfOwnGold'] for x in examples),
                'otherUnobservedCount': 0,
                'ordinaryMatchedQuarterSecond': identity['startLocalization']['0.25']['matched'],
                'observedMatchedQuarterSecond': identity['observedStartLocalization']['0.25']['matched'],
                'baselineMatchedQuarterSecond': base['startLocalization']['0.25']['matched'], 'unobservedStarts': examples}
            rows.append(row)
            correct = [x for x in examples if x['withinOneSecondOfOwnGold']]
            require(len(correct) == ordinary['matched']-observed['matched'], 'Clip counts do not explain matching delta')
            tiny = sum(x['startErrorSeconds'] <= .001+1e-12 for x in correct)
            frames = sum(x['startErrorSeconds'] <= metadata[x['recordingId']]['frameDurationSeconds']+1e-12 for x in correct)
            bins.append({'seed': cell['seed'], 'mode': mode, 'unobservedStarts': len(examples),
                'correctMatchesRemoved1s': len(correct), 'offsetAtMost1ms': tiny, 'offsetAtMostOneFrame': frames,
                'offsetOver1msAtMostOneFrame': frames-tiny, 'offsetOverOneFrameAtMost1s': len(correct)-frames,
                'offsetOver1sNotCorrectAtTolerance': len(examples)-len(correct),
                'examples': [{**x, **metadata[x['recordingId']]} for x in examples]})
    compare(rows, diagnosis['policySeedRows'], 'Independent clipping rows')
    compare(bins, resolution['policySeedRows'], 'Independent resolution bins')
    def aggregate(values):
        return {mode: {key: {'mean': math.fsum(r[key] for r in values if r['mode'] == mode)/3,
                            'min': min(r[key] for r in values if r['mode'] == mode),
                            'max': max(r[key] for r in values if r['mode'] == mode)}
                      for key, value in values[0].items() if type(value) in (int, float) and key != 'seed'}
                for mode in ('proposal_confirmation', 'full_parent')}
    first, second = aggregate(rows), aggregate(bins)
    compare(first, diagnosis['aggregates'], 'Clipping aggregate stats')
    compare(second, resolution['aggregates'], 'Resolution aggregate stats')
    paragraph = next(p for p in document.replace('**', '').split('\n\n') if p.startswith('There is an important serve-start'))
    for mode in first:
        for key in ('ordinaryRecall1s', 'observedRecall1s'):
            require(f"{100*first[mode][key]['mean']:.2f}%" in paragraph, 'Censoring prose recall differs')
        for value in (first[mode]['correctMatchesExcludedByObservationFlag1s']['mean'], second[mode]['offsetAtMost1ms']['mean']):
            require(f'{value:.2f}' in paragraph, 'Censoring prose count differs')
    require('within one millisecond' in paragraph and 'did not permit edits outside the raw parent' in paragraph,
            'Censoring limitation not explicit')
    return {'passed': True, 'seedModeCells': len(rows), 'examplesVerified': sum(len(r['unobservedStarts']) for r in rows),
            'metadataSourcesVerified': len(metadata), 'allExcludedStartsOriginalParentClipped': True,
            'ordinaryMinusObservedMatchesExactlyExplained': True, 'aggregates': first, 'resolutionAggregates': second}


def main():
    reg = read(ROOT/'registration.json'); summary = read(ROOT/'summary.json'); report = read(ROOT/'report.json')
    summary_receipt = read(ROOT/'summary-audit-v1.json')
    require(summary_receipt['passed'], 'Summary independent audit missing')
    audit.a.verify_ref(summary_receipt['summary'])
    cells = []
    for row in report['seeds']:
        audit.a.verify_ref(row['result']); cells.append(read(row['result']['path']))
    doc = DOC.read_text(encoding='utf-8'); readme = README.read_text(encoding='utf-8')
    executive = doc.split('<!-- EXECUTIVE FINDINGS START -->', 1)[1].split('<!-- EXECUTIVE FINDINGS END -->', 1)[0]
    tables = audit_exec_tables(summary, executive)
    claims = audit_claims(summary, doc, readme, read(reg['contract']['qualification']['path']))
    data = read(reg['contract']['input']['path'])
    attribution = audit_restricted_loss_attribution(data, cells)
    censoring = audit_censoring(data, cells, read(ROOT/'observed-start-diagnosis.json'),
                               read(ROOT/'clipped-start-resolution-diagnosis.json'), doc)
    generated = audit.audit_tables(summary, doc)
    for item in [*reg['contract']['sources'].values(), *reg['contract']['sourceCopies'].values()]:
        audit.a.verify_ref(item)
    receipt = {'passed': True, 'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': reg['sha256'],
               'summary': ref(ROOT/'summary.json'), 'summaryAudit': ref(ROOT/'summary-audit-v1.json'),
               'document': ref(DOC), 'readme': ref(README), 'auditor': ref(Path(__file__)),
               'executiveTableAudit': tables, 'executiveClaimsAudit': claims,
               'restrictedLossAttribution': attribution, 'generatedTableAudit': generated,
               'censoringAudit': censoring, 'observedStartDiagnosis': ref(ROOT/'observed-start-diagnosis.json'),
               'clippingResolutionDiagnosis': ref(ROOT/'clipped-start-resolution-diagnosis.json'),
               'frozenSourcesUnchanged': len(reg['contract']['sources'])}
    path = ROOT/'interpretation-audit-v1.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': ref(path), 'executiveClaims': len(claims['numericClaims']),
                      'executiveTableLines': tables['executiveTableLinesVerified'], 'generatedTableLines': generated['generatedTableLinesVerified']}))


if __name__ == '__main__':
    main()
