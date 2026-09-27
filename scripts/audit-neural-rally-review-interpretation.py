#!/usr/bin/env python3
"""Verify final review prose evidence and every generated Markdown data row."""
from __future__ import annotations
from analysis.private_ledger import private_value

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0092'))
DOC = REPO/'docs/research/neural-rally-review-proposals-results-2026-09-19.md'
README = REPO/'docs/research/README.md'
PROTOCOL = REPO/'docs/research/neural-rally-review-proposals-protocol-2026-09-19.md'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sizeBytes': path.stat().st_size}


def number(value):
    return 'n/a' if value is None else f'{value:.2f}'


def percent(value):
    return 'n/a' if value is None else f'{100*value:.2f}'


def quality(row):
    p, e, w = row['primary'], row['identity'], row.get('workload', {})
    serve = e['observedStartLocalization']['1']
    return [row['id'], *[percent(p[k]) for k in ('P_pad', 'R_core', 'F1_padP_coreR')],
            *[percent(e[k]) for k in ('eventPrecision', 'eventRecall', 'eventF1')],
            *[percent(serve[k]) for k in ('precision', 'recall', 'f1')],
            number(w.get('reviewSeconds', 0)/60), number(w.get('proposalsSelected', 0)),
            number(w.get('reviewedTrueRallies', 0)),
            'yes' if row.get('guardrailScreen', {}).get('passed') else ('baseline' if not w else 'no')]


def generated_rows(summary):
    output = [quality(x) for x in summary['automaticBaselines']]
    lookup = {x['id']: x for x in summary['arms']}
    for ids in summary['rankingsByDeclaredBudgetAndMode'].values():
        output.extend(quality(lookup[x]) for x in ids)
    for row in summary['arms']:
        output.append([row['id'], *[number(row['identity'][k]) for k in ('predictedRallies', 'completeMisses',
                       'mergedPredictions', 'splitTrueRallies', 'mergedPredictionsMaterial', 'splitTrueRalliesMaterial')],
                       *[number(row['workload'][k]) for k in ('censoredStarts', 'censoredEnds', 'unobservedStarts',
                         'unobservedEnds', 'touchedRalliesWithUneditableBoundary')]])
    for row in summary['arms']:
        output.append([row['id'], percent(row['primaryDeltas']['F1_padP_coreR']),
                       *[percent(row['identityDeltas'][k]) for k in ('eventF1', 'eventRecall', 'observedStart1sF1', 'observedStart1sRecall')],
                       *[number(row['identityDeltas'][k]) for k in ('completeMisses', 'mergedPredictionsMaterial', 'splitTrueRalliesMaterial')]])
    for row in summary['arms']:
        w = row['workload']
        output.append([row['id'], percent(row['budgetFraction']), percent(w['reviewFractionOfVideo']),
                       *[number(w[k]/60) for k in ('reviewSeconds', 'unusedBudgetSeconds', 'editSeconds')],
                       *[number(w[k]) for k in ('proposalsSelected', 'proposalsAvailable', 'decisionRegions',
                                               'reviewClips', 'reviewedTrueRallies', 'playbackTrueRallies')]])
    for row in [*summary['automaticBaselines'], *summary['arms']]:
        for pad in row['padding']:
            p = pad['metrics']
            output.append([row['id'], str(pad['paddingSecondsBeforeAndAfter']),
                           *[percent(p[k]) for k in ('P_pad', 'R_core', 'F1_padP_coreR')],
                           *[number(p[k]/60) for k in ('paddedModelExportSeconds', 'paddedHumanExportSeconds',
                               'exportDurationDifferenceSeconds', 'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds')],
                           number(p['missedCoreSeconds'])])
    for row in summary['arms']:
        for group, data in row['sourceGroups'].items():
            output.append([row['id'], group, *[percent(data['primary'][k]) for k in ('P_pad', 'R_core', 'F1_padP_coreR')],
                           percent(data['identity']['eventF1']),
                           *[percent(data['identity']['observedStartLocalization']['1'][k]) for k in ('precision', 'recall', 'f1')],
                           number(data['workload']['reviewSeconds']/60), number(data['workload']['reviewedTrueRallies'])])
    return output


def main():
    summary = read(ROOT/'summary.json')
    audit = read(ROOT/'summary-audit-v1.json')
    require(audit['passed'] and audit['summary'] == ref(ROOT/'summary.json'), 'Independent summary audit binding differs')
    require(audit['report'] == ref(ROOT/'report.json'), 'Audited report changed')
    text = DOC.read_text(encoding='utf-8')
    executive = text.split('<!-- ROOT EXECUTIVE FINDINGS START -->')[1].split('<!-- ROOT EXECUTIVE FINDINGS END -->')[0]
    body = text.split('<!-- ROOT EXECUTIVE FINDINGS END -->')[1]
    ids = {x['id'] for x in [*summary['automaticBaselines'], *summary['arms']]}
    actual = [[x.strip() for x in line.strip().strip('|').split('|')] for line in body.splitlines() if line.startswith('| ')]
    actual = [row for row in actual if row[0] in ids]
    expected = generated_rows(summary)
    require(actual == expected, 'At least one generated Markdown data row differs from immutable summary')
    baselines = {x['id']: x for x in summary['automaticBaselines']}
    arms = {(x['mode'], x['model'], x['inventory'], x['ranker'], x['budgetFraction']): x for x in summary['arms']}

    def arm(mode, model, inventory, budget, ranker='evidence'):
        return arms[mode, model, inventory, ranker, budget]

    # Every numerical executive example is captured as evidence, independently
    # rounded from the already audited per-seed summary.
    examples = []
    for mode, model, inventory, budget, ranker in (
            ('individual', 'dino_global', 'legacy', .1, 'evidence'),
            ('individual', 'dino_global', 'local_events', .1, 'evidence'),
            ('individual', 'compact_boost', 'legacy', .1, 'evidence'),
            ('individual', 'compact_boost', 'local_events', .1, 'evidence'),
            ('individual', 'dino_boost', 'legacy', .1, 'evidence'),
            ('individual', 'dino_boost', 'local_events', .1, 'evidence'),
            ('production', 'dino_global', 'local_events', .4, 'evidence'),
            ('production', 'dino_global', 'legacy', .4, 'evidence'),
            ('production', 'compact_boost', 'local_events', .4, 'evidence'),
            ('production', 'compact_boost', 'local_heads', .4, 'evidence'),
            ('production', 'compact_boost', 'local_heads', .1, 'evidence'),
            ('individual', 'dino_global', 'local_heads', .4, 'evidence'),
            ('individual', 'dino_global', 'local_heads', .4, 'chronological'),
            ('individual', 'dino_global', 'local_events', .4, 'evidence')):
        x = arm(mode, model, inventory, budget, ranker)
        examples.append({'id': x['id'], 'exportPRF': [percent(x['primary'][k]) for k in ('P_pad', 'R_core', 'F1_padP_coreR')],
                         'eventF1': percent(x['identity']['eventF1']),
                         'observedStartPRF': [percent(x['identity']['observedStartLocalization']['1'][k]) for k in ('precision', 'recall', 'f1')],
                         'playbackMinutes': number(x['workload']['reviewSeconds']/60),
                         'editRegions': number(x['workload']['decisionRegions']),
                         'trueRallies': number(x['workload']['reviewedTrueRallies']),
                         'completeMisses': number(x['identity']['completeMisses']),
                         'materialMerges': number(x['identity']['mergedPredictionsMaterial']),
                         'materialSplits': number(x['identity']['splitTrueRalliesMaterial'])})
    counts = {}
    for mode in ('production', 'individual'):
        for inventory in ('local_events', 'local_heads'):
            differences = []
            for model in ('compact_boost', 'dino_global', 'dino_boost'):
                for budget in (.05, .1, .2, .4):
                    new, old = arm(mode, model, inventory, budget), arm(mode, model, 'legacy', budget)
                    differences.append((new['identity']['eventF1']-old['identity']['eventF1'],
                                        new['identity']['observedStartLocalization']['1']['recall']-old['identity']['observedStartLocalization']['1']['recall']))
            counts[mode+'--'+inventory] = {'eventF1Improved': sum(x[0] > 1e-12 for x in differences),
                                         'observedStartRecallImproved': sum(x[1] > 1e-12 for x in differences),
                                         'observedStartRecallDecreased': sum(x[1] < -1e-12 for x in differences),
                                         'observedStartRecallTied': sum(abs(x[1]) <= 1e-12 for x in differences)}
    ranker_head_export_losses = ranker_local_event_wins = 0
    for model in ('compact_boost', 'dino_global', 'dino_boost'):
        for budget in (.05, .1, .2, .4):
            h, hc = arm('production', model, 'local_heads', budget), arm('production', model, 'local_heads', budget, 'chronological')
            e, ec = arm('production', model, 'local_events', budget), arm('production', model, 'local_events', budget, 'chronological')
            ranker_head_export_losses += h['primary']['F1_padP_coreR'] < hc['primary']['F1_padP_coreR'] - 1e-12
            ranker_local_event_wins += e['identity']['eventF1'] > ec['identity']['eventF1'] + 1e-12
    require(counts['individual--local_events']['eventF1Improved'] == counts['individual--local_heads']['eventF1Improved'] == 12,
            '24 individual event improvements claim differs')
    require(counts['production--local_events']['eventF1Improved'] == 12
            and counts['production--local_events']['observedStartRecallDecreased'] == 11
            and counts['production--local_events']['observedStartRecallTied'] == 1,
            'Production local-event paired claim differs')
    require(counts['production--local_heads']['observedStartRecallImproved'] == 12,
            'Production head-guided start recall claim differs')
    require(ranker_head_export_losses == ranker_local_event_wins == 12, 'Ordering comparison claim differs')
    require(sum(x['guardrailScreen']['passed'] for x in summary['arms']) == 144, '144 passing descriptive screens claim differs')
    new, old = arm('individual', 'dino_global', 'local_events', .1), arm('individual', 'dino_global', 'legacy', .1)
    deltas = {'exportF1Points': number(100*(new['primary']['F1_padP_coreR']-old['primary']['F1_padP_coreR'])),
              'eventF1Points': number(100*(new['identity']['eventF1']-old['identity']['eventF1'])),
              'playbackMinutesSaved': number((old['workload']['reviewSeconds']-new['workload']['reviewSeconds'])/60)}
    require(deltas == {'exportF1Points': '0.87', 'eventF1Points': '4.42', 'playbackMinutesSaved': '1.25'}, 'Headline deltas differ')
    for phrase in ('not binary keep/remove', 'not reviewer wall-clock labor', 'not equal-realized-time',
                   'not established as a better overall sorter', 'Neither is serving-side or score accuracy',
                   'no single uniformly better review policy', 'protected test was not opened'):
        require(phrase.lower() in executive.lower(), 'Required interpretation caveat missing: '+phrase)
    primary_leader = summary['rankingsByDeclaredBudgetAndMode']['individual--budget-40'][0]
    require(primary_leader == arm('individual', 'dino_global', 'local_heads', .4, 'chronological')['id'], 'Primary leader differs')
    figures = read(ROOT/'figures/manifest.json')
    require(figures['summary']['sha256'] == ref(ROOT/'summary.json')['sha256'], 'Figure summary binding differs')
    receipt = {'passed': True, 'kind': 'independent-rally-review-interpretation-audit-v1',
               'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': summary['contractSha256'],
               'registration': ref(ROOT/'registration.json'), 'report': ref(ROOT/'report.json'),
               'summary': ref(ROOT/'summary.json'), 'summaryAudit': ref(ROOT/'summary-audit-v1.json'),
               'document': ref(DOC), 'readme': ref(README), 'protocol': ref(PROTOCOL),
               'figuresManifest': ref(ROOT/'figures/manifest.json'), 'sourceScript': ref(Path(__file__)),
               'generatedMarkdownDataRowsVerified': len(expected), 'executiveExamples': examples,
               'pairedComparisonCounts': counts, 'headlineDeltas': deltas,
               'productionHeadRankerExportLosses': ranker_head_export_losses,
               'productionLocalEventRankerEventWins': ranker_local_event_wins,
               'descriptiveScreenPasses': 144, 'manualNarrativeReview': 'All displayed executive values, action limits, event/serve proxy distinctions and recommendations reviewed against the audit evidence; no score-tracking readiness claim.'}
    destination = ROOT/'interpretation-audit-v1.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': ref(destination), 'finalDocument': ref(DOC),
                      'generatedMarkdownDataRowsVerified': len(expected), 'examples': examples,
                      'pairedCounts': counts, 'headlineDeltas': deltas}), flush=True)


if __name__ == '__main__':
    main()
