#!/usr/bin/env python3
"""Verify detailed report tables against the independently audited summary.

Executive prose is additionally reviewed by the independent agent. This script
preserves that review scope and exact document identity with table checks.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(private_value('private-reference-0091'))
REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO/'docs/research/neural-perfect-human-review-results-2026-09-19.md'


def identity(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'sizeBytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value):
    return f'{value:.2f}'


def percent(value):
    return number(100*value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit():
    summary, audit_receipt = read(ROOT/'summary.json'), read(ROOT/'summary-audit-v1.json')
    require(audit_receipt['passed'] and audit_receipt['summary'] == identity(ROOT/'summary.json'), 'Numerical audit missing/changed')
    initial_document = identity(DOCUMENT)
    rows = {r['id']: r for r in summary['configurations']}
    legacy = {r['id']: r for r in summary['legacy']}
    counts, seen = Counter(), set()
    section = None
    for line in DOCUMENT.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            section = line[3:]
        if not line.startswith('| `'):
            continue
        actual = [part.strip() for part in line.strip('|').split('|')]
        key = actual[0].strip('`')
        if section == 'Complete human-assisted quality and workload matrix':
            p = rows[key]['primary']
            expected = [actual[0], percent(p['automatic']['F1_padP_coreR'])]
            expected += [percent(p[mode][field]) for mode in ('binary', 'boundary') for field in ('P_pad', 'R_core', 'F1_padP_coreR')]
            expected += [number(p['workload']['reviewSeconds']/60)]
            expected += [number(p['workload'][field]) for field in ('flaggedCandidates', 'reviewedTrueRallies', 'reviewClips')]
            item = section, key
        elif section == 'Decision composition and missed-rally guardrails':
            r = rows[key]
            fields = ('flaggedPositiveCandidates', 'flaggedNegativeCandidates', 'binaryKeptCandidates',
                      'binaryDroppedCandidates', 'mixedCandidates', 'playbackTrueRallies', 'completeRallyLossesBinary',
                      'partialRallyLossesBinary', 'completeRallyLossesBoundary', 'partialRallyLossesBoundary')
            expected = [actual[0], *[number(r['primary']['workload'][field]) for field in fields], percent(r['binaryEventF1'])]
            item = section, key
        elif section == 'Export time accounting at target padding':
            mode = actual[1]
            require(mode in ('binary', 'boundary'), 'Unknown human-action table mode')
            p = rows[key]['primary'][mode]
            expected = [actual[0], mode, *[number(p[field]/60) for field in (
                'paddedModelExportSeconds', 'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds', 'incorrectExportSeconds')],
                number(p['missedCoreSeconds'])]
            item = section, key, mode
        elif section == 'Prior fine-grained export-disagreement reference':
            p = legacy[key]['primary']
            expected = [actual[0], *[percent(p['boundary'][field]) for field in ('P_pad', 'R_core', 'F1_padP_coreR')],
                        number(p['workload']['reviewSeconds']/60), *[number(p['workload'][field]) for field in (
                            'decisionSegments', 'reviewedTrueRallies', 'reviewClips')]]
            item = section, key
        elif section == 'Required 0/1/2/3-second padding sensitivity':
            pad, mode = int(actual[1]), actual[2]
            require(pad in (0, 1, 2, 3) and mode in ('binary', 'boundary'), 'Sensitivity scope differs')
            p = next(x for x in rows[key]['padding'] if x['paddingSecondsBeforeAndAfter'] == pad)
            expected = [actual[0], str(pad), mode, *[percent(p[mode][field]) for field in ('P_pad', 'R_core', 'F1_padP_coreR')],
                        *[number(p[mode][field]) for field in ('paddedModelExportSeconds', 'paddedHumanExportSeconds',
                                                            'exportDurationDifferenceSeconds')],
                        number(p['workload']['reviewSeconds']/60)]
            item = section, key, pad, mode
        else:
            raise ValueError('Unexpected numerical report table section: '+str(section))
        require(actual == expected, f'Document values differ in {item}: {actual} != {expected}')
        require(item not in seen, 'Duplicate report table row')
        seen.add(item)
        counts[section] += 1
    expected_counts = {'Complete human-assisted quality and workload matrix': 120,
        'Decision composition and missed-rally guardrails': 120, 'Export time accounting at target padding': 240,
        'Prior fine-grained export-disagreement reference': 30, 'Required 0/1/2/3-second padding sensitivity': 960}
    require(dict(counts) == expected_counts, 'Detailed table row inventory differs')
    require(identity(DOCUMENT) == initial_document, 'Document changed during review')
    receipt = {'kind': 'independent-perfect-human-review-interpretation-audit-v1', 'passed': True,
        'createdAt': datetime.now(timezone.utc).isoformat(), 'contractSha256': summary['contractSha256'],
        'finalDocument': initial_document, 'summary': identity(ROOT/'summary.json'),
        'summaryAudit': identity(ROOT/'summary-audit-v1.json'), 'report': identity(ROOT/'report.json'),
        'implementation': identity(Path(__file__).resolve()), 'protectedTestOpened': False,
        'tableRowsVerified': sum(counts.values()), 'tableRowCounts': dict(counts),
        'executiveNarrativeReview': {'method': 'Independent manual comparison against unrounded immutable summary values',
            'checked': ['Guarded compact 10.5-second playback / one candidate / zero true rallies / 6.983-second binary FP savings',
                'Compact zero-support workload and P/R/F1; 17.5376667-second missed core versus 17.4716667 baseline',
                'DINO half-support and bidirectional-half workload, positive/negative composition and P/R/F1',
                'Compact/DINO broad bidirectional perfect-editing quality and workload',
                'All three individually highlighted positive-uncertainty model results',
                'DINO medium/wide search quality, candidate composition and footage fractions',
                'Full-positive and full-footage references distinguished from selective review',
                'Binary keep/remove versus perfect boundary correction; no exact recall-preservation claim from rounded equality',
                'Candidate decisions, distinct raw-region true rallies and playback clips kept distinct',
                'Uncalibrated existing-score wrapper; no learned needs-review head or measured device latency',
                'Development exposure, hypothetical human outcomes, no production change and no protected test',
                'Zero grid-sliver claim matches independent numerical audit'], 'requiredCorrections': []},
        'failures': []}
    destination = ROOT/'interpretation-audit-v1.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'receipt': identity(destination), 'document': initial_document,
                      'tableRowsVerified': sum(counts.values())}, indent=2))


if __name__ == '__main__':
    audit()
