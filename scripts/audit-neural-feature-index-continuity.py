#!/usr/bin/env python3
"""Verify that complete precision publication only adds references to the core panel."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(private_value('private-reference-0089'))
REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def checked(reference):
    actual = identity(reference['path'])
    require(actual['sha256'] == reference['sha256'], 'Bound file changed: '+actual['path'])
    return actual


def indexed(rows, key):
    values = {row[key]: row for row in rows}
    require(len(values) == len(rows), 'Duplicate recording identity')
    return values


def compare_publications(core, complete, core_audit, complete_audit, core_ref, complete_ref):
    require(core.get('kind') == complete.get('kind') == 'neural-generalization-features-v1', 'Wrong feature index kind')
    require(core.get('publicationMode') == 'core' and complete.get('publicationMode') == 'complete', 'Wrong publication modes')
    require(core.get('publicationScope') == complete.get('publicationScope') == 'all', 'Wrong publication scope')
    require([r['recordingId'] for r in core['records']] == [r['recordingId'] for r in complete['records']],
            'Published recording order changed')
    old, new = deepcopy(core), deepcopy(complete)
    old_rows = indexed(old.pop('records'), 'recordingId')
    new_rows = indexed(new.pop('records'), 'recordingId')
    old.pop('publicationMode'); new.pop('publicationMode')
    require(old == new, 'Shared index contract changed')
    require(set(old_rows) == set(new_rows) and len(old_rows) == 42, 'Full42 recording scope changed')
    added = []
    for key in sorted(old_rows):
        prior, current = old_rows[key], new_rows[key]
        old_dino = prior['features'].pop('dino')
        new_dino = current['features'].pop('dino')
        require(prior == current, 'Existing nonprecision feature/source/staging reference changed: '+key)
        require('fp32' in old_dino and set(new_dino) == {'fp32', 'fp16', 'int8'}, 'Precision coverage differs: '+key)
        require(set(old_dino) <= set(new_dino), 'Previously published precision arm removed: '+key)
        require(all(new_dino[arm] == reference for arm, reference in old_dino.items()), 'Previously published precision reference changed: '+key)
        added.append({'id': key, 'addedPrecisions': sorted(set(new_dino)-set(old_dino))})
    for audit, mode, reference in ((core_audit, 'core', core_ref), (complete_audit, 'complete', complete_ref)):
        require(audit.get('kind') == 'independent-expansion-feature-audit-v1' and audit.get('passed') is True,
                'Independent feature audit did not pass')
        require(audit.get('mode') == mode and audit.get('scope') == 'all' and audit.get('auditedRecordingCount') == 42,
                'Independent feature audit scope differs')
        require(audit.get('features') == reference, 'Independent feature audit binds another index')
        require(audit.get('stagePlan') == core['stagePlan'], 'Independent feature audit staging differs')
    left, right = deepcopy(core_audit), deepcopy(complete_audit)
    require([r['id'] for r in left['records']] == [r['id'] for r in right['records']], 'Audited recording order changed')
    left_rows = indexed(left.pop('records'), 'id')
    right_rows = indexed(right.pop('records'), 'id')
    require(left_rows == right_rows and set(left_rows) == set(old_rows), 'Audited source/timing/teacher associations changed')
    for audit in (left, right):
        audit.pop('mode'); audit.pop('features')
    require(left == right, 'Independent feature audit shared inputs or qualification bindings changed')
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == ROOT and Path('/mnt/freenas').is_mount(), 'Expected mounted NAS study root')
    paths = [root/name for name in ('features-core.json', 'features-complete.json',
                                   'audit-features-core.json', 'audit-features-complete.json')]
    refs = [identity(path) for path in paths]
    core, complete, core_audit, complete_audit = [json.loads(path.read_text()) for path in paths]
    added = compare_publications(core, complete, core_audit, complete_audit, refs[0], refs[1])
    require(core_audit['source'] == identity(REPO/'scripts/audit-neural-generalization-features.py'),
            'Feature audit source is not the frozen repository auditor')
    # Existing independent audits perform array verification; this additive audit checks their
    # current identities and every shared metadata binding without reloading large image arrays.
    common_refs = []
    for field in ('inputs', 'auditPlan', 'stagePlan', 'cpuEngineering', 'encoderEngineering', 'source'):
        common_refs.append(checked(core_audit[field]))
    common_refs.append(checked(core['encoderPlan']))
    require([identity(path) for path in paths] == refs, 'Index or audit changed during continuity inspection')
    output = root/'audit-core-complete-continuity-v1.json'
    report = {'kind': 'feature-index-core-complete-continuity-audit-v1', 'passed': True,
        'source': identity(__file__), 'coreIndex': refs[0], 'completeIndex': refs[1],
        'coreAudit': refs[2], 'completeAudit': refs[3], 'sharedBindings': common_refs,
        'recordingCount': 42, 'records': added, 'existingReferenceChanges': 0,
        'onlyAddedPrecisionReferences': True, 'sameGoldInputRevision': core_audit['inputs'],
        'scope': 'Exhaustive index/reference and audit identity comparison. Array checks are bound from the two passing independent feature audits.',
        'fp32RerunRequired': False, 'precisionRuntimeLimitationsUnchanged': True}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'report': identity(output)}), flush=True)


if __name__ == '__main__':
    main()
