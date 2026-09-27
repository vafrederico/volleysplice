#!/usr/bin/env python3
"""Read-only provenance check of every historical logical inner score owner.

This complements the independent interval audit: it verifies declared seed,
original numerical contract, exact ordered tier membership, both excluded
sources, encoder membership, and archive/checkpoint ownership. No neural work.
"""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep as sweep
from analysis import neural_recall_sweep_historical as study
from analysis.neural_recall_sweep_adapters import bind


def audit(root):
    protocol = study.verify(root)
    layouts = sweep.read(root / 'layouts.json')['layouts']
    manifest = sweep.read(protocol['source']['path'])
    evidence, checks, owners = {}, [], set()
    bind(sweep.identity(root / 'protocol.json'), evidence)
    bind(sweep.identity(root / 'layouts.json'), evidence)
    bind(protocol['source'], evidence)
    for layout in layouts:
        bind(layout['registration'], evidence)
        registration = sweep.read(layout['registration']['path'])
        for fold in layout['folds']:
            for owner in fold['innerOwners']:
                folder = Path(owner['fitDirectory'])
                ref = sweep.identity(folder / 'completed.json')
                bind(ref, evidence)
                meta = sweep.read(ref['path'])
                excluded = {fold['heldSourceGroup'], owner['heldInnerSourceGroup']}
                rows = {tier: [r for r in manifest[tier + 'Rows'] if r['sourceGroup'] not in excluded]
                        for tier in ('exact', 'draft', 'coverage')}
                held = [r['id'] for r in manifest['exactRows'] if r['sourceGroup'] == owner['heldInnerSourceGroup']]
                expected = {'contractSha256': registration['sha256'], 'seed': layout['seed'],
                    'trainIds': [r['id'] for r in rows['exact']],
                    'auxiliaryIds': {tier: [r['id'] for r in rows[tier]] for tier in ('draft', 'coverage')},
                    'scalerTrainIds': [r['id'] for r in rows['exact']],
                    'trainGroups': sorted({r['sourceGroup'] for r in rows['exact']}),
                    'auxiliaryGroups': {tier: sorted({r['sourceGroup'] for r in rows[tier]}) for tier in ('draft', 'coverage')},
                    'epochs': list(sweep.EPOCHS), 'lossArm': 'short_boost'}
                sweep.require(all(meta.get(key) == value for key, value in expected.items()), 'Historical owner seed/source/objective differs: ' + str(folder))
                sweep.require(set(held) <= set(meta['validationIds']) and set(meta['validationGroups']) <= excluded,
                              'Logical inner view belongs to another held population')
                expected_held = [r['id'] for r in manifest['exactRows'] if r['sourceGroup'] in set(meta['validationGroups'])]
                sweep.require(meta['validationIds'] == expected_held, 'Owner held recording order differs')
                for epoch in sweep.EPOCHS:
                    for stem in ('weights', 'predictions'):
                        name = f'{stem}-{epoch}.npz'
                        bind({'path': str(folder / name), 'sha256': meta['artifacts'][name]}, evidence)
                for student_ref in owner['additionalTrainingOwners']:
                    student = sweep.read(bind(student_ref, evidence))
                    ids = [r['id'] for tier in ('exact', 'draft', 'coverage') for r in rows[tier]]
                    sweep.require(student['seed'] == layout['seed'] and student['contractSha256'] == registration['sha256']
                                  and student['excludedGroups'] == sorted(excluded) and student['trainIds'] == ids
                                  and student['trainGroups'] == sorted({r['sourceGroup'] for tier in rows.values() for r in tier}),
                                  'Historical student owner membership differs')
                    bind(student['weights'], evidence)
                owners.add(str(folder))
                checks.append({'model': layout['model'], 'seed': layout['seed'], 'outer': fold['heldSourceGroup'],
                               'inner': owner['heldInnerSourceGroup'], 'owner': ref, 'recordingIds': held,
                               'allEpochArtifactsChecked': True})
    sweep.require(len(checks) == 216 and len(owners) == 144, 'Historical logical/physical owner inventory changed')
    for ref in list(evidence.values()):
        bind(ref)
    return {'kind': 'independent-historical-score-owner-provenance-audit-v1', 'passed': True,
            'protocol': sweep.identity(root / 'protocol.json'), 'logicalInnerViews': len(checks), 'physicalInnerOwners': len(owners),
            'checks': checks, 'references': list(evidence.values()), 'auditor': sweep.identity(Path(__file__)),
            'trainingPerformed': False, 'gpuUsed': False, 'protectedTestOpened': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sweep.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')) and not args.output.exists(), 'Audit requires new NAS output')
    sweep.write_new(args.output, audit(args.study))
