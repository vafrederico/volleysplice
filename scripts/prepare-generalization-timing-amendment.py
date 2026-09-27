#!/usr/bin/env python3
"""Register and qualify bounded nearest-frame handling for real source gaps."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import neural_generalization_feature_stage as original
from analysis import neural_generalization_gap_stage as amended
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_inputs import require, verified

ROOT = original.ROOT


def register():
    original.environment(ROOT)
    plan = original.verify_plan(ROOT/'stage-plan.json')
    diagnostics, completed, affected, examined = [], [], [], []
    for source in plan['records']:
        if source['reuse']:
            continue
        key = source['id']
        path = ROOT/'timing-diagnostics-v1'/key/'report.json'
        if path.exists():
            report = read(path)
            require(report['video'] == source['videoIdentity'] and report['labelsUsed'] is False
                    and report['outputsUsed'] is False and report['stagePlan'] == identity(ROOT/'stage-plan.json'),
                    'Label-blind timing diagnostic lineage differs')
            with np.load(verified(report['arrays']), allow_pickle=False) as values:
                times, ordinals = amended.pts_selection(values['pts'], report['duration'], 4.)
                require(np.array_equal(times, values['times']) and np.array_equal(ordinals, values['ordinals']),
                        'Amendment would change the original nearest-frame rule')
                mobile_times, mobile_ordinals = amended.pts_selection(values['pts'], report['duration'], 2.)
                require(np.array_equal(mobile_times, times[::2]) and np.array_equal(mobile_ordinals, ordinals[::2]),
                        'Amendment would change even-tick Mobile sampling')
            require(report['maximumSelectionErrorSeconds'] <= .25+1e-9, 'Source exceeds amended250ms bound')
            diagnostics.append(identity(path))
            if report['badTicks']:
                affected.append(key)
        else:
            path = ROOT/'staged'/key/'receipt.json'
            report = read(path)
            require(report['lineage'] == {'plan': identity(ROOT/'stage-plan.json'), 'source': source}
                    and report['allRequestedFramesDecoded'] is True, 'New source lacks timing qualification')
            with np.load(verified(report['outputs']['selection']), allow_pickle=False) as values:
                require(np.max(abs(values['selected_pts']-values['times'])) <= .125+1e-9,
                        'Completed strict source exceeds original tolerance')
            completed.append(identity(path))
        examined.append(key)
    require(len(examined) == 24 and affected, 'Expected all24 new sources and at least one real timing failure')
    code = {name: identity(REPO/name) for name in (
        'analysis/neural_generalization_gap_stage.py', 'scripts/prepare-generalization-timing-amendment.py',
        'analysis/tests/test_neural_generalization_gap_stage.py')}
    registration = {'kind': 'bounded-source-timing-amendment-v1', 'basePlan': identity(ROOT/'stage-plan.json'),
        'originalFullParity': identity(ROOT/'engineering-parity.json'), 'code': code,
        'diagnostics': diagnostics, 'strictCompletedSources': completed, 'allNewSourcesExamined': examined,
        'affectedRecordings': affected, 'maximumNearestErrorSeconds': .25,
        'sampling': 'Unchanged nearest native PTS, ties choose earlier frame; same4Hz pixels for AV/DINO and even2Hz ticks for Mobile.',
        'duplicates': 'Repeated selected ordinals reuse the exact decoded frame; record every duplicate tick.',
        'teacherPolicyUnchanged': True, 'labelMasksChanged': False, 'labelsUsedForAmendment': False,
        'selectionOrTrainingOutcomesUsed': False, 'noInterpolation': True,
        'missingnessCaveat': 'Camera gaps contain no new pixels. Repeated/offset frames retain actual PTS and existing Mobile quality offsets; AV/DINO gain no new missingness channel. These ticks remain valid and are not gold ignored intervals.',
        'preservation': 'Original stager and all prior outputs unchanged. An empty failed directory is moved intact to failed-staging-preserved-v1 before publishing a receipt pointing at new staged-timing-v1 arrays.',
        'qualification': 'Full unaffected fixed engineering recording must be array-bit-exact to original stager, including every selected RGB frame, AV104 value and Mobile/teacher image.'}
    folder = ROOT/'timing-amendment-v1'
    write_immutable(folder/'protocol.json', registration)
    for name, reference in code.items():
        path = folder/'registered-sources'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(Path(reference['path']).read_bytes())
    print(json.dumps({'registered': True, 'protocol': identity(folder/'protocol.json'), 'affected': affected}), flush=True)


def compare_npz(a, b):
    with np.load(verified(a), allow_pickle=False) as left, np.load(verified(b), allow_pickle=False) as right:
        require(set(left.files) == set(right.files), 'Amended archive inventory differs')
        for key in left.files:
            require(np.array_equal(left[key], right[key]), 'Amended unaffected source differs: '+key)


def qualify():
    plan = original.verify_plan(ROOT/'stage-plan.json')
    key = plan['engineeringRecord']
    before = read(ROOT/'staged'/key/'receipt.json')
    after = amended.stage_record(ROOT/'stage-plan.json', key)
    compare_npz(before['outputs']['audiovisual'], after['outputs']['audiovisual'])
    compare_npz(before['outputs']['selection'], after['outputs']['selection'])
    require(len(before['dinoRgbChunks']) == len(after['dinoRgbChunks']), 'Amended RGB chunk count differs')
    for left, right in zip(before['dinoRgbChunks'], after['dinoRgbChunks']):
        require(np.array_equal(np.load(verified(left), mmap_mode='r'), np.load(verified(right), mmap_mode='r')),
                'Amended unaffected RGB pixels differ')
    a, b = (read(verified(r['outputs']['imageInput'])) for r in (before, after))
    compare_npz(a['arrays']['timing'], b['arrays']['timing'])
    for key in ('images224', 'teaching336'):
        require(np.array_equal(np.load(verified(a['arrays'][key]), mmap_mode='r'),
                               np.load(verified(b['arrays'][key]), mmap_mode='r')), 'Amended unaffected image pixels differ')
    result = {'kind': 'bounded-source-timing-full-parity-v1', 'passed': True,
              'protocol': identity(ROOT/'timing-amendment-v1/protocol.json'),
              'originalReceipt': identity(ROOT/'staged'/plan['engineeringRecord']/'receipt.json'),
              'amendedReceipt': identity(ROOT/'staged-timing-v1'/plan['engineeringRecord']/'receipt.json'),
              'allSelectedPixelsAndFeatureArraysBitExact': True, 'labelsUsed': False}
    write_immutable(ROOT/'timing-amendment-v1/engineering-parity.json', result)
    print(json.dumps({'passed': True, 'report': identity(ROOT/'timing-amendment-v1/engineering-parity.json')}), flush=True)


def stage_affected():
    path = ROOT/'timing-amendment-v1/protocol.json'
    protocol = read(path)
    gate_path = ROOT/'timing-amendment-v1/engineering-parity.json'
    gate = read(gate_path)
    require(gate['passed'] is True and gate['protocol'] == identity(path), 'Amendment engineering gate absent')
    for key in protocol['affectedRecordings']:
        receipt = amended.stage_record(ROOT/'stage-plan.json', key)
        canonical = ROOT/'staged'/key
        original_failed = None
        if canonical.exists() and not (canonical/'receipt.json').exists():
            require(not list(canonical.iterdir()), 'Failed folder contains data requiring separate review')
            preserved = ROOT/'failed-staging-preserved-v1'/key
            require(canonical.resolve().is_relative_to((ROOT/'staged').resolve())
                    and preserved.resolve().is_relative_to(ROOT.resolve()) and not preserved.exists(),
                    'Failed-directory preservation target differs')
            preserved.parent.mkdir(exist_ok=True)
            canonical.rename(preserved)
            original_failed = {'originalPath': str(canonical), 'preservedPath': str(preserved), 'entriesBefore': []}
        write_immutable(canonical/'receipt.json', receipt)
        publication = {'protocol': identity(path), 'engineering': identity(gate_path), 'id': key,
                       'actualReceipt': identity(ROOT/'staged-timing-v1'/key/'receipt.json'),
                       'publishedReceipt': identity(canonical/'receipt.json'), 'preservedFailedDirectory': original_failed}
        write_immutable(ROOT/'timing-amendment-v1/publications'/(key+'.json'), publication)
        print(json.dumps({'amendedStagePublished': key, 'publication': publication}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'qualify', 'stage'))
    args = parser.parse_args()
    {'register': register, 'qualify': qualify, 'stage': stage_affected}[args.phase]()
