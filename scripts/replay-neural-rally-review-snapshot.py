#!/usr/bin/env python3
"""Verify an extracted rally-review archive and optionally replay frozen cells.

Original absolute NAS paths are provenance only. All reads use verified archive
entries; unchanged numerical runner.execute() writes only into a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys


CONTRACT = '7386abe4befe96ba4c9607454f5f6936b312dd8459c5dae1a453d4cadc93440d'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_entry(root, name):
    value = PurePosixPath(name)
    require(name and value.as_posix() == name and not value.is_absolute() and '..' not in value.parts
            and ':' not in name and '\\' not in name, 'Unsafe archive entry')
    path = root.joinpath(*value.parts)
    require(path.is_file() and path.resolve().is_relative_to(root.resolve())
            and not any(p.is_symlink() for p in [path, *path.parents]), 'Unsafe/missing archive file')
    return path


def verify_snapshot(root):
    root = Path(root).resolve()
    manifest_path = safe_entry(root, 'snapshot-manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['kind'] == 'rally-review-proposals-reproducible-snapshot-v1'
            and manifest['contractSha256'] == CONTRACT, 'Unexpected snapshot identity')
    entries = {}
    for ref in manifest['files']:
        name = ref['archivePath']
        require(name not in entries, 'Duplicate archive file')
        path = safe_entry(root, name)
        require(path.stat().st_size == ref['sizeBytes'] and sha(path) == ref['sha256'], 'Changed archive bytes: '+name)
        entries[name] = ref
    registration = json.loads(safe_entry(root, 'study/registration.json').read_text(encoding='utf-8'))
    require(canonical(registration['contract']) == registration['sha256'] == CONTRACT, 'Registration contract differs')
    for relative, ref in registration['contract']['sources'].items():
        entry = entries['repository/'+relative]
        require(entry['sha256'] == ref['sha256'] and entry['sizeBytes'] == ref['sizeBytes'], 'Registered source differs')
    for key, name in [('input', 'rally-review-input.json'), ('probabilities', 'probabilities.npz'),
                      ('protocol', 'protocol-initial.md'), ('qualification', 'qualification-v1.json'),
                      ('goldSemantics', 'gold-semantics-audit-v1.json')]:
        ref, entry = registration['contract'][key], entries['study/'+name]
        require(entry['sha256'] == ref['sha256'] and entry['sizeBytes'] == ref['sizeBytes'], 'Registered input differs')
    return manifest, registration, entries


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def run(root, output=None, configuration=None, seed=None, all_cells=False):
    root = Path(root).resolve()
    manifest, registration, entries = verify_snapshot(root)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.dont_write_bytecode = True
    import numpy as np
    require(np.__version__ == manifest['environment']['numpyVersion'],
            'Use the archived NumPy version for exact numerical replay')
    repository = root/'repository'
    sys.path.insert(0, str(repository))
    runner = module('frozen_rally_review_runner', repository/'scripts/run-neural-rally-review-proposals.py')
    import analysis
    require(Path(analysis.__file__).resolve().is_relative_to(repository), 'Imported a different analysis package')
    data = json.loads((root/'study/rally-review-input.json').read_text(encoding='utf-8'))
    with np.load(root/'study/probabilities.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for entry in data['entries']:
        a, t = arrays[entry['scoresKey']], arrays[entry['timesKey']]
        require(a.dtype == np.float32 and t.dtype == np.float64 and a.shape == (len(t), 4)
                and hashlib.sha256(a.tobytes()).hexdigest() == entry['scoresBytesSha256'], 'Archived score bytes differ')
    for row in data['records']:
        t = arrays[row['timesKey']]
        require(hashlib.sha256(t.tobytes()).hexdigest() == row['timelineBytesSha256'], 'Archived timestamp bytes differ')
    receipt = {'passed': True, 'mode': 'verified', 'contractSha256': CONTRACT,
               'filesVerified': len(entries), 'registeredSourcesVerified': len(registration['contract']['sources']),
               'normalizedScoreArraysVerified': len(data['entries']), 'importsResolvedFromArchive': True,
               'numpyVersion': np.__version__, 'replayedCells': []}
    if configuration is None and not all_cells:
        require(output is None and seed is None, 'Output/seed require --configuration or --all')
        print(json.dumps(receipt, indent=2))
        return receipt
    require(output is not None and (all_cells or seed in registration['contract']['seeds']), 'Replay output/seed required')
    configs = registration['contract']['configurations']
    require(all_cells or configuration in {c['id'] for c in configs}, 'Unknown configuration')
    require(not all_cells or configuration is None and seed is None, '--all excludes configuration/seed')
    output = Path(output).absolute()
    require(not output.exists() and not output.resolve().is_relative_to(root), 'Replay needs a new directory outside snapshot')
    output.mkdir(parents=True, exist_ok=False)
    auditor = module('frozen_rally_review_auditor', repository/'scripts/audit-neural-rally-review-proposals.py')
    candidate = module('frozen_rally_candidate_auditor', repository/'scripts/audit-neural-rally-review-candidates.py')
    accounting = module('frozen_rally_accounting_auditor', repository/'scripts/audit-neural-combination-accounting.py')
    runner._STATE = (output, registration, data, arrays, auditor, candidate, accounting)
    tasks = [(c, s) for c in configs for s in registration['contract']['seeds']
             if all_cells or c['id'] == configuration and s == seed]
    for config, current_seed in tasks:
        result = runner.execute((config, current_seed))
        archive_name = 'study/results/'+Path(result['path']).name
        expected = entries[archive_name]
        require(result['sha256'] == expected['sha256'] and result['sizeBytes'] == expected['sizeBytes'],
                'Replayed result bytes differ from frozen result: '+archive_name)
        receipt['replayedCells'].append({'configuration': config['id'], 'seed': current_seed,
                                         'exactBytesMatched': True, **result})
        print(f"REPLAYED {config['id']} seed {current_seed}; exact bytes match", flush=True)
    receipt['mode'] = 'replayed'
    with (output/'replay-verification.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--configuration')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--all', action='store_true', dest='all_cells')
    args = parser.parse_args()
    run(args.snapshot, args.output, args.configuration, args.seed, args.all_cells)
