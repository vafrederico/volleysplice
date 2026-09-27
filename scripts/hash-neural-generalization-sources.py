#!/usr/bin/env python3
"""Bind full source bytes for label-blind feature expansion on NAS."""
from pathlib import Path
import argparse
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_feature_stage import environment, ROOT
from analysis.neural_generalization_inputs import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args()
    root = environment(ROOT)/'source-identities-v1'
    root.mkdir(parents=True, exist_ok=True)
    inventory = read(args.inventory)
    legacy = read(ROOT/'reference-qualification-v1/inputs.json')
    old = {r['id']: r for r in legacy['records']}
    for row in inventory['records']:
        if row['environment'] == 'beach':
            continue
        source = old.get(row['id'], row)
        video = Path(source['video'])
        before = video.stat()
        path = root/(row['id']+'.json')
        if path.exists():
            receipt = read(path)
            require(receipt['video']['path'] == str(video)
                    and receipt['sourceSizeBytes'] == before.st_size
                    and receipt['sourceMtimeNs'] == before.st_mtime_ns, 'Previously hashed source changed')
        else:
            bound = identity(video)
            after = video.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source changed while hashing')
            expected = source.get('contentSha256')
            require(expected is None or bound['sha256'] == expected, 'Full source SHA differs from prior identity')
            receipt = {'id': row['id'], 'video': bound, 'sourceSizeBytes': before.st_size,
                       'sourceMtimeNs': before.st_mtime_ns, 'priorFullHashPresent': expected is not None,
                       'inventory': identity(args.inventory), 'legacyOverride': row['id'] in old,
                       'labelsUsed': False, 'source': identity(__file__)}
            write_immutable(path, receipt)
        print(json.dumps({'hashed': row['id'], 'sizeBytes': before.st_size,
                          'sha256': receipt['video']['sha256']}), flush=True)


if __name__ == '__main__':
    main()
