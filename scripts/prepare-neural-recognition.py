#!/usr/bin/env python3
"""Write immutable label-free extraction inputs from the qualified development corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    records = []
    for tier in ('exactRows', 'draftRows', 'coverageRows'):
        for row in source[tier]:
            if row['environment'] == 'beach' or row['split'] == 'test' or row['sourceGroup'] in source['protectedSourceGroups']:
                raise ValueError('Excluded source in development extraction input')
            if row['consent'].get('train') is not True:
                raise ValueError('Training authorization missing')
            record = {key: row[key] for key in ('id', 'video', 'contentSha256', 'roi', 'sourceGroup', 'split', 'environment', 'durationSeconds')}
            cache = row['featureCaches']['audiovisual']
            record['avCache'] = {key: cache[key] for key in ('path', 'sha256')}
            records.append(record)
    if len(records) != 18 or len({r['id'] for r in records}) != 18:
        raise ValueError('Unexpected extraction population')
    result = {'schemaVersion': 1, 'kind': 'label-free-recognition-extraction-input-v1',
              'sourceManifest': {'path': str(args.source), 'sha256': sha(args.source)},
              'labelsUsed': False, 'protectedTestOpened': False, 'beachIncluded': False,
              'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) != result:
            raise ValueError('Refusing changed extraction manifest')
    else:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'path': str(args.output), 'sha256': sha(args.output), 'records': len(records)}))


if __name__ == '__main__':
    main()
