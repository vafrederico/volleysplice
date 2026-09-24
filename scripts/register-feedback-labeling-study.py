#!/usr/bin/env python3
"""Publish one verified human import with a derived, additive labeling catalog."""
from analysis.private_ledger import private_value
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(private_value('private-reference-0082'))
WORKSPACE = Path(private_value('private-reference-0102'))
NODE = private_value('private-reference-0101')


def ref(path):
    value = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(value).hexdigest(), 'sizeBytes': len(value)}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    imported = json.loads((root/'feedback-import/import-receipt.json').read_text())
    assert imported['passed'] and imported['labelSchemaValidated']
    for key in ('label', 'corpusRecord', 'immutableFeedbackCopy'):
        entry = imported[key]
        assert ref(Path(entry['path']))['sha256'] == entry['sha256'], key
    record = json.loads(Path(imported['corpusRecord']['path']).read_text())
    label_path = Path(imported['label']['path'])
    label = json.loads(label_path.read_text())
    assert record['recordingId'] == label['recording']['id']
    code = """import {readFileSync} from 'node:fs';
import {parseLabelDocument} from './lib/annotations.ts';
const doc=parseLabelDocument(JSON.parse(readFileSync(process.argv[1],'utf8')));
console.log(JSON.stringify({recordingId:doc.recording.id,rallies:doc.rallies.length,serves:doc.serveMarkers.length}));"""
    validation = subprocess.run([NODE, '--experimental-strip-types', '--input-type=module', '-e', code, str(label_path)],
                                cwd=REPO, check=True, capture_output=True, text=True)
    validated = json.loads(validation.stdout)
    original = WORKSPACE/'reports/full-nas-video-corpus-v3.json'
    original_ref = ref(original)
    catalog = json.loads(original.read_text())
    assert catalog['kind'] == 'volleycut-full-nas-video-corpus-v1'
    assert not any(row['recordingId'] == record['recordingId'] for row in catalog['records'])
    draft = WORKSPACE/'labels/full-v3'/f"{record['recordingId']}.labels.json"
    derived = root/'labeling-catalog.json'
    receipt = root/'labeling-registration.json'
    assert not any(path.exists() for path in (draft, derived, receipt)), 'Existing user data must be preserved'
    count = len(catalog['records'])
    catalog['records'].append(record)
    catalog['additiveRegistration'] = {'createdAt': datetime.now(timezone.utc).isoformat(),
        'baseCatalog': original_ref, 'addedRecordingId': record['recordingId'],
        'originalCatalogUnchanged': True, 'priorRecordsUnchanged': True}
    # Publish the editable draft first so the new task never falls back to AI labels.
    draft.parent.mkdir(parents=True, exist_ok=True)
    with draft.open('xb') as stream:
        stream.write(label_path.read_bytes())
    write_new(derived, catalog)
    assert ref(original) == original_ref
    assert json.loads(derived.read_text())['records'][:-1] == json.loads(original.read_text())['records']
    result = {'passed': True, 'recordingId': record['recordingId'], 'originalCount': count,
        'registeredCount': len(catalog['records']), 'nodeSchemaValidation': validated,
        'baseCatalog': original_ref, 'catalog': ref(derived), 'humanDraft': ref(draft),
        'immutableImportedLabel': imported['label'], 'source': ref(Path(__file__)),
        'originalCatalogUnchanged': True, 'existingHumanLabelsOverwritten': False,
        'trainingManifestChanged': False}
    write_new(receipt, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
