#!/usr/bin/env python3
"""Prepare one feedback-backed human labeling task without publishing a catalog."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.annotations import load_label_document
from analysis.exported_project_dataset import sampled_fingerprint, sha256_file
from analysis.feedback_label_import import human_layers
from analysis.features import probe_video


def ref(path):
    return {'path': str(path), 'sha256': sha256_file(path), 'sizeBytes': path.stat().st_size}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedback', type=Path, required=True)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--source-group', required=True)
    args = parser.parse_args()
    feedback = args.feedback.resolve(); root = args.study_root.resolve()
    output = root/'feedback-import'
    if output.exists():
        raise FileExistsError(f'Refusing to replace prepared import: {output}')
    source_bytes = feedback.read_bytes(); payload = json.loads(source_bytes)
    if payload.get('schema') != 'volleycut-model-feedback' or payload.get('schemaVersion') != 3:
        raise ValueError('Expected feedback schema v3')
    source = payload['source']; info = source['file']; video = (feedback.parent/info['name']).resolve()
    if source['timelineCoordinates'] != 'seconds-from-start-of-source':
        raise ValueError('Unknown timeline coordinate system')
    if video.stat().st_size != info['sizeBytes'] or sampled_fingerprint(video) != info['sampledFingerprint']:
        raise ValueError('Source video size or sampled fingerprint mismatch')
    metadata = probe_video(video)
    duration = source['media']['duration']
    if abs(metadata.duration-duration) > .1:
        raise ValueError('Source duration mismatch')
    layers = human_layers(payload)
    video_ref = ref(video); feedback_ref = ref(feedback)
    identifier = f'raw-no-backup-{video.stem}'
    now = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    source_copy = output/feedback.name
    with source_copy.open('xb') as stream: stream.write(source_bytes)
    provenance = {'kind': 'human-reviewed-feedback-import-v1', 'createdAt': now,
        'sourceFeedback': feedback_ref, 'immutableFeedbackCopy': ref(source_copy), 'sourceVideo': video_ref,
        'sourceSampledFingerprint': info['sampledFingerprint'], 'generatedAt': payload.get('generatedAt'),
        'correctionsUpdatedAt': payload['corrections'].get('updatedAt'), 'author': 'user-requested-feedback-import',
        'humanReviewScope': 'reviewed production export selections and corrected score markers',
        'independentEndpointGold': False, 'llmVideoLabelingUsed': False, 'trainingAuthorizedByImport': False,
        'sourceTimeline': 'seconds-from-start-of-source', 'normalizationApplied': False,
        'importDetails': layers['importDetails'],
        'sources': [ref(Path(__file__)), ref(REPO/'analysis/feedback_label_import.py'),
                    ref(REPO/'analysis/exported_project_dataset.py')]}
    label_path = output/'human-labels'/f'{identifier}.labels.json'
    label = {'schemaVersion': 1, 'kind': 'volleycut-rally-labels', 'createdAt': now,
        'recording': {'id': identifier, 'video': str(video), 'videoFilename': video.name,
            'contentSha256': video_ref['sha256'], 'durationSeconds': duration, 'sourceGroup': args.source_group,
            'split': 'challenge', 'environment': args.environment, 'roi': source.get('featureRoi'),
            'game': {'playersPerTeam': None, 'targetPoints': None, 'format': args.environment},
            'capture': {'sourceTimeline': 'seconds-from-start-of-source', 'variableFrameRatePreserved': True}},
        'annotationPolicy': {'id': 'serve-contact-to-dead-ball-v1', 'rallyStart': 'serve-ball contact',
            'rallyEnd': 'first instant live play has ended',
            'intervalConvention': 'half-open [start,end) seconds on the original source video',
            'importLimitation': 'Imported ranges are reviewed export cores, pending independent endpoint review.'},
        'annotation': {'status': 'in-progress', 'annotator': 'user-requested-feedback-import',
            'continuousVideoReviewed': False, 'reviewedAt': None, 'humanReviewImported': True,
            'notes': 'Human layer imported from the user-reviewed project. All manual changes are retained; '
                     'independent serve-contact/dead-ball endpoint labeling is not asserted.'},
        **{key: layers[key] for key in ('rallies', 'ignoredIntervals', 'hardNegatives', 'serveMarkers', 'sideSwitches')},
        'importProvenance': provenance}
    write(label_path, label)
    # Source content was hashed above; avoid rereading 4+GB merely to validate schema.
    document = load_label_document(label_path, require_complete=False, require_video=False)
    baseline_path = output/'original-production-inference.json'
    write(baseline_path, payload['initialInference'])
    reference_path = output/'normalized-reference.json'
    write(reference_path, {'provenance': provenance, 'annotations': layers['normalizedReference']})
    record = {'recordingId': identifier, 'environment': args.environment, 'sourceGroup': args.source_group,
        'split': 'challenge', 'sourceType': 'human-reviewed-project-feedback', 'targetStatus': 'reviewed-export-coverage',
        'labelPath': str(label_path), 'labelSha256': sha256_file(label_path), 'videoPath': str(video),
        'videoFilename': video.name, 'videoSha256': video_ref['sha256'], 'durationSeconds': duration,
        'roi': source.get('featureRoi'), 'priority': 5,
        **{key: layers[key] for key in ('rallies', 'ignoredIntervals', 'serveMarkers', 'sideSwitches')},
        'candidateSource': {'kind': 'user-requested-human-feedback-import', 'feedbackPath': str(source_copy),
            'feedbackSha256': feedback_ref['sha256'], 'independentEndpointGold': False,
            'annotationStatus': 'in-progress', 'continuousVideoReviewed': False},
        'productionBaseline': ref(baseline_path)}
    record_path = output/'corpus-record.json'; write(record_path, record)
    receipt_path = output/'import-receipt.json'
    receipt = {'passed': True, **provenance, 'label': ref(label_path), 'corpusRecord': ref(record_path),
        'normalizedReference': ref(reference_path), 'productionBaseline': ref(baseline_path),
        'counts': {key: len(layers[key]) for key in ('rallies', 'ignoredIntervals', 'hardNegatives', 'serveMarkers', 'sideSwitches')},
        'labelSchemaValidated': True, 'sourceVideoFingerprintValidated': True, 'sourceVideoFullSha256Computed': True,
        'catalogPublished': False, 'warnings': list(document.warnings)}
    if feedback.read_bytes() != source_bytes:
        raise ValueError('Feedback changed during import')
    write(receipt_path, receipt)
    print(json.dumps({'passed': True, 'receipt': ref(receipt_path), 'counts': receipt['counts'], 'label': str(label_path)}))


if __name__ == '__main__': main()
