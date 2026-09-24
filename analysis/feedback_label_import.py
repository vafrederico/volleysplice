"""Import reviewed project corrections into a human labeling layer.

Coverage is not promoted to independently verified serve-to-dead-ball gold.
All coordinates remain source seconds, without frame-rate conversion or snapping.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .exported_project_dataset import normalize_feedback_annotations


def subtract(start: float, end: float, masks: list[dict[str, Any]]) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for mask in masks:
        revised = []
        for left, right in pieces:
            if mask['end'] <= left or mask['start'] >= right:
                revised.append((left, right))
            else:
                if left < mask['start']:
                    revised.append((left, mask['start']))
                if mask['end'] < right:
                    revised.append((mask['end'], right))
        pieces = revised
    return pieces


def human_layers(payload: dict[str, Any]) -> dict[str, Any]:
    """Use final export membership, corrected cores and exact corrected markers."""
    # Stale score markers on suppressed cuts must not be aligned onto surviving
    # ranges by the older weak-label importer (which can create duplicate times).
    normalization_input = deepcopy(payload)
    active = {identifier for row in payload['finalExportIntervals'] for identifier in row['cutIds']}
    known = {row['id'] for row in payload['corrections']['correctedRanges']}
    state_input = normalization_input['corrections']['scoreTracking']['state']
    removed_input = set(state_input.get('removedModelMarkerIds', []))
    state_input['serveMarkers'] = [row for row in state_input['serveMarkers']
        if row['id'] not in removed_input and (row.get('rallyId') not in known or row['rallyId'] in active)]
    normalized = normalize_feedback_annotations(normalization_input)
    corrections = payload['corrections']
    ignored = deepcopy(normalized['ignoredIntervals'])
    suppressed = [r for r in payload.get('finalExportProvenance', [])
                  if str(r.get('kind', '')).startswith('suppression-')]
    rallies = []
    for row in normalized['retainedCoreRanges']:
        masks = ignored + [r for r in suppressed if row['id'] in r.get('cutIds', [])]
        pieces = subtract(row['start'], row['end'], masks)
        for index, (start, end) in enumerate(pieces):
            rallies.append({'id': row['id'] if len(pieces) == 1 else f"{row['id']}::part:{index + 1}",
                'sourceCutId': row['id'], 'start': start, 'end': end,
                'tags': ['human-reviewed-project-coverage', 'imported-feedback',
                         'manual-addition' if row['origin'] == 'manual' else 'reviewed-model-range'],
                'notes': 'Imported corrected core coverage; endpoints have not been independently relabeled.',
                'sourceOrigin': row['origin']})
    rallies.sort(key=lambda r: (r['start'], r['end'], r['id']))
    if any(a['end'] > b['start'] for a, b in zip(rallies, rallies[1:])):
        raise ValueError('Corrected coverage overlaps; explicit human resolution is required')

    state = corrections['scoreTracking']['state']
    excluded = set(corrections['scoreTracking'].get('excludedRallyIds', []))
    removed = set(state.get('removedModelMarkerIds', []))
    retained_ids = {r['sourceCutId'] for r in rallies}
    corrected_ids = {r['id'] for r in corrections['correctedRanges']}
    serves, skipped = [], []
    for raw in state['serveMarkers']:
        timestamp = raw['timestamp']
        reason = ('excluded-rally' if raw.get('rallyId') in excluded else
                  'removed-marker' if raw['id'] in removed else
                  'ignored-time' if any(r['start'] <= timestamp < r['end'] for r in ignored) else
                  'unretained-rally' if raw.get('rallyId') in corrected_ids and raw['rallyId'] not in retained_ids else None)
        if reason:
            skipped.append({'id': raw['id'], 'reason': reason})
            continue
        marker = deepcopy(raw)
        marker['time'] = marker.pop('timestamp')
        marker['notes'] = 'Exact corrected feedback marker; original manual/model provenance retained.'
        serves.append(marker)
    serves.sort(key=lambda r: (r['time'], r['id']))
    switches = []
    for raw in state.get('sideSwitchMarkers', []):
        if raw.get('id') in removed or any(r['start'] <= raw['timestamp'] < r['end'] for r in ignored):
            continue
        row = deepcopy(raw)
        row['time'] = row.pop('timestamp')
        switches.append(row)
    switches.sort(key=lambda r: (r['time'], str(r.get('id', ''))))

    # Only explicit user false-positive labels become hard negatives. Automatic
    # suppression without human feedback is preserved as provenance, not gold.
    negatives = []
    for raw in corrections.get('labels', {}).get('falsePositives', []):
        for index, (start, end) in enumerate(subtract(raw['start'], raw['end'], ignored + rallies)):
            negatives.append({'id': f"{raw['id']}::negative:{index+1}", 'sourceCutId': raw['id'],
                'start': start, 'end': end, 'category': 'model-false-positive',
                'tags': ['human-reviewed-project-discard', 'imported-feedback'],
                'notes': 'Explicit false-positive correction from the source project.'})
    negatives.sort(key=lambda r: (r['start'], r['end']))
    if any(a['end'] > b['start'] for a, b in zip(negatives, negatives[1:])):
        raise ValueError('Explicit false-positive ranges overlap; do not silently merge identities')
    differences = [{'id': r['id'], 'sourceTime': r['rawTime'], 'normalizerAlignedTime': r['time']}
                   for r in normalized['serveEvents'] if r['wasTimeAdjusted']]
    return {'rallies': rallies, 'ignoredIntervals': ignored, 'hardNegatives': negatives,
            'serveMarkers': serves, 'sideSwitches': switches, 'normalizedReference': normalized,
            'importDetails': {'retainedCutIds': sorted(retained_ids), 'skippedServeMarkers': skipped,
                'normalizerServeTimeAdjustmentsNotApplied': differences,
                'suppressionProvenanceRows': len(suppressed), 'sourceSecondsPreserved': True,
                'frameRateConversionApplied': False, 'independentEndpointGold': False}}
