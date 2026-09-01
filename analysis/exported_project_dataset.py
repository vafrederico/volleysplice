"""Import human-reviewed production project exports as reusable weak-label data.

The production editor stores two related, but non-identical, annotation layers:

* corrected/final cut ranges describe retained rally *coverage*; and
* score-tracking serve markers describe the actual ordered serve events.

A short export join can leave one rally spread over more than one corrected range,
and a corrected range can contain more than one serve.  This module therefore does
not silently promote corrected ranges to frame-exact rally gold.  It preserves the
raw export and materializes explicit coverage, serve-event, side-switch, and ignored
time layers for downstream consumers to select deliberately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_text
from .features import probe_video


DATASET_KIND = "volleycut-exported-project-dataset-v1"
REFERENCE_KIND = "volleycut-exported-project-reference-v1"
SAMPLED_FINGERPRINT_PREFIX = "sampled-sha256-v1:"
SAMPLED_FINGERPRINT_BYTES = 1024 * 1024
MICRO_RANGE_SECONDS = 0.5
SERVE_ALIGNMENT_TOLERANCE_SECONDS = 1.0


class ExportedProjectDatasetError(ValueError):
    """Raised when a project export cannot be imported without guessing."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_fingerprint(path: Path) -> str:
    """Return the production app's sampled-sha256-v1 source fingerprint."""

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", size))
    with path.open("rb") as handle:
        if size <= 2 * SAMPLED_FINGERPRINT_BYTES:
            digest.update(handle.read())
        else:
            digest.update(handle.read(SAMPLED_FINGERPRINT_BYTES))
            handle.seek(size - SAMPLED_FINGERPRINT_BYTES)
            digest.update(handle.read(SAMPLED_FINGERPRINT_BYTES))
    return f"{SAMPLED_FINGERPRINT_PREFIX}{digest.hexdigest()}"


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportedProjectDatasetError(f"{where} must be an object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ExportedProjectDatasetError(f"{where} must be an array")
    return value


def _number(value: Any, where: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ExportedProjectDatasetError(f"{where} must be finite")
    return float(value)


def _strict_ranges(value: Any, duration: float, where: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(_sequence(value, where)):
        row = _mapping(raw, f"{where}[{index}]")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ExportedProjectDatasetError(f"{where}[{index}].id is required")
        start = _number(row.get("coreStart"), f"{where}[{index}].coreStart")
        end = _number(row.get("coreEnd"), f"{where}[{index}].coreEnd")
        if start < 0 or end <= start or end > duration + 1e-6:
            raise ExportedProjectDatasetError(
                f"{where}[{index}] must satisfy 0 <= coreStart < coreEnd <= {duration}"
            )
        rows.append(
            {
                "id": identifier,
                "start": start,
                "end": end,
                "duration": end - start,
                "included": row.get("included") is True,
                "origin": row.get("origin"),
                "confidence": row.get("confidence"),
                **(
                    {"agreement": row["agreement"]}
                    if isinstance(row.get("agreement"), str)
                    else {}
                ),
            }
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ExportedProjectDatasetError(f"{where} contains duplicate ids")
    return rows


def _time_in_ignored(time: float, ignored: Sequence[Mapping[str, Any]]) -> bool:
    return any(float(row["start"]) <= time < float(row["end"]) for row in ignored)


def _distance_to_range(time: float, row: Mapping[str, Any]) -> float:
    start, end = float(row["start"]), float(row["end"])
    if start <= time <= end:
        return 0.0
    return min(abs(time - start), abs(time - end))


def normalize_feedback_annotations(
    payload: Mapping[str, Any],
    *,
    micro_range_seconds: float = MICRO_RANGE_SECONDS,
    serve_alignment_tolerance_seconds: float = SERVE_ALIGNMENT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """Normalize one export while retaining every source timestamp and identifier."""

    source = _mapping(payload.get("source"), "source")
    media = _mapping(source.get("media"), "source.media")
    duration = _number(media.get("duration"), "source.media.duration")
    corrections = _mapping(payload.get("corrections"), "corrections")
    corrected = _strict_ranges(
        corrections.get("correctedRanges"), duration, "corrections.correctedRanges"
    )
    corrected_by_id = {row["id"]: row for row in corrected}

    ignored: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, raw in enumerate(
        _sequence(corrections.get("ignoredIntervals", []), "corrections.ignoredIntervals")
    ):
        row = _mapping(raw, f"corrections.ignoredIntervals[{index}]")
        start = _number(row.get("start"), f"corrections.ignoredIntervals[{index}].start")
        end = _number(row.get("end"), f"corrections.ignoredIntervals[{index}].end")
        if start < 0 or end <= start or end > duration or start < previous_end:
            raise ExportedProjectDatasetError("ignored intervals are invalid or overlap")
        ignored.append(
            {
                "id": row.get("id"),
                "start": start,
                "end": end,
                "reason": row.get("reason"),
            }
        )
        previous_end = end

    final_exports = _sequence(payload.get("finalExportIntervals"), "finalExportIntervals")
    effective_ids: list[str] = []
    range_export_index: dict[str, int] = {}
    normalized_exports: list[dict[str, Any]] = []
    for export_index, raw in enumerate(final_exports):
        row = _mapping(raw, f"finalExportIntervals[{export_index}]")
        start = _number(row.get("start"), f"finalExportIntervals[{export_index}].start")
        end = _number(row.get("end"), f"finalExportIntervals[{export_index}].end")
        if start < 0 or end <= start or end > duration + 1e-6:
            raise ExportedProjectDatasetError(
                f"finalExportIntervals[{export_index}] is outside the source"
            )
        cut_ids = []
        for cut_index, identifier in enumerate(
            _sequence(row.get("cutIds"), f"finalExportIntervals[{export_index}].cutIds")
        ):
            if not isinstance(identifier, str) or identifier not in corrected_by_id:
                raise ExportedProjectDatasetError(
                    f"finalExportIntervals[{export_index}].cutIds[{cut_index}] is unknown"
                )
            if not corrected_by_id[identifier]["included"]:
                raise ExportedProjectDatasetError(
                    f"final export references disabled range {identifier}"
                )
            if identifier in range_export_index:
                raise ExportedProjectDatasetError(
                    f"corrected range {identifier} appears in more than one final interval"
                )
            range_export_index[identifier] = export_index
            effective_ids.append(identifier)
            cut_ids.append(identifier)
        normalized_exports.append(
            {
                "start": start,
                "end": end,
                "cutIds": cut_ids,
                "joinedGaps": row.get("joinedGaps", []),
            }
        )

    retained = sorted(
        (corrected_by_id[identifier] for identifier in effective_ids),
        key=lambda row: (float(row["start"]), float(row["end"]), str(row["id"])),
    )
    micro_ids = {
        str(row["id"])
        for row in retained
        if float(row["duration"]) < micro_range_seconds
    }
    association_ranges = [row for row in retained if row["id"] not in micro_ids]
    if not association_ranges:
        raise ExportedProjectDatasetError("final export has no non-micro retained coverage")

    score_tracking = _mapping(corrections.get("scoreTracking"), "corrections.scoreTracking")
    state = _mapping(score_tracking.get("state"), "corrections.scoreTracking.state")
    excluded_ids = {
        str(value)
        for value in _sequence(
            score_tracking.get("excludedRallyIds", []),
            "corrections.scoreTracking.excludedRallyIds",
        )
    }
    raw_markers = []
    for index, raw in enumerate(
        _sequence(state.get("serveMarkers"), "corrections.scoreTracking.state.serveMarkers")
    ):
        marker = dict(_mapping(raw, f"serveMarkers[{index}]"))
        marker_id = marker.get("id")
        raw_rally_id = marker.get("rallyId")
        time = _number(marker.get("timestamp"), f"serveMarkers[{index}].timestamp")
        if not isinstance(marker_id, str) or not marker_id:
            raise ExportedProjectDatasetError(f"serveMarkers[{index}].id is required")
        if time < 0 or time > duration:
            raise ExportedProjectDatasetError(f"serveMarkers[{index}] is outside the source")
        if raw_rally_id in excluded_ids or _time_in_ignored(time, ignored):
            continue
        raw_markers.append({**marker, "timestamp": time})
    raw_markers.sort(key=lambda row: (float(row["timestamp"]), str(row["id"])))

    direct_non_micro_ids = {
        str(marker["rallyId"])
        for marker in raw_markers
        if marker.get("rallyId") in corrected_by_id
        and marker.get("rallyId") in range_export_index
        and marker.get("rallyId") not in micro_ids
    }
    missing_direct_ids = {
        str(row["id"]) for row in association_ranges
    } - direct_non_micro_ids

    events: list[dict[str, Any]] = []
    for marker in raw_markers:
        raw_time = float(marker["timestamp"])
        raw_rally_id = marker.get("rallyId")
        target: Mapping[str, Any] | None = None
        reason = "nearest-retained-range"

        if (
            isinstance(raw_rally_id, str)
            and raw_rally_id in corrected_by_id
            and raw_rally_id in range_export_index
            and raw_rally_id not in micro_ids
        ):
            target = corrected_by_id[raw_rally_id]
            reason = "preserved-export-rally-id"
            if _distance_to_range(raw_time, target) > serve_alignment_tolerance_seconds:
                export_index = range_export_index[raw_rally_id]
                missing_same_export = [
                    row
                    for row in association_ranges
                    if range_export_index[str(row["id"])] == export_index
                    and row["id"] in missing_direct_ids
                ]
                if missing_same_export:
                    alternative = min(
                        missing_same_export,
                        key=lambda row: (
                            _distance_to_range(raw_time, row),
                            abs(float(row["start"]) - raw_time),
                            float(row["start"]),
                        ),
                    )
                    if _distance_to_range(raw_time, alternative) < _distance_to_range(
                        raw_time, target
                    ):
                        target = alternative
                        reason = "reassigned-to-unmarked-neighbor"
        elif isinstance(raw_rally_id, str) and raw_rally_id in micro_ids:
            export_index = range_export_index[raw_rally_id]
            candidates = [
                row
                for row in association_ranges
                if range_export_index[str(row["id"])] == export_index
            ]
            if candidates:
                target = min(
                    candidates,
                    key=lambda row: (
                        _distance_to_range(raw_time, row),
                        abs(float(row["start"]) - raw_time),
                        float(row["start"]),
                    ),
                )
                reason = "reassigned-from-micro-range"

        if target is None:
            containing = [
                row
                for row in association_ranges
                if float(row["start"]) <= raw_time <= float(row["end"])
            ]
            candidates = containing or association_ranges
            target = min(
                candidates,
                key=lambda row: (
                    _distance_to_range(raw_time, row),
                    abs(float(row["start"]) - raw_time),
                    float(row["start"]),
                ),
            )
            reason = (
                "manual-marker-inside-retained-range"
                if containing and marker.get("origin") == "manual"
                else "nearest-retained-range"
            )

        aligned_time = raw_time
        if (
            raw_time < float(target["start"]) - serve_alignment_tolerance_seconds
            or raw_time > float(target["end"])
        ):
            aligned_time = float(target["start"])
            reason = f"{reason}+snapped-to-edited-range-start"

        event = {
            "id": marker["id"],
            "time": aligned_time,
            "rawTime": raw_time,
            "side": marker.get("side"),
            "origin": marker.get("origin"),
            "rawRallyId": raw_rally_id,
            "alignedRangeId": target["id"],
            "coverageRangeIds": [target["id"]],
            "alignment": reason,
            "wasTimeAdjusted": abs(aligned_time - raw_time) > 1e-9,
            "ignorePreviousPoint": marker.get("ignorePreviousPoint") is True,
        }
        for field in ("modelSide", "modelConfidence"):
            if field in marker:
                event[field] = marker[field]
        events.append(event)

    events.sort(key=lambda row: (float(row["time"]), str(row["id"])))
    if any(
        float(current["time"]) <= float(previous["time"])
        for previous, current in zip(events, events[1:], strict=False)
    ):
        raise ExportedProjectDatasetError(
            "serve-marker alignment produced non-increasing event times"
        )

    # A retained range without its own marker is often a continuation fragment
    # joined into the same export interval.  Associate it with the nearest event
    # in that final interval without inventing another serve.
    targets = {str(event["alignedRangeId"]) for event in events}
    unassociated_range_ids: list[str] = []
    for row in association_ranges:
        identifier = str(row["id"])
        if identifier in targets:
            continue
        export_index = range_export_index[identifier]
        candidates = [
            event
            for event in events
            if range_export_index.get(str(event["alignedRangeId"])) == export_index
        ]
        if not candidates:
            unassociated_range_ids.append(identifier)
            continue
        center = (float(row["start"]) + float(row["end"])) / 2.0
        event = min(
            candidates,
            key=lambda item: (abs(float(item["time"]) - center), float(item["time"])),
        )
        event["coverageRangeIds"].append(identifier)
        event["coverageRangeIds"].sort(
            key=lambda item: (
                float(corrected_by_id[item]["start"]),
                float(corrected_by_id[item]["end"]),
                item,
            )
        )

    side_switches: list[dict[str, Any]] = []
    for index, raw in enumerate(
        _sequence(
            state.get("sideSwitchMarkers", []),
            "corrections.scoreTracking.state.sideSwitchMarkers",
        )
    ):
        marker = _mapping(raw, f"sideSwitchMarkers[{index}]")
        time = _number(marker.get("timestamp"), f"sideSwitchMarkers[{index}].timestamp")
        if time < 0 or time > duration or _time_in_ignored(time, ignored):
            continue
        side_switches.append(
            {
                "id": marker.get("id"),
                "time": time,
                "origin": marker.get("origin"),
                **(
                    {"modelConfidence": marker["modelConfidence"]}
                    if "modelConfidence" in marker
                    else {}
                ),
                **(
                    {"modelEventId": marker["modelEventId"]}
                    if "modelEventId" in marker
                    else {}
                ),
                **(
                    {"rallyIds": marker["rallyIds"]}
                    if "rallyIds" in marker
                    else {}
                ),
            }
        )
    side_switches.sort(key=lambda row: (float(row["time"]), str(row["id"])))

    adjusted = [event for event in events if event["wasTimeAdjusted"]]
    return {
        "durationSeconds": duration,
        "retainedCoreRanges": retained,
        "associationCoreRanges": association_ranges,
        "microRangeArtifacts": [
            row for row in retained if str(row["id"]) in micro_ids
        ],
        "serveEvents": events,
        "sideSwitches": side_switches,
        "ignoredIntervals": ignored,
        "finalExportIntervals": normalized_exports,
        "scoreTracking": {
            "team1Name": state.get("team1Name"),
            "team2Name": state.get("team2Name"),
            "excludedRallyIds": sorted(excluded_ids),
            "derivedFinalScore": score_tracking.get("derivedFinalScore"),
        },
        "normalization": {
            "microRangeThresholdSeconds": micro_range_seconds,
            "serveAlignmentToleranceSeconds": serve_alignment_tolerance_seconds,
            "serveEventCount": len(events),
            "adjustedServeEventCount": len(adjusted),
            "adjustedServeEventIds": [event["id"] for event in adjusted],
            "microRangeArtifactIds": sorted(micro_ids),
            "unassociatedCoverageRangeIds": unassociated_range_ids,
            "llmLabelingUsed": False,
        },
    }


def _feedback_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportedProjectDatasetError(f"cannot read {path}: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "volleycut-model-feedback"
        or payload.get("schemaVersion") != 3
    ):
        raise ExportedProjectDatasetError(f"{path} is not model-feedback schema v3")
    return payload


def _reference_record(
    feedback_path: Path,
    *,
    source_group: str,
    split: str,
    environment: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _feedback_payload(feedback_path)
    source = _mapping(payload.get("source"), "source")
    file_info = _mapping(source.get("file"), "source.file")
    filename = file_info.get("name")
    if not isinstance(filename, str) or not filename:
        raise ExportedProjectDatasetError(f"{feedback_path}: source.file.name is required")
    video_path = (feedback_path.parent / filename).resolve()
    if not video_path.is_file():
        raise ExportedProjectDatasetError(f"source video does not exist: {video_path}")
    expected_size = file_info.get("sizeBytes")
    if not isinstance(expected_size, int) or video_path.stat().st_size != expected_size:
        raise ExportedProjectDatasetError(f"source size differs from {feedback_path}")
    expected_fingerprint = file_info.get("sampledFingerprint")
    actual_fingerprint = sampled_fingerprint(video_path)
    if expected_fingerprint != actual_fingerprint:
        raise ExportedProjectDatasetError(f"source fingerprint differs from {feedback_path}")

    normalized = normalize_feedback_annotations(payload)
    metadata = probe_video(video_path)
    if abs(metadata.duration - float(normalized["durationSeconds"])) > 0.1:
        raise ExportedProjectDatasetError(
            f"source duration differs from {feedback_path}: "
            f"{metadata.duration:.3f} != {normalized['durationSeconds']:.3f}"
        )
    stem = video_path.stem
    recording_id = f"raw-no-backup-{stem}"
    video_sha256 = sha256_file(video_path)
    feedback_sha256 = sha256_file(feedback_path)
    reference = {
        "schemaVersion": 1,
        "kind": REFERENCE_KIND,
        "recordingId": recording_id,
        "sourceGroup": source_group,
        "split": split,
        "environment": environment,
        "videoPath": str(video_path),
        "videoFilename": video_path.name,
        "videoSha256": video_sha256,
        "videoSizeBytes": video_path.stat().st_size,
        "sampledFingerprint": actual_fingerprint,
        "feedbackPath": str(feedback_path.resolve()),
        "feedbackSha256": feedback_sha256,
        "feedbackGeneratedAt": payload.get("generatedAt"),
        "feedbackWarnings": payload.get("warnings", []),
        "featuresRetainedInExport": payload.get("features") is not None,
        "roi": source.get("featureRoi"),
        "gameWindow": source.get("gameWindow"),
        "annotationScope": {
            "rallyCoverage": "complete-per-export-owner",
            "rallyBoundaryUse": "human-retained export coverage; not independently promoted to frame-exact rally gold",
            "serveTiming": "approximately within one second for most markers per export owner",
            "sideSwitches": "exported score-tracking markers",
            "llmLabelingUsed": False,
        },
        "annotations": normalized,
    }
    manifest_row = {
        "id": recording_id,
        "video": str(video_path),
        "split": split,
        "sourceGroup": source_group,
        "environment": environment,
        "game": {"playersPerTeam": None, "targetPoints": None, "format": "grass"},
        "consent": {"analyze": True, "train": split in {"train", "validation"}},
        "capture": {},
        "roi": source.get("featureRoi"),
        "rallies": [],
        "ignoredIntervals": normalized["ignoredIntervals"],
    }
    return reference, manifest_row


def build_exported_project_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    feedback_glob: str,
    source_group: str,
    split: str = "challenge",
    environment: str = "grass",
) -> dict[str, Any]:
    """Import an explicit labeled-feedback batch into a new immutable workspace."""

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir():
        raise ExportedProjectDatasetError(f"source root does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    if split not in {"train", "validation", "test", "challenge"}:
        raise ExportedProjectDatasetError(f"unsupported split: {split}")
    feedback_paths = sorted(source.glob(feedback_glob))
    if not feedback_paths:
        raise ExportedProjectDatasetError(
            f"no feedback files match {feedback_glob!r} under {source}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    ).resolve()
    try:
        references: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        checksums: list[tuple[str, str]] = []
        for feedback_path in feedback_paths:
            reference, manifest_row = _reference_record(
                feedback_path,
                source_group=source_group,
                split=split,
                environment=environment,
            )
            reference_path = staging / "references" / f"{reference['recordingId']}.json"
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                reference_path,
                json.dumps(reference, indent=2, allow_nan=False) + "\n",
            )
            checksums.append((sha256_file(reference_path), str(reference_path.relative_to(staging))))
            references.append(
                {
                    "recordingId": reference["recordingId"],
                    "referencePath": str(reference_path.relative_to(staging)),
                    "referenceSha256": checksums[-1][0],
                    "videoPath": reference["videoPath"],
                    "videoSha256": reference["videoSha256"],
                    "feedbackPath": reference["feedbackPath"],
                    "feedbackSha256": reference["feedbackSha256"],
                    "durationSeconds": reference["annotations"]["durationSeconds"],
                    "retainedCoreRangeCount": len(
                        reference["annotations"]["retainedCoreRanges"]
                    ),
                    "serveEventCount": len(reference["annotations"]["serveEvents"]),
                    "sideSwitchCount": len(reference["annotations"]["sideSwitches"]),
                    "ignoredIntervalCount": len(
                        reference["annotations"]["ignoredIntervals"]
                    ),
                }
            )
            manifest_rows.append(manifest_row)

        inference_manifest = {
            "schemaVersion": 1,
            "name": f"{source_group}-exported-project-inference-only",
            "annotationPolicy": {
                "id": "serve-contact-to-dead-ball-v1",
                "rallyStart": "serve-ball contact",
                "rallyEnd": "first instant live play has ended",
                "intervalConvention": "half-open [start,end) seconds on this source video",
            },
            "recordings": manifest_rows,
        }
        inference_path = staging / "manifests" / "inference-only.json"
        inference_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            inference_path,
            json.dumps(inference_manifest, indent=2, allow_nan=False) + "\n",
        )
        inference_sha = sha256_file(inference_path)
        checksums.append((inference_sha, str(inference_path.relative_to(staging))))

        dataset = {
            "schemaVersion": 1,
            "kind": DATASET_KIND,
            "createdAt": datetime.now(UTC).isoformat(),
            "name": source_group,
            "sourceRoot": str(source),
            "sourceGroup": source_group,
            "split": split,
            "environment": environment,
            "targetStatus": "reviewed-export-coverage",
            "inferenceManifestPath": str(inference_path.relative_to(staging)),
            "inferenceManifestSha256": inference_sha,
            "regeneratedInferenceIndexPath": "regenerated-inference/index.json",
            "records": references,
        }
        dataset_path = staging / "dataset.json"
        atomic_write_text(
            dataset_path,
            json.dumps(dataset, indent=2, allow_nan=False) + "\n",
        )
        checksums.append((sha256_file(dataset_path), "dataset.json"))
        atomic_write_text(
            staging / "checksums.sha256",
            "".join(f"{digest}  {relative}\n" for digest, relative in sorted(checksums, key=lambda row: row[1])),
        )
        os.replace(staging, output)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    return json.loads((output / "dataset.json").read_text(encoding="utf-8"))


def load_exported_project_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportedProjectDatasetError(f"cannot read {dataset_path}: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schemaVersion") != 1
        or payload.get("kind") != DATASET_KIND
    ):
        raise ExportedProjectDatasetError(f"{dataset_path} is not a {DATASET_KIND}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ExportedProjectDatasetError(f"{dataset_path} contains no records")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ExportedProjectDatasetError(f"records[{index}] must be an object")
        reference = dataset_path.parent / str(record.get("referencePath", ""))
        if not reference.is_file() or sha256_file(reference) != record.get("referenceSha256"):
            raise ExportedProjectDatasetError(
                f"records[{index}] reference is missing or has changed: {reference}"
            )
    return payload
