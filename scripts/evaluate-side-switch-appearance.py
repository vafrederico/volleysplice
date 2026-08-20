#!/usr/bin/env python3
"""Run label-only diagnostic variants for side-switch appearance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.nas_video_corpus import load_manifest, manifest_path_record, manifest_target_status
from analysis.side_switch_appearance import (
    AppearanceAggregate,
    COLOR_SPACE,
    aggregate_frames,
    appearance_features,
    create_hog,
    summarize_frame,
    read_frame,
)


EXPERIMENT_KIND = "volleycut-side-switch-appearance-diagnostic-v1"
DEFAULT_ENVIRONMENTS = ("beach", "grass")


@dataclass(frozen=True)
class RecordingLabels:
    label_path: Path | None
    recording_id: str
    environment: str
    source_group: str
    split: str
    source_type: str
    target_status: str
    video_path: Path
    video_filename: str
    duration_seconds: float
    rallies: tuple[tuple[float, float], ...]
    switches: tuple[float, ...]
    candidate_source: dict[str, Any]


@dataclass(frozen=True)
class TransitionEvent:
    event_id: str
    recording_id: str
    environment: str
    label: int | None
    transition_time: float
    gap_start: float
    gap_end: float
    source: str
    target_status: str
    source_type: str
    source_group: str
    split: str
    candidate_source: dict[str, Any]

    @property
    def gap_seconds(self) -> float:
        return self.gap_end - self.gap_start


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def load_recording_labels(path: Path) -> RecordingLabels:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recording = payload["recording"]
    recording_id = str(recording["id"])
    environment = str(recording.get("environment", "unknown"))
    annotation = payload.get("annotation", {})
    target_status = (
        "gold"
        if annotation.get("status") == "complete"
        else "reviewed-draft"
        if annotation.get("continuousVideoReviewed") is True
        else "candidate-only"
    )
    video = Path(str(recording["video"])).expanduser()
    if not video.is_absolute():
        video = (path.parent / video).resolve()
    else:
        video = video.resolve()
    rallies = tuple(
        (_finite(row["start"], f"{recording_id}.rallies.start"), _finite(row["end"], f"{recording_id}.rallies.end"))
        for row in payload["rallies"]
    )
    if any(start >= end for start, end in rallies):
        raise ValueError(f"{recording_id} contains an invalid rally interval")
    switches = tuple(
        _finite(row["time"], f"{recording_id}.sideSwitches.time")
        for row in payload.get("sideSwitches", [])
    )
    if tuple(sorted(switches)) != switches:
        raise ValueError(f"{recording_id} side switches are not ordered")
    return RecordingLabels(
        path,
        recording_id,
        environment,
        str(recording.get("sourceGroup", "unknown")),
        str(recording.get("split", "unknown")),
        "label-directory",
        target_status,
        video,
        str(recording.get("videoFilename", video.name)),
        _finite(recording["durationSeconds"], f"{recording_id}.durationSeconds"),
        rallies,
        switches,
        {
            "kind": "label-document",
            "annotationStatus": annotation.get("status"),
            "continuousVideoReviewed": annotation.get("continuousVideoReviewed"),
        },
    )


def load_manifest_record(record: dict[str, Any]) -> RecordingLabels:
    recording_id = str(record["recordingId"])
    rallies = tuple(
        (
            _finite(row["start"], f"{recording_id}.rallies.start"),
            _finite(row["end"], f"{recording_id}.rallies.end"),
        )
        for row in record["rallies"]
    )
    if any(start < 0 or start >= end for start, end in rallies):
        raise ValueError(f"{recording_id} contains an invalid rally interval")
    switches = tuple(
        _finite(row["time"], f"{recording_id}.sideSwitches.time")
        for row in record.get("sideSwitches", [])
    )
    label_path_value = record.get("labelPath")
    label_path = Path(label_path_value).expanduser().resolve() if isinstance(label_path_value, str) else None
    video_path = manifest_path_record(record)
    candidate_source = record.get("candidateSource")
    return RecordingLabels(
        label_path,
        recording_id,
        str(record.get("environment", "unknown")),
        str(record.get("sourceGroup", "unknown")),
        str(record.get("split", "unknown")),
        str(record.get("sourceType", "manifest")),
        manifest_target_status(record),
        video_path,
        str(record.get("videoFilename", video_path.name)),
        _finite(record["durationSeconds"], f"{recording_id}.durationSeconds"),
        rallies,
        switches,
        candidate_source if isinstance(candidate_source, dict) else {},
    )


def _gap_for_time(
    rallies: Sequence[tuple[float, float]],
    timestamp: float,
) -> tuple[int, float, float] | None:
    for index, ((_, previous_end), (next_start, _)) in enumerate(zip(rallies, rallies[1:])):
        if previous_end - 1e-6 <= timestamp <= next_start + 1e-6:
            return index, previous_end, next_start
    return None


def build_events(
    labels: RecordingLabels,
    *,
    minimum_gap_seconds: float,
    maximum_negative_events: int,
) -> tuple[TransitionEvent, ...]:
    events: list[TransitionEvent] = []
    if labels.target_status == "candidate-only":
        candidate_events: list[TransitionEvent] = []
        for index, ((_, previous_end), (next_start, _)) in enumerate(
            zip(labels.rallies, labels.rallies[1:])
        ):
            gap = next_start - previous_end
            if gap < minimum_gap_seconds:
                continue
            candidate_events.append(
                TransitionEvent(
                    event_id=f"{labels.recording_id}:candidate-gap:{index + 1}",
                    recording_id=labels.recording_id,
                    environment=labels.environment,
                    label=None,
                    transition_time=(previous_end + next_start) / 2.0,
                    gap_start=previous_end,
                    gap_end=next_start,
                    source="model-inference-gap",
                    target_status=labels.target_status,
                    source_type=labels.source_type,
                    source_group=labels.source_group,
                    split=labels.split,
                    candidate_source=labels.candidate_source,
                )
            )
        if maximum_negative_events > 0 and len(candidate_events) > maximum_negative_events:
            indexes = np.linspace(0, len(candidate_events) - 1, maximum_negative_events).round().astype(int)
            candidate_events = [candidate_events[index] for index in indexes]
        return tuple(candidate_events)

    for marker_index, timestamp in enumerate(labels.switches):
        gap = _gap_for_time(labels.rallies, timestamp)
        if gap is None:
            continue
        index, gap_start, gap_end = gap
        events.append(
            TransitionEvent(
                event_id=f"{labels.recording_id}:switch:{marker_index + 1}",
                recording_id=labels.recording_id,
                environment=labels.environment,
                label=1,
                transition_time=timestamp,
                gap_start=gap_start,
                gap_end=gap_end,
                source="sideSwitches",
                target_status=labels.target_status,
                source_type=labels.source_type,
                source_group=labels.source_group,
                split=labels.split,
                candidate_source=labels.candidate_source,
            )
        )

    negative_candidates: list[TransitionEvent] = []
    for index, ((_, previous_end), (next_start, _)) in enumerate(
        zip(labels.rallies, labels.rallies[1:])
    ):
        gap = next_start - previous_end
        if gap < minimum_gap_seconds:
            continue
        if any(previous_end - 1e-6 <= timestamp <= next_start + 1e-6 for timestamp in labels.switches):
            continue
        negative_candidates.append(
            TransitionEvent(
                event_id=f"{labels.recording_id}:control:{index + 1}",
                recording_id=labels.recording_id,
                environment=labels.environment,
                label=0,
                transition_time=(previous_end + next_start) / 2.0,
                gap_start=previous_end,
                gap_end=next_start,
                source="unmarked-inter-rally-gap",
                target_status=labels.target_status,
                source_type=labels.source_type,
                source_group=labels.source_group,
                split=labels.split,
                candidate_source=labels.candidate_source,
            )
        )
    negative_candidates.sort(key=lambda event: (event.transition_time, event.event_id))
    if maximum_negative_events > 0 and len(negative_candidates) > maximum_negative_events:
        indexes = np.linspace(
            0,
            len(negative_candidates) - 1,
            maximum_negative_events,
        ).round().astype(int)
        negative_candidates = [negative_candidates[index] for index in indexes]
    events.extend(negative_candidates)
    return tuple(sorted(events, key=lambda event: (event.recording_id, event.transition_time, event.label)))


def _segment_times(
    start: float,
    end: float,
    count: int,
) -> tuple[float, ...]:
    if end <= start or count < 1:
        return ()
    if count == 1:
        return ((start + end) / 2.0,)
    return tuple(float(value) for value in np.linspace(start, end, count))


def sample_transition_times(
    event: TransitionEvent,
    *,
    samples_per_side: int,
    flank_seconds: float,
    edge_margin_seconds: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    left = event.gap_start + edge_margin_seconds
    right = event.gap_end - edge_margin_seconds
    before_start = max(left, event.transition_time - flank_seconds)
    before_end = min(right, event.transition_time - edge_margin_seconds)
    after_start = max(left, event.transition_time + edge_margin_seconds)
    after_end = min(right, event.transition_time + flank_seconds)
    return (
        _segment_times(before_start, before_end, samples_per_side),
        _segment_times(after_start, after_end, samples_per_side),
    )


def _aggregate_dict(value: AppearanceAggregate) -> dict[str, Any]:
    return {
        "meanDetectionCount": round(value.mean_detection_count, 5),
        "meanTotalBoxAreaFraction": round(value.mean_total_box_area_fraction, 7),
        "meanMedianBoxHeightFraction": (
            None
            if value.mean_median_box_height_fraction is None
            else round(value.mean_median_box_height_fraction, 7)
        ),
        "meanDetectionScore": (
            None if value.mean_detection_score is None else round(value.mean_detection_score, 5)
        ),
        "usableFrameCount": value.usable_frame_count,
    }


def _feature_dict(features: dict[str, float | None]) -> dict[str, float | None]:
    return {
        key: None if value is None else round(float(value), 7)
        for key, value in features.items()
    }


def analyze_event(
    event: TransitionEvent,
    capture: cv2.VideoCapture,
    hog: cv2.HOGDescriptor,
    *,
    samples_per_side: int,
    flank_seconds: float,
    edge_margin_seconds: float,
) -> dict[str, Any]:
    before_times, after_times = sample_transition_times(
        event,
        samples_per_side=samples_per_side,
        flank_seconds=flank_seconds,
        edge_margin_seconds=edge_margin_seconds,
    )
    row: dict[str, Any] = {
        "eventId": event.event_id,
        "recordingId": event.recording_id,
        "environment": event.environment,
        "label": event.label,
        "source": event.source,
        "targetStatus": event.target_status,
        "sourceType": event.source_type,
        "sourceGroup": event.source_group,
        "split": event.split,
        "candidateSource": event.candidate_source,
        "transitionTime": round(event.transition_time, 5),
        "gapStart": round(event.gap_start, 5),
        "gapEnd": round(event.gap_end, 5),
        "gapSeconds": round(event.gap_seconds, 5),
        "beforeTimes": [round(value, 5) for value in before_times],
        "afterTimes": [round(value, 5) for value in after_times],
    }
    if not before_times or not after_times:
        row.update({"status": "insufficient-window", "features": {}})
        return row
    try:
        before_frames = [
            summarize_frame(read_frame(capture, timestamp), timestamp, hog)
            for timestamp in before_times
        ]
        after_frames = [
            summarize_frame(read_frame(capture, timestamp), timestamp, hog)
            for timestamp in after_times
        ]
        before = aggregate_frames(before_frames)
        after = aggregate_frames(after_frames)
        row.update(
            {
                "status": "ok",
                "before": _aggregate_dict(before),
                "after": _aggregate_dict(after),
                "features": _feature_dict(appearance_features(before, after)),
            }
        )
    except (RuntimeError, ValueError, cv2.error) as error:
        row.update({"status": "frame-error", "error": str(error), "features": {}})
    return row


def _rank_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(value == 1 for value in labels)
    negatives = sum(value == 0 for value in labels)
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    rank_sum = 0.0
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        rank_sum += average_rank * sum(labels[index] == 1 for index in order[position:end])
        position = end
    return float(
        (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(value == 1 for value in labels)
    if not positives:
        return None
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    true_positives = 0
    weighted_precision = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            true_positives += 1
            weighted_precision += true_positives / rank
    return float(weighted_precision / positives)


def _threshold_diagnostic(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    negative_scores = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not negative_scores:
        return {"threshold": None, "precision": None, "recall": None, "falsePositiveRate": None}
    threshold = float(np.percentile(negative_scores, 95.0))
    predicted = [score >= threshold for score in scores]
    true_positives = sum(pred and label == 1 for pred, label in zip(predicted, labels, strict=True))
    false_positives = sum(pred and label == 0 for pred, label in zip(predicted, labels, strict=True))
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    return {
        "threshold": round(threshold, 7),
        "precision": round(true_positives / (true_positives + false_positives), 7)
        if true_positives + false_positives
        else None,
        "recall": round(true_positives / positives, 7) if positives else None,
        "falsePositiveRate": round(false_positives / negatives, 7) if negatives else None,
        "truePositives": true_positives,
        "falsePositives": false_positives,
    }


FEATURES = (
    "fullFrameControl",
    "playerPaletteEqual",
    "playerPaletteArea",
    "playerPaletteAreaPlusGeometry",
)


def summarize_metrics(
    rows: Sequence[dict[str, Any]],
    target_statuses: set[str] | None = None,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if isinstance(row.get("label"), int)
        and (
            target_statuses is None
            or str(row.get("targetStatus", "gold")) in target_statuses
        )
    ]
    result: dict[str, Any] = {}
    environments = sorted({str(row["environment"]) for row in eligible})
    scopes = [("pooled", eligible)] + [
        (environment, [row for row in eligible if row["environment"] == environment])
        for environment in environments
    ]
    for key, values in (
        ("sourceType", sorted({str(row.get("sourceType", "unknown")) for row in eligible})),
        ("targetStatus", sorted({str(row.get("targetStatus", "gold")) for row in eligible})),
    ):
        scopes.extend(
            (
                f"{key}:{value}",
                [row for row in eligible if str(row.get(key, "unknown")) == value],
            )
            for value in values
        )
    for scope, scoped_rows in scopes:
        scope_result: dict[str, Any] = {
            "events": len(scoped_rows),
            "positives": sum(row["label"] == 1 for row in scoped_rows),
            "negatives": sum(row["label"] == 0 for row in scoped_rows),
            "features": {},
        }
        for feature in FEATURES:
            usable = [
                row
                for row in scoped_rows
                if row.get("status") == "ok" and row.get("features", {}).get(feature) is not None
            ]
            labels = [int(row["label"]) for row in usable]
            scores = [float(row["features"][feature]) for row in usable]
            threshold = _threshold_diagnostic(labels, scores)
            scope_result["features"][feature] = {
                "usableEvents": len(usable),
                "coverage": round(len(usable) / len(scoped_rows), 7) if scoped_rows else None,
                "positives": sum(value == 1 for value in labels),
                "negatives": sum(value == 0 for value in labels),
                "rocAuc": None if _rank_auc(labels, scores) is None else round(_rank_auc(labels, scores) or 0.0, 7),
                "averagePrecision": None
                if _average_precision(labels, scores) is None
                else round(_average_precision(labels, scores) or 0.0, 7),
                "negative95thPercentileDiagnostic": threshold,
            }
        result[scope] = scope_result
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run diagnostic side-switch appearance variants from completed labels."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labels-dir",
        type=Path,
        help="directory containing completed *.labels.json documents",
    )
    source.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--environments",
        nargs="+",
        default=["beach", "grass", "indoor", "unknown"],
        choices=("beach", "grass", "indoor", "broadcast", "unknown"),
    )
    parser.add_argument("--minimum-gap-seconds", type=float, default=8.0)
    parser.add_argument(
        "--maximum-negative-events-per-recording",
        type=int,
        default=0,
        help="0 keeps all eligible no-switch controls",
    )
    parser.add_argument("--samples-per-side", type=int, default=2)
    parser.add_argument("--flank-seconds", type=float, default=8.0)
    parser.add_argument("--edge-margin-seconds", type=float, default=0.75)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    labels_dir = arguments.labels_dir.expanduser().resolve() if arguments.labels_dir else None
    manifest_path = arguments.corpus_manifest.expanduser().resolve() if arguments.corpus_manifest else None
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    if arguments.minimum_gap_seconds <= 0:
        raise ValueError("--minimum-gap-seconds must be positive")
    if arguments.maximum_negative_events_per_recording < 0:
        raise ValueError("--maximum-negative-events-per-recording cannot be negative")
    if arguments.samples_per_side < 1:
        raise ValueError("--samples-per-side must be positive")
    if arguments.flank_seconds <= arguments.edge_margin_seconds:
        raise ValueError("--flank-seconds must exceed --edge-margin-seconds")

    if manifest_path:
        manifest = load_manifest(manifest_path)
        recordings = [load_manifest_record(record) for record in manifest["records"]]
    else:
        assert labels_dir is not None
        label_paths = sorted(labels_dir.glob("*.labels.json"))
        if not label_paths:
            raise FileNotFoundError(f"no label documents found in {labels_dir}")
        recordings = [load_recording_labels(path) for path in label_paths]
    recordings = [item for item in recordings if item.environment in arguments.environments]
    if not recordings:
        raise ValueError("no recordings matched --environments")

    events: list[TransitionEvent] = []
    for recording in recordings:
        events.extend(
            build_events(
                recording,
                minimum_gap_seconds=arguments.minimum_gap_seconds,
                maximum_negative_events=arguments.maximum_negative_events_per_recording,
            )
        )
    if not events:
        raise ValueError("no transition events were constructed")

    events_by_recording: dict[str, list[TransitionEvent]] = {}
    for event in events:
        events_by_recording.setdefault(event.recording_id, []).append(event)
    rows: list[dict[str, Any]] = []
    for recording in recordings:
        recording_events = events_by_recording.get(recording.recording_id, [])
        if not recording_events:
            continue
        print(
            f"Analyzing {recording.recording_id}: {len(recording_events)} events from {recording.video_path}",
            file=sys.stderr,
            flush=True,
        )
        if not recording.video_path.is_file():
            raise FileNotFoundError(f"video does not exist: {recording.video_path}")
        capture = cv2.VideoCapture(str(recording.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {recording.video_path}")
        try:
            hog = create_hog()
            for index, event in enumerate(recording_events, start=1):
                print(
                    f"  {index}/{len(recording_events)} {event.source} at {event.transition_time:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
                rows.append(
                    analyze_event(
                        event,
                        capture,
                        hog,
                        samples_per_side=arguments.samples_per_side,
                        flank_seconds=arguments.flank_seconds,
                        edge_margin_seconds=arguments.edge_margin_seconds,
                    )
                )
        finally:
            capture.release()

    evaluation_statuses = {"gold", "reviewed-draft"}
    statuses = sorted({str(row.get("targetStatus", "gold")) for row in rows})
    metrics_by_status = {
        status: summarize_metrics(rows, {status})
        for status in statuses
        if status != "candidate-only"
    }
    labels_directory = str(labels_dir) if labels_dir else str(manifest_path.parent)
    report = {
        "schemaVersion": 1,
        "kind": EXPERIMENT_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "labels": {
            "directory": labels_directory,
            "manifest": str(manifest_path) if manifest_path else None,
            "files": [
                {
                    "recordingId": recording.recording_id,
                    "environment": recording.environment,
                    "sourceGroup": recording.source_group,
                    "split": recording.split,
                    "sourceType": recording.source_type,
                    "targetStatus": recording.target_status,
                    "path": str(recording.label_path) if recording.label_path else None,
                    "sha256": _sha256(recording.label_path) if recording.label_path and recording.label_path.is_file() else None,
                    "videoPath": str(recording.video_path),
                    "videoFilename": recording.video_filename,
                    "durationSeconds": recording.duration_seconds,
                    "candidateSource": recording.candidate_source,
                    "rallies": len(recording.rallies),
                    "sideSwitches": len(recording.switches),
                }
                for recording in recordings
            ],
        },
        "protocol": {
            "environments": list(arguments.environments),
            "minimumGapSeconds": arguments.minimum_gap_seconds,
            "maximumNegativeEventsPerRecording": arguments.maximum_negative_events_per_recording,
            "samplesPerSide": arguments.samples_per_side,
            "flankSeconds": arguments.flank_seconds,
            "edgeMarginSeconds": arguments.edge_margin_seconds,
            "positiveSource": "completed sideSwitches markers",
            "negativeSource": "unmarked inter-rally gaps",
            "candidateSource": "gaps between model-inference rally candidates for candidate-only recordings",
            "evaluationTargetStatuses": sorted(evaluation_statuses),
            "corpusScope": "full NAS manifest" if manifest_path else "label directory",
            "personProposal": "opencv-hog-default-people-detector-v1",
            "colorRepresentation": COLOR_SPACE,
            "thresholdsAreDiagnostic": True,
        },
        "summary": {
            "recordings": len(recordings),
            "events": len(rows),
            "positives": sum(row.get("label") == 1 for row in rows),
            "negatives": sum(row.get("label") == 0 for row in rows),
            "candidateEvents": sum(row.get("label") is None for row in rows),
            "targetStatusCounts": {
                status: sum(row.get("targetStatus", "gold") == status for row in rows)
                for status in statuses
            },
            "sourceTypeCounts": {
                source_type: sum(row.get("sourceType") == source_type for row in rows)
                for source_type in sorted({str(row.get("sourceType", "unknown")) for row in rows})
            },
            "statuses": {
                status: sum(row.get("status") == status for row in rows)
                for status in sorted({str(row.get("status")) for row in rows})
            },
            "metrics": summarize_metrics(rows, evaluation_statuses),
            "metricsByTargetStatus": metrics_by_status,
        },
        "events": rows,
    }
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(output), "events": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
