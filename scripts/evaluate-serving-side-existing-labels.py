#!/usr/bin/env python3
"""Evaluate serving-side evidence variants using the current rally labels.

This is an audit/diagnostic report, not a trained production model.  Rally
``start`` values are the existing serve-contact anchors.  The only available
serving-side targets are conservative phrases in rally notes, so unlabeled
notes remain outside the target metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.nas_video_corpus import load_manifest, manifest_path_record, manifest_target_status
from analysis.serving_side import (
    FAR_SIDE,
    NEAR_SIDE,
    ServingSideCue,
    crop_roi,
    hog_occupancy,
    infer_note_serving_side,
    occupancy_change_margin,
    palette_change,
    pixel_motion,
)
from analysis.side_switch_appearance import create_hog, read_frame


EXPERIMENT_KIND = "volleycut-serving-side-existing-label-variants-v1"
SIDE_VALUES = (NEAR_SIDE, FAR_SIDE)
VARIANT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "pixelMotion": {
        "label": "Pixel motion",
        "description": "Near-minus-far grayscale change from pre-serve to action frames",
        "components": ("pixelMotionMargin",),
    },
    "paletteChange": {
        "label": "Palette change",
        "description": "Near-minus-far HSV palette distance from pre-serve to action frames",
        "components": ("paletteChangeMargin",),
    },
    "hogAreaChange": {
        "label": "HOG area change",
        "description": "Near-minus-far absolute person-proposal box-area change",
        "components": ("hogAreaChangeMargin",),
    },
    "motionPalette": {
        "label": "Motion + palette",
        "description": "Weighted pixel-motion and palette-change margins",
        "components": ("pixelMotionMargin", "paletteChangeMargin"),
    },
    "motionPaletteHog": {
        "label": "Motion + palette + HOG",
        "description": "Weighted pixel-motion, palette-change, and HOG area margins",
        "components": (
            "pixelMotionMargin",
            "paletteChangeMargin",
            "hogAreaChangeMargin",
        ),
    },
    "baselineMotion": {
        "label": "Baseline-band motion",
        "description": "Pixel motion restricted to the outer serving-baseline bands",
        "components": ("baselinePixelMotionMargin",),
    },
    "baselinePalette": {
        "label": "Baseline-band palette",
        "description": "Palette change restricted to the outer serving-baseline bands",
        "components": ("baselinePaletteChangeMargin",),
    },
    "baselineHogArea": {
        "label": "Baseline-band HOG area",
        "description": "HOG proposal-area change restricted to outer baseline bands",
        "components": ("baselineHogAreaChangeMargin",),
    },
    "baselineMotionPalette": {
        "label": "Baseline motion + palette",
        "description": "Weighted baseline-band pixel motion and palette change",
        "components": ("baselinePixelMotionMargin", "baselinePaletteChangeMargin"),
    },
}

BASELINE_BAND_FRACTION = 0.35


@dataclass(frozen=True)
class RallyLabel:
    index: int
    start: float
    end: float
    notes: str | None
    tags: tuple[str, ...]
    target_status: str


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
    roi: tuple[float, float, float, float] | None
    rallies: tuple[RallyLabel, ...]
    candidate_source: dict[str, Any]


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


def _read_roi(value: Any, where: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object or null")
    result = tuple(
        _finite(value[key], f"{where}.{key}")
        for key in ("x", "y", "width", "height")
    )
    if result[2] <= 0 or result[3] <= 0:
        raise ValueError(f"{where} width and height must be positive")
    return result  # type: ignore[return-value]


def load_recording_labels(path: Path) -> RecordingLabels:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recording = payload["recording"]
    recording_id = str(recording["id"])
    video = Path(str(recording["video"])).expanduser()
    if not video.is_absolute():
        video = (path.parent / video).resolve()
    else:
        video = video.resolve()
    annotation = payload.get("annotation", {})
    target_status = (
        "gold"
        if annotation.get("status") == "complete"
        else "reviewed-draft"
        if annotation.get("continuousVideoReviewed") is True
        else "candidate-only"
    )
    rallies: list[RallyLabel] = []
    previous_end = -1.0
    for index, row in enumerate(payload["rallies"], start=1):
        start = _finite(row["start"], f"{recording_id}.rallies[{index}].start")
        end = _finite(row["end"], f"{recording_id}.rallies[{index}].end")
        if start < 0 or start >= end or start < previous_end:
            raise ValueError(f"{recording_id} contains an invalid rally interval")
        notes = row.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(f"{recording_id}.rallies[{index}].notes must be a string")
        tags = row.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"{recording_id}.rallies[{index}].tags must be strings")
        rallies.append(RallyLabel(index, start, end, notes, tuple(tags), target_status))
        previous_end = end
    duration = _finite(recording["durationSeconds"], f"{recording_id}.durationSeconds")
    return RecordingLabels(
        label_path=path,
        recording_id=recording_id,
        environment=str(recording.get("environment", "unknown")),
        source_group=str(recording.get("sourceGroup", "unknown")),
        split=str(recording.get("split", "unknown")),
        source_type="label-directory",
        target_status=target_status,
        video_path=video,
        video_filename=str(recording.get("videoFilename", video.name)),
        duration_seconds=duration,
        roi=_read_roi(recording.get("roi"), f"{recording_id}.recording.roi"),
        rallies=tuple(rallies),
        candidate_source={
            "kind": "label-document",
            "annotationStatus": annotation.get("status"),
            "continuousVideoReviewed": annotation.get("continuousVideoReviewed"),
        },
    )


def load_manifest_record(record: dict[str, Any]) -> RecordingLabels:
    recording_id = str(record["recordingId"])
    target_status = manifest_target_status(record)
    rallies: list[RallyLabel] = []
    previous_end = -1.0
    for index, row in enumerate(record["rallies"], start=1):
        start = _finite(row["start"], f"{recording_id}.rallies[{index}].start")
        end = _finite(row["end"], f"{recording_id}.rallies[{index}].end")
        if start < 0 or start >= end or start < previous_end:
            raise ValueError(f"{recording_id} contains an invalid rally interval")
        notes = row.get("notes")
        if notes is not None and not isinstance(notes, str):
            notes = None
        tags = row.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"{recording_id}.rallies[{index}].tags must be strings")
        rallies.append(RallyLabel(index, start, end, notes, tuple(tags), target_status))
        previous_end = end
    label_path_value = record.get("labelPath")
    label_path = Path(label_path_value).expanduser().resolve() if isinstance(label_path_value, str) else None
    return RecordingLabels(
        label_path=label_path,
        recording_id=recording_id,
        environment=str(record.get("environment", "unknown")),
        source_group=str(record.get("sourceGroup", "unknown")),
        split=str(record.get("split", "unknown")),
        source_type=str(record.get("sourceType", "manifest")),
        target_status=target_status,
        video_path=manifest_path_record(record),
        video_filename=str(record.get("videoFilename", manifest_path_record(record).name)),
        duration_seconds=_finite(record["durationSeconds"], f"{recording_id}.durationSeconds"),
        roi=_read_roi(record.get("roi"), f"{recording_id}.recording.roi"),
        rallies=tuple(rallies),
        candidate_source=(record.get("candidateSource") if isinstance(record.get("candidateSource"), dict) else {}),
    )


def sample_times(
    rally: RallyLabel,
    duration_seconds: float,
    *,
    pre_offsets: Sequence[float] = (-1.25, -0.75, -0.35),
    action_offsets: Sequence[float] = (0.05, 0.30, 0.55),
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return stable pre-serve and immediate action frame times."""

    def bounded(offset: float) -> float:
        return max(0.0, min(duration_seconds - 0.01, rally.start + offset))

    pre = tuple(dict.fromkeys(bounded(offset) for offset in pre_offsets))
    action = tuple(dict.fromkeys(bounded(offset) for offset in action_offsets))
    return pre, action


def _round(value: float | None, digits: int = 7) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _occupancy_dict(value: Any) -> dict[str, Any]:
    return {
        "nearAreaFraction": _round(value.near_area_fraction),
        "farAreaFraction": _round(value.far_area_fraction),
        "nearCount": value.near_count,
        "farCount": value.far_count,
        "nearScore": _round(value.near_score, 5),
        "farScore": _round(value.far_score, 5),
    }


def _weighted_margin(
    features: dict[str, float | None],
    names: Sequence[str],
    weights: Sequence[float],
) -> float | None:
    values = [features.get(name) for name in names]
    usable = [
        (float(value), float(weight))
        for value, weight in zip(values, weights, strict=True)
        if value is not None and math.isfinite(float(value)) and weight > 0
    ]
    if not usable:
        return None
    return sum(value * weight for value, weight in usable) / sum(
        weight for _, weight in usable
    )


def variant_scores(features: dict[str, float | None]) -> dict[str, float | None]:
    return {
        "pixelMotion": features.get("pixelMotionMargin"),
        "paletteChange": features.get("paletteChangeMargin"),
        "hogAreaChange": features.get("hogAreaChangeMargin"),
        "motionPalette": _weighted_margin(
            features,
            ("pixelMotionMargin", "paletteChangeMargin"),
            (0.65, 0.35),
        ),
        "motionPaletteHog": _weighted_margin(
            features,
            ("pixelMotionMargin", "paletteChangeMargin", "hogAreaChangeMargin"),
            (0.55, 0.25, 0.20),
        ),
        "baselineMotion": features.get("baselinePixelMotionMargin"),
        "baselinePalette": features.get("baselinePaletteChangeMargin"),
        "baselineHogArea": features.get("baselineHogAreaChangeMargin"),
        "baselineMotionPalette": _weighted_margin(
            features,
            ("baselinePixelMotionMargin", "baselinePaletteChangeMargin"),
            (0.65, 0.35),
        ),
    }


def _prediction(score: float | None, decision_margin: float) -> dict[str, Any]:
    if score is None or not math.isfinite(score):
        return {"direction": None, "predictedSide": None, "confidence": None}
    direction = NEAR_SIDE if score >= 0 else FAR_SIDE
    predicted = direction if abs(score) >= decision_margin else None
    return {
        "direction": direction,
        "predictedSide": predicted,
        "confidence": round(abs(score), 7),
    }


def _cue_dict(cue: ServingSideCue) -> dict[str, Any]:
    return {
        "side": cue.side,
        "strength": cue.strength,
        "reason": cue.reason,
        "matches": list(cue.matches),
    }


def analyze_rally(
    recording: RecordingLabels,
    rally: RallyLabel,
    capture: cv2.VideoCapture,
    hog: cv2.HOGDescriptor,
    *,
    split_fraction: float,
    decision_margin: float,
) -> dict[str, Any]:
    cue = infer_note_serving_side(rally.notes)
    pre_times, action_times = sample_times(rally, recording.duration_seconds)
    row: dict[str, Any] = {
        "rallyId": f"{recording.recording_id}:rally:{rally.index}",
        "recordingId": recording.recording_id,
        "environment": recording.environment,
        "sourceGroup": recording.source_group,
        "split": recording.split,
        "sourceType": recording.source_type,
        "targetStatus": recording.target_status,
        "candidateSource": recording.candidate_source,
        "rallyIndex": rally.index,
        "start": _round(rally.start, 5),
        "end": _round(rally.end, 5),
        "notes": rally.notes,
        "tags": list(rally.tags),
        "weakTarget": _cue_dict(cue),
        "preTimes": [_round(value, 5) for value in pre_times],
        "actionTimes": [_round(value, 5) for value in action_times],
    }
    try:
        pre_frames = [
            crop_roi(read_frame(capture, timestamp), recording.roi)
            for timestamp in pre_times
        ]
        action_frames = [
            crop_roi(read_frame(capture, timestamp), recording.roi)
            for timestamp in action_times
        ]
        if not pre_frames or not action_frames:
            raise RuntimeError("no pre-serve or action frames were sampled")
        reference = pre_frames[-1]
        near_motion, far_motion, motion_margin = pixel_motion(
            reference,
            action_frames,
            split_fraction,
        )
        near_palette, far_palette, palette_margin = palette_change(
            reference,
            action_frames,
            split_fraction,
        )
        near_baseline_motion, far_baseline_motion, baseline_motion_margin = pixel_motion(
            reference,
            action_frames,
            split_fraction,
            BASELINE_BAND_FRACTION,
        )
        near_baseline_palette, far_baseline_palette, baseline_palette_margin = palette_change(
            reference,
            action_frames,
            split_fraction,
            BASELINE_BAND_FRACTION,
        )
        pre_occupancy = hog_occupancy(pre_frames[-1], hog, split_fraction)
        action_occupancy = hog_occupancy(action_frames[0], hog, split_fraction)
        pre_baseline_occupancy = hog_occupancy(
            pre_frames[-1],
            hog,
            split_fraction,
            BASELINE_BAND_FRACTION,
        )
        action_baseline_occupancy = hog_occupancy(
            action_frames[0],
            hog,
            split_fraction,
            BASELINE_BAND_FRACTION,
        )
        area_margin = occupancy_change_margin(
            pre_occupancy,
            action_occupancy,
            "area",
        )
        count_margin = occupancy_change_margin(
            pre_occupancy,
            action_occupancy,
            "count",
        )
        baseline_area_margin = occupancy_change_margin(
            pre_baseline_occupancy,
            action_baseline_occupancy,
            "area",
        )
        features: dict[str, float | None] = {
            "nearPixelMotion": near_motion,
            "farPixelMotion": far_motion,
            "pixelMotionMargin": motion_margin,
            "nearPaletteChange": near_palette,
            "farPaletteChange": far_palette,
            "paletteChangeMargin": palette_margin,
            "nearBaselinePixelMotion": near_baseline_motion,
            "farBaselinePixelMotion": far_baseline_motion,
            "baselinePixelMotionMargin": baseline_motion_margin,
            "nearBaselinePaletteChange": near_baseline_palette,
            "farBaselinePaletteChange": far_baseline_palette,
            "baselinePaletteChangeMargin": baseline_palette_margin,
            "hogAreaChangeMargin": area_margin,
            "hogCountChangeMargin": count_margin,
            "baselineHogAreaChangeMargin": baseline_area_margin,
            "nearHogAreaBefore": pre_occupancy.near_area_fraction,
            "farHogAreaBefore": pre_occupancy.far_area_fraction,
            "nearHogAreaAfter": action_occupancy.near_area_fraction,
            "farHogAreaAfter": action_occupancy.far_area_fraction,
        }
        scores = variant_scores(features)
        row.update(
            {
                "status": "ok",
                "features": {key: _round(value) for key, value in features.items()},
                "hogBefore": _occupancy_dict(pre_occupancy),
                "hogAfter": _occupancy_dict(action_occupancy),
                "hogBaselineBefore": _occupancy_dict(pre_baseline_occupancy),
                "hogBaselineAfter": _occupancy_dict(action_baseline_occupancy),
                "variants": {
                    name: {
                        "score": _round(score),
                        **_prediction(score, decision_margin),
                    }
                    for name, score in scores.items()
                },
            }
        )
    except (RuntimeError, ValueError, cv2.error) as error:
        row.update({"status": "frame-error", "error": str(error), "features": {}, "variants": {}})
    return row


def _class_metrics(rows: Sequence[dict[str, Any]], variant: str, decision_margin: float) -> dict[str, Any]:
    known = [row for row in rows if row.get("weakTarget", {}).get("side") in SIDE_VALUES]
    usable = [
        row
        for row in known
        if row.get("status") == "ok"
        and isinstance(row.get("variants", {}).get(variant, {}).get("score"), (int, float))
    ]
    directional = [
        (row["weakTarget"]["side"], row["variants"][variant]["direction"])
        for row in usable
    ]
    decided = [
        (row["weakTarget"]["side"], row["variants"][variant]["predictedSide"])
        for row in usable
        if row["variants"][variant]["predictedSide"] in SIDE_VALUES
    ]

    def accuracy(items: Sequence[tuple[str, str | None]]) -> float | None:
        return (
            None
            if not items
            else round(sum(expected == predicted for expected, predicted in items) / len(items), 7)
        )

    def balanced(items: Sequence[tuple[str, str | None]]) -> float | None:
        recalls = []
        for side in SIDE_VALUES:
            side_items = [(expected, predicted) for expected, predicted in items if expected == side]
            if not side_items:
                continue
            recalls.append(sum(expected == predicted for expected, predicted in side_items) / len(side_items))
        return None if not recalls else round(sum(recalls) / len(recalls), 7)

    confusion = {
        expected: {
            predicted: sum(
                target == expected and actual == predicted
                for target, actual in directional
            )
            for predicted in SIDE_VALUES
        }
        for expected in SIDE_VALUES
    }
    return {
        "knownTargets": len(known),
        "usableScores": len(usable),
        "scoreCoverage": round(len(usable) / len(known), 7) if known else None,
        "directionalAccuracy": accuracy(directional),
        "directionalBalancedAccuracy": balanced(directional),
        "directionalConfusion": confusion,
        "decisionMargin": decision_margin,
        "decidedRows": len(decided),
        "decisionCoverage": round(len(decided) / len(known), 7) if known else None,
        "decisionAccuracy": accuracy(decided),
        "decisionBalancedAccuracy": balanced(decided),
        "weakTargetStrengths": {
            strength: sum(
                row.get("weakTarget", {}).get("strength") == strength for row in known
            )
            for strength in ("strong", "medium", "weak")
        },
    }


def summarize_metrics(
    rows: Sequence[dict[str, Any]],
    decision_margin: float,
    target_statuses: set[str] | None = None,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if target_statuses is None
        or str(row.get("targetStatus", "gold")) in target_statuses
    ]
    scopes: list[tuple[str, Sequence[dict[str, Any]]]] = [("pooled", eligible)]
    for key, values in (
        ("environment", sorted({str(row["environment"]) for row in eligible})),
        ("split", sorted({str(row["split"]) for row in eligible})),
        ("sourceType", sorted({str(row.get("sourceType", "unknown")) for row in eligible})),
        ("targetStatus", sorted({str(row.get("targetStatus", "gold")) for row in eligible})),
    ):
        for value in values:
            scopes.append(
                (
                    f"{key}:{value}",
                    [row for row in eligible if str(row.get(key, "unknown")) == value],
                )
            )
    return {
        scope: {
            "rows": len(scoped_rows),
            "knownTargets": sum(
                row.get("weakTarget", {}).get("side") in SIDE_VALUES for row in scoped_rows
            ),
            "variants": {
                name: _class_metrics(scoped_rows, name, decision_margin)
                for name in VARIANT_DEFINITIONS
            },
        }
        for scope, scoped_rows in scopes
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate serving-side variants from current rally labels and weak note cues."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels-dir", type=Path)
    source.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--environments",
        nargs="+",
        default=["beach", "grass", "indoor", "broadcast", "unknown"],
        choices=("beach", "grass", "indoor", "broadcast", "unknown"),
    )
    parser.add_argument("--split-fraction", type=float, default=0.5)
    parser.add_argument(
        "--decision-margin",
        type=float,
        default=0.05,
        help="abstain when an evidence margin is within this signed-distance band",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    labels_dir = arguments.labels_dir.expanduser().resolve() if arguments.labels_dir else None
    manifest_path = arguments.corpus_manifest.expanduser().resolve() if arguments.corpus_manifest else None
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    if not 0.1 < arguments.split_fraction < 0.9:
        raise ValueError("--split-fraction must be between 0.1 and 0.9")
    if not 0 <= arguments.decision_margin < 1:
        raise ValueError("--decision-margin must be between 0 and 1")
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

    rows: list[dict[str, Any]] = []
    for recording in recordings:
        if not recording.video_path.is_file():
            raise FileNotFoundError(f"video does not exist: {recording.video_path}")
        print(
            f"Analyzing {recording.recording_id}: {len(recording.rallies)} rallies from {recording.video_path}",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(recording.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {recording.video_path}")
        try:
            hog = create_hog()
            for index, rally in enumerate(recording.rallies, start=1):
                if index == 1 or index % 10 == 0 or index == len(recording.rallies):
                    print(
                        f"  {index}/{len(recording.rallies)} at {rally.start:.3f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                rows.append(
                    analyze_rally(
                        recording,
                        rally,
                        capture,
                        hog,
                        split_fraction=arguments.split_fraction,
                        decision_margin=arguments.decision_margin,
                    )
                )
        finally:
            capture.release()

    target_rows = [
        row
        for row in rows
        if row.get("weakTarget", {}).get("side") in SIDE_VALUES
        and row.get("targetStatus", "gold") != "candidate-only"
    ]
    evaluation_statuses = {"gold", "reviewed-draft"}
    statuses = sorted({str(row.get("targetStatus", "gold")) for row in rows})
    metrics_by_status = {
        status: summarize_metrics(rows, arguments.decision_margin, {status})
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
                }
                for recording in recordings
            ],
        },
        "protocol": {
            "rallyAnchor": "completed rallies[].start (serve-contact-to-dead-ball-v1)",
            "preOffsetsSeconds": [-1.25, -0.75, -0.35],
            "actionOffsetsSeconds": [0.05, 0.30, 0.55],
            "roi": "recording.roi, or full frame when absent",
            "sideSplitFraction": arguments.split_fraction,
            "baselineBandFraction": BASELINE_BAND_FRACTION,
            "nearSideDefinition": "lower/foreground ROI half",
            "farSideDefinition": "upper/background ROI half",
            "hogProposal": "opencv-hog-default-people-detector-v1 at max width 960",
            "weakTargetSource": "strict near/far serving phrases in existing rally notes",
            "weakTargetExclusions": [
                "unmentioned serving sides",
                "far player or foreground ball phrases without a serving cue",
                "conflicting cues and far-side-or-outside notes",
            ],
            "decisionMargin": arguments.decision_margin,
            "corpusScope": "full NAS manifest" if manifest_path else "label directory",
            "evaluationTargetStatuses": sorted(evaluation_statuses),
            "thresholdsAreDiagnostic": True,
        },
        "variants": VARIANT_DEFINITIONS,
        "summary": {
            "recordings": len(recordings),
            "rallies": len(rows),
            "targetableRallies": len(target_rows),
            "unknownRallies": len(rows) - len(target_rows),
            "targetStatusCounts": {
                status: sum(row.get("targetStatus", "gold") == status for row in rows)
                for status in statuses
            },
            "sourceTypeCounts": {
                source_type: sum(row.get("sourceType") == source_type for row in rows)
                for source_type in sorted({str(row.get("sourceType", "unknown")) for row in rows})
            },
            "targetSides": {
                side: sum(row.get("weakTarget", {}).get("side") == side for row in rows)
                for side in SIDE_VALUES
            },
            "targetStrengths": {
                strength: sum(
                    row.get("weakTarget", {}).get("side") in SIDE_VALUES
                    and row.get("weakTarget", {}).get("strength") == strength
                    for row in rows
                )
                for strength in ("strong", "medium", "weak")
            },
            "statuses": {
                status: sum(row.get("status") == status for row in rows)
                for status in sorted({str(row.get("status")) for row in rows})
            },
            "metrics": summarize_metrics(rows, arguments.decision_margin, evaluation_statuses),
            "metricsByTargetStatus": metrics_by_status,
        },
        "rallies": rows,
    }
    atomic_write_text(output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(output),
                "recordings": len(recordings),
                "rallies": len(rows),
                "targetableRallies": len(target_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
