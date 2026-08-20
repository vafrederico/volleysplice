#!/usr/bin/env python3
"""Extract quantized-player and adaptive-team side-switch v6 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_player_detector import (
    MODEL_SIZE_BYTES,
    QuantizedPersonDetector,
    detector_identity,
)
from analysis.side_switch_training_policy import validate_side_switch_fit_recordings
from analysis.side_switch_v3 import FROZEN_RECORDING_SPLIT, RECORDING_ROLE
from analysis.side_switch_v6 import (
    ADAPTIVE_PROTOTYPE_RATE,
    DETECTOR_FRAMES_PER_RALLY,
    FEATURE_ARTIFACT_KIND,
    FEATURE_ARTIFACT_SCHEMA_VERSION,
    VISUAL_FEATURE_NAMES,
    build_adaptive_orientation_profile,
    build_frozen_orientation_profile,
    detected_features,
    orientation_context,
    summarize_detected_sequence,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_V4_FEATURES = (
    ROOT / "reports/side-switch/side-switch-v4-multiframe-normalized-features.json"
)
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_DETECTOR_DIR = (
    ROOT
    / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/side-switch/side-switch-v6-detected-adaptive-features.json"
)
EXPECTED_SHA256 = {
    "v4Features": "4d9ae424a48b81b630c72ad8650fbc2cf68be41b350b593339fe774564717069",
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sample_times(rally: Mapping[str, Any]) -> tuple[float, ...]:
    start = float(rally["start"])
    end = float(rally["end"])
    duration = end - start
    if duration <= 0:
        raise ValueError("rally duration must be positive")
    fractions = np.linspace(0.15, 0.85, DETECTOR_FRAMES_PER_RALLY)
    return tuple(float(start + duration * fraction) for fraction in fractions)


def _crop_roi(frame: np.ndarray, roi: Mapping[str, Any]) -> np.ndarray:
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, round(float(roi.get("x", 0.0)) * width)))
    top = max(0, min(height - 1, round(float(roi.get("y", 0.0)) * height)))
    right = max(
        left + 1,
        min(
            width,
            round(
                (float(roi.get("x", 0.0)) + float(roi.get("width", 1.0)))
                * width
            ),
        ),
    )
    bottom = max(
        top + 1,
        min(
            height,
            round(
                (float(roi.get("y", 0.0)) + float(roi.get("height", 1.0)))
                * height
            ),
        ),
    )
    return frame[top:bottom, left:right]


def _profile_audit(profile: Any) -> dict[str, Any]:
    return {
        "anchorSeparation": profile.anchor_separation,
        "robustScale": profile.robust_scale,
        "updateRate": profile.update_rate,
        "updates": profile.update_count,
        "prototypeDrift": profile.prototype_drift,
    }


def extract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "v4Features": args.v4_features.expanduser().resolve(),
        "manifest": args.manifest.expanduser().resolve(),
    }
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v6 feature artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"v6 source identity changed: {hashes}")
    v4_features = _load(paths["v4Features"])
    manifest = _load(paths["manifest"])
    detector = QuantizedPersonDetector(
        detector_dir, opencv_threads=args.opencv_threads
    )
    detector_source = detector_identity(detector_dir)
    frozen_split = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if v4_features.get("frozenRecordingSplit") != frozen_split:
        raise ValueError("v4 feature artifact does not match the frozen v6 split")
    raw_rows = v4_features.get("rows")
    raw_records = manifest.get("records")
    extraction_v4 = v4_features.get("extractionAudit")
    if (
        not isinstance(raw_rows, list)
        or not isinstance(raw_records, list)
        or not isinstance(extraction_v4, Mapping)
    ):
        raise ValueError("v6 sources do not contain rows, records, and geometry")
    records = {
        str(record["recordingId"]): record
        for record in raw_records
        if isinstance(record, Mapping)
        and str(record.get("recordingId", "")) in RECORDING_ROLE
    }
    if set(records) != set(RECORDING_ROLE):
        raise ValueError("manifest does not cover the frozen v6 split")
    validate_side_switch_fit_recordings(list(FROZEN_RECORDING_SPLIT["train"]))

    rows = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            continue
        row = {
            key: value
            for key, value in raw_row.items()
            if key
            not in {
                "features",
                "status",
                "error",
                "orientationContext",
                "frozenOrientationContext",
            }
        }
        row["v4Features"] = dict(raw_row.get("features", {}))
        rows.append(row)
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_recording[str(row["recordingId"])].append(row)

    extraction_audit: dict[str, Any] = {}
    for role in ("train", "validation", "evaluation"):
        for recording_id in FROZEN_RECORDING_SPLIT[role]:
            recording_rows = sorted(
                rows_by_recording[recording_id], key=lambda row: int(row["gapOrder"])
            )
            record = records[recording_id]
            rallies = record["rallies"]
            geometry_payload = extraction_v4[recording_id]["courtGeometry"]
            net_y_ratio = float(geometry_payload["netYRatio"])
            video_path = Path(str(record["videoPath"])).resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"missing v6 source video: {video_path}")
            print(
                f"Extracting v6 {role} {recording_id}: "
                f"{len(recording_rows)} gaps, {len(rallies)} rallies",
                file=sys.stderr,
                flush=True,
            )
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open v6 source video: {video_path}")
            summaries: dict[int, Any] = {}
            errors: dict[int, str] = {}
            try:
                for rally_number, rally in enumerate(rallies, start=1):
                    try:
                        frames = [
                            _crop_roi(read_frame(capture, timestamp), record["roi"])
                            for timestamp in _sample_times(rally)
                        ]
                        summaries[rally_number] = summarize_detected_sequence(
                            frames, net_y_ratio, detector
                        )
                    except (RuntimeError, ValueError, cv2.error) as error:
                        errors[rally_number] = str(error)
            finally:
                capture.release()

            if len(summaries) < 3:
                raise RuntimeError(f"v6 orientation extraction failed for {recording_id}")
            adaptive_profile = build_adaptive_orientation_profile(summaries)
            frozen_profile = build_frozen_orientation_profile(summaries)
            for row in recording_rows:
                gap_order = int(row["gapOrder"])
                before = summaries.get(gap_order)
                after = summaries.get(gap_order + 1)
                adaptive_context = orientation_context(adaptive_profile, gap_order)
                frozen_context = orientation_context(frozen_profile, gap_order)
                row["orientationContext"] = {
                    name: round(float(value), 8)
                    for name, value in adaptive_context.items()
                }
                row["frozenOrientationContext"] = {
                    name: round(float(value), 8)
                    for name, value in frozen_context.items()
                }
                if before is None or after is None:
                    row["status"] = "frame-error"
                    row["error"] = errors.get(gap_order) or errors.get(gap_order + 1)
                    row["features"] = {}
                else:
                    row["status"] = "ok"
                    features = detected_features(
                        before, after, row["v4Features"], adaptive_context
                    )
                    if tuple(features) != VISUAL_FEATURE_NAMES:
                        raise RuntimeError("v6 feature order changed during extraction")
                    row["features"] = {
                        name: round(float(value), 8)
                        for name, value in features.items()
                    }
                del row["v4Features"]

            summary_values = list(summaries.values())
            extraction_audit[recording_id] = {
                "role": role,
                "videoPath": str(video_path),
                "reviewedGaps": len(recording_rows),
                "wholeSetRallies": len(rallies),
                "rallySummaries": len(summaries),
                "framesPerRally": DETECTOR_FRAMES_PER_RALLY,
                "detectorCallsPerRally": DETECTOR_FRAMES_PER_RALLY * 4,
                "frameErrors": len(errors),
                "roi": record["roi"],
                "courtGeometry": geometry_payload,
                "localization": {
                    "meanSelectedPlayersPerFrame": float(
                        np.mean([value.detection_count for value in summary_values])
                    ),
                    "meanRawCandidatesPerFrame": float(
                        np.mean([value.raw_candidate_count for value in summary_values])
                    ),
                    "meanConfidence": float(
                        np.mean([value.mean_confidence for value in summary_values])
                    ),
                    "meanTemporalConsistency": float(
                        np.mean(
                            [value.temporal_consistency for value in summary_values]
                        )
                    ),
                    "ralliesWithoutSelectedPlayer": sum(
                        value.detection_count <= 0 for value in summary_values
                    ),
                    "totalDetectorInferenceMilliseconds": float(
                        np.sum(
                            [value.inference_milliseconds for value in summary_values]
                        )
                    ),
                },
                "adaptiveOrientationProfile": _profile_audit(adaptive_profile),
                "frozenOrientationAblationProfile": _profile_audit(frozen_profile),
            }

    payload = {
        "schemaVersion": FEATURE_ARTIFACT_SCHEMA_VERSION,
        "kind": FEATURE_ARTIFACT_KIND,
        "createdAt": datetime.now(UTC).isoformat(),
        "profile": {
            "name": "DETECTED-ADAPTIVE29",
            "featureNames": list(VISUAL_FEATURE_NAMES),
            "framesPerRally": DETECTOR_FRAMES_PER_RALLY,
            "sampleRegion": "15%, 50%, and 85% of every rally",
            "courtNormalization": (
                "inherits frozen v4 per-recording net geometry for detection-side assignment"
            ),
            "playerIsolation": (
                "four overlapping ownership tiles through the pinned 3.48 MB "
                "block-int8 MediaPipe person detector; torso landmarks; maximum two "
                "players per court side"
            ),
            "sideAssignment": (
                "soft canonical hip-y assignment around y=0.56 after piecewise net normalization"
            ),
            "temporalConsistency": (
                "same-side hip match within 0.18 normalized frame distance across three samples"
            ),
            "orientationState": (
                "score-zero first-three-rally initialization plus confidence-gated "
                f"online team-prototype updates at maximum rate {ADAPTIVE_PROTOTYPE_RATE}"
            ),
            "onDeviceBudget": {
                "detectorModelBytes": MODEL_SIZE_BYTES,
                "detectorInvocationsPerRally": DETECTOR_FRAMES_PER_RALLY * 4,
                "runtime": "OpenCV DNN CPU; no cloud inference",
            },
            "labelResolution": v4_features.get("profile", {}).get("labelResolution"),
        },
        "frozenRecordingSplit": frozen_split,
        "collapseAudit": v4_features.get("collapseAudit"),
        "extractionAudit": extraction_audit,
        "rows": rows,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
        "personDetector": detector_source,
        "dataPolicy": {
            "oneRecordingOneSet": True,
            "startPointTotal": 0,
            "cadencePoints": 7,
            "fitExclusions": ["beach-source-02"],
            "evaluationLabelsUsedDuringExtraction": False,
        },
    }
    if any(not math.isfinite(value) for value in [ADAPTIVE_PROTOTYPE_RATE]):
        raise AssertionError("non-finite extraction policy")
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-features", type=Path, default=DEFAULT_V4_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    counts = {
        role: sum(row["role"] == role for row in payload["rows"])
        for role in FROZEN_RECORDING_SPLIT
    }
    errors = sum(row["status"] != "ok" for row in payload["rows"])
    print(json.dumps({"rows": counts, "frameErrors": errors}, indent=2))


if __name__ == "__main__":
    main()
