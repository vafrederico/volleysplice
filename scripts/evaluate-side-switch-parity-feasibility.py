#!/usr/bin/env python3
"""Extract and evaluate rally-level side-parity evidence on reviewed phone videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_parity import (
    TRANSITION_MASK_SECONDS,
    RallyStateObservation,
    decode_persistent_flips,
    match_interval_proposals,
    overlaps_transition_mask,
    parity_state_at,
    recording_quality_floor,
    state_metrics,
)
from analysis.side_switch_v3 import FROZEN_RECORDING_SPLIT
from analysis.side_switch_v4 import FRAME_HEIGHT, FRAME_WIDTH, CourtGeometry
from analysis.side_switch_v5 import (
    FRAMES_PER_RALLY,
    build_orientation_profile,
    summarize_player_sequence,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_V4_FEATURES = REPORTS / "side-switch-v4-multiframe-normalized-features.json"
DEFAULT_MARKERS = REPORTS / "full-video-side-switch-markers-full-nas-v1.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-parity-feasibility-v1.json"
EXPECTED_SHA256 = {
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
    "v4Features": "4d9ae424a48b81b630c72ad8650fbc2cf68be41b350b593339fe774564717069",
    "markers": "50aabec1ecf5fb9d67edbdfa0a29a1997f4add5264c0b4a59a55b79ef6979e00",
}
VARIANTS = (
    {"name": "signed-persistence-1", "persistence": 1, "qualityFraction": 0.0},
    {"name": "signed-persistence-2", "persistence": 2, "qualityFraction": 0.0},
    {"name": "signed-persistence-3", "persistence": 3, "qualityFraction": 0.0},
    {
        "name": "signed-persistence-2-quality-half-median",
        "persistence": 2,
        "qualityFraction": 0.5,
    },
)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sample_times(rally: Mapping[str, Any]) -> tuple[float, ...]:
    start = float(rally["start"])
    end = float(rally["end"])
    duration = end - start
    if duration <= 0:
        raise ValueError("rally duration must be positive")
    first = start + min(0.20, duration * 0.08)
    last = end - min(0.15, duration * 0.08)
    if last <= first:
        first = start + duration * 0.20
        last = start + duration * 0.80
    return tuple(float(value) for value in np.linspace(first, last, FRAMES_PER_RALLY))


def _crop_and_resize(frame: np.ndarray, roi: Mapping[str, Any]) -> np.ndarray:
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
    return cv2.resize(
        frame[top:bottom, left:right],
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def _geometry(payload: Mapping[str, Any]) -> CourtGeometry:
    return CourtGeometry(
        net_y_ratio=float(payload["netYRatio"]),
        confidence=float(payload["confidence"]),
        detected_frames=int(payload["detectedFrames"]),
        sampled_frames=int(payload["sampledFrames"]),
    )


def _pool_matches(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposals = sum(int(value["proposals"]) for value in values)
    markers = sum(int(value["markers"]) for value in values)
    true_positives = sum(int(value["truePositives"]) for value in values)
    false_positives = sum(int(value["falsePositives"]) for value in values)
    false_negatives = sum(int(value["falseNegatives"]) for value in values)
    precision = true_positives / proposals if proposals else 0.0
    recall = true_positives / markers if markers else 0.0
    return {
        "proposals": proposals,
        "markers": markers,
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "averagePerVideo": {
            "proposals": proposals / len(values) if values else 0.0,
            "truePositives": true_positives / len(values) if values else 0.0,
            "falsePositives": false_positives / len(values) if values else 0.0,
            "falseNegatives": false_negatives / len(values) if values else 0.0,
        },
    }


def _all_boundaries(
    observations: Sequence[RallyStateObservation],
) -> list[dict[str, Any]]:
    ordered = sorted(observations, key=lambda value: value.rally_index)
    return [
        {
            "recordingId": right.recording_id,
            "start": min(left.end, right.start),
            "end": max(left.end, right.start),
            "leftRallyIndex": left.rally_index,
            "rightRallyIndex": right.rally_index,
        }
        for left, right in zip(ordered, ordered[1:], strict=False)
    ]


def _oracle_state_boundaries(
    observations: Sequence[RallyStateObservation],
) -> list[dict[str, Any]]:
    stable = [value for value in observations if not value.transition_masked]
    ordered = sorted(stable, key=lambda value: value.rally_index)
    return [
        {
            "recordingId": right.recording_id,
            "start": min(left.end, right.start),
            "end": max(left.end, right.start),
            "leftRallyIndex": left.rally_index,
            "rightRallyIndex": right.rally_index,
        }
        for left, right in zip(ordered, ordered[1:], strict=False)
        if left.parity_state != right.parity_state
    ]


def _evaluate_variant(
    name: str,
    persistence: int,
    quality_fraction: float,
    observations_by_recording: Mapping[str, Sequence[RallyStateObservation]],
    markers_by_recording: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    proposals: list[dict[str, Any]] = []
    for recording_id, observations in observations_by_recording.items():
        minimum_quality = recording_quality_floor(observations, quality_fraction)
        recording_proposals = decode_persistent_flips(
            observations,
            persistence=persistence,
            minimum_quality=minimum_quality,
        )
        proposals.extend(recording_proposals)
        per_video[recording_id] = {
            "minimumQuality": minimum_quality,
            "strict": match_interval_proposals(
                recording_proposals, markers_by_recording[recording_id], 0.0
            ),
            "margin4": match_interval_proposals(
                recording_proposals, markers_by_recording[recording_id], 4.0
            ),
            "proposals": recording_proposals,
        }
    return {
        "name": name,
        "persistence": persistence,
        "qualityFractionOfRecordingMedian": quality_fraction,
        "strict": _pool_matches([value["strict"] for value in per_video.values()]),
        "margin4": _pool_matches([value["margin4"] for value in per_video.values()]),
        "perVideo": per_video,
        "proposals": proposals,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest.expanduser().resolve(),
        "v4Features": args.v4_features.expanduser().resolve(),
        "markers": args.markers.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite parity artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"parity source identity changed: {hashes}")
    manifest = _load(paths["manifest"])
    v4_features = _load(paths["v4Features"])
    marker_payload = _load(paths["markers"])
    recording_ids = tuple(FROZEN_RECORDING_SPLIT["evaluation"])
    reviewed = {str(value) for value in marker_payload.get("reviewedRecordingIds", [])}
    missing_reviews = set(recording_ids) - reviewed
    if missing_reviews:
        raise ValueError(f"parity recordings are not fully reviewed: {missing_reviews}")
    markers_by_recording: dict[str, list[float]] = defaultdict(list)
    for marker in marker_payload.get("markers", []):
        if str(marker.get("recordingId")) in recording_ids:
            markers_by_recording[str(marker["recordingId"])].append(float(marker["time"]))
    for values in markers_by_recording.values():
        values.sort()
    if set(markers_by_recording) != set(recording_ids):
        raise ValueError("every reviewed parity recording must contain marker truth")
    if sum(map(len, markers_by_recording.values())) != 50:
        raise ValueError("expected the frozen 50-marker reviewed scope")
    records = {
        str(record["recordingId"]): record
        for record in manifest.get("records", [])
        if str(record.get("recordingId")) in recording_ids
    }
    extraction_v4 = v4_features.get("extractionAudit")
    if set(records) != set(recording_ids) or not isinstance(extraction_v4, Mapping):
        raise ValueError("manifest or v4 geometry does not cover parity scope")

    observations: list[RallyStateObservation] = []
    observation_rows: list[dict[str, Any]] = []
    extraction_audit: dict[str, Any] = {}
    for recording_id in recording_ids:
        record = records[recording_id]
        rallies = record.get("rallies", [])
        geometry = _geometry(extraction_v4[recording_id]["courtGeometry"])
        video_path = Path(str(record["videoPath"])).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"missing parity source video: {video_path}")
        print(
            f"Extracting parity {recording_id}: {len(rallies)} rallies",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open parity source video: {video_path}")
        summaries: dict[int, Any] = {}
        errors: dict[int, str] = {}
        try:
            for rally_number, rally in enumerate(rallies, start=1):
                try:
                    frames = [
                        _crop_and_resize(read_frame(capture, timestamp), record["roi"])
                        for timestamp in _sample_times(rally)
                    ]
                    summaries[rally_number] = summarize_player_sequence(frames, geometry)
                except (RuntimeError, ValueError, cv2.error) as error:
                    errors[rally_number] = str(error)
        finally:
            capture.release()
        if len(summaries) < 3:
            raise RuntimeError(f"parity extraction failed for {recording_id}")
        profile = build_orientation_profile(summaries)
        marker_times = markers_by_recording[recording_id]
        for rally_number, rally in enumerate(rallies, start=1):
            if rally_number not in summaries:
                continue
            start = float(rally["start"])
            end = float(rally["end"])
            midpoint = 0.5 * (start + end)
            value = RallyStateObservation(
                recording_id=recording_id,
                rally_index=rally_number,
                start=start,
                end=end,
                coordinate=float(profile.coordinates[rally_number]),
                quality=float(profile.qualities[rally_number]),
                parity_state=parity_state_at(midpoint, marker_times),
                transition_masked=overlaps_transition_mask(
                    start, end, marker_times, TRANSITION_MASK_SECONDS
                ),
            )
            observations.append(value)
            observation_rows.append(
                {
                    "recordingId": recording_id,
                    "rallyIndex": rally_number,
                    "start": start,
                    "end": end,
                    "midpoint": midpoint,
                    "parityState": value.parity_state,
                    "transitionMasked": value.transition_masked,
                    "orientationCoordinate": value.coordinate,
                    "orientationQuality": value.quality,
                }
            )
        extraction_audit[recording_id] = {
            "videoPath": str(video_path),
            "rallies": len(rallies),
            "summaries": len(summaries),
            "frameErrors": errors,
            "markers": marker_times,
            "courtGeometry": geometry.to_dict(),
            "anchorSeparation": profile.anchor_separation,
            "robustScale": profile.robust_scale,
        }

    observations_by_recording = {
        recording_id: [
            value for value in observations if value.recording_id == recording_id
        ]
        for recording_id in recording_ids
    }
    state_per_video = {
        recording_id: state_metrics(values)
        for recording_id, values in observations_by_recording.items()
    }
    pooled_state = state_metrics(observations)
    pooled_state["macroAccuracy"] = float(
        np.mean([value["accuracy"] for value in state_per_video.values()])
    )
    pooled_state["macroBalancedAccuracy"] = float(
        np.mean([value["balancedAccuracy"] for value in state_per_video.values()])
    )

    ceilings: dict[str, Any] = {}
    for ceiling_name, builder in (
        ("allConsecutiveRallyBoundaries", _all_boundaries),
        ("truthParityStableBoundaries", _oracle_state_boundaries),
    ):
        per_video = {
            recording_id: {
                "strict": match_interval_proposals(
                    builder(values), markers_by_recording[recording_id], 0.0
                ),
                "margin4": match_interval_proposals(
                    builder(values), markers_by_recording[recording_id], 4.0
                ),
            }
            for recording_id, values in observations_by_recording.items()
        }
        ceilings[ceiling_name] = {
            "strict": _pool_matches(
                [value["strict"] for value in per_video.values()]
            ),
            "margin4": _pool_matches(
                [value["margin4"] for value in per_video.values()]
            ),
            "perVideo": per_video,
        }

    variants = [
        _evaluate_variant(
            str(variant["name"]),
            int(variant["persistence"]),
            float(variant["qualityFraction"]),
            observations_by_recording,
            markers_by_recording,
        )
        for variant in VARIANTS
    ]
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-parity-feasibility-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": {
            "status": "opened-development-only",
            "recordingIds": list(recording_ids),
            "recordings": len(recording_ids),
            "physicalSwitchMarkers": sum(map(len, markers_by_recording.values())),
        },
        "profile": {
            "name": "V5-RALLY-PARITY-DIAGNOSTIC",
            "frameShape": [FRAME_HEIGHT, FRAME_WIDTH],
            "framesPerRally": FRAMES_PER_RALLY,
            "initialIdentity": "first-three-rally near/far palettes per recording",
            "stateDefinition": "state zero before first marker; every marker toggles parity",
            "transitionMaskSeconds": TRANSITION_MASK_SECONDS,
            "eventMatching": "one-to-one marker containment in proposal interval, strict and padded ±4 seconds",
            "runtimeIntent": "detector-free low-resolution V5 player proposals",
        },
        "stateEvaluation": {
            "pooled": pooled_state,
            "perVideo": state_per_video,
        },
        "candidateCeilings": ceilings,
        "variants": variants,
        "observations": observation_rows,
        "extractionAudit": extraction_audit,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v4-features", type=Path, default=DEFAULT_V4_FEATURES)
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = evaluate(_parser().parse_args())
    print(
        json.dumps(
            {
                "state": payload["stateEvaluation"]["pooled"],
                "ceilings": {
                    name: value["margin4"]
                    for name, value in payload["candidateCeilings"].items()
                },
                "variants": {
                    value["name"]: value["margin4"]
                    for value in payload["variants"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
