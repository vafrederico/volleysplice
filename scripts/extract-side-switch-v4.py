#!/usr/bin/env python3
"""Extract multi-frame, court-normalized side identity features for v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_training_policy import validate_side_switch_fit_recordings
from analysis.side_switch_v3 import FROZEN_RECORDING_SPLIT, RECORDING_ROLE
from analysis.side_switch_v4 import (
    FEATURE_ARTIFACT_KIND,
    FEATURE_ARTIFACT_SCHEMA_VERSION,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FRAMES_PER_RALLY,
    estimate_court_geometry,
    summarize_sequence,
    visual_features,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_V3_FEATURES = ROOT / "reports/side-switch/side-switch-v3-features.json"
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_OUTPUT = (
    ROOT / "reports/side-switch/side-switch-v4-multiframe-normalized-features.json"
)
EXPECTED_SHA256 = {
    "v3Features": "e50bad8020d0e1992b65617bd2a8d2ce7f9a5aa407fb0834c19e0921e35c05fa",
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
        min(width, round((float(roi.get("x", 0.0)) + float(roi.get("width", 1.0))) * width)),
    )
    bottom = max(
        top + 1,
        min(height, round((float(roi.get("y", 0.0)) + float(roi.get("height", 1.0))) * height)),
    )
    return cv2.resize(
        frame[top:bottom, left:right],
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def extract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "v3Features": args.v3_features.expanduser().resolve(),
        "manifest": args.manifest.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v4 feature artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"v4 source identity changed: {hashes}")
    v3_features = _load(paths["v3Features"])
    manifest = _load(paths["manifest"])
    if v3_features.get("frozenRecordingSplit") != {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }:
        raise ValueError("v3 label artifact does not match the frozen v4 split")
    raw_rows = v3_features.get("rows")
    raw_records = manifest.get("records")
    if not isinstance(raw_rows, list) or not isinstance(raw_records, list):
        raise ValueError("v4 sources do not contain rows and records")
    records = {
        str(record["recordingId"]): record
        for record in raw_records
        if isinstance(record, Mapping)
        and str(record.get("recordingId", "")) in RECORDING_ROLE
    }
    if set(records) != set(RECORDING_ROLE):
        raise ValueError("manifest does not cover the frozen v4 split")
    validate_side_switch_fit_recordings(list(FROZEN_RECORDING_SPLIT["train"]))

    rows = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            continue
        row = {
            key: value
            for key, value in raw_row.items()
            if key not in {"features", "status", "error"}
        }
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
            calibration_rallies = set(range(1, min(7, len(rallies)) + 1))
            needed_rallies = sorted(
                calibration_rallies
                | {
                    rally_index
                    for row in recording_rows
                    for rally_index in (int(row["gapOrder"]), int(row["gapOrder"]) + 1)
                }
            )
            video_path = Path(str(record["videoPath"])).resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"missing v4 source video: {video_path}")
            print(
                f"Extracting v4 {role} {recording_id}: "
                f"{len(recording_rows)} gaps, {len(needed_rallies)} rally sequences",
                file=sys.stderr,
                flush=True,
            )
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open v4 source video: {video_path}")
            sequences: dict[int, list[np.ndarray]] = {}
            errors: dict[int, str] = {}
            try:
                for rally_number in needed_rallies:
                    rally = rallies[rally_number - 1]
                    try:
                        sequences[rally_number] = [
                            _crop_and_resize(read_frame(capture, timestamp), record["roi"])
                            for timestamp in _sample_times(rally)
                        ]
                    except (RuntimeError, ValueError, cv2.error) as error:
                        errors[rally_number] = str(error)
            finally:
                capture.release()

            calibration_frames = [
                frame
                for rally_number in sorted(calibration_rallies)
                for frame in sequences.get(rally_number, [])
            ]
            if not calibration_frames:
                raise RuntimeError(f"v4 court calibration failed for {recording_id}")
            geometry = estimate_court_geometry(calibration_frames)
            summaries: dict[int, Any] = {}
            for rally_number, frames in sequences.items():
                try:
                    summaries[rally_number] = summarize_sequence(frames, geometry)
                except (RuntimeError, ValueError, cv2.error) as error:
                    errors[rally_number] = str(error)
            for row in recording_rows:
                gap_order = int(row["gapOrder"])
                before = summaries.get(gap_order)
                after = summaries.get(gap_order + 1)
                if before is None or after is None:
                    row["status"] = "frame-error"
                    row["error"] = errors.get(gap_order) or errors.get(gap_order + 1)
                    row["features"] = {}
                else:
                    row["status"] = "ok"
                    row["features"] = {
                        name: round(float(value), 8)
                        for name, value in visual_features(before, after).items()
                    }
            extraction_audit[recording_id] = {
                "role": role,
                "videoPath": str(video_path),
                "reviewedGaps": len(recording_rows),
                "rallySequences": len(summaries),
                "framesPerRally": FRAMES_PER_RALLY,
                "frameErrors": len(errors),
                "roi": record["roi"],
                "courtGeometry": geometry.to_dict(),
                "calibrationRallies": sorted(calibration_rallies),
            }

    payload = {
        "schemaVersion": FEATURE_ARTIFACT_SCHEMA_VERSION,
        "kind": FEATURE_ARTIFACT_KIND,
        "createdAt": datetime.now(UTC).isoformat(),
        "profile": {
            "name": "MULTIFRAME-NORMALIZED19",
            "frameShape": [FRAME_HEIGHT, FRAME_WIDTH],
            "framesPerRally": FRAMES_PER_RALLY,
            "sampleRegion": "8% through 92% of each rally",
            "courtCalibration": "long horizontal lines from the first seven rallies",
            "courtNormalization": "piecewise vertical warp placing the net at y=0.5",
            "sidePooling": "overlapping broad and disjoint tight court-relative bands",
            "cameraCompensation": "upper-frame phase-correlation translation",
            "proposalDetector": None,
            "operations": (
                "resize, Canny/Hough, remap, phase correlation, HSV histograms, "
                "temporal frame differences"
            ),
            "labelResolution": v3_features.get("profile", {}).get("labelResolution"),
        },
        "frozenRecordingSplit": {
            role: list(recording_ids)
            for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
        },
        "collapseAudit": v3_features.get("collapseAudit"),
        "extractionAudit": extraction_audit,
        "rows": rows,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]} for name in paths
        },
    }
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-features", type=Path, default=DEFAULT_V3_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
