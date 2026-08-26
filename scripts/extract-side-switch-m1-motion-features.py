#!/usr/bin/env python3
"""Extract immutable boundary-gap foreground-motion features for side-switch M1."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_m1_motion import (
    M1_FEATURE_NAMES,
    PAIR_COUNT,
    extract_residual_motion,
    m1_features,
    sample_pairs,
)
from analysis.side_switch_v4 import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CourtGeometry,
    normalize_court_frame,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_CANDIDATES = REPORTS / "side-switch-candidate-union-v1-evaluation.json"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_V5_FEATURES = REPORTS / "side-switch-v5-player-orientation-features.json"
DEFAULT_VISUAL_SUMMARY = REPORTS / "side-switch-visual-summary-v2-features-v1.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-m1-foreground-motion-features-v1.json"
EXPECTED_SHA256 = {
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
    "candidates": "c4717a6e056b659fc7c63541a2eac13451518b0720dd7362dfb49643688edbc2",
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "v5Features": "4406fe47de0326c1257b8d9cf353116a9495f82d2e43c92e9721b1d6764ba5db",
    "visualSummary": "6ce23b43018d04045ba783510ad86ef3c6d8767fdc435c7684481fca89ba4871",
}
MAX_WALL_SECONDS = 60.0 * 60.0
MAX_PEAK_MEMORY_MIB = 512.0


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _geometry(payload: Mapping[str, Any]) -> CourtGeometry:
    return CourtGeometry(
        net_y_ratio=float(payload["netYRatio"]),
        confidence=float(payload["confidence"]),
        detected_frames=int(payload["detectedFrames"]),
        sampled_frames=int(payload["sampledFrames"]),
    )


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


def _candidate_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["eventId"]),
        str(row["recordingId"]),
        str(row["kind"]),
        float(row["gapStart"]),
        float(row["gapEnd"]),
        float(row["transitionTime"]),
        row.get("sourceRangeId"),
    )


def extract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest.expanduser().resolve(),
        "candidates": args.candidates.expanduser().resolve(),
        "features": args.features.expanduser().resolve(),
        "v5Features": args.v5_features.expanduser().resolve(),
        "visualSummary": args.visual_summary.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite M1 feature artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"M1 source identity changed: {hashes}")

    started = time.perf_counter()
    manifest = _load(paths["manifest"])
    candidates = _load(paths["candidates"])
    source = _load(paths["features"])
    v5 = _load(paths["v5Features"])
    visual = _load(paths["visualSummary"])
    recording_ids = tuple(str(value) for value in source["scope"]["recordingIds"])
    records = {
        str(record["recordingId"]): record
        for record in manifest["records"]
        if str(record.get("recordingId")) in recording_ids
    }
    if set(records) != set(recording_ids):
        raise ValueError("manifest does not cover the frozen M1 scope")
    visual_audit = visual["extractionAudit"]
    source_rows = [dict(row) for row in source["rows"]]
    rows_by_recording = {
        recording_id: [
            row for row in source_rows if str(row["recordingId"]) == recording_id
        ]
        for recording_id in recording_ids
    }
    selected = candidates["selected"]["metrics"]["4.0"]["byRecording"]
    for recording_id in recording_ids:
        inventory = selected[recording_id]["candidateInventory"]
        if [_candidate_identity(row) for row in rows_by_recording[recording_id]] != [
            _candidate_identity(row) for row in inventory
        ]:
            raise ValueError(f"M1 candidate/source identity drifted for {recording_id}")

    output_rows: list[dict[str, Any]] = []
    extraction_audit: dict[str, Any] = {}
    eligible_rows = 0
    ineligible_rows = 0
    total_requested_frames = 0
    total_unique_frames = 0
    total_flow_pairs = 0
    for recording_number, recording_id in enumerate(recording_ids, start=1):
        record = records[recording_id]
        local_rows = rows_by_recording[recording_id]
        boundary_rows = [
            row
            for row in local_rows
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        geometry_payload = v5["extractionAudit"][recording_id]["courtGeometry"]
        geometry = _geometry(geometry_payload)
        video_path = Path(str(record["videoPath"])).resolve()
        source_video = visual_audit[recording_id]
        if (
            str(video_path) != str(Path(str(source_video["videoPath"])).resolve())
            or video_path.stat().st_size != int(source_video["videoSizeBytes"])
            or not source_video.get("videoSha256")
        ):
            raise ValueError(f"M1 video provenance changed for {recording_id}")

        candidate_pairs = {
            str(row["eventId"]): sample_pairs(
                float(row["gapStart"]), float(row["gapEnd"])
            )
            for row in boundary_rows
        }
        timestamps = sorted(
            {
                round(timestamp, 9)
                for pairs in candidate_pairs.values()
                for pair in pairs
                for timestamp in pair
            }
        )
        print(
            f"[{recording_number}/{len(recording_ids)}] {recording_id}: "
            f"{len(boundary_rows)} boundaries, {len(timestamps)} unique frames",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video: {video_path}")
        frames: dict[float, np.ndarray] = {}
        local_started = time.perf_counter()
        try:
            for timestamp in timestamps:
                frame = _crop_and_resize(read_frame(capture, timestamp), record["roi"])
                frames[timestamp] = normalize_court_frame(frame, geometry)
        finally:
            capture.release()

        local_results: dict[str, Any] = {}
        for row in boundary_rows:
            event_id = str(row["eventId"])
            pairs = candidate_pairs[event_id]
            motions = [
                extract_residual_motion(
                    frames[round(left, 9)], frames[round(right, 9)]
                )
                for left, right in pairs
            ]
            features, diagnostics = m1_features(
                motions, float(row["gapEnd"]) - float(row["gapStart"])
            )
            local_results[event_id] = {
                "features": features,
                "diagnostics": {
                    **diagnostics,
                    "samplePairs": [
                        {"before": left, "after": right} for left, right in pairs
                    ],
                },
            }

        for row in local_rows:
            event_id = str(row["eventId"])
            updated = dict(row)
            updated["features"] = {
                str(name): float(value)
                for name, value in row["features"].items()
            }
            if event_id in local_results:
                updated["features"].update(local_results[event_id]["features"])
                updated["m1Motion"] = {
                    "status": "ok",
                    **local_results[event_id]["diagnostics"],
                }
                eligible_rows += 1
            else:
                updated["m1Motion"] = {
                    "status": "not-eligible",
                    "reason": "first M1 arm is adjacent-boundary-only",
                }
                ineligible_rows += 1
            output_rows.append(updated)

        requested = len(boundary_rows) * PAIR_COUNT * 2
        flow_pairs = len(boundary_rows) * PAIR_COUNT
        total_requested_frames += requested
        total_unique_frames += len(timestamps)
        total_flow_pairs += flow_pairs
        extraction_audit[recording_id] = {
            "videoPath": str(video_path),
            "videoSizeBytes": video_path.stat().st_size,
            "videoSha256": source_video["videoSha256"],
            "videoHashSource": str(paths["visualSummary"]),
            "feedbackPath": source_video["feedbackPath"],
            "feedbackSha256": source_video["feedbackSha256"],
            "initialInferenceSha256": source_video["initialInferenceSha256"],
            "roi": record["roi"],
            "courtGeometry": geometry_payload,
            "boundaryRows": len(boundary_rows),
            "internalRows": len(local_rows) - len(boundary_rows),
            "requestedFrames": requested,
            "uniqueFrames": len(timestamps),
            "flowPairs": flow_pairs,
            "elapsedSeconds": time.perf_counter() - local_started,
            "frameErrors": 0,
        }

    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("M1 output row order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"M1 changed existing feature {name}")
    if eligible_rows != 624 or ineligible_rows != 80:
        raise ValueError("M1 boundary/internal scope changed")
    if total_requested_frames != 7488 or total_flow_pairs != 3744:
        raise ValueError("M1 frame or flow-pair budget changed")
    if any(
        name not in row["features"]
        for row in output_rows
        if row["m1Motion"]["status"] == "ok"
        for name in M1_FEATURE_NAMES
    ):
        raise ValueError("M1 left an eligible row incomplete")

    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    budget = {
        "maximumWallSeconds": MAX_WALL_SECONDS,
        "maximumPeakResidentMemoryMiB": MAX_PEAK_MEMORY_MIB,
        "wallTimePassed": elapsed <= MAX_WALL_SECONDS,
        "peakMemoryPassed": peak_memory <= MAX_PEAK_MEMORY_MIB,
    }
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_m1_motion.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-m1-foreground-motion-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-only",
        "scope": source["scope"],
        "contract": {
            "eligibleCandidateKind": "adjacent-rally-boundary",
            "pairCount": PAIR_COUNT,
            "framesPerEligibleCandidate": PAIR_COUNT * 2,
            "featureNames": list(M1_FEATURE_NAMES),
            "candidateGeneration": "unchanged",
            "labelUse": "none",
        },
        "parity": {
            "rows": len(output_rows),
            "eligibleBoundaryRows": eligible_rows,
            "ineligibleInternalRows": ineligible_rows,
            "existingFeatureValues": "exact",
            "candidateIdAndOrder": "exact",
        },
        "performance": {
            "elapsedSeconds": elapsed,
            "peakResidentMemoryMiB": peak_memory,
            "requestedFrames": total_requested_frames,
            "uniqueFrames": total_unique_frames,
            "exactTimestampReuseFraction": 1.0
            - total_unique_frames / total_requested_frames,
            "flowPairs": total_flow_pairs,
            "budget": budget,
        },
        "rows": output_rows,
        "extractionAudit": extraction_audit,
        "sources": {
            **{
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "extractor": {"path": str(script_path), "sha256": _sha256(script_path)},
            "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        },
        "limitations": [
            "All recordings belong to the repeatedly opened development scope.",
            "M1 is boundary-only and cannot recover markers outside the frozen candidate union.",
            "Product browser/Android latency has not been measured.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--v5-features", type=Path, default=DEFAULT_V5_FEATURES)
    parser.add_argument("--visual-summary", type=Path, default=DEFAULT_VISUAL_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    print(
        json.dumps(
            {
                "scope": payload["scope"],
                "parity": payload["parity"],
                "performance": payload["performance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
