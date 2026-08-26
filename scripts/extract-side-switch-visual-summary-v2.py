#!/usr/bin/env python3
"""Extract parity-checked reusable visual summaries for Q1, C1, and P1."""

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
from analysis.side_switch_full_union_features import (
    ComparisonWindow,
    candidate_windows,
    whole_rally_sample_times,
)
from analysis.side_switch_v4 import FRAME_HEIGHT, FRAME_WIDTH, CourtGeometry
from analysis.side_switch_visual_summary_v2 import (
    C1_FEATURE_NAMES,
    P1_FEATURE_NAMES,
    Q1_FEATURE_NAMES,
    c1_features,
    current_visual_features,
    observation_payload,
    p1_features,
    q1_features,
    summarize_observation,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_CANDIDATES = REPORTS / "side-switch-candidate-union-v1-evaluation.json"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_V5_FEATURES = REPORTS / "side-switch-v5-player-orientation-features.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-visual-summary-v2-features-v1.json"
EXPECTED_SHA256 = {
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
    "candidates": "c4717a6e056b659fc7c63541a2eac13451518b0720dd7362dfb49643688edbc2",
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "v5Features": "4406fe47de0326c1257b8d9cf353116a9495f82d2e43c92e9721b1d6764ba5db",
}
PARITY_TOLERANCE = 1e-8
CURRENT_VISUAL_FEATURE_NAMES = (
    "v4BroadSameAssignmentCost",
    "v4TightSameAssignmentCost",
    "v4MeanSwapMargin",
    "v4GlobalAppearanceChange",
    "v4MaximumCameraShift",
    "v4MinimumAlignmentResponse",
    "playerSameAssignmentCost",
    "playerSwappedAssignmentCost",
    "playerSwapMargin",
    "playerOrientationFlipEvidence",
    "minimumPlayerSideSeparation",
    "playerSideSeparationChange",
    "beforePlayerPaletteInstability",
    "afterPlayerPaletteInstability",
    "playerGlobalAppearanceChange",
    "minimumProposalCoverage",
    "proposalCoverageChange",
    "minimumProposalCount",
    "proposalCountChange",
    "minimumNearSupport",
    "minimumFarSupport",
    "sideSupportImbalanceChange",
)


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


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _window_key(start: float, end: float) -> tuple[float, float]:
    return round(start, 9), round(end, 9)


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
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite visual-summary artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"visual-summary source identity changed: {hashes}")

    started = time.perf_counter()
    manifest = _load(paths["manifest"])
    candidate_payload = _load(paths["candidates"])
    source_payload = _load(paths["features"])
    v5_payload = _load(paths["v5Features"])
    recording_ids = tuple(str(value) for value in source_payload["scope"]["recordingIds"])
    records = {
        str(record["recordingId"]): record
        for record in manifest["records"]
        if str(record.get("recordingId")) in recording_ids
    }
    if set(records) != set(recording_ids):
        raise ValueError("manifest does not cover the frozen recording scope")
    source_rows = [dict(row) for row in source_payload["rows"]]
    rows_by_recording = {
        recording_id: [
            row for row in source_rows if str(row["recordingId"]) == recording_id
        ]
        for recording_id in recording_ids
    }
    selected_inventory = {
        recording_id: candidate_payload["selected"]["metrics"]["4.0"]["byRecording"][recording_id]["candidateInventory"]
        for recording_id in recording_ids
    }
    for recording_id in recording_ids:
        if [_candidate_identity(row) for row in rows_by_recording[recording_id]] != [
            _candidate_identity(row) for row in selected_inventory[recording_id]
        ]:
            raise ValueError(f"candidate/source row identity drifted for {recording_id}")

    output_rows: list[dict[str, Any]] = []
    observations_by_recording: dict[str, Any] = {}
    extraction_audit: dict[str, Any] = {}
    parity_differences: list[float] = []
    total_unique_windows = 0
    total_frame_requests = 0
    total_naive_frame_requests = 0

    for recording_number, recording_id in enumerate(recording_ids, start=1):
        record = records[recording_id]
        feedback_path = Path(str(record["labelPath"])).resolve()
        feedback_sha256 = _sha256(feedback_path)
        feedback = _load(feedback_path)
        # Feature extraction reads only the frozen inference ranges, never human labels.
        initial_inference = feedback["initialInference"]
        initial_inference_sha256 = _json_sha256(initial_inference)
        expected_initial_sha256 = str(
            candidate_payload["feedbackSources"][recording_id][
                "initialInferenceSha256"
            ]
        )
        if initial_inference_sha256 != expected_initial_sha256:
            raise ValueError(f"initial inference changed for {recording_id}")
        ranges = list(initial_inference["ranges"])
        range_index = {str(value["id"]): index for index, value in enumerate(ranges)}
        if len(range_index) != len(ranges):
            raise ValueError(f"duplicate range IDs for {recording_id}")
        geometry_payload = v5_payload["extractionAudit"][recording_id]["courtGeometry"]
        geometry = _geometry(geometry_payload)

        windows: dict[tuple[float, float], dict[str, Any]] = {}
        range_window_keys: list[tuple[float, float]] = []
        for value in ranges:
            window = ComparisonWindow(float(value["start"]), float(value["end"]))
            key = _window_key(window.start, window.end)
            range_window_keys.append(key)
            windows[key] = {
                "window": window,
                "observationId": f"{recording_id}:range:{value['id']}",
                "kind": "decoded-range",
                "sourceRangeId": str(value["id"]),
            }

        candidate_context: dict[str, Any] = {}
        for row in rows_by_recording[recording_id]:
            before, after, boundary_index = candidate_windows(row, ranges)
            before_key = _window_key(before.start, before.end)
            after_key = _window_key(after.start, after.end)
            for side, key, window in (
                ("before", before_key, before),
                ("after", after_key, after),
            ):
                if key not in windows:
                    windows[key] = {
                        "window": window,
                        "observationId": f"{row['eventId']}:{side}",
                        "kind": "internal-flank",
                        "sourceRangeId": row.get("sourceRangeId"),
                    }
            candidate_context[str(row["eventId"])] = {
                "beforeKey": before_key,
                "afterKey": after_key,
                "boundaryIndex": boundary_index,
            }

        video_path = Path(str(record["videoPath"])).resolve()
        print(
            f"[{recording_number}/{len(recording_ids)}] {recording_id}: hashing "
            f"{video_path.name}",
            file=sys.stderr,
            flush=True,
        )
        video_sha256 = _sha256(video_path) if args.hash_videos else None
        print(
            f"[{recording_number}/{len(recording_ids)}] {recording_id}: extracting "
            f"{len(windows)} unique sequences",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video: {video_path}")
        summaries: dict[tuple[float, float], Any] = {}
        observation_payloads: list[dict[str, Any]] = []
        try:
            for key, metadata in sorted(windows.items()):
                window = metadata["window"]
                sample_times = whole_rally_sample_times(window.start, window.end)
                frames = [
                    _crop_and_resize(read_frame(capture, timestamp), record["roi"])
                    for timestamp in sample_times
                ]
                observation = summarize_observation(frames, geometry)
                summaries[key] = observation
                payload = observation_payload(
                    observation,
                    observation_id=str(metadata["observationId"]),
                    start=window.start,
                    end=window.end,
                    sample_times=sample_times,
                    observation_kind=str(metadata["kind"]),
                )
                payload["sourceRangeId"] = metadata["sourceRangeId"]
                observation_payloads.append(payload)
        finally:
            capture.release()

        local_maximum = 0.0
        for row in rows_by_recording[recording_id]:
            context = candidate_context[str(row["eventId"])]
            before = summaries[context["beforeKey"]]
            after = summaries[context["afterKey"]]
            reconstructed = current_visual_features(before, after)
            source_features = row["features"]
            differences = {
                name: abs(float(reconstructed[name]) - float(source_features[name]))
                for name in CURRENT_VISUAL_FEATURE_NAMES
            }
            maximum = max(differences.values(), default=0.0)
            if maximum > PARITY_TOLERANCE:
                failing = max(differences, key=differences.get)
                raise ValueError(
                    f"visual parity failed for {row['eventId']} {failing}: {maximum}"
                )
            parity_differences.append(maximum)
            local_maximum = max(local_maximum, maximum)
            new_features = {
                str(name): float(value) for name, value in source_features.items()
            }
            new_features.update(q1_features(before, after))
            new_features.update(c1_features(before, after))
            boundary_index = context["boundaryIndex"]
            persistence_diagnostics = None
            if boundary_index is not None:
                before_start = max(0, int(boundary_index) - 2)
                after_end = min(len(ranges), int(boundary_index) + 4)
                before_context = [
                    summaries[range_window_keys[index]]
                    for index in range(before_start, int(boundary_index) + 1)
                ]
                after_context = [
                    summaries[range_window_keys[index]]
                    for index in range(int(boundary_index) + 1, after_end)
                ]
                persistence, persistence_diagnostics = p1_features(
                    before_context, after_context
                )
                new_features.update(persistence)
            updated = dict(row)
            updated["features"] = new_features
            updated["visualSummaryV2"] = {
                "beforeObservationId": windows[context["beforeKey"]]["observationId"],
                "afterObservationId": windows[context["afterKey"]]["observationId"],
                "currentVisualParityMaximumAbsoluteDifference": maximum,
                "persistenceDiagnostics": persistence_diagnostics,
            }
            output_rows.append(updated)

        observations_by_recording[recording_id] = observation_payloads
        total_unique_windows += len(windows)
        total_frame_requests += len(windows) * 7
        total_naive_frame_requests += len(rows_by_recording[recording_id]) * 14
        extraction_audit[recording_id] = {
            "videoPath": str(video_path),
            "videoSizeBytes": video_path.stat().st_size,
            "videoSha256": video_sha256,
            "feedbackPath": str(feedback_path),
            "feedbackSha256": feedback_sha256,
            "manifestFeedbackSha256": str(record["labelSha256"]),
            "initialInferenceSha256": initial_inference_sha256,
            "roi": record["roi"],
            "courtGeometry": geometry_payload,
            "decodedRanges": len(ranges),
            "candidateRows": len(rows_by_recording[recording_id]),
            "uniqueSequences": len(windows),
            "frameRequests": len(windows) * 7,
            "naiveCandidateFrameRequests": len(rows_by_recording[recording_id]) * 14,
            "maximumCurrentVisualParityDifference": local_maximum,
            "frameErrors": 0,
        }

    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("visual-summary output row order changed")
    boundary_rows = [
        row for row in output_rows if str(row["kind"]) == "adjacent-rally-boundary"
    ]
    if any(name not in row["features"] for row in output_rows for name in (*Q1_FEATURE_NAMES, *C1_FEATURE_NAMES)):
        raise ValueError("Q1/C1 extraction left a candidate incomplete")
    if any(name not in row["features"] for row in boundary_rows for name in P1_FEATURE_NAMES):
        raise ValueError("P1 extraction left a boundary incomplete")

    elapsed = time.perf_counter() - started
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_visual_summary_v2.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-visual-summary-v2-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-only",
        "scope": source_payload["scope"],
        "contract": {
            "framesPerSequence": 7,
            "sampling": "frozen 8%-to-92% whole-range and fixed internal flanks",
            "persistenceContextWidth": 3,
            "missingWithinPairPolicy": "neutral-zero-plus-context-fraction",
            "currentVisualFeatureNames": list(CURRENT_VISUAL_FEATURE_NAMES),
            "q1FeatureNames": list(Q1_FEATURE_NAMES),
            "c1FeatureNames": list(C1_FEATURE_NAMES),
            "p1FeatureNames": list(P1_FEATURE_NAMES),
            "labelUse": "none; only initialInference.ranges read from feedback files",
        },
        "parity": {
            "rows": len(parity_differences),
            "featureValuesPerRow": len(CURRENT_VISUAL_FEATURE_NAMES),
            "tolerance": PARITY_TOLERANCE,
            "maximumAbsoluteDifference": max(parity_differences, default=0.0),
            "status": "exact-within-tolerance",
        },
        "performance": {
            "elapsedSeconds": elapsed,
            "peakResidentMemoryMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "uniqueSequences": total_unique_windows,
            "frameRequests": total_frame_requests,
            "naiveCandidateFrameRequests": total_naive_frame_requests,
            "frameRequestReductionFraction": 1.0 - total_frame_requests / total_naive_frame_requests,
        },
        "rows": output_rows,
        "observationsByRecording": observations_by_recording,
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
            "All recordings and labels belong to the repeatedly opened development scope.",
            "Camera statistics are research diagnostics and are not yet ported to browser or Android.",
            "P1 values are emitted only for adjacent-boundary rows; internal flanks do not imply stable multi-rally context.",
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--hash-videos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    print(
        json.dumps(
            {
                "output": str(_parser().get_default("output")),
                "scope": payload["scope"],
                "parity": payload["parity"],
                "performance": payload["performance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
