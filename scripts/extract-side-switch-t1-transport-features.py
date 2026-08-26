#!/usr/bin/env python3
"""Extract immutable label-free endpoint identity transport features for T1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_appearance import read_frame
from analysis.side_switch_feature_development import BASE_FEATURE_NAMES
from analysis.side_switch_player_detector import (
    QuantizedPersonDetector,
    detector_identity,
)
from analysis.side_switch_t1_transport import (
    ENDPOINT_FRAME_FRACTIONS,
    T1_CORE_FEATURE_NAMES,
    T1_FEATURE_NAMES,
    endpoint_sample_times,
    materialized_profile_feature,
    summarize_endpoint,
    transport_features,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_VISUAL_SUMMARY = REPORTS / "side-switch-visual-summary-v2-features-v1.json"
DEFAULT_DETECTOR_DIR = (
    ROOT / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
)
DEFAULT_OUTPUT = REPORTS / "side-switch-t1-endpoint-identity-transport-features-v1.json"
EXPECTED_SHA256 = {
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "visualSummary": "6ce23b43018d04045ba783510ad86ef3c6d8767fdc435c7684481fca89ba4871",
}
EXPECTED_ROWS = 704
EXPECTED_BOUNDARIES = 624
EXPECTED_INTERNAL = 80
EXPECTED_ENDPOINTS = 635
EXPECTED_FRAME_REQUESTS = 1905
EXPECTED_TILE_CALLS = 7620
MAX_WALL_SECONDS = 60.0 * 60.0
MAX_PEAK_MEMORY_MIB = 768.0


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
    return np.ascontiguousarray(frame[top:bottom, left:right])


def _window_key(recording_id: str, window: Mapping[str, Any]) -> tuple[str, float, float]:
    return (
        recording_id,
        round(float(window["start"]), 9),
        round(float(window["end"]), 9),
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    order = np.argsort(numeric, kind="mergesort")
    result = np.empty(len(numeric), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and numeric[order[end]] == numeric[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if (
        first.shape != second.shape
        or first.ndim != 1
        or len(first) < 2
        or float(np.ptp(first)) <= 1e-15
        or float(np.ptp(second)) <= 1e-15
    ):
        return None
    value = float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1])
    return value if math.isfinite(value) else None


def extract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest.expanduser().resolve(),
        "features": args.features.expanduser().resolve(),
        "visualSummary": args.visual_summary.expanduser().resolve(),
    }
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T1 feature artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"T1 source identity changed: {hashes}")

    started = time.perf_counter()
    # The manifest is hash-bound but deliberately not parsed: Visual Summary V2 is the
    # no-label source for video paths, hashes, ROI, and court geometry.
    source = _load(paths["features"])
    visual = _load(paths["visualSummary"])
    detector = QuantizedPersonDetector(
        detector_dir, opencv_threads=args.opencv_threads
    )
    detector_source = detector_identity(detector_dir)
    if source["scope"] != visual["scope"]:
        raise ValueError("T1 source and visual-summary scopes differ")
    source_rows = [dict(row) for row in source["rows"]]
    visual_rows = visual["rows"]
    if [str(row["eventId"]) for row in source_rows] != [
        str(row["eventId"]) for row in visual_rows
    ]:
        raise ValueError("T1 source row identity/order differs from Visual Summary V2")
    if len(source_rows) != EXPECTED_ROWS:
        raise ValueError("T1 source row count changed")
    recording_ids = tuple(str(value) for value in source["scope"]["recordingIds"])
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        rows_by_recording[str(row["recordingId"])].append(row)

    all_endpoint_summaries: dict[tuple[str, float, float], Any] = {}
    extraction_audit: dict[str, Any] = {}
    for recording_number, recording_id in enumerate(recording_ids, start=1):
        rows = rows_by_recording[recording_id]
        boundaries = [
            row for row in rows if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        windows: dict[tuple[str, float, float], Mapping[str, Any]] = {}
        for row in boundaries:
            for side in ("before", "after"):
                window = row["comparisonWindows"][side]
                windows[_window_key(recording_id, window)] = window
        audit = visual["extractionAudit"][recording_id]
        video_path = Path(str(audit["videoPath"])).resolve()
        if (
            not video_path.is_file()
            or video_path.stat().st_size != int(audit["videoSizeBytes"])
            or not str(audit["videoSha256"])
        ):
            raise ValueError(f"T1 video provenance changed for {recording_id}")
        net_y_ratio = float(audit["courtGeometry"]["netYRatio"])
        roi = audit["roi"]
        print(
            f"[{recording_number}/{len(recording_ids)}] {recording_id}: "
            f"{len(boundaries)} boundaries, {len(windows)} endpoint windows",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open T1 source video: {video_path}")
        local_started = time.perf_counter()
        local_summaries = []
        try:
            for key, window in sorted(windows.items(), key=lambda value: value[0][1:]):
                frames = [
                    _crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in endpoint_sample_times(
                        float(window["start"]), float(window["end"])
                    )
                ]
                summary = summarize_endpoint(frames, net_y_ratio, detector)
                all_endpoint_summaries[key] = summary
                local_summaries.append(summary)
        finally:
            capture.release()
        any_tracklet = [bool(value.near or value.far) for value in local_summaries]
        both_sides = [bool(value.near and value.far) for value in local_summaries]
        extraction_audit[recording_id] = {
            "videoPath": str(video_path),
            "videoSizeBytes": int(audit["videoSizeBytes"]),
            "videoSha256": str(audit["videoSha256"]),
            "videoHashSource": str(paths["visualSummary"]),
            "roi": roi,
            "courtGeometry": audit["courtGeometry"],
            "boundaryRows": len(boundaries),
            "endpointWindows": len(windows),
            "frameRequests": len(windows) * len(ENDPOINT_FRAME_FRACTIONS),
            "detectorTileCalls": len(windows)
            * len(ENDPOINT_FRAME_FRACTIONS)
            * 4,
            "frameErrors": 0,
            "endpointAnyTrackletFraction": float(np.mean(any_tracklet)),
            "endpointBothSidesFraction": float(np.mean(both_sides)),
            "meanSelectedPlayersPerFrame": float(
                np.mean(
                    [
                        count
                        for value in local_summaries
                        for count in value.selected_counts
                    ]
                )
            ),
            "meanNearTracklets": float(
                np.mean([len(value.near) for value in local_summaries])
            ),
            "meanFarTracklets": float(
                np.mean([len(value.far) for value in local_summaries])
            ),
            "detectorInferenceMilliseconds": float(
                sum(value.detector_inference_milliseconds for value in local_summaries)
            ),
            "elapsedSeconds": time.perf_counter() - local_started,
        }

    if len(all_endpoint_summaries) != EXPECTED_ENDPOINTS:
        raise ValueError("T1 unique endpoint count changed")
    output_rows: list[dict[str, Any]] = []
    boundary_feature_rows: list[dict[str, Any]] = []
    eligible = 0
    ineligible = 0
    for row in source_rows:
        updated = dict(row)
        updated["features"] = {
            str(name): float(value) for name, value in row["features"].items()
        }
        if str(row["kind"]) == "adjacent-rally-boundary":
            recording_id = str(row["recordingId"])
            before = all_endpoint_summaries[
                _window_key(recording_id, row["comparisonWindows"]["before"])
            ]
            after = all_endpoint_summaries[
                _window_key(recording_id, row["comparisonWindows"]["after"])
            ]
            values, diagnostics = transport_features(before, after)
            updated["features"].update(values)
            updated["t1Transport"] = {
                "status": "ok",
                "before": before.to_diagnostic(),
                "after": after.to_diagnostic(),
                **diagnostics,
            }
            boundary_feature_rows.append(updated)
            eligible += 1
        else:
            updated["t1Transport"] = {
                "status": "not-eligible",
                "reason": "T1 is adjacent-boundary-only",
            }
            ineligible += 1
        output_rows.append(updated)

    if eligible != EXPECTED_BOUNDARIES or ineligible != EXPECTED_INTERNAL:
        raise ValueError("T1 boundary/internal scope changed")
    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("T1 row order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T1 changed existing feature {name}")

    endpoint_values = list(all_endpoint_summaries.values())
    endpoint_any = float(np.mean([bool(value.near or value.far) for value in endpoint_values]))
    endpoint_both = float(np.mean([bool(value.near and value.far) for value in endpoint_values]))
    coverage_values = np.asarray(
        [row["features"]["transportCoverageMinimum"] for row in boundary_feature_rows],
        dtype=np.float64,
    )
    boundary_coverage = float(np.mean(coverage_values > 0.0))
    core_nonconstant = {
        name: float(
            np.ptp([row["features"][name] for row in boundary_feature_rows])
        )
        > 1e-15
        for name in T1_CORE_FEATURE_NAMES
    }
    existing_names = tuple(BASE_FEATURE_NAMES)
    if len(existing_names) != 34:
        raise ValueError("T1 expected the exact existing 34-input signature")
    correlations: list[dict[str, Any]] = []
    for index, left_name in enumerate(T1_CORE_FEATURE_NAMES):
        left_values = np.asarray(
            [row["features"][left_name] for row in boundary_feature_rows]
        )
        compare_names = (*T1_CORE_FEATURE_NAMES[index + 1 :], *existing_names)
        for right_name in compare_names:
            right_values = np.asarray(
                [
                    materialized_profile_feature(row, right_name)
                    for row in boundary_feature_rows
                ]
            )
            value = _spearman(left_values, right_values)
            if value is not None:
                correlations.append(
                    {"left": left_name, "right": right_name, "spearman": value}
                )
    strongest = max(correlations, key=lambda value: abs(value["spearman"]))
    checks = {
        "endpointAnyTrackletAtLeast90Percent": endpoint_any >= 0.90 - 1e-12,
        "endpointBothSidesAtLeast60Percent": endpoint_both >= 0.60 - 1e-12,
        "boundaryTransportCoverageAtLeast50Percent": boundary_coverage >= 0.50 - 1e-12,
        "coreFeaturesNonconstant": all(core_nonconstant.values()),
        "maximumAbsoluteSpearmanBelow98Percent": abs(strongest["spearman"])
        < 0.98 - 1e-12,
    }

    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    performance = {
        "elapsedSeconds": elapsed,
        "peakResidentMemoryMiB": peak_memory,
        "endpointWindows": len(all_endpoint_summaries),
        "frameRequests": len(all_endpoint_summaries)
        * len(ENDPOINT_FRAME_FRACTIONS),
        "detectorTileCalls": len(all_endpoint_summaries)
        * len(ENDPOINT_FRAME_FRACTIONS)
        * 4,
        "budget": {
            "maximumWallSeconds": MAX_WALL_SECONDS,
            "maximumPeakResidentMemoryMiB": MAX_PEAK_MEMORY_MIB,
            "wallTimePassed": elapsed <= MAX_WALL_SECONDS,
            "peakMemoryPassed": peak_memory <= MAX_PEAK_MEMORY_MIB,
        },
    }
    if (
        performance["frameRequests"] != EXPECTED_FRAME_REQUESTS
        or performance["detectorTileCalls"] != EXPECTED_TILE_CALLS
    ):
        raise ValueError("T1 frame/detector budget changed")
    checks.update(
        {
            "wallTimePassed": bool(performance["budget"]["wallTimePassed"]),
            "peakMemoryPassed": bool(performance["budget"]["peakMemoryPassed"]),
            "noFrameErrors": all(
                int(value["frameErrors"]) == 0
                for value in extraction_audit.values()
            ),
        }
    )
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_t1_transport.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t1-endpoint-identity-transport-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": source["scope"],
        "contract": {
            "eligibleCandidateKind": "adjacent-rally-boundary",
            "featureNames": list(T1_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T1_CORE_FEATURE_NAMES),
            "endpointFrameFractions": list(ENDPOINT_FRAME_FRACTIONS),
            "candidateGeneration": "unchanged",
            "labelUse": "none; manifest hash-only and no audit/feedback/label artifact loaded",
            "modelSelectionUse": "prohibited until new recording-held gold exists",
        },
        "parity": {
            "rows": len(output_rows),
            "eligibleBoundaryRows": eligible,
            "ineligibleInternalRows": ineligible,
            "existingFeatureValues": "exact",
            "candidateIdAndOrder": "exact",
        },
        "observability": {
            "endpointAnyTrackletFraction": endpoint_any,
            "endpointBothSidesFraction": endpoint_both,
            "boundaryNonzeroTransportCoverageFraction": boundary_coverage,
            "coreFeatureNonconstant": core_nonconstant,
            "strongestAbsoluteSpearman": strongest,
            "checks": checks,
            "correlations": correlations,
        },
        "performance": performance,
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
        "personDetector": detector_source,
        "limitations": [
            "No T1 model was fit or selected because no eligible new side-switch gold exists.",
            "Tracklets are anonymous color descriptors, not supervised person identities.",
            "Product browser/Android latency has not been measured.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--visual-summary", type=Path, default=DEFAULT_VISUAL_SUMMARY)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    print(
        json.dumps(
            {
                "engineeringDecision": payload["engineeringDecision"],
                "parity": payload["parity"],
                "observability": {
                    key: value
                    for key, value in payload["observability"].items()
                    if key != "correlations"
                },
                "performance": payload["performance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
