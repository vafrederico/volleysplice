#!/usr/bin/env python3
"""Extract T5 court-constrained temporal team features without labels."""

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
from analysis.side_switch_player_detector import QuantizedPersonDetector, detector_identity
from analysis.side_switch_t1_transport import materialized_profile_feature
from analysis.side_switch_t3_jersey_transport import endpoint_sample_times
from analysis.side_switch_t4_selective_far import T4_CORE_FEATURE_NAMES
from analysis.side_switch_t5_court_tracking import (
    T5_CORE_FEATURE_NAMES,
    T5_FEATURE_NAMES,
    court_tracked_transport_features,
    summarize_court_tracked_endpoint,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_T4 = REPORTS / "side-switch-t4-selective-far-court-detection-features-v1.json"
DEFAULT_DETECTOR_DIR = (
    ROOT / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
)
DEFAULT_OUTPUT = (
    REPORTS
    / "side-switch-t5-court-constrained-temporal-team-tracking-features-v1.json"
)
EXPECTED_T4_SHA256 = "443c0ded15cfddbb5e156f670c895375c8c43caede032a9b3ef5ae445ca4113a"
EXPECTED_ROWS = 704
EXPECTED_BOUNDARIES = 624
EXPECTED_INTERNAL = 80
EXPECTED_ENDPOINTS = 635
EXPECTED_FRAME_REQUESTS = 3175
EXPECTED_TILE_CALLS = 25400
MAX_WALL_SECONDS = 60.0 * 60.0
MAX_PEAK_MEMORY_MIB = 768.0


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    t4_path = args.t4_features.expanduser().resolve()
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T5 feature artifact: {output}")
    t4_hash = _sha256(t4_path)
    if args.enforce_source_hash and t4_hash != EXPECTED_T4_SHA256:
        raise ValueError(f"T5 T4 source identity changed: {t4_hash}")

    started = time.perf_counter()
    t4 = _load(t4_path)
    if (
        str(t4["engineeringDecision"]) != "pass"
        or int(t4["parity"]["rows"]) != EXPECTED_ROWS
        or str(t4["parity"]["priorFeatureValues"]) != "exact"
        or str(t4["parity"]["candidateIdAndOrder"]) != "exact"
    ):
        raise ValueError("T5 requires the exact passing T4 engineering artifact")
    detector = QuantizedPersonDetector(detector_dir, opencv_threads=args.opencv_threads)
    detector_source = detector_identity(detector_dir)
    source_rows = [dict(row) for row in t4["rows"]]
    recording_ids = tuple(str(value) for value in t4["scope"]["recordingIds"])
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        rows_by_recording[str(row["recordingId"])].append(row)

    endpoint_summaries: dict[tuple[str, float, float], Any] = {}
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
        prior = t4["extractionAudit"][recording_id]
        video_path = Path(str(prior["videoPath"])).resolve()
        if (
            not video_path.is_file()
            or video_path.stat().st_size != int(prior["videoSizeBytes"])
            or not str(prior["videoSha256"])
        ):
            raise ValueError(f"T5 video provenance changed for {recording_id}")
        net_y_ratio = float(prior["courtGeometry"]["netYRatio"])
        roi = prior["roi"]
        print(
            f"[{recording_number}/{len(recording_ids)}] {recording_id}: "
            f"{len(boundaries)} boundaries, {len(windows)} endpoint windows",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open T5 source video: {video_path}")
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
                summary = summarize_court_tracked_endpoint(
                    frames, net_y_ratio, detector
                )
                endpoint_summaries[key] = summary
                local_summaries.append(summary)
        finally:
            capture.release()
        any_team = [value.near.available or value.far.available for value in local_summaries]
        both_teams = [value.near.available and value.far.available for value in local_summaries]
        extraction_audit[recording_id] = {
            "videoPath": str(video_path),
            "videoSizeBytes": int(prior["videoSizeBytes"]),
            "videoSha256": str(prior["videoSha256"]),
            "videoHashSource": str(t4_path),
            "roi": roi,
            "courtGeometry": prior["courtGeometry"],
            "boundaryRows": len(boundaries),
            "endpointWindows": len(windows),
            "frameRequests": len(windows) * 5,
            "detectorTileCalls": len(windows) * 5 * 8,
            "frameErrors": 0,
            "endpointAnyQualifiedTeamFraction": float(np.mean(any_team)),
            "endpointBothQualifiedTeamsFraction": float(np.mean(both_teams)),
            "t4EndpointBothQualifiedTeamsFraction": float(
                prior["endpointBothQualifiedTeamsFraction"]
            ),
            "meanSelectedPlayersPerFrame": float(
                np.mean(
                    [count for value in local_summaries for count in value.selected_counts]
                )
            ),
            "meanNearObservedFrames": float(
                np.mean([value.near_track.observed_frames for value in local_summaries])
            ),
            "meanFarObservedFrames": float(
                np.mean([value.far_track.observed_frames for value in local_summaries])
            ),
            "meanNearTeamReliability": float(
                np.mean([value.near.reliability for value in local_summaries])
            ),
            "meanFarTeamReliability": float(
                np.mean([value.far.reliability for value in local_summaries])
            ),
            "backgroundMaskFallbacks": sum(
                value.background_fallbacks for value in local_summaries
            ),
            "fullDetectorInferenceMilliseconds": float(
                sum(value.full_detector_inference_milliseconds for value in local_summaries)
            ),
            "farDetectorInferenceMilliseconds": float(
                sum(value.far_detector_inference_milliseconds for value in local_summaries)
            ),
            "elapsedSeconds": time.perf_counter() - local_started,
        }

    if len(endpoint_summaries) != EXPECTED_ENDPOINTS:
        raise ValueError("T5 unique endpoint count changed")
    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    eligible = 0
    ineligible = 0
    for row in source_rows:
        updated = dict(row)
        updated["features"] = {
            str(name): float(value) for name, value in row["features"].items()
        }
        if str(row["kind"]) == "adjacent-rally-boundary":
            recording_id = str(row["recordingId"])
            before = endpoint_summaries[
                _window_key(recording_id, row["comparisonWindows"]["before"])
            ]
            after = endpoint_summaries[
                _window_key(recording_id, row["comparisonWindows"]["after"])
            ]
            values, diagnostics = court_tracked_transport_features(before, after)
            updated["features"].update(values)
            updated["t5CourtTrackedFarJerseyTransport"] = {
                "status": "ok",
                "before": before.to_diagnostic(),
                "after": after.to_diagnostic(),
                **diagnostics,
            }
            boundary_rows.append(updated)
            eligible += 1
        else:
            updated["t5CourtTrackedFarJerseyTransport"] = {
                "status": "not-eligible",
                "reason": "T5 is adjacent-boundary-only",
            }
            ineligible += 1
        output_rows.append(updated)
    if eligible != EXPECTED_BOUNDARIES or ineligible != EXPECTED_INTERNAL:
        raise ValueError("T5 boundary/internal scope changed")
    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("T5 row order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T5 changed prior feature {name}")

    endpoint_values = list(endpoint_summaries.values())
    endpoint_any = float(
        np.mean([value.near.available or value.far.available for value in endpoint_values])
    )
    endpoint_both = float(
        np.mean([value.near.available and value.far.available for value in endpoint_values])
    )
    minimum_recording_both = min(
        float(value["endpointBothQualifiedTeamsFraction"])
        for value in extraction_audit.values()
    )
    four_team = float(
        np.mean(
            [
                row["features"]["courtTrackedFarJerseyTeamReliabilityMinimum"] > 0.0
                and row["features"]["courtTrackedFarJerseyCrossSideSimilarityMinimum"]
                > 0.0
                for row in boundary_rows
            ]
        )
    )
    reliable_swap_nonzero = float(
        np.mean(
            [
                row["features"]["courtTrackedFarJerseyReliableSwapEvidence"] > 0.0
                for row in boundary_rows
            ]
        )
    )
    core_nonconstant = {
        name: float(np.ptp([row["features"][name] for row in boundary_rows])) > 1e-15
        for name in T5_CORE_FEATURE_NAMES
    }
    comparison_names = (*T5_CORE_FEATURE_NAMES, *T4_CORE_FEATURE_NAMES, *BASE_FEATURE_NAMES)
    correlations: list[dict[str, Any]] = []
    for index, left_name in enumerate(T5_CORE_FEATURE_NAMES):
        left_values = np.asarray([row["features"][left_name] for row in boundary_rows])
        right_names = comparison_names[index + 1 :] if index < 2 else comparison_names[3:]
        for right_name in right_names:
            if right_name in T5_CORE_FEATURE_NAMES and T5_CORE_FEATURE_NAMES.index(right_name) <= index:
                continue
            right_values = np.asarray(
                [materialized_profile_feature(row, right_name) for row in boundary_rows]
            )
            value = _spearman(left_values, right_values)
            if value is not None:
                correlations.append(
                    {"left": left_name, "right": right_name, "spearman": value}
                )
    core_correlations = [
        value for value in correlations if value["right"] in T5_CORE_FEATURE_NAMES
    ]
    t4_correlations = [
        value for value in correlations if value["right"] in T4_CORE_FEATURE_NAMES
    ]
    existing_correlations = [
        value for value in correlations if value["right"] in BASE_FEATURE_NAMES
    ]
    strongest = lambda values: max(  # noqa: E731
        values, key=lambda value: abs(value["spearman"]), default=None
    )
    strongest_core = strongest(core_correlations)
    strongest_t4 = strongest(t4_correlations)
    strongest_existing = strongest(existing_correlations)
    t4_endpoint_both = float(t4["engineering"]["endpointBothQualifiedTeamsFraction"])
    t4_four_team = float(t4["engineering"]["boundaryFourTeamObservabilityFraction"])
    t4_reliable_swap = float(t4["engineering"]["reliableSwapEvidenceNonzeroFraction"])
    checks = {
        "endpointAnyQualifiedTeamAtLeast90Percent": endpoint_any >= 0.90 - 1e-12,
        "endpointBothQualifiedTeamsGainAtLeast5pp": endpoint_both - t4_endpoint_both
        >= 0.05 - 1e-12,
        "everyRecordingBothTeamsAtLeast35Percent": minimum_recording_both
        >= 0.35 - 1e-12,
        "boundaryFourTeamVisibilityGainAtLeast5pp": four_team - t4_four_team
        >= 0.05 - 1e-12,
        "reliableSwapEvidenceNonzeroGainAtLeast3pp": reliable_swap_nonzero
        - t4_reliable_swap
        >= 0.03 - 1e-12,
        "coreFeaturesNonconstant": all(core_nonconstant.values()),
        "coreAbsoluteSpearmanBelow98Percent": strongest_core is None
        or abs(strongest_core["spearman"]) < 0.98 - 1e-12,
        "t4CoreAbsoluteSpearmanBelow98Percent": strongest_t4 is None
        or abs(strongest_t4["spearman"]) < 0.98 - 1e-12,
        "existingInputAbsoluteSpearmanBelow98Percent": strongest_existing is None
        or abs(strongest_existing["spearman"]) < 0.98 - 1e-12,
    }

    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    performance = {
        "elapsedSeconds": elapsed,
        "peakResidentMemoryMiB": peak_memory,
        "endpointWindows": len(endpoint_summaries),
        "frameRequests": len(endpoint_summaries) * 5,
        "detectorTileCalls": len(endpoint_summaries) * 5 * 8,
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
        raise ValueError("T5 frame/detector budget changed")
    checks.update(
        {
            "wallTimePassed": bool(performance["budget"]["wallTimePassed"]),
            "peakMemoryPassed": bool(performance["budget"]["peakMemoryPassed"]),
            "noFrameErrors": all(
                int(value["frameErrors"]) == 0 for value in extraction_audit.values()
            ),
        }
    )
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_t5_court_tracking.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t5-court-tracked-team-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": t4["scope"],
        "contract": {
            "eligibleCandidateKind": "adjacent-rally-boundary",
            "featureNames": list(T5_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T5_CORE_FEATURE_NAMES),
            "candidateGeneration": "unchanged",
            "labelUse": "none; T4 label-free artifact and source videos only",
            "modelSelectionUse": "opened-development only after engineering pass",
        },
        "parity": {
            "rows": len(output_rows),
            "eligibleBoundaryRows": eligible,
            "ineligibleInternalRows": ineligible,
            "priorFeatureValues": "exact",
            "candidateIdAndOrder": "exact",
        },
        "engineering": {
            "endpointAnyQualifiedTeamFraction": endpoint_any,
            "endpointBothQualifiedTeamsFraction": endpoint_both,
            "minimumRecordingBothQualifiedTeamsFraction": minimum_recording_both,
            "boundaryFourTeamVisibilityFraction": four_team,
            "reliableSwapEvidenceNonzeroFraction": reliable_swap_nonzero,
            "t4EndpointBothQualifiedTeamsFraction": t4_endpoint_both,
            "t4BoundaryFourTeamVisibilityFraction": t4_four_team,
            "t4ReliableSwapEvidenceNonzeroFraction": t4_reliable_swap,
            "endpointBothTeamGain": endpoint_both - t4_endpoint_both,
            "boundaryFourTeamGain": four_team - t4_four_team,
            "reliableSwapNonzeroGain": reliable_swap_nonzero - t4_reliable_swap,
            "coreFeatureNonconstant": core_nonconstant,
            "strongestCoreAbsoluteSpearman": strongest_core,
            "strongestT4CoreAbsoluteSpearman": strongest_t4,
            "strongestExistingInputAbsoluteSpearman": strongest_existing,
            "checks": checks,
            "correlations": correlations,
        },
        "performance": performance,
        "rows": output_rows,
        "extractionAudit": extraction_audit,
        "sources": {
            "t4Features": {"path": str(t4_path), "sha256": t4_hash},
            "extractor": {"path": str(script_path), "sha256": _sha256(script_path)},
            "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        },
        "personDetector": detector_source,
        "limitations": [
            "No label or model result was used for extraction or the engineering decision.",
            "The T4 detector, selective crop, five source frames, and jersey descriptors are unchanged.",
            "T5 tracks anonymous court-side team state; it does not claim player identity.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t4-features", type=Path, default=DEFAULT_T4)
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
                "engineering": {
                    key: value
                    for key, value in payload["engineering"].items()
                    if key != "correlations"
                },
                "performance": payload["performance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
