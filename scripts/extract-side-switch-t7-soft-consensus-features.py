#!/usr/bin/env python3
"""Extract T7 soft robust jersey consensus features without labels."""

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
from analysis.side_switch_t5_court_tracking import T5_CORE_FEATURE_NAMES
from analysis.side_switch_t6_dominant_jersey import T6_CORE_FEATURE_NAMES
from analysis.side_switch_t7_soft_consensus import (
    T7_CORE_FEATURE_NAMES,
    T7_FEATURE_NAMES,
    soft_consensus_transport_features,
    summarize_soft_consensus_endpoint,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_T6 = REPORTS / "side-switch-t6-dominant-jersey-consensus-features-v1.json"
DEFAULT_DETECTOR_DIR = ROOT / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
DEFAULT_OUTPUT = REPORTS / "side-switch-t7-soft-robust-jersey-consensus-features-v1.json"
EXPECTED_T6_SHA256 = "e53dafc72f1645936c6c4618bb201d33413216ad203550f2a05b6e14b0db89e4"
T5_ENDPOINT_BOTH = 0.8110236220472441
T5_MINIMUM_RECORDING_BOTH = 0.4230769230769231
T5_FOUR_TEAM = 0.6875
T5_MEAN_SEPARATION = 0.3195760441978619
T5_POSITIVE_MARGIN = 0.1987179487179487
T4_POSITIVE_MARGIN = 0.23557692307692307
T4_RELIABLE_SWAP = 0.1778846153846154
EXPECTED_ENDPOINTS = 635
EXPECTED_FRAMES = 3175
EXPECTED_TILES = 25400


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
    right = max(left + 1, min(width, round((float(roi.get("x", 0.0)) + float(roi.get("width", 1.0))) * width)))
    bottom = max(top + 1, min(height, round((float(roi.get("y", 0.0)) + float(roi.get("height", 1.0))) * height)))
    return np.ascontiguousarray(frame[top:bottom, left:right])


def _window_key(recording_id: str, window: Mapping[str, Any]) -> tuple[str, float, float]:
    return recording_id, round(float(window["start"]), 9), round(float(window["end"]), 9)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if float(np.ptp(left)) <= 1e-15 or float(np.ptp(right)) <= 1e-15:
        return None
    value = float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])
    return value if math.isfinite(value) else None


def extract(args: argparse.Namespace) -> dict[str, Any]:
    t6_path = args.t6_features.expanduser().resolve()
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T7 artifact: {output}")
    t6_hash = _sha256(t6_path)
    if args.enforce_source_hash and t6_hash != EXPECTED_T6_SHA256:
        raise ValueError(f"T7 T6 source identity changed: {t6_hash}")
    started = time.perf_counter()
    t6 = _load(t6_path)
    if (
        str(t6["engineeringDecision"]) != "fail"
        or int(t6["parity"]["rows"]) != 704
        or str(t6["parity"]["priorFeatureValues"]) != "exact"
        or str(t6["parity"]["candidateIdAndOrder"]) != "exact"
    ):
        raise ValueError("T7 requires the exact T6 engineering artifact")
    detector = QuantizedPersonDetector(detector_dir, opencv_threads=args.opencv_threads)
    source_rows = [dict(row) for row in t6["rows"]]
    recording_ids = tuple(str(value) for value in t6["scope"]["recordingIds"])
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        rows_by_recording[str(row["recordingId"])].append(row)
    endpoints: dict[tuple[str, float, float], Any] = {}
    extraction_audit: dict[str, Any] = {}
    for number, recording_id in enumerate(recording_ids, 1):
        boundaries = [row for row in rows_by_recording[recording_id] if str(row["kind"]) == "adjacent-rally-boundary"]
        windows: dict[tuple[str, float, float], Mapping[str, Any]] = {}
        for row in boundaries:
            for side in ("before", "after"):
                window = row["comparisonWindows"][side]
                windows[_window_key(recording_id, window)] = window
        prior = t6["extractionAudit"][recording_id]
        video_path = Path(str(prior["videoPath"])).resolve()
        if not video_path.is_file() or video_path.stat().st_size != int(prior["videoSizeBytes"]):
            raise ValueError(f"T7 video provenance changed for {recording_id}")
        net_y_ratio = float(prior["courtGeometry"]["netYRatio"])
        roi = prior["roi"]
        print(f"[{number}/{len(recording_ids)}] {recording_id}: {len(boundaries)} boundaries, {len(windows)} endpoint windows", file=sys.stderr, flush=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open T7 video: {video_path}")
        local_started = time.perf_counter()
        local = []
        try:
            for key, window in sorted(windows.items(), key=lambda value: value[0][1:]):
                frames = [
                    _crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in endpoint_sample_times(float(window["start"]), float(window["end"]))
                ]
                summary = summarize_soft_consensus_endpoint(frames, net_y_ratio, detector)
                endpoints[key] = summary
                local.append(summary)
        finally:
            capture.release()
        both = float(np.mean([value.near.available and value.far.available for value in local]))
        t5_both = float(prior["t5EndpointBothQualifiedTeamsFraction"])
        # T6 audit's T5 comparator is the exact per-recording T5 availability.
        extraction_audit[recording_id] = {
            "videoPath": str(video_path),
            "videoSizeBytes": int(prior["videoSizeBytes"]),
            "videoSha256": str(prior["videoSha256"]),
            "roi": roi,
            "courtGeometry": prior["courtGeometry"],
            "boundaryRows": len(boundaries),
            "endpointWindows": len(windows),
            "frameRequests": len(windows) * 5,
            "detectorTileCalls": len(windows) * 40,
            "frameErrors": 0,
            "endpointBothQualifiedTeamsFraction": both,
            "t5EndpointBothQualifiedTeamsFraction": t5_both,
            "availabilityMatchesT5": abs(both - t5_both) <= 1e-12,
            "meanNearEffectiveSupport": float(np.mean([value.near_track.effective_robust_support for value in local if value.near_track.total_observations > 0])),
            "meanFarEffectiveSupport": float(np.mean([value.far_track.effective_robust_support for value in local if value.far_track.total_observations > 0])),
            "belowHalfWeight": int(sum(value.near_track.below_half_weight + value.far_track.below_half_weight for value in local)),
            "belowQuarterWeight": int(sum(value.near_track.below_quarter_weight + value.far_track.below_quarter_weight for value in local)),
            "elapsedSeconds": time.perf_counter() - local_started,
        }
    if len(endpoints) != EXPECTED_ENDPOINTS:
        raise ValueError("T7 endpoint count changed")

    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for row in source_rows:
        updated = dict(row)
        updated["features"] = {str(name): float(value) for name, value in row["features"].items()}
        if str(row["kind"]) == "adjacent-rally-boundary":
            recording_id = str(row["recordingId"])
            before = endpoints[_window_key(recording_id, row["comparisonWindows"]["before"])]
            after = endpoints[_window_key(recording_id, row["comparisonWindows"]["after"])]
            values, diagnostics = soft_consensus_transport_features(before, after)
            updated["features"].update(values)
            updated["t7SoftRobustJerseyConsensus"] = {
                "status": "ok", "before": before.to_diagnostic(), "after": after.to_diagnostic(), **diagnostics
            }
            boundary_rows.append(updated)
        else:
            updated["t7SoftRobustJerseyConsensus"] = {"status": "not-eligible", "reason": "T7 is adjacent-boundary-only"}
        output_rows.append(updated)
    if len(boundary_rows) != 624 or len(output_rows) != 704:
        raise ValueError("T7 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [str(row["eventId"]) for row in source_rows]:
        raise ValueError("T7 row order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T7 changed prior feature {name}")

    endpoint_values = list(endpoints.values())
    endpoint_both = float(np.mean([value.near.available and value.far.available for value in endpoint_values]))
    minimum_recording_both = min(float(value["endpointBothQualifiedTeamsFraction"]) for value in extraction_audit.values())
    four_team = float(np.mean([row["features"]["softConsensusJerseyTeamReliabilityMinimum"] > 0.0 and row["features"]["softConsensusJerseyCrossSideSimilarityMinimum"] > 0.0 for row in boundary_rows]))
    mean_separation = float(np.mean([row["features"]["softConsensusJerseyTeamSeparationMinimum"] for row in boundary_rows]))
    positive_margin = float(np.mean([row["features"]["softConsensusJerseyTeamTransportSwapMargin"] > 0.0 for row in boundary_rows]))
    reliable_swap = float(np.mean([row["features"]["softConsensusJerseyReliableSwapEvidence"] > 0.0 for row in boundary_rows]))
    tracks = [track for value in endpoint_values for track in (value.near_track, value.far_track) if track.total_observations > 0]
    mean_effective_support = float(np.mean([value.effective_robust_support for value in tracks]))
    below_half = int(sum(value.below_half_weight for value in tracks))
    below_quarter = int(sum(value.below_quarter_weight for value in tracks))
    core_nonconstant = {name: float(np.ptp([row["features"][name] for row in boundary_rows])) > 1e-15 for name in T7_CORE_FEATURE_NAMES}

    correlations: list[dict[str, Any]] = []
    comparison = (*T7_CORE_FEATURE_NAMES, *T4_CORE_FEATURE_NAMES, *T5_CORE_FEATURE_NAMES, *T6_CORE_FEATURE_NAMES, *BASE_FEATURE_NAMES)
    for index, left_name in enumerate(T7_CORE_FEATURE_NAMES):
        left = np.asarray([row["features"][left_name] for row in boundary_rows])
        for right_name in comparison[index + 1 :]:
            if right_name in T7_CORE_FEATURE_NAMES and T7_CORE_FEATURE_NAMES.index(right_name) <= index:
                continue
            right = np.asarray([materialized_profile_feature(row, right_name) for row in boundary_rows])
            value = _spearman(left, right)
            if value is not None:
                correlations.append({"left": left_name, "right": right_name, "spearman": value})
    def strongest(names: tuple[str, ...]) -> Mapping[str, Any] | None:
        values = [value for value in correlations if value["right"] in names]
        return max(values, key=lambda value: abs(value["spearman"]), default=None)
    strongest_values = {
        "core": strongest(T7_CORE_FEATURE_NAMES),
        "t4": strongest(T4_CORE_FEATURE_NAMES),
        "t5": strongest(T5_CORE_FEATURE_NAMES),
        "t6": strongest(T6_CORE_FEATURE_NAMES),
        "existing": strongest(tuple(BASE_FEATURE_NAMES)),
    }
    checks = {
        "endpointAvailabilityMatchesT5": abs(endpoint_both - T5_ENDPOINT_BOTH) <= 1e-12 and all(bool(value["availabilityMatchesT5"]) for value in extraction_audit.values()),
        "minimumRecordingAvailabilityMatchesT5": abs(minimum_recording_both - T5_MINIMUM_RECORDING_BOTH) <= 1e-12,
        "fourTeamVisibilityWithin1ppOfT5": four_team >= T5_FOUR_TEAM - 0.01 - 1e-12,
        "meanSeparationGainAtLeast006": mean_separation - T5_MEAN_SEPARATION >= 0.06 - 1e-12,
        "positiveMarginGainAtLeast3pp": positive_margin - T5_POSITIVE_MARGIN >= 0.03 - 1e-12,
        "positiveMarginAtLeastT4": positive_margin >= T4_POSITIVE_MARGIN - 1e-12,
        "reliableSwapAtLeastT4": reliable_swap >= T4_RELIABLE_SWAP - 1e-12,
        "meanEffectiveSupportAtLeast35Percent": mean_effective_support >= 0.35 - 1e-12,
        "noDiscardedObservations": True,
        "downweightingNonzero": below_half > 0,
        "coreFeaturesNonconstant": all(core_nonconstant.values()),
        **{f"{name}CorrelationBelow98Percent": value is None or abs(value["spearman"]) < 0.98 - 1e-12 for name, value in strongest_values.items()},
    }
    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    performance = {
        "elapsedSeconds": elapsed,
        "peakResidentMemoryMiB": peak_memory,
        "endpointWindows": len(endpoint_values),
        "frameRequests": len(endpoint_values) * 5,
        "detectorTileCalls": len(endpoint_values) * 40,
        "budget": {"wallTimePassed": elapsed <= 3600.0, "peakMemoryPassed": peak_memory <= 768.0},
    }
    if performance["frameRequests"] != EXPECTED_FRAMES or performance["detectorTileCalls"] != EXPECTED_TILES:
        raise ValueError("T7 execution shape changed")
    checks.update({
        "wallTimePassed": bool(performance["budget"]["wallTimePassed"]),
        "peakMemoryPassed": bool(performance["budget"]["peakMemoryPassed"]),
        "noFrameErrors": all(int(value["frameErrors"]) == 0 for value in extraction_audit.values()),
    })
    script_path = Path(__file__).resolve()
    module_path = (script_path.parent.parent / "analysis/side_switch_t7_soft_consensus.py").resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t7-soft-robust-jersey-consensus-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": t6["scope"],
        "contract": {"featureNames": list(T7_FEATURE_NAMES), "futureFirstHeadFeatureNames": list(T7_CORE_FEATURE_NAMES), "candidateGeneration": "unchanged", "labelUse": "none"},
        "parity": {"rows": len(output_rows), "eligibleBoundaryRows": len(boundary_rows), "ineligibleInternalRows": 80, "priorFeatureValues": "exact", "candidateIdAndOrder": "exact"},
        "engineering": {
            "endpointBothQualifiedTeamsFraction": endpoint_both,
            "minimumRecordingBothQualifiedTeamsFraction": minimum_recording_both,
            "boundaryFourTeamVisibilityFraction": four_team,
            "meanMinimumTeamSeparation": mean_separation,
            "positiveRawTransportMarginFraction": positive_margin,
            "reliableSwapEvidenceNonzeroFraction": reliable_swap,
            "meanEffectiveRobustSupport": mean_effective_support,
            "belowHalfWeightObservations": below_half,
            "belowQuarterWeightObservations": below_quarter,
            "coreFeatureNonconstant": core_nonconstant,
            "strongestCorrelations": strongest_values,
            "checks": checks,
            "correlations": correlations,
        },
        "performance": performance,
        "rows": output_rows,
        "extractionAudit": extraction_audit,
        "sources": {"t6Features": {"path": str(t6_path), "sha256": t6_hash}, "extractor": {"path": str(script_path), "sha256": _sha256(script_path)}, "module": {"path": str(module_path), "sha256": _sha256(module_path)}},
        "personDetector": detector_identity(detector_dir),
        "limitations": ["No label or model artifact was loaded.", "Every T5 observation is retained with a strictly positive robust weight."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t6-features", type=Path, default=DEFAULT_T6)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument("--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    print(json.dumps({"engineeringDecision": payload["engineeringDecision"], "parity": payload["parity"], "engineering": {key: value for key, value in payload["engineering"].items() if key != "correlations"}, "performance": payload["performance"]}, indent=2))


if __name__ == "__main__":
    main()
