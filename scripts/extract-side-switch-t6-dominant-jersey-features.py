#!/usr/bin/env python3
"""Extract T6 robust dominant-jersey consensus features without labels."""

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
from analysis.side_switch_t6_dominant_jersey import (
    T6_CORE_FEATURE_NAMES,
    T6_FEATURE_NAMES,
    dominant_jersey_transport_features,
    summarize_dominant_jersey_endpoint,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_T5 = (
    REPORTS / "side-switch-t5-court-constrained-temporal-team-tracking-features-v1.json"
)
DEFAULT_DETECTOR_DIR = ROOT / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
DEFAULT_OUTPUT = REPORTS / "side-switch-t6-dominant-jersey-consensus-features-v1.json"
EXPECTED_T5_SHA256 = "25d8a23563f803058f23ba2da23df0e82283ab893afd143b5f39200d96fcb460"
EXPECTED_ENDPOINTS = 635
EXPECTED_FRAME_REQUESTS = 3175
EXPECTED_TILE_CALLS = 25400
MAX_WALL_SECONDS = 3600.0
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
    right = max(left + 1, min(width, round((float(roi.get("x", 0.0)) + float(roi.get("width", 1.0))) * width)))
    bottom = max(top + 1, min(height, round((float(roi.get("y", 0.0)) + float(roi.get("height", 1.0))) * height)))
    return np.ascontiguousarray(frame[top:bottom, left:right])


def _window_key(recording_id: str, window: Mapping[str, Any]) -> tuple[str, float, float]:
    return recording_id, round(float(window["start"]), 9), round(float(window["end"]), 9)


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
    if left.shape != right.shape or float(np.ptp(left)) <= 1e-15 or float(np.ptp(right)) <= 1e-15:
        return None
    value = float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])
    return value if math.isfinite(value) else None


def extract(args: argparse.Namespace) -> dict[str, Any]:
    t5_path = args.t5_features.expanduser().resolve()
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T6 artifact: {output}")
    t5_hash = _sha256(t5_path)
    if args.enforce_source_hash and t5_hash != EXPECTED_T5_SHA256:
        raise ValueError(f"T6 T5 source identity changed: {t5_hash}")
    started = time.perf_counter()
    t5 = _load(t5_path)
    if (
        str(t5["engineeringDecision"]) != "fail"
        or int(t5["parity"]["rows"]) != 704
        or str(t5["parity"]["priorFeatureValues"]) != "exact"
        or str(t5["parity"]["candidateIdAndOrder"]) != "exact"
    ):
        raise ValueError("T6 requires the exact T5 engineering artifact")
    detector = QuantizedPersonDetector(detector_dir, opencv_threads=args.opencv_threads)
    source_rows = [dict(row) for row in t5["rows"]]
    recording_ids = tuple(str(value) for value in t5["scope"]["recordingIds"])
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        rows_by_recording[str(row["recordingId"])].append(row)

    endpoint_summaries: dict[tuple[str, float, float], Any] = {}
    extraction_audit: dict[str, Any] = {}
    for number, recording_id in enumerate(recording_ids, 1):
        boundaries = [
            row for row in rows_by_recording[recording_id]
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        windows: dict[tuple[str, float, float], Mapping[str, Any]] = {}
        for row in boundaries:
            for side in ("before", "after"):
                window = row["comparisonWindows"][side]
                windows[_window_key(recording_id, window)] = window
        prior = t5["extractionAudit"][recording_id]
        video_path = Path(str(prior["videoPath"])).resolve()
        if not video_path.is_file() or video_path.stat().st_size != int(prior["videoSizeBytes"]):
            raise ValueError(f"T6 video provenance changed for {recording_id}")
        net_y_ratio = float(prior["courtGeometry"]["netYRatio"])
        roi = prior["roi"]
        print(
            f"[{number}/{len(recording_ids)}] {recording_id}: "
            f"{len(boundaries)} boundaries, {len(windows)} endpoint windows",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open T6 video: {video_path}")
        local_started = time.perf_counter()
        local = []
        try:
            for key, window in sorted(windows.items(), key=lambda value: value[0][1:]):
                frames = [
                    _crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in endpoint_sample_times(float(window["start"]), float(window["end"]))
                ]
                summary = summarize_dominant_jersey_endpoint(frames, net_y_ratio, detector)
                endpoint_summaries[key] = summary
                local.append(summary)
        finally:
            capture.release()
        extraction_audit[recording_id] = {
            "videoPath": str(video_path),
            "videoSizeBytes": int(prior["videoSizeBytes"]),
            "videoSha256": str(prior["videoSha256"]),
            "videoHashSource": str(t5_path),
            "roi": roi,
            "courtGeometry": prior["courtGeometry"],
            "boundaryRows": len(boundaries),
            "endpointWindows": len(windows),
            "frameRequests": len(windows) * 5,
            "detectorTileCalls": len(windows) * 40,
            "frameErrors": 0,
            "endpointBothQualifiedTeamsFraction": float(np.mean([value.near.available and value.far.available for value in local])),
            "t5EndpointBothQualifiedTeamsFraction": float(prior["endpointBothQualifiedTeamsFraction"]),
            "meanNearModeSupportFraction": float(np.mean([value.near_track.mode_support_fraction for value in local if value.near_track.total_observations > 0])),
            "meanFarModeSupportFraction": float(np.mean([value.far_track.mode_support_fraction for value in local if value.far_track.total_observations > 0])),
            "rejectedObservations": int(sum(value.near_track.rejected_observations + value.far_track.rejected_observations for value in local)),
            "elapsedSeconds": time.perf_counter() - local_started,
        }

    if len(endpoint_summaries) != EXPECTED_ENDPOINTS:
        raise ValueError("T6 endpoint count changed")
    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for row in source_rows:
        updated = dict(row)
        updated["features"] = {str(name): float(value) for name, value in row["features"].items()}
        if str(row["kind"]) == "adjacent-rally-boundary":
            recording_id = str(row["recordingId"])
            before = endpoint_summaries[_window_key(recording_id, row["comparisonWindows"]["before"])]
            after = endpoint_summaries[_window_key(recording_id, row["comparisonWindows"]["after"])]
            values, diagnostics = dominant_jersey_transport_features(before, after)
            updated["features"].update(values)
            updated["t6DominantJerseyConsensus"] = {
                "status": "ok",
                "before": before.to_diagnostic(),
                "after": after.to_diagnostic(),
                **diagnostics,
            }
            boundary_rows.append(updated)
        else:
            updated["t6DominantJerseyConsensus"] = {
                "status": "not-eligible",
                "reason": "T6 is adjacent-boundary-only",
            }
        output_rows.append(updated)
    if len(boundary_rows) != 624 or len(output_rows) - len(boundary_rows) != 80:
        raise ValueError("T6 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [str(row["eventId"]) for row in source_rows]:
        raise ValueError("T6 row order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T6 changed prior feature {name}")

    endpoints = list(endpoint_summaries.values())
    endpoint_both = float(np.mean([value.near.available and value.far.available for value in endpoints]))
    minimum_recording_both = min(float(value["endpointBothQualifiedTeamsFraction"]) for value in extraction_audit.values())
    four_team = float(np.mean([row["features"]["dominantJerseyTeamReliabilityMinimum"] > 0.0 and row["features"]["dominantJerseyCrossSideSimilarityMinimum"] > 0.0 for row in boundary_rows]))
    mean_separation = float(np.mean([row["features"]["dominantJerseyTeamSeparationMinimum"] for row in boundary_rows]))
    positive_margin = float(np.mean([row["features"]["dominantJerseyTeamTransportSwapMargin"] > 0.0 for row in boundary_rows]))
    reliable_swap = float(np.mean([row["features"]["dominantJerseyReliableSwapEvidence"] > 0.0 for row in boundary_rows]))
    nonempty_tracks = [track for value in endpoints for track in (value.near_track, value.far_track) if track.total_observations > 0]
    mean_mode_support = float(np.mean([value.mode_support_fraction for value in nonempty_tracks]))
    rejected_observations = int(sum(value.rejected_observations for value in nonempty_tracks))
    core_nonconstant = {name: float(np.ptp([row["features"][name] for row in boundary_rows])) > 1e-15 for name in T6_CORE_FEATURE_NAMES}

    correlations: list[dict[str, Any]] = []
    comparison_names = (*T6_CORE_FEATURE_NAMES, *T4_CORE_FEATURE_NAMES, *T5_CORE_FEATURE_NAMES, *BASE_FEATURE_NAMES)
    for index, left_name in enumerate(T6_CORE_FEATURE_NAMES):
        left = np.asarray([row["features"][left_name] for row in boundary_rows])
        for right_name in comparison_names[index + 1 :]:
            if right_name in T6_CORE_FEATURE_NAMES and T6_CORE_FEATURE_NAMES.index(right_name) <= index:
                continue
            right = np.asarray([materialized_profile_feature(row, right_name) for row in boundary_rows])
            value = _spearman(left, right)
            if value is not None:
                correlations.append({"left": left_name, "right": right_name, "spearman": value})
    def strongest(names: tuple[str, ...]) -> Mapping[str, Any] | None:
        values = [value for value in correlations if value["right"] in names]
        return max(values, key=lambda value: abs(value["spearman"]), default=None)
    strongest_core = strongest(T6_CORE_FEATURE_NAMES)
    strongest_t4 = strongest(T4_CORE_FEATURE_NAMES)
    strongest_t5 = strongest(T5_CORE_FEATURE_NAMES)
    strongest_existing = strongest(tuple(BASE_FEATURE_NAMES))
    t4_endpoint_both = float(t5["engineering"]["t4EndpointBothQualifiedTeamsFraction"])
    t4_four_team = float(t5["engineering"]["t4BoundaryFourTeamVisibilityFraction"])
    t5_separation = float(np.mean([row["features"]["courtTrackedFarJerseyTeamSeparationMinimum"] for row in boundary_rows]))
    t5_positive_margin = float(np.mean([row["features"]["courtTrackedFarJerseyTeamTransportSwapMargin"] > 0.0 for row in boundary_rows]))
    t5_reliable_swap = float(t5["engineering"]["reliableSwapEvidenceNonzeroFraction"])
    checks = {
        "endpointBothAtLeastT4": endpoint_both >= t4_endpoint_both - 1e-12,
        "everyRecordingBothTeamsAtLeast35Percent": minimum_recording_both >= 0.35 - 1e-12,
        "fourTeamVisibilityAtLeastT4": four_team >= t4_four_team - 1e-12,
        "meanSeparationGainAtLeast003": mean_separation - t5_separation >= 0.03 - 1e-12,
        "positiveMarginGainAtLeast3pp": positive_margin - t5_positive_margin >= 0.03 - 1e-12,
        "reliableSwapGainAtLeast3pp": reliable_swap - t5_reliable_swap >= 0.03 - 1e-12,
        "meanModeSupportAtLeast60Percent": mean_mode_support >= 0.60 - 1e-12,
        "outlierRejectionNonzero": rejected_observations > 0,
        "coreFeaturesNonconstant": all(core_nonconstant.values()),
        "coreCorrelationBelow98Percent": strongest_core is None or abs(strongest_core["spearman"]) < 0.98 - 1e-12,
        "t4CorrelationBelow98Percent": strongest_t4 is None or abs(strongest_t4["spearman"]) < 0.98 - 1e-12,
        "t5CorrelationBelow98Percent": strongest_t5 is None or abs(strongest_t5["spearman"]) < 0.98 - 1e-12,
        "existingCorrelationBelow98Percent": strongest_existing is None or abs(strongest_existing["spearman"]) < 0.98 - 1e-12,
    }
    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    performance = {
        "elapsedSeconds": elapsed,
        "peakResidentMemoryMiB": peak_memory,
        "endpointWindows": len(endpoints),
        "frameRequests": len(endpoints) * 5,
        "detectorTileCalls": len(endpoints) * 40,
        "budget": {
            "wallTimePassed": elapsed <= MAX_WALL_SECONDS,
            "peakMemoryPassed": peak_memory <= MAX_PEAK_MEMORY_MIB,
        },
    }
    if performance["frameRequests"] != EXPECTED_FRAME_REQUESTS or performance["detectorTileCalls"] != EXPECTED_TILE_CALLS:
        raise ValueError("T6 execution shape changed")
    checks.update({
        "wallTimePassed": bool(performance["budget"]["wallTimePassed"]),
        "peakMemoryPassed": bool(performance["budget"]["peakMemoryPassed"]),
        "noFrameErrors": all(int(value["frameErrors"]) == 0 for value in extraction_audit.values()),
    })
    script_path = Path(__file__).resolve()
    module_path = (script_path.parent.parent / "analysis/side_switch_t6_dominant_jersey.py").resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t6-dominant-jersey-consensus-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": t5["scope"],
        "contract": {
            "eligibleCandidateKind": "adjacent-rally-boundary",
            "featureNames": list(T6_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T6_CORE_FEATURE_NAMES),
            "candidateGeneration": "unchanged",
            "labelUse": "none",
        },
        "parity": {
            "rows": len(output_rows),
            "eligibleBoundaryRows": len(boundary_rows),
            "ineligibleInternalRows": len(output_rows) - len(boundary_rows),
            "priorFeatureValues": "exact",
            "candidateIdAndOrder": "exact",
        },
        "engineering": {
            "endpointBothQualifiedTeamsFraction": endpoint_both,
            "minimumRecordingBothQualifiedTeamsFraction": minimum_recording_both,
            "boundaryFourTeamVisibilityFraction": four_team,
            "meanMinimumTeamSeparation": mean_separation,
            "positiveRawTransportMarginFraction": positive_margin,
            "reliableSwapEvidenceNonzeroFraction": reliable_swap,
            "meanModeSupportFraction": mean_mode_support,
            "rejectedObservations": rejected_observations,
            "t5MeanMinimumTeamSeparation": t5_separation,
            "t5PositiveRawTransportMarginFraction": t5_positive_margin,
            "t5ReliableSwapEvidenceNonzeroFraction": t5_reliable_swap,
            "coreFeatureNonconstant": core_nonconstant,
            "strongestCoreAbsoluteSpearman": strongest_core,
            "strongestT4CoreAbsoluteSpearman": strongest_t4,
            "strongestT5CoreAbsoluteSpearman": strongest_t5,
            "strongestExistingInputAbsoluteSpearman": strongest_existing,
            "checks": checks,
            "correlations": correlations,
        },
        "performance": performance,
        "rows": output_rows,
        "extractionAudit": extraction_audit,
        "sources": {
            "t5Features": {"path": str(t5_path), "sha256": t5_hash},
            "extractor": {"path": str(script_path), "sha256": _sha256(script_path)},
            "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        },
        "personDetector": detector_identity(detector_dir),
        "limitations": [
            "No label or model result was loaded.",
            "T6 changes dominant jersey consensus only; detector and frames are unchanged.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t5-features", type=Path, default=DEFAULT_T5)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument("--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    print(json.dumps({
        "engineeringDecision": payload["engineeringDecision"],
        "parity": payload["parity"],
        "engineering": {key: value for key, value in payload["engineering"].items() if key != "correlations"},
        "performance": payload["performance"],
    }, indent=2))


if __name__ == "__main__":
    main()
