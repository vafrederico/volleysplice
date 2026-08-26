#!/usr/bin/env python3
"""Extract T8 explicit two-mode jersey transport features without labels."""

from __future__ import annotations

import argparse
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
from analysis.side_switch_t7_soft_consensus import T7_CORE_FEATURE_NAMES
from analysis.side_switch_t8_two_mode_transport import (
    T8_CORE_FEATURE_NAMES,
    T8_FEATURE_NAMES,
    summarize_two_mode_endpoint,
    two_mode_transport_features,
)
from scripts.extract_side_switch_helpers import crop_roi, load_json, sha256_path, spearman, window_key


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_T7 = REPORTS / "side-switch-t7-soft-robust-jersey-consensus-features-v1.json"
DEFAULT_DETECTOR_DIR = ROOT / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
DEFAULT_OUTPUT = REPORTS / "side-switch-t8-two-mode-jersey-transport-features-v1.json"
EXPECTED_T7_SHA256 = "d592961f184122afe6e88c95e6405c2bdc57bd71986733480021d46037cba57a"
T5_ENDPOINT_BOTH = 0.8110236220472441
T5_MINIMUM_RECORDING_BOTH = 0.4230769230769231
T5_FOUR_TEAM = 0.6875
T4_MEAN_SEPARATION = 0.37386011658617885
T7_POSITIVE_MARGIN = 0.2548076923076923
T7_RELIABLE_SWAP = 0.19391025641025642
EXPECTED_ENDPOINTS = 635
EXPECTED_FRAMES = 3175
EXPECTED_TILES = 25400


def extract(args: argparse.Namespace) -> dict[str, Any]:
    t7_path = args.t7_features.expanduser().resolve()
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T8 artifact: {output}")
    t7_hash = sha256_path(t7_path)
    if args.enforce_source_hash and t7_hash != EXPECTED_T7_SHA256:
        raise ValueError(f"T8 T7 source identity changed: {t7_hash}")
    started = time.perf_counter()
    t7 = load_json(t7_path)
    if (
        str(t7["engineeringDecision"]) != "fail"
        or int(t7["parity"]["rows"]) != 704
        or str(t7["parity"]["priorFeatureValues"]) != "exact"
        or str(t7["parity"]["candidateIdAndOrder"]) != "exact"
    ):
        raise ValueError("T8 requires the exact T7 engineering artifact")
    detector = QuantizedPersonDetector(detector_dir, opencv_threads=args.opencv_threads)
    source_rows = [dict(row) for row in t7["rows"]]
    recording_ids = tuple(str(value) for value in t7["scope"]["recordingIds"])
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        rows_by_recording[str(row["recordingId"])].append(row)
    endpoints: dict[tuple[str, float, float], Any] = {}
    extraction_audit: dict[str, Any] = {}
    for number, recording_id in enumerate(recording_ids, 1):
        boundaries = [
            row
            for row in rows_by_recording[recording_id]
            if str(row["kind"]) == "adjacent-rally-boundary"
        ]
        windows: dict[tuple[str, float, float], Mapping[str, Any]] = {}
        for row in boundaries:
            for side in ("before", "after"):
                window = row["comparisonWindows"][side]
                windows[window_key(recording_id, window)] = window
        prior = t7["extractionAudit"][recording_id]
        video_path = Path(str(prior["videoPath"])).resolve()
        if not video_path.is_file() or video_path.stat().st_size != int(prior["videoSizeBytes"]):
            raise ValueError(f"T8 video provenance changed for {recording_id}")
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
            raise RuntimeError(f"could not open T8 video: {video_path}")
        local_started = time.perf_counter()
        local = []
        try:
            for key, window in sorted(windows.items(), key=lambda value: value[0][1:]):
                frames = [
                    crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in endpoint_sample_times(float(window["start"]), float(window["end"]))
                ]
                summary = summarize_two_mode_endpoint(frames, net_y_ratio, detector)
                endpoints[key] = summary
                local.append(summary)
        finally:
            capture.release()
        both = float(np.mean([value.near.available and value.far.available for value in local]))
        t5_both = float(prior["t5EndpointBothQualifiedTeamsFraction"])
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
            "elapsedSeconds": time.perf_counter() - local_started,
        }
    if len(endpoints) != EXPECTED_ENDPOINTS:
        raise ValueError("T8 endpoint count changed")

    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    four_team_values: list[bool] = []
    for row in source_rows:
        updated = dict(row)
        updated["features"] = {str(name): float(value) for name, value in row["features"].items()}
        if str(row["kind"]) == "adjacent-rally-boundary":
            recording_id = str(row["recordingId"])
            before = endpoints[window_key(recording_id, row["comparisonWindows"]["before"])]
            after = endpoints[window_key(recording_id, row["comparisonWindows"]["after"])]
            values, diagnostics = two_mode_transport_features(before, after)
            updated["features"].update(values)
            updated["t8TwoModeJerseyTransport"] = {
                "status": "ok",
                "before": before.to_diagnostic(),
                "after": after.to_diagnostic(),
                **diagnostics,
            }
            four_team_values.append(
                before.near.available
                and before.far.available
                and after.near.available
                and after.far.available
            )
            boundary_rows.append(updated)
        else:
            updated["t8TwoModeJerseyTransport"] = {
                "status": "not-eligible",
                "reason": "T8 is adjacent-boundary-only",
            }
        output_rows.append(updated)
    if len(boundary_rows) != 624 or len(output_rows) != 704:
        raise ValueError("T8 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [str(row["eventId"]) for row in source_rows]:
        raise ValueError("T8 row order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T8 changed prior feature {name}")

    endpoint_values = list(endpoints.values())
    tracks = [track for value in endpoint_values for track in (value.near, value.far)]
    nonempty_tracks = [value for value in tracks if value.total_observations > 0]
    multi_observation_tracks = [value for value in tracks if value.total_observations >= 2]
    available_tracks = [value for value in tracks if value.available]
    endpoint_both = float(np.mean([value.near.available and value.far.available for value in endpoint_values]))
    minimum_recording_both = min(float(value["endpointBothQualifiedTeamsFraction"]) for value in extraction_audit.values())
    four_team = float(np.mean(four_team_values))
    mean_separation = float(np.mean([row["features"]["twoModeJerseyTeamSeparationMinimum"] for row in boundary_rows]))
    positive_margin = float(np.mean([row["features"]["twoModeJerseyTeamTransportSwapMargin"] > 0.0 for row in boundary_rows]))
    reliable_swap = float(np.mean([row["features"]["twoModeJerseyReliableSwapEvidence"] > 0.0 for row in boundary_rows]))
    all_retained = all(sum(mode.observations for mode in value.modes) == value.total_observations for value in nonempty_tracks)
    all_multi_have_two = all(len(value.modes) == 2 for value in multi_observation_tracks)
    secondary_support = [min(mode.support for mode in value.modes) if len(value.modes) == 2 else 0.0 for value in available_tracks]
    substantive_secondary = float(np.mean([value >= 0.20 for value in secondary_support]))
    mean_objective = float(np.mean([value.assignment_objective for value in nonempty_tracks]))
    mean_entropy = float(np.mean([value.support_entropy for value in nonempty_tracks]))
    core_nonconstant = {
        name: float(np.ptp([row["features"][name] for row in boundary_rows])) > 1e-15
        for name in T8_CORE_FEATURE_NAMES
    }

    correlations: list[dict[str, Any]] = []
    comparison = (
        *T8_CORE_FEATURE_NAMES,
        *T4_CORE_FEATURE_NAMES,
        *T5_CORE_FEATURE_NAMES,
        *T6_CORE_FEATURE_NAMES,
        *T7_CORE_FEATURE_NAMES,
        *BASE_FEATURE_NAMES,
    )
    for index, left_name in enumerate(T8_CORE_FEATURE_NAMES):
        left = np.asarray([row["features"][left_name] for row in boundary_rows])
        for right_name in comparison[index + 1 :]:
            if right_name in T8_CORE_FEATURE_NAMES and T8_CORE_FEATURE_NAMES.index(right_name) <= index:
                continue
            right = np.asarray([materialized_profile_feature(row, right_name) for row in boundary_rows])
            value = spearman(left, right)
            if value is not None:
                correlations.append({"left": left_name, "right": right_name, "spearman": value})

    def strongest(names: tuple[str, ...]) -> Mapping[str, Any] | None:
        values = [value for value in correlations if value["right"] in names]
        return max(values, key=lambda value: abs(value["spearman"]), default=None)

    strongest_values = {
        "core": strongest(T8_CORE_FEATURE_NAMES),
        "t4": strongest(T4_CORE_FEATURE_NAMES),
        "t5": strongest(T5_CORE_FEATURE_NAMES),
        "t6": strongest(T6_CORE_FEATURE_NAMES),
        "t7": strongest(T7_CORE_FEATURE_NAMES),
        "existing": strongest(tuple(BASE_FEATURE_NAMES)),
    }
    checks = {
        "endpointAvailabilityMatchesT5": abs(endpoint_both - T5_ENDPOINT_BOTH) <= 1e-12 and all(bool(value["availabilityMatchesT5"]) for value in extraction_audit.values()),
        "minimumRecordingAvailabilityMatchesT5": abs(minimum_recording_both - T5_MINIMUM_RECORDING_BOTH) <= 1e-12,
        "fourTeamVisibilityMatchesT5": abs(four_team - T5_FOUR_TEAM) <= 1e-12,
        "meanSeparationAtLeastT4": mean_separation >= T4_MEAN_SEPARATION - 1e-12,
        "positiveMarginAtLeastT7": positive_margin >= T7_POSITIVE_MARGIN - 1e-12,
        "reliableSwapAtLeastT7": reliable_swap >= T7_RELIABLE_SWAP - 1e-12,
        "multiObservationTracksHaveTwoModes": all_multi_have_two,
        "noDiscardedObservations": all_retained,
        "substantiveSecondaryAtLeast20Percent": substantive_secondary >= 0.20 - 1e-12,
        "coreFeaturesNonconstant": all(core_nonconstant.values()),
        **{
            f"{name}CorrelationBelow98Percent": value is None or abs(value["spearman"]) < 0.98 - 1e-12
            for name, value in strongest_values.items()
        },
    }
    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    performance = {
        "elapsedSeconds": elapsed,
        "peakResidentMemoryMiB": peak_memory,
        "endpointWindows": len(endpoint_values),
        "frameRequests": len(endpoint_values) * 5,
        "detectorTileCalls": len(endpoint_values) * 40,
        "budget": {
            "wallTimePassed": elapsed <= 3600.0,
            "peakMemoryPassed": peak_memory <= 768.0,
        },
    }
    if performance["frameRequests"] != EXPECTED_FRAMES or performance["detectorTileCalls"] != EXPECTED_TILES:
        raise ValueError("T8 execution shape changed")
    checks.update(
        {
            "wallTimePassed": bool(performance["budget"]["wallTimePassed"]),
            "peakMemoryPassed": bool(performance["budget"]["peakMemoryPassed"]),
            "noFrameErrors": all(int(value["frameErrors"]) == 0 for value in extraction_audit.values()),
        }
    )
    script_path = Path(__file__).resolve()
    module_path = (script_path.parent.parent / "analysis/side_switch_t8_two_mode_transport.py").resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t8-two-mode-jersey-transport-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": t7["scope"],
        "contract": {
            "featureNames": list(T8_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T8_CORE_FEATURE_NAMES),
            "candidateGeneration": "unchanged",
            "labelUse": "none",
        },
        "parity": {
            "rows": len(output_rows),
            "eligibleBoundaryRows": len(boundary_rows),
            "ineligibleInternalRows": 80,
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
            "multiObservationTracks": len(multi_observation_tracks),
            "multiObservationTracksWithTwoModes": sum(len(value.modes) == 2 for value in multi_observation_tracks),
            "substantiveSecondaryModeFraction": substantive_secondary,
            "meanAssignmentObjective": mean_objective,
            "meanSupportEntropy": mean_entropy,
            "coreFeatureNonconstant": core_nonconstant,
            "strongestCorrelations": strongest_values,
            "checks": checks,
            "correlations": correlations,
        },
        "performance": performance,
        "rows": output_rows,
        "extractionAudit": extraction_audit,
        "sources": {
            "t7Features": {"path": str(t7_path), "sha256": t7_hash},
            "extractor": {"path": str(script_path), "sha256": sha256_path(script_path)},
            "module": {"path": str(module_path), "sha256": sha256_path(module_path)},
        },
        "personDetector": detector_identity(detector_dir),
        "limitations": [
            "No label or model artifact was loaded.",
            "Every T5 observation is assigned to exactly one of at most two modes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-features", type=Path, default=DEFAULT_T7)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument("--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True)
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
