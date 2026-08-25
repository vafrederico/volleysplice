#!/usr/bin/env python3
"""Extract T10 equal-unit selective-far tracklet transport without labels."""

from __future__ import annotations

import argparse
import json
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
from analysis.side_switch_t8_two_mode_transport import T8_CORE_FEATURE_NAMES
from analysis.side_switch_t9_mass_transport import T9_CORE_FEATURE_NAMES
from analysis.side_switch_t10_tracklet_unit_transport import (
    T10_CORE_FEATURE_NAMES,
    T10_FEATURE_NAMES,
    summarize_tracklet_unit_endpoint,
    tracklet_unit_transport_features,
)
from scripts.extract_side_switch_helpers import (
    crop_roi,
    load_json,
    sha256_path,
    spearman,
    window_key,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_T9 = REPORTS / "side-switch-t9-mass-preserving-mode-transport-features-v1.json"
DEFAULT_DETECTOR_DIR = ROOT / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
DEFAULT_OUTPUT = REPORTS / "side-switch-t10-tracklet-unit-jersey-transport-features-v1.json"
EXPECTED_T9_SHA256 = "c21f4f5360185ae37904e622ab5caf3c0edeb13dfd58506a6ab814aa57b9c5e2"
T4_ENDPOINT_BOTH = 0.7417322834645669
T4_MINIMUM_RECORDING_BOTH = 0.3269230769230769
T4_FOUR_TEAM = 0.5833333333333334
T4_MEAN_SEPARATION = 0.37386011658617885
T7_POSITIVE_MARGIN = 0.2548076923076923
T7_RELIABLE_SWAP = 0.19391025641025642
EXPECTED_ENDPOINTS = 635
EXPECTED_FRAMES = 3175
EXPECTED_TILES = 25400


def _t4_endpoint_priors(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, float, float], Mapping[str, Any]]:
    result: dict[tuple[str, float, float], Mapping[str, Any]] = {}
    for row in rows:
        if str(row["kind"]) != "adjacent-rally-boundary":
            continue
        recording_id = str(row["recordingId"])
        diagnostic = row["t4SelectiveFarJerseyTransport"]
        for side in ("before", "after"):
            key = window_key(recording_id, row["comparisonWindows"][side])
            value = diagnostic[side]
            prior = result.get(key)
            if prior is not None:
                stable_names = (
                    "nearTeamAvailable",
                    "farTeamAvailable",
                    "nearQualifyingTracklets",
                    "farQualifyingTracklets",
                    "selectedPlayersByFrame",
                )
                if any(prior[name] != value[name] for name in stable_names):
                    raise ValueError("T10 found inconsistent stored T4 endpoint diagnostics")
            result[key] = value
    return result


def _matches_t4(summary: Any, prior: Mapping[str, Any]) -> bool:
    return (
        bool(summary.near) == bool(prior["nearTeamAvailable"])
        and bool(summary.far) == bool(prior["farTeamAvailable"])
        and len(summary.near) == int(prior["nearQualifyingTracklets"])
        and len(summary.far) == int(prior["farQualifyingTracklets"])
        and list(summary.selected_counts) == prior["selectedPlayersByFrame"]
    )


def extract(args: argparse.Namespace) -> dict[str, Any]:
    t9_path = args.t9_features.expanduser().resolve()
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T10 artifact: {output}")
    t9_hash = sha256_path(t9_path)
    if args.enforce_source_hash and t9_hash != EXPECTED_T9_SHA256:
        raise ValueError(f"T10 T9 source identity changed: {t9_hash}")
    started = time.perf_counter()
    t9 = load_json(t9_path)
    if (
        str(t9["engineeringDecision"]) != "fail"
        or int(t9["parity"]["rows"]) != 704
        or str(t9["parity"]["priorFeatureValues"]) != "exact"
        or str(t9["parity"]["candidateIdAndOrder"]) != "exact"
    ):
        raise ValueError("T10 requires the exact failed T9 engineering artifact")
    detector = QuantizedPersonDetector(detector_dir, opencv_threads=args.opencv_threads)
    source_rows = [dict(row) for row in t9["rows"]]
    t4_priors = _t4_endpoint_priors(source_rows)
    if len(t4_priors) != EXPECTED_ENDPOINTS:
        raise ValueError("T10 stored T4 endpoint scope changed")
    recording_ids = tuple(str(value) for value in t9["scope"]["recordingIds"])
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
        prior_audit = t9["extractionAudit"][recording_id]
        video_path = Path(str(prior_audit["videoPath"])).resolve()
        if (
            not video_path.is_file()
            or video_path.stat().st_size != int(prior_audit["videoSizeBytes"])
        ):
            raise ValueError(f"T10 video provenance changed for {recording_id}")
        net_y_ratio = float(prior_audit["courtGeometry"]["netYRatio"])
        roi = prior_audit["roi"]
        print(
            f"[{number}/{len(recording_ids)}] {recording_id}: "
            f"{len(boundaries)} boundaries, {len(windows)} endpoint windows",
            file=sys.stderr,
            flush=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open T10 video: {video_path}")
        local_started = time.perf_counter()
        local: list[Any] = []
        local_matches = True
        try:
            for key, window in sorted(windows.items(), key=lambda value: value[0][1:]):
                frames = [
                    crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in endpoint_sample_times(
                        float(window["start"]), float(window["end"])
                    )
                ]
                summary = summarize_tracklet_unit_endpoint(frames, net_y_ratio, detector)
                endpoints[key] = summary
                local.append(summary)
                local_matches = local_matches and _matches_t4(summary, t4_priors[key])
        finally:
            capture.release()
        both = float(np.mean([bool(value.near) and bool(value.far) for value in local]))
        t4_both = float(
            np.mean(
                [
                    bool(t4_priors[key]["nearTeamAvailable"])
                    and bool(t4_priors[key]["farTeamAvailable"])
                    for key in windows
                ]
            )
        )
        extraction_audit[recording_id] = {
            "videoPath": str(video_path),
            "videoSizeBytes": int(prior_audit["videoSizeBytes"]),
            "videoSha256": str(prior_audit["videoSha256"]),
            "roi": roi,
            "courtGeometry": prior_audit["courtGeometry"],
            "boundaryRows": len(boundaries),
            "endpointWindows": len(windows),
            "frameRequests": len(windows) * 5,
            "detectorTileCalls": len(windows) * 40,
            "frameErrors": 0,
            "endpointBothQualifiedTeamsFraction": both,
            "t4EndpointBothQualifiedTeamsFraction": t4_both,
            "availabilityMatchesT4": abs(both - t4_both) <= 1e-12,
            "trackletRepresentationMatchesT4": local_matches,
            "elapsedSeconds": time.perf_counter() - local_started,
        }
    if len(endpoints) != EXPECTED_ENDPOINTS:
        raise ValueError("T10 endpoint count changed")

    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    four_team_values: list[bool] = []
    tracklet_representation_matches_t4 = True
    for row in source_rows:
        updated = dict(row)
        updated["features"] = {
            str(name): float(value) for name, value in row["features"].items()
        }
        if str(row["kind"]) == "adjacent-rally-boundary":
            recording_id = str(row["recordingId"])
            before = endpoints[
                window_key(recording_id, row["comparisonWindows"]["before"])
            ]
            after = endpoints[
                window_key(recording_id, row["comparisonWindows"]["after"])
            ]
            values, diagnostics = tracklet_unit_transport_features(before, after)
            updated["features"].update(values)
            before_diagnostic = before.to_diagnostic()
            after_diagnostic = after.to_diagnostic()
            updated["t10TrackletUnitJerseyTransport"] = {
                "status": "ok",
                "before": before_diagnostic,
                "after": after_diagnostic,
                **diagnostics,
            }
            prior = row["t4SelectiveFarJerseyTransport"]
            tracklet_representation_matches_t4 = (
                tracklet_representation_matches_t4
                and _matches_t4(before, prior["before"])
                and _matches_t4(after, prior["after"])
            )
            four_team_values.append(
                bool(before.near)
                and bool(before.far)
                and bool(after.near)
                and bool(after.far)
            )
            boundary_rows.append(updated)
        else:
            updated["t10TrackletUnitJerseyTransport"] = {
                "status": "not-eligible",
                "reason": "T10 is adjacent-boundary-only",
            }
        output_rows.append(updated)
    if len(boundary_rows) != 624 or len(output_rows) != 704:
        raise ValueError("T10 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("T10 row order changed")
    for before_row, after_row in zip(source_rows, output_rows, strict=True):
        for name, value in before_row["features"].items():
            if float(after_row["features"][name]) != float(value):
                raise ValueError(f"T10 changed prior feature {name}")

    endpoint_values = list(endpoints.values())
    endpoint_both = float(
        np.mean([bool(value.near) and bool(value.far) for value in endpoint_values])
    )
    minimum_recording_both = min(
        float(value["endpointBothQualifiedTeamsFraction"])
        for value in extraction_audit.values()
    )
    four_team = float(np.mean(four_team_values))
    mean_separation = float(
        np.mean(
            [
                row["features"]["trackletUnitJerseyTeamSeparationMinimum"]
                for row in boundary_rows
            ]
        )
    )
    positive_margin = float(
        np.mean(
            [
                row["features"]["trackletUnitJerseyTeamTransportSwapMargin"] > 0.0
                for row in boundary_rows
            ]
        )
    )
    reliable_swap = float(
        np.mean(
            [
                row["features"]["trackletUnitJerseyReliableSwapEvidence"] > 0.0
                for row in boundary_rows
            ]
        )
    )
    cross_coverage_nonzero = float(
        np.mean(
            [
                row["features"]["trackletUnitJerseyCrossMatchCoverageMinimum"] > 0.0
                for row in boundary_rows
            ]
        )
    )
    conditional_values = np.asarray(
        [
            row["features"]["trackletUnitJerseyConditionalCrossSimilarityMinimum"]
            for row in boundary_rows
        ]
    )
    coverage_values = np.asarray(
        [
            row["features"]["trackletUnitJerseyCrossMatchCoverageMinimum"]
            for row in boundary_rows
        ]
    )
    conditional_coverage_correlation = spearman(conditional_values, coverage_values)
    core_nonconstant = {
        name: float(np.ptp([row["features"][name] for row in boundary_rows])) > 1e-15
        for name in T10_CORE_FEATURE_NAMES
    }

    correlations: list[dict[str, Any]] = []
    comparison = (
        *T10_CORE_FEATURE_NAMES,
        *T4_CORE_FEATURE_NAMES,
        *T5_CORE_FEATURE_NAMES,
        *T6_CORE_FEATURE_NAMES,
        *T7_CORE_FEATURE_NAMES,
        *T8_CORE_FEATURE_NAMES,
        *T9_CORE_FEATURE_NAMES,
        *BASE_FEATURE_NAMES,
    )
    for index, left_name in enumerate(T10_CORE_FEATURE_NAMES):
        left = np.asarray([row["features"][left_name] for row in boundary_rows])
        for right_name in comparison[index + 1 :]:
            right = np.asarray(
                [materialized_profile_feature(row, right_name) for row in boundary_rows]
            )
            value = spearman(left, right)
            if value is not None:
                correlations.append(
                    {"left": left_name, "right": right_name, "spearman": value}
                )

    def strongest(names: tuple[str, ...]) -> Mapping[str, Any] | None:
        values = [value for value in correlations if value["right"] in names]
        return max(values, key=lambda value: abs(value["spearman"]), default=None)

    strongest_values = {
        "core": strongest(T10_CORE_FEATURE_NAMES),
        "t4": strongest(T4_CORE_FEATURE_NAMES),
        "t5": strongest(T5_CORE_FEATURE_NAMES),
        "t6": strongest(T6_CORE_FEATURE_NAMES),
        "t7": strongest(T7_CORE_FEATURE_NAMES),
        "t8": strongest(T8_CORE_FEATURE_NAMES),
        "t9": strongest(T9_CORE_FEATURE_NAMES),
        "existing": strongest(tuple(BASE_FEATURE_NAMES)),
    }
    checks = {
        "endpointAvailabilityMatchesT4": abs(endpoint_both - T4_ENDPOINT_BOTH) <= 1e-12
        and all(bool(value["availabilityMatchesT4"]) for value in extraction_audit.values()),
        "minimumRecordingAvailabilityMatchesT4": abs(
            minimum_recording_both - T4_MINIMUM_RECORDING_BOTH
        )
        <= 1e-12,
        "fourTeamVisibilityMatchesT4": abs(four_team - T4_FOUR_TEAM) <= 1e-12,
        "trackletRepresentationMatchesT4": tracklet_representation_matches_t4
        and all(
            bool(value["trackletRepresentationMatchesT4"])
            for value in extraction_audit.values()
        ),
        "meanSeparationAtLeastT4": mean_separation >= T4_MEAN_SEPARATION - 1e-12,
        "positiveMarginAtLeastT7": positive_margin >= T7_POSITIVE_MARGIN - 1e-12,
        "reliableSwapAtLeastT7": reliable_swap >= T7_RELIABLE_SWAP - 1e-12,
        "crossMatchCoverageNonzeroAtLeast50Percent": cross_coverage_nonzero >= 0.50,
        "conditionalSimilarityCoverageCorrelationBelow90Percent": (
            conditional_coverage_correlation is not None
            and abs(conditional_coverage_correlation) < 0.90 - 1e-12
        ),
        "coreFeaturesNonconstant": all(core_nonconstant.values()),
        **{
            f"{name}CorrelationBelow98Percent": value is None
            or abs(value["spearman"]) < 0.98 - 1e-12
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
    if (
        performance["frameRequests"] != EXPECTED_FRAMES
        or performance["detectorTileCalls"] != EXPECTED_TILES
    ):
        raise ValueError("T10 execution shape changed")
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
        script_path.parent.parent / "analysis/side_switch_t10_tracklet_unit_transport.py"
    ).resolve()
    t3_module_path = (
        script_path.parent.parent / "analysis/side_switch_t3_jersey_transport.py"
    ).resolve()
    t4_module_path = (
        script_path.parent.parent / "analysis/side_switch_t4_selective_far.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t10-tracklet-unit-jersey-transport-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": t9["scope"],
        "contract": {
            "featureNames": list(T10_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T10_CORE_FEATURE_NAMES),
            "candidateGeneration": "unchanged",
            "labelUse": "none",
            "trackletMass": "one equal unit per qualifying T4 player tracklet",
            "unmatchedUnitCost": 1.0,
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
            "trackletRepresentationMatchesT4": tracklet_representation_matches_t4,
            "meanMinimumTeamSeparation": mean_separation,
            "positiveRawTransportMarginFraction": positive_margin,
            "reliableSwapEvidenceNonzeroFraction": reliable_swap,
            "crossMatchCoverageNonzeroFraction": cross_coverage_nonzero,
            "conditionalSimilarityCoverageSpearman": conditional_coverage_correlation,
            "coreFeatureNonconstant": core_nonconstant,
            "strongestCorrelations": strongest_values,
            "checks": checks,
            "correlations": correlations,
        },
        "performance": performance,
        "rows": output_rows,
        "extractionAudit": extraction_audit,
        "sources": {
            "t9Features": {"path": str(t9_path), "sha256": t9_hash},
            "extractor": {"path": str(script_path), "sha256": sha256_path(script_path)},
            "module": {"path": str(module_path), "sha256": sha256_path(module_path)},
            "t3Module": {"path": str(t3_module_path), "sha256": sha256_path(t3_module_path)},
            "t4Module": {"path": str(t4_module_path), "sha256": sha256_path(t4_module_path)},
        },
        "personDetector": detector_identity(detector_dir),
        "limitations": [
            "No label or model artifact was loaded.",
            "T4 detector ownership and player tracklets are reused exactly; only the team representation changes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t9-features", type=Path, default=DEFAULT_T9)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opencv-threads", type=int, default=12)
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
