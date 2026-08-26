#!/usr/bin/env python3
"""Extract T14 dominant stable-tracklet medoid transport without labels."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_feature_development import BASE_FEATURE_NAMES
from analysis.side_switch_player_detector import detector_identity
from analysis.side_switch_t1_transport import materialized_profile_feature
from analysis.side_switch_t4_selective_far import T4_CORE_FEATURE_NAMES
from analysis.side_switch_t5_court_tracking import T5_CORE_FEATURE_NAMES
from analysis.side_switch_t6_dominant_jersey import T6_CORE_FEATURE_NAMES
from analysis.side_switch_t7_soft_consensus import T7_CORE_FEATURE_NAMES
from analysis.side_switch_t8_two_mode_transport import T8_CORE_FEATURE_NAMES
from analysis.side_switch_t9_mass_transport import T9_CORE_FEATURE_NAMES
from analysis.side_switch_t10_tracklet_unit_transport import T10_CORE_FEATURE_NAMES
from analysis.side_switch_t11_fallback_tracklet_transport import T11_CORE_FEATURE_NAMES
from analysis.side_switch_t12_hierarchical_transport import T12_CORE_FEATURE_NAMES
from analysis.side_switch_t13_source_symmetric_transport import T13_CORE_FEATURE_NAMES
from analysis.side_switch_t14_dominant_tracklet_medoid import (
    T14_CORE_FEATURE_NAMES,
    T14_FEATURE_NAMES,
    dominant_tracklet_transport_features,
    summarize_dominant_tracklet_endpoint,
)
from scripts.extract_side_switch_helpers import load_json, sha256_path, spearman, window_key
from scripts.side_switch_parallel_endpoint_extraction import extract_parallel_endpoints


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_SOURCE = REPORTS / "side-switch-t13-source-symmetric-routing-features-v1.json"
DEFAULT_VIDEO_AUDIT = REPORTS / "side-switch-t11-empty-side-fallback-tracklet-transport-features-v1.json"
DEFAULT_DETECTOR_DIR = ROOT / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
DEFAULT_OUTPUT = REPORTS / "side-switch-t14-dominant-tracklet-medoid-features-v1.json"
EXPECTED_SOURCE_SHA256 = "232a26201edf16fe104dec965e73bb9f1864162baa13749187b0d3ad4970972e"
EXPECTED_VIDEO_AUDIT_SHA256 = "aa0b6abfa6324350b7944728757ee2867e7ab6c35eadfc1e085f73339e74a6d5"
T5_ENDPOINT_BOTH = 0.8110236220472441
T5_MINIMUM_RECORDING_BOTH = 0.4230769230769231
T5_FOUR_TEAM = 0.6875
T4_MEAN_SEPARATION = 0.37386011658617885
T7_POSITIVE_MARGIN = 0.2548076923076923
T7_RELIABLE_SWAP = 0.19391025641025642


def _matches_prior(summary: Any, prior: Mapping[str, Any]) -> bool:
    base = summary.base
    t10 = prior["t10TrackletUnitJerseyTransport"]
    t5 = prior["t5CourtTrackedFarJerseyTransport"]
    return (
        all(
            len(getattr(base, f"{side}_stable"))
            == int(t10[f"{side}QualifyingTracklets"])
            and bool(getattr(summary, side)) == bool(t5[f"{side}TeamAvailable"])
            for side in ("near", "far")
        )
        and list(base.selected_counts) == t10["selectedPlayersByFrame"]
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.t13_features.expanduser().resolve()
    video_audit_path = args.video_audit_features.expanduser().resolve()
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T14 artifact: {output}")
    source_hash = sha256_path(source_path)
    audit_hash = sha256_path(video_audit_path)
    if args.enforce_source_hash and source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"T14 T13 source identity changed: {source_hash}")
    if args.enforce_source_hash and audit_hash != EXPECTED_VIDEO_AUDIT_SHA256:
        raise ValueError(f"T14 video-audit identity changed: {audit_hash}")
    started = time.perf_counter()
    source = load_json(source_path)
    video_source = load_json(video_audit_path)
    source_rows = [dict(row) for row in source["rows"]]
    extraction = extract_parallel_endpoints(
        rows=source_rows,
        scope=source["scope"],
        video_audit=video_source["extractionAudit"],
        detector_dir=detector_dir,
        summarize=summarize_dominant_tracklet_endpoint,
        iteration="T14",
        recording_workers=args.recording_workers,
        opencv_threads=args.opencv_threads,
    )
    if len(extraction.endpoints) != 635:
        raise ValueError("T14 endpoint count changed")

    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    four_team_values: list[bool] = []
    prior_matches = True
    for source_row in source_rows:
        row = dict(source_row)
        row["features"] = {
            str(name): float(value) for name, value in source_row["features"].items()
        }
        if str(row["kind"]) == "adjacent-rally-boundary":
            recording_id = str(row["recordingId"])
            before = extraction.endpoints[
                window_key(recording_id, row["comparisonWindows"]["before"])
            ]
            after = extraction.endpoints[
                window_key(recording_id, row["comparisonWindows"]["after"])
            ]
            values, diagnostics = dominant_tracklet_transport_features(before, after)
            row["features"].update(values)
            row["t14DominantTrackletMedoid"] = {
                "status": "ok",
                "before": before.to_diagnostic(),
                "after": after.to_diagnostic(),
                **diagnostics,
            }
            prior_matches &= _matches_prior(
                before,
                {
                    "t10TrackletUnitJerseyTransport": row[
                        "t10TrackletUnitJerseyTransport"
                    ]["before"],
                    "t5CourtTrackedFarJerseyTransport": row[
                        "t5CourtTrackedFarJerseyTransport"
                    ]["before"],
                },
            )
            prior_matches &= _matches_prior(
                after,
                {
                    "t10TrackletUnitJerseyTransport": row[
                        "t10TrackletUnitJerseyTransport"
                    ]["after"],
                    "t5CourtTrackedFarJerseyTransport": row[
                        "t5CourtTrackedFarJerseyTransport"
                    ]["after"],
                },
            )
            four_team_values.append(
                bool(before.near)
                and bool(before.far)
                and bool(after.near)
                and bool(after.far)
            )
            boundary_rows.append(row)
        else:
            row["t14DominantTrackletMedoid"] = {
                "status": "not-eligible",
                "reason": "T14 is adjacent-boundary-only",
            }
        output_rows.append(row)
    if len(output_rows) != 704 or len(boundary_rows) != 624:
        raise ValueError("T14 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("T14 row identity/order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T14 changed prior feature {name}")

    endpoints = list(extraction.endpoints.values())
    endpoint_both = float(np.mean([bool(value.near) and bool(value.far) for value in endpoints]))
    recording_both = []
    for recording_id in source["scope"]["recordingIds"]:
        local = [
            value
            for key, value in extraction.endpoints.items()
            if key[0] == str(recording_id)
        ]
        recording_both.append(float(np.mean([bool(v.near) and bool(v.far) for v in local])))
    minimum_recording_both = min(recording_both)
    four_team = float(np.mean(four_team_values))
    selections = [
        selection
        for endpoint in endpoints
        for selection in (endpoint.near_selection, endpoint.far_selection)
    ]
    multi_selections = [value for value in selections if value.stable_tracklets >= 2]
    medoid_distances = [
        float(value.medoid_to_pool_distance)
        for value in multi_selections
        if value.medoid_to_pool_distance is not None
    ]
    fallback_sides = sum(value.kind == "pooled-fallback" for value in selections)

    def feature_values(name: str) -> np.ndarray:
        return np.asarray(
            [materialized_profile_feature(row, name) for row in boundary_rows],
            dtype=np.float64,
        )

    mean_separation = float(np.mean(feature_values("dominantTrackletJerseyTeamSeparationMinimum")))
    positive_margin = float(
        np.mean(feature_values("dominantTrackletJerseyTeamTransportSwapMargin") > 0.0)
    )
    reliable_swap = float(
        np.mean(feature_values("dominantTrackletJerseyReliableSwapEvidence") > 0.0)
    )
    mean_medoid_distance = float(np.mean(medoid_distances)) if medoid_distances else 0.0
    core_nonconstant = {
        name: float(np.ptp(feature_values(name))) > 1e-15 for name in T14_CORE_FEATURE_NAMES
    }
    prior_families = {
        "t4": T4_CORE_FEATURE_NAMES,
        "t5": T5_CORE_FEATURE_NAMES,
        "t6": T6_CORE_FEATURE_NAMES,
        "t7": T7_CORE_FEATURE_NAMES,
        "t8": T8_CORE_FEATURE_NAMES,
        "t9": T9_CORE_FEATURE_NAMES,
        "t10": T10_CORE_FEATURE_NAMES,
        "t11": T11_CORE_FEATURE_NAMES,
        "t12": T12_CORE_FEATURE_NAMES,
        "t13": T13_CORE_FEATURE_NAMES,
        "existing": tuple(BASE_FEATURE_NAMES),
    }
    comparison_names = list(T14_CORE_FEATURE_NAMES)
    for names in prior_families.values():
        comparison_names.extend(names)
    correlations: list[dict[str, Any]] = []
    for index, left_name in enumerate(T14_CORE_FEATURE_NAMES):
        left = feature_values(left_name)
        for right_name in comparison_names[index + 1 :]:
            correlation = spearman(left, feature_values(right_name))
            if correlation is not None:
                correlations.append(
                    {"left": left_name, "right": right_name, "spearman": correlation}
                )

    def strongest(names: tuple[str, ...]) -> Mapping[str, Any] | None:
        candidates = [value for value in correlations if value["right"] in names]
        return max(candidates, key=lambda value: abs(value["spearman"]), default=None)

    strongest_values = {"core": strongest(T14_CORE_FEATURE_NAMES)}
    strongest_values.update({name: strongest(names) for name, names in prior_families.items()})
    checks = {
        "priorRepresentationMatches": prior_matches,
        "endpointAvailabilityMatchesT5": abs(endpoint_both - T5_ENDPOINT_BOTH) <= 1e-12,
        "minimumRecordingAvailabilityMatchesT5": abs(
            minimum_recording_both - T5_MINIMUM_RECORDING_BOTH
        )
        <= 1e-12,
        "fourTeamVisibilityMatchesT5": abs(four_team - T5_FOUR_TEAM) <= 1e-12,
        "medoidAndFallbackActive": bool(multi_selections) and fallback_sides > 0,
        "meanMedoidToPooledDistanceAtLeast001": mean_medoid_distance >= 0.01 - 1e-12,
        "meanSeparationAtLeastT4": mean_separation >= T4_MEAN_SEPARATION - 1e-12,
        "positiveMarginAtLeastT7": positive_margin >= T7_POSITIVE_MARGIN - 1e-12,
        "reliableSwapAtLeastT7": reliable_swap >= T7_RELIABLE_SWAP - 1e-12,
        "coreFeaturesNonconstant": all(core_nonconstant.values()),
        **{
            f"{name}CorrelationBelow98Percent": value is None
            or abs(value["spearman"]) < 0.98 - 1e-12
            for name, value in strongest_values.items()
        },
    }
    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    frame_requests = sum(int(value["frameRequests"]) for value in extraction.audit.values())
    tile_calls = sum(int(value["detectorTileCalls"]) for value in extraction.audit.values())
    checks.update(
        {
            "executionShapeExact": len(endpoints) == 635
            and frame_requests == 3175
            and tile_calls == 25400,
            "wallTimePassed": elapsed <= 3600.0,
            "peakMemoryPassed": peak_memory <= 2048.0,
            "noFrameErrors": all(
                int(value["frameErrors"]) == 0 for value in extraction.audit.values()
            ),
        }
    )
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_t14_dominant_tracklet_medoid.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t14-dominant-tracklet-medoid-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": source["scope"],
        "contract": {
            "featureNames": list(T14_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T14_CORE_FEATURE_NAMES),
            "stableRepresentation": "unweighted Hellinger medoid tracklet",
            "emptySideFallback": "exact T5 pooled team",
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
            "multiTrackletSides": len(multi_selections),
            "fallbackSides": fallback_sides,
            "meanMedoidToPooledDescriptorDistance": mean_medoid_distance,
            "meanMinimumTeamSeparation": mean_separation,
            "positiveRawTransportMarginFraction": positive_margin,
            "reliableSwapEvidenceNonzeroFraction": reliable_swap,
            "coreFeatureNonconstant": core_nonconstant,
            "strongestCorrelations": strongest_values,
            "checks": checks,
            "correlations": correlations,
        },
        "performance": {
            "elapsedSeconds": elapsed,
            "peakResidentMemoryMiB": peak_memory,
            "endpointWindows": len(endpoints),
            "frameRequests": frame_requests,
            "detectorTileCalls": tile_calls,
            "recordingWorkers": extraction.recording_workers,
            "opencvThreadsPerWorker": extraction.opencv_threads_per_worker,
            "budget": {
                "wallTimePassed": elapsed <= 3600.0,
                "peakMemoryPassed": peak_memory <= 2048.0,
            },
        },
        "rows": output_rows,
        "extractionAudit": extraction.audit,
        "sources": {
            "t13Features": {"path": str(source_path), "sha256": source_hash},
            "t11VideoAudit": {"path": str(video_audit_path), "sha256": audit_hash},
            "extractor": {"path": str(script_path), "sha256": sha256_path(script_path)},
            "module": {"path": str(module_path), "sha256": sha256_path(module_path)},
        },
        "personDetector": detector_identity(detector_dir),
        "limitations": [
            "No label, audit decision, feedback, or model artifact was loaded.",
            "T14 changes only stable-side aggregation to a player-tracklet medoid.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t13-features", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--video-audit-features", type=Path, default=DEFAULT_VIDEO_AUDIT)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recording-workers", type=int, default=4)
    parser.add_argument("--opencv-threads", type=int, default=3)
    parser.add_argument("--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    payload = build(_parser().parse_args())
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
