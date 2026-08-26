#!/usr/bin/env python3
"""Extract T11 empty-side fallback tracklet transport without labels."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
from analysis.side_switch_t10_tracklet_unit_transport import T10_CORE_FEATURE_NAMES
from analysis.side_switch_t11_fallback_tracklet_transport import (
    T11_CORE_FEATURE_NAMES,
    T11_FEATURE_NAMES,
    fallback_tracklet_transport_features,
    summarize_fallback_tracklet_endpoint,
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
DEFAULT_T10 = REPORTS / "side-switch-t10-tracklet-unit-jersey-transport-features-v1.json"
DEFAULT_DETECTOR_DIR = ROOT / "models/third-party/opencv-zoo-mediapipe-person-int8bq-2023mar"
DEFAULT_OUTPUT = REPORTS / "side-switch-t11-empty-side-fallback-tracklet-transport-features-v1.json"
EXPECTED_T10_SHA256 = "10ab41ebab069f20edf7b720152dc0ec94ef982f12a88200ec564b6c63849cce"
T5_ENDPOINT_BOTH = 0.8110236220472441
T5_MINIMUM_RECORDING_BOTH = 0.4230769230769231
T5_FOUR_TEAM = 0.6875
T4_MEAN_SEPARATION = 0.37386011658617885
T7_POSITIVE_MARGIN = 0.2548076923076923
T7_RELIABLE_SWAP = 0.19391025641025642
EXPECTED_ENDPOINTS = 635
EXPECTED_FRAMES = 3175
EXPECTED_TILES = 25400


def _endpoint_priors(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, float, float], dict[str, Mapping[str, Any]]]:
    result: dict[tuple[str, float, float], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if str(row["kind"]) != "adjacent-rally-boundary":
            continue
        recording_id = str(row["recordingId"])
        for side in ("before", "after"):
            key = window_key(recording_id, row["comparisonWindows"][side])
            value = {
                "t10": row["t10TrackletUnitJerseyTransport"][side],
                "t5": row["t5CourtTrackedFarJerseyTransport"][side],
            }
            prior = result.get(key)
            if prior is not None:
                for family in ("t10", "t5"):
                    for name in (
                        "nearTeamAvailable",
                        "farTeamAvailable",
                        "selectedPlayersByFrame",
                    ):
                        if name in value[family] and prior[family][name] != value[family][name]:
                            raise ValueError("T11 found inconsistent stored endpoint diagnostics")
            result[key] = value
    return result


def _matches_prior(summary: Any, prior: Mapping[str, Mapping[str, Any]]) -> bool:
    t10 = prior["t10"]
    return (
        len(summary.near_stable) == int(t10["nearQualifyingTracklets"])
        and len(summary.far_stable) == int(t10["farQualifyingTracklets"])
        and list(summary.selected_counts) == t10["selectedPlayersByFrame"]
        and bool(summary.near) == bool(prior["t5"]["nearTeamAvailable"])
        and bool(summary.far) == bool(prior["t5"]["farTeamAvailable"])
    )


def _fallback_rule_valid(summary: Any) -> bool:
    return (
        summary.near_fallback_used
        == (not summary.near_stable and summary.near_pool.team.available)
        and summary.far_fallback_used
        == (not summary.far_stable and summary.far_pool.team.available)
        and not (summary.near_stable and summary.near_fallback_used)
        and not (summary.far_stable and summary.far_fallback_used)
    )


def extract(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.t10_features.expanduser().resolve()
    detector_dir = args.detector_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T11 artifact: {output}")
    source_hash = sha256_path(source_path)
    if args.enforce_source_hash and source_hash != EXPECTED_T10_SHA256:
        raise ValueError(f"T11 T10 source identity changed: {source_hash}")
    started = time.perf_counter()
    source = load_json(source_path)
    if (
        str(source["engineeringDecision"]) != "fail"
        or int(source["parity"]["rows"]) != 704
        or str(source["parity"]["priorFeatureValues"]) != "exact"
        or str(source["parity"]["candidateIdAndOrder"]) != "exact"
    ):
        raise ValueError("T11 requires the exact failed T10 engineering artifact")
    source_rows = [dict(row) for row in source["rows"]]
    priors = _endpoint_priors(source_rows)
    if len(priors) != EXPECTED_ENDPOINTS:
        raise ValueError("T11 stored endpoint scope changed")
    recording_ids = tuple(str(value) for value in source["scope"]["recordingIds"])
    rows_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        rows_by_recording[str(row["recordingId"])].append(row)
    if args.recording_workers < 1:
        raise ValueError("T11 recording workers must be positive")
    thread_state = threading.local()

    def process_recording(
        number: int, recording_id: str
    ) -> tuple[str, dict[tuple[str, float, float], Any], dict[str, Any]]:
        detector = getattr(thread_state, "detector", None)
        if detector is None:
            detector = QuantizedPersonDetector(
                detector_dir, opencv_threads=args.opencv_threads
            )
            thread_state.detector = detector
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
        prior_audit = source["extractionAudit"][recording_id]
        video_path = Path(str(prior_audit["videoPath"])).resolve()
        if not video_path.is_file() or video_path.stat().st_size != int(
            prior_audit["videoSizeBytes"]
        ):
            raise ValueError(f"T11 video provenance changed for {recording_id}")
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
            raise RuntimeError(f"could not open T11 video: {video_path}")
        local_started = time.perf_counter()
        local: list[Any] = []
        local_endpoints: dict[tuple[str, float, float], Any] = {}
        local_matches = True
        fallback_rules_valid = True
        fallback_sides = 0
        try:
            for key, window in sorted(windows.items(), key=lambda value: value[0][1:]):
                frames = [
                    crop_roi(read_frame(capture, timestamp), roi)
                    for timestamp in endpoint_sample_times(
                        float(window["start"]), float(window["end"])
                    )
                ]
                summary = summarize_fallback_tracklet_endpoint(
                    frames, net_y_ratio, detector
                )
                local_endpoints[key] = summary
                local.append(summary)
                local_matches = local_matches and _matches_prior(summary, priors[key])
                fallback_rules_valid = fallback_rules_valid and _fallback_rule_valid(summary)
                fallback_sides += int(summary.near_fallback_used) + int(
                    summary.far_fallback_used
                )
        finally:
            capture.release()
        both = float(np.mean([bool(value.near) and bool(value.far) for value in local]))
        t5_both = float(
            np.mean(
                [
                    bool(priors[key]["t5"]["nearTeamAvailable"])
                    and bool(priors[key]["t5"]["farTeamAvailable"])
                    for key in windows
                ]
            )
        )
        audit = {
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
            "t5EndpointBothQualifiedTeamsFraction": t5_both,
            "availabilityMatchesT5": abs(both - t5_both) <= 1e-12,
            "priorRepresentationMatches": local_matches,
            "fallbackRulesValid": fallback_rules_valid,
            "fallbackSides": fallback_sides,
            "elapsedSeconds": time.perf_counter() - local_started,
        }
        print(
            f"[{number}/{len(recording_ids)}] {recording_id}: complete",
            file=sys.stderr,
            flush=True,
        )
        return recording_id, local_endpoints, audit

    endpoints: dict[tuple[str, float, float], Any] = {}
    extraction_audit: dict[str, Any] = {}
    maximum_workers = min(args.recording_workers, len(recording_ids))
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = [
            executor.submit(process_recording, number, recording_id)
            for number, recording_id in enumerate(recording_ids, 1)
        ]
        for future in futures:
            recording_id, local_endpoints, audit = future.result()
            endpoints.update(local_endpoints)
            extraction_audit[recording_id] = audit
    if len(endpoints) != EXPECTED_ENDPOINTS:
        raise ValueError("T11 endpoint count changed")

    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    four_team_values: list[bool] = []
    prior_representation_matches = True
    fallback_rules_valid = True
    fallback_sides = 0
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
            values, diagnostics = fallback_tracklet_transport_features(before, after)
            updated["features"].update(values)
            updated["t11EmptySideFallbackTrackletTransport"] = {
                "status": "ok",
                "before": before.to_diagnostic(),
                "after": after.to_diagnostic(),
                **diagnostics,
            }
            for summary, side in ((before, "before"), (after, "after")):
                key = window_key(recording_id, row["comparisonWindows"][side])
                prior_representation_matches = prior_representation_matches and _matches_prior(
                    summary, priors[key]
                )
                fallback_rules_valid = fallback_rules_valid and _fallback_rule_valid(summary)
            fallback_sides += sum(
                int(value)
                for value in (
                    before.near_fallback_used,
                    before.far_fallback_used,
                    after.near_fallback_used,
                    after.far_fallback_used,
                )
            )
            four_team_values.append(
                bool(before.near)
                and bool(before.far)
                and bool(after.near)
                and bool(after.far)
            )
            boundary_rows.append(updated)
        else:
            updated["t11EmptySideFallbackTrackletTransport"] = {
                "status": "not-eligible",
                "reason": "T11 is adjacent-boundary-only",
            }
        output_rows.append(updated)
    if len(boundary_rows) != 624 or len(output_rows) != 704:
        raise ValueError("T11 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("T11 row order changed")
    for before_row, after_row in zip(source_rows, output_rows, strict=True):
        for name, value in before_row["features"].items():
            if float(after_row["features"][name]) != float(value):
                raise ValueError(f"T11 changed prior feature {name}")

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
                row["features"]["fallbackTrackletUnitJerseyTeamSeparationMinimum"]
                for row in boundary_rows
            ]
        )
    )
    positive_margin = float(
        np.mean(
            [
                row["features"]["fallbackTrackletUnitJerseyTeamTransportSwapMargin"] > 0.0
                for row in boundary_rows
            ]
        )
    )
    reliable_swap = float(
        np.mean(
            [
                row["features"]["fallbackTrackletUnitJerseyReliableSwapEvidence"] > 0.0
                for row in boundary_rows
            ]
        )
    )
    cross_coverage = np.asarray(
        [
            row["features"]["fallbackTrackletUnitJerseyCrossMatchCoverageMinimum"]
            for row in boundary_rows
        ]
    )
    conditional_similarity = np.asarray(
        [
            row["features"]["fallbackTrackletUnitJerseyConditionalCrossSimilarityMinimum"]
            for row in boundary_rows
        ]
    )
    cross_coverage_nonzero = float(np.mean(cross_coverage > 0.0))
    conditional_coverage_correlation = spearman(conditional_similarity, cross_coverage)
    core_nonconstant = {
        name: float(np.ptp([row["features"][name] for row in boundary_rows])) > 1e-15
        for name in T11_CORE_FEATURE_NAMES
    }

    prior_families = {
        "t4": T4_CORE_FEATURE_NAMES,
        "t5": T5_CORE_FEATURE_NAMES,
        "t6": T6_CORE_FEATURE_NAMES,
        "t7": T7_CORE_FEATURE_NAMES,
        "t8": T8_CORE_FEATURE_NAMES,
        "t9": T9_CORE_FEATURE_NAMES,
        "t10": T10_CORE_FEATURE_NAMES,
        "existing": tuple(BASE_FEATURE_NAMES),
    }
    correlations: list[dict[str, Any]] = []
    comparison = (*T11_CORE_FEATURE_NAMES, *(name for values in prior_families.values() for name in values))
    for index, left_name in enumerate(T11_CORE_FEATURE_NAMES):
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

    strongest_values = {"core": strongest(T11_CORE_FEATURE_NAMES)}
    strongest_values.update({name: strongest(values) for name, values in prior_families.items()})
    checks = {
        "endpointAvailabilityMatchesT5": abs(endpoint_both - T5_ENDPOINT_BOTH) <= 1e-12
        and all(bool(value["availabilityMatchesT5"]) for value in extraction_audit.values()),
        "minimumRecordingAvailabilityMatchesT5": abs(
            minimum_recording_both - T5_MINIMUM_RECORDING_BOTH
        )
        <= 1e-12,
        "fourTeamVisibilityMatchesT5": abs(four_team - T5_FOUR_TEAM) <= 1e-12,
        "priorRepresentationMatches": prior_representation_matches
        and all(bool(value["priorRepresentationMatches"]) for value in extraction_audit.values()),
        "fallbackUsedAndRulesValid": fallback_sides > 0
        and fallback_rules_valid
        and all(bool(value["fallbackRulesValid"]) for value in extraction_audit.values()),
        "meanSeparationAtLeastT4": mean_separation >= T4_MEAN_SEPARATION - 1e-12,
        "positiveMarginAtLeastT7": positive_margin >= T7_POSITIVE_MARGIN - 1e-12,
        "reliableSwapAtLeastT7": reliable_swap >= T7_RELIABLE_SWAP - 1e-12,
        "crossMatchCoverageNonzeroAtLeast60Percent": cross_coverage_nonzero >= 0.60,
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
        "recordingWorkers": maximum_workers,
        "opencvThreadsPerWorker": args.opencv_threads,
        "budget": {
            "wallTimePassed": elapsed <= 3600.0,
            "peakMemoryPassed": peak_memory <= 2048.0,
        },
    }
    if performance["frameRequests"] != EXPECTED_FRAMES or performance["detectorTileCalls"] != EXPECTED_TILES:
        raise ValueError("T11 execution shape changed")
    checks.update(
        {
            "wallTimePassed": bool(performance["budget"]["wallTimePassed"]),
            "peakMemoryPassed": bool(performance["budget"]["peakMemoryPassed"]),
            "noFrameErrors": all(int(value["frameErrors"]) == 0 for value in extraction_audit.values()),
        }
    )
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_t11_fallback_tracklet_transport.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t11-empty-side-fallback-tracklet-transport-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": source["scope"],
        "contract": {
            "featureNames": list(T11_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T11_CORE_FEATURE_NAMES),
            "candidateGeneration": "unchanged",
            "labelUse": "none",
            "stableUnits": "exact T10",
            "emptySideFallback": "one exact T5 pooled team unit",
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
            "priorRepresentationMatches": prior_representation_matches,
            "fallbackSides": fallback_sides,
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
            "t10Features": {"path": str(source_path), "sha256": source_hash},
            "extractor": {"path": str(script_path), "sha256": sha256_path(script_path)},
            "module": {"path": str(module_path), "sha256": sha256_path(module_path)},
        },
        "personDetector": detector_identity(detector_dir),
        "limitations": [
            "No label or model artifact was loaded.",
            "Only an empty stable-unit side may receive one exact T5 pooled fallback unit.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t10-features", type=Path, default=DEFAULT_T10)
    parser.add_argument("--detector-dir", type=Path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recording-workers", type=int, default=4)
    parser.add_argument("--opencv-threads", type=int, default=3)
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
