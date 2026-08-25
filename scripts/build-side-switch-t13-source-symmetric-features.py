#!/usr/bin/env python3
"""Build T13 source-symmetric routed transport without labels or video."""

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
from analysis.side_switch_t13_source_symmetric_transport import (
    T13_CORE_FEATURE_NAMES,
    T13_FEATURE_NAMES,
    source_symmetric_transport_features,
)
from scripts.extract_side_switch_helpers import load_json, sha256_path, spearman


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_SOURCE = REPORTS / "side-switch-t12-type-consistent-hierarchical-transport-features-v1.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-t13-source-symmetric-routing-features-v1.json"
EXPECTED_SOURCE_SHA256 = "8941f5f369d98c3e25f8070f6fc13fb206858729750d930e8c5b55648cf4275d"
T4_MEAN_SEPARATION = 0.37386011658617885
T7_POSITIVE_MARGIN = 0.2548076923076923
T7_RELIABLE_SWAP = 0.19391025641025642


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.t12_features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T13 artifact: {output}")
    source_hash = sha256_path(source_path)
    if args.enforce_source_hash and source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"T13 T12 source identity changed: {source_hash}")
    started = time.perf_counter()
    source = load_json(source_path)
    failed = [
        name for name, value in source["engineering"]["checks"].items() if not bool(value)
    ]
    if str(source["engineeringDecision"]) != "fail" or failed != [
        "t11CorrelationBelow98Percent"
    ]:
        raise ValueError("T13 requires the exact T11-correlation-only T12 failure")

    source_rows = source["rows"]
    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    source_route_counts = {"stable-player-units": 0, "pooled-team": 0}
    route_symmetry_exact = True
    for source_row in source_rows:
        row = dict(source_row)
        row["features"] = {
            str(name): float(value) for name, value in source_row["features"].items()
        }
        if str(row["kind"]) == "adjacent-rally-boundary":
            values, diagnostics = source_symmetric_transport_features(row)
            row["features"].update(values)
            row["t13SourceSymmetricRouting"] = {"status": "ok", **diagnostics}
            reductions = diagnostics["teamReductions"]
            route_symmetry_exact &= (
                reductions["nearNear"]["route"] == reductions["nearFar"]["route"]
                and reductions["farFar"]["route"] == reductions["farNear"]["route"]
            )
            for route in diagnostics["sourceRoutes"].values():
                source_route_counts[str(route)] += 1
            boundary_rows.append(row)
        else:
            row["t13SourceSymmetricRouting"] = {
                "status": "not-eligible",
                "reason": "T13 is adjacent-boundary-only",
            }
        output_rows.append(row)
    if len(output_rows) != 704 or len(boundary_rows) != 624:
        raise ValueError("T13 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("T13 row identity/order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T13 changed prior feature {name}")

    def values(name: str) -> np.ndarray:
        return np.asarray(
            [materialized_profile_feature(row, name) for row in boundary_rows],
            dtype=np.float64,
        )

    mean_separation = float(
        np.mean(values("sourceSymmetricTrackletJerseyTeamSeparationMinimum"))
    )
    positive_margin = float(
        np.mean(values("sourceSymmetricTrackletJerseyTeamTransportSwapMargin") > 0.0)
    )
    reliable_swap = float(
        np.mean(values("sourceSymmetricTrackletJerseyReliableSwapEvidence") > 0.0)
    )
    core_nonconstant = {
        name: float(np.ptp(values(name))) > 1e-15 for name in T13_CORE_FEATURE_NAMES
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
        "existing": tuple(BASE_FEATURE_NAMES),
    }
    comparison_names = list(T13_CORE_FEATURE_NAMES)
    for names in prior_families.values():
        comparison_names.extend(names)
    correlations: list[dict[str, Any]] = []
    for index, left_name in enumerate(T13_CORE_FEATURE_NAMES):
        left = values(left_name)
        for right_name in comparison_names[index + 1 :]:
            correlation = spearman(left, values(right_name))
            if correlation is not None:
                correlations.append(
                    {"left": left_name, "right": right_name, "spearman": correlation}
                )

    def strongest(names: tuple[str, ...]) -> Mapping[str, Any] | None:
        candidates = [value for value in correlations if value["right"] in names]
        return max(candidates, key=lambda value: abs(value["spearman"]), default=None)

    strongest_values = {"core": strongest(T13_CORE_FEATURE_NAMES)}
    strongest_values.update({name: strongest(names) for name, names in prior_families.items()})
    checks = {
        "sourceRouteSymmetryExact": route_symmetry_exact,
        "bothRoutesUsed": all(value > 0 for value in source_route_counts.values()),
        "availabilityMatchesT5": all(
            bool(source["engineering"]["checks"][name])
            for name in (
                "endpointAvailabilityMatchesT5",
                "minimumRecordingAvailabilityMatchesT5",
                "fourTeamVisibilityMatchesT5",
            )
        ),
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
    checks.update(
        {
            "wallTimePassed": elapsed <= 60.0,
            "peakMemoryPassed": peak_memory <= 512.0,
            "noVideoOrDetectorLoaded": True,
        }
    )
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_t13_source_symmetric_transport.py"
    ).resolve()
    availability = source["engineering"]
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t13-source-symmetric-routing-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": source["scope"],
        "contract": {
            "featureNames": list(T13_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T13_CORE_FEATURE_NAMES),
            "routing": "one frozen route per before-side source across both destinations",
            "labelUse": "none",
            "videoOrDetectorUse": "none; immutable T12 JSON transformation only",
        },
        "parity": {
            "rows": len(output_rows),
            "eligibleBoundaryRows": len(boundary_rows),
            "ineligibleInternalRows": 80,
            "priorFeatureValues": "exact",
            "candidateIdAndOrder": "exact",
        },
        "engineering": {
            "endpointBothQualifiedTeamsFraction": availability["endpointBothQualifiedTeamsFraction"],
            "minimumRecordingBothQualifiedTeamsFraction": availability["minimumRecordingBothQualifiedTeamsFraction"],
            "boundaryFourTeamVisibilityFraction": availability["boundaryFourTeamVisibilityFraction"],
            "sourceRouteCounts": source_route_counts,
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
            "maximumWallSeconds": 60.0,
            "maximumPeakResidentMemoryMiB": 512.0,
        },
        "rows": output_rows,
        "sources": {
            "t12Features": {"path": str(source_path), "sha256": source_hash},
            "transformer": {"path": str(script_path), "sha256": sha256_path(script_path)},
            "module": {"path": str(module_path), "sha256": sha256_path(module_path)},
        },
        "limitations": [
            "No video, detector, label, audit, feedback, or model artifact was loaded.",
            "Each before-side source uses one representation route for both destinations.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t12-features", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
