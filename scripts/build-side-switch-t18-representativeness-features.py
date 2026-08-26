#!/usr/bin/env python3
"""Build T18 medoid representativeness transport without labels or video."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from analysis.side_switch_t13_source_symmetric_transport import T13_CORE_FEATURE_NAMES
from analysis.side_switch_t14_dominant_tracklet_medoid import T14_CORE_FEATURE_NAMES
from analysis.side_switch_t15_bilateral_consensus import T15_CORE_FEATURE_NAMES
from analysis.side_switch_t16_source_resolved import T16_CORE_FEATURE_NAMES
from analysis.side_switch_t17_relative_contrast import T17_CORE_FEATURE_NAMES
from analysis.side_switch_t18_representativeness import (
    T18_CORE_FEATURE_NAMES,
    T18_FEATURE_NAMES,
    representativeness_features,
)
from scripts.extract_side_switch_helpers import load_json, sha256_path, spearman


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_SOURCE = REPORTS / "side-switch-t17-relative-source-contrast-features-v1.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-t18-medoid-representativeness-features-v1.json"
EXPECTED_SOURCE_SHA256 = "200ac7445e6e8b8b64b97aa07b25164761da3526a59a2eebca2e0b7dae5c35dd"


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.t17_features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T18 artifact: {output}")
    source_hash = sha256_path(source_path)
    if args.enforce_source_hash and source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"T18 T17 source identity changed: {source_hash}")
    started = time.perf_counter()
    source = load_json(source_path)
    source_rows = source["rows"]
    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    maximum_formula_error = 0.0
    for source_row in source_rows:
        row = dict(source_row)
        row["features"] = {str(name): float(value) for name, value in source_row["features"].items()}
        if str(row["kind"]) == "adjacent-rally-boundary":
            values = representativeness_features(row)
            row["features"].update(values)
            row["t18MedoidRepresentativeness"] = {"status": "ok"}
            expected = float(row["features"]["dominantTrackletJerseyTeamTransportSwapMargin"]) * values["representativeMedoidJerseyGateMinimum"]
            maximum_formula_error = max(maximum_formula_error, abs(expected - values[T18_CORE_FEATURE_NAMES[0]]))
            boundary_rows.append(row)
        else:
            row["features"].update({name: 0.0 for name in T18_FEATURE_NAMES})
            row["t18MedoidRepresentativeness"] = {"status": "not-eligible", "reason": "T18 is adjacent-boundary-only"}
        output_rows.append(row)
    if len(output_rows) != 704 or len(boundary_rows) != 624:
        raise ValueError("T18 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [str(row["eventId"]) for row in source_rows]:
        raise ValueError("T18 row identity/order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T18 changed prior feature {name}")

    def values(name: str) -> np.ndarray:
        return np.asarray([materialized_profile_feature(row, name) for row in boundary_rows], dtype=np.float64)

    core = values(T18_CORE_FEATURE_NAMES[0])
    gate = values("representativeMedoidJerseyGateMinimum")
    raw = values("dominantTrackletJerseyTeamTransportSwapMargin")
    sign_matches = bool(np.array_equal(np.sign(core[gate > 0.0]), np.sign(raw[gate > 0.0])))
    prior_families = {
        "t4": T4_CORE_FEATURE_NAMES, "t5": T5_CORE_FEATURE_NAMES,
        "t6": T6_CORE_FEATURE_NAMES, "t7": T7_CORE_FEATURE_NAMES,
        "t8": T8_CORE_FEATURE_NAMES, "t9": T9_CORE_FEATURE_NAMES,
        "t10": T10_CORE_FEATURE_NAMES, "t11": T11_CORE_FEATURE_NAMES,
        "t12": T12_CORE_FEATURE_NAMES, "t13": T13_CORE_FEATURE_NAMES,
        "t14": T14_CORE_FEATURE_NAMES, "t15": T15_CORE_FEATURE_NAMES,
        "t16": T16_CORE_FEATURE_NAMES, "t17": T17_CORE_FEATURE_NAMES,
        "existing": tuple(BASE_FEATURE_NAMES),
    }
    correlations: list[dict[str, Any]] = []
    for family, names in prior_families.items():
        for right_name in names:
            correlation = spearman(core, values(right_name))
            if correlation is not None:
                correlations.append({"right": right_name, "family": family, "spearman": correlation})
    strongest_values = {
        family: max((value for value in correlations if value["family"] == family), key=lambda value: abs(value["spearman"]), default=None)
        for family in prior_families
    }
    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    checks = {
        "priorFeatureValuesExact": True,
        "candidateIdAndOrderExact": True,
        "formulaReconstructionWithin1e12": maximum_formula_error <= 1e-12,
        "coreFiniteAndNonconstant": bool(np.all(np.isfinite(core))) and float(np.ptp(core)) > 1e-15,
        "gateWithinUnitInterval": bool(np.all((gate >= 0.0) & (gate <= 1.0))),
        "gateBelow099AtLeast20Percent": float(np.mean(gate < 0.99)) >= 0.20 - 1e-12,
        "gatePositiveAtLeast50Percent": float(np.mean(gate > 0.0)) >= 0.50 - 1e-12,
        "transformedMarginNonzeroAtLeast15Percent": float(np.mean(core != 0.0)) >= 0.15 - 1e-12,
        "signAgreementExactWhenGatePositive": sign_matches,
        **{f"{family}CorrelationBelow98Percent": value is None or abs(float(value["spearman"])) < 0.98 - 1e-12 for family, value in strongest_values.items()},
        "wallTimePassed": elapsed <= 60.0,
        "peakMemoryPassed": peak_memory <= 2048.0,
        "noVideoOrDetectorLoaded": True,
    }
    script_path = Path(__file__).resolve()
    module_path = (script_path.parent.parent / "analysis/side_switch_t18_representativeness.py").resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t18-medoid-representativeness-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "adaptive-opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": source["scope"],
        "contract": {"featureNames": list(T18_FEATURE_NAMES), "futureFirstHeadFeatureNames": list(T18_CORE_FEATURE_NAMES), "gate": "minimum four-side medoid-to-pooled representativeness", "labelUseDuringTransformation": "none", "videoOrDetectorUse": "none"},
        "parity": {"rows": len(output_rows), "eligibleBoundaryRows": len(boundary_rows), "ineligibleInternalRows": 80, "priorFeatureValues": "exact", "candidateIdAndOrder": "exact"},
        "engineering": {"gatePositiveFraction": float(np.mean(gate > 0.0)), "gateBelow099Fraction": float(np.mean(gate < 0.99)), "transformedMarginNonzeroFraction": float(np.mean(core != 0.0)), "meanGate": float(np.mean(gate)), "maximumFormulaReconstructionError": maximum_formula_error, "strongestCorrelations": strongest_values, "checks": checks, "correlations": correlations},
        "performance": {"elapsedSeconds": elapsed, "peakResidentMemoryMiB": peak_memory, "maximumWallSeconds": 60.0, "maximumPeakResidentMemoryMiB": 2048.0},
        "rows": output_rows,
        "sources": {"t17Features": {"path": str(source_path), "sha256": source_hash}, "transformer": {"path": str(script_path), "sha256": sha256_path(script_path)}, "module": {"path": str(module_path), "sha256": sha256_path(module_path)}},
        "limitations": ["No video, detector, label, audit, feedback, or model artifact was loaded by the transformation.", "The hypothesis was selected adaptively after T14-T17."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t17-features", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    payload = build(_parser().parse_args())
    print(json.dumps({"engineeringDecision": payload["engineeringDecision"], "parity": payload["parity"], "engineering": {key: value for key, value in payload["engineering"].items() if key != "correlations"}, "performance": payload["performance"]}, indent=2))


if __name__ == "__main__":
    main()
