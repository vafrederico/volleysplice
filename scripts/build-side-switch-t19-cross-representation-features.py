#!/usr/bin/env python3
"""Build T19 T4/T14 jersey consensus without labels or video."""

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
from analysis.side_switch_t18_representativeness import T18_CORE_FEATURE_NAMES
from analysis.side_switch_t19_cross_representation import (
    T4_MARGIN,
    T14_MARGIN,
    T19_CORE_FEATURE_NAMES,
    cross_representation_features,
)
from scripts.extract_side_switch_helpers import load_json, sha256_path, spearman


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_SOURCE = REPORTS / "side-switch-t18-medoid-representativeness-features-v1.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-t19-cross-representation-consensus-features-v1.json"
EXPECTED_SOURCE_SHA256 = "0efab13db25849a74413fc595e2753c735d1c9eb00ae4afaf8b73b9007ef8949"


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.t18_features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T19 artifact: {output}")
    source_hash = sha256_path(source_path)
    if args.enforce_source_hash and source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"T19 T18 source identity changed: {source_hash}")
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
            values = cross_representation_features(row)
            row["features"].update(values)
            row["t19CrossRepresentationConsensus"] = {"status": "ok"}
            t4 = float(row["features"][T4_MARGIN])
            t14 = float(row["features"][T14_MARGIN])
            maximum_formula_error = max(maximum_formula_error, abs(0.5 * (t4 + t14) - values[T19_CORE_FEATURE_NAMES[0]]), abs(abs(t4 - t14) - values[T19_CORE_FEATURE_NAMES[1]]))
            boundary_rows.append(row)
        else:
            row["features"].update({name: 0.0 for name in T19_CORE_FEATURE_NAMES})
            row["t19CrossRepresentationConsensus"] = {"status": "not-eligible", "reason": "T19 is adjacent-boundary-only"}
        output_rows.append(row)
    if len(output_rows) != 704 or len(boundary_rows) != 624:
        raise ValueError("T19 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [str(row["eventId"]) for row in source_rows]:
        raise ValueError("T19 row identity/order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T19 changed prior feature {name}")

    def values(name: str) -> np.ndarray:
        return np.asarray([materialized_profile_feature(row, name) for row in boundary_rows], dtype=np.float64)

    consensus = values(T19_CORE_FEATURE_NAMES[0])
    disagreement = values(T19_CORE_FEATURE_NAMES[1])
    core_correlation = spearman(consensus, disagreement)
    raw_correlations = {T4_MARGIN: spearman(consensus, values(T4_MARGIN)), T14_MARGIN: spearman(consensus, values(T14_MARGIN))}
    prior_families = {
        "t4": T4_CORE_FEATURE_NAMES, "t5": T5_CORE_FEATURE_NAMES,
        "t6": T6_CORE_FEATURE_NAMES, "t7": T7_CORE_FEATURE_NAMES,
        "t8": T8_CORE_FEATURE_NAMES, "t9": T9_CORE_FEATURE_NAMES,
        "t10": T10_CORE_FEATURE_NAMES, "t11": T11_CORE_FEATURE_NAMES,
        "t12": T12_CORE_FEATURE_NAMES, "t13": T13_CORE_FEATURE_NAMES,
        "t14": T14_CORE_FEATURE_NAMES, "t15": T15_CORE_FEATURE_NAMES,
        "t16": T16_CORE_FEATURE_NAMES, "t17": T17_CORE_FEATURE_NAMES,
        "t18": T18_CORE_FEATURE_NAMES, "existing": tuple(BASE_FEATURE_NAMES),
    }
    correlations: list[dict[str, Any]] = []
    for left_name in T19_CORE_FEATURE_NAMES:
        for family, names in prior_families.items():
            for right_name in names:
                correlation = spearman(values(left_name), values(right_name))
                if correlation is not None:
                    correlations.append({"left": left_name, "right": right_name, "family": family, "spearman": correlation})
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
        "coreFeaturesFiniteAndNonconstant": all(np.all(np.isfinite(values(name))) and float(np.ptp(values(name))) > 1e-15 for name in T19_CORE_FEATURE_NAMES),
        "consensusPositiveAtLeast15Percent": float(np.mean(consensus > 0.0)) >= 0.15 - 1e-12,
        "consensusNegativeAtLeast15Percent": float(np.mean(consensus < 0.0)) >= 0.15 - 1e-12,
        "disagreementNonzeroAtLeast20Percent": float(np.mean(disagreement > 0.0)) >= 0.20 - 1e-12,
        "coreCorrelationBelow98Percent": core_correlation is not None and abs(core_correlation) < 0.98 - 1e-12,
        "consensusRawT4AndT14CorrelationsBelow95Percent": all(value is not None and abs(value) < 0.95 - 1e-12 for value in raw_correlations.values()),
        **{f"{family}CorrelationBelow98Percent": value is None or abs(float(value["spearman"])) < 0.98 - 1e-12 for family, value in strongest_values.items()},
        "wallTimePassed": elapsed <= 60.0,
        "peakMemoryPassed": peak_memory <= 2048.0,
        "noVideoOrDetectorLoaded": True,
    }
    script_path = Path(__file__).resolve()
    module_path = (script_path.parent.parent / "analysis/side_switch_t19_cross_representation.py").resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t19-cross-representation-consensus-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "adaptive-opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": source["scope"],
        "contract": {"featureNames": list(T19_CORE_FEATURE_NAMES), "futureFirstHeadFeatureNames": list(T19_CORE_FEATURE_NAMES), "fusion": "equal-weight T4/T14 signed consensus plus absolute disagreement", "labelUseDuringTransformation": "none", "videoOrDetectorUse": "none"},
        "parity": {"rows": len(output_rows), "eligibleBoundaryRows": len(boundary_rows), "ineligibleInternalRows": 80, "priorFeatureValues": "exact", "candidateIdAndOrder": "exact"},
        "engineering": {"consensusPositiveFraction": float(np.mean(consensus > 0.0)), "consensusNegativeFraction": float(np.mean(consensus < 0.0)), "disagreementNonzeroFraction": float(np.mean(disagreement > 0.0)), "coreSpearman": core_correlation, "consensusRawCorrelations": raw_correlations, "maximumFormulaReconstructionError": maximum_formula_error, "strongestCorrelations": strongest_values, "checks": checks, "correlations": correlations},
        "performance": {"elapsedSeconds": elapsed, "peakResidentMemoryMiB": peak_memory, "maximumWallSeconds": 60.0, "maximumPeakResidentMemoryMiB": 2048.0},
        "rows": output_rows,
        "sources": {"t18Features": {"path": str(source_path), "sha256": source_hash}, "transformer": {"path": str(script_path), "sha256": sha256_path(script_path)}, "module": {"path": str(module_path), "sha256": sha256_path(module_path)}},
        "limitations": ["No video, detector, label, audit, feedback, or model artifact was loaded by the transformation.", "The hypothesis was selected adaptively after T14-T18."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t18-features", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    payload = build(_parser().parse_args())
    print(json.dumps({"engineeringDecision": payload["engineeringDecision"], "parity": payload["parity"], "engineering": {key: value for key, value in payload["engineering"].items() if key != "correlations"}, "performance": payload["performance"]}, indent=2))


if __name__ == "__main__":
    main()
