#!/usr/bin/env python3
"""Build immutable label-free T2 features from the T1 transport artifact."""

from __future__ import annotations

import argparse
import hashlib
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
from analysis.side_switch_t2_transport import (
    CONDITIONAL_FEATURE_NAME,
    T2_CORE_FEATURE_NAMES,
    spearman,
    t2_features,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_T1_FEATURES = REPORTS / "side-switch-t1-endpoint-identity-transport-features-v1.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-t2-conditional-identity-transport-features-v1.json"
EXPECTED_T1_SHA256 = "982c6955b9eda279bbd98a74e274e7644d8fa348fdd59da4949e11a9b2b0e5f4"
MAX_WALL_SECONDS = 60.0
MAX_PEAK_MEMORY_MIB = 256.0


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _values(rows: list[Mapping[str, Any]], name: str) -> np.ndarray:
    return np.asarray(
        [materialized_profile_feature(row, name) for row in rows],
        dtype=np.float64,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.t1_features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T2 feature artifact: {output}")
    source_hash = _sha256(source_path)
    if args.enforce_source_hash and source_hash != EXPECTED_T1_SHA256:
        raise ValueError(f"T2 input identity changed: {source_hash}")
    started = time.perf_counter()
    source = _load(source_path)
    t1_checks = source["observability"]["checks"]
    expected_failed_check = "maximumAbsoluteSpearmanBelow98Percent"
    if (
        str(source["engineeringDecision"]) != "fail"
        or bool(t1_checks[expected_failed_check])
        or not all(
            bool(value)
            for name, value in t1_checks.items()
            if name != expected_failed_check
        )
    ):
        raise ValueError("T2 source is not the exact non-redundancy-only T1 failure")

    source_rows = source["rows"]
    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    eligible = 0
    ineligible = 0
    zero_semantics_exact = True
    for source_row in source_rows:
        row = dict(source_row)
        row["features"] = {
            str(name): float(value)
            for name, value in source_row["features"].items()
        }
        if str(row["kind"]) == "adjacent-rally-boundary":
            values, diagnostics = t2_features(row)
            row["features"][CONDITIONAL_FEATURE_NAME] = values[CONDITIONAL_FEATURE_NAME]
            row["t2ConditionalTransport"] = {"status": "ok", **diagnostics}
            for direction, diagnostic_name in (
                ("nearFar", "nearBeforeToFarAfterConditionalSimilarity"),
                ("farNear", "farBeforeToNearAfterConditionalSimilarity"),
            ):
                reduction = row["t1Transport"]["reductions"][direction]
                if float(reduction["coverage"]) == 0.0:
                    zero_semantics_exact &= diagnostics[diagnostic_name] == 0.0
            boundary_rows.append(row)
            eligible += 1
        else:
            row["t2ConditionalTransport"] = {
                "status": "not-eligible",
                "reason": "T2 is adjacent-boundary-only",
            }
            ineligible += 1
        output_rows.append(row)

    if len(output_rows) != 704 or eligible != 624 or ineligible != 80:
        raise ValueError("T2 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("T2 row identity/order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T2 changed prior feature {name}")

    core_values = {name: _values(boundary_rows, name) for name in T2_CORE_FEATURE_NAMES}
    coverage_values = _values(boundary_rows, "transportCoverageMinimum")
    core_nonconstant = {
        name: float(np.ptp(values)) > 1e-15
        for name, values in core_values.items()
    }
    conditional = core_values[CONDITIONAL_FEATURE_NAME]
    if np.any((conditional < 0.0) | (conditional > 1.0)):
        raise ValueError("T2 conditional similarity escaped [0,1]")
    core_correlation = spearman(
        core_values[T2_CORE_FEATURE_NAMES[0]],
        core_values[T2_CORE_FEATURE_NAMES[1]],
    )
    coverage_correlation = spearman(conditional, coverage_values)
    if core_correlation is None or coverage_correlation is None:
        raise ValueError("T2 core/coverage correlation is undefined")
    existing_correlations = []
    for core_name, values in core_values.items():
        for existing_name in BASE_FEATURE_NAMES:
            correlation = spearman(values, _values(boundary_rows, existing_name))
            if correlation is not None:
                existing_correlations.append(
                    {
                        "coreFeature": core_name,
                        "existingFeature": existing_name,
                        "spearman": correlation,
                    }
                )
    strongest_existing = max(
        existing_correlations, key=lambda value: abs(value["spearman"])
    )
    checks = {
        "coreFeaturesNonconstant": all(core_nonconstant.values()),
        "conditionalSimilarityNonzeroAtLeast50Percent": float(
            np.mean(conditional > 0.0)
        )
        >= 0.50 - 1e-12,
        "coreAbsoluteSpearmanBelow90Percent": abs(core_correlation) < 0.90 - 1e-12,
        "conditionalCoverageAbsoluteSpearmanBelow90Percent": abs(coverage_correlation)
        < 0.90 - 1e-12,
        "existingInputAbsoluteSpearmanBelow98Percent": abs(
            strongest_existing["spearman"]
        )
        < 0.98 - 1e-12,
        "conditionalRangePassed": bool(
            np.all((conditional >= 0.0) & (conditional <= 1.0))
        ),
        "zeroSemanticsExact": bool(zero_semantics_exact),
    }
    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    checks.update(
        {
            "wallTimePassed": elapsed <= MAX_WALL_SECONDS,
            "peakMemoryPassed": peak_memory <= MAX_PEAK_MEMORY_MIB,
        }
    )
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_t2_transport.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t2-conditional-identity-transport-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": source["scope"],
        "contract": {
            "eligibleCandidateKind": "adjacent-rally-boundary",
            "futureCoreFeatureNames": list(T2_CORE_FEATURE_NAMES),
            "coverageUse": "diagnostic only; excluded from T2 core",
            "labelUse": "none",
            "videoOrDetectorUse": "none; immutable T1 JSON transformation only",
            "modelSelectionUse": "prohibited until new recording-held gold exists",
        },
        "parity": {
            "rows": len(output_rows),
            "eligibleBoundaryRows": eligible,
            "ineligibleInternalRows": ineligible,
            "priorFeatureValues": "exact",
            "candidateIdAndOrder": "exact",
        },
        "engineering": {
            "checks": checks,
            "coreFeatureNonconstant": core_nonconstant,
            "conditionalSimilarityNonzeroFraction": float(
                np.mean(conditional > 0.0)
            ),
            "coreSpearman": core_correlation,
            "conditionalCoverageSpearman": coverage_correlation,
            "strongestExistingInputSpearman": strongest_existing,
            "existingInputCorrelations": existing_correlations,
            "conditionalDistribution": {
                "minimum": float(np.min(conditional)),
                "p25": float(np.quantile(conditional, 0.25)),
                "median": float(np.median(conditional)),
                "p75": float(np.quantile(conditional, 0.75)),
                "maximum": float(np.max(conditional)),
                "mean": float(np.mean(conditional)),
            },
        },
        "performance": {
            "elapsedSeconds": elapsed,
            "peakResidentMemoryMiB": peak_memory,
            "maximumWallSeconds": MAX_WALL_SECONDS,
            "maximumPeakResidentMemoryMiB": MAX_PEAK_MEMORY_MIB,
        },
        "rows": output_rows,
        "sources": {
            "t1Features": {"path": str(source_path), "sha256": source_hash},
            "transformer": {"path": str(script_path), "sha256": _sha256(script_path)},
            "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        },
        "limitations": [
            "No model was fit or selected because no eligible new side-switch gold exists.",
            "T2 engineering novelty does not establish label separability.",
            "Coverage remains heterogeneous by recording as documented by T1.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-features", type=Path, default=DEFAULT_T1_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
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
                    if key != "existingInputCorrelations"
                },
                "performance": payload["performance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
