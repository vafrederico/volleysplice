#!/usr/bin/env python3
"""Build T20 compact T14/T19 features without labels or video."""

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
from analysis.side_switch_t1_transport import materialized_profile_feature
from analysis.side_switch_t20_compact_candidate import (
    T14_SOURCE_NAME,
    T19_SOURCE_NAME,
    T20_CORE_FEATURE_NAMES,
    compact_candidate_features,
)
from scripts.extract_side_switch_helpers import load_json, sha256_path


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_SOURCE = REPORTS / "side-switch-t19-cross-representation-consensus-features-v1.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-t20-compact-medoid-disagreement-features-v1.json"
EXPECTED_SOURCE_SHA256 = "ab4b474fdbaee20bf884093a37eba25e79b6e93172f0dbfe619c3b0546c421e5"


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.t19_features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T20 artifact: {output}")
    source_hash = sha256_path(source_path)
    if args.enforce_source_hash and source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"T20 T19 source identity changed: {source_hash}")
    started = time.perf_counter()
    source = load_json(source_path)
    source_rows = source["rows"]
    output_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    source_copy_exact = True
    internal_zero_exact = True
    for source_row in source_rows:
        row = dict(source_row)
        row["features"] = {
            str(name): float(value) for name, value in source_row["features"].items()
        }
        if str(row["kind"]) == "adjacent-rally-boundary":
            values = compact_candidate_features(row)
            source_copy_exact &= (
                values[T20_CORE_FEATURE_NAMES[0]] == float(row["features"][T14_SOURCE_NAME])
                and values[T20_CORE_FEATURE_NAMES[1]] == float(row["features"][T19_SOURCE_NAME])
            )
            row["features"].update(values)
            row["t20CompactMedoidDisagreement"] = {
                "status": "ok",
                "sourceCopy": {
                    T20_CORE_FEATURE_NAMES[0]: T14_SOURCE_NAME,
                    T20_CORE_FEATURE_NAMES[1]: T19_SOURCE_NAME,
                },
            }
            boundary_rows.append(row)
        else:
            zeros = {name: 0.0 for name in T20_CORE_FEATURE_NAMES}
            row["features"].update(zeros)
            internal_zero_exact &= all(value == 0.0 for value in zeros.values())
            row["t20CompactMedoidDisagreement"] = {
                "status": "not-eligible",
                "reason": "T20 is adjacent-boundary-only",
            }
        output_rows.append(row)
    if len(output_rows) != 704 or len(boundary_rows) != 624:
        raise ValueError("T20 row scope changed")
    if [str(row["eventId"]) for row in output_rows] != [
        str(row["eventId"]) for row in source_rows
    ]:
        raise ValueError("T20 row identity/order changed")
    for before, after in zip(source_rows, output_rows, strict=True):
        for name, value in before["features"].items():
            if float(after["features"][name]) != float(value):
                raise ValueError(f"T20 changed prior feature {name}")

    def values(name: str) -> np.ndarray:
        return np.asarray(
            [materialized_profile_feature(row, name) for row in boundary_rows],
            dtype=np.float64,
        )

    margin = values(T20_CORE_FEATURE_NAMES[0])
    disagreement = values(T20_CORE_FEATURE_NAMES[1])
    positive_margin_count = int(np.sum(margin > 0.0))
    nonzero_disagreement_count = int(np.sum(disagreement > 0.0))
    elapsed = time.perf_counter() - started
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    checks = {
        "priorFeatureValuesExact": True,
        "candidateIdAndOrderExact": True,
        "sourceValuesCopiedBitExactly": source_copy_exact,
        "internalRowsZeroExact": internal_zero_exact,
        "coreFeaturesFiniteAndNonconstant": all(
            np.all(np.isfinite(values(name))) and float(np.ptp(values(name))) > 1e-15
            for name in T20_CORE_FEATURE_NAMES
        ),
        "positiveMarginCountMatchesT14": positive_margin_count == 206,
        "nonzeroDisagreementCountMatchesT19": nonzero_disagreement_count == 617,
        "wallTimePassed": elapsed <= 60.0,
        "peakMemoryPassed": peak_memory <= 2048.0,
        "noVideoDetectorOrLabelSourceLoaded": True,
    }
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_t20_compact_candidate.py"
    ).resolve()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-t20-compact-medoid-disagreement-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "adaptive-opened-development-features-only",
        "engineeringDecision": "pass" if all(checks.values()) else "fail",
        "scope": source["scope"],
        "contract": {
            "featureNames": list(T20_CORE_FEATURE_NAMES),
            "futureFirstHeadFeatureNames": list(T20_CORE_FEATURE_NAMES),
            "sourceFeatureNames": [T14_SOURCE_NAME, T19_SOURCE_NAME],
            "assembly": "bit-exact source copies; no interaction or rescaling",
            "labelUseDuringTransformation": "none",
            "videoOrDetectorUse": "none; immutable T19 JSON transformation only",
        },
        "parity": {
            "rows": len(output_rows),
            "eligibleBoundaryRows": len(boundary_rows),
            "ineligibleInternalRows": 80,
            "priorFeatureValues": "exact",
            "candidateIdAndOrder": "exact",
        },
        "engineering": {
            "positiveMarginCount": positive_margin_count,
            "positiveMarginFraction": positive_margin_count / len(boundary_rows),
            "nonzeroDisagreementCount": nonzero_disagreement_count,
            "nonzeroDisagreementFraction": nonzero_disagreement_count / len(boundary_rows),
            "checks": checks,
        },
        "performance": {
            "elapsedSeconds": elapsed,
            "peakResidentMemoryMiB": peak_memory,
            "maximumWallSeconds": 60.0,
            "maximumPeakResidentMemoryMiB": 2048.0,
        },
        "rows": output_rows,
        "sources": {
            "t19Features": {"path": str(source_path), "sha256": source_hash},
            "transformer": {"path": str(script_path), "sha256": sha256_path(script_path)},
            "module": {"path": str(module_path), "sha256": sha256_path(module_path)},
        },
        "limitations": [
            "No video, detector, label, audit, feedback, or model artifact was loaded by the transformation.",
            "The exact feature assembly was selected adaptively from opened T14/T19 results.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t19-features", type=Path, default=DEFAULT_SOURCE)
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
                "engineering": payload["engineering"],
                "performance": payload["performance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
