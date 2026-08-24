#!/usr/bin/env python3
"""Run versioned side-switch feature-development profiles with nested LOO."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_feature_development import (
    BASELINE_PROFILE,
    DECODER,
    concise_metrics,
    evaluate_profile,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_MODEL = ROOT / "models/side-switch-hard-negative-mining-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-hard-negative-mining-v1-evaluation.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-feature-development-e0-baseline-v1.json"
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "audit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "model": "c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3",
    "evaluation": "e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b",
}


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


def _metric_counts(value: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(
        int(value[name])
        for name in ("proposals", "truePositives", "falsePositives", "falseNegatives")
    )


def _validate_e0(
    result: Mapping[str, Any],
    current_model: Mapping[str, Any],
    current_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    expected = current_evaluation["fixedVariantOuterResults"]["union34-top2-x2"]
    if _metric_counts(result["primary"]) != _metric_counts(expected["primary"]):
        raise ValueError("E0 primary event metrics do not reproduce the promoted control")
    if _metric_counts(result["strict"]) != _metric_counts(expected["strict"]):
        raise ValueError("E0 strict event metrics do not reproduce the promoted control")
    if not np.isclose(
        float(result["rowAveragePrecision"]),
        float(expected["rowAveragePrecision"]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("E0 row AP does not reproduce the promoted control")

    expected_thresholds = {
        str(fold["heldRecordingId"]): float(
            next(
                row
                for row in fold["leaderboard"]
                if row["variant"]["id"] == "union34-top2-x2"
            )["threshold"]
        )
        for fold in current_evaluation["outerFolds"]
    }
    actual_thresholds = {
        str(fold["heldRecordingId"]): float(fold["threshold"])
        for fold in result["outerFolds"]
    }
    if actual_thresholds.keys() != expected_thresholds.keys() or any(
        not np.isclose(actual_thresholds[key], expected_thresholds[key], atol=1e-12, rtol=0.0)
        for key in actual_thresholds
    ):
        raise ValueError("E0 outer thresholds do not reproduce the promoted control")

    classifier = result["fullDevelopment"]["classifier"]
    expected_classifier = current_model["classifier"]
    if classifier["featureNames"] != expected_classifier["featureNames"]:
        raise ValueError("E0 final feature signature changed")
    for name in ("impute", "mean", "scale", "weights"):
        if not np.allclose(
            np.asarray(classifier[name], dtype=np.float64),
            np.asarray(expected_classifier[name], dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"E0 final classifier changed for {name}")
    for name in ("bias", "l2"):
        if not np.isclose(
            float(classifier[name]),
            float(expected_classifier[name]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"E0 final classifier changed for {name}")
    if not np.isclose(
        float(classifier["threshold"]),
        float(current_model["threshold"]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("E0 selected threshold changed")
    return {
        "status": "exact",
        "primaryCounts": list(_metric_counts(result["primary"])),
        "strictCounts": list(_metric_counts(result["strict"])),
        "rowAveragePrecision": result["rowAveragePrecision"],
        "outerThresholdsExact": True,
        "finalClassifierExact": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "audit": args.audit.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature-development artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"feature-development source identity changed: {hashes}")

    feature_payload = _load(paths["features"])
    audit = _load(paths["audit"])
    current_model = _load(paths["model"])
    current_evaluation = _load(paths["evaluation"])
    recording_ids = tuple(
        str(value) for value in feature_payload["scope"]["recordingIds"]
    )
    markers = {
        recording_id: [
            {"time": float(value)}
            for value in audit["scope"]["humanEventsByRecording"][recording_id]
        ]
        for recording_id in recording_ids
    }

    result = evaluate_profile(feature_payload["rows"], markers, BASELINE_PROFILE)
    parity = _validate_e0(result, current_model, current_evaluation)
    script_path = Path(__file__).resolve()
    module_path = (script_path.parent.parent / "analysis/side_switch_feature_development.py").resolve()
    created_at = datetime.now(UTC).isoformat()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-feature-development-e0-v1",
        "createdAt": created_at,
        "status": "opened-development-only",
        "experiment": {
            "id": "E0",
            "name": "freeze-and-reproduce-control",
            "decision": "baseline-only",
        },
        "scope": {
            **feature_payload["scope"],
            "humanMarkers": sum(len(value) for value in markers.values()),
            "positiveCandidateLabels": result["positiveCandidateLabels"],
        },
        "protocol": {
            "outer": "leave one recording out",
            "inner": "grouped leave-one-recording-out threshold selection on outer-fit recordings",
            "classifier": "fixed square-root logistic, L2 0.1, top-2/2x recording-balanced hard negatives",
            "decoder": DECODER.to_dict(),
            "primaryPaddingSeconds": 4.0,
        },
        "profiles": {BASELINE_PROFILE.identifier: result},
        "parity": parity,
        "sources": {
            **{
                name: {"path": str(paths[name]), "sha256": hashes[name]}
                for name in paths
            },
            "runner": {"path": str(script_path), "sha256": _sha256(script_path)},
            "module": {"path": str(module_path), "sha256": _sha256(module_path)},
        },
        "limitations": [
            "All 11 recordings and 50 markers are opened development data.",
            "E0 validates experiment machinery; it is not a new model result.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = run(_parser().parse_args())
    result = payload["profiles"][BASELINE_PROFILE.identifier]
    print(
        json.dumps(
            {
                "experiment": payload["experiment"],
                "parity": payload["parity"],
                "metrics": concise_metrics(result),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
