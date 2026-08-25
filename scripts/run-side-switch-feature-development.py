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
    GAP_SHAPE_PROFILE,
    INTERACTION_PROFILE,
    Q1_PROFILE,
    concise_metrics,
    evaluate_profile,
    metric_delta,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_MODEL = ROOT / "models/side-switch-hard-negative-mining-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-hard-negative-mining-v1-evaluation.json"
DEFAULT_IMPORTANCE = REPORTS / "side-switch-feature-importance-2026-08-24.json"
DEFAULT_GAP_FEATURES = REPORTS / "side-switch-gap-shape-features-v1.json"
DEFAULT_VISUAL_FEATURES = REPORTS / "side-switch-visual-summary-v2-features-v1.json"
DEFAULT_OUTPUTS = {
    "E0": REPORTS / "side-switch-feature-development-e0-baseline-v1.json",
    "E1": REPORTS / "side-switch-feature-development-e1-interactions-v1.json",
    "E2": REPORTS / "side-switch-feature-development-e2-gap-shape-v1.json",
    "E4": REPORTS / "side-switch-feature-development-e4-directional-q1-v1.json",
}
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "audit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "model": "c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3",
    "evaluation": "e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b",
    "importance": "b2c501c61e9f7b3aeb2bbb04cf73f3f9793e1831053a7762c08993924daea14d",
    "gapFeatures": "0f89dbbf7d7cda896100e3f0730a17ecbaba5dfd1cd0525993199a5246c5a777",
    "visualFeatures": "6ce23b43018d04045ba783510ad86ef3c6d8767fdc435c7684481fca89ba4871",
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
        not np.isclose(
            actual_thresholds[key], expected_thresholds[key], atol=1e-12, rtol=0.0
        )
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


def _selected_ids(result: Mapping[str, Any]) -> set[str]:
    return {
        str(row["eventId"])
        for fold in result["outerFolds"]
        for row in fold["heldCandidateScores"]
        if bool(row["selected"])
    }


def _recording_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for recording_id, before in baseline["primary"]["byRecording"].items():
        after = candidate["primary"]["byRecording"][recording_id]
        result[recording_id] = {
            name: int(after[name]) - int(before[name])
            for name in (
                "proposals",
                "truePositives",
                "falsePositives",
                "falseNegatives",
            )
        }
    return result


def _standalone_checks(delta: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "primaryF1GainAtLeast2pp": delta["primaryF1"] >= 0.02 - 1e-12,
        "precisionDeclineNoMoreThan2pp": delta["primaryPrecision"] >= -0.02 - 1e-12,
        "recallDeclineNoMoreThan2pp": delta["primaryRecall"] >= -0.02 - 1e-12,
        "strictF1DeclineNoMoreThan1pp": delta["strictF1"] >= -0.01 - 1e-12,
    }


def _evaluate_e1_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    importance: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    high_player_change_ids = {
        str(row["candidateId"])
        for row in importance["errorAnalysis"]["falsePositiveProposals"]
        if any(
            str(item["feature"]) == "playerGlobalAppearanceChange"
            for item in row["topPositiveContributors"]
        )
    }
    baseline_selected = _selected_ids(baseline)
    candidate_selected = _selected_ids(candidate)
    baseline_slice = len(high_player_change_ids & baseline_selected)
    candidate_slice = len(high_player_change_ids & candidate_selected)
    checks = {
        **_standalone_checks(delta),
        "highPlayerChangeFalsePositiveSliceReduced": candidate_slice < baseline_slice,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlice": {
            "definition": (
                "baseline false proposals where playerGlobalAppearanceChange was "
                "among the five strongest positive logit contributors"
            ),
            "frozenCandidateIds": sorted(high_player_change_ids),
            "baselineSelected": baseline_slice,
            "candidateSelected": candidate_slice,
            "change": candidate_slice - baseline_slice,
        },
        "byRecordingDelta": _recording_deltas(baseline, candidate),
    }


def _evaluate_e2_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    importance: Mapping[str, Any],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    gap_supported_ids = {
        str(row["candidateId"])
        for row in importance["errorAnalysis"]["falsePositiveProposals"]
        if any(
            str(item["feature"])
            in {
                "productionGapMeanDeadStateScore",
                "productionGapPeakDeadStateScore",
                "productionGapDurationSeconds",
            }
            for item in row["topPositiveContributors"]
        )
    }
    baseline_selected = _selected_ids(baseline)
    candidate_selected = _selected_ids(candidate)
    baseline_slice = len(gap_supported_ids & baseline_selected)
    candidate_slice = len(gap_supported_ids & candidate_selected)
    checks = {
        **_standalone_checks(delta),
        "gapSupportedFalsePositiveSliceReduced": candidate_slice < baseline_slice,
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlice": {
            "definition": (
                "baseline false proposals where a current dead-state mean, peak, or "
                "gap-duration input was among the five strongest positive contributors"
            ),
            "frozenCandidateIds": sorted(gap_supported_ids),
            "baselineSelected": baseline_slice,
            "candidateSelected": candidate_slice,
            "change": candidate_slice - baseline_slice,
        },
        "byRecordingDelta": _recording_deltas(baseline, candidate),
    }


def _evaluate_e4_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    delta = metric_delta(baseline, candidate)
    baseline_false_selected = {
        str(row["eventId"])
        for fold in baseline["outerFolds"]
        for row in fold["heldCandidateScores"]
        if bool(row["selected"]) and int(row["label"]) == 0
    }
    post_observation_collapse = {
        str(row["eventId"])
        for row in rows
        if (
            float(row["features"]["afterProposalCoverage"])
            < float(row["features"]["beforeProposalCoverage"])
            or float(row["features"]["afterProposalCount"])
            < float(row["features"]["beforeProposalCount"])
        )
    }
    frozen_ids = baseline_false_selected & post_observation_collapse
    candidate_selected = _selected_ids(candidate)
    checks = {
        **_standalone_checks(delta),
        "postObservationCollapseFalsePositiveSliceReduced": len(
            frozen_ids & candidate_selected
        )
        < len(frozen_ids),
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "delta": delta,
        "targetSlice": {
            "definition": (
                "baseline false proposals where after-window proposal coverage or "
                "proposal count is lower than the before window"
            ),
            "frozenCandidateIds": sorted(frozen_ids),
            "baselineSelected": len(frozen_ids),
            "candidateSelected": len(frozen_ids & candidate_selected),
            "change": len(frozen_ids & candidate_selected) - len(frozen_ids),
        },
        "byRecordingDelta": _recording_deltas(baseline, candidate),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "audit": args.audit.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
    }
    if args.experiment in {"E1", "E2"}:
        paths["importance"] = args.importance.expanduser().resolve()
    if args.experiment == "E2":
        paths["gapFeatures"] = args.gap_features.expanduser().resolve()
    if args.experiment == "E4":
        paths["visualFeatures"] = args.visual_features.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else DEFAULT_OUTPUTS[args.experiment].resolve()
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature-development artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected_hashes = {
        name: EXPECTED_SHA256[name]
        for name in paths
    }
    if args.enforce_source_hash and hashes != expected_hashes:
        raise ValueError(f"feature-development source identity changed: {hashes}")

    feature_payload = _load(paths["features"])
    experiment_rows = feature_payload["rows"]
    if args.experiment == "E2":
        gap_payload = _load(paths["gapFeatures"])
        if gap_payload["scope"] != feature_payload["scope"] or [
            str(row["eventId"]) for row in gap_payload["rows"]
        ] != [str(row["eventId"]) for row in feature_payload["rows"]]:
            raise ValueError("E2 gap feature artifact does not match the frozen row universe")
        experiment_rows = gap_payload["rows"]
    elif args.experiment == "E4":
        visual_payload = _load(paths["visualFeatures"])
        if visual_payload["scope"] != feature_payload["scope"] or [
            str(row["eventId"]) for row in visual_payload["rows"]
        ] != [str(row["eventId"]) for row in feature_payload["rows"]]:
            raise ValueError(
                "E4 visual-summary artifact does not match the frozen row universe"
            )
        if visual_payload["parity"]["status"] != "exact-within-tolerance" or float(
            visual_payload["parity"]["maximumAbsoluteDifference"]
        ) > 1e-8:
            raise ValueError("E4 visual-summary artifact did not pass current parity")
        experiment_rows = visual_payload["rows"]
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

    baseline = evaluate_profile(experiment_rows, markers, BASELINE_PROFILE)
    parity = _validate_e0(baseline, current_model, current_evaluation)
    profiles: dict[str, Any] = {BASELINE_PROFILE.identifier: baseline}
    comparison: dict[str, Any] | None = None
    if args.experiment == "E1":
        interaction = evaluate_profile(
            feature_payload["rows"], markers, INTERACTION_PROFILE
        )
        profiles[INTERACTION_PROFILE.identifier] = interaction
        comparison = _evaluate_e1_gate(
            baseline, interaction, _load(paths["importance"])
        )
    elif args.experiment == "E2":
        gap_shape = evaluate_profile(experiment_rows, markers, GAP_SHAPE_PROFILE)
        profiles[GAP_SHAPE_PROFILE.identifier] = gap_shape
        comparison = _evaluate_e2_gate(
            baseline, gap_shape, _load(paths["importance"])
        )
    elif args.experiment == "E4":
        directional = evaluate_profile(experiment_rows, markers, Q1_PROFILE)
        profiles[Q1_PROFILE.identifier] = directional
        comparison = _evaluate_e4_gate(baseline, directional, experiment_rows)
    script_path = Path(__file__).resolve()
    module_path = (script_path.parent.parent / "analysis/side_switch_feature_development.py").resolve()
    created_at = datetime.now(UTC).isoformat()
    payload = {
        "schemaVersion": 1,
        "kind": f"volleycut-side-switch-feature-development-{args.experiment.lower()}-v1",
        "createdAt": created_at,
        "status": "opened-development-only",
        "experiment": {
            "id": args.experiment,
            "name": {
                "E0": "freeze-and-reproduce-control",
                "E1": "swap-specific-interactions",
                "E2": "production-gap-consensus-and-shape",
                "E4": "directional-observation-quality-replacement",
            }[args.experiment],
            "decision": "baseline-only" if comparison is None else comparison["decision"],
        },
        "scope": {
            **feature_payload["scope"],
            "humanMarkers": sum(len(value) for value in markers.values()),
            "positiveCandidateLabels": baseline["positiveCandidateLabels"],
        },
        "protocol": {
            "outer": "leave one recording out",
            "inner": "grouped leave-one-recording-out threshold selection on outer-fit recordings",
            "classifier": "fixed square-root logistic, L2 0.1, top-2/2x recording-balanced hard negatives",
            "decoder": DECODER.to_dict(),
            "primaryPaddingSeconds": 4.0,
        },
        "profiles": profiles,
        "parity": parity,
        "comparison": comparison,
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
            (
                "E0 validates experiment machinery; it is not a new model result."
                if args.experiment == "E0"
                else (
                    f"{args.experiment} formulae and gates were frozen in the checked-in "
                    "development plan before this run."
                )
            ),
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
    parser.add_argument("--importance", type=Path, default=DEFAULT_IMPORTANCE)
    parser.add_argument("--gap-features", type=Path, default=DEFAULT_GAP_FEATURES)
    parser.add_argument("--visual-features", type=Path, default=DEFAULT_VISUAL_FEATURES)
    parser.add_argument("--experiment", choices=tuple(DEFAULT_OUTPUTS), default="E0")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = run(_parser().parse_args())
    metrics = {
        name: concise_metrics(result)
        for name, result in payload["profiles"].items()
    }
    print(
        json.dumps(
            {
                "experiment": payload["experiment"],
                "parity": payload["parity"],
                "metrics": metrics,
                "comparison": payload["comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
