#!/usr/bin/env python3
"""Evaluate source-group-balanced fitting and conservative probability blends."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.serving_side_development_eval import (
    candidate_rank,
    configuration_matrix,
    group_metrics,
    labels,
    matrix,
    prediction_rows,
    tied_recording_ranks,
)
from analysis.serving_side_review_slices import rescore_reviewed_predictions
from analysis.serving_side_specialist import binary_metrics, select_threshold
from analysis.serving_side_v2 import LogisticModel, fit_logistic
from analysis.serving_side_weighted_logistic import fit_weighted_logistic


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_CENTER = ROOT / "features/serving-side-flight-v3/development.json"
DEFAULT_REVIEW_SLICES = (
    ROOT / "reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/serving-side/serving-side-group-balance-v1-development.json"
)
FLIGHT_CONFIGURATION = "192x108-r4c6"
BASELINE_L2 = 0.1
GROUP_L2_GRID = (0.01, 0.1, 1.0, 10.0)
BLEND_ALPHAS = (0.25, 0.50, 0.75, 1.00)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_development_eval.py",
    REPOSITORY_ROOT / "analysis/serving_side_review_slices.py",
    REPOSITORY_ROOT / "analysis/serving_side_weighted_logistic.py",
    Path(__file__).resolve(),
)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_group_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts: dict[str, int] = {}
    for row in rows:
        group = str(row["sourceGroup"])
        counts[group] = counts.get(group, 0) + 1
    result = np.asarray(
        [1.0 / counts[str(row["sourceGroup"])] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise AssertionError("source-group weights are invalid")
    return result


def _audit(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray
) -> dict[str, Any]:
    truth = labels(rows)
    threshold_selection = select_threshold(truth, probabilities)
    threshold = float(threshold_selection["threshold"])
    predicted = probabilities >= threshold
    by_source_group = group_metrics(rows, predicted, "sourceGroup")
    balanced = [
        float(metrics["balancedAccuracy"])
        for metrics in by_source_group.values()
        if metrics["balancedAccuracy"] is not None
    ]
    return {
        "thresholdSelection": threshold_selection,
        "pooledMetrics": binary_metrics(truth, predicted),
        "sourceGroupMacroBalancedAccuracy": float(np.mean(balanced)),
        "worstSourceGroupBalancedAccuracy": float(np.min(balanced)),
        "bySourceGroup": by_source_group,
        "byEnvironment": group_metrics(rows, predicted, "environment"),
        "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
    }


def _paired(
    truth: np.ndarray, candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, int]:
    return {
        "bothCorrect": int(np.sum((candidate == truth) & (baseline == truth))),
        "candidateOnlyCorrect": int(
            np.sum((candidate == truth) & (baseline != truth))
        ),
        "baselineOnlyCorrect": int(
            np.sum((candidate != truth) & (baseline == truth))
        ),
        "bothWrong": int(np.sum((candidate != truth) & (baseline != truth))),
    }


def _model_dict(model: LogisticModel, names: Sequence[str]) -> dict[str, Any]:
    parameters = model.to_dict()
    return {
        "fingerprint": hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "featureNames": list(names),
        "parameters": parameters,
    }


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    center_path = args.center.resolve()
    review_path = args.review_slices.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite group balance report: {output_path}")
    center_hash = _sha256(center_path)
    review_hash = _sha256(review_path)
    center = _load(center_path)
    review = _load(review_path)
    if (
        center.get("kind") != "volleycut-serving-side-flight-feature-development-v1"
        or center.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or center.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("group balance requires the complete development center bank")
    raw_rows = center.get("rows")
    reviewed_rows = review.get("reviewedPredictions")
    if not isinstance(raw_rows, list) or not raw_rows or not isinstance(
        reviewed_rows, list
    ):
        raise ValueError("group balance parent rows are unavailable")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows) or any(row.get("sourceSplit") == "test" for row in rows):
        raise ValueError("group balance rows are invalid or protected")
    v2_names = [str(name) for name in center["v2FeatureNames"]]
    flight_configuration = next(
        item
        for item in center["configurations"]
        if item["name"] == FLIGHT_CONFIGURATION
    )
    flight_names = [str(name) for name in flight_configuration["featureNames"]]
    v2 = matrix(rows, "v2RecordingRankFeatures", v2_names)
    flight_rank = tied_recording_ranks(
        rows, configuration_matrix(rows, FLIGHT_CONFIGURATION, flight_names)
    )
    values = np.column_stack((v2, flight_rank))
    names = [
        *(f"v2:{name}" for name in v2_names),
        *(f"flight:{name}" for name in flight_names),
    ]
    truth = labels(rows)
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    baseline_probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    group_probabilities = {
        l2: np.full(len(rows), np.nan, dtype=np.float64) for l2 in GROUP_L2_GRID
    }
    fold_audits = []
    for held_group in groups:
        held = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] == held_group]
        )
        train_indices = np.asarray(
            [index for index, row in enumerate(rows) if row["sourceGroup"] != held_group]
        )
        train_rows = [rows[index] for index in train_indices]
        baseline_model = fit_logistic(
            values[train_indices], truth[train_indices], l2=BASELINE_L2
        )
        baseline_probabilities[held] = baseline_model.predict_proba(values[held])
        observation_weights = _source_group_weights(train_rows)
        for l2 in GROUP_L2_GRID:
            model = fit_weighted_logistic(
                values[train_indices],
                truth[train_indices],
                observation_weights,
                l2=l2,
            )
            group_probabilities[l2][held] = model.predict_proba(values[held])
        fold_audits.append(
            {
                "heldSourceGroup": held_group,
                "trainRows": len(train_indices),
                "heldRows": len(held),
                "trainingSourceGroups": len(groups) - 1,
                "weightPerTrainingSourceGroup": 1.0,
            }
        )
    if not np.isfinite(baseline_probabilities).all() or not all(
        np.isfinite(probabilities).all()
        for probabilities in group_probabilities.values()
    ):
        raise AssertionError("group balance cross-fit left rows unscored")

    candidate_probabilities = {"fixed-flight-baseline": baseline_probabilities}
    for l2, probabilities in group_probabilities.items():
        candidate_probabilities[f"source-group-balanced-l2-{l2:g}"] = probabilities
        for alpha in BLEND_ALPHAS[:-1]:
            candidate_probabilities[
                f"baseline-group-balanced-l2-{l2:g}-blend-{alpha:.2f}"
            ] = (1.0 - alpha) * baseline_probabilities + alpha * probabilities
    leaderboard = []
    for name, probabilities in candidate_probabilities.items():
        l2 = (
            BASELINE_L2
            if name == "fixed-flight-baseline"
            else float(name.split("l2-", 1)[1].split("-", 1)[0])
        )
        leaderboard.append(
            {
                "featureFamily": name,
                "featureCount": len(names),
                "l2": l2,
                "evaluation": _audit(rows, probabilities),
            }
        )
    leaderboard.sort(key=candidate_rank, reverse=True)
    selected = leaderboard[0]
    baseline = next(
        candidate
        for candidate in leaderboard
        if candidate["featureFamily"] == "fixed-flight-baseline"
    )
    selected_name = str(selected["featureFamily"])
    baseline_threshold = float(
        baseline["evaluation"]["thresholdSelection"]["threshold"]
    )
    baseline_choices = baseline_probabilities >= baseline_threshold
    predictions_by_candidate = {}
    slices_by_candidate = {}
    paired_by_candidate = {}
    for candidate in leaderboard:
        name = str(candidate["featureFamily"])
        threshold = float(
            candidate["evaluation"]["thresholdSelection"]["threshold"]
        )
        predictions = prediction_rows(
            rows, candidate_probabilities[name], threshold, detailed=True
        )
        rescored = rescore_reviewed_predictions(reviewed_rows, predictions)
        predictions_by_candidate[name] = predictions
        slices_by_candidate[name] = {
            "overall": rescored["overall"],
            "slices": rescored["slices"],
        }
        paired_by_candidate[name] = _paired(
            truth,
            candidate_probabilities[name] >= threshold,
            baseline_choices,
        )

    full_group_weights = _source_group_weights(rows)
    selected_threshold = float(
        selected["evaluation"]["thresholdSelection"]["threshold"]
    )
    final_bundle: dict[str, Any] = {
        "featureFamily": selected_name,
        "threshold": selected_threshold,
        "trainingRows": len(rows),
        "sourceGroupWeighting": "equal total weight per source group",
    }
    if selected_name == "fixed-flight-baseline":
        model = replace(
            fit_logistic(values, truth, l2=BASELINE_L2),
            threshold=selected_threshold,
        )
        final_bundle["model"] = _model_dict(model, names)
    else:
        selected_l2 = float(selected["l2"])
        group_model = fit_weighted_logistic(
            values, truth, full_group_weights, l2=selected_l2
        )
        if "blend" in selected_name:
            alpha = float(selected_name.rsplit("-", 1)[1])
            baseline_model = fit_logistic(values, truth, l2=BASELINE_L2)
            final_bundle.update(
                {
                    "blendAlpha": alpha,
                    "baselineModel": _model_dict(baseline_model, names),
                    "groupBalancedModel": _model_dict(group_model, names),
                }
            )
        else:
            final_bundle["model"] = _model_dict(
                replace(group_model, threshold=selected_threshold), names
            )
    selected_macro = float(
        selected["evaluation"]["sourceGroupMacroBalancedAccuracy"]
    )
    baseline_macro = float(
        baseline["evaluation"]["sourceGroupMacroBalancedAccuracy"]
    )
    selected_worst = float(
        selected["evaluation"]["worstSourceGroupBalancedAccuracy"]
    )
    baseline_worst = float(
        baseline["evaluation"]["worstSourceGroupBalancedAccuracy"]
    )
    eligible = bool(
        selected_name != "fixed-flight-baseline"
        and selected_macro > baseline_macro
        and selected_worst >= baseline_worst
    )
    if _sha256(center_path) != center_hash or _sha256(review_path) != review_hash:
        raise RuntimeError("a frozen group-balance source changed during evaluation")
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-group-balance-development-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "predeclaration": {
            "baselineL2": BASELINE_L2,
            "groupBalancedL2Grid": list(GROUP_L2_GRID),
            "blendAlphas": list(BLEND_ALPHAS),
            "groupWeighting": "each training source group has total weight 1",
        },
        "selection": {
            "primaryMetric": "mean source-group balanced accuracy",
            "tieBreaks": [
                "pooled balanced accuracy",
                "pooled macro-F1",
                "worst source-group balanced accuracy",
                "fewer features",
            ],
            "crossValidation": "leave-one-source-group-out development predictions",
            "protectedTest": "not loaded, scored, or used",
            "selectedCandidate": selected,
            "baselineCandidate": baseline,
            "eligibleForNextStage": eligible,
        },
        "leaderboard": leaderboard,
        "pairedAgainstBaselineByCandidate": paired_by_candidate,
        "reviewedSliceEstimatesByCandidate": slices_by_candidate,
        "predictionsByCandidate": predictions_by_candidate,
        "foldAudits": fold_audits,
        "finalModelBundle": final_bundle,
        "counts": center.get("counts"),
        "sources": {
            "centerDevelopmentDataset": {
                "path": str(center_path),
                "sha256": center_hash,
            },
            "reviewedSlices": {"path": str(review_path), "sha256": review_hash},
            "humanLabelCorrections": center.get("sources", {}).get(
                "humanLabelCorrections"
            ),
            "sourceQualityExclusions": center.get("sources", {}).get(
                "sourceQualityExclusions"
            ),
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ],
        },
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"selected {selected_name}; eligible={eligible}")
    print(f"wrote {output_path}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--center", type=Path, default=DEFAULT_CENTER)
    result.add_argument("--review-slices", type=Path, default=DEFAULT_REVIEW_SLICES)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
