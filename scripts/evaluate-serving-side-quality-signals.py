#!/usr/bin/env python3
"""Evaluate inference-time visibility and contact-quality signals."""

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
    configuration_matrix,
    matrix,
    tied_recording_ranks,
)
from analysis.serving_side_weighted_logistic import (
    fit_weighted_logistic,
    select_weighted_threshold,
    weighted_binary_metrics,
    weighted_brier_score,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_CENTER = ROOT / "features/serving-side-flight-v3/development.json"
DEFAULT_TRAJECTORY = ROOT / "features/serving-side-trajectory-v1/development.json"
DEFAULT_PATCHES = ROOT / "features/serving-side-selective-highres-v2/development.json"
DEFAULT_PATCH_EVALUATION = (
    ROOT
    / "reports/serving-side/serving-side-selective-highres-v2-ablation-development.json"
)
DEFAULT_REVIEW_SLICES = (
    ROOT / "reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json"
)
DEFAULT_OUTPUT = ROOT / "reports/serving-side/serving-side-quality-v1-development.json"
FLIGHT_CONFIGURATION = "192x108-r4c6"
REFERENCE_PATCH_CONFIGURATION = "patch-20pct-96"
L2_GRID = (0.1, 1.0, 10.0, 100.0)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_development_eval.py",
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


def _target_value(target: str, row: Mapping[str, Any]) -> int | None:
    annotation = row.get("annotation")
    annotation = annotation if isinstance(annotation, Mapping) else {}
    if target == "server-visible":
        value = annotation.get("serverVisibility")
        if value == "visible":
            return 1
        if value in ("partial", "offscreen"):
            return 0
        return None
    if target == "contact-on-anchor":
        value = annotation.get("contactTiming")
        if value == "on-anchor":
            return 1
        if value in ("before-anchor", "after-anchor"):
            return 0
        return None
    raise ValueError(f"unknown quality target: {target}")


def _group_metrics(
    review_rows: Sequence[Mapping[str, Any]],
    truth: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for group in sorted({str(row["sourceGroup"]) for row in review_rows}):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(review_rows)
                if row["sourceGroup"] == group
            ]
        )
        result[group] = weighted_binary_metrics(
            truth[indices], predicted[indices], weights[indices]
        )
    return result


def _cross_fit(
    review_rows: Sequence[Mapping[str, Any]],
    values: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> tuple[dict[str, Any], np.ndarray]:
    groups = sorted({str(row["sourceGroup"]) for row in review_rows})
    probabilities = np.full(len(review_rows), np.nan, dtype=np.float64)
    folds = []
    for group in groups:
        held = np.asarray(
            [
                index
                for index, row in enumerate(review_rows)
                if row["sourceGroup"] == group
            ]
        )
        train = np.asarray(
            [
                index
                for index, row in enumerate(review_rows)
                if row["sourceGroup"] != group
            ]
        )
        model = fit_weighted_logistic(
            values[train], truth[train], weights[train], l2=l2
        )
        probabilities[held] = model.predict_proba(values[held])
        folds.append(
            {
                "heldSourceGroup": group,
                "trainRows": len(train),
                "heldRows": len(held),
            }
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("quality cross-fit left reviewed rows unscored")
    threshold = select_weighted_threshold(truth, probabilities, weights)
    predicted = probabilities >= float(threshold["threshold"])
    by_group = _group_metrics(review_rows, truth, predicted, weights)
    group_balanced = [
        float(metrics["balancedAccuracy"])
        for metrics in by_group.values()
        if metrics["balancedAccuracy"] is not None
    ]
    return (
        {
            "thresholdSelection": threshold,
            "pooledMetrics": weighted_binary_metrics(truth, predicted, weights),
            "sourceGroupMacroBalancedAccuracy": float(np.mean(group_balanced)),
            "worstSupportedSourceGroupBalancedAccuracy": float(
                np.min(group_balanced)
            ),
            "supportedSourceGroups": len(group_balanced),
            "bySourceGroup": by_group,
            "weightedBrierScore": weighted_brier_score(
                truth, probabilities, weights
            ),
            "folds": folds,
            "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
        },
        probabilities,
    )


def _rank(candidate: Mapping[str, Any]) -> tuple[float, float, float, int]:
    evaluation = candidate["evaluation"]
    return (
        float(evaluation["sourceGroupMacroBalancedAccuracy"]),
        float(evaluation["pooledMetrics"]["balancedAccuracy"] or 0.0),
        -float(evaluation["weightedBrierScore"]),
        -int(candidate["featureCount"]),
    )


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    paths = {
        "center": args.center.resolve(),
        "trajectory": args.trajectory.resolve(),
        "patches": args.patches.resolve(),
        "patchEvaluation": args.patch_evaluation.resolve(),
        "reviewSlices": args.review_slices.resolve(),
    }
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite quality evaluation: {output_path}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    center = _load(paths["center"])
    trajectory = _load(paths["trajectory"])
    patches = _load(paths["patches"])
    patch_evaluation = _load(paths["patchEvaluation"])
    review = _load(paths["reviewSlices"])
    if (
        center.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or trajectory.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or patches.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or patch_evaluation.get("selection", {}).get("protectedTest")
        != "not loaded, scored, or used"
    ):
        raise ValueError("quality selection requires development-only parent artifacts")
    raw_center_rows = center.get("rows")
    raw_trajectory_rows = trajectory.get("rows")
    raw_patch_rows = patches.get("rows")
    raw_review_rows = review.get("reviewedPredictions")
    if not all(
        isinstance(rows, list) and rows
        for rows in (
            raw_center_rows,
            raw_trajectory_rows,
            raw_patch_rows,
            raw_review_rows,
        )
    ):
        raise ValueError("quality parent rows are unavailable")
    center_rows = [row for row in raw_center_rows if isinstance(row, Mapping)]
    trajectory_by_id = {
        str(row["rallyId"]): row
        for row in raw_trajectory_rows
        if isinstance(row, Mapping)
    }
    patch_by_id = {
        str(row["rallyId"]): row for row in raw_patch_rows if isinstance(row, Mapping)
    }
    if len(center_rows) != len(trajectory_by_id) or len(center_rows) != len(patch_by_id):
        raise ValueError("quality parent artifacts differ in row count")
    selected_patch = patch_evaluation.get("selection", {}).get(
        "bestPatchCandidate", {}
    ).get("configuration")
    if not patch_evaluation.get("selection", {}).get("eligibleForNextStage"):
        selected_patch = REFERENCE_PATCH_CONFIGURATION
    if not isinstance(selected_patch, str):
        raise ValueError("quality evaluation cannot resolve its patch configuration")
    rows = []
    for center_row in center_rows:
        rally_id = str(center_row["rallyId"])
        trajectory_row = trajectory_by_id.get(rally_id)
        patch_row = patch_by_id.get(rally_id)
        if trajectory_row is None or patch_row is None:
            raise ValueError(f"quality feature banks lack {rally_id}")
        if any(
            center_row[field] != trajectory_row[field]
            or center_row[field] != patch_row[field]
            for field in ("recordingId", "sourceGroup", "environment", "label")
        ):
            raise ValueError(f"quality feature identities differ for {rally_id}")
        rows.append(
            {
                **center_row,
                "trajectoryFeatures": trajectory_row["trajectoryFeatures"],
                "patchFeatures": patch_row["configurations"][selected_patch],
            }
        )
    row_index = {str(row["rallyId"]): index for index, row in enumerate(rows)}

    v2_names = [str(name) for name in center["v2FeatureNames"]]
    flight_configuration = next(
        item
        for item in center["configurations"]
        if item["name"] == FLIGHT_CONFIGURATION
    )
    flight_names = [str(name) for name in flight_configuration["featureNames"]]
    trajectory_names = [str(name) for name in trajectory["featureNames"]]
    patch_configuration = next(
        item for item in patches["configurations"] if item["name"] == selected_patch
    )
    patch_names = [str(name) for name in patch_configuration["featureNames"]]
    v2 = matrix(rows, "v2RecordingRankFeatures", v2_names)
    flight_rank = tied_recording_ranks(
        rows, configuration_matrix(rows, FLIGHT_CONFIGURATION, flight_names)
    )
    trajectory_rank = tied_recording_ranks(
        rows, matrix(rows, "trajectoryFeatures", trajectory_names)
    )
    patch_absolute = matrix(rows, "patchFeatures", patch_names)
    families = {
        "v2-rank": (v2, [f"v2:{name}" for name in v2_names]),
        "flight-rank": (
            flight_rank,
            [f"flight:{name}" for name in flight_names],
        ),
        "trajectory-rank": (
            trajectory_rank,
            [f"trajectory:{name}" for name in trajectory_names],
        ),
        "patch-absolute": (
            patch_absolute,
            [f"patch:{name}" for name in patch_names],
        ),
        "trajectory-plus-patch": (
            np.column_stack((trajectory_rank, patch_absolute)),
            [
                *(f"trajectory:{name}" for name in trajectory_names),
                *(f"patch:{name}" for name in patch_names),
            ],
        ),
        "v2-plus-trajectory-plus-patch": (
            np.column_stack((v2, trajectory_rank, patch_absolute)),
            [
                *(f"v2:{name}" for name in v2_names),
                *(f"trajectory:{name}" for name in trajectory_names),
                *(f"patch:{name}" for name in patch_names),
            ],
        ),
    }

    output_targets = {}
    for target in ("server-visible", "contact-on-anchor"):
        target_review = [
            row
            for row in raw_review_rows
            if isinstance(row, Mapping) and _target_value(target, row) is not None
        ]
        indices = np.asarray([row_index[str(row["rallyId"])] for row in target_review])
        truth = np.asarray(
            [_target_value(target, row) for row in target_review], dtype=np.int64
        )
        weights = np.asarray([float(row["weight"]) for row in target_review])
        leaderboard = []
        probabilities_by_key: dict[tuple[str, float], np.ndarray] = {}
        for family, (all_values, names) in families.items():
            for l2 in L2_GRID:
                print(f"evaluating {target} / {family} / l2={l2}", flush=True)
                audit, probabilities = _cross_fit(
                    target_review, all_values[indices], truth, weights, l2
                )
                leaderboard.append(
                    {
                        "featureFamily": family,
                        "featureCount": len(names),
                        "l2": l2,
                        "evaluation": audit,
                    }
                )
                probabilities_by_key[(family, l2)] = probabilities
        leaderboard.sort(key=_rank, reverse=True)
        selected = leaderboard[0]
        selected_key = (str(selected["featureFamily"]), float(selected["l2"]))
        selected_values, selected_names = families[selected_key[0]]
        threshold = float(
            selected["evaluation"]["thresholdSelection"]["threshold"]
        )
        model = replace(
            fit_weighted_logistic(
                selected_values[indices], truth, weights, l2=selected_key[1]
            ),
            threshold=threshold,
        )
        parameters = model.to_dict()
        reviewed_probabilities = probabilities_by_key[selected_key]
        all_probabilities = model.predict_proba(selected_values)
        output_targets[target] = {
            "labelContract": (
                "positive=visible; negative=partial-or-offscreen; unclear excluded"
                if target == "server-visible"
                else "positive=on-anchor; negative=before-or-after-anchor; unclear excluded"
            ),
            "counts": {
                "reviewedRows": len(target_review),
                "positiveObserved": int(np.sum(truth == 1)),
                "negativeObserved": int(np.sum(truth == 0)),
                "positiveWeighted": float(np.sum(weights[truth == 1])),
                "negativeWeighted": float(np.sum(weights[truth == 0])),
            },
            "selectedCandidate": selected,
            "leaderboard": leaderboard,
            "finalModel": {
                "fingerprint": hashlib.sha256(
                    json.dumps(
                        parameters, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "featureFamily": selected_key[0],
                "featureNames": selected_names,
                "parameters": parameters,
                "trainingRows": len(target_review),
            },
            "reviewedCrossFitPredictions": [
                {
                    "rallyId": row["rallyId"],
                    "sourceGroup": row["sourceGroup"],
                    "truth": int(label),
                    "weight": float(weight),
                    "probabilityPositive": float(probability),
                    "prediction": bool(probability >= threshold),
                }
                for row, label, weight, probability in zip(
                    target_review,
                    truth,
                    weights,
                    reviewed_probabilities,
                    strict=True,
                )
            ],
            "allRowInference": [
                {
                    "rallyId": row["rallyId"],
                    "recordingId": row["recordingId"],
                    "sourceGroup": row["sourceGroup"],
                    "probabilityPositive": float(probability),
                }
                for row, probability in zip(rows, all_probabilities, strict=True)
            ],
        }
    if any(_sha256(path) != hashes[name] for name, path in paths.items()):
        raise RuntimeError("a frozen quality source changed during evaluation")
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-quality-signal-development-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "selection": {
            "primaryMetric": "weighted source-group macro balanced accuracy",
            "tieBreaks": [
                "weighted pooled balanced accuracy",
                "lower weighted Brier score",
                "fewer features",
            ],
            "crossValidation": "leave-one-source-group-out reviewed rows",
            "sampling": "frozen error census plus weighted stratified controls",
            "protectedTest": "not loaded, scored, or used",
            "humanLabelsAtInference": False,
            "patchConfiguration": selected_patch,
        },
        "targets": output_targets,
        "limitations": {
            "offscreenFarObserved": 0,
            "visibilityUse": (
                "quality gating only; insufficient support for a trusted two-sided "
                "offscreen serving-side specialist"
            ),
        },
        "sources": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        }
        | {
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in IMPLEMENTATION_PATHS
            ]
        },
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for target, result in output_targets.items():
        selected = result["selectedCandidate"]
        print(
            f"selected {target}: {selected['featureFamily']} / l2={selected['l2']}"
        )
    print(f"wrote {output_path}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--center", type=Path, default=DEFAULT_CENTER)
    result.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    result.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    result.add_argument(
        "--patch-evaluation", type=Path, default=DEFAULT_PATCH_EVALUATION
    )
    result.add_argument("--review-slices", type=Path, default=DEFAULT_REVIEW_SLICES)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
