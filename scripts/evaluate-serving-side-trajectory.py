#!/usr/bin/env python3
"""Evaluate explicit residual-component trajectory features on development data."""

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
from analysis.serving_side_review_slices import rescore_reviewed_predictions
from analysis.serving_side_specialist import binary_metrics, select_threshold
from analysis.serving_side_v2 import fit_logistic


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_DATASET = ROOT / "features/serving-side-trajectory-v1/development.json"
DEFAULT_REVIEW_SLICES = (
    ROOT / "reports/serving-side/serving-side-flight-v2-reviewed-slices-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/serving-side/serving-side-trajectory-v1-development.json"
)
CONFIGURATION = "192x108-r4c6"
L2_GRID = (0.01, 0.1, 1.0, 10.0)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    REPOSITORY_ROOT / "analysis/serving_side_review_slices.py",
    REPOSITORY_ROOT / "analysis/serving_side_trajectory.py",
    REPOSITORY_ROOT / "analysis/serving_side_v2.py",
    REPOSITORY_ROOT / "scripts/extract-serving-side-trajectory-features.py",
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


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def _matrix(
    rows: Sequence[Mapping[str, Any]], field: str, names: Sequence[str]
) -> np.ndarray:
    values = []
    for row in rows:
        features = row.get(field)
        if not isinstance(features, Mapping):
            raise ValueError(f"row {row.get('rallyId')} lacks {field}")
        values.append([float(features[name]) for name in names])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} contains non-finite features")
    return result


def _flight_matrix(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> np.ndarray:
    values = []
    for row in rows:
        configurations = row.get("configurations")
        features = (
            configurations.get(CONFIGURATION)
            if isinstance(configurations, Mapping)
            else None
        )
        if not isinstance(features, Mapping):
            raise ValueError(f"row {row.get('rallyId')} lacks {CONFIGURATION}")
        values.append([float(features[name]) for name in names])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("fixed-flight features contain non-finite values")
    return result


def _tied_recording_ranks(
    rows: Sequence[Mapping[str, Any]], values: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["recordingId"] == recording_id
            ]
        )
        count = len(indices)
        local = values[indices]
        for feature in range(values.shape[1]):
            order = np.argsort(local[:, feature], kind="stable")
            sorted_values = local[order, feature]
            ranks = np.zeros(count, dtype=np.float64)
            start = 0
            while start < count:
                end = start + 1
                while end < count and sorted_values[end] == sorted_values[start]:
                    end += 1
                rank = (
                    0.5
                    if count == 1
                    else ((start + end - 1) / 2.0) / (count - 1)
                )
                ranks[order[start:end]] = rank
                start = end
            result[indices, feature] = ranks
    return result


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, key: str
) -> dict[str, Any]:
    truth = _labels(rows)
    result = {}
    for group in sorted({str(row[key]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row[key] == group]
        )
        result[group] = binary_metrics(truth[indices], predicted[indices])
    return result


def _cross_fit(
    rows: Sequence[Mapping[str, Any]], values: np.ndarray, l2: float
) -> tuple[dict[str, Any], np.ndarray]:
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    if len(groups) < 3:
        raise ValueError("trajectory evaluation requires at least three source groups")
    truth = _labels(rows)
    probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    folds = []
    for group in groups:
        held = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["sourceGroup"] == group
            ]
        )
        train = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["sourceGroup"] != group
            ]
        )
        model = fit_logistic(values[train], truth[train], l2=l2)
        probabilities[held] = model.predict_proba(values[held])
        folds.append(
            {"heldSourceGroup": group, "trainRows": len(train), "heldRows": len(held)}
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("source-group cross-fit left rows unscored")
    threshold_selection = select_threshold(truth, probabilities)
    threshold = float(threshold_selection["threshold"])
    predicted = probabilities >= threshold
    by_source_group = _group_metrics(rows, predicted, "sourceGroup")
    balanced = [
        float(metrics["balancedAccuracy"])
        for metrics in by_source_group.values()
        if metrics["balancedAccuracy"] is not None
    ]
    return (
        {
            "thresholdSelection": threshold_selection,
            "pooledMetrics": binary_metrics(truth, predicted),
            "sourceGroupMacroBalancedAccuracy": float(np.mean(balanced)),
            "worstSourceGroupBalancedAccuracy": float(np.min(balanced)),
            "bySourceGroup": by_source_group,
            "byEnvironment": _group_metrics(rows, predicted, "environment"),
            "folds": folds,
            "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
        },
        probabilities,
    )


def _rank(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    evaluation = candidate["evaluation"]
    pooled = evaluation["pooledMetrics"]
    return (
        float(evaluation["sourceGroupMacroBalancedAccuracy"]),
        float(pooled["balancedAccuracy"] or 0.0),
        float(pooled["macroF1"] or 0.0),
        float(evaluation["worstSourceGroupBalancedAccuracy"]),
        -int(candidate["featureCount"]),
    )


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    *,
    detailed: bool,
) -> list[dict[str, Any]]:
    truth = _labels(rows)
    predicted = probabilities >= threshold
    output = []
    for row, probability, choice, label in zip(
        rows, probabilities, predicted, truth, strict=True
    ):
        item: dict[str, Any] = {
            "rallyId": row["rallyId"],
            "prediction": "near" if choice else "far",
        }
        if detailed:
            item.update(
                {
                    "recordingId": row["recordingId"],
                    "sourceGroup": row["sourceGroup"],
                    "environment": row["environment"],
                    "decision": row["decision"],
                    "probabilityNear": float(probability),
                    "correct": bool(choice == label),
                }
            )
        output.append(item)
    return output


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset_path = args.dataset.resolve()
    review_path = args.review_slices.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite trajectory evaluation: {output_path}")
    dataset_hash = _sha256(dataset_path)
    dataset = _load(dataset_path)
    if (
        dataset.get("kind")
        != "volleycut-serving-side-trajectory-feature-development-v1"
        or dataset.get("scope") != "development"
        or dataset.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or dataset.get("dataPolicy", {}).get("humanCorrectedAnchorsUsed") is not False
        or dataset.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("trajectory selection requires the complete development artifact")
    raw_trajectory_rows = dataset.get("rows")
    if not isinstance(raw_trajectory_rows, list) or not raw_trajectory_rows:
        raise ValueError("trajectory dataset has no rows")
    trajectory_rows = [
        row for row in raw_trajectory_rows if isinstance(row, Mapping)
    ]
    if len(trajectory_rows) != len(raw_trajectory_rows) or any(
        row.get("sourceSplit") == "test" for row in trajectory_rows
    ):
        raise ValueError("trajectory dataset contains invalid or protected rows")

    center_source = dataset.get("sources", {}).get("centerFlightDataset")
    if not isinstance(center_source, Mapping):
        raise ValueError("trajectory dataset lacks its center feature source")
    center_path = Path(str(center_source["path"])).resolve()
    if _sha256(center_path) != center_source.get("sha256"):
        raise ValueError("center feature dataset changed since trajectory extraction")
    center = _load(center_path)
    if (
        center.get("kind") != "volleycut-serving-side-flight-feature-development-v1"
        or center.get("dataPolicy", {}).get("protectedTestIncluded") is not False
        or center.get("dataPolicy", {}).get("limitPerRecording") is not None
    ):
        raise ValueError("trajectory evaluation requires the complete development center bank")
    center_by_id = {
        str(row["rallyId"]): row
        for row in center.get("rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("rallyId"), str)
    }
    if len(center_by_id) != len(trajectory_rows):
        raise ValueError("center and trajectory feature banks differ in row count")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trajectory in trajectory_rows:
        rally_id = str(trajectory["rallyId"])
        if rally_id in seen:
            raise ValueError(f"duplicate trajectory row: {rally_id}")
        seen.add(rally_id)
        center_row = center_by_id.get(rally_id)
        if center_row is None:
            raise ValueError(f"center feature bank lacks {rally_id}")
        if any(
            trajectory[field] != center_row[field]
            for field in (
                "recordingId",
                "sourceGroup",
                "environment",
                "sourceSplit",
                "serveAnchor",
                "decision",
                "label",
            )
        ):
            raise ValueError(f"trajectory and center identities differ for {rally_id}")
        rows.append(
            {**center_row, "trajectoryFeatures": trajectory["trajectoryFeatures"]}
        )

    trajectory_names = dataset.get("featureNames")
    v2_names = center.get("v2FeatureNames")
    configurations = center.get("configurations")
    configuration = next(
        (
            item
            for item in configurations
            if isinstance(item, Mapping) and item.get("name") == CONFIGURATION
        ),
        None,
    ) if isinstance(configurations, list) else None
    flight_names = (
        configuration.get("featureNames")
        if isinstance(configuration, Mapping)
        else None
    )
    if not all(
        isinstance(names, list) and names
        for names in (trajectory_names, v2_names, flight_names)
    ):
        raise ValueError("trajectory evaluation feature contracts are unavailable")
    trajectory_names = [str(name) for name in trajectory_names]
    v2_names = [str(name) for name in v2_names]
    flight_names = [str(name) for name in flight_names]

    v2 = _matrix(rows, "v2RecordingRankFeatures", v2_names)
    flight_raw = _flight_matrix(rows, flight_names)
    trajectory_raw = _matrix(rows, "trajectoryFeatures", trajectory_names)
    flight_rank = _tied_recording_ranks(rows, flight_raw)
    trajectory_rank = _tied_recording_ranks(rows, trajectory_raw)
    v2_labels = [f"v2:{name}" for name in v2_names]
    flight_labels = [f"flight:{name}" for name in flight_names]
    trajectory_labels = [f"trajectory:{name}" for name in trajectory_names]
    families = {
        "v2-plus-fixed-flight-rank": (
            np.column_stack((v2, flight_rank)),
            [*v2_labels, *flight_labels],
        ),
        "trajectory-absolute": (trajectory_raw, trajectory_labels),
        "trajectory-recording-rank": (trajectory_rank, trajectory_labels),
        "v2-plus-trajectory-rank": (
            np.column_stack((v2, trajectory_rank)),
            [*v2_labels, *trajectory_labels],
        ),
        "fixed-flight-plus-trajectory-rank": (
            np.column_stack((flight_rank, trajectory_rank)),
            [*flight_labels, *trajectory_labels],
        ),
        "v2-plus-fixed-flight-plus-trajectory-rank": (
            np.column_stack((v2, flight_rank, trajectory_rank)),
            [*v2_labels, *flight_labels, *trajectory_labels],
        ),
    }

    leaderboard: list[dict[str, Any]] = []
    probabilities_by_key: dict[tuple[str, float], np.ndarray] = {}
    for family, (values, names) in families.items():
        for l2 in L2_GRID:
            print(f"evaluating {family} / l2={l2}", flush=True)
            audit, probabilities = _cross_fit(rows, values, l2)
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
    baseline = max(
        (
            candidate
            for candidate in leaderboard
            if candidate["featureFamily"] == "v2-plus-fixed-flight-rank"
        ),
        key=_rank,
    )
    best_trajectory = max(
        (
            candidate
            for candidate in leaderboard
            if "trajectory" in str(candidate["featureFamily"])
        ),
        key=_rank,
    )
    selected_key = (str(selected["featureFamily"]), float(selected["l2"]))
    baseline_key = (str(baseline["featureFamily"]), float(baseline["l2"]))
    best_trajectory_key = (
        str(best_trajectory["featureFamily"]),
        float(best_trajectory["l2"]),
    )

    selected_probabilities = probabilities_by_key[selected_key]
    selected_threshold = float(
        selected["evaluation"]["thresholdSelection"]["threshold"]
    )
    selected_choices = selected_probabilities >= selected_threshold
    baseline_probabilities = probabilities_by_key[baseline_key]
    baseline_threshold = float(
        baseline["evaluation"]["thresholdSelection"]["threshold"]
    )
    baseline_choices = baseline_probabilities >= baseline_threshold
    truth = _labels(rows)
    paired = {
        "bothCorrect": int(
            np.sum((selected_choices == truth) & (baseline_choices == truth))
        ),
        "selectedOnlyCorrect": int(
            np.sum((selected_choices == truth) & (baseline_choices != truth))
        ),
        "baselineOnlyCorrect": int(
            np.sum((selected_choices != truth) & (baseline_choices == truth))
        ),
        "bothWrong": int(
            np.sum((selected_choices != truth) & (baseline_choices != truth))
        ),
    }

    selected_values, selected_names = families[selected_key[0]]
    final_model = replace(
        fit_logistic(selected_values, truth, l2=selected_key[1]),
        threshold=selected_threshold,
    )
    final_parameters = final_model.to_dict()
    fingerprint = hashlib.sha256(
        json.dumps(final_parameters, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    top_weights = sorted(
        (
            {
                "feature": name,
                "weight": float(weight),
                "absoluteWeight": abs(float(weight)),
            }
            for name, weight in zip(
                selected_names, final_model.weights, strict=True
            )
        ),
        key=lambda item: (-item["absoluteWeight"], item["feature"]),
    )[:30]

    review = _load(review_path)
    reviewed_rows = review.get("reviewedPredictions")
    if not isinstance(reviewed_rows, list):
        raise ValueError("review-slice artifact has no frozen reviewed sample")
    selected_predictions = _prediction_rows(
        rows, selected_probabilities, selected_threshold, detailed=True
    )
    selected_review = rescore_reviewed_predictions(
        reviewed_rows, selected_predictions
    )
    best_trajectory_predictions = _prediction_rows(
        rows,
        probabilities_by_key[best_trajectory_key],
        float(best_trajectory["evaluation"]["thresholdSelection"]["threshold"]),
        detailed=False,
    )
    best_trajectory_review = rescore_reviewed_predictions(
        reviewed_rows, best_trajectory_predictions
    )
    if _sha256(dataset_path) != dataset_hash:
        raise RuntimeError("trajectory feature artifact changed during evaluation")

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-trajectory-development-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "selection": {
            "primaryMetric": "mean source-group balanced accuracy",
            "tieBreaks": [
                "pooled balanced accuracy",
                "pooled macro-F1",
                "worst source-group balanced accuracy",
                "fewer features",
            ],
            "crossValidation": (
                "leave-one-source-group-out over correction-clean development rows"
            ),
            "protectedTest": "not loaded, scored, or used",
            "reviewSlices": "reported after selection; not a selection metric",
            "selectedCandidate": selected,
            "fixedFlightBaselineCandidate": baseline,
            "bestTrajectoryCandidate": best_trajectory,
        },
        "leaderboard": leaderboard,
        "pairedAgainstFixedFlightBaseline": paired,
        "reviewedSliceEstimate": {
            "overall": selected_review["overall"],
            "slices": selected_review["slices"],
        },
        "bestTrajectoryReviewedSliceEstimate": {
            "overall": best_trajectory_review["overall"],
            "slices": best_trajectory_review["slices"],
        },
        "selectedTopWeights": top_weights,
        "finalModel": {
            "fingerprint": fingerprint,
            "featureFamily": selected_key[0],
            "featureNames": selected_names,
            "parameters": final_parameters,
            "trainingRows": len(rows),
        },
        "selectedPredictions": selected_predictions,
        "counts": dataset.get("counts"),
        "sources": {
            "trajectoryDevelopmentDataset": {
                "path": str(dataset_path),
                "sha256": dataset_hash,
            },
            "centerFlightDataset": center_source,
            "reviewedSlices": {
                "path": str(review_path),
                "sha256": _sha256(review_path),
            },
            "humanLabelCorrections": dataset.get("sources", {}).get(
                "humanLabelCorrections"
            ),
            "sourceQualityExclusions": dataset.get("sources", {}).get(
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
    print(f"selected {selected_key[0]} / l2={selected_key[1]}")
    print(
        f"best trajectory {best_trajectory_key[0]} / l2={best_trajectory_key[1]}"
    )
    print(f"wrote {output_path}")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    result.add_argument("--review-slices", type=Path, default=DEFAULT_REVIEW_SLICES)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
