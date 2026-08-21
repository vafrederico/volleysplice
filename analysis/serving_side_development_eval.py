"""Shared development-only evaluation helpers for serving-side experiments."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .serving_side_specialist import binary_metrics, select_threshold
from .serving_side_v2 import fit_logistic


def labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def matrix(
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


def configuration_matrix(
    rows: Sequence[Mapping[str, Any]],
    configuration: str,
    names: Sequence[str],
) -> np.ndarray:
    values = []
    for row in rows:
        configurations = row.get("configurations")
        features = (
            configurations.get(configuration)
            if isinstance(configurations, Mapping)
            else None
        )
        if not isinstance(features, Mapping):
            raise ValueError(f"row {row.get('rallyId')} lacks {configuration}")
        values.append([float(features[name]) for name in names])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{configuration} contains non-finite features")
    return result


def tied_recording_ranks(
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


def group_metrics(
    rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, key: str
) -> dict[str, Any]:
    truth = labels(rows)
    result = {}
    for group in sorted({str(row[key]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row[key] == group]
        )
        result[group] = binary_metrics(truth[indices], predicted[indices])
    return result


def cross_fit(
    rows: Sequence[Mapping[str, Any]], values: np.ndarray, l2: float
) -> tuple[dict[str, Any], np.ndarray]:
    groups = sorted({str(row["sourceGroup"]) for row in rows})
    if len(groups) < 3:
        raise ValueError("development evaluation requires at least three source groups")
    truth = labels(rows)
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
    by_source_group = group_metrics(rows, predicted, "sourceGroup")
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
            "byEnvironment": group_metrics(rows, predicted, "environment"),
            "folds": folds,
            "predictionDigest": hashlib.sha256(probabilities.tobytes()).hexdigest(),
        },
        probabilities,
    )


def candidate_rank(
    candidate: Mapping[str, Any],
) -> tuple[float, float, float, float, int]:
    evaluation = candidate["evaluation"]
    pooled = evaluation["pooledMetrics"]
    return (
        float(evaluation["sourceGroupMacroBalancedAccuracy"]),
        float(pooled["balancedAccuracy"] or 0.0),
        float(pooled["macroF1"] or 0.0),
        float(evaluation["worstSourceGroupBalancedAccuracy"]),
        -int(candidate["featureCount"]),
    )


def prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    *,
    detailed: bool,
) -> list[dict[str, Any]]:
    truth = labels(rows)
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
