"""Weighted logistic fitting and binary metrics for reviewed serving-side slices."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .serving_side_v2 import LogisticModel


def _validate(
    values: np.ndarray,
    labels: np.ndarray,
    observation_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64)
    if (
        matrix.ndim != 2
        or len(matrix) != len(truth)
        or truth.shape != weights.shape
        or not len(truth)
        or not np.isin(truth, (0.0, 1.0)).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.any(truth == 0)
        or not np.any(truth == 1)
    ):
        raise ValueError("weighted logistic data must be aligned with both classes")
    return matrix, truth, weights


def fit_weighted_logistic(
    values: np.ndarray,
    labels: np.ndarray,
    observation_weights: np.ndarray,
    *,
    l2: float,
) -> LogisticModel:
    if l2 <= 0:
        raise ValueError("weighted logistic L2 must be positive")
    matrix, truth, weights = _validate(values, labels, observation_weights)
    impute = np.asarray(
        [
            float(np.median(column[np.isfinite(column)]))
            if np.any(np.isfinite(column))
            else 0.0
            for column in matrix.T
        ]
    )
    filled = np.where(np.isfinite(matrix), matrix, impute)
    normalized_weights = weights / np.sum(weights)
    mean = np.sum(filled * normalized_weights[:, None], axis=0)
    scale = np.sqrt(
        np.sum(((filled - mean) ** 2) * normalized_weights[:, None], axis=0)
    )
    scale[scale < 1e-6] = 1.0
    normalized = (filled - mean) / scale
    positive_weight = float(np.sum(weights[truth == 1]))
    negative_weight = float(np.sum(weights[truth == 0]))
    total_weight = positive_weight + negative_weight
    class_weights = np.where(
        truth == 1,
        total_weight / (2.0 * positive_weight),
        total_weight / (2.0 * negative_weight),
    )
    fitting_weights = weights * class_weights
    dimensions = normalized.shape[1]
    coefficients = np.zeros(dimensions, dtype=np.float64)
    bias = 0.0
    design = np.column_stack((normalized, np.ones(len(normalized))))
    regularizer = np.diag(np.r_[np.full(dimensions, l2), 0.0])

    def loss(candidate_weights: np.ndarray, candidate_bias: float) -> float:
        logits = normalized @ candidate_weights + candidate_bias
        losses = np.logaddexp(0.0, logits) - truth * logits
        return float(np.sum(losses * fitting_weights) / np.sum(fitting_weights)) + (
            0.5 * l2 * float(candidate_weights @ candidate_weights)
        )

    for _ in range(100):
        logits = np.clip(normalized @ coefficients + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        errors = (probabilities - truth) * fitting_weights
        gradient = design.T @ errors / np.sum(fitting_weights)
        gradient[:-1] += l2 * coefficients
        curvature = probabilities * (1.0 - probabilities) * fitting_weights
        hessian = (design.T * curvature) @ design / np.sum(fitting_weights)
        hessian += regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        current_loss = loss(coefficients, bias)
        step_scale = 1.0
        while step_scale > 1e-5:
            candidate_coefficients = coefficients - step_scale * step[:-1]
            candidate_bias = bias - step_scale * float(step[-1])
            if loss(candidate_coefficients, candidate_bias) <= current_loss + 1e-12:
                break
            step_scale *= 0.5
        coefficients = candidate_coefficients
        bias = candidate_bias
        if float(np.max(np.abs(step_scale * step))) < 1e-8:
            break
    return LogisticModel(impute, mean, scale, coefficients, bias, l2)


def weighted_binary_metrics(
    labels: np.ndarray,
    predicted: np.ndarray,
    observation_weights: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    guesses = np.asarray(predicted, dtype=bool)
    weights = np.asarray(observation_weights, dtype=np.float64)
    if (
        truth.shape != guesses.shape
        or truth.shape != weights.shape
        or not np.isin(truth, (0, 1)).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("weighted metrics require aligned finite binary rows")
    true_positive = float(np.sum(weights[(truth == 1) & guesses]))
    false_negative = float(np.sum(weights[(truth == 1) & ~guesses]))
    false_positive = float(np.sum(weights[(truth == 0) & guesses]))
    true_negative = float(np.sum(weights[(truth == 0) & ~guesses]))

    def divide(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    positive_precision = divide(true_positive, true_positive + false_positive)
    positive_recall = divide(true_positive, true_positive + false_negative)
    negative_precision = divide(true_negative, true_negative + false_negative)
    negative_recall = divide(true_negative, true_negative + false_positive)

    def f1(precision: float | None, recall: float | None) -> float | None:
        if precision is None or recall is None:
            return None
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    balanced = (
        (positive_recall + negative_recall) / 2
        if positive_recall is not None and negative_recall is not None
        else None
    )
    return {
        "observedRows": len(truth),
        "estimatedPopulationRows": float(np.sum(weights)),
        "confusion": {
            "positive": {"positive": true_positive, "negative": false_negative},
            "negative": {"positive": false_positive, "negative": true_negative},
        },
        "accuracy": divide(true_positive + true_negative, float(np.sum(weights))),
        "balancedAccuracy": balanced,
        "positivePrecision": positive_precision,
        "positiveRecall": positive_recall,
        "positiveF1": f1(positive_precision, positive_recall),
        "negativePrecision": negative_precision,
        "negativeRecall": negative_recall,
        "negativeF1": f1(negative_precision, negative_recall),
    }


def select_weighted_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    observation_weights: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64)
    if (
        truth.shape != scores.shape
        or truth.shape != weights.shape
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("weighted threshold inputs must be aligned probabilities")
    unique = np.unique(scores)
    thresholds = np.unique(
        np.r_[0.0, (unique[:-1] + unique[1:]) / 2.0, 1.0]
    )
    candidates = []
    for threshold in thresholds:
        metrics = weighted_binary_metrics(truth, scores >= threshold, weights)
        candidates.append(
            {
                "threshold": float(threshold),
                "distanceFromHalf": abs(float(threshold) - 0.5),
                **metrics,
            }
        )
    return max(
        candidates,
        key=lambda item: (
            float(item["balancedAccuracy"] or 0.0),
            float(item["accuracy"] or 0.0),
            -float(item["distanceFromHalf"]),
            -float(item["threshold"]),
        ),
    )


def weighted_brier_score(
    labels: np.ndarray,
    probabilities: np.ndarray,
    observation_weights: np.ndarray,
) -> float:
    truth = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64)
    if truth.shape != scores.shape or truth.shape != weights.shape:
        raise ValueError("weighted Brier inputs must be aligned")
    value = float(np.sum(weights * (scores - truth) ** 2) / np.sum(weights))
    if not math.isfinite(value):
        raise ValueError("weighted Brier score is not finite")
    return value
