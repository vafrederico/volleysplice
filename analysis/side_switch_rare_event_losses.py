"""Focal and effective-number objectives for the linear side-switch head."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from analysis.side_switch_full_union_ranker import fit_weighted_logistic
from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v6 import V6Model, labels_for, matrix_for


def effective_number_class_weights(
    positive_count: int, negative_count: int, beta: float
) -> tuple[float, float]:
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("effective-number weighting needs both classes")
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("effective-number beta must stay in [0, 1)")
    positive = (1.0 - beta) / (1.0 - beta**positive_count)
    negative = (1.0 - beta) / (1.0 - beta**negative_count)
    return positive, negative


def fit_effective_number_logistic(
    events: Sequence[V3Event],
    l2: float,
    feature_names: Sequence[str],
    beta: float,
    extra_sample_weights: np.ndarray | None = None,
) -> V6Model:
    labels = labels_for(events)
    positive, negative = effective_number_class_weights(
        int(np.sum(labels == 1)), int(np.sum(labels == 0)), beta
    )
    weights = np.where(labels == 1, positive, negative)
    if extra_sample_weights is not None:
        extra = np.asarray(extra_sample_weights, dtype=np.float64)
        if (
            extra.shape != (len(events),)
            or not np.isfinite(extra).all()
            or np.any(extra <= 0)
        ):
            raise ValueError("extra sample weights must be finite, positive, and aligned")
        weights *= extra
    return fit_weighted_logistic(events, l2, feature_names, 0.0, weights)


def fit_focal_logistic(
    events: Sequence[V3Event],
    l2: float,
    feature_names: Sequence[str],
    class_balance_exponent: float,
    gamma: float,
    extra_sample_weights: np.ndarray | None = None,
) -> V6Model:
    """Fit exact binary focal loss with deterministic BFGS."""

    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("focal gamma must be finite and nonnegative")
    if gamma == 0:
        return fit_weighted_logistic(
            events,
            l2,
            feature_names,
            class_balance_exponent,
            extra_sample_weights,
        )
    if l2 <= 0 or not math.isfinite(l2):
        raise ValueError("l2 must be finite and positive")
    if not 0 <= class_balance_exponent <= 1 or not math.isfinite(
        class_balance_exponent
    ):
        raise ValueError("class-balance exponent must stay in [0, 1]")

    names = tuple(feature_names)
    values = matrix_for(events, names)
    labels = labels_for(events)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise ValueError("focal fit needs both classes")
    impute = np.asarray(
        [
            float(np.median(column[np.isfinite(column)]))
            if np.any(np.isfinite(column))
            else 0.0
            for column in values.T
        ]
    )
    filled = np.where(np.isfinite(values), values, impute)
    mean = np.mean(filled, axis=0)
    scale = np.std(filled, axis=0)
    scale[scale < 1e-6] = 1.0
    matrix = (filled - mean) / scale
    positive_weight = (len(labels) / (2.0 * positives)) ** class_balance_exponent
    negative_weight = (len(labels) / (2.0 * negatives)) ** class_balance_exponent
    sample_weights = np.where(labels == 1, positive_weight, negative_weight)
    if extra_sample_weights is not None:
        extra = np.asarray(extra_sample_weights, dtype=np.float64)
        if (
            extra.shape != (len(events),)
            or not np.isfinite(extra).all()
            or np.any(extra <= 0)
        ):
            raise ValueError("extra sample weights must be finite, positive, and aligned")
        sample_weights *= extra
    sample_weights /= float(np.mean(sample_weights))
    signed_labels = 2.0 * labels - 1.0

    initial = fit_weighted_logistic(
        events,
        l2,
        names,
        class_balance_exponent,
        extra_sample_weights,
    )
    parameters = np.r_[initial.weights, initial.bias].astype(np.float64)

    def objective_and_gradient(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        weights = candidate[:-1]
        bias = float(candidate[-1])
        logits = np.clip(matrix @ weights + bias, -30.0, 30.0)
        true_probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(signed_labels * logits, -30.0, 30.0))
        )
        true_probabilities = np.clip(true_probabilities, 1e-12, 1.0 - 1e-12)
        modulation = (1.0 - true_probabilities) ** gamma
        losses = -modulation * np.log(true_probabilities)
        objective = float(
            np.sum(losses * sample_weights) / np.sum(sample_weights)
            + 0.5 * l2 * (weights @ weights)
        )
        derivative = signed_labels * modulation * (
            gamma * true_probabilities * np.log(true_probabilities)
            - (1.0 - true_probabilities)
        )
        weighted = derivative * sample_weights / np.sum(sample_weights)
        gradient = np.r_[matrix.T @ weighted + l2 * weights, np.sum(weighted)]
        return objective, gradient

    dimensions = len(parameters)
    inverse_hessian = np.eye(dimensions)
    objective, gradient = objective_and_gradient(parameters)
    for _ in range(200):
        if float(np.max(np.abs(gradient))) < 1e-7:
            break
        direction = -(inverse_hessian @ gradient)
        if float(direction @ gradient) >= -1e-12:
            direction = -gradient
            inverse_hessian = np.eye(dimensions)
        step_scale = 1.0
        directional = float(gradient @ direction)
        while step_scale > 1e-8:
            candidate = parameters + step_scale * direction
            candidate_objective, candidate_gradient = objective_and_gradient(candidate)
            if candidate_objective <= objective + 1e-4 * step_scale * directional:
                break
            step_scale *= 0.5
        if step_scale <= 1e-8:
            break
        step = candidate - parameters
        gradient_change = candidate_gradient - gradient
        curvature = float(step @ gradient_change)
        if curvature > 1e-10:
            rho = 1.0 / curvature
            identity = np.eye(dimensions)
            left = identity - rho * np.outer(step, gradient_change)
            inverse_hessian = (
                left @ inverse_hessian @ left.T + rho * np.outer(step, step)
            )
        else:
            inverse_hessian = np.eye(dimensions)
        parameters = candidate
        objective = candidate_objective
        gradient = candidate_gradient
    return V6Model(
        names,
        impute,
        mean,
        scale,
        parameters[:-1],
        float(parameters[-1]),
        0.5,
        l2,
    )
