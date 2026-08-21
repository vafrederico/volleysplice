"""Selective near/far decisions for the serving-side specialist."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def selective_metrics(
    truth: Sequence[int] | np.ndarray,
    near_probabilities: Sequence[float] | np.ndarray,
    *,
    far_threshold: float,
    near_threshold: float,
) -> dict[str, Any]:
    """Measure precision, all-row recall, and coverage with an abstention band."""
    labels = np.asarray(truth, dtype=np.int64)
    scores = np.asarray(near_probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or scores.ndim != 1
        or len(labels) != len(scores)
        or not len(labels)
        or not np.isfinite(scores).all()
        or not set(labels.tolist()).issubset({0, 1})
        or not math.isfinite(far_threshold)
        or not math.isfinite(near_threshold)
        or not 0 <= far_threshold <= near_threshold <= 1
    ):
        raise ValueError("invalid selective serving-side inputs")

    predicted_far = scores < far_threshold
    predicted_near = scores >= near_threshold
    abstained = ~(predicted_far | predicted_near)
    actual_near = labels == 1
    actual_far = labels == 0
    near_near = int(np.sum(predicted_near & actual_near))
    far_near = int(np.sum(predicted_near & actual_far))
    near_far = int(np.sum(predicted_far & actual_near))
    far_far = int(np.sum(predicted_far & actual_far))
    abstained_near = int(np.sum(abstained & actual_near))
    abstained_far = int(np.sum(abstained & actual_far))
    emitted = near_near + far_near + near_far + far_far
    near_total = int(np.sum(actual_near))
    far_total = int(np.sum(actual_far))
    near_recall = near_near / near_total if near_total else 0.0
    far_recall = far_far / far_total if far_total else 0.0
    return {
        "rows": len(labels),
        "near": near_total,
        "far": far_total,
        "farThreshold": float(far_threshold),
        "nearThreshold": float(near_threshold),
        "predictedNear": near_near + far_near,
        "predictedFar": near_far + far_far,
        "abstained": abstained_near + abstained_far,
        "coverage": emitted / len(labels),
        "confusion": {
            "near": {
                "near": near_near,
                "far": near_far,
                "abstain": abstained_near,
            },
            "far": {
                "near": far_near,
                "far": far_far,
                "abstain": abstained_far,
            },
        },
        "nearPrecision": _ratio(near_near, near_near + far_near),
        "nearRecall": near_recall,
        "farPrecision": _ratio(far_far, far_far + near_far),
        "farRecall": far_recall,
        "balancedRecall": (near_recall + far_recall) / 2,
        "selectiveAccuracy": _ratio(near_near + far_far, emitted),
    }


def select_precision_operating_point(
    truth: Sequence[int] | np.ndarray,
    near_probabilities: Sequence[float] | np.ndarray,
    *,
    center_threshold: float,
    precision_floor: float,
    minimum_predictions_per_side: int,
) -> dict[str, Any] | None:
    """Choose maximum coverage while only abstaining around the frozen cutoff."""
    labels = np.asarray(truth, dtype=np.int64)
    scores = np.asarray(near_probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or scores.ndim != 1
        or len(labels) != len(scores)
        or not len(labels)
        or not 0 < precision_floor <= 1
        or not 0 <= center_threshold <= 1
        or minimum_predictions_per_side < 1
    ):
        raise ValueError("invalid precision operating-point inputs")

    ordered = np.argsort(scores, kind="stable")
    sorted_scores = scores[ordered]
    sorted_labels = labels[ordered]
    cumulative_near = np.r_[0, np.cumsum(sorted_labels)]
    total_near = int(cumulative_near[-1])
    total_far = len(labels) - total_near
    lower_candidates = sorted(
        {0.0, float(center_threshold), *(float(score) for score in sorted_scores if score <= center_threshold)}
    )
    upper_candidates = sorted(
        {float(center_threshold), 1.0, *(float(score) for score in sorted_scores if score >= center_threshold)}
    )
    best: tuple[tuple[float, ...], float, float] | None = None
    for lower in lower_candidates:
        far_end = int(np.searchsorted(sorted_scores, lower, side="left"))
        near_in_far = int(cumulative_near[far_end])
        far_far = far_end - near_in_far
        predicted_far = far_end
        far_precision = _ratio(far_far, predicted_far)
        if (
            predicted_far < minimum_predictions_per_side
            or far_precision is None
            or far_precision < precision_floor
        ):
            continue
        for upper in upper_candidates:
            near_start = int(np.searchsorted(sorted_scores, upper, side="left"))
            near_near = total_near - int(cumulative_near[near_start])
            far_in_near = total_far - (near_start - int(cumulative_near[near_start]))
            predicted_near = len(labels) - near_start
            near_precision = _ratio(near_near, predicted_near)
            if (
                predicted_near < minimum_predictions_per_side
                or near_precision is None
                or near_precision < precision_floor
            ):
                continue
            emitted = predicted_far + predicted_near
            near_recall = near_near / total_near if total_near else 0.0
            far_recall = far_far / total_far if total_far else 0.0
            rank = (
                emitted / len(labels),
                min(near_recall, far_recall),
                (near_recall + far_recall) / 2,
                (near_near + far_far) / emitted,
                -(upper - lower),
            )
            if best is None or rank > best[0]:
                best = (rank, lower, upper)
    if best is None:
        return None
    return selective_metrics(
        labels,
        scores,
        far_threshold=best[1],
        near_threshold=best[2],
    )
