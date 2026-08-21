"""Probability calibration helpers for serving-side development predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


_EPSILON = 1e-9


def logit(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("probabilities must be a non-empty finite vector")
    clipped = np.clip(values, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class PlattCalibrator:
    mean: float
    scale: float
    coefficient: float
    bias: float
    l2: float

    def predict(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        values = (logit(probabilities) - self.mean) / self.scale
        logits = np.clip(values * self.coefficient + self.bias, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "scale": self.scale,
            "coefficient": self.coefficient,
            "bias": self.bias,
            "l2": self.l2,
        }


def fit_platt(
    probabilities: Sequence[float] | np.ndarray,
    truth: Sequence[int] | np.ndarray,
    *,
    l2: float,
) -> PlattCalibrator:
    """Fit unweighted Platt scaling without changing the observed class prior."""
    scores = logit(probabilities)
    labels = np.asarray(truth, dtype=np.float64)
    if (
        labels.shape != scores.shape
        or not np.isin(labels, (0.0, 1.0)).all()
        or not np.any(labels == 0)
        or not np.any(labels == 1)
        or not math.isfinite(l2)
        or l2 <= 0
    ):
        raise ValueError("Platt fitting requires aligned binary data and positive L2")
    mean = float(np.mean(scores))
    scale = float(np.std(scores))
    if scale < 1e-6:
        scale = 1.0
    values = (scores - mean) / scale
    design = np.column_stack((values, np.ones(len(values))))
    parameters = np.asarray([1.0, 0.0], dtype=np.float64)
    regularizer = np.diag([l2, 0.0])

    def loss(candidate: np.ndarray) -> float:
        logits = values * candidate[0] + candidate[1]
        return float(np.mean(np.logaddexp(0.0, logits) - labels * logits)) + (
            0.5 * l2 * float(candidate[0] ** 2)
        )

    for _ in range(100):
        logits = np.clip(design @ parameters, -30.0, 30.0)
        predicted = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (predicted - labels) / len(labels)
        gradient[0] += l2 * parameters[0]
        curvature = predicted * (1.0 - predicted)
        hessian = (design.T * curvature) @ design / len(labels) + regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        current = loss(parameters)
        step_scale = 1.0
        candidate = parameters - step
        while step_scale > 1e-6:
            candidate = parameters - step_scale * step
            if loss(candidate) <= current + 1e-12:
                break
            step_scale *= 0.5
        parameters = candidate
        if float(np.max(np.abs(step_scale * step))) < 1e-9:
            break
    return PlattCalibrator(
        mean=mean,
        scale=scale,
        coefficient=float(parameters[0]),
        bias=float(parameters[1]),
        l2=float(l2),
    )


def calibration_metrics(
    truth: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    labels = np.asarray(truth, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or labels.shape != scores.shape
        or not len(labels)
        or not np.isin(labels, (0.0, 1.0)).all()
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or bins < 2
    ):
        raise ValueError("invalid calibration metrics inputs")
    clipped = np.clip(scores, _EPSILON, 1.0 - _EPSILON)
    bin_indices = np.minimum((scores * bins).astype(np.int64), bins - 1)
    reliability = []
    calibration_error = 0.0
    for index in range(bins):
        selected = bin_indices == index
        if not np.any(selected):
            continue
        count = int(np.sum(selected))
        average_probability = float(np.mean(scores[selected]))
        observed_near_rate = float(np.mean(labels[selected]))
        calibration_error += count / len(labels) * abs(
            average_probability - observed_near_rate
        )
        reliability.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "rows": count,
                "averageProbability": average_probability,
                "observedNearRate": observed_near_rate,
            }
        )
    return {
        "rows": len(labels),
        "brier": float(np.mean((scores - labels) ** 2)),
        "logLoss": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
        "expectedCalibrationError": calibration_error,
        "bins": reliability,
    }
