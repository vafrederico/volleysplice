"""Rare-event scoring and decoding utilities for the full side-switch union."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v6 import V6Model, labels_for, matrix_for


DERIVED_FEATURE_NAMES = (
    "candidateIsInternalDeadStatePeak",
    "candidateGeneratorScore",
)
_EPSILON = 1e-9


@dataclass(frozen=True)
class UnionDecoderSettings:
    minimum_index_separation: int = 0
    minimum_time_separation_seconds: float = 0.0
    free_predictions_per_recording: int = 6
    count_penalty_logit: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_index_separation < 0:
            raise ValueError("minimum index separation cannot be negative")
        if (
            not math.isfinite(self.minimum_time_separation_seconds)
            or self.minimum_time_separation_seconds < 0
        ):
            raise ValueError("minimum time separation must be finite and nonnegative")
        if self.free_predictions_per_recording < 1:
            raise ValueError("soft count needs at least one free prediction")
        if not math.isfinite(self.count_penalty_logit) or self.count_penalty_logit < 0:
            raise ValueError("count penalty must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimumIndexSeparation": self.minimum_index_separation,
            "minimumTimeSeparationSeconds": self.minimum_time_separation_seconds,
            "freePredictionsPerRecording": self.free_predictions_per_recording,
            "countPenaltyLogit": self.count_penalty_logit,
            "usesCadence": False,
            "reanchorOnSelection": False,
            "hardMaximumPredictions": None,
        }


def add_derived_features(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    features = dict(row.get("features", {}))
    internal = str(row.get("kind")) == "internal-dead-state-peak"
    raw_score = row.get("score")
    score = float(raw_score) if raw_score is not None else 0.0
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("candidate generator score must stay in [0, 1]")
    features[DERIVED_FEATURE_NAMES[0]] = float(internal)
    features[DERIVED_FEATURE_NAMES[1]] = score
    result["features"] = features
    return result


def _logit(value: float) -> float:
    clipped = min(1.0 - _EPSILON, max(_EPSILON, value))
    return math.log(clipped / (1.0 - clipped))


def decode_ranked_candidates(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    settings: UnionDecoderSettings,
) -> np.ndarray:
    """Decode score-ranked local peaks with an optional soft count penalty."""

    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        scores.shape != (len(rows),)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("candidate probabilities must be a finite aligned vector")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must stay in [0, 1]")
    predictions = np.zeros(len(rows), dtype=bool)
    threshold_logit = _logit(threshold)
    recording_ids = sorted({str(row.get("recordingId", "")) for row in rows})
    if not recording_ids or "" in recording_ids:
        raise ValueError("candidate rows need recording IDs")
    for recording_id in recording_ids:
        indexes = [
            index
            for index, row in enumerate(rows)
            if str(row["recordingId"]) == recording_id
        ]
        chronological = sorted(
            indexes,
            key=lambda index: (
                float(rows[index]["transitionTime"]),
                str(rows[index]["eventId"]),
            ),
        )
        ordinal = {index: order for order, index in enumerate(chronological)}
        ranked = sorted(
            indexes,
            key=lambda index: (
                -float(scores[index]),
                float(rows[index]["transitionTime"]),
                str(rows[index]["eventId"]),
            ),
        )
        selected: list[int] = []
        for index in ranked:
            timestamp = float(rows[index]["transitionTime"])
            if any(
                (
                    settings.minimum_index_separation > 0
                    and abs(ordinal[index] - ordinal[other])
                    < settings.minimum_index_separation
                )
                or (
                    settings.minimum_time_separation_seconds > 0
                    and abs(timestamp - float(rows[other]["transitionTime"]))
                    < settings.minimum_time_separation_seconds
                )
                for other in selected
            ):
                continue
            excess = max(
                0,
                len(selected) + 1 - settings.free_predictions_per_recording,
            )
            adjusted_margin = (
                _logit(float(scores[index]))
                - threshold_logit
                - settings.count_penalty_logit * excess
            )
            if adjusted_margin >= -1e-12:
                selected.append(index)
                predictions[index] = True
    return predictions


def predict_classifier(
    classifier: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    names = tuple(str(value) for value in classifier["featureNames"])
    matrix = np.asarray(
        [
            [float(row.get("features", {}).get(name, math.nan)) for name in names]
            for row in rows
        ],
        dtype=np.float64,
    )
    impute = np.asarray(classifier["impute"], dtype=np.float64)
    mean = np.asarray(classifier["mean"], dtype=np.float64)
    scale = np.asarray(classifier["scale"], dtype=np.float64)
    weights = np.asarray(classifier["weights"], dtype=np.float64)
    expected = (len(names),)
    if any(value.shape != expected for value in (impute, mean, scale, weights)):
        raise ValueError("classifier parameter shape drifted")
    filled = np.where(np.isfinite(matrix), matrix, impute)
    logits = np.clip(
        ((filled - mean) / scale) @ weights + float(classifier["bias"]),
        -30.0,
        30.0,
    )
    return 1.0 / (1.0 + np.exp(-logits))


def calibrate_recording_scores(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    method: str,
) -> np.ndarray:
    """Apply label-free within-recording score normalization."""

    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        scores.shape != (len(rows),)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("calibration scores must be finite aligned probabilities")
    if method not in {"raw", "robust-logit", "percentile"}:
        raise ValueError(f"unsupported recording calibration: {method}")
    if method == "raw":
        return scores.copy()
    result = np.full(len(rows), np.nan, dtype=np.float64)
    for recording_id in sorted({str(row.get("recordingId", "")) for row in rows}):
        if not recording_id:
            raise ValueError("calibration rows need recording IDs")
        indexes = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["recordingId"]) == recording_id
            ],
            dtype=np.int64,
        )
        values = scores[indexes]
        if method == "robust-logit":
            logits = np.asarray([_logit(float(value)) for value in values])
            median = float(np.median(logits))
            mad = 1.4826 * float(np.median(np.abs(logits - median)))
            scale = max(mad, 0.25 * float(np.std(logits)), 1e-6)
            normalized = np.clip((logits - median) / scale, -30.0, 30.0)
            result[indexes] = 1.0 / (1.0 + np.exp(-normalized))
        else:
            if len(values) == 1:
                result[indexes] = 0.5
                continue
            for local_index, value in enumerate(values):
                lower = int(np.sum(values < value))
                upper = int(np.sum(values <= value)) - 1
                result[indexes[local_index]] = 0.5 * (lower + upper) / (
                    len(values) - 1
                )
    if not np.isfinite(result).all():
        raise ValueError("recording calibration left scores unnormalized")
    return result


def penalize_internal_candidates(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    penalty_logit: float,
) -> np.ndarray:
    """Apply a soft logit penalty to internal peaks without making a hard gate."""

    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        scores.shape != (len(rows),)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("internal penalty scores must be finite aligned probabilities")
    if not math.isfinite(penalty_logit) or penalty_logit < 0:
        raise ValueError("internal candidate penalty must be finite and nonnegative")
    logits = np.asarray([_logit(float(value)) for value in scores])
    internal = np.asarray(
        [str(row.get("kind")) == "internal-dead-state-peak" for row in rows]
    )
    adjusted = np.clip(logits - penalty_logit * internal, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-adjusted))


def fit_weighted_logistic(
    events: Sequence[V3Event],
    l2: float,
    feature_names: Sequence[str],
    class_balance_exponent: float,
    extra_sample_weights: np.ndarray | None = None,
) -> V6Model:
    """Fit a linear head from natural to fully class-balanced weighting.

    Exponent zero gives ordinary empirical-prevalence logistic loss; exponent one
    gives the existing equal-total-class loss; 0.5 is the square-root compromise.
    """

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
        raise ValueError("weighted logistic fit needs both classes")
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

    dimensions = matrix.shape[1]
    weights = np.zeros(dimensions)
    bias = 0.0
    design = np.column_stack((matrix, np.ones(len(matrix))))
    regularizer = np.diag(np.r_[np.full(dimensions, l2), 0.0])
    for _ in range(100):
        logits = np.clip(matrix @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = (probabilities - labels) * sample_weights
        gradient = design.T @ error / np.sum(sample_weights)
        gradient[:-1] += l2 * weights
        curvature = probabilities * (1.0 - probabilities) * sample_weights
        hessian = (design.T * curvature) @ design / np.sum(sample_weights)
        hessian += regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        current_logits = matrix @ weights + bias
        current_loss = float(
            np.sum(
                (np.logaddexp(0.0, current_logits) - labels * current_logits)
                * sample_weights
            )
            / np.sum(sample_weights)
            + 0.5 * l2 * (weights @ weights)
        )
        step_scale = 1.0
        while step_scale > 1e-5:
            candidate_weights = weights - step_scale * step[:-1]
            candidate_bias = bias - step_scale * float(step[-1])
            candidate_logits = matrix @ candidate_weights + candidate_bias
            candidate_loss = float(
                np.sum(
                    (
                        np.logaddexp(0.0, candidate_logits)
                        - labels * candidate_logits
                    )
                    * sample_weights
                )
                / np.sum(sample_weights)
                + 0.5 * l2 * (candidate_weights @ candidate_weights)
            )
            if candidate_loss <= current_loss + 1e-12:
                break
            step_scale *= 0.5
        weights = candidate_weights
        bias = candidate_bias
        if float(np.max(np.abs(step_scale * step))) < 1e-8:
            break
    return V6Model(names, impute, mean, scale, weights, bias, 0.5, l2)


def fit_within_recording_pairwise_logistic(
    events: Sequence[V3Event],
    l2: float,
    feature_names: Sequence[str],
    class_balance_exponent: float,
    pairwise_strength: float,
    pairwise_margin: float = 0.0,
    extra_sample_weights: np.ndarray | None = None,
) -> V6Model:
    """Fit pointwise logistic loss plus a recording-balanced pairwise rank loss.

    Every positive/negative pair is formed only within its recording. Each recording
    contributes the same total pair weight, so longer games do not dominate the AUC
    surrogate. A zero pairwise strength delegates to the ordinary fitter exactly.
    """

    if not math.isfinite(pairwise_strength) or pairwise_strength < 0:
        raise ValueError("pairwise strength must be finite and nonnegative")
    if not math.isfinite(pairwise_margin) or pairwise_margin < 0:
        raise ValueError("pairwise margin must be finite and nonnegative")
    if pairwise_strength == 0:
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
        raise ValueError("pairwise logistic fit needs both classes")
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

    pair_differences: list[np.ndarray] = []
    pair_weights: list[float] = []
    recording_ids = sorted({event.recording_id for event in events})
    eligible_recordings = []
    indexes_by_recording: dict[str, tuple[list[int], list[int]]] = {}
    for recording_id in recording_ids:
        positive_indexes = [
            index
            for index, event in enumerate(events)
            if event.recording_id == recording_id and event.label == 1
        ]
        negative_indexes = [
            index
            for index, event in enumerate(events)
            if event.recording_id == recording_id and event.label == 0
        ]
        if positive_indexes and negative_indexes:
            eligible_recordings.append(recording_id)
            indexes_by_recording[recording_id] = (positive_indexes, negative_indexes)
    if not eligible_recordings:
        raise ValueError("pairwise logistic fit needs an in-recording positive/negative pair")
    recording_weight = 1.0 / len(eligible_recordings)
    for recording_id in eligible_recordings:
        positive_indexes, negative_indexes = indexes_by_recording[recording_id]
        local_weight = recording_weight / (
            len(positive_indexes) * len(negative_indexes)
        )
        for positive_index in positive_indexes:
            for negative_index in negative_indexes:
                pair_differences.append(matrix[positive_index] - matrix[negative_index])
                pair_weights.append(local_weight)
    pair_matrix = np.asarray(pair_differences, dtype=np.float64)
    pair_weight_array = np.asarray(pair_weights, dtype=np.float64)

    dimensions = matrix.shape[1]
    weights = np.zeros(dimensions)
    bias = 0.0
    design = np.column_stack((matrix, np.ones(len(matrix))))
    regularizer = np.diag(np.r_[np.full(dimensions, l2), 0.0])

    def objective(candidate_weights: np.ndarray, candidate_bias: float) -> float:
        logits = matrix @ candidate_weights + candidate_bias
        pointwise = float(
            np.sum(
                (np.logaddexp(0.0, logits) - labels * logits) * sample_weights
            )
            / np.sum(sample_weights)
        )
        pair_arguments = pairwise_margin - pair_matrix @ candidate_weights
        pairwise = float(
            np.sum(np.logaddexp(0.0, pair_arguments) * pair_weight_array)
            / np.sum(pair_weight_array)
        )
        return (
            pointwise
            + pairwise_strength * pairwise
            + 0.5 * l2 * float(candidate_weights @ candidate_weights)
        )

    for _ in range(100):
        logits = np.clip(matrix @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = (probabilities - labels) * sample_weights
        gradient = design.T @ error / np.sum(sample_weights)
        curvature = probabilities * (1.0 - probabilities) * sample_weights
        hessian = (design.T * curvature) @ design / np.sum(sample_weights)

        pair_arguments = np.clip(pairwise_margin - pair_matrix @ weights, -30.0, 30.0)
        pair_probabilities = 1.0 / (1.0 + np.exp(-pair_arguments))
        pair_scale = pair_weight_array / np.sum(pair_weight_array)
        gradient[:-1] -= pairwise_strength * (
            pair_matrix.T @ (pair_probabilities * pair_scale)
        )
        pair_curvature = pair_probabilities * (1.0 - pair_probabilities) * pair_scale
        hessian[:-1, :-1] += pairwise_strength * (
            (pair_matrix.T * pair_curvature) @ pair_matrix
        )
        gradient[:-1] += l2 * weights
        hessian += regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        current_loss = objective(weights, bias)
        step_scale = 1.0
        while step_scale > 1e-5:
            candidate_weights = weights - step_scale * step[:-1]
            candidate_bias = bias - step_scale * float(step[-1])
            if objective(candidate_weights, candidate_bias) <= current_loss + 1e-12:
                break
            step_scale *= 0.5
        weights = candidate_weights
        bias = candidate_bias
        if float(np.max(np.abs(step_scale * step))) < 1e-8:
            break
    return V6Model(names, impute, mean, scale, weights, bias, 0.5, l2)
