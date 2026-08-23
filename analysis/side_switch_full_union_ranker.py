"""Rare-event scoring and decoding utilities for the full side-switch union."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


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

