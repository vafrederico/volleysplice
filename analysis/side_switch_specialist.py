"""Small, leakage-aware specialist for reviewed side-switch marker features.

The specialist consumes the fixed feature rows produced by
``evaluate-side-switch-appearance.py``.  It intentionally has no dependency on
scikit-learn: the repository's NumPy runtime is enough for deterministic,
class-balanced logistic fitting and immutable JSON artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_KIND = "volleycut-side-switch-specialist-v1"
MODEL_SCHEMA_VERSION = 1
DECISIONS = {"switch": 1, "no-switch": 0}
ELIGIBLE_ENVIRONMENTS = frozenset({"beach", "grass", "unknown"})
EVALUATION_SPLITS = frozenset({"challenge", "non-training", "test"})

APPEARANCE_FEATURES = (
    "fullFrameControl",
    "playerPaletteEqual",
    "playerPaletteArea",
    "playerPaletteAreaPlusGeometry",
    "detectionCountChange",
    "boxAreaChange",
    "medianBoxHeightChange",
)
AGGREGATE_FEATURES = (
    "meanDetectionCount",
    "meanTotalBoxAreaFraction",
    "meanMedianBoxHeightFraction",
    "meanDetectionScore",
    "usableFrameCount",
)
FEATURE_SETS = {
    "palette-area": ("feature:playerPaletteArea",),
    "appearance-bank": tuple(f"feature:{name}" for name in APPEARANCE_FEATURES),
    "appearance-context": (
        *(f"feature:{name}" for name in APPEARANCE_FEATURES),
        *(f"before:{name}" for name in AGGREGATE_FEATURES),
        *(f"after:{name}" for name in AGGREGATE_FEATURES),
        "context:logGapSeconds",
    ),
}


class SideSwitchSpecialistError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReviewedEvent:
    event_id: str
    recording_id: str
    environment: str
    split: str
    source_group: str
    source_type: str
    target_status: str
    decision: str
    label: int
    event: Mapping[str, Any]


@dataclass(frozen=True)
class SpecialistModel:
    feature_set: str
    feature_names: tuple[str, ...]
    impute: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    l2: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise SideSwitchSpecialistError(
                "model expects "
                f"(*, {len(self.feature_names)}) features, got {matrix.shape}"
            )
        filled = np.where(np.isfinite(matrix), matrix, self.impute)
        normalized = (filled - self.mean) / self.scale
        logits = np.clip(normalized @ self.weights + self.bias, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.predict_proba(values) >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "featureSet": self.feature_set,
            "featureNames": list(self.feature_names),
            "impute": self.impute.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpecialistModel":
        feature_set = value.get("featureSet")
        names = value.get("featureNames")
        if feature_set not in FEATURE_SETS or not isinstance(names, list) or not names:
            raise SideSwitchSpecialistError("invalid specialist feature signature")
        expected_names = feature_names(str(feature_set))
        if tuple(names) != expected_names:
            raise SideSwitchSpecialistError(
                "specialist feature names do not match the declared feature set"
            )
        model = cls(
            feature_set=str(feature_set),
            feature_names=tuple(str(name) for name in names),
            impute=np.asarray(value.get("impute"), dtype=np.float64),
            mean=np.asarray(value.get("mean"), dtype=np.float64),
            scale=np.asarray(value.get("scale"), dtype=np.float64),
            weights=np.asarray(value.get("weights"), dtype=np.float64),
            bias=float(value.get("bias")),
            threshold=float(value.get("threshold")),
            l2=float(value.get("l2")),
        )
        expected = (len(model.feature_names),)
        if any(
            array.shape != expected
            for array in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise SideSwitchSpecialistError("specialist parameter shapes do not agree")
        if not all(
            np.isfinite(array).all()
            for array in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise SideSwitchSpecialistError("specialist parameters must be finite")
        if (
            np.any(model.scale <= 0)
            or not math.isfinite(model.bias)
            or not 0 <= model.threshold <= 1
            or model.l2 <= 0
        ):
            raise SideSwitchSpecialistError("specialist scalar parameters are invalid")
        return model


def _record(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return math.nan


def _raw_value(event: Mapping[str, Any], specification: str) -> float:
    group, name = specification.split(":", 1)
    if group == "feature":
        return _finite_number(_record(event.get("features")).get(name))
    if group in {"before", "after"}:
        return _finite_number(_record(event.get(group)).get(name))
    if group == "context" and name == "logGapSeconds":
        gap = _finite_number(event.get("gapSeconds"))
        return math.log1p(max(0.0, gap)) if math.isfinite(gap) else math.nan
    raise SideSwitchSpecialistError(f"unknown feature specification: {specification}")


def feature_names(feature_set: str) -> tuple[str, ...]:
    specifications = FEATURE_SETS.get(feature_set)
    if specifications is None:
        raise SideSwitchSpecialistError(f"unknown feature set: {feature_set}")
    names: list[str] = []
    for specification in specifications:
        names.extend((specification, f"{specification}:missing"))
    return tuple(names)


def event_vector(event: Mapping[str, Any], feature_set: str) -> np.ndarray:
    specifications = FEATURE_SETS.get(feature_set)
    if specifications is None:
        raise SideSwitchSpecialistError(f"unknown feature set: {feature_set}")
    values: list[float] = []
    for specification in specifications:
        value = _raw_value(event, specification)
        values.extend((value, 0.0 if math.isfinite(value) else 1.0))
    return np.asarray(values, dtype=np.float64)


def reviewed_events(
    report: Mapping[str, Any], decisions: Mapping[str, Any]
) -> tuple[list[ReviewedEvent], dict[str, int]]:
    report_kind = report.get("kind")
    report_created_at = report.get("createdAt")
    if (
        decisions.get("reportKind") != report_kind
        or decisions.get("reportCreatedAt") != report_created_at
    ):
        raise SideSwitchSpecialistError(
            "side-switch decisions belong to a different appearance report"
        )
    raw_events = report.get("events")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_events, list) or not isinstance(raw_decisions, Mapping):
        raise SideSwitchSpecialistError(
            "report or decision payload has an invalid schema"
        )
    event_ids = {
        event.get("eventId")
        for event in raw_events
        if isinstance(event, Mapping) and isinstance(event.get("eventId"), str)
    }
    unknown_ids = sorted(set(raw_decisions) - event_ids)
    if unknown_ids:
        raise SideSwitchSpecialistError(
            f"decision file contains {len(unknown_ids)} unknown event IDs"
        )

    rows: list[ReviewedEvent] = []
    counts = {
        "events": len(raw_events),
        "switch": 0,
        "no-switch": 0,
        "unclear": 0,
        "missing": 0,
    }
    for event in raw_events:
        if not isinstance(event, Mapping) or not isinstance(event.get("eventId"), str):
            raise SideSwitchSpecialistError(
                "appearance report contains an invalid event"
            )
        event_id = str(event["eventId"])
        decision = raw_decisions.get(event_id)
        if decision is None:
            counts["missing"] += 1
            continue
        if decision == "unclear":
            counts["unclear"] += 1
            continue
        if decision not in DECISIONS:
            raise SideSwitchSpecialistError(f"invalid review decision for {event_id}")
        counts[str(decision)] += 1
        rows.append(
            ReviewedEvent(
                event_id=event_id,
                recording_id=str(event.get("recordingId", "")),
                environment=str(event.get("environment", "unknown")),
                split=str(event.get("split", "unknown")),
                source_group=str(event.get("sourceGroup", "unknown")),
                source_type=str(event.get("sourceType", "unknown")),
                target_status=str(event.get("targetStatus", "unknown")),
                decision=str(decision),
                label=DECISIONS[str(decision)],
                event=event,
            )
        )
    return rows, counts


def matrix_for(rows: Sequence[ReviewedEvent], feature_set: str) -> np.ndarray:
    if not rows:
        return np.empty((0, len(feature_names(feature_set))), dtype=np.float64)
    return np.stack([event_vector(row.event, feature_set) for row in rows])


def labels_for(rows: Sequence[ReviewedEvent]) -> np.ndarray:
    return np.asarray([row.label for row in rows], dtype=np.float64)


def _fit_preprocessor(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or not len(values):
        raise SideSwitchSpecialistError(
            "training matrix must be non-empty and two-dimensional"
        )
    impute = np.zeros(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        finite = values[np.isfinite(values[:, index]), index]
        impute[index] = float(np.median(finite)) if len(finite) else 0.0
    filled = np.where(np.isfinite(values), values, impute)
    mean = np.mean(filled, axis=0)
    scale = np.std(filled, axis=0)
    scale[scale < 1e-6] = 1.0
    return impute, mean, scale


def _weighted_log_loss(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    weights: np.ndarray,
    bias: float,
    l2: float,
) -> float:
    logits = matrix @ weights + bias
    losses = np.logaddexp(0.0, logits) - labels * logits
    return float(np.sum(losses * sample_weights) / np.sum(sample_weights)) + (
        0.5 * l2 * float(weights @ weights)
    )


def fit_specialist(
    rows: Sequence[ReviewedEvent], feature_set: str, l2: float
) -> SpecialistModel:
    if l2 <= 0:
        raise SideSwitchSpecialistError("l2 must be positive")
    values = matrix_for(rows, feature_set)
    labels = labels_for(rows)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise SideSwitchSpecialistError("training rows must contain both decisions")
    impute, mean, scale = _fit_preprocessor(values)
    matrix = (np.where(np.isfinite(values), values, impute) - mean) / scale
    sample_weights = np.where(
        labels == 1,
        len(labels) / (2.0 * positives),
        len(labels) / (2.0 * negatives),
    )
    dimensions = matrix.shape[1]
    weights = np.zeros(dimensions, dtype=np.float64)
    bias = 0.0
    regularizer = np.diag(np.r_[np.full(dimensions, l2), 0.0])
    design = np.column_stack((matrix, np.ones(len(matrix), dtype=np.float64)))

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
        current_loss = _weighted_log_loss(
            matrix, labels, sample_weights, weights, bias, l2
        )
        step_scale = 1.0
        candidate_weights = weights - step[:-1]
        candidate_bias = bias - float(step[-1])
        while step_scale > 1e-5:
            candidate_weights = weights - step_scale * step[:-1]
            candidate_bias = bias - step_scale * float(step[-1])
            candidate_loss = _weighted_log_loss(
                matrix,
                labels,
                sample_weights,
                candidate_weights,
                candidate_bias,
                l2,
            )
            if candidate_loss <= current_loss + 1e-12:
                break
            step_scale *= 0.5
        weights = candidate_weights
        bias = candidate_bias
        if float(np.max(np.abs(step_scale * step))) < 1e-8:
            break

    return SpecialistModel(
        feature_set=feature_set,
        feature_names=feature_names(feature_set),
        impute=impute,
        mean=mean,
        scale=scale,
        weights=weights,
        bias=bias,
        threshold=0.5,
        l2=l2,
    )


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    guesses = np.asarray(predicted, dtype=bool)
    if truth.shape != guesses.shape:
        raise SideSwitchSpecialistError("metric labels and predictions are not aligned")
    tp = int(np.sum((truth == 1) & guesses))
    fp = int(np.sum((truth == 0) & guesses))
    fn = int(np.sum((truth == 1) & ~guesses))
    tn = int(np.sum((truth == 0) & ~guesses))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    return {
        "rows": len(truth),
        "positives": int(np.sum(truth == 1)),
        "negatives": int(np.sum(truth == 0)),
        "truePositives": tp,
        "falsePositives": fp,
        "falseNegatives": fn,
        "trueNegatives": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balancedAccuracy": balanced,
        "accuracy": (tp + tn) / len(truth) if len(truth) else None,
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.shape != scores.shape or not len(truth) or not np.isfinite(scores).all():
        raise SideSwitchSpecialistError(
            "threshold inputs must be finite, non-empty, and aligned"
        )
    thresholds = sorted({float(value) for value in scores}, reverse=True)
    thresholds.insert(0, math.nextafter(thresholds[0], math.inf))
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = binary_metrics(truth, scores >= threshold)
        candidates.append({"threshold": threshold, **metrics})
    return max(
        candidates,
        key=lambda row: (
            float(row["f1"] or 0.0),
            float(row["precision"] or 0.0),
            float(row["recall"] or 0.0),
            float(row["threshold"]),
        ),
    )


def grouped_cross_fit(
    rows: Sequence[ReviewedEvent], feature_set: str, l2: float
) -> tuple[np.ndarray, dict[str, Any]]:
    recording_ids = sorted({row.recording_id for row in rows})
    if len(recording_ids) < 2:
        raise SideSwitchSpecialistError(
            "grouped cross-fit needs at least two recordings"
        )
    probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for recording_id in recording_ids:
        train_indices = [
            index for index, row in enumerate(rows) if row.recording_id != recording_id
        ]
        held_indices = [
            index for index, row in enumerate(rows) if row.recording_id == recording_id
        ]
        train_rows = [rows[index] for index in train_indices]
        held_rows = [rows[index] for index in held_indices]
        model = fit_specialist(train_rows, feature_set, l2)
        held_probabilities = model.predict_proba(matrix_for(held_rows, feature_set))
        probabilities[held_indices] = held_probabilities
        folds.append(
            {
                "heldRecordingId": recording_id,
                "trainRows": len(train_rows),
                "heldRows": len(held_rows),
                "heldPositives": sum(row.label for row in held_rows),
            }
        )
    if not np.isfinite(probabilities).all():
        raise SideSwitchSpecialistError("grouped cross-fit left rows without scores")
    threshold = select_threshold(labels_for(rows), probabilities)
    return probabilities, {"folds": folds, "selectedThresholdMetrics": threshold}


def model_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
