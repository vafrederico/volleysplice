"""Deterministic baseline specialist for reviewed serving-side evidence.

The specialist consumes the fixed feature rows produced by
``evaluate-serving-side-existing-labels.py`` and the complete NAS-backed review.
Near is the positive class and far is the negative class, but selection uses
balanced accuracy and macro-F1 because the two camera-space labels are symmetric.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_KIND = "volleycut-serving-side-specialist-v1"
MODEL_SCHEMA_VERSION = 1
DECISIONS = {"near": 1, "far": 0}
EVALUATION_SPLITS = frozenset({"challenge", "non-training", "test"})

MOTION_PALETTE_FEATURES = (
    "pixelMotionMargin",
    "paletteChangeMargin",
)
BASELINE_MOTION_PALETTE_FEATURES = (
    "baselinePixelMotionMargin",
    "baselinePaletteChangeMargin",
)
MARGIN_FEATURES = (
    *MOTION_PALETTE_FEATURES,
    *BASELINE_MOTION_PALETTE_FEATURES,
    "hogAreaChangeMargin",
    "hogCountChangeMargin",
    "baselineHogAreaChangeMargin",
)
FULL_EXISTING_FEATURES = (
    "nearPixelMotion",
    "farPixelMotion",
    "pixelMotionMargin",
    "nearPaletteChange",
    "farPaletteChange",
    "paletteChangeMargin",
    "nearBaselinePixelMotion",
    "farBaselinePixelMotion",
    "baselinePixelMotionMargin",
    "nearBaselinePaletteChange",
    "farBaselinePaletteChange",
    "baselinePaletteChangeMargin",
    "nearHogAreaBefore",
    "farHogAreaBefore",
    "nearHogAreaAfter",
    "farHogAreaAfter",
    "hogAreaChangeMargin",
    "hogCountChangeMargin",
    "baselineHogAreaChangeMargin",
)
FEATURE_SETS = {
    "motion-palette": tuple(
        f"feature:{name}" for name in MOTION_PALETTE_FEATURES
    ),
    "baseline-motion-palette": tuple(
        f"feature:{name}" for name in BASELINE_MOTION_PALETTE_FEATURES
    ),
    "combined-motion-palette": tuple(
        f"feature:{name}"
        for name in (*MOTION_PALETTE_FEATURES, *BASELINE_MOTION_PALETTE_FEATURES)
    ),
    "margin-bank": tuple(f"feature:{name}" for name in MARGIN_FEATURES),
    "full-existing-bank": tuple(
        f"feature:{name}" for name in FULL_EXISTING_FEATURES
    ),
}


class ServingSideSpecialistError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReviewedRally:
    rally_id: str
    recording_id: str
    environment: str
    split: str
    source_group: str
    source_type: str
    target_status: str
    decision: str
    label: int
    rally: Mapping[str, Any]


@dataclass(frozen=True)
class ServingSideModel:
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
            raise ServingSideSpecialistError(
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
            "positiveDecision": "near",
            "negativeDecision": "far",
            "impute": self.impute.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ServingSideModel":
        feature_set = value.get("featureSet")
        names = value.get("featureNames")
        if feature_set not in FEATURE_SETS or not isinstance(names, list) or not names:
            raise ServingSideSpecialistError("invalid specialist feature signature")
        expected_names = feature_names(str(feature_set))
        if tuple(names) != expected_names:
            raise ServingSideSpecialistError(
                "specialist feature names do not match the declared feature set"
            )
        if (
            value.get("positiveDecision") != "near"
            or value.get("negativeDecision") != "far"
        ):
            raise ServingSideSpecialistError("specialist decision mapping is invalid")
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
            raise ServingSideSpecialistError("specialist parameter shapes do not agree")
        if not all(
            np.isfinite(array).all()
            for array in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise ServingSideSpecialistError("specialist parameters must be finite")
        if (
            np.any(model.scale <= 0)
            or not math.isfinite(model.bias)
            or not 0 <= model.threshold <= 1
            or model.l2 <= 0
        ):
            raise ServingSideSpecialistError("specialist scalar parameters are invalid")
        return model


def _record(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return math.nan


def _raw_value(rally: Mapping[str, Any], specification: str) -> float:
    group, name = specification.split(":", 1)
    if group == "feature":
        return _finite_number(_record(rally.get("features")).get(name))
    raise ServingSideSpecialistError(
        f"unknown feature specification: {specification}"
    )


def feature_names(feature_set: str) -> tuple[str, ...]:
    specifications = FEATURE_SETS.get(feature_set)
    if specifications is None:
        raise ServingSideSpecialistError(f"unknown feature set: {feature_set}")
    names: list[str] = []
    for specification in specifications:
        names.extend((specification, f"{specification}:missing"))
    return tuple(names)


def rally_vector(rally: Mapping[str, Any], feature_set: str) -> np.ndarray:
    specifications = FEATURE_SETS.get(feature_set)
    if specifications is None:
        raise ServingSideSpecialistError(f"unknown feature set: {feature_set}")
    values: list[float] = []
    for specification in specifications:
        value = _raw_value(rally, specification)
        values.extend((value, 0.0 if math.isfinite(value) else 1.0))
    return np.asarray(values, dtype=np.float64)


def reviewed_rallies(
    report: Mapping[str, Any], decisions: Mapping[str, Any]
) -> tuple[list[ReviewedRally], dict[str, int]]:
    if (
        decisions.get("reportKind") != report.get("kind")
        or decisions.get("reportCreatedAt") != report.get("createdAt")
    ):
        raise ServingSideSpecialistError(
            "serving-side decisions belong to a different diagnostic report"
        )
    raw_rallies = report.get("rallies")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_rallies, list) or not isinstance(raw_decisions, Mapping):
        raise ServingSideSpecialistError(
            "report or decision payload has an invalid schema"
        )
    rally_ids = [
        rally.get("rallyId")
        for rally in raw_rallies
        if isinstance(rally, Mapping) and isinstance(rally.get("rallyId"), str)
    ]
    if len(rally_ids) != len(raw_rallies) or len(set(rally_ids)) != len(rally_ids):
        raise ServingSideSpecialistError(
            "serving-side report contains invalid or duplicate rally IDs"
        )
    unknown_ids = sorted(set(raw_decisions) - set(rally_ids))
    if unknown_ids:
        raise ServingSideSpecialistError(
            f"decision file contains {len(unknown_ids)} unknown rally IDs"
        )

    rows: list[ReviewedRally] = []
    counts = {
        "rallies": len(raw_rallies),
        "near": 0,
        "far": 0,
        "unclear": 0,
        "missing": 0,
    }
    for rally in raw_rallies:
        assert isinstance(rally, Mapping)
        rally_id = str(rally["rallyId"])
        decision = raw_decisions.get(rally_id)
        if decision is None:
            counts["missing"] += 1
            continue
        if decision == "unclear":
            counts["unclear"] += 1
            continue
        if decision not in DECISIONS:
            raise ServingSideSpecialistError(
                f"invalid review decision for {rally_id}"
            )
        counts[str(decision)] += 1
        rows.append(
            ReviewedRally(
                rally_id=rally_id,
                recording_id=str(rally.get("recordingId", "")),
                environment=str(rally.get("environment", "unknown")),
                split=str(rally.get("split", "unknown")),
                source_group=str(rally.get("sourceGroup", "unknown")),
                source_type=str(rally.get("sourceType", "unknown")),
                target_status=str(rally.get("targetStatus", "unknown")),
                decision=str(decision),
                label=DECISIONS[str(decision)],
                rally=rally,
            )
        )
    return rows, counts


def apply_review_corrections(
    report: Mapping[str, Any],
    decisions: Mapping[str, Any],
    corrections: Mapping[str, Any],
    *,
    base_decision_sha256: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply the NAS correction overlay before building a feature dataset.

    Side corrections replace the reviewed decision. A ``not-serve`` correction
    removes that rally from the serving-side training/evaluation universe.
    """
    if (
        corrections.get("schemaVersion") != 1
        or corrections.get("reportKind") != report.get("kind")
        or corrections.get("reportCreatedAt") != report.get("createdAt")
        or corrections.get("baseDecisionSha256") != base_decision_sha256
    ):
        raise ServingSideSpecialistError(
            "serving-side corrections belong to a different label revision"
        )
    raw_decisions = decisions.get("decisions")
    raw_corrections = corrections.get("corrections")
    if not isinstance(raw_decisions, Mapping) or not isinstance(
        raw_corrections, Mapping
    ):
        raise ServingSideSpecialistError(
            "serving-side correction payload has an invalid schema"
        )
    effective_decisions = dict(raw_decisions)
    counts = {"applied": 0, "near": 0, "far": 0, "notServe": 0}
    for rally_id, correction in raw_corrections.items():
        original = raw_decisions.get(rally_id)
        if not isinstance(rally_id, str) or original not in DECISIONS:
            raise ServingSideSpecialistError(
                f"serving-side correction targets an unknown or unclear rally: {rally_id}"
            )
        if correction == "not-serve":
            effective_decisions.pop(rally_id, None)
            counts["notServe"] += 1
        elif correction in DECISIONS:
            effective_decisions[rally_id] = correction
            counts[str(correction)] += 1
        else:
            raise ServingSideSpecialistError(
                f"invalid serving-side correction for {rally_id}"
            )
        counts["applied"] += 1
    return {**decisions, "decisions": effective_decisions}, counts


def matrix_for(rows: Sequence[ReviewedRally], feature_set: str) -> np.ndarray:
    if not rows:
        return np.empty((0, len(feature_names(feature_set))), dtype=np.float64)
    return np.stack([rally_vector(row.rally, feature_set) for row in rows])


def labels_for(rows: Sequence[ReviewedRally]) -> np.ndarray:
    return np.asarray([row.label for row in rows], dtype=np.float64)


def _fit_preprocessor(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or not len(values):
        raise ServingSideSpecialistError(
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
    rows: Sequence[ReviewedRally], feature_set: str, l2: float
) -> ServingSideModel:
    if l2 <= 0:
        raise ServingSideSpecialistError("l2 must be positive")
    values = matrix_for(rows, feature_set)
    labels = labels_for(rows)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ServingSideSpecialistError("training rows must contain both decisions")
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

    return ServingSideModel(
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


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    guesses = np.asarray(predicted, dtype=bool)
    if truth.shape != guesses.shape:
        raise ServingSideSpecialistError(
            "metric labels and predictions are not aligned"
        )
    near_near = int(np.sum((truth == 1) & guesses))
    far_near = int(np.sum((truth == 0) & guesses))
    near_far = int(np.sum((truth == 1) & ~guesses))
    far_far = int(np.sum((truth == 0) & ~guesses))
    near_precision = _ratio(near_near, near_near + far_near)
    near_recall = _ratio(near_near, near_near + near_far)
    far_precision = _ratio(far_far, far_far + near_far)
    far_recall = _ratio(far_far, far_far + far_near)
    near_f1 = _f1(near_precision, near_recall)
    far_f1 = _f1(far_precision, far_recall)
    macro_f1 = (
        (near_f1 + far_f1) / 2.0
        if near_f1 is not None and far_f1 is not None
        else None
    )
    balanced = (
        (near_recall + far_recall) / 2.0
        if near_recall is not None and far_recall is not None
        else None
    )
    return {
        "rows": len(truth),
        "near": int(np.sum(truth == 1)),
        "far": int(np.sum(truth == 0)),
        "confusion": {
            "near": {"near": near_near, "far": near_far},
            "far": {"near": far_near, "far": far_far},
        },
        "nearPrecision": near_precision,
        "nearRecall": near_recall,
        "nearF1": near_f1,
        "farPrecision": far_precision,
        "farRecall": far_recall,
        "farF1": far_f1,
        "macroF1": macro_f1,
        "balancedAccuracy": balanced,
        "accuracy": (
            (near_near + far_far) / len(truth) if len(truth) else None
        ),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        truth.shape != scores.shape
        or not len(truth)
        or not np.isfinite(scores).all()
        or len(set(truth.tolist())) != 2
    ):
        raise ServingSideSpecialistError(
            "threshold inputs must be finite, aligned, and contain both decisions"
        )
    thresholds = sorted({float(value) for value in scores}, reverse=True)
    thresholds.insert(0, math.nextafter(thresholds[0], math.inf))
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        candidates.append(
            {"threshold": threshold, **binary_metrics(truth, scores >= threshold)}
        )
    return max(
        candidates,
        key=lambda row: (
            float(row["balancedAccuracy"] or 0.0),
            float(row["macroF1"] or 0.0),
            float(row["accuracy"] or 0.0),
            -abs(float(row["threshold"]) - 0.5),
        ),
    )


def grouped_cross_fit(
    rows: Sequence[ReviewedRally], feature_set: str, l2: float
) -> tuple[np.ndarray, dict[str, Any]]:
    recording_ids = sorted({row.recording_id for row in rows})
    if len(recording_ids) < 2:
        raise ServingSideSpecialistError(
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
        probabilities[held_indices] = model.predict_proba(
            matrix_for(held_rows, feature_set)
        )
        folds.append(
            {
                "heldRecordingId": recording_id,
                "trainRows": len(train_rows),
                "heldRows": len(held_rows),
                "heldNear": sum(row.label for row in held_rows),
                "heldFar": sum(1 - row.label for row in held_rows),
            }
        )
    if not np.isfinite(probabilities).all():
        raise ServingSideSpecialistError("grouped cross-fit left rows without scores")
    threshold = select_threshold(labels_for(rows), probabilities)
    return probabilities, {"folds": folds, "selectedThresholdMetrics": threshold}


def model_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
