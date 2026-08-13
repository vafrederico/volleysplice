"""Deterministic joint softmax model for the four public multistate targets.

The model is deliberately small and experiment-local.  Its primary objective is
unweighted categorical cross-entropy, which learns the fold-training empirical
class prior directly.  An explicitly requested inverse-frequency balanced mode
is also available as a secondary diagnostic; only that mode permits fold-local
empirical-prior correction at prediction time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .config import FeatureConfig, TrainingConfig
from .model import ModelError
from .multistate import STATE_ORDER


_CLASS_COUNT = len(STATE_ORDER)
CLASS_WEIGHTING_MODES = ("empirical", "balanced")


def _validated_values(
    values: np.ndarray,
    *,
    dimensions: int,
    label: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise ModelError(
            f"{label} expects (*, {dimensions}) features, got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ModelError(f"{label} features must be finite")
    return matrix


def _validated_targets(
    targets: np.ndarray,
    *,
    samples: int,
    label: str,
) -> np.ndarray:
    raw = np.asarray(targets)
    if raw.ndim != 1 or raw.shape != (samples,):
        raise ModelError(f"{label} targets must have one value per feature row")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ModelError(f"{label} targets must be integer state indexes")
    result = raw.astype(np.int64, copy=False)
    if np.any((result < 0) | (result >= _CLASS_COUNT)):
        raise ModelError(f"{label} targets must follow STATE_ORDER")
    return result


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != _CLASS_COUNT:
        raise ValueError("softmax logits have an invalid shape")
    if not len(logits):
        return np.empty(logits.shape, dtype=np.float64)
    maximum = np.max(logits, axis=1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _class_counts_from_summary(summary: Mapping[str, Any]) -> np.ndarray:
    raw = summary.get("classSamples")
    if not isinstance(raw, Mapping):
        raise ModelError("softmax training summary has no classSamples mapping")
    try:
        counts = np.asarray([raw[state.name] for state in STATE_ORDER], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError("softmax classSamples do not follow STATE_ORDER") from error
    if counts.shape != (_CLASS_COUNT,) or not np.isfinite(counts).all() or np.any(
        counts <= 0.0
    ):
        raise ModelError("softmax classSamples must be finite and positive")
    return counts


def _class_weighting_from_summary(summary: Mapping[str, Any]) -> str:
    mode = summary.get("classWeighting")
    if mode not in CLASS_WEIGHTING_MODES:
        raise ModelError(
            "softmax training summary has no valid classWeighting mode"
        )
    return str(mode)


@dataclass(frozen=True)
class MultistateSoftmaxModel:
    """One linear joint classifier with columns in ``STATE_ORDER``."""

    feature_config: FeatureConfig
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    training_summary: dict[str, Any]

    def __post_init__(self) -> None:
        self.feature_config.validate()
        dimensions = len(self.feature_names)
        if not dimensions or len(set(self.feature_names)) != dimensions or any(
            not isinstance(name, str) or not name for name in self.feature_names
        ):
            raise ModelError("softmax feature names must be non-empty and unique")
        if self.mean.shape != (dimensions,) or self.scale.shape != (dimensions,):
            raise ModelError("softmax normalization dimensions do not match features")
        if self.weights.shape != (dimensions, _CLASS_COUNT):
            raise ModelError("softmax weights must have one STATE_ORDER column")
        if self.bias.shape != (_CLASS_COUNT,):
            raise ModelError("softmax bias must have one value per STATE_ORDER class")
        if not all(
            np.isfinite(values).all()
            for values in (self.mean, self.scale, self.weights, self.bias)
        ):
            raise ModelError("softmax parameters must be finite")
        if np.any(self.scale <= 0.0):
            raise ModelError("softmax normalization scales must be positive")
        if not isinstance(self.training_summary, dict):
            raise ModelError("softmax training summary must be an object")

    def predict_logits(self, values: np.ndarray) -> np.ndarray:
        """Return native balanced-prior logits in ``STATE_ORDER``."""

        matrix = _validated_values(
            values,
            dimensions=len(self.feature_names),
            label="softmax model",
        )
        normalized = (
            matrix.astype(np.float64, copy=False)
            - self.mean.astype(np.float64, copy=False)
        ) / self.scale.astype(np.float64, copy=False)
        logits = normalized @ self.weights.astype(np.float64, copy=False)
        logits += self.bias.astype(np.float64, copy=False)
        if not np.isfinite(logits).all():
            raise ModelError("softmax prediction produced non-finite logits")
        return logits

    def _prediction_logits(
        self, values: np.ndarray, *, prior_corrected: bool
    ) -> np.ndarray:
        logits = self.predict_logits(values)
        if prior_corrected:
            if _class_weighting_from_summary(self.training_summary) != "balanced":
                raise ModelError(
                    "softmax prior correction is only valid for balanced training"
                )
            counts = _class_counts_from_summary(self.training_summary)
            logits = logits + np.log(counts / np.sum(counts))
        return logits

    def predict_log_scores(
        self,
        values: np.ndarray,
        *,
        prior_corrected: bool = False,
    ) -> np.ndarray:
        """Return normalized log probabilities in ``STATE_ORDER``."""

        return _log_softmax(
            self._prediction_logits(values, prior_corrected=prior_corrected)
        )

    def predict_probabilities(
        self,
        values: np.ndarray,
        *,
        prior_corrected: bool = False,
    ) -> np.ndarray:
        """Return normalized probabilities in ``STATE_ORDER``."""

        return np.exp(
            self.predict_log_scores(values, prior_corrected=prior_corrected)
        )


def _normalization(sequences: Sequence[np.ndarray], dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(dimensions, dtype=np.float64)
    total_squared = np.zeros(dimensions, dtype=np.float64)
    samples = 0
    for values in sequences:
        matrix = _validated_values(
            values,
            dimensions=dimensions,
            label="softmax training",
        )
        total += np.sum(matrix, axis=0, dtype=np.float64)
        total_squared += np.sum(
            np.square(matrix, dtype=np.float64), axis=0, dtype=np.float64
        )
        samples += len(matrix)
    if not samples:
        raise ModelError("softmax training sequences contain no samples")
    mean = total / samples
    variance = np.maximum(total_squared / samples - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _weighted_loss(
    values: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    class_weights: np.ndarray,
    l2: float,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for matrix, labels in zip(values, targets, strict=True):
        normalized = (
            matrix.astype(np.float64, copy=False)
            - mean.astype(np.float64, copy=False)
        ) / scale.astype(np.float64, copy=False)
        logits = normalized @ weights.astype(np.float64, copy=False)
        logits += bias.astype(np.float64, copy=False)
        log_probabilities = _log_softmax(logits)
        sample_weights = class_weights[labels]
        numerator -= float(
            np.sum(sample_weights * log_probabilities[np.arange(len(labels)), labels])
        )
        denominator += float(np.sum(sample_weights))
    return numerator / max(denominator, 1.0) + 0.5 * l2 * float(
        np.sum(np.square(weights, dtype=np.float64))
    )


def train_multistate_softmax(
    train_values: Sequence[np.ndarray],
    train_targets: Sequence[np.ndarray],
    validation_values: Sequence[np.ndarray],
    validation_targets: Sequence[np.ndarray],
    feature_config: FeatureConfig,
    feature_names: tuple[str, ...],
    training_config: TrainingConfig,
    *,
    class_weighting: str = "empirical",
) -> MultistateSoftmaxModel:
    """Fit a four-state linear softmax model.

    The default ``empirical`` mode minimizes ordinary categorical
    cross-entropy with unit sample weights. ``balanced`` uses inverse-frequency
    fold-training class weights and makes ``prior_corrected=True`` available at
    inference. Normalization, weights, and priors use training rows only.
    Validation rows affect only early stopping. Passing empty validation
    sequences uses training loss, which lets an outer refit run for a
    caller-frozen epoch cap by setting patience above that cap.
    """

    feature_config.validate()
    training_config.validate()
    if class_weighting not in CLASS_WEIGHTING_MODES:
        raise ValueError(
            "softmax class_weighting must be 'empirical' or 'balanced'"
        )
    dimensions = len(feature_names)
    if not dimensions or len(set(feature_names)) != dimensions or any(
        not isinstance(name, str) or not name for name in feature_names
    ):
        raise ModelError("softmax feature names must be non-empty and unique")
    if len(train_values) != len(train_targets) or not train_values:
        raise ModelError("softmax training features and targets must be aligned")
    if len(validation_values) != len(validation_targets):
        raise ModelError("softmax validation features and targets must be aligned")

    validated_train_values: list[np.ndarray] = []
    validated_train_targets: list[np.ndarray] = []
    for values, targets in zip(train_values, train_targets, strict=True):
        matrix = _validated_values(
            values, dimensions=dimensions, label="softmax training"
        )
        labels = _validated_targets(
            targets, samples=len(matrix), label="softmax training"
        )
        validated_train_values.append(matrix)
        validated_train_targets.append(labels)

    validated_validation_values: list[np.ndarray] = []
    validated_validation_targets: list[np.ndarray] = []
    for values, targets in zip(validation_values, validation_targets, strict=True):
        matrix = _validated_values(
            values, dimensions=dimensions, label="softmax validation"
        )
        labels = _validated_targets(
            targets, samples=len(matrix), label="softmax validation"
        )
        validated_validation_values.append(matrix)
        validated_validation_targets.append(labels)

    mean, scale = _normalization(validated_train_values, dimensions)
    class_counts = np.zeros(_CLASS_COUNT, dtype=np.int64)
    for labels in validated_train_targets:
        class_counts += np.bincount(labels, minlength=_CLASS_COUNT)
    if np.any(class_counts <= 0):
        missing = [
            state.name
            for state, count in zip(STATE_ORDER, class_counts, strict=True)
            if count <= 0
        ]
        raise ModelError(f"softmax training requires every state class: {missing}")
    samples = int(np.sum(class_counts))
    class_weights = (
        np.ones(_CLASS_COUNT, dtype=np.float64)
        if class_weighting == "empirical"
        else samples / (_CLASS_COUNT * class_counts.astype(np.float64))
    )

    rng = np.random.default_rng(training_config.seed)
    weights = np.zeros((dimensions, _CLASS_COUNT), dtype=np.float32)
    bias = np.zeros(_CLASS_COUNT, dtype=np.float32)
    moment_w = np.zeros_like(weights)
    velocity_w = np.zeros_like(weights)
    moment_b = np.zeros_like(bias)
    velocity_b = np.zeros_like(bias)
    beta_one, beta_two = 0.9, 0.999
    update = 0
    best_loss = math.inf
    best_weights = weights.copy()
    best_bias = bias.copy()
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, training_config.epochs + 1):
        order = rng.permutation(len(validated_train_values))
        for sequence_index in order:
            matrix = validated_train_values[int(sequence_index)]
            labels = validated_train_targets[int(sequence_index)]
            indexes = rng.permutation(len(matrix))
            for offset in range(0, len(indexes), training_config.batch_size):
                batch_indexes = indexes[offset : offset + training_config.batch_size]
                batch = (
                    matrix[batch_indexes].astype(np.float64, copy=False)
                    - mean.astype(np.float64, copy=False)
                ) / scale.astype(np.float64, copy=False)
                batch_targets = labels[batch_indexes]
                logits = batch @ weights.astype(np.float64, copy=False)
                logits += bias.astype(np.float64, copy=False)
                probabilities = np.exp(_log_softmax(logits))
                probabilities[np.arange(len(batch_targets)), batch_targets] -= 1.0
                sample_weights = class_weights[batch_targets]
                error = probabilities * sample_weights[:, None]
                denominator = max(float(np.sum(sample_weights)), 1.0)
                gradient_w = batch.T @ error / denominator
                gradient_w += training_config.l2 * weights
                gradient_b = np.sum(error, axis=0) / denominator

                update += 1
                moment_w = beta_one * moment_w + (1.0 - beta_one) * gradient_w
                velocity_w = beta_two * velocity_w + (1.0 - beta_two) * np.square(
                    gradient_w
                )
                moment_b = beta_one * moment_b + (1.0 - beta_one) * gradient_b
                velocity_b = beta_two * velocity_b + (1.0 - beta_two) * np.square(
                    gradient_b
                )
                corrected_w = moment_w / (1.0 - beta_one**update)
                corrected_vw = velocity_w / (1.0 - beta_two**update)
                corrected_b = moment_b / (1.0 - beta_one**update)
                corrected_vb = velocity_b / (1.0 - beta_two**update)
                weights -= (
                    training_config.learning_rate
                    * corrected_w
                    / (np.sqrt(corrected_vw) + 1e-8)
                )
                bias -= (
                    training_config.learning_rate
                    * corrected_b
                    / (np.sqrt(corrected_vb) + 1e-8)
                )

        evaluation_values = (
            validated_validation_values
            if validated_validation_values
            else validated_train_values
        )
        evaluation_targets = (
            validated_validation_targets
            if validated_validation_targets
            else validated_train_targets
        )
        loss = _weighted_loss(
            evaluation_values,
            evaluation_targets,
            mean,
            scale,
            weights,
            bias,
            class_weights,
            training_config.l2,
        )
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, "validationLoss": round(loss, 7)})
        if epoch == 1 or loss < best_loss - 1e-6:
            best_loss = loss
            best_weights = weights.copy()
            best_bias = bias.copy()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= training_config.patience:
                break

    class_samples = {
        state.name: int(count)
        for state, count in zip(STATE_ORDER, class_counts, strict=True)
    }
    class_priors = {
        state.name: float(count / samples)
        for state, count in zip(STATE_ORDER, class_counts, strict=True)
    }
    balanced_weights = {
        state.name: float(weight)
        for state, weight in zip(STATE_ORDER, class_weights, strict=True)
    }
    return MultistateSoftmaxModel(
        feature_config=feature_config,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        weights=best_weights.astype(np.float32, copy=False),
        bias=best_bias.astype(np.float32, copy=False),
        training_summary={
            "seed": training_config.seed,
            "epochsRequested": training_config.epochs,
            "epochsCompleted": epoch,
            "bestEpoch": best_epoch,
            "bestValidationLoss": round(best_loss, 7),
            "trainSamples": samples,
            "validationSamples": sum(
                len(labels) for labels in validated_validation_targets
            ),
            "validationUsedForEarlyStopping": bool(validated_validation_values),
            "stateOrder": [state.name for state in STATE_ORDER],
            "classSamples": class_samples,
            "classPriors": class_priors,
            "classWeights": balanced_weights,
            "classWeighting": class_weighting,
            "balancedClassObjective": class_weighting == "balanced",
            "priorCorrectionAvailable": class_weighting == "balanced",
            "config": training_config.to_dict(),
            "lossHistory": history,
        },
    )


def multistate_softmax_fingerprint(model: MultistateSoftmaxModel) -> str:
    """Return a stable fingerprint for nested-fold reproduction checks."""

    digest = hashlib.sha256()
    for values in (model.mean, model.scale, model.weights, model.bias):
        digest.update(np.ascontiguousarray(values).tobytes())
    digest.update("\0".join(model.feature_names).encode("utf-8"))
    digest.update("\0".join(state.name for state in STATE_ORDER).encode("utf-8"))
    digest.update(
        _class_weighting_from_summary(model.training_summary).encode("utf-8")
    )
    # Fold-local class counts affect prior-corrected inference even when the
    # fitted parameter arrays happen to be identical.
    digest.update(
        np.ascontiguousarray(
            _class_counts_from_summary(model.training_summary), dtype=np.float64
        ).tobytes()
    )
    digest.update(
        json.dumps(model.feature_config.to_dict(), sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()


__all__ = [
    "CLASS_WEIGHTING_MODES",
    "MultistateSoftmaxModel",
    "multistate_softmax_fingerprint",
    "train_multistate_softmax",
]
