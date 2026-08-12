from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import (
    AUDIO_NORMALIZED_FEATURE_VERSION,
    FEATURE_VERSION,
    LEGACY_FEATURE_VERSIONS,
    MODEL_TYPE,
    DecoderConfig,
    FeatureConfig,
    TrainingConfig,
    feature_version_for_config,
)
from .version import __version__


MODEL_SCHEMA_VERSION = 1
RALLY_LIVE_TASK = "rally-live"
SERVE_CONTACT_TASK = "serve-contact"
STACKED_RALLY_TASK = "rally-live-stacked-serve"
DEAD_BALL_TASK = "dead-ball-boundary"
PREDICTION_TASKS = {
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    STACKED_RALLY_TASK,
    DEAD_BALL_TASK,
}


class ModelError(RuntimeError):
    pass


@dataclass
class LogisticModel:
    feature_config: FeatureConfig
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    decoder: DecoderConfig
    training_summary: dict[str, Any]
    artifact_sha256: str | None = None
    feature_version: str = FEATURE_VERSION
    prediction_task: str = RALLY_LIVE_TASK

    def predict(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ModelError(
                f"model expects (*, {len(self.feature_names)}) features, got {values.shape}"
            )
        normalized = (values.astype(np.float32, copy=False) - self.mean) / self.scale
        logits = np.clip(normalized @ self.weights + self.bias, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def save(self, destination: str | Path) -> Path:
        if self.prediction_task not in PREDICTION_TASKS:
            raise ModelError(f"unsupported prediction task: {self.prediction_task!r}")
        compatible_feature_versions = {feature_version_for_config(self.feature_config)}
        if (
            not self.feature_config.use_audio
            and not self.feature_config.use_advanced_visual
        ):
            compatible_feature_versions.add("court-motion-flow-v1")
        if self.feature_version not in compatible_feature_versions:
            raise ModelError(
                "model feature version does not match its feature configuration"
            )
        model_dir = Path(destination).expanduser().resolve()
        if model_dir.exists():
            if not model_dir.is_dir() or any(model_dir.iterdir()):
                raise ModelError(f"model destination is not an empty directory: {model_dir}")
        created_directory = not model_dir.exists()
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / "model.json"
        weights_path = model_dir / "weights.npz"
        weights_descriptor, temporary_weights_name = tempfile.mkstemp(
            prefix=".weights-", suffix=".npz", dir=model_dir
        )
        metadata_descriptor, temporary_metadata_name = tempfile.mkstemp(
            prefix=".model-", suffix=".json", dir=model_dir
        )
        os.close(weights_descriptor)
        os.close(metadata_descriptor)
        temporary_weights = Path(temporary_weights_name)
        temporary_metadata = Path(temporary_metadata_name)
        try:
            np.savez_compressed(
                temporary_weights,
                mean=self.mean.astype(np.float32),
                scale=self.scale.astype(np.float32),
                weights=self.weights.astype(np.float32),
                bias=np.asarray([self.bias], dtype=np.float32),
            )
            weights_bytes = temporary_weights.read_bytes()
            metadata = {
                "schemaVersion": MODEL_SCHEMA_VERSION,
                "producer": f"volleycut-analysis/{__version__}",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "modelType": MODEL_TYPE,
                "featureVersion": self.feature_version,
                "predictionTask": self.prediction_task,
                "featureConfig": self.feature_config.to_dict(),
                "featureNames": list(self.feature_names),
                "decoder": self.decoder.to_dict(),
                "training": self.training_summary,
                "weightsFile": weights_path.name,
                "weightsSha256": hashlib.sha256(weights_bytes).hexdigest(),
            }
            metadata_bytes = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode(
                "utf-8"
            )
            temporary_metadata.write_bytes(metadata_bytes)
            temporary_weights.replace(weights_path)
            try:
                temporary_metadata.replace(metadata_path)
            except OSError:
                weights_path.unlink(missing_ok=True)
                raise
            self.artifact_sha256 = _artifact_sha256(metadata_bytes, weights_bytes)
        except Exception:
            temporary_weights.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
            if created_directory:
                try:
                    model_dir.rmdir()
                except OSError:
                    pass
            raise
        return model_dir


def load_model(path: str | Path) -> LogisticModel:
    supplied = Path(path).expanduser().resolve()
    metadata_path = supplied / "model.json" if supplied.is_dir() else supplied
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ModelError(f"cannot load model metadata {metadata_path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ModelError("model metadata root must be an object")
    if metadata.get("schemaVersion") != MODEL_SCHEMA_VERSION:
        raise ModelError(f"unsupported model schemaVersion: {metadata.get('schemaVersion')!r}")
    feature_version = metadata.get("featureVersion")
    if (
        metadata.get("modelType") != MODEL_TYPE
        or feature_version
        not in {
            FEATURE_VERSION,
            AUDIO_NORMALIZED_FEATURE_VERSION,
            *LEGACY_FEATURE_VERSIONS,
        }
    ):
        raise ModelError("unsupported model or feature type")
    weights_file = metadata.get("weightsFile")
    if not isinstance(weights_file, str) or Path(weights_file).name != weights_file:
        raise ModelError("model weightsFile must be a local filename")
    weights_path = metadata_path.parent / weights_file
    expected_digest = metadata.get("weightsSha256")
    try:
        weights_bytes = weights_path.read_bytes()
    except OSError as error:
        raise ModelError(f"cannot read model weights {weights_path}: {error}") from error
    if not isinstance(expected_digest, str) or hashlib.sha256(weights_bytes).hexdigest() != expected_digest:
        raise ModelError("model weights digest is missing or does not match")
    try:
        with np.load(io.BytesIO(weights_bytes), allow_pickle=False) as artifact:
            mean = artifact["mean"].astype(np.float32, copy=False)
            scale = artifact["scale"].astype(np.float32, copy=False)
            weights = artifact["weights"].astype(np.float32, copy=False)
            bias_array = artifact["bias"].astype(np.float32, copy=False)
    except (OSError, KeyError, ValueError, IndexError) as error:
        raise ModelError(f"cannot load model weights {weights_path}: {error}") from error
    raw_names = metadata.get("featureNames")
    if (
        not isinstance(raw_names, list)
        or not raw_names
        or any(not isinstance(item, str) or not item for item in raw_names)
        or len(set(raw_names)) != len(raw_names)
    ):
        raise ModelError("model featureNames must be a non-empty unique string array")
    names = tuple(raw_names)
    expected_shape = (len(names),)
    if mean.shape != expected_shape or scale.shape != expected_shape or weights.shape != expected_shape:
        raise ModelError("model parameter and feature dimensions do not agree")
    if bias_array.shape != (1,):
        raise ModelError("model bias must contain exactly one value")
    bias = float(bias_array[0])
    if not all(np.isfinite(array).all() for array in (mean, scale, weights)) or not math.isfinite(bias):
        raise ModelError("model contains non-finite parameters")
    if np.any(scale <= 0):
        raise ModelError("model normalization scales must be positive")
    try:
        feature_config = FeatureConfig.from_dict(metadata["featureConfig"])
        decoder = DecoderConfig.from_dict(metadata["decoder"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(f"invalid model configuration: {error}") from error
    training_summary = metadata.get("training", {})
    if not isinstance(training_summary, dict):
        raise ModelError("model training metadata must be an object")
    prediction_task = metadata.get("predictionTask", RALLY_LIVE_TASK)
    if prediction_task not in PREDICTION_TASKS:
        raise ModelError(f"unsupported model predictionTask: {prediction_task!r}")
    compatible_feature_versions = {feature_version_for_config(feature_config)}
    if not feature_config.use_audio and not feature_config.use_advanced_visual:
        compatible_feature_versions.add("court-motion-flow-v1")
    if feature_version not in compatible_feature_versions:
        raise ModelError("model feature version does not match its feature configuration")
    return LogisticModel(
        feature_config=feature_config,
        feature_names=names,
        mean=mean,
        scale=scale,
        weights=weights,
        bias=bias,
        decoder=decoder,
        training_summary=dict(training_summary),
        artifact_sha256=_artifact_sha256(metadata_bytes, weights_bytes),
        feature_version=str(feature_version),
        prediction_task=prediction_task,
    )


def _artifact_sha256(metadata_bytes: bytes, weights_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(metadata_bytes)
    digest.update(weights_bytes)
    return digest.hexdigest()


def _normalization(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ModelError("training requires at least one feature sequence")
    dimensions = sequences[0].shape[1]
    total = np.zeros(dimensions, dtype=np.float64)
    total_squared = np.zeros(dimensions, dtype=np.float64)
    count = 0
    for values in sequences:
        if values.ndim != 2 or values.shape[1] != dimensions:
            raise ModelError("all training feature sequences must share a signature")
        total += np.sum(values, axis=0, dtype=np.float64)
        total_squared += np.sum(np.square(values, dtype=np.float64), axis=0, dtype=np.float64)
        count += len(values)
    if count == 0:
        raise ModelError("training sequences contain no samples")
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _weighted_loss(
    values: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    bias: float,
    positive_weight: float,
    negative_weight: float,
    l2: float,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for matrix, targets in zip(values, labels, strict=True):
        normalized = (matrix - mean) / scale
        logits = normalized @ weights + bias
        sample_weights = np.where(targets > 0.5, positive_weight, negative_weight)
        losses = np.logaddexp(0.0, logits) - targets * logits
        numerator += float(np.sum(losses * sample_weights))
        denominator += float(np.sum(sample_weights))
    return numerator / max(denominator, 1.0) + 0.5 * l2 * float(weights @ weights)


def train_logistic_model(
    train_values: Sequence[np.ndarray],
    train_labels: Sequence[np.ndarray],
    validation_values: Sequence[np.ndarray],
    validation_labels: Sequence[np.ndarray],
    feature_config: FeatureConfig,
    feature_names: tuple[str, ...],
    decoder: DecoderConfig,
    config: TrainingConfig,
    *,
    prediction_task: str = RALLY_LIVE_TASK,
) -> LogisticModel:
    if prediction_task not in PREDICTION_TASKS:
        raise ModelError(f"unsupported prediction task: {prediction_task!r}")
    config.validate()
    feature_config.validate()
    if len(train_values) != len(train_labels) or not train_values:
        raise ModelError("training features and labels must be non-empty and aligned")
    if len(validation_values) != len(validation_labels):
        raise ModelError("validation features and labels must be aligned")
    for matrix, labels in zip((*train_values, *validation_values), (*train_labels, *validation_labels), strict=True):
        if matrix.ndim != 2 or labels.shape != (len(matrix),):
            raise ModelError("each feature sequence must have one label per row")
        if matrix.shape[1] != len(feature_names):
            raise ModelError("feature signature does not match feature names")

    mean, scale = _normalization(train_values)
    positives = sum(float(np.sum(labels > 0.5)) for labels in train_labels)
    samples = sum(len(labels) for labels in train_labels)
    negatives = samples - positives
    if positives == 0 or negatives == 0:
        raise ModelError("training data must contain both live and dead samples")
    positive_weight = samples / (2.0 * positives)
    negative_weight = samples / (2.0 * negatives)

    rng = np.random.default_rng(config.seed)
    dimensions = len(feature_names)
    weights = np.zeros(dimensions, dtype=np.float32)
    bias = 0.0
    moment_w = np.zeros_like(weights)
    velocity_w = np.zeros_like(weights)
    moment_b = 0.0
    velocity_b = 0.0
    beta_one, beta_two = 0.9, 0.999
    update = 0
    best_loss = math.inf
    best_weights = weights.copy()
    best_bias = bias
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(train_values))
        for sequence_index in order:
            matrix = train_values[int(sequence_index)]
            targets = train_labels[int(sequence_index)]
            indices = rng.permutation(len(matrix))
            for offset in range(0, len(indices), config.batch_size):
                batch_indices = indices[offset : offset + config.batch_size]
                batch = (matrix[batch_indices] - mean) / scale
                batch_targets = targets[batch_indices]
                logits = np.clip(batch @ weights + bias, -30.0, 30.0)
                probabilities = 1.0 / (1.0 + np.exp(-logits))
                sample_weights = np.where(
                    batch_targets > 0.5, positive_weight, negative_weight
                ).astype(np.float32)
                error = (probabilities - batch_targets) * sample_weights
                denominator = max(float(np.sum(sample_weights)), 1.0)
                gradient_w = batch.T @ error / denominator + config.l2 * weights
                gradient_b = float(np.sum(error) / denominator)

                update += 1
                moment_w = beta_one * moment_w + (1 - beta_one) * gradient_w
                velocity_w = beta_two * velocity_w + (1 - beta_two) * np.square(gradient_w)
                moment_b = beta_one * moment_b + (1 - beta_one) * gradient_b
                velocity_b = beta_two * velocity_b + (1 - beta_two) * gradient_b * gradient_b
                corrected_w = moment_w / (1 - beta_one**update)
                corrected_vw = velocity_w / (1 - beta_two**update)
                corrected_b = moment_b / (1 - beta_one**update)
                corrected_vb = velocity_b / (1 - beta_two**update)
                weights -= config.learning_rate * corrected_w / (np.sqrt(corrected_vw) + 1e-8)
                bias -= config.learning_rate * corrected_b / (math.sqrt(corrected_vb) + 1e-8)

        evaluation_values = validation_values or train_values
        evaluation_labels = validation_labels or train_labels
        loss = _weighted_loss(
            evaluation_values,
            evaluation_labels,
            mean,
            scale,
            weights,
            bias,
            positive_weight,
            negative_weight,
            config.l2,
        )
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, "validationLoss": round(loss, 7)})
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_weights = weights.copy()
            best_bias = bias
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    summary: dict[str, Any] = {
        "seed": config.seed,
        "epochsRequested": config.epochs,
        "epochsCompleted": epoch,
        "bestEpoch": best_epoch,
        "bestValidationLoss": round(best_loss, 7),
        "trainSamples": samples,
        "validationSamples": sum(len(labels) for labels in validation_labels),
        "positiveSamples": int(positives),
        "negativeSamples": int(negatives),
        "positiveClassWeight": round(positive_weight, 6),
        "negativeClassWeight": round(negative_weight, 6),
        "config": config.to_dict(),
        "lossHistory": history,
    }
    return LogisticModel(
        feature_config=feature_config,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        weights=best_weights,
        bias=float(best_bias),
        decoder=decoder,
        training_summary=summary,
        feature_version=feature_version_for_config(feature_config),
        prediction_task=prediction_task,
    )
