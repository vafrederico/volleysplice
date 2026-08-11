from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import FEATURE_VERSION, DecoderConfig, FeatureConfig, TrainingConfig
from .crop_evaluation import RecordingIntervals, evaluate_crop_padding
from .decoder import decode_probabilities
from .feature_families import (
    FEATURE_FAMILY_KINDS,
    base_feature_name,
    feature_family,
    grouped_feature_indexes,
)
from .features import FeatureSequence, contextualize
from .metrics import aggregate_evaluations, aggregate_outcome_slices
from .model import LogisticModel, load_model, train_logistic_model
from .pipeline import (
    PreparedRecording,
    _evaluate_prepared_probabilities,
    _evaluate_prepared_probabilities_for_selection,
    _manifest_digest,
    _prepare_many,
    _tune_decoder,
    evaluate_dataset,
    train_dataset,
)
from .schema import DatasetManifest, Interval, Recording, load_manifest


EXPERIMENT_SCHEMA_VERSION = 1
DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
PROTECTED_SPLITS = frozenset({"test", "challenge"})
DEFAULT_PADDING_SECONDS = (0.0, 1.0, 2.0, 3.0)
OBJECTIVE_WEIGHTS = {
    "eventF1": 0.55,
    "timeIoU": 0.30,
    "liveTimeRecall": 0.15,
}
DEFAULT_OBJECTIVE_MARGIN = 0.01
DEFAULT_SIGN_CONSISTENCY = 0.75


class FeatureExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class InnerFold:
    validation_group: str
    training_groups: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "validationSourceGroup": self.validation_group,
            "trainingSourceGroups": list(self.training_groups),
        }


@dataclass(frozen=True)
class OuterFold:
    held_out_group: str
    training_groups: tuple[str, ...]
    inner_folds: tuple[InnerFold, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "heldOutSourceGroup": self.held_out_group,
            "trainingSourceGroups": list(self.training_groups),
            "innerFolds": [item.to_dict() for item in self.inner_folds],
        }


@dataclass(frozen=True)
class FeatureSet:
    name: str
    indexes: tuple[int, ...]
    families: tuple[str, ...]
    interpretation_subject: str

    def to_dict(self, feature_names: Sequence[str]) -> dict[str, Any]:
        return {
            "name": self.name,
            "featureCount": len(self.indexes),
            "families": list(self.families),
            "featureNames": [feature_names[index] for index in self.indexes],
            "interpretationSubject": self.interpretation_subject,
        }


@dataclass
class FoldArtifact:
    fold: OuterFold
    feature_set: FeatureSet
    model: LogisticModel
    decoder: DecoderConfig
    held_prepared: list[PreparedRecording]
    probabilities: list[np.ndarray]
    per_recording: list[dict[str, Any]]
    aggregate: dict[str, Any]
    inner_selection: dict[str, Any]
    epoch_cap: int
    model_fingerprint: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def objective(metrics: dict[str, Any]) -> float:
    return sum(float(metrics[key]) * weight for key, weight in OBJECTIVE_WEIGHTS.items())


def build_fold_plan(recordings: Sequence[Recording]) -> tuple[OuterFold, ...]:
    development = [item for item in recordings if item.split in DEVELOPMENT_SPLITS]
    groups = sorted({item.source_group for item in development})
    if len(groups) < 4:
        raise FeatureExperimentError(
            "feature experiments require at least four development source groups"
        )
    folds: list[OuterFold] = []
    for held_out in groups:
        training = tuple(item for item in groups if item != held_out)
        inner = tuple(
            InnerFold(
                validation_group=validation,
                training_groups=tuple(item for item in training if item != validation),
            )
            for validation in training
        )
        folds.append(
            OuterFold(
                held_out_group=held_out,
                training_groups=training,
                inner_folds=inner,
            )
        )
    return tuple(folds)


def build_feature_sets(feature_names: Sequence[str]) -> tuple[FeatureSet, ...]:
    grouped = grouped_feature_indexes(feature_names)
    unknown = set(grouped) - set(FEATURE_FAMILY_KINDS)
    if unknown:
        raise FeatureExperimentError(f"unknown feature families: {sorted(unknown)}")
    all_indexes = tuple(range(len(feature_names)))
    legacy_families = tuple(
        family for family in sorted(grouped) if FEATURE_FAMILY_KINDS[family] == "legacy"
    )
    added_families = tuple(
        family for family in sorted(grouped) if FEATURE_FAMILY_KINDS[family] == "added"
    )

    def indexes_for(families: Sequence[str]) -> tuple[int, ...]:
        selected = {index for family in families for index in grouped[family]}
        return tuple(index for index in all_indexes if index in selected)

    if not legacy_families or not added_families:
        raise FeatureExperimentError("both legacy and added feature families are required")
    result = [
        FeatureSet("full", all_indexes, tuple(sorted(grouped)), "all features"),
        FeatureSet(
            "legacy_only",
            indexes_for(legacy_families),
            legacy_families,
            "added feature families collectively",
        ),
        FeatureSet(
            "added_only",
            indexes_for(added_families),
            added_families,
            "legacy feature families collectively",
        ),
    ]
    for family in sorted(grouped):
        retained = tuple(item for item in sorted(grouped) if item != family)
        result.append(
            FeatureSet(
                name=f"full_minus_{family}",
                indexes=indexes_for(retained),
                families=retained,
                interpretation_subject=family,
            )
        )
    if any(not item.indexes for item in result):
        raise FeatureExperimentError("a requested feature set contains no columns")
    return tuple(result)


def classify_paired_deltas(
    deltas: Sequence[float],
    *,
    margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
) -> dict[str, Any]:
    if not deltas:
        raise ValueError("at least one paired delta is required")
    if margin < 0 or not 0.5 < sign_consistency <= 1:
        raise ValueError("invalid classification margin or sign consistency")
    values = [float(item) for item in deltas]
    required = max(1, math.ceil(sign_consistency * len(values) - 1e-12))
    positive = sum(item > 0 for item in values)
    negative = sum(item < 0 for item in values)
    neutral = sum(abs(item) < margin for item in values)
    mean = float(np.mean(values))
    median = float(np.median(values))
    if mean >= margin and median >= margin and positive >= required:
        classification = "helpful"
    elif mean <= -margin and median <= -margin and negative >= required:
        classification = "harmful"
    elif abs(mean) < margin and neutral >= required:
        classification = "neutral"
    else:
        classification = "uncertain"
    return {
        "classification": classification,
        "margin": margin,
        "requiredConsistentGroups": required,
        "groups": len(values),
        "positiveGroups": positive,
        "negativeGroups": negative,
        "neutralGroups": neutral,
        "meanDelta": mean,
        "medianDelta": median,
        "minimumDelta": min(values),
        "maximumDelta": max(values),
    }


def _seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _prepared_for_groups(
    prepared: Sequence[PreparedRecording], groups: Sequence[str]
) -> list[PreparedRecording]:
    selected = set(groups)
    return [item for item in prepared if item.recording.source_group in selected]


def _subset_prepared(
    prepared: Sequence[PreparedRecording], indexes: Sequence[int]
) -> list[PreparedRecording]:
    selected = np.asarray(indexes, dtype=np.int64)
    return [
        replace(
            item,
            contextual_values=np.ascontiguousarray(item.contextual_values[:, selected]),
            contextual_names=tuple(item.contextual_names[index] for index in indexes),
        )
        for item in prepared
    ]


def _training_arrays(
    prepared: Sequence[PreparedRecording],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return (
        [item.contextual_values[item.sample_mask] for item in prepared],
        [item.labels[item.sample_mask] for item in prepared],
    )


def _fit(
    training: Sequence[PreparedRecording],
    validation: Sequence[PreparedRecording],
    *,
    feature_config: FeatureConfig,
    decoder: DecoderConfig,
    config: TrainingConfig,
    seed: int,
) -> LogisticModel:
    if not training:
        raise FeatureExperimentError("an experiment fold has no training recordings")
    train_values, train_labels = _training_arrays(training)
    validation_values, validation_labels = _training_arrays(validation)
    return train_logistic_model(
        train_values,
        train_labels,
        validation_values,
        validation_labels,
        feature_config,
        training[0].contextual_names,
        decoder,
        replace(config, seed=seed),
    )


class _FrozenProbabilityModel:
    def __init__(
        self,
        prepared: Sequence[PreparedRecording],
        probabilities: Sequence[np.ndarray],
    ) -> None:
        if len(prepared) != len(probabilities):
            raise ValueError("prepared recordings and probabilities must be aligned")
        self._probabilities = {
            id(item.contextual_values): value
            for item, value in zip(prepared, probabilities, strict=True)
        }

    def predict(self, values: np.ndarray) -> np.ndarray:
        try:
            return self._probabilities[id(values)]
        except KeyError as error:
            raise FeatureExperimentError("decoder selection requested unknown values") from error


def _select_decoder(
    prepared: Sequence[PreparedRecording],
    probabilities: Sequence[np.ndarray],
    base: DecoderConfig,
) -> tuple[DecoderConfig, dict[str, Any]]:
    proxy = _FrozenProbabilityModel(prepared, probabilities)
    return _tune_decoder(prepared, proxy, base)  # type: ignore[arg-type]


def _model_fingerprint(model: LogisticModel) -> str:
    digest = hashlib.sha256()
    for array in (model.mean, model.scale, model.weights):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(np.asarray([model.bias], dtype=np.float32).tobytes())
    digest.update("\0".join(model.feature_names).encode("utf-8"))
    digest.update(
        json.dumps(model.feature_config.to_dict(), sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()


def _inner_folds(
    fold: OuterFold, inner_fold_limit: int | None
) -> tuple[InnerFold, ...]:
    if inner_fold_limit is None or inner_fold_limit <= 0:
        return fold.inner_folds
    return fold.inner_folds[:inner_fold_limit]


def _run_candidate_fold(
    prepared: Sequence[PreparedRecording],
    fold: OuterFold,
    feature_set: FeatureSet,
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    decoder_config: DecoderConfig,
    inner_fold_limit: int | None,
) -> FoldArtifact:
    outer_training_full = _prepared_for_groups(prepared, fold.training_groups)
    held_full = _prepared_for_groups(prepared, (fold.held_out_group,))
    outer_training = _subset_prepared(outer_training_full, feature_set.indexes)
    held = _subset_prepared(held_full, feature_set.indexes)
    if not held:
        raise FeatureExperimentError(
            f"outer source group {fold.held_out_group!r} has no recordings"
        )

    inner_prepared: list[PreparedRecording] = []
    inner_probabilities: list[np.ndarray] = []
    inner_summaries: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    for inner in _inner_folds(fold, inner_fold_limit):
        inner_training = _prepared_for_groups(outer_training, inner.training_groups)
        inner_validation = _prepared_for_groups(
            outer_training, (inner.validation_group,)
        )
        inner_seed = _seed(
            training_config.seed,
            "inner",
            fold.held_out_group,
            inner.validation_group,
        )
        model = _fit(
            inner_training,
            inner_validation,
            feature_config=feature_config,
            decoder=decoder_config,
            config=training_config,
            seed=inner_seed,
        )
        probabilities = [model.predict(item.contextual_values) for item in inner_validation]
        inner_prepared.extend(inner_validation)
        inner_probabilities.extend(probabilities)
        best_epoch = int(model.training_summary["bestEpoch"])
        best_epochs.append(best_epoch)
        inner_summaries.append(
            {
                **inner.to_dict(),
                "seed": inner_seed,
                "bestEpoch": best_epoch,
                "epochsCompleted": int(model.training_summary["epochsCompleted"]),
                "bestValidationLoss": float(
                    model.training_summary["bestValidationLoss"]
                ),
                "modelFingerprint": _model_fingerprint(model),
            }
        )
    if not inner_prepared:
        raise FeatureExperimentError("nested selection produced no validation predictions")
    decoder, decoder_selection = _select_decoder(
        inner_prepared, inner_probabilities, decoder_config
    )
    epoch_cap = max(1, int(round(float(np.median(best_epochs)))))
    refit_config = replace(
        training_config,
        epochs=epoch_cap,
        patience=max(training_config.patience, epoch_cap + 1),
    )
    outer_seed = _seed(training_config.seed, "outer-refit", fold.held_out_group)
    model = _fit(
        outer_training,
        (),
        feature_config=feature_config,
        decoder=decoder,
        config=refit_config,
        seed=outer_seed,
    )
    model.decoder = decoder
    model.training_summary.update(
        {
            "experimentCheckpointSelection": (
                "minimum training loss up to the median inner-fold best-epoch cap"
            ),
            "innerBestEpochs": best_epochs,
            "epochCap": epoch_cap,
            "outerTrainingSourceGroups": list(fold.training_groups),
            "outerHeldOutSourceGroup": fold.held_out_group,
        }
    )
    probabilities = [model.predict(item.contextual_values) for item in held]
    per_recording, aggregate = _evaluate_prepared_probabilities(
        held, probabilities, decoder
    )
    for row, item in zip(per_recording, held, strict=True):
        row["sourceGroup"] = item.recording.source_group
    aggregate["objective"] = objective(aggregate)
    return FoldArtifact(
        fold=fold,
        feature_set=feature_set,
        model=model,
        decoder=decoder,
        held_prepared=held,
        probabilities=probabilities,
        per_recording=per_recording,
        aggregate=aggregate,
        inner_selection={
            "foldsUsed": inner_summaries,
            "bestEpochs": best_epochs,
            "selectedEpochCap": epoch_cap,
            "decoder": decoder_selection,
        },
        epoch_cap=epoch_cap,
        model_fingerprint=_model_fingerprint(model),
    )


def _aggregate_artifacts(artifacts: Sequence[FoldArtifact]) -> dict[str, Any]:
    per_recording = [row for artifact in artifacts for row in artifact.per_recording]
    aggregate = aggregate_evaluations(per_recording)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [row["outcomeSlices"] for row in per_recording]
    )
    aggregate["objective"] = objective(aggregate)
    group_metrics = {
        artifact.fold.held_out_group: artifact.aggregate for artifact in artifacts
    }
    return {
        "aggregate": aggregate,
        "macroSourceGroup": {
            "sourceGroups": len(group_metrics),
            "objective": float(
                np.mean([float(item["objective"]) for item in group_metrics.values()])
            ),
            "eventF1": float(
                np.mean([float(item["eventF1"]) for item in group_metrics.values()])
            ),
            "timeIoU": float(
                np.mean([float(item["timeIoU"]) for item in group_metrics.values()])
            ),
            "liveTimeRecall": float(
                np.mean(
                    [float(item["liveTimeRecall"]) for item in group_metrics.values()]
                )
            ),
        },
        "bySourceGroup": group_metrics,
        "recordings": per_recording,
    }


def _fold_report(artifact: FoldArtifact) -> dict[str, Any]:
    return {
        **artifact.fold.to_dict(),
        "featureCount": len(artifact.feature_set.indexes),
        "innerSelection": artifact.inner_selection,
        "outerRefit": {
            "seed": int(artifact.model.training_summary["seed"]),
            "epochCap": artifact.epoch_cap,
            "selectedCheckpointEpoch": int(
                artifact.model.training_summary["bestEpoch"]
            ),
            "modelFingerprint": artifact.model_fingerprint,
        },
        "decoder": artifact.decoder.to_dict(),
        "metrics": artifact.aggregate,
        "recordings": artifact.per_recording,
    }


def circular_shift_family(
    prepared: PreparedRecording,
    family: str,
    shift_samples: int,
    feature_config: FeatureConfig,
) -> PreparedRecording:
    grouped = grouped_feature_indexes(prepared.sequence.names)
    if family not in grouped:
        raise ValueError(f"feature family {family!r} is not present")
    if not len(prepared.sequence.times):
        raise ValueError("cannot permute an empty feature sequence")
    shift = int(shift_samples) % len(prepared.sequence.times)
    values = prepared.sequence.values.copy()
    indexes = np.asarray(grouped[family], dtype=np.int64)
    values[:, indexes] = np.roll(values[:, indexes], shift, axis=0)
    sequence = FeatureSequence(
        times=prepared.sequence.times,
        values=values,
        names=prepared.sequence.names,
        metadata=prepared.sequence.metadata,
    )
    contextual_values, contextual_names = contextualize(sequence, feature_config)
    if contextual_names != prepared.contextual_names:
        raise FeatureExperimentError("permutation changed the feature signature")
    return replace(
        prepared,
        sequence=sequence,
        contextual_values=contextual_values,
    )


def circular_shift_base_feature(
    prepared: PreparedRecording,
    name: str,
    shift_samples: int,
    feature_config: FeatureConfig,
) -> PreparedRecording:
    """Shift one base channel and rebuild all of its temporal context columns."""
    try:
        index = prepared.sequence.names.index(name)
    except ValueError as error:
        raise ValueError(f"base feature {name!r} is not present") from error
    if not len(prepared.sequence.times):
        raise ValueError("cannot permute an empty feature sequence")
    shift = int(shift_samples) % len(prepared.sequence.times)
    values = prepared.sequence.values.copy()
    values[:, index] = np.roll(values[:, index], shift)
    sequence = FeatureSequence(
        times=prepared.sequence.times,
        values=values,
        names=prepared.sequence.names,
        metadata=prepared.sequence.metadata,
    )
    contextual_values, contextual_names = contextualize(sequence, feature_config)
    if contextual_names != prepared.contextual_names:
        raise FeatureExperimentError("permutation changed the feature signature")
    return replace(
        prepared,
        sequence=sequence,
        contextual_values=contextual_values,
    )


def _random_shift(
    item: PreparedRecording,
    *,
    minimum_shift_seconds: float,
    rng: np.random.Generator,
) -> int:
    count = len(item.sequence.times)
    if count <= 1:
        return 0
    if count > 1:
        step = float(np.median(np.diff(item.sequence.times)))
    else:
        step = 1.0
    minimum = max(1, int(round(minimum_shift_seconds / max(step, 1e-9))))
    if count > 2 * minimum:
        return int(rng.integers(minimum, count - minimum + 1))
    return int(rng.integers(1, count))


def _permutation_importance(
    full_artifacts: Sequence[FoldArtifact],
    *,
    feature_config: FeatureConfig,
    repeats: int,
    seed: int,
    minimum_shift_seconds: float,
    margin: float,
    sign_consistency: float,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("permutation repeats must be positive")
    families = sorted(grouped_feature_indexes(full_artifacts[0].model.feature_names))
    report: dict[str, Any] = {}
    for family in families:
        fold_rows: list[dict[str, Any]] = []
        for artifact in full_artifacts:
            repeat_rows: list[dict[str, Any]] = []
            for repeat_index in range(repeats):
                permuted: list[PreparedRecording] = []
                shifts: dict[str, int] = {}
                for item in artifact.held_prepared:
                    local_seed = _seed(
                        seed,
                        "permutation",
                        artifact.fold.held_out_group,
                        family,
                        str(repeat_index),
                        item.recording.id,
                    )
                    shift = _random_shift(
                        item,
                        minimum_shift_seconds=minimum_shift_seconds,
                        rng=np.random.default_rng(local_seed),
                    )
                    shifts[item.recording.id] = shift
                    permuted.append(
                        circular_shift_family(item, family, shift, feature_config)
                    )
                probabilities = [
                    artifact.model.predict(item.contextual_values) for item in permuted
                ]
                metrics = _evaluate_prepared_probabilities_for_selection(
                    permuted, probabilities, artifact.decoder
                )
                permuted_objective = objective(metrics)
                repeat_rows.append(
                    {
                        "repeat": repeat_index,
                        "shiftsSamples": shifts,
                        "objective": permuted_objective,
                        "deltaObjectiveBaselineMinusPermuted": float(
                            artifact.aggregate["objective"] - permuted_objective
                        ),
                        "eventF1": float(metrics["eventF1"]),
                        "timeIoU": float(metrics["timeIoU"]),
                        "liveTimeRecall": float(metrics["liveTimeRecall"]),
                    }
                )
            mean_delta = float(
                np.mean(
                    [
                        item["deltaObjectiveBaselineMinusPermuted"]
                        for item in repeat_rows
                    ]
                )
            )
            fold_rows.append(
                {
                    "heldOutSourceGroup": artifact.fold.held_out_group,
                    "baselineObjective": float(artifact.aggregate["objective"]),
                    "meanDeltaObjective": mean_delta,
                    "repeats": repeat_rows,
                }
            )
        fold_deltas = [float(item["meanDeltaObjective"]) for item in fold_rows]
        report[family] = {
            "family": family,
            "method": (
                "joint within-recording circular shift of all base family channels; "
                "context is rebuilt; fitted model and decoder are frozen"
            ),
            "classification": classify_paired_deltas(
                fold_deltas, margin=margin, sign_consistency=sign_consistency
            ),
            "meanDeltaObjectiveAcrossGroups": float(np.mean(fold_deltas)),
            "folds": fold_rows,
        }
    return report


def _base_feature_permutation_importance(
    full_artifacts: Sequence[FoldArtifact],
    *,
    feature_config: FeatureConfig,
    repeats: int,
    seed: int,
    minimum_shift_seconds: float,
    margin: float,
    sign_consistency: float,
) -> dict[str, Any]:
    """Rank every base signal by held-group circular-shift importance."""
    if repeats < 1:
        raise ValueError("permutation repeats must be positive")
    if not full_artifacts or any(
        item.feature_set.name != "full" for item in full_artifacts
    ):
        raise FeatureExperimentError(
            "base-feature importance requires full-model outer-fold artifacts"
        )
    names = full_artifacts[0].held_prepared[0].sequence.names
    if any(
        item.sequence.names != names
        for artifact in full_artifacts
        for item in artifact.held_prepared
    ):
        raise FeatureExperimentError("base-feature signatures differ across folds")
    by_feature: dict[str, Any] = {}
    for name in names:
        fold_rows: list[dict[str, Any]] = []
        for artifact in full_artifacts:
            repeat_rows: list[dict[str, Any]] = []
            for repeat_index in range(repeats):
                permuted: list[PreparedRecording] = []
                shifts: dict[str, int] = {}
                for item in artifact.held_prepared:
                    local_seed = _seed(
                        seed,
                        "base-permutation",
                        artifact.fold.held_out_group,
                        name,
                        str(repeat_index),
                        item.recording.id,
                    )
                    shift = _random_shift(
                        item,
                        minimum_shift_seconds=minimum_shift_seconds,
                        rng=np.random.default_rng(local_seed),
                    )
                    shifts[item.recording.id] = shift
                    permuted.append(
                        circular_shift_base_feature(
                            item, name, shift, feature_config
                        )
                    )
                probabilities = [
                    artifact.model.predict(item.contextual_values)
                    for item in permuted
                ]
                metrics = _evaluate_prepared_probabilities_for_selection(
                    permuted, probabilities, artifact.decoder
                )
                permuted_objective = objective(metrics)
                repeat_rows.append(
                    {
                        "repeat": repeat_index,
                        "shiftsSamples": shifts,
                        "objective": permuted_objective,
                        "deltaObjectiveBaselineMinusPermuted": float(
                            artifact.aggregate["objective"] - permuted_objective
                        ),
                    }
                )
            fold_rows.append(
                {
                    "heldOutSourceGroup": artifact.fold.held_out_group,
                    "baselineObjective": float(artifact.aggregate["objective"]),
                    "meanDeltaObjective": float(
                        np.mean(
                            [
                                item["deltaObjectiveBaselineMinusPermuted"]
                                for item in repeat_rows
                            ]
                        )
                    ),
                    "repeats": repeat_rows,
                }
            )
        deltas = [float(item["meanDeltaObjective"]) for item in fold_rows]
        by_feature[name] = {
            "feature": name,
            "family": feature_family(name),
            "classification": classify_paired_deltas(
                deltas, margin=margin, sign_consistency=sign_consistency
            ),
            "meanDeltaObjectiveAcrossGroups": float(np.mean(deltas)),
            "medianDeltaObjectiveAcrossGroups": float(np.median(deltas)),
            "folds": fold_rows,
        }
    ranking = sorted(
        (
            {
                "rank": 0,
                "feature": name,
                "family": row["family"],
                "classification": row["classification"]["classification"],
                "meanDeltaObjectiveAcrossGroups": row[
                    "meanDeltaObjectiveAcrossGroups"
                ],
                "medianDeltaObjectiveAcrossGroups": row[
                    "medianDeltaObjectiveAcrossGroups"
                ],
            }
            for name, row in by_feature.items()
        ),
        key=lambda row: (
            -float(row["meanDeltaObjectiveAcrossGroups"]), str(row["feature"])
        ),
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return {
        "method": (
            "within-recording circular shift of one base channel, rebuilding all "
            "temporal offsets while freezing each outer-fold full model and decoder"
        ),
        "interpretation": (
            "Supporting predictive importance, not causal ablation; correlated or "
            "deterministically related channels can mask or exaggerate one another."
        ),
        "features": len(names),
        "ranking": ranking,
        "byFeature": by_feature,
    }


def _standardized_coefficient_profiles(
    full_artifacts: Sequence[FoldArtifact],
) -> dict[str, Any]:
    """Summarize standardized logistic weights by base signal and outer fold."""
    if not full_artifacts:
        raise FeatureExperimentError("coefficient profiles require full artifacts")
    contextual_names = full_artifacts[0].model.feature_names
    base_names = full_artifacts[0].held_prepared[0].sequence.names
    indexes = {
        name: tuple(
            index
            for index, contextual_name in enumerate(contextual_names)
            if base_feature_name(contextual_name) == name
        )
        for name in base_names
    }
    if any(not selected for selected in indexes.values()):
        raise FeatureExperimentError("a base feature has no contextual coefficient")
    by_feature: dict[str, Any] = {}
    for name in base_names:
        folds: list[dict[str, Any]] = []
        selected = np.asarray(indexes[name], dtype=np.int64)
        for artifact in full_artifacts:
            weights = artifact.model.weights[selected].astype(np.float64)
            folds.append(
                {
                    "heldOutSourceGroup": artifact.fold.held_out_group,
                    "signedWeightsByContext": {
                        contextual_names[index]: float(artifact.model.weights[index])
                        for index in indexes[name]
                    },
                    "signedSum": float(np.sum(weights)),
                    "l1": float(np.sum(np.abs(weights))),
                    "l2": float(np.linalg.norm(weights)),
                    "maximumAbsolute": float(np.max(np.abs(weights))),
                }
            )
        by_feature[name] = {
            "feature": name,
            "family": feature_family(name),
            "medianL1AcrossGroups": float(np.median([row["l1"] for row in folds])),
            "medianL2AcrossGroups": float(np.median([row["l2"] for row in folds])),
            "medianMaximumAbsoluteAcrossGroups": float(
                np.median([row["maximumAbsolute"] for row in folds])
            ),
            "folds": folds,
        }
    ranking = sorted(
        (
            {
                "rank": 0,
                "feature": name,
                "family": row["family"],
                "medianL1AcrossGroups": row["medianL1AcrossGroups"],
                "medianL2AcrossGroups": row["medianL2AcrossGroups"],
            }
            for name, row in by_feature.items()
        ),
        key=lambda row: (-float(row["medianL1AcrossGroups"]), str(row["feature"])),
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return {
        "method": (
            "absolute coefficients in each outer logistic model after model-standardized "
            "inputs, grouped over temporal context offsets"
        ),
        "interpretation": (
            "Supporting evidence only; correlated channels can divide or cancel weights, "
            "so coefficients do not determine helpful/harmful classifications."
        ),
        "ranking": ranking,
        "byFeature": by_feature,
    }


def _scored_predictions(
    item: PreparedRecording,
    probabilities: np.ndarray,
    decoder: DecoderConfig,
) -> tuple[Interval, ...]:
    analysis_fps = (
        1.0 / float(np.median(np.diff(item.sequence.times)))
        if len(item.sequence.times) > 1
        else 1.0
    )
    decoded, _ = decode_probabilities(
        item.sequence.times,
        probabilities,
        item.sequence.metadata.duration,
        decoder,
        analysis_fps,
    )
    predictions = [Interval(item.start, item.end) for item in decoded]
    for ignored in item.recording.ignored_intervals:
        fragments: list[Interval] = []
        for prediction in predictions:
            if prediction.end <= ignored.start or prediction.start >= ignored.end:
                fragments.append(prediction)
            else:
                if prediction.start < ignored.start:
                    fragments.append(Interval(prediction.start, ignored.start))
                if prediction.end > ignored.end:
                    fragments.append(Interval(ignored.end, prediction.end))
        predictions = fragments
    return tuple(predictions)


def _padding_report(
    artifacts: Sequence[FoldArtifact], padding_seconds: Sequence[float]
) -> dict[str, Any]:
    recordings: list[RecordingIntervals] = []
    source_groups: dict[str, str] = {}
    predictions_report: list[dict[str, Any]] = []
    for artifact in artifacts:
        for item, probabilities in zip(
            artifact.held_prepared, artifact.probabilities, strict=True
        ):
            predictions = _scored_predictions(item, probabilities, artifact.decoder)
            source_groups[item.recording.id] = item.recording.source_group
            recordings.append(
                RecordingIntervals(
                    id=item.recording.id,
                    split="development-oof",
                    duration=item.sequence.metadata.duration,
                    truth=item.recording.rallies,
                    predictions=predictions,
                )
            )
            predictions_report.append(
                {
                    "id": item.recording.id,
                    "sourceGroup": item.recording.source_group,
                    "outerFoldModelFingerprint": artifact.model_fingerprint,
                    "intervals": [item.to_dict() for item in predictions],
                }
            )
    by_group = {
        group: evaluate_crop_padding(
            [item for item in recordings if source_groups[item.id] == group],
            padding_seconds,
        )
        for group in sorted(set(source_groups.values()))
    }
    return {
        "paddingSecondsBeforeAndAfter": [float(item) for item in padding_seconds],
        "policy": {
            "symmetric": True,
            "clipToVideoBounds": True,
            "mergeTouchingOrOverlappingCrops": True,
            "selectionWarning": "Select padding from development OOF results only.",
        },
        "aggregate": evaluate_crop_padding(recordings, padding_seconds),
        "bySourceGroup": by_group,
        "oofPredictions": predictions_report,
    }


def _comparison_report(
    full: Sequence[FoldArtifact],
    candidate: Sequence[FoldArtifact],
    *,
    subject: str,
    margin: float,
    sign_consistency: float,
) -> dict[str, Any]:
    full_by_group = {item.fold.held_out_group: item for item in full}
    candidate_by_group = {item.fold.held_out_group: item for item in candidate}
    if set(full_by_group) != set(candidate_by_group):
        raise FeatureExperimentError("paired candidates do not share the same outer folds")
    rows: list[dict[str, Any]] = []
    for group in sorted(full_by_group):
        baseline = full_by_group[group].aggregate
        comparison = candidate_by_group[group].aggregate
        rows.append(
            {
                "sourceGroup": group,
                "fullObjective": float(baseline["objective"]),
                "candidateObjective": float(comparison["objective"]),
                "deltaObjectiveFullMinusCandidate": float(
                    baseline["objective"] - comparison["objective"]
                ),
                "deltaEventF1FullMinusCandidate": float(
                    baseline["eventF1"] - comparison["eventF1"]
                ),
                "deltaTimeIoUFullMinusCandidate": float(
                    baseline["timeIoU"] - comparison["timeIoU"]
                ),
                "deltaLiveTimeRecallFullMinusCandidate": float(
                    baseline["liveTimeRecall"] - comparison["liveTimeRecall"]
                ),
            }
        )
    deltas = [float(item["deltaObjectiveFullMinusCandidate"]) for item in rows]
    return {
        "interpretationSubject": subject,
        "deltaDirection": (
            "positive means the interpretation subject helps the full model; "
            "negative means its removal/absence performs better"
        ),
        "classification": classify_paired_deltas(
            deltas, margin=margin, sign_consistency=sign_consistency
        ),
        "pairedSourceGroups": rows,
    }


def _short_event_component_ablation(
    full: Sequence[FoldArtifact],
    *,
    margin: float,
    sign_consistency: float,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    per_recording: list[dict[str, Any]] = []
    for artifact in full:
        disabled = replace(
            artifact.decoder,
            short_event_min_seconds=artifact.decoder.min_live_seconds,
            short_event_threshold=1.0,
        )
        fold_recordings, metrics = _evaluate_prepared_probabilities(
            artifact.held_prepared, artifact.probabilities, disabled
        )
        for row, item in zip(
            fold_recordings, artifact.held_prepared, strict=True
        ):
            row["sourceGroup"] = item.recording.source_group
        per_recording.extend(fold_recordings)
        ablated_objective = objective(metrics)
        fold_rows.append(
            {
                "sourceGroup": artifact.fold.held_out_group,
                "selectedDecoder": artifact.decoder.to_dict(),
                "disabledDecoder": disabled.to_dict(),
                "fullObjective": float(artifact.aggregate["objective"]),
                "disabledObjective": ablated_objective,
                "deltaObjectiveFullMinusDisabled": float(
                    artifact.aggregate["objective"] - ablated_objective
                ),
                "fullShortEventRecall": artifact.aggregate["outcomeSlices"].get(
                    "shortAtMost3Seconds", {}
                ),
                "disabledShortEventRecall": metrics["outcomeSlices"].get(
                    "shortAtMost3Seconds", {}
                ),
            }
        )
    aggregate = aggregate_evaluations(per_recording)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [item["outcomeSlices"] for item in per_recording]
    )
    aggregate["objective"] = objective(aggregate)
    deltas = [float(item["deltaObjectiveFullMinusDisabled"]) for item in fold_rows]
    return {
        "component": "short-event decoder path",
        "method": (
            "keep fitted probabilities frozen and set short_event_min_seconds to "
            "min_live_seconds with short_event_threshold=1.0"
        ),
        "classification": classify_paired_deltas(
            deltas, margin=margin, sign_consistency=sign_consistency
        ),
        "aggregateDisabled": aggregate,
        "pairedSourceGroups": fold_rows,
    }


def _protected_summary(recordings: Sequence[Recording]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in sorted(PROTECTED_SPLITS):
        selected = [item for item in recordings if item.split == split]
        result[split] = {
            "recordingIds": [item.id for item in selected],
            "sourceGroups": sorted({item.source_group for item in selected}),
            "labelsUsedForDevelopmentSelection": False,
        }
    return result


def code_provenance() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    tracked = (
        package / "feature_experiments.py",
        package / "feature_families.py",
        package / "features.py",
        package / "model.py",
        package / "pipeline.py",
        package / "metrics.py",
        package / "decoder.py",
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=package.parent,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        head = None
        dirty = None
    return {
        "gitHead": head,
        "gitDirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "filesSha256": {
            str(path.relative_to(package.parent)): sha256_file(path)
            for path in tracked
            if path.is_file()
        },
    }


def run_development_experiments(
    manifest: DatasetManifest,
    prepared: Sequence[PreparedRecording],
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    decoder_config: DecoderConfig,
    permutation_repeats: int = 10,
    padding_seconds: Sequence[float] = DEFAULT_PADDING_SECONDS,
    minimum_shift_seconds: float = 5.0,
    inner_fold_limit: int | None = None,
    objective_margin: float = DEFAULT_OBJECTIVE_MARGIN,
    sign_consistency: float = DEFAULT_SIGN_CONSISTENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    development_rows = [
        item for item in manifest.recordings if item.split in DEVELOPMENT_SPLITS
    ]
    expected_ids = {item.id for item in development_rows}
    prepared_ids = {item.recording.id for item in prepared}
    if prepared_ids != expected_ids:
        raise FeatureExperimentError(
            "prepared recordings must contain exactly train+validation development rows"
        )
    if any(item.recording.split not in DEVELOPMENT_SPLITS for item in prepared):
        raise FeatureExperimentError("protected test/challenge data entered development")
    if not prepared:
        raise FeatureExperimentError("development data is empty")
    signature = prepared[0].contextual_names
    if any(item.contextual_names != signature for item in prepared):
        raise FeatureExperimentError("development feature signatures differ")
    paddings = tuple(sorted({float(item) for item in padding_seconds}))
    if not paddings or paddings[0] < 0:
        raise ValueError("padding values must be non-negative")
    folds = build_fold_plan(manifest.recordings)
    feature_sets = build_feature_sets(signature)
    artifacts_by_candidate: dict[str, list[FoldArtifact]] = {}
    candidate_reports: dict[str, Any] = {}
    total = len(feature_sets) * len(folds)
    completed = 0
    for feature_set in feature_sets:
        artifacts: list[FoldArtifact] = []
        for fold in folds:
            completed += 1
            if progress is not None:
                progress(
                    f"Feature experiment {completed}/{total}: {feature_set.name}, "
                    f"hold out {fold.held_out_group}"
                )
            artifacts.append(
                _run_candidate_fold(
                    prepared,
                    fold,
                    feature_set,
                    feature_config=feature_config,
                    training_config=training_config,
                    decoder_config=decoder_config,
                    inner_fold_limit=inner_fold_limit,
                )
            )
        artifacts_by_candidate[feature_set.name] = artifacts
        candidate_reports[feature_set.name] = {
            "featureSet": feature_set.to_dict(signature),
            "oof": _aggregate_artifacts(artifacts),
            "outerFolds": [_fold_report(item) for item in artifacts],
        }

    full_artifacts = artifacts_by_candidate["full"]
    comparisons = {
        feature_set.name: _comparison_report(
            full_artifacts,
            artifacts_by_candidate[feature_set.name],
            subject=feature_set.interpretation_subject,
            margin=objective_margin,
            sign_consistency=sign_consistency,
        )
        for feature_set in feature_sets
        if feature_set.name != "full"
    }
    if progress is not None:
        progress("Computing grouped circular-shift permutation importance")
    permutation = _permutation_importance(
        full_artifacts,
        feature_config=feature_config,
        repeats=permutation_repeats,
        seed=training_config.seed,
        minimum_shift_seconds=minimum_shift_seconds,
        margin=objective_margin,
        sign_consistency=sign_consistency,
    )
    if progress is not None:
        progress("Computing per-signal circular-shift importance")
    base_feature_importance = _base_feature_permutation_importance(
        full_artifacts,
        feature_config=feature_config,
        repeats=permutation_repeats,
        seed=training_config.seed,
        minimum_shift_seconds=minimum_shift_seconds,
        margin=objective_margin,
        sign_consistency=sign_consistency,
    )
    coefficient_profiles = _standardized_coefficient_profiles(full_artifacts)
    family_assessments = {
        family: {
            "classification": comparisons[f"full_minus_{family}"]["classification"][
                "classification"
            ],
            "primaryEvidence": "retrained full-minus-family paired outer folds",
            "ablation": comparisons[f"full_minus_{family}"]["classification"],
            "permutation": permutation[family]["classification"],
            "interpretation": (
                "The headline classification follows retrained ablation. Permutation is "
                "supporting evidence and may be muted by correlated or recording-constant "
                "signals."
            ),
        }
        for family in sorted(grouped_feature_indexes(signature))
    }
    padding = _padding_report(full_artifacts, paddings)
    component_ablations = {
        "shortEventLogicDisabled": _short_event_component_ablation(
            full_artifacts,
            margin=objective_margin,
            sign_consistency=sign_consistency,
        )
    }
    grouped = grouped_feature_indexes(signature)
    return {
        "schemaVersion": EXPERIMENT_SCHEMA_VERSION,
        "kind": "volleycut-feature-experiment-development",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "development-only-source-group-out-of-fold-selection",
        "testLabelsUsed": False,
        "dataset": manifest.name,
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "manifestSnapshotSha256": _manifest_digest(manifest),
        "recordingContentSha256": {
            item.id: item.content_sha256 for item in manifest.recordings
        },
        "development": {
            "splits": sorted(DEVELOPMENT_SPLITS),
            "recordingIds": sorted(expected_ids),
            "sourceGroups": sorted(
                {item.source_group for item in development_rows}
            ),
        },
        "protected": _protected_summary(manifest.recordings),
        "featureVersion": FEATURE_VERSION,
        "selectedCandidateForFinalTest": "full",
        "featureConfig": feature_config.to_dict(),
        "featureSignature": {
            "count": len(signature),
            "names": list(signature),
            "families": {
                family: {
                    "kind": FEATURE_FAMILY_KINDS[family],
                    "featureCount": len(indexes),
                }
                for family, indexes in grouped.items()
            },
        },
        "trainingConfig": training_config.to_dict(),
        "baseDecoderConfig": decoder_config.to_dict(),
        "selectionProtocol": {
            "outer": "leave one development sourceGroup out",
            "inner": "leave one remaining sourceGroup out",
            "innerFoldLimit": inner_fold_limit,
            "epochRefit": (
                "median inner best epoch used as a cap; outer refit checkpoint uses "
                "training loss only and never outer labels"
            ),
            "decoder": "selected from pooled inner out-of-fold probabilities",
            "objective": OBJECTIVE_WEIGHTS,
            "objectiveMargin": objective_margin,
            "signConsistency": sign_consistency,
            "folds": [item.to_dict() for item in folds],
        },
        "candidates": candidate_reports,
        "pairedComparisonsAgainstFull": comparisons,
        "featureFamilyAssessments": family_assessments,
        "decoderComponentAblations": component_ablations,
        "permutationImportance": permutation,
        "baseFeaturePermutationImportance": base_feature_importance,
        "standardizedCoefficientProfiles": coefficient_profiles,
        "oofPadding": padding,
        "provenance": code_provenance(),
        "processing": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "featureCacheNote": "Superset features are prepared once and column-subset in memory.",
        },
        "limitations": [
            "Only four independent development source groups are available.",
            "Classification labels are practical evidence categories, not significance tests.",
            "Permutation cannot identify recording-constant signals within a recording.",
            "Individual signal permutation is predictive importance, not causal ablation.",
            "Correlated and derived channels can mask or exaggerate one another.",
            "Neutral means neutral for this corpus, target, and weighted-logistic architecture.",
        ],
    }


def run_fixed_split_final_test(
    manifest_path: str | Path,
    development_report_path: str | Path,
    cache_dir: str | Path,
    model_destination: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    report_path = Path(development_report_path).expanduser().resolve()
    try:
        development = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureExperimentError(
            f"cannot read development report {report_path}: {error}"
        ) from error
    if (
        development.get("schemaVersion") != EXPERIMENT_SCHEMA_VERSION
        or development.get("kind") != "volleycut-feature-experiment-development"
        or development.get("testLabelsUsed") is not False
    ):
        raise FeatureExperimentError("development report is not an unopened-test experiment")
    manifest = load_manifest(manifest_path)
    if development.get("manifestFileSha256") != sha256_file(manifest.path):
        raise FeatureExperimentError("development report does not match the supplied manifest")
    if development.get("manifestSnapshotSha256") != _manifest_digest(manifest):
        raise FeatureExperimentError(
            "recording snapshots changed after the development report was frozen"
        )
    if development.get("recordingContentSha256") != {
        item.id: item.content_sha256 for item in manifest.recordings
    }:
        raise FeatureExperimentError(
            "recording content identities changed after development"
        )
    if development.get("featureVersion") != FEATURE_VERSION:
        raise FeatureExperimentError("feature implementation version changed after development")
    if development.get("selectedCandidateForFinalTest") != "full":
        raise FeatureExperimentError("the frozen final candidate is not the full feature set")
    current_provenance = code_provenance()
    if development.get("provenance", {}).get("filesSha256") != current_provenance.get(
        "filesSha256"
    ):
        raise FeatureExperimentError("experiment code changed after development")
    feature_config = FeatureConfig.from_dict(development["featureConfig"])
    training_config = TrainingConfig(**development["trainingConfig"])
    decoder_config = DecoderConfig.from_dict(development["baseDecoderConfig"])
    paddings = tuple(
        float(item)
        for item in development["oofPadding"]["paddingSecondsBeforeAndAfter"]
    )
    trained = train_dataset(
        manifest.path,
        model_destination,
        cache_dir,
        feature_config=feature_config,
        training_config=training_config,
        decoder_config=decoder_config,
        progress=progress,
    )
    evaluated = evaluate_dataset(
        manifest.path,
        model_destination,
        cache_dir,
        split="test",
        progress=progress,
    )
    model = load_model(model_destination)
    expected_signature = tuple(development["featureSignature"]["names"])
    if model.feature_names != expected_signature:
        raise FeatureExperimentError(
            "trained final model does not match the frozen feature signature"
        )
    test_rows = manifest.for_split("test")
    prepared = (
        _prepare_many(test_rows, feature_config, cache_dir)
        if progress is None
        else _prepare_many(test_rows, feature_config, cache_dir, progress=progress)
    )
    probabilities = [model.predict(item.contextual_values) for item in prepared]
    padding_recordings = [
        RecordingIntervals(
            id=item.recording.id,
            split="test",
            duration=item.sequence.metadata.duration,
            truth=item.recording.rallies,
            predictions=_scored_predictions(item, values, model.decoder),
        )
        for item, values in zip(prepared, probabilities, strict=True)
    ]
    return {
        "schemaVersion": EXPERIMENT_SCHEMA_VERSION,
        "kind": "volleycut-feature-experiment-final-test",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "assessmentRole": "explicit-single-source-regression-test-access",
        "testLabelsOpened": True,
        "developmentReport": str(report_path),
        "developmentReportSha256": sha256_file(report_path),
        "manifest": str(manifest.path),
        "manifestFileSha256": sha256_file(manifest.path),
        "model": trained,
        "modelArtifactSha256": model.artifact_sha256,
        "test": evaluated,
        "testPadding": evaluate_crop_padding(padding_recordings, paddings),
        "provenance": current_provenance,
        "warning": (
            "This one indoor source group is a regression test, not an external "
            "generalization benchmark. Do not use it to revise features or thresholds."
        ),
    }
