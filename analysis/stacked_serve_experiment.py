from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .artifacts import atomic_write_text
from .config import TrainingConfig
from .model import (
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    STACKED_RALLY_TASK,
    LogisticModel,
    ModelError,
    load_model,
    train_logistic_model,
)
from .pipeline import (
    PreparedRecording,
    _evaluate_prepared,
    _manifest_digest,
    _prepare_many,
    _tune_decoder,
)
from .schema import ManifestError, load_manifest
from .serve import serve_labels_for_times
from .serve_experiment import _validate_model_pair
from .version import __version__


STACKED_EXPERIMENT_ID = "rally-with-frozen-serve-feature-v1"
STACKED_FEATURE_NAME = "derived/serve_contact_probability"


def _ensure_new_model_destination(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ModelError(f"model destination is not an empty directory: {destination}")
    return destination


def _stack_prepared(
    prepared: Sequence[PreparedRecording],
    serve_model: LogisticModel,
    *,
    score_overrides: dict[str, np.ndarray] | None = None,
) -> list[PreparedRecording]:
    stacked: list[PreparedRecording] = []
    for item in prepared:
        if item.contextual_names != serve_model.feature_names:
            raise ModelError(f"feature signature mismatch for {item.recording.id}")
        serve_probability = (
            score_overrides[item.recording.id]
            if score_overrides is not None
            else serve_model.predict(item.contextual_values)
        )
        if serve_probability.shape != (len(item.contextual_values),):
            raise ModelError(f"serve score shape mismatch for {item.recording.id}")
        values = np.column_stack(
            (item.contextual_values, serve_probability)
        ).astype(np.float32, copy=False)
        stacked.append(
            replace(
                item,
                contextual_values=values,
                contextual_names=(*item.contextual_names, STACKED_FEATURE_NAME),
            )
        )
    return stacked


def _parameter_sha256(model: LogisticModel) -> str:
    digest = hashlib.sha256()
    for value in (model.mean, model.scale, model.weights):
        digest.update(np.asarray(value, dtype=np.float32).tobytes())
    digest.update(np.asarray([model.bias], dtype=np.float32).tobytes())
    return digest.hexdigest()


def _crossfit_training_scores(
    prepared: Sequence[PreparedRecording],
    serve_model: LogisticModel,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    groups = sorted({item.recording.source_group for item in prepared})
    if len(groups) < 2:
        raise ManifestError("serve-score cross-fitting requires at least two training source groups")
    try:
        best_epoch = int(serve_model.training_summary["bestEpoch"])
        target_radius = float(serve_model.training_summary["serveTarget"]["radiusSeconds"])
        source_config = dict(serve_model.training_summary["config"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError("serve specialist lacks cross-fitting metadata") from error
    source_config.update({"epochs": best_epoch, "patience": max(1, best_epoch)})
    fold_config = TrainingConfig(**source_config)
    scores: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []
    for index, held_out_group in enumerate(groups, start=1):
        fold_train = [
            item for item in prepared if item.recording.source_group != held_out_group
        ]
        fold_test = [
            item for item in prepared if item.recording.source_group == held_out_group
        ]
        if progress is not None:
            progress(
                f"Cross-fitting serve scores {index}/{len(groups)}: {held_out_group}"
            )
        labels = [
            serve_labels_for_times(
                item.sequence.times, item.recording.rallies, target_radius
            )[item.sample_mask]
            for item in fold_train
        ]
        fold_model = train_logistic_model(
            [item.contextual_values[item.sample_mask] for item in fold_train],
            labels,
            [],
            [],
            serve_model.feature_config,
            serve_model.feature_names,
            serve_model.decoder,
            fold_config,
            prediction_task=SERVE_CONTACT_TASK,
        )
        for item in fold_test:
            scores[item.recording.id] = fold_model.predict(item.contextual_values)
        summaries.append(
            {
                "heldOutSourceGroup": held_out_group,
                "trainingSourceGroups": sorted(
                    {item.recording.source_group for item in fold_train}
                ),
                "heldOutRecordingIds": [item.recording.id for item in fold_test],
                "parameterSha256": _parameter_sha256(fold_model),
                "bestEpochByTrainingLoss": fold_model.training_summary["bestEpoch"],
                "epochsCompleted": fold_model.training_summary["epochsCompleted"],
            }
        )
    if set(scores) != {item.recording.id for item in prepared}:
        raise ModelError("cross-fitted serve scores do not cover every training recording")
    return scores, summaries


def _validate_live_validation(prepared: Sequence[PreparedRecording]) -> None:
    live = sum(float(np.sum(item.labels[item.sample_mask] > 0.5)) for item in prepared)
    samples = sum(int(np.sum(item.sample_mask)) for item in prepared)
    if samples == 0 or live == 0 or live == samples:
        raise ManifestError("validation data must contain both live and dead samples")


def _validate_stacked_models(
    baseline_model: LogisticModel,
    serve_model: LogisticModel,
    control_model: LogisticModel,
    stacked_model: LogisticModel,
    *,
    manifest_sha256: str | None = None,
) -> None:
    _validate_model_pair(
        baseline_model, serve_model, manifest_sha256=manifest_sha256
    )
    if control_model.prediction_task != RALLY_LIVE_TASK:
        raise ModelError("stacking control must predict rally-live state")
    if stacked_model.prediction_task != STACKED_RALLY_TASK:
        raise ModelError("stacked model must predict rally-live-stacked-serve")
    for name, model in (("control", control_model), ("stacked", stacked_model)):
        if model.feature_version != baseline_model.feature_version:
            raise ModelError(f"{name} model uses a different feature version")
        if model.feature_config != baseline_model.feature_config:
            raise ModelError(f"{name} model uses a different feature configuration")
        if model.training_summary.get("manifestSha256") != baseline_model.training_summary.get(
            "manifestSha256"
        ):
            raise ModelError(f"{name} model was trained from a different manifest")
        if model.training_summary.get("baselineRallyModelSha256") != baseline_model.artifact_sha256:
            raise ModelError(f"{name} model was trained against a different baseline model")
    if control_model.feature_names != baseline_model.feature_names:
        raise ModelError("stacking control feature signature differs from the baseline")
    expected_stacked_names = (*baseline_model.feature_names, STACKED_FEATURE_NAME)
    if stacked_model.feature_names != expected_stacked_names:
        raise ModelError("stacked model feature signature is invalid")
    if stacked_model.training_summary.get("serveModelSha256") != serve_model.artifact_sha256:
        raise ModelError("stacked model was trained against a different serve specialist")
    if stacked_model.training_summary.get("controlModelSha256") != control_model.artifact_sha256:
        raise ModelError("stacked model was selected against a different control model")
    if manifest_sha256 is not None and baseline_model.training_summary.get(
        "manifestSha256"
    ) != manifest_sha256:
        raise ModelError("stacked model set differs from the immutable evaluation manifest")


def _metric_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "predictedRallies",
        "matchedRallies",
        "eventPrecision",
        "eventRecall",
        "eventF1",
        "timeIoU",
        "liveTimeRecall",
        "liveTimePrecision",
        "missedLiveSeconds",
        "deadSecondsRetained",
    )
    result = {field: candidate[field] - reference[field] for field in fields}
    result["outcomeStrictRecall"] = {}
    for name, candidate_slice in candidate["outcomeSlices"].items():
        reference_slice = reference["outcomeSlices"].get(name, {})
        candidate_recall = candidate_slice.get("strictMatchRecall")
        reference_recall = reference_slice.get("strictMatchRecall")
        result["outcomeStrictRecall"][name] = (
            candidate_recall - reference_recall
            if candidate_recall is not None and reference_recall is not None
            else None
        )
    return result


def _serve_feature_summary(model: LogisticModel) -> dict[str, Any]:
    index = model.feature_names.index(STACKED_FEATURE_NAME)
    coefficient = float(model.weights[index])
    rank = 1 + int(
        np.sum(np.abs(model.weights) > abs(coefficient) + 1e-12)
    )
    return {
        "name": STACKED_FEATURE_NAME,
        "construction": (
            "Frozen serve specialist probability evaluated on the same contextual feature row; "
            "this is a nonlinear derived feature, not an independently calibrated probability."
        ),
        "standardizedCoefficient": coefficient,
        "absoluteCoefficientRank": rank,
        "featureCount": len(model.feature_names),
        "baseFeatureCount": len(model.feature_names) - 1,
    }


def _build_report(
    prepared: Sequence[PreparedRecording],
    stacked_prepared: Sequence[PreparedRecording],
    baseline_model: LogisticModel,
    serve_model: LogisticModel,
    control_model: LogisticModel,
    stacked_model: LogisticModel,
    *,
    dataset: str,
    manifest_sha256: str,
    split: str,
) -> dict[str, Any]:
    baseline_recordings, baseline = _evaluate_prepared(
        prepared, baseline_model, baseline_model.decoder
    )
    control_recordings, control = _evaluate_prepared(
        prepared, control_model, control_model.decoder
    )
    frozen_decoder_recordings, frozen_decoder = _evaluate_prepared(
        stacked_prepared, stacked_model, baseline_model.decoder
    )
    stacked_recordings, stacked = _evaluate_prepared(
        stacked_prepared, stacked_model, stacked_model.decoder
    )
    clamped_prepared = [
        replace(
            item,
            contextual_values=np.column_stack(
                (
                    item.contextual_values[:, :-1],
                    np.full(
                        len(item.contextual_values),
                        stacked_model.mean[-1],
                        dtype=np.float32,
                    ),
                )
            ).astype(np.float32, copy=False),
        )
        for item in stacked_prepared
    ]
    clamped_recordings, clamped = _evaluate_prepared(
        clamped_prepared, stacked_model, stacked_model.decoder
    )
    return {
        "schemaVersion": 1,
        "experiment": STACKED_EXPERIMENT_ID,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "producer": f"volleycut-analysis/{__version__}",
        "dataset": dataset,
        "manifestSha256": manifest_sha256,
        "split": split,
        "assessmentRole": "tuning-only" if split == "validation" else "held-out-evaluation",
        "matching": {"minimumIntervalIoU": 0.5},
        "models": {
            "frozenBaseline": {
                "sha256": baseline_model.artifact_sha256,
                "task": baseline_model.prediction_task,
            },
            "frozenServe": {
                "sha256": serve_model.artifact_sha256,
                "task": serve_model.prediction_task,
            },
            "retrainedControl": {
                "sha256": control_model.artifact_sha256,
                "task": control_model.prediction_task,
            },
            "stackedRally": {
                "sha256": stacked_model.artifact_sha256,
                "task": stacked_model.prediction_task,
            },
        },
        "stackedFeature": _serve_feature_summary(stacked_model),
        "trainingScoreProtocol": stacked_model.training_summary.get("stackedFeature"),
        "frozenBaseline": {
            "aggregate": baseline,
            "recordings": baseline_recordings,
        },
        "retrainedControl": {
            "aggregate": control,
            "recordings": control_recordings,
        },
        "stackedFrozenDecoder": {
            "aggregate": frozen_decoder,
            "recordings": frozen_decoder_recordings,
            "decoder": baseline_model.decoder.to_dict(),
        },
        "stackedRally": {
            "aggregate": stacked,
            "recordings": stacked_recordings,
            "decoder": stacked_model.decoder.to_dict(),
        },
        "stackedServeFeatureClamped": {
            "aggregate": clamped,
            "recordings": clamped_recordings,
            "clampedRawValue": float(stacked_model.mean[-1]),
        },
        "deltaVsFrozenBaseline": _metric_delta(stacked, baseline),
        "deltaVsRetrainedControl": _metric_delta(stacked, control),
        "deltaVsClampedFeature": _metric_delta(stacked, clamped),
        "limitations": [
            (
                "Training rows use leave-one-source-group-out serve scores; validation and test "
                "use the frozen full-training serve specialist, so score calibration can differ."
            ),
            (
                "The validation source selected the earlier baseline and serve specialist, both "
                "new rally-head epochs, and both new rally decoders."
            ),
            "The one-source test split is a retrospective regression set inspected in prior work.",
            "The serve score is derived from the same 450 inputs and adds nonlinear capacity, not new sensor data.",
        ],
    }


def _training_metadata(
    prepared_train: Sequence[PreparedRecording],
    prepared_validation: Sequence[PreparedRecording],
    *,
    dataset: str,
    manifest_sha256: str,
    baseline_model: LogisticModel,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "manifestSha256": manifest_sha256,
        "baselineRallyModelSha256": baseline_model.artifact_sha256,
        "trainingRecordingIds": [item.recording.id for item in prepared_train],
        "validationRecordingIds": [item.recording.id for item in prepared_validation],
        "trainingSourceGroups": sorted(
            {item.recording.source_group for item in prepared_train}
        ),
        "validationSourceGroups": sorted(
            {item.recording.source_group for item in prepared_validation}
        ),
        "recordingContentSha256": {
            item.recording.id: item.recording.content_sha256
            for item in (*prepared_train, *prepared_validation)
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
            "wallClockSeconds": round(runtime_seconds, 3),
        },
    }


def train_stacked_rally_dataset(
    manifest_path: str | Path,
    baseline_model_path: str | Path,
    serve_model_path: str | Path,
    control_destination: str | Path,
    stacked_destination: str | Path,
    cache_dir: str | Path,
    *,
    training_config: TrainingConfig | None = None,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    control_target = _ensure_new_model_destination(control_destination)
    stacked_target = _ensure_new_model_destination(stacked_destination)
    if (
        control_target == stacked_target
        or control_target in stacked_target.parents
        or stacked_target in control_target.parents
    ):
        raise ModelError("control and stacked model destinations must be separate")
    report_target = Path(output_path).expanduser().resolve() if output_path else None
    if report_target is not None and report_target.exists():
        raise ModelError(f"evaluation output already exists: {report_target}")
    if report_target is not None and any(
        target == report_target or target in report_target.parents
        for target in (control_target, stacked_target)
    ):
        raise ModelError("evaluation output must not be inside a model destination")

    started = time.perf_counter()
    baseline_model = load_model(baseline_model_path)
    serve_model = load_model(serve_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    _validate_model_pair(
        baseline_model, serve_model, manifest_sha256=manifest_sha256
    )
    train_rows = manifest.for_split("train")
    validation_rows = manifest.for_split("validation")
    if not train_rows or not validation_rows:
        raise ManifestError("stacked training requires both train and validation recordings")
    training = _prepare_many(
        train_rows, baseline_model.feature_config, cache_dir, progress=progress
    )
    validation = _prepare_many(
        validation_rows, baseline_model.feature_config, cache_dir, progress=progress
    )
    for item in (*training, *validation):
        if item.contextual_names != baseline_model.feature_names:
            raise ModelError(f"feature signature mismatch for {item.recording.id}")
    _validate_live_validation(validation)
    crossfit_scores, crossfit_folds = _crossfit_training_scores(
        training, serve_model, progress=progress
    )
    stacked_training = _stack_prepared(
        training, serve_model, score_overrides=crossfit_scores
    )
    stacked_validation = _stack_prepared(validation, serve_model)
    config = training_config or TrainingConfig()

    if progress is not None:
        progress("Fitting matched base-feature rally control")
    control_model = train_logistic_model(
        [item.contextual_values[item.sample_mask] for item in training],
        [item.labels[item.sample_mask] for item in training],
        [item.contextual_values[item.sample_mask] for item in validation],
        [item.labels[item.sample_mask] for item in validation],
        baseline_model.feature_config,
        baseline_model.feature_names,
        baseline_model.decoder,
        config,
        prediction_task=RALLY_LIVE_TASK,
    )
    if progress is not None:
        progress("Fitting rally head with frozen serve probability feature")
    stacked_model = train_logistic_model(
        [item.contextual_values[item.sample_mask] for item in stacked_training],
        [item.labels[item.sample_mask] for item in stacked_training],
        [item.contextual_values[item.sample_mask] for item in stacked_validation],
        [item.labels[item.sample_mask] for item in stacked_validation],
        baseline_model.feature_config,
        stacked_training[0].contextual_names,
        baseline_model.decoder,
        config,
        prediction_task=STACKED_RALLY_TASK,
    )
    control_model.feature_version = baseline_model.feature_version
    stacked_model.feature_version = baseline_model.feature_version
    if progress is not None:
        progress("Selecting matched control decoder on validation")
    control_decoder, control_selection = _tune_decoder(
        validation, control_model, baseline_model.decoder
    )
    if progress is not None:
        progress("Selecting stacked rally decoder on validation")
    stacked_decoder, stacked_selection = _tune_decoder(
        stacked_validation, stacked_model, baseline_model.decoder
    )
    control_model.decoder = control_decoder
    stacked_model.decoder = stacked_decoder
    runtime_seconds = time.perf_counter() - started
    common = _training_metadata(
        training,
        validation,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        baseline_model=baseline_model,
        runtime_seconds=runtime_seconds,
    )
    control_model.training_summary.update(
        {
            **common,
            "modelVariant": "matched-base-feature-control-v1",
            "decoderSelection": control_selection,
        }
    )
    control_model.save(control_target)
    stacked_model.training_summary.update(
        {
            **common,
            "modelVariant": STACKED_EXPERIMENT_ID,
            "serveModelSha256": serve_model.artifact_sha256,
            "controlModelSha256": control_model.artifact_sha256,
            "stackedFeature": {
                "name": STACKED_FEATURE_NAME,
                "serveModelSha256": serve_model.artifact_sha256,
                "trainingScoreProtocol": "leave-one-training-source-group-out-v1",
                "validationAndInferenceScoreProtocol": "frozen-full-training-serve-specialist",
                "serveTarget": serve_model.training_summary.get("serveTarget"),
                "crossFitFolds": crossfit_folds,
            },
            "decoderSelection": stacked_selection,
        }
    )
    stacked_model.save(stacked_target)
    _validate_stacked_models(
        baseline_model,
        serve_model,
        control_model,
        stacked_model,
        manifest_sha256=manifest_sha256,
    )
    report = _build_report(
        validation,
        stacked_validation,
        baseline_model,
        serve_model,
        control_model,
        stacked_model,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split="validation",
    )
    report["controlModelPath"] = str(control_target)
    report["stackedModelPath"] = str(stacked_target)
    if report_target is not None:
        atomic_write_text(report_target, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def evaluate_stacked_rally_dataset(
    manifest_path: str | Path,
    baseline_model_path: str | Path,
    serve_model_path: str | Path,
    control_model_path: str | Path,
    stacked_model_path: str | Path,
    cache_dir: str | Path,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    report_target = Path(output_path).expanduser().resolve() if output_path else None
    if report_target is not None and report_target.exists():
        raise ModelError(f"evaluation output already exists: {report_target}")
    baseline_model = load_model(baseline_model_path)
    serve_model = load_model(serve_model_path)
    control_model = load_model(control_model_path)
    stacked_model = load_model(stacked_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    _validate_stacked_models(
        baseline_model,
        serve_model,
        control_model,
        stacked_model,
        manifest_sha256=manifest_sha256,
    )
    recordings = manifest.for_split(split)
    if not recordings:
        raise ManifestError(f"manifest has no recordings in split {split!r}")
    evaluation_groups = {recording.source_group for recording in recordings}
    for model in (baseline_model, serve_model, control_model, stacked_model):
        protected = set(model.training_summary.get("trainingSourceGroups", []))
        if split != "validation":
            protected |= set(model.training_summary.get("validationSourceGroups", []))
        overlap = protected & evaluation_groups
        if overlap:
            raise ModelError(
                f"evaluation split leaks trained/tuned source groups: {sorted(overlap)}"
            )
    prepared = _prepare_many(
        recordings, baseline_model.feature_config, cache_dir, progress=progress
    )
    stacked_prepared = _stack_prepared(prepared, serve_model)
    report = _build_report(
        prepared,
        stacked_prepared,
        baseline_model,
        serve_model,
        control_model,
        stacked_model,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split=split,
    )
    if report_target is not None:
        atomic_write_text(report_target, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
