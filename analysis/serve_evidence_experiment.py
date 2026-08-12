from __future__ import annotations

import json
import math
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
from .serve import (
    ServeCompositionConfig,
    ServeDecoderConfig,
    ServeDetection,
    decode_serve_probabilities,
    match_serve_contacts,
)
from .serve_experiment import (
    PairedPrediction,
    _interval_report,
    _prediction_inputs,
    _serve_metrics,
    _validate_model_pair,
)
from .stacked_serve_experiment import (
    _crossfit_training_scores,
    _ensure_new_model_destination,
    _metric_delta,
    _validate_live_validation,
)
from .version import __version__


EXPERIMENT_ID = "serve-evidence-gate-and-peak-window-v1"
MODEL_VARIANT = "rally-with-serve-peak-window-v1"
PEAK_RADIUS_SECONDS = 2.0
PEAK_FEATURE_NAMES = (
    "derived/serve_peak_within_2s",
    "derived/serve_peak_proximity_2s",
    "derived/serve_peak_signed_offset_2s",
    "derived/serve_peak_confidence_2s",
)


def serve_peak_features(
    times: np.ndarray,
    detections: Sequence[ServeDetection],
    radius_seconds: float = PEAK_RADIUS_SECONDS,
) -> np.ndarray:
    """Create deterministic per-sample features around decoded serve peaks."""
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("serve peak feature times must be finite and one-dimensional")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("serve peak feature times must be strictly increasing")
    if not math.isfinite(radius_seconds) or radius_seconds <= 0:
        raise ValueError("serve peak feature radius must be positive")
    result = np.zeros((len(times), len(PEAK_FEATURE_NAMES)), dtype=np.float32)
    best_distance = np.full(len(times), np.inf, dtype=np.float64)
    best_confidence = np.full(len(times), -np.inf, dtype=np.float64)
    best_time = np.full(len(times), np.inf, dtype=np.float64)
    for detection in detections:
        if (
            not math.isfinite(detection.time)
            or not math.isfinite(detection.confidence)
            or not 0.0 <= detection.confidence <= 1.0
        ):
            raise ValueError("serve detections must have finite times and bounded confidence")
        delta = times - detection.time
        distance = np.abs(delta)
        eligible = distance <= radius_seconds + 1e-9
        better = eligible & (
            (distance < best_distance - 1e-12)
            | (
                np.abs(distance - best_distance) <= 1e-12
            )
            & (
                (detection.confidence > best_confidence + 1e-12)
                | (
                    abs(detection.confidence - best_confidence) <= 1e-12
                )
                & (detection.time < best_time)
            )
        )
        if not np.any(better):
            continue
        best_distance[better] = distance[better]
        best_confidence[better] = detection.confidence
        best_time[better] = detection.time
        result[better, 0] = 1.0
        result[better, 1] = (1.0 - distance[better] / radius_seconds).astype(
            np.float32
        )
        result[better, 2] = (delta[better] / radius_seconds).astype(np.float32)
        result[better, 3] = np.float32(detection.confidence)
    return result


def _stack_peak_features(
    prepared: Sequence[PreparedRecording],
    serve_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    *,
    score_overrides: dict[str, np.ndarray] | None = None,
) -> list[PreparedRecording]:
    expected_ids = {item.recording.id for item in prepared}
    if score_overrides is not None and set(score_overrides) != expected_ids:
        missing = sorted(expected_ids - set(score_overrides))
        extra = sorted(set(score_overrides) - expected_ids)
        raise ModelError(
            "serve score overrides differ from prepared recordings; "
            f"missing={missing}, extra={extra}"
        )
    stacked: list[PreparedRecording] = []
    for item in prepared:
        if item.contextual_names != serve_model.feature_names:
            raise ModelError(f"feature signature mismatch for {item.recording.id}")
        scores = (
            score_overrides[item.recording.id]
            if score_overrides is not None
            else serve_model.predict(item.contextual_values)
        )
        if scores.shape != (len(item.contextual_values),):
            raise ModelError(f"serve score shape mismatch for {item.recording.id}")
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ModelError(f"serve scores are invalid for {item.recording.id}")
        detections = decode_serve_probabilities(
            item.sequence.times,
            scores,
            serve_decoder,
            duration=item.sequence.metadata.duration,
        )
        derived = serve_peak_features(item.sequence.times, detections)
        stacked.append(
            replace(
                item,
                contextual_values=np.column_stack(
                    (item.contextual_values, derived)
                ).astype(np.float32, copy=False),
                contextual_names=(*item.contextual_names, *PEAK_FEATURE_NAMES),
            )
        )
    return stacked


def _clamp_peak_features(
    prepared: Sequence[PreparedRecording], model: LogisticModel
) -> list[PreparedRecording]:
    count = len(PEAK_FEATURE_NAMES)
    clamped: list[PreparedRecording] = []
    for item in prepared:
        values = item.contextual_values.copy()
        values[:, -count:] = model.mean[-count:]
        clamped.append(replace(item, contextual_values=values))
    return clamped


def _gated_composition(composition: ServeCompositionConfig) -> ServeCompositionConfig:
    """Fixed requested ablation: +/-2 s association and no unconditional fallback."""
    return replace(
        composition,
        association_seconds=PEAK_RADIUS_SECONDS,
        fallback_seconds=0.0,
    )


def _prediction_triplet(
    prepared: Sequence[PreparedRecording],
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    composition: ServeCompositionConfig,
) -> tuple[list[PairedPrediction], list[PairedPrediction]]:
    original = [
        _prediction_inputs(item, rally_model, serve_model, serve_decoder, composition)
        for item in prepared
    ]
    gated_config = _gated_composition(composition)
    gated = [
        _prediction_inputs(item, rally_model, serve_model, serve_decoder, gated_config)
        for item in prepared
    ]
    return original, gated


def _evidence_audit(
    original: Sequence[PairedPrediction],
    gated: Sequence[PairedPrediction],
    composition: ServeCompositionConfig,
) -> dict[str, Any]:
    gated_config = _gated_composition(composition)
    counts: dict[str, dict[str, int]] = {
        "matchedWithin2Seconds": {},
        "unmatchedWithin2Seconds": {},
    }
    for source, gated_prediction in zip(original, gated, strict=True):
        recording = source.prepared.recording
        detections = tuple(
            detection
            for detection in source.serves
            if not any(
                ignored.start <= detection.time < ignored.end
                for ignored in recording.ignored_intervals
            )
        )
        matches = match_serve_contacts(
            [rally.start for rally in recording.rallies],
            detections,
            PEAK_RADIUS_SECONDS,
        )
        matched = {prediction_index for _, prediction_index, _ in matches}
        for index, detection in enumerate(detections):
            if any(
                interval.start - gated_config.association_seconds
                <= detection.time
                < interval.end
                for interval in source.primary
            ):
                category = "primaryAssociated"
            else:
                # A gated interval that does not overlap any original primary is a
                # permissive-evidence rescue. Unsupported peaks disappear entirely.
                category = (
                    "permissiveRescue"
                    if any(
                        abs(interval.start - detection.time) <= 1e-9
                        for interval in gated_prediction.composed
                    )
                    else "unsupportedDropped"
                )
            bucket_name = (
                "matchedWithin2Seconds"
                if index in matched
                else "unmatchedWithin2Seconds"
            )
            bucket = counts[bucket_name]
            bucket[category] = bucket.get(category, 0) + 1
    for bucket in counts.values():
        bucket["total"] = sum(bucket.values())
    return counts


def _validate_models(
    baseline_model: LogisticModel,
    serve_model: LogisticModel,
    control_model: LogisticModel,
    peak_model: LogisticModel,
    *,
    manifest_sha256: str | None = None,
) -> tuple[ServeDecoderConfig, ServeCompositionConfig]:
    serve_decoder, composition = _validate_model_pair(
        baseline_model, serve_model, manifest_sha256=manifest_sha256
    )
    if control_model.prediction_task != RALLY_LIVE_TASK:
        raise ModelError("peak-window control must predict rally-live state")
    if peak_model.prediction_task != STACKED_RALLY_TASK:
        raise ModelError("peak-window model must predict rally-live-stacked-serve")
    for name, model in (("control", control_model), ("peak-window", peak_model)):
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
        raise ModelError("peak-window control feature signature differs from the baseline")
    if peak_model.feature_names != (*baseline_model.feature_names, *PEAK_FEATURE_NAMES):
        raise ModelError("peak-window model feature signature is invalid")
    if peak_model.training_summary.get("serveModelSha256") != serve_model.artifact_sha256:
        raise ModelError("peak-window model was trained against a different serve specialist")
    if peak_model.training_summary.get("controlModelSha256") != control_model.artifact_sha256:
        raise ModelError("peak-window model was selected against a different control model")
    if peak_model.training_summary.get("modelVariant") != MODEL_VARIANT:
        raise ModelError("peak-window model records a different experiment variant")
    transform = peak_model.training_summary.get("peakFeatureTransform")
    if not isinstance(transform, dict):
        raise ModelError("peak-window model lacks feature-transform metadata")
    if transform.get("names") != list(PEAK_FEATURE_NAMES):
        raise ModelError("peak-window model records a different feature bundle")
    if transform.get("id") != "decoded-serve-peak-window-v1":
        raise ModelError("peak-window model records a different feature transform")
    try:
        recorded_radius = float(transform.get("radiusSeconds", -1.0))
    except (TypeError, ValueError) as error:
        raise ModelError("peak-window model records an invalid peak radius") from error
    if recorded_radius != PEAK_RADIUS_SECONDS:
        raise ModelError("peak-window model records a different peak radius")
    if transform.get("serveDecoder") != serve_decoder.to_dict():
        raise ModelError("peak-window model records a different serve decoder")
    if transform.get("trainingScoreProtocol") != "leave-one-training-source-group-out-v1":
        raise ModelError("peak-window model records a different training score protocol")
    if (
        transform.get("validationAndInferenceScoreProtocol")
        != "frozen-full-training-serve-specialist"
    ):
        raise ModelError("peak-window model records a different inference score protocol")
    folds = transform.get("crossFitFolds")
    training_groups = set(peak_model.training_summary.get("trainingSourceGroups", []))
    if not isinstance(folds, list) or not all(isinstance(fold, dict) for fold in folds):
        raise ModelError("peak-window model records invalid cross-fit folds")
    held_out_groups = {fold.get("heldOutSourceGroup") for fold in folds}
    if held_out_groups != training_groups:
        raise ModelError("peak-window model records invalid cross-fit fold coverage")
    for fold in folds:
        fold_training = fold.get("trainingSourceGroups")
        if not isinstance(fold_training, list) or not all(
            isinstance(group, str) for group in fold_training
        ):
            raise ModelError("peak-window model records invalid cross-fit training groups")
        if fold.get("heldOutSourceGroup") in set(fold_training):
            raise ModelError("peak-window model records a leaking cross-fit fold")
    if peak_model.training_summary.get("evidenceGate") != _gated_composition(
        composition
    ).to_dict():
        raise ModelError("peak-window model records a different evidence gate")
    if manifest_sha256 is not None and baseline_model.training_summary.get(
        "manifestSha256"
    ) != manifest_sha256:
        raise ModelError("peak-window model set differs from the immutable manifest")
    return serve_decoder, composition


def _slice_matches(metrics: dict[str, Any], name: str) -> int:
    row = metrics["outcomeSlices"][name]
    return round(float(row.get("strictMatchRecall", 0.0)) * int(row["rallies"]))


def _validation_decision(
    control: dict[str, Any],
    v4: dict[str, Any],
    gated: dict[str, Any],
    peak: dict[str, Any],
    clamped: dict[str, Any],
) -> dict[str, Any]:
    gate_checks = {
        "preservesOverallStrictMatches": gated["matchedRallies"] >= v4["matchedRallies"],
        "preservesShortStrictMatches": _slice_matches(
            gated, "shortAtMost3Seconds"
        )
        >= _slice_matches(v4, "shortAtMost3Seconds"),
        "preservesOrdinaryLongStrictMatches": _slice_matches(
            gated, "ordinaryLong"
        )
        >= _slice_matches(v4, "ordinaryLong"),
        "improvesEventF1": gated["eventF1"] > v4["eventF1"],
    }
    peak_checks = {
        "doesNotRegressEventF1": peak["eventF1"] >= control["eventF1"],
        "doesNotRegressLiveTimeRecall": peak["liveTimeRecall"] >= control["liveTimeRecall"],
        "preservesOrdinaryLongStrictMatches": _slice_matches(
            peak, "ordinaryLong"
        )
        >= _slice_matches(control, "ordinaryLong"),
        "gainsShortStrictMatch": _slice_matches(peak, "shortAtMost3Seconds")
        > _slice_matches(control, "shortAtMost3Seconds"),
        "dynamicBundleAddsStrictMatches": peak["matchedRallies"] > clamped["matchedRallies"],
    }
    return {
        "selectedOn": "validation",
        "gate": {
            "promote": all(gate_checks.values()),
            "checks": gate_checks,
        },
        "peakWindowRallyHead": {
            "promote": all(peak_checks.values()),
            "checks": peak_checks,
        },
        "testPolicy": "retrospective-regression-only; do not revise this decision from test",
    }


def _evaluate_variants(
    prepared: Sequence[PreparedRecording],
    peak_prepared: Sequence[PreparedRecording],
    baseline_model: LogisticModel,
    serve_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    composition: ServeCompositionConfig,
    control_model: LogisticModel,
    peak_model: LogisticModel,
) -> dict[str, Any]:
    original, gated = _prediction_triplet(
        prepared, baseline_model, serve_model, serve_decoder, composition
    )
    baseline_recordings, baseline = _evaluate_prepared(
        prepared, baseline_model, baseline_model.decoder
    )
    control_recordings, control = _evaluate_prepared(
        prepared, control_model, control_model.decoder
    )
    frozen_recordings, frozen = _evaluate_prepared(
        peak_prepared, peak_model, baseline_model.decoder
    )
    peak_recordings, peak = _evaluate_prepared(
        peak_prepared, peak_model, peak_model.decoder
    )
    clamped_prepared = _clamp_peak_features(peak_prepared, peak_model)
    clamped_recordings, clamped = _evaluate_prepared(
        clamped_prepared, peak_model, peak_model.decoder
    )
    return {
        "frozenBaseline": {"aggregate": baseline, "recordings": baseline_recordings},
        "matchedControl": {"aggregate": control, "recordings": control_recordings},
        "v4Composition": _interval_report(original, "composed"),
        "evidenceGatedComposition": _interval_report(gated, "composed"),
        "peakWindowFrozenDecoder": {
            "aggregate": frozen,
            "recordings": frozen_recordings,
            "decoder": baseline_model.decoder.to_dict(),
        },
        "peakWindowRally": {"aggregate": peak, "recordings": peak_recordings},
        "peakFeaturesClamped": {
            "aggregate": clamped,
            "recordings": clamped_recordings,
            "clampedRawValues": [float(value) for value in peak_model.mean[-4:]],
        },
        "serveSpottingAt2Seconds": _serve_metrics(original, PEAK_RADIUS_SECONDS),
        "peakEvidenceAudit": _evidence_audit(original, gated, composition),
    }


def _feature_summary(model: LogisticModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in PEAK_FEATURE_NAMES:
        index = model.feature_names.index(name)
        coefficient = float(model.weights[index])
        rows.append(
            {
                "name": name,
                "standardizedCoefficient": coefficient,
                "absoluteCoefficientRank": 1
                + int(np.sum(np.abs(model.weights) > abs(coefficient) + 1e-12)),
            }
        )
    return rows


def _build_report(
    prepared: Sequence[PreparedRecording],
    peak_prepared: Sequence[PreparedRecording],
    baseline_model: LogisticModel,
    serve_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    composition: ServeCompositionConfig,
    control_model: LogisticModel,
    peak_model: LogisticModel,
    *,
    dataset: str,
    manifest_sha256: str,
    split: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    variants = _evaluate_variants(
        prepared,
        peak_prepared,
        baseline_model,
        serve_model,
        serve_decoder,
        composition,
        control_model,
        peak_model,
    )
    v4 = variants["v4Composition"]["aggregate"]
    gated = variants["evidenceGatedComposition"]["aggregate"]
    peak = variants["peakWindowRally"]["aggregate"]
    control = variants["matchedControl"]["aggregate"]
    clamped = variants["peakFeaturesClamped"]["aggregate"]
    return {
        "schemaVersion": 1,
        "experiment": EXPERIMENT_ID,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "producer": f"volleycut-analysis/{__version__}",
        "dataset": dataset,
        "manifestSha256": manifest_sha256,
        "split": split,
        "assessmentRole": (
            "tuning-only"
            if split == "validation"
            else "retrospective-regression"
            if split == "test"
            else "held-out-evaluation"
        ),
        "matching": {"minimumIntervalIoU": 0.5, "serveToleranceSeconds": 2.0},
        "models": {
            "frozenBaseline": {"sha256": baseline_model.artifact_sha256},
            "frozenServe": {"sha256": serve_model.artifact_sha256},
            "matchedControl": {"sha256": control_model.artifact_sha256},
            "peakWindowRally": {"sha256": peak_model.artifact_sha256},
        },
        "evidenceGate": {
            "construction": (
                "Use the frozen v4 serve and permissive-live decoders, accept a peak up to two "
                "seconds before a predicted interval or anywhere inside it, and omit every "
                "unused peak that lacks permissive rally evidence."
            ),
            "composition": _gated_composition(composition).to_dict(),
            "notAFeature": "Gold +/-2-second match status is used only for evaluation.",
        },
        "peakFeatureTransform": peak_model.training_summary.get("peakFeatureTransform"),
        "peakFeatureCoefficients": _feature_summary(peak_model),
        "validationDecision": peak_model.training_summary.get("validationDecision"),
        **variants,
        "deltaGateVsV4": _metric_delta(gated, v4),
        "deltaPeakVsControl": _metric_delta(peak, control),
        "deltaPeakVsClamped": _metric_delta(peak, clamped),
        "processing": {
            "predictionAndDecodeWallClockSeconds": time.perf_counter() - started,
            "note": "Feature extraction and cache loading are excluded.",
        },
        "limitations": [
            "The validation source has already selected earlier rally and serve artifacts.",
            "The one-source test has been inspected repeatedly and is retrospective only.",
            (
                "The symmetric two-second feature window is suitable for offline cutting, "
                "not causal live inference."
            ),
            (
                "The composition association rule is not symmetric: it also accepts a peak "
                "anywhere inside an interval."
            ),
            (
                "Peak evidence comes from the same 450 audiovisual inputs and is not a new "
                "sensor modality."
            ),
        ],
    }


def _check_evaluation_groups(
    split: str,
    recordings: Sequence[Any],
    models: Sequence[LogisticModel],
) -> None:
    groups = {recording.source_group for recording in recordings}
    for model in models:
        protected = set(model.training_summary.get("trainingSourceGroups", []))
        if split != "validation":
            protected |= set(model.training_summary.get("validationSourceGroups", []))
        overlap = protected & groups
        if overlap:
            raise ModelError(
                f"evaluation split leaks trained/tuned source groups: {sorted(overlap)}"
            )


def train_serve_evidence_dataset(
    manifest_path: str | Path,
    baseline_model_path: str | Path,
    serve_model_path: str | Path,
    control_model_path: str | Path,
    model_destination: str | Path,
    cache_dir: str | Path,
    *,
    training_config: TrainingConfig | None = None,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    destination = _ensure_new_model_destination(model_destination)
    report_destination = Path(output_path).expanduser().resolve() if output_path else None
    if report_destination is not None and report_destination.exists():
        raise ModelError(f"evaluation output already exists: {report_destination}")
    if report_destination is not None and (
        report_destination == destination or destination in report_destination.parents
    ):
        raise ModelError("evaluation output must not be inside the model destination")
    started = time.perf_counter()
    baseline_model = load_model(baseline_model_path)
    serve_model = load_model(serve_model_path)
    control_model = load_model(control_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    serve_decoder, composition = _validate_model_pair(
        baseline_model, serve_model, manifest_sha256=manifest_sha256
    )
    if control_model.prediction_task != RALLY_LIVE_TASK:
        raise ModelError("peak-window control must predict rally-live state")
    if control_model.feature_version != baseline_model.feature_version:
        raise ModelError("peak-window control uses a different feature version")
    if control_model.feature_config != baseline_model.feature_config:
        raise ModelError("peak-window control uses a different feature configuration")
    if control_model.feature_names != baseline_model.feature_names:
        raise ModelError("peak-window control feature signature differs from baseline")
    if control_model.training_summary.get("manifestSha256") != manifest_sha256:
        raise ModelError("peak-window control was trained from a different manifest")
    if control_model.training_summary.get(
        "baselineRallyModelSha256"
    ) != baseline_model.artifact_sha256:
        raise ModelError("peak-window control was trained against a different baseline")
    train_rows = manifest.for_split("train")
    validation_rows = manifest.for_split("validation")
    if not train_rows or not validation_rows:
        raise ManifestError("peak-window training requires train and validation recordings")
    training = _prepare_many(
        train_rows, baseline_model.feature_config, cache_dir, progress=progress
    )
    validation = _prepare_many(
        validation_rows, baseline_model.feature_config, cache_dir, progress=progress
    )
    _validate_live_validation(validation)
    crossfit_scores, folds = _crossfit_training_scores(
        training, serve_model, progress=progress
    )
    peak_training = _stack_peak_features(
        training, serve_model, serve_decoder, score_overrides=crossfit_scores
    )
    peak_validation = _stack_peak_features(validation, serve_model, serve_decoder)
    if progress is not None:
        progress("Fitting rally head with decoded +/-2-second serve-peak features")
    peak_model = train_logistic_model(
        [item.contextual_values[item.sample_mask] for item in peak_training],
        [item.labels[item.sample_mask] for item in peak_training],
        [item.contextual_values[item.sample_mask] for item in peak_validation],
        [item.labels[item.sample_mask] for item in peak_validation],
        baseline_model.feature_config,
        peak_training[0].contextual_names,
        baseline_model.decoder,
        training_config or TrainingConfig(),
        prediction_task=STACKED_RALLY_TASK,
    )
    peak_model.feature_version = baseline_model.feature_version
    if progress is not None:
        progress("Selecting peak-window rally decoder on validation")
    peak_model.decoder, decoder_selection = _tune_decoder(
        peak_validation, peak_model, baseline_model.decoder
    )
    provisional = _evaluate_variants(
        validation,
        peak_validation,
        baseline_model,
        serve_model,
        serve_decoder,
        composition,
        control_model,
        peak_model,
    )
    decision = _validation_decision(
        provisional["matchedControl"]["aggregate"],
        provisional["v4Composition"]["aggregate"],
        provisional["evidenceGatedComposition"]["aggregate"],
        provisional["peakWindowRally"]["aggregate"],
        provisional["peakFeaturesClamped"]["aggregate"],
    )
    peak_model.training_summary.update(
        {
            "dataset": manifest.name,
            "manifestSha256": manifest_sha256,
            "modelVariant": MODEL_VARIANT,
            "baselineRallyModelSha256": baseline_model.artifact_sha256,
            "serveModelSha256": serve_model.artifact_sha256,
            "controlModelSha256": control_model.artifact_sha256,
            "trainingRecordingIds": [item.recording.id for item in training],
            "validationRecordingIds": [item.recording.id for item in validation],
            "trainingSourceGroups": sorted(
                {item.recording.source_group for item in training}
            ),
            "validationSourceGroups": sorted(
                {item.recording.source_group for item in validation}
            ),
            "recordingContentSha256": {
                item.recording.id: item.recording.content_sha256
                for item in (*training, *validation)
            },
            "peakFeatureTransform": {
                "id": "decoded-serve-peak-window-v1",
                "names": list(PEAK_FEATURE_NAMES),
                "radiusSeconds": PEAK_RADIUS_SECONDS,
                "serveDecoder": serve_decoder.to_dict(),
                "trainingScoreProtocol": "leave-one-training-source-group-out-v1",
                "validationAndInferenceScoreProtocol": "frozen-full-training-serve-specialist",
                "crossFitFolds": folds,
            },
            "evidenceGate": _gated_composition(composition).to_dict(),
            "decoderSelection": decoder_selection,
            "validationDecision": decision,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": __import__("cv2").__version__,
                "wallClockSeconds": round(time.perf_counter() - started, 3),
            },
        }
    )
    peak_model.save(destination)
    _validate_models(
        baseline_model,
        serve_model,
        control_model,
        peak_model,
        manifest_sha256=manifest_sha256,
    )
    report = _build_report(
        validation,
        peak_validation,
        baseline_model,
        serve_model,
        serve_decoder,
        composition,
        control_model,
        peak_model,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split="validation",
    )
    report["peakWindowModelPath"] = str(destination)
    if report_destination is not None:
        atomic_write_text(
            report_destination, json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def evaluate_serve_evidence_dataset(
    manifest_path: str | Path,
    baseline_model_path: str | Path,
    serve_model_path: str | Path,
    control_model_path: str | Path,
    peak_model_path: str | Path,
    cache_dir: str | Path,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    destination = Path(output_path).expanduser().resolve() if output_path else None
    if destination is not None and destination.exists():
        raise ModelError(f"evaluation output already exists: {destination}")
    baseline_model = load_model(baseline_model_path)
    serve_model = load_model(serve_model_path)
    control_model = load_model(control_model_path)
    peak_model = load_model(peak_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    serve_decoder, composition = _validate_models(
        baseline_model,
        serve_model,
        control_model,
        peak_model,
        manifest_sha256=manifest_sha256,
    )
    recordings = manifest.for_split(split)
    if not recordings:
        raise ManifestError(f"manifest has no recordings in split {split!r}")
    _check_evaluation_groups(
        split, recordings, (baseline_model, serve_model, control_model, peak_model)
    )
    prepared = _prepare_many(
        recordings, baseline_model.feature_config, cache_dir, progress=progress
    )
    peak_prepared = _stack_peak_features(prepared, serve_model, serve_decoder)
    report = _build_report(
        prepared,
        peak_prepared,
        baseline_model,
        serve_model,
        serve_decoder,
        composition,
        control_model,
        peak_model,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split=split,
    )
    if destination is not None:
        atomic_write_text(destination, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
