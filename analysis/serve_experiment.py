from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .artifacts import atomic_write_text
from .config import DecoderConfig, TrainingConfig
from .decoder import DecodedInterval, decode_probabilities
from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    ordered_interval_matches,
    outcome_slice_metrics,
)
from .model import (
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
    ModelError,
    load_model,
    train_logistic_model,
)
from .pipeline import (
    PreparedRecording,
    _manifest_digest,
    _prepare_many,
    assessment_role_for_split,
)
from .schema import Interval, ManifestError, Recording, load_manifest
from .serve import (
    ServeCompositionConfig,
    ServeDecoderConfig,
    ServeDetection,
    compose_serve_anchored_intervals,
    decode_serve_probabilities,
    match_serve_contacts,
    serve_labels_for_times,
)
from .version import __version__


SERVE_TARGET_ID = "serve-contact-window-v1"
SHORT_RALLY_SECONDS = 3.0
FULL_SERVE_INPUT_PROFILE = "full"
NO_LEGACY_AUDIO_SERVE_INPUT_PROFILE = "visual-plus-normalized-band-audio"
NEW_AUDIO_ONLY_SERVE_INPUT_PROFILE = "normalized-band-audio-only"
SERVE_INPUT_PROFILES = {
    FULL_SERVE_INPUT_PROFILE,
    NO_LEGACY_AUDIO_SERVE_INPUT_PROFILE,
    NEW_AUDIO_ONLY_SERVE_INPUT_PROFILE,
}


def _is_normalized_band_audio(name: str) -> bool:
    base = name.split("/", 1)[-1]
    return base.startswith("audio_band_") or base in {
        "audio_noise_removed_broadband",
        "audio_noise_normalized_flux",
    }


def _serve_input_mask(
    feature_names: Sequence[str], profile: str
) -> np.ndarray:
    if profile not in SERVE_INPUT_PROFILES:
        raise ValueError(f"unsupported serve input profile: {profile!r}")
    normalized_audio = np.asarray(
        [_is_normalized_band_audio(name) for name in feature_names], dtype=np.bool_
    )
    if profile != FULL_SERVE_INPUT_PROFILE and not np.any(normalized_audio):
        raise ModelError(
            "normalized-band serve input profiles require normalized audio features"
        )
    if profile == FULL_SERVE_INPUT_PROFILE:
        return np.ones(len(feature_names), dtype=np.bool_)
    if profile == NEW_AUDIO_ONLY_SERVE_INPUT_PROFILE:
        return normalized_audio
    legacy_audio = np.asarray(
        [
            name.split("/", 1)[-1].startswith("audio_")
            and not normalized_audio[index]
            for index, name in enumerate(feature_names)
        ],
        dtype=np.bool_,
    )
    return ~legacy_audio


def _masked_serve_values(values: np.ndarray, retained: np.ndarray) -> np.ndarray:
    if retained.shape != (values.shape[1],):
        raise ModelError("serve input mask does not match the feature matrix")
    if np.all(retained):
        return values
    masked = values.copy()
    masked[:, ~retained] = 0.0
    return masked


@dataclass(frozen=True)
class PairedPrediction:
    prepared: PreparedRecording
    primary: tuple[DecodedInterval, ...]
    serves: tuple[ServeDetection, ...]
    composed: tuple[DecodedInterval, ...]


def _effective_analysis_fps(prepared: PreparedRecording) -> float:
    times = prepared.sequence.times
    return 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 1.0


def _validate_model_pair(
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    *,
    manifest_sha256: str | None = None,
) -> tuple[ServeDecoderConfig, ServeCompositionConfig]:
    if rally_model.prediction_task != RALLY_LIVE_TASK:
        raise ModelError("the primary model must predict rally-live state")
    if serve_model.prediction_task != SERVE_CONTACT_TASK:
        raise ModelError("the specialist model must predict serve-contact")
    if rally_model.feature_version != serve_model.feature_version:
        raise ModelError("rally and serve models use different feature versions")
    if rally_model.feature_config != serve_model.feature_config:
        raise ModelError("rally and serve models require different feature configurations")
    if rally_model.feature_names != serve_model.feature_names:
        raise ModelError("rally and serve models require different feature signatures")
    if serve_model.training_summary.get("rallyModelSha256") != rally_model.artifact_sha256:
        raise ModelError("serve specialist was selected against a different rally model")
    rally_manifest = rally_model.training_summary.get("manifestSha256")
    serve_manifest = serve_model.training_summary.get("manifestSha256")
    if rally_manifest != serve_manifest:
        raise ModelError("rally and serve models were trained from different manifests")
    if manifest_sha256 is not None and rally_manifest != manifest_sha256:
        raise ModelError("model pair differs from the immutable evaluation manifest")
    try:
        serve_decoder = ServeDecoderConfig.from_dict(
            serve_model.training_summary["serveDecoder"]
        )
        composition = ServeCompositionConfig.from_dict(
            serve_model.training_summary["composition"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(f"serve specialist selection metadata is invalid: {error}") from error
    return serve_decoder, composition


def _prediction_inputs(
    prepared: PreparedRecording,
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    composition: ServeCompositionConfig,
) -> PairedPrediction:
    if prepared.contextual_names != rally_model.feature_names:
        raise ModelError(f"feature signature mismatch for {prepared.recording.id}")
    live_probabilities = rally_model.predict(prepared.contextual_values)
    serve_probabilities = serve_model.predict(prepared.contextual_values)
    duration = prepared.sequence.metadata.duration
    fps = _effective_analysis_fps(prepared)
    primary, _ = decode_probabilities(
        prepared.sequence.times,
        live_probabilities,
        duration,
        rally_model.decoder,
        fps,
    )
    permissive, _ = decode_probabilities(
        prepared.sequence.times,
        live_probabilities,
        duration,
        composition.permissive_decoder,
        fps,
    )
    serves = decode_serve_probabilities(
        prepared.sequence.times,
        serve_probabilities,
        serve_decoder,
        duration=duration,
    )
    composed = compose_serve_anchored_intervals(
        primary,
        permissive,
        serves,
        duration,
        composition,
        sample_seconds=1.0 / fps,
    )
    return PairedPrediction(
        prepared=prepared,
        primary=tuple(primary),
        serves=tuple(serves),
        composed=tuple(composed),
    )


def _fast_matched_count(
    truth: Sequence[Interval], predictions: Sequence[Interval | DecodedInterval]
) -> int:
    """Fast exact count for ordinary disjoint interval sets used during selection."""
    if not truth or not predictions:
        return 0
    truth_start = np.asarray([item.start for item in truth], dtype=np.float64)[:, None]
    truth_end = np.asarray([item.end for item in truth], dtype=np.float64)[:, None]
    predicted_start = np.asarray([item.start for item in predictions], dtype=np.float64)[None, :]
    predicted_end = np.asarray([item.end for item in predictions], dtype=np.float64)[None, :]
    intersection = np.maximum(
        0.0,
        np.minimum(truth_end, predicted_end) - np.maximum(truth_start, predicted_start),
    )
    union = np.maximum(truth_end, predicted_end) - np.minimum(truth_start, predicted_start)
    eligible = intersection / np.maximum(union, 1e-12) >= 0.5
    if np.all(np.sum(eligible, axis=0) <= 1) and np.all(np.sum(eligible, axis=1) <= 1):
        return int(np.sum(eligible))
    return len(ordered_interval_matches(truth, predictions, 0.5))


def _short_truth(recording: Recording) -> tuple[Interval, ...]:
    return tuple(
        rally
        for rally in recording.rallies
        if rally.end - rally.start <= SHORT_RALLY_SECONDS + 1e-9
    )


def _validate_serve_validation_targets(labels: Sequence[np.ndarray]) -> None:
    samples = sum(len(item) for item in labels)
    positives = sum(int(np.sum(item > 0.5)) for item in labels)
    if samples == 0 or positives == 0 or positives == samples:
        raise ManifestError(
            "serve validation data must contain both contact-window and non-contact samples"
        )


def _selection_score(predictions: Sequence[PairedPrediction]) -> tuple[tuple[float, ...], dict[str, Any]]:
    true_count = predicted_count = matched_count = 0
    short_count = short_matched = 0
    true_seconds = predicted_seconds = intersection_seconds = 0.0
    for prediction in predictions:
        truth = prediction.prepared.recording.rallies
        short = _short_truth(prediction.prepared.recording)
        scored = _clip_ignored(
            prediction.composed, prediction.prepared.recording.ignored_intervals
        )
        true_count += len(truth)
        predicted_count += len(scored)
        matched_count += _fast_matched_count(truth, scored)
        short_count += len(short)
        short_matched += _fast_matched_count(short, scored)
        true_seconds += sum(item.end - item.start for item in truth)
        predicted_seconds += sum(item.end - item.start for item in scored)
        truth_index = prediction_index = 0
        while truth_index < len(truth) and prediction_index < len(scored):
            actual = truth[truth_index]
            proposed = scored[prediction_index]
            intersection_seconds += max(
                0.0, min(actual.end, proposed.end) - max(actual.start, proposed.start)
            )
            if actual.end <= proposed.end:
                truth_index += 1
            else:
                prediction_index += 1
    precision = matched_count / predicted_count if predicted_count else 0.0
    recall = matched_count / true_count if true_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    short_recall = short_matched / short_count if short_count else 1.0
    union_seconds = true_seconds + predicted_seconds - intersection_seconds
    time_iou = intersection_seconds / union_seconds if union_seconds else 1.0
    live_precision = (
        intersection_seconds / predicted_seconds
        if predicted_seconds
        else (1.0 if not true_seconds else 0.0)
    )
    dead_seconds = max(0.0, predicted_seconds - intersection_seconds)
    objective = 0.55 * short_recall + 0.35 * f1 + 0.10 * recall
    metrics = {
        "objectiveValue": objective,
        "shortStrictRecall": short_recall,
        "eventF1": f1,
        "eventRecall": recall,
        "eventPrecision": precision,
        "timeIoU": time_iou,
        "liveTimePrecision": live_precision,
        "deadSecondsRetained": dead_seconds,
        "trueRallies": true_count,
        "predictedRallies": predicted_count,
        "matchedRallies": matched_count,
        "shortRallies": short_count,
        "matchedShortRallies": short_matched,
    }
    tie_break = (
        objective,
        short_recall,
        f1,
        recall,
        precision,
        time_iou,
        live_precision,
        -dead_seconds,
        -float(predicted_count),
    )
    return tie_break, metrics


def _tune_composition(
    prepared: Sequence[PreparedRecording],
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[ServeDecoderConfig, ServeCompositionConfig, dict[str, Any]]:
    if not prepared:
        raise ManifestError("serve composition selection requires validation recordings")
    cached: list[tuple[PreparedRecording, np.ndarray, np.ndarray, tuple[DecodedInterval, ...]]] = []
    for item in prepared:
        live = rally_model.predict(item.contextual_values)
        serve = serve_model.predict(item.contextual_values)
        primary, _ = decode_probabilities(
            item.sequence.times,
            live,
            item.sequence.metadata.duration,
            rally_model.decoder,
            _effective_analysis_fps(item),
        )
        cached.append((item, live, serve, tuple(primary)))

    serve_configs = [
        ServeDecoderConfig(threshold, separation, offset)
        for threshold in (0.70, 0.75, 0.80, 0.85, 0.90)
        for separation in (6.0, 8.0, 10.0)
        for offset in (-0.25, 0.0, 0.25)
    ]
    permissive_configs = [
        replace(
            rally_model.decoder,
            enter_threshold=enter,
            exit_threshold=round(max(0.05, enter - 0.10), 2),
            min_live_seconds=minimum,
            bridge_gap_seconds=bridge,
        )
        for enter in (0.35, 0.40, 0.45, 0.50)
        for minimum in (0.25, 0.50)
        for bridge in (0.0, 0.50)
    ]
    serve_cache = {
        config: [
            tuple(
                decode_serve_probabilities(
                    item.sequence.times,
                    probabilities,
                    config,
                    duration=item.sequence.metadata.duration,
                )
            )
            for item, _, probabilities, _ in cached
        ]
        for config in serve_configs
    }
    permissive_cache: dict[DecoderConfig, list[tuple[DecodedInterval, ...]]] = {}
    for config in permissive_configs:
        permissive_cache[config] = [
            tuple(
                decode_probabilities(
                    item.sequence.times,
                    live,
                    item.sequence.metadata.duration,
                    config,
                    _effective_analysis_fps(item),
                )[0]
            )
            for item, live, _, _ in cached
        ]

    if progress is not None:
        progress("Selecting serve peak and rally-rescue parameters on validation data")
    best_key: tuple[float, ...] | None = None
    best_serve: ServeDecoderConfig | None = None
    best_composition: ServeCompositionConfig | None = None
    best_metrics: dict[str, Any] | None = None
    candidate_count = 0
    for serve_config in serve_configs:
        for permissive_config in permissive_configs:
            for association in (0.5, 1.0, 1.5):
                for fallback in (0.0, 1.5, 2.0, 2.5):
                    composition = ServeCompositionConfig(
                        association_seconds=association,
                        fallback_seconds=fallback,
                        max_rescue_seconds=5.0,
                        permissive_decoder=permissive_config,
                    )
                    predictions: list[PairedPrediction] = []
                    for index, (item, _, _, primary) in enumerate(cached):
                        composed = compose_serve_anchored_intervals(
                            primary,
                            permissive_cache[permissive_config][index],
                            serve_cache[serve_config][index],
                            item.sequence.metadata.duration,
                            composition,
                            sample_seconds=1.0 / _effective_analysis_fps(item),
                        )
                        predictions.append(
                            PairedPrediction(
                                item,
                                primary,
                                serve_cache[serve_config][index],
                                tuple(composed),
                            )
                        )
                    key, metrics = _selection_score(predictions)
                    candidate_count += 1
                    if best_key is None or key > best_key:
                        best_key = key
                        best_serve = serve_config
                        best_composition = composition
                        best_metrics = metrics
    assert best_serve is not None and best_composition is not None and best_metrics is not None
    return best_serve, best_composition, {
        "status": "selected-on-validation",
        "objective": "0.55*shortIoU0.5Recall + 0.35*overallEventF1 + 0.10*overallEventRecall",
        "tieBreak": (
            "higher short recall, event F1/recall/precision, time IoU and live-time precision; "
            "then less retained dead time and fewer predicted rallies"
        ),
        "shortRallyDefinitionSeconds": SHORT_RALLY_SECONDS,
        "candidateCount": candidate_count,
        "serveDecoder": best_serve.to_dict(),
        "composition": best_composition.to_dict(),
        "validationSelectionMetrics": best_metrics,
    }


def _clip_ignored(
    predictions: Sequence[DecodedInterval], ignored: Sequence[Interval]
) -> list[Interval]:
    fragments = [Interval(item.start, item.end) for item in predictions]
    for blocked in ignored:
        clipped: list[Interval] = []
        for prediction in fragments:
            if prediction.end <= blocked.start or prediction.start >= blocked.end:
                clipped.append(prediction)
            else:
                if prediction.start < blocked.start:
                    clipped.append(Interval(prediction.start, blocked.start))
                if prediction.end > blocked.end:
                    clipped.append(Interval(blocked.end, prediction.end))
        fragments = clipped
    return fragments


def _slice_truth(recording: Recording) -> dict[str, tuple[Interval, ...]]:
    rows = tuple((rally, set(rally.tags)) for rally in recording.rallies)
    return {
        "all": tuple(rally for rally, _ in rows),
        "shortAtMost3Seconds": tuple(
            rally for rally, _ in rows if rally.end - rally.start <= SHORT_RALLY_SECONDS
        ),
        "ace": tuple(rally for rally, tags in rows if "ace" in tags),
        "serviceFault": tuple(
            rally for rally, tags in rows if "service-fault" in tags
        ),
        "ordinaryLong": tuple(
            rally
            for rally, tags in rows
            if rally.end - rally.start > SHORT_RALLY_SECONDS
            and "ace" not in tags
            and "service-fault" not in tags
        ),
    }


def _interval_report(predictions: Sequence[PairedPrediction], field: str) -> dict[str, Any]:
    per_recording: list[dict[str, Any]] = []
    for prediction in predictions:
        item = prediction.prepared
        scored = _clip_ignored(getattr(prediction, field), item.recording.ignored_intervals)
        metrics = evaluate_intervals(item.recording.rallies, scored)
        metrics["outcomeSlices"] = outcome_slice_metrics(item.recording.rallies, scored)
        metrics.update({"id": item.recording.id, "environment": item.recording.environment})
        per_recording.append(metrics)
    aggregate = aggregate_evaluations(per_recording)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(
        [item["outcomeSlices"] for item in per_recording]
    )
    return {"aggregate": aggregate, "recordings": per_recording}


def _serve_metrics(
    predictions: Sequence[PairedPrediction], tolerance: float
) -> dict[str, Any]:
    truth_count = predicted_count = matched_count = 0
    errors: list[float] = []
    dead_seconds = 0.0
    slice_totals: dict[str, list[int]] = {}
    for prediction in predictions:
        recording = prediction.prepared.recording
        truth = [rally.start for rally in recording.rallies]
        serves = tuple(
            serve
            for serve in prediction.serves
            if not any(
                blocked.start <= serve.time < blocked.end
                for blocked in recording.ignored_intervals
            )
        )
        matches = match_serve_contacts(truth, serves, tolerance)
        truth_count += len(truth)
        predicted_count += len(serves)
        matched_count += len(matches)
        errors.extend(error for _, _, error in matches)
        live_seconds = sum(rally.end - rally.start for rally in recording.rallies)
        ignored_seconds = sum(item.end - item.start for item in recording.ignored_intervals)
        dead_seconds += max(
            0.0, prediction.prepared.sequence.metadata.duration - live_seconds - ignored_seconds
        )
        for name, rallies in _slice_truth(recording).items():
            matched = len(
                match_serve_contacts(
                    [rally.start for rally in rallies], serves, tolerance
                )
            )
            bucket = slice_totals.setdefault(name, [0, 0])
            bucket[0] += matched
            bucket[1] += len(rallies)
    precision = matched_count / predicted_count if predicted_count else 0.0
    recall = matched_count / truth_count if truth_count else 1.0
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "toleranceSeconds": tolerance,
        "trueServes": truth_count,
        "predictedServes": predicted_count,
        "matchedServes": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "timingMaeSeconds": float(np.mean(absolute)) if len(absolute) else None,
        "timingP90Seconds": float(np.percentile(absolute, 90)) if len(absolute) else None,
        "falseDetectionsPerDeadMinute": (
            (predicted_count - matched_count) / (dead_seconds / 60.0) if dead_seconds else None
        ),
        "sliceRecall": {
            name: {"serves": total, "recall": matched / total if total else 1.0}
            for name, (matched, total) in slice_totals.items()
        },
    }


def _top_features(model: LogisticModel, limit: int = 15) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(model.weights))[::-1][:limit]
    return [
        {
            "feature": model.feature_names[int(index)],
            "standardizedCoefficient": float(model.weights[int(index)]),
            "absoluteCoefficient": float(abs(model.weights[int(index)])),
        }
        for index in order
    ]


def _delta(composed: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "eventPrecision",
        "eventRecall",
        "eventF1",
        "timeIoU",
        "liveTimeRecall",
        "liveTimePrecision",
        "missedLiveSeconds",
        "deadSecondsRetained",
    )
    result = {key: composed[key] - baseline[key] for key in keys}
    result["outcomeStrictRecall"] = {}
    for name, composed_slice in composed["outcomeSlices"].items():
        baseline_slice = baseline["outcomeSlices"].get(name, {})
        composed_recall = composed_slice.get("strictMatchRecall")
        baseline_recall = baseline_slice.get("strictMatchRecall")
        result["outcomeStrictRecall"][name] = (
            composed_recall - baseline_recall
            if composed_recall is not None and baseline_recall is not None
            else None
        )
    return result


def _serve_input_profile_metadata(serve_model: LogisticModel) -> dict[str, Any]:
    stored = serve_model.training_summary.get("serveInputProfile")
    if isinstance(stored, dict):
        return stored
    return {
        "id": FULL_SERVE_INPUT_PROFILE,
        "retainedInputs": len(serve_model.feature_names),
        "removedInputs": 0,
        "legacyDefault": True,
    }


def _build_report(
    prepared: Sequence[PreparedRecording],
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    composition: ServeCompositionConfig,
    *,
    dataset: str,
    manifest_sha256: str,
    split: str,
    retrospective: bool = False,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    predictions = [
        _prediction_inputs(item, rally_model, serve_model, serve_decoder, composition)
        for item in prepared
    ]
    baseline = _interval_report(predictions, "primary")
    composed = _interval_report(predictions, "composed")
    video_seconds = sum(item.sequence.metadata.duration for item in prepared)
    processing_seconds = time.perf_counter() - started
    return {
        "schemaVersion": 1,
        "experiment": "serve-specialist-composition-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "producer": f"volleycut-analysis/{__version__}",
        "dataset": dataset,
        "manifestSha256": manifest_sha256,
        "split": split,
        "assessmentRole": assessment_role_for_split(
            split, retrospective=retrospective
        ),
        "models": {
            "rally": {"sha256": rally_model.artifact_sha256, "task": rally_model.prediction_task},
            "serve": {"sha256": serve_model.artifact_sha256, "task": serve_model.prediction_task},
        },
        "serveTarget": serve_model.training_summary.get("serveTarget"),
        "serveInputProfile": _serve_input_profile_metadata(serve_model),
        "serveDecoder": serve_decoder.to_dict(),
        "composition": composition.to_dict(),
        "selection": selection,
        "serveSpotting": {
            "at0.5Seconds": _serve_metrics(predictions, 0.5),
            "at1Second": _serve_metrics(predictions, 1.0),
            "at2Seconds": _serve_metrics(predictions, 2.0),
        },
        "baseline": baseline,
        "composed": composed,
        "delta": _delta(composed["aggregate"], baseline["aggregate"]),
        "featureImportance": {
            "interpretation": (
                "Absolute standardized logistic coefficients are associations, not causal "
                "importance; correlated temporal features can exchange weight."
            ),
            "rallyTop": _top_features(rally_model),
            "serveTop": _top_features(serve_model),
        },
        "processing": {
            "pairedPredictionWallClockSeconds": processing_seconds,
            "videoSeconds": video_seconds,
            "pairedPredictionToVideoRatio": (
                processing_seconds / video_seconds if video_seconds else None
            ),
            "note": "This excludes feature extraction/cache loading and measures both heads plus decoding.",
        },
        "limitations": [
            "The fixed validation split contains one grass source group and is tuning evidence only.",
            "The one-source test split is a retrospective regression set that has been inspected before.",
            "Logistic scores are ranking confidences, not calibrated probabilities.",
        ],
    }


def train_serve_dataset(
    manifest_path: str | Path,
    rally_model_path: str | Path,
    model_destination: str | Path,
    cache_dir: str | Path,
    *,
    target_radius_seconds: float = 1.0,
    serve_input_profile: str = FULL_SERVE_INPUT_PROFILE,
    training_config: TrainingConfig | None = None,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(target_radius_seconds) or target_radius_seconds < 0:
        raise ValueError("serve target radius must be non-negative")
    if serve_input_profile not in SERVE_INPUT_PROFILES:
        raise ValueError(
            f"serve_input_profile must be one of {sorted(SERVE_INPUT_PROFILES)}"
        )
    destination = Path(model_destination).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ModelError(f"model destination is not an empty directory: {destination}")
    report_destination = Path(output_path).expanduser().resolve() if output_path else None
    if report_destination is not None and report_destination.exists():
        raise ModelError(f"evaluation output already exists: {report_destination}")
    if report_destination is not None and (
        report_destination == destination or destination in report_destination.parents
    ):
        raise ModelError("evaluation output must not be inside the model destination")

    started = time.perf_counter()
    rally_model = load_model(rally_model_path)
    if rally_model.prediction_task != RALLY_LIVE_TASK:
        raise ModelError("serve training requires a rally-live primary model")
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    if rally_model.training_summary.get("manifestSha256") != manifest_sha256:
        raise ModelError("rally model differs from the immutable serve-training manifest")
    training_rows = manifest.for_split("train")
    validation_rows = manifest.for_split("validation")
    if not training_rows or not validation_rows:
        raise ManifestError("serve training requires both train and validation recordings")
    training = _prepare_many(
        training_rows, rally_model.feature_config, cache_dir, progress=progress
    )
    validation = _prepare_many(
        validation_rows, rally_model.feature_config, cache_dir, progress=progress
    )
    for item in (*training, *validation):
        if item.contextual_names != rally_model.feature_names:
            raise ModelError(f"feature signature mismatch for {item.recording.id}")

    def targets(item: PreparedRecording) -> np.ndarray:
        return serve_labels_for_times(
            item.sequence.times, item.recording.rallies, target_radius_seconds
        )

    train_labels = [targets(item)[item.sample_mask] for item in training]
    validation_labels = [targets(item)[item.sample_mask] for item in validation]
    _validate_serve_validation_targets(validation_labels)
    retained_inputs = _serve_input_mask(rally_model.feature_names, serve_input_profile)
    train_values = [
        _masked_serve_values(
            item.contextual_values[item.sample_mask], retained_inputs
        )
        for item in training
    ]
    validation_values = [
        _masked_serve_values(
            item.contextual_values[item.sample_mask], retained_inputs
        )
        for item in validation
    ]
    if progress is not None:
        progress("Fitting the serve-contact specialist")
    serve_model = train_logistic_model(
        train_values,
        train_labels,
        validation_values,
        validation_labels,
        rally_model.feature_config,
        rally_model.feature_names,
        rally_model.decoder,
        training_config or TrainingConfig(),
        prediction_task=SERVE_CONTACT_TASK,
    )
    serve_model.feature_version = rally_model.feature_version
    serve_model.training_summary.update(
        {
            "dataset": manifest.name,
            "manifestSha256": manifest_sha256,
            "rallyModelSha256": rally_model.artifact_sha256,
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
            "serveTarget": {
                "id": SERVE_TARGET_ID,
                "radiusSeconds": target_radius_seconds,
                "contactDefinition": "rally start under serve-contact-to-dead-ball-v1",
            },
            "serveInputProfile": {
                "id": serve_input_profile,
                "retainedInputs": int(np.sum(retained_inputs)),
                "removedInputs": int(np.sum(~retained_inputs)),
                "removedFeatureNames": [
                    name
                    for name, retained in zip(
                        rally_model.feature_names, retained_inputs, strict=True
                    )
                    if not retained
                ],
                "masking": (
                    "removed inputs are zeroed before normalization during fitting; "
                    "their persisted standardized coefficients are exactly zero"
                ),
            },
        }
    )
    serve_decoder, composition, selection = _tune_composition(
        validation, rally_model, serve_model, progress=progress
    )
    serve_model.training_summary.update(
        {
            "serveDecoder": serve_decoder.to_dict(),
            "composition": composition.to_dict(),
            "selection": selection,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": __import__("cv2").__version__,
                "wallClockSeconds": round(time.perf_counter() - started, 3),
            },
        }
    )
    serve_model.save(destination)
    report = _build_report(
        validation,
        rally_model,
        serve_model,
        serve_decoder,
        composition,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split="validation",
        selection=selection,
    )
    report["serveModelPath"] = str(destination)
    if report_destination is not None:
        atomic_write_text(
            report_destination, json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def evaluate_serve_dataset(
    manifest_path: str | Path,
    rally_model_path: str | Path,
    serve_model_path: str | Path,
    cache_dir: str | Path,
    *,
    split: str = "test",
    retrospective: bool = False,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    assessment_role_for_split(split, retrospective=retrospective)
    destination = Path(output_path).expanduser().resolve() if output_path else None
    if destination is not None and destination.exists():
        raise ModelError(f"evaluation output already exists: {destination}")
    rally_model = load_model(rally_model_path)
    serve_model = load_model(serve_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    serve_decoder, composition = _validate_model_pair(
        rally_model, serve_model, manifest_sha256=manifest_sha256
    )
    recordings = manifest.for_split(split)
    if not recordings:
        raise ManifestError(f"manifest has no recordings in split {split!r}")
    for model in (rally_model, serve_model):
        protected = set(model.training_summary.get("trainingSourceGroups", []))
        if split != "validation":
            protected |= set(model.training_summary.get("validationSourceGroups", []))
        overlap = protected & {recording.source_group for recording in recordings}
        if overlap:
            raise ModelError(
                f"evaluation split leaks trained/tuned source groups: {sorted(overlap)}"
            )
    prepared = _prepare_many(
        recordings, rally_model.feature_config, cache_dir, progress=progress
    )
    report = _build_report(
        prepared,
        rally_model,
        serve_model,
        serve_decoder,
        composition,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split=split,
        retrospective=retrospective,
        selection=serve_model.training_summary.get("selection"),
    )
    if destination is not None:
        atomic_write_text(destination, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
