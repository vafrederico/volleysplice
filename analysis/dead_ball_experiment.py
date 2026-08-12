from __future__ import annotations

import json
import math
import platform
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .artifacts import atomic_write_text
from .config import TrainingConfig
from .dead_ball import (
    DeadBallDecoderConfig,
    DeadBallDetection,
    dead_ball_labels_for_times,
    decode_dead_ball_probabilities,
    select_dead_ball_after_serve,
)
from .decoder import DecodedInterval, decode_probabilities
from .model import (
    DEAD_BALL_TASK,
    LogisticModel,
    ModelError,
    load_model,
    train_logistic_model,
)
from .pipeline import PreparedRecording, _manifest_digest, _prepare_many
from .schema import Interval, ManifestError, load_manifest
from .serve import (
    ServeCompositionConfig,
    ServeDecoderConfig,
    ServeDetection,
    compose_serve_anchored_intervals,
    decode_serve_probabilities,
    match_serve_contacts,
)
from .serve_experiment import (
    _clip_ignored,
    _delta,
    _effective_analysis_fps,
    _interval_report,
    _top_features,
    _validate_model_pair,
)
from .version import __version__


DEAD_BALL_TARGET_ID = "dead-ball-boundary-window-v1"
DEAD_BALL_EXPERIMENT_ID = "serve-anchored-dead-ball-v1"
SHORT_RALLY_SECONDS = 3.0


@dataclass(frozen=True)
class DeadBallCompositionConfig:
    method: str
    association_seconds: float = 1.0
    min_after_serve_seconds: float = 0.5
    max_after_serve_seconds: float = 4.0
    fixed_duration_seconds: float | None = None

    def validate(self) -> None:
        methods = {
            "v4-noop",
            "gated-learned-end",
            "refine-v4-end",
            "refine-v4-rescue-end",
            "gated-fixed-duration",
        }
        if self.method not in methods:
            raise ValueError(f"unsupported dead-ball composition method: {self.method!r}")
        for label, value in (
            ("association", self.association_seconds),
            ("minimum after serve", self.min_after_serve_seconds),
            ("maximum after serve", self.max_after_serve_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"dead-ball {label} seconds must be non-negative")
        if self.max_after_serve_seconds < self.min_after_serve_seconds:
            raise ValueError("dead-ball maximum after serve must be at least the minimum")
        if self.method == "gated-fixed-duration":
            if (
                self.fixed_duration_seconds is None
                or not math.isfinite(self.fixed_duration_seconds)
                or self.fixed_duration_seconds <= 0
            ):
                raise ValueError("fixed-duration composition requires a positive duration")
        elif self.fixed_duration_seconds is not None:
            raise ValueError("fixed duration is only valid for fixed-duration composition")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": self.method,
            "associationSeconds": self.association_seconds,
            "minAfterServeSeconds": self.min_after_serve_seconds,
            "maxAfterServeSeconds": self.max_after_serve_seconds,
        }
        if self.fixed_duration_seconds is not None:
            result["fixedDurationSeconds"] = self.fixed_duration_seconds
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DeadBallCompositionConfig":
        result = cls(
            method=str(value["method"]),
            association_seconds=float(value.get("associationSeconds", 1.0)),
            min_after_serve_seconds=float(value.get("minAfterServeSeconds", 0.5)),
            max_after_serve_seconds=float(value.get("maxAfterServeSeconds", 4.0)),
            fixed_duration_seconds=(
                float(value["fixedDurationSeconds"])
                if "fixedDurationSeconds" in value
                else None
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DeadBallInputs:
    prepared: PreparedRecording
    primary: tuple[DecodedInterval, ...]
    permissive: tuple[DecodedInterval, ...]
    serves: tuple[ServeDetection, ...]
    v4_composed: tuple[DecodedInterval, ...]
    dead_probabilities: np.ndarray


@dataclass(frozen=True)
class DeadBallPrediction:
    prepared: PreparedRecording
    primary: tuple[DecodedInterval, ...]
    v4_composed: tuple[DecodedInterval, ...]
    candidate: tuple[DecodedInterval, ...]
    serves: tuple[ServeDetection, ...]
    dead_balls: tuple[DeadBallDetection, ...]


def _merge_rows(
    rows: Sequence[Sequence[float]], duration: float
) -> tuple[DecodedInterval, ...]:
    ordered = sorted(
        (max(0.0, row[0]), min(duration, row[1]), row[2])
        for row in rows
        if row[1] > row[0] and row[0] < duration and row[1] > 0
    )
    merged: list[list[float]] = []
    for start, end, confidence in ordered:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], confidence)
        else:
            merged.append([start, end, confidence])
    return tuple(DecodedInterval(*row) for row in merged)


def _live_evidence(
    permissive: Sequence[DecodedInterval],
    start: float,
    end: float,
    association_seconds: float,
    sample_seconds: float,
) -> bool:
    return any(
        item.start <= start + association_seconds
        and item.end >= start - association_seconds
        and min(item.end, end) - max(item.start, start) >= sample_seconds - 1e-9
        for item in permissive
    )


def _compose_candidate(
    inputs: DeadBallInputs,
    detections: Sequence[DeadBallDetection],
    config: DeadBallCompositionConfig,
) -> tuple[DecodedInterval, ...]:
    config.validate()
    if config.method == "v4-noop":
        return inputs.v4_composed
    duration = inputs.prepared.sequence.metadata.duration
    sample_seconds = 1.0 / _effective_analysis_fps(inputs.prepared)
    refine_methods = {"refine-v4-end", "refine-v4-rescue-end"}
    source = inputs.v4_composed if config.method in refine_methods else inputs.primary
    rows = [
        [item.start, item.end, item.confidence]
        for item in source
    ]
    unused: list[tuple[int, ServeDetection]] = []
    for serve_index, serve in enumerate(inputs.serves):
        associated = [
            index
            for index, row in enumerate(rows)
            if abs(row[0] - serve.time) <= config.association_seconds + 1e-9
        ]
        if not associated:
            unused.append((serve_index, serve))
            continue
        row_index = min(associated, key=lambda index: abs(rows[index][0] - serve.time))
        if config.method in {"gated-learned-end", "gated-fixed-duration"}:
            # The primary rally decoder owns established intervals. This specialist may
            # rescue a missing short event, but it must not truncate ordinary rallies.
            continue
        if config.method == "refine-v4-rescue-end" and any(
            primary.start < rows[row_index][1] and rows[row_index][0] < primary.end
            for primary in inputs.primary
        ):
            continue
        if config.method == "gated-fixed-duration":
            end = serve.time + float(config.fixed_duration_seconds)
            detection = None
        else:
            next_serve = (
                inputs.serves[serve_index + 1].time
                if serve_index + 1 < len(inputs.serves)
                else None
            )
            detection = select_dead_ball_after_serve(
                detections,
                serve.time,
                config.min_after_serve_seconds,
                config.max_after_serve_seconds,
                before_time=next_serve,
            )
            if detection is None:
                continue
            end = detection.time
        start = min(rows[row_index][0], serve.time)
        if end < start + sample_seconds - 1e-9:
            continue
        rows[row_index][0] = start
        rows[row_index][1] = min(duration, end)
        rows[row_index][2] = max(
            rows[row_index][2],
            serve.confidence,
            detection.confidence if detection is not None else 0.0,
        )

    if config.method not in refine_methods:
        for serve_index, serve in unused:
            if config.method == "gated-fixed-duration":
                end = serve.time + float(config.fixed_duration_seconds)
                detection = None
            else:
                next_serve = (
                    inputs.serves[serve_index + 1].time
                    if serve_index + 1 < len(inputs.serves)
                    else None
                )
                detection = select_dead_ball_after_serve(
                    detections,
                    serve.time,
                    config.min_after_serve_seconds,
                    config.max_after_serve_seconds,
                    before_time=next_serve,
                )
                if detection is None:
                    continue
                end = detection.time
            end = min(duration, end)
            if end < serve.time + sample_seconds - 1e-9 or not _live_evidence(
                inputs.permissive,
                serve.time,
                end,
                config.association_seconds,
                sample_seconds,
            ):
                continue
            rows.append(
                [
                    serve.time,
                    end,
                    max(
                        serve.confidence,
                        detection.confidence if detection is not None else 0.0,
                    ),
                ]
            )
    return _merge_rows(rows, duration)


def _prediction_inputs(
    prepared: PreparedRecording,
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    dead_ball_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    serve_composition: ServeCompositionConfig,
) -> DeadBallInputs:
    if prepared.contextual_names != rally_model.feature_names:
        raise ModelError(f"feature signature mismatch for {prepared.recording.id}")
    live = rally_model.predict(prepared.contextual_values)
    serve = serve_model.predict(prepared.contextual_values)
    dead = dead_ball_model.predict(prepared.contextual_values)
    duration = prepared.sequence.metadata.duration
    fps = _effective_analysis_fps(prepared)
    primary, _ = decode_probabilities(
        prepared.sequence.times, live, duration, rally_model.decoder, fps
    )
    permissive, _ = decode_probabilities(
        prepared.sequence.times,
        live,
        duration,
        serve_composition.permissive_decoder,
        fps,
    )
    serves = decode_serve_probabilities(
        prepared.sequence.times, serve, serve_decoder, duration=duration
    )
    v4_composed = compose_serve_anchored_intervals(
        primary,
        permissive,
        serves,
        duration,
        serve_composition,
        sample_seconds=1.0 / fps,
    )
    return DeadBallInputs(
        prepared,
        tuple(primary),
        tuple(permissive),
        tuple(serves),
        tuple(v4_composed),
        dead,
    )


def _predictions_for(
    inputs: Sequence[DeadBallInputs],
    decoder: DeadBallDecoderConfig,
    composition: DeadBallCompositionConfig,
) -> list[DeadBallPrediction]:
    predictions: list[DeadBallPrediction] = []
    for item in inputs:
        detections = tuple(
            decode_dead_ball_probabilities(
                item.prepared.sequence.times,
                item.dead_probabilities,
                decoder,
                duration=item.prepared.sequence.metadata.duration,
            )
        )
        predictions.append(
            DeadBallPrediction(
                item.prepared,
                item.primary,
                item.v4_composed,
                _compose_candidate(item, detections, composition),
                item.serves,
                detections,
            )
        )
    return predictions


def _strict_count(slice_metrics: dict[str, Any]) -> int:
    return int(round(slice_metrics["rallies"] * slice_metrics["strictMatchRecall"]))


def _selection_key(aggregate: dict[str, Any]) -> tuple[float, ...]:
    short = aggregate["outcomeSlices"]["shortAtMost3Seconds"]
    return (
        float(_strict_count(short)),
        aggregate["eventF1"],
        aggregate["timeIoU"],
        aggregate["liveTimeRecall"],
        aggregate["liveTimePrecision"],
        -aggregate["deadSecondsRetained"],
        -float(aggregate["predictedRallies"]),
    )


def _is_feasible(aggregate: dict[str, Any], v4: dict[str, Any]) -> bool:
    ordinary = aggregate["outcomeSlices"]["ordinaryLong"]
    v4_ordinary = v4["outcomeSlices"]["ordinaryLong"]
    return (
        aggregate["matchedRallies"] >= v4["matchedRallies"]
        and _strict_count(ordinary) >= _strict_count(v4_ordinary)
        and aggregate["liveTimeRecall"] >= v4["liveTimeRecall"] - 0.01 - 1e-12
    )


def _tune_composition(
    inputs: Sequence[DeadBallInputs],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    DeadBallDecoderConfig,
    DeadBallDecoderConfig,
    DeadBallCompositionConfig,
    dict[str, tuple[DeadBallDecoderConfig, DeadBallCompositionConfig]],
    dict[str, Any],
]:
    if not inputs:
        raise ManifestError("dead-ball selection requires validation recordings")
    noop_decoder = DeadBallDecoderConfig(0.95, 0.0)
    noop_config = DeadBallCompositionConfig("v4-noop")
    noop_predictions = _predictions_for(inputs, noop_decoder, noop_config)
    v4 = _interval_report(noop_predictions, "candidate")["aggregate"]
    candidates: list[
        tuple[str, DeadBallDecoderConfig, DeadBallCompositionConfig, dict[str, Any]]
    ] = [("noop", noop_decoder, noop_config, v4)]
    if progress is not None:
        progress("Selecting dead-ball threshold and serve-gated composition on validation")
    for threshold in (0.70, 0.80, 0.90, 0.95):
        for offset in (-0.25, 0.0, 0.25):
            decoder = DeadBallDecoderConfig(threshold, offset)
            for minimum in (0.25, 0.50):
                for family, method in (
                    ("learned", "gated-learned-end"),
                    ("refined", "refine-v4-end"),
                    ("safeRefined", "refine-v4-rescue-end"),
                ):
                    config = DeadBallCompositionConfig(
                        method,
                        min_after_serve_seconds=minimum,
                    )
                    aggregate = _interval_report(
                        _predictions_for(inputs, decoder, config), "candidate"
                    )["aggregate"]
                    candidates.append((family, decoder, config, aggregate))
    for duration in (1.5, 2.0, 2.5, 3.0):
        for minimum in (0.25, 0.50):
            config = DeadBallCompositionConfig(
                "gated-fixed-duration",
                min_after_serve_seconds=minimum,
                fixed_duration_seconds=duration,
            )
            aggregate = _interval_report(
                _predictions_for(inputs, noop_decoder, config), "candidate"
            )["aggregate"]
            candidates.append(("fixed", noop_decoder, config, aggregate))

    family_best: dict[
        str, tuple[DeadBallDecoderConfig, DeadBallCompositionConfig, dict[str, Any]]
    ] = {}
    for family, decoder, config, aggregate in candidates:
        current = family_best.get(family)
        if current is None or _selection_key(aggregate) > _selection_key(current[2]):
            family_best[family] = (decoder, config, aggregate)
    feasible = [row for row in candidates if _is_feasible(row[3], v4)]
    selected = max(feasible, key=lambda row: _selection_key(row[3]))
    family_configs = {
        family: (decoder, config)
        for family, (decoder, config, _) in family_best.items()
    }
    selection = {
        "status": "selected-on-validation",
        "objective": (
            "guard overall and ordinary-long strict matches plus live recall; then maximize "
            "short strict matches, event F1, time IoU, live precision, and minimize dead time"
        ),
        "guardrails": {
            "minimumOverallStrictMatches": v4["matchedRallies"],
            "minimumOrdinaryLongStrictMatches": _strict_count(
                v4["outcomeSlices"]["ordinaryLong"]
            ),
            "minimumLiveTimeRecall": v4["liveTimeRecall"] - 0.01,
        },
        "candidateCount": len(candidates),
        "feasibleCandidateCount": len(feasible),
        "selectedFamily": selected[0],
        "deadBallDecoder": selected[1].to_dict(),
        "composition": selected[2].to_dict(),
        "validationMetrics": selected[3],
        "familySelections": {
            family: {
                "deadBallDecoder": decoder.to_dict(),
                "composition": config.to_dict(),
                "validationMetrics": family_best[family][2],
            }
            for family, (decoder, config) in family_configs.items()
        },
    }
    reporting_decoder = family_configs["learned"][0]
    return reporting_decoder, selected[1], selected[2], family_configs, selection


def _validate_dead_ball_validation(labels: Sequence[np.ndarray]) -> None:
    samples = sum(len(item) for item in labels)
    positives = sum(int(np.sum(item > 0.5)) for item in labels)
    if not samples or not positives or positives == samples:
        raise ManifestError("dead-ball validation must contain positive and negative samples")


def _crossfit_epoch_cap(
    prepared: Sequence[PreparedRecording],
    labels: dict[str, np.ndarray],
    rally_model: LogisticModel,
    config: TrainingConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    groups = sorted({item.recording.source_group for item in prepared})
    if len(groups) < 3:
        raise ManifestError("dead-ball epoch selection requires three training source groups")
    summaries: list[dict[str, Any]] = []
    epochs: list[int] = []
    for index, held_out in enumerate(groups, start=1):
        train = [item for item in prepared if item.recording.source_group != held_out]
        validation = [item for item in prepared if item.recording.source_group == held_out]
        if progress is not None:
            progress(f"Selecting dead-ball epoch {index}/{len(groups)}: {held_out}")
        model = train_logistic_model(
            [item.contextual_values[item.sample_mask] for item in train],
            [labels[item.recording.id][item.sample_mask] for item in train],
            [item.contextual_values[item.sample_mask] for item in validation],
            [labels[item.recording.id][item.sample_mask] for item in validation],
            rally_model.feature_config,
            rally_model.feature_names,
            rally_model.decoder,
            config,
            prediction_task=DEAD_BALL_TASK,
        )
        best_epoch = int(model.training_summary["bestEpoch"])
        epochs.append(best_epoch)
        summaries.append(
            {
                "heldOutSourceGroup": held_out,
                "trainingSourceGroups": sorted(
                    {item.recording.source_group for item in train}
                ),
                "bestEpoch": best_epoch,
                "epochsCompleted": model.training_summary["epochsCompleted"],
                "bestValidationLoss": model.training_summary["bestValidationLoss"],
            }
        )
        del model
    return int(statistics.median(epochs)), summaries


def _validate_model_triplet(
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    dead_ball_model: LogisticModel,
    *,
    manifest_sha256: str | None = None,
) -> tuple[
    ServeDecoderConfig,
    ServeCompositionConfig,
    DeadBallDecoderConfig,
    DeadBallDecoderConfig,
    DeadBallCompositionConfig,
    dict[str, tuple[DeadBallDecoderConfig, DeadBallCompositionConfig]],
]:
    serve_decoder, serve_composition = _validate_model_pair(
        rally_model, serve_model, manifest_sha256=manifest_sha256
    )
    if dead_ball_model.prediction_task != DEAD_BALL_TASK:
        raise ModelError("dead-ball specialist must predict dead-ball-boundary")
    if dead_ball_model.feature_version != rally_model.feature_version:
        raise ModelError("dead-ball specialist uses a different feature version")
    if dead_ball_model.feature_config != rally_model.feature_config:
        raise ModelError("dead-ball specialist uses a different feature configuration")
    if dead_ball_model.feature_names != rally_model.feature_names:
        raise ModelError("dead-ball specialist uses a different feature signature")
    summary = dead_ball_model.training_summary
    if summary.get("rallyModelSha256") != rally_model.artifact_sha256:
        raise ModelError("dead-ball specialist is bound to a different rally model")
    if summary.get("serveModelSha256") != serve_model.artifact_sha256:
        raise ModelError("dead-ball specialist is bound to a different serve model")
    if summary.get("manifestSha256") != rally_model.training_summary.get("manifestSha256"):
        raise ModelError("dead-ball specialist was trained from a different manifest")
    if manifest_sha256 is not None and summary.get("manifestSha256") != manifest_sha256:
        raise ModelError("dead-ball specialist differs from the immutable evaluation manifest")
    try:
        decoder = DeadBallDecoderConfig.from_dict(summary["deadBallDecoder"])
        selected_decoder = DeadBallDecoderConfig.from_dict(
            summary["selectedDeadBallDecoder"]
        )
        selected = DeadBallCompositionConfig.from_dict(summary["selectedComposition"])
        families = {
            name: (
                DeadBallDecoderConfig.from_dict(value["deadBallDecoder"]),
                DeadBallCompositionConfig.from_dict(value["composition"]),
            )
            for name, value in summary["familySelections"].items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(f"dead-ball selection metadata is invalid: {error}") from error
    return (
        serve_decoder,
        serve_composition,
        decoder,
        selected_decoder,
        selected,
        families,
    )


def _end_metrics(
    predictions: Sequence[DeadBallPrediction], tolerance: float
) -> dict[str, Any]:
    truth_count = prediction_count = matched_count = 0
    errors: list[float] = []
    for prediction in predictions:
        recording = prediction.prepared.recording
        detections = [
            item
            for item in prediction.dead_balls
            if not any(
                blocked.start <= item.time < blocked.end
                for blocked in recording.ignored_intervals
            )
        ]
        matches = match_serve_contacts(
            [item.end for item in recording.rallies], detections, tolerance
        )
        truth_count += len(recording.rallies)
        prediction_count += len(detections)
        matched_count += len(matches)
        errors.extend(error for _, _, error in matches)
    precision = matched_count / prediction_count if prediction_count else 0.0
    recall = matched_count / truth_count if truth_count else 1.0
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "toleranceSeconds": tolerance,
        "trueEnds": truth_count,
        "predictedEnds": prediction_count,
        "matchedEnds": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "timingMaeSeconds": float(np.mean(absolute)) if len(absolute) else None,
        "timingP90Seconds": float(np.percentile(absolute, 90)) if len(absolute) else None,
        "timingMeanSignedSeconds": float(np.mean(errors)) if errors else None,
    }


def _conditioned_end_metrics(
    inputs: Sequence[DeadBallInputs],
    predictions: Sequence[DeadBallPrediction],
    decoder: DeadBallDecoderConfig,
    composition: DeadBallCompositionConfig,
) -> dict[str, Any]:
    eligible = emitted = within_one = 0
    errors: list[float] = []
    for raw, prediction in zip(inputs, predictions, strict=True):
        matches = match_serve_contacts(
            [item.start for item in raw.prepared.recording.rallies], raw.serves, 1.0
        )
        for truth_index, serve_index, _ in matches:
            eligible += 1
            serve = raw.serves[serve_index]
            next_serve = (
                raw.serves[serve_index + 1].time
                if serve_index + 1 < len(raw.serves)
                else None
            )
            end = select_dead_ball_after_serve(
                prediction.dead_balls,
                serve.time,
                composition.min_after_serve_seconds,
                composition.max_after_serve_seconds,
                before_time=next_serve,
            )
            if end is None:
                continue
            emitted += 1
            error = end.time - raw.prepared.recording.rallies[truth_index].end
            errors.append(error)
            within_one += abs(error) <= 1.0 + 1e-12
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "serveToleranceSeconds": 1.0,
        "eligibleMatchedServes": eligible,
        "emittedEnds": emitted,
        "emissionRate": emitted / eligible if eligible else 1.0,
        "endsWithin1Second": within_one,
        "recallWithin1Second": within_one / eligible if eligible else 1.0,
        "timingMaeSeconds": float(np.mean(absolute)) if len(absolute) else None,
        "timingP90Seconds": float(np.percentile(absolute, 90)) if len(absolute) else None,
        "timingMeanSignedSeconds": float(np.mean(errors)) if errors else None,
        "decoder": decoder.to_dict(),
    }


def _report_variant(
    inputs: Sequence[DeadBallInputs],
    decoder: DeadBallDecoderConfig,
    config: DeadBallCompositionConfig,
) -> tuple[list[DeadBallPrediction], dict[str, Any]]:
    predictions = _predictions_for(inputs, decoder, config)
    return predictions, _interval_report(predictions, "candidate")


def _build_report(
    prepared: Sequence[PreparedRecording],
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    dead_ball_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    serve_composition: ServeCompositionConfig,
    reporting_decoder: DeadBallDecoderConfig,
    selected_decoder: DeadBallDecoderConfig,
    selected: DeadBallCompositionConfig,
    families: dict[str, tuple[DeadBallDecoderConfig, DeadBallCompositionConfig]],
    *,
    dataset: str,
    manifest_sha256: str,
    split: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = [
        _prediction_inputs(
            item,
            rally_model,
            serve_model,
            dead_ball_model,
            serve_decoder,
            serve_composition,
        )
        for item in prepared
    ]
    noop_decoder = DeadBallDecoderConfig(0.95, 0.0)
    noop = DeadBallCompositionConfig("v4-noop")
    noop_predictions, v4 = _report_variant(inputs, noop_decoder, noop)
    baseline = _interval_report(noop_predictions, "primary")
    variants: dict[str, dict[str, Any]] = {}
    variant_predictions: dict[str, list[DeadBallPrediction]] = {}
    for family in ("learned", "safeRefined", "refined", "fixed"):
        decoder, config = families[family]
        predictions, report = _report_variant(inputs, decoder, config)
        variant_predictions[family] = predictions
        variants[family] = {
            **report,
            "deadBallDecoder": decoder.to_dict(),
            "composition": config.to_dict(),
        }
    selected_predictions, selected_report = _report_variant(
        inputs, selected_decoder, selected
    )
    reporting_predictions = _predictions_for(
        inputs, reporting_decoder, families["learned"][1]
    )
    video_seconds = sum(item.sequence.metadata.duration for item in prepared)
    processing_seconds = time.perf_counter() - started
    return {
        "schemaVersion": 1,
        "experiment": DEAD_BALL_EXPERIMENT_ID,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "producer": f"volleycut-analysis/{__version__}",
        "dataset": dataset,
        "manifestSha256": manifest_sha256,
        "split": split,
        "assessmentRole": "tuning-only" if split == "validation" else "retrospective-regression",
        "matching": {"minimumIntervalIoU": 0.5},
        "models": {
            "rally": {"sha256": rally_model.artifact_sha256, "task": rally_model.prediction_task},
            "serve": {"sha256": serve_model.artifact_sha256, "task": serve_model.prediction_task},
            "deadBall": {
                "sha256": dead_ball_model.artifact_sha256,
                "task": dead_ball_model.prediction_task,
            },
        },
        "target": dead_ball_model.training_summary.get("deadBallTarget"),
        "selection": dead_ball_model.training_summary.get("selection"),
        "endSpotting": {
            "at0.25Seconds": _end_metrics(reporting_predictions, 0.25),
            "at0.5Seconds": _end_metrics(reporting_predictions, 0.5),
            "at1Second": _end_metrics(reporting_predictions, 1.0),
            "serveConditioned": _conditioned_end_metrics(
                inputs,
                reporting_predictions,
                reporting_decoder,
                families["learned"][1],
            ),
        },
        "baseline": baseline,
        "v4Composition": v4,
        "variants": variants,
        "selected": {
            **selected_report,
            "deadBallDecoder": selected_decoder.to_dict(),
            "composition": selected.to_dict(),
        },
        "deltaSelectedVsV4": _delta(
            selected_report["aggregate"], v4["aggregate"]
        ),
        "featureImportance": {
            "interpretation": (
                "Standardized coefficients are associations, not causal importance; the end "
                "head uses centered future context and is offline."
            ),
            "deadBallTop": _top_features(dead_ball_model),
        },
        "processing": {
            "predictionWallClockSeconds": processing_seconds,
            "videoSeconds": video_seconds,
            "predictionToVideoRatio": (
                processing_seconds / video_seconds if video_seconds else None
            ),
            "note": "This excludes feature extraction/cache loading.",
        },
        "limitations": [
            "Validation is one reused grass source group and remains tuning evidence only.",
            "The one-source indoor test was inspected in prior experiments and is retrospective.",
            "The endpoint head is gated by the frozen serve detector and cannot recover missed serves.",
            "The centered feature context makes this an offline, non-causal boundary model.",
        ],
    }


def train_dead_ball_dataset(
    manifest_path: str | Path,
    rally_model_path: str | Path,
    serve_model_path: str | Path,
    model_destination: str | Path,
    cache_dir: str | Path,
    *,
    target_radius_seconds: float = 0.5,
    training_config: TrainingConfig | None = None,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(target_radius_seconds) or target_radius_seconds < 0:
        raise ValueError("dead-ball target radius must be non-negative")
    destination = Path(model_destination).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ModelError(f"model destination is not an empty directory: {destination}")
    report_destination = Path(output_path).expanduser().resolve() if output_path else None
    if report_destination is not None and report_destination.exists():
        raise ModelError(f"evaluation output already exists: {report_destination}")
    started = time.perf_counter()
    rally_model = load_model(rally_model_path)
    serve_model = load_model(serve_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    serve_decoder, serve_composition = _validate_model_pair(
        rally_model, serve_model, manifest_sha256=manifest_sha256
    )
    training_rows = manifest.for_split("train")
    validation_rows = manifest.for_split("validation")
    if not training_rows or not validation_rows:
        raise ManifestError("dead-ball training requires train and validation recordings")
    training = _prepare_many(
        training_rows, rally_model.feature_config, cache_dir, progress=progress
    )
    validation = _prepare_many(
        validation_rows, rally_model.feature_config, cache_dir, progress=progress
    )
    for item in (*training, *validation):
        if item.contextual_names != rally_model.feature_names:
            raise ModelError(f"feature signature mismatch for {item.recording.id}")
    labels = {
        item.recording.id: dead_ball_labels_for_times(
            item.sequence.times, item.recording.rallies, target_radius_seconds
        )
        for item in (*training, *validation)
    }
    _validate_dead_ball_validation(
        [labels[item.recording.id][item.sample_mask] for item in validation]
    )
    config = training_config or TrainingConfig(epochs=120)
    epoch_cap, epoch_folds = _crossfit_epoch_cap(
        training, labels, rally_model, config, progress=progress
    )
    if progress is not None:
        progress(f"Fitting final dead-ball head for {epoch_cap} fixed epochs")
    final_config = replace(config, epochs=epoch_cap, patience=epoch_cap)
    dead_ball_model = train_logistic_model(
        [item.contextual_values[item.sample_mask] for item in training],
        [labels[item.recording.id][item.sample_mask] for item in training],
        [],
        [],
        rally_model.feature_config,
        rally_model.feature_names,
        rally_model.decoder,
        final_config,
        prediction_task=DEAD_BALL_TASK,
    )
    dead_ball_model.feature_version = rally_model.feature_version
    validation_inputs = [
        _prediction_inputs(
            item,
            rally_model,
            serve_model,
            dead_ball_model,
            serve_decoder,
            serve_composition,
        )
        for item in validation
    ]
    (
        reporting_decoder,
        selected_decoder,
        selected,
        families,
        selection,
    ) = _tune_composition(validation_inputs, progress=progress)
    dead_ball_model.training_summary.update(
        {
            "dataset": manifest.name,
            "manifestSha256": manifest_sha256,
            "rallyModelSha256": rally_model.artifact_sha256,
            "serveModelSha256": serve_model.artifact_sha256,
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
            "deadBallTarget": {
                "id": DEAD_BALL_TARGET_ID,
                "radiusSeconds": target_radius_seconds,
                "boundaryDefinition": "rally end under serve-contact-to-dead-ball-v1",
            },
            "epochSelection": {
                "method": "leave-one-training-source-group-out-median-best-epoch-v1",
                "selectedEpoch": epoch_cap,
                "folds": epoch_folds,
            },
            "deadBallDecoder": reporting_decoder.to_dict(),
            "selectedDeadBallDecoder": selected_decoder.to_dict(),
            "selectedComposition": selected.to_dict(),
            "familySelections": {
                name: {
                    "deadBallDecoder": decoder.to_dict(),
                    "composition": composition.to_dict(),
                }
                for name, (decoder, composition) in families.items()
            },
            "selection": selection,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": __import__("cv2").__version__,
                "wallClockSeconds": round(time.perf_counter() - started, 3),
            },
        }
    )
    dead_ball_model.save(destination)
    report = _build_report(
        validation,
        rally_model,
        serve_model,
        dead_ball_model,
        serve_decoder,
        serve_composition,
        reporting_decoder,
        selected_decoder,
        selected,
        families,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split="validation",
    )
    report["deadBallModelPath"] = str(destination)
    if report_destination is not None:
        atomic_write_text(
            report_destination, json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def evaluate_dead_ball_dataset(
    manifest_path: str | Path,
    rally_model_path: str | Path,
    serve_model_path: str | Path,
    dead_ball_model_path: str | Path,
    cache_dir: str | Path,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    destination = Path(output_path).expanduser().resolve() if output_path else None
    if destination is not None and destination.exists():
        raise ModelError(f"evaluation output already exists: {destination}")
    rally_model = load_model(rally_model_path)
    serve_model = load_model(serve_model_path)
    dead_ball_model = load_model(dead_ball_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    (
        serve_decoder,
        serve_composition,
        reporting_decoder,
        selected_decoder,
        selected,
        families,
    ) = _validate_model_triplet(
        rally_model,
        serve_model,
        dead_ball_model,
        manifest_sha256=manifest_sha256,
    )
    recordings = manifest.for_split(split)
    if not recordings:
        raise ManifestError(f"manifest has no recordings in split {split!r}")
    evaluation_groups = {item.source_group for item in recordings}
    for model in (rally_model, serve_model, dead_ball_model):
        protected = set(model.training_summary.get("trainingSourceGroups", []))
        if split != "validation":
            protected |= set(model.training_summary.get("validationSourceGroups", []))
        overlap = protected & evaluation_groups
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
        dead_ball_model,
        serve_decoder,
        serve_composition,
        reporting_decoder,
        selected_decoder,
        selected,
        families,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split=split,
    )
    if destination is not None:
        atomic_write_text(destination, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
