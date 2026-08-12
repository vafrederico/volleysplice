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
from .config import DecoderConfig, TrainingConfig
from .dead_ball_experiment import (
    DEAD_BALL_INPUT_PROFILES,
    FULL_DEAD_BALL_INPUT_PROFILE,
    DeadBallInputs,
    _dead_ball_input_mask,
    _masked_dead_ball_values,
    _prediction_inputs,
)
from .dead_state import (
    DeadStateDecoderConfig,
    DeadStateDetection,
    decode_dead_state_after_serve,
    end_transition_labels_for_times,
)
from .decoder import DecodedInterval
from .model import (
    DEAD_STATE_TASK,
    LogisticModel,
    ModelError,
    load_model,
    train_logistic_model,
)
from .pipeline import (
    PreparedRecording,
    _evaluate_prepared_probabilities,
    _manifest_digest,
    _prepare_many,
    _tune_decoder,
    assessment_role_for_split,
)
from .schema import (
    ManifestError,
    labels_for_times as rally_labels_for_times,
    load_manifest,
)
from .serve import ServeCompositionConfig, ServeDecoderConfig, match_serve_contacts
from .serve_experiment import (
    _delta,
    _effective_analysis_fps,
    _interval_report,
    _slice_truth,
    _top_features,
    _validate_model_pair,
)
from .version import __version__


DEAD_STATE_TARGET_ID = "post-rally-dead-transition-v1"
GLOBAL_DEAD_STATE_TARGET_ID = "global-dead-inverse-rally-live-v1"
DEAD_STATE_EXPERIMENT_ID = "v5-end-transition-refinement-v1"
GLOBAL_DEAD_STATE_EXPERIMENT_ID = "v5-global-dead-end-refinement-control-v1"
END_TRANSITION_TARGET_MODE = "end-transition"
GLOBAL_DEAD_TARGET_MODE = "global-dead"
DEAD_STATE_TARGET_MODES = frozenset(
    {END_TRANSITION_TARGET_MODE, GLOBAL_DEAD_TARGET_MODE}
)
DEFAULT_BEFORE_END_SECONDS = 2.0
DEFAULT_AFTER_END_SECONDS = 2.0
DEFAULT_PRE_SERVE_SETUP_SECONDS = 1.0


@dataclass(frozen=True)
class DeadStateRefinementConfig:
    method: str
    end_window_seconds: float = 1.0

    def validate(self) -> None:
        if self.method not in {"v5-noop", "refine-v5-end"}:
            raise ValueError(f"unsupported dead-state refinement method: {self.method!r}")
        if not math.isfinite(self.end_window_seconds) or self.end_window_seconds < 0:
            raise ValueError("dead-state end window must be non-negative")
        if self.method == "refine-v5-end" and self.end_window_seconds <= 0:
            raise ValueError("dead-state refinement requires a positive end window")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "endWindowSeconds": self.end_window_seconds,
            "missingTransition": "exact-noop",
            "rescueMissingIntervals": False,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DeadStateRefinementConfig:
        result = cls(
            method=str(value["method"]),
            end_window_seconds=float(value.get("endWindowSeconds", 1.0)),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DeadStatePrediction:
    prepared: PreparedRecording
    primary: tuple[DecodedInterval, ...]
    v5_composed: tuple[DecodedInterval, ...]
    candidate: tuple[DecodedInterval, ...]
    transitions: tuple[DeadStateDetection, ...]


class _InverseDeadStateModel:
    """Expose a dead-state head as its live-probability complement."""

    def __init__(self, model: LogisticModel) -> None:
        self.model = model

    def predict(self, values: np.ndarray) -> np.ndarray:
        return 1.0 - self.model.predict(values)


def _transition_target(
    times: np.ndarray,
    rallies: Sequence[Any],
    *,
    before_end_seconds: float,
    after_end_seconds: float,
    pre_serve_setup_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels, mask = end_transition_labels_for_times(
        times,
        rallies,
        before_seconds=before_end_seconds,
        after_seconds=after_end_seconds,
    )
    if not math.isfinite(pre_serve_setup_seconds) or pre_serve_setup_seconds < 0:
        raise ValueError("pre-serve setup seconds must be non-negative")
    if pre_serve_setup_seconds == 0:
        return labels, mask
    # Already-dead setup is deliberately negative: the head represents a new
    # live-to-dead transition, not the algebraic complement of rally-live.
    for rally in rallies:
        setup = (
            (times >= max(0.0, float(rally.start) - pre_serve_setup_seconds))
            & (times < float(rally.start))
        )
        mask |= setup
        labels[setup] = 0.0
    return labels, mask


def _global_dead_target(
    times: np.ndarray,
    rallies: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact sample-wise complement of the rally-live target."""

    live = rally_labels_for_times(times, rallies)
    return (1.0 - live).astype(np.float32, copy=False), np.ones(len(times), dtype=bool)


def _target_for_mode(
    times: np.ndarray,
    rallies: Sequence[Any],
    *,
    target_mode: str,
    before_end_seconds: float,
    after_end_seconds: float,
    pre_serve_setup_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if target_mode == END_TRANSITION_TARGET_MODE:
        return _transition_target(
            times,
            rallies,
            before_end_seconds=before_end_seconds,
            after_end_seconds=after_end_seconds,
            pre_serve_setup_seconds=pre_serve_setup_seconds,
        )
    if target_mode == GLOBAL_DEAD_TARGET_MODE:
        return _global_dead_target(times, rallies)
    raise ValueError(
        f"target_mode must be one of {sorted(DEAD_STATE_TARGET_MODES)}"
    )


def _target_metadata(
    target_mode: str,
    *,
    before_end_seconds: float,
    after_end_seconds: float,
    pre_serve_setup_seconds: float,
) -> dict[str, Any]:
    if target_mode == GLOBAL_DEAD_TARGET_MODE:
        return {
            "id": GLOBAL_DEAD_STATE_TARGET_ID,
            "mode": GLOBAL_DEAD_TARGET_MODE,
            "scope": "all valid samples after ignored-interval masking",
            "positiveDefinition": "samples outside every gold rally interval",
            "negativeDefinition": "samples inside a gold rally interval",
            "boundaryDefinition": "half-open gold rally intervals under serve-contact-to-dead-ball-v1",
            "algebraicRedundancy": {
                "isInverseRallyLiveControl": True,
                "identity": "global-dead = 1 - rally-live on every valid sample",
                "note": (
                    "With identical inputs, the same inverse-frequency class-weighting "
                    "rule, and matched optimization, this is the sign-inverted rally-live "
                    "logistic objective (p_dead = 1 - p_live at its optimum). It is a "
                    "control, not additional model capacity."
                ),
            },
        }
    if target_mode == END_TRANSITION_TARGET_MODE:
        return {
            "id": DEAD_STATE_TARGET_ID,
            "mode": END_TRANSITION_TARGET_MODE,
            "scope": "local windows around rally ends plus pre-serve negative controls",
            "beforeEndSeconds": before_end_seconds,
            "afterEndSeconds": after_end_seconds,
            "preServeSetupSeconds": pre_serve_setup_seconds,
            "positiveDefinition": "samples at/after a rally end in its local end window",
            "negativeDefinition": (
                "samples before the end plus pre-serve setup controls; this is a "
                "transition target, not inverse rally-live"
            ),
            "boundaryDefinition": "rally end under serve-contact-to-dead-ball-v1",
            "algebraicRedundancy": {
                "isInverseRallyLiveControl": False,
                "note": "This local transition target is not the complement of rally-live.",
            },
        }
    raise ValueError(
        f"target_mode must be one of {sorted(DEAD_STATE_TARGET_MODES)}"
    )


def _decode_near_end(
    times: np.ndarray,
    probabilities: np.ndarray,
    end: float,
    next_start: float | None,
    decoder: DeadStateDecoderConfig,
    refinement: DeadStateRefinementConfig,
    duration: float,
) -> DeadStateDetection | None:
    anchor = max(0.0, end - refinement.end_window_seconds)
    local_decoder = replace(
        decoder,
        min_after_serve_seconds=0.0,
        max_after_serve_seconds=(end + refinement.end_window_seconds - anchor),
    )
    detection = decode_dead_state_after_serve(
        times,
        probabilities,
        anchor,
        local_decoder,
        next_serve_time=(
            next_start if next_start is not None and next_start > anchor else None
        ),
        duration=duration,
    )
    if detection is None:
        return None
    # Keep the learned time offset inside the validation-selected local window.
    return DeadStateDetection(
        min(
            duration,
            end + refinement.end_window_seconds,
            max(end - refinement.end_window_seconds, detection.time),
        ),
        detection.confidence,
    )


def _refine_v5_ends(
    intervals: Sequence[DecodedInterval],
    transitions: Sequence[DeadStateDetection | None],
    *,
    duration: float,
    sample_seconds: float,
) -> tuple[DecodedInterval, ...]:
    if len(intervals) != len(transitions):
        raise ValueError("one dead-state transition slot is required per v5 interval")
    rows: list[DecodedInterval] = []
    for index, (interval, transition) in enumerate(
        zip(intervals, transitions, strict=True)
    ):
        if transition is None:
            rows.append(interval)
            continue
        next_start = intervals[index + 1].start if index + 1 < len(intervals) else duration
        refined_end = min(duration, next_start, transition.time)
        if refined_end < interval.start + sample_seconds - 1e-9:
            rows.append(interval)
            continue
        rows.append(
            DecodedInterval(
                interval.start,
                refined_end,
                max(interval.confidence, transition.confidence),
            )
        )
    return tuple(rows)


def _predictions_for(
    inputs: Sequence[DeadBallInputs],
    decoder: DeadStateDecoderConfig,
    refinement: DeadStateRefinementConfig,
) -> list[DeadStatePrediction]:
    decoder.validate()
    refinement.validate()
    predictions: list[DeadStatePrediction] = []
    for item in inputs:
        if refinement.method == "v5-noop":
            predictions.append(
                DeadStatePrediction(
                    item.prepared,
                    item.primary,
                    item.v4_composed,
                    item.v4_composed,
                    (),
                )
            )
            continue
        transition_slots: list[DeadStateDetection | None] = []
        for index, interval in enumerate(item.v4_composed):
            next_start = (
                item.v4_composed[index + 1].start
                if index + 1 < len(item.v4_composed)
                else None
            )
            transition_slots.append(
                _decode_near_end(
                    item.prepared.sequence.times,
                    item.dead_probabilities,
                    interval.end,
                    next_start,
                    decoder,
                    refinement,
                    item.prepared.sequence.metadata.duration,
                )
            )
        transitions = tuple(item for item in transition_slots if item is not None)
        candidate = _refine_v5_ends(
            item.v4_composed,
            transition_slots,
            duration=item.prepared.sequence.metadata.duration,
            sample_seconds=1.0 / _effective_analysis_fps(item.prepared),
        )
        predictions.append(
            DeadStatePrediction(
                item.prepared,
                item.primary,
                item.v4_composed,
                candidate,
                transitions,
            )
        )
    return predictions


def _strict_count(slice_metrics: dict[str, Any]) -> int:
    return int(round(slice_metrics["rallies"] * slice_metrics["strictMatchRecall"]))


def _endpoint_metrics(
    predictions: Sequence[DeadStatePrediction],
    field: str,
    tolerance: float,
) -> dict[str, Any]:
    truth_count = prediction_count = matched_count = 0
    errors: list[float] = []
    slice_totals: dict[str, list[int]] = {}
    for prediction in predictions:
        recording = prediction.prepared.recording
        predicted = [
            DeadStateDetection(item.end, item.confidence)
            for item in getattr(prediction, field)
            if not any(
                blocked.start <= item.end < blocked.end
                for blocked in recording.ignored_intervals
            )
        ]
        matches = match_serve_contacts(
            [item.end for item in recording.rallies], predicted, tolerance
        )
        truth_count += len(recording.rallies)
        prediction_count += len(predicted)
        matched_count += len(matches)
        errors.extend(error for _, _, error in matches)
        for name, rallies in _slice_truth(recording).items():
            matched = len(
                match_serve_contacts(
                    [item.end for item in rallies], predicted, tolerance
                )
            )
            bucket = slice_totals.setdefault(name, [0, 0])
            bucket[0] += matched
            bucket[1] += len(rallies)
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
        "sliceRecall": {
            name: {
                "ends": total,
                "matchedEnds": matched,
                "recall": matched / total if total else 1.0,
            }
            for name, (matched, total) in slice_totals.items()
        },
    }


def _transition_metrics(
    predictions: Sequence[DeadStatePrediction], tolerance: float
) -> dict[str, Any]:
    truth_count = prediction_count = matched_count = 0
    errors: list[float] = []
    for prediction in predictions:
        recording = prediction.prepared.recording
        detected = [
            item
            for item in prediction.transitions
            if not any(
                blocked.start <= item.time < blocked.end
                for blocked in recording.ignored_intervals
            )
        ]
        matches = match_serve_contacts(
            [item.end for item in recording.rallies], detected, tolerance
        )
        truth_count += len(recording.rallies)
        prediction_count += len(detected)
        matched_count += len(matches)
        errors.extend(error for _, _, error in matches)
    precision = matched_count / prediction_count if prediction_count else 0.0
    recall = matched_count / truth_count if truth_count else 1.0
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "toleranceSeconds": tolerance,
        "trueEnds": truth_count,
        "emittedTransitions": prediction_count,
        "matchedTransitions": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "timingMaeSeconds": float(np.mean(absolute)) if len(absolute) else None,
        "timingP90Seconds": float(np.percentile(absolute, 90)) if len(absolute) else None,
    }


def _is_feasible(aggregate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        aggregate["matchedRallies"] >= baseline["matchedRallies"]
        and _strict_count(aggregate["outcomeSlices"]["ordinaryLong"])
        >= _strict_count(baseline["outcomeSlices"]["ordinaryLong"])
        and aggregate["liveTimeRecall"] >= baseline["liveTimeRecall"] - 0.01 - 1e-12
        and aggregate["liveTimePrecision"]
        >= baseline["liveTimePrecision"] - 0.01 - 1e-12
    )


def _selection_key(
    predictions: Sequence[DeadStatePrediction], aggregate: dict[str, Any]
) -> tuple[float, ...]:
    endpoint = _endpoint_metrics(predictions, "candidate", 0.5)
    short = aggregate["outcomeSlices"]["shortAtMost3Seconds"]
    return (
        float(endpoint["matchedEnds"]),
        float(_strict_count(short)),
        aggregate["eventF1"],
        aggregate["timeIoU"],
        -aggregate["deadSecondsRetained"],
        aggregate["liveTimeRecall"],
        aggregate["liveTimePrecision"],
        -float(aggregate["predictedRallies"]),
    )


def _tune_refinement(
    inputs: Sequence[DeadBallInputs],
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[DeadStateDecoderConfig, DeadStateRefinementConfig, dict[str, Any]]:
    if not inputs:
        raise ManifestError("dead-state refinement selection requires validation recordings")
    noop_decoder = DeadStateDecoderConfig(
        dead_threshold=0.95,
        live_reset_threshold=0.4,
        minimum_live_samples=1,
        minimum_dead_samples=2,
        min_after_serve_seconds=0.0,
        max_after_serve_seconds=1.0,
    )
    noop = DeadStateRefinementConfig("v5-noop", 0.0)
    noop_predictions = _predictions_for(inputs, noop_decoder, noop)
    baseline = _interval_report(noop_predictions, "candidate")["aggregate"]
    candidates: list[
        tuple[
            DeadStateDecoderConfig,
            DeadStateRefinementConfig,
            list[DeadStatePrediction],
            dict[str, Any],
        ]
    ] = [(noop_decoder, noop, noop_predictions, baseline)]
    if progress is not None:
        progress("Selecting low-to-high dead-state end refinement on validation")
    for dead_threshold in (0.60, 0.75, 0.90):
        for live_reset in (0.20, 0.40):
            if live_reset >= dead_threshold:
                continue
            for persistence in (1, 2):
                for window in (0.75, 1.50):
                    for offset in (-0.25, 0.0, 0.25):
                        decoder = DeadStateDecoderConfig(
                            dead_threshold=dead_threshold,
                            live_reset_threshold=live_reset,
                            minimum_live_samples=1,
                            minimum_dead_samples=persistence,
                            min_after_serve_seconds=0.0,
                            max_after_serve_seconds=2.0 * window,
                            time_offset_seconds=offset,
                        )
                        refinement = DeadStateRefinementConfig(
                            "refine-v5-end", window
                        )
                        predictions = _predictions_for(inputs, decoder, refinement)
                        aggregate = _interval_report(predictions, "candidate")[
                            "aggregate"
                        ]
                        candidates.append(
                            (decoder, refinement, predictions, aggregate)
                        )
    feasible = [row for row in candidates if _is_feasible(row[3], baseline)]
    selected = max(feasible, key=lambda row: _selection_key(row[2], row[3]))
    endpoint = _endpoint_metrics(selected[2], "candidate", 0.5)
    selection = {
        "status": "selected-on-validation",
        "objective": (
            "guard overall and ordinary-long strict matches plus live-time recall and "
            "precision within 0.01; then maximize endpoints within 0.5s, short strict "
            "matches, event F1, time IoU, and minimize retained dead time"
        ),
        "guardrails": {
            "minimumOverallStrictMatches": baseline["matchedRallies"],
            "minimumOrdinaryLongStrictMatches": _strict_count(
                baseline["outcomeSlices"]["ordinaryLong"]
            ),
            "minimumLiveTimeRecall": baseline["liveTimeRecall"] - 0.01,
            "minimumLiveTimePrecision": baseline["liveTimePrecision"] - 0.01,
        },
        "candidateCount": len(candidates),
        "feasibleCandidateCount": len(feasible),
        "deadStateDecoder": selected[0].to_dict(),
        "refinement": selected[1].to_dict(),
        "validationMetrics": selected[3],
        "validationEndpointAt0.5Seconds": endpoint,
    }
    return selected[0], selected[1], selection


def _validate_target(labels: Sequence[np.ndarray]) -> None:
    samples = sum(len(item) for item in labels)
    positives = sum(int(np.sum(item > 0.5)) for item in labels)
    if not samples or not positives or positives == samples:
        raise ManifestError(
            "dead-state transition target must contain positive and negative samples"
        )


def _eligible_target_mask(
    sample_mask: np.ndarray, target_mask: np.ndarray
) -> np.ndarray:
    if (
        sample_mask.ndim != 1
        or target_mask.ndim != 1
        or sample_mask.shape != target_mask.shape
    ):
        raise ValueError("sample and dead-state target masks must be aligned vectors")
    return sample_mask.astype(bool, copy=False) & target_mask.astype(bool, copy=False)


def _crossfit_epoch_cap(
    prepared: Sequence[PreparedRecording],
    labels: dict[str, np.ndarray],
    target_masks: dict[str, np.ndarray],
    rally_model: LogisticModel,
    config: TrainingConfig,
    retained_inputs: np.ndarray,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    groups = sorted({item.recording.source_group for item in prepared})
    if len(groups) < 3:
        raise ManifestError("dead-state epoch selection requires three training source groups")
    summaries: list[dict[str, Any]] = []
    epochs: list[int] = []
    for index, held_out in enumerate(groups, start=1):
        train = [item for item in prepared if item.recording.source_group != held_out]
        validation = [
            item for item in prepared if item.recording.source_group == held_out
        ]
        if progress is not None:
            progress(f"Selecting dead-state epoch {index}/{len(groups)}: {held_out}")

        def eligible(item: PreparedRecording) -> np.ndarray:
            return _eligible_target_mask(
                item.sample_mask, target_masks[item.recording.id]
            )

        model = train_logistic_model(
            [
                _masked_dead_ball_values(
                    item.contextual_values[eligible(item)], retained_inputs
                )
                for item in train
            ],
            [labels[item.recording.id][eligible(item)] for item in train],
            [
                _masked_dead_ball_values(
                    item.contextual_values[eligible(item)], retained_inputs
                )
                for item in validation
            ],
            [labels[item.recording.id][eligible(item)] for item in validation],
            rally_model.feature_config,
            rally_model.feature_names,
            rally_model.decoder,
            config,
            prediction_task=DEAD_STATE_TASK,
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
    dead_state_model: LogisticModel,
    *,
    manifest_sha256: str | None = None,
) -> tuple[
    ServeDecoderConfig,
    ServeCompositionConfig,
    DeadStateDecoderConfig,
    DeadStateRefinementConfig,
]:
    serve_decoder, serve_composition = _validate_model_pair(
        rally_model, serve_model, manifest_sha256=manifest_sha256
    )
    if dead_state_model.prediction_task != DEAD_STATE_TASK:
        raise ModelError("dead-state specialist must predict dead-time-state")
    if dead_state_model.feature_version != rally_model.feature_version:
        raise ModelError("dead-state specialist uses a different feature version")
    if dead_state_model.feature_config != rally_model.feature_config:
        raise ModelError("dead-state specialist uses a different feature configuration")
    if dead_state_model.feature_names != rally_model.feature_names:
        raise ModelError("dead-state specialist uses a different feature signature")
    summary = dead_state_model.training_summary
    if summary.get("rallyModelSha256") != rally_model.artifact_sha256:
        raise ModelError("dead-state specialist is bound to a different rally model")
    if summary.get("serveModelSha256") != serve_model.artifact_sha256:
        raise ModelError("dead-state specialist is bound to a different serve model")
    if summary.get("manifestSha256") != rally_model.training_summary.get(
        "manifestSha256"
    ):
        raise ModelError("dead-state specialist was trained from a different manifest")
    if manifest_sha256 is not None and summary.get("manifestSha256") != manifest_sha256:
        raise ModelError("dead-state specialist differs from the immutable evaluation manifest")
    try:
        decoder = DeadStateDecoderConfig.from_dict(summary["selectedDeadStateDecoder"])
        refinement = DeadStateRefinementConfig.from_dict(
            summary["selectedRefinement"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(f"dead-state selection metadata is invalid: {error}") from error
    return serve_decoder, serve_composition, decoder, refinement


def _input_profile_metadata(model: LogisticModel) -> dict[str, Any]:
    stored = model.training_summary.get("deadStateInputProfile")
    if isinstance(stored, dict):
        return stored
    return {
        "id": FULL_DEAD_BALL_INPUT_PROFILE,
        "retainedInputs": len(model.feature_names),
        "removedInputs": 0,
        "legacyDefault": True,
    }


def _binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predicted = probabilities >= threshold
    positive = labels > 0.5
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    true_negative = int(np.sum(~predicted & ~positive))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 1.0
    return {
        "threshold": threshold,
        "samples": len(labels),
        "positiveSamples": int(np.sum(positive)),
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (true_positive + true_negative) / len(labels) if len(labels) else 1.0,
        "balancedAccuracy": (recall + specificity) / 2.0,
    }


def _local_target_metrics(
    prepared: Sequence[PreparedRecording],
    model: LogisticModel,
    decoder: DeadStateDecoderConfig,
) -> dict[str, Any]:
    target = model.training_summary["deadStateTarget"]
    target_mode = str(
        target.get(
            "mode",
            (
                GLOBAL_DEAD_TARGET_MODE
                if target.get("id") == GLOBAL_DEAD_STATE_TARGET_ID
                else END_TRANSITION_TARGET_MODE
            ),
        )
    )
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for item in prepared:
        item_labels, target_mask = _target_for_mode(
            item.sequence.times,
            item.recording.rallies,
            target_mode=target_mode,
            before_end_seconds=float(
                target.get("beforeEndSeconds", DEFAULT_BEFORE_END_SECONDS)
            ),
            after_end_seconds=float(
                target.get("afterEndSeconds", DEFAULT_AFTER_END_SECONDS)
            ),
            pre_serve_setup_seconds=float(
                target.get(
                    "preServeSetupSeconds", DEFAULT_PRE_SERVE_SETUP_SECONDS
                )
            ),
        )
        eligible = _eligible_target_mask(item.sample_mask, target_mask)
        labels.append(item_labels[eligible])
        probabilities.append(model.predict(item.contextual_values)[eligible])
    joined_labels = np.concatenate(labels) if labels else np.empty(0, dtype=np.float32)
    joined_probabilities = (
        np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
    )
    return {
        "targetMode": target_mode,
        "scope": (
            "all-valid-samples"
            if target_mode == GLOBAL_DEAD_TARGET_MODE
            else "local-end-transition-samples"
        ),
        "at0.5": _binary_metrics(joined_labels, joined_probabilities, 0.5),
        "atSelectedDeadThreshold": _binary_metrics(
            joined_labels, joined_probabilities, decoder.dead_threshold
        ),
    }


def _global_dead_as_rally_report(
    prepared: Sequence[PreparedRecording],
    dead_state_model: LogisticModel,
    rally_decoder: DecoderConfig,
) -> dict[str, Any]:
    probabilities = [
        1.0 - dead_state_model.predict(item.contextual_values) for item in prepared
    ]
    recordings, aggregate = _evaluate_prepared_probabilities(
        prepared, probabilities, rally_decoder
    )
    return {
        "decoder": rally_decoder.to_dict(),
        "recordings": recordings,
        "aggregate": aggregate,
    }


def _global_dead_rally_decoder(
    dead_state_model: LogisticModel,
) -> tuple[DecoderConfig, dict[str, Any]] | None:
    stored = dead_state_model.training_summary.get("globalDeadAsRallyControl")
    if stored is None:
        return None
    if not isinstance(stored, dict):
        raise ModelError("global-dead inverse-rally control metadata must be an object")
    transform = stored.get("probabilityTransform")
    if transform is not None and transform != "one-minus-dead-probability":
        raise ModelError("global-dead inverse-rally probability transform is invalid")
    try:
        decoder = DecoderConfig.from_dict(stored["selectedDecoder"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(
            f"global-dead inverse-rally decoder metadata is invalid: {error}"
        ) from error
    return decoder, stored


def _build_report(
    prepared: Sequence[PreparedRecording],
    rally_model: LogisticModel,
    serve_model: LogisticModel,
    dead_state_model: LogisticModel,
    serve_decoder: ServeDecoderConfig,
    serve_composition: ServeCompositionConfig,
    decoder: DeadStateDecoderConfig,
    refinement: DeadStateRefinementConfig,
    *,
    dataset: str,
    manifest_sha256: str,
    split: str,
    retrospective: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = [
        _prediction_inputs(
            item,
            rally_model,
            serve_model,
            dead_state_model,
            serve_decoder,
            serve_composition,
        )
        for item in prepared
    ]
    noop_decoder = DeadStateDecoderConfig(
        dead_threshold=0.95,
        live_reset_threshold=0.4,
        minimum_live_samples=1,
        minimum_dead_samples=2,
        min_after_serve_seconds=0.0,
        max_after_serve_seconds=1.0,
    )
    noop = DeadStateRefinementConfig("v5-noop", 0.0)
    v5_predictions = _predictions_for(inputs, noop_decoder, noop)
    selected_predictions = _predictions_for(inputs, decoder, refinement)
    baseline = _interval_report(v5_predictions, "primary")
    v5 = _interval_report(v5_predictions, "candidate")
    selected = _interval_report(selected_predictions, "candidate")
    video_seconds = sum(item.sequence.metadata.duration for item in prepared)
    processing_seconds = time.perf_counter() - started
    target = dead_state_model.training_summary.get("deadStateTarget", {})
    target_mode = str(target.get("mode", END_TRANSITION_TARGET_MODE))
    limitations = [
        "Validation is one reused grass source group and remains tuning evidence only.",
        "Any previously inspected non-validation result must be marked retrospective.",
        "The v1 dead-state head only refines existing v5 intervals and cannot rescue a missing rally.",
        "The centered feature context makes this an offline, non-causal boundary model.",
    ]
    if target_mode == GLOBAL_DEAD_TARGET_MODE:
        limitations.append(
            "The global-dead target is exactly 1 - rally-live on valid samples; with "
            "identical inputs and matched optimization it adds no model capacity."
        )
    else:
        limitations.append(
            "Pre-serve setup is a negative transition-control target, not a claim that "
            "setup time is live play."
        )
    report = {
        "schemaVersion": 1,
        "experiment": (
            GLOBAL_DEAD_STATE_EXPERIMENT_ID
            if target_mode == GLOBAL_DEAD_TARGET_MODE
            else DEAD_STATE_EXPERIMENT_ID
        ),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "producer": f"volleycut-analysis/{__version__}",
        "dataset": dataset,
        "manifestSha256": manifest_sha256,
        "split": split,
        "assessmentRole": assessment_role_for_split(
            split, retrospective=retrospective
        ),
        "matching": {"minimumIntervalIoU": 0.5},
        "models": {
            "rally": {
                "sha256": rally_model.artifact_sha256,
                "task": rally_model.prediction_task,
            },
            "serve": {
                "sha256": serve_model.artifact_sha256,
                "task": serve_model.prediction_task,
            },
            "deadState": {
                "sha256": dead_state_model.artifact_sha256,
                "task": dead_state_model.prediction_task,
            },
        },
        "target": target,
        "deadStateInputProfile": _input_profile_metadata(dead_state_model),
        "selection": dead_state_model.training_summary.get("selection"),
        "deadStateDecoder": decoder.to_dict(),
        "refinement": refinement.to_dict(),
        "localTarget": _local_target_metrics(prepared, dead_state_model, decoder),
        "endpointAccuracy": {
            "v5Composition": {
                "at0.25Seconds": _endpoint_metrics(v5_predictions, "candidate", 0.25),
                "at0.5Seconds": _endpoint_metrics(v5_predictions, "candidate", 0.5),
                "at1Second": _endpoint_metrics(v5_predictions, "candidate", 1.0),
            },
            "selected": {
                "at0.25Seconds": _endpoint_metrics(selected_predictions, "candidate", 0.25),
                "at0.5Seconds": _endpoint_metrics(selected_predictions, "candidate", 0.5),
                "at1Second": _endpoint_metrics(selected_predictions, "candidate", 1.0),
            },
            "emittedTransitions": {
                "at0.25Seconds": _transition_metrics(selected_predictions, 0.25),
                "at0.5Seconds": _transition_metrics(selected_predictions, 0.5),
                "at1Second": _transition_metrics(selected_predictions, 1.0),
            },
        },
        "baseline": baseline,
        "v5Composition": v5,
        "selected": {
            **selected,
            "deadStateDecoder": decoder.to_dict(),
            "refinement": refinement.to_dict(),
        },
        "deltaSelectedVsV5": _delta(selected["aggregate"], v5["aggregate"]),
        "featureImportance": {
            "interpretation": (
                "Standardized coefficients are associations, not causal importance; the "
                "dead-state head uses centered future context and is offline."
            ),
            "deadStateTop": _top_features(dead_state_model),
        },
        "processing": {
            "predictionWallClockSeconds": processing_seconds,
            "videoSeconds": video_seconds,
            "predictionToVideoRatio": (
                processing_seconds / video_seconds if video_seconds else None
            ),
            "note": "This excludes feature extraction/cache loading.",
        },
        "limitations": limitations,
    }
    if target_mode == GLOBAL_DEAD_TARGET_MODE:
        inverse_control = _global_dead_rally_decoder(dead_state_model)
        if inverse_control is None:
            report["globalDeadAsRally"] = {
                "interpretation": (
                    "This global-dead artifact predates persisted inverse-rally decoder "
                    "selection metadata."
                ),
            }
        else:
            selected_live_decoder, inverse_metadata = inverse_control
            report["globalDeadAsRally"] = {
                "interpretation": (
                    "One minus the global-dead probability, decoded as rally-live. This "
                    "is a complement/parity control, not additional model capacity."
                ),
                "selection": inverse_metadata.get("selection"),
                "frozenEnhancedRallyDecoder": _global_dead_as_rally_report(
                    prepared, dead_state_model, rally_model.decoder
                ),
                "validationSelectedDecoder": _global_dead_as_rally_report(
                    prepared, dead_state_model, selected_live_decoder
                ),
            }
    return report


def train_dead_state_dataset(
    manifest_path: str | Path,
    rally_model_path: str | Path,
    serve_model_path: str | Path,
    model_destination: str | Path,
    cache_dir: str | Path,
    *,
    before_end_seconds: float = DEFAULT_BEFORE_END_SECONDS,
    after_end_seconds: float = DEFAULT_AFTER_END_SECONDS,
    pre_serve_setup_seconds: float = DEFAULT_PRE_SERVE_SETUP_SECONDS,
    target_mode: str = END_TRANSITION_TARGET_MODE,
    dead_state_input_profile: str = FULL_DEAD_BALL_INPUT_PROFILE,
    training_config: TrainingConfig | None = None,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if target_mode not in DEAD_STATE_TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(DEAD_STATE_TARGET_MODES)}"
        )
    for label, value in (
        ("before-end", before_end_seconds),
        ("after-end", after_end_seconds),
        ("pre-serve setup", pre_serve_setup_seconds),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"dead-state {label} seconds must be non-negative")
    if (
        target_mode == END_TRANSITION_TARGET_MODE
        and before_end_seconds == 0
        and after_end_seconds == 0
    ):
        raise ValueError("dead-state target needs a positive end-transition window")
    if dead_state_input_profile not in DEAD_BALL_INPUT_PROFILES:
        raise ValueError(
            "dead_state_input_profile must be one of "
            f"{sorted(DEAD_BALL_INPUT_PROFILES)}"
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
    serve_model = load_model(serve_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    serve_decoder, serve_composition = _validate_model_pair(
        rally_model, serve_model, manifest_sha256=manifest_sha256
    )
    training_rows = manifest.for_split("train")
    validation_rows = manifest.for_split("validation")
    if not training_rows or not validation_rows:
        raise ManifestError("dead-state training requires train and validation recordings")
    training = _prepare_many(
        training_rows, rally_model.feature_config, cache_dir, progress=progress
    )
    validation = _prepare_many(
        validation_rows, rally_model.feature_config, cache_dir, progress=progress
    )
    for item in (*training, *validation):
        if item.contextual_names != rally_model.feature_names:
            raise ModelError(f"feature signature mismatch for {item.recording.id}")

    targets = {
        item.recording.id: _target_for_mode(
            item.sequence.times,
            item.recording.rallies,
            target_mode=target_mode,
            before_end_seconds=before_end_seconds,
            after_end_seconds=after_end_seconds,
            pre_serve_setup_seconds=pre_serve_setup_seconds,
        )
        for item in (*training, *validation)
    }
    labels = {name: value[0] for name, value in targets.items()}
    target_masks = {name: value[1] for name, value in targets.items()}
    _validate_target(
        [
            labels[item.recording.id][
                _eligible_target_mask(
                    item.sample_mask, target_masks[item.recording.id]
                )
            ]
            for item in validation
        ]
    )
    retained_inputs = _dead_ball_input_mask(
        rally_model.feature_names, dead_state_input_profile
    )
    config = training_config or TrainingConfig(epochs=120)
    epoch_cap, epoch_folds = _crossfit_epoch_cap(
        training,
        labels,
        target_masks,
        rally_model,
        config,
        retained_inputs,
        progress=progress,
    )
    if progress is not None:
        progress(f"Fitting final dead-state head for {epoch_cap} fixed epochs")
    final_config = replace(config, epochs=epoch_cap, patience=epoch_cap)

    def eligible(item: PreparedRecording) -> np.ndarray:
        return _eligible_target_mask(
            item.sample_mask, target_masks[item.recording.id]
        )

    dead_state_model = train_logistic_model(
        [
            _masked_dead_ball_values(
                item.contextual_values[eligible(item)], retained_inputs
            )
            for item in training
        ],
        [labels[item.recording.id][eligible(item)] for item in training],
        [],
        [],
        rally_model.feature_config,
        rally_model.feature_names,
        rally_model.decoder,
        final_config,
        prediction_task=DEAD_STATE_TASK,
    )
    dead_state_model.feature_version = rally_model.feature_version
    inverse_rally_control: dict[str, Any] | None = None
    if target_mode == GLOBAL_DEAD_TARGET_MODE:
        if progress is not None:
            progress("Selecting the inverse global-dead rally decoder on validation")
        inverse_decoder, inverse_selection = _tune_decoder(
            validation,
            _InverseDeadStateModel(dead_state_model),
            rally_model.decoder,
        )
        inverse_rally_control = {
            "probabilityTransform": "one-minus-dead-probability",
            "selectedDecoder": inverse_decoder.to_dict(),
            "selection": inverse_selection,
        }
    validation_inputs = [
        _prediction_inputs(
            item,
            rally_model,
            serve_model,
            dead_state_model,
            serve_decoder,
            serve_composition,
        )
        for item in validation
    ]
    selected_decoder, selected_refinement, selection = _tune_refinement(
        validation_inputs, progress=progress
    )
    profile = {
        "id": dead_state_input_profile,
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
            "removed inputs are zeroed before normalization during fitting; their "
            "persisted standardized coefficients are exactly zero"
        ),
    }
    dead_state_model.training_summary.update(
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
            "deadStateTarget": _target_metadata(
                target_mode,
                before_end_seconds=before_end_seconds,
                after_end_seconds=after_end_seconds,
                pre_serve_setup_seconds=pre_serve_setup_seconds,
            ),
            "deadStateInputProfile": profile,
            "epochSelection": {
                "method": "leave-one-training-source-group-out-median-best-epoch-v1",
                "selectedEpoch": epoch_cap,
                "folds": epoch_folds,
            },
            "selectedDeadStateDecoder": selected_decoder.to_dict(),
            "selectedRefinement": selected_refinement.to_dict(),
            "selection": selection,
            **(
                {"globalDeadAsRallyControl": inverse_rally_control}
                if inverse_rally_control is not None
                else {}
            ),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": __import__("cv2").__version__,
                "wallClockSeconds": round(time.perf_counter() - started, 3),
            },
        }
    )
    dead_state_model.save(destination)
    report = _build_report(
        validation,
        rally_model,
        serve_model,
        dead_state_model,
        serve_decoder,
        serve_composition,
        selected_decoder,
        selected_refinement,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split="validation",
    )
    report["deadStateModelPath"] = str(destination)
    if report_destination is not None:
        atomic_write_text(
            report_destination, json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def evaluate_dead_state_dataset(
    manifest_path: str | Path,
    rally_model_path: str | Path,
    serve_model_path: str | Path,
    dead_state_model_path: str | Path,
    cache_dir: str | Path,
    *,
    split: str = "test",
    retrospective: bool = False,
    output_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    destination = Path(output_path).expanduser().resolve() if output_path else None
    if destination is not None and destination.exists():
        raise ModelError(f"evaluation output already exists: {destination}")
    rally_model = load_model(rally_model_path)
    serve_model = load_model(serve_model_path)
    dead_state_model = load_model(dead_state_model_path)
    manifest = load_manifest(manifest_path)
    manifest_sha256 = _manifest_digest(manifest)
    serve_decoder, serve_composition, decoder, refinement = _validate_model_triplet(
        rally_model,
        serve_model,
        dead_state_model,
        manifest_sha256=manifest_sha256,
    )
    recordings = manifest.for_split(split)
    if not recordings:
        raise ManifestError(f"manifest has no recordings in split {split!r}")
    evaluation_groups = {item.source_group for item in recordings}
    for model in (rally_model, serve_model, dead_state_model):
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
        dead_state_model,
        serve_decoder,
        serve_composition,
        decoder,
        refinement,
        dataset=manifest.name,
        manifest_sha256=manifest_sha256,
        split=split,
        retrospective=retrospective,
    )
    if destination is not None:
        atomic_write_text(destination, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
