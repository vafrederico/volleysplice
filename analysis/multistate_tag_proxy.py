"""Weak ace/service-fault proxy helpers for the multistate result study.

The target is intentionally narrow: one row at each annotated serve, positive
only when that rally has an ``ace`` or ``service-fault`` tag.  Durations and
future transition annotations never create proxy labels.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .config import DecoderConfig, FeatureConfig, TrainingConfig
from .feature_experiments import FeatureExperimentError, _model_fingerprint
from .model import LogisticModel, train_logistic_model
from .multistate import MultistateState
from .multistate_feature_study import state_targets
from .pipeline import PreparedRecording


TAG_PROXY_FEATURE_NAMES = tuple(
    f"t+{offset}s/{name}"
    for offset in (1, 2)
    for name in (
        "player_motion_active_fraction",
        "player_motion_collapse",
        "synchronized_stand_down",
        "receiving_formation_change_proxy",
        "audio_contact_like_transient",
        "audio_onset_cadence",
        "audio_cadence_collapse",
    )
) + (
    "transition/interaction/terminal_transient_x_motion_collapse",
    "transition/interaction/cadence_collapse_x_formation_contraction",
    "transition/interaction/visibility_x_residual_player_motion",
)
TAG_PROXY_EPOCHS = 60
TAG_PROXY_MIN_PROBABILITY = 0.10
TAG_PROXY_MAX_PROBABILITY = 0.90


def _seed(base: int, parts: Sequence[str]) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _feature_indexes(item: PreparedRecording) -> tuple[int, ...]:
    by_name = {name: index for index, name in enumerate(item.contextual_names)}
    missing = [name for name in TAG_PROXY_FEATURE_NAMES if name not in by_name]
    if missing:
        raise FeatureExperimentError(
            f"tag proxy feature signature is unavailable: {missing[:5]}"
        )
    return tuple(by_name[name] for name in TAG_PROXY_FEATURE_NAMES)


def _outcome(tags: Sequence[str]) -> tuple[float, str]:
    values = set(tags)
    ace = "ace" in values
    fault = "service-fault" in values
    if ace and fault:
        raise FeatureExperimentError(
            "an immediate-result proxy rally cannot be both ace and service-fault"
        )
    if ace:
        return 1.0, "ace"
    if fault:
        return 1.0, "serviceFault"
    return 0.0, "ordinary"


def _serve_indexes(item: PreparedRecording) -> tuple[int, ...]:
    targets = state_targets(item)
    serve = np.flatnonzero(
        item.sample_mask & (targets == int(MultistateState.SERVE))
    )
    result: list[int] = []
    for rally in item.recording.rallies:
        matches = serve[
            (item.sequence.times[serve] >= rally.start - 0.5)
            & (item.sequence.times[serve] < rally.end)
        ]
        if len(matches) != 1:
            raise FeatureExperimentError(
                f"{item.recording.id}: rally at {rally.start:.3f}s has "
                f"{len(matches)} usable serve samples"
            )
        result.append(int(matches[0]))
    if len(set(result)) != len(result):
        raise FeatureExperimentError("tag proxy serve samples must be unique")
    return tuple(result)


@dataclass(frozen=True)
class TagProxyAnchors:
    values: np.ndarray
    labels: np.ndarray
    recording_ids: tuple[str, ...]
    source_groups: tuple[str, ...]
    outcome_tags: tuple[str, ...]
    serve_sample_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = len(self.labels)
        if self.values.shape != (rows, len(TAG_PROXY_FEATURE_NAMES)):
            raise ValueError("tag proxy anchor feature shape is invalid")
        if not np.isin(self.labels, (0.0, 1.0)).all():
            raise ValueError("tag proxy labels must be binary")
        if any(
            len(values) != rows
            for values in (
                self.recording_ids,
                self.source_groups,
                self.outcome_tags,
                self.serve_sample_indexes,
            )
        ):
            raise ValueError("tag proxy anchor metadata is misaligned")

    def summary(self) -> dict[str, Any]:
        groups = sorted(set(self.source_groups))
        outcomes = ("ace", "serviceFault", "ordinary")
        return {
            "samples": len(self.labels),
            "positive": int(np.sum(self.labels)),
            "negative": int(len(self.labels) - np.sum(self.labels)),
            "prevalence": float(np.mean(self.labels)) if len(self.labels) else None,
            "bySourceGroup": {
                group: {
                    "samples": sum(value == group for value in self.source_groups),
                    "positive": int(
                        sum(
                            source == group and label > 0.5
                            for source, label in zip(
                                self.source_groups, self.labels, strict=True
                            )
                        )
                    ),
                }
                for group in groups
            },
            "byOutcome": {
                outcome: sum(value == outcome for value in self.outcome_tags)
                for outcome in outcomes
            },
        }


def extract_tag_proxy_anchors(
    prepared: Sequence[PreparedRecording],
) -> TagProxyAnchors:
    values: list[np.ndarray] = []
    labels: list[float] = []
    recording_ids: list[str] = []
    source_groups: list[str] = []
    outcomes: list[str] = []
    sample_indexes: list[int] = []
    for item in prepared:
        feature_indexes = _feature_indexes(item)
        serves = _serve_indexes(item)
        for rally, serve_index in zip(item.recording.rallies, serves, strict=True):
            label, outcome = _outcome(rally.tags)
            values.append(item.contextual_values[serve_index, feature_indexes])
            labels.append(label)
            recording_ids.append(item.recording.id)
            source_groups.append(item.recording.source_group)
            outcomes.append(outcome)
            sample_indexes.append(serve_index)
    if not values:
        raise FeatureExperimentError("tag proxy requires at least one rally anchor")
    return TagProxyAnchors(
        values=np.ascontiguousarray(np.asarray(values, dtype=np.float32)),
        labels=np.asarray(labels, dtype=np.float32),
        recording_ids=tuple(recording_ids),
        source_groups=tuple(source_groups),
        outcome_tags=tuple(outcomes),
        serve_sample_indexes=tuple(sample_indexes),
    )


def tag_proxy_values(item: PreparedRecording) -> np.ndarray:
    return np.ascontiguousarray(item.contextual_values[:, _feature_indexes(item)])


def fit_tag_proxy(
    training: Sequence[PreparedRecording],
    *,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    seed_parts: Sequence[str],
) -> LogisticModel:
    anchors = extract_tag_proxy_anchors(training)
    positives = int(np.sum(anchors.labels))
    if positives <= 0 or positives >= len(anchors.labels):
        raise FeatureExperimentError("tag proxy training needs both target classes")
    config = replace(
        training_config,
        epochs=TAG_PROXY_EPOCHS,
        patience=TAG_PROXY_EPOCHS + 1,
        seed=_seed(training_config.seed, seed_parts),
    )
    return train_logistic_model(
        [anchors.values],
        [anchors.labels],
        [],
        [],
        feature_config,
        TAG_PROXY_FEATURE_NAMES,
        DecoderConfig(),
        config,
    )


def _prior_correct(model: LogisticModel, probabilities: np.ndarray) -> np.ndarray:
    positive = float(model.training_summary["positiveSamples"])
    negative = float(model.training_summary["negativeSamples"])
    if positive <= 0 or negative <= 0:
        raise FeatureExperimentError("tag proxy class counts must be positive")
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1 - 1e-8)
    logits = np.log(values) - np.log1p(-values) + math.log(positive / negative)
    corrected = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return np.clip(
        corrected, TAG_PROXY_MIN_PROBABILITY, TAG_PROXY_MAX_PROBABILITY
    )


def predict_tag_proxy(model: LogisticModel, anchors: TagProxyAnchors) -> np.ndarray:
    return _prior_correct(model, model.predict(anchors.values))


def predict_tag_proxy_sequence(
    model: LogisticModel, item: PreparedRecording
) -> np.ndarray:
    return _prior_correct(model, model.predict(tag_proxy_values(item)))


def tag_proxy_fingerprint(model: LogisticModel) -> str:
    return _model_fingerprint(model)


def _rank_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    ranks = np.arange(1, len(labels) + 1, dtype=np.float64)
    start = 0
    while start < len(ranks):
        end = start + 1
        while end < len(ranks) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = float(np.mean(ranks[start:end]))
        start = end
    restored = np.empty_like(ranks)
    restored[order] = ranks
    positives = labels > 0.5
    positive_count = int(np.sum(positives))
    negative_count = len(labels) - positive_count
    return float(
        (np.sum(restored[positives]) - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="mergesort")
    truth = labels[order] > 0.5
    positives = int(np.sum(truth))
    cumulative = np.cumsum(truth)
    precision = cumulative / np.arange(1, len(truth) + 1)
    return float(np.sum(precision[truth]) / positives)


def tag_proxy_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    baseline_prevalence: float,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or scores.shape != truth.shape or not len(truth):
        raise ValueError("tag proxy metric arrays must be aligned and non-empty")
    if not np.isin(truth, (0.0, 1.0)).all() or not np.isfinite(scores).all():
        raise ValueError("tag proxy metric inputs are invalid")
    if not 0.0 < baseline_prevalence < 1.0:
        raise ValueError("tag proxy baseline prevalence must be between zero and one")
    scores = np.clip(scores, 1e-8, 1.0 - 1e-8)
    baseline = np.full(len(truth), baseline_prevalence, dtype=np.float64)

    def losses(values: np.ndarray) -> tuple[float, float]:
        return (
            float(
                -np.mean(
                    truth * np.log(values) + (1.0 - truth) * np.log1p(-values)
                )
            ),
            float(np.mean(np.square(values - truth))),
        )

    log_loss, brier = losses(scores)
    baseline_log_loss, baseline_brier = losses(baseline)
    positives = int(np.sum(truth))
    negatives = len(truth) - positives
    return {
        "samples": len(truth),
        "positive": positives,
        "negative": negatives,
        "prevalence": float(np.mean(truth)),
        "meanProbability": float(np.mean(scores)),
        "logLoss": log_loss,
        "brier": brier,
        "baselinePrevalence": baseline_prevalence,
        "baselineLogLoss": baseline_log_loss,
        "baselineBrier": baseline_brier,
        "deltaLogLossProxyMinusBaseline": log_loss - baseline_log_loss,
        "deltaBrierProxyMinusBaseline": brier - baseline_brier,
        "rocAuc": _rank_auc(truth, scores) if positives and negatives else None,
        "averagePrecision": (
            _average_precision(truth, scores) if positives and negatives else None
        ),
    }


@dataclass(frozen=True)
class TagConditionedLiveRuns:
    ordinary: tuple[int, ...]
    immediate_result: tuple[int, ...]
    excluded: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        def describe(values: tuple[int, ...]) -> dict[str, Any]:
            array = np.asarray(values, dtype=np.float64)
            return {
                "runs": len(values),
                "meanSamples": float(np.mean(array)) if len(array) else None,
                "medianSamples": float(np.median(array)) if len(array) else None,
                "minimumSamples": int(np.min(array)) if len(array) else None,
                "maximumSamples": int(np.max(array)) if len(array) else None,
            }

        return {
            "ordinary": describe(self.ordinary),
            "immediateResult": describe(self.immediate_result),
            "excluded": list(self.excluded),
        }


def tag_conditioned_live_runs(
    prepared: Sequence[PreparedRecording],
) -> TagConditionedLiveRuns:
    ordinary: list[int] = []
    immediate: list[int] = []
    excluded: list[dict[str, Any]] = []
    for item in prepared:
        targets = state_targets(item)
        times = item.sequence.times
        for rally_index, rally in enumerate(item.recording.rallies):
            label, outcome = _outcome(rally.tags)
            covered = np.flatnonzero((times >= rally.start) & (times < rally.end))
            closing = np.flatnonzero(times >= rally.end)
            reason: str | None = None
            if not len(covered):
                reason = "no-live-grid-samples"
            elif not len(closing):
                reason = "recording-edge-censored"
            elif not np.all(item.sample_mask[covered]) or not item.sample_mask[closing[0]]:
                reason = "ignored-span-censored"
            live_count = int(
                np.sum(
                    item.sample_mask[covered]
                    & (targets[covered] == int(MultistateState.LIVE))
                )
            ) if len(covered) else 0
            if reason is None and live_count < 1:
                reason = "no-public-live-samples"
            if reason is not None:
                excluded.append(
                    {
                        "recordingId": item.recording.id,
                        "rallyIndex": rally_index,
                        "start": rally.start,
                        "end": rally.end,
                        "outcome": outcome,
                        "reason": reason,
                    }
                )
                continue
            (immediate if label > 0.5 else ordinary).append(live_count)
    if not ordinary or not immediate:
        raise FeatureExperimentError(
            "tag-conditioned duration priors require both rally outcomes"
        )
    return TagConditionedLiveRuns(
        ordinary=tuple(ordinary),
        immediate_result=tuple(immediate),
        excluded=tuple(excluded),
    )


__all__ = [
    "TAG_PROXY_EPOCHS",
    "TAG_PROXY_FEATURE_NAMES",
    "TAG_PROXY_MAX_PROBABILITY",
    "TAG_PROXY_MIN_PROBABILITY",
    "TagConditionedLiveRuns",
    "TagProxyAnchors",
    "extract_tag_proxy_anchors",
    "fit_tag_proxy",
    "predict_tag_proxy",
    "predict_tag_proxy_sequence",
    "tag_conditioned_live_runs",
    "tag_proxy_fingerprint",
    "tag_proxy_metrics",
    "tag_proxy_values",
]
