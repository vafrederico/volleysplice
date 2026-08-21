"""Cadence-free local-peak and soft-count decoding for side-switch scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.side_switch_v3 import V3Event


DECODER_KIND = "volleycut-side-switch-peak-cleanup-decoder-v1"
_PROBABILITY_EPSILON = 1e-9


class SideSwitchPeakCleanupError(ValueError):
    pass


def _probability_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise SideSwitchPeakCleanupError(f"{name} must be a finite vector")
    if np.any((result < 0.0) | (result > 1.0)):
        raise SideSwitchPeakCleanupError(f"{name} must stay in [0, 1]")
    return result


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class PeakCleanupSettings:
    """Score-ranked cleanup without cadence windows or selection re-anchoring."""

    minimum_gap_separation: int = 0
    minimum_time_separation_seconds: float = 0.0
    free_predictions_per_recording: int = 6
    count_penalty_logit: float = 0.0
    production_context_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_gap_separation < 0:
            raise SideSwitchPeakCleanupError(
                "minimum gap separation cannot be negative"
            )
        if (
            not math.isfinite(self.minimum_time_separation_seconds)
            or self.minimum_time_separation_seconds < 0.0
        ):
            raise SideSwitchPeakCleanupError(
                "minimum time separation must be finite and nonnegative"
            )
        if self.free_predictions_per_recording < 1:
            raise SideSwitchPeakCleanupError(
                "soft count prior needs at least one free prediction"
            )
        if (
            not math.isfinite(self.count_penalty_logit)
            or self.count_penalty_logit < 0.0
        ):
            raise SideSwitchPeakCleanupError(
                "count penalty must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.production_context_weight)
            or self.production_context_weight < 0.0
        ):
            raise SideSwitchPeakCleanupError(
                "production-context weight must be finite and nonnegative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": DECODER_KIND,
            "selectionRule": (
                "score-ranked local-peak suppression followed by a soft "
                "post-free-count logit penalty"
            ),
            "usesCadence": False,
            "reanchorOnSelection": False,
            "minimumGapSeparation": self.minimum_gap_separation,
            "minimumTimeSeparationSeconds": self.minimum_time_separation_seconds,
            "freePredictionsPerRecording": self.free_predictions_per_recording,
            "countPenaltyLogit": self.count_penalty_logit,
            "hardMaximumPredictions": None,
            "productionContextWeight": self.production_context_weight,
            "productionContextIsHardGate": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PeakCleanupSettings":
        if payload.get("kind") != DECODER_KIND:
            raise SideSwitchPeakCleanupError("peak-cleanup decoder kind changed")
        if (
            payload.get("usesCadence") is not False
            or payload.get("reanchorOnSelection") is not False
            or payload.get("hardMaximumPredictions") is not None
            or payload.get("productionContextIsHardGate") is not False
        ):
            raise SideSwitchPeakCleanupError(
                "peak cleanup cannot contain cadence, re-anchoring, a hard cap, "
                "or a hard context gate"
            )
        value = cls(
            minimum_gap_separation=int(payload["minimumGapSeparation"]),
            minimum_time_separation_seconds=float(
                payload["minimumTimeSeparationSeconds"]
            ),
            free_predictions_per_recording=int(
                payload["freePredictionsPerRecording"]
            ),
            count_penalty_logit=float(payload["countPenaltyLogit"]),
            production_context_weight=float(payload["productionContextWeight"]),
        )
        if value.to_dict() != dict(payload):
            raise SideSwitchPeakCleanupError(
                "peak-cleanup decoder contains unknown or changed fields"
            )
        return value


def combine_soft_context(
    primary_probabilities: np.ndarray,
    production_context_probabilities: np.ndarray,
    production_context_weight: float,
) -> np.ndarray:
    """Add production compatibility in log-odds space; never apply a hard gate."""

    primary = _probability_vector(primary_probabilities, "primary probabilities")
    context = _probability_vector(
        production_context_probabilities, "production-context probabilities"
    )
    if primary.shape != context.shape:
        raise SideSwitchPeakCleanupError(
            "primary and production-context probabilities are not aligned"
        )
    if (
        not math.isfinite(production_context_weight)
        or production_context_weight < 0.0
    ):
        raise SideSwitchPeakCleanupError(
            "production-context weight must be finite and nonnegative"
        )
    if production_context_weight == 0.0:
        return primary.copy()
    return _sigmoid(
        _logit(primary) + production_context_weight * _logit(context)
    )


def _transition_time(event: V3Event) -> float:
    value = float(event.row.get("transitionTime", math.nan))
    if not math.isfinite(value):
        raise SideSwitchPeakCleanupError(
            f"event has no finite transition time: {event.event_id}"
        )
    return value


def decode_peak_cleanup(
    events: Sequence[V3Event],
    primary_probabilities: np.ndarray,
    production_context_probabilities: np.ndarray,
    threshold: float,
    settings: PeakCleanupSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected gaps and their soft-context-combined probabilities."""

    primary = _probability_vector(primary_probabilities, "primary probabilities")
    context = _probability_vector(
        production_context_probabilities, "production-context probabilities"
    )
    if len(events) != len(primary) or primary.shape != context.shape:
        raise SideSwitchPeakCleanupError("events and probabilities are not aligned")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SideSwitchPeakCleanupError("threshold must stay in [0, 1]")
    combined = combine_soft_context(
        primary, context, settings.production_context_weight
    )
    combined_logits = _logit(combined)
    threshold_logit = float(_logit(np.asarray([threshold]))[0])
    predictions = np.zeros(len(events), dtype=bool)

    for recording_id in sorted({event.recording_id for event in events}):
        recording_indexes = [
            index
            for index, event in enumerate(events)
            if event.recording_id == recording_id
        ]
        if len({events[index].gap_order for index in recording_indexes}) != len(
            recording_indexes
        ):
            raise SideSwitchPeakCleanupError(
                "decoder expects one reviewed row per recording/gap"
            )
        ordered = sorted(
            recording_indexes,
            key=lambda index: (
                -float(combined_logits[index]),
                _transition_time(events[index]),
                events[index].gap_order,
                events[index].event_id,
            ),
        )
        selected_indexes: list[int] = []
        for index in ordered:
            event = events[index]
            event_time = _transition_time(event)
            if any(
                (
                    settings.minimum_gap_separation > 0
                    and abs(event.gap_order - events[other].gap_order)
                    < settings.minimum_gap_separation
                )
                or (
                    settings.minimum_time_separation_seconds > 0.0
                    and abs(event_time - _transition_time(events[other]))
                    < settings.minimum_time_separation_seconds
                )
                for other in selected_indexes
            ):
                continue
            next_count = len(selected_indexes) + 1
            excess_count = max(
                0, next_count - settings.free_predictions_per_recording
            )
            adjusted_margin = (
                float(combined_logits[index])
                - threshold_logit
                - settings.count_penalty_logit * excess_count
            )
            if adjusted_margin >= -1e-12:
                selected_indexes.append(index)
                predictions[index] = True

    return predictions, combined


def selected_time_structure(
    events: Sequence[V3Event], predictions: np.ndarray
) -> dict[str, Any]:
    selected = np.asarray(predictions, dtype=bool)
    if selected.ndim != 1 or len(events) != len(selected):
        raise SideSwitchPeakCleanupError("events and predictions are not aligned")
    spacings: list[float] = []
    by_recording: dict[str, Any] = {}
    for recording_id in sorted({event.recording_id for event in events}):
        times = sorted(
            _transition_time(event)
            for index, event in enumerate(events)
            if event.recording_id == recording_id and selected[index]
        )
        local_spacings = [right - left for left, right in zip(times, times[1:])]
        spacings.extend(local_spacings)
        by_recording[recording_id] = {
            "selectedCount": len(times),
            "minimumSelectedTimeSpacingSeconds": (
                min(local_spacings) if local_spacings else None
            ),
        }
    return {
        "selectedCount": int(np.sum(selected)),
        "minimumSelectedTimeSpacingSeconds": min(spacings) if spacings else None,
        "byRecording": by_recording,
    }


__all__ = [
    "DECODER_KIND",
    "PeakCleanupSettings",
    "SideSwitchPeakCleanupError",
    "combine_soft_context",
    "decode_peak_cleanup",
    "selected_time_structure",
]
