"""High-recall side-switch candidates from production ranges and internal dead peaks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateUnionConfig:
    signal: str
    threshold: float
    minimum_peak_separation_seconds: float
    range_edge_exclusion_seconds: float = 4.0
    proposal_half_width_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.signal not in {"deadState", "maxDeadOrInverseRally"}:
            raise ValueError(f"unsupported internal-peak signal: {self.signal}")
        values = (
            self.threshold,
            self.minimum_peak_separation_seconds,
            self.range_edge_exclusion_seconds,
            self.proposal_half_width_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate-union configuration must be finite")
        if not 0 <= self.threshold <= 1:
            raise ValueError("candidate threshold must stay in [0, 1]")
        if (
            self.minimum_peak_separation_seconds <= 0
            or self.range_edge_exclusion_seconds < 0
            or self.proposal_half_width_seconds <= 0
        ):
            raise ValueError("candidate-union time settings are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "threshold": self.threshold,
            "minimumPeakSeparationSeconds": self.minimum_peak_separation_seconds,
            "rangeEdgeExclusionSeconds": self.range_edge_exclusion_seconds,
            "proposalHalfWidthSeconds": self.proposal_half_width_seconds,
        }

    @property
    def identifier(self) -> str:
        return (
            f"{self.signal}-threshold-{self.threshold:g}-"
            f"separation-{self.minimum_peak_separation_seconds:g}"
        )


def _finite_range(value: Mapping[str, Any]) -> tuple[str, float, float]:
    range_id = str(value.get("id", ""))
    start = float(value.get("start", math.nan))
    end = float(value.get("end", math.nan))
    if not range_id or not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError(f"invalid production range: {value}")
    return range_id, start, end


def boundary_candidates(
    recording_id: str, ranges: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return every adjacent decoded-range boundary as an internal candidate."""

    parsed = [_finite_range(value) for value in ranges]
    result: list[dict[str, Any]] = []
    for left, right in zip(parsed, parsed[1:], strict=False):
        if right[1] < left[2]:
            raise ValueError("production ranges must be ordered and non-overlapping")
        result.append(
            {
                "eventId": f"{recording_id}:boundary:{left[0]}:{right[0]}",
                "recordingId": recording_id,
                "kind": "adjacent-rally-boundary",
                "gapStart": left[2],
                "gapEnd": right[1],
                "transitionTime": 0.5 * (left[2] + right[1]),
                "score": None,
                "sourceRangeId": None,
            }
        )
    return result


def internal_peak_candidates(
    recording_id: str,
    ranges: Sequence[Mapping[str, Any]],
    timestamps: np.ndarray,
    rally_probabilities: np.ndarray,
    dead_state_probabilities: np.ndarray,
    config: CandidateUnionConfig,
) -> list[dict[str, Any]]:
    """Select score-ranked, separated dead-state peaks inside decoded rally ranges."""

    times = np.asarray(timestamps, dtype=np.float64)
    rally = np.asarray(rally_probabilities, dtype=np.float64)
    dead = np.asarray(dead_state_probabilities, dtype=np.float64)
    if (
        times.ndim != 1
        or times.shape != rally.shape
        or times.shape != dead.shape
        or len(times) == 0
        or not np.isfinite(times).all()
        or not np.isfinite(rally).all()
        or not np.isfinite(dead).all()
        or np.any(np.diff(times) <= 0)
        or np.any((rally < 0) | (rally > 1))
        or np.any((dead < 0) | (dead > 1))
    ):
        raise ValueError("internal-peak arrays are invalid")
    score = (
        dead
        if config.signal == "deadState"
        else np.maximum(dead, 1.0 - rally)
    )
    result: list[dict[str, Any]] = []
    for raw_range in ranges:
        range_id, start, end = _finite_range(raw_range)
        eligible = np.flatnonzero(
            (times >= start + config.range_edge_exclusion_seconds)
            & (times <= end - config.range_edge_exclusion_seconds)
            & (score >= config.threshold)
        )
        selected: list[int] = []
        for index in sorted(
            eligible,
            key=lambda value: (-float(score[value]), float(times[value]), int(value)),
        ):
            if all(
                abs(float(times[index] - times[other]))
                >= config.minimum_peak_separation_seconds
                for other in selected
            ):
                selected.append(int(index))
        for index in sorted(selected, key=lambda value: float(times[value])):
            timestamp = float(times[index])
            result.append(
                {
                    "eventId": (
                        f"{recording_id}:internal-dead-peak:{range_id}:"
                        f"{round(timestamp * 1000)}"
                    ),
                    "recordingId": recording_id,
                    "kind": "internal-dead-state-peak",
                    "gapStart": max(start, timestamp - config.proposal_half_width_seconds),
                    "gapEnd": min(end, timestamp + config.proposal_half_width_seconds),
                    "transitionTime": timestamp,
                    "score": float(score[index]),
                    "sourceRangeId": range_id,
                }
            )
    return result


def candidate_union(
    recording_id: str,
    ranges: Sequence[Mapping[str, Any]],
    timestamps: np.ndarray,
    rally_probabilities: np.ndarray,
    dead_state_probabilities: np.ndarray,
    config: CandidateUnionConfig,
) -> list[dict[str, Any]]:
    result = [
        *boundary_candidates(recording_id, ranges),
        *internal_peak_candidates(
            recording_id,
            ranges,
            timestamps,
            rally_probabilities,
            dead_state_probabilities,
            config,
        ),
    ]
    event_ids = [str(value["eventId"]) for value in result]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("candidate union produced duplicate event IDs")
    return sorted(
        result,
        key=lambda value: (
            float(value["transitionTime"]),
            str(value["kind"]),
            str(value["eventId"]),
        ),
    )
