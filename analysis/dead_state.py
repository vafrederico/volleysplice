from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np


class TimeInterval(Protocol):
    start: float
    end: float


@dataclass(frozen=True)
class DeadStateDetection:
    time: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {"time": self.time, "confidence": self.confidence}


@dataclass(frozen=True)
class DeadStateDecoderConfig:
    """Decode a live-to-dead transition after a serve anchor.

    The decoder arms only after ``minimum_live_samples`` consecutive samples
    at or below ``live_reset_threshold``. It then emits the first run of
    ``minimum_dead_samples`` consecutive samples at or above
    ``dead_threshold``. This prevents an anchor in already-dead time from
    creating an endpoint solely because dead-state confidence is high.
    """

    dead_threshold: float = 0.8
    live_reset_threshold: float = 0.4
    minimum_live_samples: int = 1
    minimum_dead_samples: int = 2
    min_after_serve_seconds: float = 0.25
    max_after_serve_seconds: float = 4.0
    time_offset_seconds: float = 0.0

    def validate(self) -> None:
        if (
            not math.isfinite(self.dead_threshold)
            or not 0 < self.dead_threshold < 1
        ):
            raise ValueError("dead-state threshold must be between zero and one")
        if (
            not math.isfinite(self.live_reset_threshold)
            or not 0 < self.live_reset_threshold < self.dead_threshold
        ):
            raise ValueError(
                "dead-state live reset threshold must be between zero and the dead threshold"
            )
        for label, value in (
            ("minimum live samples", self.minimum_live_samples),
            ("minimum dead samples", self.minimum_dead_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"dead-state {label} must be a positive integer")
        if (
            not math.isfinite(self.min_after_serve_seconds)
            or not math.isfinite(self.max_after_serve_seconds)
            or self.min_after_serve_seconds < 0
            or self.max_after_serve_seconds < self.min_after_serve_seconds
        ):
            raise ValueError(
                "dead-state window must satisfy 0 <= minimum after <= maximum after"
            )
        if not math.isfinite(self.time_offset_seconds):
            raise ValueError("dead-state time offset must be finite")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "deadThreshold": self.dead_threshold,
            "liveResetThreshold": self.live_reset_threshold,
            "minimumLiveSamples": self.minimum_live_samples,
            "minimumDeadSamples": self.minimum_dead_samples,
            "minAfterServeSeconds": self.min_after_serve_seconds,
            "maxAfterServeSeconds": self.max_after_serve_seconds,
            "timeOffsetSeconds": self.time_offset_seconds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DeadStateDecoderConfig:
        result = cls(
            dead_threshold=float(value["deadThreshold"]),
            live_reset_threshold=float(value["liveResetThreshold"]),
            minimum_live_samples=int(value.get("minimumLiveSamples", 1)),
            minimum_dead_samples=int(value.get("minimumDeadSamples", 2)),
            min_after_serve_seconds=float(value.get("minAfterServeSeconds", 0.25)),
            max_after_serve_seconds=float(value.get("maxAfterServeSeconds", 4.0)),
            time_offset_seconds=float(value.get("timeOffsetSeconds", 0.0)),
        )
        result.validate()
        return result


def _validate_times(times: np.ndarray, label: str) -> None:
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError(f"{label} times must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{label} times must be strictly increasing")


def _validated_intervals(rallies: Iterable[TimeInterval]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for rally in rallies:
        start = float(rally.start)
        end = float(rally.end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("dead-state rally boundaries must be finite")
        if end <= start:
            raise ValueError("dead-state rally ends must be after their starts")
        intervals.append((start, end))
    return sorted(intervals)


def _labels_with_live_precedence(
    times: np.ndarray,
    windows: Iterable[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(len(times), dtype=bool)
    live = np.zeros(len(times), dtype=bool)
    dead = np.zeros(len(times), dtype=bool)
    for lower, end, upper in windows:
        active = (times >= lower) & (times < upper)
        mask |= active
        live |= active & (times < end)
        dead |= active & (times >= end)

    labels = (dead & ~live).astype(np.float32)
    return labels, mask


def dead_state_labels_for_times(
    times: np.ndarray,
    rallies: Iterable[TimeInterval],
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a serve-anchored, early dead-state target.

    Each rally contributes the half-open window ``[start, start + horizon)``.
    Samples before its end are live (zero), while samples at or after its end
    are dead (one). When windows overlap, any live assignment wins over a dead
    assignment so the result is independent of input rally order.
    """

    _validate_times(times, "dead-state label")
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("dead-state horizon must be positive")
    intervals = _validated_intervals(rallies)
    return _labels_with_live_precedence(
        times,
        ((start, end, start + horizon) for start, end in intervals),
    )


def end_transition_labels_for_times(
    times: np.ndarray,
    rallies: Iterable[TimeInterval],
    before_seconds: float = 2.0,
    after_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a local live-to-dead target around each annotated rally end.

    A rally contributes ``[max(start, end - before), min(end + after,
    next_start))``. The half-open upper bound keeps its dead examples out of
    the following rally. As with the serve-anchored target, live assignments
    take precedence when unusual or overlapping annotations share samples.
    """

    _validate_times(times, "end-transition label")
    if not math.isfinite(before_seconds) or before_seconds < 0:
        raise ValueError("end-transition before seconds must be non-negative")
    if not math.isfinite(after_seconds) or after_seconds < 0:
        raise ValueError("end-transition after seconds must be non-negative")
    intervals = _validated_intervals(rallies)
    windows: list[tuple[float, float, float]] = []
    for index, (start, end) in enumerate(intervals):
        lower = max(start, end - before_seconds)
        upper = end + after_seconds
        if index + 1 < len(intervals):
            upper = min(upper, intervals[index + 1][0])
        windows.append((lower, end, upper))
    return _labels_with_live_precedence(times, windows)


def decode_dead_state_after_serve(
    times: np.ndarray,
    probabilities: np.ndarray,
    serve_time: float,
    config: DeadStateDecoderConfig,
    *,
    next_serve_time: float | None = None,
    duration: float | None = None,
) -> DeadStateDetection | None:
    """Return the first stable live-to-dead transition after one serve.

    Live evidence can occur from the serve onward, including before the
    configured minimum detection lag. A dead run begins only within the
    inclusive lag window, and the next serve is an exclusive upper bound.
    """

    config.validate()
    _validate_times(times, "dead-state")
    if probabilities.ndim != 1 or len(times) != len(probabilities):
        raise ValueError(
            "dead-state times and probabilities must be aligned one-dimensional arrays"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("dead-state probabilities must be finite")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("dead-state probabilities must be between zero and one")
    if not math.isfinite(serve_time):
        raise ValueError("dead-state serve time must be finite")
    if next_serve_time is not None and (
        not math.isfinite(next_serve_time) or next_serve_time <= serve_time
    ):
        raise ValueError("dead-state next serve time must be finite and after the serve")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("dead-state decode duration must be positive when supplied")

    detection_lower = serve_time + config.min_after_serve_seconds
    detection_upper = serve_time + config.max_after_serve_seconds
    live_run = 0
    armed = False
    dead_run_start: int | None = None
    dead_run_length = 0

    for index, (time, probability) in enumerate(zip(times, probabilities, strict=True)):
        if time < serve_time:
            continue
        if time > detection_upper:
            break
        if next_serve_time is not None and time >= next_serve_time:
            break

        if not armed:
            if probability <= config.live_reset_threshold:
                live_run += 1
                if live_run >= config.minimum_live_samples:
                    armed = True
            else:
                live_run = 0

        if not armed or time < detection_lower:
            dead_run_start = None
            dead_run_length = 0
            continue

        if probability >= config.dead_threshold:
            if dead_run_start is None:
                dead_run_start = index
            dead_run_length += 1
        else:
            dead_run_start = None
            dead_run_length = 0

        if dead_run_length < config.minimum_dead_samples:
            continue

        assert dead_run_start is not None
        run = probabilities[dead_run_start : index + 1]
        detection_time = float(times[dead_run_start] + config.time_offset_seconds)
        if duration is not None:
            detection_time = min(duration, max(0.0, detection_time))
        return DeadStateDetection(
            time=detection_time,
            confidence=float(np.min(run)),
        )

    return None
