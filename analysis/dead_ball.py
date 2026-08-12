from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import numpy as np


class TimeInterval(Protocol):
    start: float
    end: float


@dataclass(frozen=True)
class DeadBallDetection:
    time: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {"time": self.time, "confidence": self.confidence}


@dataclass(frozen=True)
class DeadBallDecoderConfig:
    threshold: float = 0.8
    time_offset_seconds: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("dead-ball threshold must be between zero and one")
        if not math.isfinite(self.time_offset_seconds):
            raise ValueError("dead-ball time offset must be finite")

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "timeOffsetSeconds": self.time_offset_seconds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DeadBallDecoderConfig":
        result = cls(
            threshold=float(value["threshold"]),
            time_offset_seconds=float(value.get("timeOffsetSeconds", 0.0)),
        )
        result.validate()
        return result


def _validate_times(times: np.ndarray, label: str) -> None:
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError(f"{label} times must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{label} times must be strictly increasing")


def dead_ball_labels_for_times(
    times: np.ndarray,
    rallies: Iterable[TimeInterval],
    radius: float,
) -> np.ndarray:
    """Label a narrow pulse around every annotated dead-ball boundary.

    The nearest sample is always marked, even when the boundary falls between
    samples and the configured radius selects no point on the sampling grid.
    """
    _validate_times(times, "dead-ball label")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("dead-ball target radius must be non-negative")

    result = np.zeros(len(times), dtype=np.float32)
    if len(times) == 0:
        return result
    for rally in rallies:
        end = float(rally.end)
        if not math.isfinite(end):
            raise ValueError("dead-ball rally ends must be finite")
        distance = np.abs(times - end)
        selected = distance <= radius + 1e-9
        if not np.any(selected):
            selected[int(np.argmin(distance))] = True
        result[selected] = 1.0
    return result


def decode_dead_ball_probabilities(
    times: np.ndarray,
    probabilities: np.ndarray,
    config: DeadBallDecoderConfig,
    *,
    duration: float | None = None,
) -> list[DeadBallDetection]:
    """Collapse every threshold island to its strongest, earliest peak."""
    config.validate()
    if probabilities.ndim != 1 or len(times) != len(probabilities):
        raise ValueError(
            "dead-ball times and probabilities must be aligned one-dimensional arrays"
        )
    _validate_times(times, "dead-ball")
    if not np.isfinite(probabilities).all():
        raise ValueError("dead-ball probabilities must be finite")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("dead-ball probabilities must be between zero and one")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("dead-ball decode duration must be positive when supplied")

    detections: list[DeadBallDetection] = []
    active_start: int | None = None
    above = probabilities >= config.threshold
    for index, active in enumerate(above):
        if active and active_start is None:
            active_start = index
        closes = active_start is not None and (not active or index == len(above) - 1)
        if not closes:
            continue
        end = index if not active else index + 1
        peak_index = active_start + int(np.argmax(probabilities[active_start:end]))
        detection_time = float(times[peak_index] + config.time_offset_seconds)
        if duration is not None:
            detection_time = min(duration, max(0.0, detection_time))
        detections.append(
            DeadBallDetection(
                time=detection_time,
                confidence=float(probabilities[peak_index]),
            )
        )
        active_start = None
    return detections


def select_dead_ball_after_serve(
    detections: Sequence[DeadBallDetection],
    serve_time: float,
    min_after: float,
    max_after: float,
    before_time: float | None = None,
) -> DeadBallDetection | None:
    """Select the strongest endpoint in a serve-relative, half-open window.

    The lower and duration-derived upper bounds are inclusive. ``before_time``
    is exclusive so a detection at the next serve is not assigned backward.
    Confidence wins first; equal-confidence candidates use the earliest time.
    """
    if not math.isfinite(serve_time):
        raise ValueError("serve time must be finite")
    if (
        not math.isfinite(min_after)
        or not math.isfinite(max_after)
        or min_after < 0
        or max_after < min_after
    ):
        raise ValueError(
            "dead-ball window must satisfy 0 <= minimum after <= maximum after"
        )
    if before_time is not None and not math.isfinite(before_time):
        raise ValueError("dead-ball before time must be finite when supplied")

    start = serve_time + min_after
    end = serve_time + max_after
    eligible: list[DeadBallDetection] = []
    for detection in detections:
        if (
            not math.isfinite(detection.time)
            or not math.isfinite(detection.confidence)
            or not 0 <= detection.confidence <= 1
        ):
            raise ValueError("dead-ball detections must have finite times and confidences")
        if detection.time < start or detection.time > end:
            continue
        if before_time is not None and detection.time >= before_time:
            continue
        eligible.append(detection)
    if not eligible:
        return None
    return min(eligible, key=lambda item: (-item.confidence, item.time))
