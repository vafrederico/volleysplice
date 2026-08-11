from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from .config import DecoderConfig
from .decoder import DecodedInterval


class TimeInterval(Protocol):
    start: float
    end: float


@dataclass(frozen=True)
class ServeDetection:
    time: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {"time": self.time, "confidence": self.confidence}


@dataclass(frozen=True)
class ServeDecoderConfig:
    threshold: float = 0.8
    min_separation_seconds: float = 8.0
    time_offset_seconds: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("serve threshold must be between zero and one")
        if (
            not math.isfinite(self.min_separation_seconds)
            or self.min_separation_seconds < 0
        ):
            raise ValueError("serve minimum separation must be non-negative")
        if not math.isfinite(self.time_offset_seconds):
            raise ValueError("serve time offset must be finite")

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "minSeparationSeconds": self.min_separation_seconds,
            "timeOffsetSeconds": self.time_offset_seconds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ServeDecoderConfig":
        result = cls(
            threshold=float(value["threshold"]),
            min_separation_seconds=float(value["minSeparationSeconds"]),
            time_offset_seconds=float(value.get("timeOffsetSeconds", 0.0)),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ServeCompositionConfig:
    association_seconds: float
    fallback_seconds: float
    max_rescue_seconds: float
    permissive_decoder: DecoderConfig

    def validate(self) -> None:
        for label, value in (
            ("association", self.association_seconds),
            ("fallback", self.fallback_seconds),
            ("maximum rescue", self.max_rescue_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"serve {label} seconds must be non-negative")
        if self.max_rescue_seconds <= 0:
            raise ValueError("serve maximum rescue seconds must be positive")
        self.permissive_decoder.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "serve-anchor-permissive-live-fallback-v1",
            "associationSeconds": self.association_seconds,
            "fallbackSeconds": self.fallback_seconds,
            "maxRescueSeconds": self.max_rescue_seconds,
            "permissiveDecoder": self.permissive_decoder.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ServeCompositionConfig":
        if value.get("method") != "serve-anchor-permissive-live-fallback-v1":
            raise ValueError(f"unsupported serve composition method: {value.get('method')!r}")
        result = cls(
            association_seconds=float(value["associationSeconds"]),
            fallback_seconds=float(value["fallbackSeconds"]),
            max_rescue_seconds=float(value["maxRescueSeconds"]),
            permissive_decoder=DecoderConfig.from_dict(value["permissiveDecoder"]),
        )
        result.validate()
        return result


def serve_labels_for_times(
    times: np.ndarray,
    rallies: Iterable[TimeInterval],
    radius_seconds: float,
) -> np.ndarray:
    """Label a narrow pulse around each serve contact.

    At least the nearest sample is marked for every contact, including when a small
    radius and the sampling grid would otherwise leave the contact unlabeled.
    """
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("serve label times must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("serve label times must be strictly increasing")
    if not math.isfinite(radius_seconds) or radius_seconds < 0:
        raise ValueError("serve target radius must be non-negative")
    result = np.zeros(len(times), dtype=np.float32)
    if len(times) == 0:
        return result
    for rally in rallies:
        distance = np.abs(times - float(rally.start))
        selected = distance <= radius_seconds + 1e-9
        if not np.any(selected):
            selected[int(np.argmin(distance))] = True
        result[selected] = 1.0
    return result


def decode_serve_probabilities(
    times: np.ndarray,
    probabilities: np.ndarray,
    config: ServeDecoderConfig,
    *,
    duration: float | None = None,
) -> list[ServeDetection]:
    """Collapse thresholded probability islands into score-first NMS contact points."""
    config.validate()
    if (
        times.ndim != 1
        or probabilities.ndim != 1
        or len(times) != len(probabilities)
    ):
        raise ValueError("serve times and probabilities must be aligned one-dimensional arrays")
    if not np.isfinite(times).all() or not np.isfinite(probabilities).all():
        raise ValueError("serve times and probabilities must be finite")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("serve probabilities must be between zero and one")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("serve times must be strictly increasing")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("serve decode duration must be positive when supplied")

    candidates: list[ServeDetection] = []
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
        contact_time = float(times[peak_index] + config.time_offset_seconds)
        if duration is not None:
            contact_time = min(duration, max(0.0, contact_time))
        candidates.append(
            ServeDetection(contact_time, float(probabilities[peak_index]))
        )
        active_start = None

    retained: list[ServeDetection] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.time)):
        if all(
            abs(candidate.time - previous.time) >= config.min_separation_seconds
            for previous in retained
        ):
            retained.append(candidate)
    return sorted(retained, key=lambda item: item.time)


def match_serve_contacts(
    truth_times: Sequence[float],
    detections: Sequence[ServeDetection],
    tolerance_seconds: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality chronological matching, then minimum absolute error."""
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("serve match tolerance must be non-negative")
    truth = np.asarray(truth_times, dtype=np.float64)
    predicted = np.asarray([item.time for item in detections], dtype=np.float64)
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("serve match times must be finite")
    if len(truth) > 1 and np.any(np.diff(truth) < 0):
        raise ValueError("serve truth times must be ordered")
    if len(predicted) > 1 and np.any(np.diff(predicted) < 0):
        raise ValueError("serve detections must be ordered")

    rows, columns = len(truth), len(predicted)
    counts = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    errors = np.zeros((rows + 1, columns + 1), dtype=np.float64)
    actions = np.zeros((rows + 1, columns + 1), dtype=np.int8)

    def better(left: tuple[int, float], right: tuple[int, float]) -> bool:
        return left[0] > right[0] or (left[0] == right[0] and left[1] < right[1] - 1e-12)

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            best = (int(counts[row - 1, column]), float(errors[row - 1, column]))
            action = 1
            skip_prediction = (
                int(counts[row, column - 1]),
                float(errors[row, column - 1]),
            )
            if better(skip_prediction, best):
                best = skip_prediction
                action = 2
            absolute_error = abs(predicted[column - 1] - truth[row - 1])
            if absolute_error <= tolerance_seconds + 1e-12:
                matched = (
                    int(counts[row - 1, column - 1]) + 1,
                    float(errors[row - 1, column - 1]) + float(absolute_error),
                )
                if better(matched, best):
                    best = matched
                    action = 3
            counts[row, column], errors[row, column] = best
            actions[row, column] = action

    matches: list[tuple[int, int, float]] = []
    row, column = rows, columns
    while row > 0 and column > 0:
        action = actions[row, column]
        if action == 3:
            error = float(predicted[column - 1] - truth[row - 1])
            matches.append((row - 1, column - 1, error))
            row -= 1
            column -= 1
        elif action == 1:
            row -= 1
        else:
            column -= 1
    matches.reverse()
    return matches


def evaluate_serve_contacts(
    truth_times: Sequence[float],
    detections: Sequence[ServeDetection],
    tolerance_seconds: float,
) -> dict[str, Any]:
    matches = match_serve_contacts(truth_times, detections, tolerance_seconds)
    matched = len(matches)
    precision = (
        matched / len(detections)
        if len(detections) > 0
        else (1.0 if len(truth_times) == 0 else 0.0)
    )
    recall = (
        matched / len(truth_times)
        if len(truth_times) > 0
        else (1.0 if len(detections) == 0 else 0.0)
    )
    absolute_errors = [abs(error) for _, _, error in matches]
    return {
        "trueServes": len(truth_times),
        "predictedServes": len(detections),
        "matchedServes": matched,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "timingMaeSeconds": float(np.mean(absolute_errors)) if absolute_errors else None,
        "timingP90Seconds": (
            float(np.percentile(absolute_errors, 90)) if absolute_errors else None
        ),
        "errorsSeconds": [error for _, _, error in matches],
    }


def compose_serve_anchored_intervals(
    primary: Sequence[TimeInterval],
    permissive: Sequence[TimeInterval],
    serves: Sequence[ServeDetection],
    duration: float,
    config: ServeCompositionConfig,
    *,
    sample_seconds: float,
) -> list[DecodedInterval]:
    """Anchor existing starts and rescue serve-gated short permissive intervals."""
    config.validate()
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("composition duration must be positive")
    if not math.isfinite(sample_seconds) or sample_seconds <= 0:
        raise ValueError("composition sample width must be positive")

    rows = [
        [float(item.start), float(item.end), float(getattr(item, "confidence", 0.0))]
        for item in primary
    ]
    unused: list[ServeDetection] = []
    for serve in serves:
        associated = [
            index
            for index, (start, end, _) in enumerate(rows)
            if start - config.association_seconds <= serve.time < end
        ]
        if not associated:
            unused.append(serve)
            continue
        index = min(associated, key=lambda candidate: abs(rows[candidate][0] - serve.time))
        if serve.time < rows[index][0]:
            rows[index][0] = serve.time
            rows[index][2] = max(rows[index][2], serve.confidence)

    for serve in unused:
        eligible = [
            item
            for item in permissive
            if item.end >= serve.time - config.association_seconds
            and item.start <= serve.time + config.association_seconds
            and item.end - item.start <= config.max_rescue_seconds
        ]
        if eligible:
            selected = min(
                eligible,
                key=lambda item: min(
                    abs(float(item.start) - serve.time),
                    abs(float(item.end) - serve.time),
                ),
            )
            rows.append(
                [
                    serve.time,
                    max(serve.time + sample_seconds, float(selected.end)),
                    max(serve.confidence, float(getattr(selected, "confidence", 0.0))),
                ]
            )
        elif config.fallback_seconds > 0:
            rows.append(
                [
                    serve.time,
                    serve.time + config.fallback_seconds,
                    serve.confidence,
                ]
            )

    ordered = sorted(
        (
            max(0.0, start),
            min(duration, end),
            confidence,
        )
        for start, end, confidence in rows
        if end > start and start < duration and end > 0
    )
    merged: list[list[float]] = []
    for start, end, confidence in ordered:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], confidence)
        else:
            merged.append([start, end, confidence])
    return [DecodedInterval(start, end, confidence) for start, end, confidence in merged]
