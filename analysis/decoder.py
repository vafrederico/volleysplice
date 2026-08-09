from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DecoderConfig


@dataclass(frozen=True)
class DecodedInterval:
    start: float
    end: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "confidence": self.confidence}


def smooth_probabilities(probabilities: np.ndarray, window_samples: int) -> np.ndarray:
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if len(probabilities) == 0 or window_samples <= 1:
        return probabilities.astype(np.float32, copy=True)
    window_samples = min(window_samples, len(probabilities))
    kernel = np.ones(window_samples, dtype=np.float32) / window_samples
    left_padding = window_samples // 2
    right_padding = window_samples - 1 - left_padding
    padded = np.pad(probabilities, (left_padding, right_padding), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def hysteresis_mask(
    probabilities: np.ndarray,
    enter_threshold: float,
    exit_threshold: float,
) -> np.ndarray:
    live = False
    result = np.zeros(len(probabilities), dtype=bool)
    for index, probability in enumerate(probabilities):
        if not live and probability >= enter_threshold:
            live = True
        elif live and probability < exit_threshold:
            live = False
        result[index] = live
    return result


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, item in enumerate(mask):
        if bool(item) == value and start is None:
            start = index
        if bool(item) != value and start is not None:
            result.append((start, index))
            start = None
    if start is not None:
        result.append((start, len(mask)))
    return result


def clean_mask(
    mask: np.ndarray,
    *,
    min_live_samples: int,
    bridge_gap_samples: int,
) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    if bridge_gap_samples > 0:
        for start, end in _runs(result, False):
            if start > 0 and end < len(result) and end - start <= bridge_gap_samples:
                result[start:end] = True
    if min_live_samples > 1:
        for start, end in _runs(result, True):
            if end - start < min_live_samples:
                result[start:end] = False
    return result


def decode_probabilities(
    times: np.ndarray,
    probabilities: np.ndarray,
    duration: float,
    config: DecoderConfig,
    analysis_fps: float,
) -> tuple[list[DecodedInterval], np.ndarray]:
    config.validate()
    if times.ndim != 1 or probabilities.ndim != 1 or len(times) != len(probabilities):
        raise ValueError("times and probabilities must be aligned one-dimensional arrays")
    if len(times) == 0:
        return [], probabilities.astype(np.float32, copy=True)
    smoothing_samples = max(1, round(config.smoothing_seconds * analysis_fps))
    smoothed = smooth_probabilities(probabilities, smoothing_samples)
    mask = hysteresis_mask(smoothed, config.enter_threshold, config.exit_threshold)
    mask = clean_mask(
        mask,
        min_live_samples=max(1, round(config.min_live_seconds * analysis_fps)),
        bridge_gap_samples=max(0, round(config.bridge_gap_seconds * analysis_fps)),
    )
    sample_width = 1.0 / analysis_fps
    intervals: list[DecodedInterval] = []
    for start_index, end_index in _runs(mask, True):
        start = max(0.0, float(times[start_index] - sample_width / 2))
        end = min(duration, float(times[end_index - 1] + sample_width / 2))
        if end <= start:
            continue
        confidence = float(np.mean(smoothed[start_index:end_index]))
        intervals.append(DecodedInterval(start=start, end=end, confidence=confidence))
    return intervals, smoothed
