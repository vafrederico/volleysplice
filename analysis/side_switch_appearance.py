"""Diagnostic player-appearance features for side-switch experiments.

This module intentionally contains no learned team model.  It extracts cheap,
interpretable appearance summaries from high-resolution frames and compares
stable samples before and after a candidate dead-time transition.  The HOG
person proposals are diagnostic and must not be treated as ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


COLOR_SPACE = "HSV hue-saturation histogram plus value histogram"
HUE_BINS = 12
SATURATION_BINS = 4
VALUE_BINS = 8


@dataclass(frozen=True)
class PersonDetection:
    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def area(self) -> float:
        return float(self.width * self.height)


@dataclass(frozen=True)
class FrameAppearance:
    timestamp: float
    full_palette: np.ndarray
    player_equal_palette: np.ndarray | None
    player_area_palette: np.ndarray | None
    detection_count: int
    total_box_area_fraction: float
    median_box_height_fraction: float | None
    mean_detection_score: float | None


@dataclass(frozen=True)
class AppearanceAggregate:
    full_palette: np.ndarray
    player_equal_palette: np.ndarray | None
    player_area_palette: np.ndarray | None
    mean_detection_count: float
    mean_total_box_area_fraction: float
    mean_median_box_height_fraction: float | None
    mean_detection_score: float | None
    usable_frame_count: int


def _finite_vector(values: np.ndarray, where: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{where} must be a non-empty finite vector")
    total = float(np.sum(result))
    if total <= 0:
        raise ValueError(f"{where} must have positive mass")
    return result / total


def palette_vector(image: np.ndarray) -> np.ndarray:
    """Return a compact color-presence vector for a BGR crop or frame."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("palette image must be a BGR HxWx3 array")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("palette image is too small")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    chromatic = (saturation >= 18) & (value >= 16)
    if int(np.sum(chromatic)) >= 8:
        hue_saturation = np.histogram2d(
            hue[chromatic].astype(np.float64),
            saturation[chromatic].astype(np.float64),
            bins=(HUE_BINS, SATURATION_BINS),
            range=((0.0, 180.0), (0.0, 256.0)),
        )[0]
    else:
        hue_saturation = np.zeros((HUE_BINS, SATURATION_BINS), dtype=np.float64)
    brightness = np.histogram(
        value.astype(np.float64),
        bins=VALUE_BINS,
        range=(0.0, 256.0),
    )[0]
    # A small floor makes sparse crops comparable without allowing empty bins to
    # dominate the Hellinger distance.
    vector = np.concatenate((hue_saturation.reshape(-1), brightness)) + 1e-6
    return _finite_vector(vector, "palette vector")


def hellinger_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Hellinger distance between two normalized histograms."""

    first = _finite_vector(left, "left histogram")
    second = _finite_vector(right, "right histogram")
    if first.shape != second.shape:
        raise ValueError("histograms must have the same shape")
    affinity = float(np.sum(np.sqrt(first * second)))
    return math.sqrt(max(0.0, min(1.0, 1.0 - affinity)))


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    first = _finite_vector(left, "left vector")
    second = _finite_vector(right, "right vector")
    if first.shape != second.shape:
        raise ValueError("vectors must have the same shape")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0:
        return 1.0
    return float(1.0 - np.dot(first, second) / denominator)


def _intersection_over_union(left: PersonDetection, right: PersonDetection) -> float:
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.width, right.x + right.width)
    y1 = min(left.y + left.height, right.y + right.height)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = left.area + right.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def non_maximum_suppression(
    detections: Iterable[PersonDetection],
    *,
    overlap_threshold: float = 0.45,
    maximum: int = 12,
) -> tuple[PersonDetection, ...]:
    ordered = sorted(
        detections,
        key=lambda item: (-item.score, -item.area, item.y, item.x),
    )
    retained: list[PersonDetection] = []
    for candidate in ordered:
        if all(
            _intersection_over_union(candidate, existing) < overlap_threshold
            for existing in retained
        ):
            retained.append(candidate)
            if len(retained) >= maximum:
                break
    return tuple(retained)


def detect_people(
    frame: np.ndarray,
    hog: cv2.HOGDescriptor,
    *,
    minimum_score: float = 0.0,
) -> tuple[PersonDetection, ...]:
    """Run the fixed OpenCV HOG proposal detector on one frame."""

    height, width = frame.shape[:2]
    boxes, weights = hog.detectMultiScale(
        frame,
        hitThreshold=minimum_score,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    raw: list[PersonDetection] = []
    for box, weight in zip(boxes, weights, strict=False):
        x, y, box_width, box_height = (int(value) for value in box)
        score = float(np.asarray(weight).reshape(-1)[0])
        if box_width <= 0 or box_height <= 0 or not math.isfinite(score):
            continue
        if box_height < max(32, round(height * 0.055)):
            continue
        if box_height > round(height * 0.78):
            continue
        aspect = box_width / box_height
        if aspect < 0.22 or aspect > 0.95:
            continue
        if x + box_width <= round(width * 0.02) or x >= round(width * 0.98):
            continue
        if y + box_height <= round(height * 0.12) or y >= height:
            continue
        raw.append(PersonDetection(x, y, box_width, box_height, score))
    return non_maximum_suppression(raw)


def _torso_crop(frame: np.ndarray, detection: PersonDetection) -> np.ndarray | None:
    height, width = frame.shape[:2]
    left = max(0, round(detection.x + detection.width * 0.16))
    right = min(width, round(detection.x + detection.width * 0.84))
    top = max(0, round(detection.y + detection.height * 0.10))
    bottom = min(height, round(detection.y + detection.height * 0.66))
    if right - left < 6 or bottom - top < 8:
        return None
    return frame[top:bottom, left:right]


def _weighted_palette(
    vectors: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> np.ndarray | None:
    if not vectors:
        return None
    matrix = np.stack(vectors, axis=0).astype(np.float64, copy=False)
    if weights is None:
        result = np.mean(matrix, axis=0)
    else:
        numeric = np.asarray(weights, dtype=np.float64)
        if numeric.shape != (len(vectors),) or not np.isfinite(numeric).all():
            raise ValueError("palette weights are invalid")
        numeric = np.maximum(numeric, 1e-9)
        result = np.average(matrix, axis=0, weights=numeric)
    return _finite_vector(result, "weighted palette")


def summarize_frame(
    frame: np.ndarray,
    timestamp: float,
    hog: cv2.HOGDescriptor,
) -> FrameAppearance:
    """Extract full-frame and player-proposal appearance summaries."""

    height, width = frame.shape[:2]
    broad = frame[
        round(height * 0.08) : round(height * 0.96),
        round(width * 0.04) : round(width * 0.96),
    ]
    full = palette_vector(broad)
    detections = detect_people(frame, hog)
    vectors: list[np.ndarray] = []
    areas: list[float] = []
    for detection in detections:
        crop = _torso_crop(frame, detection)
        if crop is None:
            continue
        vectors.append(palette_vector(crop))
        areas.append(detection.area)
    equal = _weighted_palette(vectors)
    area = _weighted_palette(vectors, areas)
    total_area_fraction = sum(item.area for item in detections) / max(1, height * width)
    heights = [item.height / height for item in detections]
    scores = [item.score for item in detections]
    return FrameAppearance(
        timestamp=float(timestamp),
        full_palette=full,
        player_equal_palette=equal,
        player_area_palette=area,
        detection_count=len(detections),
        total_box_area_fraction=float(total_area_fraction),
        median_box_height_fraction=(float(np.median(heights)) if heights else None),
        mean_detection_score=(float(np.mean(scores)) if scores else None),
    )


def _mean_vectors(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    return _finite_vector(np.mean(np.stack(vectors, axis=0), axis=0), "mean palette")


def aggregate_frames(frames: Sequence[FrameAppearance]) -> AppearanceAggregate:
    if not frames:
        raise ValueError("cannot aggregate an empty frame sequence")
    equal = _mean_vectors(
        [frame.player_equal_palette for frame in frames if frame.player_equal_palette is not None]
    )
    area = _mean_vectors(
        [frame.player_area_palette for frame in frames if frame.player_area_palette is not None]
    )
    heights = [
        frame.median_box_height_fraction
        for frame in frames
        if frame.median_box_height_fraction is not None
    ]
    scores = [frame.mean_detection_score for frame in frames if frame.mean_detection_score is not None]
    return AppearanceAggregate(
        full_palette=_mean_vectors([frame.full_palette for frame in frames]),  # type: ignore[arg-type]
        player_equal_palette=equal,
        player_area_palette=area,
        mean_detection_count=float(np.mean([frame.detection_count for frame in frames])),
        mean_total_box_area_fraction=float(
            np.mean([frame.total_box_area_fraction for frame in frames])
        ),
        mean_median_box_height_fraction=(float(np.mean(heights)) if heights else None),
        mean_detection_score=(float(np.mean(scores)) if scores else None),
        usable_frame_count=sum(
            frame.player_area_palette is not None for frame in frames
        ),
    )


def relative_change(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or not math.isfinite(left) or not math.isfinite(right):
        return None
    denominator = max(abs(left) + abs(right), 1e-9)
    return float(abs(right - left) / denominator)


def appearance_features(
    before: AppearanceAggregate,
    after: AppearanceAggregate,
) -> dict[str, float | None]:
    """Compare two stable windows using the predeclared diagnostic feature bank."""

    full_distance = hellinger_distance(before.full_palette, after.full_palette)
    equal_distance = (
        hellinger_distance(before.player_equal_palette, after.player_equal_palette)
        if before.player_equal_palette is not None and after.player_equal_palette is not None
        else None
    )
    area_distance = (
        hellinger_distance(before.player_area_palette, after.player_area_palette)
        if before.player_area_palette is not None and after.player_area_palette is not None
        else None
    )
    count_change = relative_change(before.mean_detection_count, after.mean_detection_count)
    area_change = relative_change(
        before.mean_total_box_area_fraction,
        after.mean_total_box_area_fraction,
    )
    height_change = relative_change(
        before.mean_median_box_height_fraction,
        after.mean_median_box_height_fraction,
    )
    if area_distance is None:
        bundle = None
    else:
        bundle = area_distance
        for value, weight in ((area_change, 0.35), (count_change, 0.20), (height_change, 0.20)):
            if value is not None:
                bundle += weight * min(1.0, value)
        bundle += 0.15 * full_distance
    return {
        "fullFrameControl": full_distance,
        "playerPaletteEqual": equal_distance,
        "playerPaletteArea": area_distance,
        "playerPaletteAreaPlusGeometry": bundle,
        "detectionCountChange": count_change,
        "boxAreaChange": area_change,
        "medianBoxHeightChange": height_change,
    }


def create_hog() -> cv2.HOGDescriptor:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def read_frame(capture: cv2.VideoCapture, timestamp: float) -> np.ndarray:
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("frame timestamp must be finite and non-negative")
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame at {timestamp:.3f}s")
    return frame

