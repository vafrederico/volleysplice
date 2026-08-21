"""Selective source-resolution patch features around residual-motion tracks."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .serving_side_flight import (
    OFFSETS_SECONDS,
    ResidualMotion,
    _affine_camera_flow,
)
from .serving_side_trajectory import (
    PAIR_COUNT,
    ComponentTrack,
    link_component_tracks,
    motion_components,
)


FEATURE_VERSION = "serving-side-selective-track-patch-v1"
PATCH_SIZE = 96
SOURCE_PATCH_FRACTION = 0.20
SELECTORS = ("persistent", "compact")
PATCH_STATISTICS = (
    "residualMean",
    "residualActiveFraction",
    "residualCenterFraction",
    "residualPeakFraction",
    "residualEntropy",
    "residualSmallComponentFraction",
    "alignmentResponse",
    "alignmentMagnitude",
    "contrast",
    "sharpness",
    "centerContrast",
)
SLOPE_STATISTICS = (
    "residualMean",
    "residualCenterFraction",
    "centerContrast",
)
_EPSILON = 1e-12


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for selector in SELECTORS:
        names.append(f"{selector}:coverage")
        for statistic in PATCH_STATISTICS:
            names.extend(
                (
                    f"{selector}:{statistic}:mean",
                    f"{selector}:{statistic}:max",
                )
            )
            if statistic in SLOPE_STATISTICS:
                names.append(f"{selector}:{statistic}:slope")
    return tuple(names)


FEATURE_NAMES = feature_names()


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required; run `npm run analysis:setup` first"
        ) from error
    return cv2


def _gray(frame: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    value = np.asarray(frame)
    if value.ndim == 2:
        result = value.astype(np.uint8, copy=False)
    elif value.ndim == 3 and value.shape[2] == 3:
        result = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("high-resolution frames must be grayscale or BGR")
    if min(result.shape) < 32:
        raise ValueError("high-resolution frames are too small")
    return result


def _full_flow_and_camera(
    before: np.ndarray, after: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    flow = cv2.calcOpticalFlowFarneback(
        before,
        after,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.1,
        0,
    )
    return flow, _affine_camera_flow(flow)


def _track_compact_score(track: ComponentTrack) -> float:
    components = track.components
    if not components:
        return 0.0
    persistence = len(components) / PAIR_COUNT
    energy = float(np.mean([item.energy_fraction for item in components]))
    area = float(np.mean([item.area_fraction for item in components]))
    vectors = [
        math.hypot(
            after.centroid_x - before.centroid_x,
            after.centroid_y - before.centroid_y,
        )
        for before, after in zip(components[:-1], components[1:], strict=True)
    ]
    speed = float(np.mean(vectors)) if vectors else math.hypot(
        components[0].flow_x, components[0].flow_y
    )
    compactness = math.exp(-area / 0.0125)
    motion_bonus = 0.5 + min(speed / 0.05, 1.0)
    return persistence * math.sqrt(max(energy, 0.0)) * compactness * motion_bonus


def select_tracks(
    motions: Sequence[ResidualMotion],
) -> Mapping[str, ComponentTrack | None]:
    if len(motions) != PAIR_COUNT:
        raise ValueError("selective high-resolution input has the wrong pair count")
    components = [
        motion_components(motion, index)
        for index, motion in enumerate(motions)
    ]
    tracks = link_component_tracks(components)
    return {
        "persistent": tracks[0] if tracks else None,
        "compact": max(
            tracks,
            key=lambda track: (
                _track_compact_score(track),
                len(track.components),
                -track.components[0].pair_index,
            ),
            default=None,
        ),
    }


def _source_patch(
    gray: np.ndarray,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    cv2 = _cv2()
    height, width = gray.shape
    side = min(
        min(height, width),
        max(32, int(round(min(height, width) * SOURCE_PATCH_FRACTION))),
    )
    patch = cv2.getRectSubPix(
        gray,
        (side, side),
        (
            float(np.clip(center_x, 0.0, 1.0) * max(width - 1, 1)),
            float(np.clip(center_y, 0.0, 1.0) * max(height - 1, 1)),
        ),
    )
    return cv2.resize(
        patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA
    )


def _patch_statistics(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    cv2 = _cv2()
    before_float = before.astype(np.float32) / 255.0
    after_float = after.astype(np.float32) / 255.0
    window = cv2.createHanningWindow((PATCH_SIZE, PATCH_SIZE), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(before_float, after_float, window)
    transform = np.asarray(
        [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32
    )
    aligned_before = cv2.warpAffine(
        before_float,
        transform,
        (PATCH_SIZE, PATCH_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    difference = np.abs(after_float - aligned_before)
    threshold = max(2.0 / 255.0, float(np.percentile(difference, 90)))
    residual = np.maximum(difference - threshold, 0.0)
    total = float(np.sum(residual))
    active = residual > 0
    inner = residual[
        PATCH_SIZE // 4 : 3 * PATCH_SIZE // 4,
        PATCH_SIZE // 4 : 3 * PATCH_SIZE // 4,
    ]
    if total > 0:
        probabilities = residual[active] / total
        entropy = float(
            -np.sum(probabilities * np.log(probabilities))
            / max(math.log(residual.size), 1.0)
        )
        flat = np.sort(residual.reshape(-1))
        peak_count = max(1, int(round(residual.size * 0.01)))
        peak_fraction = float(np.sum(flat[-peak_count:]) / total)
    else:
        entropy = peak_fraction = 0.0
    small_component_fraction = 0.0
    if np.any(active) and total > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            active.astype(np.uint8), connectivity=8
        )
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            small_labels = np.flatnonzero(areas <= residual.size * 0.02) + 1
            small_component_fraction = float(
                np.sum(residual[np.isin(labels, small_labels)]) / total
            )
    after_blur = cv2.GaussianBlur(after_float, (0, 0), 2.0)
    central = after_blur[
        3 * PATCH_SIZE // 8 : 5 * PATCH_SIZE // 8,
        3 * PATCH_SIZE // 8 : 5 * PATCH_SIZE // 8,
    ]
    ring_mask = np.ones_like(after_blur, dtype=bool)
    ring_mask[
        PATCH_SIZE // 4 : 3 * PATCH_SIZE // 4,
        PATCH_SIZE // 4 : 3 * PATCH_SIZE // 4,
    ] = False
    center_contrast = abs(float(np.mean(central)) - float(np.mean(after_blur[ring_mask])))
    return {
        "residualMean": float(np.mean(residual)),
        "residualActiveFraction": float(np.mean(active)),
        "residualCenterFraction": float(np.sum(inner) / total) if total > 0 else 0.0,
        "residualPeakFraction": peak_fraction,
        "residualEntropy": entropy,
        "residualSmallComponentFraction": small_component_fraction,
        "alignmentResponse": float(np.clip(response, 0.0, 1.0)),
        "alignmentMagnitude": math.hypot(float(shift[0]), float(shift[1])) / PATCH_SIZE,
        "contrast": float(np.std(after_float)),
        "sharpness": float(np.mean(np.abs(cv2.Laplacian(after_float, cv2.CV_32F)))),
        "centerContrast": center_contrast,
    }


def _slope(pair_indices: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2 or np.all(pair_indices == pair_indices[0]):
        return 0.0
    centered = pair_indices - np.mean(pair_indices)
    denominator = float(np.sum(centered * centered))
    return float(np.sum(centered * (values - np.mean(values))) / denominator)


def extract_selective_highres_features(
    source_frames: Sequence[np.ndarray],
    low_resolution_frames: Sequence[np.ndarray],
    motions: Sequence[ResidualMotion],
) -> dict[str, float]:
    if len(source_frames) != len(OFFSETS_SECONDS) or len(low_resolution_frames) != len(
        OFFSETS_SECONDS
    ):
        raise ValueError("selective high-resolution extraction needs nine aligned frames")
    if len(motions) != PAIR_COUNT:
        raise ValueError("selective high-resolution extraction needs eight motions")
    source_gray = [_gray(frame) for frame in source_frames]
    low_gray = [_gray(frame) for frame in low_resolution_frames]
    if any(frame.shape != source_gray[0].shape for frame in source_gray) or any(
        frame.shape != low_gray[0].shape for frame in low_gray
    ):
        raise ValueError("aligned frames must share their respective source shapes")
    camera_fields = [
        _full_flow_and_camera(before, after)[1]
        for before, after in zip(low_gray[:-1], low_gray[1:], strict=True)
    ]
    low_height, low_width = low_gray[0].shape
    tracks = select_tracks(motions)
    output: dict[str, float] = {}
    for selector in SELECTORS:
        track = tracks[selector]
        observations: list[tuple[int, dict[str, float]]] = []
        if track is not None:
            for component in track.components:
                x = int(round(component.centroid_x * max(low_width - 1, 1)))
                y = int(round(component.centroid_y * max(low_height - 1, 1)))
                x = int(np.clip(x, 0, low_width - 1))
                y = int(np.clip(y, 0, low_height - 1))
                camera = camera_fields[component.pair_index][y, x]
                after_x = component.centroid_x + float(camera[0]) / low_width + component.flow_x
                after_y = component.centroid_y + float(camera[1]) / low_height + component.flow_y
                before_patch = _source_patch(
                    source_gray[component.pair_index],
                    component.centroid_x,
                    component.centroid_y,
                )
                after_patch = _source_patch(
                    source_gray[component.pair_index + 1], after_x, after_y
                )
                observations.append(
                    (
                        component.pair_index,
                        _patch_statistics(before_patch, after_patch),
                    )
                )
        output[f"{selector}:coverage"] = len(observations) / PAIR_COUNT
        pair_indices = np.asarray(
            [item[0] for item in observations], dtype=np.float64
        )
        for statistic in PATCH_STATISTICS:
            values = np.asarray(
                [item[1][statistic] for item in observations], dtype=np.float64
            )
            output[f"{selector}:{statistic}:mean"] = (
                float(np.mean(values)) if len(values) else 0.0
            )
            output[f"{selector}:{statistic}:max"] = (
                float(np.max(values)) if len(values) else 0.0
            )
            if statistic in SLOPE_STATISTICS:
                output[f"{selector}:{statistic}:slope"] = _slope(
                    pair_indices, values
                )
    if tuple(output) != FEATURE_NAMES or not all(
        math.isfinite(value) for value in output.values()
    ):
        raise AssertionError("selective high-resolution feature contract is invalid")
    return output
