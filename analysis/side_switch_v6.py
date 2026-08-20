"""Detected-player identity and adaptive team prototypes for side-switch v6."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.side_switch_player_detector import (
    PlayerDetection,
    QuantizedPersonDetector,
)
from analysis.side_switch_v3 import V3Event


MODEL_KIND = "volleycut-side-switch-specialist-v6"
MODEL_SCHEMA_VERSION = 6
FEATURE_ARTIFACT_KIND = "volleycut-side-switch-features-v6"
FEATURE_ARTIFACT_SCHEMA_VERSION = 6
DETECTOR_FRAMES_PER_RALLY = 3
MAXIMUM_PLAYERS_PER_SIDE = 2
ADAPTIVE_PROTOTYPE_RATE = 0.12

V4_CARRY_FEATURES = {
    "v4BroadSameAssignmentCost": "broadSameAssignmentCost",
    "v4TightSameAssignmentCost": "tightSameAssignmentCost",
    "v4MeanSwapMargin": "meanSwapMargin",
    "v4GlobalAppearanceChange": "globalAppearanceChange",
    "v4MaximumCameraShift": "maximumCameraShift",
    "v4MinimumAlignmentResponse": "minimumAlignmentResponse",
}

DETECTED_FEATURE_NAMES = (
    *V4_CARRY_FEATURES,
    "detectedSameAssignmentCost",
    "detectedSwappedAssignmentCost",
    "detectedSwapMargin",
    "detectedOrientationFlipEvidence",
    "minimumDetectedSideSeparation",
    "detectedSideSeparationChange",
    "beforeDetectedPaletteInstability",
    "afterDetectedPaletteInstability",
    "detectedGlobalAppearanceChange",
    "minimumDetectionCoverage",
    "detectionCoverageChange",
    "minimumDetectionCount",
    "detectionCountChange",
    "minimumMeanDetectionConfidence",
    "meanDetectionConfidenceChange",
    "minimumNearSupport",
    "minimumFarSupport",
    "sideSupportImbalanceChange",
    "minimumTemporalConsistency",
    "temporalConsistencyChange",
)

ADAPTIVE_FEATURE_NAMES = (
    "adaptiveOrientationFlipMagnitude",
    "adaptiveOrientationFlipAgreement",
    "adaptiveOrientationQuality",
)

VISUAL_FEATURE_NAMES = (*DETECTED_FEATURE_NAMES, *ADAPTIVE_FEATURE_NAMES)


class SideSwitchV6Error(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectedSequenceSummary:
    near_palette: np.ndarray
    far_palette: np.ndarray
    global_palette: np.ndarray
    palette_instability: float
    detection_coverage: float
    detection_count: float
    mean_confidence: float
    near_support: float
    far_support: float
    temporal_consistency: float
    raw_candidate_count: float
    inference_milliseconds: float


@dataclass(frozen=True)
class AdaptiveOrientationProfile:
    coordinates: Mapping[int, float]
    qualities: Mapping[int, float]
    assignment_margins: Mapping[int, float]
    anchor_separation: float
    robust_scale: float
    update_count: int
    prototype_drift: float
    update_rate: float


@dataclass(frozen=True)
class V6Model:
    feature_names: tuple[str, ...]
    impute: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    l2: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise SideSwitchV6Error(
                f"model expects (*, {len(self.feature_names)}) features, got {matrix.shape}"
            )
        filled = np.where(np.isfinite(matrix), matrix, self.impute)
        normalized = (filled - self.mean) / self.scale
        logits = np.clip(normalized @ self.weights + self.bias, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "featureNames": list(self.feature_names),
            "impute": self.impute.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_names: Sequence[str] = VISUAL_FEATURE_NAMES,
    ) -> "V6Model":
        names = tuple(str(value) for value in payload.get("featureNames", []))
        expected_names = tuple(expected_names)
        if names != expected_names:
            raise SideSwitchV6Error("v6 model feature signature is invalid")
        model = cls(
            feature_names=names,
            impute=np.asarray(payload.get("impute"), dtype=np.float64),
            mean=np.asarray(payload.get("mean"), dtype=np.float64),
            scale=np.asarray(payload.get("scale"), dtype=np.float64),
            weights=np.asarray(payload.get("weights"), dtype=np.float64),
            bias=float(payload.get("bias")),
            threshold=float(payload.get("threshold")),
            l2=float(payload.get("l2")),
        )
        expected_shape = (len(names),)
        if any(
            value.shape != expected_shape
            for value in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise SideSwitchV6Error("v6 model parameter shapes do not agree")
        if not all(
            np.isfinite(value).all()
            for value in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise SideSwitchV6Error("v6 model parameters must be finite")
        if (
            np.any(model.scale <= 0)
            or not math.isfinite(model.bias)
            or not 0 <= model.threshold <= 1
            or model.l2 <= 0
        ):
            raise SideSwitchV6Error("v6 model scalars are invalid")
        return model


@dataclass(frozen=True)
class _LocalizedPalette:
    detection: PlayerDetection
    palette: np.ndarray
    x_ratio: float
    canonical_hip_y: float
    near_probability: float
    temporal_consistency: float


def _hellinger(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.sqrt(np.maximum(left, 0)) - np.sqrt(np.maximum(right, 0)))
        / math.sqrt(2.0)
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def _normalized_palette(value: np.ndarray) -> np.ndarray:
    numeric = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = float(np.sum(numeric))
    if total <= 1e-12:
        return np.full(52, 1.0 / 52.0, dtype=np.float64)
    return numeric / total


def _weighted_palette(hsv: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pixels = hsv.reshape(-1, 3)
    numeric_weights = weights.reshape(-1).astype(np.float64)
    if float(np.sum(numeric_weights)) < 1e-8:
        return np.full(52, 1.0 / 52.0, dtype=np.float64)
    hs, _, _ = np.histogram2d(
        pixels[:, 0],
        pixels[:, 1],
        bins=(12, 4),
        range=((0.0, 180.0), (0.0, 256.0)),
        weights=numeric_weights,
    )
    value, _ = np.histogram(
        pixels[:, 2], bins=4, range=(0.0, 256.0), weights=numeric_weights
    )
    return _normalized_palette(np.concatenate((hs.reshape(-1), value)))


def _pooled_palette(
    weighted_palettes: Sequence[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, float]:
    if not weighted_palettes:
        return np.full(52, 1.0 / 52.0, dtype=np.float64), 0.0
    weights = np.asarray([max(value[1], 1e-8) for value in weighted_palettes])
    matrix = np.stack([value[0] for value in weighted_palettes])
    pooled = _normalized_palette(np.average(matrix, axis=0, weights=weights))
    instability = float(
        np.median([_hellinger(value[0], pooled) for value in weighted_palettes])
    )
    return pooled, instability


def canonical_y(y_ratio: float, net_y_ratio: float) -> float:
    net = float(np.clip(net_y_ratio, 0.20, 0.80))
    if y_ratio <= net:
        return 0.5 * y_ratio / net
    return 0.5 + 0.5 * (y_ratio - net) / (1.0 - net)


def _detection_palette(frame: np.ndarray, detection: PlayerDetection) -> np.ndarray:
    height, width = frame.shape[:2]
    x0 = max(0, min(width - 1, int(math.floor(detection.x))))
    y0 = max(0, min(height - 1, int(math.floor(detection.y))))
    x1 = max(x0 + 1, min(width, int(math.ceil(detection.x + detection.width))))
    y1 = max(y0 + 1, min(height, int(math.ceil(detection.y + detection.height))))
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    local_height, local_width = hsv.shape[:2]
    yy, xx = np.ogrid[:local_height, :local_width]
    radius_x = max(local_width * 0.5, 1.0)
    radius_y = max(local_height * 0.5, 1.0)
    ellipse = (
        ((xx - (local_width - 1) * 0.5) / radius_x) ** 2
        + ((yy - (local_height - 1) * 0.5) / radius_y) ** 2
        <= 1.0
    )
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0
    weights = ellipse.astype(np.float64) * (0.75 + 0.25 * saturation)
    return _weighted_palette(hsv, weights)


def _select_frame_detections(
    frame: np.ndarray,
    detections: Sequence[PlayerDetection],
    net_y_ratio: float,
) -> list[tuple[PlayerDetection, float, float]]:
    height, width = frame.shape[:2]
    candidates: list[tuple[PlayerDetection, float, float]] = []
    for detection in detections:
        x_ratio = detection.hip_x / width
        y_ratio = canonical_y(detection.hip_y / height, net_y_ratio)
        if not 0.06 <= x_ratio <= 0.94 or not 0.18 <= y_ratio <= 0.98:
            continue
        near_probability = 1.0 / (1.0 + math.exp(-(y_ratio - 0.56) / 0.065))
        candidates.append((detection, x_ratio, near_probability))

    selected: list[tuple[PlayerDetection, float, float]] = []
    for near_side in (False, True):
        side = [
            value
            for value in candidates
            if (value[2] >= 0.5) == near_side
        ]
        side.sort(
            key=lambda value: (
                value[0].score * math.sqrt(max(value[0].width * value[0].height, 1.0)),
                value[0].score,
            ),
            reverse=True,
        )
        selected.extend(side[:MAXIMUM_PLAYERS_PER_SIDE])
    return selected


def summarize_detected_sequence(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> DetectedSequenceSummary:
    """Pool detector-confirmed torso colors over three frames of one rally."""

    if len(frames) != DETECTOR_FRAMES_PER_RALLY:
        raise SideSwitchV6Error(
            f"v6 expects exactly {DETECTOR_FRAMES_PER_RALLY} frames per rally"
        )
    if any(frame.ndim != 3 or frame.shape[2] != 3 or not frame.size for frame in frames):
        raise SideSwitchV6Error("v6 frames must be non-empty BGR images")

    per_frame: list[list[tuple[PlayerDetection, float, float]]] = []
    raw_counts: list[int] = []
    inference_times: list[float] = []
    coverages: list[float] = []
    confidence_values: list[float] = []
    for frame in frames:
        result = detector.detect(frame)
        selected = _select_frame_detections(frame, result.detections, net_y_ratio)
        per_frame.append(selected)
        raw_counts.append(result.raw_candidates)
        inference_times.append(result.inference_milliseconds)
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for detection, _, _ in selected:
            x0 = max(0, int(math.floor(detection.x)))
            y0 = max(0, int(math.floor(detection.y)))
            x1 = min(width, int(math.ceil(detection.x + detection.width)))
            y1 = min(height, int(math.ceil(detection.y + detection.height)))
            mask[y0:y1, x0:x1] = 1
            confidence_values.append(detection.score)
        coverages.append(float(np.mean(mask)))

    localized: list[_LocalizedPalette] = []
    consistency_values: list[float] = []
    for frame_index, (frame, selected) in enumerate(zip(frames, per_frame, strict=True)):
        height, width = frame.shape[:2]
        for detection, x_ratio, near_probability in selected:
            canonical_hip_y = canonical_y(detection.hip_y / height, net_y_ratio)
            matches = 0
            for other_index, other in enumerate(per_frame):
                if other_index == frame_index:
                    continue
                distances = [
                    math.hypot(
                        x_ratio - candidate_x,
                        canonical_hip_y
                        - canonical_y(candidate.hip_y / frames[other_index].shape[0], net_y_ratio),
                    )
                    for candidate, candidate_x, candidate_near in other
                    if (candidate_near >= 0.5) == (near_probability >= 0.5)
                ]
                if distances and min(distances) <= 0.18:
                    matches += 1
            consistency = matches / max(len(frames) - 1, 1)
            consistency_values.append(consistency)
            localized.append(
                _LocalizedPalette(
                    detection=detection,
                    palette=_detection_palette(frame, detection),
                    x_ratio=x_ratio,
                    canonical_hip_y=canonical_hip_y,
                    near_probability=near_probability,
                    temporal_consistency=consistency,
                )
            )

    near_values: list[tuple[np.ndarray, float]] = []
    far_values: list[tuple[np.ndarray, float]] = []
    global_values: list[tuple[np.ndarray, float]] = []
    near_support = 0.0
    far_support = 0.0
    for value in localized:
        support = value.detection.score * (0.5 + 0.5 * value.temporal_consistency)
        near_weight = support * value.near_probability
        far_weight = support * (1.0 - value.near_probability)
        if value.near_probability >= 0.05:
            near_values.append((value.palette, near_weight))
            near_support += near_weight
        if value.near_probability <= 0.95:
            far_values.append((value.palette, far_weight))
            far_support += far_weight
        global_values.append((value.palette, support))
    near, near_instability = _pooled_palette(near_values)
    far, far_instability = _pooled_palette(far_values)
    global_palette, _ = _pooled_palette(global_values)
    total_support = near_support + far_support
    return DetectedSequenceSummary(
        near_palette=near,
        far_palette=far,
        global_palette=global_palette,
        palette_instability=0.5 * (near_instability + far_instability),
        detection_coverage=float(np.mean(coverages)),
        detection_count=float(np.mean([len(value) for value in per_frame])),
        mean_confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
        near_support=near_support / max(total_support, 1e-12),
        far_support=far_support / max(total_support, 1e-12),
        temporal_consistency=(
            float(np.mean(consistency_values)) if consistency_values else 0.0
        ),
        raw_candidate_count=float(np.mean(raw_counts)),
        inference_milliseconds=float(np.sum(inference_times)),
    )


def _weighted_summary_palette(
    summaries: Mapping[int, DetectedSequenceSummary],
    numbers: Sequence[int],
    side: str,
) -> np.ndarray:
    values = []
    for number in numbers:
        summary = summaries[number]
        palette = summary.near_palette if side == "near" else summary.far_palette
        support = summary.near_support if side == "near" else summary.far_support
        weight = support * max(summary.mean_confidence, 0.05)
        values.append((palette, max(weight, 1e-6)))
    pooled, _ = _pooled_palette(values)
    return pooled


def _build_orientation_profile(
    summaries: Mapping[int, DetectedSequenceSummary],
    *,
    update_rate: float,
) -> AdaptiveOrientationProfile:
    rally_numbers = sorted(summaries)
    if len(rally_numbers) < 3:
        raise SideSwitchV6Error("orientation profile needs at least three rallies")
    anchor_numbers = rally_numbers[:3]
    initial_near = _weighted_summary_palette(summaries, anchor_numbers, "near")
    initial_far = _weighted_summary_palette(summaries, anchor_numbers, "far")
    prototype_near = initial_near.copy()
    prototype_far = initial_far.copy()
    anchor_separation = _hellinger(initial_near, initial_far)
    raw_values: dict[int, float] = {}
    qualities: dict[int, float] = {}
    margins: dict[int, float] = {}
    updates = 0
    for rally_number in rally_numbers:
        summary = summaries[rally_number]
        same = 0.5 * (
            _hellinger(summary.near_palette, prototype_near)
            + _hellinger(summary.far_palette, prototype_far)
        )
        swapped = 0.5 * (
            _hellinger(summary.near_palette, prototype_far)
            + _hellinger(summary.far_palette, prototype_near)
        )
        raw = swapped - same
        assignment_confidence = abs(raw) / max(same + swapped, 1e-6)
        balance = 2.0 * min(summary.near_support, summary.far_support)
        side_separation = _hellinger(summary.near_palette, summary.far_palette)
        localization = math.sqrt(
            max(summary.mean_confidence, 0.0)
            * max(summary.temporal_consistency, 0.0)
        )
        quality = float(
            np.clip(
                balance
                * localization
                * min(1.0, side_separation / max(anchor_separation, 0.08)),
                0.0,
                1.0,
            )
        )
        raw_values[rally_number] = raw
        qualities[rally_number] = quality
        margins[rally_number] = assignment_confidence

        if update_rate <= 0 or quality < 0.02 or assignment_confidence < 0.05:
            continue
        if raw >= 0:
            team_near = summary.near_palette
            team_far = summary.far_palette
        else:
            team_near = summary.far_palette
            team_far = summary.near_palette
        alpha = update_rate * min(1.0, quality / 0.25) * min(
            1.0, assignment_confidence / 0.25
        )
        prototype_near = _normalized_palette(
            (1.0 - alpha) * prototype_near + alpha * team_near
        )
        prototype_far = _normalized_palette(
            (1.0 - alpha) * prototype_far + alpha * team_far
        )
        updates += 1

    numeric_raw = np.asarray(list(raw_values.values()), dtype=np.float64)
    robust_scale = max(float(np.median(np.abs(numeric_raw))), 1e-6)
    coordinates = {
        number: float(np.tanh(raw / (2.0 * robust_scale)))
        for number, raw in raw_values.items()
    }
    prototype_drift = 0.5 * (
        _hellinger(initial_near, prototype_near)
        + _hellinger(initial_far, prototype_far)
    )
    return AdaptiveOrientationProfile(
        coordinates=coordinates,
        qualities=qualities,
        assignment_margins=margins,
        anchor_separation=anchor_separation,
        robust_scale=robust_scale,
        update_count=updates,
        prototype_drift=prototype_drift,
        update_rate=update_rate,
    )


def build_adaptive_orientation_profile(
    summaries: Mapping[int, DetectedSequenceSummary]
) -> AdaptiveOrientationProfile:
    return _build_orientation_profile(
        summaries, update_rate=ADAPTIVE_PROTOTYPE_RATE
    )


def build_frozen_orientation_profile(
    summaries: Mapping[int, DetectedSequenceSummary]
) -> AdaptiveOrientationProfile:
    return _build_orientation_profile(summaries, update_rate=0.0)


def orientation_context(
    profile: AdaptiveOrientationProfile, gap_order: int
) -> dict[str, float]:
    before_numbers = [
        number
        for number in range(max(1, gap_order - 2), gap_order + 1)
        if number in profile.coordinates
    ]
    after_numbers = [
        number
        for number in range(gap_order + 1, gap_order + 4)
        if number in profile.coordinates
    ]
    if not before_numbers or not after_numbers:
        return {"before": 0.0, "after": 0.0, "quality": 0.0, "margin": 0.0}
    before = float(np.median([profile.coordinates[number] for number in before_numbers]))
    after = float(np.median([profile.coordinates[number] for number in after_numbers]))
    quality = min(
        float(np.mean([profile.qualities[number] for number in before_numbers])),
        float(np.mean([profile.qualities[number] for number in after_numbers])),
    )
    margin = min(
        float(np.mean([profile.assignment_margins[number] for number in before_numbers])),
        float(np.mean([profile.assignment_margins[number] for number in after_numbers])),
    )
    return {"before": before, "after": after, "quality": quality, "margin": margin}


def detected_features(
    before: DetectedSequenceSummary,
    after: DetectedSequenceSummary,
    v4_features: Mapping[str, Any],
    adaptive_context: Mapping[str, Any],
) -> dict[str, float]:
    same = 0.5 * (
        _hellinger(before.near_palette, after.near_palette)
        + _hellinger(before.far_palette, after.far_palette)
    )
    swapped = 0.5 * (
        _hellinger(before.near_palette, after.far_palette)
        + _hellinger(before.far_palette, after.near_palette)
    )
    before_orientation = before.near_palette - before.far_palette
    after_orientation = after.near_palette - after.far_palette
    before_separation = _hellinger(before.near_palette, before.far_palette)
    after_separation = _hellinger(after.near_palette, after.far_palette)
    context_before = float(adaptive_context.get("before", 0.0))
    context_after = float(adaptive_context.get("after", 0.0))
    context_quality = float(adaptive_context.get("quality", 0.0))
    result = {
        output_name: float(v4_features.get(source_name, math.nan))
        for output_name, source_name in V4_CARRY_FEATURES.items()
    }
    result.update(
        {
            "detectedSameAssignmentCost": same,
            "detectedSwappedAssignmentCost": swapped,
            "detectedSwapMargin": same - swapped,
            "detectedOrientationFlipEvidence": -_cosine(
                before_orientation, after_orientation
            ),
            "minimumDetectedSideSeparation": min(
                before_separation, after_separation
            ),
            "detectedSideSeparationChange": abs(
                before_separation - after_separation
            ),
            "beforeDetectedPaletteInstability": before.palette_instability,
            "afterDetectedPaletteInstability": after.palette_instability,
            "detectedGlobalAppearanceChange": _hellinger(
                before.global_palette, after.global_palette
            ),
            "minimumDetectionCoverage": min(
                before.detection_coverage, after.detection_coverage
            ),
            "detectionCoverageChange": abs(
                before.detection_coverage - after.detection_coverage
            ),
            "minimumDetectionCount": min(
                before.detection_count, after.detection_count
            ),
            "detectionCountChange": abs(
                before.detection_count - after.detection_count
            ),
            "minimumMeanDetectionConfidence": min(
                before.mean_confidence, after.mean_confidence
            ),
            "meanDetectionConfidenceChange": abs(
                before.mean_confidence - after.mean_confidence
            ),
            "minimumNearSupport": min(before.near_support, after.near_support),
            "minimumFarSupport": min(before.far_support, after.far_support),
            "sideSupportImbalanceChange": abs(
                abs(before.near_support - before.far_support)
                - abs(after.near_support - after.far_support)
            ),
            "minimumTemporalConsistency": min(
                before.temporal_consistency, after.temporal_consistency
            ),
            "temporalConsistencyChange": abs(
                before.temporal_consistency - after.temporal_consistency
            ),
            "adaptiveOrientationFlipMagnitude": abs(
                context_before - context_after
            )
            * context_quality,
            "adaptiveOrientationFlipAgreement": max(
                0.0, -context_before * context_after
            )
            * context_quality,
            "adaptiveOrientationQuality": context_quality,
        }
    )
    return result


def event_vector(
    row: Mapping[str, Any], feature_names: Sequence[str] = VISUAL_FEATURE_NAMES
) -> np.ndarray:
    raw = row.get("features") if isinstance(row.get("features"), Mapping) else {}
    return np.asarray(
        [float(raw.get(name, math.nan)) for name in feature_names], dtype=np.float64
    )


def matrix_for(
    events: Sequence[V3Event], feature_names: Sequence[str] = VISUAL_FEATURE_NAMES
) -> np.ndarray:
    names = tuple(feature_names)
    if not events:
        return np.empty((0, len(names)), dtype=np.float64)
    return np.stack([event_vector(event.row, names) for event in events])


def labels_for(events: Sequence[V3Event]) -> np.ndarray:
    return np.asarray([event.label for event in events], dtype=np.float64)


def _fit_preprocessor(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    impute = np.zeros(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        finite = values[np.isfinite(values[:, index]), index]
        impute[index] = float(np.median(finite)) if len(finite) else 0.0
    filled = np.where(np.isfinite(values), values, impute)
    mean = np.mean(filled, axis=0)
    scale = np.std(filled, axis=0)
    scale[scale < 1e-6] = 1.0
    return impute, mean, scale


def fit_model(
    events: Sequence[V3Event],
    l2: float,
    feature_names: Sequence[str] = VISUAL_FEATURE_NAMES,
) -> V6Model:
    if l2 <= 0:
        raise SideSwitchV6Error("l2 must be positive")
    names = tuple(feature_names)
    values = matrix_for(events, names)
    labels = labels_for(events)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise SideSwitchV6Error("training events must contain both classes")
    impute, mean, scale = _fit_preprocessor(values)
    matrix = (np.where(np.isfinite(values), values, impute) - mean) / scale
    sample_weights = np.where(
        labels == 1,
        len(labels) / (2.0 * positives),
        len(labels) / (2.0 * negatives),
    )
    dimensions = matrix.shape[1]
    weights = np.zeros(dimensions)
    bias = 0.0
    design = np.column_stack((matrix, np.ones(len(matrix))))
    regularizer = np.diag(np.r_[np.full(dimensions, l2), 0.0])
    for _ in range(100):
        logits = np.clip(matrix @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = (probabilities - labels) * sample_weights
        gradient = design.T @ error / np.sum(sample_weights)
        gradient[:-1] += l2 * weights
        curvature = probabilities * (1.0 - probabilities) * sample_weights
        hessian = (design.T * curvature) @ design / np.sum(sample_weights)
        hessian += regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        current_logits = matrix @ weights + bias
        current_loss = float(
            np.sum(
                (np.logaddexp(0.0, current_logits) - labels * current_logits)
                * sample_weights
            )
            / np.sum(sample_weights)
            + 0.5 * l2 * (weights @ weights)
        )
        step_scale = 1.0
        while step_scale > 1e-5:
            candidate_weights = weights - step_scale * step[:-1]
            candidate_bias = bias - step_scale * float(step[-1])
            candidate_logits = matrix @ candidate_weights + candidate_bias
            candidate_loss = float(
                np.sum(
                    (np.logaddexp(0.0, candidate_logits) - labels * candidate_logits)
                    * sample_weights
                )
                / np.sum(sample_weights)
                + 0.5 * l2 * (candidate_weights @ candidate_weights)
            )
            if candidate_loss <= current_loss + 1e-12:
                break
            step_scale *= 0.5
        weights = candidate_weights
        bias = candidate_bias
        if float(np.max(np.abs(step_scale * step))) < 1e-8:
            break
    return V6Model(names, impute, mean, scale, weights, bias, 0.5, l2)


def grouped_cross_fit(
    events: Sequence[V3Event],
    l2: float,
    feature_names: Sequence[str] = VISUAL_FEATURE_NAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    names = tuple(feature_names)
    recording_ids = sorted({event.recording_id for event in events})
    if len(recording_ids) < 2:
        raise SideSwitchV6Error("grouped cross-fit needs at least two recordings")
    probabilities = np.full(len(events), np.nan)
    folds: list[dict[str, Any]] = []
    for held_id in recording_ids:
        train_indexes = [
            index for index, event in enumerate(events) if event.recording_id != held_id
        ]
        held_indexes = [
            index for index, event in enumerate(events) if event.recording_id == held_id
        ]
        train_events = [events[index] for index in train_indexes]
        held_events = [events[index] for index in held_indexes]
        model = fit_model(train_events, l2, names)
        probabilities[held_indexes] = model.predict_proba(matrix_for(held_events, names))
        folds.append(
            {
                "heldRecordingId": held_id,
                "trainRows": len(train_events),
                "heldRows": len(held_events),
                "heldPositives": sum(event.label for event in held_events),
            }
        )
    if not np.isfinite(probabilities).all():
        raise SideSwitchV6Error("grouped cross-fit left events unscored")
    return probabilities, {"folds": folds}


def with_threshold(model: V6Model, threshold: float) -> V6Model:
    return replace(model, threshold=float(threshold))


def fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ADAPTIVE_FEATURE_NAMES",
    "ADAPTIVE_PROTOTYPE_RATE",
    "DETECTED_FEATURE_NAMES",
    "DETECTOR_FRAMES_PER_RALLY",
    "FEATURE_ARTIFACT_KIND",
    "FEATURE_ARTIFACT_SCHEMA_VERSION",
    "MODEL_KIND",
    "MODEL_SCHEMA_VERSION",
    "AdaptiveOrientationProfile",
    "DetectedSequenceSummary",
    "V6Model",
    "VISUAL_FEATURE_NAMES",
    "build_adaptive_orientation_profile",
    "build_frozen_orientation_profile",
    "canonical_y",
    "detected_features",
    "event_vector",
    "fingerprint",
    "fit_model",
    "grouped_cross_fit",
    "labels_for",
    "matrix_for",
    "orientation_context",
    "summarize_detected_sequence",
    "with_threshold",
]
