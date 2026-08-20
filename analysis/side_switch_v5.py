"""Player-isolated side identity and persistent-orientation decoding for v5."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.side_switch_training_policy import (
    SIDE_SWITCH_CADENCE_POINTS,
    SIDE_SWITCH_MAX_OPPORTUNITIES,
)
from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v4 import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CourtGeometry,
    _align_frames,
    normalize_court_frame,
)


MODEL_KIND = "volleycut-side-switch-specialist-v5"
MODEL_SCHEMA_VERSION = 5
FEATURE_ARTIFACT_KIND = "volleycut-side-switch-features-v5"
FEATURE_ARTIFACT_SCHEMA_VERSION = 5
FRAMES_PER_RALLY = 7

V4_CARRY_FEATURES = {
    "v4BroadSameAssignmentCost": "broadSameAssignmentCost",
    "v4TightSameAssignmentCost": "tightSameAssignmentCost",
    "v4MeanSwapMargin": "meanSwapMargin",
    "v4GlobalAppearanceChange": "globalAppearanceChange",
    "v4MaximumCameraShift": "maximumCameraShift",
    "v4MinimumAlignmentResponse": "minimumAlignmentResponse",
}

VISUAL_FEATURE_NAMES = (
    *V4_CARRY_FEATURES,
    "playerSameAssignmentCost",
    "playerSwappedAssignmentCost",
    "playerSwapMargin",
    "playerOrientationFlipEvidence",
    "minimumPlayerSideSeparation",
    "playerSideSeparationChange",
    "beforePlayerPaletteInstability",
    "afterPlayerPaletteInstability",
    "playerGlobalAppearanceChange",
    "minimumProposalCoverage",
    "proposalCoverageChange",
    "minimumProposalCount",
    "proposalCountChange",
    "minimumNearSupport",
    "minimumFarSupport",
    "sideSupportImbalanceChange",
)


class SideSwitchV5Error(RuntimeError):
    pass


@dataclass(frozen=True)
class PlayerSequenceSummary:
    near_palette: np.ndarray
    far_palette: np.ndarray
    global_palette: np.ndarray
    palette_instability: float
    proposal_coverage: float
    proposal_count: float
    near_support: float
    far_support: float


@dataclass(frozen=True)
class OrientationProfile:
    coordinates: Mapping[int, float]
    qualities: Mapping[int, float]
    anchor_separation: float
    robust_scale: float


@dataclass(frozen=True)
class V5Model:
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
            raise SideSwitchV5Error(
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "V5Model":
        names = tuple(str(value) for value in payload.get("featureNames", []))
        if names != VISUAL_FEATURE_NAMES:
            raise SideSwitchV5Error("v5 model feature signature is invalid")
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
        expected = (len(VISUAL_FEATURE_NAMES),)
        if any(
            value.shape != expected
            for value in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise SideSwitchV5Error("v5 model parameter shapes do not agree")
        if not all(
            np.isfinite(value).all()
            for value in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise SideSwitchV5Error("v5 model parameters must be finite")
        if (
            np.any(model.scale <= 0)
            or not math.isfinite(model.bias)
            or not 0 <= model.threshold <= 1
            or model.l2 <= 0
        ):
            raise SideSwitchV5Error("v5 model scalars are invalid")
        return model


@dataclass(frozen=True)
class OrientationDecoderSettings:
    candidate_margin: int
    distance_penalty: float
    orientation_weight: float
    maximum_opportunities: int = SIDE_SWITCH_MAX_OPPORTUNITIES
    reanchor_on_selection: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidateMargin": self.candidate_margin,
            "distancePenalty": self.distance_penalty,
            "orientationWeight": self.orientation_weight,
            "maximumOpportunities": self.maximum_opportunities,
            "reanchorOnSelection": self.reanchor_on_selection,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrientationDecoderSettings":
        value = cls(
            candidate_margin=int(payload["candidateMargin"]),
            distance_penalty=float(payload["distancePenalty"]),
            orientation_weight=float(payload["orientationWeight"]),
            maximum_opportunities=int(
                payload.get("maximumOpportunities", SIDE_SWITCH_MAX_OPPORTUNITIES)
            ),
            reanchor_on_selection=bool(payload.get("reanchorOnSelection", True)),
        )
        if value.candidate_margin not in {1, 2, 3, 4}:
            raise SideSwitchV5Error("candidate margin must be one of 1, 2, 3, or 4")
        if (
            value.distance_penalty < 0
            or value.orientation_weight < 0
            or not math.isfinite(value.distance_penalty)
            or not math.isfinite(value.orientation_weight)
        ):
            raise SideSwitchV5Error("decoder weights must be finite and non-negative")
        if value.maximum_opportunities < 1:
            raise SideSwitchV5Error("maximum opportunities must be positive")
        return value


def _hellinger(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.sqrt(np.maximum(left, 0)) - np.sqrt(np.maximum(right, 0)))
        / math.sqrt(2.0)
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


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
        pixels[:, 2],
        bins=4,
        range=(0.0, 256.0),
        weights=numeric_weights,
    )
    result = np.concatenate((hs.reshape(-1), value)).astype(np.float64)
    return result / max(float(np.sum(result)), 1e-12)


def _proposal_boxes(frame_difference: np.ndarray) -> list[tuple[int, int, int, int]]:
    numeric = np.clip(frame_difference * 255.0, 0.0, 255.0).astype(np.uint8)
    threshold = max(12, int(np.percentile(numeric, 94)))
    mask = np.where(numeric >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        bottom = y + height
        aspect = height / max(width, 1)
        if (
            area < 10
            or area > FRAME_WIDTH * FRAME_HEIGHT * 0.035
            or width < 2
            or height < 4
            or width > FRAME_WIDTH * 0.18
            or height > FRAME_HEIGHT * 0.48
            or not 0.45 <= aspect <= 5.5
            or bottom < FRAME_HEIGHT * 0.35
            or y > FRAME_HEIGHT * 0.95
        ):
            continue
        pad_x = max(1, round(width * 0.18))
        pad_y = max(1, round(height * 0.10))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(FRAME_WIDTH, x + width + pad_x)
        lower = min(FRAME_HEIGHT, bottom + pad_y)
        vertical_shape = min(aspect, 2.5)
        score = area * (0.50 + lower / FRAME_HEIGHT) ** 2 * vertical_shape
        candidates.append((score, (left, top, right - left, lower - top)))
    candidates.sort(reverse=True)

    def overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        first_x, first_y, first_w, first_h = first
        second_x, second_y, second_w, second_h = second
        intersection_w = max(
            0, min(first_x + first_w, second_x + second_w) - max(first_x, second_x)
        )
        intersection_h = max(
            0, min(first_y + first_h, second_y + second_h) - max(first_y, second_y)
        )
        intersection = intersection_w * intersection_h
        union = first_w * first_h + second_w * second_h - intersection
        return intersection / union if union else 0.0

    selected: list[tuple[int, int, int, int]] = []
    for _, box in candidates:
        if all(overlap(box, existing) < 0.35 for existing in selected):
            selected.append(box)
        if len(selected) == 6:
            break
    return selected


def _pooled_palette(
    weighted_palettes: Sequence[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, float]:
    if not weighted_palettes:
        return np.full(52, 1.0 / 52.0, dtype=np.float64), 0.0
    weights = np.asarray([value[1] for value in weighted_palettes], dtype=np.float64)
    matrix = np.stack([value[0] for value in weighted_palettes])
    pooled = np.average(matrix, axis=0, weights=weights)
    pooled /= max(float(np.sum(pooled)), 1e-12)
    instability = float(
        np.median([_hellinger(value[0], pooled) for value in weighted_palettes])
    )
    return pooled, instability


def summarize_player_sequence(
    frames: Sequence[np.ndarray], geometry: CourtGeometry
) -> PlayerSequenceSummary:
    """Create near/far palettes from motion-component player proposals."""

    if len(frames) < 3:
        raise SideSwitchV5Error("a v5 rally sequence needs at least three frames")
    if any(frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH) for frame in frames):
        raise SideSwitchV5Error("v5 frames must use the frozen 256x144 shape")
    normalized = [normalize_court_frame(frame, geometry) for frame in frames]
    aligned, _, _ = _align_frames(normalized)
    hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in aligned]
    gray_stack = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in aligned]
    ).astype(np.float64)
    median_gray = np.median(gray_stack, axis=0)
    motion = (np.max(gray_stack, axis=0) - np.min(gray_stack, axis=0)) / 255.0

    near_palettes: list[tuple[np.ndarray, float]] = []
    far_palettes: list[tuple[np.ndarray, float]] = []
    global_palettes: list[tuple[np.ndarray, float]] = []
    coverages: list[float] = []
    proposal_counts: list[int] = []
    near_support = 0.0
    far_support = 0.0
    for hsv, gray in zip(hsv_frames, gray_stack, strict=True):
        difference = np.abs(gray - median_gray) / 255.0
        boxes = _proposal_boxes(difference)
        proposal_counts.append(len(boxes))
        covered = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=bool)
        saturation = hsv[:, :, 1].astype(np.float64) / 255.0
        for x, y, width, height in boxes:
            region = np.s_[y : y + height, x : x + width]
            covered[region] = True
            local_weights = (
                difference[region]
                + 0.15 * motion[region]
                + 0.01 * saturation[region]
            )
            palette = _weighted_palette(hsv[region], local_weights)
            support = max(float(np.sum(local_weights)), 1e-8)
            foot_y = (y + height) / FRAME_HEIGHT
            near_probability = 1.0 / (1.0 + math.exp(-(foot_y - 0.63) / 0.06))
            far_probability = 1.0 - near_probability
            if near_probability >= 0.08:
                weight = support * near_probability
                near_palettes.append((palette, weight))
                near_support += weight
            if far_probability >= 0.08:
                weight = support * far_probability
                far_palettes.append((palette, weight))
                far_support += weight
            global_palettes.append((palette, support))
        coverages.append(float(np.mean(covered)))

    near, near_instability = _pooled_palette(near_palettes)
    far, far_instability = _pooled_palette(far_palettes)
    global_palette, _ = _pooled_palette(global_palettes)
    total_support = near_support + far_support
    return PlayerSequenceSummary(
        near_palette=near,
        far_palette=far,
        global_palette=global_palette,
        palette_instability=0.5 * (near_instability + far_instability),
        proposal_coverage=float(np.mean(coverages)),
        proposal_count=float(np.mean(proposal_counts)),
        near_support=near_support / max(total_support, 1e-12),
        far_support=far_support / max(total_support, 1e-12),
    )


def build_orientation_profile(
    summaries: Mapping[int, PlayerSequenceSummary]
) -> OrientationProfile:
    """Compare every rally with team-side palettes anchored before the first switch."""

    rally_numbers = sorted(summaries)
    if len(rally_numbers) < 3:
        raise SideSwitchV5Error("orientation profile needs at least three rallies")
    anchor_numbers = rally_numbers[:3]
    anchor_near = np.mean(
        np.stack([summaries[number].near_palette for number in anchor_numbers]), axis=0
    )
    anchor_far = np.mean(
        np.stack([summaries[number].far_palette for number in anchor_numbers]), axis=0
    )
    anchor_near /= max(float(np.sum(anchor_near)), 1e-12)
    anchor_far /= max(float(np.sum(anchor_far)), 1e-12)
    anchor_separation = _hellinger(anchor_near, anchor_far)
    raw_values = []
    quality_values = []
    for rally_number in rally_numbers:
        summary = summaries[rally_number]
        same = 0.5 * (
            _hellinger(summary.near_palette, anchor_near)
            + _hellinger(summary.far_palette, anchor_far)
        )
        swapped = 0.5 * (
            _hellinger(summary.near_palette, anchor_far)
            + _hellinger(summary.far_palette, anchor_near)
        )
        raw_values.append(swapped - same)
        balance = 2.0 * min(summary.near_support, summary.far_support)
        quality_values.append(float(np.clip(balance * anchor_separation, 0.0, 1.0)))
    raw = np.asarray(raw_values, dtype=np.float64)
    robust_scale = float(np.median(np.abs(raw)))
    robust_scale = max(robust_scale, 1e-6)
    coordinates = np.tanh(raw / (2.0 * robust_scale))
    return OrientationProfile(
        coordinates={
            rally_number: float(coordinate)
            for rally_number, coordinate in zip(
                rally_numbers, coordinates, strict=True
            )
        },
        qualities={
            rally_number: float(quality)
            for rally_number, quality in zip(
                rally_numbers, quality_values, strict=True
            )
        },
        anchor_separation=anchor_separation,
        robust_scale=robust_scale,
    )


def orientation_context(
    profile: OrientationProfile, gap_order: int
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
        return {"before": 0.0, "after": 0.0, "quality": 0.0}
    before = float(np.median([profile.coordinates[number] for number in before_numbers]))
    after = float(np.median([profile.coordinates[number] for number in after_numbers]))
    quality = min(
        float(np.mean([profile.qualities[number] for number in before_numbers])),
        float(np.mean([profile.qualities[number] for number in after_numbers])),
    )
    return {"before": before, "after": after, "quality": quality}


def player_features(
    before: PlayerSequenceSummary,
    after: PlayerSequenceSummary,
    v4_features: Mapping[str, Any],
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
    result = {
        output_name: float(v4_features.get(source_name, math.nan))
        for output_name, source_name in V4_CARRY_FEATURES.items()
    }
    result.update(
        {
            "playerSameAssignmentCost": same,
            "playerSwappedAssignmentCost": swapped,
            "playerSwapMargin": same - swapped,
            "playerOrientationFlipEvidence": -_cosine(
                before_orientation, after_orientation
            ),
            "minimumPlayerSideSeparation": min(
                before_separation, after_separation
            ),
            "playerSideSeparationChange": abs(
                before_separation - after_separation
            ),
            "beforePlayerPaletteInstability": before.palette_instability,
            "afterPlayerPaletteInstability": after.palette_instability,
            "playerGlobalAppearanceChange": _hellinger(
                before.global_palette, after.global_palette
            ),
            "minimumProposalCoverage": min(
                before.proposal_coverage, after.proposal_coverage
            ),
            "proposalCoverageChange": abs(
                before.proposal_coverage - after.proposal_coverage
            ),
            "minimumProposalCount": min(
                before.proposal_count, after.proposal_count
            ),
            "proposalCountChange": abs(
                before.proposal_count - after.proposal_count
            ),
            "minimumNearSupport": min(before.near_support, after.near_support),
            "minimumFarSupport": min(before.far_support, after.far_support),
            "sideSupportImbalanceChange": abs(
                abs(before.near_support - before.far_support)
                - abs(after.near_support - after.far_support)
            ),
        }
    )
    return result


def event_vector(row: Mapping[str, Any]) -> np.ndarray:
    raw = row.get("features") if isinstance(row.get("features"), Mapping) else {}
    return np.asarray(
        [float(raw.get(name, math.nan)) for name in VISUAL_FEATURE_NAMES],
        dtype=np.float64,
    )


def matrix_for(events: Sequence[V3Event]) -> np.ndarray:
    if not events:
        return np.empty((0, len(VISUAL_FEATURE_NAMES)), dtype=np.float64)
    return np.stack([event_vector(event.row) for event in events])


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


def fit_model(events: Sequence[V3Event], l2: float) -> V5Model:
    if l2 <= 0:
        raise SideSwitchV5Error("l2 must be positive")
    values = matrix_for(events)
    labels = labels_for(events)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise SideSwitchV5Error("training events must contain both classes")
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
    return V5Model(
        VISUAL_FEATURE_NAMES,
        impute,
        mean,
        scale,
        weights,
        bias,
        0.5,
        l2,
    )


def grouped_cross_fit(
    events: Sequence[V3Event], l2: float
) -> tuple[np.ndarray, dict[str, Any]]:
    recording_ids = sorted({event.recording_id for event in events})
    if len(recording_ids) < 2:
        raise SideSwitchV5Error("grouped cross-fit needs at least two recordings")
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
        model = fit_model(train_events, l2)
        probabilities[held_indexes] = model.predict_proba(matrix_for(held_events))
        folds.append(
            {
                "heldRecordingId": held_id,
                "trainRows": len(train_events),
                "heldRows": len(held_events),
                "heldPositives": sum(event.label for event in held_events),
            }
        )
    if not np.isfinite(probabilities).all():
        raise SideSwitchV5Error("grouped cross-fit left events unscored")
    return probabilities, {"folds": folds}


def _orientation_selection_score(event: V3Event, current_sign: float) -> float:
    context = event.row.get("orientationContext")
    if not isinstance(context, Mapping):
        return 0.0
    before = float(context.get("before", 0.0))
    after = float(context.get("after", 0.0))
    quality = float(context.get("quality", 0.0))
    if not all(math.isfinite(value) for value in (before, after, quality)):
        return 0.0
    flip_agreement = 0.5 * current_sign * (before - after)
    return quality * (flip_agreement - 0.25)


def decode_orientation_opportunities(
    events: Sequence[V3Event],
    probabilities: np.ndarray,
    threshold: float,
    settings: OrientationDecoderSettings,
) -> np.ndarray:
    """Decode cadence opportunities while carrying whole-set orientation parity."""

    if len(events) != len(probabilities):
        raise SideSwitchV5Error("decoder events and probabilities are not aligned")
    if not events:
        return np.empty(0, dtype=bool)
    ordered = sorted(range(len(events)), key=lambda index: events[index].gap_order)
    if len({events[index].gap_order for index in ordered}) != len(events):
        raise SideSwitchV5Error("decoder expects one reviewed row per rally gap")
    threshold_logit = math.log(
        max(threshold, 1e-9) / max(1.0 - threshold, 1e-9)
    )
    maximum_gap = max(event.gap_order for event in events)
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (-1, SIDE_SWITCH_CADENCE_POINTS): (0.0, ())
    }
    for _ in range(settings.maximum_opportunities):
        next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for (last_gap, opportunity), (prior_score, selected) in states.items():
            if opportunity - settings.candidate_margin > maximum_gap:
                existing = next_states.get((last_gap, opportunity))
                if existing is None or prior_score > existing[0]:
                    next_states[(last_gap, opportunity)] = (prior_score, selected)
                continue
            skip_key = (last_gap, opportunity + SIDE_SWITCH_CADENCE_POINTS)
            existing = next_states.get(skip_key)
            if existing is None or prior_score > existing[0]:
                next_states[skip_key] = (prior_score, selected)
            current_sign = 1.0 if len(selected) % 2 == 0 else -1.0
            for index in ordered:
                gap = events[index].gap_order
                if (
                    gap <= last_gap
                    or abs(gap - opportunity) > settings.candidate_margin
                ):
                    continue
                probability = float(np.clip(probabilities[index], 1e-9, 1.0 - 1e-9))
                logit = math.log(probability / (1.0 - probability))
                score = (
                    prior_score
                    + logit
                    - threshold_logit
                    - settings.distance_penalty * abs(gap - opportunity)
                    + settings.orientation_weight
                    * _orientation_selection_score(events[index], current_sign)
                )
                next_center = (
                    gap + SIDE_SWITCH_CADENCE_POINTS
                    if settings.reanchor_on_selection
                    else opportunity + SIDE_SWITCH_CADENCE_POINTS
                )
                key = (gap, next_center)
                candidate = (score, (*selected, index))
                existing = next_states.get(key)
                if existing is None or score > existing[0] + 1e-12:
                    next_states[key] = candidate
        states = next_states
    selected = max(states.values(), key=lambda value: value[0])[1] if states else ()
    predictions = np.zeros(len(events), dtype=bool)
    predictions[list(selected)] = True
    return predictions


def decode_all(
    events: Sequence[V3Event],
    probabilities: np.ndarray,
    threshold: float,
    settings: OrientationDecoderSettings,
) -> np.ndarray:
    predictions = np.zeros(len(events), dtype=bool)
    for recording_id in sorted({event.recording_id for event in events}):
        indexes = [
            index for index, event in enumerate(events) if event.recording_id == recording_id
        ]
        recording_events = [events[index] for index in indexes]
        predictions[indexes] = decode_orientation_opportunities(
            recording_events, probabilities[indexes], threshold, settings
        )
    return predictions


def with_threshold(model: V5Model, threshold: float) -> V5Model:
    return replace(model, threshold=float(threshold))


def fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
