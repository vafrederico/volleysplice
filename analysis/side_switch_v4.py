"""Multi-frame, court-normalized side identity features for side-switch research."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from analysis.side_switch_v3 import V3Event


MODEL_KIND = "volleycut-side-switch-specialist-v4"
MODEL_SCHEMA_VERSION = 4
FEATURE_ARTIFACT_KIND = "volleycut-side-switch-features-v4"
FEATURE_ARTIFACT_SCHEMA_VERSION = 4
FRAME_WIDTH = 256
FRAME_HEIGHT = 144
FRAMES_PER_RALLY = 7
CANONICAL_NET_Y = 0.50

VISUAL_FEATURE_NAMES = (
    "broadSameAssignmentCost",
    "broadSwappedAssignmentCost",
    "broadSwapMargin",
    "broadOrientationFlipEvidence",
    "tightSameAssignmentCost",
    "tightSwappedAssignmentCost",
    "tightSwapMargin",
    "tightOrientationFlipEvidence",
    "meanSwapMargin",
    "crossScaleSwapDisagreement",
    "minimumSideSeparation",
    "sideSeparationChange",
    "beforeSidePaletteInstability",
    "afterSidePaletteInstability",
    "globalAppearanceChange",
    "minimumForegroundCoverage",
    "foregroundCoverageChange",
    "maximumCameraShift",
    "minimumAlignmentResponse",
)


class SideSwitchV4Error(RuntimeError):
    pass


@dataclass(frozen=True)
class CourtGeometry:
    net_y_ratio: float
    confidence: float
    detected_frames: int
    sampled_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "netYRatio": self.net_y_ratio,
            "confidence": self.confidence,
            "detectedFrames": self.detected_frames,
            "sampledFrames": self.sampled_frames,
        }


@dataclass(frozen=True)
class SidePaletteSummary:
    near: np.ndarray
    far: np.ndarray
    instability: float


@dataclass(frozen=True)
class SequenceSummary:
    broad: SidePaletteSummary
    tight: SidePaletteSummary
    global_palette: np.ndarray
    foreground_coverage: float
    maximum_camera_shift: float
    minimum_alignment_response: float


@dataclass(frozen=True)
class V4Model:
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
            raise SideSwitchV4Error(
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "V4Model":
        names = tuple(str(value) for value in payload.get("featureNames", []))
        if names != VISUAL_FEATURE_NAMES:
            raise SideSwitchV4Error("v4 model feature signature is invalid")
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
            raise SideSwitchV4Error("v4 model parameter shapes do not agree")
        if not all(
            np.isfinite(value).all()
            for value in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise SideSwitchV4Error("v4 model parameters must be finite")
        if (
            np.any(model.scale <= 0)
            or not math.isfinite(model.bias)
            or not 0 <= model.threshold <= 1
            or model.l2 <= 0
        ):
            raise SideSwitchV4Error("v4 model scalars are invalid")
        return model


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
        numeric_weights = np.ones(len(pixels), dtype=np.float64)
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


def _horizontal_line_candidate(frame: np.ndarray) -> tuple[float, float] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 110)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(24, FRAME_WIDTH // 10),
        minLineLength=FRAME_WIDTH * 0.32,
        maxLineGap=FRAME_WIDTH * 0.12,
    )
    if lines is None:
        return None
    candidates: list[tuple[float, float]] = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in raw)
        length = math.hypot(x2 - x1, y2 - y1)
        coverage = abs(x2 - x1) / FRAME_WIDTH
        y_ratio = (y1 + y2) / (2.0 * FRAME_HEIGHT)
        slope = abs(y2 - y1) / max(abs(x2 - x1), 1.0)
        if 0.25 <= y_ratio <= 0.68 and slope <= 0.08 and coverage >= 0.30:
            # The foreground net is normally the lowest long horizontal court line.
            score = coverage * (1.0 + 0.35 * y_ratio) + 0.10 * length / FRAME_WIDTH
            candidates.append((score, y_ratio))
    if not candidates:
        return None
    score, y_ratio = max(candidates)
    return y_ratio, min(1.0, score)


def estimate_court_geometry(frames: Sequence[np.ndarray]) -> CourtGeometry:
    """Estimate a recording-level net height from early low-resolution frames."""

    if not frames:
        raise SideSwitchV4Error("court calibration needs at least one frame")
    if any(frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH) for frame in frames):
        raise SideSwitchV4Error("v4 calibration frames must use the frozen shape")
    candidates = [
        candidate
        for frame in frames
        if (candidate := _horizontal_line_candidate(frame)) is not None
    ]
    if not candidates:
        return CourtGeometry(0.50, 0.0, 0, len(frames))
    positions = np.asarray([value[0] for value in candidates], dtype=np.float64)
    scores = np.asarray([value[1] for value in candidates], dtype=np.float64)
    center = float(np.median(positions))
    deviations = np.abs(positions - center)
    retained = deviations <= max(0.035, float(np.median(deviations)) * 2.5)
    if np.any(retained):
        center = float(np.average(positions[retained], weights=scores[retained]))
    detection_fraction = len(candidates) / len(frames)
    dispersion = float(np.median(np.abs(positions - np.median(positions))))
    confidence = detection_fraction * max(0.0, 1.0 - dispersion / 0.10)
    return CourtGeometry(
        net_y_ratio=float(np.clip(center, 0.25, 0.68)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        detected_frames=len(candidates),
        sampled_frames=len(frames),
    )


def normalize_court_frame(frame: np.ndarray, geometry: CourtGeometry) -> np.ndarray:
    """Warp vertical coordinates so the detected net has a stable normalized height."""

    if frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise SideSwitchV4Error("v4 frames must use the frozen 256x144 shape")
    net = float(np.clip(geometry.net_y_ratio, 0.20, 0.80))
    output_y = np.arange(FRAME_HEIGHT, dtype=np.float32) / max(FRAME_HEIGHT - 1, 1)
    source_y = np.where(
        output_y <= CANONICAL_NET_Y,
        output_y * net / CANONICAL_NET_Y,
        net
        + (output_y - CANONICAL_NET_Y)
        * (1.0 - net)
        / (1.0 - CANONICAL_NET_Y),
    )
    map_y = np.repeat((source_y * (FRAME_HEIGHT - 1))[:, None], FRAME_WIDTH, axis=1)
    map_x = np.repeat(
        np.arange(FRAME_WIDTH, dtype=np.float32)[None, :], FRAME_HEIGHT, axis=0
    )
    return cv2.remap(
        frame,
        map_x,
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _align_frames(
    frames: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], float, float]:
    reference_index = len(frames) // 2
    reference_gray = cv2.cvtColor(frames[reference_index], cv2.COLOR_BGR2GRAY)
    calibration_height = round(FRAME_HEIGHT * 0.42)
    reference = cv2.GaussianBlur(
        reference_gray[:calibration_height], (7, 7), 0
    ).astype(np.float32)
    window = cv2.createHanningWindow(
        (FRAME_WIDTH, calibration_height), cv2.CV_32F
    )
    aligned: list[np.ndarray] = []
    shifts: list[float] = []
    responses: list[float] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = cv2.GaussianBlur(
            gray[:calibration_height], (7, 7), 0
        ).astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(reference, current, window)
        normalized_shift = math.hypot(dx / FRAME_WIDTH, dy / FRAME_HEIGHT)
        if (
            math.isfinite(normalized_shift)
            and math.isfinite(response)
            and response >= 0.02
            and normalized_shift <= 0.12
        ):
            transform = np.asarray([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
            aligned_frame = cv2.warpAffine(
                frame,
                transform,
                (FRAME_WIDTH, FRAME_HEIGHT),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            shifts.append(normalized_shift)
        else:
            aligned_frame = frame
            shifts.append(0.0)
        responses.append(float(response) if math.isfinite(response) else 0.0)
        aligned.append(aligned_frame)
    return aligned, max(shifts, default=0.0), min(responses, default=0.0)


def _pooled_palette(
    hsv_frames: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    region: tuple[slice, slice],
) -> tuple[np.ndarray, float]:
    palettes = [
        _weighted_palette(hsv[region], weight[region])
        for hsv, weight in zip(hsv_frames, weights, strict=True)
    ]
    pooled = np.mean(np.stack(palettes), axis=0)
    pooled /= max(float(np.sum(pooled)), 1e-12)
    instability = float(np.median([_hellinger(value, pooled) for value in palettes]))
    return pooled, instability


def summarize_sequence(
    frames: Sequence[np.ndarray], geometry: CourtGeometry
) -> SequenceSummary:
    """Aggregate seven court-normalized frames into broad and tight side identities."""

    if len(frames) < 3:
        raise SideSwitchV4Error("a v4 rally sequence needs at least three frames")
    normalized = [normalize_court_frame(frame, geometry) for frame in frames]
    aligned, maximum_shift, minimum_response = _align_frames(normalized)
    hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in aligned]
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in aligned]
    gray_stack = np.stack(gray_frames).astype(np.float64)
    median_gray = np.median(gray_stack, axis=0)
    motion = (np.max(gray_stack, axis=0) - np.min(gray_stack, axis=0)) / 255.0
    weights = []
    for hsv, gray in zip(hsv_frames, gray_stack, strict=True):
        frame_difference = np.abs(gray - median_gray) / 255.0
        weights.append(
            np.maximum(frame_difference - (2.0 / 255.0), 0.0)
            + 0.20 * np.maximum(motion - (5.0 / 255.0), 0.0)
        )

    def region(top: float, bottom: float) -> tuple[slice, slice]:
        return (
            slice(round(FRAME_HEIGHT * top), round(FRAME_HEIGHT * bottom)),
            slice(round(FRAME_WIDTH * 0.06), round(FRAME_WIDTH * 0.94)),
        )

    broad_far, broad_far_instability = _pooled_palette(
        hsv_frames, weights, region(0.15, 0.60)
    )
    broad_near, broad_near_instability = _pooled_palette(
        hsv_frames, weights, region(0.44, 0.98)
    )
    tight_far, tight_far_instability = _pooled_palette(
        hsv_frames, weights, region(0.27, 0.53)
    )
    tight_near, tight_near_instability = _pooled_palette(
        hsv_frames, weights, region(0.56, 0.91)
    )
    global_palette, _ = _pooled_palette(
        hsv_frames, weights, region(0.08, 0.98)
    )
    return SequenceSummary(
        broad=SidePaletteSummary(
            near=broad_near,
            far=broad_far,
            instability=0.5 * (broad_near_instability + broad_far_instability),
        ),
        tight=SidePaletteSummary(
            near=tight_near,
            far=tight_far,
            instability=0.5 * (tight_near_instability + tight_far_instability),
        ),
        global_palette=global_palette,
        foreground_coverage=float(np.mean(motion > (10.0 / 255.0))),
        maximum_camera_shift=maximum_shift,
        minimum_alignment_response=minimum_response,
    )


def _assignment_features(
    before: SidePaletteSummary, after: SidePaletteSummary
) -> tuple[float, float, float, float]:
    same = 0.5 * (
        _hellinger(before.near, after.near) + _hellinger(before.far, after.far)
    )
    swapped = 0.5 * (
        _hellinger(before.near, after.far) + _hellinger(before.far, after.near)
    )
    before_orientation = before.near - before.far
    after_orientation = after.near - after.far
    return same, swapped, same - swapped, -_cosine(
        before_orientation, after_orientation
    )


def visual_features(
    before: SequenceSummary, after: SequenceSummary
) -> dict[str, float]:
    broad = _assignment_features(before.broad, after.broad)
    tight = _assignment_features(before.tight, after.tight)
    before_separation = 0.5 * (
        _hellinger(before.broad.near, before.broad.far)
        + _hellinger(before.tight.near, before.tight.far)
    )
    after_separation = 0.5 * (
        _hellinger(after.broad.near, after.broad.far)
        + _hellinger(after.tight.near, after.tight.far)
    )
    return {
        "broadSameAssignmentCost": broad[0],
        "broadSwappedAssignmentCost": broad[1],
        "broadSwapMargin": broad[2],
        "broadOrientationFlipEvidence": broad[3],
        "tightSameAssignmentCost": tight[0],
        "tightSwappedAssignmentCost": tight[1],
        "tightSwapMargin": tight[2],
        "tightOrientationFlipEvidence": tight[3],
        "meanSwapMargin": 0.5 * (broad[2] + tight[2]),
        "crossScaleSwapDisagreement": abs(broad[2] - tight[2]),
        "minimumSideSeparation": min(before_separation, after_separation),
        "sideSeparationChange": abs(before_separation - after_separation),
        "beforeSidePaletteInstability": 0.5
        * (before.broad.instability + before.tight.instability),
        "afterSidePaletteInstability": 0.5
        * (after.broad.instability + after.tight.instability),
        "globalAppearanceChange": _hellinger(
            before.global_palette, after.global_palette
        ),
        "minimumForegroundCoverage": min(
            before.foreground_coverage, after.foreground_coverage
        ),
        "foregroundCoverageChange": abs(
            before.foreground_coverage - after.foreground_coverage
        ),
        "maximumCameraShift": max(
            before.maximum_camera_shift, after.maximum_camera_shift
        ),
        "minimumAlignmentResponse": min(
            before.minimum_alignment_response, after.minimum_alignment_response
        ),
    }


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


def fit_model(events: Sequence[V3Event], l2: float) -> V4Model:
    if l2 <= 0:
        raise SideSwitchV4Error("l2 must be positive")
    values = matrix_for(events)
    labels = labels_for(events)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise SideSwitchV4Error("training events must contain both classes")
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
    return V4Model(
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
        raise SideSwitchV4Error("grouped cross-fit needs at least two recordings")
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
        raise SideSwitchV4Error("grouped cross-fit left events unscored")
    return probabilities, {"folds": folds}


def with_threshold(model: V4Model, threshold: float) -> V4Model:
    return replace(model, threshold=float(threshold))
