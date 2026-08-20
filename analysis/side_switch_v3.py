"""Cadence-first, low-resolution side-switch specialist for on-device research."""

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


MODEL_KIND = "volleycut-side-switch-specialist-v3"
MODEL_SCHEMA_VERSION = 3
FEATURE_ARTIFACT_KIND = "volleycut-side-switch-features-v3"
FEATURE_ARTIFACT_SCHEMA_VERSION = 3
FRAME_WIDTH = 192
FRAME_HEIGHT = 108
FRAMES_PER_RALLY = 3

FROZEN_RECORDING_SPLIT = {
    "train": (
        "beach-source-01",
        "grass-source-02",
        "grass-source-06",
        "grass-source-01",
        "grass-source-05",
        "grass-source-09",
    ),
    "validation": (
        "grass-source-03",
        "grass-source-04",
        "grass-source-08",
        "grass-source-10",
    ),
    "evaluation": (
        "raw-no-backup-PXL_20260816_160023210",
        "raw-no-backup-PXL_20260816_161923155",
        "raw-no-backup-PXL_20260816_164327879",
        "raw-no-backup-PXL_20260816_171720964",
        "raw-no-backup-PXL_20260816_180646590",
        "raw-no-backup-PXL_20260816_183701800",
        "raw-no-backup-PXL_20260816_190429172",
        "raw-no-backup-PXL_20260816_193307688",
        "raw-no-backup-PXL_20260816_203801418",
        "raw-no-backup-PXL_20260816_210449857",
        "raw-no-backup-PXL_20260816_212717581",
    ),
}
RECORDING_ROLE = {
    recording_id: role
    for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    for recording_id in recording_ids
}

VISUAL_FEATURE_NAMES = (
    "sameAssignmentCost",
    "swappedAssignmentCost",
    "swapMargin",
    "orientationFlipEvidence",
    "minimumMotionCoverage",
    "motionCoverageChange",
    "beforeWithinFrameChange",
    "afterWithinFrameChange",
    "globalAppearanceChange",
    "minimumBlurLog",
    "lumaChange",
    "edgeDensityChange",
)


class SideSwitchV3Error(RuntimeError):
    pass


@dataclass(frozen=True)
class SequenceSummary:
    near_palette: np.ndarray
    far_palette: np.ndarray
    global_palette: np.ndarray
    motion_coverage: float
    within_frame_change: float
    blur_log: float
    luma: float
    edge_density: float


@dataclass(frozen=True)
class V3Event:
    event_id: str
    recording_id: str
    role: str
    gap_order: int
    label: int
    row: Mapping[str, Any]


@dataclass(frozen=True)
class V3Model:
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
            raise SideSwitchV3Error(
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "V3Model":
        names = tuple(str(value) for value in payload.get("featureNames", []))
        if names != VISUAL_FEATURE_NAMES:
            raise SideSwitchV3Error("v3 model feature signature is invalid")
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
            raise SideSwitchV3Error("v3 model parameter shapes do not agree")
        if not all(
            np.isfinite(value).all()
            for value in (model.impute, model.mean, model.scale, model.weights)
        ):
            raise SideSwitchV3Error("v3 model parameters must be finite")
        if (
            np.any(model.scale <= 0)
            or not math.isfinite(model.bias)
            or not 0 <= model.threshold <= 1
            or model.l2 <= 0
        ):
            raise SideSwitchV3Error("v3 model scalars are invalid")
        return model


@dataclass(frozen=True)
class DecoderSettings:
    candidate_margin: int
    distance_penalty: float
    maximum_opportunities: int = SIDE_SWITCH_MAX_OPPORTUNITIES
    reanchor_on_selection: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidateMargin": self.candidate_margin,
            "distancePenalty": self.distance_penalty,
            "maximumOpportunities": self.maximum_opportunities,
            "reanchorOnSelection": self.reanchor_on_selection,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecoderSettings":
        value = cls(
            candidate_margin=int(payload["candidateMargin"]),
            distance_penalty=float(payload["distancePenalty"]),
            maximum_opportunities=int(
                payload.get("maximumOpportunities", SIDE_SWITCH_MAX_OPPORTUNITIES)
            ),
            reanchor_on_selection=bool(payload.get("reanchorOnSelection", True)),
        )
        if value.candidate_margin not in {1, 2, 3, 4}:
            raise SideSwitchV3Error("candidate margin must be one of 1, 2, 3, or 4")
        if value.distance_penalty < 0 or not math.isfinite(value.distance_penalty):
            raise SideSwitchV3Error("distance penalty must be finite and non-negative")
        if value.maximum_opportunities < 1:
            raise SideSwitchV3Error("maximum opportunities must be positive")
        return value


def _weighted_histogram(hsv: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pixels = hsv.reshape(-1, 3)
    numeric_weights = weights.reshape(-1).astype(np.float64)
    if float(np.sum(numeric_weights)) < 1e-8:
        numeric_weights = np.ones(len(pixels), dtype=np.float64)
    parts = []
    for values, bins, limit in (
        (pixels[:, 0], 8, 180.0),
        (pixels[:, 1], 4, 256.0),
        (pixels[:, 2], 4, 256.0),
    ):
        histogram, _ = np.histogram(
            values,
            bins=bins,
            range=(0.0, limit),
            weights=numeric_weights,
        )
        parts.append(histogram.astype(np.float64))
    result = np.concatenate(parts)
    return result / max(float(np.sum(result)), 1e-12)


def _hellinger(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.sqrt(np.maximum(left, 0)) - np.sqrt(np.maximum(right, 0)))
        / math.sqrt(2.0)
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def summarize_sequence(frames: Sequence[np.ndarray]) -> SequenceSummary:
    """Summarize three already-cropped 192×108 BGR frames with cheap operations."""

    if len(frames) < 2:
        raise SideSwitchV3Error("a rally sequence needs at least two frames")
    if any(frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH) for frame in frames):
        raise SideSwitchV3Error("v3 frames must use the frozen 192x108 shape")
    hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames]
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    gray_stack = np.stack(gray_frames).astype(np.float64)
    motion = (np.max(gray_stack, axis=0) - np.min(gray_stack, axis=0)) / 255.0
    median_hsv = np.median(np.stack(hsv_frames), axis=0).astype(np.uint8)
    saturation = median_hsv[:, :, 1].astype(np.float64) / 255.0
    weights = motion + 0.10 * saturation
    top = max(0, round(FRAME_HEIGHT * 0.12))
    divider = round(FRAME_HEIGHT * 0.56)
    left = round(FRAME_WIDTH * 0.08)
    right = round(FRAME_WIDTH * 0.92)
    far_slice = np.s_[top:divider, left:right]
    near_slice = np.s_[divider:FRAME_HEIGHT, left:right]
    pair_changes = [
        float(np.mean(cv2.absdiff(first, second))) / 255.0
        for first, second in zip(gray_frames, gray_frames[1:])
    ]
    blur_values = [
        math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        for gray in gray_frames
    ]
    edge_densities = [
        float(np.mean(cv2.Canny(gray, 60, 120) > 0)) for gray in gray_frames
    ]
    return SequenceSummary(
        near_palette=_weighted_histogram(
            median_hsv[near_slice], weights[near_slice]
        ),
        far_palette=_weighted_histogram(median_hsv[far_slice], weights[far_slice]),
        global_palette=_weighted_histogram(median_hsv, weights),
        motion_coverage=float(np.mean(motion > (8.0 / 255.0))),
        within_frame_change=float(np.mean(pair_changes)),
        blur_log=float(np.median(blur_values)),
        luma=float(np.mean(gray_stack)) / 255.0,
        edge_density=float(np.mean(edge_densities)),
    )


def visual_features(
    before: SequenceSummary, after: SequenceSummary
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
    return {
        "sameAssignmentCost": same,
        "swappedAssignmentCost": swapped,
        "swapMargin": same - swapped,
        "orientationFlipEvidence": -_cosine(
            before_orientation, after_orientation
        ),
        "minimumMotionCoverage": min(
            before.motion_coverage, after.motion_coverage
        ),
        "motionCoverageChange": abs(
            before.motion_coverage - after.motion_coverage
        ),
        "beforeWithinFrameChange": before.within_frame_change,
        "afterWithinFrameChange": after.within_frame_change,
        "globalAppearanceChange": _hellinger(
            before.global_palette, after.global_palette
        ),
        "minimumBlurLog": min(before.blur_log, after.blur_log),
        "lumaChange": abs(before.luma - after.luma),
        "edgeDensityChange": abs(before.edge_density - after.edge_density),
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


def fit_model(events: Sequence[V3Event], l2: float) -> V3Model:
    if l2 <= 0:
        raise SideSwitchV3Error("l2 must be positive")
    values = matrix_for(events)
    labels = labels_for(events)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise SideSwitchV3Error("training events must contain both classes")
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
    return V3Model(
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
        raise SideSwitchV3Error("grouped cross-fit needs at least two recordings")
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
        raise SideSwitchV3Error("grouped cross-fit left events unscored")
    return probabilities, {"folds": folds}


def cadence_opportunities(max_gap_order: int, candidate_margin: int) -> tuple[int, ...]:
    if max_gap_order < 1:
        return ()
    return tuple(
        range(
            SIDE_SWITCH_CADENCE_POINTS,
            max_gap_order + candidate_margin + 1,
            SIDE_SWITCH_CADENCE_POINTS,
        )
    )


def decode_opportunities(
    events: Sequence[V3Event],
    probabilities: np.ndarray,
    threshold: float,
    settings: DecoderSettings,
    *,
    force_each_opportunity: bool = False,
) -> np.ndarray:
    """Decode monotonic one-to-one opportunity/gap assignments, including ±4 overlap."""

    if len(events) != len(probabilities):
        raise SideSwitchV3Error("decoder events and probabilities are not aligned")
    if not events:
        return np.empty(0, dtype=bool)
    ordered = sorted(range(len(events)), key=lambda index: events[index].gap_order)
    if len({events[index].gap_order for index in ordered}) != len(events):
        raise SideSwitchV3Error("decoder expects one reviewed row per rally gap")
    threshold_logit = math.log(
        max(threshold, 1e-9) / max(1.0 - threshold, 1e-9)
    )
    maximum_gap = max(event.gap_order for event in events)
    # State is (last selected gap, next opportunity center); paths remain monotonic.
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
            candidates = [
                index
                for index in ordered
                if events[index].gap_order > last_gap
                and abs(events[index].gap_order - opportunity)
                <= settings.candidate_margin
            ]
            if not force_each_opportunity or not candidates:
                next_key = (last_gap, opportunity + SIDE_SWITCH_CADENCE_POINTS)
                next_states[next_key] = max(
                    next_states.get(next_key, (-math.inf, ())),
                    (prior_score, selected),
                    key=lambda value: value[0],
                )
            for index in candidates:
                probability = float(np.clip(probabilities[index], 1e-9, 1.0 - 1e-9))
                logit = math.log(probability / (1.0 - probability))
                distance = abs(events[index].gap_order - opportunity)
                score = (
                    prior_score
                    + logit
                    - threshold_logit
                    - settings.distance_penalty * distance
                )
                gap = events[index].gap_order
                candidate = (score, (*selected, index))
                next_center = (
                    gap + SIDE_SWITCH_CADENCE_POINTS
                    if settings.reanchor_on_selection
                    else opportunity + SIDE_SWITCH_CADENCE_POINTS
                )
                key = (gap, next_center)
                existing = next_states.get(key)
                if existing is None or candidate[0] > existing[0] + 1e-12:
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
    settings: DecoderSettings,
    *,
    force_each_opportunity: bool = False,
) -> np.ndarray:
    predictions = np.zeros(len(events), dtype=bool)
    for recording_id in sorted({event.recording_id for event in events}):
        indexes = [
            index for index, event in enumerate(events) if event.recording_id == recording_id
        ]
        recording_events = [events[index] for index in indexes]
        predictions[indexes] = decode_opportunities(
            recording_events,
            probabilities[indexes],
            threshold,
            settings,
            force_each_opportunity=force_each_opportunity,
        )
    return predictions


def _maximum_tolerance_matches(
    truth: Sequence[int], predicted: Sequence[int], tolerance: int
) -> int:
    first = sorted(truth)
    second = sorted(predicted)
    dynamic = np.zeros((len(first) + 1, len(second) + 1), dtype=np.int64)
    for i, truth_gap in enumerate(first, start=1):
        for j, predicted_gap in enumerate(second, start=1):
            dynamic[i, j] = max(dynamic[i - 1, j], dynamic[i, j - 1])
            if abs(truth_gap - predicted_gap) <= tolerance:
                dynamic[i, j] = max(dynamic[i, j], dynamic[i - 1, j - 1] + 1)
    return int(dynamic[-1, -1])


def event_metrics(
    events: Sequence[V3Event], predictions: np.ndarray, *, tolerance: int = 0
) -> dict[str, Any]:
    if len(events) != len(predictions):
        raise SideSwitchV3Error("metric events and predictions are not aligned")
    true_positives = 0
    truth_count = 0
    prediction_count = 0
    exact_count_recordings = 0
    recording_ids = sorted({event.recording_id for event in events})
    for recording_id in recording_ids:
        indexes = [
            index for index, event in enumerate(events) if event.recording_id == recording_id
        ]
        truth = [events[index].gap_order for index in indexes if events[index].label == 1]
        predicted = [
            events[index].gap_order for index in indexes if bool(predictions[index])
        ]
        true_positives += _maximum_tolerance_matches(truth, predicted, tolerance)
        truth_count += len(truth)
        prediction_count += len(predicted)
        exact_count_recordings += len(truth) == len(predicted)
    false_positives = prediction_count - true_positives
    false_negatives = truth_count - true_positives
    precision = (
        true_positives / prediction_count if prediction_count else None
    )
    recall = true_positives / truth_count if truth_count else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "recordings": len(recording_ids),
        "truthEvents": truth_count,
        "predictedEvents": prediction_count,
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rallyTolerance": tolerance,
        "exactSwitchCountRecordings": exact_count_recordings,
        "exactSwitchCountAccuracy": (
            exact_count_recordings / len(recording_ids) if recording_ids else None
        ),
    }


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    positives = int(np.sum(truth == 1))
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = truth[order]
    precisions = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precisions * ranked) / positives)


def with_threshold(model: V3Model, threshold: float) -> V3Model:
    return replace(model, threshold=float(threshold))


def fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
