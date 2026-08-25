"""Anonymous endpoint player-identity transport for side-switch T1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Sequence

import cv2
import numpy as np

from analysis.side_switch_player_detector import (
    PlayerDetection,
    QuantizedPersonDetector,
)
from analysis.side_switch_v6 import canonical_y


ENDPOINT_FRAME_FRACTIONS = (0.15, 0.50, 0.85)
MAXIMUM_PLAYERS_PER_SIDE = 2
MAXIMUM_TRACKLETS_PER_SIDE = 3
TRACK_LINK_THRESHOLD = 0.55
TRACK_SPATIAL_SCALE = 0.50
UNMATCHED_TRANSPORT_COST = 0.70
T1_CORE_FEATURE_NAMES = (
    "appearanceTransportSwapMargin",
    "bidirectionalMatchedIdentityMinimum",
    "transportCoverageMinimum",
)
T1_RELIABILITY_FEATURE_NAMES = (
    "sameSideIdentityRetentionPenalty",
    "endpointTeamSeparationMinimum",
    "sceneContinuityConfidence",
)
T1_FEATURE_NAMES = (*T1_CORE_FEATURE_NAMES, *T1_RELIABILITY_FEATURE_NAMES)


@dataclass(frozen=True)
class AppearanceObservation:
    descriptor: np.ndarray
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class AppearanceTracklet:
    descriptor: np.ndarray
    x: float
    y: float
    reliability: float
    observed_frames: int


@dataclass(frozen=True)
class EndpointSummary:
    near: tuple[AppearanceTracklet, ...]
    far: tuple[AppearanceTracklet, ...]
    selected_counts: tuple[int, ...]
    raw_candidate_counts: tuple[int, ...]
    detector_inference_milliseconds: float

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "nearTracklets": len(self.near),
            "farTracklets": len(self.far),
            "selectedPlayersByFrame": list(self.selected_counts),
            "rawCandidatesByFrame": list(self.raw_candidate_counts),
            "detectorInferenceMilliseconds": self.detector_inference_milliseconds,
        }


@dataclass(frozen=True)
class TransportReduction:
    cost: float
    matched_identity_mass: float
    coverage: float
    matched_pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost": self.cost,
            "matchedIdentityMass": self.matched_identity_mass,
            "coverage": self.coverage,
            "matchedPairs": self.matched_pairs,
        }


def endpoint_sample_times(start: float, end: float) -> tuple[float, ...]:
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("T1 endpoint window must be finite and positive")
    duration = end - start
    return tuple(start + duration * fraction for fraction in ENDPOINT_FRAME_FRACTIONS)


def _normalize_histogram(values: np.ndarray) -> np.ndarray:
    numeric = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(np.sum(numeric))
    if total <= 1e-12:
        return np.full(len(numeric), 1.0 / max(len(numeric), 1), dtype=np.float64)
    return numeric / total


def appearance_descriptor(frame: np.ndarray, detection: PlayerDetection) -> np.ndarray:
    """Create the frozen 64-value torso color descriptor."""

    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("T1 descriptor frame must be a non-empty BGR image")
    height, width = frame.shape[:2]
    x0 = max(0, min(width - 1, int(math.floor(detection.x))))
    y0 = max(0, min(height - 1, int(math.floor(detection.y))))
    x1 = max(x0 + 1, min(width, int(math.ceil(detection.x + detection.width))))
    y1 = max(y0 + 1, min(height, int(math.ceil(detection.y + detection.height))))
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    local_height, local_width = crop.shape[:2]
    yy, xx = np.ogrid[:local_height, :local_width]
    ellipse = (
        ((xx - (local_width - 1) * 0.5) / max(local_width * 0.5, 1.0)) ** 2
        + ((yy - (local_height - 1) * 0.5) / max(local_height * 0.5, 1.0)) ** 2
        <= 1.0
    )
    weights = ellipse.astype(np.float64) * (
        0.75 + 0.25 * hsv[:, :, 1].astype(np.float64) / 255.0
    )
    hs, _, _ = np.histogram2d(
        hsv[:, :, 0].reshape(-1),
        hsv[:, :, 1].reshape(-1),
        bins=(12, 4),
        range=((0.0, 180.0), (0.0, 256.0)),
        weights=weights.reshape(-1),
    )
    value, _ = np.histogram(
        hsv[:, :, 2], bins=4, range=(0.0, 256.0), weights=weights
    )
    hsv_block = _normalize_histogram(np.concatenate((hs.reshape(-1), value)))
    lab_blocks = [
        _normalize_histogram(
            np.histogram(
                lab[:, :, channel],
                bins=4,
                range=(0.0, 256.0),
                weights=weights,
            )[0]
        )
        for channel in range(3)
    ]
    descriptor = _normalize_histogram(
        np.concatenate((hsv_block, *lab_blocks))
    )
    if descriptor.shape != (64,) or not np.isfinite(descriptor).all():
        raise AssertionError("T1 descriptor contract changed")
    return descriptor


def hellinger(left: np.ndarray, right: np.ndarray) -> float:
    first = _normalize_histogram(left)
    second = _normalize_histogram(right)
    if first.shape != second.shape:
        raise ValueError("T1 descriptors must have identical shapes")
    return float(np.linalg.norm(np.sqrt(first) - np.sqrt(second)) / math.sqrt(2.0))


def _minimum_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the tiny rectangular assignment exactly without a new dependency."""

    matrix = np.asarray(costs, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("T1 assignment matrix must be finite and non-empty")
    rows, columns = matrix.shape
    best_cost = math.inf
    best_pairs: tuple[tuple[int, int], ...] | None = None
    if rows <= columns:
        for column_order in permutations(range(columns), rows):
            pairs = tuple((row, column_order[row]) for row in range(rows))
            cost = sum(matrix[row, column] for row, column in pairs)
            if cost < best_cost:
                best_cost = float(cost)
                best_pairs = pairs
    else:
        for row_order in permutations(range(rows), columns):
            pairs = tuple((row_order[column], column) for column in range(columns))
            cost = sum(matrix[row, column] for row, column in pairs)
            if cost < best_cost:
                best_cost = float(cost)
                best_pairs = pairs
    if best_pairs is None:
        raise AssertionError("T1 assignment solver failed")
    return (
        np.asarray([value[0] for value in best_pairs], dtype=np.int64),
        np.asarray([value[1] for value in best_pairs], dtype=np.int64),
    )


def _selected_observations(
    frame: np.ndarray,
    detections: Sequence[PlayerDetection],
    net_y_ratio: float,
) -> tuple[list[AppearanceObservation], list[AppearanceObservation]]:
    height, width = frame.shape[:2]
    grouped: dict[str, list[tuple[float, AppearanceObservation]]] = {
        "near": [],
        "far": [],
    }
    for detection in detections:
        x_ratio = detection.hip_x / width
        y_ratio = canonical_y(detection.hip_y / height, net_y_ratio)
        if not 0.06 <= x_ratio <= 0.94 or not 0.18 <= y_ratio <= 0.98:
            continue
        near_probability = 1.0 / (1.0 + math.exp(-(y_ratio - 0.56) / 0.065))
        side = "near" if near_probability >= 0.5 else "far"
        observation = AppearanceObservation(
            descriptor=appearance_descriptor(frame, detection),
            x=float(x_ratio),
            y=float(y_ratio),
            confidence=float(detection.score),
        )
        priority = detection.score * math.sqrt(
            max(detection.width * detection.height, 1.0)
        )
        grouped[side].append((priority, observation))
    for values in grouped.values():
        values.sort(key=lambda value: (value[0], value[1].confidence), reverse=True)
    return (
        [value[1] for value in grouped["near"][:MAXIMUM_PLAYERS_PER_SIDE]],
        [value[1] for value in grouped["far"][:MAXIMUM_PLAYERS_PER_SIDE]],
    )


def _link_cost(left: AppearanceObservation, right: AppearanceObservation) -> float:
    spatial = min(
        math.hypot(left.x - right.x, left.y - right.y) / TRACK_SPATIAL_SCALE,
        1.0,
    )
    return 0.75 * hellinger(left.descriptor, right.descriptor) + 0.25 * spatial


def build_tracklets(
    per_frame: Sequence[Sequence[AppearanceObservation]],
) -> tuple[AppearanceTracklet, ...]:
    if len(per_frame) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T1 expects exactly three ordered observation frames")
    tracks: list[list[AppearanceObservation]] = []
    for observations in per_frame:
        current = list(observations)
        if not tracks:
            tracks.extend([[value] for value in current])
            continue
        if not current:
            continue
        costs = np.asarray(
            [[_link_cost(track[-1], value) for value in current] for track in tracks],
            dtype=np.float64,
        )
        left_indexes, right_indexes = _minimum_assignment(costs)
        used: set[int] = set()
        for left_index, right_index in zip(left_indexes, right_indexes, strict=True):
            if costs[left_index, right_index] <= TRACK_LINK_THRESHOLD:
                tracks[left_index].append(current[right_index])
                used.add(int(right_index))
        tracks.extend(
            [value] for index, value in enumerate(current) if index not in used
        )

    result: list[AppearanceTracklet] = []
    for track in tracks:
        confidences = np.asarray(
            [max(value.confidence, 1e-6) for value in track], dtype=np.float64
        )
        descriptor = _normalize_histogram(
            np.average(
                np.stack([value.descriptor for value in track]),
                axis=0,
                weights=confidences,
            )
        )
        observed_frames = len(track)
        reliability = float(
            np.mean(confidences)
            * (0.5 + 0.5 * observed_frames / len(ENDPOINT_FRAME_FRACTIONS))
        )
        result.append(
            AppearanceTracklet(
                descriptor=descriptor,
                x=float(np.average([value.x for value in track], weights=confidences)),
                y=float(np.average([value.y for value in track], weights=confidences)),
                reliability=reliability,
                observed_frames=observed_frames,
            )
        )
    result.sort(
        key=lambda value: (value.reliability, value.observed_frames), reverse=True
    )
    return tuple(result[:MAXIMUM_TRACKLETS_PER_SIDE])


def summarize_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> EndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T1 endpoint requires exactly three frames")
    near_frames: list[list[AppearanceObservation]] = []
    far_frames: list[list[AppearanceObservation]] = []
    selected_counts: list[int] = []
    raw_counts: list[int] = []
    inference_milliseconds = 0.0
    for frame in frames:
        result = detector.detect(frame)
        near, far = _selected_observations(frame, result.detections, net_y_ratio)
        near_frames.append(near)
        far_frames.append(far)
        selected_counts.append(len(near) + len(far))
        raw_counts.append(result.raw_candidates)
        inference_milliseconds += result.inference_milliseconds
    return EndpointSummary(
        near=build_tracklets(near_frames),
        far=build_tracklets(far_frames),
        selected_counts=tuple(selected_counts),
        raw_candidate_counts=tuple(raw_counts),
        detector_inference_milliseconds=float(inference_milliseconds),
    )


def transport_reduction(
    left: Sequence[AppearanceTracklet], right: Sequence[AppearanceTracklet]
) -> TransportReduction:
    first = tuple(left)
    second = tuple(right)
    if not first or not second:
        return TransportReduction(1.0, 0.0, 0.0, 0)
    costs = np.asarray(
        [
            [hellinger(left_value.descriptor, right_value.descriptor) for right_value in second]
            for left_value in first
        ],
        dtype=np.float64,
    )
    left_indexes, right_indexes = _minimum_assignment(costs)
    left_mass = float(sum(value.reliability for value in first))
    right_mass = float(sum(value.reliability for value in second))
    denominator = max(left_mass, right_mass, 1e-12)
    matched_mass = 0.0
    weighted_cost = 0.0
    identity_mass = 0.0
    for left_index, right_index in zip(left_indexes, right_indexes, strict=True):
        reliability = min(
            first[left_index].reliability, second[right_index].reliability
        )
        cost = float(costs[left_index, right_index])
        matched_mass += reliability
        weighted_cost += reliability * cost
        identity_mass += reliability * max(1.0 - cost, 0.0)
    unmatched_mass = max(denominator - matched_mass, 0.0)
    cost = (weighted_cost + UNMATCHED_TRANSPORT_COST * unmatched_mass) / denominator
    coverage = min(
        matched_mass / max(left_mass, 1e-12),
        matched_mass / max(right_mass, 1e-12),
    )
    return TransportReduction(
        cost=float(np.clip(cost, 0.0, 1.0)),
        matched_identity_mass=float(np.clip(identity_mass / denominator, 0.0, 1.0)),
        coverage=float(np.clip(coverage, 0.0, 1.0)),
        matched_pairs=len(left_indexes),
    )


def transport_features(
    before: EndpointSummary, after: EndpointSummary
) -> tuple[dict[str, float], dict[str, Any]]:
    reductions = {
        "nearNear": transport_reduction(before.near, after.near),
        "farFar": transport_reduction(before.far, after.far),
        "nearFar": transport_reduction(before.near, after.far),
        "farNear": transport_reduction(before.far, after.near),
        "beforeNearFar": transport_reduction(before.near, before.far),
        "afterNearFar": transport_reduction(after.near, after.far),
        "global": transport_reduction(
            (*before.near, *before.far), (*after.near, *after.far)
        ),
    }
    same_cost = 0.5 * (
        reductions["nearNear"].cost + reductions["farFar"].cost
    )
    swapped_cost = 0.5 * (
        reductions["nearFar"].cost + reductions["farNear"].cost
    )
    values = {
        "appearanceTransportSwapMargin": same_cost - swapped_cost,
        "bidirectionalMatchedIdentityMinimum": min(
            reductions["nearFar"].matched_identity_mass,
            reductions["farNear"].matched_identity_mass,
        ),
        "transportCoverageMinimum": min(
            reductions["nearFar"].coverage, reductions["farNear"].coverage
        ),
        "sameSideIdentityRetentionPenalty": 0.5
        * (
            reductions["nearNear"].matched_identity_mass
            + reductions["farFar"].matched_identity_mass
        ),
        "endpointTeamSeparationMinimum": min(
            reductions["beforeNearFar"].cost,
            reductions["afterNearFar"].cost,
        ),
        "sceneContinuityConfidence": reductions["global"].matched_identity_mass,
    }
    if tuple(values) != T1_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T1 feature signature or values changed")
    return values, {
        "sameAssignmentCost": same_cost,
        "swappedAssignmentCost": swapped_cost,
        "reductions": {
            name: value.to_dict() for name, value in reductions.items()
        },
    }
