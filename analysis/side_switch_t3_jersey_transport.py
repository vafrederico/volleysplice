"""Team-isolated jersey transport features for side-switch T3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Sequence

import cv2
import numpy as np

from analysis.side_switch_player_detector import PlayerDetection, QuantizedPersonDetector
from analysis.side_switch_t1_transport import hellinger
from analysis.side_switch_v6 import canonical_y


ENDPOINT_FRAME_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
MAXIMUM_PLAYERS_PER_SIDE = 3
MAXIMUM_TRACKLETS_PER_SIDE = 3
MINIMUM_TRACKLET_FRAMES = 2
TRACK_LINK_THRESHOLD = 0.50
TRACK_SPATIAL_SCALE = 0.50
BACKGROUND_LAB_DISTANCE = 10.0
MINIMUM_MASK_PIXELS = 32
MINIMUM_MASK_FRACTION = 0.15
T3_CORE_FEATURE_NAMES = (
    "jerseyTeamTransportSwapMargin",
    "jerseyReliableSwapEvidence",
    "jerseyReliableContinuityEvidence",
)
T3_DIAGNOSTIC_FEATURE_NAMES = (
    "jerseyCrossSideSimilarityMinimum",
    "jerseySameSideSimilarityMinimum",
    "jerseyTeamReliabilityMinimum",
    "jerseyTeamSeparationMinimum",
    "jerseyTeamCohesionMinimum",
    "jerseyBaseReliabilityGate",
    "jerseySwapReliabilityGate",
    "jerseyContinuityReliabilityGate",
)
T3_FEATURE_NAMES = (*T3_CORE_FEATURE_NAMES, *T3_DIAGNOSTIC_FEATURE_NAMES)


@dataclass(frozen=True)
class JerseyObservation:
    descriptor: np.ndarray
    x: float
    y: float
    scale: float
    confidence: float
    support: float
    background_fallback: bool


@dataclass(frozen=True)
class JerseyTracklet:
    descriptor: np.ndarray
    x: float
    y: float
    scale: float
    reliability: float
    observed_frames: int


@dataclass(frozen=True)
class TeamSummary:
    descriptor: np.ndarray | None
    reliability: float
    cohesion: float
    qualifying_tracklets: int

    @property
    def available(self) -> bool:
        return self.descriptor is not None


@dataclass(frozen=True)
class EndpointSummary:
    near: TeamSummary
    far: TeamSummary
    selected_counts: tuple[int, ...]
    raw_candidate_counts: tuple[int, ...]
    descriptor_counts: tuple[int, ...]
    background_fallbacks: int
    detector_inference_milliseconds: float

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "nearTeamAvailable": self.near.available,
            "farTeamAvailable": self.far.available,
            "nearTeamReliability": self.near.reliability,
            "farTeamReliability": self.far.reliability,
            "nearTeamCohesion": self.near.cohesion,
            "farTeamCohesion": self.far.cohesion,
            "nearQualifyingTracklets": self.near.qualifying_tracklets,
            "farQualifyingTracklets": self.far.qualifying_tracklets,
            "selectedPlayersByFrame": list(self.selected_counts),
            "descriptorPlayersByFrame": list(self.descriptor_counts),
            "rawCandidatesByFrame": list(self.raw_candidate_counts),
            "backgroundMaskFallbacks": self.background_fallbacks,
            "detectorInferenceMilliseconds": self.detector_inference_milliseconds,
        }


def endpoint_sample_times(start: float, end: float) -> tuple[float, ...]:
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("T3 endpoint window must be finite and positive")
    duration = end - start
    return tuple(start + duration * value for value in ENDPOINT_FRAME_FRACTIONS)


def _normalize(values: np.ndarray) -> np.ndarray:
    numeric = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(np.sum(numeric))
    if total <= 1e-12:
        return np.full(len(numeric), 1.0 / max(len(numeric), 1), dtype=np.float64)
    return numeric / total


def _clip_box(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(math.floor(x0))))
    top = max(0, min(height - 1, int(math.floor(y0))))
    right = max(left + 1, min(width, int(math.ceil(x1))))
    bottom = max(top + 1, min(height, int(math.ceil(y1))))
    return left, top, right, bottom


def _torso_box(
    detection: PlayerDetection, width: int, height: int
) -> tuple[int, int, int, int]:
    span = detection.hip_y - detection.shoulder_y
    valid = (
        math.isfinite(span)
        and span >= 0.12 * detection.height
        and detection.shoulder_x >= detection.x - 0.1 * detection.width
        and detection.shoulder_x <= detection.x + 1.1 * detection.width
        and detection.hip_x >= detection.x - 0.1 * detection.width
        and detection.hip_x <= detection.x + 1.1 * detection.width
    )
    if valid:
        center_x = 0.5 * (detection.shoulder_x + detection.hip_x)
        half_width = max(2.0, min(0.28 * detection.width, 0.42 * span))
        return _clip_box(
            center_x - half_width,
            detection.shoulder_y + 0.10 * span,
            center_x + half_width,
            detection.shoulder_y + 0.95 * span,
            width,
            height,
        )
    return _clip_box(
        detection.x + 0.25 * detection.width,
        detection.y + 0.18 * detection.height,
        detection.x + 0.75 * detection.width,
        detection.y + 0.58 * detection.height,
        width,
        height,
    )


def _background_lab(frame: np.ndarray, detection: PlayerDetection) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _clip_box(
        detection.x,
        detection.y,
        detection.x + detection.width,
        detection.y + detection.height,
        width,
        height,
    )
    crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
    local_height, local_width = crop.shape[:2]
    ring_x = max(1, int(math.ceil(local_width * 0.125)))
    ring_y = max(1, int(math.ceil(local_height * 0.125)))
    mask = np.ones((local_height, local_width), dtype=bool)
    if local_height > 2 * ring_y and local_width > 2 * ring_x:
        mask[ring_y:-ring_y, ring_x:-ring_x] = False
    pixels = crop[mask]
    if not len(pixels):
        pixels = crop.reshape(-1, 3)
    return np.median(pixels.astype(np.float64), axis=0)


def jersey_descriptor(
    frame: np.ndarray, detection: PlayerDetection
) -> tuple[np.ndarray, float, bool]:
    """Return the frozen jersey descriptor, support, and background fallback."""

    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("T3 descriptor frame must be a non-empty BGR image")
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _torso_box(detection, width, height)
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    local_height, local_width = crop.shape[:2]
    yy, xx = np.ogrid[:local_height, :local_width]
    ellipse = (
        ((xx - (local_width - 1) * 0.5) / max(local_width * 0.5, 1.0)) ** 2
        + ((yy - (local_height - 1) * 0.5) / max(local_height * 0.5, 1.0)) ** 2
        <= 1.0
    )
    skin = (
        (ycrcb[:, :, 1] >= 133)
        & (ycrcb[:, :, 1] <= 173)
        & (ycrcb[:, :, 2] >= 77)
        & (ycrcb[:, :, 2] <= 127)
    )
    base = ellipse & ~skin
    background = _background_lab(frame, detection)
    distance = np.linalg.norm(lab.astype(np.float64) - background, axis=2)
    retained = base & (distance >= BACKGROUND_LAB_DISTANCE)
    ellipse_pixels = max(int(np.sum(ellipse)), 1)
    fallback = (
        int(np.sum(retained)) < MINIMUM_MASK_PIXELS
        or float(np.sum(retained)) / ellipse_pixels < MINIMUM_MASK_FRACTION
    )
    if fallback:
        retained = base
    support = float(np.sum(retained)) / ellipse_pixels
    weights = retained.astype(np.float64) * (
        0.75 + 0.25 * hsv[:, :, 1].astype(np.float64) / 255.0
    )
    hs, _, _ = np.histogram2d(
        hsv[:, :, 0].reshape(-1),
        hsv[:, :, 1].reshape(-1),
        bins=(12, 4),
        range=((0.0, 180.0), (0.0, 256.0)),
        weights=weights.reshape(-1),
    )
    value = np.histogram(
        hsv[:, :, 2], bins=4, range=(0.0, 256.0), weights=weights
    )[0]
    hsv_block = _normalize(np.concatenate((hs.reshape(-1), value)))
    lab_blocks = [
        _normalize(
            np.histogram(
                lab[:, :, channel],
                bins=4,
                range=(0.0, 256.0),
                weights=weights,
            )[0]
        )
        for channel in range(3)
    ]
    descriptor = _normalize(np.concatenate((hsv_block, *lab_blocks)))
    if descriptor.shape != (64,) or not np.isfinite(descriptor).all():
        raise AssertionError("T3 jersey descriptor contract changed")
    return descriptor, support, fallback


def _minimum_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(costs, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("T3 assignment matrix must be finite and non-empty")
    rows, columns = matrix.shape
    best_cost = math.inf
    best: tuple[tuple[int, int], ...] | None = None
    if rows <= columns:
        for order in permutations(range(columns), rows):
            pairs = tuple((row, order[row]) for row in range(rows))
            cost = sum(matrix[row, column] for row, column in pairs)
            if cost < best_cost:
                best_cost, best = float(cost), pairs
    else:
        for order in permutations(range(rows), columns):
            pairs = tuple((order[column], column) for column in range(columns))
            cost = sum(matrix[row, column] for row, column in pairs)
            if cost < best_cost:
                best_cost, best = float(cost), pairs
    if best is None:
        raise AssertionError("T3 assignment solver failed")
    return (
        np.asarray([value[0] for value in best], dtype=np.int64),
        np.asarray([value[1] for value in best], dtype=np.int64),
    )


def _link_cost(left: JerseyObservation, right: JerseyObservation) -> float:
    spatial = min(
        math.hypot(left.x - right.x, left.y - right.y) / TRACK_SPATIAL_SCALE,
        1.0,
    )
    scale = min(abs(math.log(max(left.scale, 1e-6) / max(right.scale, 1e-6))) / math.log(2.0), 1.0)
    return 0.60 * hellinger(left.descriptor, right.descriptor) + 0.25 * spatial + 0.15 * scale


def build_tracklets(
    per_frame: Sequence[Sequence[JerseyObservation]],
) -> tuple[JerseyTracklet, ...]:
    if len(per_frame) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T3 expects exactly five ordered observation frames")
    tracks: list[list[JerseyObservation]] = []
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
    result: list[JerseyTracklet] = []
    for track in tracks:
        if len(track) < MINIMUM_TRACKLET_FRAMES:
            continue
        weights = np.asarray(
            [max(value.confidence * value.support, 1e-6) for value in track],
            dtype=np.float64,
        )
        descriptor = _normalize(
            np.average(
                np.stack([value.descriptor for value in track]),
                axis=0,
                weights=weights,
            )
        )
        reliability = float(
            np.mean([value.confidence for value in track])
            * (0.40 + 0.60 * len(track) / len(ENDPOINT_FRAME_FRACTIONS))
            * math.sqrt(max(float(np.mean([value.support for value in track])), 0.0))
        )
        result.append(
            JerseyTracklet(
                descriptor=descriptor,
                x=float(np.average([value.x for value in track], weights=weights)),
                y=float(np.average([value.y for value in track], weights=weights)),
                scale=float(np.average([value.scale for value in track], weights=weights)),
                reliability=reliability,
                observed_frames=len(track),
            )
        )
    result.sort(key=lambda value: (value.reliability, value.observed_frames), reverse=True)
    return tuple(result[:MAXIMUM_TRACKLETS_PER_SIDE])


def summarize_team(tracklets: Sequence[JerseyTracklet]) -> TeamSummary:
    selected = tuple(tracklets)[:MAXIMUM_TRACKLETS_PER_SIDE]
    if not selected:
        return TeamSummary(None, 0.0, 0.0, 0)
    weights = np.asarray([max(value.reliability, 1e-6) for value in selected])
    descriptor = _normalize(
        np.average(
            np.stack([value.descriptor for value in selected]), axis=0, weights=weights
        )
    )
    cohesion = 1.0 - float(
        np.average(
            [hellinger(value.descriptor, descriptor) for value in selected],
            weights=weights,
        )
    )
    return TeamSummary(
        descriptor=descriptor,
        reliability=float(np.clip(np.sum(weights) / 2.0, 0.0, 1.0)),
        cohesion=float(np.clip(cohesion, 0.0, 1.0)),
        qualifying_tracklets=len(selected),
    )


def _selected_observations(
    frame: np.ndarray,
    detections: Sequence[PlayerDetection],
    net_y_ratio: float,
) -> tuple[list[JerseyObservation], list[JerseyObservation], int]:
    height, width = frame.shape[:2]
    grouped: dict[str, list[tuple[float, JerseyObservation]]] = {"near": [], "far": []}
    fallbacks = 0
    for detection in detections:
        x_ratio = detection.hip_x / width
        y_ratio = canonical_y(detection.hip_y / height, net_y_ratio)
        if not 0.06 <= x_ratio <= 0.94 or not 0.18 <= y_ratio <= 0.98:
            continue
        descriptor, support, fallback = jersey_descriptor(frame, detection)
        if support <= 1e-6:
            continue
        fallbacks += int(fallback)
        side = "near" if y_ratio >= 0.56 else "far"
        observation = JerseyObservation(
            descriptor=descriptor,
            x=float(x_ratio),
            y=float(y_ratio),
            scale=float(detection.height / height),
            confidence=float(detection.score),
            support=support,
            background_fallback=fallback,
        )
        priority = detection.score * math.sqrt(max(detection.width * detection.height, 1.0))
        grouped[side].append((priority, observation))
    for values in grouped.values():
        values.sort(key=lambda value: (value[0], value[1].confidence), reverse=True)
    return (
        [value[1] for value in grouped["near"][:MAXIMUM_PLAYERS_PER_SIDE]],
        [value[1] for value in grouped["far"][:MAXIMUM_PLAYERS_PER_SIDE]],
        fallbacks,
    )


def summarize_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> EndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T3 endpoint requires exactly five frames")
    near_frames: list[list[JerseyObservation]] = []
    far_frames: list[list[JerseyObservation]] = []
    selected_counts: list[int] = []
    descriptor_counts: list[int] = []
    raw_counts: list[int] = []
    fallbacks = 0
    inference_milliseconds = 0.0
    for frame in frames:
        result = detector.detect(frame)
        near, far, local_fallbacks = _selected_observations(
            frame, result.detections, net_y_ratio
        )
        near_frames.append(near)
        far_frames.append(far)
        selected_counts.append(len(near) + len(far))
        descriptor_counts.append(len(near) + len(far))
        raw_counts.append(result.raw_candidates)
        fallbacks += local_fallbacks
        inference_milliseconds += result.inference_milliseconds
    return EndpointSummary(
        near=summarize_team(build_tracklets(near_frames)),
        far=summarize_team(build_tracklets(far_frames)),
        selected_counts=tuple(selected_counts),
        raw_candidate_counts=tuple(raw_counts),
        descriptor_counts=tuple(descriptor_counts),
        background_fallbacks=fallbacks,
        detector_inference_milliseconds=float(inference_milliseconds),
    )


def _team_cost(left: TeamSummary, right: TeamSummary) -> float:
    if left.descriptor is None or right.descriptor is None:
        return 1.0
    return hellinger(left.descriptor, right.descriptor)


def team_transport_features(
    before: EndpointSummary, after: EndpointSummary
) -> tuple[dict[str, float], dict[str, Any]]:
    costs = {
        "nearNear": _team_cost(before.near, after.near),
        "farFar": _team_cost(before.far, after.far),
        "nearFar": _team_cost(before.near, after.far),
        "farNear": _team_cost(before.far, after.near),
        "beforeNearFar": _team_cost(before.near, before.far),
        "afterNearFar": _team_cost(after.near, after.far),
    }
    same_cost = 0.5 * (costs["nearNear"] + costs["farFar"])
    swapped_cost = 0.5 * (costs["nearFar"] + costs["farNear"])
    margin = same_cost - swapped_cost
    cross_similarity = min(1.0 - costs["nearFar"], 1.0 - costs["farNear"])
    reliability = min(
        before.near.reliability,
        before.far.reliability,
        after.near.reliability,
        after.far.reliability,
    )
    separation = min(costs["beforeNearFar"], costs["afterNearFar"])
    cohesion = min(
        before.near.cohesion,
        before.far.cohesion,
        after.near.cohesion,
        after.far.cohesion,
    )
    same_similarity = min(1.0 - costs["nearNear"], 1.0 - costs["farFar"])
    base_gate = min(reliability, separation, cohesion)
    swap_gate = min(base_gate, cross_similarity)
    continuity_gate = min(base_gate, same_similarity)
    values = {
        "jerseyTeamTransportSwapMargin": margin,
        "jerseyReliableSwapEvidence": max(margin, 0.0) * swap_gate,
        "jerseyReliableContinuityEvidence": max(-margin, 0.0) * continuity_gate,
        "jerseyCrossSideSimilarityMinimum": cross_similarity,
        "jerseySameSideSimilarityMinimum": same_similarity,
        "jerseyTeamReliabilityMinimum": reliability,
        "jerseyTeamSeparationMinimum": separation,
        "jerseyTeamCohesionMinimum": cohesion,
        "jerseyBaseReliabilityGate": base_gate,
        "jerseySwapReliabilityGate": swap_gate,
        "jerseyContinuityReliabilityGate": continuity_gate,
    }
    if tuple(values) != T3_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T3 team transport feature contract changed")
    return values, {
        "sameAssignmentCost": same_cost,
        "swappedAssignmentCost": swapped_cost,
        "teamCosts": costs,
    }
