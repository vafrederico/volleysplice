"""Explicit two-mode jersey transport for side-switch T8."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from analysis.side_switch_player_detector import QuantizedPersonDetector
from analysis.side_switch_t1_transport import hellinger
from analysis.side_switch_t3_jersey_transport import (
    ENDPOINT_FRAME_FRACTIONS,
    T3_FEATURE_NAMES,
    JerseyObservation,
    _normalize,
    _selected_observations,
)
from analysis.side_switch_t4_selective_far import (
    _far_detections,
    _near_detections,
    far_crop_bounds,
)
from analysis.side_switch_t5_court_tracking import track_court_team


def _t8_name(t3_name: str) -> str:
    if not t3_name.startswith("jersey"):
        raise ValueError(f"unexpected jersey feature name: {t3_name}")
    return "twoModeJersey" + t3_name[len("jersey") :]


T8_FEATURE_NAMES = tuple(_t8_name(name) for name in T3_FEATURE_NAMES)
T8_CORE_FEATURE_NAMES = T8_FEATURE_NAMES[:3]


@dataclass(frozen=True)
class AppearanceMode:
    descriptor: np.ndarray
    support: float
    observations: int
    observed_frames: int
    medoid_frame: int
    medoid_index: int
    dispersion: float


@dataclass(frozen=True)
class TwoModeTrack:
    modes: tuple[AppearanceMode, ...]
    available: bool
    reliability: float
    cohesion: float
    observed_frames: int
    total_observations: int
    assignment_objective: float
    support_entropy: float


@dataclass(frozen=True)
class TwoModeEndpointSummary:
    near: TwoModeTrack
    far: TwoModeTrack
    selected_counts: tuple[int, ...]
    full_raw_candidate_counts: tuple[int, ...]
    far_raw_candidate_counts: tuple[int, ...]
    background_fallbacks: int
    full_detector_inference_milliseconds: float
    far_detector_inference_milliseconds: float
    far_crop_top: int
    far_crop_bottom: int
    frame_height: int

    def to_diagnostic(self) -> dict[str, Any]:
        def track(value: TwoModeTrack) -> dict[str, Any]:
            return {
                "teamAvailable": value.available,
                "teamReliability": value.reliability,
                "teamCohesion": value.cohesion,
                "observedFrames": value.observed_frames,
                "totalObservations": value.total_observations,
                "assignmentObjective": value.assignment_objective,
                "supportEntropy": value.support_entropy,
                "modes": [
                    {
                        "support": mode.support,
                        "observations": mode.observations,
                        "observedFrames": mode.observed_frames,
                        "medoidFrame": mode.medoid_frame,
                        "medoidIndex": mode.medoid_index,
                        "dispersion": mode.dispersion,
                    }
                    for mode in value.modes
                ],
            }

        return {
            "near": track(self.near),
            "far": track(self.far),
            "selectedPlayersByFrame": list(self.selected_counts),
            "fullRawCandidatesByFrame": list(self.full_raw_candidate_counts),
            "farRawCandidatesByFrame": list(self.far_raw_candidate_counts),
            "backgroundMaskFallbacks": self.background_fallbacks,
            "fullDetectorInferenceMilliseconds": self.full_detector_inference_milliseconds,
            "farDetectorInferenceMilliseconds": self.far_detector_inference_milliseconds,
            "farCrop": {
                "top": self.far_crop_top,
                "bottom": self.far_crop_bottom,
                "frameHeight": self.frame_height,
            },
        }


def _assignment(
    flat: Sequence[tuple[int, int, JerseyObservation]], medoids: tuple[int, ...]
) -> tuple[list[int], float]:
    base_weights = [max(value.confidence * value.support, 1e-6) for _, _, value in flat]
    assigned: list[int] = []
    weighted_cost = 0.0
    for item_index, (_, _, value) in enumerate(flat):
        distances = [hellinger(value.descriptor, flat[index][2].descriptor) for index in medoids]
        mode_index = min(range(len(distances)), key=lambda index: (distances[index], index))
        assigned.append(mode_index)
        weighted_cost += base_weights[item_index] * distances[mode_index]
    return assigned, float(weighted_cost / sum(base_weights))


def track_two_modes(
    per_frame: Sequence[Sequence[JerseyObservation]],
) -> TwoModeTrack:
    if len(per_frame) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T8 expects exactly five ordered observation frames")
    base = track_court_team(per_frame)
    flat = [
        (frame_index, observation_index, observation)
        for frame_index, observations in enumerate(per_frame)
        for observation_index, observation in enumerate(observations)
    ]
    if not flat:
        return TwoModeTrack((), False, 0.0, 0.0, 0, 0, 0.0, 0.0)
    if len(flat) == 1:
        medoids = (0,)
        assigned, objective = [0], 0.0
    else:
        best: tuple[tuple[float, int, int, int, int], tuple[int, int], list[int], float] | None = None
        for left, right in combinations(range(len(flat)), 2):
            local_assignment, local_objective = _assignment(flat, (left, right))
            left_frame, left_observation, _ = flat[left]
            right_frame, right_observation, _ = flat[right]
            key = (
                local_objective,
                left_frame,
                left_observation,
                right_frame,
                right_observation,
            )
            if best is None or key < best[0]:
                best = (key, (left, right), local_assignment, local_objective)
        if best is None:
            raise AssertionError("T8 two-medoid selection failed")
        _, medoids, assigned, objective = best
    base_weights = np.asarray(
        [max(value.confidence * value.support, 1e-6) for _, _, value in flat],
        dtype=np.float64,
    )
    total_weight = float(np.sum(base_weights))
    modes: list[AppearanceMode] = []
    for mode_index, medoid_index in enumerate(medoids):
        indexes = [index for index, value in enumerate(assigned) if value == mode_index]
        if not indexes:
            continue
        weights = base_weights[indexes]
        descriptor = _normalize(
            np.average(
                np.stack([flat[index][2].descriptor for index in indexes]),
                axis=0,
                weights=weights,
            )
        )
        dispersion = float(
            np.average(
                [hellinger(flat[index][2].descriptor, descriptor) for index in indexes],
                weights=weights,
            )
        )
        frame_index, observation_index, _ = flat[medoid_index]
        modes.append(
            AppearanceMode(
                descriptor=descriptor,
                support=float(np.sum(weights) / total_weight),
                observations=len(indexes),
                observed_frames=len({flat[index][0] for index in indexes}),
                medoid_frame=frame_index,
                medoid_index=observation_index,
                dispersion=dispersion,
            )
        )
    supports = np.asarray([mode.support for mode in modes], dtype=np.float64)
    entropy = 0.0
    if len(supports) > 1:
        entropy = float(-np.sum(supports * np.log(np.maximum(supports, 1e-12))) / math.log(len(supports)))
    return TwoModeTrack(
        modes=tuple(modes),
        available=base.team.available,
        reliability=base.team.reliability if base.team.available else 0.0,
        cohesion=float(np.clip(1.0 - objective, 0.0, 1.0)) if base.team.available else 0.0,
        observed_frames=base.observed_frames,
        total_observations=len(flat),
        assignment_objective=objective,
        support_entropy=entropy,
    )


def summarize_two_mode_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> TwoModeEndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T8 endpoint requires exactly five frames")
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("T8 endpoint frames must share one shape")
    frame_height = int(frames[0].shape[0])
    top, bottom = far_crop_bounds(frame_height, net_y_ratio)
    near_frames: list[list[JerseyObservation]] = []
    far_frames: list[list[JerseyObservation]] = []
    selected_counts: list[int] = []
    full_raw_counts: list[int] = []
    far_raw_counts: list[int] = []
    fallbacks = 0
    full_inference = 0.0
    far_inference = 0.0
    for frame in frames:
        full_result = detector.detect(frame)
        far_result = detector.detect(np.ascontiguousarray(frame[top:bottom]))
        near, _, near_fallbacks = _selected_observations(
            frame, _near_detections(full_result, frame_height, net_y_ratio), net_y_ratio
        )
        _, far, far_fallbacks = _selected_observations(
            frame, _far_detections(far_result, top, frame_height, net_y_ratio), net_y_ratio
        )
        near_frames.append(near)
        far_frames.append(far)
        selected_counts.append(len(near) + len(far))
        full_raw_counts.append(full_result.raw_candidates)
        far_raw_counts.append(far_result.raw_candidates)
        fallbacks += near_fallbacks + far_fallbacks
        full_inference += full_result.inference_milliseconds
        far_inference += far_result.inference_milliseconds
    return TwoModeEndpointSummary(
        near=track_two_modes(near_frames),
        far=track_two_modes(far_frames),
        selected_counts=tuple(selected_counts),
        full_raw_candidate_counts=tuple(full_raw_counts),
        far_raw_candidate_counts=tuple(far_raw_counts),
        background_fallbacks=fallbacks,
        full_detector_inference_milliseconds=float(full_inference),
        far_detector_inference_milliseconds=float(far_inference),
        far_crop_top=top,
        far_crop_bottom=bottom,
        frame_height=frame_height,
    )


def mode_set_cost(left: TwoModeTrack, right: TwoModeTrack) -> float:
    if not left.available or not right.available or not left.modes or not right.modes:
        return 1.0
    left_directed = sum(
        mode.support * min(hellinger(mode.descriptor, other.descriptor) for other in right.modes)
        for mode in left.modes
    )
    right_directed = sum(
        mode.support * min(hellinger(mode.descriptor, other.descriptor) for other in left.modes)
        for mode in right.modes
    )
    return float(0.5 * (left_directed + right_directed))


def two_mode_transport_features(
    before: TwoModeEndpointSummary, after: TwoModeEndpointSummary
) -> tuple[dict[str, float], dict[str, Any]]:
    costs = {
        "nearNear": mode_set_cost(before.near, after.near),
        "farFar": mode_set_cost(before.far, after.far),
        "nearFar": mode_set_cost(before.near, after.far),
        "farNear": mode_set_cost(before.far, after.near),
        "beforeNearFar": mode_set_cost(before.near, before.far),
        "afterNearFar": mode_set_cost(after.near, after.far),
    }
    same_cost = 0.5 * (costs["nearNear"] + costs["farFar"])
    swapped_cost = 0.5 * (costs["nearFar"] + costs["farNear"])
    margin = same_cost - swapped_cost
    cross_similarity = min(1.0 - costs["nearFar"], 1.0 - costs["farNear"])
    reliability = min(before.near.reliability, before.far.reliability, after.near.reliability, after.far.reliability)
    separation = min(costs["beforeNearFar"], costs["afterNearFar"])
    cohesion = min(before.near.cohesion, before.far.cohesion, after.near.cohesion, after.far.cohesion)
    same_similarity = min(1.0 - costs["nearNear"], 1.0 - costs["farFar"])
    base_gate = min(reliability, separation, cohesion)
    swap_gate = min(base_gate, cross_similarity)
    continuity_gate = min(base_gate, same_similarity)
    values = {
        "twoModeJerseyTeamTransportSwapMargin": margin,
        "twoModeJerseyReliableSwapEvidence": max(margin, 0.0) * swap_gate,
        "twoModeJerseyReliableContinuityEvidence": max(-margin, 0.0) * continuity_gate,
        "twoModeJerseyCrossSideSimilarityMinimum": cross_similarity,
        "twoModeJerseySameSideSimilarityMinimum": same_similarity,
        "twoModeJerseyTeamReliabilityMinimum": reliability,
        "twoModeJerseyTeamSeparationMinimum": separation,
        "twoModeJerseyTeamCohesionMinimum": cohesion,
        "twoModeJerseyBaseReliabilityGate": base_gate,
        "twoModeJerseySwapReliabilityGate": swap_gate,
        "twoModeJerseyContinuityReliabilityGate": continuity_gate,
    }
    if tuple(values) != T8_FEATURE_NAMES or not all(math.isfinite(value) for value in values.values()):
        raise AssertionError("T8 team transport feature contract changed")
    return values, {
        "sameAssignmentCost": same_cost,
        "swappedAssignmentCost": swapped_cost,
        "teamCosts": costs,
    }
