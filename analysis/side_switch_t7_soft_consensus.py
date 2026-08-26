"""Soft robust jersey consensus for side-switch T7."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from analysis.side_switch_player_detector import QuantizedPersonDetector
from analysis.side_switch_t1_transport import hellinger
from analysis.side_switch_t3_jersey_transport import (
    ENDPOINT_FRAME_FRACTIONS,
    T3_FEATURE_NAMES,
    JerseyObservation,
    TeamSummary,
    _normalize,
    _selected_observations,
    team_transport_features,
)
from analysis.side_switch_t4_selective_far import (
    _far_detections,
    _near_detections,
    far_crop_bounds,
)
from analysis.side_switch_t6_dominant_jersey import CONSENSUS_RADIUS, track_dominant_jersey


def _t7_name(t3_name: str) -> str:
    if not t3_name.startswith("jersey"):
        raise ValueError(f"unexpected jersey feature name: {t3_name}")
    return "softConsensusJersey" + t3_name[len("jersey") :]


T7_FEATURE_NAMES = tuple(_t7_name(name) for name in T3_FEATURE_NAMES)
T7_CORE_FEATURE_NAMES = T7_FEATURE_NAMES[:3]


@dataclass(frozen=True)
class SoftConsensusTrack:
    team: TeamSummary
    observed_frames: int
    total_observations: int
    mean_robust_weight: float
    effective_robust_support: float
    below_half_weight: int
    below_quarter_weight: int
    medoid_frame: int | None
    medoid_index: int | None
    distances: tuple[float, ...]
    robust_weights: tuple[float, ...]


@dataclass(frozen=True)
class SoftConsensusEndpointSummary:
    near: TeamSummary
    far: TeamSummary
    near_track: SoftConsensusTrack
    far_track: SoftConsensusTrack
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
        def track(value: SoftConsensusTrack) -> dict[str, Any]:
            return {
                "teamAvailable": value.team.available,
                "teamReliability": value.team.reliability,
                "teamCohesion": value.team.cohesion,
                "observedFrames": value.observed_frames,
                "totalObservations": value.total_observations,
                "meanRobustWeight": value.mean_robust_weight,
                "effectiveRobustSupport": value.effective_robust_support,
                "belowHalfWeight": value.below_half_weight,
                "belowQuarterWeight": value.below_quarter_weight,
                "medoidFrame": value.medoid_frame,
                "medoidIndex": value.medoid_index,
                "distanceMinimum": min(value.distances) if value.distances else 0.0,
                "distanceMedian": float(np.median(value.distances)) if value.distances else 0.0,
                "distanceMaximum": max(value.distances) if value.distances else 0.0,
            }

        return {
            "near": track(self.near_track),
            "far": track(self.far_track),
            "selectedPlayersByFrame": list(self.selected_counts),
            "fullRawCandidatesByFrame": list(self.full_raw_candidate_counts),
            "farRawCandidatesByFrame": list(self.far_raw_candidate_counts),
            "backgroundMaskFallbacks": self.background_fallbacks,
            "fullDetectorInferenceMilliseconds": self.full_detector_inference_milliseconds,
            "farDetectorInferenceMilliseconds": self.far_detector_inference_milliseconds,
        }


def robust_weight(distance: float) -> float:
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("T7 robust distance must be finite and nonnegative")
    return 1.0 / (1.0 + (distance / CONSENSUS_RADIUS) ** 2)


def track_soft_consensus(
    per_frame: Sequence[Sequence[JerseyObservation]],
) -> SoftConsensusTrack:
    if len(per_frame) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T7 expects exactly five ordered observation frames")
    dominant = track_dominant_jersey(per_frame)
    flat = [
        (frame_index, observation_index, observation)
        for frame_index, observations in enumerate(per_frame)
        for observation_index, observation in enumerate(observations)
    ]
    if not flat:
        return SoftConsensusTrack(
            TeamSummary(None, 0.0, 0.0, 0), 0, 0, 0.0, 0.0, 0, 0,
            None, None, (), (),
        )
    if dominant.medoid_frame is None or dominant.medoid_index is None:
        raise AssertionError("T7 requires a deterministic medoid")
    medoid = per_frame[dominant.medoid_frame][dominant.medoid_index]
    distances = tuple(hellinger(medoid.descriptor, value.descriptor) for _, _, value in flat)
    robust = tuple(robust_weight(value) for value in distances)
    base_weights = tuple(max(value.confidence * value.support, 1e-6) for _, _, value in flat)
    effective_support = float(
        sum(weight * local for weight, local in zip(base_weights, robust, strict=True))
        / sum(base_weights)
    )
    frame_descriptors: list[np.ndarray] = []
    frame_reliabilities: list[float] = []
    for observations in per_frame:
        values = tuple(observations)
        if not values:
            continue
        local_distances = [hellinger(medoid.descriptor, value.descriptor) for value in values]
        local_robust = [robust_weight(value) for value in local_distances]
        weights = np.asarray(
            [
                max(value.confidence * value.support * local, 1e-6)
                for value, local in zip(values, local_robust, strict=True)
            ],
            dtype=np.float64,
        )
        frame_descriptors.append(
            _normalize(np.average(np.stack([value.descriptor for value in values]), axis=0, weights=weights))
        )
        t5_reliability = float(
            np.mean([value.confidence for value in values])
            * math.sqrt(max(float(np.mean([value.support for value in values])), 0.0))
        )
        frame_reliabilities.append(t5_reliability * math.sqrt(float(np.mean(local_robust))))
    observed_frames = len(frame_descriptors)
    if observed_frames < 2:
        team = TeamSummary(None, 0.0, 0.0, 0)
    else:
        reliability_weights = np.asarray(frame_reliabilities, dtype=np.float64)
        descriptor = _normalize(
            np.average(np.stack(frame_descriptors), axis=0, weights=reliability_weights)
        )
        cohesion = 1.0 - float(
            np.average(
                [hellinger(value, descriptor) for value in frame_descriptors],
                weights=reliability_weights,
            )
        )
        reliability = float(
            np.mean(reliability_weights)
            * (0.40 + 0.60 * observed_frames / len(ENDPOINT_FRAME_FRACTIONS))
        )
        team = TeamSummary(
            descriptor,
            float(np.clip(reliability, 0.0, 1.0)),
            float(np.clip(cohesion, 0.0, 1.0)),
            observed_frames,
        )
    return SoftConsensusTrack(
        team=team,
        observed_frames=observed_frames,
        total_observations=len(flat),
        mean_robust_weight=float(np.mean(robust)),
        effective_robust_support=effective_support,
        below_half_weight=sum(value < 0.5 for value in robust),
        below_quarter_weight=sum(value < 0.25 for value in robust),
        medoid_frame=dominant.medoid_frame,
        medoid_index=dominant.medoid_index,
        distances=distances,
        robust_weights=robust,
    )


def summarize_soft_consensus_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> SoftConsensusEndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T7 endpoint requires exactly five frames")
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
    near_track = track_soft_consensus(near_frames)
    far_track = track_soft_consensus(far_frames)
    return SoftConsensusEndpointSummary(
        near_track.team, far_track.team, near_track, far_track,
        tuple(selected_counts), tuple(full_raw_counts), tuple(far_raw_counts),
        fallbacks, float(full_inference), float(far_inference), top, bottom, frame_height,
    )


def soft_consensus_transport_features(
    before: SoftConsensusEndpointSummary,
    after: SoftConsensusEndpointSummary,
) -> tuple[dict[str, float], dict[str, Any]]:
    t3_values, diagnostics = team_transport_features(before, after)  # type: ignore[arg-type]
    values = {_t7_name(name): float(value) for name, value in t3_values.items()}
    if tuple(values) != T7_FEATURE_NAMES:
        raise AssertionError("T7 feature signature changed")
    return values, diagnostics
