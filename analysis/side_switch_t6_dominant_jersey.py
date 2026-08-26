"""Robust dominant-jersey temporal consensus for side-switch T6."""

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
    _selected_observations,
    team_transport_features,
)
from analysis.side_switch_t4_selective_far import (
    _far_detections,
    _near_detections,
    far_crop_bounds,
)
from analysis.side_switch_t5_court_tracking import CourtTeamTrack, track_court_team


CONSENSUS_RADIUS = 0.38


def _t6_name(t3_name: str) -> str:
    if not t3_name.startswith("jersey"):
        raise ValueError(f"unexpected jersey feature name: {t3_name}")
    return "dominantJersey" + t3_name[len("jersey") :]


T6_FEATURE_NAMES = tuple(_t6_name(name) for name in T3_FEATURE_NAMES)
T6_CORE_FEATURE_NAMES = T6_FEATURE_NAMES[:3]


@dataclass(frozen=True)
class DominantJerseyTrack:
    team: TeamSummary
    observed_frames: int
    total_observations: int
    mode_observations: int
    rejected_observations: int
    mode_support_fraction: float
    medoid_frame: int | None
    medoid_index: int | None
    medoid_weighted_mean_distance: float
    base_track: CourtTeamTrack


@dataclass(frozen=True)
class DominantJerseyEndpointSummary:
    near: TeamSummary
    far: TeamSummary
    near_track: DominantJerseyTrack
    far_track: DominantJerseyTrack
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
        def track(value: DominantJerseyTrack) -> dict[str, Any]:
            return {
                "teamAvailable": value.team.available,
                "teamReliability": value.team.reliability,
                "teamCohesion": value.team.cohesion,
                "observedFrames": value.observed_frames,
                "totalObservations": value.total_observations,
                "modeObservations": value.mode_observations,
                "rejectedObservations": value.rejected_observations,
                "modeSupportFraction": value.mode_support_fraction,
                "medoidFrame": value.medoid_frame,
                "medoidIndex": value.medoid_index,
                "medoidWeightedMeanDistance": value.medoid_weighted_mean_distance,
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
            "farCrop": {
                "top": self.far_crop_top,
                "bottom": self.far_crop_bottom,
                "frameHeight": self.frame_height,
            },
        }


def track_dominant_jersey(
    per_frame: Sequence[Sequence[JerseyObservation]],
) -> DominantJerseyTrack:
    if len(per_frame) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T6 expects exactly five ordered observation frames")
    flat = [
        (frame_index, observation_index, observation)
        for frame_index, observations in enumerate(per_frame)
        for observation_index, observation in enumerate(observations)
    ]
    empty = track_court_team(tuple(() for _ in ENDPOINT_FRAME_FRACTIONS))
    if not flat:
        return DominantJerseyTrack(empty.team, 0, 0, 0, 0, 0.0, None, None, 0.0, empty)
    weights = [max(value.confidence * value.support, 1e-6) for _, _, value in flat]
    total_weight = float(sum(weights))
    best: tuple[tuple[float, ...], int, list[int], float] | None = None
    for seed_index, (seed_frame, seed_observation, seed) in enumerate(flat):
        members = [
            index
            for index, (_, _, value) in enumerate(flat)
            if hellinger(seed.descriptor, value.descriptor) <= CONSENSUS_RADIUS
        ]
        represented_frames = len({flat[index][0] for index in members})
        member_weight = float(sum(weights[index] for index in members))
        mean_distance = float(
            sum(
                weights[index] * hellinger(seed.descriptor, flat[index][2].descriptor)
                for index in members
            )
            / max(member_weight, 1e-12)
        )
        key = (
            float(represented_frames),
            member_weight,
            -mean_distance,
            -float(seed_frame),
            -float(seed_observation),
        )
        if best is None or key > best[0]:
            best = (key, seed_index, members, mean_distance)
    if best is None:
        raise AssertionError("T6 medoid selection failed")
    _, seed_index, members, mean_distance = best
    selected_by_frame: list[list[JerseyObservation]] = [
        [] for _ in ENDPOINT_FRAME_FRACTIONS
    ]
    for index in members:
        frame_index, _, observation = flat[index]
        selected_by_frame[frame_index].append(observation)
    base = track_court_team(selected_by_frame)
    support_fraction = float(sum(weights[index] for index in members) / total_weight)
    if base.team.available:
        team = TeamSummary(
            descriptor=base.team.descriptor,
            reliability=base.team.reliability * math.sqrt(support_fraction),
            cohesion=base.team.cohesion,
            qualifying_tracklets=base.team.qualifying_tracklets,
        )
    else:
        team = base.team
    seed_frame, seed_observation, _ = flat[seed_index]
    return DominantJerseyTrack(
        team=team,
        observed_frames=base.observed_frames,
        total_observations=len(flat),
        mode_observations=len(members),
        rejected_observations=len(flat) - len(members),
        mode_support_fraction=support_fraction,
        medoid_frame=seed_frame,
        medoid_index=seed_observation,
        medoid_weighted_mean_distance=mean_distance,
        base_track=base,
    )


def summarize_dominant_jersey_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> DominantJerseyEndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T6 endpoint requires exactly five frames")
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("T6 endpoint frames must share one shape")
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
            frame,
            _far_detections(far_result, top, frame_height, net_y_ratio),
            net_y_ratio,
        )
        near_frames.append(near)
        far_frames.append(far)
        selected_counts.append(len(near) + len(far))
        full_raw_counts.append(full_result.raw_candidates)
        far_raw_counts.append(far_result.raw_candidates)
        fallbacks += near_fallbacks + far_fallbacks
        full_inference += full_result.inference_milliseconds
        far_inference += far_result.inference_milliseconds
    near_track = track_dominant_jersey(near_frames)
    far_track = track_dominant_jersey(far_frames)
    return DominantJerseyEndpointSummary(
        near=near_track.team,
        far=far_track.team,
        near_track=near_track,
        far_track=far_track,
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


def dominant_jersey_transport_features(
    before: DominantJerseyEndpointSummary,
    after: DominantJerseyEndpointSummary,
) -> tuple[dict[str, float], dict[str, Any]]:
    t3_values, diagnostics = team_transport_features(before, after)  # type: ignore[arg-type]
    values = {_t6_name(name): float(value) for name, value in t3_values.items()}
    if tuple(values) != T6_FEATURE_NAMES:
        raise AssertionError("T6 feature signature changed")
    return values, diagnostics
