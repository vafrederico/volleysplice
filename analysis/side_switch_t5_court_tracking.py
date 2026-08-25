"""Court-constrained temporal team tracking for side-switch T5."""

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


MINIMUM_TEAM_FRAMES = 2


def _t5_name(t3_name: str) -> str:
    if not t3_name.startswith("jersey"):
        raise ValueError(f"unexpected jersey feature name: {t3_name}")
    return "courtTrackedFarJersey" + t3_name[len("jersey") :]


T5_FEATURE_NAMES = tuple(_t5_name(name) for name in T3_FEATURE_NAMES)
T5_CORE_FEATURE_NAMES = T5_FEATURE_NAMES[:3]


@dataclass(frozen=True)
class CourtTeamTrack:
    team: TeamSummary
    observed_frames: int
    total_observations: int
    observations_by_frame: tuple[int, ...]
    frame_reliabilities: tuple[float, ...]


@dataclass(frozen=True)
class CourtTrackedEndpointSummary:
    near: TeamSummary
    far: TeamSummary
    near_track: CourtTeamTrack
    far_track: CourtTeamTrack
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
        return {
            "nearTeamAvailable": self.near.available,
            "farTeamAvailable": self.far.available,
            "nearTeamReliability": self.near.reliability,
            "farTeamReliability": self.far.reliability,
            "nearTeamCohesion": self.near.cohesion,
            "farTeamCohesion": self.far.cohesion,
            "nearObservedFrames": self.near_track.observed_frames,
            "farObservedFrames": self.far_track.observed_frames,
            "nearTotalObservations": self.near_track.total_observations,
            "farTotalObservations": self.far_track.total_observations,
            "nearObservationsByFrame": list(self.near_track.observations_by_frame),
            "farObservationsByFrame": list(self.far_track.observations_by_frame),
            "nearFrameReliabilities": list(self.near_track.frame_reliabilities),
            "farFrameReliabilities": list(self.far_track.frame_reliabilities),
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
                "heightFraction": (self.far_crop_bottom - self.far_crop_top)
                / self.frame_height,
            },
        }


def track_court_team(
    per_frame: Sequence[Sequence[JerseyObservation]],
) -> CourtTeamTrack:
    """Track one anonymous volleyball team on one court side across five frames."""

    if len(per_frame) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T5 expects exactly five ordered observation frames")
    frame_descriptors: list[np.ndarray] = []
    frame_reliabilities: list[float] = []
    counts: list[int] = []
    for observations in per_frame:
        values = tuple(observations)
        counts.append(len(values))
        if not values:
            frame_reliabilities.append(0.0)
            continue
        weights = np.asarray(
            [max(value.confidence * value.support, 1e-6) for value in values],
            dtype=np.float64,
        )
        frame_descriptors.append(
            _normalize(
                np.average(
                    np.stack([value.descriptor for value in values]),
                    axis=0,
                    weights=weights,
                )
            )
        )
        frame_reliabilities.append(
            float(
                np.mean([value.confidence for value in values])
                * math.sqrt(max(float(np.mean([value.support for value in values])), 0.0))
            )
        )
    observed_frames = len(frame_descriptors)
    if observed_frames < MINIMUM_TEAM_FRAMES:
        return CourtTeamTrack(
            team=TeamSummary(None, 0.0, 0.0, 0),
            observed_frames=observed_frames,
            total_observations=sum(counts),
            observations_by_frame=tuple(counts),
            frame_reliabilities=tuple(frame_reliabilities),
        )
    available_reliabilities = np.asarray(
        [value for value in frame_reliabilities if value > 0.0], dtype=np.float64
    )
    descriptor = _normalize(
        np.average(
            np.stack(frame_descriptors), axis=0, weights=available_reliabilities
        )
    )
    cohesion = 1.0 - float(
        np.average(
            [hellinger(value, descriptor) for value in frame_descriptors],
            weights=available_reliabilities,
        )
    )
    reliability = float(
        np.mean(available_reliabilities)
        * (0.40 + 0.60 * observed_frames / len(ENDPOINT_FRAME_FRACTIONS))
    )
    return CourtTeamTrack(
        team=TeamSummary(
            descriptor=descriptor,
            reliability=float(np.clip(reliability, 0.0, 1.0)),
            cohesion=float(np.clip(cohesion, 0.0, 1.0)),
            qualifying_tracklets=observed_frames,
        ),
        observed_frames=observed_frames,
        total_observations=sum(counts),
        observations_by_frame=tuple(counts),
        frame_reliabilities=tuple(frame_reliabilities),
    )


def summarize_court_tracked_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> CourtTrackedEndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T5 endpoint requires exactly five frames")
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("T5 endpoint frames must share one shape")
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
            frame,
            _near_detections(full_result, frame_height, net_y_ratio),
            net_y_ratio,
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
    near_track = track_court_team(near_frames)
    far_track = track_court_team(far_frames)
    return CourtTrackedEndpointSummary(
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


def court_tracked_transport_features(
    before: CourtTrackedEndpointSummary,
    after: CourtTrackedEndpointSummary,
) -> tuple[dict[str, float], dict[str, Any]]:
    t3_values, diagnostics = team_transport_features(before, after)  # type: ignore[arg-type]
    values = {_t5_name(name): float(value) for name, value in t3_values.items()}
    if tuple(values) != T5_FEATURE_NAMES:
        raise AssertionError("T5 feature signature changed")
    return values, diagnostics
