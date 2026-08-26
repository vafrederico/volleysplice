"""Empty-side pooled fallback over stable tracklet units for side-switch T11."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from analysis.side_switch_player_detector import QuantizedPersonDetector
from analysis.side_switch_t3_jersey_transport import (
    ENDPOINT_FRAME_FRACTIONS,
    JerseyObservation,
    JerseyTracklet,
    TeamSummary,
    _selected_observations,
    build_tracklets,
)
from analysis.side_switch_t4_selective_far import (
    _far_detections,
    _near_detections,
    far_crop_bounds,
)
from analysis.side_switch_t5_court_tracking import CourtTeamTrack, track_court_team
from analysis.side_switch_t10_tracklet_unit_transport import (
    T10_CORE_FEATURE_NAMES,
    T10_FEATURE_NAMES,
    TrackletUnitEndpointSummary,
    tracklet_unit_transport_features,
)


def _t11_name(name: str) -> str:
    prefix = "trackletUnitJersey"
    if not name.startswith(prefix):
        raise ValueError(f"unexpected T10 feature name: {name}")
    return "fallbackTrackletUnitJersey" + name[len(prefix) :]


T11_FEATURE_NAMES = tuple(_t11_name(name) for name in T10_FEATURE_NAMES)
T11_CORE_FEATURE_NAMES = tuple(_t11_name(name) for name in T10_CORE_FEATURE_NAMES)


@dataclass(frozen=True)
class FallbackAppearanceUnit:
    descriptor: np.ndarray
    reliability: float
    observed_frames: int
    x: float
    y: float
    scale: float
    kind: str


@dataclass(frozen=True)
class FallbackTrackletEndpointSummary:
    near: tuple[JerseyTracklet | FallbackAppearanceUnit, ...]
    far: tuple[JerseyTracklet | FallbackAppearanceUnit, ...]
    near_stable: tuple[JerseyTracklet, ...]
    far_stable: tuple[JerseyTracklet, ...]
    near_pool: CourtTeamTrack
    far_pool: CourtTeamTrack
    selected_counts: tuple[int, ...]
    full_raw_candidate_counts: tuple[int, ...]
    far_raw_candidate_counts: tuple[int, ...]
    background_fallbacks: int
    full_detector_inference_milliseconds: float
    far_detector_inference_milliseconds: float
    far_crop_top: int
    far_crop_bottom: int
    frame_height: int

    @property
    def near_fallback_used(self) -> bool:
        return not self.near_stable and bool(self.near)

    @property
    def far_fallback_used(self) -> bool:
        return not self.far_stable and bool(self.far)

    def as_t10(self) -> TrackletUnitEndpointSummary:
        return TrackletUnitEndpointSummary(
            near=self.near,  # type: ignore[arg-type]
            far=self.far,  # type: ignore[arg-type]
            near_team=self.near_pool.team,
            far_team=self.far_pool.team,
            selected_counts=self.selected_counts,
            full_raw_candidate_counts=self.full_raw_candidate_counts,
            far_raw_candidate_counts=self.far_raw_candidate_counts,
            background_fallbacks=self.background_fallbacks,
            full_detector_inference_milliseconds=self.full_detector_inference_milliseconds,
            far_detector_inference_milliseconds=self.far_detector_inference_milliseconds,
            far_crop_top=self.far_crop_top,
            far_crop_bottom=self.far_crop_bottom,
            frame_height=self.frame_height,
        )

    def to_diagnostic(self) -> dict[str, Any]:
        base = self.as_t10().to_diagnostic()
        base.update(
            {
                "nearStableTracklets": len(self.near_stable),
                "farStableTracklets": len(self.far_stable),
                "nearFallbackUsed": self.near_fallback_used,
                "farFallbackUsed": self.far_fallback_used,
                "nearUnitKinds": [
                    value.kind if isinstance(value, FallbackAppearanceUnit) else "stable-tracklet"
                    for value in self.near
                ],
                "farUnitKinds": [
                    value.kind if isinstance(value, FallbackAppearanceUnit) else "stable-tracklet"
                    for value in self.far
                ],
                "nearPooledTeamAvailable": self.near_pool.team.available,
                "farPooledTeamAvailable": self.far_pool.team.available,
                "nearPooledObservedFrames": self.near_pool.observed_frames,
                "farPooledObservedFrames": self.far_pool.observed_frames,
            }
        )
        return base


def _fallback_unit(team: TeamSummary, observed_frames: int) -> FallbackAppearanceUnit:
    if not team.available or team.descriptor is None:
        raise ValueError("T11 fallback requires an available pooled team")
    return FallbackAppearanceUnit(
        descriptor=team.descriptor,
        reliability=team.reliability,
        observed_frames=observed_frames,
        x=0.0,
        y=0.0,
        scale=0.0,
        kind="pooled-fallback",
    )


def side_units(
    per_frame: Sequence[Sequence[JerseyObservation]],
) -> tuple[
    tuple[JerseyTracklet | FallbackAppearanceUnit, ...],
    tuple[JerseyTracklet, ...],
    CourtTeamTrack,
]:
    stable = build_tracklets(per_frame)
    pool = track_court_team(per_frame)
    if stable:
        units: tuple[JerseyTracklet | FallbackAppearanceUnit, ...] = stable
    elif pool.team.available:
        units = (_fallback_unit(pool.team, pool.observed_frames),)
    else:
        units = ()
    return units, stable, pool


def summarize_fallback_tracklet_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> FallbackTrackletEndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T11 endpoint requires exactly five frames")
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("T11 endpoint frames must share one shape")
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
    near_units, near_stable, near_pool = side_units(near_frames)
    far_units, far_stable, far_pool = side_units(far_frames)
    return FallbackTrackletEndpointSummary(
        near=near_units,
        far=far_units,
        near_stable=near_stable,
        far_stable=far_stable,
        near_pool=near_pool,
        far_pool=far_pool,
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


def fallback_tracklet_transport_features(
    before: FallbackTrackletEndpointSummary,
    after: FallbackTrackletEndpointSummary,
) -> tuple[dict[str, float], dict[str, Any]]:
    t10_values, diagnostics = tracklet_unit_transport_features(
        before.as_t10(), after.as_t10()
    )
    values = {_t11_name(name): float(value) for name, value in t10_values.items()}
    if tuple(values) != T11_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T11 feature signature or values changed")
    return values, diagnostics
