"""Selective far-court detection using the pinned T3 jersey transport stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from analysis.side_switch_player_detector import (
    PersonDetectionResult,
    PlayerDetection,
    QuantizedPersonDetector,
)
from analysis.side_switch_t3_jersey_transport import (
    ENDPOINT_FRAME_FRACTIONS,
    T3_FEATURE_NAMES,
    TeamSummary,
    _selected_observations,
    build_tracklets,
    summarize_team,
    team_transport_features,
)
from analysis.side_switch_v6 import canonical_y


FAR_CANONICAL_TOP = 0.10
FAR_CANONICAL_BOTTOM = 0.62


def _t4_name(t3_name: str) -> str:
    if not t3_name.startswith("jersey"):
        raise ValueError(f"unexpected T3 jersey feature name: {t3_name}")
    return "selectiveFarJersey" + t3_name[len("jersey") :]


T4_FEATURE_NAMES = tuple(_t4_name(name) for name in T3_FEATURE_NAMES)
T4_CORE_FEATURE_NAMES = T4_FEATURE_NAMES[:3]


@dataclass(frozen=True)
class SelectiveFarEndpointSummary:
    near: TeamSummary
    far: TeamSummary
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
            "nearQualifyingTracklets": self.near.qualifying_tracklets,
            "farQualifyingTracklets": self.far.qualifying_tracklets,
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
                "canonicalTop": FAR_CANONICAL_TOP,
                "canonicalBottom": FAR_CANONICAL_BOTTOM,
            },
        }


def far_crop_bounds(frame_height: int, net_y_ratio: float) -> tuple[int, int]:
    if frame_height < 2 or not math.isfinite(net_y_ratio):
        raise ValueError("T4 far crop requires finite court geometry")
    net = float(np.clip(net_y_ratio, 0.20, 0.80))
    top_ratio = 2.0 * FAR_CANONICAL_TOP * net
    bottom_ratio = net + 2.0 * (FAR_CANONICAL_BOTTOM - 0.5) * (1.0 - net)
    top = max(0, min(frame_height - 1, round(top_ratio * frame_height)))
    bottom = max(top + 1, min(frame_height, round(bottom_ratio * frame_height)))
    return top, bottom


def map_far_detection(detection: PlayerDetection, top: int) -> PlayerDetection:
    offset = float(top)
    return PlayerDetection(
        x=detection.x,
        y=detection.y + offset,
        width=detection.width,
        height=detection.height,
        hip_x=detection.hip_x,
        hip_y=detection.hip_y + offset,
        shoulder_x=detection.shoulder_x,
        shoulder_y=detection.shoulder_y + offset,
        score=detection.score,
    )


def _near_detections(
    result: PersonDetectionResult, frame_height: int, net_y_ratio: float
) -> tuple[PlayerDetection, ...]:
    return tuple(
        value
        for value in result.detections
        if canonical_y(value.hip_y / frame_height, net_y_ratio) >= 0.56
    )


def _far_detections(
    result: PersonDetectionResult,
    top: int,
    frame_height: int,
    net_y_ratio: float,
) -> tuple[PlayerDetection, ...]:
    mapped = tuple(map_far_detection(value, top) for value in result.detections)
    return tuple(
        value
        for value in mapped
        if canonical_y(value.hip_y / frame_height, net_y_ratio) < 0.56
    )


def summarize_selective_far_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> SelectiveFarEndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T4 endpoint requires exactly five frames")
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("T4 endpoint frames must share one shape")
    frame_height = int(frames[0].shape[0])
    top, bottom = far_crop_bounds(frame_height, net_y_ratio)
    near_frames = []
    far_frames = []
    selected_counts = []
    full_raw_counts = []
    far_raw_counts = []
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
            _far_detections(
                far_result,
                top,
                frame_height,
                net_y_ratio,
            ),
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
    return SelectiveFarEndpointSummary(
        near=summarize_team(build_tracklets(near_frames)),
        far=summarize_team(build_tracklets(far_frames)),
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


def selective_far_transport_features(
    before: SelectiveFarEndpointSummary,
    after: SelectiveFarEndpointSummary,
) -> tuple[dict[str, float], dict[str, Any]]:
    t3_values, diagnostics = team_transport_features(before, after)
    values = {_t4_name(name): float(value) for name, value in t3_values.items()}
    if tuple(values) != T4_FEATURE_NAMES:
        raise AssertionError("T4 feature signature changed")
    return values, diagnostics
