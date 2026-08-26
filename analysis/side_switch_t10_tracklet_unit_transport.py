"""Equal-unit selective-far player-tracklet transport for side-switch T10."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from analysis.side_switch_player_detector import QuantizedPersonDetector
from analysis.side_switch_t1_transport import hellinger
from analysis.side_switch_t3_jersey_transport import (
    ENDPOINT_FRAME_FRACTIONS,
    JerseyObservation,
    JerseyTracklet,
    TeamSummary,
    _minimum_assignment,
    _selected_observations,
    build_tracklets,
    summarize_team,
)
from analysis.side_switch_t4_selective_far import (
    _far_detections,
    _near_detections,
    far_crop_bounds,
)


T10_CORE_FEATURE_NAMES = (
    "trackletUnitJerseyTeamTransportSwapMargin",
    "trackletUnitJerseyConditionalCrossSimilarityMinimum",
    "trackletUnitJerseyCrossMatchCoverageMinimum",
    "trackletUnitJerseyReliableSwapEvidence",
    "trackletUnitJerseyReliableContinuityEvidence",
)
T10_DIAGNOSTIC_FEATURE_NAMES = (
    "trackletUnitJerseyConditionalSameSimilarityMinimum",
    "trackletUnitJerseySameMatchCoverageMinimum",
    "trackletUnitJerseyTeamReliabilityMinimum",
    "trackletUnitJerseyTeamSeparationMinimum",
    "trackletUnitJerseyBaseReliabilityGate",
    "trackletUnitJerseySwapReliabilityGate",
    "trackletUnitJerseyContinuityReliabilityGate",
)
T10_FEATURE_NAMES = (*T10_CORE_FEATURE_NAMES, *T10_DIAGNOSTIC_FEATURE_NAMES)


@dataclass(frozen=True)
class UnitMatchReduction:
    cost: float
    coverage: float
    conditional_similarity: float
    matched_pairs: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "cost": self.cost,
            "coverage": self.coverage,
            "conditionalSimilarity": self.conditional_similarity,
            "matchedPairs": self.matched_pairs,
        }


@dataclass(frozen=True)
class TrackletUnitEndpointSummary:
    near: tuple[JerseyTracklet, ...]
    far: tuple[JerseyTracklet, ...]
    near_team: TeamSummary
    far_team: TeamSummary
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
        def tracklets(values: Sequence[JerseyTracklet]) -> list[dict[str, float | int]]:
            return [
                {
                    "reliability": value.reliability,
                    "observedFrames": value.observed_frames,
                    "x": value.x,
                    "y": value.y,
                    "scale": value.scale,
                }
                for value in values
            ]

        return {
            "nearTeamAvailable": bool(self.near),
            "farTeamAvailable": bool(self.far),
            "nearQualifyingTracklets": len(self.near),
            "farQualifyingTracklets": len(self.far),
            "nearTeamReliability": self.near_team.reliability,
            "farTeamReliability": self.far_team.reliability,
            "nearTracklets": tracklets(self.near),
            "farTracklets": tracklets(self.far),
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


def unit_match_reduction(
    left: Sequence[JerseyTracklet], right: Sequence[JerseyTracklet]
) -> UnitMatchReduction:
    first = tuple(left)
    second = tuple(right)
    if not first or not second:
        return UnitMatchReduction(1.0, 0.0, 0.0, 0)
    costs = np.asarray(
        [
            [hellinger(left_value.descriptor, right_value.descriptor) for right_value in second]
            for left_value in first
        ],
        dtype=np.float64,
    )
    left_indexes, right_indexes = _minimum_assignment(costs)
    matched_cost = float(
        np.mean(
            [
                costs[left_index, right_index]
                for left_index, right_index in zip(
                    left_indexes, right_indexes, strict=True
                )
            ]
        )
    )
    coverage = min(len(first), len(second)) / max(len(first), len(second))
    conditional_similarity = float(np.clip(1.0 - matched_cost, 0.0, 1.0))
    cost = float(np.clip(1.0 - coverage * conditional_similarity, 0.0, 1.0))
    return UnitMatchReduction(
        cost=cost,
        coverage=float(coverage),
        conditional_similarity=conditional_similarity,
        matched_pairs=len(left_indexes),
    )


def _mean_reliability(values: Sequence[JerseyTracklet]) -> float:
    return float(np.mean([value.reliability for value in values])) if values else 0.0


def summarize_tracklet_unit_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> TrackletUnitEndpointSummary:
    if len(frames) != len(ENDPOINT_FRAME_FRACTIONS):
        raise ValueError("T10 endpoint requires exactly five frames")
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("T10 endpoint frames must share one shape")
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
    near = build_tracklets(near_frames)
    far = build_tracklets(far_frames)
    return TrackletUnitEndpointSummary(
        near=near,
        far=far,
        near_team=summarize_team(near),
        far_team=summarize_team(far),
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


def tracklet_unit_transport_features(
    before: TrackletUnitEndpointSummary,
    after: TrackletUnitEndpointSummary,
) -> tuple[dict[str, float], dict[str, Any]]:
    reductions = {
        "nearNear": unit_match_reduction(before.near, after.near),
        "farFar": unit_match_reduction(before.far, after.far),
        "nearFar": unit_match_reduction(before.near, after.far),
        "farNear": unit_match_reduction(before.far, after.near),
        "beforeNearFar": unit_match_reduction(before.near, before.far),
        "afterNearFar": unit_match_reduction(after.near, after.far),
    }
    same_cost = 0.5 * (reductions["nearNear"].cost + reductions["farFar"].cost)
    swapped_cost = 0.5 * (reductions["nearFar"].cost + reductions["farNear"].cost)
    margin = same_cost - swapped_cost
    cross_similarity = min(
        reductions["nearFar"].conditional_similarity,
        reductions["farNear"].conditional_similarity,
    )
    cross_coverage = min(
        reductions["nearFar"].coverage, reductions["farNear"].coverage
    )
    same_similarity = min(
        reductions["nearNear"].conditional_similarity,
        reductions["farFar"].conditional_similarity,
    )
    same_coverage = min(
        reductions["nearNear"].coverage, reductions["farFar"].coverage
    )
    reliability = min(
        _mean_reliability(before.near),
        _mean_reliability(before.far),
        _mean_reliability(after.near),
        _mean_reliability(after.far),
    )
    separation = min(
        reductions["beforeNearFar"].cost, reductions["afterNearFar"].cost
    )
    base_gate = min(reliability, separation)
    swap_gate = min(base_gate, cross_similarity, cross_coverage)
    continuity_gate = min(base_gate, same_similarity, same_coverage)
    values = {
        "trackletUnitJerseyTeamTransportSwapMargin": margin,
        "trackletUnitJerseyConditionalCrossSimilarityMinimum": cross_similarity,
        "trackletUnitJerseyCrossMatchCoverageMinimum": cross_coverage,
        "trackletUnitJerseyReliableSwapEvidence": max(margin, 0.0) * swap_gate,
        "trackletUnitJerseyReliableContinuityEvidence": max(-margin, 0.0)
        * continuity_gate,
        "trackletUnitJerseyConditionalSameSimilarityMinimum": same_similarity,
        "trackletUnitJerseySameMatchCoverageMinimum": same_coverage,
        "trackletUnitJerseyTeamReliabilityMinimum": reliability,
        "trackletUnitJerseyTeamSeparationMinimum": separation,
        "trackletUnitJerseyBaseReliabilityGate": base_gate,
        "trackletUnitJerseySwapReliabilityGate": swap_gate,
        "trackletUnitJerseyContinuityReliabilityGate": continuity_gate,
    }
    if tuple(values) != T10_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T10 feature signature or values changed")
    return values, {
        "sameAssignmentCost": same_cost,
        "swappedAssignmentCost": swapped_cost,
        "teamReductions": {
            name: value.to_dict() for name, value in reductions.items()
        },
    }
