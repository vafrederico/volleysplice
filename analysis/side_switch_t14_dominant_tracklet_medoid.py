"""Dominant stable-tracklet jersey medoid transport for side-switch T14."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from analysis.side_switch_player_detector import QuantizedPersonDetector
from analysis.side_switch_t1_transport import hellinger
from analysis.side_switch_t3_jersey_transport import JerseyTracklet
from analysis.side_switch_t11_fallback_tracklet_transport import (
    FallbackAppearanceUnit,
    FallbackTrackletEndpointSummary,
    summarize_fallback_tracklet_endpoint,
)
from analysis.side_switch_t10_tracklet_unit_transport import (
    TrackletUnitEndpointSummary,
    tracklet_unit_transport_features,
)


T14_CORE_FEATURE_NAMES = (
    "dominantTrackletJerseyTeamTransportSwapMargin",
    "dominantTrackletJerseyReliableSwapEvidence",
    "dominantTrackletJerseyReliableContinuityEvidence",
)
T14_DIAGNOSTIC_FEATURE_NAMES = (
    "dominantTrackletJerseyConditionalCrossSimilarityMinimum",
    "dominantTrackletJerseyCrossMatchCoverageMinimum",
    "dominantTrackletJerseyConditionalSameSimilarityMinimum",
    "dominantTrackletJerseySameMatchCoverageMinimum",
    "dominantTrackletJerseyTeamReliabilityMinimum",
    "dominantTrackletJerseyTeamSeparationMinimum",
    "dominantTrackletJerseyBaseReliabilityGate",
    "dominantTrackletJerseySwapReliabilityGate",
    "dominantTrackletJerseyContinuityReliabilityGate",
)
T14_FEATURE_NAMES = (*T14_CORE_FEATURE_NAMES, *T14_DIAGNOSTIC_FEATURE_NAMES)


@dataclass(frozen=True)
class DominantSelection:
    unit: JerseyTracklet | FallbackAppearanceUnit | None
    kind: str
    medoid_index: int | None
    stable_tracklets: int
    medoid_to_pool_distance: float | None


@dataclass(frozen=True)
class DominantTrackletEndpointSummary:
    base: FallbackTrackletEndpointSummary
    near_selection: DominantSelection
    far_selection: DominantSelection

    @property
    def near(self) -> tuple[JerseyTracklet | FallbackAppearanceUnit, ...]:
        return (self.near_selection.unit,) if self.near_selection.unit is not None else ()

    @property
    def far(self) -> tuple[JerseyTracklet | FallbackAppearanceUnit, ...]:
        return (self.far_selection.unit,) if self.far_selection.unit is not None else ()

    def as_t10(self) -> TrackletUnitEndpointSummary:
        return TrackletUnitEndpointSummary(
            near=self.near,  # type: ignore[arg-type]
            far=self.far,  # type: ignore[arg-type]
            near_team=self.base.near_pool.team,
            far_team=self.base.far_pool.team,
            selected_counts=self.base.selected_counts,
            full_raw_candidate_counts=self.base.full_raw_candidate_counts,
            far_raw_candidate_counts=self.base.far_raw_candidate_counts,
            background_fallbacks=self.base.background_fallbacks,
            full_detector_inference_milliseconds=self.base.full_detector_inference_milliseconds,
            far_detector_inference_milliseconds=self.base.far_detector_inference_milliseconds,
            far_crop_top=self.base.far_crop_top,
            far_crop_bottom=self.base.far_crop_bottom,
            frame_height=self.base.frame_height,
        )

    def to_diagnostic(self) -> dict[str, Any]:
        result = self.base.to_diagnostic()
        for side, selection in (
            ("near", self.near_selection),
            ("far", self.far_selection),
        ):
            result[f"{side}DominantKind"] = selection.kind
            result[f"{side}DominantMedoidIndex"] = selection.medoid_index
            result[f"{side}StableTracklets"] = selection.stable_tracklets
            result[f"{side}MedoidToPooledDistance"] = selection.medoid_to_pool_distance
        return result


def dominant_selection(
    stable: Sequence[JerseyTracklet],
    fallback_units: Sequence[JerseyTracklet | FallbackAppearanceUnit],
    pooled_descriptor: np.ndarray | None,
) -> DominantSelection:
    values = tuple(stable)
    if values:
        objectives = [
            sum(hellinger(candidate.descriptor, other.descriptor) for other in values)
            for candidate in values
        ]
        index = min(range(len(values)), key=lambda value: (objectives[value], value))
        unit = values[index]
        distance = (
            hellinger(unit.descriptor, pooled_descriptor)
            if pooled_descriptor is not None
            else None
        )
        return DominantSelection(unit, "stable-medoid", index, len(values), distance)
    fallback = tuple(fallback_units)
    if fallback:
        if len(fallback) != 1 or not isinstance(fallback[0], FallbackAppearanceUnit):
            raise ValueError("T14 empty stable side expects exactly one typed fallback")
        return DominantSelection(fallback[0], "pooled-fallback", None, 0, None)
    return DominantSelection(None, "unavailable", None, 0, None)


def summarize_dominant_tracklet_endpoint(
    frames: Sequence[np.ndarray],
    net_y_ratio: float,
    detector: QuantizedPersonDetector,
) -> DominantTrackletEndpointSummary:
    base = summarize_fallback_tracklet_endpoint(frames, net_y_ratio, detector)
    return DominantTrackletEndpointSummary(
        base=base,
        near_selection=dominant_selection(
            base.near_stable,
            base.near,
            base.near_pool.team.descriptor,
        ),
        far_selection=dominant_selection(
            base.far_stable,
            base.far,
            base.far_pool.team.descriptor,
        ),
    )


def dominant_tracklet_transport_features(
    before: DominantTrackletEndpointSummary,
    after: DominantTrackletEndpointSummary,
) -> tuple[dict[str, float], dict[str, Any]]:
    base, diagnostics = tracklet_unit_transport_features(before.as_t10(), after.as_t10())
    values = {
        "dominantTrackletJerseyTeamTransportSwapMargin": base[
            "trackletUnitJerseyTeamTransportSwapMargin"
        ],
        "dominantTrackletJerseyReliableSwapEvidence": base[
            "trackletUnitJerseyReliableSwapEvidence"
        ],
        "dominantTrackletJerseyReliableContinuityEvidence": base[
            "trackletUnitJerseyReliableContinuityEvidence"
        ],
        "dominantTrackletJerseyConditionalCrossSimilarityMinimum": base[
            "trackletUnitJerseyConditionalCrossSimilarityMinimum"
        ],
        "dominantTrackletJerseyCrossMatchCoverageMinimum": base[
            "trackletUnitJerseyCrossMatchCoverageMinimum"
        ],
        "dominantTrackletJerseyConditionalSameSimilarityMinimum": base[
            "trackletUnitJerseyConditionalSameSimilarityMinimum"
        ],
        "dominantTrackletJerseySameMatchCoverageMinimum": base[
            "trackletUnitJerseySameMatchCoverageMinimum"
        ],
        "dominantTrackletJerseyTeamReliabilityMinimum": base[
            "trackletUnitJerseyTeamReliabilityMinimum"
        ],
        "dominantTrackletJerseyTeamSeparationMinimum": base[
            "trackletUnitJerseyTeamSeparationMinimum"
        ],
        "dominantTrackletJerseyBaseReliabilityGate": base[
            "trackletUnitJerseyBaseReliabilityGate"
        ],
        "dominantTrackletJerseySwapReliabilityGate": base[
            "trackletUnitJerseySwapReliabilityGate"
        ],
        "dominantTrackletJerseyContinuityReliabilityGate": base[
            "trackletUnitJerseyContinuityReliabilityGate"
        ],
    }
    if tuple(values) != T14_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T14 feature signature or values changed")
    return values, diagnostics
