"""T4/T14 jersey-transport consensus and disagreement for side-switch T19."""

from __future__ import annotations

import math
from typing import Any, Mapping


T19_CORE_FEATURE_NAMES = (
    "crossRepresentationJerseySwapConsensus",
    "crossRepresentationJerseySwapDisagreement",
)
T4_MARGIN = "selectiveFarJerseyTeamTransportSwapMargin"
T14_MARGIN = "dominantTrackletJerseyTeamTransportSwapMargin"


def cross_representation_features(row: Mapping[str, Any]) -> dict[str, float]:
    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T19 features are adjacent-boundary-only")
    t4_margin = float(row["features"][T4_MARGIN])
    t14_margin = float(row["features"][T14_MARGIN])
    values = {
        T19_CORE_FEATURE_NAMES[0]: 0.5 * (t4_margin + t14_margin),
        T19_CORE_FEATURE_NAMES[1]: abs(t4_margin - t14_margin),
    }
    if tuple(values) != T19_CORE_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T19 feature signature or values changed")
    return values
