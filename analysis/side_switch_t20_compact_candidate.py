"""Compact T14 direction plus T19 disagreement candidate for side-switch T20."""

from __future__ import annotations

import math
from typing import Any, Mapping


T20_CORE_FEATURE_NAMES = (
    "compactMedoidJerseySwapMargin",
    "compactCrossRepresentationJerseyDisagreement",
)
T14_SOURCE_NAME = "dominantTrackletJerseyTeamTransportSwapMargin"
T19_SOURCE_NAME = "crossRepresentationJerseySwapDisagreement"


def compact_candidate_features(row: Mapping[str, Any]) -> dict[str, float]:
    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T20 features are adjacent-boundary-only")
    values = {
        T20_CORE_FEATURE_NAMES[0]: float(row["features"][T14_SOURCE_NAME]),
        T20_CORE_FEATURE_NAMES[1]: float(row["features"][T19_SOURCE_NAME]),
    }
    if tuple(values) != T20_CORE_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T20 feature signature or values changed")
    return values
