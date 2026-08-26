"""Source-resolved one-medoid assignment advantages for side-switch T16."""

from __future__ import annotations

import math
from typing import Any, Mapping


T16_CORE_FEATURE_NAMES = (
    "sourceResolvedMedoidJerseyNearSwapAdvantage",
    "sourceResolvedMedoidJerseyFarSwapAdvantage",
)


def source_resolved_features(row: Mapping[str, Any]) -> dict[str, float]:
    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T16 features are adjacent-boundary-only")
    source = row["t15BilateralMedoidConsensus"]["sourceAdvantages"]
    values = {
        T16_CORE_FEATURE_NAMES[0]: float(source["near"]),
        T16_CORE_FEATURE_NAMES[1]: float(source["far"]),
    }
    if tuple(values) != T16_CORE_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T16 feature signature or values changed")
    return values
