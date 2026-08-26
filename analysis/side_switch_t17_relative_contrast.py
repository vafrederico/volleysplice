"""Scale-free per-source medoid assignment contrast for side-switch T17."""

from __future__ import annotations

import math
from typing import Any, Mapping


T17_CORE_FEATURE_NAMES = (
    "relativeMedoidJerseyNearSwapContrast",
    "relativeMedoidJerseyFarSwapContrast",
)


def relative_contrast(same_cost: float, swapped_cost: float) -> float:
    denominator = same_cost + swapped_cost
    return (same_cost - swapped_cost) / denominator if denominator > 0.0 else 0.0


def relative_source_features(row: Mapping[str, Any]) -> dict[str, float]:
    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T17 features are adjacent-boundary-only")
    reductions = row["t14DominantTrackletMedoid"]["teamReductions"]
    values = {
        T17_CORE_FEATURE_NAMES[0]: relative_contrast(
            float(reductions["nearNear"]["cost"]),
            float(reductions["nearFar"]["cost"]),
        ),
        T17_CORE_FEATURE_NAMES[1]: relative_contrast(
            float(reductions["farFar"]["cost"]),
            float(reductions["farNear"]["cost"]),
        ),
    }
    if tuple(values) != T17_CORE_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T17 feature signature or values changed")
    return values
