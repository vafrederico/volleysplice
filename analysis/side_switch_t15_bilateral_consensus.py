"""Bilateral one-medoid assignment consensus for side-switch T15."""

from __future__ import annotations

import math
from typing import Any, Mapping


T15_CORE_FEATURE_NAMES = (
    "bilateralMedoidJerseySwapEvidence",
    "bilateralMedoidJerseyContinuityEvidence",
)
T15_DIAGNOSTIC_FEATURE_NAMES = (
    "bilateralMedoidJerseyNearSourceAdvantage",
    "bilateralMedoidJerseyFarSourceAdvantage",
)
T15_FEATURE_NAMES = (*T15_CORE_FEATURE_NAMES, *T15_DIAGNOSTIC_FEATURE_NAMES)


def bilateral_consensus_features(
    row: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return evidence only when both T14 source sides agree on direction."""

    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T15 features are adjacent-boundary-only")
    reductions = row["t14DominantTrackletMedoid"]["teamReductions"]
    near_advantage = float(reductions["nearNear"]["cost"]) - float(
        reductions["nearFar"]["cost"]
    )
    far_advantage = float(reductions["farFar"]["cost"]) - float(
        reductions["farNear"]["cost"]
    )
    values = {
        "bilateralMedoidJerseySwapEvidence": max(
            min(near_advantage, far_advantage), 0.0
        ),
        "bilateralMedoidJerseyContinuityEvidence": max(
            min(-near_advantage, -far_advantage), 0.0
        ),
        "bilateralMedoidJerseyNearSourceAdvantage": near_advantage,
        "bilateralMedoidJerseyFarSourceAdvantage": far_advantage,
    }
    if tuple(values) != T15_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T15 feature signature or values changed")
    return values, {
        "sourceAdvantages": {
            "near": near_advantage,
            "far": far_advantage,
        },
        "agreement": (
            "swap"
            if min(near_advantage, far_advantage) > 0.0
            else "continuity"
            if max(near_advantage, far_advantage) < 0.0
            else "disagree-or-tie"
        ),
    }
