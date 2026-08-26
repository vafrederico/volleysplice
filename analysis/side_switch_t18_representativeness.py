"""Medoid-to-team representativeness gated transport for side-switch T18."""

from __future__ import annotations

import math
from typing import Any, Mapping


T18_CORE_FEATURE_NAMES = ("representativeMedoidJerseySwapMargin",)
T18_DIAGNOSTIC_FEATURE_NAMES = (
    "representativeMedoidJerseyGateMinimum",
    "representativeMedoidJerseyDistanceMaximum",
)
T18_FEATURE_NAMES = (*T18_CORE_FEATURE_NAMES, *T18_DIAGNOSTIC_FEATURE_NAMES)


def _selection_representativeness(endpoint: Mapping[str, Any], side: str) -> float:
    kind = str(endpoint[f"{side}DominantKind"])
    distance = endpoint[f"{side}MedoidToPooledDistance"]
    if kind == "stable-medoid":
        if distance is None:
            raise ValueError("stable medoid requires pooled distance")
        return 1.0 - min(max(float(distance), 0.0), 1.0)
    if kind == "pooled-fallback":
        return 1.0
    if kind == "unavailable":
        return 0.0
    raise ValueError(f"unexpected T14 dominant kind: {kind}")


def representativeness_features(row: Mapping[str, Any]) -> dict[str, float]:
    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T18 features are adjacent-boundary-only")
    diagnostic = row["t14DominantTrackletMedoid"]
    representativeness = [
        _selection_representativeness(diagnostic[endpoint], side)
        for endpoint in ("before", "after")
        for side in ("near", "far")
    ]
    gate = min(representativeness)
    distances = [1.0 - value for value in representativeness]
    raw_margin = float(
        row["features"]["dominantTrackletJerseyTeamTransportSwapMargin"]
    )
    values = {
        "representativeMedoidJerseySwapMargin": raw_margin * gate,
        "representativeMedoidJerseyGateMinimum": gate,
        "representativeMedoidJerseyDistanceMaximum": max(distances),
    }
    if tuple(values) != T18_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T18 feature signature or values changed")
    return values
