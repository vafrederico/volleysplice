"""Coverage-conditional identity similarity for side-switch T2."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


CONDITIONAL_FEATURE_NAME = "conditionalCrossSideIdentitySimilarityMinimum"
T2_CORE_FEATURE_NAMES = (
    "appearanceTransportSwapMargin",
    CONDITIONAL_FEATURE_NAME,
)


def directional_conditional_similarity(reduction: Mapping[str, Any]) -> float:
    coverage = float(reduction["coverage"])
    mass = float(reduction["matchedIdentityMass"])
    if (
        not math.isfinite(coverage)
        or not math.isfinite(mass)
        or not 0.0 <= coverage <= 1.0
        or not 0.0 <= mass <= 1.0
    ):
        raise ValueError("T2 directional reduction must be finite in [0,1]")
    if coverage == 0.0:
        if mass != 0.0:
            raise ValueError("T2 zero coverage must have zero matched identity mass")
        return 0.0
    value = mass / coverage
    if not math.isfinite(value) or not -1e-12 <= value <= 1.0 + 1e-12:
        raise ValueError("T2 conditional similarity escaped [0,1]")
    return float(np.clip(value, 0.0, 1.0))


def t2_features(row: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T2 features are adjacent-boundary-only")
    t1 = row.get("t1Transport")
    if not isinstance(t1, Mapping) or str(t1.get("status")) != "ok":
        raise ValueError("T2 requires a complete T1 boundary diagnostic")
    reductions = t1["reductions"]
    near_far = directional_conditional_similarity(reductions["nearFar"])
    far_near = directional_conditional_similarity(reductions["farNear"])
    values = {
        "appearanceTransportSwapMargin": float(
            row["features"]["appearanceTransportSwapMargin"]
        ),
        CONDITIONAL_FEATURE_NAME: min(near_far, far_near),
    }
    if tuple(values) != T2_CORE_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T2 feature signature or values changed")
    return values, {
        "nearBeforeToFarAfterConditionalSimilarity": near_far,
        "farBeforeToNearAfterConditionalSimilarity": far_near,
        "transportCoverageMinimum": float(
            row["features"]["transportCoverageMinimum"]
        ),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or not len(numeric) or not np.isfinite(numeric).all():
        raise ValueError("T2 ranks require a finite vector")
    order = np.argsort(numeric, kind="mergesort")
    result = np.empty(len(numeric), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and numeric[order[end]] == numeric[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if (
        first.shape != second.shape
        or first.ndim != 1
        or len(first) < 2
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or float(np.ptp(first)) <= 1e-15
        or float(np.ptp(second)) <= 1e-15
    ):
        return None
    value = float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])
    return value if math.isfinite(value) else None

