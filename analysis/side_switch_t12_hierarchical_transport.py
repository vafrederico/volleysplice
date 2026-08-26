"""Type-consistent stable-player/pooled-team routing for side-switch T12."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


T12_CORE_FEATURE_NAMES = (
    "hierarchicalTrackletJerseyTeamTransportSwapMargin",
    "hierarchicalTrackletJerseyConditionalCrossSimilarityMinimum",
    "hierarchicalTrackletJerseyCrossMatchCoverageMinimum",
    "hierarchicalTrackletJerseyReliableSwapEvidence",
    "hierarchicalTrackletJerseyReliableContinuityEvidence",
)
T12_DIAGNOSTIC_FEATURE_NAMES = (
    "hierarchicalTrackletJerseyConditionalSameSimilarityMinimum",
    "hierarchicalTrackletJerseySameMatchCoverageMinimum",
    "hierarchicalTrackletJerseyTeamReliabilityMinimum",
    "hierarchicalTrackletJerseyTeamSeparationMinimum",
    "hierarchicalTrackletJerseyBaseReliabilityGate",
    "hierarchicalTrackletJerseySwapReliabilityGate",
    "hierarchicalTrackletJerseyContinuityReliabilityGate",
)
T12_FEATURE_NAMES = (*T12_CORE_FEATURE_NAMES, *T12_DIAGNOSTIC_FEATURE_NAMES)


@dataclass(frozen=True)
class HierarchicalReduction:
    cost: float
    coverage: float
    conditional_similarity: float
    route: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "cost": self.cost,
            "coverage": self.coverage,
            "conditionalSimilarity": self.conditional_similarity,
            "route": self.route,
        }


def select_reduction(
    *,
    left_stable: bool,
    right_stable: bool,
    stable_reduction: Mapping[str, Any],
    pooled_cost: float,
    left_pooled_available: bool,
    right_pooled_available: bool,
) -> HierarchicalReduction:
    if left_stable and right_stable:
        result = HierarchicalReduction(
            cost=float(stable_reduction["cost"]),
            coverage=float(stable_reduction["coverage"]),
            conditional_similarity=float(stable_reduction["conditionalSimilarity"]),
            route="stable-player-units",
        )
    else:
        available = left_pooled_available and right_pooled_available
        cost = float(pooled_cost)
        result = HierarchicalReduction(
            cost=cost,
            coverage=1.0 if available else 0.0,
            conditional_similarity=max(1.0 - cost, 0.0) if available else 0.0,
            route="pooled-team",
        )
    if not all(
        math.isfinite(value)
        for value in (result.cost, result.coverage, result.conditional_similarity)
    ):
        raise ValueError("T12 reduction must be finite")
    return result


def _side_state(row: Mapping[str, Any], endpoint: str, side: str) -> dict[str, Any]:
    t11 = row["t11EmptySideFallbackTrackletTransport"][endpoint]
    t5 = row["t5CourtTrackedFarJerseyTransport"][endpoint]
    units = t11[f"{side}Tracklets"]
    reliability = (
        sum(float(value["reliability"]) for value in units) / len(units)
        if units
        else 0.0
    )
    return {
        "stable": int(t11[f"{side}StableTracklets"]) > 0,
        "pooledAvailable": bool(t5[f"{side}TeamAvailable"]),
        "reliability": reliability,
    }


def hierarchical_transport_features(
    row: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T12 features are adjacent-boundary-only")
    sides = {
        (endpoint, side): _side_state(row, endpoint, side)
        for endpoint in ("before", "after")
        for side in ("near", "far")
    }
    pair_sides = {
        "nearNear": (("before", "near"), ("after", "near")),
        "farFar": (("before", "far"), ("after", "far")),
        "nearFar": (("before", "near"), ("after", "far")),
        "farNear": (("before", "far"), ("after", "near")),
        "beforeNearFar": (("before", "near"), ("before", "far")),
        "afterNearFar": (("after", "near"), ("after", "far")),
    }
    stable_reductions = row["t10TrackletUnitJerseyTransport"]["teamReductions"]
    pooled_costs = row["t5CourtTrackedFarJerseyTransport"]["teamCosts"]
    reductions: dict[str, HierarchicalReduction] = {}
    for name, (left_key, right_key) in pair_sides.items():
        left = sides[left_key]
        right = sides[right_key]
        reductions[name] = select_reduction(
            left_stable=bool(left["stable"]),
            right_stable=bool(right["stable"]),
            stable_reduction=stable_reductions[name],
            pooled_cost=float(pooled_costs[name]),
            left_pooled_available=bool(left["pooledAvailable"]),
            right_pooled_available=bool(right["pooledAvailable"]),
        )
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
    reliability = min(float(value["reliability"]) for value in sides.values())
    separation = min(
        reductions["beforeNearFar"].cost, reductions["afterNearFar"].cost
    )
    base_gate = min(reliability, separation)
    swap_gate = min(base_gate, cross_similarity, cross_coverage)
    continuity_gate = min(base_gate, same_similarity, same_coverage)
    values = {
        "hierarchicalTrackletJerseyTeamTransportSwapMargin": margin,
        "hierarchicalTrackletJerseyConditionalCrossSimilarityMinimum": cross_similarity,
        "hierarchicalTrackletJerseyCrossMatchCoverageMinimum": cross_coverage,
        "hierarchicalTrackletJerseyReliableSwapEvidence": max(margin, 0.0) * swap_gate,
        "hierarchicalTrackletJerseyReliableContinuityEvidence": max(-margin, 0.0)
        * continuity_gate,
        "hierarchicalTrackletJerseyConditionalSameSimilarityMinimum": same_similarity,
        "hierarchicalTrackletJerseySameMatchCoverageMinimum": same_coverage,
        "hierarchicalTrackletJerseyTeamReliabilityMinimum": reliability,
        "hierarchicalTrackletJerseyTeamSeparationMinimum": separation,
        "hierarchicalTrackletJerseyBaseReliabilityGate": base_gate,
        "hierarchicalTrackletJerseySwapReliabilityGate": swap_gate,
        "hierarchicalTrackletJerseyContinuityReliabilityGate": continuity_gate,
    }
    if tuple(values) != T12_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T12 feature signature or values changed")
    return values, {
        "sameAssignmentCost": same_cost,
        "swappedAssignmentCost": swapped_cost,
        "teamReductions": {
            name: value.to_dict() for name, value in reductions.items()
        },
    }
