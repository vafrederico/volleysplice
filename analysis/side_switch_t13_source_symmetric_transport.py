"""Source-symmetric stable/pooled routing for side-switch T13."""

from __future__ import annotations

import math
from typing import Any, Mapping

from analysis.side_switch_t12_hierarchical_transport import (
    HierarchicalReduction,
    _side_state,
    select_reduction,
)


T13_CORE_FEATURE_NAMES = (
    "sourceSymmetricTrackletJerseyTeamTransportSwapMargin",
    "sourceSymmetricTrackletJerseyConditionalCrossSimilarityMinimum",
    "sourceSymmetricTrackletJerseyCrossMatchCoverageMinimum",
    "sourceSymmetricTrackletJerseyReliableSwapEvidence",
    "sourceSymmetricTrackletJerseyReliableContinuityEvidence",
)
T13_DIAGNOSTIC_FEATURE_NAMES = (
    "sourceSymmetricTrackletJerseyConditionalSameSimilarityMinimum",
    "sourceSymmetricTrackletJerseySameMatchCoverageMinimum",
    "sourceSymmetricTrackletJerseyTeamReliabilityMinimum",
    "sourceSymmetricTrackletJerseyTeamSeparationMinimum",
    "sourceSymmetricTrackletJerseyBaseReliabilityGate",
    "sourceSymmetricTrackletJerseySwapReliabilityGate",
    "sourceSymmetricTrackletJerseyContinuityReliabilityGate",
)
T13_FEATURE_NAMES = (*T13_CORE_FEATURE_NAMES, *T13_DIAGNOSTIC_FEATURE_NAMES)


def source_symmetric_transport_features(
    row: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    if str(row.get("kind")) != "adjacent-rally-boundary":
        raise ValueError("T13 features are adjacent-boundary-only")
    sides = {
        (endpoint, side): _side_state(row, endpoint, side)
        for endpoint in ("before", "after")
        for side in ("near", "far")
    }
    stable_reductions = row["t10TrackletUnitJerseyTransport"]["teamReductions"]
    pooled_costs = row["t5CourtTrackedFarJerseyTransport"]["teamCosts"]

    def reduction(
        name: str,
        left_key: tuple[str, str],
        right_key: tuple[str, str],
        stable_route: bool,
    ) -> HierarchicalReduction:
        left = sides[left_key]
        right = sides[right_key]
        return select_reduction(
            left_stable=stable_route,
            right_stable=stable_route,
            stable_reduction=stable_reductions[name],
            pooled_cost=float(pooled_costs[name]),
            left_pooled_available=bool(left["pooledAvailable"]),
            right_pooled_available=bool(right["pooledAvailable"]),
        )

    after_both_stable = bool(
        sides[("after", "near")]["stable"]
        and sides[("after", "far")]["stable"]
    )
    near_route = bool(sides[("before", "near")]["stable"] and after_both_stable)
    far_route = bool(sides[("before", "far")]["stable"] and after_both_stable)
    reductions = {
        "nearNear": reduction(
            "nearNear", ("before", "near"), ("after", "near"), near_route
        ),
        "nearFar": reduction(
            "nearFar", ("before", "near"), ("after", "far"), near_route
        ),
        "farFar": reduction(
            "farFar", ("before", "far"), ("after", "far"), far_route
        ),
        "farNear": reduction(
            "farNear", ("before", "far"), ("after", "near"), far_route
        ),
        "beforeNearFar": reduction(
            "beforeNearFar",
            ("before", "near"),
            ("before", "far"),
            bool(
                sides[("before", "near")]["stable"]
                and sides[("before", "far")]["stable"]
            ),
        ),
        "afterNearFar": reduction(
            "afterNearFar",
            ("after", "near"),
            ("after", "far"),
            after_both_stable,
        ),
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
    reliability = min(float(value["reliability"]) for value in sides.values())
    separation = min(
        reductions["beforeNearFar"].cost, reductions["afterNearFar"].cost
    )
    base_gate = min(reliability, separation)
    swap_gate = min(base_gate, cross_similarity, cross_coverage)
    continuity_gate = min(base_gate, same_similarity, same_coverage)
    values = {
        "sourceSymmetricTrackletJerseyTeamTransportSwapMargin": margin,
        "sourceSymmetricTrackletJerseyConditionalCrossSimilarityMinimum": cross_similarity,
        "sourceSymmetricTrackletJerseyCrossMatchCoverageMinimum": cross_coverage,
        "sourceSymmetricTrackletJerseyReliableSwapEvidence": max(margin, 0.0) * swap_gate,
        "sourceSymmetricTrackletJerseyReliableContinuityEvidence": max(-margin, 0.0)
        * continuity_gate,
        "sourceSymmetricTrackletJerseyConditionalSameSimilarityMinimum": same_similarity,
        "sourceSymmetricTrackletJerseySameMatchCoverageMinimum": same_coverage,
        "sourceSymmetricTrackletJerseyTeamReliabilityMinimum": reliability,
        "sourceSymmetricTrackletJerseyTeamSeparationMinimum": separation,
        "sourceSymmetricTrackletJerseyBaseReliabilityGate": base_gate,
        "sourceSymmetricTrackletJerseySwapReliabilityGate": swap_gate,
        "sourceSymmetricTrackletJerseyContinuityReliabilityGate": continuity_gate,
    }
    if tuple(values) != T13_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("T13 feature signature or values changed")
    return values, {
        "sameAssignmentCost": same_cost,
        "swappedAssignmentCost": swapped_cost,
        "sourceRoutes": {
            "beforeNear": "stable-player-units" if near_route else "pooled-team",
            "beforeFar": "stable-player-units" if far_route else "pooled-team",
        },
        "teamReductions": {
            name: value.to_dict() for name, value in reductions.items()
        },
    }
