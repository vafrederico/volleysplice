"""Mass-preserving two-mode jersey transport for side-switch T9."""

from __future__ import annotations

import math
from typing import Any

from analysis.side_switch_t3_jersey_transport import T3_FEATURE_NAMES
from analysis.side_switch_t8_two_mode_transport import (
    TwoModeEndpointSummary,
    TwoModeTrack,
    mode_set_cost,
)
from analysis.side_switch_t1_transport import hellinger


def _t9_name(t3_name: str) -> str:
    if not t3_name.startswith("jersey"):
        raise ValueError(f"unexpected jersey feature name: {t3_name}")
    return "massTransportJersey" + t3_name[len("jersey") :]


T9_FEATURE_NAMES = tuple(_t9_name(name) for name in T3_FEATURE_NAMES)
T9_CORE_FEATURE_NAMES = T9_FEATURE_NAMES[:3]


def mass_transport_cost(left: TwoModeTrack, right: TwoModeTrack) -> float:
    """Return exact support-mass optimal transport for one or two modes per side."""

    if not left.available or not right.available or not left.modes or not right.modes:
        return 1.0
    if len(left.modes) > 2 or len(right.modes) > 2:
        raise ValueError("T9 supports at most two appearance modes per team")
    costs = [
        [hellinger(left_mode.descriptor, right_mode.descriptor) for right_mode in right.modes]
        for left_mode in left.modes
    ]
    if len(left.modes) == 1:
        return float(sum(mode.support * costs[0][index] for index, mode in enumerate(right.modes)))
    if len(right.modes) == 1:
        return float(sum(mode.support * costs[index][0] for index, mode in enumerate(left.modes)))
    a0 = float(left.modes[0].support)
    b0 = float(right.modes[0].support)
    lower = max(0.0, a0 + b0 - 1.0)
    upper = min(a0, b0)
    slope = costs[0][0] - costs[0][1] - costs[1][0] + costs[1][1]
    flow00 = lower if slope >= 0.0 else upper
    flow01 = a0 - flow00
    flow10 = b0 - flow00
    flow11 = 1.0 - a0 - b0 + flow00
    flows = ((flow00, flow01), (flow10, flow11))
    if min(value for row in flows for value in row) < -1e-10:
        raise AssertionError("T9 analytic transport produced negative mass")
    return float(sum(max(flows[i][j], 0.0) * costs[i][j] for i in range(2) for j in range(2)))


def mass_transport_features(
    before: TwoModeEndpointSummary, after: TwoModeEndpointSummary
) -> tuple[dict[str, float], dict[str, Any]]:
    pairs = {
        "nearNear": (before.near, after.near),
        "farFar": (before.far, after.far),
        "nearFar": (before.near, after.far),
        "farNear": (before.far, after.near),
        "beforeNearFar": (before.near, before.far),
        "afterNearFar": (after.near, after.far),
    }
    costs = {name: mass_transport_cost(*pair) for name, pair in pairs.items()}
    chamfer_costs = {name: mode_set_cost(*pair) for name, pair in pairs.items()}
    same_cost = 0.5 * (costs["nearNear"] + costs["farFar"])
    swapped_cost = 0.5 * (costs["nearFar"] + costs["farNear"])
    margin = same_cost - swapped_cost
    cross_similarity = min(1.0 - costs["nearFar"], 1.0 - costs["farNear"])
    reliability = min(
        before.near.reliability,
        before.far.reliability,
        after.near.reliability,
        after.far.reliability,
    )
    separation = min(costs["beforeNearFar"], costs["afterNearFar"])
    cohesion = min(
        before.near.cohesion,
        before.far.cohesion,
        after.near.cohesion,
        after.far.cohesion,
    )
    same_similarity = min(1.0 - costs["nearNear"], 1.0 - costs["farFar"])
    base_gate = min(reliability, separation, cohesion)
    swap_gate = min(base_gate, cross_similarity)
    continuity_gate = min(base_gate, same_similarity)
    values = {
        "massTransportJerseyTeamTransportSwapMargin": margin,
        "massTransportJerseyReliableSwapEvidence": max(margin, 0.0) * swap_gate,
        "massTransportJerseyReliableContinuityEvidence": max(-margin, 0.0) * continuity_gate,
        "massTransportJerseyCrossSideSimilarityMinimum": cross_similarity,
        "massTransportJerseySameSideSimilarityMinimum": same_similarity,
        "massTransportJerseyTeamReliabilityMinimum": reliability,
        "massTransportJerseyTeamSeparationMinimum": separation,
        "massTransportJerseyTeamCohesionMinimum": cohesion,
        "massTransportJerseyBaseReliabilityGate": base_gate,
        "massTransportJerseySwapReliabilityGate": swap_gate,
        "massTransportJerseyContinuityReliabilityGate": continuity_gate,
    }
    if tuple(values) != T9_FEATURE_NAMES or not all(math.isfinite(value) for value in values.values()):
        raise AssertionError("T9 team transport feature contract changed")
    return values, {
        "sameAssignmentCost": same_cost,
        "swappedAssignmentCost": swapped_cost,
        "teamCosts": costs,
        "t8ChamferTeamCosts": chamfer_costs,
    }
