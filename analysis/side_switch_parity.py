"""Persistent side-parity diagnostics for rare side-switch events.

The parity identity is deliberately recording-local: state zero means the initial
near/far team assignment and every reviewed physical switch toggles the state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TRANSITION_MASK_SECONDS = 4.0


@dataclass(frozen=True)
class RallyStateObservation:
    recording_id: str
    rally_index: int
    start: float
    end: float
    coordinate: float
    quality: float
    parity_state: int
    transition_masked: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RallyStateObservation":
        return cls(
            recording_id=str(payload["recordingId"]),
            rally_index=int(payload["rallyIndex"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            coordinate=float(payload["orientationCoordinate"]),
            quality=float(payload["orientationQuality"]),
            parity_state=int(payload["parityState"]),
            transition_masked=bool(payload.get("transitionMasked", False)),
        )


def parity_state_at(timestamp: float, marker_times: Sequence[float]) -> int:
    """Return recording-local parity after applying every marker at/before time."""

    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")
    markers = [float(value) for value in marker_times]
    if any(not math.isfinite(value) for value in markers):
        raise ValueError("marker times must be finite")
    return sum(value <= timestamp for value in markers) % 2


def overlaps_transition_mask(
    start: float,
    end: float,
    marker_times: Sequence[float],
    margin_seconds: float = TRANSITION_MASK_SECONDS,
) -> bool:
    """Whether an observation interval touches a masked physical transition."""

    if not all(math.isfinite(value) for value in (start, end, margin_seconds)):
        raise ValueError("transition-mask inputs must be finite")
    if end < start or margin_seconds < 0:
        raise ValueError("transition-mask interval or margin is invalid")
    return any(
        start <= float(marker) + margin_seconds
        and end >= float(marker) - margin_seconds
        for marker in marker_times
    )


def predicted_state(coordinate: float) -> int:
    """Map the score-zero-anchored orientation sign to parity state."""

    if not math.isfinite(coordinate):
        raise ValueError("orientation coordinate must be finite")
    return 0 if coordinate >= 0 else 1


def state_metrics(
    observations: Sequence[RallyStateObservation],
    minimum_quality: float = 0.0,
) -> dict[str, Any]:
    """Measure state accuracy after masking transitions and low-quality anchors."""

    if minimum_quality < 0 or not math.isfinite(minimum_quality):
        raise ValueError("minimum quality must be finite and non-negative")
    eligible = [
        value
        for value in observations
        if not value.transition_masked
        and math.isfinite(value.coordinate)
        and math.isfinite(value.quality)
    ]
    evaluated = [value for value in eligible if value.quality >= minimum_quality]
    by_state: dict[int, list[RallyStateObservation]] = {
        state: [value for value in evaluated if value.parity_state == state]
        for state in (0, 1)
    }
    correct_by_state = {
        state: sum(predicted_state(value.coordinate) == state for value in values)
        for state, values in by_state.items()
    }
    recalls = [
        correct_by_state[state] / len(by_state[state])
        for state in (0, 1)
        if by_state[state]
    ]
    correct = sum(correct_by_state.values())
    return {
        "eligibleObservations": len(eligible),
        "evaluatedObservations": len(evaluated),
        "coverage": len(evaluated) / len(eligible) if eligible else 0.0,
        "correct": correct,
        "accuracy": correct / len(evaluated) if evaluated else 0.0,
        "balancedAccuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "state0": {
            "observations": len(by_state[0]),
            "correct": correct_by_state[0],
            "recall": (
                correct_by_state[0] / len(by_state[0]) if by_state[0] else 0.0
            ),
        },
        "state1": {
            "observations": len(by_state[1]),
            "correct": correct_by_state[1],
            "recall": (
                correct_by_state[1] / len(by_state[1]) if by_state[1] else 0.0
            ),
        },
    }


def recording_quality_floor(
    observations: Sequence[RallyStateObservation], fraction_of_median: float
) -> float:
    """Create a label-independent, recording-relative quality threshold."""

    if fraction_of_median < 0 or not math.isfinite(fraction_of_median):
        raise ValueError("quality fraction must be finite and non-negative")
    qualities = sorted(
        value.quality
        for value in observations
        if not value.transition_masked and math.isfinite(value.quality)
    )
    if not qualities:
        return math.inf
    middle = len(qualities) // 2
    median = (
        qualities[middle]
        if len(qualities) % 2
        else 0.5 * (qualities[middle - 1] + qualities[middle])
    )
    return fraction_of_median * median


def decode_persistent_flips(
    observations: Sequence[RallyStateObservation],
    *,
    persistence: int,
    minimum_quality: float = 0.0,
) -> list[dict[str, Any]]:
    """Decode alternating persistent states into dead-time proposal intervals.

    State zero is known at the beginning of each recording. A flip is confirmed only
    after ``persistence`` qualified observations agree on the opposite state. The
    emitted interval is backdated to the boundary between the last observation of the
    accepted state and the first observation of the confirming run.
    """

    if persistence < 1:
        raise ValueError("persistence must be positive")
    if minimum_quality < 0 or not math.isfinite(minimum_quality):
        raise ValueError("minimum quality must be finite and non-negative")
    ordered = sorted(observations, key=lambda value: value.rally_index)
    accepted_state = 0
    last_accepted: RallyStateObservation | None = None
    run: list[RallyStateObservation] = []
    proposals: list[dict[str, Any]] = []
    for value in ordered:
        if (
            value.transition_masked
            or not math.isfinite(value.coordinate)
            or not math.isfinite(value.quality)
            or value.quality < minimum_quality
        ):
            continue
        observed_state = predicted_state(value.coordinate)
        if observed_state == accepted_state:
            last_accepted = value
            run = []
            continue
        run.append(value)
        if len(run) < persistence:
            continue
        onset = run[0]
        if last_accepted is None:
            start = end = onset.start
            previous_rally_index = None
        else:
            start = min(last_accepted.end, onset.start)
            end = max(last_accepted.end, onset.start)
            previous_rally_index = last_accepted.rally_index
        proposals.append(
            {
                "recordingId": onset.recording_id,
                "start": start,
                "end": end,
                "fromState": accepted_state,
                "toState": observed_state,
                "onsetRallyIndex": onset.rally_index,
                "confirmationRallyIndex": run[-1].rally_index,
                "previousAcceptedRallyIndex": previous_rally_index,
                "minimumRunQuality": min(item.quality for item in run),
            }
        )
        accepted_state = observed_state
        last_accepted = run[-1]
        run = []
    return proposals


def match_interval_proposals(
    proposals: Sequence[Mapping[str, Any]],
    marker_times: Sequence[float],
    margin_seconds: float = TRANSITION_MASK_SECONDS,
) -> dict[str, Any]:
    """Greedily one-to-one match marker points to padded proposal intervals."""

    if margin_seconds < 0 or not math.isfinite(margin_seconds):
        raise ValueError("matching margin must be finite and non-negative")
    unmatched = set(range(len(marker_times)))
    matches: list[dict[str, Any]] = []
    normalized = sorted(
        proposals,
        key=lambda value: (float(value["start"]), float(value["end"])),
    )
    for proposal_index, proposal in enumerate(normalized):
        start = float(proposal["start"]) - margin_seconds
        end = float(proposal["end"]) + margin_seconds
        center = 0.5 * (start + end)
        candidates = [
            marker_index
            for marker_index in unmatched
            if start <= float(marker_times[marker_index]) <= end
        ]
        if not candidates:
            continue
        marker_index = min(
            candidates,
            key=lambda index: abs(float(marker_times[index]) - center),
        )
        unmatched.remove(marker_index)
        matches.append(
            {
                "proposalIndex": proposal_index,
                "markerIndex": marker_index,
                "markerTime": float(marker_times[marker_index]),
            }
        )
    true_positives = len(matches)
    false_positives = len(normalized) - true_positives
    false_negatives = len(marker_times) - true_positives
    precision = true_positives / len(normalized) if normalized else 0.0
    recall = true_positives / len(marker_times) if marker_times else 0.0
    return {
        "proposals": len(normalized),
        "markers": len(marker_times),
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "matches": matches,
    }
