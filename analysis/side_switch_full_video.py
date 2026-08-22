"""Full-video point-marker matching for side-switch proposal audits.

The side-switch specialists emit inter-rally proposal intervals, while the
exhaustive review UI stores one point inside each physical switch.  This module
implements the shared one-to-one matching contract without depending on model
labels or cadence state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence


class SideSwitchFullVideoEvaluationError(ValueError):
    """Raised when a proposal or marker violates the evaluation contract."""


@dataclass(frozen=True)
class MatchPair:
    proposal_index: int
    marker_index: int
    anchor_distance_seconds: float


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[MatchPair, ...]
    unmatched_proposal_indices: tuple[int, ...]
    unmatched_marker_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Plan:
    pairs: tuple[tuple[int, int], ...]
    anchor_distance_seconds: float


def _number(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise SideSwitchFullVideoEvaluationError(
            f"missing or invalid {key}: {row!r}"
        ) from error
    if not isfinite(value):
        raise SideSwitchFullVideoEvaluationError(
            f"non-finite {key}: {row!r}"
        )
    return value


def _better(current: _Plan | None, candidate: _Plan) -> _Plan:
    if current is None:
        return candidate
    current_key = (
        -len(current.pairs),
        current.anchor_distance_seconds,
        current.pairs,
    )
    candidate_key = (
        -len(candidate.pairs),
        candidate.anchor_distance_seconds,
        candidate.pairs,
    )
    return candidate if candidate_key < current_key else current


def monotonic_interval_match(
    proposals: Sequence[Mapping[str, object]],
    markers: Sequence[Mapping[str, object]],
    padding_seconds: float,
) -> MatchResult:
    """Match proposal intervals to point markers one-to-one in temporal order.

    A marker is eligible when it is inside ``[gapStart-padding,
    gapEnd+padding]``.  The dynamic program first maximizes cardinality, then
    minimizes total distance from each proposal's ``transitionTime`` anchor.
    Temporal ordering prevents one proposal or marker from being reused and
    gives deterministic behavior when padded neighboring gaps overlap.
    """

    padding = float(padding_seconds)
    if not isfinite(padding) or padding < 0:
        raise SideSwitchFullVideoEvaluationError(
            f"padding_seconds must be finite and non-negative: {padding_seconds}"
        )

    ordered_proposals: list[tuple[int, Mapping[str, object], float, float, float]] = []
    for index, proposal in enumerate(proposals):
        start = _number(proposal, "gapStart")
        end = _number(proposal, "gapEnd")
        anchor = _number(proposal, "transitionTime")
        if start > end:
            raise SideSwitchFullVideoEvaluationError(
                f"proposal gapStart exceeds gapEnd: {proposal!r}"
            )
        ordered_proposals.append((index, proposal, start, end, anchor))
    ordered_proposals.sort(key=lambda item: (item[4], item[2], item[3], item[0]))

    ordered_markers: list[tuple[int, Mapping[str, object], float]] = []
    for index, marker in enumerate(markers):
        ordered_markers.append((index, marker, _number(marker, "time")))
    ordered_markers.sort(key=lambda item: (item[2], item[0]))

    proposal_count = len(ordered_proposals)
    marker_count = len(ordered_markers)
    plans: list[list[_Plan | None]] = [
        [None] * (marker_count + 1) for _ in range(proposal_count + 1)
    ]
    plans[0][0] = _Plan((), 0.0)

    for proposal_index in range(proposal_count + 1):
        for marker_index in range(marker_count + 1):
            current = plans[proposal_index][marker_index]
            if current is None:
                continue
            if proposal_index < proposal_count:
                plans[proposal_index + 1][marker_index] = _better(
                    plans[proposal_index + 1][marker_index], current
                )
            if marker_index < marker_count:
                plans[proposal_index][marker_index + 1] = _better(
                    plans[proposal_index][marker_index + 1], current
                )
            if proposal_index >= proposal_count or marker_index >= marker_count:
                continue

            _, _, start, end, anchor = ordered_proposals[proposal_index]
            _, _, marker_time = ordered_markers[marker_index]
            if start - padding <= marker_time <= end + padding:
                matched = _Plan(
                    (
                        *current.pairs,
                        (proposal_index, marker_index),
                    ),
                    current.anchor_distance_seconds + abs(marker_time - anchor),
                )
                plans[proposal_index + 1][marker_index + 1] = _better(
                    plans[proposal_index + 1][marker_index + 1], matched
                )

    selected = plans[proposal_count][marker_count]
    if selected is None:  # pragma: no cover - the empty plan always exists
        raise AssertionError("interval matching produced no plan")

    pairs = tuple(
        MatchPair(
            proposal_index=ordered_proposals[proposal_index][0],
            marker_index=ordered_markers[marker_index][0],
            anchor_distance_seconds=abs(
                ordered_markers[marker_index][2]
                - ordered_proposals[proposal_index][4]
            ),
        )
        for proposal_index, marker_index in selected.pairs
    )
    matched_proposals = {pair.proposal_index for pair in pairs}
    matched_markers = {pair.marker_index for pair in pairs}
    return MatchResult(
        pairs=pairs,
        unmatched_proposal_indices=tuple(
            index for index in range(len(proposals)) if index not in matched_proposals
        ),
        unmatched_marker_indices=tuple(
            index for index in range(len(markers)) if index not in matched_markers
        ),
    )


def event_metric_counts(result: MatchResult) -> dict[str, int | float | None]:
    """Return event TP/FP/FN and derived metrics for one match result."""

    true_positives = len(result.pairs)
    false_positives = len(result.unmatched_proposal_indices)
    false_negatives = len(result.unmatched_marker_indices)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else None
    )
    if precision is None or recall is None:
        f1 = None
    elif precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
