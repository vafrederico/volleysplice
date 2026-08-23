"""Recording-normalized same-side continuity verifier utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PLAYER_MARGIN = "playerSwapMargin"
V4_MARGIN = "v4MeanSwapMargin"
QUALITY_FEATURES = (
    "minimumPlayerSideSeparation",
    "minimumFarSupport",
    "minimumProposalCoverage",
    "v4MinimumAlignmentResponse",
)


@dataclass(frozen=True)
class VerifierRow:
    event_id: str
    recording_id: str
    player_switch_evidence: float
    v4_switch_evidence: float
    agreement_switch_evidence: float
    quality: float
    label: int
    selected_by_control: bool
    source: Mapping[str, Any]

    def signal(self, name: str) -> float:
        values = {
            "player": self.player_switch_evidence,
            "v4": self.v4_switch_evidence,
            "agreement": self.agreement_switch_evidence,
        }
        try:
            return values[name]
        except KeyError as error:
            raise ValueError(f"unknown continuity signal: {name}") from error


def _finite_feature(row: Mapping[str, Any], name: str) -> float:
    features = row.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("continuity row has no feature mapping")
    try:
        value = float(features[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"continuity row has invalid feature {name}") from error
    if not math.isfinite(value):
        raise ValueError(f"continuity feature must be finite: {name}")
    return value


def robust_standardize(values: Sequence[float]) -> np.ndarray:
    """Robustly center a recording's evidence with a nonzero fallback scale."""

    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or len(numeric) == 0 or not np.isfinite(numeric).all():
        raise ValueError("standardization requires a nonempty finite vector")
    median = float(np.median(numeric))
    mad_scale = 1.4826 * float(np.median(np.abs(numeric - median)))
    scale = max(mad_scale, 0.25 * float(np.std(numeric)), 1e-6)
    return (numeric - median) / scale


def build_verifier_rows(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    selected_event_ids: set[str],
) -> list[VerifierRow]:
    """Create label-bound verifier rows with label-free recording normalization."""

    if not rows:
        return []
    recording_ids = {str(row.get("recordingId", "")) for row in rows}
    if len(recording_ids) != 1 or "" in recording_ids:
        raise ValueError("build_verifier_rows expects exactly one recording")
    event_ids = [str(row.get("eventId", "")) for row in rows]
    if len(set(event_ids)) != len(event_ids) or "" in event_ids:
        raise ValueError("continuity event IDs must be present and unique")
    if any(event_id not in labels for event_id in event_ids):
        raise ValueError("continuity labels do not cover every event")
    player_raw = [_finite_feature(row, PLAYER_MARGIN) for row in rows]
    v4_raw = [_finite_feature(row, V4_MARGIN) for row in rows]
    player = robust_standardize(player_raw)
    v4 = robust_standardize(v4_raw)
    quality_medians = {
        name: max(float(np.median([_finite_feature(row, name) for row in rows])), 1e-6)
        for name in QUALITY_FEATURES
    }
    result: list[VerifierRow] = []
    for index, row in enumerate(rows):
        quality = min(
            float(
                np.clip(
                    _finite_feature(row, name) / quality_medians[name], 0.0, 1.0
                )
            )
            for name in QUALITY_FEATURES
        )
        event_id = event_ids[index]
        label = int(labels[event_id])
        if label not in {0, 1}:
            raise ValueError("continuity labels must be binary")
        result.append(
            VerifierRow(
                event_id=event_id,
                recording_id=next(iter(recording_ids)),
                player_switch_evidence=float(player[index]),
                v4_switch_evidence=float(v4[index]),
                agreement_switch_evidence=float(0.5 * (player[index] + v4[index])),
                quality=quality,
                label=label,
                selected_by_control=event_id in selected_event_ids,
                source=row,
            )
        )
    return result


def _f1(true_positives: int, false_positives: int, marker_count: int) -> float:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = true_positives / marker_count if marker_count else 0.0
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _selection_counts(
    rows: Sequence[VerifierRow], selected_event_ids: set[str], marker_count: int
) -> dict[str, int | float]:
    true_positives = sum(
        row.label for row in rows if row.event_id in selected_event_ids
    )
    proposals = sum(row.event_id in selected_event_ids for row in rows)
    false_positives = proposals - true_positives
    return {
        "proposals": proposals,
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": marker_count - true_positives,
        "f1": _f1(true_positives, false_positives, marker_count),
    }


def select_veto_threshold(
    rows: Sequence[VerifierRow],
    *,
    signal: str,
    minimum_quality: float,
    marker_count: int,
    minimum_true_positive_retention: float = 0.90,
) -> dict[str, Any]:
    """Select a conservative continuity veto on development recordings.

    Lower switch evidence means stronger same-side continuity. Candidate thresholds
    maximize event F1 while retaining the configured fraction of control true positives.
    """

    if not 0 <= minimum_quality <= 1 or not math.isfinite(minimum_quality):
        raise ValueError("minimum quality must stay in [0, 1]")
    if not 0 <= minimum_true_positive_retention <= 1:
        raise ValueError("true-positive retention must stay in [0, 1]")
    control = [row for row in rows if row.selected_by_control]
    if not control:
        raise ValueError("veto selection needs control proposals")
    eligible_scores = [
        row.signal(signal) for row in control if row.quality >= minimum_quality
    ]
    lower_noop = min(eligible_scores, default=0.0) - 1e-9
    thresholds = [lower_noop, *sorted(set(eligible_scores))]
    baseline_true_positives = sum(row.label for row in control)
    required = math.ceil(
        minimum_true_positive_retention * baseline_true_positives - 1e-12
    )
    best: tuple[tuple[float, int, int, float], float, set[str], dict[str, Any]] | None = None
    for threshold in thresholds:
        retained = {
            row.event_id
            for row in control
            if not (
                row.quality >= minimum_quality
                and row.signal(signal) <= threshold
            )
        }
        counts = _selection_counts(control, retained, marker_count)
        if int(counts["truePositives"]) < required:
            continue
        key = (
            float(counts["f1"]),
            int(counts["truePositives"]),
            -int(counts["falsePositives"]),
            -threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, retained, counts)
    if best is None:  # pragma: no cover - the no-op threshold always qualifies
        raise AssertionError("no continuity veto threshold satisfied retention")
    _, threshold, retained, counts = best
    return {
        "threshold": threshold,
        "minimumQuality": minimum_quality,
        "minimumTruePositiveRetention": minimum_true_positive_retention,
        "baselineTruePositives": baseline_true_positives,
        "requiredTruePositives": required,
        "retainedEventIds": sorted(retained),
        "trainingCounts": counts,
    }


def select_positive_threshold(
    rows: Sequence[VerifierRow],
    *,
    signal: str,
    minimum_quality: float,
    marker_count: int,
) -> dict[str, Any]:
    """Select a switch-evidence threshold for verifier-alone/add-only diagnostics."""

    if not rows:
        raise ValueError("positive threshold selection needs rows")
    if not 0 <= minimum_quality <= 1 or not math.isfinite(minimum_quality):
        raise ValueError("minimum quality must stay in [0, 1]")
    eligible_scores = [
        row.signal(signal) for row in rows if row.quality >= minimum_quality
    ]
    upper_noop = max(eligible_scores, default=0.0) + 1e-9
    thresholds = [upper_noop, *sorted(set(eligible_scores), reverse=True)]
    best: tuple[tuple[float, int, int, float], float, set[str], dict[str, Any]] | None = None
    for threshold in thresholds:
        selected = {
            row.event_id
            for row in rows
            if row.quality >= minimum_quality and row.signal(signal) >= threshold
        }
        counts = _selection_counts(rows, selected, marker_count)
        key = (
            float(counts["f1"]),
            int(counts["truePositives"]),
            -int(counts["falsePositives"]),
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, selected, counts)
    if best is None:  # pragma: no cover - rows guarantee a candidate
        raise AssertionError("positive threshold selection produced no result")
    _, threshold, selected, counts = best
    return {
        "threshold": threshold,
        "minimumQuality": minimum_quality,
        "selectedEventIds": sorted(selected),
        "trainingCounts": counts,
    }


def apply_veto(
    rows: Sequence[VerifierRow],
    *,
    signal: str,
    threshold: float,
    minimum_quality: float,
) -> set[str]:
    return {
        row.event_id
        for row in rows
        if row.selected_by_control
        and not (
            row.quality >= minimum_quality and row.signal(signal) <= threshold
        )
    }


def apply_positive(
    rows: Sequence[VerifierRow],
    *,
    signal: str,
    threshold: float,
    minimum_quality: float,
) -> set[str]:
    return {
        row.event_id
        for row in rows
        if row.quality >= minimum_quality and row.signal(signal) >= threshold
    }
