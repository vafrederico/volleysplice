"""Internal dead-state candidate representation and specialist head."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.side_switch_full_union_ranker import fit_weighted_logistic
from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v6 import matrix_for


DERIVED_INTERNAL_FEATURE_NAMES = (
    "internalRangeDurationSeconds",
    "internalSecondsSinceRangeStart",
    "internalSecondsToRangeEnd",
    "internalNormalizedRangePosition",
    "internalSecondsSinceServeAnchor",
    "internalNearestBoundarySeconds",
    "internalPeaksInSourceRange",
    "internalInverseScoreRankInRange",
    "internalPaletteInstabilityDifference",
)
TRANSITION_FEATURE_NAMES = (
    "v4MeanSwapMargin",
    "v4GlobalAppearanceChange",
    "playerSwapMargin",
    "playerOrientationFlipEvidence",
    "minimumPlayerSideSeparation",
    "playerSideSeparationChange",
    "playerGlobalAppearanceChange",
    "proposalCoverageChange",
    "proposalCountChange",
    "sideSupportImbalanceChange",
    "internalPaletteInstabilityDifference",
    "productionMinimumAdjacentServeConfidence",
)
GEOMETRY_FEATURE_NAMES = (
    "candidateGeneratorScore",
    "internalRangeDurationSeconds",
    "internalSecondsSinceRangeStart",
    "internalSecondsToRangeEnd",
    "internalNormalizedRangePosition",
    "internalSecondsSinceServeAnchor",
    "internalNearestBoundarySeconds",
    "internalPeaksInSourceRange",
    "internalInverseScoreRankInRange",
    "productionMaximumAdjacentAnchorErrorSeconds",
    "productionDeadToNextServeSeconds",
    "productionGapMeanDeadStateScore",
)
COMBINED_FEATURE_NAMES = tuple(dict.fromkeys((*TRANSITION_FEATURE_NAMES, *GEOMETRY_FEATURE_NAMES)))
FEATURE_GROUPS = {
    "transition12": TRANSITION_FEATURE_NAMES,
    "geometry12": GEOMETRY_FEATURE_NAMES,
    "combined24": COMBINED_FEATURE_NAMES,
}


def enrich_internal_candidates(
    rows: Sequence[Mapping[str, Any]],
    all_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    boundaries_by_recording: dict[str, list[float]] = {}
    for candidate in all_candidates:
        if str(candidate.get("kind")) == "adjacent-rally-boundary":
            boundaries_by_recording.setdefault(str(candidate["recordingId"]), []).append(
                float(candidate["transitionTime"])
            )
    peers: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("kind")) != "internal-dead-state-peak":
            raise ValueError("internal specialist received a non-internal candidate")
        key = (str(row["recordingId"]), str(row.get("sourceRangeId", "")))
        peers.setdefault(key, []).append(row)
    result = []
    for row in rows:
        recording_id = str(row["recordingId"])
        source_range = str(row.get("sourceRangeId", ""))
        context = row.get("productionStateContext", {})
        before = context.get("before", {}) if isinstance(context, Mapping) else {}
        start = float(before.get("sourceStart", row["gapStart"]))
        end = float(before.get("sourceEnd", row["gapEnd"]))
        anchor = float(before.get("anchorTime", start))
        timestamp = float(row["transitionTime"])
        duration = max(end - start, 1e-6)
        local_peers = peers[(recording_id, source_range)]
        ranked = sorted(
            local_peers,
            key=lambda value: (-float(value.get("score", 0.0)), str(value["eventId"])),
        )
        rank = next(
            index
            for index, value in enumerate(ranked, 1)
            if str(value["eventId"]) == str(row["eventId"])
        )
        boundaries = boundaries_by_recording.get(recording_id, [])
        nearest_boundary = min(
            (abs(timestamp - value) for value in boundaries), default=duration
        )
        features = dict(row.get("features", {}))
        before_instability = float(features.get("beforePlayerPaletteInstability", 0.0))
        after_instability = float(features.get("afterPlayerPaletteInstability", 0.0))
        features.update(
            {
                "candidateGeneratorScore": float(row.get("score", 0.0)),
                "internalRangeDurationSeconds": duration,
                "internalSecondsSinceRangeStart": timestamp - start,
                "internalSecondsToRangeEnd": end - timestamp,
                "internalNormalizedRangePosition": (timestamp - start) / duration,
                "internalSecondsSinceServeAnchor": timestamp - anchor,
                "internalNearestBoundarySeconds": nearest_boundary,
                "internalPeaksInSourceRange": float(len(local_peers)),
                "internalInverseScoreRankInRange": 1.0 / rank,
                "internalPaletteInstabilityDifference": abs(
                    after_instability - before_instability
                ),
            }
        )
        enriched = dict(row)
        enriched["features"] = features
        result.append(enriched)
    return result


def fit_internal_head(
    events: Sequence[V3Event], feature_names: Sequence[str], l2: float
) -> tuple[Any, dict[str, Any]]:
    initial = fit_weighted_logistic(events, l2, feature_names, 0.5)
    scores = initial.predict_proba(matrix_for(events, feature_names))
    multipliers = np.ones(len(events), dtype=np.float64)
    selected: dict[str, list[str]] = {}
    for recording_id in sorted({event.recording_id for event in events}):
        negatives = [
            index
            for index, event in enumerate(events)
            if event.recording_id == recording_id and event.label == 0
        ]
        chosen = sorted(
            negatives,
            key=lambda index: (-float(scores[index]), events[index].event_id),
        )[:2]
        multipliers[chosen] = 2.0
        selected[recording_id] = [events[index].event_id for index in chosen]
    model = fit_weighted_logistic(events, l2, feature_names, 0.5, multipliers)
    return model, {
        "hardNegativesPerRecording": 2,
        "hardNegativeMultiplier": 2.0,
        "selectedEventIds": selected,
    }


def crossfit_internal_scores(
    events: Sequence[V3Event], feature_names: Sequence[str], l2: float
) -> np.ndarray:
    scores = np.full(len(events), np.nan, dtype=np.float64)
    for held_id in sorted({event.recording_id for event in events}):
        fit = [index for index, event in enumerate(events) if event.recording_id != held_id]
        held = [index for index, event in enumerate(events) if event.recording_id == held_id]
        model, _ = fit_internal_head([events[index] for index in fit], feature_names, l2)
        scores[held] = model.predict_proba(
            matrix_for([events[index] for index in held], feature_names)
        )
    if not np.isfinite(scores).all():
        raise ValueError("internal specialist cross-fit left candidates unscored")
    return scores


def select_internal_candidates(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    threshold: float,
    boundary_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_separation_seconds: float = 10.0,
    boundary_duplicate_seconds: float = 4.0,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.shape != (len(rows),)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("internal selection needs finite aligned scores")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("internal threshold must stay in [0, 1]")
    selected = np.zeros(len(rows), dtype=bool)
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        indexes = [
            index
            for index, row in enumerate(rows)
            if str(row["recordingId"]) == recording_id
        ]
        boundary_times = [
            float(row["transitionTime"])
            for row in boundary_rows
            if str(row["recordingId"]) == recording_id
        ]
        chosen: list[int] = []
        for index in sorted(
            indexes,
            key=lambda value: (
                -float(values[value]),
                float(rows[value]["transitionTime"]),
                str(rows[value]["eventId"]),
            ),
        ):
            if values[index] < threshold:
                continue
            timestamp = float(rows[index]["transitionTime"])
            if any(abs(timestamp - value) <= boundary_duplicate_seconds for value in boundary_times):
                continue
            if any(
                abs(timestamp - float(rows[other]["transitionTime"]))
                < minimum_separation_seconds
                for other in chosen
            ):
                continue
            chosen.append(index)
            selected[index] = True
    return selected
