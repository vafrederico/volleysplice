"""Shared evaluation contract for experiments against the side-switch research winner."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from analysis.side_switch_full_union_ranker import (
    DERIVED_FEATURE_NAMES,
    UnionDecoderSettings,
    decode_ranked_candidates,
    fit_weighted_logistic,
)
from analysis.side_switch_full_video import event_metric_counts, monotonic_interval_match
from analysis.side_switch_production_state import STATE_GATE_FEATURE_NAMES
from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v5 import VISUAL_FEATURE_NAMES
from analysis.side_switch_v6 import matrix_for


FEATURE_NAMES = (
    *VISUAL_FEATURE_NAMES,
    *STATE_GATE_FEATURE_NAMES,
    *DERIVED_FEATURE_NAMES,
)
L2 = 0.1
CLASS_BALANCE_EXPONENT = 0.5
HARD_NEGATIVES_PER_RECORDING = 2
HARD_NEGATIVE_MULTIPLIER = 2.0
PADDING = 4.0
THRESHOLD_QUANTILES = 65
DECODER = UnionDecoderSettings(
    minimum_index_separation=2,
    free_predictions_per_recording=6,
    count_penalty_logit=0.5,
)
FitFunction = Callable[[Sequence[V3Event], np.ndarray], Any]


def proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eventId": str(row["eventId"]),
        "recordingId": str(row["recordingId"]),
        "kind": str(row["kind"]),
        "gapStart": float(row["gapStart"]),
        "gapEnd": float(row["gapEnd"]),
        "transitionTime": float(row["transitionTime"]),
    }


def candidate_labels(
    rows: Sequence[Mapping[str, Any]], markers: Mapping[str, Any]
) -> dict[str, int]:
    result = {str(row["eventId"]): 0 for row in rows}
    for recording_id, truth in markers.items():
        local = [row for row in rows if str(row["recordingId"]) == recording_id]
        match = monotonic_interval_match([proposal(row) for row in local], truth, PADDING)
        for pair in match.pairs:
            result[str(local[pair.proposal_index]["eventId"])] = 1
    return result


def events(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int]
) -> list[V3Event]:
    order: dict[str, int] = {}
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        local = sorted(
            (row for row in rows if str(row["recordingId"]) == recording_id),
            key=lambda row: (float(row["transitionTime"]), str(row["eventId"])),
        )
        order.update({str(row["eventId"]): index for index, row in enumerate(local, 1)})
    return [
        V3Event(
            str(row["eventId"]),
            str(row["recordingId"]),
            "opened-development",
            order[str(row["eventId"])],
            labels[str(row["eventId"])],
            row,
        )
        for row in rows
    ]


def fit_promoted_head(
    fit_events: Sequence[V3Event],
    final_fitter: FitFunction | None = None,
) -> tuple[Any, dict[str, Any]]:
    initial = fit_weighted_logistic(
        fit_events,
        L2,
        FEATURE_NAMES,
        CLASS_BALANCE_EXPONENT,
    )
    initial_scores = initial.predict_proba(matrix_for(fit_events, FEATURE_NAMES))
    multipliers = np.ones(len(fit_events), dtype=np.float64)
    selected_ids: dict[str, list[str]] = {}
    for recording_id in sorted({event.recording_id for event in fit_events}):
        negatives = [
            index
            for index, event in enumerate(fit_events)
            if event.recording_id == recording_id and event.label == 0
        ]
        chosen = sorted(
            negatives,
            key=lambda index: (
                -float(initial_scores[index]),
                fit_events[index].event_id,
            ),
        )[:HARD_NEGATIVES_PER_RECORDING]
        multipliers[chosen] = HARD_NEGATIVE_MULTIPLIER
        selected_ids[recording_id] = [fit_events[index].event_id for index in chosen]
    model = (
        fit_weighted_logistic(
            fit_events,
            L2,
            FEATURE_NAMES,
            CLASS_BALANCE_EXPONENT,
            multipliers,
        )
        if final_fitter is None
        else final_fitter(fit_events, multipliers)
    )
    return model, {
        "selectedHardNegatives": int(np.sum(multipliers > 1.0)),
        "byRecording": selected_ids,
    }


def crossfit_scores(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    final_fitter: FitFunction | None = None,
) -> np.ndarray:
    all_events = events(rows, labels)
    scores = np.full(len(rows), np.nan)
    for held_id in sorted({event.recording_id for event in all_events}):
        fit = [
            index
            for index, event in enumerate(all_events)
            if event.recording_id != held_id
        ]
        held = [
            index
            for index, event in enumerate(all_events)
            if event.recording_id == held_id
        ]
        model, _ = fit_promoted_head(
            [all_events[index] for index in fit], final_fitter
        )
        scores[held] = model.predict_proba(
            matrix_for([all_events[index] for index in held], FEATURE_NAMES)
        )
    if not np.isfinite(scores).all():
        raise ValueError("winner cross-fit left rows unscored")
    return scores


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    markers: Mapping[str, Any],
    padding: float,
    *,
    inventory: bool = False,
) -> dict[str, Any]:
    per_video: dict[str, Any] = {}
    for recording_id, truth in markers.items():
        proposals = sorted(
            [
                proposal(row)
                for row, selected in zip(rows, predictions, strict=True)
                if selected and str(row["recordingId"]) == recording_id
            ],
            key=lambda row: (row["transitionTime"], row["eventId"]),
        )
        match = monotonic_interval_match(proposals, truth, padding)
        counts = event_metric_counts(match)
        per_video[recording_id] = {
            "humanEvents": len(truth),
            "proposals": len(proposals),
            **counts,
            "missedHumanTimes": [
                float(truth[index]["time"]) for index in match.unmatched_marker_indices
            ],
        }
        if inventory:
            per_video[recording_id]["proposalInventory"] = proposals
    tp = sum(int(value["truePositives"]) for value in per_video.values())
    fp = sum(int(value["falsePositives"]) for value in per_video.values())
    fn = sum(int(value["falseNegatives"]) for value in per_video.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "paddingSeconds": padding,
        "proposals": tp + fp,
        "truePositives": tp,
        "falsePositives": fp,
        "falseNegatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "macroPerVideoPrecision": float(
            np.mean([float(value["precision"] or 0.0) for value in per_video.values()])
        ),
        "macroPerVideoRecall": float(
            np.mean([float(value["recall"] or 0.0) for value in per_video.values()])
        ),
        "byRecording": per_video,
    }


def thresholds(scores: np.ndarray) -> tuple[float, ...]:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, THRESHOLD_QUANTILES))
    above = min(1.0, math.nextafter(float(np.max(scores)), math.inf))
    return tuple(sorted({above, *(float(value) for value in quantiles)}, reverse=True))


def metric_rank(metrics: Mapping[str, Any], threshold: float) -> tuple[float, ...]:
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(metrics["proposals"]),
        float(threshold),
    )


def select_threshold(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    markers: Mapping[str, Any],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in thresholds(scores):
        predictions = decode_ranked_candidates(rows, scores, threshold, DECODER)
        metrics = evaluate(rows, predictions, markers, PADDING)
        value = {"threshold": threshold, "metrics": metrics}
        if best is None or metric_rank(metrics, threshold) > metric_rank(
            best["metrics"], float(best["threshold"])
        ):
            best = value
    if best is None:
        raise AssertionError("winner threshold selection failed")
    return best


def logit(value: float) -> float:
    clipped = min(1.0 - 1e-9, max(1e-9, value))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))
