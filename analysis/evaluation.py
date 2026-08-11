from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Interval:
    start: float
    end: float


def interval_iou(first: Interval, second: Interval) -> float:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = max(first.end, second.end) - min(first.start, second.start)
    return intersection / union if union > 0 else 0.0


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: (item.start, item.end)):
        if interval.end <= interval.start:
            continue
        if merged and interval.start < merged[-1].end:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, interval.end))
        else:
            merged.append(interval)
    return merged


def ordered_interval_matches(
    truth: Sequence[Interval],
    predictions: Sequence[Interval],
    minimum_iou: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality chronological matching, then maximum total IoU."""
    rows, columns = len(truth), len(predictions)
    counts = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    overlaps = np.zeros((rows + 1, columns + 1), dtype=np.float64)
    actions = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            best = (int(counts[row - 1, column]), float(overlaps[row - 1, column]))
            action = 1
            left = (int(counts[row, column - 1]), float(overlaps[row, column - 1]))
            if left > best:
                best = left
                action = 2
            overlap = interval_iou(truth[row - 1], predictions[column - 1])
            if overlap >= minimum_iou:
                matched = (
                    int(counts[row - 1, column - 1]) + 1,
                    float(overlaps[row - 1, column - 1]) + overlap,
                )
                if matched > best:
                    best = matched
                    action = 3
            counts[row, column], overlaps[row, column] = best
            actions[row, column] = action

    matches: list[tuple[int, int, float]] = []
    row, column = rows, columns
    while row > 0 and column > 0:
        action = actions[row, column]
        if action == 3:
            matches.append((row - 1, column - 1, interval_iou(truth[row - 1], predictions[column - 1])))
            row -= 1
            column -= 1
        elif action == 1:
            row -= 1
        else:
            column -= 1
    matches.reverse()
    return matches


def _total(intervals: Iterable[Interval]) -> float:
    return sum(item.end - item.start for item in intervals)


def _intersection_seconds(truth: Sequence[Interval], predictions: Sequence[Interval]) -> float:
    first = second = 0
    total = 0.0
    while first < len(truth) and second < len(predictions):
        left, right = truth[first], predictions[second]
        total += max(0.0, min(left.end, right.end) - max(left.start, right.start))
        if left.end <= right.end:
            first += 1
        else:
            second += 1
    return total


def evaluate_intervals(
    truth: Sequence[Interval],
    predictions: Sequence[Interval],
    *,
    minimum_iou: float = 0.5,
) -> dict[str, Any]:
    normalized_truth = merge_intervals(truth)
    normalized_predictions = merge_intervals(predictions)
    true_seconds = _total(normalized_truth)
    predicted_seconds = _total(normalized_predictions)
    intersection = _intersection_seconds(normalized_truth, normalized_predictions)
    union = true_seconds + predicted_seconds - intersection
    matches = ordered_interval_matches(normalized_truth, normalized_predictions, minimum_iou)
    matched_count = len(matches)
    precision = matched_count / len(normalized_predictions) if normalized_predictions else 0.0
    recall = matched_count / len(normalized_truth) if normalized_truth else 0.0
    start_errors = [
        normalized_predictions[predicted].start - normalized_truth[actual].start
        for actual, predicted, _ in matches
    ]
    end_errors = [
        normalized_predictions[predicted].end - normalized_truth[actual].end
        for actual, predicted, _ in matches
    ]
    return {
        "trueRallies": len(normalized_truth),
        "predictedRallies": len(normalized_predictions),
        "matchedRallies": matched_count,
        "exactRallyCount": len(normalized_truth) == len(normalized_predictions),
        "eventPrecision": precision,
        "eventRecall": recall,
        "eventF1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "timeIoU": intersection / union if union else 1.0,
        "liveTimeRecall": intersection / true_seconds if true_seconds else 1.0,
        "liveTimePrecision": intersection / predicted_seconds if predicted_seconds else 0.0,
        "trueLiveSeconds": true_seconds,
        "predictedLiveSeconds": predicted_seconds,
        "intersectionSeconds": intersection,
        "unionSeconds": union,
        "missedLiveSeconds": max(0.0, true_seconds - intersection),
        "deadSecondsRetained": max(0.0, predicted_seconds - intersection),
        "matchedIoUs": [overlap for _, _, overlap in matches],
        "startErrorsSeconds": start_errors,
        "endErrorsSeconds": end_errors,
        "matches": [
            {"truthIndex": actual, "predictionIndex": predicted, "iou": overlap}
            for actual, predicted, overlap in matches
        ],
    }


def aggregate_evaluations(per_recording: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not per_recording:
        raise ValueError("cannot aggregate an empty evaluation")

    def summed(key: str) -> float:
        return sum(float(item[key]) for item in per_recording)

    true_count = int(summed("trueRallies"))
    predicted_count = int(summed("predictedRallies"))
    matched_count = int(summed("matchedRallies"))
    true_seconds = summed("trueLiveSeconds")
    predicted_seconds = summed("predictedLiveSeconds")
    intersection = summed("intersectionSeconds")
    union = summed("unionSeconds")
    precision = matched_count / predicted_count if predicted_count else 0.0
    recall = matched_count / true_count if true_count else 0.0
    start_errors = [error for item in per_recording for error in item["startErrorsSeconds"]]
    end_errors = [error for item in per_recording for error in item["endErrorsSeconds"]]
    matched_ious = [value for item in per_recording for value in item["matchedIoUs"]]
    boundary_errors = start_errors + end_errors

    def mean_absolute(values: list[float]) -> float | None:
        return float(np.mean(np.abs(values))) if values else None

    def percentile_absolute(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.abs(values), percentile)) if values else None

    return {
        "recordings": len(per_recording),
        "trueRallies": true_count,
        "predictedRallies": predicted_count,
        "matchedRallies": matched_count,
        "exactRallyCountRate": sum(bool(item["exactRallyCount"]) for item in per_recording) / len(per_recording),
        "eventPrecision": precision,
        "eventRecall": recall,
        "eventF1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "timeIoU": intersection / union if union else 1.0,
        "liveTimeRecall": intersection / true_seconds if true_seconds else 1.0,
        "liveTimePrecision": intersection / predicted_seconds if predicted_seconds else 0.0,
        "trueLiveSeconds": true_seconds,
        "predictedLiveSeconds": predicted_seconds,
        "missedLiveSeconds": max(0.0, true_seconds - intersection),
        "deadSecondsRetained": max(0.0, predicted_seconds - intersection),
        "matchedRallyMeanIoU": float(np.mean(matched_ious)) if matched_ious else None,
        "startBoundaryMaeSeconds": mean_absolute(start_errors),
        "startBoundaryMeanSignedSeconds": float(np.mean(start_errors)) if start_errors else None,
        "startBoundaryP90Seconds": percentile_absolute(start_errors, 90),
        "endBoundaryMaeSeconds": mean_absolute(end_errors),
        "endBoundaryMeanSignedSeconds": float(np.mean(end_errors)) if end_errors else None,
        "endBoundaryP90Seconds": percentile_absolute(end_errors, 90),
        "boundariesWithin1SecondRate": (
            sum(abs(value) <= 1 for value in boundary_errors) / len(boundary_errors)
            if boundary_errors else None
        ),
        "boundariesWithin2SecondsRate": (
            sum(abs(value) <= 2 for value in boundary_errors) / len(boundary_errors)
            if boundary_errors else None
        ),
        "macroEventF1": float(np.mean([item["eventF1"] for item in per_recording])),
        "macroTimeIoU": float(np.mean([item["timeIoU"] for item in per_recording])),
    }
