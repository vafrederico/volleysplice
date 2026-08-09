from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence

import numpy as np


class TimeInterval(Protocol):
    start: float
    end: float


def interval_iou(first: TimeInterval, second: TimeInterval) -> float:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = max(first.end, second.end) - min(first.start, second.start)
    return intersection / union if union > 0 else 0.0


def _total(intervals: Iterable[TimeInterval]) -> float:
    return sum(max(0.0, item.end - item.start) for item in intervals)


def _intersection_seconds(
    truth: Sequence[TimeInterval], predictions: Sequence[TimeInterval]
) -> float:
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


def ordered_interval_matches(
    truth: Sequence[TimeInterval],
    predictions: Sequence[TimeInterval],
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


def evaluate_intervals(
    truth: Sequence[TimeInterval],
    predictions: Sequence[TimeInterval],
    *,
    minimum_iou: float = 0.5,
) -> dict[str, Any]:
    true_seconds = _total(truth)
    predicted_seconds = _total(predictions)
    intersection = _intersection_seconds(truth, predictions)
    union = true_seconds + predicted_seconds - intersection
    matches = ordered_interval_matches(truth, predictions, minimum_iou)
    matched_count = len(matches)
    precision = matched_count / len(predictions) if predictions else (1.0 if not truth else 0.0)
    recall = matched_count / len(truth) if truth else (1.0 if not predictions else 0.0)
    event_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    start_errors = [predictions[pred].start - truth[actual].start for actual, pred, _ in matches]
    end_errors = [predictions[pred].end - truth[actual].end for actual, pred, _ in matches]
    matched_ious = [overlap for _, _, overlap in matches]
    return {
        "trueRallies": len(truth),
        "predictedRallies": len(predictions),
        "matchedRallies": matched_count,
        "exactRallyCount": len(truth) == len(predictions),
        "eventPrecision": precision,
        "eventRecall": recall,
        "eventF1": event_f1,
        "timeIoU": intersection / union if union else 1.0,
        "liveTimeRecall": intersection / true_seconds if true_seconds else 1.0,
        "liveTimePrecision": intersection / predicted_seconds if predicted_seconds else (1.0 if not truth else 0.0),
        "trueLiveSeconds": true_seconds,
        "predictedLiveSeconds": predicted_seconds,
        "intersectionSeconds": intersection,
        "unionSeconds": union,
        "missedLiveSeconds": max(0.0, true_seconds - intersection),
        "deadSecondsRetained": max(0.0, predicted_seconds - intersection),
        "matchedIoUs": matched_ious,
        "startErrorsSeconds": start_errors,
        "endErrorsSeconds": end_errors,
    }


def aggregate_evaluations(per_recording: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not per_recording:
        raise ValueError("cannot aggregate an empty evaluation")
    summed = lambda key: sum(float(item[key]) for item in per_recording)
    true_count = int(summed("trueRallies"))
    predicted_count = int(summed("predictedRallies"))
    matched_count = int(summed("matchedRallies"))
    true_seconds = summed("trueLiveSeconds")
    predicted_seconds = summed("predictedLiveSeconds")
    intersection = summed("intersectionSeconds")
    union = summed("unionSeconds")
    precision = matched_count / predicted_count if predicted_count else (1.0 if not true_count else 0.0)
    recall = matched_count / true_count if true_count else (1.0 if not predicted_count else 0.0)
    start_errors = [error for item in per_recording for error in item["startErrorsSeconds"]]
    end_errors = [error for item in per_recording for error in item["endErrorsSeconds"]]
    matched_ious = [value for item in per_recording for value in item["matchedIoUs"]]

    def mean_absolute(values: list[float]) -> float | None:
        return float(np.mean(np.abs(values))) if values else None

    def percentile_absolute(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.abs(values), percentile)) if values else None

    def signed_mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    boundary_errors = start_errors + end_errors

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
        "liveTimePrecision": intersection / predicted_seconds if predicted_seconds else (1.0 if not true_count else 0.0),
        "trueLiveSeconds": true_seconds,
        "predictedLiveSeconds": predicted_seconds,
        "missedLiveSeconds": max(0.0, true_seconds - intersection),
        "deadSecondsRetained": max(0.0, predicted_seconds - intersection),
        "matchedRallyMeanIoU": float(np.mean(matched_ious)) if matched_ious else None,
        "startBoundaryMaeSeconds": mean_absolute(start_errors),
        "startBoundaryMeanSignedSeconds": signed_mean(start_errors),
        "startBoundaryP50Seconds": percentile_absolute(start_errors, 50),
        "startBoundaryP90Seconds": percentile_absolute(start_errors, 90),
        "endBoundaryMaeSeconds": mean_absolute(end_errors),
        "endBoundaryMeanSignedSeconds": signed_mean(end_errors),
        "endBoundaryP50Seconds": percentile_absolute(end_errors, 50),
        "endBoundaryP90Seconds": percentile_absolute(end_errors, 90),
        "boundariesWithin1SecondRate": (
            sum(abs(value) <= 1.0 for value in boundary_errors) / len(boundary_errors)
            if boundary_errors
            else None
        ),
        "boundariesWithin2SecondsRate": (
            sum(abs(value) <= 2.0 for value in boundary_errors) / len(boundary_errors)
            if boundary_errors
            else None
        ),
        "macroEventF1": float(np.mean([item["eventF1"] for item in per_recording])),
        "macroTimeIoU": float(np.mean([item["timeIoU"] for item in per_recording])),
    }
