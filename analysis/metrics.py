from __future__ import annotations

import math
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
    matches_at_03 = ordered_interval_matches(truth, predictions, 0.3)
    matches_at_07 = ordered_interval_matches(truth, predictions, 0.7)
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
        "matchedRalliesAtIou03": len(matches_at_03),
        "matchedRalliesAtIou07": len(matches_at_07),
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
        "matches": [
            {"truthIndex": actual, "predictionIndex": prediction, "iou": overlap}
            for actual, prediction, overlap in matches
        ],
    }


def evaluate_interval_selection(
    truth: Sequence[TimeInterval],
    predictions: Sequence[TimeInterval],
    *,
    minimum_iou: float = 0.5,
) -> dict[str, float | int]:
    """Return only the sufficient statistics used while selecting a decoder.

    Decoder search evaluates thousands of candidates. The full evaluation also
    computes outcome slices, boundary errors, and matches at additional IoU
    thresholds; none of those values participates in selection.
    """
    return {
        "trueRallies": len(truth),
        "predictedRallies": len(predictions),
        "matchedRallies": len(
            ordered_interval_matches(truth, predictions, minimum_iou)
        ),
        "trueLiveSeconds": _total(truth),
        "predictedLiveSeconds": _total(predictions),
        "intersectionSeconds": _intersection_seconds(truth, predictions),
    }


def aggregate_interval_selection(
    per_recording: Sequence[dict[str, float | int]],
) -> dict[str, float | int]:
    """Aggregate decoder-selection statistics with full-metric parity."""
    if not per_recording:
        raise ValueError("cannot aggregate an empty evaluation")
    true_count = sum(int(item["trueRallies"]) for item in per_recording)
    predicted_count = sum(int(item["predictedRallies"]) for item in per_recording)
    matched_count = sum(int(item["matchedRallies"]) for item in per_recording)
    true_seconds = sum(float(item["trueLiveSeconds"]) for item in per_recording)
    predicted_seconds = sum(
        float(item["predictedLiveSeconds"]) for item in per_recording
    )
    intersection = sum(float(item["intersectionSeconds"]) for item in per_recording)
    union = true_seconds + predicted_seconds - intersection
    precision = (
        matched_count / predicted_count
        if predicted_count
        else (1.0 if not true_count else 0.0)
    )
    recall = (
        matched_count / true_count
        if true_count
        else (1.0 if not predicted_count else 0.0)
    )
    return {
        "trueRallies": true_count,
        "predictedRallies": predicted_count,
        "matchedRallies": matched_count,
        "eventF1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "timeIoU": intersection / union if union else 1.0,
        "liveTimeRecall": intersection / true_seconds if true_seconds else 1.0,
        "liveTimePrecision": (
            intersection / predicted_seconds
            if predicted_seconds
            else (1.0 if not true_count else 0.0)
        ),
    }


def aggregate_evaluations(per_recording: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not per_recording:
        raise ValueError("cannot aggregate an empty evaluation")
    summed = lambda key: sum(float(item[key]) for item in per_recording)
    true_count = int(summed("trueRallies"))
    predicted_count = int(summed("predictedRallies"))
    matched_count = int(summed("matchedRallies"))
    matched_count_03 = int(summed("matchedRalliesAtIou03"))
    matched_count_07 = int(summed("matchedRalliesAtIou07"))
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

    def event_f1_for_matches(count: int) -> float:
        precision_at_threshold = count / predicted_count if predicted_count else (
            1.0 if not true_count else 0.0
        )
        recall_at_threshold = count / true_count if true_count else (
            1.0 if not predicted_count else 0.0
        )
        return (
            2
            * precision_at_threshold
            * recall_at_threshold
            / (precision_at_threshold + recall_at_threshold)
            if precision_at_threshold + recall_at_threshold
            else 0.0
        )

    return {
        "recordings": len(per_recording),
        "trueRallies": true_count,
        "predictedRallies": predicted_count,
        "matchedRallies": matched_count,
        "matchedRalliesAtIou03": matched_count_03,
        "matchedRalliesAtIou07": matched_count_07,
        "exactRallyCountRate": sum(bool(item["exactRallyCount"]) for item in per_recording) / len(per_recording),
        "eventPrecision": precision,
        "eventRecall": recall,
        "eventF1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "eventF1AtIou03": event_f1_for_matches(matched_count_03),
        "eventF1AtIou07": event_f1_for_matches(matched_count_07),
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
        "boundariesWithin025SecondRate": (
            sum(abs(value) <= 0.25 for value in boundary_errors) / len(boundary_errors)
            if boundary_errors
            else None
        ),
        "boundariesWithin05SecondRate": (
            sum(abs(value) <= 0.5 for value in boundary_errors) / len(boundary_errors)
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


def truth_slice_metrics(
    truth: Sequence[TimeInterval],
    predictions: Sequence[TimeInterval],
    truth_indexes: Sequence[int],
    *,
    minimum_iou: float = 0.5,
) -> dict[str, Any]:
    """Measure recall/coverage for a truth subset without relabeling other events negative."""
    selected = tuple(dict.fromkeys(int(index) for index in truth_indexes))
    if any(index < 0 or index >= len(truth) for index in selected):
        raise ValueError("truth slice index is out of range")
    matches = {
        actual: (prediction, overlap)
        for actual, prediction, overlap in ordered_interval_matches(
            truth, predictions, minimum_iou
        )
    }
    rows: list[dict[str, Any]] = []
    for index in selected:
        actual = truth[index]
        duration = actual.end - actual.start
        intersections = sorted(
            (
                max(actual.start, item.start),
                min(actual.end, item.end),
                prediction_index,
            )
            for prediction_index, item in enumerate(predictions)
            if item.start < actual.end and actual.start < item.end
        )
        merged: list[tuple[float, float]] = []
        for start, end, _ in intersections:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        covered_seconds = sum(end - start for start, end in merged)
        if intersections:
            _, _, boundary_prediction_index = max(
                intersections, key=lambda item: (item[1] - item[0], -item[2])
            )
            boundary_prediction = predictions[boundary_prediction_index]
            start_error = abs(boundary_prediction.start - actual.start)
            end_error = abs(boundary_prediction.end - actual.end)
        elif predictions:
            boundary_prediction = min(
                predictions, key=lambda item: abs(item.start - actual.start)
            )
            start_error = abs(boundary_prediction.start - actual.start)
            end_error = abs(boundary_prediction.end - actual.end)
        else:
            start_error = end_error = math.inf
        rows.append(
            {
                "strictMatch": index in matches,
                "anyOverlap": covered_seconds > 0,
                "coverage": covered_seconds / duration,
                "fullyContained": any(
                    item.start <= actual.start and item.end >= actual.end
                    for item in predictions
                ),
                "startError": start_error,
                "endError": end_error,
            }
        )
    count = len(rows)
    if not count:
        return {"rallies": 0}
    return {
        "rallies": count,
        "strictMatchRecall": sum(row["strictMatch"] for row in rows) / count,
        "anyOverlapRecall": sum(row["anyOverlap"] for row in rows) / count,
        "meanCoverage": float(np.mean([row["coverage"] for row in rows])),
        "coverageAtLeast95Rate": sum(row["coverage"] >= 0.95 for row in rows) / count,
        "fullyContainedRate": sum(row["fullyContained"] for row in rows) / count,
        "startsWithin025SecondRate": sum(row["startError"] <= 0.25 for row in rows) / count,
        "startsWithin05SecondRate": sum(row["startError"] <= 0.5 for row in rows) / count,
        "startsWithin1SecondRate": sum(row["startError"] <= 1.0 for row in rows) / count,
        "endsWithin025SecondRate": sum(row["endError"] <= 0.25 for row in rows) / count,
        "endsWithin05SecondRate": sum(row["endError"] <= 0.5 for row in rows) / count,
        "endsWithin1SecondRate": sum(row["endError"] <= 1.0 for row in rows) / count,
    }


def outcome_slice_metrics(
    truth: Sequence[TimeInterval],
    predictions: Sequence[TimeInterval],
) -> dict[str, dict[str, Any]]:
    def tags(index: int) -> set[str]:
        raw = getattr(truth[index], "tags", ())
        return {str(tag) for tag in raw}

    slices = {
        "all": list(range(len(truth))),
        "shortAtMost3Seconds": [
            index
            for index, item in enumerate(truth)
            if item.end - item.start <= 3.0
        ],
        "ace": [index for index in range(len(truth)) if "ace" in tags(index)],
        "serviceFault": [
            index for index in range(len(truth)) if "service-fault" in tags(index)
        ],
        "ordinaryLong": [
            index
            for index, item in enumerate(truth)
            if item.end - item.start > 3.0
            and not ({"ace", "service-fault"} & tags(index))
        ],
    }
    return {
        name: truth_slice_metrics(truth, predictions, indexes)
        for name, indexes in slices.items()
    }


def aggregate_outcome_slices(
    per_recording: Sequence[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if not per_recording:
        return {}
    names = sorted({name for item in per_recording for name in item})
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        rows = [item[name] for item in per_recording if name in item]
        total = sum(int(row.get("rallies", 0)) for row in rows)
        aggregate: dict[str, Any] = {"rallies": total}
        keys = sorted({key for row in rows for key in row if key != "rallies"})
        for key in keys:
            if not total:
                continue
            aggregate[key] = sum(
                float(row.get(key, 0.0)) * int(row.get("rallies", 0))
                for row in rows
            ) / total
        result[name] = aggregate
    return result
