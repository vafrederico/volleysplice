from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    outcome_slice_metrics,
)
from .schema import Interval


@dataclass(frozen=True)
class RecordingIntervals:
    id: str
    split: str
    duration: float
    truth: tuple[Interval, ...]
    predictions: tuple[Interval, ...]
    ignored_intervals: tuple[Interval, ...] = ()


def pad_and_merge_intervals(
    intervals: Iterable[Interval],
    duration: float,
    padding_seconds: float,
) -> tuple[Interval, ...]:
    """Apply symmetric edit padding and merge crops that now touch or overlap."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if padding_seconds < 0:
        raise ValueError("padding_seconds cannot be negative")
    merged: list[Interval] = []
    for item in sorted(intervals, key=lambda interval: (interval.start, interval.end)):
        start = max(0.0, item.start - padding_seconds)
        end = min(duration, item.end + padding_seconds)
        if end <= start:
            continue
        if merged and start <= merged[-1].end:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, end))
        else:
            merged.append(Interval(start, end))
    return tuple(merged)


def subtract_intervals(
    intervals: Iterable[Interval],
    excluded: Iterable[Interval],
) -> tuple[Interval, ...]:
    """Return the interval union after removing the excluded interval union."""
    source = _merge_intervals(intervals)
    ignored = _merge_intervals(excluded)
    if not ignored:
        return source
    kept: list[Interval] = []
    ignored_index = 0
    for item in source:
        cursor = item.start
        while ignored_index < len(ignored) and ignored[ignored_index].end <= cursor:
            ignored_index += 1
        index = ignored_index
        while index < len(ignored) and ignored[index].start < item.end:
            cut = ignored[index]
            if cut.start > cursor:
                kept.append(Interval(cursor, min(cut.start, item.end), item.tags))
            cursor = max(cursor, cut.end)
            if cursor >= item.end:
                break
            index += 1
        if cursor < item.end:
            kept.append(Interval(cursor, item.end, item.tags))
    return tuple(kept)


def _merge_intervals(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    merged: list[Interval] = []
    for item in sorted(intervals, key=lambda interval: (interval.start, interval.end)):
        if item.end <= item.start:
            continue
        if merged and item.start <= merged[-1].end:
            tags = tuple(dict.fromkeys((*merged[-1].tags, *item.tags)))
            merged[-1] = Interval(
                merged[-1].start,
                max(merged[-1].end, item.end),
                tags,
            )
        else:
            merged.append(Interval(item.start, item.end, item.tags))
    return tuple(merged)


def _duration(intervals: Iterable[Interval]) -> float:
    return sum(item.end - item.start for item in intervals)


def _intersection_duration(
    left: Iterable[Interval],
    right: Iterable[Interval],
) -> float:
    first = _merge_intervals(left)
    second = _merge_intervals(right)
    first_index = 0
    second_index = 0
    total = 0.0
    while first_index < len(first) and second_index < len(second):
        first_item = first[first_index]
        second_item = second[second_index]
        total += max(
            0.0,
            min(first_item.end, second_item.end)
            - max(first_item.start, second_item.start),
        )
        if first_item.end <= second_item.end:
            first_index += 1
        else:
            second_index += 1
    return total


def evaluate_f1_pad_p_core_r(
    recordings: Sequence[RecordingIntervals],
    padding_seconds: Sequence[float],
) -> list[dict[str, Any]]:
    """Calculate the pooled model-ranking metric for symmetric export padding."""
    if not recordings:
        raise ValueError("cannot evaluate an empty recording set")
    if not padding_seconds:
        raise ValueError("at least one padding value is required")
    rows: list[dict[str, Any]] = []
    for padding in padding_seconds:
        if padding < 0:
            raise ValueError("padding values cannot be negative")
        precision_numerator = 0.0
        precision_denominator = 0.0
        recall_numerator = 0.0
        recall_denominator = 0.0
        padded_human_seconds = 0.0
        output_crop_count = 0
        per_recording: list[dict[str, Any]] = []
        for recording in recordings:
            core_human = subtract_intervals(
                recording.truth,
                recording.ignored_intervals,
            )
            core_human_seconds = _duration(core_human)
            if core_human_seconds <= 0:
                raise ValueError(
                    f"recording {recording.id!r} has no evaluable core human-label time"
                )
            padded_model = subtract_intervals(
                pad_and_merge_intervals(
                    recording.predictions,
                    recording.duration,
                    float(padding),
                ),
                recording.ignored_intervals,
            )
            padded_human = subtract_intervals(
                pad_and_merge_intervals(
                    recording.truth,
                    recording.duration,
                    float(padding),
                ),
                recording.ignored_intervals,
            )
            model_seconds = _duration(padded_model)
            human_padded_seconds = _duration(padded_human)
            padded_intersection = _intersection_duration(padded_model, padded_human)
            core_intersection = _intersection_duration(padded_model, core_human)
            precision_numerator += padded_intersection
            precision_denominator += model_seconds
            recall_numerator += core_intersection
            recall_denominator += core_human_seconds
            padded_human_seconds += human_padded_seconds
            output_crop_count += len(padded_model)
            per_recording.append(
                {
                    "id": recording.id,
                    "paddedPrecisionIntersectionSeconds": padded_intersection,
                    "paddedModelExportSeconds": model_seconds,
                    "coreRecallIntersectionSeconds": core_intersection,
                    "coreHumanSeconds": core_human_seconds,
                    "paddedHumanExportSeconds": human_padded_seconds,
                }
            )
        padded_precision = (
            precision_numerator / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        core_recall = recall_numerator / recall_denominator
        f1 = (
            2 * padded_precision * core_recall / (padded_precision + core_recall)
            if padded_precision + core_recall > 0
            else 0.0
        )
        rows.append(
            {
                "paddingSecondsBeforeAndAfter": float(padding),
                "P_pad": padded_precision,
                "R_core": core_recall,
                "F1_padP_coreR": f1,
                "paddedPrecisionIntersectionSeconds": precision_numerator,
                "paddedModelExportSeconds": precision_denominator,
                "coreRecallIntersectionSeconds": recall_numerator,
                "coreHumanSeconds": recall_denominator,
                "paddedHumanExportSeconds": padded_human_seconds,
                "exportDurationDifferenceSeconds": (
                    precision_denominator - padded_human_seconds
                ),
                "inputCropCount": sum(len(item.predictions) for item in recordings),
                "outputCropCount": output_crop_count,
                "recordingCount": len(recordings),
                "recordings": per_recording,
            }
        )
    return rows


def evaluate_crop_padding(
    recordings: Sequence[RecordingIntervals],
    padding_seconds: Sequence[float],
) -> list[dict[str, Any]]:
    """Evaluate practical exported crops at each symmetric padding setting."""
    if not recordings:
        raise ValueError("cannot evaluate an empty recording set")
    if not padding_seconds:
        raise ValueError("at least one padding value is required")
    total_video_seconds = sum(item.duration for item in recordings)
    rows: list[dict[str, Any]] = []
    for padding in padding_seconds:
        if padding < 0:
            raise ValueError("padding values cannot be negative")
        per_recording: list[dict[str, Any]] = []
        input_crop_count = 0
        output_crop_count = 0
        for recording in recordings:
            input_crop_count += len(recording.predictions)
            predictions = pad_and_merge_intervals(
                recording.predictions,
                recording.duration,
                float(padding),
            )
            output_crop_count += len(predictions)
            metrics = evaluate_intervals(recording.truth, predictions)
            metrics["id"] = recording.id
            metrics["outcomeSlices"] = outcome_slice_metrics(
                recording.truth, predictions
            )
            per_recording.append(metrics)
        aggregate = aggregate_evaluations(per_recording)
        aggregate["outcomeSlices"] = aggregate_outcome_slices(
            [item["outcomeSlices"] for item in per_recording]
        )
        retained = float(aggregate["predictedLiveSeconds"])
        rows.append(
            {
                "paddingSecondsBeforeAndAfter": float(padding),
                "aggregate": aggregate,
                "retainedVideoSeconds": retained,
                "retainedVideoRate": retained / total_video_seconds,
                "removedVideoSeconds": total_video_seconds - retained,
                "removedVideoRate": 1.0 - retained / total_video_seconds,
                "inputCropCount": input_crop_count,
                "outputCropCount": output_crop_count,
                "cropMergeRate": (
                    1.0 - output_crop_count / input_crop_count
                    if input_crop_count
                    else 0.0
                ),
                "recordings": per_recording,
            }
        )

    baseline = rows[0]
    baseline_metrics = baseline["aggregate"]
    for row in rows:
        metrics = row["aggregate"]
        row["deltaFromFirstSetting"] = {
            "liveTimeRecallPoints": 100
            * (metrics["liveTimeRecall"] - baseline_metrics["liveTimeRecall"]),
            "liveTimePrecisionPoints": 100
            * (metrics["liveTimePrecision"] - baseline_metrics["liveTimePrecision"]),
            "timeIoUPoints": 100 * (metrics["timeIoU"] - baseline_metrics["timeIoU"]),
            "eventF1Points": 100 * (metrics["eventF1"] - baseline_metrics["eventF1"]),
            "missedLiveSeconds": metrics["missedLiveSeconds"]
            - baseline_metrics["missedLiveSeconds"],
            "deadSecondsRetained": metrics["deadSecondsRetained"]
            - baseline_metrics["deadSecondsRetained"],
            "retainedVideoSeconds": row["retainedVideoSeconds"]
            - baseline["retainedVideoSeconds"],
        }
    return rows
