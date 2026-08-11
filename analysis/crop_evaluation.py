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
