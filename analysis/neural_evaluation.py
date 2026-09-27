"""Current-contract, model-independent evaluation for neural experiments.

``evaluate_predictions`` accepts mappings with ``id``, ``sourceGroup``,
``durationSeconds``, ``rallies``, ``predictions``, and optional ``ignoredIntervals``.
Intervals may be schema.Interval objects, start/end mappings, or two-item sequences.
Only supplied intervals are evaluated; this module never opens data or selects models.

The primary objective is pooled F1_padP_coreR at a declared symmetric padding.
The canonical crop helpers own padding, strict short-gap joining, and ignored-time
subtraction. All four required padding cases are always reported. A recording with
no evaluable human time is invalid, including an entirely ignored recording.

Event diagnostics exclude original rallies touched by ignored time, rather than
inventing short rallies from their fragments. Their spans are also censored from
predictions for event diagnostics. Time and original-rally coverage diagnostics
still include every remaining nonignored core second. Boundary errors are measured
only on matched, uncensored events; they are not unconditional boundary accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

from .crop_evaluation import (
    DEFAULT_JOIN_GAP_SECONDS,
    RecordingIntervals,
    evaluate_f1_pad_p_core_r,
    pad_and_merge_intervals,
    subtract_intervals,
)
from .metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    outcome_slice_metrics,
)
from .schema import Interval


PADDING_SECONDS = (0.0, 1.0, 2.0, 3.0)
PRIMARY_PADDING_SECONDS = 2.0


@dataclass(frozen=True)
class _Record:
    intervals: RecordingIntervals
    source_group: str


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _intervals(values: Iterable[Any], duration: float, name: str) -> tuple[Interval, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an interval iterable")
    result: list[Interval] = []
    for index, item in enumerate(values):
        tags: Sequence[str] = ()
        if isinstance(item, Interval):
            start, end, tags = item.start, item.end, item.tags
        elif isinstance(item, Mapping):
            start, end = item.get("start"), item.get("end")
            tags = item.get("tags", ())
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError(f"{name}[{index}] must contain start and end")
        start = _number(start, f"{name}[{index}].start")
        end = _number(end, f"{name}[{index}].end")
        if end <= start:
            raise ValueError(f"{name}[{index}] must have start < end")
        if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence) or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise ValueError(f"{name}[{index}].tags must be a sequence of nonempty strings")
        start, end = max(0.0, start), min(duration, end)
        if start < end:
            result.append(Interval(start, end, tuple(dict.fromkeys(tags))))
    return tuple(sorted(result, key=lambda item: (item.start, item.end)))


def _read_record(row: Mapping[str, Any]) -> _Record:
    for key in ("id", "sourceGroup"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"recording {key} must be a nonempty string")
    duration = _number(row.get("durationSeconds"), "durationSeconds")
    if duration <= 0:
        raise ValueError("durationSeconds must be positive")
    for key in ("rallies", "predictions"):
        if key not in row:
            raise ValueError(f"recording {row['id']!r} is missing {key}")
    truth = _intervals(row["rallies"], duration, "rallies")
    if any(right.start < left.end for left, right in zip(truth, truth[1:])):
        raise ValueError("rallies must not overlap; preserve original event identities")
    return _Record(
        RecordingIntervals(
            id=row["id"],
            split=str(row.get("split", "development")),
            duration=duration,
            truth=truth,
            predictions=_intervals(row["predictions"], duration, "predictions"),
            ignored_intervals=_intervals(row.get("ignoredIntervals", ()), duration, "ignoredIntervals"),
        ),
        source_group=row["sourceGroup"],
    )


def _duration(intervals: Iterable[Interval]) -> float:
    return sum(item.end - item.start for item in intervals)


def _overlaps(left: Interval, right: Interval) -> bool:
    return left.start < right.end and right.start < left.end


def _coverage(record: RecordingIntervals, output: tuple[Interval, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, rally in enumerate(record.truth):
        core = subtract_intervals((rally,), record.ignored_intervals)
        seconds = _duration(core)
        if seconds <= 0:
            continue
        missed = _duration(subtract_intervals(core, output))
        retained = max(0.0, seconds - missed)
        fully_covered = isclose(missed, 0.0, abs_tol=1e-9)
        completely_lost = retained <= 1e-9
        rows.append({
            "recordingId": record.id,
            "truthIndex": index,
            "start": rally.start,
            "end": rally.end,
            "tags": list(rally.tags),
            "evaluableCoreSeconds": seconds,
            "retainedCoreSeconds": retained,
            "coverage": retained / seconds,
            "fullyCovered": fully_covered,
            "completelyLost": completely_lost,
            "partiallyLost": not fully_covered and not completely_lost,
        })
    seconds = sum(item["evaluableCoreSeconds"] for item in rows)
    retained = sum(item["retainedCoreSeconds"] for item in rows)
    return {
        "originalRallies": len(record.truth),
        "evaluableRallies": len(rows),
        "fullyIgnoredRallies": len(record.truth) - len(rows),
        "completeRallyLosses": sum(item["completelyLost"] for item in rows),
        "partialRallyLosses": sum(item["partiallyLost"] for item in rows),
        "fullyCoveredRallies": sum(item["fullyCovered"] for item in rows),
        "evaluableCoreSeconds": seconds,
        "retainedCoreSeconds": retained,
        "coreRecall": retained / seconds if seconds else 0.0,
        "rallies": rows,
    }


def _record_guardrails(record: RecordingIntervals, padding: float, join_gap: float) -> dict[str, Any]:
    core = subtract_intervals(record.truth, record.ignored_intervals)
    output = subtract_intervals(record.predictions, record.ignored_intervals)
    censored_truth = tuple(
        item for item in record.truth
        if any(_overlaps(item, ignored) for ignored in record.ignored_intervals)
    )
    event_truth = tuple(item for item in record.truth if item not in censored_truth)
    event_output = subtract_intervals(
        record.predictions, (*record.ignored_intervals, *censored_truth)
    )
    event_metrics = evaluate_intervals(event_truth, event_output)
    event_metrics["outcomeSlices"] = outcome_slice_metrics(event_truth, event_output)
    exported = subtract_intervals(
        pad_and_merge_intervals(record.predictions, record.duration, padding, join_gap),
        record.ignored_intervals,
    )
    return {
        "eventMetrics": event_metrics,
        "ignoredTouchedRalliesExcludedFromEvents": len(censored_truth),
        "timeMetrics": evaluate_intervals(core, output),
        "coreCoverage": _coverage(record, output),
        "primaryExportCoverage": _coverage(record, exported),
    }


def _pool_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "originalRallies", "evaluableRallies", "fullyIgnoredRallies",
        "completeRallyLosses", "partialRallyLosses", "fullyCoveredRallies",
        "evaluableCoreSeconds", "retainedCoreSeconds",
    )
    result: dict[str, Any] = {key: sum(row[key] for row in rows) for key in keys}
    seconds = result["evaluableCoreSeconds"]
    result["coreRecall"] = result["retainedCoreSeconds"] / seconds if seconds else 0.0
    result["rallies"] = [item for row in rows for item in row["rallies"]]
    return result


def _pool_guardrails(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events = aggregate_evaluations([row["eventMetrics"] for row in rows])
    events["outcomeSlices"] = aggregate_outcome_slices(
        [row["eventMetrics"]["outcomeSlices"] for row in rows]
    )
    time_metrics = aggregate_evaluations([row["timeMetrics"] for row in rows])
    time_coverage = {
        key: time_metrics[key] for key in (
            "timeIoU", "liveTimeRecall", "liveTimePrecision", "trueLiveSeconds",
            "predictedLiveSeconds", "missedLiveSeconds", "deadSecondsRetained",
        )
    }
    if not events["trueRallies"]:
        # An all-censored event set is not evidence of perfect event detection.
        for key in ("eventPrecision", "eventRecall", "eventF1", "eventF1AtIou03", "eventF1AtIou07"):
            events[key] = None
    return {
        **events,
        **time_coverage,
        "eventMetricsAvailable": bool(events["trueRallies"]),
        "eventScope": "original rallies untouched by ignored intervals; censored spans excluded",
        "boundaryScope": "matched uncensored events at IoU >= 0.5",
        "timeScope": "all nonignored core time; no padding or short-gap joining",
        "ignoredTouchedRalliesExcludedFromEvents": sum(
            row["ignoredTouchedRalliesExcludedFromEvents"] for row in rows
        ),
        "timeCoverage": time_coverage,
        "coreCoverage": _pool_coverage([row["coreCoverage"] for row in rows]),
        "primaryExportCoverage": _pool_coverage([row["primaryExportCoverage"] for row in rows]),
    }


def _summary(
    records: Sequence[_Record],
    guards: Mapping[str, Mapping[str, Any]],
    primary_padding: float,
    join_gap: float,
) -> dict[str, Any]:
    padding = evaluate_f1_pad_p_core_r(
        [row.intervals for row in records], PADDING_SECONDS, join_gap_seconds=join_gap
    )
    primary = next(row for row in padding if row["paddingSecondsBeforeAndAfter"] == primary_padding)
    return {
        "recordingCount": len(records),
        "sourceGroupCount": len({row.source_group for row in records}),
        "objective": primary["F1_padP_coreR"],
        "primary": primary,
        "padding": padding,
        "guardrails": _pool_guardrails([guards[row.intervals.id] for row in records]),
    }


def _prepare(
    records: Iterable[Mapping[str, Any]], primary_padding_seconds: float, join_gap_seconds: float
) -> tuple[tuple[_Record, ...], float, float]:
    primary_padding = _number(primary_padding_seconds, "primary_padding_seconds")
    join_gap = _number(join_gap_seconds, "join_gap_seconds")
    if primary_padding not in PADDING_SECONDS:
        raise ValueError("primary_padding_seconds must be one of 0, 1, 2, 3")
    if join_gap < 0:
        raise ValueError("join_gap_seconds must be nonnegative")
    parsed = tuple(_read_record(row) for row in records)
    if not parsed:
        raise ValueError("cannot evaluate an empty recording set")
    ids = [row.intervals.id for row in parsed]
    if len(set(ids)) != len(ids):
        raise ValueError("recording IDs must be unique; do not pool repeated seeds as recordings")
    return parsed, primary_padding, join_gap


def score_predictions(
    records: Iterable[Mapping[str, Any]],
    *,
    primary_padding_seconds: float = PRIMARY_PADDING_SECONDS,
    join_gap_seconds: float = DEFAULT_JOIN_GAP_SECONDS,
) -> float:
    """Return only the canonical primary score for inner epoch/decoder selection.

    This skips event matching and diagnostic sweeps. Use evaluate_predictions for
    every reportable candidate; this scalar is not a complete evaluation artifact.
    """
    parsed, primary_padding, join_gap = _prepare(records, primary_padding_seconds, join_gap_seconds)
    row = evaluate_f1_pad_p_core_r(
        [record.intervals for record in parsed], (primary_padding,), join_gap_seconds=join_gap
    )[0]
    return float(row["F1_padP_coreR"])


def evaluate_predictions(
    records: Iterable[Mapping[str, Any]],
    *,
    primary_padding_seconds: float = PRIMARY_PADDING_SECONDS,
    join_gap_seconds: float = DEFAULT_JOIN_GAP_SECONDS,
) -> dict[str, Any]:
    """Evaluate an explicit comparable recording scope, pooling duration statistics.

    ``objective`` is exactly ``primary['F1_padP_coreR']``. ``sourceGroups`` and
    ``recordings`` are diagnostics, never averages used to calculate the objective.
    Metadata/consent, fold isolation and label-revision checks belong to the caller.
    No protected-test filesystem access is performed or implied by this pure API.
    """
    parsed, primary_padding, join_gap = _prepare(records, primary_padding_seconds, join_gap_seconds)
    # Run the canonical validator before constructing diagnostic guardrails.
    evaluate_f1_pad_p_core_r(
        [row.intervals for row in parsed], (primary_padding,), join_gap_seconds=join_gap
    )
    guards = {
        row.intervals.id: _record_guardrails(row.intervals, primary_padding, join_gap)
        for row in parsed
    }
    result = _summary(parsed, guards, primary_padding, join_gap)
    result["metricContract"] = {
        "version": "neural-evaluation-v1",
        "rankingMetric": "F1_padP_coreR",
        "primaryPaddingSecondsBeforeAndAfter": primary_padding,
        "symmetricPaddingCasesSeconds": list(PADDING_SECONDS),
        "joinGapSeconds": join_gap,
        "joinRule": "positive gap strictly less than threshold; overlaps and touching merge",
        "ignoredRule": "subtract after padding/joining; never rejoin fragments",
        "aggregation": "pooled intersection numerators and duration denominators",
        "eventIouThreshold": 0.5,
    }
    result["sourceGroups"] = {
        group: _summary(
            [row for row in parsed if row.source_group == group], guards, primary_padding, join_gap
        )
        for group in sorted({row.source_group for row in parsed})
    }
    result["recordings"] = [
        {
            "id": row.intervals.id,
            "sourceGroup": row.source_group,
            **_summary((row,), guards, primary_padding, join_gap),
        }
        for row in parsed
    ]
    return result
