"""Versioned source-quality exclusions shared by serving-side pipelines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXCLUSION_KIND = "volleycut-serving-side-source-quality-exclusions-v1"


@dataclass(frozen=True)
class SourceQualityInterval:
    start: float
    end: float
    reason: str


def _number(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def load_source_quality_exclusions(
    path: Path,
) -> dict[str, tuple[SourceQualityInterval, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schemaVersion") != 1
        or payload.get("kind") != EXCLUSION_KIND
        or not isinstance(payload.get("records"), list)
    ):
        raise ValueError(f"invalid serving-side source exclusions: {path}")
    result: dict[str, tuple[SourceQualityInterval, ...]] = {}
    for record_index, record in enumerate(payload["records"]):
        where = f"records[{record_index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{where} must be an object")
        recording_id = record.get("recordingId")
        intervals = record.get("intervals")
        duration = _number(record.get("durationSeconds"), f"{where}.durationSeconds")
        if not isinstance(recording_id, str) or not recording_id or recording_id in result:
            raise ValueError(f"{where}.recordingId is invalid or duplicated")
        if duration <= 0 or not isinstance(intervals, list):
            raise ValueError(f"{where} has an invalid duration or interval list")
        parsed: list[SourceQualityInterval] = []
        previous_end = -1.0
        for interval_index, interval in enumerate(intervals):
            interval_where = f"{where}.intervals[{interval_index}]"
            if not isinstance(interval, Mapping):
                raise ValueError(f"{interval_where} must be an object")
            start = _number(interval.get("start"), f"{interval_where}.start")
            end = _number(interval.get("end"), f"{interval_where}.end")
            reason = interval.get("reason")
            if (
                start < 0
                or start >= end
                or end > duration
                or start < previous_end
                or not isinstance(reason, str)
                or not reason
            ):
                raise ValueError(f"{interval_where} is invalid or overlaps")
            parsed.append(SourceQualityInterval(start, end, reason))
            previous_end = end
        result[recording_id] = tuple(parsed)
    return result


def samples_touch_source_exclusion(
    exclusions: Mapping[str, Sequence[SourceQualityInterval]],
    recording_id: str,
    sample_times: Sequence[float],
) -> bool:
    """Return true when any sampled instant is outside the usable source view."""

    return any(
        interval.start <= float(sample_time) < interval.end
        for interval in exclusions.get(recording_id, ())
        for sample_time in sample_times
    )
