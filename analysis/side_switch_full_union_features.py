"""Sampling contracts for score-compatible full-union side-switch features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FRAMES_PER_SEQUENCE = 7
INTERNAL_FLANK_OUTER_SECONDS = 4.0
INTERNAL_FLANK_INNER_SECONDS = 1.0
SUPPORTED_CANDIDATE_KINDS = (
    "adjacent-rally-boundary",
    "internal-dead-state-peak",
)


@dataclass(frozen=True)
class ComparisonWindow:
    start: float
    end: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.end <= self.start
        ):
            raise ValueError("comparison window must be finite and positive")

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}


def whole_rally_sample_times(
    start: float, end: float, frames: int = FRAMES_PER_SEQUENCE
) -> tuple[float, ...]:
    """Mirror the frozen V4/V5 8%-to-92% whole-rally sampling contract."""

    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("rally interval must be finite and positive")
    if frames < 3:
        raise ValueError("a side comparison needs at least three frames")
    duration = end - start
    first = start + min(0.20, duration * 0.08)
    last = end - min(0.15, duration * 0.08)
    if last <= first:
        first = start + duration * 0.20
        last = start + duration * 0.80
    return tuple(float(value) for value in np.linspace(first, last, frames))


def internal_flank_windows(
    transition_time: float,
    source_start: float,
    source_end: float,
    *,
    outer_seconds: float = INTERNAL_FLANK_OUTER_SECONDS,
    inner_seconds: float = INTERNAL_FLANK_INNER_SECONDS,
) -> tuple[ComparisonWindow, ComparisonWindow]:
    """Return fixed stable flanks wholly contained in one decoded range."""

    values = (transition_time, source_start, source_end, outer_seconds, inner_seconds)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("internal flank inputs must be finite")
    if source_end <= source_start or outer_seconds <= inner_seconds or inner_seconds < 0:
        raise ValueError("internal flank geometry is invalid")
    before = ComparisonWindow(
        transition_time - outer_seconds, transition_time - inner_seconds
    )
    after = ComparisonWindow(
        transition_time + inner_seconds, transition_time + outer_seconds
    )
    tolerance = 1e-6
    if before.start < source_start - tolerance or after.end > source_end + tolerance:
        raise ValueError("internal candidate lacks the declared stable flanks")
    return before, after


def candidate_windows(
    candidate: Mapping[str, Any], ranges: Sequence[Mapping[str, Any]]
) -> tuple[ComparisonWindow, ComparisonWindow, int | None]:
    """Resolve visual comparison windows and the associated range boundary index."""

    kind = str(candidate.get("kind", ""))
    if kind not in SUPPORTED_CANDIDATE_KINDS:
        raise ValueError(f"unsupported side-switch candidate kind: {kind}")
    parsed = [
        (
            str(value.get("id", "")),
            float(value.get("start", math.nan)),
            float(value.get("end", math.nan)),
        )
        for value in ranges
    ]
    if any(
        not range_id or not math.isfinite(start) or not math.isfinite(end) or end <= start
        for range_id, start, end in parsed
    ):
        raise ValueError("decoded ranges are invalid")

    if kind == "adjacent-rally-boundary":
        event_id = str(candidate.get("eventId", ""))
        matches = [
            index
            for index, (left, right) in enumerate(zip(parsed, parsed[1:], strict=False))
            if event_id.endswith(f":{left[0]}:{right[0]}")
        ]
        if len(matches) != 1:
            raise ValueError("boundary candidate does not identify one adjacent pair")
        index = matches[0]
        left, right = parsed[index], parsed[index + 1]
        return (
            ComparisonWindow(left[1], left[2]),
            ComparisonWindow(right[1], right[2]),
            index,
        )

    source_id = str(candidate.get("sourceRangeId", ""))
    matches = [value for value in parsed if value[0] == source_id]
    if len(matches) != 1:
        raise ValueError("internal candidate does not identify one decoded range")
    _, source_start, source_end = matches[0]
    before, after = internal_flank_windows(
        float(candidate["transitionTime"]), source_start, source_end
    )
    return before, after, None

