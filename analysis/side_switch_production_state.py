"""Production rally-state evidence for side-switch appearance specialists.

The helpers in this module are deliberately model-agnostic.  Callers supply the
two production bundles' decoded ranges, serve detections, and score traces.  This
keeps the side-switch feature contract testable without loading the production
artifacts and makes the separation between frozen production evidence and
side-switch labels explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PRODUCTION_MODEL_SOURCES = ("all-labels-v2", "previous-production")
GROUNDING_WINDOW_BEFORE_SECONDS = 1.25
GROUNDING_WINDOW_AFTER_SECONDS = 0.75
SERVE_ASSOCIATION_BEFORE_SECONDS = 0.75
SERVE_ASSOCIATION_AFTER_SECONDS = 3.0
SERVE_CONSENSUS_TOLERANCE_SECONDS = 1.0

STATE_GATE_FEATURE_NAMES = (
    "productionBeforeSupportCount",
    "productionAfterSupportCount",
    "productionMinimumAdjacentSupportCount",
    "productionMinimumAdjacentRallyPeak",
    "productionGapLiveFraction",
    "productionGapMeanRallyScore",
    "productionGapPeakRallyScore",
    "productionGapMeanDeadStateScore",
    "productionGapPeakDeadStateScore",
    "productionGapDurationSeconds",
)

SERVE_ANCHOR_FEATURE_NAMES = (
    "productionBeforeServeSupportCount",
    "productionAfterServeSupportCount",
    "productionMinimumAdjacentServeSupportCount",
    "productionBeforeServeConfidence",
    "productionAfterServeConfidence",
    "productionMinimumAdjacentServeConfidence",
    "productionBeforeAnchorErrorSeconds",
    "productionAfterAnchorErrorSeconds",
    "productionMaximumAdjacentAnchorErrorSeconds",
    "productionDeadToNextServeSeconds",
)

PRODUCTION_STATE_FEATURE_NAMES = (
    *STATE_GATE_FEATURE_NAMES,
    *SERVE_ANCHOR_FEATURE_NAMES,
)

SUPPRESSION_DIAGNOSTIC_NAMES = (
    "suppressionGapMeanScore",
    "suppressionGapPeakScore",
)


class ProductionStateError(ValueError):
    pass


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.end <= self.start
        ):
            raise ProductionStateError("time range must be finite and positive")


@dataclass(frozen=True)
class ScoredTime:
    time: float
    confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.time):
            raise ProductionStateError("scored time must be finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ProductionStateError("scored confidence must be in [0, 1]")


@dataclass(frozen=True)
class ProductionComponent:
    start: float
    end: float
    sources: tuple[str, ...]

    @property
    def support_count(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class RallyEvidence:
    source_start: float
    source_end: float
    component: ProductionComponent | None
    support_count: int
    rally_peak: float
    serve_support_count: int
    serve_confidence: float
    serve_disagreement_seconds: float
    anchor_time: float
    anchor_source: str
    comparison_start: float
    comparison_end: float


@dataclass(frozen=True)
class ProductionTrace:
    times: np.ndarray
    duration: float
    rally_scores: Mapping[str, np.ndarray]
    serve_scores: Mapping[str, np.ndarray]
    dead_state_scores: Mapping[str, np.ndarray]
    ranges: Mapping[str, tuple[TimeRange, ...]]
    serves: Mapping[str, tuple[ScoredTime, ...]]
    suppression_scores: np.ndarray | None = None

    def validate(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ProductionStateError("production times must be a finite vector")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ProductionStateError("production times must increase")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ProductionStateError("production duration must be positive")
        for group_name, group in (
            ("rally", self.rally_scores),
            ("serve", self.serve_scores),
            ("dead-state", self.dead_state_scores),
        ):
            if tuple(sorted(group)) != tuple(sorted(PRODUCTION_MODEL_SOURCES)):
                raise ProductionStateError(
                    f"{group_name} scores must cover both production sources"
                )
            for source, values in group.items():
                array = np.asarray(values, dtype=np.float64)
                if array.shape != times.shape or not np.isfinite(array).all():
                    raise ProductionStateError(
                        f"{group_name} scores for {source} are malformed"
                    )
                if np.any((array < 0.0) | (array > 1.0)):
                    raise ProductionStateError(
                        f"{group_name} scores for {source} leave [0, 1]"
                    )
        if tuple(sorted(self.ranges)) != tuple(sorted(PRODUCTION_MODEL_SOURCES)):
            raise ProductionStateError("decoded ranges must cover both sources")
        if tuple(sorted(self.serves)) != tuple(sorted(PRODUCTION_MODEL_SOURCES)):
            raise ProductionStateError("serve detections must cover both sources")
        if self.suppression_scores is not None:
            suppression = np.asarray(self.suppression_scores, dtype=np.float64)
            if suppression.shape != times.shape or not np.isfinite(suppression).all():
                raise ProductionStateError("suppression scores are malformed")
            if np.any((suppression < 0.0) | (suppression > 1.0)):
                raise ProductionStateError("suppression scores leave [0, 1]")


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def merge_production_components(
    ranges: Mapping[str, Sequence[TimeRange]],
) -> tuple[ProductionComponent, ...]:
    """Mirror the production overlap-connected union while retaining support."""

    if tuple(sorted(ranges)) != tuple(sorted(PRODUCTION_MODEL_SOURCES)):
        raise ProductionStateError("component merge requires both production sources")
    tagged = sorted(
        (float(item.start), float(item.end), source)
        for source, source_ranges in ranges.items()
        for item in source_ranges
    )
    clusters: list[tuple[float, float, set[str]]] = []
    for start, end, source in tagged:
        if not clusters or start >= clusters[-1][1]:
            clusters.append((start, end, {source}))
            continue
        cluster_start, cluster_end, sources = clusters[-1]
        clusters[-1] = (cluster_start, max(cluster_end, end), sources | {source})
    return tuple(
        ProductionComponent(start, end, tuple(sorted(sources)))
        for start, end, sources in clusters
    )


def select_component(
    source: TimeRange, components: Sequence[ProductionComponent]
) -> ProductionComponent | None:
    """Choose the greatest-overlap component; do not invent nearest support."""

    candidates = [
        (
            _overlap(source.start, source.end, component.start, component.end),
            -abs(source.start - component.start),
            -component.start,
            component,
        )
        for component in components
    ]
    supported = [candidate for candidate in candidates if candidate[0] > 0.0]
    return max(supported, key=lambda candidate: candidate[:3])[3] if supported else None


def _comparison_window(anchor: float, duration: float) -> tuple[float, float]:
    requested = GROUNDING_WINDOW_BEFORE_SECONDS + GROUNDING_WINDOW_AFTER_SECONDS
    start = anchor - GROUNDING_WINDOW_BEFORE_SECONDS
    end = anchor + GROUNDING_WINDOW_AFTER_SECONDS
    if start < 0.0:
        end = min(duration, end - start)
        start = 0.0
    if end > duration:
        start = max(0.0, start - (end - duration))
        end = duration
    if end - start < min(requested, duration) - 1e-6:
        raise ProductionStateError("could not construct the grounded comparison window")
    return float(start), float(end)


def _score_peak(
    times: np.ndarray, values: np.ndarray, start: float, end: float
) -> float:
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        center = (start + end) / 2.0
        return float(values[int(np.argmin(np.abs(times - center)))])
    return float(np.max(values[mask]))


def rally_evidence(
    source: TimeRange,
    trace: ProductionTrace,
    components: Sequence[ProductionComponent] | None = None,
) -> RallyEvidence:
    trace.validate()
    available_components = (
        tuple(components)
        if components is not None
        else merge_production_components(trace.ranges)
    )
    component = select_component(source, available_components)
    support_count = component.support_count if component is not None else 0
    evidence_start = component.start if component is not None else source.start
    evidence_end = component.end if component is not None else source.end
    supported_sources = component.sources if component is not None else ()
    rally_peak = (
        min(
            _score_peak(
                trace.times,
                np.asarray(trace.rally_scores[source_name]),
                evidence_start,
                evidence_end,
            )
            for source_name in supported_sources
        )
        if supported_sources
        else 0.0
    )

    contacts: list[tuple[str, ScoredTime]] = []
    association_end = min(
        evidence_end, evidence_start + SERVE_ASSOCIATION_AFTER_SECONDS
    )
    association_start = evidence_start - SERVE_ASSOCIATION_BEFORE_SECONDS
    for source_name in PRODUCTION_MODEL_SOURCES:
        candidates = [
            item
            for item in trace.serves[source_name]
            if association_start <= item.time <= association_end
        ]
        if candidates:
            contacts.append(
                (
                    source_name,
                    max(candidates, key=lambda item: (item.confidence, -item.time)),
                )
            )

    serve_support_count = len(contacts)
    serve_confidence = (
        min(item.confidence for _, item in contacts) if contacts else 0.0
    )
    disagreement = (
        max(item.time for _, item in contacts) - min(item.time for _, item in contacts)
        if len(contacts) > 1
        else 0.0
    )
    if len(contacts) > 1 and disagreement <= SERVE_CONSENSUS_TOLERANCE_SECONDS:
        confidence_sum = sum(item.confidence for _, item in contacts)
        anchor = sum(item.time * item.confidence for _, item in contacts) / max(
            confidence_sum, 1e-9
        )
        anchor_source = "serve-consensus"
    elif contacts:
        anchor_source_name, selected = max(
            contacts, key=lambda item: (item[1].confidence, -item[1].time)
        )
        anchor = selected.time
        anchor_source = f"serve-{anchor_source_name}"
    elif component is not None:
        anchor = component.start
        anchor_source = "ensemble-start"
    else:
        anchor = source.start
        anchor_source = "source-start-fallback"
    comparison_start, comparison_end = _comparison_window(anchor, trace.duration)
    return RallyEvidence(
        source_start=source.start,
        source_end=source.end,
        component=component,
        support_count=support_count,
        rally_peak=rally_peak,
        serve_support_count=serve_support_count,
        serve_confidence=serve_confidence,
        serve_disagreement_seconds=float(disagreement),
        anchor_time=float(anchor),
        anchor_source=anchor_source,
        comparison_start=comparison_start,
        comparison_end=comparison_end,
    )


def _interval_union_duration(
    start: float, end: float, ranges: Sequence[TimeRange]
) -> float:
    clipped = sorted(
        (max(start, item.start), min(end, item.end))
        for item in ranges
        if _overlap(start, end, item.start, item.end) > 0.0
    )
    if not clipped:
        return 0.0
    total = 0.0
    active_start, active_end = clipped[0]
    for item_start, item_end in clipped[1:]:
        if item_start <= active_end:
            active_end = max(active_end, item_end)
        else:
            total += active_end - active_start
            active_start, active_end = item_start, item_end
    return total + active_end - active_start


def _window_scores(
    times: np.ndarray, values: np.ndarray, start: float, end: float
) -> tuple[float, float]:
    mask = (times >= start) & (times < end)
    selected = values[mask]
    if not len(selected):
        selected = values[[int(np.argmin(np.abs(times - (start + end) / 2.0)))]]
    return float(np.mean(selected)), float(np.max(selected))


def gap_state_features(
    before: RallyEvidence,
    after: RallyEvidence,
    gap_start: float,
    gap_end: float,
    trace: ProductionTrace,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return deployable production-state inputs and quarantined suppression scores."""

    trace.validate()
    if not math.isfinite(gap_start) or not math.isfinite(gap_end) or gap_end <= gap_start:
        raise ProductionStateError("candidate gap must be finite and positive")
    gap_duration = gap_end - gap_start
    union_ranges = [
        item for source_ranges in trace.ranges.values() for item in source_ranges
    ]
    live_fraction = _interval_union_duration(
        gap_start, gap_end, union_ranges
    ) / gap_duration
    rally_values = np.maximum(
        np.asarray(trace.rally_scores[PRODUCTION_MODEL_SOURCES[0]]),
        np.asarray(trace.rally_scores[PRODUCTION_MODEL_SOURCES[1]]),
    )
    dead_values = np.maximum(
        np.asarray(trace.dead_state_scores[PRODUCTION_MODEL_SOURCES[0]]),
        np.asarray(trace.dead_state_scores[PRODUCTION_MODEL_SOURCES[1]]),
    )
    rally_mean, rally_peak = _window_scores(
        trace.times, rally_values, gap_start, gap_end
    )
    dead_mean, dead_peak = _window_scores(
        trace.times, dead_values, gap_start, gap_end
    )
    before_anchor_error = abs(before.anchor_time - before.source_start)
    after_anchor_error = abs(after.anchor_time - after.source_start)
    before_end = (
        before.component.end if before.component is not None else before.source_end
    )
    features = {
        "productionBeforeSupportCount": float(before.support_count),
        "productionAfterSupportCount": float(after.support_count),
        "productionMinimumAdjacentSupportCount": float(
            min(before.support_count, after.support_count)
        ),
        "productionMinimumAdjacentRallyPeak": float(
            min(before.rally_peak, after.rally_peak)
        ),
        "productionGapLiveFraction": float(live_fraction),
        "productionGapMeanRallyScore": rally_mean,
        "productionGapPeakRallyScore": rally_peak,
        "productionGapMeanDeadStateScore": dead_mean,
        "productionGapPeakDeadStateScore": dead_peak,
        "productionGapDurationSeconds": float(gap_duration),
        "productionBeforeServeSupportCount": float(before.serve_support_count),
        "productionAfterServeSupportCount": float(after.serve_support_count),
        "productionMinimumAdjacentServeSupportCount": float(
            min(before.serve_support_count, after.serve_support_count)
        ),
        "productionBeforeServeConfidence": float(before.serve_confidence),
        "productionAfterServeConfidence": float(after.serve_confidence),
        "productionMinimumAdjacentServeConfidence": float(
            min(before.serve_confidence, after.serve_confidence)
        ),
        "productionBeforeAnchorErrorSeconds": float(before_anchor_error),
        "productionAfterAnchorErrorSeconds": float(after_anchor_error),
        "productionMaximumAdjacentAnchorErrorSeconds": float(
            max(before_anchor_error, after_anchor_error)
        ),
        "productionDeadToNextServeSeconds": float(after.anchor_time - before_end),
    }
    if tuple(features) != PRODUCTION_STATE_FEATURE_NAMES:
        raise ProductionStateError("production-state feature order changed")
    if not all(math.isfinite(value) for value in features.values()):
        raise ProductionStateError("production-state features must be finite")

    suppression: dict[str, float] = {}
    if trace.suppression_scores is not None:
        suppression_mean, suppression_peak = _window_scores(
            trace.times,
            np.asarray(trace.suppression_scores),
            gap_start,
            gap_end,
        )
        suppression = {
            "suppressionGapMeanScore": suppression_mean,
            "suppressionGapPeakScore": suppression_peak,
        }
    return features, suppression


def rally_evidence_payload(value: RallyEvidence) -> dict[str, Any]:
    return {
        "sourceStart": value.source_start,
        "sourceEnd": value.source_end,
        "component": (
            {
                "start": value.component.start,
                "end": value.component.end,
                "sources": list(value.component.sources),
                "supportCount": value.component.support_count,
            }
            if value.component is not None
            else None
        ),
        "supportCount": value.support_count,
        "rallyPeak": value.rally_peak,
        "serveSupportCount": value.serve_support_count,
        "serveConfidence": value.serve_confidence,
        "serveDisagreementSeconds": value.serve_disagreement_seconds,
        "anchorTime": value.anchor_time,
        "anchorSource": value.anchor_source,
        "comparisonWindow": {
            "start": value.comparison_start,
            "end": value.comparison_end,
        },
    }
