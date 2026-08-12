"""Pure, leakage-auditable overlap-component selection primitives.

This module deliberately stops at the component-selector boundary.  Callers supply
already-generated, source-group-out-of-fold candidates and numeric summaries.  Gold
labels are accepted only by :func:`fit_component_selector`, separately from feature
rows; this module never derives features from annotations and never trains any of the
candidate generators.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .decoder import DecodedInterval
from .dual_serve_fusion_experiment import BoundarySelectorConfig


SELECTOR_SCHEMA_VERSION = 1
FEATURE_SPEC_ID = "component-selector-features-v1"

KEEP_V4 = "keep-v4"
KEEP_V5 = "keep-v5"
INTERSECTION = "intersection"
UNION = "union"
LOCAL_REFINED = "local-refined"

# The order is part of the serialized feature specification and is also the exact
# tie-break order.  Conservative v4 is intentionally first.
ACTION_ORDER = (KEEP_V4, KEEP_V5, INTERSECTION, UNION, LOCAL_REFINED)
ACTION_PRIORITY = {action: index for index, action in enumerate(ACTION_ORDER)}

TOPOLOGY_ORDER = (
    "v4-only",
    "v5-only",
    "one-to-one",
    "one-to-many",
    "many-to-one",
    "many-to-many",
    "local-only",
)

FROZEN_INTERSECTION_CONFIG = BoundarySelectorConfig(
    action=INTERSECTION,
    minimum_pair_iou=0.30,
    maximum_start_delta_seconds=1.0,
    minimum_v5_end_earlier_seconds=1.0,
    minimum_both_confidence=0.80,
)


class LeakageError(ValueError):
    """Raised when a generated feature was trained on its held-out source group."""


def _require_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, order=True)
class GeneratorProvenance:
    """The exact fold artifact which generated an input to a selector row."""

    generator_id: str
    artifact_sha256: str
    training_source_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.generator_id, "generator_id")
        _require_identifier(self.artifact_sha256, "artifact_sha256")
        groups = tuple(sorted(set(self.training_source_groups)))
        if any(not isinstance(group, str) or not group.strip() for group in groups):
            raise ValueError("training_source_groups must contain non-empty strings")
        object.__setattr__(self, "training_source_groups", groups)

    def validate_held_out(self, source_group: str) -> None:
        if source_group in self.training_source_groups:
            raise LeakageError(
                f"generator {self.generator_id!r} ({self.artifact_sha256}) was "
                f"trained on held-out source group {source_group!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generatorId": self.generator_id,
            "artifactSha256": self.artifact_sha256,
            "trainingSourceGroups": list(self.training_source_groups),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratorProvenance:
        try:
            groups = payload.get("trainingSourceGroups", [])
            if not isinstance(groups, list) or any(
                not isinstance(group, str) for group in groups
            ):
                raise TypeError("trainingSourceGroups must be a string array")
            return cls(
                generator_id=str(payload["generatorId"]),
                artifact_sha256=str(payload["artifactSha256"]),
                training_source_groups=tuple(groups),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid generator provenance: {error}") from error


@dataclass(frozen=True, order=True)
class BoundaryProvenance:
    """Inference-time provenance for the two boundaries of a proposal."""

    start_generator: str
    end_generator: str
    operation: str = "decoded"
    parent_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.start_generator, "start_generator")
        _require_identifier(self.end_generator, "end_generator")
        _require_identifier(self.operation, "operation")
        parents = tuple(sorted(set(self.parent_candidate_ids)))
        if any(not isinstance(item, str) or not item.strip() for item in parents):
            raise ValueError("parent_candidate_ids must contain non-empty strings")
        object.__setattr__(self, "parent_candidate_ids", parents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "startGenerator": self.start_generator,
            "endGenerator": self.end_generator,
            "operation": self.operation,
            "parentCandidateIds": list(self.parent_candidate_ids),
        }


@dataclass(frozen=True, order=True)
class RawScoreSummary:
    """A named raw score channel summarized without calibration assumptions."""

    model_id: str
    signal: str
    minimum: float
    maximum: float
    mean: float
    at_start: float
    at_end: float
    sample_count: int

    def __post_init__(self) -> None:
        _require_identifier(self.model_id, "model_id")
        _require_identifier(self.signal, "signal")
        for name in ("minimum", "maximum", "mean", "at_start", "at_end"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.minimum > self.maximum:
            raise ValueError("raw score minimum cannot exceed maximum")
        if self.mean < self.minimum - 1e-12 or self.mean > self.maximum + 1e-12:
            raise ValueError("raw score mean must lie between minimum and maximum")

    @classmethod
    def from_values(
        cls,
        model_id: str,
        signal: str,
        values: Sequence[float],
        *,
        at_start: float | None = None,
        at_end: float | None = None,
    ) -> RawScoreSummary:
        numeric = tuple(_finite(value, "raw score") for value in values)
        if not numeric:
            raise ValueError("raw score values cannot be empty")
        return cls(
            model_id=model_id,
            signal=signal,
            minimum=min(numeric),
            maximum=max(numeric),
            mean=sum(numeric) / len(numeric),
            at_start=numeric[0] if at_start is None else at_start,
            at_end=numeric[-1] if at_end is None else at_end,
            sample_count=len(numeric),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelId": self.model_id,
            "signal": self.signal,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "atStart": self.at_start,
            "atEnd": self.at_end,
            "sampleCount": self.sample_count,
        }


@dataclass(frozen=True)
class TransitionQualitySummary:
    """Caller-computed inference signals around proposed start/end boundaries."""

    start_before: float = 0.0
    start_after: float = 0.0
    end_before: float = 0.0
    end_after: float = 0.0
    end_persistence: float = 0.0
    visibility: float = 0.0
    audio_availability: float = 0.0
    camera_quality: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "start_before",
            "start_after",
            "end_before",
            "end_after",
            "end_persistence",
            "visibility",
            "audio_availability",
            "camera_quality",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 0.0 <= self.audio_availability <= 1.0:
            raise ValueError("audio_availability must be between zero and one")

    def feature_values(self) -> dict[str, float]:
        return {
            "transition:start-before": self.start_before,
            "transition:start-after": self.start_after,
            "transition:start-delta": self.start_after - self.start_before,
            "transition:end-before": self.end_before,
            "transition:end-after": self.end_after,
            "transition:end-delta": self.end_after - self.end_before,
            "transition:end-persistence": self.end_persistence,
            "quality:visibility": self.visibility,
            "quality:audio-availability": self.audio_availability,
            "quality:camera": self.camera_quality,
        }

    def to_dict(self) -> dict[str, float]:
        return {
            "startBefore": self.start_before,
            "startAfter": self.start_after,
            "endBefore": self.end_before,
            "endAfter": self.end_after,
            "endPersistence": self.end_persistence,
            "visibility": self.visibility,
            "audioAvailability": self.audio_availability,
            "cameraQuality": self.camera_quality,
        }


@dataclass(frozen=True, order=True)
class IntervalCandidate:
    """A generated interval.  It contains no gold-derived attributes."""

    candidate_id: str
    generator_id: str
    start: float
    end: float
    confidence: float
    boundary_provenance: BoundaryProvenance

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        _require_identifier(self.generator_id, "generator_id")
        object.__setattr__(self, "start", _finite(self.start, "start"))
        object.__setattr__(self, "end", _finite(self.end, "end"))
        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence"))
        if self.start < 0.0 or self.end <= self.start:
            raise ValueError("candidate interval must have 0 <= start < end")

    @classmethod
    def from_decoded(
        cls,
        candidate_id: str,
        generator_id: str,
        interval: DecodedInterval,
    ) -> IntervalCandidate:
        return cls(
            candidate_id=candidate_id,
            generator_id=generator_id,
            start=interval.start,
            end=interval.end,
            confidence=interval.confidence,
            boundary_provenance=BoundaryProvenance(
                start_generator=generator_id,
                end_generator=generator_id,
                operation="decoded",
                parent_candidate_ids=(candidate_id,),
            ),
        )

    def to_decoded(self) -> DecodedInterval:
        return DecodedInterval(self.start, self.end, self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidateId": self.candidate_id,
            "generatorId": self.generator_id,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "boundaryProvenance": self.boundary_provenance.to_dict(),
        }


def _candidate_key(item: IntervalCandidate) -> tuple[float, float, str, str]:
    return (item.start, item.end, item.generator_id, item.candidate_id)


def _topology(v4_count: int, v5_count: int, local_count: int) -> str:
    if v4_count == 0 and v5_count == 0:
        if local_count:
            return "local-only"
        raise ValueError("an overlap component cannot be empty")
    if v4_count == 0:
        return "v5-only"
    if v5_count == 0:
        return "v4-only"
    if v4_count == 1 and v5_count == 1:
        return "one-to-one"
    if v4_count == 1:
        return "one-to-many"
    if v5_count == 1:
        return "many-to-one"
    return "many-to-many"


def _envelope(items: Sequence[IntervalCandidate]) -> tuple[float, float] | None:
    if not items:
        return None
    return min(item.start for item in items), max(item.end for item in items)


def _interval_overlap(
    first: tuple[float, float] | None, second: tuple[float, float] | None
) -> float:
    if first is None or second is None:
        return 0.0
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


@dataclass(frozen=True)
class ComponentFeatureRow:
    """One strict-overlap component and its inference-time feature summaries."""

    recording_id: str
    source_group: str
    component_id: str
    v4: tuple[IntervalCandidate, ...]
    v5: tuple[IntervalCandidate, ...]
    local_refined: tuple[IntervalCandidate, ...] = ()
    raw_scores: tuple[RawScoreSummary, ...] = ()
    transition_quality: TransitionQualitySummary = field(
        default_factory=TransitionQualitySummary
    )
    generators: tuple[GeneratorProvenance, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.recording_id, "recording_id")
        _require_identifier(self.source_group, "source_group")
        _require_identifier(self.component_id, "component_id")
        for name in ("v4", "v5", "local_refined"):
            ordered = tuple(sorted(tuple(getattr(self, name)), key=_candidate_key))
            object.__setattr__(self, name, ordered)
        scores = tuple(sorted(tuple(self.raw_scores)))
        if len({(row.model_id, row.signal) for row in scores}) != len(scores):
            raise ValueError("raw score model/signal pairs must be unique within a row")
        object.__setattr__(self, "raw_scores", scores)
        generators = tuple(
            sorted(tuple(self.generators), key=lambda row: row.generator_id)
        )
        if len({row.generator_id for row in generators}) != len(generators):
            raise ValueError("generator ids must be unique within a row")
        object.__setattr__(self, "generators", generators)

        all_candidates = self.v4 + self.v5 + self.local_refined
        identities = [(row.generator_id, row.candidate_id) for row in all_candidates]
        if not all_candidates:
            raise ValueError("an overlap component cannot be empty")
        if len(set(identities)) != len(identities):
            raise ValueError("a generated candidate cannot be assigned twice")
        known = {row.generator_id for row in generators}
        referenced = {
            name
            for row in all_candidates
            for name in (
                row.generator_id,
                row.boundary_provenance.start_generator,
                row.boundary_provenance.end_generator,
            )
        } | {row.model_id for row in scores}
        missing = sorted(referenced - known)
        if missing:
            raise ValueError(
                "component is missing generator provenance for: " + ", ".join(missing)
            )

    @property
    def topology(self) -> str:
        return _topology(len(self.v4), len(self.v5), len(self.local_refined))

    @property
    def cardinality(self) -> tuple[int, int, int]:
        return len(self.v4), len(self.v5), len(self.local_refined)

    @property
    def component_envelope(self) -> tuple[float, float]:
        items = self.v4 + self.v5 + self.local_refined
        return min(row.start for row in items), max(row.end for row in items)

    @property
    def duration_seconds(self) -> float:
        start, end = self.component_envelope
        return end - start

    @property
    def start_disagreement_seconds(self) -> float:
        v4_envelope, v5_envelope = _envelope(self.v4), _envelope(self.v5)
        if v4_envelope is None or v5_envelope is None:
            return 0.0
        return v5_envelope[0] - v4_envelope[0]

    @property
    def end_disagreement_seconds(self) -> float:
        v4_envelope, v5_envelope = _envelope(self.v4), _envelope(self.v5)
        if v4_envelope is None or v5_envelope is None:
            return 0.0
        return v5_envelope[1] - v4_envelope[1]

    @property
    def v4_containment(self) -> float:
        """Fraction of the v4 envelope contained by the v5 envelope."""
        v4_envelope, v5_envelope = _envelope(self.v4), _envelope(self.v5)
        if v4_envelope is None:
            return 0.0
        return _interval_overlap(v4_envelope, v5_envelope) / (
            v4_envelope[1] - v4_envelope[0]
        )

    @property
    def v5_containment(self) -> float:
        """Fraction of the v5 envelope contained by the v4 envelope."""
        v4_envelope, v5_envelope = _envelope(self.v4), _envelope(self.v5)
        if v5_envelope is None:
            return 0.0
        return _interval_overlap(v4_envelope, v5_envelope) / (
            v5_envelope[1] - v5_envelope[0]
        )

    @property
    def generator_training_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (row.generator_id, row.training_source_groups) for row in self.generators
        )

    @property
    def generator_hashes(self) -> tuple[tuple[str, str], ...]:
        return tuple((row.generator_id, row.artifact_sha256) for row in self.generators)

    def validate_oof(self) -> None:
        for generator in self.generators:
            generator.validate_held_out(self.source_group)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordingId": self.recording_id,
            "sourceGroup": self.source_group,
            "componentId": self.component_id,
            "topology": self.topology,
            "cardinality": {
                "v4": len(self.v4),
                "v5": len(self.v5),
                "localRefined": len(self.local_refined),
            },
            "v4": [row.to_dict() for row in self.v4],
            "v5": [row.to_dict() for row in self.v5],
            "localRefined": [row.to_dict() for row in self.local_refined],
            "rawScores": [row.to_dict() for row in self.raw_scores],
            "startDisagreementSeconds": self.start_disagreement_seconds,
            "endDisagreementSeconds": self.end_disagreement_seconds,
            "durationSeconds": self.duration_seconds,
            "v4Containment": self.v4_containment,
            "v5Containment": self.v5_containment,
            "transitionQuality": self.transition_quality.to_dict(),
            "generators": [row.to_dict() for row in self.generators],
        }


def build_component_rows(
    v4: Sequence[IntervalCandidate],
    v5: Sequence[IntervalCandidate],
    *,
    recording_id: str,
    source_group: str,
    generators: Sequence[GeneratorProvenance],
    local_refined: Sequence[IntervalCandidate] = (),
) -> tuple[ComponentFeatureRow, ...]:
    """Build deterministic strict-overlap connected components.

    Every supplied interval participates in exactly one component.  Touching
    boundaries do not overlap; transitive strict overlaps do share a component.
    Local refinements take part in connectivity, so one refinement can never be
    silently attached to two pre-existing components.
    """

    tagged: list[tuple[IntervalCandidate, str]] = [
        *((row, "v4") for row in v4),
        *((row, "v5") for row in v5),
        *((row, LOCAL_REFINED) for row in local_refined),
    ]
    identities = [(row.generator_id, row.candidate_id) for row, _ in tagged]
    if len(set(identities)) != len(identities):
        raise ValueError("a generated candidate cannot be supplied more than once")
    tagged.sort(key=lambda pair: (*_candidate_key(pair[0]), pair[1]))

    partitions: list[list[tuple[IntervalCandidate, str]]] = []
    current: list[tuple[IntervalCandidate, str]] = []
    current_end = -math.inf
    for item, role in tagged:
        if current and item.start >= current_end:
            partitions.append(current)
            current = []
            current_end = -math.inf
        current.append((item, role))
        current_end = max(current_end, item.end)
    if current:
        partitions.append(current)

    rows = tuple(
        ComponentFeatureRow(
            recording_id=recording_id,
            source_group=source_group,
            component_id=f"{recording_id}:component:{index:04d}",
            v4=tuple(item for item, role in partition if role == "v4"),
            v5=tuple(item for item, role in partition if role == "v5"),
            local_refined=tuple(
                item for item, role in partition if role == LOCAL_REFINED
            ),
            generators=tuple(generators),
        )
        for index, partition in enumerate(partitions)
    )
    assert_no_double_assignment(rows, expected_candidate_count=len(tagged))
    return rows


# Explicit alias for callers searching for the overlap terminology used by v4/v5.
build_overlap_component_rows = build_component_rows


def assert_no_double_assignment(
    components: Sequence[ComponentFeatureRow],
    *,
    expected_candidate_count: int | None = None,
) -> None:
    component_ids: set[tuple[str, str]] = set()
    assignments: set[tuple[str, str]] = set()
    for component in components:
        component_key = (component.recording_id, component.component_id)
        if component_key in component_ids:
            raise ValueError(f"duplicate component id: {component.component_id!r}")
        component_ids.add(component_key)
        for item in component.v4 + component.v5 + component.local_refined:
            identity = (item.generator_id, item.candidate_id)
            if identity in assignments:
                raise ValueError(
                    f"candidate {item.candidate_id!r} was assigned to multiple "
                    "components"
                )
            assignments.add(identity)
    if (
        expected_candidate_count is not None
        and len(assignments) != expected_candidate_count
    ):
        raise ValueError(
            f"expected {expected_candidate_count} assigned candidates, got "
            f"{len(assignments)}"
        )


@dataclass(frozen=True, order=True)
class ProposedInterval:
    start: float
    end: float
    confidence: float
    boundary_provenance: BoundaryProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _finite(self.start, "start"))
        object.__setattr__(self, "end", _finite(self.end, "end"))
        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence"))
        if self.start < 0.0 or self.end <= self.start:
            raise ValueError("proposed interval must have 0 <= start < end")

    @classmethod
    def from_candidate(cls, candidate: IntervalCandidate) -> ProposedInterval:
        return cls(
            candidate.start,
            candidate.end,
            candidate.confidence,
            candidate.boundary_provenance,
        )

    def to_decoded(self) -> DecodedInterval:
        return DecodedInterval(self.start, self.end, self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "boundaryProvenance": self.boundary_provenance.to_dict(),
        }


def _merged_ranges(
    intervals: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    ordered = sorted(intervals)
    if not ordered:
        return ()
    result: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        prior_start, prior_end = result[-1]
        if start <= prior_end:
            result[-1] = prior_start, max(prior_end, end)
        else:
            result.append((start, end))
    return tuple(result)


def _total_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merged_ranges(intervals))


def _overlap_duration(
    first: Iterable[tuple[float, float]], second: Iterable[tuple[float, float]]
) -> float:
    left, right = _merged_ranges(first), _merged_ranges(second)
    first_index = second_index = 0
    total = 0.0
    while first_index < len(left) and second_index < len(right):
        a_start, a_end = left[first_index]
        b_start, b_end = right[second_index]
        total += max(0.0, min(a_end, b_end) - max(a_start, b_start))
        if a_end <= b_end:
            first_index += 1
        else:
            second_index += 1
    return total


@dataclass(frozen=True)
class CandidateFeatureRow:
    """One selectable action for a component; targets are intentionally absent."""

    recording_id: str
    source_group: str
    component_id: str
    candidate_id: str
    action: str
    intervals: tuple[ProposedInterval, ...]
    topology: str
    v4_count: int
    v5_count: int
    local_refined_count: int
    raw_scores: tuple[RawScoreSummary, ...]
    start_disagreement_seconds: float
    end_disagreement_seconds: float
    component_duration_seconds: float
    duration_seconds: float
    containment_by_v4: float
    containment_by_v5: float
    coverage_of_v4: float
    coverage_of_v5: float
    transition_quality: TransitionQualitySummary
    boundary_provenance: tuple[BoundaryProvenance, ...]
    generators: tuple[GeneratorProvenance, ...]
    uncovered_addition: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.recording_id, "recording_id")
        _require_identifier(self.source_group, "source_group")
        _require_identifier(self.component_id, "component_id")
        _require_identifier(self.candidate_id, "candidate_id")
        if self.action not in ACTION_PRIORITY:
            raise ValueError(f"unsupported component action: {self.action!r}")
        if self.topology not in TOPOLOGY_ORDER:
            raise ValueError(f"unsupported overlap topology: {self.topology!r}")
        for name in ("v4_count", "v5_count", "local_refined_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        intervals = tuple(
            sorted(tuple(self.intervals), key=lambda row: (row.start, row.end))
        )
        object.__setattr__(self, "intervals", intervals)
        provenance = tuple(row.boundary_provenance for row in intervals)
        supplied_provenance = tuple(self.boundary_provenance)
        if supplied_provenance and supplied_provenance != provenance:
            raise ValueError("boundary_provenance must align with proposed intervals")
        object.__setattr__(self, "boundary_provenance", provenance)
        scores = tuple(sorted(tuple(self.raw_scores)))
        if len({(row.model_id, row.signal) for row in scores}) != len(scores):
            raise ValueError("raw score model/signal pairs must be unique within a row")
        object.__setattr__(self, "raw_scores", scores)
        generators = tuple(
            sorted(tuple(self.generators), key=lambda row: row.generator_id)
        )
        if len({row.generator_id for row in generators}) != len(generators):
            raise ValueError("generator ids must be unique within a row")
        object.__setattr__(self, "generators", generators)
        known = {row.generator_id for row in generators}
        referenced = {
            name
            for row in provenance
            for name in (row.start_generator, row.end_generator)
        } | {row.model_id for row in scores}
        missing = sorted(referenced - known)
        if missing:
            raise ValueError(
                "candidate row is missing generator provenance for: "
                + ", ".join(missing)
            )
        for name in (
            "start_disagreement_seconds",
            "end_disagreement_seconds",
            "component_duration_seconds",
            "duration_seconds",
            "containment_by_v4",
            "containment_by_v5",
            "coverage_of_v4",
            "coverage_of_v5",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.component_duration_seconds <= 0.0 or self.duration_seconds < 0.0:
            raise ValueError(
                "component duration must be positive and action duration non-negative"
            )
        for name in (
            "containment_by_v4",
            "containment_by_v5",
            "coverage_of_v4",
            "coverage_of_v5",
        ):
            if not -1e-12 <= getattr(self, name) <= 1.0 + 1e-12:
                raise ValueError(f"{name} must be between zero and one")
        if self.uncovered_addition and (self.v4_count != 0 or not self.intervals):
            raise ValueError(
                "uncovered additions require no v4 input and a non-empty output"
            )

    @property
    def generator_training_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (row.generator_id, row.training_source_groups) for row in self.generators
        )

    @property
    def generator_hashes(self) -> tuple[tuple[str, str], ...]:
        return tuple((row.generator_id, row.artifact_sha256) for row in self.generators)

    def validate_oof(self) -> None:
        for generator in self.generators:
            generator.validate_held_out(self.source_group)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordingId": self.recording_id,
            "sourceGroup": self.source_group,
            "componentId": self.component_id,
            "candidateId": self.candidate_id,
            "action": self.action,
            "intervals": [row.to_dict() for row in self.intervals],
            "topology": self.topology,
            "cardinality": {
                "v4": self.v4_count,
                "v5": self.v5_count,
                "localRefined": self.local_refined_count,
            },
            "rawScores": [row.to_dict() for row in self.raw_scores],
            "startDisagreementSeconds": self.start_disagreement_seconds,
            "endDisagreementSeconds": self.end_disagreement_seconds,
            "componentDurationSeconds": self.component_duration_seconds,
            "durationSeconds": self.duration_seconds,
            "containmentByV4": self.containment_by_v4,
            "containmentByV5": self.containment_by_v5,
            "coverageOfV4": self.coverage_of_v4,
            "coverageOfV5": self.coverage_of_v5,
            "transitionQuality": self.transition_quality.to_dict(),
            "boundaryProvenance": [row.to_dict() for row in self.boundary_provenance],
            "generators": [row.to_dict() for row in self.generators],
            "uncoveredAddition": self.uncovered_addition,
        }


def _provenance_for_combination(
    action: str,
    parents: Sequence[IntervalCandidate],
    *,
    choose_start: str,
    choose_end: str,
) -> BoundaryProvenance:
    start_ordered = sorted(
        parents,
        key=lambda row: (
            row.start if choose_start == "minimum" else -row.start,
            0 if row in parents[: len(parents) // 2] else 1,
            row.generator_id,
            row.candidate_id,
        ),
    )
    end_ordered = sorted(
        parents,
        key=lambda row: (
            row.end if choose_end == "minimum" else -row.end,
            0 if row in parents[: len(parents) // 2] else 1,
            row.generator_id,
            row.candidate_id,
        ),
    )
    return BoundaryProvenance(
        start_generator=start_ordered[0].boundary_provenance.start_generator,
        end_generator=end_ordered[0].boundary_provenance.end_generator,
        operation=action,
        parent_candidate_ids=tuple(row.candidate_id for row in parents),
    )


def _action_interval_sets(
    component: ComponentFeatureRow,
    *,
    permit_union: bool,
    allow_uncovered_additions: bool,
) -> list[tuple[str, tuple[ProposedInterval, ...]]]:
    result: list[tuple[str, tuple[ProposedInterval, ...]]] = [
        (KEEP_V4, tuple(ProposedInterval.from_candidate(row) for row in component.v4))
    ]
    covered = bool(component.v4)
    if component.v5 and (covered or allow_uncovered_additions):
        result.append(
            (
                KEEP_V5,
                tuple(ProposedInterval.from_candidate(row) for row in component.v5),
            )
        )
    if len(component.v4) == 1 and len(component.v5) == 1:
        v4, v5 = component.v4[0], component.v5[0]
        start, end = max(v4.start, v5.start), min(v4.end, v5.end)
        if end > start:
            parents = (v4, v5)
            result.append(
                (
                    INTERSECTION,
                    (
                        ProposedInterval(
                            start,
                            end,
                            min(v4.confidence, v5.confidence),
                            _provenance_for_combination(
                                INTERSECTION,
                                parents,
                                choose_start="maximum",
                                choose_end="minimum",
                            ),
                        ),
                    ),
                )
            )
    if permit_union and component.v4 and component.v5:
        parents = component.v4 + component.v5
        result.append(
            (
                UNION,
                (
                    ProposedInterval(
                        min(row.start for row in parents),
                        max(row.end for row in parents),
                        max(row.confidence for row in parents),
                        _provenance_for_combination(
                            UNION,
                            parents,
                            choose_start="minimum",
                            choose_end="maximum",
                        ),
                    ),
                ),
            )
        )
    if component.local_refined and (covered or allow_uncovered_additions):
        result.append(
            (
                LOCAL_REFINED,
                tuple(
                    ProposedInterval.from_candidate(row)
                    for row in component.local_refined
                ),
            )
        )
    result.sort(key=lambda row: ACTION_PRIORITY[row[0]])
    return result


def candidate_action_rows(
    component: ComponentFeatureRow,
    *,
    permit_union: bool = False,
    allow_uncovered_additions: bool = False,
    raw_scores_by_action: Mapping[str, Sequence[RawScoreSummary]] | None = None,
    transition_quality_by_action: Mapping[str, TransitionQualitySummary] | None = None,
) -> tuple[CandidateFeatureRow, ...]:
    """Materialize deterministic candidate actions for one component.

    The conservative defaults make v5/local-only components produce just an empty
    ``keep-v4`` row.  Consequently a learned selector cannot add uncovered intervals
    unless both row construction and model policy explicitly opt in.
    """

    raw_scores_by_action = raw_scores_by_action or {}
    transition_quality_by_action = transition_quality_by_action or {}
    v4_ranges = tuple((row.start, row.end) for row in component.v4)
    v5_ranges = tuple((row.start, row.end) for row in component.v5)
    v4_duration, v5_duration = _total_duration(v4_ranges), _total_duration(v5_ranges)
    rows: list[CandidateFeatureRow] = []
    for action, intervals in _action_interval_sets(
        component,
        permit_union=permit_union,
        allow_uncovered_additions=allow_uncovered_additions,
    ):
        proposed_ranges = tuple((row.start, row.end) for row in intervals)
        proposed_duration = _total_duration(proposed_ranges)
        overlap_v4 = _overlap_duration(proposed_ranges, v4_ranges)
        overlap_v5 = _overlap_duration(proposed_ranges, v5_ranges)
        rows.append(
            CandidateFeatureRow(
                recording_id=component.recording_id,
                source_group=component.source_group,
                component_id=component.component_id,
                candidate_id=f"{component.component_id}:{action}",
                action=action,
                intervals=intervals,
                topology=component.topology,
                v4_count=len(component.v4),
                v5_count=len(component.v5),
                local_refined_count=len(component.local_refined),
                raw_scores=tuple(
                    raw_scores_by_action.get(action, component.raw_scores)
                ),
                start_disagreement_seconds=component.start_disagreement_seconds,
                end_disagreement_seconds=component.end_disagreement_seconds,
                component_duration_seconds=component.duration_seconds,
                duration_seconds=proposed_duration,
                containment_by_v4=(
                    overlap_v4 / proposed_duration if proposed_duration else 0.0
                ),
                containment_by_v5=(
                    overlap_v5 / proposed_duration if proposed_duration else 0.0
                ),
                coverage_of_v4=overlap_v4 / v4_duration if v4_duration else 0.0,
                coverage_of_v5=overlap_v5 / v5_duration if v5_duration else 0.0,
                transition_quality=transition_quality_by_action.get(
                    action, component.transition_quality
                ),
                boundary_provenance=tuple(
                    row.boundary_provenance for row in intervals
                ),
                generators=component.generators,
                uncovered_addition=not component.v4 and bool(intervals),
            )
        )
    return tuple(rows)


# Short alias for experiment code.
candidate_actions = candidate_action_rows


def validate_oof_rows(
    rows: Sequence[ComponentFeatureRow | CandidateFeatureRow],
) -> None:
    """Reject any row whose held-out source was seen by a feature generator."""

    if not rows:
        raise ValueError("at least one out-of-fold row is required")
    recording_sources: dict[str, str] = {}
    artifact_training_groups: dict[tuple[str, str], tuple[str, ...]] = {}
    for row in rows:
        row.validate_oof()
        prior_source = recording_sources.setdefault(row.recording_id, row.source_group)
        if prior_source != row.source_group:
            raise LeakageError(
                f"recording {row.recording_id!r} has inconsistent source groups"
            )
        for generator in row.generators:
            key = (generator.generator_id, generator.artifact_sha256)
            prior_groups = artifact_training_groups.setdefault(
                key, generator.training_source_groups
            )
            if prior_groups != generator.training_source_groups:
                raise LeakageError(
                    f"generator artifact {key!r} has inconsistent training groups"
                )


def _decoded_iou(first: DecodedInterval, second: DecodedInterval) -> float:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = max(first.end, second.end) - min(first.start, second.start)
    return overlap / union if union > 0.0 else 0.0


def frozen_intersection_action(
    component: ComponentFeatureRow,
    config: BoundarySelectorConfig = FROZEN_INTERSECTION_CONFIG,
) -> str:
    """Return the action selected by the frozen conservative v4/v5 rule."""

    config.validate()
    if config.action not in {KEEP_V4, INTERSECTION}:
        raise ValueError("frozen intersection helper requires keep-v4 or intersection")
    if config.action == KEEP_V4 or len(component.v4) != 1 or len(component.v5) != 1:
        return KEEP_V4
    v4, v5 = component.v4[0].to_decoded(), component.v5[0].to_decoded()
    eligible = (
        _decoded_iou(v4, v5) + 1e-12 >= config.minimum_pair_iou
        and abs(v5.start - v4.start)
        <= config.maximum_start_delta_seconds + 1e-12
        and v4.end - v5.end + 1e-12 >= config.minimum_v5_end_earlier_seconds
        and min(v4.confidence, v5.confidence) + 1e-12
        >= config.minimum_both_confidence
    )
    return INTERSECTION if eligible else KEEP_V4


def apply_frozen_intersection(
    v4: Sequence[DecodedInterval],
    v5: Sequence[DecodedInterval],
    config: BoundarySelectorConfig = FROZEN_INTERSECTION_CONFIG,
) -> tuple[list[DecodedInterval], int]:
    """Pure parity implementation of the frozen ``apply_boundary_selector`` rule."""

    config.validate()
    if config.action not in {KEEP_V4, INTERSECTION}:
        raise ValueError("frozen intersection helper requires keep-v4 or intersection")
    tagged = sorted(
        [(row.start, row.end, 0, row) for row in v4]
        + [(row.start, row.end, 1, row) for row in v5],
        key=lambda row: (row[0], row[1], row[2]),
    )
    partitions: list[list[tuple[float, float, int, DecodedInterval]]] = []
    current: list[tuple[float, float, int, DecodedInterval]] = []
    current_end = -math.inf
    for node in tagged:
        if current and node[0] >= current_end:
            partitions.append(current)
            current = []
            current_end = -math.inf
        current.append(node)
        current_end = max(current_end, node[1])
    if current:
        partitions.append(current)

    selected: list[DecodedInterval] = []
    changed = 0
    for partition in partitions:
        left = [row[3] for row in partition if row[2] == 0]
        right = [row[3] for row in partition if row[2] == 1]
        if config.action == KEEP_V4 or len(left) != 1 or len(right) != 1:
            selected.extend(left)
            continue
        original, candidate = left[0], right[0]
        eligible = (
            _decoded_iou(original, candidate) + 1e-12 >= config.minimum_pair_iou
            and abs(candidate.start - original.start)
            <= config.maximum_start_delta_seconds + 1e-12
            and original.end - candidate.end + 1e-12
            >= config.minimum_v5_end_earlier_seconds
            and min(original.confidence, candidate.confidence) + 1e-12
            >= config.minimum_both_confidence
        )
        if not eligible:
            selected.append(original)
            continue
        intersection = DecodedInterval(
            max(original.start, candidate.start),
            min(original.end, candidate.end),
            min(original.confidence, candidate.confidence),
        )
        if intersection.end <= intersection.start:
            selected.append(original)
            continue
        selected.append(intersection)
        changed += 1
    return sorted(selected, key=lambda row: (row.start, row.end)), changed


# Name which reads naturally in reports/tests.
frozen_intersection_parity = apply_frozen_intersection


def _row_base_features(row: CandidateFeatureRow) -> dict[str, float]:
    confidences = [interval.confidence for interval in row.intervals]
    result = {
        "cardinality:v4": float(row.v4_count),
        "cardinality:v5": float(row.v5_count),
        "cardinality:local-refined": float(row.local_refined_count),
        "cardinality:total": float(
            row.v4_count + row.v5_count + row.local_refined_count
        ),
        "component:duration": row.component_duration_seconds,
        "component:start-disagreement": row.start_disagreement_seconds,
        "component:abs-start-disagreement": abs(row.start_disagreement_seconds),
        "component:end-disagreement": row.end_disagreement_seconds,
        "component:abs-end-disagreement": abs(row.end_disagreement_seconds),
        "candidate:duration": row.duration_seconds,
        "candidate:relative-duration": (
            row.duration_seconds / row.component_duration_seconds
        ),
        "candidate:output-count": float(len(row.intervals)),
        "candidate:empty": float(not row.intervals),
        "candidate:uncovered-addition": float(row.uncovered_addition),
        "containment:by-v4": row.containment_by_v4,
        "containment:by-v5": row.containment_by_v5,
        "containment:coverage-v4": row.coverage_of_v4,
        "containment:coverage-v5": row.coverage_of_v5,
        "confidence:minimum": min(confidences) if confidences else 0.0,
        "confidence:maximum": max(confidences) if confidences else 0.0,
        "confidence:mean": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
    }
    result.update(
        {f"topology:{name}": float(row.topology == name) for name in TOPOLOGY_ORDER}
    )
    result.update(row.transition_quality.feature_values())
    for summary in row.raw_scores:
        prefix = f"raw:{summary.model_id}:{summary.signal}"
        result.update(
            {
                f"{prefix}:present": 1.0,
                f"{prefix}:minimum": summary.minimum,
                f"{prefix}:maximum": summary.maximum,
                f"{prefix}:mean": summary.mean,
                f"{prefix}:at-start": summary.at_start,
                f"{prefix}:at-end": summary.at_end,
                f"{prefix}:sample-count": float(summary.sample_count),
            }
        )
    if row.boundary_provenance:
        divisor = float(len(row.boundary_provenance))
        start_counts = Counter(item.start_generator for item in row.boundary_provenance)
        end_counts = Counter(item.end_generator for item in row.boundary_provenance)
        operation_counts = Counter(item.operation for item in row.boundary_provenance)
        for name, count in sorted(start_counts.items()):
            result[f"boundary:start:{name}"] = count / divisor
        for name, count in sorted(end_counts.items()):
            result[f"boundary:end:{name}"] = count / divisor
        for name, count in sorted(operation_counts.items()):
            result[f"boundary:operation:{name}"] = count / divisor
    return result


def _feature_names(rows: Sequence[CandidateFeatureRow]) -> tuple[str, ...]:
    base_names = sorted(
        {name for row in rows for name in _row_base_features(row)},
    )
    return tuple(
        [f"action={action}" for action in ACTION_ORDER]
        + [
            f"action={action}|{name}"
            for action in ACTION_ORDER
            for name in base_names
        ]
    )


def feature_spec_hash(feature_names: Sequence[str]) -> str:
    payload = {
        "featureSpecId": FEATURE_SPEC_ID,
        "schemaVersion": SELECTOR_SCHEMA_VERSION,
        "actionOrder": list(ACTION_ORDER),
        "featureNames": list(feature_names),
        "scoreSemantics": "uncalibrated-raw-summaries",
        "targetsInRows": False,
        "componentOverlap": "strict-transitive-touching-separate",
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _expanded_features(row: CandidateFeatureRow) -> dict[str, float]:
    prefix = f"action={row.action}"
    result = {prefix: 1.0}
    result.update(
        {f"{prefix}|{name}": value for name, value in _row_base_features(row).items()}
    )
    return result


def _vectorize(
    rows: Sequence[CandidateFeatureRow], feature_names: Sequence[str]
) -> np.ndarray:
    indices = {name: index for index, name in enumerate(feature_names)}
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for name, value in _expanded_features(row).items():
            column = indices.get(name)
            if column is not None:
                matrix[row_index, column] = value
    return matrix


@dataclass(frozen=True, order=True)
class SelectorTarget:
    """A separate fit-only target; never embedded in feature rows."""

    recording_id: str
    component_id: str
    selected_action: str

    def __post_init__(self) -> None:
        _require_identifier(self.recording_id, "recording_id")
        _require_identifier(self.component_id, "component_id")
        if self.selected_action not in ACTION_PRIORITY:
            raise ValueError(f"unsupported selected action: {self.selected_action!r}")


def _component_key(row: CandidateFeatureRow) -> tuple[str, str]:
    return row.recording_id, row.component_id


def _ordered_policy_rows(
    rows: Sequence[CandidateFeatureRow],
    *,
    permit_union: bool,
    allow_uncovered_additions: bool,
) -> tuple[CandidateFeatureRow, ...]:
    allowed = [
        row
        for row in rows
        if (permit_union or row.action != UNION)
        and (allow_uncovered_additions or not row.uncovered_addition)
    ]
    allowed.sort(
        key=lambda row: (
            row.recording_id,
            row.component_id,
            ACTION_PRIORITY[row.action],
            row.candidate_id,
        )
    )
    if not allowed:
        raise ValueError("selector policy removed every candidate row")
    seen: set[tuple[str, str, str]] = set()
    sources: dict[tuple[str, str], str] = {}
    provenance: dict[
        tuple[str, str], tuple[tuple[str, str, tuple[str, ...]], ...]
    ] = {}
    for row in allowed:
        key = _component_key(row)
        prior_source = sources.setdefault(key, row.source_group)
        if prior_source != row.source_group:
            raise ValueError(
                "all action rows for a component must share a source group"
            )
        signature = tuple(
            (
                generator.generator_id,
                generator.artifact_sha256,
                generator.training_source_groups,
            )
            for generator in row.generators
        )
        prior_signature = provenance.setdefault(key, signature)
        if prior_signature != signature:
            raise LeakageError(
                "all action rows for a component must share generator provenance"
            )
        action_key = (*key, row.action)
        if action_key in seen:
            raise ValueError(
                f"component {row.component_id!r} has duplicate action {row.action!r}"
            )
        seen.add(action_key)
    return tuple(allowed)


def _resolve_targets(
    rows: Sequence[CandidateFeatureRow],
    targets: Mapping[Any, str] | Sequence[SelectorTarget] | Sequence[str],
) -> dict[tuple[str, str], str]:
    keys = sorted({_component_key(row) for row in rows})
    resolved: dict[tuple[str, str], str] = {}
    if isinstance(targets, Mapping):
        component_id_counts = Counter(component_id for _, component_id in keys)
        consumed: set[Any] = set()
        for key in keys:
            recording_id, component_id = key
            if key in targets:
                value = targets[key]
                consumed.add(key)
            elif component_id in targets and component_id_counts[component_id] == 1:
                value = targets[component_id]
                consumed.add(component_id)
            else:
                raise ValueError(f"missing target for component {key!r}")
            resolved[key] = str(value)
        extras = set(targets) - consumed
        if extras:
            raise ValueError(
                "targets include unknown components: "
                f"{sorted(extras, key=repr)!r}"
            )
    else:
        supplied = tuple(targets)
        if supplied and all(isinstance(item, SelectorTarget) for item in supplied):
            for item in supplied:
                assert isinstance(item, SelectorTarget)
                key = (item.recording_id, item.component_id)
                if key in resolved:
                    raise ValueError(f"duplicate target for component {key!r}")
                resolved[key] = item.selected_action
            if set(resolved) != set(keys):
                raise ValueError("selector targets do not exactly cover component rows")
        else:
            if len(supplied) != len(keys) or any(
                not isinstance(item, str) for item in supplied
            ):
                raise ValueError(
                    "targets must be a mapping, SelectorTarget sequence, or one "
                    "action string per sorted component"
                )
            resolved = dict(zip(keys, supplied, strict=True))
    available = defaultdict(set)
    for row in rows:
        available[_component_key(row)].add(row.action)
    for key, action in resolved.items():
        if action not in ACTION_PRIORITY:
            raise ValueError(f"unsupported target action: {action!r}")
        if action not in available[key]:
            raise ValueError(f"target action {action!r} is unavailable for {key!r}")
    return resolved


def _conditional_loss_gradient(
    weights: np.ndarray,
    matrix: np.ndarray,
    groups: Sequence[np.ndarray],
    target_indices: Sequence[int],
    l2: float,
) -> tuple[float, np.ndarray]:
    loss = 0.5 * l2 * float(weights @ weights)
    gradient = l2 * weights
    divisor = float(len(groups))
    for indices, target_index in zip(groups, target_indices, strict=True):
        logits = matrix[indices] @ weights
        logits -= float(np.max(logits))
        exp_logits = np.exp(logits)
        probabilities = exp_logits / float(np.sum(exp_logits))
        target_position = int(np.flatnonzero(indices == target_index)[0])
        loss -= math.log(max(float(probabilities[target_position]), 1e-300)) / divisor
        residual = probabilities
        residual[target_position] -= 1.0
        gradient += (matrix[indices].T @ residual) / divisor
    return loss, gradient


@dataclass(frozen=True)
class ComponentSelectorModel:
    """A deterministic L2-regularized conditional linear selector."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    feature_spec_sha256: str
    l2: float
    iterations: int
    converged: bool
    training_source_groups: tuple[str, ...]
    generator_provenance: tuple[GeneratorProvenance, ...]
    training_component_count: int
    training_row_count: int
    target_action_counts: tuple[tuple[str, int], ...]
    permit_union: bool = False
    allow_uncovered_additions: bool = False

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        object.__setattr__(self, "feature_names", names)
        if not names or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty and unique")
        for field_name in ("mean", "scale", "weights"):
            values = tuple(
                _finite(value, field_name) for value in getattr(self, field_name)
            )
            if len(values) != len(names):
                raise ValueError("selector parameter dimensions do not agree")
            object.__setattr__(self, field_name, values)
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("selector feature scales must be positive")
        expected_hash = feature_spec_hash(names)
        if self.feature_spec_sha256 != expected_hash:
            raise ValueError("selector feature specification hash does not match")
        object.__setattr__(
            self,
            "training_source_groups",
            tuple(sorted(set(self.training_source_groups))),
        )
        object.__setattr__(
            self,
            "generator_provenance",
            tuple(sorted(set(self.generator_provenance))),
        )
        counts = tuple(sorted(tuple(self.target_action_counts)))
        if any(action not in ACTION_PRIORITY or count < 0 for action, count in counts):
            raise ValueError("invalid target action counts")
        object.__setattr__(self, "target_action_counts", counts)
        if self.l2 <= 0.0 or not math.isfinite(self.l2):
            raise ValueError("selector L2 regularization must be finite and positive")
        if self.iterations < 0 or self.training_component_count <= 0:
            raise ValueError("invalid selector training counts")
        if self.training_row_count < self.training_component_count:
            raise ValueError(
                "training row count cannot be smaller than component count"
            )

    @property
    def spec_hash(self) -> str:
        return self.feature_spec_sha256

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    def score_rows(
        self, rows: Sequence[CandidateFeatureRow]
    ) -> tuple[tuple[CandidateFeatureRow, float], ...]:
        if not rows:
            return ()
        matrix = _vectorize(rows, self.feature_names)
        normalized = (matrix - np.asarray(self.mean)) / np.asarray(self.scale)
        scores = normalized @ np.asarray(self.weights)
        return tuple(
            (row, float(score)) for row, score in zip(rows, scores, strict=True)
        )

    def select(
        self, rows: Sequence[CandidateFeatureRow]
    ) -> tuple[CandidateFeatureRow, ...]:
        ordered = _ordered_policy_rows(
            rows,
            permit_union=self.permit_union,
            allow_uncovered_additions=self.allow_uncovered_additions,
        )
        scored = self.score_rows(ordered)
        grouped: dict[tuple[str, str], list[tuple[CandidateFeatureRow, float]]] = (
            defaultdict(list)
        )
        for row, score in scored:
            grouped[_component_key(row)].append((row, score))
        return tuple(
            deterministic_choice(
                [row for row, _ in grouped[key]],
                [score for _, score in grouped[key]],
            )
            for key in sorted(grouped)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SELECTOR_SCHEMA_VERSION,
            "modelType": "regularized-conditional-linear-component-selector",
            "featureSpecId": FEATURE_SPEC_ID,
            "featureSpecSha256": self.feature_spec_sha256,
            "actionOrder": list(ACTION_ORDER),
            "featureNames": list(self.feature_names),
            "normalization": {
                "mean": list(self.mean),
                "scale": list(self.scale),
            },
            "weights": list(self.weights),
            "policy": {
                "permitUnion": self.permit_union,
                "allowUncoveredAdditions": self.allow_uncovered_additions,
            },
            "training": {
                "fitInput": "caller-supplied-source-group-OOF-rows",
                "l2": self.l2,
                "iterations": self.iterations,
                "converged": self.converged,
                "componentCount": self.training_component_count,
                "rowCount": self.training_row_count,
                "sourceGroups": list(self.training_source_groups),
                "targetActionCounts": {
                    action: count for action, count in self.target_action_counts
                },
                "generatorProvenance": [
                    row.to_dict() for row in self.generator_provenance
                ],
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComponentSelectorModel:
        try:
            if payload.get("schemaVersion") != SELECTOR_SCHEMA_VERSION:
                raise ValueError("unsupported selector schema version")
            if (
                payload.get("modelType")
                != "regularized-conditional-linear-component-selector"
            ):
                raise ValueError("unsupported selector model type")
            if payload.get("featureSpecId") != FEATURE_SPEC_ID:
                raise ValueError("unsupported selector feature specification")
            if payload.get("actionOrder") != list(ACTION_ORDER):
                raise ValueError("selector action order does not match")
            feature_names = payload["featureNames"]
            normalization = payload["normalization"]
            policy = payload["policy"]
            training = payload["training"]
            provenance = training["generatorProvenance"]
            counts = training["targetActionCounts"]
            if not isinstance(feature_names, list) or any(
                not isinstance(name, str) for name in feature_names
            ):
                raise TypeError("featureNames must be a string array")
            if not isinstance(normalization, Mapping):
                raise TypeError("normalization must be an object")
            if not isinstance(policy, Mapping) or not isinstance(training, Mapping):
                raise TypeError("policy and training must be objects")
            if not isinstance(provenance, list) or not isinstance(counts, Mapping):
                raise TypeError("invalid training provenance metadata")
            if not isinstance(policy.get("permitUnion"), bool) or not isinstance(
                policy.get("allowUncoveredAdditions"), bool
            ):
                raise TypeError("selector policy flags must be booleans")
            return cls(
                feature_names=tuple(feature_names),
                mean=tuple(float(row) for row in normalization["mean"]),
                scale=tuple(float(row) for row in normalization["scale"]),
                weights=tuple(float(row) for row in payload["weights"]),
                feature_spec_sha256=str(payload["featureSpecSha256"]),
                l2=float(training["l2"]),
                iterations=int(training["iterations"]),
                converged=bool(training["converged"]),
                training_source_groups=tuple(training["sourceGroups"]),
                generator_provenance=tuple(
                    GeneratorProvenance.from_dict(row) for row in provenance
                ),
                training_component_count=int(training["componentCount"]),
                training_row_count=int(training["rowCount"]),
                target_action_counts=tuple(
                    (str(action), int(count)) for action, count in counts.items()
                ),
                permit_union=bool(policy["permitUnion"]),
                allow_uncovered_additions=bool(
                    policy["allowUncoveredAdditions"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid component selector model: {error}") from error


def deterministic_choice(
    rows: Sequence[CandidateFeatureRow], scores: Sequence[float]
) -> CandidateFeatureRow:
    """Choose the highest score; exact ties use ACTION_ORDER then candidate id."""

    if not rows or len(rows) != len(scores):
        raise ValueError("rows and scores must be non-empty and aligned")
    keys = {_component_key(row) for row in rows}
    if len(keys) != 1:
        raise ValueError("deterministic_choice requires exactly one component")
    numeric = tuple(_finite(score, "selector score") for score in scores)
    return min(
        zip(rows, numeric, strict=True),
        key=lambda pair: (
            -pair[1],
            ACTION_PRIORITY[pair[0].action],
            pair[0].candidate_id,
        ),
    )[0]


def fit_component_selector(
    rows: Sequence[CandidateFeatureRow],
    targets: Mapping[Any, str] | Sequence[SelectorTarget] | Sequence[str],
    *,
    l2: float = 0.1,
    max_iterations: int = 5000,
    tolerance: float = 1e-6,
    permit_union: bool = False,
    allow_uncovered_additions: bool = False,
) -> ComponentSelectorModel:
    """Fit a conditional linear selector on caller-supplied OOF rows only.

    Target actions are consumed only after row provenance has passed the held-out
    check.  Full-batch gradient descent with deterministic backtracking minimizes
    grouped softmax loss plus positive L2 regularization.
    """

    if not math.isfinite(l2) or l2 <= 0.0:
        raise ValueError("l2 must be finite and positive")
    if max_iterations <= 0 or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive")
    validate_oof_rows(rows)
    ordered = _ordered_policy_rows(
        rows,
        permit_union=permit_union,
        allow_uncovered_additions=allow_uncovered_additions,
    )
    resolved_targets = _resolve_targets(ordered, targets)
    names = _feature_names(ordered)
    raw_matrix = _vectorize(ordered, names)
    mean = np.mean(raw_matrix, axis=0)
    scale = np.std(raw_matrix, axis=0)
    scale[scale < 1e-12] = 1.0
    matrix = (raw_matrix - mean) / scale

    row_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        row_indices[_component_key(row)].append(index)
    groups: list[np.ndarray] = []
    target_indices: list[int] = []
    for key in sorted(row_indices):
        indices = np.asarray(row_indices[key], dtype=np.int64)
        target_action = resolved_targets[key]
        target_index = next(
            index for index in indices if ordered[int(index)].action == target_action
        )
        groups.append(indices)
        target_indices.append(int(target_index))

    weights = np.zeros(len(names), dtype=np.float64)
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        loss, gradient = _conditional_loss_gradient(
            weights, matrix, groups, target_indices, l2
        )
        gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if gradient_norm <= tolerance:
            converged = True
            break
        squared_norm = float(gradient @ gradient)
        step = 1.0
        while step >= 2.0**-30:
            proposal = weights - step * gradient
            proposal_loss, _ = _conditional_loss_gradient(
                proposal, matrix, groups, target_indices, l2
            )
            if proposal_loss <= loss - 1e-4 * step * squared_norm:
                weights = proposal
                break
            step *= 0.5
        else:
            break

    generator_provenance = tuple(
        sorted({generator for row in ordered for generator in row.generators})
    )
    target_counts = Counter(resolved_targets.values())
    return ComponentSelectorModel(
        feature_names=names,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
        feature_spec_sha256=feature_spec_hash(names),
        l2=float(l2),
        iterations=iterations,
        converged=converged,
        training_source_groups=tuple(sorted({row.source_group for row in ordered})),
        generator_provenance=generator_provenance,
        training_component_count=len(groups),
        training_row_count=len(ordered),
        target_action_counts=tuple(sorted(target_counts.items())),
        permit_union=permit_union,
        allow_uncovered_additions=allow_uncovered_additions,
    )


# Concise aliases for experiment code without obscuring the explicit public names.
fit_selector = fit_component_selector
SelectorModel = ComponentSelectorModel


def materialize_selected(
    selected: Sequence[CandidateFeatureRow],
) -> tuple[DecodedInterval, ...]:
    """Materialize one chosen action per component in deterministic time order."""

    seen_components: set[tuple[str, str]] = set()
    intervals: list[DecodedInterval] = []
    for row in selected:
        key = _component_key(row)
        if key in seen_components:
            raise ValueError(
                f"component {row.component_id!r} was selected more than once"
            )
        seen_components.add(key)
        intervals.extend(interval.to_decoded() for interval in row.intervals)
    return tuple(
        sorted(intervals, key=lambda row: (row.start, row.end, row.confidence))
    )


__all__ = [
    "ACTION_ORDER",
    "BoundaryProvenance",
    "CandidateFeatureRow",
    "ComponentFeatureRow",
    "ComponentSelectorModel",
    "FEATURE_SPEC_ID",
    "FROZEN_INTERSECTION_CONFIG",
    "GeneratorProvenance",
    "INTERSECTION",
    "IntervalCandidate",
    "KEEP_V4",
    "KEEP_V5",
    "LOCAL_REFINED",
    "LeakageError",
    "ProposedInterval",
    "RawScoreSummary",
    "SELECTOR_SCHEMA_VERSION",
    "SelectorModel",
    "SelectorTarget",
    "TransitionQualitySummary",
    "UNION",
    "apply_frozen_intersection",
    "assert_no_double_assignment",
    "build_component_rows",
    "build_overlap_component_rows",
    "candidate_action_rows",
    "candidate_actions",
    "deterministic_choice",
    "feature_spec_hash",
    "fit_component_selector",
    "fit_selector",
    "frozen_intersection_action",
    "frozen_intersection_parity",
    "materialize_selected",
    "validate_oof_rows",
]
