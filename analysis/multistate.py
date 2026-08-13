from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np


class MultistateState(IntEnum):
    """Ordered categorical states used by the short-event sequence model."""

    DEAD = 0
    SETUP = 1
    SERVE = 2
    LIVE = 3


STATE_ORDER = tuple(MultistateState)

ALLOWED_TRANSITIONS: Mapping[MultistateState, frozenset[MultistateState]] = (
    MappingProxyType(
        {
            MultistateState.DEAD: frozenset(
                (MultistateState.DEAD, MultistateState.SETUP)
            ),
            MultistateState.SETUP: frozenset(
                (
                    MultistateState.SETUP,
                    MultistateState.SERVE,
                    MultistateState.DEAD,
                )
            ),
            MultistateState.SERVE: frozenset(
                (MultistateState.LIVE, MultistateState.DEAD)
            ),
            MultistateState.LIVE: frozenset(
                (MultistateState.LIVE, MultistateState.DEAD)
            ),
        }
    )
)


class TimeInterval(Protocol):
    start: float
    end: float


@dataclass(frozen=True)
class StateDurationPrior:
    """Fold-estimated log scores for a single contiguous state run.

    ``log_scores[d - 1]`` is the total prior log score for a run of ``d``
    samples. Durations beyond the supplied table add ``tail_log_score`` per
    additional sample. A finite maximum is a hard constraint. This compact
    representation lets the decoder retain exact semi-Markov duration scores
    without a quadratic scan over every possible segment start.
    """

    log_scores: tuple[float, ...] = (0.0,)
    tail_log_score: float = 0.0
    minimum_samples: int = 1
    maximum_samples: int | None = None

    def validate(self, state: MultistateState | None = None) -> None:
        label = state.name if state is not None else "state"
        if not self.log_scores or any(
            not math.isfinite(float(value)) for value in self.log_scores
        ):
            raise ValueError(f"{label} duration log scores must be finite and non-empty")
        if not math.isfinite(self.tail_log_score):
            raise ValueError(f"{label} duration tail log score must be finite")
        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(self.minimum_samples, int)
            or self.minimum_samples < 1
        ):
            raise ValueError(f"{label} minimum duration must be a positive integer")
        if self.maximum_samples is not None and (
            isinstance(self.maximum_samples, bool)
            or not isinstance(self.maximum_samples, int)
            or self.maximum_samples < self.minimum_samples
        ):
            raise ValueError(
                f"{label} maximum duration must be an integer at least the minimum"
            )
        if state == MultistateState.SERVE and (
            self.minimum_samples != 1 or self.maximum_samples != 1
        ):
            raise ValueError("SERVE duration must be fixed at exactly one sample")

    def score_at(self, samples: int) -> float:
        if samples < 1:
            raise ValueError("duration must contain at least one sample")
        if self.maximum_samples is not None and samples > self.maximum_samples:
            return -math.inf
        if samples <= len(self.log_scores):
            return float(self.log_scores[samples - 1])
        return float(
            self.log_scores[-1]
            + (samples - len(self.log_scores)) * self.tail_log_score
        )

    def increment_at(self, samples: int) -> float:
        """Return the prior increment incurred by extending to ``samples``."""

        if samples == 1:
            return self.score_at(1)
        return self.score_at(samples) - self.score_at(samples - 1)


@dataclass(frozen=True)
class MultistateDecoderConfig:
    """All non-emission decoder values, supplied by an outer training fold."""

    duration_priors: Mapping[MultistateState, StateDurationPrior]
    transition_log_scores: Mapping[
        tuple[MultistateState, MultistateState], float
    ] = field(default_factory=dict)

    def validate(self) -> None:
        supplied_states = set(self.duration_priors)
        expected_states = set(STATE_ORDER)
        if supplied_states != expected_states:
            missing = sorted(state.name for state in expected_states - supplied_states)
            extra = sorted(str(state) for state in supplied_states - expected_states)
            raise ValueError(
                "duration priors must contain every multistate state exactly once "
                f"(missing={missing}, extra={extra})"
            )
        for state in STATE_ORDER:
            prior = self.duration_priors[state]
            if not isinstance(prior, StateDurationPrior):
                raise ValueError(f"{state.name} duration prior has the wrong type")
            prior.validate(state)

        for transition, score in self.transition_log_scores.items():
            if not isinstance(transition, tuple) or len(transition) != 2:
                raise ValueError("transition log-score keys must be (from_state, to_state)")
            source, destination = transition
            if source not in expected_states or destination not in expected_states:
                raise ValueError("transition log-score keys must use multistate states")
            if destination not in ALLOWED_TRANSITIONS[source]:
                raise ValueError(
                    f"transition {source.name}->{destination.name} is not allowed"
                )
            if not math.isfinite(float(score)):
                raise ValueError("transition log scores must be finite")

    def transition_score(
        self,
        source: MultistateState,
        destination: MultistateState,
    ) -> float:
        return float(self.transition_log_scores.get((source, destination), 0.0))


@dataclass(frozen=True)
class MultistateInterval:
    """A half-open ``[start, end)`` serve-contact-to-dead-ball interval."""

    start: float
    end: float

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class MultistateDecode:
    intervals: tuple[MultistateInterval, ...]
    states: tuple[MultistateState, ...]
    path_log_score: float


def _validate_times(times: np.ndarray, label: str) -> None:
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError(f"{label} times must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{label} times must be strictly increasing")


def _validated_intervals(
    rallies: Sequence[TimeInterval],
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for rally in rallies:
        start = float(rally.start)
        end = float(rally.end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("multistate rally boundaries must be finite")
        if end <= start:
            raise ValueError("multistate rally ends must be after their starts")
        intervals.append((start, end))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:], strict=False):
        if current[0] < previous[1]:
            raise ValueError("multistate rallies must not overlap")
    return intervals


def build_state_targets(
    times: np.ndarray,
    rallies: Sequence[TimeInterval],
    *,
    setup_radius_seconds: float,
    serve_radius_seconds: float,
) -> np.ndarray:
    """Build categorical state targets from rally boundaries only.

    The fixed setup window is ``[serve - setup_radius, serve)`` and the fixed
    serve pulse is ``[serve - serve_radius, serve + serve_radius)``. Rally live
    time is the annotation's half-open ``[start, end)`` interval. Preparation
    for a later rally is clamped to the preceding rally end, so a future serve
    cannot relabel the preceding rally. Overlap precedence is deterministic:
    ``SERVE`` wins over ``LIVE``, which wins over ``SETUP``, which wins over
    ``DEAD``. If a serve pulse misses the sampling grid, its nearest eligible
    sample is selected, matching the existing serve-target behavior.

    Outcome tags and other interval attributes are deliberately not read.
    """

    _validate_times(times, "multistate target")
    for label, value in (
        ("setup", setup_radius_seconds),
        ("serve", serve_radius_seconds),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"multistate {label} radius must be non-negative")

    intervals = _validated_intervals(rallies)
    setup = np.zeros(len(times), dtype=bool)
    live = np.zeros(len(times), dtype=bool)
    serve = np.zeros(len(times), dtype=bool)

    for index, (start, end) in enumerate(intervals):
        previous_end = intervals[index - 1][1] if index else -math.inf
        preparation_lower = max(previous_end, start - setup_radius_seconds)
        serve_lower = max(previous_end, start - serve_radius_seconds)
        serve_upper = min(end, start + serve_radius_seconds)

        setup |= (times >= preparation_lower) & (times < start)
        live |= (times >= start) & (times < end)
        selected_serve = (times >= serve_lower) & (times < serve_upper)
        if not np.any(selected_serve) and len(times):
            eligible = np.flatnonzero((times >= previous_end) & (times < end))
            if len(eligible):
                nearest = eligible[int(np.argmin(np.abs(times[eligible] - start)))]
                selected_serve[nearest] = True
        serve |= selected_serve

    targets = np.full(len(times), int(MultistateState.DEAD), dtype=np.int8)
    targets[setup] = int(MultistateState.SETUP)
    targets[live] = int(MultistateState.LIVE)
    targets[serve] = int(MultistateState.SERVE)
    return targets


def _duration_slot_caps(
    config: MultistateDecoderConfig,
) -> dict[MultistateState, int]:
    caps: dict[MultistateState, int] = {}
    for state in STATE_ORDER:
        prior = config.duration_priors[state]
        if state not in ALLOWED_TRANSITIONS[state]:
            caps[state] = 1
        elif prior.maximum_samples is not None:
            caps[state] = prior.maximum_samples
        else:
            # The final bucket represents this age and every greater age. It
            # is exact because all later increments use the linear tail score.
            caps[state] = max(len(prior.log_scores) + 1, prior.minimum_samples)
    return caps


def _intervals_from_states(
    times: np.ndarray,
    states: Sequence[MultistateState],
) -> tuple[MultistateInterval, ...]:
    intervals: list[MultistateInterval] = []
    open_start: float | None = None
    for index, state in enumerate(states):
        if state == MultistateState.SERVE:
            if open_start is not None:
                raise RuntimeError("decoded a serve before the preceding event closed")
            open_start = float(times[index])
        elif state == MultistateState.DEAD and open_start is not None:
            end = float(times[index])
            if end <= open_start:
                raise RuntimeError("decoded a non-positive multistate interval")
            intervals.append(MultistateInterval(start=open_start, end=end))
            open_start = None
    if open_start is not None:
        raise RuntimeError("decoded an event without a closing DEAD transition")
    return tuple(intervals)


def decode_multistate(
    times: np.ndarray,
    state_log_scores: np.ndarray,
    config: MultistateDecoderConfig,
) -> MultistateDecode:
    """Decode the best legal state path and its half-open rally intervals.

    Score columns must follow ``STATE_ORDER`` (equivalently, the integer value
    of ``MultistateState``). The decoder always starts and finishes in ``DEAD``.
    It consumes only state log scores and fold-supplied configuration; labels,
    outcomes, and annotated boundaries are not accepted at inference time.
    Exact ties retain the first path encountered in state/duration order.
    """

    config.validate()
    _validate_times(times, "multistate decode")
    scores = np.asarray(state_log_scores, dtype=np.float64)
    expected_shape = (len(times), len(STATE_ORDER))
    if scores.shape != expected_shape:
        raise ValueError(
            "multistate log scores must have shape "
            f"{expected_shape} in STATE_ORDER column order"
        )
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("multistate log scores may be finite or negative infinity")
    if len(times) == 0:
        return MultistateDecode(intervals=(), states=(), path_log_score=0.0)

    caps = _duration_slot_caps(config)
    slot_states: list[MultistateState] = []
    slot_ages: list[int] = []
    slot_for: dict[tuple[MultistateState, int], int] = {}
    for state in STATE_ORDER:
        for age in range(1, caps[state] + 1):
            slot_for[state, age] = len(slot_states)
            slot_states.append(state)
            slot_ages.append(age)

    slot_count = len(slot_states)
    backpointers = np.full((len(times), slot_count), -1, dtype=np.int32)
    previous_scores = np.full(slot_count, -math.inf, dtype=np.float64)
    dead_slot = slot_for[MultistateState.DEAD, 1]
    dead_prior = config.duration_priors[MultistateState.DEAD]
    previous_scores[dead_slot] = (
        scores[0, int(MultistateState.DEAD)] + dead_prior.score_at(1)
    )

    for sample_index in range(1, len(times)):
        current_scores = np.full(slot_count, -math.inf, dtype=np.float64)
        for previous_slot, previous_score in enumerate(previous_scores):
            if np.isneginf(previous_score):
                continue
            source = slot_states[previous_slot]
            age = slot_ages[previous_slot]
            prior = config.duration_priors[source]
            cap = caps[source]

            if source in ALLOWED_TRANSITIONS[source]:
                if prior.maximum_samples is None:
                    next_age = min(age + 1, cap)
                    increment = (
                        prior.tail_log_score
                        if age == cap
                        else prior.increment_at(age + 1)
                    )
                    destination_slot = slot_for[source, next_age]
                    candidate = (
                        previous_score
                        + config.transition_score(source, source)
                        + increment
                        + scores[sample_index, int(source)]
                    )
                    if candidate > current_scores[destination_slot]:
                        current_scores[destination_slot] = candidate
                        backpointers[sample_index, destination_slot] = previous_slot
                elif age < prior.maximum_samples:
                    destination_slot = slot_for[source, age + 1]
                    candidate = (
                        previous_score
                        + config.transition_score(source, source)
                        + prior.increment_at(age + 1)
                        + scores[sample_index, int(source)]
                    )
                    if candidate > current_scores[destination_slot]:
                        current_scores[destination_slot] = candidate
                        backpointers[sample_index, destination_slot] = previous_slot

            if age < prior.minimum_samples:
                continue
            for destination in STATE_ORDER:
                if destination == source or destination not in ALLOWED_TRANSITIONS[source]:
                    continue
                destination_slot = slot_for[destination, 1]
                destination_prior = config.duration_priors[destination]
                candidate = (
                    previous_score
                    + config.transition_score(source, destination)
                    + destination_prior.score_at(1)
                    + scores[sample_index, int(destination)]
                )
                if candidate > current_scores[destination_slot]:
                    current_scores[destination_slot] = candidate
                    backpointers[sample_index, destination_slot] = previous_slot
        previous_scores = current_scores

    final_slots = [
        slot_for[MultistateState.DEAD, age]
        for age in range(
            config.duration_priors[MultistateState.DEAD].minimum_samples,
            caps[MultistateState.DEAD] + 1,
        )
    ]
    best_final_slot = final_slots[0]
    for candidate_slot in final_slots[1:]:
        if previous_scores[candidate_slot] > previous_scores[best_final_slot]:
            best_final_slot = candidate_slot
    best_score = float(previous_scores[best_final_slot])
    if np.isneginf(best_score):
        raise ValueError(
            "multistate scores and duration constraints admit no DEAD-to-DEAD path"
        )

    decoded_slots = np.empty(len(times), dtype=np.int32)
    decoded_slots[-1] = best_final_slot
    for sample_index in range(len(times) - 1, 0, -1):
        previous_slot = int(backpointers[sample_index, decoded_slots[sample_index]])
        if previous_slot < 0:
            raise RuntimeError("multistate decoder encountered an incomplete backpointer")
        decoded_slots[sample_index - 1] = previous_slot
    decoded_states = tuple(slot_states[int(slot)] for slot in decoded_slots)
    intervals = _intervals_from_states(times, decoded_states)
    return MultistateDecode(
        intervals=intervals,
        states=decoded_states,
        path_log_score=best_score,
    )
