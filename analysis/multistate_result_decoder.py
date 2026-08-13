"""Latent immediate-result branches for the serve-anchored state decoder.

The public model remains ``DEAD/SETUP/SERVE/LIVE``.  This decoder privately
duplicates ``LIVE`` into ordinary and immediate-result branches so each branch
can have its own duration prior.  Both branches consume the same public LIVE
emission and project back to public LIVE in the returned path.

The branch probability is consumed exactly once, at the decoded SERVE sample:
``log(1 - q_t)`` scores the ordinary branch and ``log(q_t)`` scores the result
branch.  The decoder deliberately does not clip ignored intervals; callers can
apply the same post-decode clipping used by the base multistate pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .multistate import (
    STATE_ORDER,
    MultistateDecoderConfig,
    MultistateInterval,
    MultistateState,
    StateDurationPrior,
)


class ResultLatentState(IntEnum):
    """Private decoder states; values are stable only within this module."""

    DEAD = 0
    SETUP = 1
    SERVE = 2
    LIVE_ORDINARY = 3
    LIVE_RESULT = 4


_LATENT_ORDER = tuple(ResultLatentState)
_PUBLIC_STATE: Mapping[ResultLatentState, MultistateState] = MappingProxyType(
    {
        ResultLatentState.DEAD: MultistateState.DEAD,
        ResultLatentState.SETUP: MultistateState.SETUP,
        ResultLatentState.SERVE: MultistateState.SERVE,
        ResultLatentState.LIVE_ORDINARY: MultistateState.LIVE,
        ResultLatentState.LIVE_RESULT: MultistateState.LIVE,
    }
)
_ALLOWED_TRANSITIONS: Mapping[
    ResultLatentState, frozenset[ResultLatentState]
] = MappingProxyType(
    {
        ResultLatentState.DEAD: frozenset(
            (ResultLatentState.DEAD, ResultLatentState.SETUP)
        ),
        ResultLatentState.SETUP: frozenset(
            (
                ResultLatentState.SETUP,
                ResultLatentState.SERVE,
                ResultLatentState.DEAD,
            )
        ),
        ResultLatentState.SERVE: frozenset(
            (
                ResultLatentState.LIVE_ORDINARY,
                ResultLatentState.LIVE_RESULT,
            )
        ),
        ResultLatentState.LIVE_ORDINARY: frozenset(
            (ResultLatentState.LIVE_ORDINARY, ResultLatentState.DEAD)
        ),
        ResultLatentState.LIVE_RESULT: frozenset(
            (ResultLatentState.LIVE_RESULT, ResultLatentState.DEAD)
        ),
    }
)


@dataclass(frozen=True)
class ResultDecoderConfig:
    """Base decoder values plus separate priors for the two LIVE branches."""

    base: MultistateDecoderConfig
    ordinary_live_prior: StateDurationPrior
    result_live_prior: StateDurationPrior

    def validate(self) -> None:
        if not isinstance(self.base, MultistateDecoderConfig):
            raise ValueError("result decoder base has the wrong type")
        self.base.validate()
        for label, prior in (
            ("ordinary", self.ordinary_live_prior),
            ("result", self.result_live_prior),
        ):
            if not isinstance(prior, StateDurationPrior):
                raise ValueError(f"{label} LIVE duration prior has the wrong type")
            prior.validate(MultistateState.LIVE)


@dataclass(frozen=True)
class ResultDecode:
    """Best latent path, its public projection, and branch diagnostics.

    Duration tuples count only samples assigned to the corresponding latent
    LIVE branch; the preceding SERVE sample is not included.
    """

    intervals: tuple[MultistateInterval, ...]
    states: tuple[MultistateState, ...]
    latent_states: tuple[ResultLatentState, ...]
    ordinary_branch_count: int
    result_branch_count: int
    ordinary_durations_samples: tuple[int, ...]
    result_durations_samples: tuple[int, ...]
    path_log_score: float


def _validate_times(times: np.ndarray) -> None:
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("result decoder times must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("result decoder times must be strictly increasing")


def _duration_priors(
    config: ResultDecoderConfig,
) -> dict[ResultLatentState, StateDurationPrior]:
    return {
        ResultLatentState.DEAD: config.base.duration_priors[MultistateState.DEAD],
        ResultLatentState.SETUP: config.base.duration_priors[MultistateState.SETUP],
        ResultLatentState.SERVE: config.base.duration_priors[MultistateState.SERVE],
        ResultLatentState.LIVE_ORDINARY: config.ordinary_live_prior,
        ResultLatentState.LIVE_RESULT: config.result_live_prior,
    }


def _duration_slot_caps(
    priors: Mapping[ResultLatentState, StateDurationPrior],
) -> dict[ResultLatentState, int]:
    caps: dict[ResultLatentState, int] = {}
    for state in _LATENT_ORDER:
        prior = priors[state]
        if state not in _ALLOWED_TRANSITIONS[state]:
            caps[state] = 1
        elif prior.maximum_samples is not None:
            caps[state] = prior.maximum_samples
        else:
            # The final slot represents this age and every greater age because
            # all later extensions use the constant linear tail increment.
            caps[state] = max(len(prior.log_scores) + 1, prior.minimum_samples)
    return caps


def _base_transition_score(
    config: ResultDecoderConfig,
    source: ResultLatentState,
    destination: ResultLatentState,
) -> float:
    return config.base.transition_score(
        _PUBLIC_STATE[source], _PUBLIC_STATE[destination]
    )


def _branch_log_score(
    source: ResultLatentState,
    destination: ResultLatentState,
    serve_sample: int,
    result_probabilities: np.ndarray,
) -> float:
    if source != ResultLatentState.SERVE:
        return 0.0
    probability = float(result_probabilities[serve_sample])
    if destination == ResultLatentState.LIVE_ORDINARY:
        return math.log1p(-probability)
    if destination == ResultLatentState.LIVE_RESULT:
        return math.log(probability)
    raise RuntimeError("latent SERVE may transition only to a LIVE branch")


def _intervals_from_public_states(
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
            intervals.append(MultistateInterval(open_start, end))
            open_start = None
    if open_start is not None:
        raise RuntimeError("decoded an event without a closing DEAD transition")
    return tuple(intervals)


def _branch_durations(
    states: Sequence[ResultLatentState],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordinary: list[int] = []
    result: list[int] = []
    start = 0
    while start < len(states):
        state = states[start]
        end = start + 1
        while end < len(states) and states[end] == state:
            end += 1
        if state == ResultLatentState.LIVE_ORDINARY:
            ordinary.append(end - start)
        elif state == ResultLatentState.LIVE_RESULT:
            result.append(end - start)
        start = end
    return tuple(ordinary), tuple(result)


def decode_multistate_result(
    times: np.ndarray,
    state_log_scores: np.ndarray,
    result_probabilities: np.ndarray,
    config: ResultDecoderConfig,
) -> ResultDecode:
    """Decode a legal path with one latent LIVE branch after every SERVE.

    ``state_log_scores`` retains the public ``(samples, 4)`` ``STATE_ORDER``
    shape. Both latent LIVE states consume its LIVE column. The best path must
    begin and end in DEAD, and direct ``SERVE -> DEAD`` is never legal.
    """

    config.validate()
    time_values = np.asarray(times, dtype=np.float64)
    _validate_times(time_values)
    scores = np.asarray(state_log_scores, dtype=np.float64)
    expected_shape = (len(time_values), len(STATE_ORDER))
    if scores.shape != expected_shape:
        raise ValueError(
            "result decoder state log scores must have shape "
            f"{expected_shape} in STATE_ORDER column order"
        )
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("result decoder scores may be finite or negative infinity")

    probabilities = np.asarray(result_probabilities, dtype=np.float64)
    if probabilities.shape != (len(time_values),):
        raise ValueError(
            "result probabilities must have one value per decoder sample"
        )
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities <= 0.0) | (probabilities >= 1.0)
    ):
        raise ValueError(
            "result probabilities must be finite and strictly between zero and one"
        )

    if not len(time_values):
        return ResultDecode(
            intervals=(),
            states=(),
            latent_states=(),
            ordinary_branch_count=0,
            result_branch_count=0,
            ordinary_durations_samples=(),
            result_durations_samples=(),
            path_log_score=0.0,
        )

    priors = _duration_priors(config)
    caps = _duration_slot_caps(priors)
    slot_states: list[ResultLatentState] = []
    slot_ages: list[int] = []
    slot_for: dict[tuple[ResultLatentState, int], int] = {}
    for state in _LATENT_ORDER:
        for age in range(1, caps[state] + 1):
            slot_for[state, age] = len(slot_states)
            slot_states.append(state)
            slot_ages.append(age)

    backpointers = np.full(
        (len(time_values), len(slot_states)), -1, dtype=np.int32
    )
    previous_scores = np.full(len(slot_states), -math.inf, dtype=np.float64)
    dead_slot = slot_for[ResultLatentState.DEAD, 1]
    previous_scores[dead_slot] = (
        scores[0, int(MultistateState.DEAD)]
        + priors[ResultLatentState.DEAD].score_at(1)
    )

    for sample_index in range(1, len(time_values)):
        current_scores = np.full(len(slot_states), -math.inf, dtype=np.float64)
        for previous_slot, previous_score in enumerate(previous_scores):
            if np.isneginf(previous_score):
                continue
            source = slot_states[previous_slot]
            age = slot_ages[previous_slot]
            prior = priors[source]
            cap = caps[source]

            if source in _ALLOWED_TRANSITIONS[source]:
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
                        + _base_transition_score(config, source, source)
                        + increment
                        + scores[sample_index, int(_PUBLIC_STATE[source])]
                    )
                    if candidate > current_scores[destination_slot]:
                        current_scores[destination_slot] = candidate
                        backpointers[sample_index, destination_slot] = previous_slot
                elif age < prior.maximum_samples:
                    destination_slot = slot_for[source, age + 1]
                    candidate = (
                        previous_score
                        + _base_transition_score(config, source, source)
                        + prior.increment_at(age + 1)
                        + scores[sample_index, int(_PUBLIC_STATE[source])]
                    )
                    if candidate > current_scores[destination_slot]:
                        current_scores[destination_slot] = candidate
                        backpointers[sample_index, destination_slot] = previous_slot

            if age < prior.minimum_samples:
                continue
            for destination in _LATENT_ORDER:
                if (
                    destination == source
                    or destination not in _ALLOWED_TRANSITIONS[source]
                ):
                    continue
                destination_slot = slot_for[destination, 1]
                candidate = (
                    previous_score
                    + _base_transition_score(config, source, destination)
                    + _branch_log_score(
                        source,
                        destination,
                        sample_index - 1,
                        probabilities,
                    )
                    + priors[destination].score_at(1)
                    + scores[sample_index, int(_PUBLIC_STATE[destination])]
                )
                if candidate > current_scores[destination_slot]:
                    current_scores[destination_slot] = candidate
                    backpointers[sample_index, destination_slot] = previous_slot
        previous_scores = current_scores

    final_slots = [
        slot_for[ResultLatentState.DEAD, age]
        for age in range(
            priors[ResultLatentState.DEAD].minimum_samples,
            caps[ResultLatentState.DEAD] + 1,
        )
    ]
    best_final_slot = final_slots[0]
    for candidate_slot in final_slots[1:]:
        if previous_scores[candidate_slot] > previous_scores[best_final_slot]:
            best_final_slot = candidate_slot
    best_score = float(previous_scores[best_final_slot])
    if np.isneginf(best_score):
        raise ValueError(
            "result decoder scores and duration constraints admit no DEAD-to-DEAD path"
        )

    decoded_slots = np.empty(len(time_values), dtype=np.int32)
    decoded_slots[-1] = best_final_slot
    for sample_index in range(len(time_values) - 1, 0, -1):
        previous_slot = int(backpointers[sample_index, decoded_slots[sample_index]])
        if previous_slot < 0:
            raise RuntimeError("result decoder encountered an incomplete backpointer")
        decoded_slots[sample_index - 1] = previous_slot

    latent_states = tuple(slot_states[int(slot)] for slot in decoded_slots)
    public_states = tuple(_PUBLIC_STATE[state] for state in latent_states)
    ordinary_durations, result_durations = _branch_durations(latent_states)
    return ResultDecode(
        intervals=_intervals_from_public_states(time_values, public_states),
        states=public_states,
        latent_states=latent_states,
        ordinary_branch_count=len(ordinary_durations),
        result_branch_count=len(result_durations),
        ordinary_durations_samples=ordinary_durations,
        result_durations_samples=result_durations,
        path_log_score=best_score,
    )


__all__ = [
    "ResultDecode",
    "ResultDecoderConfig",
    "ResultLatentState",
    "decode_multistate_result",
]
