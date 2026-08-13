"""Optional edge-local evidence for the public four-state decoder.

The legacy decoder remains the source of truth when no evidence is supplied.
When evidence is present, ``evidence[(source, destination)][t]`` is added only
when the path changes from ``source`` at ``t - 1`` to ``destination`` at ``t``.
Self-transition duration scoring is deliberately unchanged.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .multistate import (
    ALLOWED_TRANSITIONS,
    STATE_ORDER,
    MultistateDecode,
    MultistateDecoderConfig,
    MultistateState,
    _duration_slot_caps,
    _intervals_from_states,
    _validate_times,
    decode_multistate,
)


TransitionEdge = tuple[MultistateState, MultistateState]
TransitionEvidence = Mapping[TransitionEdge, np.ndarray]


def _validated_transition_evidence(
    evidence: TransitionEvidence,
    sample_count: int,
) -> dict[TransitionEdge, np.ndarray]:
    validated: dict[TransitionEdge, np.ndarray] = {}
    for edge, raw_values in evidence.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("transition evidence keys must be (source, destination)")
        source, destination = edge
        if source not in STATE_ORDER or destination not in STATE_ORDER:
            raise ValueError("transition evidence edges must use multistate states")
        if destination not in ALLOWED_TRANSITIONS[source]:
            raise ValueError(
                f"transition evidence edge {source.name}->{destination.name} is not allowed"
            )
        if source == destination:
            raise ValueError("transition evidence is only defined for non-self edges")
        values = np.asarray(raw_values, dtype=np.float64)
        if values.shape != (sample_count,):
            raise ValueError(
                "transition evidence arrays must have shape "
                f"({sample_count},), got {values.shape} for "
                f"{source.name}->{destination.name}"
            )
        if not np.isfinite(values).all():
            raise ValueError("transition evidence arrays must contain only finite values")
        validated[(source, destination)] = values
    return validated


def decode_multistate_with_transition_evidence(
    times: np.ndarray,
    state_log_scores: np.ndarray,
    config: MultistateDecoderConfig,
    transition_evidence: TransitionEvidence | None = None,
) -> MultistateDecode:
    """Decode with optional evidence on non-self transition edges.

    Evidence is indexed by the destination sample: an edge taken from state
    ``source`` at sample ``t - 1`` into ``destination`` at sample ``t`` adds
    ``transition_evidence[(source, destination)][t]`` to that candidate path.
    Passing ``None`` or an empty mapping delegates directly to
    :func:`decode_multistate`, preserving its output exactly.
    """

    if not transition_evidence:
        return decode_multistate(times, state_log_scores, config)

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
    evidence = _validated_transition_evidence(transition_evidence, len(times))
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
                edge_values = evidence.get((source, destination))
                edge_score = (
                    float(edge_values[sample_index]) if edge_values is not None else 0.0
                )
                candidate = (
                    previous_score
                    + config.transition_score(source, destination)
                    + edge_score
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
    return MultistateDecode(
        intervals=_intervals_from_states(times, decoded_states),
        states=decoded_states,
        path_log_score=best_score,
    )
