from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from analysis.multistate import (
    ALLOWED_TRANSITIONS,
    STATE_ORDER,
    MultistateDecoderConfig,
    MultistateInterval,
    MultistateState,
    StateDurationPrior,
    build_state_targets,
    decode_multistate,
)


@dataclass(frozen=True)
class IntervalValue:
    start: float
    end: float
    tags: tuple[str, ...] = ()


def flat_config(
    *,
    setup_minimum: int = 1,
    live_minimum: int = 1,
) -> MultistateDecoderConfig:
    return MultistateDecoderConfig(
        duration_priors={
            MultistateState.DEAD: StateDurationPrior(),
            MultistateState.SETUP: StateDurationPrior(
                minimum_samples=setup_minimum
            ),
            MultistateState.SERVE: StateDurationPrior(maximum_samples=1),
            MultistateState.LIVE: StateDurationPrior(
                minimum_samples=live_minimum
            ),
        }
    )


def scores_for_path(states: list[MultistateState]) -> np.ndarray:
    scores = np.full((len(states), len(STATE_ORDER)), -100.0, dtype=np.float64)
    for index, state in enumerate(states):
        scores[index, int(state)] = 0.0
    return scores


class MultistateTargetTests(unittest.TestCase):
    def test_targets_use_half_open_windows_and_explicit_precedence(self) -> None:
        times = np.arange(0.0, 4.0, 0.25)

        targets = build_state_targets(
            times,
            [IntervalValue(1.0, 2.0, tags=("service-fault",))],
            setup_radius_seconds=0.75,
            serve_radius_seconds=0.25,
        )

        expected = np.full(len(times), int(MultistateState.DEAD), dtype=np.int8)
        expected[(times >= 0.25) & (times < 0.75)] = int(MultistateState.SETUP)
        expected[(times >= 1.25) & (times < 2.0)] = int(MultistateState.LIVE)
        # SERVE wins over SETUP at 0.75 and over LIVE at the 1.0 contact.
        expected[(times >= 0.75) & (times < 1.25)] = int(
            MultistateState.SERVE
        )
        np.testing.assert_array_equal(targets, expected)
        self.assertEqual(targets[np.where(times == 2.0)[0][0]], MultistateState.DEAD)

    def test_next_serve_preparation_never_overwrites_preceding_live_time(self) -> None:
        times = np.arange(0.0, 5.0, 0.25)

        targets = build_state_targets(
            times,
            [IntervalValue(0.5, 2.5), IntervalValue(3.0, 4.0)],
            setup_radius_seconds=2.0,
            serve_radius_seconds=1.0,
        )

        self.assertEqual(
            targets[np.where(times == 2.25)[0][0]], MultistateState.LIVE
        )
        self.assertEqual(
            targets[np.where(times == 2.5)[0][0]], MultistateState.SERVE
        )
        self.assertEqual(
            targets[np.where(times == 3.0)[0][0]], MultistateState.SERVE
        )
        self.assertEqual(
            targets[np.where(times == 4.0)[0][0]], MultistateState.DEAD
        )

    def test_targets_are_order_independent_and_do_not_read_outcome_tags(self) -> None:
        times = np.arange(0.0, 6.0, 0.25)
        tagged = [
            IntervalValue(1.0, 2.0, tags=("ace",)),
            IntervalValue(4.0, 5.0, tags=("service-fault",)),
        ]
        untagged_reversed = [
            IntervalValue(4.0, 5.0),
            IntervalValue(1.0, 2.0),
        ]

        first = build_state_targets(
            times,
            tagged,
            setup_radius_seconds=0.5,
            serve_radius_seconds=0.0,
        )
        second = build_state_targets(
            times,
            untagged_reversed,
            setup_radius_seconds=0.5,
            serve_radius_seconds=0.0,
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[np.where(times == 1.0)[0][0]], MultistateState.SERVE)
        self.assertEqual(first[np.where(times == 4.0)[0][0]], MultistateState.SERVE)

    def test_target_validation_rejects_overlap_and_invalid_times(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            build_state_targets(
                np.arange(4.0),
                [IntervalValue(0.0, 2.0), IntervalValue(1.0, 3.0)],
                setup_radius_seconds=1.0,
                serve_radius_seconds=0.25,
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_state_targets(
                np.asarray([0.0, 1.0, 0.5]),
                [],
                setup_radius_seconds=1.0,
                serve_radius_seconds=0.25,
            )


class MultistateDecoderTests(unittest.TestCase):
    def test_transition_graph_is_exact(self) -> None:
        self.assertEqual(
            ALLOWED_TRANSITIONS,
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
            },
        )

    def test_serve_to_dead_opens_a_one_sample_immediate_event(self) -> None:
        states = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.DEAD,
        ]
        times = np.arange(len(states), dtype=np.float64) * 0.25

        decoded = decode_multistate(times, scores_for_path(states), flat_config())

        self.assertEqual(decoded.states, tuple(states))
        self.assertEqual(decoded.intervals, (MultistateInterval(0.5, 0.75),))

    def test_multiple_events_are_chronological_nonoverlapping_and_use_next_dead(self) -> None:
        states = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.DEAD,
            MultistateState.DEAD,
        ]
        times = np.arange(len(states), dtype=np.float64)

        decoded = decode_multistate(times, scores_for_path(states), flat_config())

        self.assertEqual(
            decoded.intervals,
            (MultistateInterval(2.0, 4.0), MultistateInterval(6.0, 7.0)),
        )
        self.assertLessEqual(decoded.intervals[0].end, decoded.intervals[1].start)

    def test_duration_prior_minimum_overrides_a_locally_preferred_early_serve(self) -> None:
        times = np.arange(5, dtype=np.float64)
        scores = scores_for_path(
            [
                MultistateState.DEAD,
                MultistateState.SETUP,
                MultistateState.SERVE,
                MultistateState.SERVE,
                MultistateState.DEAD,
            ]
        )
        # The emission peak favors SERVE at sample two. A two-sample SETUP
        # duration supplied by the fold makes that path illegal.
        scores[2, int(MultistateState.SETUP)] = -1.0
        scores[3, int(MultistateState.SERVE)] = -1.0

        decoded = decode_multistate(
            times,
            scores,
            flat_config(setup_minimum=2),
        )

        self.assertEqual(
            decoded.states,
            (
                MultistateState.DEAD,
                MultistateState.SETUP,
                MultistateState.SETUP,
                MultistateState.SERVE,
                MultistateState.DEAD,
            ),
        )
        self.assertEqual(decoded.intervals, (MultistateInterval(3.0, 4.0),))

    def test_fold_duration_log_scores_choose_between_legal_setup_lengths(self) -> None:
        times = np.arange(6, dtype=np.float64)
        scores = np.full((len(times), len(STATE_ORDER)), -100.0)
        scores[0, int(MultistateState.DEAD)] = 0.0
        scores[1, int(MultistateState.SETUP)] = 0.0
        scores[2, int(MultistateState.SETUP)] = 0.0
        scores[2, int(MultistateState.SERVE)] = 0.0
        scores[3, int(MultistateState.SERVE)] = 0.0
        scores[3:, int(MultistateState.DEAD)] = 0.0
        config = MultistateDecoderConfig(
            duration_priors={
                MultistateState.DEAD: StateDurationPrior(),
                MultistateState.SETUP: StateDurationPrior(
                    log_scores=(-5.0, 0.0),
                    tail_log_score=-5.0,
                ),
                MultistateState.SERVE: StateDurationPrior(maximum_samples=1),
                MultistateState.LIVE: StateDurationPrior(),
            }
        )

        decoded = decode_multistate(times, scores, config)

        self.assertEqual(decoded.states[1:4], (
            MultistateState.SETUP,
            MultistateState.SETUP,
            MultistateState.SERVE,
        ))
        self.assertEqual(decoded.intervals, (MultistateInterval(3.0, 4.0),))

    def test_disallowed_direct_dead_to_serve_is_never_decoded(self) -> None:
        times = np.arange(3, dtype=np.float64)
        scores = np.full((3, len(STATE_ORDER)), -10.0)
        scores[0, int(MultistateState.DEAD)] = 0.0
        scores[1, int(MultistateState.SERVE)] = 100.0
        scores[2, int(MultistateState.DEAD)] = 0.0

        decoded = decode_multistate(times, scores, flat_config())

        self.assertNotEqual(decoded.states[1], MultistateState.SERVE)
        self.assertEqual(decoded.intervals, ())

    def test_ties_are_deterministic(self) -> None:
        times = np.arange(6, dtype=np.float64)
        scores = np.zeros((len(times), len(STATE_ORDER)), dtype=np.float64)

        first = decode_multistate(times, scores, flat_config())
        second = decode_multistate(times, scores, flat_config())

        self.assertEqual(first, second)

    def test_decoder_validates_fold_config_and_score_alignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "SERVE duration"):
            flat_config().duration_priors[MultistateState.SERVE].__class__().validate(
                MultistateState.SERVE
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_multistate(
                np.arange(2, dtype=np.float64),
                np.zeros((2, 3)),
                flat_config(),
            )
        with self.assertRaisesRegex(ValueError, "no DEAD-to-DEAD path"):
            impossible = np.full((2, len(STATE_ORDER)), -math.inf)
            decode_multistate(np.arange(2, dtype=np.float64), impossible, flat_config())


if __name__ == "__main__":
    unittest.main()
