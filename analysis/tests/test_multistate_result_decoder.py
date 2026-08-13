from __future__ import annotations

import math
import unittest

import numpy as np

from analysis.multistate import (
    STATE_ORDER,
    MultistateDecoderConfig,
    MultistateInterval,
    MultistateState,
    StateDurationPrior,
)
from analysis.multistate_result_decoder import (
    ResultDecoderConfig,
    ResultLatentState,
    decode_multistate_result,
)


def result_config(
    *,
    ordinary: StateDurationPrior | None = None,
    result: StateDurationPrior | None = None,
    transitions: dict[tuple[MultistateState, MultistateState], float] | None = None,
) -> ResultDecoderConfig:
    base = MultistateDecoderConfig(
        duration_priors={
            MultistateState.DEAD: StateDurationPrior(),
            MultistateState.SETUP: StateDurationPrior(),
            MultistateState.SERVE: StateDurationPrior(maximum_samples=1),
            MultistateState.LIVE: StateDurationPrior(),
        },
        transition_log_scores=transitions or {},
    )
    return ResultDecoderConfig(
        base=base,
        ordinary_live_prior=ordinary or StateDurationPrior(),
        result_live_prior=result or StateDurationPrior(),
    )


def scores_for_public_path(states: list[MultistateState]) -> np.ndarray:
    scores = np.full((len(states), len(STATE_ORDER)), -100.0, dtype=np.float64)
    for index, state in enumerate(states):
        scores[index, int(state)] = 0.0
    return scores


class MultistateResultDecoderTests(unittest.TestCase):
    def test_result_probability_at_serve_selects_branch_once(self) -> None:
        public_path = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
        ]
        times = np.arange(len(public_path), dtype=np.float64)
        probabilities = np.full(len(times), 0.01, dtype=np.float64)
        probabilities[2] = 0.8

        decoded = decode_multistate_result(
            times,
            scores_for_public_path(public_path),
            probabilities,
            result_config(),
        )

        self.assertEqual(decoded.states, tuple(public_path))
        self.assertEqual(
            decoded.latent_states,
            (
                ResultLatentState.DEAD,
                ResultLatentState.SETUP,
                ResultLatentState.SERVE,
                ResultLatentState.LIVE_RESULT,
                ResultLatentState.DEAD,
            ),
        )
        self.assertEqual(decoded.intervals, (MultistateInterval(2.0, 4.0),))
        self.assertEqual(decoded.ordinary_branch_count, 0)
        self.assertEqual(decoded.result_branch_count, 1)
        self.assertEqual(decoded.result_durations_samples, (1,))
        self.assertAlmostEqual(decoded.path_log_score, math.log(0.8))

    def test_low_result_probability_selects_ordinary_branch(self) -> None:
        public_path = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
        ]
        probabilities = np.full(len(public_path), 0.9, dtype=np.float64)
        probabilities[2] = 0.2

        decoded = decode_multistate_result(
            np.arange(len(public_path), dtype=np.float64),
            scores_for_public_path(public_path),
            probabilities,
            result_config(),
        )

        self.assertEqual(decoded.latent_states[3], ResultLatentState.LIVE_ORDINARY)
        self.assertEqual(decoded.ordinary_branch_count, 1)
        self.assertEqual(decoded.ordinary_durations_samples, (1,))
        self.assertEqual(decoded.result_branch_count, 0)
        self.assertAlmostEqual(decoded.path_log_score, math.log(0.8))

    def test_direct_serve_to_dead_is_impossible_even_with_base_bonus(self) -> None:
        public_path = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.DEAD,
            MultistateState.DEAD,
        ]
        scores = scores_for_public_path(public_path)
        # A direct SERVE -> DEAD path would take the locally perfect DEAD row.
        # Keep a slightly weaker LIVE alternative so the legal latent path is
        # still preferable to avoiding the strongly supported SERVE entirely.
        scores[3, int(MultistateState.LIVE)] = -1.0
        decoded = decode_multistate_result(
            np.arange(len(public_path), dtype=np.float64),
            scores,
            np.full(len(public_path), 0.5),
            result_config(
                transitions={(MultistateState.SERVE, MultistateState.DEAD): 1_000.0}
            ),
        )

        self.assertIn(
            decoded.latent_states[3],
            (ResultLatentState.LIVE_ORDINARY, ResultLatentState.LIVE_RESULT),
        )
        self.assertEqual(decoded.states[3], MultistateState.LIVE)
        self.assertEqual(decoded.intervals, (MultistateInterval(2.0, 4.0),))

    def test_distinct_duration_priors_can_override_branch_probability(self) -> None:
        public_path = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.LIVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
        ]
        probabilities = np.full(len(public_path), 0.5, dtype=np.float64)
        probabilities[2] = 0.99
        decoded = decode_multistate_result(
            np.arange(len(public_path), dtype=np.float64),
            scores_for_public_path(public_path),
            probabilities,
            result_config(
                ordinary=StateDurationPrior(),
                result=StateDurationPrior(maximum_samples=1),
            ),
        )

        self.assertEqual(
            decoded.latent_states[3:6],
            (ResultLatentState.LIVE_ORDINARY,) * 3,
        )
        self.assertEqual(decoded.ordinary_durations_samples, (3,))
        self.assertEqual(decoded.result_durations_samples, ())

    def test_multiple_branches_report_counts_and_live_run_durations(self) -> None:
        public_path = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
        ]
        probabilities = np.full(len(public_path), 0.5, dtype=np.float64)
        probabilities[2] = 0.1
        probabilities[6] = 0.9

        decoded = decode_multistate_result(
            np.arange(len(public_path), dtype=np.float64) * 0.25,
            scores_for_public_path(public_path),
            probabilities,
            result_config(),
        )

        self.assertEqual(decoded.ordinary_branch_count, 1)
        self.assertEqual(decoded.result_branch_count, 1)
        self.assertEqual(decoded.ordinary_durations_samples, (1,))
        self.assertEqual(decoded.result_durations_samples, (2,))
        self.assertEqual(
            decoded.intervals,
            (MultistateInterval(0.5, 1.0), MultistateInterval(1.5, 2.25)),
        )

    def test_exact_branch_tie_is_deterministic_and_prefers_ordinary(self) -> None:
        public_path = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
        ]
        arguments = (
            np.arange(len(public_path), dtype=np.float64),
            scores_for_public_path(public_path),
            np.full(len(public_path), 0.5),
            result_config(),
        )

        first = decode_multistate_result(*arguments)
        second = decode_multistate_result(*arguments)

        self.assertEqual(first, second)
        self.assertEqual(first.latent_states[3], ResultLatentState.LIVE_ORDINARY)

    def test_result_probability_validation_is_strict(self) -> None:
        times = np.arange(2, dtype=np.float64)
        scores = np.zeros((2, len(STATE_ORDER)), dtype=np.float64)
        for probabilities in (
            np.asarray([0.5]),
            np.asarray([0.0, 0.5]),
            np.asarray([1.0, 0.5]),
            np.asarray([math.nan, 0.5]),
            np.asarray([math.inf, 0.5]),
        ):
            with self.subTest(probabilities=probabilities):
                with self.assertRaisesRegex(ValueError, "result probabilities"):
                    decode_multistate_result(
                        times, scores, probabilities, result_config()
                    )

    def test_empty_input_returns_empty_summary(self) -> None:
        decoded = decode_multistate_result(
            np.asarray([], dtype=np.float64),
            np.empty((0, len(STATE_ORDER)), dtype=np.float64),
            np.asarray([], dtype=np.float64),
            result_config(),
        )

        self.assertEqual(decoded.intervals, ())
        self.assertEqual(decoded.states, ())
        self.assertEqual(decoded.latent_states, ())
        self.assertEqual(decoded.ordinary_branch_count, 0)
        self.assertEqual(decoded.result_branch_count, 0)
        self.assertEqual(decoded.path_log_score, 0.0)


if __name__ == "__main__":
    unittest.main()
