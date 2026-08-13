from __future__ import annotations

import unittest

import numpy as np

from analysis.multistate import (
    STATE_ORDER,
    MultistateDecoderConfig,
    MultistateState,
    StateDurationPrior,
    decode_multistate,
)
from analysis.multistate_transition_evidence import (
    decode_multistate_with_transition_evidence,
)


def _config() -> MultistateDecoderConfig:
    return MultistateDecoderConfig(
        duration_priors={
            MultistateState.DEAD: StateDurationPrior(),
            MultistateState.SETUP: StateDurationPrior(),
            MultistateState.SERVE: StateDurationPrior(maximum_samples=1),
            MultistateState.LIVE: StateDurationPrior(),
        }
    )


def _scores(states: list[MultistateState], *, preferred: float = 0.0) -> np.ndarray:
    result = np.full((len(states), len(STATE_ORDER)), -100.0, dtype=np.float64)
    for index, state in enumerate(states):
        result[index, int(state)] = preferred
    return result


class MultistateTransitionEvidenceTests(unittest.TestCase):
    def test_omitted_and_empty_evidence_exactly_match_legacy_decoder(self) -> None:
        states = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
        ]
        times = np.arange(len(states), dtype=np.float64) * 0.25
        scores = _scores(states)
        legacy = decode_multistate(times, scores, _config())

        for evidence in (
            None,
            {},
            {
                (MultistateState.SETUP, MultistateState.SERVE): np.zeros(
                    len(times), dtype=np.float64
                ),
                (MultistateState.LIVE, MultistateState.DEAD): np.zeros(
                    len(times), dtype=np.float64
                ),
            },
        ):
            candidate = decode_multistate_with_transition_evidence(
                times, scores, _config(), evidence
            )
            self.assertEqual(candidate, legacy)

    def test_evidence_is_indexed_at_transition_destination_sample(self) -> None:
        states = [
            MultistateState.DEAD,
            MultistateState.SETUP,
            MultistateState.SERVE,
            MultistateState.LIVE,
            MultistateState.DEAD,
        ]
        times = np.arange(len(states), dtype=np.float64)
        scores = _scores(states)
        setup_serve = np.zeros(len(times), dtype=np.float64)
        setup_serve[2] = 1.25
        live_dead = np.zeros(len(times), dtype=np.float64)
        live_dead[4] = 2.5

        decoded = decode_multistate_with_transition_evidence(
            times,
            scores,
            _config(),
            {
                (MultistateState.SETUP, MultistateState.SERVE): setup_serve,
                (MultistateState.LIVE, MultistateState.DEAD): live_dead,
            },
        )

        self.assertEqual(decoded.states, tuple(states))
        self.assertAlmostEqual(decoded.path_log_score, 3.75)

    def test_positive_terminal_evidence_changes_the_best_legal_path(self) -> None:
        times = np.arange(6, dtype=np.float64)
        scores = _scores(
            [
                MultistateState.DEAD,
                MultistateState.SETUP,
                MultistateState.SERVE,
                MultistateState.LIVE,
                MultistateState.LIVE,
                MultistateState.DEAD,
            ]
        )
        # Make LIVE and DEAD equally plausible at sample four. Legacy tie
        # behavior stays LIVE; edge evidence makes an early terminal transition win.
        scores[4, int(MultistateState.DEAD)] = 0.0
        evidence = np.zeros(len(times), dtype=np.float64)
        evidence[4] = 1.0

        legacy = decode_multistate(times, scores, _config())
        changed = decode_multistate_with_transition_evidence(
            times,
            scores,
            _config(),
            {(MultistateState.LIVE, MultistateState.DEAD): evidence},
        )

        self.assertEqual(legacy.states[4], MultistateState.LIVE)
        self.assertEqual(changed.states[4], MultistateState.DEAD)

    def test_invalid_edges_shapes_and_values_are_rejected(self) -> None:
        times = np.arange(4, dtype=np.float64)
        scores = _scores(
            [
                MultistateState.DEAD,
                MultistateState.SETUP,
                MultistateState.SERVE,
                MultistateState.DEAD,
            ]
        )
        cases = (
            ({(MultistateState.DEAD, MultistateState.SERVE): np.zeros(4)}, "not allowed"),
            ({(MultistateState.LIVE, MultistateState.LIVE): np.zeros(4)}, "non-self"),
            ({(MultistateState.SETUP, MultistateState.SERVE): np.zeros(3)}, "shape"),
            ({(MultistateState.SETUP, MultistateState.SERVE): np.asarray([0.0, 0.0, np.nan, 0.0])}, "finite"),
        )
        for evidence, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                decode_multistate_with_transition_evidence(
                    times, scores, _config(), evidence
                )


if __name__ == "__main__":
    unittest.main()
