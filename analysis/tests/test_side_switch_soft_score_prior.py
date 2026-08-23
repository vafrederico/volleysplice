import unittest

import numpy as np

from analysis.side_switch_soft_score_prior import (
    PointTransition,
    apply_soft_score_prior,
    cadence_hazards,
    latent_count_distributions,
    opportunity_index,
)


def _boundary(index: int) -> dict:
    return {
        "eventId": f"video:boundary:R{index:03d}:R{index + 1:03d}",
        "recordingId": "video",
    }


class SideSwitchSoftScorePriorTests(unittest.TestCase):
    def test_opportunity_index_supports_boundaries_and_internal_peaks(self) -> None:
        self.assertEqual(opportunity_index(_boundary(7)), 7)
        self.assertEqual(
            opportunity_index(
                {
                    "eventId": "video:internal-dead-peak:R014:1000",
                    "recordingId": "video",
                    "sourceRangeId": "R014",
                }
            ),
            14,
        )

    def test_exact_count_prior_peaks_at_multiples_of_seven(self) -> None:
        transition = PointTransition(0.0, 1.0, 0.0, 0.75)
        hazards = cadence_hazards(
            [_boundary(6), _boundary(7), _boundary(8), _boundary(14)], transition
        )
        self.assertGreater(hazards[1], hazards[0])
        self.assertGreater(hazards[1], hazards[2])
        self.assertAlmostEqual(hazards[1], hazards[3])

    def test_uncertain_transition_distribution_remains_normalized(self) -> None:
        transition = PointTransition(0.05, 0.9, 0.05, 1.0)
        for distribution in latent_count_distributions(30, transition):
            self.assertAlmostEqual(float(np.sum(distribution)), 1.0)

    def test_zero_weight_preserves_scores_and_positive_prior_raises_them(self) -> None:
        scores = np.asarray([0.4, 0.4])
        prior = np.asarray([-1.0, 1.0])
        np.testing.assert_array_equal(apply_soft_score_prior(scores, prior, 0.0), scores)
        adjusted = apply_soft_score_prior(scores, prior, 0.5)
        self.assertLess(adjusted[0], scores[0])
        self.assertGreater(adjusted[1], scores[1])


if __name__ == "__main__":
    unittest.main()
