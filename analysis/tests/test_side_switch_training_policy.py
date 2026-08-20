from __future__ import annotations

import unittest

from analysis.side_switch_training_policy import (
    SIDE_SWITCH_CADENCE_POINTS,
    SIDE_SWITCH_DEFAULT_RALLY_MARGIN,
    SIDE_SWITCH_MAX_OPPORTUNITIES,
    SIDE_SWITCH_ONE_SET_PER_RECORDING,
    SIDE_SWITCH_RECORDING_START_POINT,
    SideSwitchTrainingPolicyError,
    expected_switch_gap_windows,
    expected_switch_point_totals,
    validate_side_switch_fit_recordings,
)


class SideSwitchTrainingPolicyTests(unittest.TestCase):
    def test_one_set_recording_starts_at_zero_with_seven_point_cadence(self) -> None:
        self.assertTrue(SIDE_SWITCH_ONE_SET_PER_RECORDING)
        self.assertEqual(SIDE_SWITCH_RECORDING_START_POINT, 0)
        self.assertEqual(SIDE_SWITCH_CADENCE_POINTS, 7)
        self.assertEqual(SIDE_SWITCH_DEFAULT_RALLY_MARGIN, 2)
        self.assertEqual(SIDE_SWITCH_MAX_OPPORTUNITIES, 6)
        self.assertEqual(expected_switch_point_totals(41), (7, 14, 21, 28, 35))

    def test_rally_proxy_uses_inclusive_candidate_margins(self) -> None:
        self.assertEqual(
            expected_switch_gap_windows(22),
            ((5, 9), (12, 16), (19, 22)),
        )

    def test_candidate_window_opens_before_nominal_gap(self) -> None:
        self.assertEqual(expected_switch_gap_windows(6), ((5, 6),))

    def test_candidate_windows_respect_one_set_opportunity_cap(self) -> None:
        self.assertEqual(len(expected_switch_gap_windows(70, rally_margin=4)), 6)

    def test_rally_margin_validation(self) -> None:
        with self.assertRaisesRegex(SideSwitchTrainingPolicyError, "cannot be negative"):
            expected_switch_gap_windows(21, rally_margin=-1)

    def test_blurry_beach_recording_is_rejected_from_new_fits(self) -> None:
        with self.assertRaisesRegex(
            SideSwitchTrainingPolicyError, "beach-source-02"
        ):
            validate_side_switch_fit_recordings(
                ["grass-source-09", "beach-source-02"]
            )

    def test_other_recordings_are_allowed(self) -> None:
        validate_side_switch_fit_recordings(
            ["beach-source-01", "grass-source-09"]
        )


if __name__ == "__main__":
    unittest.main()
