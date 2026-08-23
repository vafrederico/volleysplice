from __future__ import annotations

import unittest

from analysis.side_switch_parity import (
    RallyStateObservation,
    decode_persistent_flips,
    match_interval_proposals,
    overlaps_transition_mask,
    parity_state_at,
    recording_quality_floor,
    state_metrics,
)


def _observation(
    index: int,
    coordinate: float,
    parity: int,
    *,
    quality: float = 1.0,
    masked: bool = False,
) -> RallyStateObservation:
    return RallyStateObservation(
        recording_id="r",
        rally_index=index,
        start=float(index * 10),
        end=float(index * 10 + 5),
        coordinate=coordinate,
        quality=quality,
        parity_state=parity,
        transition_masked=masked,
    )


class SideSwitchParityTests(unittest.TestCase):
    def test_markers_toggle_parity_and_mask_overlapping_rally(self) -> None:
        markers = [25.0, 55.0]
        self.assertEqual(parity_state_at(24.9, markers), 0)
        self.assertEqual(parity_state_at(25.0, markers), 1)
        self.assertEqual(parity_state_at(60.0, markers), 0)
        self.assertTrue(overlaps_transition_mask(20.0, 23.0, markers, 2.0))
        self.assertFalse(overlaps_transition_mask(10.0, 20.0, markers, 2.0))

    def test_state_metrics_exclude_masked_and_low_quality_observations(self) -> None:
        observations = [
            _observation(1, 0.8, 0),
            _observation(2, -0.7, 1),
            _observation(3, 0.6, 1, quality=0.1),
            _observation(4, -0.5, 0, masked=True),
        ]
        metrics = state_metrics(observations, minimum_quality=0.5)
        self.assertEqual(metrics["eligibleObservations"], 3)
        self.assertEqual(metrics["evaluatedObservations"], 2)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balancedAccuracy"], 1.0)

    def test_relative_quality_floor_does_not_use_labels(self) -> None:
        observations = [
            _observation(1, 1.0, 0, quality=0.2),
            _observation(2, -1.0, 1, quality=0.4),
            _observation(3, 1.0, 0, quality=0.8),
        ]
        self.assertAlmostEqual(recording_quality_floor(observations, 0.5), 0.2)

    def test_persistence_backdates_alternating_state_flips(self) -> None:
        observations = [
            _observation(1, 0.9, 0),
            _observation(2, 0.8, 0),
            _observation(3, -0.7, 1),
            _observation(4, -0.8, 1),
            _observation(5, 0.6, 0, masked=True),
            _observation(6, 0.7, 0),
            _observation(7, 0.8, 0),
        ]
        proposals = decode_persistent_flips(observations, persistence=2)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0]["start"], 25.0)
        self.assertEqual(proposals[0]["end"], 30.0)
        self.assertEqual(proposals[0]["confirmationRallyIndex"], 4)
        self.assertEqual(proposals[1]["start"], 45.0)
        self.assertEqual(proposals[1]["end"], 60.0)

    def test_one_to_one_matching_does_not_reuse_a_marker(self) -> None:
        proposals = [
            {"start": 10.0, "end": 20.0},
            {"start": 14.0, "end": 18.0},
            {"start": 40.0, "end": 45.0},
        ]
        metrics = match_interval_proposals(proposals, [16.0, 50.0], 0.0)
        self.assertEqual(metrics["truePositives"], 1)
        self.assertEqual(metrics["falsePositives"], 2)
        self.assertEqual(metrics["falseNegatives"], 1)


if __name__ == "__main__":
    unittest.main()
