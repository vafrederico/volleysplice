from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_candidate_union import (
    CandidateUnionConfig,
    boundary_candidates,
    internal_peak_candidates,
)


class SideSwitchCandidateUnionTests(unittest.TestCase):
    def test_all_adjacent_boundaries_are_preserved(self) -> None:
        ranges = [
            {"id": "R1", "start": 0.0, "end": 5.0},
            {"id": "R2", "start": 8.0, "end": 12.0},
            {"id": "R3", "start": 15.0, "end": 20.0},
        ]
        candidates = boundary_candidates("r", ranges)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["gapStart"], 5.0)
        self.assertEqual(candidates[0]["gapEnd"], 8.0)
        self.assertEqual(candidates[1]["eventId"], "r:boundary:R2:R3")

    def test_internal_peak_nms_respects_range_edges_and_separation(self) -> None:
        times = np.arange(0.0, 31.0, 1.0)
        rally = np.full(len(times), 0.9)
        dead = np.zeros(len(times))
        dead[2] = 1.0  # excluded by the four-second range edge
        dead[10] = 0.99
        dead[12] = 1.0  # wins the nearby score-ranked suppression cluster
        dead[27] = 1.0  # excluded by the far range edge
        candidates = internal_peak_candidates(
            "r",
            [{"id": "R1", "start": 0.0, "end": 30.0}],
            times,
            rally,
            dead,
            CandidateUnionConfig("deadState", 0.98, 6.0),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["transitionTime"], 12.0)
        self.assertEqual(candidates[0]["gapStart"], 11.0)
        self.assertEqual(candidates[0]["gapEnd"], 13.0)

    def test_combined_signal_can_use_inverse_rally(self) -> None:
        times = np.arange(0.0, 21.0, 1.0)
        rally = np.full(len(times), 1.0)
        rally[10] = 0.0
        dead = np.zeros(len(times))
        ranges = [{"id": "R1", "start": 0.0, "end": 20.0}]
        dead_only = internal_peak_candidates(
            "r",
            ranges,
            times,
            rally,
            dead,
            CandidateUnionConfig("deadState", 0.98, 6.0),
        )
        combined = internal_peak_candidates(
            "r",
            ranges,
            times,
            rally,
            dead,
            CandidateUnionConfig("maxDeadOrInverseRally", 0.98, 6.0),
        )
        self.assertEqual(dead_only, [])
        self.assertEqual(len(combined), 1)


if __name__ == "__main__":
    unittest.main()
