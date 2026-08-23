import unittest

from analysis.side_switch_full_union_features import (
    candidate_windows,
    internal_flank_windows,
    whole_rally_sample_times,
)


class SideSwitchFullUnionFeatureTests(unittest.TestCase):
    def test_whole_rally_sampling_matches_frozen_contract(self) -> None:
        self.assertEqual(
            whole_rally_sample_times(10.0, 20.0),
            (10.2, 11.808333333333334, 13.416666666666666, 15.025,
             16.633333333333333, 18.241666666666667, 19.85),
        )

    def test_internal_flanks_use_fixed_three_second_windows(self) -> None:
        before, after = internal_flank_windows(20.0, 16.0, 24.0)
        self.assertEqual((before.start, before.end), (16.0, 19.0))
        self.assertEqual((after.start, after.end), (21.0, 24.0))

    def test_internal_flanks_reject_crossing_range_edge(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks the declared stable flanks"):
            internal_flank_windows(20.0, 16.1, 24.0)

    def test_candidate_windows_resolve_boundary_and_internal_peak(self) -> None:
        ranges = [
            {"id": "R001", "start": 5.0, "end": 12.0},
            {"id": "R002", "start": 16.0, "end": 30.0},
        ]
        boundary = {
            "eventId": "video:boundary:R001:R002",
            "kind": "adjacent-rally-boundary",
        }
        before, after, index = candidate_windows(boundary, ranges)
        self.assertEqual((before.start, before.end), (5.0, 12.0))
        self.assertEqual((after.start, after.end), (16.0, 30.0))
        self.assertEqual(index, 0)

        internal = {
            "eventId": "video:internal-dead-peak:R002:22000",
            "kind": "internal-dead-state-peak",
            "transitionTime": 22.0,
            "sourceRangeId": "R002",
        }
        before, after, index = candidate_windows(internal, ranges)
        self.assertEqual((before.start, before.end), (18.0, 21.0))
        self.assertEqual((after.start, after.end), (23.0, 26.0))
        self.assertIsNone(index)


if __name__ == "__main__":
    unittest.main()
