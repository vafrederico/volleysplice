from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_t3_jersey_transport import JerseyTracklet, summarize_team
from analysis.side_switch_t10_tracklet_unit_transport import (
    T10_CORE_FEATURE_NAMES,
    TrackletUnitEndpointSummary,
    tracklet_unit_transport_features,
    unit_match_reduction,
)


def _tracklet(index: int, reliability: float = 0.8) -> JerseyTracklet:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return JerseyTracklet(descriptor, 0.5, 0.7, 0.1, reliability, 3)


def _endpoint(
    near: tuple[JerseyTracklet, ...], far: tuple[JerseyTracklet, ...]
) -> TrackletUnitEndpointSummary:
    return TrackletUnitEndpointSummary(
        near=near,
        far=far,
        near_team=summarize_team(near),
        far_team=summarize_team(far),
        selected_counts=(0, 0, 0, 0, 0),
        full_raw_candidate_counts=(0, 0, 0, 0, 0),
        far_raw_candidate_counts=(0, 0, 0, 0, 0),
        background_fallbacks=0,
        full_detector_inference_milliseconds=0.0,
        far_detector_inference_milliseconds=0.0,
        far_crop_top=0,
        far_crop_bottom=1,
        frame_height=1,
    )


class SideSwitchT10TrackletUnitTransportTest(unittest.TestCase):
    def test_permuted_equal_unit_sets_match_exactly(self) -> None:
        reduction = unit_match_reduction(
            (_tracklet(1), _tracklet(9)),
            (_tracklet(9), _tracklet(1)),
        )
        self.assertEqual(reduction.matched_pairs, 2)
        self.assertAlmostEqual(reduction.coverage, 1.0)
        self.assertAlmostEqual(reduction.conditional_similarity, 1.0)
        self.assertAlmostEqual(reduction.cost, 0.0)

    def test_unmatched_player_unit_receives_maximum_cost(self) -> None:
        reduction = unit_match_reduction(
            (_tracklet(1), _tracklet(9)),
            (_tracklet(1),),
        )
        self.assertEqual(reduction.matched_pairs, 1)
        self.assertAlmostEqual(reduction.coverage, 0.5)
        self.assertAlmostEqual(reduction.conditional_similarity, 1.0)
        self.assertAlmostEqual(reduction.cost, 0.5)

    def test_unavailable_set_has_zero_support_and_unit_cost(self) -> None:
        reduction = unit_match_reduction((), (_tracklet(1),))
        self.assertEqual(reduction.cost, 1.0)
        self.assertEqual(reduction.coverage, 0.0)
        self.assertEqual(reduction.conditional_similarity, 0.0)

    def test_swapped_endpoint_has_positive_margin_and_reliable_evidence(self) -> None:
        before = _endpoint((_tracklet(1),), (_tracklet(9),))
        after = _endpoint((_tracklet(9),), (_tracklet(1),))
        values, diagnostics = tracklet_unit_transport_features(before, after)
        self.assertEqual(tuple(values)[: len(T10_CORE_FEATURE_NAMES)], T10_CORE_FEATURE_NAMES)
        self.assertGreater(values["trackletUnitJerseyTeamTransportSwapMargin"], 0.0)
        self.assertGreater(values["trackletUnitJerseyReliableSwapEvidence"], 0.0)
        self.assertAlmostEqual(values["trackletUnitJerseyCrossMatchCoverageMinimum"], 1.0)
        self.assertLess(diagnostics["swappedAssignmentCost"], diagnostics["sameAssignmentCost"])


if __name__ == "__main__":
    unittest.main()
