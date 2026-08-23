import unittest

import numpy as np

from analysis.side_switch_internal_specialist import (
    DERIVED_INTERNAL_FEATURE_NAMES,
    enrich_internal_candidates,
    select_internal_candidates,
)


def _internal(event_id: str, time: float, score: float) -> dict:
    return {
        "eventId": event_id,
        "recordingId": "video",
        "kind": "internal-dead-state-peak",
        "gapStart": time - 1,
        "gapEnd": time + 1,
        "transitionTime": time,
        "score": score,
        "sourceRangeId": "R001",
        "features": {
            "beforePlayerPaletteInstability": 0.2,
            "afterPlayerPaletteInstability": 0.4,
        },
        "productionStateContext": {
            "before": {
                "sourceStart": 0.0,
                "sourceEnd": 30.0,
                "anchorTime": 2.0,
            }
        },
    }


class SideSwitchInternalSpecialistTests(unittest.TestCase):
    def test_enrichment_adds_range_and_peak_geometry(self) -> None:
        rows = [_internal("a", 10.0, 0.9), _internal("b", 20.0, 0.8)]
        boundary = {
            "eventId": "boundary",
            "recordingId": "video",
            "kind": "adjacent-rally-boundary",
            "transitionTime": 5.0,
        }
        enriched = enrich_internal_candidates(rows, [*rows, boundary])
        for name in DERIVED_INTERNAL_FEATURE_NAMES:
            self.assertIn(name, enriched[0]["features"])
        self.assertEqual(enriched[0]["features"]["internalPeaksInSourceRange"], 2.0)
        self.assertEqual(enriched[0]["features"]["internalInverseScoreRankInRange"], 1.0)

    def test_internal_selection_suppresses_boundary_duplicates_and_local_peaks(self) -> None:
        rows = [
            _internal("near-boundary", 10.0, 0.9),
            _internal("strong", 30.0, 0.8),
            _internal("weak-neighbor", 35.0, 0.7),
        ]
        boundary = {
            "eventId": "boundary",
            "recordingId": "video",
            "kind": "adjacent-rally-boundary",
            "transitionTime": 12.0,
        }
        selected = select_internal_candidates(
            rows, np.asarray([0.9, 0.8, 0.7]), 0.5, [boundary]
        )
        self.assertEqual(selected.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
