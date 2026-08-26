from __future__ import annotations

import unittest

from analysis.side_switch_t13_source_symmetric_transport import (
    source_symmetric_transport_features,
)


def _row(stable: dict[tuple[str, str], bool]) -> dict:
    endpoints = {}
    t5_endpoints = {}
    for endpoint in ("before", "after"):
        diagnostic = {}
        pooled = {}
        for side in ("near", "far"):
            is_stable = stable[(endpoint, side)]
            diagnostic[f"{side}StableTracklets"] = int(is_stable)
            diagnostic[f"{side}Tracklets"] = [{"reliability": 0.8}]
            pooled[f"{side}TeamAvailable"] = True
        endpoints[endpoint] = diagnostic
        t5_endpoints[endpoint] = pooled
    names = (
        "nearNear",
        "farFar",
        "nearFar",
        "farNear",
        "beforeNearFar",
        "afterNearFar",
    )
    return {
        "kind": "adjacent-rally-boundary",
        "t11EmptySideFallbackTrackletTransport": endpoints,
        "t10TrackletUnitJerseyTransport": {
            "teamReductions": {
                name: {"cost": 0.2, "coverage": 1.0, "conditionalSimilarity": 0.8}
                for name in names
            }
        },
        "t5CourtTrackedFarJerseyTransport": {
            **t5_endpoints,
            "teamCosts": {name: 0.6 for name in names},
        },
    }


class SideSwitchT13SourceSymmetricTransportTest(unittest.TestCase):
    def test_source_routes_are_symmetric_across_destinations(self) -> None:
        stable = {
            ("before", "near"): True,
            ("before", "far"): False,
            ("after", "near"): True,
            ("after", "far"): True,
        }
        _, diagnostics = source_symmetric_transport_features(_row(stable))
        reductions = diagnostics["teamReductions"]
        self.assertEqual(reductions["nearNear"]["route"], reductions["nearFar"]["route"])
        self.assertEqual(reductions["farFar"]["route"], reductions["farNear"]["route"])
        self.assertEqual(reductions["nearNear"]["route"], "stable-player-units")
        self.assertEqual(reductions["farFar"]["route"], "pooled-team")

    def test_missing_after_stable_unit_forces_both_source_routes_to_pool(self) -> None:
        stable = {
            ("before", "near"): True,
            ("before", "far"): True,
            ("after", "near"): True,
            ("after", "far"): False,
        }
        _, diagnostics = source_symmetric_transport_features(_row(stable))
        self.assertEqual(set(diagnostics["sourceRoutes"].values()), {"pooled-team"})


if __name__ == "__main__":
    unittest.main()
