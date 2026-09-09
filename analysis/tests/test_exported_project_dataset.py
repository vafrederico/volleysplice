from __future__ import annotations

import unittest

from analysis.exported_project_dataset import normalize_feedback_annotations


def payload(
    ranges: list[dict[str, object]],
    markers: list[dict[str, object]],
    *,
    cut_ids: list[str],
    excluded: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source": {"media": {"duration": 100.0}},
        "corrections": {
            "correctedRanges": ranges,
            "ignoredIntervals": [],
            "scoreTracking": {
                "state": {
                    "team1Name": "A",
                    "team2Name": "B",
                    "serveMarkers": markers,
                    "sideSwitchMarkers": [],
                },
                "excludedRallyIds": excluded or [],
                "derivedFinalScore": {},
            },
        },
        "finalExportIntervals": [
            {"start": 8.0, "end": 32.0, "cutIds": cut_ids}
        ],
    }


def corrected(identifier: str, start: float, end: float) -> dict[str, object]:
    return {
        "id": identifier,
        "coreStart": start,
        "coreEnd": end,
        "included": True,
        "origin": "cached-label",
        "confidence": 0.8,
    }


def marker(
    identifier: str,
    time: float,
    rally_id: str | None,
    *,
    origin: str = "model",
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": identifier,
        "timestamp": time,
        "side": "near",
        "origin": origin,
        "ignorePreviousPoint": False,
    }
    if rally_id is not None:
        result["rallyId"] = rally_id
    return result


class ExportedProjectDatasetTests(unittest.TestCase):
    def test_moved_serve_uses_corrected_timestamp_instead_of_original_model_anchor(self) -> None:
        for moved_start in (12.0, 8.0):
            with self.subTest(moved_start=moved_start):
                value = payload(
                    [corrected("R001", moved_start, 20.0)],
                    [marker("serve-R001", moved_start, "R001")],
                    cut_ids=["R001"],
                )
                value["inference"] = {
                    "servingSide": {"candidates": [{"id": "R001", "anchor": 10.0}]}
                }

                result = normalize_feedback_annotations(value)

                event = result["serveEvents"][0]
                self.assertEqual(event["rawTime"], moved_start)
                self.assertEqual(event["time"], moved_start)
                self.assertEqual(event["alignedRangeId"], "R001")
                self.assertEqual(event["time"], result["associationCoreRanges"][0]["start"])
                self.assertFalse(event["wasTimeAdjusted"])
                self.assertEqual(event["alignment"], "preserved-export-rally-id")

    def test_marker_from_micro_range_moves_to_unmarked_neighbor(self) -> None:
        value = payload(
            [corrected("R001", 10.0, 10.1), corrected("R002", 14.0, 20.0)],
            [marker("serve-R001", 9.0, "R001")],
            cut_ids=["R001", "R002"],
        )

        result = normalize_feedback_annotations(value)

        self.assertEqual(result["normalization"]["microRangeArtifactIds"], ["R001"])
        self.assertEqual(result["serveEvents"][0]["alignedRangeId"], "R002")
        self.assertEqual(result["serveEvents"][0]["time"], 14.0)
        self.assertTrue(result["serveEvents"][0]["wasTimeAdjusted"])

    def test_unmarked_joined_range_is_coverage_not_an_invented_serve(self) -> None:
        value = payload(
            [corrected("R001", 10.0, 15.0), corrected("R002", 16.0, 20.0)],
            [marker("serve-R001", 10.0, "R001")],
            cut_ids=["R001", "R002"],
        )

        result = normalize_feedback_annotations(value)

        self.assertEqual(len(result["serveEvents"]), 1)
        self.assertEqual(
            result["serveEvents"][0]["coverageRangeIds"], ["R001", "R002"]
        )
        self.assertEqual(result["normalization"]["unassociatedCoverageRangeIds"], [])

    def test_manual_second_serve_inside_one_coverage_range_is_preserved(self) -> None:
        value = payload(
            [corrected("R001", 10.0, 30.0)],
            [
                marker("serve-R001", 10.0, "R001"),
                marker("S001", 20.0, None, origin="manual"),
            ],
            cut_ids=["R001"],
        )

        result = normalize_feedback_annotations(value)

        self.assertEqual([event["time"] for event in result["serveEvents"]], [10.0, 20.0])
        self.assertEqual(
            [event["alignedRangeId"] for event in result["serveEvents"]],
            ["R001", "R001"],
        )
        self.assertFalse(result["serveEvents"][1]["wasTimeAdjusted"])

    def test_excluded_rally_marker_is_not_imported(self) -> None:
        value = payload(
            [corrected("R001", 10.0, 15.0), corrected("R002", 20.0, 25.0)],
            [marker("serve-R001", 10.0, "R001"), marker("serve-R002", 20.0, "R002")],
            cut_ids=["R001"],
            excluded=["R002"],
        )

        result = normalize_feedback_annotations(value)

        self.assertEqual([event["id"] for event in result["serveEvents"]], ["serve-R001"])


if __name__ == "__main__":
    unittest.main()
