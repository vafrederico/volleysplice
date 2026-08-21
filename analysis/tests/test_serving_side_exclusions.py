from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.serving_side_exclusions import (
    load_source_quality_exclusions,
    samples_touch_source_exclusion,
)


class ServingSideSourceExclusionsTest(unittest.TestCase):
    def test_loads_versioned_intervals_and_uses_half_open_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exclusions.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "volleycut-serving-side-source-quality-exclusions-v1",
                        "createdAt": "2026-08-21T00:00:00Z",
                        "records": [
                            {
                                "recordingId": "camera-hit",
                                "durationSeconds": 20,
                                "intervals": [
                                    {
                                        "start": 10,
                                        "end": 20,
                                        "reason": "rotated-view",
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            exclusions = load_source_quality_exclusions(path)
            self.assertFalse(
                samples_touch_source_exclusion(exclusions, "camera-hit", [9.999])
            )
            self.assertTrue(
                samples_touch_source_exclusion(exclusions, "camera-hit", [10.0])
            )
            self.assertFalse(
                samples_touch_source_exclusion(exclusions, "camera-hit", [20.0])
            )

    def test_rejects_overlapping_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exclusions.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "volleycut-serving-side-source-quality-exclusions-v1",
                        "records": [
                            {
                                "recordingId": "camera-hit",
                                "durationSeconds": 20,
                                "intervals": [
                                    {"start": 5, "end": 12, "reason": "first"},
                                    {"start": 10, "end": 20, "reason": "second"},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid or overlaps"):
                load_source_quality_exclusions(path)


if __name__ == "__main__":
    unittest.main()
