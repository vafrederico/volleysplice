from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.production_eval_inference import (
    _requested_frame_indexes,
    _tied_percentile_ranks,
    generate_side_switch_candidates,
    merge_production_ranges,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "regenerate-exported-project-eval-inference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "exported_project_eval_inference_regeneration", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT_MODULE)


class ProductionEvalInferenceTests(unittest.TestCase):
    def test_production_range_merge_matches_overlap_union_contract(self) -> None:
        merged = merge_production_ranges(
            {
                "all-labels-v2": [
                    {"start": 1.0, "end": 3.0, "confidence": 0.8},
                    {"start": 5.0, "end": 6.0, "confidence": 0.9},
                ],
                "previous-production": [
                    {"start": 2.0, "end": 4.0, "confidence": 0.6},
                    # Touching intervals stay separate in the production ensemble.
                    {"start": 6.0, "end": 7.0, "confidence": 0.5},
                ],
            }
        )

        self.assertEqual(
            [(row["id"], row["start"], row["end"], row["agreement"]) for row in merged],
            [
                ("R001", 1.0, 4.0, "both-models"),
                ("R002", 5.0, 6.0, "all-labels-v2-only"),
                ("R003", 6.0, 7.0, "previous-production-only"),
            ],
        )
        self.assertAlmostEqual(merged[0]["confidence"], 0.7)
        self.assertAlmostEqual(merged[1]["confidence"], 0.49)
        self.assertAlmostEqual(merged[2]["confidence"], 0.3)

    def test_specialist_sampling_uses_latest_frame_at_or_before_target(self) -> None:
        indexes = _requested_frame_indexes(
            [0.0, 0.016, 0.017, 1.999, 99.0],
            duration=2.1,
            fps=60.0,
            frame_count=126,
        )

        self.assertEqual(tuple(indexes), (0, 1, 119, 125))
        self.assertEqual(len(indexes[0]), 2)
        self.assertEqual(len(indexes[1]), 1)

    def test_serving_side_tied_ranks_are_within_recording_midranks(self) -> None:
        ranked = _tied_percentile_ranks(
            np.asarray([[2.0, 4.0], [1.0, 4.0], [2.0, 9.0]], dtype=np.float64)
        )

        np.testing.assert_allclose(
            ranked,
            np.asarray([[0.75, 0.25], [0.0, 0.25], [0.75, 1.0]]),
        )

    def test_side_switch_candidates_depend_only_on_predicted_ranges_and_scores(self) -> None:
        runtime = {
            "candidateGenerator": {
                "internalPeakThreshold": 0.98,
                "internalPeakMinimumSeparationSeconds": 14.0,
                "internalPeakRangeEdgeExclusionSeconds": 4.0,
                "internalPeakProposalHalfWidthSeconds": 1.0,
            }
        }
        ranges = [
            {"id": "R001", "start": 0.0, "end": 20.0, "included": True},
            {"id": "R002", "start": 30.0, "end": 40.0, "included": True},
        ]
        times = np.arange(0.0, 40.25, 0.25)
        dead = np.zeros_like(times)
        dead[np.where(times == 10.0)[0][0]] = 0.99

        candidates = generate_side_switch_candidates(ranges, times, dead, runtime)

        self.assertEqual(
            [candidate["kind"] for candidate in candidates],
            ["internal-dead-state-peak", "adjacent-rally-boundary"],
        )
        self.assertEqual(candidates[0]["sourceRangeIds"], ["R001"])
        self.assertEqual(candidates[1]["sourceRangeIds"], ["R001", "R002"])

    def test_nvdec_workers_start_only_above_eight_logical_cpus(self) -> None:
        with patch.object(SCRIPT_MODULE.os, "cpu_count", return_value=8):
            self.assertEqual(
                SCRIPT_MODULE._worker_layout(
                    7, 0, SCRIPT_MODULE.NVDEC_VIDEO_DECODER
                ),
                (1, 8, 0),
            )
        with patch.object(SCRIPT_MODULE.os, "cpu_count", return_value=24):
            self.assertEqual(
                SCRIPT_MODULE._worker_layout(
                    7, 0, SCRIPT_MODULE.NVDEC_VIDEO_DECODER
                ),
                (7, 3, 2),
            )

    def test_cpu_specialist_decoding_is_always_serial(self) -> None:
        with patch.object(SCRIPT_MODULE.os, "cpu_count", return_value=24):
            self.assertEqual(
                SCRIPT_MODULE._worker_layout(
                    7, 0, SCRIPT_MODULE.OPENCV_VIDEO_DECODER
                ),
                (1, 24, 0),
            )


if __name__ == "__main__":
    unittest.main()
