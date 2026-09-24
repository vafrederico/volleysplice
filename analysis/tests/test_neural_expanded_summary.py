from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/summarize-neural-expanded.py"
SPEC = importlib.util.spec_from_file_location("expanded_summary_tested", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def selection(recall=.96):
    return {"recallEligibilityPassed": recall >= .95,
            "recallEligibilityFloor": .95, "innerR_core": recall}


class ExpandedSummaryTests(unittest.TestCase):
    def test_empty_source_group_outcome_slice_is_unavailable(self):
        evaluation = summary.evaluate_predictions([{"id": "x", "sourceGroup": "a", "durationSeconds": 10,
                                                    "rallies": [{"start": 2, "end": 4}],
                                                    "predictions": [{"start": 2, "end": 4}]}])
        compact = summary.compact_evaluation(evaluation["sourceGroups"]["a"])
        self.assertEqual(compact["outcomeRecall"]["ace"]["rallies"], 0)
        self.assertIsNone(compact["outcomeRecall"]["ace"]["strictMatchRecall"])

    def test_outer_floor_is_diagnostic_not_an_unregistered_gate(self):
        candidates = [{"evaluation": {"primary": {"R_core": .93}}, "selections": [selection()]} for _ in range(3)]
        pairs = [{"primaryDelta": {"F1_padP_coreR": .03, "R_core": -.004},
                  "sourceGroups": {"a": {"F1_padP_coreRDelta": .02}, "b": {"F1_padP_coreRDelta": .01},
                                   "c": {"F1_padP_coreRDelta": -.01}}} for _ in range(3)]
        result = summary.feasibility(candidates, pairs, .95)
        self.assertEqual(len(result["checks"]), 4)
        self.assertTrue(result["passed"])
        self.assertTrue(result["screenPassedAndInnerFeasible"])
        self.assertFalse(result["outerRecallFloorDiagnostic"]["allCandidateSeedsAtLeastInnerFloor"])
        self.assertFalse(result["productionPromotionAllowed"])
        candidates[0]["selections"] = [selection(.94)]
        result = summary.feasibility(candidates, pairs, .95)
        self.assertTrue(result["passed"])
        self.assertFalse(result["allCandidateInnerSelectionsFeasible"])
        self.assertFalse(result["screenPassedAndInnerFeasible"])

    def test_contradictory_selected_recall_is_rejected(self):
        row = selection(.94)
        row["recallEligibilityPassed"] = True
        with self.assertRaisesRegex(ValueError, "contradicts"):
            summary.selection_feasible(row, .95)

    def test_paired_exposure_accepts_common_prefix_but_detects_drift(self):
        first = {"trainIds": ["x"], "scalerTrainIds": ["x"], "positiveWeight": [1, 2, 3, 4],
                 "history": [{"epoch": i, "optimizerSteps": i*10,
                              "exposureSha256": {"exact": f"e{i}", "draft": f"d{i}"}} for i in (1, 2)]}
        second = copy.deepcopy(first)
        second["history"].append({"epoch": 3, "optimizerSteps": 30, "exposureSha256": {"exact": "e3", "draft": "d3"}})
        self.assertEqual(summary.audit_exposure_pair(first, second, compare_draft=True), 2)
        for field in ("exact", "draft"):
            changed = copy.deepcopy(second)
            changed["history"][0]["exposureSha256"][field] = "different"
            with self.assertRaisesRegex(ValueError, "exposure differs"):
                summary.audit_exposure_pair(first, changed, compare_draft=True)
        changed = copy.deepcopy(second)
        changed["history"][1]["optimizerSteps"] += 1
        with self.assertRaisesRegex(ValueError, "optimizer-step"):
            summary.audit_exposure_pair(first, changed, compare_draft=True)

    def test_coverage_has_no_live_negatives_and_respects_game_window_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez(path, times=np.arange(0, 5, .25), metadata_json=json.dumps({"duration": 5}))
            row = {"featureCaches": {"audiovisual": {"path": str(path)}},
                   "ignoredIntervals": [{"start": 2, "end": 3}], "gameWindow": {"start": 1, "end": 4},
                   "keepTargets": [{"start": 1.5, "end": 3.5}]}
            counts = summary.expected_supervision(row, "coverage")
            self.assertEqual(counts["valid"], [0, 0, 0, 8])
            self.assertEqual(counts["positiveMass"], [0, 0, 0, 4])

    def test_short_draft_interval_is_unknown_not_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez(path, times=np.arange(0, 5, .25), metadata_json=json.dumps({"duration": 5}))
            row = {"featureCaches": {"audiovisual": {"path": str(path)}},
                   "rallies": [{"start": 2, "end": 2.5}], "ignoredIntervals": []}
            counts = summary.expected_supervision(row, "draft")
            # [1,3.5] is unknown; only nine outside ticks are negative supervision.
            self.assertEqual(counts["valid"], [9, 0, 0, 0])
            self.assertEqual(counts["positiveMass"], [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
