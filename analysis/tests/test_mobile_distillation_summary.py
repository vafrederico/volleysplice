import copy
import importlib.util
from pathlib import Path
import unittest

from analysis.neural_evaluation import evaluate_predictions

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/summarize-neural-mobile-distillation.py"
spec = importlib.util.spec_from_file_location("mobile_distillation_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture(infeasible_last=False):
    rows = [{"id": f"record-{i}", "sourceGroup": f"group-{i // 2}", "durationSeconds": 30,
             "rallies": [{"start": 5, "end": 15}], "predictions": [{"start": 5, "end": 15}],
             "ignoredIntervals": []} for i in range(8)]
    cells = []
    for seed in summary.SEEDS:
        complete = not (infeasible_last and seed == summary.SEEDS[-1])
        selected = {"epoch": 5, "innerR_core": 1., "innerF1_padP_coreR": 1., "decoder": {}}
        selections = [{"heldSourceGroup": f"group-{i}", "recallEligibilityFloor": .99,
            "candidateCount": 192, "eligibleCandidateCount": 192 if complete or i != 3 else 0,
            "feasible": complete or i != 3, "maximumInnerRecall": 1. if complete or i != 3 else .98,
            "selected": selected if complete or i != 3 else None} for i in range(4)]
        predictions = rows if complete else rows[:6]
        cells.append({"seed": seed, "complete": complete, "selections": selections,
                      "predictions": predictions, "evaluation": evaluate_predictions(rows) if complete else None})
    gold = summary.signature(rows)
    return cells, gold, summary.universe_seconds(gold)


class DistillationSummaryTests(unittest.TestCase):
    def test_infeasible_third_seed_is_visible_and_forbids_subset_mean(self):
        cells, gold, universe = fixture(True)
        result = summary.describe_model(cells, gold, universe)
        self.assertEqual(result["completeSeeds"], [3407, 1729])
        self.assertFalse(result["modelRankEligible"])
        self.assertIsNone(result["meanAcrossAllThreeSeeds"])
        self.assertEqual(result["seeds"][2]["status"], "infeasible-inner-recall")
        self.assertIsNone(result["seeds"][2]["metrics"])
        full_cells, _, _ = fixture()
        full = summary.describe_model(full_cells, gold, universe)
        paired = summary.paired_differences(full, result)
        self.assertIsNone(paired["meanAcrossAllThreeSeeds"])
        self.assertEqual([r["bothCompleteEvaluationScopes"] for r in paired["seeds"]], [True, True, False])

    def test_infeasible_seed_cannot_supply_partial_pooled_evaluation(self):
        cells, gold, universe = fixture(True)
        cells[-1]["evaluation"] = evaluate_predictions(cells[-1]["predictions"])
        with self.assertRaisesRegex(ValueError, "Incomplete source scope"):
            summary.describe_model(cells, gold, universe)

    def test_mismatched_ignored_revision_is_rejected(self):
        cells, gold, universe = fixture()
        cells = copy.deepcopy(cells)
        cells[0]["predictions"][0]["ignoredIntervals"] = [{"start": 0, "end": 1}]
        with self.assertRaisesRegex(ValueError, "Gold revision"):
            summary.describe_model(cells, gold, universe)

    def test_original_long_rally_remains_long_after_ignored_fragmentation(self):
        rows = [{"id": "r", "sourceGroup": "g", "durationSeconds": 30,
                 "rallies": [{"start": 5, "end": 15}], "predictions": [{"start": 5, "end": 7}],
                 "ignoredIntervals": [{"start": 7, "end": 13}]}]
        evaluated = evaluate_predictions(rows)
        metrics = summary.describe_evaluation(evaluated, 24)
        self.assertEqual(metrics["primaryExportCoverage"]["shortOriginalRalliesAtMost3Seconds"]["evaluableRallies"], 0)
        self.assertEqual(metrics["primaryExportCoverage"]["longOriginalRalliesOver3Seconds"]["evaluableRallies"], 1)
        self.assertEqual(metrics["primaryExportCoverage"]["partialRallyLosses"], 1)
        self.assertEqual(metrics["event"]["ignoredTouchedRalliesExcludedFromEvents"], 1)
        self.assertEqual([p["paddingSecondsBeforeAndAfter"] for p in metrics["padding"]], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
