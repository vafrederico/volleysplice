from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/summarize-neural-development.py"
SPEC = importlib.util.spec_from_file_location("neural_summary_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class NeuralSummaryRevisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "id": "recording", "sourceGroup": "group", "durationSeconds": 20.0,
            "rallies": [{"start": 5.0, "end": 10.0, "tags": ["ace"]}],
            "ignoredIntervals": [{"start": 15.0, "end": 16.0}],
        }
        self.manifest = {"recordings": [copy.deepcopy(self.row)]}

    def test_exact_revision_passes(self) -> None:
        summary.assert_result_revision(self.manifest, [self.row], "candidate")

    def test_outside_core_ignored_revision_mismatch_is_rejected(self) -> None:
        changed = copy.deepcopy(self.row)
        changed["ignoredIntervals"] = [{"start": 17.0, "end": 18.0}]
        # Both versions leave all five core seconds untouched.
        with self.assertRaisesRegex(ValueError, "exact ignoredIntervals revision"):
            summary.assert_result_revision(self.manifest, [changed], "candidate")

    def test_equal_duration_shifted_ignored_hole_is_rejected(self) -> None:
        self.manifest["recordings"][0]["ignoredIntervals"] = [{"start": 6.0, "end": 7.0}]
        changed = copy.deepcopy(self.row)
        changed["ignoredIntervals"] = [{"start": 8.0, "end": 9.0}]
        # Both leave exactly four core seconds, but the evaluation universes differ.
        with self.assertRaisesRegex(ValueError, "exact ignoredIntervals revision"):
            summary.assert_result_revision(self.manifest, [changed], "candidate")

    def test_outcome_tag_revision_and_duplicate_recording_are_rejected(self) -> None:
        changed = copy.deepcopy(self.row)
        changed["rallies"][0]["tags"] = ["service-fault"]
        with self.assertRaisesRegex(ValueError, "exact rallies revision"):
            summary.assert_result_revision(self.manifest, [changed], "candidate")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summary.assert_result_revision(self.manifest, [self.row, self.row], "candidate")


class NeuralSummaryScopeTests(unittest.TestCase):
    def test_three_group_six_recording_summary_uses_frozen_counts(self) -> None:
        from analysis.neural_evaluation import evaluate_predictions

        groups = ["grass-a", "grass-b", "indoor-c"]
        seeds = [3407, 1729, 20260918]
        rallies = [{"start": 3, "end": 4, "tags": ["ace"]},
                   {"start": 8, "end": 9, "tags": ["service-fault"]},
                   {"start": 13, "end": 18}]
        rows = [{"id": f"recording-{index}", "sourceGroup": group, "durationSeconds": 20,
                 "rallies": rallies, "predictions": rallies}
                for index, group in enumerate(groups * 2)]
        evaluation = evaluate_predictions(rows)
        contract = {"groups": groups, "seeds": seeds, "kinds": ["linear"], "manifestSha256": "manifest"}
        contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report = {"contractSha256": contract_hash, "records": 6, "sourceGroups": groups,
                  "results": [{"kind": "linear", "seed": seed, "evaluation": evaluation,
                               "predictions": rows, "selections": []} for seed in seeds]}
        baseline = {"manifestSha256": "manifest", "results": {"shipped-union": evaluation}}
        verified = {"recordingCount": 6, "sourceGroups": groups}

        def load(path: Path) -> dict:
            return ({"contract": contract, "sha256": contract_hash} if path.name == "preregistration.json"
                    else report if path.name == "report.json" else baseline)

        with patch.object(summary, "load", side_effect=load), patch.object(Path, "exists", return_value=True), \
                patch.object(summary, "verify_frozen_inputs", return_value=verified), \
                patch.object(summary, "digest", return_value="artifact"):
            result = summary.summarize(Path("study"), Path("baseline.json"))
        self.assertEqual(result["scope"], {"recordingCount": 6, "sourceGroupCount": 3,
                                           "sourceGroups": groups, "seedCount": 3})
        text = summary.markdown(result)
        self.assertIn("3 source groups, 6 recordings", text)
        self.assertIn("Only 3 independent source groups", text)
        self.assertIn("average 3 separately pooled seed results", text)
        self.assertIn("at least 2 of 3 seeds", text)
        self.assertNotIn("Four source groups", text)
        self.assertNotIn("eight recordings", text)

        # Rendering another declared seed count uses the same unchanged 2/3 rule.
        result["scope"]["seedCount"] = 4
        text = summary.markdown(result)
        self.assertIn("average 4 separately pooled seed results", text)
        self.assertIn("at least 3 of 4 seeds", text)

    def test_conflicting_report_counts_cannot_relabel_the_frozen_scope(self) -> None:
        contract = {"groups": ["a", "b", "c"], "seeds": [7]}
        verified = {"recordingCount": 6, "sourceGroups": ["a", "b", "c"]}
        for report in ({"records": 8, "sourceGroups": ["a", "b", "c"]},
                       {"records": 6, "sourceGroups": ["a", "b", "different"]}):
            with self.subTest(report=report), self.assertRaisesRegex(ValueError, "counts differ"):
                summary.study_scope(contract, report, verified)


if __name__ == "__main__":
    unittest.main()
