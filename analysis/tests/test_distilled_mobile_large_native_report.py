"""Protect native report scope, timing and saved-evidence requirements."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("distilled_native_report",
    Path(__file__).resolve().parents[2] / "scripts/report-distilled-mobile-large-native.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class NativeReportTest(unittest.TestCase):
    def test_excerpt_clips_boundaries_and_retains_original_numbers(self):
        labels = {"rallies": [{"start": 170, "end": 185}, {"start": 200, "end": 210},
                               {"start": 298, "end": 305}, {"start": 310, "end": 320}],
                  "ignoredIntervals": [{"start": 175, "end": 182}]}
        result = REPORT.shifted_labels(labels, 180, 120)
        self.assertEqual(result, {"rallies": [
            {"start": 0, "end": 5, "sourceHumanRallyNumber": 1},
            {"start": 20, "end": 30, "sourceHumanRallyNumber": 2},
            {"start": 118, "end": 120, "sourceHumanRallyNumber": 3}],
            "ignoredIntervals": [{"start": 0, "end": 2}]})

    def test_wholly_missed_is_export_coverage_not_event_match(self):
        row = {"rallies": [{"start": 4, "end": 5}]}
        gold = {"rallies": [{"start": 6, "end": 7}, {"start": 15, "end": 16},
                            {"start": 21, "end": 22}],
                "ignoredIntervals": [{"start": 20, "end": 23}]}
        result = REPORT.evaluate(row, gold, 30)
        self.assertEqual(result["whollyMissedSavedHumanRallies"], 1)
        self.assertEqual(result["whollyMissedHumanRallyRanges"][0]["humanRallyNumber"], 2)
        self.assertEqual(result["event"]["matchedRallies"], 0)
        self.assertEqual(result["primary"]["R_core"], .5)
        self.assertEqual([p["paddingSecondsBeforeAndAfter"] for p in result["padding"]], [0, 1, 2, 3])

    def test_timing_uses_wall_time_and_does_not_sum_nested_profiles(self):
        result = REPORT.timings({"totalMs": 10000,
            "stagesMs": {"score_specialists": 3000, "video_decode_and_features": 4000,
                         "audio_decode_and_features": 1000, "contextualize": 100},
            "profileMs": {"neural/embedding_video_pass": 1500, "score/shared_decode_wall": 2500},
            "neural": {"video": {"decodeAndOtherMs": 1000, "prepareMs": 300,
                                   "encoderAndReadbackMs": 200}}})
        self.assertEqual(result["ralliesReadySeconds"], 7)
        self.assertEqual(result["allReadySeconds"], 10)
        self.assertEqual(result["embeddingVideoPassSeconds"], 1.5)
        self.assertEqual(result["embeddingEncoderSeconds"], .2)

    def test_saved_neural_tensor_must_exist_with_actual_expected_size(self):
        row = {"id": "case", "sampleRows": 2,
               "neural": {"featureRows": 2, "featureDimension": 3952,
                          "video": {"sampleCount": 1}}}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Saved neural tensor"):
                REPORT.storage(Path(folder), row)

    def test_complete_result_requires_matching_run_and_no_partial_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "result.json").write_text(json.dumps({"status": "running", "runId": "a"}))
            (root / "pipeline-plan.json").write_text(json.dumps({"runId": "a", "cases": []}))
            with self.assertRaisesRegex(ValueError, "Completed matching"):
                REPORT.completed_rows(root, ["production"], 120, 3)

    def test_native_report_rejects_gold_changes_before_loading_benchmarks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root / "labels.json"
            labels.write_text('{"rallies": []}')
            with self.assertRaisesRegex(ValueError, "gold revision changed"):
                REPORT.build(root, labels, root, root, root)

    def test_provenance_accepts_equivalent_contract_formatting_and_production_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); graphs = root / "graphs"; graphs.mkdir()
            contract = {"roi": {"x": 0, "y": 0, "width": 1, "height": 1}}
            (graphs / "input-contract.json").write_text(json.dumps(contract) + "\n")
            (root / "input-contract.json").write_text(json.dumps(contract, indent=2))
            (root / "result.json").write_text(json.dumps({"results": [
                {"rallies": []}, {"neural": {"decoder": {"enter": .2}}}]}))
            (root / "pipeline-plan.json").write_text('{"runId": "run"}')
            (root / "temporal-decoder-parity.json").write_text('{"passed": true}')
            identity = {"decoder": {"enter": .2}, "graphFiles": [{"component": "graph.onnx", "sha256": "abc"}]}
            receipt = {"runId": "run", "apkSha256": REPORT.APK_SHA256,
                       "inputContractSha256": REPORT.digest(root / "input-contract.json"),
                       "planSha256": REPORT.digest(root / "pipeline-plan.json"),
                       "resultSha256": REPORT.digest(root / "result.json"),
                       "remoteGraphDigestVerifiedByDriverBeforeLaunch": True,
                       "graphHashes": {"graph.onnx": "abc"}}
            (root / "run-provenance.json").write_text(json.dumps(receipt))
            self.assertIn("runProvenanceSha256", REPORT.run_identity(root, {"runId": "run"}, identity, graphs))
            receipt["graphHashes"]["graph.onnx"] = "different-selected-weights"
            (root / "run-provenance.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "graph hashes differ"):
                REPORT.run_identity(root, {"runId": "run"}, identity, graphs)

    def test_reserved_group_does_not_imply_actual_selection_exposure(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps({"records": [{"id": "training", "sourceGroup": "fit"}]}))
            task = {"manifest": {"path": str(manifest), "sha256": REPORT.digest(manifest)},
                    "trainIds": ["training"], "calibrationIds": [], "commonEvaluationGroups": ["reserved"]}
            record = {"id": "benchmark", "sourceGroup": "reserved"}
            result = REPORT.exposure_for_task(record, task, [{"id": "selected", "sourceGroup": "other"}])
            self.assertTrue(result["reservedEvaluationGroup"])
            self.assertFalse(result["directTraining"])
            self.assertFalse(result["sourceGroupModelSelection"])
            self.assertFalse(result["directModelSelection"])


if __name__ == "__main__":
    unittest.main()
