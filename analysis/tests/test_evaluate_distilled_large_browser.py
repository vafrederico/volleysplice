"""Integrity and temporal-alignment guards for independent browser evaluation."""
import hashlib
import importlib.util
import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "browser_evaluation", Path(__file__).resolve().parents[2] / "scripts/evaluate-distilled-large-browser.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class BrowserEvaluationTests(unittest.TestCase):
    @staticmethod
    def file_timing_receipt():
        return dict(source=dict(transport="file"),
                    timingIntegrity=dict(clean=True, sourceTransport="file",
                                         sourceCachePolicy="BlobSource filesystem reads; no HTTP media transport"),
                    sourceIo=dict(requests=0, sourceBytesRead=None, streamErrors=[], rejectedRanges=0,
                                  failedBrowserRequests=[]), consoleDiagnostics=[])

    def test_file_timing_accepts_unmeasured_filesystem_reads(self):
        receipt = self.file_timing_receipt()
        actual = evaluation.timing_integrity(receipt)
        self.assertTrue(actual["clean"])
        self.assertEqual(actual["sourceTransport"], "file")
        self.assertEqual(actual["sourceCachePolicy"], receipt["timingIntegrity"]["sourceCachePolicy"])

    def test_file_timing_rejects_contradictory_transport_or_http_reads(self):
        original = self.file_timing_receipt()
        for section, field, value, message in (
            ("timingIntegrity", "sourceTransport", "url", "transport disagree"),
            ("sourceIo", "requests", 1, "must not claim HTTP media reads"),
            ("sourceIo", "sourceBytesRead", 0, "must not claim HTTP media reads"),
        ):
            with self.subTest(field=field):
                receipt = copy.deepcopy(original)
                receipt[section][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    evaluation.timing_integrity(receipt)

    def test_file_timing_recomputes_dirty_status_from_diagnostics(self):
        original = self.file_timing_receipt()
        for section, field, value in (
            ("sourceIo", "streamErrors", ["stream failed"]),
            ("sourceIo", "rejectedRanges", 1),
            ("sourceIo", "failedBrowserRequests", [dict(error="net::ERR_FAILED")]),
            (None, "consoleDiagnostics", [dict(message="Retrying failed fetch")]),
        ):
            with self.subTest(field=field):
                receipt = copy.deepcopy(original)
                target = receipt[section] if section else receipt
                target[field] = value
                with self.assertRaisesRegex(ValueError, "status differs"):
                    evaluation.timing_integrity(receipt)
                receipt["timingIntegrity"]["clean"] = False
                self.assertFalse(evaluation.timing_integrity(receipt)["clean"])

    def test_artifact_identity_and_shape_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = np.array([1, 2, 3], dtype="<f4").tobytes()
            path = root / "values.f32"
            path.write_bytes(raw)
            entries = {path.name: dict(sizeBytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())}
            artifacts = evaluation.Artifacts(root, entries)
            np.testing.assert_array_equal(artifacts.array(path.name, "<f4", (3,)), [1, 2, 3])
            with self.assertRaisesRegex(ValueError, "size differs"):
                artifacts.array(path.name, "<f4", (2,))
            path.write_bytes(raw[:-4])
            with self.assertRaisesRegex(ValueError, "identity differs"):
                evaluation.Artifacts(root, entries)

    def test_unregistered_and_parent_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "filename must be local"):
                evaluation.Artifacts(Path(temporary), {"../outside.f32": {}})

    def test_rank_preserves_absolute_columns_and_ties(self):
        names = ["ordinary"] * 104
        names[3] = "audio_available"
        raw = np.tile(np.array([8, 2, 2, 9], np.float32)[:, None], (1, 104))
        raw[:, 3] = [1, 0, 1, 1]
        ranked = evaluation.rank_av(raw, names)
        np.testing.assert_array_equal(ranked[:, 3], raw[:, 3])
        np.testing.assert_allclose(ranked[:, 0], [2 / 3, 1 / 6, 1 / 6, 1], atol=1e-7)

    def test_hold_uses_nominal_grid_not_nearest_embedding(self):
        times = np.array([0., .25, .5, .75])
        embedding_times = np.array([0., .5])
        tokens = np.stack((np.ones(3840, np.float32), np.full(3840, 2, np.float32)))
        config = dict(mean=[0.] * 112, scale=[1.] * 112)
        actual = evaluation.fused_inputs(times, np.zeros((4, 104), np.float32), embedding_times,
                                         tokens, np.zeros((2, 6), np.float32), config)
        np.testing.assert_array_equal(actual[:, 104], [1, 1, 2, 2])
        np.testing.assert_array_equal(actual[:, -2], [0, .25, 0, .25])
        np.testing.assert_array_equal(actual[:, -1], 1)
        with self.assertRaisesRegex(ValueError, "Incomplete encoder coverage"):
            evaluation.fused_inputs(np.array([1.25]), np.zeros((1, 104), np.float32), embedding_times,
                                    tokens, np.zeros((2, 6), np.float32), config)


if __name__ == "__main__":
    unittest.main()
