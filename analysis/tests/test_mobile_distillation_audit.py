import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit-neural-mobile-distillation.py"
spec = importlib.util.spec_from_file_location("mobile_distillation_independent_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def record(identifier, group):
    return SimpleNamespace(example=SimpleNamespace(id=identifier, group=group))


class DistillationAuditTests(unittest.TestCase):
    def test_vfr_seek_rounding_requires_exact_sha_and_pts(self):
        class RoundedCapture:
            position = 0
            def set(self, _, position):
                self.position = position + 1 if position else 0
            def read(self):
                self.grab()
                return self.retrieve()
            def grab(self):
                self.last = self.position
                self.position += 1
                return True
            def retrieve(self):
                return True, np.full((2, 2, 3), self.last, dtype=np.uint8)
            def get(self, _):
                return self.last * 1000 / 30
        expected = np.full((2, 2, 3), 20, dtype=np.uint8)
        digest = hashlib.sha256(expected.tobytes()).hexdigest()
        actual = audit.exact_source_frame(RoundedCapture(), 20, 20 / 30, digest)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "Exact sampled source"):
            audit.exact_source_frame(RoundedCapture(), 20, 20 / 30, "wrong-pixels")
        with self.assertRaisesRegex(ValueError, "Exact sampled source"):
            audit.exact_source_frame(RoundedCapture(), 20, 20 / 30 + .001, digest)

    def test_both_inner_held_sources_excluded_even_as_auxiliary(self):
        data = {"exact": [record("outer", "g0"), record("inner", "g1"), record("fit", "g2")],
                "draft": [record("raw-g0", "g0"), record("project-g1", "g1"), record("other", "g3")],
                "coverage": [record("export-g0", "g0"), record("export-g1", "g1"), record("raw-fit", "g2")]}
        meta = {"contractSha256": "study", "seed": 7, "excludedGroups": ["g0", "g1"],
                "trainIds": ["fit", "other", "raw-fit"], "trainGroups": ["g2", "g3"]}
        _, permitted, held = audit.check_student_membership(meta, data, {"g0", "g1"}, 7, "study")
        self.assertEqual([r.example.id for r in permitted], ["fit", "other", "raw-fit"])
        self.assertEqual([r.example.id for r in held], ["outer", "inner"])
        for forbidden in ("raw-g0", "project-g1", "export-g0", "export-g1"):
            broken = {**meta, "trainIds": meta["trainIds"] + [forbidden]}
            with self.assertRaisesRegex(ValueError, "membership"):
                audit.check_student_membership(broken, data, {"g0", "g1"}, 7, "study")

    def test_large_vfr_seek_error_falls_back_and_reuses_sequential_cursor(self):
        class BadSeekCapture:
            position = 0
            seeks = 0
            def set(self, _, position):
                self.seeks += 1
                self.position = position + 100 if position else 0
            def grab(self):
                self.last = self.position
                self.position += 1
                return True
            def retrieve(self):
                return True, np.full((1, 1, 3), self.last, dtype=np.uint8)
            def read(self):
                self.grab()
                return self.retrieve()
            def get(self, _):
                return self.last * 1000 / 30
        capture, state = BadSeekCapture(), {}
        for ordinal in (20, 45):
            expected = np.full((1, 1, 3), ordinal, dtype=np.uint8)
            actual = audit.exact_source_frame(capture, ordinal, ordinal / 30,
                hashlib.sha256(expected.tobytes()).hexdigest(), state)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(state["sequentialOrdinal"], ordinal)
        self.assertEqual(capture.seeks, 2)

    def test_99_percent_is_inclusive_but_no_infeasible_fallback(self):
        candidates = [{"innerR_core": .989, "innerF1_padP_coreR": .95, "epoch": i} for i in range(192)]
        result = audit.select99(candidates)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["selected"])
        candidates[3] = {"innerR_core": .99, "innerF1_padP_coreR": .7, "epoch": 3}
        self.assertEqual(audit.select99(candidates)["selected"]["epoch"], 3)

    def test_teacher_selection_excludes_existing_invalid_spans_and_caps_128(self):
        valid_times = np.arange(1600) / 4.
        times = np.arange(800) / 2.
        valid = (valid_times >= 25) & ((valid_times < 50) | (valid_times >= 200))
        actual = audit.independent_teaching_indexes(times, valid_times, valid)
        self.assertEqual(len(actual), 128)
        self.assertEqual(actual[0], 50)
        self.assertEqual(actual[-1], 799)
        self.assertTrue(np.all(valid[actual * 2]))

    def test_wrong_source_groups_are_rejected_even_when_train_ids_match(self):
        data = {"exact": [record("held", "g0"), record("fit", "g1")], "draft": [], "coverage": []}
        meta = {"contractSha256": "c", "seed": 1, "excludedGroups": ["g0"], "trainIds": ["fit"], "trainGroups": ["g0"]}
        with self.assertRaisesRegex(ValueError, "trainGroups"):
            audit.check_student_membership(meta, data, {"g0"}, 1, "c")


if __name__ == "__main__":
    unittest.main()
