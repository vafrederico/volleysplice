from analysis.private_ledger import private_value
import unittest
from pathlib import Path

from analysis.source_exposure_inventory import (COVERAGE, choose_feedback, eligible_for_training,
                                                 exposure_status, feedback_targets, label_tier)


class InventoryTests(unittest.TestCase):
    def test_complete_weak_export_never_becomes_exact(self):
        doc = {"annotation": {"status": "complete", "annotator": "Vini", "continuousVideoReviewed": True,
                              "notes": "Imported weak coverage"}, "rallies": [{"start": 1, "end": 3}]}
        self.assertEqual(label_tier(doc), COVERAGE)

    def test_protected_group_never_training_even_auxiliary(self):
        row = {"environment": "indoor", "sourceGroup": private_value('source-group-008'), "tier": COVERAGE}
        self.assertFalse(eligible_for_training(row))

    def test_project_alias_and_group_related_exposure(self):
        head = {"id": "suppression", "trainingRecordingIds": ["project-abc"],
                "trainingSourceGroups": ["session"], "fitScopeComplete": True, "calibrationScopeComplete": True}
        alias = {"id": "raw-abc", "sourceGroup": "session", "aliases": ["project-abc"]}
        related = {"id": "raw-other", "sourceGroup": "session", "aliases": []}
        self.assertEqual(exposure_status(alias, [head])["status"], "same-source-training")
        self.assertEqual(exposure_status(related, [head])["status"], "same-group-training")
        self.assertFalse(exposure_status(alias, [head])["primaryTrainingClean"])

    def test_calibration_related_is_training_clean_but_not_strict(self):
        head = {"id": "suppression", "trainingRecordingIds": [], "calibrationRecordingIds": ["other-set"],
                "calibrationSourceGroups": ["protected"], "fitScopeComplete": True, "calibrationScopeComplete": True}
        row = {"id": "protected-set", "sourceGroup": "protected", "aliases": []}
        result = exposure_status(row, [head])
        self.assertTrue(result["primaryTrainingClean"])
        self.assertFalse(result["strictNoFitOrCalibration"])

    def test_missing_lineage_is_unknown_not_clean(self):
        result = exposure_status({"id": "new", "aliases": [], "sourceGroup": "g"}, [{"id": "head"}])
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["primaryTrainingClean"])

    def test_feedback_timestamp_collision_rejected(self):
        base = {"source": {}, "generatedAt": "a", "corrections": {"updatedAt": "b"}, "finalExportIntervals": []}
        changed = {**base, "finalExportIntervals": [{"start": 1, "end": 2}]}
        with self.assertRaises(ValueError):
            choose_feedback([(Path("a"), base), (Path("b"), changed)])

    def test_export_mapping_preserves_seams_not_new_rallies(self):
        doc = {"source": {"media": {"duration": 20}, "gameWindow": {"start": 2, "end": 19}},
               "corrections": {"correctedRanges": [
                   {"id": "a", "coreStart": 3, "coreEnd": 5, "included": True},
                   {"id": "b", "coreStart": 10, "coreEnd": 12, "included": True},
                   {"id": "c", "coreStart": 15, "coreEnd": 16, "included": False}]},
               "finalExportIntervals": [{"start": 2, "end": 7, "cutIds": ["a"]},
                                        {"start": 8, "end": 14, "cutIds": ["b"]}]}
        cores, keep, ignored, mapping = feedback_targets(doc)
        self.assertEqual([(r["start"], r["end"]) for r in cores], [(3, 5), (10, 12)])
        self.assertEqual(keep, [{"start": 2., "end": 7.}, {"start": 8., "end": 14.}])
        self.assertEqual(mapping[1]["outputStart"], 5.)
        self.assertEqual(mapping[1]["sourceStart"], 8.)
        self.assertEqual(len(ignored), 2)


if __name__ == "__main__":
    unittest.main()
