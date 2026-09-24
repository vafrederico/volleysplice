from analysis.private_ledger import private_value
import unittest

from analysis.export_rally_proxy_inventory import derive_proxy_record


class ProxyTests(unittest.TestCase):
    def inputs(self):
        row = {"id": "raw", "sourceGroup": private_value('source-group-006'), "environment": "grass",
               "video": "raw.mp4", "roi": None, "durationSeconds": 20, "protected": False,
               "ignoredIntervals": [{"start": 0, "end": 4}], "gameWindow": {"start": 0, "end": 20},
               "keepTargets": [{"start": 1, "end": 15}],
               "coverageReview": {"exhaustiveKeptDiscardedReviewConfirmed": True},
               "feedback": {"path": "feedback.json", "sha256": "x", "sizeBytes": 20}}
        feedback = {"corrections": {"correctedRanges": [
            {"id": "a", "coreStart": 2, "coreEnd": 6, "included": True},
            {"id": "b", "coreStart": 10, "coreEnd": 13, "included": True},
            {"id": "micro", "coreStart": 14, "coreEnd": 14.1, "included": True}]},
            "finalExportIntervals": [{"start": 1, "end": 15, "cutIds": ["a", "b", "micro"]}]}
        return row, feedback

    def test_ignored_mask_never_invents_serve_boundary(self):
        row, feedback = self.inputs()
        result = derive_proxy_record(row, feedback)
        self.assertEqual([(r["id"], r["start"], r["end"]) for r in result["rallies"]], [("a", 2., 6.), ("b", 10., 13.)])
        self.assertTrue(result["derivation"]["retained"][0]["startInsideIgnored"])
        self.assertEqual(result["derivation"]["excluded"][0]["reason"], "micro-range-under-0.25s")

    def test_reserved_source_rejected(self):
        row, feedback = self.inputs(); row["sourceGroup"] = private_value('source-group-001')
        with self.assertRaises(ValueError):
            derive_proxy_record(row, feedback)

    def test_joined_export_cannot_supply_core_identity(self):
        row, feedback = self.inputs(); feedback["finalExportIntervals"][0]["cutIds"] = []
        with self.assertRaises(ValueError):
            derive_proxy_record(row, feedback)

    def test_duplicate_core_is_explicitly_dropped(self):
        row, feedback = self.inputs(); feedback["corrections"]["correctedRanges"].append(dict(feedback["corrections"]["correctedRanges"][1]))
        result = derive_proxy_record(row, feedback)
        self.assertEqual(len(result["rallies"]), 2)
        self.assertIn("duplicate-core-identity-or-range", [r["reason"] for r in result["derivation"]["excluded"]])


if __name__ == "__main__":
    unittest.main()
