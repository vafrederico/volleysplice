import hashlib
import json
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
WINNER_PATH = REPOSITORY / "data/side-switch-current-research-winner-v1.json"


class SideSwitchCurrentWinnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(WINNER_PATH.read_text())
        self.evaluation = self.manifest["primaryEvaluation"]
        self.rows = list(self.evaluation["byRecording"].values())

    def test_promoted_identity_is_research_only(self) -> None:
        self.assertEqual(
            self.manifest["winner"]["id"],
            "side-switch-hard-negative-mining-v1/union34-top2-x2",
        )
        self.assertEqual(self.manifest["status"], "research-only")
        self.assertFalse(self.manifest["automaticProductionUse"])
        self.assertFalse(self.manifest["onDevicePortAvailable"])

    def test_per_recording_counts_reproduce_pooled_metrics(self) -> None:
        recordings = self.evaluation["recordings"]
        self.assertEqual(recordings, len(self.rows))
        for key in (
            "humanEvents",
            "proposals",
            "truePositives",
            "falsePositives",
            "falseNegatives",
        ):
            self.assertEqual(self.evaluation[key], sum(row[key] for row in self.rows))

        true_positives = self.evaluation["truePositives"]
        proposals = self.evaluation["proposals"]
        human_events = self.evaluation["humanEvents"]
        precision = true_positives / proposals
        recall = true_positives / human_events
        f1 = 2.0 * precision * recall / (precision + recall)
        self.assertAlmostEqual(self.evaluation["pooledPrecision"], precision)
        self.assertAlmostEqual(self.evaluation["pooledRecall"], recall)
        self.assertAlmostEqual(self.evaluation["pooledF1"], f1)

    def test_per_video_averages_are_reproducible(self) -> None:
        count = len(self.rows)
        self.assertAlmostEqual(
            self.evaluation["macroPerVideoPrecision"],
            sum(row["precision"] for row in self.rows) / count,
        )
        self.assertAlmostEqual(
            self.evaluation["macroPerVideoRecall"],
            sum(row["recall"] for row in self.rows) / count,
        )
        for key in (
            "proposals",
            "truePositives",
            "falsePositives",
            "falseNegatives",
        ):
            self.assertAlmostEqual(
                self.evaluation["averagePerVideo"][key],
                self.evaluation[key] / count,
            )

    def test_previous_winner_archive_is_exact(self) -> None:
        previous = self.manifest["previousWinner"]
        archive = REPOSITORY / previous["manifestPath"]
        self.assertTrue(archive.is_file())
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), previous["manifestSha256"])


if __name__ == "__main__":
    unittest.main()
