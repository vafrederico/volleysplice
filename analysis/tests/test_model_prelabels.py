from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.cli import build_parser
from analysis.model_prelabels import materialize_model_prelabels
from analysis.schema import ManifestError


class ModelPrelabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tasks = self.root / "tasks" / "full"
        self.candidates = self.root / "candidates"
        self.output = self.root / "prelabels" / "sol-xhigh"
        self.proxy = self.root / "proxies" / "grass" / "match-full.mp4"
        self.tasks.mkdir(parents=True)
        self.candidates.mkdir(parents=True)
        self.proxy.parent.mkdir(parents=True)
        self.proxy.write_bytes(b"proxy")
        task = {
            "schemaVersion": 1,
            "kind": "volleycut-rally-labels",
            "createdAt": "2026-08-10T00:00:00+00:00",
            "recording": {
                "id": "match-full",
                "video": "../../proxies/grass/match-full.mp4",
                "videoFilename": "match-full.mp4",
                "contentSha256": "0" * 64,
                "durationSeconds": 100.0,
                "sourceGroup": "match",
                "split": "train",
                "environment": "grass",
                "game": {"playersPerTeam": 2, "targetPoints": None, "format": "2v2"},
                "capture": {},
                "roi": None,
            },
            "annotationPolicy": {
                "id": "serve-contact-to-dead-ball-v1",
                "rallyStart": "serve-ball contact",
                "rallyEnd": "first instant live play has ended",
                "intervalConvention": "half-open [start,end) seconds on this normalized video",
            },
            "annotation": {
                "status": "not-started",
                "annotator": "",
                "continuousVideoReviewed": False,
                "reviewedAt": None,
                "notes": "",
            },
            "rallies": [],
            "ignoredIntervals": [],
            "hardNegatives": [],
        }
        (self.tasks / "match-full.labels.json").write_text(json.dumps(task), encoding="utf-8")
        self.candidate = {
            "schemaVersion": 1,
            "recordingId": "match-full",
            "videoPath": str(self.proxy),
            "durationSeconds": 100.0,
            "analysisMethod": "blind-gpt-5.6-sol-xhigh-audiovisual",
            "events": [
                {
                    "serveContact": 10.04,
                    "rallyEnd": 20.06,
                    "serveConfidence": "high",
                    "endConfidence": "medium",
                    "notes": "clear serve; whistle near the end",
                }
            ],
            "ambiguities": [],
            "analyzedAt": "2026-08-10T00:00:00Z",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_candidate(self) -> None:
        (self.candidates / "match-full.candidates.json").write_text(
            json.dumps(self.candidate),
            encoding="utf-8",
        )

    def test_materializes_isolated_unvalidated_label_and_reuses_it(self) -> None:
        self.write_candidate()

        first = materialize_model_prelabels(self.candidates, self.tasks, self.output)
        second = materialize_model_prelabels(self.candidates, self.tasks, self.output)

        self.assertEqual((first["created"], first["reused"], first["rallies"]), (1, 0, 1))
        self.assertEqual((second["created"], second["reused"]), (0, 1))
        payload = json.loads((self.output / "match-full.labels.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["annotation"]["status"], "in-progress")
        self.assertEqual(
            payload["annotation"]["annotator"],
            "GPT-5.6 Sol xhigh (unvalidated)",
        )
        self.assertFalse(payload["annotation"]["continuousVideoReviewed"])
        self.assertEqual(payload["rallies"][0]["start"], 10.04)
        self.assertIn("serve-confidence:high", payload["rallies"][0]["tags"])
        self.assertEqual(
            payload["prelabel"]["analysisMethod"],
            "blind-gpt-5.6-sol-xhigh-audiovisual",
        )

    def test_materializes_explicit_high_effort_provenance_and_matching_annotator(
        self,
    ) -> None:
        method = "blind-gpt-5.6-sol-high-audiovisual"
        self.candidate["analysisMethod"] = method
        self.write_candidate()

        result = materialize_model_prelabels(
            self.candidates,
            self.tasks,
            self.output,
            analysis_method=method,
        )

        payload = json.loads(
            (self.output / "match-full.labels.json").read_text(encoding="utf-8")
        )
        self.assertEqual(result["analysisMethod"], method)
        self.assertEqual(payload["prelabel"]["analysisMethod"], method)
        self.assertEqual(
            payload["annotation"]["annotator"],
            "GPT-5.6 Sol high (unvalidated)",
        )

    def test_rejects_high_effort_candidate_without_explicit_method(self) -> None:
        self.candidate["analysisMethod"] = "blind-gpt-5.6-sol-high-audiovisual"
        self.write_candidate()

        with self.assertRaisesRegex(ManifestError, "analysisMethod must be"):
            materialize_model_prelabels(self.candidates, self.tasks, self.output)

    def test_import_cli_accepts_explicit_high_effort_method(self) -> None:
        parsed = build_parser().parse_args(
            [
                "import-model-prelabels",
                "--candidates-dir",
                "candidates",
                "--tasks-dir",
                "tasks",
                "--output-dir",
                "output",
                "--analysis-method",
                "blind-gpt-5.6-sol-high-audiovisual",
            ]
        )
        self.assertEqual(
            parsed.analysis_method,
            "blind-gpt-5.6-sol-high-audiovisual",
        )

    def test_rejects_overlapping_candidate_rallies(self) -> None:
        self.candidate["events"].append(
            {
                "serveContact": 19.0,
                "rallyEnd": 30.0,
                "serveConfidence": "medium",
                "endConfidence": "low",
                "notes": "",
            }
        )
        self.write_candidate()

        with self.assertRaisesRegex(ManifestError, "ordered and non-overlapping"):
            materialize_model_prelabels(self.candidates, self.tasks, self.output)

    def test_rejects_candidate_for_a_different_video(self) -> None:
        self.candidate["videoPath"] = str(self.root / "other.mp4")
        self.write_candidate()

        with self.assertRaisesRegex(ManifestError, "videoPath does not match"):
            materialize_model_prelabels(self.candidates, self.tasks, self.output)

    def test_materializes_zero_event_candidate_for_human_review(self) -> None:
        self.candidate["events"] = []
        self.candidate["ambiguities"] = ["No plausible rallies found in the blind pass."]
        self.write_candidate()

        result = materialize_model_prelabels(self.candidates, self.tasks, self.output)

        self.assertEqual(result["rallies"], 0)
        payload = json.loads((self.output / "match-full.labels.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["rallies"], [])
        self.assertEqual(payload["annotation"]["status"], "in-progress")
        self.assertFalse(payload["annotation"]["continuousVideoReviewed"])


if __name__ == "__main__":
    unittest.main()
