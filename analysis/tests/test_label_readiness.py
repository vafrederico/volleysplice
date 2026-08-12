from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.label_readiness import build_label_readiness_report


class LabelReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="volleycut-readiness-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_label(self, name: str, *, extended: bool) -> Path:
        rallies = [
            {"start": 5.0, "end": 10.0, "tags": ["ace"]},
            {"start": 20.0, "end": 30.0, "tags": []},
        ]
        if extended:
            rallies[0].update(
                {
                    "receiverReactionTime": 5.2,
                    "collectiveStandDownTime": 10.1,
                    "terminalCue": "no-recovery",
                    "endObservability": "observable",
                    "startConfidence": 0.9,
                    "endConfidence": 0.8,
                    "verifiedImmediateResult": True,
                    "playerTracklets": [
                        {
                            "trackId": track_id,
                            "window": window,
                            "team": team,
                            "courtSide": side,
                            "observations": [
                                {
                                    "time": first_time,
                                    "footpoint": {"x": x, "y": 0.7},
                                    "state": "ready" if window == "serve" else "playing",
                                },
                                {
                                    "time": second_time,
                                    "footpoint": {"x": x + 0.01, "y": 0.69},
                                    "state": (
                                        "playing" if window == "serve" else "stand-down"
                                    ),
                                },
                            ],
                        }
                        for window, first_time, second_time in (
                            ("serve", 4.8, 5.2),
                            ("rally-end", 9.8, 10.2),
                        )
                        for track_id, team, side, x in (
                            ("P1", "team-a", "near", 0.3),
                            ("P2", "team-b", "far", 0.7),
                        )
                    ],
                }
            )
        payload = {
            "schemaVersion": 1,
            "kind": "volleycut-rally-labels",
            "createdAt": "2026-08-12T00:00:00Z",
            "recording": {
                "id": name,
                "video": "missing.mp4",
                "videoFilename": "missing.mp4",
                "contentSha256": "0" * 64,
                "durationSeconds": 60,
                "sourceGroup": name,
                "split": "train",
                "environment": "grass",
                "game": {"playersPerTeam": 2, "targetPoints": 21, "format": "2v2"},
                "capture": {},
                "roi": None,
                **(
                    {
                        "courtGeometry": {
                            "corners": {
                                "nearLeft": {"x": 0.1, "y": 0.9},
                                "nearRight": {"x": 0.9, "y": 0.9},
                                "farLeft": {"x": 0.35, "y": 0.2},
                                "farRight": {"x": 0.65, "y": 0.2},
                            },
                            "netAnchors": {
                                "left": {"x": 0.2, "y": 0.5},
                                "right": {"x": 0.8, "y": 0.5},
                            },
                        }
                    }
                    if extended
                    else {}
                ),
            },
            "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
            "annotation": {
                "status": "in-progress",
                "annotator": "reviewer",
                "continuousVideoReviewed": False,
                "reviewedAt": None,
                "notes": "",
            },
            "rallies": rallies,
            "ignoredIntervals": [],
            "hardNegatives": (
                [
                    {"start": 35, "end": 37, "category": "walking-ball-retrieval"},
                    {"start": 40, "end": 42, "category": "celebration-huddle"},
                    {"start": 45, "end": 47, "category": "random-dead-control"},
                ]
                if extended
                else []
            ),
            "sideSwitches": [],
        }
        path = self.root / f"{name}.labels.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_report_tracks_per_recording_and_aggregate_label_debt(self) -> None:
        extended = self.write_label("extended", extended=True)
        legacy = self.write_label("legacy", extended=False)

        report = build_label_readiness_report([extended, legacy])

        self.assertEqual(report["documentsScanned"], 2)
        self.assertEqual(report["checklist"]["geometry"]["minimumReadyRecordings"], 1)
        self.assertEqual(report["checklist"]["geometry"]["missingCornerClicks"], 4)
        self.assertEqual(
            report["checklist"]["hardNegatives"]["additionalIntervalsToMinimumThree"],
            3,
        )
        transitions = report["checklist"]["transitions"]
        self.assertEqual(transitions["rallies"], 4)
        self.assertEqual(transitions["additionalRalliesToPilotTarget"], 9)
        self.assertEqual(transitions["recordingsMeetingPilotTarget"], 0)
        self.assertEqual(transitions["fields"]["terminalCue"]["labeled"], 1)
        self.assertEqual(
            transitions["byOutcomeStratum"]["ace"]["fullyLabeled"]["labeled"],
            1,
        )
        tracklets = report["checklist"]["playerTracklets"]
        self.assertEqual(tracklets["tracklets"], 4)
        self.assertEqual(tracklets["usableTracklets"], 4)
        self.assertEqual(tracklets["ralliesWithBothReadyWindows"], 1)
        self.assertEqual(tracklets["additionalRalliesToPilotTarget"], 9)
        self.assertEqual(
            report["recordings"][0]["playerTracklets"]["readyRallyWindows"],
            2,
        )

    def test_invalid_documents_are_reported_without_aborting_scan(self) -> None:
        valid = self.write_label("valid", extended=False)
        invalid = self.root / "invalid.labels.json"
        invalid.write_text("not json", encoding="utf-8")

        report = build_label_readiness_report([valid, invalid])

        self.assertEqual(report["checklist"]["validRecordings"], 1)
        self.assertEqual(report["checklist"]["invalidDocuments"], 1)
        self.assertIn("cannot read label document", report["invalidDocuments"][0]["error"])


if __name__ == "__main__":
    unittest.main()
