from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.transition_label_gate import (
    TRANSITION_GATE_KIND,
    build_transition_label_gate,
    require_transition_development_ready,
    write_transition_gate,
)


TRANSITION_VALUES = {
    "receiverReactionTime": 5.2,
    "collectiveStandDownTime": 10.1,
    "terminalCue": "no-recovery",
    "endObservability": "observable",
    "startConfidence": 0.9,
    "endConfidence": 0.8,
    "verifiedImmediateResult": False,
}


class TransitionLabelGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="volleycut-transition-gate-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_label(
        self,
        recording_id: str,
        *,
        split: str,
        cued: int,
        hard_negatives: int = 0,
        source_group: str | None = None,
    ) -> Path:
        rallies = []
        for index in range(6):
            start = 5.0 + 12.0 * index
            rally = {"start": start, "end": start + 5.0, "tags": []}
            if index < cued:
                rally.update(
                    {
                        **TRANSITION_VALUES,
                        "receiverReactionTime": start + 0.2,
                        "collectiveStandDownTime": start + 5.1,
                    }
                )
            rallies.append(rally)
        categories = (
            "walking-ball-retrieval",
            "celebration-huddle",
            "model-false-positive",
        )
        payload = {
            "schemaVersion": 1,
            "kind": "volleycut-rally-labels",
            "createdAt": "2026-08-12T00:00:00Z",
            "recording": {
                "id": recording_id,
                "video": "missing.mp4",
                "videoFilename": "missing.mp4",
                "contentSha256": "0" * 64,
                "durationSeconds": 90.0,
                "sourceGroup": source_group or recording_id,
                "split": split,
                "environment": "grass",
                "game": {"playersPerTeam": 2, "targetPoints": 21, "format": "2v2"},
                "capture": {},
                "roi": None,
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
            "hardNegatives": [
                {
                    "start": 78.0 + index * 2.0,
                    "end": 79.0 + index * 2.0,
                    "category": categories[index],
                }
                for index in range(hard_negatives)
            ],
            "sideSwitches": [],
        }
        path = self.root / f"{recording_id}-{split}.labels.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_gate_separates_development_from_protected_debt(self) -> None:
        train = self.write_label("train", split="train", cued=5, hard_negatives=3)
        validation = self.write_label("validation", split="validation", cued=3)
        protected = self.write_label("protected", split="test", cued=0)

        report = build_transition_label_gate([train, validation, protected])

        self.assertEqual(report["kind"], TRANSITION_GATE_KIND)
        self.assertTrue(report["readOnly"])
        self.assertEqual(report["development"]["recordings"], 2)
        self.assertEqual(report["development"]["fullyCuedRallies"], 8)
        self.assertEqual(report["development"]["pilotTargetRallies"], 10)
        self.assertEqual(report["development"]["additionalRalliesToPilotTarget"], 2)
        self.assertEqual(report["development"]["additionalHardNegativesToThree"], 3)
        self.assertFalse(report["development"]["allRecordingsMeetHardNegativeMinimum"])
        self.assertEqual(report["protected"]["recordings"], 1)
        self.assertEqual(report["protected"]["additionalRalliesToPilotTarget"], 5)
        self.assertEqual(report["protected"]["additionalHardNegativesToThree"], 3)
        self.assertTrue(report["protected"]["sealedForDevelopment"])
        self.assertFalse(report["developmentExperimentReady"])
        self.assertFalse(report["developmentHardNegativeExperimentReady"])
        self.assertFalse(report["allRegisteredDevelopmentExperimentsReady"])
        with self.assertRaisesRegex(RuntimeError, "additional fully cued rallies: 2"):
            require_transition_development_ready(report)

    def test_ready_gate_requires_only_development_pilot_labels(self) -> None:
        train = self.write_label("train", split="train", cued=5, hard_negatives=3)
        validation = self.write_label(
            "validation", split="validation", cued=5, hard_negatives=3
        )
        protected = self.write_label("protected", split="test", cued=0)

        report = build_transition_label_gate([train, validation, protected])

        self.assertTrue(report["developmentExperimentReady"])
        self.assertTrue(report["developmentHardNegativeExperimentReady"])
        self.assertTrue(report["allRegisteredDevelopmentExperimentsReady"])
        require_transition_development_ready(report)

    def test_invalid_and_duplicate_documents_prevent_readiness(self) -> None:
        first = self.write_label(
            "duplicate", split="train", cued=5, hard_negatives=3, source_group="a"
        )
        second = self.write_label(
            "duplicate",
            split="validation",
            cued=5,
            hard_negatives=3,
            source_group="b",
        )
        invalid = self.root / "broken.labels.json"
        invalid.write_text("not json", encoding="utf-8")

        report = build_transition_label_gate([first, second, invalid])

        self.assertEqual(report["duplicateRecordingIds"], ["duplicate"])
        self.assertEqual(len(report["invalidDocuments"]), 1)
        self.assertFalse(report["developmentExperimentReady"])

    def test_writer_is_atomic_and_never_overwrites(self) -> None:
        source = self.write_label("train", split="train", cued=0)
        report = build_transition_label_gate([source])
        destination = self.root / "gate.json"

        self.assertEqual(write_transition_gate(destination, report), destination.resolve())
        self.assertEqual(json.loads(destination.read_text()), report)
        with self.assertRaises(FileExistsError):
            write_transition_gate(destination, report)


if __name__ == "__main__":
    unittest.main()
