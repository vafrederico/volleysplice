from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.annotations import (
    build_manifest_from_labels,
    create_label_draft,
    load_label_document,
)
from analysis.features import VideoMetadata
from analysis.schema import ManifestError, load_manifest


class AnnotationDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="volleycut-label-test-")
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.video = self.root / "proxy.mp4"
        self.video.write_bytes(b"synthetic-proxy")
        self.labels = self.root / "labels" / "match.labels.json"
        metadata = VideoMetadata(
            duration=60.0,
            width=1280,
            height=720,
            fps=30.0,
            frame_count=1800,
            has_audio=True,
        )
        with patch("analysis.annotations.probe_video", return_value=metadata):
            create_label_draft(
                self.video,
                self.labels,
                recording_id="match-1",
                source_group="source-1",
                split="train",
                environment="grass",
                players_per_team=2,
                target_points=None,
                format_name="grass 2v2",
                roi=(0.0, 0.1, 1.0, 0.9),
                capture={
                    "position": "centered-behind-endline",
                    "stationary": True,
                    "fullCourtVisible": True,
                    "serviceAreasVisible": True,
                },
            )

    def complete_payload(self) -> dict[str, object]:
        payload = json.loads(self.labels.read_text(encoding="utf-8"))
        payload["annotation"] = {
            "status": "complete",
            "annotator": "reviewer-1",
            "continuousVideoReviewed": True,
            "reviewedAt": "2026-08-09T12:00:00Z",
            "notes": "",
        }
        payload["rallies"] = [
            {"start": 5.0, "end": 12.0, "tags": []},
            {"start": 22.0, "end": 30.0, "tags": ["service-error"]},
        ]
        payload["ignoredIntervals"] = [{"start": 0.0, "end": 2.0, "reason": "partial-rally"}]
        payload["hardNegatives"] = [
            {"start": 35.0, "end": 42.0, "category": "foreground-crossing"}
        ]
        payload["sideSwitches"] = [
            {"time": 18.0, "notes": "teams cross after the point"},
            {"time": 48.0},
        ]
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_draft_requires_explicit_completion(self) -> None:
        draft = load_label_document(
            self.labels,
            require_complete=False,
            require_video=False,
        )
        self.assertEqual(draft.recording_id, "match-1")
        with self.assertRaisesRegex(ManifestError, "status must be 'complete'"):
            load_label_document(self.labels, require_video=False)

    def test_completed_document_validates_intervals_and_metadata(self) -> None:
        self.complete_payload()
        document = load_label_document(self.labels, require_video=False)
        self.assertEqual(len(document.rallies), 2)
        self.assertEqual(len(document.ignored_intervals), 1)
        self.assertEqual(len(document.hard_negatives), 1)
        self.assertEqual([marker.time for marker in document.side_switches], [18.0, 48.0])
        self.assertIn("target points are unknown", document.warnings)

    def test_side_switches_are_optional_but_validated_when_present(self) -> None:
        legacy_payload = json.loads(self.labels.read_text(encoding="utf-8"))
        legacy_payload.pop("sideSwitches")
        self.labels.write_text(json.dumps(legacy_payload), encoding="utf-8")
        draft = load_label_document(
            self.labels,
            require_complete=False,
            require_video=False,
        )
        self.assertEqual(draft.side_switches, ())

        payload = self.complete_payload()
        payload["sideSwitches"] = [{"time": 18.0}, {"time": 18.0}]
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "strictly ordered"):
            load_label_document(self.labels, require_video=False)

    def test_rejects_overlap_between_rally_and_ignored_time(self) -> None:
        payload = self.complete_payload()
        payload["ignoredIntervals"] = [{"start": 11.5, "end": 13.0, "reason": "ambiguous"}]
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "must not overlap rallies"):
            load_label_document(self.labels, require_video=False)

    def test_completed_labels_build_a_training_manifest(self) -> None:
        self.complete_payload()
        manifest_path = self.root / "dataset.json"
        payload = build_manifest_from_labels(
            [self.labels],
            manifest_path,
            name="completed-labels",
            require_videos=False,
        )
        self.assertEqual(len(payload["recordings"]), 1)
        manifest = load_manifest(manifest_path, require_videos=False)
        self.assertEqual(len(manifest.recordings[0].rallies), 2)
        self.assertEqual(len(manifest.recordings[0].ignored_intervals), 1)
        self.assertEqual(payload["recordings"][0]["sideSwitches"][0]["time"], 18.0)


if __name__ == "__main__":
    unittest.main()
