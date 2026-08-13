from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.annotations import (
    build_manifest_from_labels,
    create_label_draft,
    freeze_label_snapshot,
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

    def test_completed_document_rejects_touching_rallies(self) -> None:
        payload = self.complete_payload()
        payload["rallies"][1]["start"] = payload["rallies"][0]["end"]
        self.labels.write_text(json.dumps(payload), encoding="utf-8")

        draft = load_label_document(
            self.labels,
            require_complete=False,
            require_video=False,
        )
        self.assertEqual(len(draft.rallies), 2)
        with self.assertRaisesRegex(ManifestError, "positive dead time"):
            load_label_document(self.labels, require_video=False)

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

    def test_extended_geometry_transition_and_negative_labels_are_preserved(self) -> None:
        payload = self.complete_payload()
        payload["recording"]["courtGeometry"] = {
            "corners": {
                "nearLeft": {"x": 0.1, "y": 0.9},
                "nearRight": {"x": 0.9, "y": 0.9},
                "farLeft": {"x": 0.35, "y": 0.2},
                "farRight": {"x": 0.65, "y": 0.2},
            },
            "netAnchors": {
                "left": {"x": 0.25, "y": 0.55},
                "right": {"x": 0.75, "y": 0.55},
            },
            "serviceZoneAnchors": {
                "near": {"x": 0.5, "y": 0.95},
                "far": {"x": 0.5, "y": 0.12},
            },
        }
        payload["rallies"][0].update(
            {
                "receiverReactionTime": 5.4,
                "collectiveStandDownTime": 12.4,
                "terminalCue": "ball-down-or-out",
                "endObservability": "observable",
                "startConfidence": 0.9,
                "endConfidence": 0.8,
                "verifiedImmediateResult": False,
                "playerTracklets": [
                    {
                        "trackId": "P1",
                        "window": "serve",
                        "team": "team-a",
                        "courtSide": "near",
                        "observations": [
                            {
                                "time": 4.8,
                                "footpoint": {"x": 0.4, "y": 0.85},
                                "state": "ready",
                            },
                            {
                                "time": 5.2,
                                "box": {
                                    "x": 0.35,
                                    "y": 0.45,
                                    "width": 0.1,
                                    "height": 0.4,
                                },
                                "state": "playing",
                            },
                        ],
                    }
                ],
            }
        )
        payload["hardNegatives"] = [
            {"start": 35.0, "end": 38.0, "category": "walking-ball-retrieval"},
            {"start": 40.0, "end": 42.0, "category": "celebration-huddle"},
        ]
        self.labels.write_text(json.dumps(payload), encoding="utf-8")

        document = load_label_document(self.labels, require_video=False)
        self.assertEqual(len(document.court_geometry["corners"]), 4)
        self.assertEqual(document.rally_transitions[0].terminal_cue, "ball-down-or-out")
        self.assertFalse(document.rally_transitions[0].verified_immediate_result)
        self.assertEqual(document.player_tracklets[0].track_id, "P1")
        self.assertEqual(document.player_tracklets[0].rally_index, 0)
        self.assertEqual(document.player_tracklets[0].observations[1].state, "playing")

        manifest_path = self.root / "extended.json"
        manifest = build_manifest_from_labels(
            [self.labels], manifest_path, name="extended", require_videos=False
        )
        row = manifest["recordings"][0]
        self.assertEqual(row["courtGeometry"], payload["recording"]["courtGeometry"])
        self.assertEqual(row["rallies"][0]["receiverReactionTime"], 5.4)
        self.assertEqual(row["rallies"][0]["playerTracklets"][0]["trackId"], "P1")
        self.assertEqual(row["hardNegatives"][1]["category"], "celebration-huddle")

    def test_player_tracklets_are_optional_but_strictly_anonymous_and_bounded(self) -> None:
        legacy = self.complete_payload()
        self.labels.write_text(json.dumps(legacy), encoding="utf-8")
        self.assertEqual(
            load_label_document(self.labels, require_video=False).player_tracklets,
            (),
        )

        payload = self.complete_payload()
        payload["rallies"][0]["playerTracklets"] = [
            {
                "trackId": "P1",
                "window": "rally-end",
                "team": "team-b",
                "courtSide": "far",
                "observations": [
                    {
                        "time": 11.7,
                        "footpoint": {"x": 0.6, "y": 0.4},
                        "state": "stand-down",
                    },
                    {
                        "time": 12.3,
                        "footpoint": {"x": 0.61, "y": 0.42},
                    },
                ],
                "playerName": "must-not-be-stored",
            }
        ]
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "unrecognized fields"):
            load_label_document(self.labels, require_video=False)

        del payload["rallies"][0]["playerTracklets"][0]["playerName"]
        payload["rallies"][0]["playerTracklets"][0]["trackId"] = "ALICE"
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "anonymous token"):
            load_label_document(self.labels, require_video=False)

        payload["rallies"][0]["playerTracklets"][0]["trackId"] = "P1"
        payload["rallies"][0]["playerTracklets"][0]["observations"][1]["time"] = 16.0
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "inside its boundary window"):
            load_label_document(self.labels, require_video=False)

        observation = payload["rallies"][0]["playerTracklets"][0]["observations"][1]
        observation["time"] = 12.3
        observation["box"] = {"x": 0.9, "y": 0.3, "width": 0.2, "height": 0.4}
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "positive normalized frame box"):
            load_label_document(self.labels, require_video=False)

    def test_partial_geometry_is_draft_safe_but_not_completion_safe(self) -> None:
        payload = self.complete_payload()
        payload["recording"]["courtGeometry"] = {
            "corners": {"nearLeft": {"x": 0.1, "y": 0.9}}
        }
        self.labels.write_text(json.dumps(payload), encoding="utf-8")

        draft = load_label_document(
            self.labels, require_complete=False, require_video=False
        )
        self.assertEqual(len(draft.court_geometry["corners"]), 1)
        with self.assertRaisesRegex(ManifestError, "courtGeometry.corners"):
            load_label_document(self.labels, require_video=False)

    def test_transition_annotations_are_bounded_and_related_to_rally(self) -> None:
        payload = self.complete_payload()
        payload["rallies"][0]["receiverReactionTime"] = 11.0
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "within five seconds after rally start"):
            load_label_document(self.labels, require_video=False)

        payload["rallies"][0]["receiverReactionTime"] = 5.2
        payload["rallies"][0]["endConfidence"] = 1.1
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "endConfidence must be between"):
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

    def test_reviewed_draft_freezes_without_mutating_source(self) -> None:
        payload = self.complete_payload()
        payload["annotation"]["status"] = "in-progress"
        payload["annotation"]["annotator"] = "inherited-model-source"
        payload["annotation"]["reviewedAt"] = None
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        source_before = self.labels.read_bytes()
        snapshot = self.root / "completed" / "full-v1"

        documents = freeze_label_snapshot(
            [self.labels],
            snapshot,
            annotator="reviewer-1",
            require_videos=False,
        )

        self.assertEqual(self.labels.read_bytes(), source_before)
        self.assertEqual(len(documents), 1)
        frozen = load_label_document(
            snapshot / self.labels.name,
            require_video=False,
        )
        self.assertEqual(frozen.payload["annotation"]["status"], "complete")
        self.assertEqual(frozen.payload["annotation"]["annotator"], "reviewer-1")
        self.assertTrue(frozen.payload["annotation"]["continuousVideoReviewed"])
        self.assertIsInstance(frozen.payload["annotation"]["reviewedAt"], str)
        self.assertEqual(frozen.video, self.video.resolve())
        ledger = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(ledger["kind"], "volleycut-completed-label-snapshot")
        self.assertEqual(ledger["recordings"][0]["recordingId"], "match-1")
        self.assertEqual((snapshot / self.labels.name).stat().st_mode & 0o777, 0o444)
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            freeze_label_snapshot(
                [self.labels],
                snapshot,
                annotator="reviewer-1",
                require_videos=False,
            )

    def test_unreviewed_draft_cannot_be_frozen(self) -> None:
        with self.assertRaisesRegex(ManifestError, "continuousVideoReviewed must be true"):
            freeze_label_snapshot(
                [self.labels],
                self.root / "completed" / "full-v1",
                annotator="reviewer-1",
                require_videos=False,
            )

    def test_freeze_can_audit_and_drop_touching_duplicate_tail(self) -> None:
        payload = self.complete_payload()
        payload["annotation"]["status"] = "in-progress"
        payload["annotation"]["reviewedAt"] = None
        payload["rallies"] = [
            {
                "start": 5.0,
                "end": 12.0,
                "tags": ["ai-prelabel", "serve-confidence:high"],
                "notes": "copied candidate note",
            },
            {
                "start": 12.0,
                "end": 12.4,
                "tags": ["ai-prelabel", "serve-confidence:high"],
                "notes": "copied candidate note",
            },
            {"start": 22.0, "end": 30.0, "tags": []},
        ]
        self.labels.write_text(json.dumps(payload), encoding="utf-8")
        snapshot = self.root / "completed" / "full-v1"

        documents = freeze_label_snapshot(
            [self.labels],
            snapshot,
            annotator="reviewer-1",
            drop_touching_duplicate_tails=True,
            require_videos=False,
        )

        self.assertEqual(len(documents[0].rallies), 2)
        ledger = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
        transformations = ledger["recordings"][0]["transformations"]
        self.assertEqual(len(transformations), 1)
        self.assertEqual(transformations[0]["sourceRallyIndex"], 1)
        self.assertEqual(transformations[0]["removed"]["end"], 12.4)


if __name__ == "__main__":
    unittest.main()
