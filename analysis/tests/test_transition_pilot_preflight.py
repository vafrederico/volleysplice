from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from analysis.annotations import build_manifest_from_labels
from analysis.schema import ManifestError, load_manifest
from analysis.transition_label_gate import build_transition_label_gate, write_transition_gate
from analysis.transition_pilot_preflight import (
    TRANSITION_PREFLIGHT_KIND,
    build_transition_pilot_preflight,
    extract_transition_pilot,
    hard_negative_mask_for_times,
)


class TransitionPilotPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="volleycut-pilot-preflight-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _label(self, recording_id: str, split: str, byte: bytes) -> Path:
        video = self.root / f"{recording_id}.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (32, 32)
        )
        self.assertTrue(writer.isOpened())
        color = int(byte[0])
        for _ in range(45):
            writer.write(np.full((32, 32, 3), color, dtype=np.uint8))
        writer.release()
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        rallies = []
        for index in range(5):
            start = 2.0 + 8.0 * index
            rallies.append(
                {
                    "start": start,
                    "end": start + 4.0,
                    "tags": [],
                    "receiverReactionTime": start + 0.2,
                    "collectiveStandDownTime": start + 4.1,
                    "terminalCue": "no-recovery",
                    "endObservability": "observable",
                    "startConfidence": 0.9,
                    "endConfidence": 0.8,
                    "verifiedImmediateResult": index == 0,
                }
            )
        payload = {
            "schemaVersion": 1,
            "kind": "volleycut-rally-labels",
            "createdAt": "2026-08-12T00:00:00Z",
            "recording": {
                "id": recording_id,
                "video": str(video),
                "videoFilename": video.name,
                "contentSha256": digest,
                "durationSeconds": 45.0,
                "sourceGroup": recording_id,
                "split": split,
                "environment": "grass",
                "game": {},
                "capture": {},
                "roi": None,
            },
            "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
            "annotation": {
                "status": "complete",
                "annotator": "reviewer",
                "continuousVideoReviewed": True,
                "reviewedAt": "2026-08-12T00:00:00Z",
                "notes": "",
            },
            "rallies": rallies,
            "ignoredIntervals": [],
            "hardNegatives": [],
            "sideSwitches": [],
        }
        path = self.root / f"{recording_id}.labels.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _ready_fixture(self) -> dict[str, Path]:
        sources = [
            self._label("train", "train", b"train-video"),
            self._label("validation", "validation", b"validation-video"),
            self._label("protected", "test", b"protected-video"),
        ]
        baseline = self.root / "baseline.json"
        build_manifest_from_labels(sources, baseline, name="baseline")
        gate_path = self.root / "gate.json"
        write_transition_gate(gate_path, build_transition_label_gate(sources))
        snapshot_dir = self.root / "dev-snapshot"
        snapshot_dir.mkdir()
        ledger_rows = []
        snapshots = []
        for source in sources[:2]:
            snapshot = snapshot_dir / source.name
            snapshot.write_bytes(source.read_bytes())
            payload = json.loads(source.read_text())
            ledger_rows.append(
                {
                    "recordingId": payload["recording"]["id"],
                    "sourceDraft": str(source),
                    "sourceDraftSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "snapshotFile": snapshot.name,
                    "snapshotSha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    "video": payload["recording"]["video"],
                    "videoContentSha256": payload["recording"]["contentSha256"],
                    "transformations": [],
                }
            )
            snapshots.append(snapshot)
        ledger = snapshot_dir / "snapshot.json"
        ledger.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "volleycut-completed-label-snapshot",
                    "recordings": ledger_rows,
                }
            ),
            encoding="utf-8",
        )
        manifest = self.root / "pilot.json"
        build_manifest_from_labels(snapshots, manifest, name="pilot")
        return {
            "baseline": baseline,
            "gate": gate_path,
            "ledger": ledger,
            "manifest": manifest,
            "protected": sources[2],
            "train": sources[0],
        }

    def test_ready_lineage_and_pure_target_extraction(self) -> None:
        paths = self._ready_fixture()
        paths["protected"].unlink()  # Development preflight must never open it.

        report = build_transition_pilot_preflight(
            baseline_manifest_path=paths["baseline"],
            manifest_path=paths["manifest"],
            snapshot_ledger_path=paths["ledger"],
            gate_path=paths["gate"],
            candidate="reaction-supervised-serve-edge",
        )

        self.assertEqual(report["kind"], TRANSITION_PREFLIGHT_KIND)
        self.assertFalse(report["featureRowsPrepared"])
        self.assertFalse(report["protectedSplitsPrepared"])
        self.assertEqual(report["targetCounts"]["reaction"], 10)
        extracted = extract_transition_pilot(load_manifest(paths["manifest"]).recordings[0])
        self.assertIn(False, [row.value for row in extracted.immediate_results])

    def test_mutated_draft_or_snapshot_is_rejected(self) -> None:
        paths = self._ready_fixture()
        paths["train"].write_text(paths["train"].read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "draft changed"):
            build_transition_pilot_preflight(
                baseline_manifest_path=paths["baseline"],
                manifest_path=paths["manifest"],
                snapshot_ledger_path=paths["ledger"],
                gate_path=paths["gate"],
                candidate="reaction-supervised-serve-edge",
            )

    def test_unready_gate_fails_before_manifest_loading(self) -> None:
        source = self._label("train", "train", b"train")
        payload = json.loads(source.read_text())
        for rally in payload["rallies"]:
            rally.pop("receiverReactionTime")
        source.write_text(json.dumps(payload), encoding="utf-8")
        baseline = self.root / "baseline.json"
        build_manifest_from_labels([source], baseline, name="baseline")
        gate = self.root / "gate.json"
        write_transition_gate(gate, build_transition_label_gate([source]))
        with patch("analysis.transition_pilot_preflight.load_manifest") as loader:
            with self.assertRaisesRegex(RuntimeError, "waiting"):
                build_transition_pilot_preflight(
                    baseline_manifest_path=baseline,
                    manifest_path=self.root / "poison.json",
                    snapshot_ledger_path=self.root / "poison-ledger.json",
                    gate_path=gate,
                    candidate="reaction-supervised-serve-edge",
                )
            loader.assert_not_called()

    def test_hard_negative_mask_is_half_open(self) -> None:
        from analysis.transition_pilot_preflight import HardNegativeTarget

        mask = hard_negative_mask_for_times(
            np.asarray([0.9, 1.0, 1.5, 2.0]),
            (HardNegativeTarget(1.0, 2.0, "random-dead-control"),),
        )
        np.testing.assert_array_equal(mask, [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
