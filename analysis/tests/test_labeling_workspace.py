from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.labeling_workspace import prepare_labeling_workspace
from analysis.schema import ManifestError


class LabelingWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="volleycut-label-workspace-")
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.video = self.root / "source.mkv"
        self.video.write_bytes(b"source")
        self.plan = self.root / "plan.json"
        self.plan.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "recordings": [
                        {
                            "id": "match-1",
                            "video": self.video.name,
                            "sourceGroup": "source-1",
                            "split": "train",
                            "environment": "grass",
                            "game": {"playersPerTeam": 2, "targetPoints": None},
                            "capture": {},
                            "roi": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def fake_normalize(_source: Path, output: Path, **_kwargs: object) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"proxy")
        output.with_suffix(output.suffix + ".provenance.json").write_text("{}", encoding="utf-8")

    @staticmethod
    def fake_task(_video: Path, output: Path, **_kwargs: object) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")

    def test_batch_preparation_is_resumable(self) -> None:
        workspace = self.root / "workspace"
        with (
            patch("analysis.labeling_workspace.normalize_video", side_effect=self.fake_normalize) as normalize,
            patch("analysis.labeling_workspace.create_label_draft", side_effect=self.fake_task) as task,
        ):
            first = prepare_labeling_workspace(self.plan, workspace)
            second = prepare_labeling_workspace(self.plan, workspace)

        self.assertEqual(first["proxiesCreated"], 1)
        self.assertEqual(first["tasksCreated"], 1)
        self.assertEqual(second["proxiesReused"], 1)
        self.assertEqual(second["tasksReused"], 1)
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual(task.call_count, 1)

    def test_does_not_duplicate_full_suffix_from_plan_id(self) -> None:
        payload = json.loads(self.plan.read_text(encoding="utf-8"))
        payload["recordings"][0]["id"] = "match-1-full"
        self.plan.write_text(json.dumps(payload), encoding="utf-8")
        workspace = self.root / "workspace"

        with (
            patch("analysis.labeling_workspace.normalize_video", side_effect=self.fake_normalize),
            patch("analysis.labeling_workspace.create_label_draft", side_effect=self.fake_task),
        ):
            result = prepare_labeling_workspace(self.plan, workspace)

        self.assertEqual(
            result["items"][0]["proxy"],
            str(workspace / "proxies" / "grass" / "match-1-full.mp4"),
        )

    def test_rejects_source_group_split_leakage_before_processing_second_row(self) -> None:
        payload = json.loads(self.plan.read_text(encoding="utf-8"))
        second = dict(payload["recordings"][0])
        second.update({"id": "match-2", "split": "test"})
        payload["recordings"].append(second)
        self.plan.write_text(json.dumps(payload), encoding="utf-8")

        with (
            patch("analysis.labeling_workspace.normalize_video", side_effect=self.fake_normalize),
            patch("analysis.labeling_workspace.create_label_draft", side_effect=self.fake_task),
        ):
            with self.assertRaisesRegex(ManifestError, "sourceGroup.*crosses"):
                prepare_labeling_workspace(self.plan, self.root / "workspace")

    def test_rejects_a_missing_later_source_before_any_transcode(self) -> None:
        payload = json.loads(self.plan.read_text(encoding="utf-8"))
        second = dict(payload["recordings"][0])
        second.update({"id": "match-2", "video": "missing.mkv"})
        payload["recordings"].append(second)
        self.plan.write_text(json.dumps(payload), encoding="utf-8")

        with (
            patch("analysis.labeling_workspace.normalize_video", side_effect=self.fake_normalize) as normalize,
            patch("analysis.labeling_workspace.create_label_draft", side_effect=self.fake_task),
        ):
            with self.assertRaisesRegex(ManifestError, r"recordings\[1\].*does not exist"):
                prepare_labeling_workspace(self.plan, self.root / "workspace")

        normalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
