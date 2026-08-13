from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from analysis.schema import DatasetManifest, Interval, Recording
from analysis.unsloth_dataset import (
    UnslothDatasetError,
    Window,
    aligned_windows,
    build_row,
    build_rows,
    export_dataset,
    intervals_overlap_window,
    labels_for_window,
)


def recording(*, split: str = "train") -> Recording:
    return Recording(
        id="match-1",
        video=Path("/videos/match-1.mp4"),
        split=split,
        source_group=f"group-{split}",
        environment="indoor",
        game={"playersPerTeam": 6},
        rallies=(
            Interval(22.0, 28.0, ("ace", "ai-prelabel")),
            Interval(50.0, 60.0, ("service-fault",)),
        ),
        ignored_intervals=(),
        roi=None,
        capture={},
        consent={"analyze": True, "train": True},
        content_sha256="a" * 64,
        raw={
            "hardNegatives": [{"start": 30.0, "end": 34.0, "category": "timeout"}]
        },
    )


class UnslothDatasetTests(unittest.TestCase):
    def test_aligned_windows_are_exhaustive_and_not_gold_centered(self) -> None:
        self.assertEqual(
            aligned_windows(70.0),
            (Window(0.0, 32.0), Window(24.0, 56.0), Window(48.0, 70.0)),
        )
        self.assertEqual(
            aligned_windows(48.1),
            (Window(0.0, 32.0), Window(24.0, 48.1)),
        )
        with self.assertRaisesRegex(UnslothDatasetError, "stride"):
            aligned_windows(10.0, window_seconds=4.0, stride_seconds=5.0)

    def test_window_labels_preserve_censoring_and_outcomes(self) -> None:
        labels = labels_for_window(recording(), Window(24.0, 56.0))
        self.assertEqual(
            labels,
            {
                "liveAtStart": True,
                "liveAtEnd": True,
                "rallies": [
                    {"start": 0.0, "end": 4.0, "outcome": "ace"},
                    {"start": 26.0, "end": 32.0, "outcome": "service-fault"},
                ],
            },
        )

    def test_row_is_directly_shaped_for_unsloth_video_collator(self) -> None:
        row = build_row(recording(), Window(24.0, 56.0), sample_fps=1.0)
        video = row["messages"][0]["content"][0]
        self.assertEqual(
            video,
            {
                "type": "video",
                "video": "/videos/match-1.mp4",
                "video_start": 24.0,
                "video_end": 56.0,
                "fps": 1.0,
                "min_frames": 4,
                "max_frames": 32,
            },
        )
        target = json.loads(row["messages"][1]["content"][0]["text"])
        self.assertTrue(target["liveAtStart"])
        self.assertEqual(row["metadata"]["hardNegatives"][0]["category"], "timeout")
        self.assertEqual(row["metadata"]["sourceRallies"][0]["tags"], ["ace", "ai-prelabel"])

    def test_half_open_overlap_does_not_include_touching_interval(self) -> None:
        self.assertFalse(intervals_overlap_window((Interval(2.0, 4.0),), Window(4.0, 8.0)))
        self.assertTrue(intervals_overlap_window((Interval(3.999, 5.0),), Window(4.0, 8.0)))

    def test_export_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-unsloth-test-") as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            manifest = DatasetManifest(
                path=manifest_path,
                name="test",
                recordings=(recording(),),
                raw={},
            )
            output = root / "already-there"
            output.mkdir()
            with self.assertRaisesRegex(UnslothDatasetError, "overwrite"):
                export_dataset(manifest, output)

    def test_build_rows_can_exclude_an_environment_without_changing_labels(self) -> None:
        beach = replace(
            recording(),
            id="beach-1",
            video=Path("/videos/beach-1.mp4"),
            environment="beach",
            source_group="beach-group",
        )
        manifest = DatasetManifest(
            path=Path("/manifest.json"),
            name="test",
            recordings=(recording(), beach),
            raw={},
        )
        with patch("analysis.unsloth_dataset.probe", return_value={"duration": 32.0}):
            rows = build_rows(manifest, excluded_environments=[" BEACH "])
        self.assertEqual(
            [row["metadata"]["recordingId"] for row in rows["train"]],
            ["match-1"],
        )

    def test_rejects_unknown_excluded_environment(self) -> None:
        manifest = DatasetManifest(
            path=Path("/manifest.json"),
            name="test",
            recordings=(recording(),),
            raw={},
        )
        with self.assertRaisesRegex(UnslothDatasetError, "unknown excluded"):
            build_rows(manifest, excluded_environments=["court"])

    def test_build_rows_can_rebase_video_paths_for_windows(self) -> None:
        source_root = Path("/nas/volleycut")
        source = replace(
            recording(),
            video=source_root / "labeling/proxies/indoor/match-1.mp4",
        )
        manifest = DatasetManifest(
            path=source_root / "labeling/manifests/full.json",
            name="test",
            recordings=(source,),
            raw={},
        )
        with patch("analysis.unsloth_dataset.probe", return_value={"duration": 32.0}):
            rows = build_rows(
                manifest,
                source_data_root=source_root,
                consumer_data_root="Z:/volleycut",
            )
        row = rows["train"][0]
        expected = "Z:/volleycut/labeling/proxies/indoor/match-1.mp4"
        self.assertEqual(row["messages"][0]["content"][0]["video"], expected)
        self.assertEqual(row["metadata"]["sourceVideo"]["path"], expected)

    def test_rebase_requires_both_absolute_roots(self) -> None:
        manifest = DatasetManifest(
            path=Path("/manifest.json"),
            name="test",
            recordings=(recording(),),
            raw={},
        )
        with self.assertRaisesRegex(UnslothDatasetError, "supplied together"):
            build_rows(manifest, consumer_data_root="Z:/volleycut")
        with self.assertRaisesRegex(UnslothDatasetError, "absolute"):
            with patch("analysis.unsloth_dataset.probe", return_value={"duration": 32.0}):
                build_rows(
                    manifest,
                    source_data_root=Path("/"),
                    consumer_data_root="relative/path",
                )


if __name__ == "__main__":
    unittest.main()
