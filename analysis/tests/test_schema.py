from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from analysis.schema import Interval, ManifestError, labels_for_times, load_manifest, mask_for_times


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="volleycut-manifest-test-")
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    @staticmethod
    def recording(
        recording_id: str,
        *,
        split: str = "train",
        source_group: str | None = None,
        video: str | None = None,
        rallies: list[dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": recording_id,
            "video": video or f"videos/{recording_id}.mp4",
            "split": split,
            "sourceGroup": source_group or f"source-{recording_id}",
            "environment": "beach",
            "game": {"playersPerTeam": 2, "targetPoints": None, "format": "2v2"},
            "consent": {"analyze": True, "train": split in {"train", "validation"}},
            "capture": {"position": "centered-behind-endline", "stationary": True},
            "rallies": rallies if rallies is not None else [],
        }

    def write_manifest(self, recordings: list[dict[str, Any]]) -> Path:
        path = self.root / "dataset.json"
        path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "name": "unit-test",
                    "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
                    "recordings": recordings,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_touching_ordered_intervals_are_valid(self) -> None:
        path = self.write_manifest(
            [
                self.recording(
                    "match-1",
                    rallies=[
                        {"start": 0.0, "end": 2.5},
                        {"start": 2.5, "end": 7.0},
                    ],
                )
            ]
        )

        manifest = load_manifest(path, require_videos=False)

        self.assertEqual(
            manifest.recordings[0].rallies,
            (Interval(start=0.0, end=2.5), Interval(start=2.5, end=7.0)),
        )
        self.assertEqual(manifest.recordings[0].video, (self.root / "videos/match-1.mp4").resolve())
        self.assertEqual(manifest.recordings[0].game["playersPerTeam"], 2)

    def test_validates_game_metadata_without_guessing_unknown_target(self) -> None:
        row = self.recording("match-1")
        row["game"] = {"playersPerTeam": 4, "targetPoints": None, "format": "reverse 4s"}
        manifest = load_manifest(self.write_manifest([row]), require_videos=False)
        self.assertEqual(manifest.recordings[0].game["targetPoints"], None)

        invalid_values = [0, 7, 2.5, True, "2"]
        for value in invalid_values:
            with self.subTest(players_per_team=value):
                invalid = self.recording("match-1")
                invalid["game"] = {"playersPerTeam": value}
                with self.assertRaisesRegex(ManifestError, "playersPerTeam"):
                    load_manifest(self.write_manifest([invalid]), require_videos=False)

    def test_requires_the_versioned_annotation_policy(self) -> None:
        invalid_policies: list[Any] = [
            None,
            {},
            "serve-contact-to-dead-ball-v1",
            {"id": "serve-contact-to-dead-ball-v0"},
        ]
        for policy in invalid_policies:
            with self.subTest(policy=policy):
                path = self.write_manifest([self.recording("match-1")])
                payload = json.loads(path.read_text(encoding="utf-8"))
                if policy is None:
                    payload.pop("annotationPolicy")
                else:
                    payload["annotationPolicy"] = policy
                path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ManifestError, "annotationPolicy.id"):
                    load_manifest(path, require_videos=False)

    def test_rejects_overlapping_or_non_positive_intervals(self) -> None:
        invalid_rallies = {
            "overlap": [{"start": 1.0, "end": 3.0}, {"start": 2.9, "end": 4.0}],
            "reverse": [{"start": 3.0, "end": 2.0}],
            "zero length": [{"start": 3.0, "end": 3.0}],
            "negative start": [{"start": -0.1, "end": 1.0}],
        }
        for description, rallies in invalid_rallies.items():
            with self.subTest(description=description):
                path = self.write_manifest([self.recording("match-1", rallies=rallies)])
                with self.assertRaises(ManifestError):
                    load_manifest(path, require_videos=False)

    def test_rejects_non_finite_interval_boundaries(self) -> None:
        invalid_rallies = {
            "NaN start": [{"start": math.nan, "end": 2.0}],
            "NaN end": [{"start": 1.0, "end": math.nan}],
            "infinite end": [{"start": 1.0, "end": math.inf}],
        }
        for description, rallies in invalid_rallies.items():
            with self.subTest(description=description):
                path = self.write_manifest([self.recording("match-1", rallies=rallies)])
                with self.assertRaises(ManifestError):
                    load_manifest(path, require_videos=False)

    def test_ignored_intervals_must_not_overlap_rallies(self) -> None:
        row = self.recording("match-1", rallies=[{"start": 2.0, "end": 5.0}])
        row["ignoredIntervals"] = [{"start": 0.0, "end": 2.0}]
        manifest = load_manifest(self.write_manifest([row]), require_videos=False)
        self.assertEqual(manifest.recordings[0].ignored_intervals, (Interval(0.0, 2.0),))

        row["ignoredIntervals"] = [{"start": 1.9, "end": 2.1}]
        with self.assertRaisesRegex(ManifestError, "must not overlap"):
            load_manifest(self.write_manifest([row]), require_videos=False)

    def test_rejects_source_group_crossing_splits(self) -> None:
        path = self.write_manifest(
            [
                self.recording("match-1-set-1", split="train", source_group="match-1"),
                self.recording("match-1-set-2", split="test", source_group="match-1"),
            ]
        )

        with self.assertRaisesRegex(ManifestError, "sourceGroup.*crosses"):
            load_manifest(path, require_videos=False)

    def test_rejects_source_group_whitespace_leakage_bypass(self) -> None:
        path = self.write_manifest(
            [
                self.recording("match-1-set-1", split="train", source_group="match-1"),
                self.recording("match-1-set-2", split="test", source_group=" match-1 "),
            ]
        )

        with self.assertRaises(ManifestError):
            load_manifest(path, require_videos=False)

    def test_rejects_video_reuse_even_with_distinct_ids_and_groups(self) -> None:
        path = self.write_manifest(
            [
                self.recording("clip-a", video="videos/shared.mp4"),
                self.recording("clip-b", video="videos/shared.mp4"),
            ]
        )

        with self.assertRaisesRegex(ManifestError, "video .* is reused"):
            load_manifest(path, require_videos=False)


class LabelTests(unittest.TestCase):
    def test_labels_use_half_open_intervals(self) -> None:
        times = np.asarray([0.999, 1.0, 1.5, 2.0, 2.999, 3.0, 4.0], dtype=np.float64)

        labels = labels_for_times(times, [Interval(1.0, 2.0), Interval(3.0, 4.0)])

        np.testing.assert_array_equal(labels, np.asarray([0, 1, 1, 0, 0, 1, 0], dtype=np.float32))
        self.assertEqual(labels.dtype, np.float32)

    def test_mask_excludes_half_open_ignored_intervals(self) -> None:
        times = np.asarray([0.0, 1.0, 1.999, 2.0, 3.0], dtype=np.float64)
        mask = mask_for_times(times, [Interval(1.0, 2.0)])
        np.testing.assert_array_equal(mask, np.asarray([True, False, False, True, True]))


if __name__ == "__main__":
    unittest.main()
