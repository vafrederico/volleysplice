from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from analysis.ball_annotation import (
    ANNOTATION_POLICY,
    BALL_ANNOTATION_SCHEMA_VERSION,
    BALL_ANNOTATION_TASK_TYPE,
    BALL_SAMPLING_POLICY_ID,
    FRAMES_PER_WINDOW,
    SAMPLE_FPS,
    STRATA,
    BallAnnotationError,
    _immutable_digest,
    _recording_artifact_paths,
    assert_immutable_provenance_unchanged,
    select_sampling_windows,
    validate_ball_annotation_task,
)
from analysis.features import VideoMetadata
from analysis.schema import Interval, Recording


MANIFEST_DIGEST = hashlib.sha256(b"manifest").hexdigest()


def recording(*, timeout: bool = True, rallies: tuple[Interval, ...] | None = None) -> Recording:
    hard_negatives = (
        [{"start": 95.0, "end": 110.0, "category": "timeout"}] if timeout else []
    )
    return Recording(
        id="development-recording",
        video=Path("/tmp/development-recording.mp4"),
        split="train",
        source_group="source-a",
        environment="indoor",
        game={},
        rallies=(
            rallies
            if rallies is not None
            else (
                Interval(10.0, 19.0),
                Interval(30.0, 41.0),
                Interval(55.0, 67.0),
                Interval(75.0, 84.0),
            )
        ),
        ignored_intervals=(),
        roi=None,
        capture={},
        consent={"analyze": True, "train": True},
        content_sha256="a" * 64,
        raw={"hardNegatives": hard_negatives},
    )


METADATA = VideoMetadata(
    duration=120.0,
    width=960,
    height=540,
    fps=30.0,
    frame_count=3600,
    has_audio=True,
)


def task_fixture() -> dict[str, object]:
    windows = []
    frames = []
    annotations = {}
    for window_index, stratum in enumerate(STRATA):
        start_sample = window_index * 90
        window_id = f"{window_index + 1:02d}-{stratum}"
        windows.append(
            {
                "id": window_id,
                "requestedStratum": stratum,
                "actualSource": "fixture",
                "startSampleIndex": start_sample,
                "startSeconds": start_sample / SAMPLE_FPS,
                "endSeconds": (start_sample + FRAMES_PER_WINDOW) / SAMPLE_FPS,
                "centerSeconds": (start_sample + FRAMES_PER_WINDOW / 2) / SAMPLE_FPS,
                "reference": {},
            }
        )
        for offset in range(FRAMES_PER_WINDOW):
            source_index = (start_sample + offset) * 2
            frame_id = f"f{source_index:09d}"
            frames.append(
                {
                    "id": frame_id,
                    "windowId": window_id,
                    "sampleOffset": offset,
                    "sourceFrameIndex": source_index,
                    "sourceTimestampSeconds": source_index / 30,
                    "image": {
                        "path": f"../images/{frame_id}.png",
                        "sha256": "b" * 64,
                        "width": 960,
                        "height": 540,
                        "format": "png",
                    },
                }
            )
            annotations[frame_id] = {
                "status": "unreviewed",
                "primaryBallState": None,
                "objects": [],
                "notes": "",
            }
    immutable = {
        "manifest": {
            "name": "fixture",
            "filename": "fixture.json",
            "pathHint": "/tmp/fixture.json",
            "sha256": MANIFEST_DIGEST,
        },
        "recording": {
            "id": "development-recording",
            "split": "train",
            "sourceGroup": "source-a",
            "environment": "indoor",
        },
        "source": {
            "proxy": {
                "filename": "proxy.mp4",
                "pathHint": "/tmp/proxy.mp4",
                "sizeBytes": 123,
                "sha256": "a" * 64,
                "width": 960,
                "height": 540,
                "fps": 30.0,
                "frameCount": 3600,
                "durationSeconds": 120.0,
            },
            "normalizationProvenance": None,
        },
        "sampling": {
            "policyId": BALL_SAMPLING_POLICY_ID,
            "round": 1,
            "sampleFps": SAMPLE_FPS,
            "windowSeconds": 3,
            "framesPerWindow": FRAMES_PER_WINDOW,
            "minimumCenterSeparationSeconds": 5,
            "seedMaterial": "manifest SHA-256 + recording id + round",
            "frameRule": "fixture",
        },
        "annotationPolicy": copy.deepcopy(ANNOTATION_POLICY),
        "windows": windows,
        "frames": frames,
    }
    digest = _immutable_digest(immutable)
    immutable["taskId"] = f"ball-presence-{digest[:24]}"
    immutable["digestSha256"] = digest
    return {
        "schemaVersion": BALL_ANNOTATION_SCHEMA_VERSION,
        "taskType": BALL_ANNOTATION_TASK_TYPE,
        "immutable": immutable,
        "suggestions": {"status": "empty", "model": None, "frames": {}},
        "annotations": {
            "review": {
                "status": "unreviewed",
                "annotator": None,
                "reviewedAt": None,
                "notes": "",
            },
            "frames": annotations,
        },
    }


def resign(task: dict[str, object]) -> None:
    immutable = task["immutable"]
    digest = _immutable_digest(immutable)
    immutable["taskId"] = f"ball-presence-{digest[:24]}"
    immutable["digestSha256"] = digest


class BallAnnotationSamplingTests(unittest.TestCase):
    def test_recording_artifact_paths_reject_path_components(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ball-paths-") as directory:
            root = Path(directory)
            task_path, image_root = _recording_artifact_paths(
                root, "beach-safe_ID-01"
            )
            self.assertEqual(task_path.parent, (root / "tasks").resolve())
            self.assertEqual(image_root.parent, (root / "images").resolve())

            for unsafe in ("../escape", "/tmp/escape", "nested/id", "space id", "."):
                with self.subTest(recording_id=unsafe):
                    with self.assertRaisesRegex(BallAnnotationError, "recording id"):
                        _recording_artifact_paths(root, unsafe)

    def test_six_strata_are_deterministic_separated_and_use_distinct_mid_rallies(self) -> None:
        first = select_sampling_windows(
            recording(), METADATA, manifest_sha256=MANIFEST_DIGEST, round_index=1
        )
        second = select_sampling_windows(
            recording(), METADATA, manifest_sha256=MANIFEST_DIGEST, round_index=1
        )

        self.assertEqual(first, second)
        self.assertEqual(tuple(item.requested_stratum for item in first), STRATA)
        self.assertEqual(len(first), 6)
        self.assertEqual(first[-1].actual_source, "timeout_hard_negative")
        midpoint_rallies = {
            first[1].reference["rallyIndex"],
            first[2].reference["rallyIndex"],
        }
        self.assertEqual(len(midpoint_rallies), 2)
        for index, left in enumerate(first):
            for right in first[index + 1 :]:
                self.assertGreaterEqual(abs(left.center_seconds - right.center_seconds), 5)

    def test_uses_highest_scored_dead_window_without_timeout(self) -> None:
        item = recording(timeout=False)
        target_start = 90 * SAMPLE_FPS
        windows = select_sampling_windows(
            item,
            METADATA,
            manifest_sha256=MANIFEST_DIGEST,
            round_index=1,
            motion_scores={0: 0.1, target_start: 0.9},
        )

        hard_negative = windows[-1]
        self.assertEqual(hard_negative.actual_source, "highest_frame_difference_dead")
        self.assertEqual(hard_negative.start_sample_index, target_start)
        self.assertEqual(hard_negative.reference["motionMeanAbsDiff"], 0.9)
        self.assertIn("no eligible timeout", hard_negative.fallback_reason or "")

    def test_named_strata_record_uniform_fallbacks(self) -> None:
        item = recording(timeout=False, rallies=())
        windows = select_sampling_windows(
            item,
            METADATA,
            manifest_sha256=MANIFEST_DIGEST,
            round_index=2,
            motion_scores={0: 0.5},
        )

        self.assertEqual(tuple(window.requested_stratum for window in windows), STRATA)
        self.assertEqual(windows[0].actual_source, "uniform_fallback")
        self.assertIsNotNone(windows[0].fallback_reason)

    def test_rejects_test_split_and_non_multiple_frame_rate(self) -> None:
        test_recording = copy.copy(recording())
        object.__setattr__(test_recording, "split", "test")
        with self.assertRaisesRegex(BallAnnotationError, "development-only"):
            select_sampling_windows(
                test_recording,
                METADATA,
                manifest_sha256=MANIFEST_DIGEST,
                round_index=1,
            )
        bad_metadata = VideoMetadata(120, 960, 540, 29.97, 3596, True)
        with self.assertRaisesRegex(BallAnnotationError, "integer multiple"):
            select_sampling_windows(
                recording(),
                bad_metadata,
                manifest_sha256=MANIFEST_DIGEST,
                round_index=1,
            )


class BallAnnotationSchemaTests(unittest.TestCase):
    def test_validates_unreviewed_task(self) -> None:
        task = task_fixture()
        validated = validate_ball_annotation_task(task)
        self.assertEqual(validated["immutable"]["taskId"], task["immutable"]["taskId"])

    def test_detects_any_immutable_provenance_change(self) -> None:
        original = task_fixture()
        changed = copy.deepcopy(original)
        changed["immutable"]["frames"][0]["sourceFrameIndex"] += 1

        with self.assertRaisesRegex(BallAnnotationError, "digest"):
            validate_ball_annotation_task(changed)
        with self.assertRaises(BallAnnotationError):
            assert_immutable_provenance_unchanged(original, changed)

    def test_semantic_checks_reject_resigned_derived_provenance(self) -> None:
        mutations = (
            lambda task: task["immutable"]["windows"][0].__setitem__("centerSeconds", 9.0),
            lambda task: task["immutable"]["windows"][0].__setitem__("startSampleIndex", -1),
            lambda task: task["immutable"]["frames"][0].__setitem__("sourceFrameIndex", 2),
            lambda task: task["immutable"]["frames"][0].__setitem__(
                "sourceTimestampSeconds", 1.0
            ),
            lambda task: task["immutable"]["frames"][0]["image"].__setitem__("width", 959),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                task = task_fixture()
                mutate(task)
                resign(task)
                with self.assertRaises(BallAnnotationError):
                    validate_ball_annotation_task(task)

    def test_semantic_checks_reject_resigned_annotation_policy(self) -> None:
        task = task_fixture()
        task["immutable"]["annotationPolicy"]["negativeRule"] = "logos may be balls"
        resign(task)

        with self.assertRaisesRegex(BallAnnotationError, "annotationPolicy"):
            validate_ball_annotation_task(task)

    def test_rejects_resigned_protected_or_unscoped_recording(self) -> None:
        for key, value in (("split", "test"), ("id", ""), ("sourceGroup", "")):
            with self.subTest(key=key):
                task = task_fixture()
                task["immutable"]["recording"][key] = value
                resign(task)
                with self.assertRaises(BallAnnotationError):
                    validate_ball_annotation_task(task)

    def test_annotations_and_detector_suggestions_remain_separate(self) -> None:
        task = task_fixture()
        frame_id = next(iter(task["annotations"]["frames"]))
        task["suggestions"] = {
            "status": "partial",
            "model": {"id": "detector-v0", "sha256": "c" * 64},
            "frames": {
                frame_id: {
                    "ballPresenceProbability": 0.75,
                    "detections": [
                        {
                            "confidence": 0.75,
                            "bbox": {"x": 0.4, "y": 0.2, "width": 0.01, "height": 0.02},
                        }
                    ],
                }
            },
        }
        validate_ball_annotation_task(task)
        self.assertIsNone(task["annotations"]["frames"][frame_id]["primaryBallState"])

        invalid = copy.deepcopy(task)
        invalid["annotations"]["frames"][frame_id]["primaryBallState"] = "localizable"
        with self.assertRaisesRegex(BallAnnotationError, "before review"):
            validate_ball_annotation_task(invalid)

    def test_primary_ball_state_requires_one_primary_object_but_allows_other_court(self) -> None:
        task = task_fixture()
        frame_ids = iter(task["annotations"]["frames"])
        localizable_id = next(frame_ids)
        out_of_frame_id = next(frame_ids)
        task["annotations"]["frames"][localizable_id] = {
            "status": "reviewed",
            "primaryBallState": "localizable",
            "objects": [
                {
                    "id": "primary-track-1",
                    "category": "volleyball",
                    "role": "primary-court",
                    "bbox": {"x": 0.4, "y": 0.2, "width": 0.01, "height": 0.02},
                    "visibility": "motion-blurred",
                    "truncated": False,
                }
            ],
            "notes": "",
        }
        task["annotations"]["frames"][out_of_frame_id] = {
            "status": "reviewed",
            "primaryBallState": "out_of_frame",
            "objects": [
                {
                    "id": "adjacent-track-1",
                    "category": "volleyball",
                    "role": "other-court",
                    "bbox": {"x": 0.8, "y": 0.1, "width": 0.02, "height": 0.03},
                    "visibility": "clear",
                    "truncated": False,
                }
            ],
            "notes": "",
        }
        task["annotations"]["review"] = {
            "status": "in_progress",
            "annotator": None,
            "reviewedAt": None,
            "notes": "",
        }
        validate_ball_annotation_task(task)

        invalid = copy.deepcopy(task)
        invalid["annotations"]["frames"][localizable_id]["objects"] = []
        with self.assertRaisesRegex(BallAnnotationError, "exactly one primary-court"):
            validate_ball_annotation_task(invalid)

    def test_review_status_must_match_frame_progress_and_metadata(self) -> None:
        task = task_fixture()
        frame_id = next(iter(task["annotations"]["frames"]))
        task["annotations"]["frames"][frame_id] = {
            "status": "reviewed",
            "primaryBallState": "out_of_frame",
            "objects": [],
            "notes": "",
        }
        with self.assertRaisesRegex(BallAnnotationError, "unreviewed task"):
            validate_ball_annotation_task(task)

        task["annotations"]["review"] = {
            "status": "in_progress",
            "annotator": "reviewer",
            "reviewedAt": None,
            "notes": "",
        }
        validate_ball_annotation_task(task)

        task["annotations"]["review"]["status"] = "complete"
        task["annotations"]["review"]["reviewedAt"] = "2026-08-11T12:00:00+00:00"
        with self.assertRaisesRegex(BallAnnotationError, "every frame"):
            validate_ball_annotation_task(task)

    def test_optional_image_verification_is_sha_pinned_and_confined(self) -> None:
        task = task_fixture()
        with tempfile.TemporaryDirectory(prefix="ball-task-") as directory:
            root = Path(directory)
            task_path = root / "tasks" / "task.json"
            task_path.parent.mkdir()
            first = task["immutable"]["frames"][0]
            image = root / "images" / "image.png"
            image.parent.mkdir()
            image.write_bytes(b"png")
            first["image"]["path"] = "../images/image.png"
            first["image"]["sha256"] = hashlib.sha256(b"png").hexdigest()
            digest = _immutable_digest(task["immutable"])
            task["immutable"]["taskId"] = f"ball-presence-{digest[:24]}"
            task["immutable"]["digestSha256"] = digest

            with self.assertRaisesRegex(BallAnnotationError, "missing or has changed"):
                validate_ball_annotation_task(task, task_path=task_path, verify_images=True)


if __name__ == "__main__":
    unittest.main()
