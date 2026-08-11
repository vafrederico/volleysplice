from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from analysis.ball_annotation import _immutable_digest, validate_ball_annotation_task
from analysis.ball_detector import (
    BallDetection,
    BallDetectorError,
    BallDetectorError,
    FrameDetections,
    _download_verified,
    build_suggestion_index,
    decode_sports_ball_output,
    infer_annotation_task,
    infer_video_sidecar,
    letterbox_rgb,
    sha256_file,
)
from analysis.ball_presence import load_ball_presence_sidecar
from analysis.tests.test_ball_annotation import task_fixture


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.offset = 0

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class _FakeDetector:
    def __init__(self, model_dir: str | Path, **_: object) -> None:
        self.model_dir = Path(model_dir)

    def detect(self, _: np.ndarray) -> FrameDetections:
        return FrameDetections(
            detections=(BallDetection(1.0, 2.0, 3.0, 4.0, 0.75),),
            maximum_raw_score=0.75,
            raw_candidates_above_floor=2,
            inference_milliseconds=12.5,
        )


class BallDetectorTests(unittest.TestCase):
    def test_letterbox_uses_rgb_top_left_without_normalization(self) -> None:
        frame = np.zeros((320, 640, 3), dtype=np.uint8)
        frame[0, 0] = (1, 2, 3)
        blob, ratio = letterbox_rgb(frame)
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob.dtype, np.float32)
        self.assertEqual(ratio, 1.0)
        np.testing.assert_array_equal(blob[0, :, 0, 0], [3.0, 2.0, 1.0])
        np.testing.assert_array_equal(blob[0, :, 500, 500], [114.0] * 3)

    def test_decode_filters_to_sports_ball_and_maps_box(self) -> None:
        output = np.zeros((1, 8400, 85), dtype=np.float32)
        # First stride-8 cell: center=(10, 10)*8 and size=(4, 2)*8.
        output[0, 0, :4] = [10.0, 10.0, np.log(4.0), np.log(2.0)]
        output[0, 0, 4] = 0.9
        output[0, 0, 5 + 32] = 0.8
        # A confident person score must not affect the sports-ball result.
        output[0, 1, 4] = 0.99
        output[0, 1, 5] = 0.99
        detections, maximum, raw_count = decode_sports_ball_output(
            output,
            image_width=320,
            image_height=180,
            letterbox_ratio=2.0,
            score_floor=0.1,
            nms_threshold=0.5,
            maximum_detections=5,
        )
        self.assertEqual(raw_count, 1)
        self.assertAlmostEqual(maximum, 0.72, places=6)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertAlmostEqual(detection.x, 32.0, places=4)
        self.assertAlmostEqual(detection.y, 36.0, places=4)
        self.assertAlmostEqual(detection.width, 16.0, places=4)
        self.assertAlmostEqual(detection.height, 8.0, places=4)

    def test_verified_download_is_atomic_and_rejects_bad_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = b"pinned bytes"
            destination = root / "model.bin"
            _download_verified(
                "https://invalid.example/model",
                destination,
                expected_sha256=hashlib.sha256(content).hexdigest(),
                expected_size=len(content),
                opener=lambda _: _Response(content),
            )
            self.assertEqual(destination.read_bytes(), content)
            bad_destination = root / "bad.bin"
            with self.assertRaises(BallDetectorError):
                _download_verified(
                    "https://invalid.example/model",
                    bad_destination,
                    expected_sha256="0" * 64,
                    opener=lambda _: _Response(content),
                )
            self.assertFalse(bad_destination.exists())

    def test_task_inference_keeps_suggestions_separate_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "frame.png"
            image_path.parent.mkdir()
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            task = task_fixture()
            task["immutable"]["source"]["proxy"]["width"] = 10
            task["immutable"]["source"]["proxy"]["height"] = 8
            for frame in task["immutable"]["frames"]:
                frame["image"] = {
                    "path": "../images/frame.png",
                    "sha256": image_sha,
                    "width": 10,
                    "height": 8,
                    "format": "png",
                }
            digest = _immutable_digest(task["immutable"])
            task["immutable"]["taskId"] = f"ball-presence-{digest[:24]}"
            task["immutable"]["digestSha256"] = digest
            task_path = root / "tasks" / "task.json"
            task_path.parent.mkdir()
            task_path.write_text(json.dumps(task), encoding="utf-8")
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model.json").write_text("{}", encoding="utf-8")
            output = root / "suggestions" / "task.with-suggestions.json"
            with patch("analysis.ball_detector.YoloXBallDetector", _FakeDetector), patch(
                "analysis.ball_detector.sha256_file",
                side_effect=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            ):
                infer_annotation_task(task_path, model_dir, output, limit=1)
            payload = json.loads(output.read_text(encoding="utf-8"))
            validate_ball_annotation_task(payload, task_path=output, verify_images=True)
            self.assertEqual(payload["suggestions"]["status"], "partial")
            self.assertEqual(payload["suggestions"]["summary"]["processedFrames"], 1)
            suggested = next(iter(payload["suggestions"]["frames"].values()))
            self.assertEqual(suggested["detections"][0]["confidence"], 0.75)
            self.assertEqual(
                payload["annotations"]["frames"][next(iter(payload["annotations"]["frames"]))]["status"],
                "unreviewed",
            )
            incompatible = root / "nested" / "suggestions" / "bad.json"
            with patch("analysis.ball_detector.YoloXBallDetector", _FakeDetector), patch(
                "analysis.ball_detector.sha256_file",
                side_effect=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            ):
                with self.assertRaisesRegex(BallDetectorError, "relative image paths"):
                    infer_annotation_task(
                        task_path,
                        model_dir,
                        incompatible,
                        limit=1,
                    )
            self.assertFalse(incompatible.exists())

            complete = root / "complete" / "task.json"
            with patch("analysis.ball_detector.YoloXBallDetector", _FakeDetector), patch(
                "analysis.ball_detector.sha256_file",
                side_effect=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            ):
                infer_annotation_task(task_path, model_dir, complete)
            pilot_index = root / "index.json"
            pilot_index.write_text(
                json.dumps(
                    {
                        "artifactType": "volleycut-ball-presence-pilot-index",
                        "developmentOnly": True,
                        "manifest": {"sha256": "a" * 64},
                        "round": 1,
                        "tasks": [
                            {
                                "recordingId": "development-recording",
                                "task": "tasks/task.json",
                                "taskId": task["immutable"]["taskId"],
                                "initialTaskSha256": hashlib.sha256(
                                    task_path.read_bytes()
                                ).hexdigest(),
                                "frameCount": 270,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            suggestion_index = build_suggestion_index(
                pilot_index,
                complete.parent,
                root / "proposal-index.json",
            )
            indexed = json.loads(suggestion_index.read_text(encoding="utf-8"))
            self.assertEqual(indexed["summary"]["frames"], 270)
            self.assertEqual(indexed["summary"]["framesWithDetection"], 270)

    def test_video_generator_matches_strict_sidecar_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "fixture.avi"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"MJPG"),
                8.0,
                (64, 32),
            )
            self.assertTrue(writer.isOpened())
            for index in range(8):
                writer.write(np.full((32, 64, 3), index * 10, dtype=np.uint8))
            writer.release()
            output = root / "sidecar.npz"
            video_sha = sha256_file(video)
            with patch("analysis.ball_detector.YoloXBallDetector", _FakeDetector):
                infer_video_sidecar(
                    video,
                    root / "unused-model",
                    output,
                    recording_id="recording-1",
                    source_group="match-1",
                    split="train",
                    expected_video_sha256=video_sha,
                    detector_fps=8.0,
                )
            loaded = load_ball_presence_sidecar(
                output,
                expected_recording_id="recording-1",
                expected_source_group="match-1",
                expected_split="train",
                expected_video_sha256=video_sha,
                expected_duration_seconds=1.0,
                expected_width=64,
                expected_height=32,
                expected_detector_id="opencv-zoo-yolox-s-2022nov",
                expected_detector_sha256=(
                    "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
                ),
            )
            self.assertEqual(len(loaded.times), 8)
            self.assertTrue(loaded.frame_available.all())
            self.assertTrue((loaded.candidate_count == 1).all())


if __name__ == "__main__":
    unittest.main()
