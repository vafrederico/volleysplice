from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.ball_presence import (
    BALL_PRESENCE_FEATURE_NAMES,
    BALL_PRESENCE_KIND,
    BALL_PRESENCE_SCHEMA_VERSION,
    BallPresenceError,
    aggregate_ball_presence,
    contextualize_ball_presence,
    load_ball_presence_sidecar,
    sha256_file,
)


VIDEO_SHA256 = "a" * 64
DETECTOR_SHA256 = "b" * 64
RECORDING_ID = "indoor-set-01"


class BallPresenceSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="volleycut-ball-presence-test-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    @staticmethod
    def metadata(
        *,
        duration: float = 1.0,
        sample_fps: float = 8.0,
        status: str = "available",
        unavailable_reason: str | None = None,
        detector_sha256: str | None = DETECTOR_SHA256,
    ) -> dict[str, object]:
        return {
            "schemaVersion": BALL_PRESENCE_SCHEMA_VERSION,
            "kind": BALL_PRESENCE_KIND,
            "status": status,
            "unavailableReason": unavailable_reason,
            "recordingId": RECORDING_ID,
            "sourceGroup": "source-a",
            "split": "train",
            "sourceVideo": {
                "contentSha256": VIDEO_SHA256,
                "durationSeconds": duration,
                "width": 960,
                "height": 540,
                "fps": 30.0,
                "frameCount": int(round(duration * 30)),
                "timeBase": "video-start-seconds",
            },
            "detector": {
                "id": "frozen-sports-ball-v1",
                "artifactSha256": detector_sha256,
                "sampleFps": sample_fps,
                "scoreFloor": 0.01,
                "nmsThreshold": 0.5,
                "maximumDetections": 20,
                "opencvVersion": "4.12.0",
                "implementationSha256": "c" * 64,
                "sampleFrameRule": "round(sampleIndex * sourceFps / sampleFps)",
                "roi": {"x": 0.02, "y": 0.1, "width": 0.96, "height": 0.88},
            },
        }

    def write_sidecar(
        self,
        name: str,
        *,
        metadata: dict[str, object] | None = None,
        times: np.ndarray | None = None,
        frame_available: np.ndarray | None = None,
        observability: np.ndarray | None = None,
        best_score: np.ndarray | None = None,
        candidate_count: np.ndarray | None = None,
        extra: dict[str, np.ndarray] | None = None,
    ) -> Path:
        payload = metadata or self.metadata()
        detector = payload["detector"]
        source = payload["sourceVideo"]
        assert isinstance(detector, dict)
        assert isinstance(source, dict)
        if times is None:
            if payload["status"] == "available":
                count = int(
                    np.ceil(
                        float(source["durationSeconds"])
                        * float(detector["sampleFps"])
                        - 1e-9
                    )
                )
                times = np.arange(count, dtype=np.float64) / float(
                    detector["sampleFps"]
                )
            else:
                times = np.empty(0, dtype=np.float64)
        length = len(times)
        frame_available = (
            np.ones(length, dtype=np.bool_)
            if frame_available is None
            else frame_available
        )
        observability = (
            np.ones(length, dtype=np.float32)
            if observability is None
            else observability
        )
        best_score = (
            np.zeros(length, dtype=np.float32)
            if best_score is None
            else best_score
        )
        candidate_count = (
            np.zeros(length, dtype=np.int16)
            if candidate_count is None
            else candidate_count
        )
        path = self.root / name
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(payload, allow_nan=False)),
            times=times,
            frame_available=frame_available,
            observability=observability,
            best_score=best_score,
            candidate_count=candidate_count,
            **(extra or {}),
        )
        return path

    def load(self, path: Path, **overrides: object):
        arguments: dict[str, object] = {
            "expected_recording_id": RECORDING_ID,
            "expected_source_group": "source-a",
            "expected_split": "train",
            "expected_video_sha256": VIDEO_SHA256,
            "expected_duration_seconds": 1.0,
            "expected_width": 960,
            "expected_height": 540,
            "expected_detector_id": "frozen-sports-ball-v1",
            "expected_detector_sha256": DETECTOR_SHA256,
            "expected_roi": (0.02, 0.1, 0.96, 0.88),
        }
        arguments.update(overrides)
        return load_ball_presence_sidecar(path, **arguments)

    def test_loads_strict_provenance_and_aggregates_centered_windows(self) -> None:
        path = self.write_sidecar(
            "available.npz",
            observability=np.asarray(
                [1.0, 0.5, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            best_score=np.asarray(
                [0.0, 0.8, 0.4, 0.0, 0.9, 0.1, 0.0, 0.0],
                dtype=np.float32,
            ),
            candidate_count=np.asarray([0, 1, 2, 0, 1, 1, 0, 0], dtype=np.int16),
        )
        artifact_sha256 = sha256_file(path)

        sidecar = self.load(path, expected_sidecar_sha256=artifact_sha256)
        features = aggregate_ball_presence(
            sidecar,
            np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float64),
            analysis_fps=4.0,
            detection_threshold=0.5,
        )

        self.assertEqual(features.names, BALL_PRESENCE_FEATURE_NAMES)
        self.assertEqual(features.sidecar_sha256, artifact_sha256)
        self.assertEqual(features.detector_artifact_sha256, DETECTOR_SHA256)
        self.assertEqual(features.values.shape, (4, 8))
        indexes = {name: BALL_PRESENCE_FEATURE_NAMES.index(name) for name in features.names}
        np.testing.assert_array_equal(
            features.values[:, indexes["ball_detector_available"]],
            np.ones(4, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            features.values[:, indexes["ball_observation_fraction"]],
            np.ones(4, dtype=np.float32),
        )
        np.testing.assert_allclose(
            features.values[:, indexes["ball_observability_quality"]],
            np.asarray([1.0, 0.75, 0.9, 1.0]),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            features.values[:, indexes["ball_presence_probability"]],
            np.asarray([0.0, 0.8, 0.9, 0.1]),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            features.values[:, indexes["ball_detected_fraction"]],
            np.asarray([0.0, 0.5, 0.5, 0.0]),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            features.values[:, indexes["ball_candidate_count_normalized"]],
            np.asarray([0.0, 0.5, 1.0 / 6.0, 1.0 / 6.0]),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            features.values[:, indexes["ball_seconds_since_detection"]],
            np.asarray([10.0, 0.125, 0.0, 0.25]),
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            features.values[:, indexes["ball_quality_gated_presence"]],
            np.asarray([0.0, 0.4, 0.72, 0.1]),
            rtol=0.0,
            atol=1e-7,
        )
        self.assertFalse(features.times.flags.writeable)
        self.assertFalse(features.values.flags.writeable)

    def test_seconds_since_detection_is_causal_while_presence_window_is_centered(self) -> None:
        metadata = self.metadata(duration=0.5, sample_fps=12.0)
        scores = np.zeros(6, dtype=np.float32)
        counts = np.zeros(6, dtype=np.int16)
        scores[4] = 0.9  # 0.333s: future evidence inside the 0.25s centered window.
        counts[4] = 1
        path = self.write_sidecar(
            "causal.npz",
            metadata=metadata,
            best_score=scores,
            candidate_count=counts,
        )
        sidecar = self.load(path, expected_duration_seconds=0.5)

        features = aggregate_ball_presence(
            sidecar,
            np.asarray([0.0, 0.25]),
            analysis_fps=4.0,
            detection_threshold=0.5,
        )
        presence = features.values[:, 3]
        elapsed = features.values[:, 6]

        self.assertAlmostEqual(float(presence[1]), 0.9, places=6)
        self.assertEqual(float(elapsed[1]), 10.0)

    def test_decode_gap_reduces_observation_fraction_without_becoming_no_ball(self) -> None:
        metadata = self.metadata(duration=0.5, sample_fps=8.0)
        path = self.write_sidecar(
            "gap.npz",
            metadata=metadata,
            frame_available=np.asarray([True, False, True, True]),
            observability=np.asarray([1.0, 0.0, 0.8, 1.0], dtype=np.float32),
        )
        sidecar = self.load(path, expected_duration_seconds=0.5)

        values = aggregate_ball_presence(
            sidecar,
            np.asarray([0.0, 0.25]),
            analysis_fps=4.0,
            detection_threshold=0.5,
        ).values

        self.assertEqual(float(values[1, 0]), 1.0)
        self.assertEqual(float(values[1, 1]), 0.5)
        self.assertAlmostEqual(float(values[1, 2]), 0.8, places=6)
        self.assertEqual(float(values[1, 3]), 0.0)
        self.assertEqual(float(values[1, 6]), 10.0)

    def test_unavailable_detector_and_available_no_detection_are_distinct(self) -> None:
        unavailable_metadata = self.metadata(
            duration=0.5,
            status="unavailable",
            unavailable_reason="model-missing",
            detector_sha256=None,
        )
        unavailable_path = self.write_sidecar(
            "unavailable.npz", metadata=unavailable_metadata
        )
        unavailable = self.load(
            unavailable_path,
            expected_duration_seconds=0.5,
            expected_detector_sha256=None,
        )
        missing_values = aggregate_ball_presence(
            unavailable,
            np.asarray([0.0, 0.25]),
            analysis_fps=4.0,
            detection_threshold=0.5,
        ).values

        available_metadata = self.metadata(duration=0.5)
        available_path = self.write_sidecar(
            "no-detection.npz", metadata=available_metadata
        )
        available = self.load(available_path, expected_duration_seconds=0.5)
        no_detection_values = aggregate_ball_presence(
            available,
            np.asarray([0.0, 0.25]),
            analysis_fps=4.0,
            detection_threshold=0.5,
        ).values

        np.testing.assert_array_equal(missing_values, np.zeros_like(missing_values))
        np.testing.assert_array_equal(no_detection_values[:, 0], np.ones(2))
        np.testing.assert_array_equal(no_detection_values[:, 1], np.ones(2))
        np.testing.assert_array_equal(no_detection_values[:, 3:6], np.zeros((2, 3)))
        np.testing.assert_array_equal(no_detection_values[:, 6], np.full(2, 10.0))

        missing_features = aggregate_ball_presence(
            unavailable,
            np.asarray([0.0, 0.25]),
            analysis_fps=4.0,
            detection_threshold=0.5,
        )
        contextual, names = contextualize_ball_presence(
            missing_features,
            (-0.25, 0.0, 0.25),
        )
        self.assertEqual(contextual.shape, (2, 3 * len(BALL_PRESENCE_FEATURE_NAMES)))
        self.assertTrue((contextual == 0).all())
        self.assertEqual(names[0], "t-0.25s/ball_detector_available")

    def test_rejects_provenance_and_artifact_hash_mismatches(self) -> None:
        path = self.write_sidecar("provenance.npz")
        cases = {
            "recording id": {"expected_recording_id": "different"},
            "source group": {"expected_source_group": "different"},
            "split": {"expected_split": "validation"},
            "source video": {"expected_video_sha256": "c" * 64},
            "source duration": {"expected_duration_seconds": 2.0},
            "source width": {"expected_width": 1280},
            "source height": {"expected_height": 720},
            "detector id": {"expected_detector_id": "different-detector"},
            "detector artifact": {"expected_detector_sha256": "d" * 64},
            "detector sample fps": {"expected_detector_sample_fps": 12.0},
            "detector score floor": {"expected_score_floor": 0.02},
            "detector NMS": {"expected_nms_threshold": 0.4},
            "detector max count": {"expected_maximum_detections": 10},
            "detector implementation": {
                "expected_implementation_sha256": "d" * 64
            },
            "sidecar artifact": {"expected_sidecar_sha256": "e" * 64},
            "detector ROI": {"expected_roi": (0.0, 0.0, 1.0, 1.0)},
        }
        for description, overrides in cases.items():
            with self.subTest(description=description):
                with self.assertRaises(BallPresenceError):
                    self.load(path, **overrides)

    def test_rejects_nonuniform_or_semantically_invalid_frame_arrays(self) -> None:
        invalid_times = np.arange(8, dtype=np.float64) / 8.0
        invalid_times[3] += 0.02
        paths = {
            "nonuniform time grid": self.write_sidecar(
                "bad-time.npz", times=invalid_times
            ),
            "score without candidate": self.write_sidecar(
                "bad-score.npz",
                best_score=np.asarray(
                    [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
            ),
            "nonzero unavailable frame": self.write_sidecar(
                "bad-gap.npz",
                frame_available=np.asarray(
                    [True, False, True, True, True, True, True, True]
                ),
            ),
            "unknown archive key": self.write_sidecar(
                "extra-key.npz", extra={"unexpected": np.zeros(1)}
            ),
            "count above configured maximum": self.write_sidecar(
                "too-many.npz",
                best_score=np.asarray([0.5] + [0.0] * 7, dtype=np.float32),
                candidate_count=np.asarray([21] + [0] * 7, dtype=np.int16),
            ),
            "candidate score below extraction floor": self.write_sidecar(
                "below-floor.npz",
                best_score=np.asarray([0.005] + [0.0] * 7, dtype=np.float32),
                candidate_count=np.asarray([1] + [0] * 7, dtype=np.int16),
            ),
        }
        for description, path in paths.items():
            with self.subTest(description=description):
                with self.assertRaises(BallPresenceError):
                    self.load(path)

    def test_rejects_unavailable_status_with_rows_or_detector_hash(self) -> None:
        metadata = self.metadata(
            status="unavailable",
            unavailable_reason="decode-failed",
            detector_sha256=DETECTOR_SHA256,
        )
        path = self.write_sidecar("bad-unavailable.npz", metadata=metadata)

        with self.assertRaisesRegex(BallPresenceError, "must be null"):
            self.load(path, expected_detector_sha256=None)

        metadata["detector"] = dict(metadata["detector"])  # type: ignore[arg-type]
        metadata["detector"]["artifactSha256"] = None  # type: ignore[index]
        rows = np.arange(8, dtype=np.float64) / 8.0
        path = self.write_sidecar(
            "bad-unavailable-rows.npz", metadata=metadata, times=rows
        )
        with self.assertRaisesRegex(BallPresenceError, "empty arrays"):
            self.load(path, expected_detector_sha256=None)

    def test_rejects_partial_or_misaligned_target_timeline(self) -> None:
        sidecar = self.load(self.write_sidecar("alignment.npz"))
        cases = (
            np.asarray([0.0, 0.25, 0.5]),
            np.asarray([0.1, 0.35, 0.6, 0.85]),
            np.asarray([0.0, 0.25, 0.5, 0.5]),
        )
        for times in cases:
            with self.subTest(times=times.tolist()):
                with self.assertRaises(BallPresenceError):
                    aggregate_ball_presence(
                        sidecar,
                        times,
                        analysis_fps=4.0,
                        detection_threshold=0.5,
                    )

        with self.assertRaisesRegex(BallPresenceError, "at least twice"):
            aggregate_ball_presence(
                sidecar,
                np.arange(5, dtype=np.float64) / 5.0,
                analysis_fps=5.0,
                detection_threshold=0.5,
            )
        with self.assertRaisesRegex(BallPresenceError, "score floor"):
            aggregate_ball_presence(
                sidecar,
                np.asarray([0.0, 0.25, 0.5, 0.75]),
                analysis_fps=4.0,
                detection_threshold=0.001,
            )


if __name__ == "__main__":
    unittest.main()
