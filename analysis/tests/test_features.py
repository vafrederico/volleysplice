from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.config import FeatureConfig
from analysis.features import (
    FeatureSequence,
    VideoMetadata,
    cached_features,
    extract_features,
    feature_names,
)


try:
    import cv2
except Exception:  # dependency test must also tolerate binary/ABI import failures
    cv2 = None


class CacheIntegrityTests(unittest.TestCase):
    def test_same_size_same_mtime_content_change_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-cache-integrity-") as directory:
            root = Path(directory)
            video = root / "source.bin"
            cache_dir = root / "cache"
            video.write_bytes(b"first")
            original_stat = video.stat()
            config = FeatureConfig(use_optical_flow=False, context_offsets_seconds=(0.0,))
            metadata = VideoMetadata(1.0, 1280, 720, 30.0, 30, False)

            def sequence(value: float) -> FeatureSequence:
                return FeatureSequence(
                    times=np.asarray([0.0], dtype=np.float64),
                    values=np.full((1, len(feature_names(config))), value, dtype=np.float32),
                    names=feature_names(config),
                    metadata=metadata,
                )

            with patch("analysis.features.extract_features", return_value=sequence(1.0)):
                first = cached_features("source", video, config, None, cache_dir)
            video.write_bytes(b"other")
            os.utime(video, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            with patch("analysis.features.extract_features", return_value=sequence(2.0)) as extractor:
                second = cached_features("source", video, config, None, cache_dir)

            extractor.assert_called_once()
            self.assertEqual(len(list(cache_dir.glob("*.npz"))), 2)
            self.assertEqual(float(first.values[0, 0]), 1.0)
            self.assertEqual(float(second.values[0, 0]), 2.0)


@unittest.skipUnless(cv2 is not None, "OpenCV is not available to the test interpreter")
class RealFeatureExtractionTests(unittest.TestCase):
    def make_video(self, path: Path) -> None:
        assert cv2 is not None
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            8.0,
            (160, 96),
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG VideoWriter is unavailable")
        try:
            for index in range(8):
                frame = np.zeros((96, 160, 3), dtype=np.uint8)
                frame[:, :, 1] = 35
                cv2.rectangle(frame, (10 + 12 * index, 30), (34 + 12 * index, 60), (255, 255, 255), -1)
                writer.write(frame)
        finally:
            writer.release()

    def test_real_extraction_valid_cache_hit_and_invalid_cache_recovery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-feature-test-") as directory:
            root = Path(directory)
            video = root / "moving-box.avi"
            cache_dir = root / "cache"
            self.make_video(video)
            config = FeatureConfig(
                analysis_fps=4.0,
                resize_width=64,
                resize_height=36,
                grid_size=2,
                use_optical_flow=False,
                context_offsets_seconds=(0.0,),
            )

            extracted = extract_features(video, config, roi=(0.05, 0.05, 0.9, 0.9))

            self.assertEqual(extracted.names, feature_names(config))
            self.assertEqual(extracted.values.shape, (len(extracted.times), len(extracted.names)))
            self.assertGreaterEqual(len(extracted.times), 3)
            self.assertTrue(np.all(np.diff(extracted.times) > 0))
            self.assertTrue(np.isfinite(extracted.values).all())
            self.assertEqual(extracted.metadata.width, 160)
            self.assertEqual(extracted.metadata.height, 96)
            self.assertAlmostEqual(extracted.metadata.fps, 8.0, places=1)

            cached = cached_features(
                "moving-box",
                video,
                config,
                (0.05, 0.05, 0.9, 0.9),
                cache_dir,
            )
            cache_files = list(cache_dir.glob("*.npz"))
            self.assertEqual(len(cache_files), 1)
            cache_path = cache_files[0]
            with patch(
                "analysis.features.extract_features",
                side_effect=AssertionError("valid cache should avoid extraction"),
            ):
                cache_hit = cached_features(
                    "moving-box",
                    video,
                    config,
                    (0.05, 0.05, 0.9, 0.9),
                    cache_dir,
                )
            np.testing.assert_array_equal(cache_hit.times, cached.times)
            np.testing.assert_array_equal(cache_hit.values, cached.values)

            np.savez_compressed(
                cache_path,
                times=cached.times[::-1],
                values=cached.values[::-1],
                names=np.asarray(cached.names),
                metadata_json=json.dumps(asdict(cached.metadata), sort_keys=True),
            )
            with patch("analysis.features.extract_features", wraps=extract_features) as extractor:
                repaired = cached_features(
                    "moving-box",
                    video,
                    config,
                    (0.05, 0.05, 0.9, 0.9),
                    cache_dir,
                )
            extractor.assert_called_once()
            self.assertTrue(np.all(np.diff(repaired.times) > 0))
            self.assertTrue(np.isfinite(repaired.values).all())


if __name__ == "__main__":
    unittest.main()
