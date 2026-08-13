from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.dinov2_embeddings import (
    DINO_CACHE_SCHEMA_VERSION,
    DINO_EMBEDDING_DIMENSION,
    DINO_TOKEN_COUNT,
    DinoEmbeddingError,
    DinoExtractorConfig,
    _patch_pool,
    letterbox_frame,
    load_dino_cache,
    preprocess_frames,
    sample_timestamps,
)


class DinoEmbeddingContractTests(unittest.TestCase):
    def test_sample_timestamps_uses_exact_grid_without_duration_overrun(self) -> None:
        values = sample_timestamps(1.01, 4.0)
        np.testing.assert_array_equal(values, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))

    def test_letterbox_and_preprocess_preserve_aspect_ratio_and_normalize(self) -> None:
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[:, :, 0] = 10
        frame[:, :, 1] = 20
        frame[:, :, 2] = 30
        image = letterbox_frame(frame, 28)
        self.assertEqual(image.shape, (28, 28, 3))
        self.assertEqual(int(image[0, 0, 0]), 0)
        values = preprocess_frames((frame,), input_size=28)
        self.assertEqual(values.shape, (1, 3, 28, 28))
        # BGR is converted to RGB before ImageNet normalization.
        self.assertAlmostEqual(float(values[0, 0, 14, 14]), (30 / 255 - 0.485) / 0.229, places=4)
        self.assertAlmostEqual(float(values[0, 1, 14, 14]), (20 / 255 - 0.456) / 0.224, places=4)
        self.assertAlmostEqual(float(values[0, 2, 14, 14]), (10 / 255 - 0.406) / 0.225, places=4)

    def test_cache_loader_rejects_incomplete_or_wrong_shape_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.npz"
            timestamps = np.asarray([0.0, 0.25], dtype=np.float64)
            tokens = np.zeros((2, DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION), dtype=np.float16)
            metadata = {
                "schemaVersion": DINO_CACHE_SCHEMA_VERSION,
                "completed": False,
                "recordingId": "synthetic",
            }
            np.savez_compressed(
                path,
                timestamps=timestamps,
                tokens=tokens,
                metadata_json=json.dumps(metadata),
            )
            with self.assertRaises(DinoEmbeddingError):
                load_dino_cache(path)

    def test_valid_cache_round_trips_float16_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.npz"
            timestamps = np.asarray([0.0, 0.25], dtype=np.float64)
            tokens = np.ones((2, DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION), dtype=np.float16)
            metadata = {
                "schemaVersion": DINO_CACHE_SCHEMA_VERSION,
                "completed": True,
                "recordingId": "synthetic",
                "recordingContentSha256": "a" * 64,
                "extractorConfigSha256": "b" * 64,
                "backbone": {"modelName": "dinov2_vits14"},
            }
            np.savez_compressed(path, timestamps=timestamps, tokens=tokens, metadata_json=json.dumps(metadata))
            loaded = load_dino_cache(
                path,
                recording_id="synthetic",
                recording_content_sha256="a" * 64,
                expected_config_sha256="b" * 64,
                expected_backbone={"modelName": "dinov2_vits14"},
            )
            self.assertEqual(loaded.tokens.shape, (2, DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION))
            self.assertEqual(loaded.tokens.dtype, np.float32)

    @unittest.skipUnless(
        __import__("importlib.util").util.find_spec("torch") is not None,
        "PyTorch is not installed in this test interpreter",
    )
    def test_patch_pool_produces_nine_regions(self) -> None:
        import torch

        patches = torch.arange(16 * 384, dtype=torch.float32).reshape(1, 16, 384)
        pooled = _patch_pool(torch, patches, 56)
        self.assertEqual(tuple(pooled.shape), (1, 9, 384))
        self.assertTrue(bool(torch.isfinite(pooled).all()))


if __name__ == "__main__":
    unittest.main()
