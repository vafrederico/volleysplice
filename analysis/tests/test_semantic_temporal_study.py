from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.dinov2_embeddings import DinoCache
from analysis.features import FeatureSequence, VideoMetadata
from analysis.schema import Interval, Recording
from analysis.semantic_temporal_study import (
    CHUNK_LENGTH_TICKS,
    CHUNK_STRIDE_TICKS,
    align_dino_to_audiovisual,
    build_sequence_example,
    candidate_specs,
    chunk_example,
)


def _recording() -> Recording:
    return Recording(
        id="synthetic",
        video=Path("synthetic.mp4"),
        split="train",
        source_group="source-a",
        environment="indoor",
        game={},
        rallies=(Interval(1.0, 1.5, ("ordinary",)), Interval(3.0, 3.5, ("ace",))),
        ignored_intervals=(Interval(1.75, 2.0),),
        roi=None,
        capture={},
        consent={"analyze": True, "train": True},
        content_sha256="a" * 64,
        raw={},
    )


class SemanticTemporalStudyTests(unittest.TestCase):
    def test_alignment_requires_nearest_dino_tick_within_half_analysis_tick(self) -> None:
        tokens = np.zeros((3, 10, 384), dtype=np.float32)
        cache = DinoCache(
            path=Path("dino.npz"),
            timestamps=np.asarray([0.0, 0.25, 0.5]),
            tokens=tokens,
            metadata={},
        )
        aligned = align_dino_to_audiovisual(cache, np.asarray([0.0, 0.24, 0.5]))
        self.assertEqual(aligned.shape, (3, 10, 384))
        with self.assertRaisesRegex(RuntimeError, "not aligned"):
            align_dino_to_audiovisual(cache, np.asarray([0.0, 0.7, 0.5]))

    def test_sequence_targets_and_chunking_preserve_censored_ticks(self) -> None:
        times = np.arange(8.0, dtype=np.float64) / 4.0
        audiovisual = FeatureSequence(
            times=times,
            values=np.zeros((len(times), 90), dtype=np.float32),
            names=tuple(f"signal-{index}" for index in range(90)),
            metadata=VideoMetadata(2.0, 960, 540, 30.0, 60, True),
        )
        cache = DinoCache(
            path=Path("dino.npz"),
            timestamps=times,
            tokens=np.zeros((len(times), 10, 384), dtype=np.float32),
            metadata={},
        )
        example = build_sequence_example(_recording(), cache, audiovisual)
        self.assertEqual(float(example.live_target.max()), 1.0)
        self.assertFalse(bool(example.valid_mask[-1]))
        chunks = chunk_example(example, chunk_length=4, stride=2)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].dino_tokens.shape, (4, 10, 384))
        covered = np.zeros(len(times), dtype=bool)
        for chunk in chunks:
            covered[chunk.start_index : chunk.start_index + chunk.length] = True
        self.assertTrue(bool(covered.all()))

    def test_candidate_order_matches_preregistered_track_t_sequence(self) -> None:
        self.assertEqual(
            [item.name for item in candidate_specs()],
            [
                "audiovisual_tcn_control",
                "dino_class_only",
                "dino_class_plus_regions",
                "dino_regions_plus_audiovisual",
                "dino_regions_plus_audiovisual_boundary_heads",
                "dino_regions_plus_audiovisual_short_path",
            ],
        )
        self.assertEqual(CHUNK_LENGTH_TICKS, 256)
        self.assertEqual(CHUNK_STRIDE_TICKS, 128)


if __name__ == "__main__":
    unittest.main()
