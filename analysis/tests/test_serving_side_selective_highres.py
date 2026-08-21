from __future__ import annotations

import unittest

import cv2
import numpy as np

from analysis.serving_side_flight import extract_motion_sequence
from analysis.serving_side_selective_highres import (
    FEATURE_NAMES,
    extract_selective_highres_features,
    extract_selective_highres_from_context,
    prepare_selective_highres_context,
    select_tracks,
)


def moving_frames() -> tuple[list[np.ndarray], list[np.ndarray]]:
    source = []
    for index in range(9):
        frame = np.full((432, 768, 3), 96, dtype=np.uint8)
        cv2.circle(frame, (180 + 22 * index, 100 + 13 * index), 7, (240, 240, 240), -1)
        cv2.line(frame, (0, 300), (767, 300), (110, 110, 110), 2)
        source.append(frame)
    low = [cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA) for frame in source]
    return source, low


class SelectiveHighresTests(unittest.TestCase):
    def test_moving_compact_object_produces_finite_features(self) -> None:
        source, low = moving_frames()
        motions = extract_motion_sequence(low)
        tracks = select_tracks(motions)
        self.assertIsNotNone(tracks["persistent"])
        self.assertIsNotNone(tracks["compact"])
        features = extract_selective_highres_features(source, low, motions)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertGreater(features["compact:coverage"], 0)
        self.assertGreater(features["compact:contrast:max"], 0)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_static_frames_have_a_stable_zero_contract(self) -> None:
        source = [np.full((432, 768, 3), 96, dtype=np.uint8) for _ in range(9)]
        low = [cv2.resize(frame, (192, 108)) for frame in source]
        motions = extract_motion_sequence(low)
        first = extract_selective_highres_features(source, low, motions)
        second = extract_selective_highres_features(source, low, motions)
        self.assertEqual(first, second)
        self.assertEqual(tuple(first), FEATURE_NAMES)
        self.assertTrue(all(value == 0 for value in first.values()))

    def test_rejects_misaligned_frame_counts(self) -> None:
        source, low = moving_frames()
        motions = extract_motion_sequence(low)
        with self.assertRaises(ValueError):
            extract_selective_highres_features(source[:-1], low, motions)

    def test_one_prepared_context_supports_predeclared_crop_sizes(self) -> None:
        source, low = moving_frames()
        motions = extract_motion_sequence(low)
        context = prepare_selective_highres_context(source, low, motions)
        small = extract_selective_highres_from_context(
            context, source_patch_fraction=0.12, patch_size=96
        )
        large = extract_selective_highres_from_context(
            context, source_patch_fraction=0.30, patch_size=96
        )
        self.assertEqual(tuple(small), FEATURE_NAMES)
        self.assertEqual(tuple(large), FEATURE_NAMES)
        self.assertNotEqual(small, large)


if __name__ == "__main__":
    unittest.main()
