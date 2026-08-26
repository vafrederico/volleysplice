from __future__ import annotations

import unittest

import cv2
import numpy as np

from analysis.serving_side_flight import ResidualMotion
from analysis.side_switch_m1_motion import (
    M1_FEATURE_NAMES,
    PAIR_COUNT,
    extract_residual_motion,
    m1_features,
    sample_pairs,
)


def _motion(*, upward: bool, downward: bool) -> ResidualMotion:
    height, width = 72, 128
    energy = np.zeros((height, width), dtype=np.float32)
    flow_y = np.zeros_like(energy)
    if upward:
        energy[42:62, 16:112] = 1.0
        flow_y[42:62, 16:112] = -0.02
    if downward:
        energy[12:32, 16:112] = 1.0
        flow_y[12:32, 16:112] = 0.02
    zeros = np.zeros_like(energy)
    return ResidualMotion(
        energy=energy,
        flow_x=zeros,
        flow_y=flow_y,
        divergence=zeros,
    )


class SideSwitchM1MotionTest(unittest.TestCase):
    def test_sampling_uses_six_local_pairs_across_long_gap(self) -> None:
        pairs = sample_pairs(10.0, 20.0)
        self.assertEqual(len(pairs), PAIR_COUNT)
        self.assertEqual(pairs[0], (10.0, 10.25))
        self.assertEqual(pairs[-1], (19.75, 20.0))
        self.assertTrue(all(abs(right - left - 0.25) < 1e-12 for left, right in pairs))

    def test_sampling_scales_pair_delta_for_short_gap(self) -> None:
        pairs = sample_pairs(1.0, 1.125)
        self.assertEqual(pairs[0], (1.0, 1.0625))
        self.assertEqual(pairs[-1], (1.0625, 1.125))
        self.assertTrue(all(1.0 <= value <= 1.125 for pair in pairs for value in pair))

    def test_bidirectional_motion_activates_exchange_bundle(self) -> None:
        values, diagnostics = m1_features(
            [_motion(upward=True, downward=True)] * PAIR_COUNT,
            12.0,
        )
        self.assertEqual(tuple(values), M1_FEATURE_NAMES)
        self.assertGreater(values["nearToFarForegroundFluxMean"], 0.001)
        self.assertGreater(values["farToNearForegroundFluxMean"], 0.001)
        self.assertGreater(values["bidirectionalExchangeMinimum"], 0.001)
        self.assertEqual(values["coordinatedExchangePairFraction"], 1.0)
        self.assertEqual(values["coordinatedExchangeDurationSeconds"], 12.0)
        self.assertEqual(diagnostics["pairCount"], PAIR_COUNT)

    def test_one_way_motion_has_zero_bidirectional_exchange(self) -> None:
        values, _ = m1_features(
            [_motion(upward=True, downward=False)] * PAIR_COUNT,
            5.0,
        )
        self.assertGreater(values["nearToFarForegroundFluxMean"], 0.001)
        self.assertEqual(values["farToNearForegroundFluxMean"], 0.0)
        self.assertEqual(values["bidirectionalExchangeMinimum"], 0.0)
        self.assertEqual(values["coordinatedExchangePairFraction"], 0.0)

    def test_affine_camera_translation_leaves_little_residual_energy(self) -> None:
        base = np.zeros((144, 256), dtype=np.uint8)
        cv2.rectangle(base, (20, 20), (220, 120), 180, -1)
        cv2.circle(base, (90, 70), 17, 255, -1)
        translated = cv2.warpAffine(
            base,
            np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]]),
            (256, 144),
            borderMode=cv2.BORDER_REFLECT,
        )
        motion = extract_residual_motion(base, translated)
        self.assertLess(float(np.mean(motion.energy)), 0.001)
        self.assertTrue(np.isfinite(motion.energy).all())


if __name__ == "__main__":
    unittest.main()
