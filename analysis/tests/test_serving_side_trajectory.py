from __future__ import annotations

import unittest

import numpy as np

from analysis.serving_side_flight import ResidualMotion
from analysis.serving_side_trajectory import (
    FEATURE_NAMES,
    PAIR_COUNT,
    extract_trajectory_features,
    link_component_tracks,
    motion_components,
)


def moving_motion(pair_index: int, *, growing: bool = True) -> ResidualMotion:
    height, width = 40, 60
    energy = np.zeros((height, width), dtype=np.float32)
    radius = 1 + (pair_index // 3 if growing else 0)
    center_y = 8 + pair_index * 2
    center_x = 18 + pair_index
    energy[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ] = 1.0
    flow_x = np.zeros_like(energy)
    flow_y = np.zeros_like(energy)
    flow_x[energy > 0] = 1 / width
    flow_y[energy > 0] = 2 / height
    divergence = np.zeros_like(energy)
    return ResidualMotion(energy, flow_x, flow_y, divergence)


class ServingSideTrajectoryTests(unittest.TestCase):
    def test_links_a_persistent_moving_and_expanding_component(self) -> None:
        motions = [moving_motion(index) for index in range(PAIR_COUNT)]
        components = [
            motion_components(motion, index)
            for index, motion in enumerate(motions)
        ]
        tracks = link_component_tracks(components)
        self.assertEqual(len(tracks[0].components), PAIR_COUNT)
        features = extract_trajectory_features(motions)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(features["best:persistence"], 1.0)
        self.assertGreater(features["best:displacementY"], 0.25)
        self.assertGreater(features["best:logAreaChange"], 0)
        self.assertGreater(features["best:straightness"], 0.99)
        self.assertGreater(features["best:directionConsistency"], 0.99)
        self.assertGreater(features["best:flowAlignment"], 0.99)

    def test_empty_motion_has_a_stable_finite_zero_contract(self) -> None:
        zeros = np.zeros((40, 60), dtype=np.float32)
        motions = [ResidualMotion(zeros, zeros, zeros, zeros) for _ in range(PAIR_COUNT)]
        features = extract_trajectory_features(motions)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertTrue(all(value == 0 for value in features.values()))

    def test_linking_is_deterministic_with_a_distractor(self) -> None:
        motions = []
        for pair_index in range(PAIR_COUNT):
            motion = moving_motion(pair_index, growing=False)
            energy = motion.energy.copy()
            energy[30:32, 45:47] = 0.35
            motions.append(
                ResidualMotion(
                    energy,
                    motion.flow_x,
                    motion.flow_y,
                    motion.divergence,
                )
            )
        first = extract_trajectory_features(motions)
        second = extract_trajectory_features(motions)
        self.assertEqual(first, second)
        self.assertEqual(first["best:persistence"], 1.0)
        self.assertGreaterEqual(first["tracks:persistentCount"], 2)

    def test_rejects_misaligned_motion_arrays(self) -> None:
        energy = np.zeros((40, 60), dtype=np.float32)
        wrong = np.zeros((39, 60), dtype=np.float32)
        with self.assertRaises(ValueError):
            motion_components(ResidualMotion(energy, wrong, energy, energy), 0)


if __name__ == "__main__":
    unittest.main()
