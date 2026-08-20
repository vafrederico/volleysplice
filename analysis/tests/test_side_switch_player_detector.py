from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_player_detector import (
    INPUT_SIZE,
    _Tile,
    _decode_tile,
    detector_tiles,
    mediapipe_anchors,
)


class SideSwitchPlayerDetectorTests(unittest.TestCase):
    def test_anchor_generator_matches_frozen_graph_layout(self) -> None:
        anchors = mediapipe_anchors()

        self.assertEqual(anchors.shape, (2254, 2))
        np.testing.assert_allclose(anchors[0], [0.5 / 28.0, 0.5 / 28.0])
        np.testing.assert_allclose(anchors[1568], [0.5 / 14.0, 0.5 / 14.0])
        np.testing.assert_allclose(anchors[1960], [0.5 / 7.0, 0.5 / 7.0])
        np.testing.assert_allclose(anchors[-1], [6.5 / 7.0, 6.5 / 7.0])

    def test_four_tiles_overlap_but_own_disjoint_quadrants(self) -> None:
        tiles = detector_tiles(1000, 500)

        self.assertEqual(len(tiles), 4)
        self.assertEqual({(tile.owner_column, tile.owner_row) for tile in tiles}, {
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
        })
        self.assertTrue(all(tile.width == 620 for tile in tiles))
        self.assertTrue(all(tile.height == 310 for tile in tiles))

    def test_decoder_builds_torso_box_from_landmarks(self) -> None:
        anchors = mediapipe_anchors()
        index = int(np.argmin(np.linalg.norm(anchors - [0.5, 0.5], axis=1)))
        geometry = np.zeros((1, 2254, 12), dtype=np.float32)
        scores = np.full((1, 2254, 1), -100.0, dtype=np.float32)
        scores[0, index, 0] = 10.0
        targets = np.asarray(
            [[112.0, 140.0], [112.0, 200.0], [112.0, 100.0], [112.0, 70.0]]
        )
        geometry[0, index, 4:] = (
            targets - anchors[index] * INPUT_SIZE
        ).reshape(-1)

        detections, raw_count = _decode_tile(
            geometry,
            scores,
            tile=_Tile(0, 0, INPUT_SIZE, INPUT_SIZE, 1, 1),
            frame_width=INPUT_SIZE,
            frame_height=INPUT_SIZE,
            pad_x=0.0,
            pad_y=0.0,
        )

        self.assertEqual(raw_count, 1)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].hip_y, 140.0, places=4)
        self.assertLess(detections[0].y, detections[0].shoulder_y)
        self.assertGreater(detections[0].height, 40.0)


if __name__ == "__main__":
    unittest.main()
