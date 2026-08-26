from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_t3_jersey_transport import JerseyTracklet
from analysis.side_switch_t14_dominant_tracklet_medoid import dominant_selection


def _tracklet(index: int, reliability: float = 0.8) -> JerseyTracklet:
    descriptor = np.zeros(64, dtype=np.float64)
    descriptor[index] = 1.0
    return JerseyTracklet(descriptor, 0.5, 0.7, 0.1, reliability, 3)


class SideSwitchT14DominantTrackletMedoidTest(unittest.TestCase):
    def test_majority_descriptor_is_medoid(self) -> None:
        stable = (_tracklet(1), _tracklet(1), _tracklet(9))
        selection = dominant_selection(stable, stable, stable[0].descriptor)
        self.assertEqual(selection.kind, "stable-medoid")
        self.assertEqual(selection.medoid_index, 0)
        self.assertIs(selection.unit, stable[0])

    def test_tied_medoids_use_existing_order(self) -> None:
        stable = (_tracklet(1), _tracklet(9))
        selection = dominant_selection(stable, stable, stable[0].descriptor)
        self.assertEqual(selection.medoid_index, 0)

    def test_unavailable_side_remains_unavailable(self) -> None:
        selection = dominant_selection((), (), None)
        self.assertEqual(selection.kind, "unavailable")
        self.assertIsNone(selection.unit)


if __name__ == "__main__":
    unittest.main()
