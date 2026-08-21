from __future__ import annotations

import unittest

import numpy as np

from analysis.serving_side_flight import (
    FEATURE_VERSION,
    OFFSETS_SECONDS,
    extract_flight_features,
    feature_names,
)


class ServingSideFlightTests(unittest.TestCase):
    @staticmethod
    def _moving_dot(direction: int) -> list[np.ndarray]:
        rng = np.random.default_rng(11)
        background = rng.integers(0, 40, (108, 192), dtype=np.uint8)
        frames = []
        for index in range(len(OFFSETS_SECONDS)):
            frame = np.repeat(background[:, :, None], 3, axis=2)
            center_y = 54 + direction * (index - 4) * 5
            frame[center_y - 3 : center_y + 4, 92:99] = 255
            frames.append(frame)
        return frames

    def test_feature_contract_is_stable_and_finite(self) -> None:
        values = extract_flight_features(self._moving_dot(1), 3, 3)

        self.assertEqual(FEATURE_VERSION, "serving-side-concentrated-flight-grid-v1")
        self.assertEqual(tuple(values), feature_names(3, 3))
        self.assertTrue(np.isfinite(list(values.values())).all())
        for phase in ("launch", "early", "late"):
            mass = sum(
                value
                for name, value in values.items()
                if name.startswith(f"{phase}:grid:")
            )
            self.assertAlmostEqual(mass, 1.0, places=6)

    def test_centroid_trajectory_distinguishes_toward_and_away_motion(self) -> None:
        toward = extract_flight_features(self._moving_dot(1), 4, 4)
        away = extract_flight_features(self._moving_dot(-1), 4, 4)

        self.assertGreater(
            toward["trajectory:launchToEarly:centroidY"], 0.05
        )
        self.assertLess(away["trajectory:launchToEarly:centroidY"], -0.05)
        self.assertGreater(
            toward["trajectory:earlyToLate:centroidY"], 0.05
        )
        self.assertLess(away["trajectory:earlyToLate:centroidY"], -0.05)

    def test_rejects_invalid_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            feature_names(1, 3)


if __name__ == "__main__":
    unittest.main()
