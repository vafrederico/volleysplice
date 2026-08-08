from __future__ import annotations

import unittest

from analysis.rallies import detect_rallies


def timeline(*active_ranges: tuple[float, float], duration: float = 24, fps: int = 4):
    samples = []
    for index in range(duration * fps):
        timestamp = index / fps
        active = any(start <= timestamp < end for start, end in active_ranges)
        samples.append({
            "time": timestamp,
            "motion": 0.8 if active else 0.02,
            "audio": 0.6 if active else 0.01,
            "activity": 0.75 if active else 0.03,
        })
    return samples


class RallyDetectionTests(unittest.TestCase):
    def test_detects_and_pads_one_sustained_activity_interval(self):
        rallies = detect_rallies(timeline((5, 10)), 24, camera_stability=0.95)
        self.assertEqual(len(rallies), 1)
        self.assertEqual(rallies[0]["id"], "R01")
        self.assertAlmostEqual(rallies[0]["start"], 4.25)
        self.assertAlmostEqual(rallies[0]["end"], 11.0)

    def test_closes_a_short_internal_gap(self):
        rallies = detect_rallies(timeline((5, 8), (9, 12)), 24, camera_stability=0.95)
        self.assertEqual(len(rallies), 1)
        self.assertLessEqual(rallies[0]["start"], 5)
        self.assertGreaterEqual(rallies[0]["end"], 12)

    def test_keeps_well_separated_activity_as_two_candidates(self):
        rallies = detect_rallies(timeline((3, 7), (14, 19)), 24, camera_stability=0.8)
        self.assertEqual([rally["id"] for rally in rallies], ["R01", "R02"])

    def test_rejects_a_tiny_burst_and_an_empty_signal(self):
        self.assertEqual(detect_rallies(timeline((5, 5.5)), 24, 1), [])
        self.assertEqual(detect_rallies([], 24, 1), [])


if __name__ == "__main__":
    unittest.main()
