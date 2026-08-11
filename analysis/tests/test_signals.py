from __future__ import annotations

import unittest

import numpy as np

from analysis.signals import combine_signals, robust_normalize


class SignalTests(unittest.TestCase):
    def test_constant_and_tiny_spread_signals_do_not_become_activity(self):
        np.testing.assert_array_equal(robust_normalize(np.ones(10)), np.zeros(10))
        np.testing.assert_array_equal(
            robust_normalize(np.linspace(0, 0.001, 10), minimum_spread=0.004),
            np.zeros(10),
        )

    def test_combined_signal_aligns_audio_to_motion_timebase(self):
        combined = combine_signals(
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
        self.assertEqual([sample["time"] for sample in combined], [0.0, 0.5, 1.0])
        self.assertGreater(combined[1]["audio"], combined[0]["audio"])


if __name__ == "__main__":
    unittest.main()
