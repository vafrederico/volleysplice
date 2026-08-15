from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.train_environment_specialists import hard_negative_augmented


class EnvironmentSpecialistExperimentTests(unittest.TestCase):
    def test_hard_negative_rows_are_repeated_without_relabeling(self) -> None:
        item = SimpleNamespace(
            sequence=SimpleNamespace(times=np.asarray([0.0, 1.0, 2.0, 3.0])),
            recording=SimpleNamespace(
                raw={"hardNegatives": [{"start": 1.0, "end": 3.0}]}
            ),
            sample_mask=np.asarray([True, True, True, True]),
            labels=np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        values = np.arange(8, dtype=np.float32).reshape(4, 2)
        labels = item.labels.copy()

        expanded_values, expanded_labels, hard_samples = hard_negative_augmented(
            item, values, labels, multiplier=4
        )

        self.assertEqual(hard_samples, 2)
        self.assertEqual(expanded_values.shape, (10, 2))
        self.assertEqual(int(np.sum(expanded_labels == 0.0)), 8)
        np.testing.assert_array_equal(expanded_values[-2:], values[1:3])


if __name__ == "__main__":
    unittest.main()
