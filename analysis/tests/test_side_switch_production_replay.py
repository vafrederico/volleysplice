import base64
import unittest

import numpy as np

from analysis.side_switch_production_replay import numeric_array


class SideSwitchProductionReplayTests(unittest.TestCase):
    def test_numeric_array_decodes_frozen_feedback_encoding(self) -> None:
        values = np.asarray([0.25, 0.75], dtype="<f4")
        payload = {
            "encoding": "base64",
            "byteOrder": "little-endian",
            "dataType": "float32",
            "shape": [2],
            "data": base64.b64encode(values.tobytes()).decode("ascii"),
        }
        np.testing.assert_array_equal(numeric_array(payload, "float32", (2,)), values)

    def test_numeric_array_rejects_shape_drift(self) -> None:
        payload = {
            "encoding": "base64",
            "byteOrder": "little-endian",
            "dataType": "float32",
            "shape": [1],
            "data": base64.b64encode(np.asarray([1.0], dtype="<f4").tobytes()).decode(
                "ascii"
            ),
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            numeric_array(payload, "float32", (2,))


if __name__ == "__main__":
    unittest.main()
