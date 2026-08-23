import unittest

import numpy as np

from analysis.side_switch_full_union_ranker import fit_weighted_logistic
from analysis.side_switch_rare_event_losses import (
    effective_number_class_weights,
    fit_effective_number_logistic,
    fit_focal_logistic,
)
from analysis.side_switch_v3 import V3Event
from analysis.side_switch_v6 import matrix_for


def _events() -> list[V3Event]:
    result = []
    for index, (value, label) in enumerate(((-2.0, 0), (-1.0, 0), (1.0, 1), (2.0, 1))):
        row = {
            "eventId": str(index),
            "recordingId": "video",
            "features": {"x": value},
        }
        result.append(V3Event(str(index), "video", "research", index + 1, label, row))
    return result


class SideSwitchRareEventLossTests(unittest.TestCase):
    def test_focal_gamma_zero_is_exact_weighted_logistic(self) -> None:
        events = _events()
        control = fit_weighted_logistic(events, 0.1, ("x",), 0.5)
        focal = fit_focal_logistic(events, 0.1, ("x",), 0.5, 0.0)
        self.assertEqual(control.to_dict(), focal.to_dict())

    def test_focal_fit_is_finite_and_orders_simple_classes(self) -> None:
        events = _events()
        model = fit_focal_logistic(events, 0.1, ("x",), 0.5, 2.0)
        scores = model.predict_proba(matrix_for(events, ("x",)))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(np.mean(scores[2:])), float(np.mean(scores[:2])))

    def test_effective_number_increases_minority_class_weight(self) -> None:
        positive, negative = effective_number_class_weights(2, 20, 0.99)
        self.assertGreater(positive, negative)
        model = fit_effective_number_logistic(_events(), 0.1, ("x",), 0.99)
        self.assertTrue(
            np.isfinite(model.predict_proba(matrix_for(_events(), ("x",)))).all()
        )


if __name__ == "__main__":
    unittest.main()
