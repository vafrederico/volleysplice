import unittest

import numpy as np

from analysis.serving_side import (
    FAR_SIDE,
    NEAR_SIDE,
    SideOccupancy,
    infer_note_serving_side,
    occupancy_change_margin,
    pixel_motion,
    side_margin,
)


class ServingSideCueTests(unittest.TestCase):
    def test_extracts_explicit_near_and_far_serving_phrases(self) -> None:
        near = infer_note_serving_side("Near-side serve contact is visible and audible.")
        far = infer_note_serving_side("Far-side contact is inferred from receiver movement.")

        self.assertEqual(near.side, NEAR_SIDE)
        self.assertEqual(near.strength, "strong")
        self.assertEqual(far.side, FAR_SIDE)
        self.assertEqual(far.strength, "weak")

    def test_does_not_promote_unrelated_player_or_ball_phrases(self) -> None:
        cue = infer_note_serving_side("The far player goes down and the foreground ball is retrieved.")
        self.assertIsNone(cue.side)

    def test_conflicting_or_outside_phrase_is_unknown(self) -> None:
        cue = infer_note_serving_side("Serve originates at the far side or outside the clearest view.")
        self.assertIsNone(cue.side)
        self.assertEqual(cue.strength, "ambiguous")


class ServingSideFeatureTests(unittest.TestCase):
    def test_side_margin_is_signed_near_minus_far(self) -> None:
        self.assertAlmostEqual(side_margin(3.0, 1.0) or 0.0, 0.5)
        self.assertAlmostEqual(side_margin(1.0, 3.0) or 0.0, -0.5)
        self.assertEqual(side_margin(0.0, 0.0), 0.0)

    def test_pixel_motion_identifies_changed_half(self) -> None:
        reference = np.zeros((20, 10, 3), dtype=np.uint8)
        near_action = reference.copy()
        near_action[10:, :] = 255
        far_action = reference.copy()
        far_action[:10, :] = 255

        _, _, near_margin = pixel_motion(reference, [near_action])
        _, _, far_margin = pixel_motion(reference, [far_action])

        self.assertGreater(near_margin or 0.0, 0.9)
        self.assertLess(far_margin or 0.0, -0.9)

    def test_occupancy_change_identifies_larger_side_change(self) -> None:
        before = SideOccupancy(0.10, 0.20, 2, 2, 0.5, 0.5)
        after = SideOccupancy(0.30, 0.22, 2, 2, 0.5, 0.5)
        self.assertGreater(occupancy_change_margin(before, after, "area") or 0.0, 0.8)


if __name__ == "__main__":
    unittest.main()
