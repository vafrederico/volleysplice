from __future__ import annotations

import cv2
import numpy as np
import unittest

from analysis.side_switch_v4 import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    CourtGeometry,
    SequenceSummary,
    SidePaletteSummary,
)
from analysis.side_switch_v5 import PlayerSequenceSummary
from analysis.side_switch_visual_summary_v2 import (
    C1_FEATURE_NAMES,
    P1_FEATURE_NAMES,
    Q1_FEATURE_NAMES,
    CameraSummary,
    SideObservationV2,
    c1_features,
    p1_features,
    q1_features,
    summarize_observation,
)


def _palette(index: int) -> np.ndarray:
    value = np.zeros(52, dtype=np.float64)
    value[index] = 1.0
    return value


def _observation(*, swapped: bool = False) -> SideObservationV2:
    near = _palette(1 if swapped else 0)
    far = _palette(0 if swapped else 1)
    side = SidePaletteSummary(near=near, far=far, instability=0.1)
    player = PlayerSequenceSummary(
        near_palette=near,
        far_palette=far,
        global_palette=0.5 * (near + far),
        palette_instability=0.2,
        proposal_coverage=0.3 if swapped else 0.4,
        proposal_count=2.0 if swapped else 3.0,
        near_support=0.6,
        far_support=0.4,
    )
    camera = CameraSummary(
        translation_vectors=((0.0, 0.0),) * 7,
        alignment_responses=(0.8,) * 7,
        shift_dispersion=0.0,
        residual_p90=0.01,
        background_palette=_palette(2),
        frame_global_palettes=(_palette(3),) * 7,
    )
    return SideObservationV2(
        v4=SequenceSummary(
            broad=side,
            tight=side,
            global_palette=0.5 * (near + far),
            foreground_coverage=0.2,
            maximum_camera_shift=0.02 if swapped else 0.01,
            minimum_alignment_response=0.7 if swapped else 0.9,
        ),
        player=player,
        camera=camera,
    )


class SideSwitchVisualSummaryV2Test(unittest.TestCase):
    def test_q1_preserves_directional_values_in_frozen_order(self) -> None:
        values = q1_features(_observation(), _observation(swapped=True))
        self.assertEqual(tuple(values), Q1_FEATURE_NAMES)
        self.assertEqual(values["beforeProposalCount"], 3.0)
        self.assertEqual(values["afterProposalCount"], 2.0)
        self.assertEqual(values["beforeV4MinimumAlignmentResponse"], 0.9)
        self.assertEqual(values["afterV4MinimumAlignmentResponse"], 0.7)

    def test_c1_returns_finite_frozen_bundle(self) -> None:
        values = c1_features(_observation(), _observation(swapped=True))
        self.assertEqual(tuple(values), C1_FEATURE_NAMES)
        self.assertTrue(all(np.isfinite(value) for value in values.values()))

    def test_p1_perfect_persistent_swap_is_positive(self) -> None:
        values, diagnostics = p1_features(
            [_observation(), _observation(), _observation()],
            [
                _observation(swapped=True),
                _observation(swapped=True),
                _observation(swapped=True),
            ],
        )
        self.assertEqual(tuple(values), P1_FEATURE_NAMES)
        self.assertGreater(values["playerCrossSwapQ25"], 0.99)
        self.assertGreater(values["playerWithinContinuityMinimum"], 0.99)
        self.assertGreater(values["playerPersistentSwapMinimum"], 0.99)
        self.assertGreater(values["courtPersistentSwapMinimum"], 0.99)
        self.assertEqual(diagnostics["contextFraction"], 1.0)

    def test_p1_perfect_continuity_is_negative(self) -> None:
        values, _ = p1_features(
            [_observation(), _observation()],
            [_observation(), _observation()],
        )
        self.assertLess(values["playerCrossSwapQ25"], -0.99)
        self.assertLess(values["playerPersistentSwapMinimum"], -0.99)

    def test_p1_missing_within_context_is_neutral_and_explicit(self) -> None:
        values, diagnostics = p1_features(
            [_observation()], [_observation(swapped=True)]
        )
        self.assertEqual(values["playerWithinContinuityMinimum"], 0.0)
        self.assertEqual(values["playerPersistentSwapMinimum"], 0.0)
        self.assertEqual(values["minimumPersistentContextFraction"], 1.0 / 3.0)
        self.assertEqual(
            diagnostics["missingWithinPairPolicy"],
            "neutral-zero-plus-context-fraction",
        )

    def test_camera_summary_is_stable_under_small_translation(self) -> None:
        base = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(base, (30, 20), (90, 95), (20, 180, 230), -1)
        cv2.circle(base, (180, 80), 18, (210, 60, 30), -1)
        frames = []
        for dx in (-3, -2, -1, 0, 1, 2, 3):
            transform = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, 0.0]])
            frames.append(
                cv2.warpAffine(
                    base,
                    transform,
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    borderMode=cv2.BORDER_REFLECT,
                )
            )
        observation = summarize_observation(
            frames, CourtGeometry(0.5, 1.0, 7, 7)
        )
        self.assertLess(observation.camera.shift_dispersion, 0.02)
        self.assertLess(observation.camera.residual_p90, 0.02)
        self.assertEqual(len(observation.camera.translation_vectors), 7)
        self.assertEqual(len(observation.camera.alignment_responses), 7)
