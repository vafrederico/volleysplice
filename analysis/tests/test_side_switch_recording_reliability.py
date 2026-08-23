import unittest

import numpy as np

from analysis.side_switch_recording_reliability import (
    RELIABILITY_FEATURE_NAMES,
    adjust_scores,
    fit_reliability_head,
    recording_summary,
)


def _row(event_id: str, recording_id: str, time: float, scale: float) -> dict:
    return {
        "eventId": event_id,
        "recordingId": recording_id,
        "transitionTime": time,
        "features": {
            "v4MaximumCameraShift": 0.01 * scale,
            "v4MinimumAlignmentResponse": 0.5 * scale,
            "minimumPlayerSideSeparation": 0.1 * scale,
            "minimumProposalCoverage": 0.2 * scale,
            "beforePlayerPaletteInstability": 0.3 * scale,
            "afterPlayerPaletteInstability": 0.4 * scale,
            "playerGlobalAppearanceChange": 0.1 * scale,
            "productionMaximumAdjacentAnchorErrorSeconds": 1.0 * scale,
            "productionMinimumAdjacentServeConfidence": 0.8 * scale,
        },
    }


class SideSwitchRecordingReliabilityTests(unittest.TestCase):
    def test_summary_is_finite_and_label_independent(self) -> None:
        rows = [_row("a", "video", 30.0, 1.0), _row("b", "video", 60.0, 2.0)]
        summary = recording_summary(rows, np.asarray([0.2, 0.8]), 0.5)
        self.assertEqual(tuple(summary), RELIABILITY_FEATURE_NAMES)
        self.assertTrue(np.isfinite(list(summary.values())).all())
        self.assertEqual(summary["fractionAboveBaseThreshold"], 0.5)

    def test_positive_threshold_offset_suppresses_recording_scores(self) -> None:
        rows = [_row("a", "video", 30.0, 1.0)]
        scores = np.asarray([0.8])
        self.assertEqual(
            adjust_scores(rows, scores, {"video": 1.0}, 0.0).tolist(),
            scores.tolist(),
        )
        self.assertLess(adjust_scores(rows, scores, {"video": 1.0}, 1.0)[0], 0.8)

    def test_ridge_head_fits_finite_recording_targets(self) -> None:
        summaries = []
        for value in (1.0, 2.0, 3.0):
            summaries.append(
                recording_summary(
                    [_row(str(value), str(value), 60.0, value)],
                    np.asarray([0.1 * value]),
                    0.5,
                )
            )
        head = fit_reliability_head(summaries, np.asarray([-1.0, 0.0, 1.0]), 1.0)
        predictions = head.predict(
            np.asarray([[summary[name] for name in RELIABILITY_FEATURE_NAMES] for summary in summaries])
        )
        self.assertTrue(np.isfinite(predictions).all())
        self.assertGreater(predictions[-1], predictions[0])


if __name__ == "__main__":
    unittest.main()
