from __future__ import annotations

import unittest

import numpy as np

from analysis.serving_side_v2 import (
    FEATURE_NAMES,
    BoostedStumpModel,
    LogisticModel,
    extract_window_features,
    fit_boosted_stumps,
    fit_logistic,
    service_zone_masks,
)


class ServingSideV2Tests(unittest.TestCase):
    def test_roi_fallback_produces_disjoint_end_bands(self) -> None:
        masks, provenance = service_zone_masks(
            100, 160, roi=(0.0, 0.0, 1.0, 1.0), court_geometry=None
        )

        self.assertEqual(provenance, "roi-relative-end-bands")
        self.assertFalse(np.any(masks["near"] & masks["far"]))
        self.assertTrue(masks["near"][-1].all())
        self.assertTrue(masks["far"][0].all())

    def test_annotated_service_anchors_override_fallback(self) -> None:
        masks, provenance = service_zone_masks(
            100,
            160,
            roi=(0.1, 0.1, 0.8, 0.8),
            court_geometry={
                "serviceZoneAnchors": {
                    "near": {"x": 0.5, "y": 0.8},
                    "far": {"x": 0.5, "y": 0.2},
                }
            },
        )

        self.assertEqual(provenance, "annotated-service-zone-anchors")
        self.assertGreater(np.mean(np.where(masks["near"])[0]), np.mean(np.where(masks["far"])[0]))

    def test_temporal_features_detect_near_band_motion(self) -> None:
        frames = []
        for index in range(8):
            frame = np.zeros((108, 192, 3), dtype=np.uint8)
            if index >= 3:
                left = min(150, 20 + index * 12)
                frame[78:100, left : left + 16] = 255
            frames.append(frame)
        masks, _ = service_zone_masks(108, 192, roi=(0, 0, 1, 1), court_geometry=None)

        values = extract_window_features(frames, masks)

        self.assertEqual(tuple(values), FEATURE_NAMES)
        self.assertGreater(values["contact:near:activeFraction"], values["contact:far:activeFraction"])

    def test_small_models_separate_and_round_trip(self) -> None:
        values = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0, 0, 1, 1])

        logistic = fit_logistic(values, labels, l2=0.1)
        boosted = fit_boosted_stumps(values, labels, estimators=5, learning_rate=0.1)

        self.assertLess(logistic.predict_proba(values)[1], logistic.predict_proba(values)[2])
        self.assertLess(boosted.predict_proba(values)[1], boosted.predict_proba(values)[2])
        np.testing.assert_allclose(
            LogisticModel.from_dict(logistic.to_dict()).predict_proba(values),
            logistic.predict_proba(values),
        )
        np.testing.assert_allclose(
            BoostedStumpModel.from_dict(boosted.to_dict()).predict_proba(values),
            boosted.predict_proba(values),
        )


if __name__ == "__main__":
    unittest.main()
