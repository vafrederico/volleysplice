from __future__ import annotations

import unittest

import numpy as np

from analysis.side_switch_v2 import (
    FEATURE_SETS,
    CourtPlayer,
    DecoderSettings,
    SideFrame,
    V2Event,
    V2Model,
    bind_recording_normalization,
    calibrate_side_divider,
    decode_sequence,
    derived_existing_features,
    fit_model,
    grouped_cross_fit,
    matrix_for,
    side_pair_features,
)


def _palette(index: int) -> np.ndarray:
    value = np.full(4, 1e-4, dtype=np.float64)
    value[index] = 1.0
    return value / np.sum(value)


def _event(
    recording_id: str,
    rally_order: int,
    label: int,
    value: float,
    *,
    before_orientation: float = 1.0,
    after_orientation: float = 1.0,
) -> V2Event:
    row = {
        "eventId": f"{recording_id}:{rally_order}",
        "recordingId": recording_id,
        "rallyOrder": rally_order,
        "features": {name: value for name in FEATURE_SETS["derived-existing"]},
        "normalizedFeatures": {
            name: value for name in FEATURE_SETS["derived-existing"]
        },
        "beforeOrientation": before_orientation,
        "afterOrientation": after_orientation,
    }
    return V2Event(
        event_id=str(row["eventId"]),
        recording_id=recording_id,
        role="train",
        rally_order=rally_order,
        decision="switch" if label else "no-switch",
        label=label,
        row=row,
    )


class SideSwitchV2Tests(unittest.TestCase):
    def test_recording_geometry_calibration_adapts_to_camera_distance(self) -> None:
        frames = [
            [
                CourtPlayer(_palette(0), 100.0, 0.4, far),
                CourtPlayer(_palette(1), 180.0, 0.6, near),
            ]
            for far, near in zip(
                np.linspace(0.48, 0.52, 8),
                np.linspace(0.67, 0.72, 8),
                strict=True,
            )
        ]

        divider, audit = calibrate_side_divider(frames)

        self.assertEqual(audit["source"], "unlabeled-weighted-two-means")
        self.assertGreater(divider, 0.58)
        self.assertLess(divider, 0.64)

    def test_side_pair_cost_prefers_swapped_assignment(self) -> None:
        before = SideFrame(1.0, _palette(0), _palette(1), _palette(0) - _palette(1), 2)
        after = SideFrame(2.0, _palette(1), _palette(0), _palette(1) - _palette(0), 2)

        features, before_moment, after_moment = side_pair_features([before], [after])

        self.assertGreater(features["sideSwapMarginMedian"] or 0.0, 0.9)
        self.assertEqual(features["sideSwapSupportFraction"], 1.0)
        self.assertEqual(features["sideMinimumCoverage"], 1.0)
        self.assertIsNotNone(before_moment)
        self.assertIsNotNone(after_moment)

    def test_frame_consistency_records_missing_side_coverage(self) -> None:
        complete = SideFrame(1.0, _palette(0), _palette(1), _palette(0), 2)
        missing = SideFrame(2.0, _palette(0), None, _palette(0), 1)

        features, _, _ = side_pair_features(
            [complete, missing], [complete, missing]
        )

        self.assertEqual(features["sideUsablePairCount"], 1.0)
        self.assertEqual(features["sideMinimumCoverage"], 0.5)

    def test_derived_features_include_predeclared_interaction(self) -> None:
        features = derived_existing_features(
            {
                "features": {
                    "playerPaletteEqual": 0.4,
                    "playerPaletteArea": 0.6,
                    "fullFrameControl": 0.2,
                    "detectionCountChange": 0.1,
                    "boxAreaChange": 0.2,
                    "medianBoxHeightChange": 0.3,
                }
            }
        )

        self.assertAlmostEqual(features["paletteDistanceMean"] or 0.0, 0.5)
        self.assertAlmostEqual(features["geometryStability"] or 0.0, 0.8)
        self.assertAlmostEqual(
            features["paletteByGeometryStability"] or 0.0, 0.4
        )

    def test_normalization_uses_complete_recording_without_labels(self) -> None:
        rows = [
            {
                "eventId": f"r:{index}",
                "recordingId": "r",
                "features": {name: float(index) for name in FEATURE_SETS["combined"]},
            }
            for index in range(3)
        ]

        metadata = bind_recording_normalization(rows)

        self.assertEqual(metadata["r"]["candidateCount"], 3)
        self.assertEqual(rows[1]["normalizedFeatures"]["paletteDistanceMean"], 0.0)
        self.assertNotIn("decision", metadata["r"])

    def test_gap_duration_is_absent_from_every_learned_feature_set(self) -> None:
        self.assertTrue(
            all(
                "gap" not in feature.lower()
                for features in FEATURE_SETS.values()
                for feature in features
            )
        )

    def test_grouped_fit_round_trips(self) -> None:
        events = []
        for recording_id in ("a", "b", "c"):
            events.extend(
                [
                    _event(recording_id, 1, 0, 0.05),
                    _event(recording_id, 2, 0, 0.15),
                    _event(recording_id, 3, 1, 0.85),
                    _event(recording_id, 4, 1, 0.95),
                ]
            )
        model = fit_model(events, "derived-existing", 0.1)
        restored = V2Model.from_dict(model.to_dict())
        scores, audit = grouped_cross_fit(events, "derived-existing", 0.1)

        np.testing.assert_allclose(
            model.predict_proba(matrix_for(events, model.feature_set)),
            restored.predict_proba(matrix_for(events, model.feature_set)),
        )
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(len(audit["folds"]), 3)

    def test_decoder_applies_spacing_and_toggle_consistency(self) -> None:
        events = [
            _event("r", 1, 1, 0.0, before_orientation=1.0, after_orientation=-1.0),
            _event("r", 2, 0, 0.0, before_orientation=-1.0, after_orientation=-1.0),
        ]
        settings = DecoderSettings(
            minimum_spacing_rallies=4,
            close_switch_penalty=4.0,
            orientation_weight=1.0,
            extra_switch_penalty=0.0,
        )

        predictions, scores, _ = decode_sequence(
            events, np.asarray([0.8, 0.8]), 0.5, settings
        )

        self.assertEqual(predictions.tolist(), [True, False])
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
